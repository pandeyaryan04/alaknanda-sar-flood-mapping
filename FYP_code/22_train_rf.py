r"""
22_train_rf.py
==============
Trains the Random Forest flood model on Sen1Floods11.

WHAT IT DOES
------------
1. Takes the training chips from data\models\chip_index.csv: 'hand' = the
   hand-labelled train chips; 'hand_weak' = those plus the kept weak chips.
   Never a valid, test or Bolivia chip.
2. From each chip, samples up to N water and N not-water pixels (default
   1000 each, random, seeded), so every chip and both classes count. The
   total is capped so the finished forest fits in --max-ram-gb (default 12):
   the cap is an upper bound on the forest's size, printed before training.
   With hand_weak, the weak chips give as many rows as the hand chips
   (50/50, the same mix as the U-Net's batches). The first run of 7 October
   gave the weak chips the whole rest of the row budget instead (65 % of the
   rows); --weak-fill reproduces it.
3. Features: VV, VH (dB), HAND (m), slope (degrees), clipped with the bounds
   in norm_stats.json; a missing value becomes the channel mean.
4. RandomForestClassifier: 300 trees, min_samples_leaf 5, max_features sqrt,
   all CPU cores but one, random_state = --seed.
5. Chooses the probability threshold that gives the best water IoU on the
   hand VALID chips (and records the IoU at 0.5 too). Corridor maps use the
   valid-chosen threshold.
6. Saves model.joblib (compress 3), model_meta.json (with the scikit-learn
   version), a copy of norm_stats.json, train_chips.txt and run_record.json
   into data\models\rf_<trainset>\.

USAGE: HOW TO RUN THIS  (copy one line, paste into Anaconda Prompt, press Enter)
================================================================================
    conda activate fyp
    cd /d "<project folder>\04 Code\FYP_code"

    python 22_train_rf.py --selftest
    python 22_train_rf.py --trainset hand
    python 22_train_rf.py --trainset hand_weak

On the 16 GB laptop keep the default --max-ram-gb 12. The last line says
PASS and prints the valid IoU and the chosen threshold.
================================================================================
"""

__stage__ = "Models: Random Forest training"
__reads__ = ["data/models/chip_index.csv", "data/models/norm_stats.json", "Sen1Floods11 chips"]
__writes__ = ["data/models/rf_<trainset>/model.joblib", "model_meta.json", "norm_stats.json",
              "train_chips.txt", "run_record.json"]

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import models_common as mc

try:
    import joblib
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
except ImportError as e:
    sys.exit(f"\nMissing package: {e}.\nActivate the environment first:  conda activate fyp\n")


def quotas(hand, weak, n_per_class, cap, hand_fraction, weak_fill=False):
    """Per-chip samples per class for hand and weak chips under a total row cap.

    Default: the weak chips give about as many rows as the hand chips plan to
    (hand_fraction of the rows are hand rows). weak_fill: the weak chips take
    the whole rest of the row budget (the first run of 7 October).
    """
    if not weak:
        return min(n_per_class, cap // max(2 * len(hand), 1)), 0
    qh = min(n_per_class, int(cap * hand_fraction) // (2 * len(hand)))
    planned = sum(min(qh, int(r["labelled_pixels"] * r["water_share"]))
                  + min(qh, int(r["labelled_pixels"] * (1 - r["water_share"]))) for r in hand)
    room = max(cap - planned, 0)
    if not weak_fill:
        room = min(room, int(planned * (1 - hand_fraction) / hand_fraction))
    qw = min(n_per_class, room // (2 * len(weak)))
    return qh, qw


def sample_rows(rows, quota, data_root, stats, seed, offset, prog):
    feats, labels, used = [], [], []
    for i, r in enumerate(rows):
        x, y = mc.read_chip(r, data_root)
        rng = np.random.default_rng([seed, offset + i])
        tv = mc.train_valid(x, y)
        pick = []
        for cls in (1, 0):
            idx = np.flatnonzero(tv & (y == cls))
            if idx.size > quota:
                idx = np.sort(rng.choice(idx, size=quota, replace=False))
            pick.append(idx)
        idx = np.concatenate(pick)
        if idx.size:
            feats.append(x.reshape(4, -1)[:, idx].T)
            labels.append(y.ravel()[idx])
            used.append(r["chip"])
        prog.tick()
    return feats, labels, used


def fit_forest(X, y, args):
    """Grow the forest a tenth at a time (warm start), printing progress and an ETA.

    scikit-learn draws every tree's seed from random_state in order, so the
    forest is the same as one grown in a single call.
    """
    clf = RandomForestClassifier(n_estimators=1, min_samples_leaf=args.min_samples_leaf,
                                 max_features="sqrt", n_jobs=args.n_jobs,
                                 random_state=args.seed, warm_start=True)
    step = max(1, args.n_estimators // 10)
    t0, done = time.time(), 0
    while done < args.n_estimators:
        done = min(done + step, args.n_estimators)
        clf.set_params(n_estimators=done)
        clf.fit(X, y)
        el = time.time() - t0
        print(f"    {done}/{args.n_estimators} trees, {el / 60:.1f} min so far, about "
              f"{el / done * (args.n_estimators - done) / 60:.1f} min left", flush=True)
    clf.set_params(warm_start=False)
    return clf


def valid_threshold(clf_pred, rows, data_root, stats):
    hist = mc.ThresholdHist()
    prog = mc.Progress(len(rows), "valid chips predicted", every=120)
    for r in rows:
        x, y = mc.read_chip(r, data_root)
        tv = mc.train_valid(x, y)
        if tv.any():
            hist.add(clf_pred(mc.rf_features(x[:, tv].T, stats)), y[tv] == 1)
        prog.tick()
    return hist


def main(argv=None):
    mc.setup_console()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mc.add_common_args(ap)
    ap.add_argument("--trainset", choices=mc.TRAINSETS, default="hand")
    ap.add_argument("--n-per-class", type=int, default=1000,
                    help="pixels per class per chip, at most (default 1000)")
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--min-samples-leaf", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--hand-fraction", type=float, default=0.5,
                    help="hand_weak: share of the training rows that are hand rows (default 0.5)")
    ap.add_argument("--weak-fill", action="store_true",
                    help="hand_weak: give the weak chips the whole rest of the row budget, as "
                         "the first run of 7 October did (65 %% weak rows)")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.smoke:
        args.n_estimators = min(args.n_estimators, 20)
        args.n_per_class = min(args.n_per_class, 200)
    paths = mc.paths_from_args(args)
    out = paths.model_dir("rf", args.trainset)
    print(f"22_train_rf.py: Random Forest, trainset {args.trainset}"
          f"{' (smoke)' if args.smoke else ''}")
    if mc.guard(out / "run_record.json", args, f"The Random Forest '{args.trainset}'"):
        return 0
    t0 = time.time()
    index = mc.load_chip_index(paths)
    stats = mc.load_norm_stats(paths.norm_stats)
    hand, weak = mc.training_rows(index, args.trainset)
    valid_rows = mc.split_rows(index, "valid")
    if not hand:
        mc.fail("No hand train chips in the chip index. Run 21_prepare_chips.py again.")

    cap = mc.rf_row_cap(args.max_ram_gb, args.n_estimators, args.min_samples_leaf)
    if not 0 < args.hand_fraction < 1:
        mc.fail(f"--hand-fraction must be between 0 and 1, not {args.hand_fraction}.")
    qh, qw = quotas(hand, weak, args.n_per_class, cap, args.hand_fraction, args.weak_fill)
    print(f"  chips: {len(hand)} hand train" + (f" + {len(weak)} weak" if weak else ""))
    print(f"  row budget for --max-ram-gb {args.max_ram_gb:g}: {cap:,} rows "
          f"(upper bound {mc.rf_bytes_per_row(args.n_estimators, args.min_samples_leaf):,.0f} "
          "bytes per row for the finished forest)")
    print(f"  per chip, per class: up to {qh} (hand)" + (f", {qw} (weak)" if weak else ""))
    if qh < 1 or (weak and qw < 1):
        mc.fail(f"--max-ram-gb {args.max_ram_gb:g} is too small for even one pixel per chip. "
                "Use a larger value (12 is the default).")

    prog = mc.Progress(len(hand) + len(weak), "chips sampled", every=120)
    fh, lh, uh = sample_rows(hand, qh, paths.data, stats, args.seed, 0, prog)
    fw, lw, uw = sample_rows(weak, qw, paths.data, stats, args.seed, len(hand), prog)
    X = mc.rf_features(np.concatenate(fh + fw), stats)
    y = np.concatenate(lh + lw).astype("int8")
    n_water = int((y == 1).sum())
    hand_rows, weak_rows = sum(map(len, lh)), sum(map(len, lw))
    print(f"  training rows: {len(y):,}  water {n_water:,} ({100 * n_water / len(y):.1f} %), "
          f"not water {len(y) - n_water:,}; hand rows {hand_rows:,}, weak rows {weak_rows:,}"
          + (f" ({100 * weak_rows / len(y):.0f} % weak)" if weak else ""))
    print(f"  feature table in RAM: {X.nbytes / 1e9:.3f} GB; forest upper bound "
          f"{len(y) * mc.rf_bytes_per_row(args.n_estimators, args.min_samples_leaf) / 1e9:.2f} GB")
    classes_note = ""
    if len(np.unique(y)) < 2:
        classes_note = "only one class in the training rows; the forest predicts it everywhere"
        print(f"  WARNING: {classes_note}")

    print(f"  fitting {args.n_estimators} trees on {args.n_jobs} CPU cores ...", flush=True)
    t_fit = time.time()
    clf = fit_forest(X, y, args)
    t_fit = time.time() - t_fit
    nodes = sum(t.tree_.node_count for t in clf.estimators_)
    print(f"  fitted in {t_fit / 60:.1f} min; forest holds {nodes:,} nodes, about "
          f"{nodes * 80 / 1e9:.2f} GB in RAM")
    del X

    classes = list(clf.classes_)

    def pred(f):
        out = np.empty(len(f), "float32")
        for i in range(0, len(f), mc.CHUNK):
            out[i:i + mc.CHUNK] = mc.rf_water_prob(clf, f[i:i + mc.CHUNK], args.n_jobs)
        return out

    print(f"  choosing the threshold on {len(valid_rows)} hand valid chips")
    hist = valid_threshold(pred, valid_rows, paths.data, stats)
    thr, iou_thr = hist.best()
    iou05 = hist.iou(0.5)
    print(f"  valid water IoU: {iou05:.4f} at threshold 0.5; {iou_thr:.4f} at the chosen "
          f"threshold {thr:.2f}")

    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "model.joblib.part"
    joblib.dump(clf, tmp, compress=3)
    os.replace(tmp, out / "model.joblib")
    shutil.copy2(paths.norm_stats, out / "norm_stats.json")
    (out / "train_chips.txt").write_text("\n".join(uh + uw) + "\n", encoding="utf-8")
    meta = {"model": "rf", "trainset": args.trainset, "sklearn_version": sklearn.__version__,
            "channels": list(mc.CHANNELS), "threshold_default": 0.5,
            "threshold_valid": thr,
            "threshold_how": "highest water IoU on the hand valid chips, 0.01 to 0.99 in 0.01",
            "iou_valid_at_0.5": iou05, "iou_valid_at_threshold": iou_thr,
            "n_estimators": args.n_estimators, "min_samples_leaf": args.min_samples_leaf,
            "max_features": "sqrt", "rows": int(len(y)), "water_rows": n_water,
            "forest_nodes": int(nodes), "hand_rows": int(hand_rows), "weak_rows": int(weak_rows),
            "weak_rule": ("" if not weak else "rest of the row budget (--weak-fill)"
                          if args.weak_fill else f"hand fraction {args.hand_fraction:g} of rows"),
            "classes": [int(c) for c in classes], "note": classes_note}
    mc.save_json(out / "model_meta.json", meta)
    rec = mc.run_record(args, paths, model="rf", trainset=args.trainset, seed=args.seed,
                        norm_file=str(paths.norm_stats), norm_sha256=mc.sha256(paths.norm_stats),
                        hand_chips=len(uh), weak_chips=len(uw), train_chip_ids=uh + uw,
                        per_chip_quota={"hand": qh, "weak": qw}, row_cap=cap,
                        rows=int(len(y)), water_rows=n_water, hand_rows=int(hand_rows),
                        weak_rows=int(weak_rows), weak_rule=meta["weak_rule"],
                        threshold=thr, threshold_how=meta["threshold_how"],
                        iou_valid_at_0_5=iou05, iou_valid_at_threshold=iou_thr,
                        forest_nodes=int(nodes), fit_seconds=round(t_fit, 1),
                        seconds=round(time.time() - t0, 1),
                        outputs=[mc.file_entry(out / f) for f in
                                 ("model.joblib", "model_meta.json", "norm_stats.json",
                                  "train_chips.txt")])
    mc.save_json(out / "run_record.json", rec)
    print(f"  wrote {out}")
    print(f"PASS  Random Forest '{args.trainset}': valid IoU {iou_thr:.3f} at threshold {thr:.2f}")
    return 0


def selftest():
    import models_synthetic as syn
    print("22_train_rf.py self-test")
    fast = ["--n-estimators", "15", "--n-per-class", "300", "--n-jobs", "2"]
    with tempfile.TemporaryDirectory(prefix="fyp selftest ", ignore_cleanup_errors=True) as td:
        proj = Path(td) / "Final Year Project"
        syn.build_project(proj, corridor=False)
        ok = mc.run_script("21_prepare_chips.py", ["--project", str(proj)]) == 0
        ok &= main(["--project", str(proj), "--trainset", "hand_weak"] + fast) == 0
        paths = mc.Paths(proj)
        p = mc.load_predictor("rf", paths.model_dir("rf", "hand_weak"))
        row = mc.split_rows(mc.load_chip_index(paths), "test")[0]
        x, y = mc.read_chip(row, paths.data)
        tv = mc.train_valid(x, y)
        prob = p.predict_image(x, mc.input_valid(x))
        iou = mc.metrics_from_confusion(*mc.confusion(prob[tv] >= p.threshold, y[tv] == 1))["iou"]
        a = prob.copy()
        ok2 = main(["--project", str(proj), "--trainset", "hand_weak", "--force"] + fast) == 0
        b = mc.load_predictor("rf", paths.model_dir("rf", "hand_weak")).predict_image(
            x, mc.input_valid(x))
    checks = [("trained and saved", ok), ("test chip IoU above 0.5", iou > 0.5),
              ("same seed, identical predictions", ok2 and np.array_equal(a, b, equal_nan=True))]
    for name, c in checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {name}")
    good = all(c for _, c in checks)
    print("SELFTEST PASS" if good else "SELFTEST FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(mc.cli(main))
