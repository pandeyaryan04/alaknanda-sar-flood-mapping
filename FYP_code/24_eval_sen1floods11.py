r"""
24_eval_sen1floods11.py
=======================
Scores a saved model on the Sen1Floods11 hand-labelled valid (89), test (90)
and Bolivia (15) chips. This is the final evaluation: the test and Bolivia
labels are read here and nowhere else.

For each split it reports, pooled over every scored pixel (VV and VH finite,
label 0 or 1): water IoU, F1, precision, recall and overall accuracy, at the
threshold chosen on valid and at 0.5; plus the mean of the per-chip water
IoUs (chips with no water in truth and none predicted have no IoU and are
left out of that mean; the count is given). Bolivia is an event the models
never saw: report it separately.

Writes 06 Outputs\tables\s1f11_<model>_<trainset>\metrics.csv and metrics.md.

USAGE: HOW TO RUN THIS  (copy one line, paste into Anaconda Prompt, press Enter)
================================================================================
    conda activate fyp
    cd /d "<project folder>\04 Code\FYP_code"

    python 24_eval_sen1floods11.py --selftest
    python 24_eval_sen1floods11.py --model rf --trainset hand
    python 24_eval_sen1floods11.py --model unet --trainset hand_weak
    python 24_eval_sen1floods11.py --model tm_s1t --trainset hand

About 2 minutes for the U-Net or TerraMind and 10 to 30 minutes for a Random
Forest.
The last line says PASS with the test IoU.
================================================================================
"""

__stage__ = "Models: evaluation on Sen1Floods11"
__reads__ = ["data/models/<model>_<trainset>/", "data/models/chip_index.csv",
             "Sen1Floods11 valid, test and Bolivia chips"]
__writes__ = ["06 Outputs/tables/s1f11_<model>_<trainset>/metrics.csv, metrics.md, "
              "run_record.json"]

import argparse
import csv
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import models_common as mc

SPLITS = ("valid", "test", "bolivia")
COLS = ["split", "threshold_kind", "threshold", "chips", "pixels", "water_pixels", "tp", "fp",
        "fn", "tn", "iou", "f1", "precision", "recall", "oa", "mean_chip_iou", "chips_in_mean"]


def evaluate(pred, rows, data_root, thresholds):
    conf = {k: np.zeros(4, "int64") for k in thresholds}
    chip_iou = {k: [] for k in thresholds}
    prog = mc.Progress(len(rows), "chips", every=120)
    for r in rows:
        x, y = mc.read_chip(r, data_root)
        v = mc.input_valid(x)
        prob = pred.predict_image(x, v)
        tv = v & (y >= 0)
        p, t = prob[tv], y[tv] == 1
        for k, thr in thresholds.items():
            c = mc.confusion(p >= thr, t)
            conf[k] += c
            iou = mc.metrics_from_confusion(*c)["iou"]
            if np.isfinite(iou):
                chip_iou[k].append(iou)
        prog.tick()
    return conf, chip_iou


def main(argv=None):
    mc.setup_console()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mc.add_common_args(ap, seed=False)
    ap.add_argument("--model", choices=mc.MODELS, required=False, default="rf")
    ap.add_argument("--trainset", choices=mc.TRAINSETS, default="hand")
    ap.add_argument("--cpu-ok", action="store_true", help="allow the U-Net or TerraMind on the CPU")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    mc.add_run_tag_arg(ap)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    mc.check_tile_args(args.tile, args.overlap)
    tag = mc.check_run_tag(args, seed_used=False)
    paths = mc.paths_from_args(args)
    name = f"s1f11_{args.model}_{args.trainset}" + (f"_{tag}" if tag else "")
    out = paths.tables / name
    print(f"24_eval_sen1floods11.py: {args.model} {args.trainset}"
          f"{' (smoke)' if args.smoke else ''}")
    if mc.guard(out / "metrics.md", args, f"The evaluation {name}"):
        return 0
    t0 = time.time()
    index = mc.load_chip_index(paths)
    mdir = paths.model_dir(args.model, args.trainset, tag)
    pred = mc.load_predictor(args.model, mdir, cpu_ok=args.cpu_ok or args.smoke, max_ram_gb=args.max_ram_gb,
                             tile=args.tile, overlap=args.overlap)
    thresholds = {"valid_chosen": pred.threshold, "default_0.5": 0.5}
    rows_out = []
    splits = ("valid",) if args.smoke else SPLITS
    if args.smoke:
        print("  smoke: valid chips only. Test and Bolivia labels are read only by the real, "
              "final evaluation.")
    for split in splits:
        rows = mc.split_rows(index, split)
        print(f"  {split}: {len(rows)} chips")
        conf, chip_iou = evaluate(pred, rows, paths.data, thresholds)
        for k, thr in thresholds.items():
            tp, fp, fn, tn = (int(v) for v in conf[k])
            m = mc.metrics_from_confusion(tp, fp, fn, tn)
            rows_out.append({"split": split, "threshold_kind": k, "threshold": thr,
                             "chips": len(rows), "pixels": tp + fp + fn + tn,
                             "water_pixels": tp + fn, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                             **{q: round(m[q], 4) for q in ("iou", "f1", "precision", "recall",
                                                             "oa")},
                             "mean_chip_iou": round(float(np.mean(chip_iou[k])), 4)
                             if chip_iou[k] else float("nan"),
                             "chips_in_mean": len(chip_iou[k])})
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows_out)
    lines = [f"# Sen1Floods11 evaluation: {args.model}, trained on {args.trainset}"
             + (f", repeat run {tag}" if tag else ""), "",
             f"Model: `{mdir}`", f"Threshold chosen on valid: {pred.threshold:.2f}", "",
             "Pooled over every scored pixel (VV and VH finite, label 0 or 1). Water class.",
             "Bolivia is an event never seen in training; report it separately.", "",
             "| Split | Threshold | Chips | Water px | IoU | F1 | Precision | Recall | OA | "
             "Mean chip IoU (n) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows_out:
        lines.append(f"| {r['split']} | {r['threshold_kind']} ({r['threshold']:.2f}) | "
                     f"{r['chips']} | {r['water_pixels']} | {r['iou']} | {r['f1']} | "
                     f"{r['precision']} | {r['recall']} | {r['oa']} | {r['mean_chip_iou']} "
                     f"({r['chips_in_mean']}) |")
    rec = mc.run_record(args, paths, step="evaluation", model=args.model,
                        trainset=args.trainset, run_tag=tag, model_dir=str(mdir),
                        threshold=pred.threshold, threshold_how="chosen on valid by 22, 23 or 30",
                        seconds=round(time.time() - t0, 1),
                        outputs=[mc.file_entry(out / "metrics.csv")])
    rec.update(getattr(pred, "record", {}))        # TerraMind: its weights and normalisation
    mc.save_json(out / "run_record.json", rec)
    tmp = out / "metrics.md.part"
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, out / "metrics.md")
    print("\n".join(lines[7:]))
    last = [r for r in rows_out if r["threshold_kind"] == "valid_chosen"][-1 if args.smoke else 1]
    print(f"\n  wrote {out}")
    print(f"PASS  {args.model} {args.trainset}{' ' + tag if tag else ''}: "
          f"{last['split']} IoU {last['iou']}")
    return 0


def selftest():
    import models_synthetic as syn
    print("24_eval_sen1floods11.py self-test")
    with tempfile.TemporaryDirectory(prefix="fyp selftest ", ignore_cleanup_errors=True) as td:
        proj = Path(td) / "Final Year Project"
        syn.build_project(proj, corridor=False)
        ok = mc.run_script("21_prepare_chips.py", ["--project", str(proj)]) == 0
        ok &= mc.run_script("22_train_rf.py", ["--project", str(proj), "--n-estimators", "10",
                                               "--n-per-class", "200"]) == 0
        ok &= mc.run_script("23_train_unet.py", ["--project", str(proj), "--base-width", "8",
                                                 "--crop", "32", "--batch-size", "4",
                                                 "--steps-per-epoch", "2", "--epochs", "1",
                                                 "--tile", "64", "--overlap", "16",
                                                 "--workers", "0", "--cpu-ok"]) == 0
        ok &= main(["--project", str(proj), "--model", "rf"]) == 0
        ok &= main(["--project", str(proj), "--model", "unet", "--cpu-ok", "--tile", "64",
                    "--overlap", "16"]) == 0
        paths = mc.Paths(proj)
        with open(paths.tables / "s1f11_rf_hand" / "metrics.csv", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        # recount the RF test IoU by hand
        p = mc.load_predictor("rf", paths.model_dir("rf", "hand"))
        tp = fp = fn = 0
        for r in mc.split_rows(mc.load_chip_index(paths), "test"):
            x, y = mc.read_chip(r, paths.data)
            v = mc.input_valid(x)
            tv = v & (y >= 0)
            pr = p.predict_image(x, v)[tv] >= p.threshold
            tp += int((pr & (y[tv] == 1)).sum())
            fp += int((pr & (y[tv] == 0)).sum())
            fn += int((~pr & (y[tv] == 1)).sum())
        got = next(float(r["iou"]) for r in rows
                   if r["split"] == "test" and r["threshold_kind"] == "valid_chosen")
    checks = [("both models evaluated", ok), ("6 rows", len(rows) == 6),
              ("test IoU recounted", abs(got - round(tp / (tp + fp + fn), 4)) < 1e-9)]
    for name, c in checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {name}")
    good = all(c for _, c in checks)
    print("SELFTEST PASS" if good else "SELFTEST FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(mc.cli(main))
