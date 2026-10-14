r"""
31_calibration.py
=================
How well do the models' water probabilities mean what they say? A check for
the paper. It changes no map, no model and no score.

WHY
---
The U-Net's best threshold on the Sen1Floods11 valid chips is 0.01 to 0.07,
not near 0.5, and its IoU at 0.5 jumps from epoch to epoch: its
probabilities are badly scaled ("calibrated"). The maps are still sound,
because every threshold is chosen on the valid chips. This script measures
the problem and shows how much a simple rescaling fixes it.

WHAT IT DOES, for every trained model it finds (default: the U-Net and both
TerraMind variants, each trained on hand and on hand_weak; add rf with
--models)
------------------------------------------------------------------------
1. Predicts the Sen1Floods11 valid, test and Bolivia chips and keeps a
   random, seeded sample of their scored pixels (VV, VH finite and label
   0 or 1): probability and true label.
2. On the VALID sample only, fits two rescalings of the model's output z
   (z = log(p / (1 - p))):
     temperature  p' = sigmoid(z / T)          (one number, T)
     Platt        p' = sigmoid(a * z + b)      (two numbers, a and b)
   Both keep the order of the pixels, so the best IoU over all thresholds
   does not change; only the threshold that gives it moves.
3. For valid, test and Bolivia, before and after each rescaling: the
   expected calibration error (ECE: in 15 equal bins of probability, the
   gap between the mean probability and the share that is really water,
   weighted by the pixels in the bin), the Brier score, the log loss, and
   the threshold with the best IoU. The test and Bolivia numbers show
   whether a rescaling fitted on valid holds on chips it never saw.
4. Writes, per model, 06 Outputs\models\calibration\<model>_<trainset>\
   calibration.json and reliability.png (a reliability diagram: predicted
   probability against the share that is water, before and after), and one
   table for all models: 06 Outputs\tables\calibration_summary.md and .csv.

USAGE: HOW TO RUN THIS  (copy one line, paste into Anaconda Prompt, press Enter)
================================================================================
    conda activate fyp
    cd /d "<project folder>\04 Code\FYP_code"

    python 31_calibration.py --selftest
    python 31_calibration.py
    python 31_calibration.py --models unet,tm_s1,tm_s1t,rf

About 5 to 15 minutes on the RTX 3050 for the six networks (estimate); the
Random Forest adds about 15 minutes per training set. A model that is not
trained yet is skipped with a line saying so. The last line says PASS.
================================================================================
"""

__stage__ = "Models: calibration of the water probabilities (Sen1Floods11, no map changed)"
__reads__ = ["data/models/<model>_<trainset>/ (trained models)", "data/models/chip_index.csv",
             "Sen1Floods11 valid, test and Bolivia chips"]
__writes__ = ["06 Outputs/models/calibration/<model>_<trainset>/calibration.json, reliability.png",
              "06 Outputs/tables/calibration_summary.md, calibration_summary.csv"]

import argparse
import csv
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import models_common as mc

try:
    from scipy.optimize import minimize, minimize_scalar
except ImportError as e:
    sys.exit(f"\nMissing package: {e}.\nActivate the environment first:  conda activate fyp\n")

SPLITS = ("valid", "test", "bolivia")
EPS = 1e-6
BINS = 15
SUMMARY_COLS = ["model", "trainset", "split", "pixels", "water_share", "ece_raw", "ece_temp",
                "ece_platt", "brier_raw", "brier_platt", "logloss_raw", "logloss_platt",
                "best_thr_raw", "best_thr_platt", "best_iou", "temperature", "platt_a",
                "platt_b"]


def logit(p):
    p = np.clip(np.asarray(p, "float64"), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def nll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def fit_temperature(z, y):
    """T > 0 that minimises the log loss of sigmoid(z / T)."""
    res = minimize_scalar(lambda lt: nll(sigmoid(z / math.exp(lt)), y), bounds=(-6, 6),
                          method="bounded")
    return float(math.exp(res.x))


def fit_platt(z, y):
    """a, b that minimise the log loss of sigmoid(a z + b)."""
    res = minimize(lambda ab: nll(sigmoid(ab[0] * z + ab[1]), y), x0=[1.0, 0.0],
                   method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 4000})
    return float(res.x[0]), float(res.x[1])


def reliability(p, y, bins=BINS):
    """Per bin: mean probability, share that is water, pixels; and the ECE."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    n = np.bincount(idx, minlength=bins).astype("float64")
    sp = np.bincount(idx, weights=p, minlength=bins)
    sy = np.bincount(idx, weights=y, minlength=bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mp, fy = sp / n, sy / n
    ece = float(np.nansum(n * np.abs(mp - fy)) / max(n.sum(), 1))
    return {"mean_prob": mp, "water_share": fy, "pixels": n, "ece": ece}


def best_threshold(p, y):
    h = mc.ThresholdHist()
    h.add(p, y.astype(bool))
    return h.best()


def scores(p, y):
    r = reliability(p, y)
    thr, iou = best_threshold(p, y)
    return {"ece": r["ece"], "brier": float(np.mean((p - y) ** 2)), "logloss": nll(p, y),
            "best_thr": float(thr), "best_iou": float(iou), "rel": r}


def sample_split(pred, rows, data_root, per_chip, rng):
    ps, ys = [], []
    prog = mc.Progress(len(rows), "chips", every=120)
    for r in rows:
        x, y = mc.read_chip(r, data_root)
        v = mc.input_valid(x)
        prob = pred.predict_image(x, v)
        tv = v & (y >= 0) & np.isfinite(prob)
        idx = np.flatnonzero(tv)
        if idx.size > per_chip:
            idx = rng.choice(idx, size=per_chip, replace=False)
        ps.append(prob.ravel()[idx].astype("float64"))
        ys.append((y.ravel()[idx] == 1).astype("float64"))
        prog.tick()
    return (np.concatenate(ps), np.concatenate(ys)) if ps else (np.zeros(0), np.zeros(0))


def plot(path, title, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=150)
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", label="perfect calibration")
    for label, rel, style in curves:
        ok = rel["pixels"] > 0
        ax.plot(rel["mean_prob"][ok], rel["water_share"][ok], style, lw=1.6, ms=4,
                label=f"{label} (ECE {rel['ece']:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted water probability (bin mean)")
    ax.set_ylabel("Share of pixels that are water")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_model(args, paths, index, model, trainset, tag=""):
    mdir = paths.model_dir(model, trainset, tag)
    if not (mdir / "model_meta.json").exists():
        print(f"  {model} {trainset}: not trained yet, skipped")
        return None
    out = paths.out / "models" / "calibration" / mdir.name
    if (out / "calibration.json").exists() and not args.force:
        print(f"  {model} {trainset}: already done, kept ({out.name}); --force redoes it")
        return mc.load_json(out / "calibration.json")["rows"]
    t0 = time.time()
    pred = mc.load_predictor(model, mdir, cpu_ok=args.cpu_ok, max_ram_gb=args.max_ram_gb,
                             tile=args.tile, overlap=args.overlap)
    rng = np.random.default_rng([args.seed, mc.MODELS.index(model), mc.TRAINSETS.index(trainset)])
    data = {}
    for split in SPLITS:
        rows = mc.split_rows(index, split)
        print(f"  {model} {trainset}, {split}: {len(rows)} chips", flush=True)
        data[split] = sample_split(pred, rows, paths.data, args.per_chip, rng)
    zv, yv = logit(data["valid"][0]), data["valid"][1]
    if zv.size == 0 or yv.min() == yv.max():
        print(f"  {model} {trainset}: valid sample holds one class only; skipped")
        return None
    T = fit_temperature(zv, yv)
    a, b = fit_platt(zv, yv)
    rows, rec = [], {"model": model, "trainset": trainset, "run_tag": tag, "model_dir": str(mdir),
                     "threshold_used_for_maps": pred.threshold, "temperature": T,
                     "platt_a": a, "platt_b": b, "fitted_on": "valid", "bins": BINS,
                     "pixels_per_chip": args.per_chip, "seed": args.seed}
    curves = []
    for split in SPLITS:
        p, y = data[split]
        if p.size == 0:
            continue
        z = logit(p)
        raw, tmp, plt_ = scores(p, y), scores(sigmoid(z / T), y), scores(sigmoid(a * z + b), y)
        rows.append({"model": model, "trainset": trainset + (f"_{tag}" if tag else ""),
                     "split": split, "pixels": int(p.size),
                     "water_share": round(float(y.mean()), 4),
                     "ece_raw": round(raw["ece"], 4), "ece_temp": round(tmp["ece"], 4),
                     "ece_platt": round(plt_["ece"], 4),
                     "brier_raw": round(raw["brier"], 4), "brier_platt": round(plt_["brier"], 4),
                     "logloss_raw": round(raw["logloss"], 4),
                     "logloss_platt": round(plt_["logloss"], 4),
                     "best_thr_raw": raw["best_thr"], "best_thr_platt": plt_["best_thr"],
                     "best_iou": round(raw["best_iou"], 4), "temperature": round(T, 4),
                     "platt_a": round(a, 4), "platt_b": round(b, 4)})
        rec[split] = {k: {q: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for q, v in s["rel"].items()} for k, s in
                      (("raw", raw), ("temperature", tmp), ("platt", plt_))}
        if split in ("valid", "test"):
            curves.append((f"{split}, as trained", raw["rel"], "o-" if split == "valid" else "s-"))
        if split == "test":
            curves.append(("test, after Platt (fitted on valid)", plt_["rel"], "^-"))
    out.mkdir(parents=True, exist_ok=True)
    plot(out / "reliability.png", f"{model} {trainset}: reliability on Sen1Floods11", curves)
    rec["rows"] = rows
    rec["seconds"] = round(time.time() - t0, 1)
    mc.save_json(out / "calibration.json", mc.run_record(args, paths, **rec))
    for r in rows:
        print(f"    {r['split']:7s} ECE {r['ece_raw']:.3f} -> {r['ece_platt']:.3f} (Platt), "
              f"{r['ece_temp']:.3f} (temperature); best threshold {r['best_thr_raw']:.2f} -> "
              f"{r['best_thr_platt']:.2f}; best IoU {r['best_iou']:.3f}")
    print(f"    T {T:.3f}; Platt a {a:.3f}, b {b:.3f}; wrote {out}", flush=True)
    return rows


def main(argv=None):
    mc.setup_console()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mc.add_common_args(ap, smoke=False)
    ap.add_argument("--models", default="unet,tm_s1,tm_s1t",
                    help="comma-separated, from rf, unet, tm_s1, tm_s1t (default the networks)")
    ap.add_argument("--trainsets", default="hand,hand_weak")
    ap.add_argument("--per-chip", type=int, default=20000,
                    help="scored pixels sampled per chip, at most (default 20000)")
    ap.add_argument("--cpu-ok", action="store_true", help="allow the networks on the CPU")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    mc.add_run_tag_arg(ap)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    trainsets = [t.strip() for t in args.trainsets.split(",") if t.strip()]
    bad = [m for m in models if m not in mc.MODELS] + [t for t in trainsets
                                                         if t not in mc.TRAINSETS]
    if bad:
        mc.fail(f"Unknown model or training set: {', '.join(bad)}. Models: "
                f"{', '.join(mc.MODELS)}; training sets: {', '.join(mc.TRAINSETS)}.")
    mc.check_tile_args(args.tile, args.overlap)
    tag = mc.check_run_tag(args, seed_used=False)
    sfx = f"_{tag}" if tag else ""
    paths = mc.paths_from_args(args)
    print("31_calibration.py: calibration of the water probabilities, Sen1Floods11 "
          f"({', '.join(models)}; {', '.join(trainsets)})")
    index = mc.load_chip_index(paths)
    allrows = []
    for model in models:
        for ts in trainsets:
            r = run_model(args, paths, index, model, ts, tag)
            allrows += r or []
    if not allrows:
        mc.fail("No trained model was found for these --models and --trainsets. Train them "
                "first (RUN ALL MODELS.bat).")
    paths.tables.mkdir(parents=True, exist_ok=True)
    with open(paths.tables / f"calibration_summary{sfx}.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_COLS)
        w.writeheader()
        w.writerows(allrows)
    lines = ["# Calibration of the water probabilities (Sen1Floods11)"
             + (f", repeat run {tag}" if tag else ""), "",
             "Rescalings fitted on the valid chips only; test and Bolivia show whether they "
             "hold on chips never used. ECE: expected calibration error, 15 equal bins "
             "(0 = perfect). Best IoU is the same before and after (the rescalings keep the "
             "order of the pixels); only the threshold that gives it moves. No map or score "
             "of Tables A and D changes.", "",
             "| Model | Trained on | Split | ECE as trained | ECE, temperature | ECE, Platt | "
             "Best threshold as trained | after Platt | Best IoU |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in allrows:
        lines.append(f"| {r['model']} | {r['trainset']} | {r['split']} | {r['ece_raw']:.3f} | "
                     f"{r['ece_temp']:.3f} | {r['ece_platt']:.3f} | {r['best_thr_raw']:.2f} | "
                     f"{r['best_thr_platt']:.2f} | {r['best_iou']:.3f} |")
    lines += ["", "Reliability diagrams: `06 Outputs\\models\\calibration\\<model>_<trainset>"
              f"{sfx}\\reliability.png`.", ""]
    (paths.tables / f"calibration_summary{sfx}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {paths.tables / f'calibration_summary{sfx}.md'}")
    print(f"PASS  calibration: {len(allrows)} rows")
    return 0


def selftest():
    import models_synthetic as syn
    print("31_calibration.py self-test")
    rng = np.random.default_rng(0)
    # 1. maths: a model whose logits are 3x too large is fixed by T = 3 and a = 1/3
    zt = rng.normal(0, 2, 200000)
    y = (rng.random(zt.size) < sigmoid(zt)).astype("float64")
    T = fit_temperature(3 * zt, y)
    a, b = fit_platt(3 * zt - 1.0, y)
    p_bad, p_good = sigmoid(3 * zt), sigmoid(zt)
    ece_bad, ece_good = reliability(p_bad, y)["ece"], reliability(p_good, y)["ece"]
    same_iou = abs(best_threshold(p_bad, y)[1] - best_threshold(sigmoid(3 * zt / T), y)[1]) < 0.01
    with tempfile.TemporaryDirectory(prefix="fyp selftest ", ignore_cleanup_errors=True) as td:
        proj = Path(td) / "Final Year Project"
        syn.build_project(proj, corridor=False)
        ok = mc.run_script("21_prepare_chips.py", ["--project", str(proj)]) == 0
        ok &= mc.run_script("23_train_unet.py", ["--project", str(proj), "--base-width", "8",
                                                 "--crop", "32", "--batch-size", "4",
                                                 "--steps-per-epoch", "3", "--epochs", "2",
                                                 "--tile", "64", "--overlap", "16",
                                                 "--workers", "0", "--cpu-ok"]) == 0
        ok &= main(["--project", str(proj), "--cpu-ok", "--tile", "64", "--overlap", "16",
                    "--per-chip", "500"]) == 0
        paths = mc.Paths(proj)
        out = paths.out / "models" / "calibration" / "unet_hand"
        files = (out / "calibration.json").exists() and (out / "reliability.png").exists() and \
            (paths.tables / "calibration_summary.md").exists()
        with open(paths.tables / "calibration_summary.csv", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        again = main(["--project", str(proj), "--cpu-ok", "--tile", "64", "--overlap", "16"]) == 0
    checks = [("temperature finds the 3x scale", abs(T - 3) < 0.15),
              ("Platt finds a = 1/3, b = 1/3", abs(a - 1 / 3) < 0.02 and abs(b - 1 / 3) < 0.05),
              ("ECE: badly scaled high, well scaled near 0", ece_bad > 0.05 > ece_good),
              ("rescaling keeps the best IoU", same_iou),
              ("tiny U-Net: files written", ok and files),
              ("rows for valid, test, Bolivia; models not trained skipped", len(rows) == 3),
              ("a second run keeps what is done", again)]
    for name, c in checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {name}")
    good = all(c for _, c in checks)
    print("SELFTEST PASS" if good else "SELFTEST FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(mc.cli(main))
