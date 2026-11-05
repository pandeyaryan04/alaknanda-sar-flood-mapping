"""Jal-SAR extensions: why the Random Forest's SLOPE SHAP sign is positive (M4), against physics.

M4 found that a steeper cell pushes the forest's water probability UP (Spearman +0.874
between SLOPE and its SHAP value). Physics expects the opposite. This script tests the
three explanations that fit what is known, without changing any result:

  1. Out of range. The forest learnt from Sen1Floods11 (flat floodplains: training SLOPE
     mean 2.9 degrees, 99.5th percentile 33.5), but half the explained corridor cells are
     steeper than 18 degrees. Beyond the training data a forest repeats whatever its few
     steep training pixels said. Test: the share of the corridor cells beyond the training
     percentiles.
  2. Learnt from the training labels. In flat floodplain chips the few steep pixels may sit
     on river banks, embankments or DEM steps next to water, so steep and wet go together
     there. Test: water share of the Sen1Floods11 training pixels by slope class, overall
     and with HAND under 3 m.
  3. A SHAP artefact. Interventional SHAP (M4) swaps a cell's SLOPE for background values
     while keeping its own HAND, so it asks the forest about low-HAND, steep combinations
     that hardly exist (SLOPE and HAND correlate at rho 0.66 in the explained cells). Tests:
     (a) the forest's response to SLOPE with the other three inputs held at each cell's own
     values (individual conditional expectation, averaged by HAND class); (b) SHAP again on
     a subsample with a background drawn from cells of the same HAND class (closer to a
     conditional explanation). If the sign turns negative in (b) but not in M4, the M4
     sign is mostly the background; if (a) rises with slope, the forest itself does it.

Reads: 06 Outputs\\jalsar\\extensions\\m4_shap_rf\\shap_values.csv (the M4 cells and inputs), the
rf_hand forest, and the hand training chips (through the chip index).
Writes: 06 Outputs\\jalsar\\extensions\\m4_shap_rf\\slope_check\\ (M4_SLOPE_CHECK.md first, then
training_slope_water.csv, ice_slope.csv, shap_matched.csv, fig_*.png, run_record.json).
Nothing in M4 is overwritten.

Run:  python m4_slope_check.py              (about 10 to 30 minutes)
      python m4_slope_check.py --smoke      (a few minutes)
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "FYP_code"))
import models_common as mc  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FEATS = list(mc.CHANNELS)                      # VV, VH, HAND, SLOPE
SLOPE_BINS = [0, 2, 5, 10, 15, 20, 25, 30, 90]
HAND_BINS = [0, 1, 3, 10, 30, 1e9]
HAND_NAMES = ["0-1 m", "1-3 m", "3-10 m", "10-30 m", "30 m and more"]
SLOPE_GRID = [0, 1, 2, 4, 6, 8, 10, 12, 15, 18, 21, 24, 27, 30, 33.4]


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def read_m4(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    X = np.array([[float(r[k]) for k in FEATS] for r in rows], "float32")
    S = np.array([[float(r[f"shap_{k}"]) for k in FEATS] for r in rows], "float32")
    return rows, X, S


def training_pixels(paths, n_per_chip, rng):
    """(VV, VH, HAND, SLOPE, label) of up to n_per_chip training-valid pixels per hand train chip."""
    index = mc.load_chip_index(paths)
    hand, _ = mc.training_rows(index, "hand")
    out = []
    for r in hand:
        x, y = mc.read_chip(r, paths.data)
        ok = mc.train_valid(x, y) & np.isfinite(x[2]) & np.isfinite(x[3])
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            continue
        if idx.size > n_per_chip:
            idx = rng.choice(idx, size=n_per_chip, replace=False)
        out.append(np.column_stack([x.reshape(4, -1)[:, idx].T, y.reshape(-1)[idx]]))
    return np.concatenate(out) if out else np.zeros((0, 5)), len(hand)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-matched", type=int, default=600, help="cells re-explained with a matched background")
    ap.add_argument("--background", type=int, default=60, help="background cells per HAND class")
    ap.add_argument("--n-per-chip", type=int, default=4000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_matched, args.background, args.n_per_chip = 60, 12, 300
    t0 = time.time()
    paths = mc.Paths()
    m4 = paths.o06 / "jalsar" / "extensions" / "m4_shap_rf"
    out = m4 / ("slope_check_smoke" if args.smoke else "slope_check")
    out.mkdir(parents=True, exist_ok=True)
    if not (m4 / "shap_values.csv").exists():
        print(f"PROBLEM: {m4 / 'shap_values.csv'} is missing (run m4_shap_rf.py first)")
        return 1
    rows, X, S = read_m4(m4 / "shap_values.csv")
    rng = np.random.default_rng([mc.SEED, 9])
    rf = mc.RFPredictor(paths.model_dir("rf", "hand"))
    st = rf.stats
    j_s, j_h = FEATS.index("SLOPE"), FEATS.index("HAND")
    print(f"m4_slope_check.py: {len(X):,} M4 cells; forest threshold {rf.threshold}")

    # 1. out of range
    lo_s, hi_s, mean_s, sd_s = st["clip_lo"][j_s], st["clip_hi"][j_s], st["mean"][j_s], st["std"][j_s]
    sl = X[:, j_s]
    share_above = {k: float((sl > v).mean()) for k, v in
                   (("mean_plus_2sd", mean_s + 2 * sd_s), ("mean_plus_3sd", mean_s + 3 * sd_s),
                    ("at_clip_top", hi_s - 1e-3))}

    # 2. training labels by slope class
    print("  reading the hand training chips ...")
    tp, n_chips = training_pixels(paths, args.n_per_chip, rng)
    tr_rows = []
    if len(tp):
        ts, th, ty = tp[:, 3], tp[:, 2], tp[:, 4]
        for sub, msk in (("all training pixels", np.ones(len(tp), bool)), ("HAND under 3 m", th < 3)):
            for a, b in zip(SLOPE_BINS[:-1], SLOPE_BINS[1:]):
                m = msk & (ts >= a) & (ts < b)
                tr_rows.append({"subset": sub, "slope_from": a, "slope_to": b, "pixels": int(m.sum()),
                                "share_of_subset": round(float(m.sum() / max(msk.sum(), 1)), 5),
                                "water_share": round(float(ty[m].mean()), 4) if m.any() else None})
        rho_tr = spearman(ts, ty) if len(tp) > 10 else float("nan")
    else:
        rho_tr = float("nan")
    with open(out / "training_slope_water.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tr_rows[0].keys()) if tr_rows else ["subset"])
        w.writeheader(); w.writerows(tr_rows)

    # 3a. individual conditional expectation: each cell's own VV, VH, HAND; SLOPE swept
    print("  forest response to slope, other inputs fixed ...")
    ice = np.zeros((len(X), len(SLOPE_GRID)), "float32")
    for k, s in enumerate(SLOPE_GRID):
        Xk = X.copy()
        Xk[:, j_s] = np.clip(s, lo_s, hi_s)
        ice[:, k] = rf.prob_features(np.ascontiguousarray(Xk))
    hcls = np.digitize(X[:, j_h], HAND_BINS[1:-1])
    ice_rows = []
    for c, name in enumerate(HAND_NAMES):
        m = hcls == c
        if m.any():
            mean = ice[m].mean(0)
            ice_rows.append({"hand_class": name, "cells": int(m.sum()),
                             **{f"slope_{s:g}": round(float(v), 4) for s, v in zip(SLOPE_GRID, mean)}})
    with open(out / "ice_slope.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ice_rows[0].keys()))
        w.writeheader(); w.writerows(ice_rows)
    # slope of each cell's own curve over the range seen in training (0 to 10 degrees) and beyond
    g = np.array(SLOPE_GRID)
    rise_in = float(np.mean(ice[:, g <= 10][:, -1] - ice[:, 0]))
    rise_out = float(np.mean(ice[:, -1] - ice[:, g <= 10][:, -1]))

    # 3b. SHAP with a background from the same HAND class
    print("  SHAP again with a HAND-matched background ...")
    import shap
    pick = np.sort(rng.choice(len(X), size=min(args.n_matched, len(X)), replace=False))
    Sm = np.full((len(pick), len(FEATS)), np.nan, "float32")
    for c in range(len(HAND_NAMES)):
        sel = pick[hcls[pick] == c]
        pool = np.flatnonzero(hcls == c)
        if sel.size == 0 or pool.size < 5:
            continue
        bg = X[rng.choice(pool, size=min(args.background, pool.size), replace=False)]
        ex = shap.explainers.Exact(lambda a: rf.prob_features(np.ascontiguousarray(a, dtype="float32")),
                                   shap.maskers.Independent(bg, max_samples=len(bg)))
        e = ex(X[sel])
        Sm[np.searchsorted(pick, sel)] = e.values
    ok = np.isfinite(Sm[:, j_s])
    rho_m4_sub = spearman(X[pick, j_s], S[pick, j_s])
    rho_matched = spearman(X[pick][ok, j_s], Sm[ok, j_s]) if ok.sum() > 10 else float("nan")
    rho_matched_by = {}
    for c, name in enumerate(HAND_NAMES):
        mm = ok & (hcls[pick] == c)
        if mm.sum() > 10:
            rho_matched_by[name] = round(spearman(X[pick][mm, j_s], Sm[mm, j_s]), 3)
    with open(out / "shap_matched.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["m4_row"] + FEATS + ["hand_class"] + [f"shap_m4_{k}" for k in FEATS] + [f"shap_matched_{k}" for k in FEATS])
        for i, p in enumerate(pick):
            w.writerow([int(p)] + [round(float(v), 4) for v in X[p]] + [HAND_NAMES[hcls[p]]]
                       + [round(float(v), 5) for v in S[p]] + [None if not np.isfinite(v) else round(float(v), 5) for v in Sm[i]])

    # figures
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for r in ice_rows:
        ax.plot(SLOPE_GRID, [r[f"slope_{s:g}"] for s in SLOPE_GRID], marker="o", ms=3,
                label=f"HAND {r['hand_class']} ({r['cells']})")
    ax.axvspan(mean_s + 2 * sd_s, SLOPE_GRID[-1], color="#e5e7eb", zorder=0)
    ax.text(mean_s + 2 * sd_s + 0.3, ax.get_ylim()[1] * 0.95, "beyond most training pixels", fontsize=7, va="top")
    ax.set_xlabel("SLOPE set to, degrees (other inputs: the cell's own)")
    ax.set_ylabel("mean water probability")
    ax.set_title("Random Forest response to slope, by HAND class", fontsize=9)
    ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(out / "fig_ice_slope.png", dpi=180); plt.close(fig)
    if tr_rows:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        for sub, col in (("all training pixels", "#1f2937"), ("HAND under 3 m", "#2563eb")):
            rr = [r for r in tr_rows if r["subset"] == sub and r["water_share"] is not None]
            ax.plot([(r["slope_from"] + min(r["slope_to"], 35)) / 2 for r in rr], [r["water_share"] for r in rr],
                    marker="o", color=col, label=sub)
        ax.set_xlabel("slope class, degrees"); ax.set_ylabel("share labelled water")
        ax.set_title("Sen1Floods11 hand training pixels: water share by slope", fontsize=9)
        ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "fig_training_slope.png", dpi=180); plt.close(fig)

    # verdicts, worded from the numbers
    f3 = lambda v: "n/a" if v is None or not np.isfinite(v) else f"{v:.3f}"  # noqa: E731
    tr_all = [r for r in tr_rows if r["subset"] == "all training pixels" and r["water_share"] is not None]
    steep_tr = [r for r in tr_all if r["slope_from"] >= 10]
    flat_tr = [r for r in tr_all if r["slope_to"] <= 5]
    wet_flat = np.average([r["water_share"] for r in flat_tr], weights=[r["pixels"] for r in flat_tr]) if flat_tr else np.nan
    wet_steep = np.average([r["water_share"] for r in steep_tr], weights=[r["pixels"] for r in steep_tr]) if steep_tr else np.nan
    v1 = (f"{share_above['mean_plus_2sd'] * 100:.0f} % of the explained cells are steeper than the training mean plus two "
          f"standard deviations ({mean_s + 2 * sd_s:.1f} degrees); {share_above['at_clip_top'] * 100:.0f} % sit on the "
          f"clip value ({hi_s:.1f} degrees), so for them the forest cannot tell any steeper slope apart.")
    v2 = (f"Training water share: {f3(wet_flat)} below 5 degrees, {f3(wet_steep)} at 10 degrees and above "
          f"(Spearman slope vs label {f3(rho_tr)}). "
          + ("Steep training pixels are wetter, so the forest learnt steep-and-wet from Sen1Floods11."
             if np.isfinite(wet_flat) and np.isfinite(wet_steep) and wet_steep > wet_flat else
             "Steep training pixels are not wetter, so the training labels do not explain the sign."
             if np.isfinite(wet_flat) and np.isfinite(wet_steep) else "No training pixels read."))
    v3a = (f"Holding each cell's VV, VH and HAND, moving SLOPE from 0 to 10 degrees changes the water probability by "
           f"{rise_in:+.3f} on average, and from 10 to {SLOPE_GRID[-1]:g} degrees by {rise_out:+.3f}. "
           + ("The forest itself raises the probability with slope." if rise_in + rise_out > 0.01 else
              "The forest itself does not raise the probability with slope." if rise_in + rise_out < -0.01 else
              "The forest is almost flat in slope."))
    v3b = (f"On the same {len(pick)} cells, Spearman between SLOPE and its SHAP value is {f3(rho_m4_sub)} with the M4 "
           f"background and {f3(rho_matched)} with a HAND-matched background"
           + (f" (by HAND class: {', '.join(f'{k} {v:+.2f}' for k, v in rho_matched_by.items())})" if rho_matched_by else "")
           + ". " + ("The sign turns negative with the matched background, so most of the M4 sign comes from "
                     "explaining against an unrealistic (steep, high-HAND) background."
                     if np.isfinite(rho_matched) and rho_matched < 0 < rho_m4_sub else
                     "The sign stays positive with the matched background, so it is not mainly a background artefact."
                     if np.isfinite(rho_matched) else ""))
    md = ["# Why the SLOPE SHAP sign is positive in M4", "",
          "Read with M4_SHAP_RF.md. Nothing in M4 changes; this only explains its SLOPE row. Every sentence "
          "below is worded from the numbers this run produced; check them before quoting.", "",
          "## 1. Out of the training range", "",
          f"Training SLOPE (Sen1Floods11 hand train chips, norm_stats.json): mean {mean_s:.2f}, standard deviation "
          f"{sd_s:.2f}, clipped to {lo_s:.1f} to {hi_s:.1f} degrees (0.5 and 99.5 percentiles). Explained corridor "
          f"cells: median {np.median(sl):.1f}, quarter at or above {np.percentile(sl, 75):.1f} degrees.", "", v1, "",
          "## 2. What the training labels say", "",
          f"{len(tp):,} pixels from {n_chips} hand training chips (up to {args.n_per_chip} each).", "",
          "| Subset | Slope class (degrees) | Pixels | Share of subset | Labelled water |", "|---|---|---|---|---|"]
    for r in tr_rows:
        md.append(f"| {r['subset']} | {r['slope_from']} to {r['slope_to']} | {r['pixels']:,} | "
                  f"{r['share_of_subset'] * 100:.2f} % | {f3(r['water_share'])} |")
    md += ["", v2, "", "## 3a. The forest's own response to slope", "",
           "Mean water probability with SLOPE set to each value and VV, VH and HAND kept at each cell's own values.", "",
           "| HAND class | Cells | " + " | ".join(f"{s:g}" for s in SLOPE_GRID) + " |",
           "|---|---|" + "---|" * len(SLOPE_GRID)]
    for r in ice_rows:
        md.append(f"| {r['hand_class']} | {r['cells']} | " + " | ".join(f"{r[f'slope_{s:g}']:.2f}" for s in SLOPE_GRID) + " |")
    md += ["", v3a, "", "## 3b. SHAP with a background of the same HAND class", "", v3b, "",
           "## What to write", "",
           "Use these results to choose the sentence for the report; nothing is written about the sign until you "
           "have read them. If 1 and 2 hold, the honest wording is that the forest was trained on flat "
           "floodplains and its slope response in steep Himalayan terrain is outside what it learnt, so the SLOPE "
           "SHAP sign describes the model, not the river. If 3b turns negative, add that the M4 sign is partly an "
           "effect of the explanation method.", "",
           "Figures: fig_ice_slope.png, fig_training_slope.png.", ""]
    (out / "M4_SLOPE_CHECK.md").write_text("\n".join(md), encoding="utf-8")
    mc.save_json(out / "run_record.json", {"script": "m4_slope_check.py", "smoke": args.smoke, "cells": len(X),
                                           "matched_cells": int(len(pick)), "background_per_class": args.background,
                                           "training_pixels": int(len(tp)), "training_chips": n_chips,
                                           "share_above": share_above, "rho_training": rho_tr,
                                           "ice_rise_0_10": rise_in, "ice_rise_10_top": rise_out,
                                           "rho_m4_subsample": rho_m4_sub, "rho_matched": rho_matched,
                                           "rho_matched_by_hand": rho_matched_by,
                                           "minutes": round((time.time() - t0) / 60, 1)})
    print(f"  {v1}\n  {v2}\n  {v3a}\n  {v3b}")
    print(f"  wrote {out / 'M4_SLOPE_CHECK.md'} ({(time.time() - t0) / 60:.1f} min)")
    print("PASS  slope check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
