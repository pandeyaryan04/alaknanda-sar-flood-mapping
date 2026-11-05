"""Jal-SAR extensions, item M4: SHAP values for the Random Forest (a review suggestion).

What is explained: the Random Forest's water probability (rf_hand, the Table A
forest), for corridor cells inside the 2021 and 2023 reference windows.

Cells are sampled by outcome against the reference: hits, false alarms, misses
and correct dry cells, up to --n-per-outcome of each per year. Only cells the
model actually decided are used: permanent water and layover or shadow cells
are left out, because the flood map sets those to 0 whatever the model says.

Method: exact Shapley values (shap.explainers.Exact) with an independent
(interventional) background. With four inputs (VV, VH, HAND, SLOPE) every one
of the 2^4 = 16 input subsets is evaluated, so nothing is approximated except
the background, which is --background random corridor cells (half from each
year's scene). TreeSHAP was not used: the forest's 300 deep trees make it far
slower than this for four inputs. Base value + the four SHAP values = the
model's probability for that cell (checked and reported).

Outputs in 06 Outputs\\jalsar\\extensions\\m4_shap_rf\\ (smoke runs: m4_shap_rf_smoke\\):
  shap_values.csv        one row per explained cell
  importance.csv         mean |SHAP| per input, overall and by outcome, year, reach
  M4_SHAP_RF.md          the summary to read first
  fig_*.png              importance, beeswarm, dependence plots
  run_record.json        settings, counts, timings, versions

Run:  python m4_shap_rf.py --smoke     (a few minutes, checks everything works)
      python m4_shap_rf.py             (about 30 to 90 minutes)
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
UNITS = {"VV": "dB", "VH": "dB", "HAND": "m", "SLOPE": "degrees"}
OUTCOMES = {"hit": (1, 1), "false_alarm": (1, 0), "miss": (0, 1), "correct_dry": (0, 0)}
COLOURS = {"hit": "#1b7837", "false_alarm": "#c51b7d", "miss": "#2166ac", "correct_dry": "#999999"}


def reach_by_col(profile, strata_dir):
    """Gorge / open valley for every column, by the rule in 04_stratified_metrics.py."""
    from rasterio.warp import transform as warp_transform
    meta = json.loads((strata_dir / "strata_meta.json").read_text(encoding="utf-8"))
    lon = float(meta.get("reach_split_longitude", 78.5981))
    x_split = warp_transform("EPSG:4326", profile["crs"], [lon], [30.15])[0][0]
    t = profile["transform"]
    xs = t.c + t.a * (np.arange(profile["width"]) + 0.5)
    return np.where(xs >= x_split, "gorge", "open_valley")


def read_on_grid(path, shape):
    import rasterio
    with rasterio.open(path) as src:
        a = src.read(1)
    if a.shape != shape:
        mc.fail(f"{Path(path).name} is {a.shape}, the scene is {shape}. They must share one grid.")
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trainset", default="hand", choices=mc.TRAINSETS)
    ap.add_argument("--n-per-outcome", type=int, default=1000)
    ap.add_argument("--background", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=250, help="cells explained per call (progress)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_per_outcome, args.background, args.chunk = 15, 12, 30
    import shap
    import sklearn
    t0 = time.time()
    paths = mc.Paths()
    out = paths.o06 / "jalsar" / "extensions" / ("m4_shap_rf_smoke" if args.smoke else "m4_shap_rf")
    out.mkdir(parents=True, exist_ok=True)
    print(f"m4_shap_rf.py: SHAP for rf_{args.trainset}{' (smoke)' if args.smoke else ''}")

    rf = mc.RFPredictor(paths.model_dir("rf", args.trainset))
    print(f"  forest loaded: threshold {rf.threshold}, scikit-learn {sklearn.__version__}, shap {shap.__version__}")

    rows, bg_parts, counts = [], [], {}
    for year in mc.YEARS:
        inp = mc.corridor_inputs(paths, year)
        shape = inp["valid"].shape
        fmap = read_on_grid(paths.flood_map(year, "rf", args.trainset), shape)
        truth = read_on_grid(paths.reference(year), shape)
        decided = inp["valid"] & ~inp["perm"] & ~inp["distort"]
        scored = np.isin(fmap, (0, 1)) & np.isin(truth, (0, 1)) & decided
        reach = reach_by_col(inp["profile"], paths.strata(year))
        rng = np.random.default_rng([mc.SEED, year])
        counts[year] = {}
        for name, (p, t) in OUTCOMES.items():
            rr, cc = np.nonzero(scored & (fmap == p) & (truth == t))
            counts[year][name] = int(rr.size)
            if rr.size == 0:
                continue
            k = min(args.n_per_outcome, rr.size)
            pick = np.sort(rng.choice(rr.size, size=k, replace=False))
            for r, c in zip(rr[pick], cc[pick]):
                rows.append({"year": year, "reach": reach[c], "outcome": name, "row": int(r), "col": int(c)})
        print(f"  {year}: cells the model decided, by outcome: "
              + ", ".join(f"{n} {v:,}" for n, v in counts[year].items()))
        # Background: random decided cells of the whole corridor scene.
        br, bc = np.nonzero(decided)
        pick = rng.choice(br.size, size=args.background // 2, replace=False)
        bg_parts.append(mc.rf_features(inp["x"][:, br[pick], bc[pick]].T, rf.stats))
        # Inputs of the sampled cells, read while this year's scene is in memory.
        mine = [r for r in rows if r["year"] == year]
        xs = mc.rf_features(inp["x"][:, [r["row"] for r in mine], [r["col"] for r in mine]].T, rf.stats)
        for r, v in zip(mine, xs):
            r["x"] = v
        del inp

    X = np.stack([r["x"] for r in rows]).astype("float32")
    bg = np.concatenate(bg_parts).astype("float32")
    print(f"  explaining {len(X):,} cells against a background of {len(bg)} cells "
          f"({2 ** len(FEATS)} input subsets each)")

    def f(a):
        return rf.prob_features(np.ascontiguousarray(a, dtype="float32"))

    explainer = shap.explainers.Exact(f, shap.maskers.Independent(bg, max_samples=len(bg)))
    vals, base = np.zeros_like(X), np.zeros(len(X), "float64")
    t1 = time.time()
    for i in range(0, len(X), args.chunk):
        e = explainer(X[i:i + args.chunk])
        vals[i:i + args.chunk] = e.values
        base[i:i + args.chunk] = e.base_values
        done = min(i + args.chunk, len(X))
        el = time.time() - t1
        print(f"    {done:,}/{len(X):,} cells, {el / 60:.1f} min, about "
              f"{el / done * (len(X) - done) / 60:.1f} min left", flush=True)
    prob = f(X)
    add_err = float(np.max(np.abs(base + vals.sum(1) - prob)))
    print(f"  additivity check: base + SHAP equals the probability to within {add_err:.2e}")

    # One table, one row per cell.
    cols = ["year", "reach", "outcome", "row", "col"] + FEATS + ["prob", "base"] + [f"shap_{k}" for k in FEATS]
    with open(out / "shap_values.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r, x, p, b, s in zip(rows, X, prob, base, vals):
            w.writerow([r["year"], r["reach"], r["outcome"], r["row"], r["col"]]
                       + [round(float(v), 4) for v in x] + [round(float(p), 4), round(float(b), 4)]
                       + [round(float(v), 5) for v in s])

    # Mean |SHAP| by group.
    yr = np.array([r["year"] for r in rows])
    oc = np.array([r["outcome"] for r in rows])
    rc = np.array([r["reach"] for r in rows])
    groups = [("all", "all", np.ones(len(rows), bool))]
    groups += [("outcome", o, oc == o) for o in OUTCOMES]
    groups += [("year", str(y), yr == y) for y in mc.YEARS]
    groups += [("reach", g, rc == g) for g in ("gorge", "open_valley")]
    imp = []
    for kind, name, m in groups:
        if m.any():
            mabs = np.abs(vals[m]).mean(0)
            mean = vals[m].mean(0)
            imp.append({"group": kind, "value": name, "cells": int(m.sum()),
                        **{f"mean_abs_{k}": round(float(v), 4) for k, v in zip(FEATS, mabs)},
                        **{f"mean_{k}": round(float(v), 4) for k, v in zip(FEATS, mean)}})
    with open(out / "importance.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(imp[0].keys()))
        w.writeheader()
        w.writerows(imp)

    # Physics check: does each input push the way physics says it should?
    def spearman(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])
    rho = {k: round(spearman(X[:, j], vals[:, j]), 3) for j, k in enumerate(FEATS)}
    expect = {"VV": "negative (darker radar, more water)", "VH": "negative (darker radar, more water)",
              "HAND": "negative (lower above the river, more water)", "SLOPE": "negative (flatter, more water)"}

    # Figures.
    order = np.argsort(-np.abs(vals).mean(0))
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh([FEATS[j] for j in order][::-1], np.abs(vals).mean(0)[order][::-1], color="#555555")
    ax.set_xlabel("mean |SHAP| (change in water probability)")
    ax.set_title("Random Forest: what the map relies on")
    fig.tight_layout(); fig.savefig(out / "fig_importance.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    wbar = 0.2
    for i, o in enumerate(OUTCOMES):
        m = oc == o
        if m.any():
            ax.bar(np.arange(4) + (i - 1.5) * wbar, np.abs(vals[m]).mean(0), wbar, label=o.replace("_", " "),
                   color=COLOURS[o])
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(FEATS)
    ax.set_ylabel("mean |SHAP|"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("By outcome against the reference")
    fig.tight_layout(); fig.savefig(out / "fig_importance_by_outcome.png", dpi=200); plt.close(fig)

    try:
        expl = shap.Explanation(values=vals, base_values=base, data=X, feature_names=FEATS)
        shap.plots.beeswarm(expl, show=False)
        plt.gcf().set_size_inches(7, 3.8); plt.tight_layout()
        plt.savefig(out / "fig_beeswarm.png", dpi=200); plt.close("all")
    except Exception as exc:                     # a plotting problem must not lose the results
        print(f"  NOTE: beeswarm plot skipped ({exc})")

    for j, k in enumerate(FEATS):
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for o in OUTCOMES:
            m = oc == o
            if m.any():
                ax.scatter(X[m, j], vals[m, j], s=4, alpha=0.5, color=COLOURS[o], label=o.replace("_", " "))
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xlabel(f"{k} ({UNITS[k]})"); ax.set_ylabel(f"SHAP value of {k}")
        ax.legend(frameon=False, fontsize=8, markerscale=3)
        fig.tight_layout(); fig.savefig(out / f"fig_dependence_{k}.png", dpi=200); plt.close(fig)

    # Summary.
    def table(kind):
        lines = ["| Group | Cells | " + " | ".join(f"{k} mean abs SHAP" for k in FEATS) + " |",
                 "|---|---|" + "---|" * len(FEATS)]
        for r in imp:
            if r["group"] == kind:
                lines.append(f"| {r['value']} | {r['cells']} | "
                             + " | ".join(str(r[f'mean_abs_{k}']) for k in FEATS) + " |")
        return lines

    md = [f"# M4: SHAP for the Random Forest (rf_{args.trainset}){' (SMOKE RUN)' if args.smoke else ''}", "",
          "What is explained: the forest's water probability for corridor cells inside the 2021 and 2023 "
          "reference windows. Permanent water, layover and shadow cells are left out (the map sets them to 0 "
          "whatever the model says).", "",
          f"Method: exact Shapley values, all 16 subsets of the four inputs, against {len(bg)} random corridor "
          "cells as background (shap.explainers.Exact). A SHAP value is how much that input moved this cell's "
          "water probability away from the background average. Base value plus the four SHAP values equals "
          f"the probability, to within {add_err:.1e}.", "",
          f"Base value (average water probability of the background): {round(float(base.mean()), 4)}.", "",
          "Cells per outcome that the model decided (the whole windows), and how many were explained:", "",
          "| Year | " + " | ".join(o.replace('_', ' ') for o in OUTCOMES) + " |",
          "|---|" + "---|" * len(OUTCOMES)]
    for y in mc.YEARS:
        md.append(f"| {y} | " + " | ".join(
            f"{counts[y].get(o, 0):,} ({int(((yr == y) & (oc == o)).sum())})" for o in OUTCOMES) + " |")
    md += ["", "## Importance overall", ""] + table("all")
    md += ["", "## By outcome", ""] + table("outcome")
    md += ["", "## By year", ""] + table("year")
    md += ["", "## By reach", ""] + table("reach")
    md += ["", "## Physics check", "",
           "Spearman correlation between each input's value and its SHAP value (all explained cells):", "",
           "| Input | Correlation | Physics expects |", "|---|---|---|"]
    md += [f"| {k} | {rho[k]} | {expect[k]} |" for k in FEATS]
    md += ["", "SHAP explains the model, not the cause of the flood. A large SHAP value for an input means the "
           "forest leans on it, not that the input caused the water.", "",
           "Figures: fig_importance.png, fig_importance_by_outcome.png, fig_beeswarm.png, "
           "fig_dependence_<input>.png."]
    (out / "M4_SHAP_RF.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    mc.save_json(out / "run_record.json", {
        "script": "m4_shap_rf.py", "smoke": args.smoke, "trainset": args.trainset,
        "n_per_outcome": args.n_per_outcome, "background": len(bg), "cells": len(X),
        "counts_by_outcome": counts, "additivity_max_error": add_err, "spearman": rho,
        "threshold": rf.threshold, "model_dir": str(rf.model_dir), "seed": mc.SEED,
        "shap_version": shap.__version__, "sklearn_version": sklearn.__version__,
        "minutes": round((time.time() - t0) / 60, 1)})
    if add_err > 1e-3:
        print(f"PROBLEM: additivity error {add_err:.2e} is too large")
        return 1
    print(f"  wrote {out}")
    print(f"PASS  M4 SHAP Random Forest{' (smoke)' if args.smoke else ''}: {len(X):,} cells, "
          f"{(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
