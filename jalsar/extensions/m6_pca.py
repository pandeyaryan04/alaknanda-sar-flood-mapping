"""Jal-SAR extensions, item M6: principal component analysis of the four model inputs.

Part 1, the inputs explained. The same Sen1Floods11 training pixels the
Random Forest rf_hand learns from (same chips, same per-chip quota, same seed,
the code of 22_train_rf.py), clipped and standardised exactly as the models
see them. PCA on them gives how much of the variation each component carries
and how each component mixes VV, VH, HAND and SLOPE.

Part 2, does PCA help the Random Forest? Two forests are trained on the
components, with the settings of rf_hand (300 trees, min_samples_leaf 5,
max_features sqrt, seed 20261007): one on the fewest components that cover 95
per cent of the variation, one on all four (a pure rotation; a forest is not
rotation invariant, so this can differ from rf_hand). Each gets its threshold
on the Sen1Floods11 valid chips, is scored on test and Bolivia, predicts both
corridor scenes in the flood-map format, and is scored with
04_stratified_metrics.py, like every other map. (If 95 per cent needs all four,
a two-component forest is added instead.)

U-Net and TerraMind are not retrained on components: TerraMind was pretrained
on real radar bands, so components mean nothing to it. The report says so.

Outputs:
  06 Outputs\\jalsar\\extensions\\m6_pca\\        M6_PCA.md (read first), pca_*.csv, fig_*.png, run_record.json
  04 Code\\FYP_code\\data\\models\\rf_pca<k>_hand\\   the forests and the PCA used
  04 Code\\FYP_code\\data\\flood_<year>_rf_pca<k>_hand\\  corridor flood maps
  06 Outputs\\tables_m1\\<year>_rf_pca<k>_hand\\   their scores
--smoke: 20 chips, 20 trees, everything into 06 Outputs\\jalsar\\extensions\\m6_pca_smoke\\.

Run:  python m6_pca.py --smoke   (a few minutes)
      python m6_pca.py           (about 1.5 to 3 hours)
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1] / "FYP_code"
sys.path.insert(0, str(CODE))
import models_common as mc  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FEATS = list(mc.CHANNELS)


def standardise(f, stats):
    """Clipped raw features (rf_features) -> the models' standardised values."""
    return (f - np.asarray(stats["mean"], "float32")) / np.asarray(stats["std"], "float32")


def scores(hist, thr):
    tp, fp, fn, tn = hist.counts(thr)
    m = mc.metrics_from_confusion(tp, fp, fn, tn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "iou_csi": round(m["iou"], 4), "f1": round(m["f1"], 4),
            "hit_rate": round(m["recall"], 4),
            "far": round(fp / (tp + fp), 4) if tp + fp else float("nan")}


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def pooled(table_dir):
    f = Path(table_dir) / "metrics_by_stratum.csv"
    if not f.exists():
        return None
    out = {}
    for r in read_rows(f):
        if r["table"] == "reported" and r["dimension"] in ("ALL", "reach"):
            out[r["stratum"]] = r
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trainset", default="hand", choices=("hand",))
    ap.add_argument("--n-per-class", type=int, default=1000)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--min-samples-leaf", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--max-ram-gb", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=mc.SEED)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_estimators, args.n_per_class = 20, 200
    import joblib
    import sklearn
    from sklearn.decomposition import PCA
    t0 = time.time()
    tr = mc.load_script("22_train_rf.py")
    paths = mc.Paths()
    out = paths.o06 / "jalsar" / "extensions" / ("m6_pca_smoke" if args.smoke else "m6_pca")
    out.mkdir(parents=True, exist_ok=True)
    print(f"m6_pca.py: PCA of the inputs and Random Forests on the components{' (smoke)' if args.smoke else ''}")

    index = mc.load_chip_index(paths)
    stats = mc.load_norm_stats(paths.norm_stats)
    hand, _ = mc.training_rows(index, args.trainset)
    valid_rows = mc.split_rows(index, "valid")
    test_rows = mc.split_rows(index, "test")
    bol_rows = mc.split_rows(index, "bolivia")
    if args.smoke:
        hand, valid_rows, test_rows, bol_rows = hand[:20], valid_rows[:6], test_rows[:6], bol_rows[:3]
    cap = mc.rf_row_cap(args.max_ram_gb, args.n_estimators, args.min_samples_leaf)
    qh, _ = tr.quotas(hand, [], args.n_per_class, cap, 0.5)
    print(f"  {len(hand)} hand train chips, up to {qh} pixels per class per chip (as rf_hand)")
    prog = mc.Progress(len(hand), "chips sampled", every=120)
    fh, lh, _ = tr.sample_rows(hand, qh, paths.data, stats, args.seed, 0, prog)
    X = mc.rf_features(np.concatenate(fh), stats)
    y = np.concatenate(lh).astype("int8")
    Z = standardise(X, stats)
    print(f"  {len(y):,} training pixels, water {int((y == 1).sum()):,}")

    # ---- Part 1: PCA of the inputs ----------------------------------------
    pca = PCA(n_components=4).fit(Z)
    var = pca.explained_variance_ratio_
    cum = np.cumsum(var)
    k95 = int(np.searchsorted(cum, 0.95) + 1)
    corr = np.corrcoef(Z.T)
    load = pca.components_                                 # (component, input)
    with open(out / "pca_variance.csv", "w", newline="", encoding="utf-8") as fh_:
        w = csv.writer(fh_)
        w.writerow(["component", "variance_share", "cumulative_share", "eigenvalue"])
        for i in range(4):
            w.writerow([f"PC{i + 1}", round(float(var[i]), 4), round(float(cum[i]), 4),
                        round(float(pca.explained_variance_[i]), 4)])
    with open(out / "pca_loadings.csv", "w", newline="", encoding="utf-8") as fh_:
        w = csv.writer(fh_)
        w.writerow(["component"] + FEATS)
        for i in range(4):
            w.writerow([f"PC{i + 1}"] + [round(float(v), 4) for v in load[i]])
    with open(out / "input_correlation.csv", "w", newline="", encoding="utf-8") as fh_:
        w = csv.writer(fh_)
        w.writerow([""] + FEATS)
        for i, k in enumerate(FEATS):
            w.writerow([k] + [round(float(v), 4) for v in corr[i]])
    # Mean component score of water and not-water pixels (does a component separate them?)
    P = pca.transform(Z)
    sep = []
    for i in range(4):
        a, b = P[y == 1, i], P[y == 0, i]
        sd = np.sqrt((a.var() + b.var()) / 2) or 1.0
        sep.append(round(float((a.mean() - b.mean()) / sd), 3))

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.bar([f"PC{i + 1}" for i in range(4)], var * 100, color="#777777", label="each component")
    ax.plot([f"PC{i + 1}" for i in range(4)], cum * 100, "o-", color="black", label="cumulative")
    ax.axhline(95, ls="--", lw=0.8, color="#c51b7d")
    ax.set_ylabel("share of variation (%)"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_pca_variance.png", dpi=200); plt.close(fig)
    for name, M, rows_, title in (("fig_pca_loadings.png", load, [f"PC{i + 1}" for i in range(4)], "Loadings"),
                                  ("fig_input_correlation.png", corr, FEATS, "Correlation between inputs")):
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(4)); ax.set_xticklabels(FEATS); ax.set_yticks(range(4)); ax.set_yticklabels(rows_)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(title); fig.colorbar(im, fraction=0.046)
        fig.tight_layout(); fig.savefig(out / name, dpi=200); plt.close(fig)
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(y), size=min(20000, len(y)), replace=False)
    fig, ax = plt.subplots(figsize=(5, 4))
    for cls, col, lab in ((0, "#bbbbbb", "not water"), (1, "#2166ac", "water")):
        m = y[pick] == cls
        ax.scatter(P[pick][m, 0], P[pick][m, 1], s=2, alpha=0.4, color=col, label=lab)
    ax.set_xlabel(f"PC1 ({var[0] * 100:.1f} %)"); ax.set_ylabel(f"PC2 ({var[1] * 100:.1f} %)")
    ax.legend(frameon=False, markerscale=4)
    fig.tight_layout(); fig.savefig(out / "fig_pc1_pc2.png", dpi=200); plt.close(fig)
    print(f"  PCA: variance {', '.join(f'{v * 100:.1f}' for v in var)} %; 95 % needs {k95} component(s)")

    # ---- Part 2: forests on the components ---------------------------------
    variants = sorted({k95, 4})
    if variants == [4]:
        variants = [2, 4]
    ns = argparse.Namespace(n_estimators=args.n_estimators, min_samples_leaf=args.min_samples_leaf,
                            n_jobs=args.n_jobs, seed=args.seed)
    results = []
    for k in variants:
        name = f"rf_pca{k}"
        print(f"\n  forest on {k} component(s): fitting {args.n_estimators} trees ...", flush=True)
        t1 = time.time()
        clf = tr.fit_forest(np.ascontiguousarray(P[:, :k], dtype="float32"), y, ns)
        fit_min = (time.time() - t1) / 60

        def pred(f, clf=clf, k=k):
            f = np.ascontiguousarray(pca.transform(standardise(f, stats))[:, :k], dtype="float32")
            o = np.empty(len(f), "float32")
            for i in range(0, len(f), mc.CHUNK):
                o[i:i + mc.CHUNK] = mc.rf_water_prob(clf, f[i:i + mc.CHUNK], args.n_jobs)
            return o

        thr, iou_v = tr.valid_threshold(pred, valid_rows, paths.data, stats).best()
        s_test = scores(tr.valid_threshold(pred, test_rows, paths.data, stats), thr)
        s_bol = scores(tr.valid_threshold(pred, bol_rows, paths.data, stats), thr)
        print(f"    threshold {thr:.2f} (valid IoU {iou_v:.4f}); test IoU {s_test['iou_csi']}, "
              f"Bolivia IoU {s_bol['iou_csi']}")
        res = {"model": name, "components": k, "threshold": thr, "valid_iou": round(iou_v, 4),
               "test": s_test, "bolivia": s_bol, "fit_minutes": round(fit_min, 1), "corridor": {}}
        if not args.smoke:
            md_dir = paths.models / f"{name}_{args.trainset}"
            md_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump({"forest": clf, "pca_mean": pca.mean_, "pca_components": pca.components_,
                         "k": k, "norm_stats": stats}, md_dir / "model_pca.joblib", compress=3)
            mc.save_json(md_dir / "model_meta.json", {"model": name, "trainset": args.trainset, "k": k,
                         "threshold_valid": thr, "iou_valid_at_threshold": iou_v,
                         "sklearn_version": sklearn.__version__, "n_estimators": args.n_estimators,
                         "min_samples_leaf": args.min_samples_leaf, "seed": args.seed,
                         "note": "Stage M6. Inputs: rf_features -> standardise -> PCA (first k)."})
        for year in mc.YEARS:
            inp = mc.corridor_inputs(paths, year)
            prob = np.full(inp["valid"].shape, np.nan, "float32")
            prob[inp["valid"]] = pred(mc.rf_features(inp["x"][:, inp["valid"]].T, stats))
            fmap, diag = mc.make_flood_map(prob, inp["valid"], thr, inp["perm"], inp["distort"])
            fname = f"flood_{year}_{name}_{args.trainset}"
            mdir = (out / "maps" / fname) if args.smoke else (paths.data / fname)
            mc.write_flood_map(mdir / f"{fname}.tif", fmap, inp["profile"])
            diag["flood_km2"] = round(diag["flood_cells"] * mc.cell_km2(inp["profile"]), 3)
            mc.save_json(mdir / f"{fname}_diagnostics.json", diag)
            tdir = (out / "tables" if args.smoke else paths.o06 / "tables_m1") / f"{year}_{name}_{args.trainset}"
            tdir.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "04_stratified_metrics.py", "--pred", str(mdir / f"{fname}.tif"),
                   "--truth", f"data/reference_{year}.tif", "--strata", f"data/strata_{year}_v3",
                   "--windows", f"data/reference_windows_{year}/reference_windows.geojson",
                   "--label", f"Stage M6, {name} {args.trainset}, {year}", "--out", str(tdir)]
            with open(tdir / "run_log.txt", "w", encoding="utf-8") as log:
                rc = subprocess.run(cmd, cwd=CODE, stdout=log, stderr=subprocess.STDOUT).returncode
            res["corridor"][year] = {"flood_km2": diag["flood_km2"], "scored": rc == 0,
                                     "rows": pooled(tdir) if rc == 0 else None}
            print(f"    {year}: {diag['flood_km2']} km2 of flood; scoring {'done' if rc == 0 else 'FAILED'}")
            del inp, prob
        results.append(res)
        del clf

    # ---- Summary -----------------------------------------------------------
    base_rows = {}
    for year in mc.YEARS:
        for d in (paths.o06 / "tables_m1" / f"{year}_rf_{args.trainset}",
                  paths.o06 / "tables" / f"{year}_rf_{args.trainset}"):
            p = pooled(d)
            if p:
                base_rows[year] = p
                break
    base_s1 = {r["split"]: r for r in read_rows(paths.o06 / "tables" / f"s1f11_rf_{args.trainset}" / "metrics.csv")
               if r["threshold_kind"] == "valid_chosen"} if (paths.o06 / "tables" / f"s1f11_rf_{args.trainset}"
                                                             / "metrics.csv").exists() else {}

    def cell(r, k):
        return r.get(k, "") if r else ""

    md = [f"# M6: PCA of the model inputs{' (SMOKE RUN)' if args.smoke else ''}", "",
          f"Training pixels: {len(y):,} from {len(hand)} Sen1Floods11 hand train chips, the same chips, quota "
          f"and seed as rf_hand, clipped and standardised as the models see them.", "",
          "## How the inputs overlap", "",
          "| | " + " | ".join(FEATS) + " |", "|---|" + "---|" * 4]
    md += [f"| {k} | " + " | ".join(f"{corr[i, j]:.2f}" for j in range(4)) + " |" for i, k in enumerate(FEATS)]
    md += ["", "## Components", "",
           "| Component | Share of variation | Cumulative | " + " | ".join(f"{k} loading" for k in FEATS)
           + " | Water vs not water (standardised gap) |",
           "|---|---|---|" + "---|" * 4 + "---|"]
    md += [f"| PC{i + 1} | {var[i] * 100:.1f} % | {cum[i] * 100:.1f} % | "
           + " | ".join(f"{v:.2f}" for v in load[i]) + f" | {sep[i]} |" for i in range(4)]
    md += ["", f"95 per cent of the variation needs {k95} component(s).", "",
           "## Random Forests on the components, against rf_hand", "",
           "Sen1Floods11 scores are for water at each forest's own valid-chosen threshold. Corridor rows are "
           "Table A style: trained only on Sen1Floods11, both reaches pooled.", "",
           "| Model | Inputs | Threshold | S1F11 test IoU | Bolivia IoU | 2021 F1 | 2021 CSI | 2021 hit rate | "
           "2021 FAR | 2023 F1 | 2023 CSI | 2023 hit rate | 2023 FAR | Fit (min) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    b21, b23 = (base_rows.get(y_, {}).get("both reaches pooled") for y_ in mc.YEARS)

    def far(r):
        if not r:
            return ""
        if r.get("far"):
            return r["far"]
        return round(1 - float(r["precision"]), 4) if r.get("precision") not in (None, "", "nan") else ""

    md.append(f"| rf_{args.trainset} (original) | VV, VH, HAND, SLOPE | | {cell(base_s1.get('test'), 'iou')} | "
              f"{cell(base_s1.get('bolivia'), 'iou')} | {cell(b21, 'f1')} | {cell(b21, 'csi')} | {cell(b21, 'recall')} | "
              f"{far(b21)} | {cell(b23, 'f1')} | {cell(b23, 'csi')} | {cell(b23, 'recall')} | {far(b23)} | |")
    for r in results:
        c21 = (r["corridor"].get(2021, {}).get("rows") or {}).get("both reaches pooled")
        c23 = (r["corridor"].get(2023, {}).get("rows") or {}).get("both reaches pooled")
        md.append(f"| {r['model']} | first {r['components']} component(s) | {r['threshold']:.2f} | "
                  f"{r['test']['iou_csi']} | {r['bolivia']['iou_csi']} | {cell(c21, 'f1')} | {cell(c21, 'csi')} | "
                  f"{cell(c21, 'hit_rate')} | {cell(c21, 'far')} | {cell(c23, 'f1')} | {cell(c23, 'csi')} | "
                  f"{cell(c23, 'hit_rate')} | {cell(c23, 'far')} | {r['fit_minutes']} |")
    md += ["", "Reach rows and every terrain class are in each run's metrics_summary.md "
           "(06 Outputs\\tables_m1\\<year>_rf_pca<k>_hand\\). A 'no gain' result is still a result: with four "
           "inputs there is little for PCA to remove.", "",
           "Figures: fig_pca_variance.png, fig_pca_loadings.png, fig_input_correlation.png, fig_pc1_pc2.png."]
    (out / "M6_PCA.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    mc.save_json(out / "run_record.json", {"script": "m6_pca.py", "smoke": args.smoke, "pixels": int(len(y)),
                 "chips": len(hand), "quota": qh, "variance": var.tolist(), "loadings": load.tolist(),
                 "k95": k95, "variants": variants, "results": results, "seed": args.seed,
                 "sklearn_version": sklearn.__version__, "minutes": round((time.time() - t0) / 60, 1)})
    bad = [f"{r['model']} {y_}" for r in results for y_, c in r["corridor"].items() if not c["scored"]]
    if bad:
        print(f"PROBLEM: scoring failed for {', '.join(bad)} (see the run_log.txt in each table folder)")
        return 1
    print(f"\n  wrote {out}")
    print(f"PASS  M6 PCA{' (smoke)' if args.smoke else ''}: {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
