"""Jal-SAR extensions, item M5: SHAP for the U-Net and both TerraMind models (needs the GPU).

TreeSHAP works only for tree models, so the networks use the SHAP library's
GradientExplainer (expected gradients): attributions are averaged over
--nsamples random points between the cell's inputs and background patches.
They are approximate; the sum of a cell's attributions comes close to, but is
not exactly, its probability minus the background average (checked and
reported). They are not directly the same numbers as the Random Forest's exact
Shapley values (M4), though both answer "which input pushed this cell towards
flood".

What is explained, for each model: the water probability of ONE cell, the
centre of a 224 x 224 patch around it (224 is TerraMind's training crop; the
U-Net gets the same patch so the two are comparable). Cells are sampled as in
M4: hits, false alarms, misses and correct dry cells of the model's own Table A
map against the reference, inside the windows, never permanent water, layover
or shadow; up to --n-per-outcome of each per year.

Two results per cell:
  - the attribution of each input (VV, VH, HAND, SLOPE), summed over the patch;
  - how much of the attribution comes from the cell itself and its close
    neighbours (the centre 5 x 5 cells, 50 m by 50 m) and how much from further
    away: the networks can use the neighbourhood, the Random Forest cannot.

Models: unet_hand, tm_s1t_hand (TerraMind with terrain, the Jal-SAR TerraMind),
tm_s1_hand (TerraMind radar only; it never uses HAND and SLOPE, so those are 0).

Outputs in 06 Outputs\\jalsar\\extensions\\m5_shap_networks\\ (smoke: m5_shap_networks_smoke\\):
  shap_cells.csv, importance.csv, M5_SHAP_NETWORKS.md (read first), fig_*.png,
  examples\\*.png (attribution maps around a few cells), run_record.json

Run:  python m5_shap_networks.py --smoke    (a few minutes)
      python m5_shap_networks.py            (about 1 to 2 hours on the RTX 3050)
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

FEATS = list(mc.CHANNELS)
OUTCOMES = {"hit": (1, 1), "false_alarm": (1, 0), "miss": (0, 1), "correct_dry": (0, 0)}
COLOURS = {"hit": "#1b7837", "false_alarm": "#c51b7d", "miss": "#2166ac", "correct_dry": "#999999"}
MODELS = ["unet", "tm_s1t", "tm_s1"]
NAMES = {"unet": "U-Net", "tm_s1t": "TerraMind with terrain", "tm_s1": "TerraMind radar only"}
PATCH = 224
NEAR = 2                     # centre (2 * NEAR + 1)^2 cells count as "the cell and its neighbours"


def reach_by_col(profile, strata_dir):
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


def patch(xn, r, c, p=PATCH):
    """(4, p, p) patch of the normalised inputs centred on (r, c); 0 (the mean) outside."""
    h = p // 2
    out = np.zeros((xn.shape[0], p, p), "float32")
    r0, c0 = r - h, c - h
    rr0, cc0 = max(r0, 0), max(c0, 0)
    rr1, cc1 = min(r0 + p, xn.shape[1]), min(c0 + p, xn.shape[2])
    out[:, rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] = xn[:, rr0:rr1, cc0:cc1]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--n-per-outcome", type=int, default=100)
    ap.add_argument("--background", type=int, default=24)
    ap.add_argument("--nsamples", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--cpu-ok", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_per_outcome, args.background, args.nsamples = 3, 4, 8
    import shap
    import torch
    t0 = time.time()
    paths = mc.Paths()
    out = paths.o06 / "jalsar" / "extensions" / ("m5_shap_networks_smoke" if args.smoke else "m5_shap_networks")
    (out / "examples").mkdir(parents=True, exist_ok=True)
    device = mc.pick_device(args.cpu_ok)
    print(f"m5_shap_networks.py{' (smoke)' if args.smoke else ''}: device {device}, shap {shap.__version__}")

    # Inputs of both scenes once (raw values; each model normalises its own way).
    scenes = {}
    for year in mc.YEARS:
        inp = mc.corridor_inputs(paths, year)
        truth = read_on_grid(paths.reference(year), inp["valid"].shape)
        scenes[year] = {"inp": inp, "truth": truth,
                        "reach": reach_by_col(inp["profile"], paths.strata(year))}

    rows, imp_rows, record = [], [], {"models": {}}
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        mdir = paths.model_dir(model, "hand")
        if not (mdir / "best.pt").exists():
            print(f"  {model}: no trained model in {mdir}, skipped")
            continue
        pred = mc.load_predictor(model, mdir, cpu_ok=args.cpu_ok)
        net = pred.model.float().eval()
        print(f"\n  {NAMES[model]} ({model}_hand), threshold {pred.threshold}")

        class Centre(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                z = self.m(x)
                if z.dim() == 4:                   # (B, 1, H, W) or (B, H, W): both give (B, 1)
                    z = z[:, 0]
                return torch.sigmoid(z[:, PATCH // 2, PATCH // 2]).unsqueeze(1)

        wrap = Centre(net).to(device).eval()
        X, meta, bg = [], [], []
        for year, sc in scenes.items():
            inp = sc["inp"]
            xn = mc.normalise(inp["x"], pred.stats)
            fmap = read_on_grid(paths.flood_map(year, model, "hand"), inp["valid"].shape)
            decided = inp["valid"] & ~inp["perm"] & ~inp["distort"]
            scored = np.isin(fmap, (0, 1)) & np.isin(sc["truth"], (0, 1)) & decided
            rng = np.random.default_rng([mc.SEED, year, MODELS.index(model)])
            for name, (p, t) in OUTCOMES.items():
                rr, cc = np.nonzero(scored & (fmap == p) & (sc["truth"] == t))
                if rr.size == 0:
                    continue
                pick = rng.choice(rr.size, size=min(args.n_per_outcome, rr.size), replace=False)
                for r, c in zip(rr[pick], cc[pick]):
                    X.append(patch(xn, r, c))
                    meta.append({"model": model, "year": year, "reach": sc["reach"][c], "outcome": name,
                                 "row": int(r), "col": int(c)})
            br, bc = np.nonzero(decided)
            for k in rng.choice(br.size, size=max(1, args.background // 2), replace=False):
                bg.append(patch(xn, br[k], bc[k]))
        X = np.stack(X)
        bgt = torch.from_numpy(np.stack(bg)).to(device)
        print(f"    {len(X)} cells to explain; background {len(bg)} patches; {args.nsamples} samples each")
        explainer = shap.GradientExplainer(wrap, bgt, batch_size=args.batch)
        with torch.no_grad():
            base = float(wrap(bgt).mean())
        att = np.zeros((len(X), 4), "float64")
        near = np.zeros(len(X), "float64")
        prob = np.zeros(len(X), "float64")
        err = np.zeros(len(X), "float64")
        keep_maps = {}
        t1 = time.time()
        step = max(1, args.batch)
        for i in range(0, len(X), step):
            xb = torch.from_numpy(X[i:i + step]).to(device)
            sv = explainer.shap_values(xb, nsamples=args.nsamples)
            sv = np.asarray(sv[0] if isinstance(sv, list) else sv)
            if sv.ndim == 5:                       # (B, 4, P, P, outputs)
                sv = sv[..., 0]
            with torch.no_grad():
                pb = wrap(xb).float().cpu().numpy()[:, 0]
            c0, c1 = PATCH // 2 - NEAR, PATCH // 2 + NEAR + 1
            for j in range(sv.shape[0]):
                k = i + j
                att[k] = sv[j].sum(axis=(1, 2))
                tot = np.abs(sv[j]).sum()
                near[k] = np.abs(sv[j][:, c0:c1, c0:c1]).sum() / tot if tot > 0 else np.nan
                prob[k] = pb[j]
                err[k] = abs(att[k].sum() - (pb[j] - base))
                o = meta[k]["outcome"]
                if keep_maps.get(o, 0) < 2:
                    keep_maps[o] = keep_maps.get(o, 0) + 1
                    fig, axs = plt.subplots(1, 3, figsize=(9.5, 3.2))
                    axs[0].imshow(X[k][0], cmap="gray"); axs[0].set_title("VV (normalised)")
                    a = np.abs(sv[j]).sum(0)
                    axs[1].imshow(a, cmap="magma"); axs[1].set_title("where the attribution sits")
                    axs[2].bar(FEATS, att[k], color=["#555", "#888", "#1f77b4", "#2ca02c"])
                    axs[2].axhline(0, color="black", lw=0.6); axs[2].set_title("by input")
                    for ax in axs[:2]:
                        ax.plot(PATCH // 2, PATCH // 2, "c+", ms=12)
                        ax.set_xticks([]); ax.set_yticks([])
                    fig.suptitle(f"{NAMES[model]}, {meta[k]['year']}, {o.replace('_', ' ')}, "
                                 f"probability {pb[j]:.2f}", fontsize=10)
                    fig.tight_layout()
                    fig.savefig(out / "examples" / f"{model}_{o}_{keep_maps[o]}.png", dpi=150)
                    plt.close(fig)
            done = min(i + step, len(X))
            el = time.time() - t1
            print(f"      {done}/{len(X)} cells, {el / 60:.1f} min, about {el / done * (len(X) - done) / 60:.1f} "
                  "min left", flush=True)
        for m_, a, n_, p_, e_ in zip(meta, att, near, prob, err):
            rows.append({**m_, "prob": round(float(p_), 4), "base": round(base, 4),
                         **{f"att_{k}": round(float(v), 5) for k, v in zip(FEATS, a)},
                         "near_share": round(float(n_), 4) if np.isfinite(n_) else "",
                         "completeness_error": round(float(e_), 4)})
        oc = np.array([m_["outcome"] for m_ in meta])
        for kind, val, msk in [("all", "all", np.ones(len(meta), bool))] + \
                [("outcome", o, oc == o) for o in OUTCOMES]:
            if msk.any():
                imp_rows.append({"model": model, "group": kind, "value": val, "cells": int(msk.sum()),
                                 **{f"mean_abs_{k}": round(float(v), 4) for k, v in
                                    zip(FEATS, np.abs(att[msk]).mean(0))},
                                 "near_share_mean": round(float(np.nanmean(near[msk])), 3)})
        record["models"][model] = {"cells": len(X), "base": base,
                                   "completeness_error_median": float(np.median(err)),
                                   "completeness_error_p90": float(np.percentile(err, 90)),
                                   "near_share_median": float(np.nanmedian(near))}
        print(f"    done: completeness error median {np.median(err):.3f}; cell and close neighbours give "
              f"{np.nanmedian(near) * 100:.0f} % of the attribution (median)")
        del pred, net, wrap, explainer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not rows:
        print("PROBLEM: no network model was found to explain")
        return 1
    with open(out / "shap_cells.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out / "importance.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(imp_rows[0].keys()))
        w.writeheader()
        w.writerows(imp_rows)

    # Comparison figure, with the Random Forest from M4 when it exists.
    comp = {r["model"]: [r[f"mean_abs_{k}"] for k in FEATS] for r in imp_rows if r["group"] == "all"}
    rf_imp = paths.o06 / "jalsar" / "extensions" / "m4_shap_rf" / "importance.csv"
    if rf_imp.exists():
        with open(rf_imp, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["group"] == "all":
                    comp = {"rf": [float(r[f"mean_abs_{k}"]) for k in FEATS], **comp}
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    wbar = 0.8 / max(len(comp), 1)
    for i, (m_, v) in enumerate(comp.items()):
        tot = sum(v) or 1.0
        ax.bar(np.arange(4) + (i - (len(comp) - 1) / 2) * wbar, [x / tot for x in v], wbar,
               label={"rf": "Random Forest", **NAMES}.get(m_, m_))
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(FEATS)
    ax.set_ylabel("share of mean |attribution|"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("What each model relies on")
    fig.tight_layout(); fig.savefig(out / "fig_compare_models.png", dpi=200); plt.close(fig)

    md = [f"# M5: SHAP for the networks{' (SMOKE RUN)' if args.smoke else ''}", "",
          "Method: shap.GradientExplainer (expected gradients) on the water probability of the centre cell of "
          f"a {PATCH} x {PATCH} patch, {args.nsamples} samples per cell, {args.background} background patches "
          "per model. Approximate: the completeness error below is how far the attributions' sum is from the "
          "probability minus the background average.", "",
          "| Model | Cells | VV | VH | HAND | SLOPE | Cell and close neighbours (5 x 5) | Completeness error (median) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in imp_rows:
        if r["group"] == "all":
            rec = record["models"][r["model"]]
            md.append(f"| {NAMES[r['model']]} | {r['cells']} | " + " | ".join(str(r[f'mean_abs_{k}']) for k in FEATS)
                      + f" | {r['near_share_mean'] * 100:.0f} % | {rec['completeness_error_median']:.3f} |")
    md += ["", "Values are mean |attribution| (change in water probability). The 5 x 5 share is the part of the "
           "attribution that comes from the cell itself and the cells within 20 m of it; the rest comes from "
           "further away in the patch, which the Random Forest cannot use.", "", "## By outcome", "",
           "| Model | Outcome | Cells | VV | VH | HAND | SLOPE | Near share |", "|---|---|---|---|---|---|---|---|"]
    for r in imp_rows:
        if r["group"] == "outcome":
            md.append(f"| {NAMES[r['model']]} | {r['value'].replace('_', ' ')} | {r['cells']} | "
                      + " | ".join(str(r[f'mean_abs_{k}']) for k in FEATS) + f" | {r['near_share_mean'] * 100:.0f} % |")
    md += ["", "Figures: fig_compare_models.png (with the Random Forest from M4 when it has run), and "
           "examples\\<model>_<outcome>_<n>.png (the VV patch, where the attribution sits, and the four inputs).",
           "", "SHAP explains the model, not the cause of the flood."]
    (out / "M5_SHAP_NETWORKS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    record.update({"script": "m5_shap_networks.py", "smoke": args.smoke, "patch": PATCH, "near": NEAR,
                   "nsamples": args.nsamples, "background": args.background, "seed": mc.SEED,
                   "shap_version": shap.__version__, "torch_version": torch.__version__,
                   "device": str(device), "minutes": round((time.time() - t0) / 60, 1)})
    mc.save_json(out / "run_record.json", record)
    print(f"\n  wrote {out}")
    print(f"PASS  M5 SHAP networks{' (smoke)' if args.smoke else ''}: {len(rows)} cells, "
          f"{(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
