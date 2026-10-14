r"""
27_compare_maps.py
==================
Paired comparison of two flood maps, A and B, against the same reference and
windows.

Only cells where A, B and the reference are all 0 or 1 count, so both maps
are scored on exactly the same cells. For each row (pooled, each reach, each
terrain class with the same classes and thin-class merging as
04_stratified_metrics.py) it gives F1, IoU and MCC for A, for B, and for
B minus A, with PAIRED 95 % intervals: the same 2000 window resamples
(seed 20261007) are applied to both maps' per-window counts, so the
interval is on the difference itself. The counting, the metrics and the
window raster, and the score() that gives each map's own F1, IoU, MCC and
intervals, are imported from 04_stratified_metrics.py, so the definitions
cannot drift apart.

Writes comparison.csv and comparison.md (which says in words which map is
better and whether the interval excludes 0) into --out.

USAGE: HOW TO RUN THIS  (copy one line, paste into Anaconda Prompt, press Enter)
================================================================================
    conda activate fyp
    cd /d "<project folder>\04 Code\FYP_code"

    python 27_compare_maps.py --selftest
    python 27_compare_maps.py --a data/flood_2021_rf_hand/flood_2021_rf_hand.tif --b data/flood_2021_unet_hand/flood_2021_unet_hand.tif --truth data/reference_2021.tif --strata data/strata_2021_v3 --windows data/reference_windows_2021/reference_windows.geojson --label-a "RF hand" --label-b "U-Net hand" --out "..\..\06 Outputs\tables\compare_2021_rf_hand_vs_unet_hand"

About 1 to 3 minutes. The last line says PASS.
================================================================================
"""

__stage__ = "Models: paired comparison of two flood maps"
__reads__ = ["two flood maps", "reference_<year>.tif", "strata_<year>_v3/",
             "reference_windows.geojson"]
__writes__ = ["comparison.csv", "comparison.md"]

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

import models_common as mc

try:
    import rasterio
    from rasterio.warp import transform as warp_transform
except ImportError as e:
    sys.exit(f"\nMissing package: {e}.\nActivate the environment first:  conda activate fyp\n")

_sc = mc.load_script("04_stratified_metrics.py")
window_counts = _sc.window_counts
metrics_from_counts = _sc.metrics_from_counts
score = _sc.score
window_raster = _sc.window_raster
read_onto = _sc.read_onto
BINS, N_BOOT, BOOT_SEED = _sc.BINS, _sc.N_BOOT, _sc.BOOT_SEED
MIN_WINDOWS_CI, MIN_FLOOD_PIXELS = _sc.MIN_WINDOWS_CI, _sc.MIN_FLOOD_PIXELS
REACHES = ("gorge_above_devprayag", "open_valley_below_devprayag")
KEYS = ("f1", "iou", "mcc")


def boot_weights(n_win):
    """The scorer's window resamples: the same draws, so intervals are comparable."""
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, n_win, size=(N_BOOT, n_win))
    return np.stack([np.bincount(d, minlength=n_win) for d in draws])


def codes(arr, edges):
    """Class number per cell with 04's rule (edges[i] <= v < edges[i+1]); -1 for none."""
    out = np.full(arr.shape, -1, "int8")
    for i in range(len(edges) - 1):
        out[(arr >= edges[i]) & (arr < edges[i + 1])] = i
    return out


def paired(wa, wb, weights):
    """One row: A, B and B - A for F1, IoU, MCC with paired window-bootstrap intervals.

    A's and B's own values and intervals come from 04's score(), unchanged.
    The difference uses the same resamples for both maps, with 04's rule for
    when an interval is given.
    """
    sa, sb = score(wa, weights), score(wb, weights)
    if sa is None:
        return None
    a = [int(v) for v in wa.sum(axis=0)]
    b = [int(v) for v in wb.sum(axis=0)]
    ma, mb = metrics_from_counts(*a), metrics_from_counts(*b)
    wwf = sa["windows_with_flood"]
    row = {"pixels": sa["pixels"], "flood_pixels_truth": sa["flood_pixels_truth"],
           "windows_with_flood": wwf, "thin": sa["thin"]}
    ba, bb = weights @ wa[1:], weights @ wb[1:]
    bma = metrics_from_counts(ba[:, 0], ba[:, 1], ba[:, 2], ba[:, 3])
    bmb = metrics_from_counts(bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3])
    for k in KEYS:
        for pre, s in (("a", sa), ("b", sb)):
            row[f"{pre}_{k}"], row[f"{pre}_{k}_lo"], row[f"{pre}_{k}_hi"] = \
                s[k], s[f"{k}_lo"], s[f"{k}_hi"]
        row[f"diff_{k}"] = round(float(mb[k] - ma[k]), 4)
        v = bmb[k] - bma[k]
        v = v[np.isfinite(v)]
        if wwf >= MIN_WINDOWS_CI and v.size >= 0.5 * len(weights):
            lo, hi = np.percentile(v, [2.5, 97.5])
            row[f"diff_{k}_lo"], row[f"diff_{k}_hi"] = round(float(lo), 4), round(float(hi), 4)
        else:
            row[f"diff_{k}_lo"] = row[f"diff_{k}_hi"] = float("nan")
    return row


def compare(a, b, truth, strata, windows, profile):
    """All rows for maps a and b (arrays on the reference grid)."""
    shape, tf, crs = truth.shape, profile["transform"], profile["crs"]
    base = np.isin(a, (0, 1)) & np.isin(b, (0, 1)) & np.isin(truth, (0, 1))
    strata = Path(strata)
    dims = {"incidence": codes(read_onto(str(strata / "strata_theta_loc.tif"), shape, tf, crs),
                               BINS["incidence"][0]),
            "hand": codes(read_onto(str(strata / "strata_hand_m.tif"), shape, tf, crs),
                          BINS["hand"][0]),
            "valley_width": codes(read_onto(str(strata / "strata_valley_width_m.tif"), shape,
                                            tf, crs), BINS["valley_width"][0]),
            "layover_shadow": np.clip(np.nan_to_num(
                read_onto(str(strata / "strata_layover_shadow.tif"), shape, tf, crs, True),
                nan=0).astype("uint8"), 0, 2).astype("int8")}
    names = {"incidence": BINS["incidence"][1], "hand": BINS["hand"][1],
             "valley_width": BINS["valley_width"][1],
             "layover_shadow": ["clean", "layover", "shadow"]}
    meta = json.loads((strata / "strata_meta.json").read_text(encoding="utf-8"))
    split_lon = float(meta.get("reach_split_longitude", 78.5981))
    split_x = warp_transform("EPSG:4326", crs, [split_lon], [30.15])[0][0]
    xs = tf.c + tf.a * (np.arange(shape[1]) + 0.5)
    gorge = np.broadcast_to(xs >= split_x, shape)
    win, n_win = window_raster(str(windows), shape, tf, crs)
    weights = boot_weights(n_win)

    def wc(sel):
        return window_counts(a, truth, sel, win, n_win), window_counts(b, truth, sel, win, n_win)

    rows = []

    def add(sel_counts, rch, dim, stratum, merged=""):
        r = paired(*sel_counts, weights)
        if r is not None:
            rows.append({"reach": rch, "dimension": dim, "stratum": stratum,
                         "merged_from": merged, **r})

    add(wc(base), "pooled", "ALL", "both reaches pooled")
    for rch, m in zip(REACHES, (gorge, ~gorge)):
        add(wc(base & m), rch, "reach", rch)
    for rch, rm in (("pooled", None), (REACHES[0], gorge), (REACHES[1], ~gorge)):
        inr = base if rm is None else base & rm
        for dim, cd in dims.items():
            per = {}
            for i, nm in enumerate(names[dim]):
                sel = inr & (cd == i)
                if sel.any():
                    per[nm] = wc(sel)
            flood = {c: int(v[0][:, [0, 2]].sum()) for c, v in per.items()}
            groups = (_sc.merge_layover(flood) if dim == "layover_shadow"
                      else _sc.merge_ordinal(names[dim], flood))
            for g in groups:
                ca = sum(per[c][0] for c in g)
                cb = sum(per[c][1] for c in g)
                nm = "+".join(g)
                add((ca, cb), rch, dim, nm, nm if len(g) > 1 else "")
    return rows, n_win


def _read(path, what):
    p = Path(path)
    if not p.exists():
        mc.fail(f"{what} not found:\n  {p}")
    with rasterio.open(p) as s:
        return s.read(1), s.profile


def main(argv=None):
    mc.setup_console()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", help="map A (the reference point of the difference)")
    ap.add_argument("--b", help="map B; differences are B minus A")
    ap.add_argument("--truth")
    ap.add_argument("--strata")
    ap.add_argument("--windows")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--out")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--max-ram-gb", type=float, default=12.0,
                    help="RAM the comparison may plan for (default 12)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    for k in ("a", "b", "truth", "strata", "windows", "out"):
        if not getattr(args, k):
            mc.fail(f"--{k} is required (see the command at the top of this file).")
    out = Path(args.out)
    print(f"27_compare_maps.py: {args.label_a} vs {args.label_b}")
    if mc.guard(out / "comparison.md", args, f"The comparison in {out.name}"):
        return 0
    a, pa = _read(args.a, "Map A")
    b, pb = _read(args.b, "Map B")
    t, pt = _read(args.truth, "The reference")
    for nm, arr, p in (("A", a, pa), ("B", b, pb)):
        if not mc.same_grid(t.shape, pt["transform"], pt["crs"], arr.shape, p["transform"],
                            p["crs"]):
            mc.fail(f"Map {nm} is not on the reference grid ({arr.shape} vs {t.shape}). "
                    "Both maps and the reference must share the corridor 10 m grid.")
    need = t.size * 70 / 1e9          # three maps, four strata layers, classes, windows
    if need > args.max_ram_gb:
        mc.fail(f"This comparison needs about {need:.1f} GB of RAM, more than --max-ram-gb "
                f"{args.max_ram_gb:g}. Close other programs and raise --max-ram-gb.")
    rows, n_win = compare(a, b, t, args.strata, args.windows, pt)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["reach", "dimension", "stratum", "merged_from", "pixels", "flood_pixels_truth",
            "windows_with_flood"]
    for k in KEYS:
        for pre in ("a", "b", "diff"):
            cols += [f"{pre}_{k}", f"{pre}_{k}_lo", f"{pre}_{k}_hi"]
    cols.append("thin")
    with open(out / "comparison.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    lines = [f"# Paired comparison: {args.label_b} (B) against {args.label_a} (A)", "",
             f"A: `{args.a}`  ", f"B: `{args.b}`  ", f"Reference: `{args.truth}`", "",
             "Cells where A, B and the reference are all 0 or 1. Differences are B minus A. "
             f"Brackets: paired 95 % interval from {N_BOOT} resamples of the {n_win} windows "
             f"(seed {BOOT_SEED}), the same resamples for both maps. No interval when the "
             f"reference flood sits in fewer than {MIN_WINDOWS_CI} windows.", ""]

    def verdict(r, k="f1"):
        d, lo, hi = r[f"diff_{k}"], r[f"diff_{k}_lo"], r[f"diff_{k}_hi"]
        if np.isnan(d):
            return f"{k.upper()} is undefined here."
        if d == 0 and (np.isnan(lo) or lo == hi == 0):
            return f"{args.label_a} and {args.label_b} have the same {k.upper()} ({r[f'a_{k}']})."
        who = (f"{args.label_b} is better than {args.label_a}" if d > 0
               else f"{args.label_b} is worse than {args.label_a}")
        if np.isnan(lo):
            return f"{who} on {k.upper()} by {abs(d):.4f}; no interval (too few windows with flood)."
        sure = ("the interval excludes 0, so the difference is unlikely to be chance"
                if lo > 0 or hi < 0 else "the interval includes 0: no clear difference")
        return f"{who} on {k.upper()} by {abs(d):.4f} [{lo}, {hi}]; {sure}."

    pooled = rows[0]
    lines += ["## In words", "", f"- Pooled: {verdict(pooled)}"]
    for r in rows[1:3]:
        lines.append(f"- {r['stratum']}: {verdict(r)}")
    lines += ["", "## Table", "",
              "| Reach | Dimension | Stratum | Flood px | Windows | F1 A | F1 B | F1 B-A [95% CI] "
              "| IoU B-A [95% CI] | MCC B-A [95% CI] |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ci = {k: (f"{r[f'diff_{k}']} [{r[f'diff_{k}_lo']}, {r[f'diff_{k}_hi']}]"
                  if not np.isnan(r[f"diff_{k}_lo"]) else f"{r[f'diff_{k}']}") for k in KEYS}
        lines.append(f"| {r['reach']} | {r['dimension']} | {r['stratum']}{' *' if r['thin'] else ''}"
                     f" | {r['flood_pixels_truth']} | {r['windows_with_flood']} | {r['a_f1']} | "
                     f"{r['b_f1']} | {ci['f1']} | {ci['iou']} | {ci['mcc']} |")
    lines += ["", f"`*` fewer than {MIN_FLOOD_PIXELS} reference flood pixels after merging."]
    tmp = out / "comparison.md.part"
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, out / "comparison.md")
    print("  " + lines[lines.index("## In words") + 2])
    print(f"  wrote {out}")
    print("PASS")
    return 0


def selftest():
    import models_synthetic as syn
    print("27_compare_maps.py self-test")
    with tempfile.TemporaryDirectory(prefix="fyp selftest ", ignore_cleanup_errors=True) as td:
        proj = Path(td) / "Final Year Project"
        syn.build_project(proj, chips=False, years=(2021,))
        p = mc.Paths(proj)
        common = ["--truth", str(p.reference(2021)), "--strata", str(p.strata(2021)),
                  "--windows", str(p.windows(2021))]
        o1, o2 = Path(td) / "cmp same", Path(td) / "cmp other"
        ok = main(["--a", str(p.otsu(2021)), "--b", str(p.otsu(2021)), "--out", str(o1)]
                  + common) == 0
        ok &= main(["--a", str(p.otsu(2021)), "--b", str(p.otsu(2021, "chfeb")),
                    "--out", str(o2)] + common) == 0
        with open(o1 / "comparison.csv", encoding="utf-8") as fh:
            same = next(csv.DictReader(fh))
        with open(o2 / "comparison.csv", encoding="utf-8") as fh:
            other = next(csv.DictReader(fh))
        a, _ = _read(p.otsu(2021), "A")
        t, _ = _read(p.reference(2021), "truth")
        sc = np.isin(a, (0, 1)) & np.isin(t, (0, 1))
        tp, fp, fn = (int(((a == 1) & (t == 1) & sc).sum()), int(((a == 1) & (t == 0) & sc).sum()),
                      int(((a == 0) & (t == 1) & sc).sum()))
    checks = [("ran", ok),
              ("A vs A: difference 0, interval [0, 0]",
               float(same["diff_f1"]) == 0 and float(same["diff_f1_lo"]) == 0 == float(same["diff_f1_hi"])),
              ("A's F1 recounted", abs(float(other["a_f1"]) - round(2 * tp / (2 * tp + fp + fn), 4)) < 1e-9)]
    for name, c in checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {name}")
    good = all(c for _, c in checks)
    print("SELFTEST PASS" if good else "SELFTEST FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(mc.cli(main))
