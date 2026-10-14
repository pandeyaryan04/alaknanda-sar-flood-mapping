r"""
04_stratified_metrics.py
========================
Score a predicted flood map against a reference, inside every terrain stratum
rather than once over the whole corridor.

WHY THIS EXISTS
---------------
Overall accuracy on a mostly-dry image is close to meaningless. Bonafilia et al.
(2020) reported 0.9389 overall accuracy while omitting 37.56 per cent of the
flood water, because most of every chip is dry land. Rachit et al. (2026) show
the same thing in this exact terrain: 97.12 per cent accuracy alongside a
critical success index of 68.44.

So this script never prints a single headline number. It prints precision,
recall, F1, IoU and critical success index for every stratum produced by
03_terrain_strata.py, and it prints the flood-class pixel count alongside them,
because a stratum with forty flood pixels in it does not get to carry an
argument.

USAGE
-----
  python 04_stratified_metrics.py --pred data/otsu_2023.tif \
                                  --truth data/reference_2023.tif \
                                  --strata data/strata_2023 \
                                  --windows data/reference_windows_2023/reference_windows.geojson \
                                  --label "Otsu baseline, 2023 event" \
                                  --out outputs/tables

Predicted and reference rasters must be single-band and on the same grid as
each other, with 1 for flood and 0 for not flood. Anything else is treated as
nodata and excluded from every count.

The strata do NOT have to match that grid. They are built at the DEM's 30 m and
resampled here onto whatever the prediction is, which for Sentinel-1 is 10 m.

================================================================================
HOW TO RUN THIS  (copy one line, paste into Anaconda Prompt, press Enter)
================================================================================
    conda activate fyp
    cd /d "<project folder>\04 Code\FYP_code"

Then, once you have a flood map to score:

    python 04_stratified_metrics.py --pred data/otsu_2023.tif --truth data/reference_2023.tif --strata data/strata_2023 --windows data/reference_windows_2023/reference_windows.geojson --label "Otsu baseline, 2023 event" --out ../../06 Outputs/tables

The exact commands for each event are in RUN ORDER.md, step 4.9. The scoring was
written before any result existed, so the metric could not be chosen to flatter
the answer.

REACH (added 7 October 2026). The project reports every figure for the gorge
above Devprayag and the open valley below it separately, never averaged. The
reach is worked out here with the same rule as 03_terrain_strata.py: cells east
of the Devprayag longitude in strata_meta.json are the gorge. Every dimension is
reported once pooled and once inside each reach.

THIN STRATA, MCC AND CONFIDENCE INTERVALS (added 7 October 2026). This is the
rest of the 23 September rule.
  - Matthews correlation (MCC) is a column. Landwehr et al. (2024) recommend it.
  - A stratum with fewer than MIN_FLOOD_PIXELS (100) reference flood pixels is
    merged with a neighbour. The merge is decided from the REFERENCE counts
    only, never from a score, so every flood map of one event is scored on the
    same merged classes. For incidence, HAND and valley width the thin class
    joins the next class up or down, whichever holds fewer flood pixels; this
    repeats until no class is thin. Layover and shadow are joined into one
    class; they are never merged into "clean". A class still thin after that
    is kept and marked.
  - 95 per cent confidence intervals come from resampling the digitising
    windows (the sampling unit) with replacement, 2000 times, seed 20261007.
    With 8 windows these intervals are wide, and that is the honest answer.
    A row whose reference flood sits in fewer than 3 windows gets no interval.
  - F1 is now 0, not nan, where flood exists but none of it overlaps (the same
    as IoU). Before 7 October such rows printed nan; no other value changes.
  - The merged table is the one to quote. The table as classified, before any
    merge, is kept below it in metrics_summary.md and in the CSV
    (column table = as_classified), so nothing is hidden.
================================================================================
"""

__stage__ = "Stratified accuracy assessment"
__reads__ = ["predicted flood raster", "reference flood raster", "strata_*.tif"]
__writes__ = ["metrics_by_stratum.csv", "metrics_summary.md"]

import argparse
import csv
import json
import os
import sys

import numpy as np

try:
    import rasterio
except ImportError as e:
    sys.exit(f"\nMissing package: {e}.\nActivate the environment first:  conda activate fyp\n")

MIN_FLOOD_PIXELS = 100      # below this a stratum is thin: merged, or flagged if it cannot be
N_BOOT = 2000               # window resamples for the confidence intervals
BOOT_SEED = 20261007        # fixed, so a re-run gives the same intervals
MIN_WINDOWS_CI = 3          # no interval for a row whose flood sits in fewer windows

BINS = {
    "incidence": ([0, 30, 40, 50, 181], ["lt30", "30_40", "40_50", "gt50"]),
    "hand": ([0, 5, 15, 50, 1e9], ["lt5m", "5_15m", "15_50m", "gt50m"]),
    "valley_width": ([0, 100, 300, 1000, 1e9], ["lt100m", "100_300m", "300_1000m", "gt1000m"]),
}


def read_band(path, shape=None):
    """Read band 1. If a shape is given it must match exactly."""
    with rasterio.open(path) as src:
        a = src.read(1)
    if shape is not None and a.shape != shape:
        sys.exit(f"\n{os.path.basename(path)} is {a.shape}, expected {shape}.\n"
                 "The prediction and the reference must be on the same grid as "
                 "each other. Strata are resampled automatically; these two are not.\n")
    return a


def read_onto(path, shape, transform, crs, categorical=False):
    """Read a stratum raster and resample it onto the prediction grid.

    The strata are built at the DEM's resolution, 30 m, because slope and HAND
    from a 30 m DEM mean nothing at 10 m. The flood maps are at Sentinel-1's
    resolution, 10 m. Rather than force one onto the other up front, the strata
    are resampled here, at scoring time.

    Nearest neighbour for the categorical layover/shadow mask, because
    interpolating a class label produces a class that does not exist. Bilinear
    for the continuous ones.
    """
    from rasterio.warp import reproject, Resampling
    with rasterio.open(path) as src:
        out = np.empty(shape, dtype="float64")
        reproject(source=rasterio.band(src, 1), destination=out,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform, dst_crs=crs,
                  resampling=Resampling.nearest if categorical else Resampling.bilinear,
                  src_nodata=src.nodata, dst_nodata=np.nan)
    return out


def classify(arr, edges, labels):
    out = np.full(arr.shape, "", dtype=object)
    for i, lab in enumerate(labels):
        out[(arr >= edges[i]) & (arr < edges[i + 1])] = lab
    return out


def metrics_from_counts(tp, fp, fn, tn):
    """Precision, recall, F1, IoU and MCC from confusion counts.

    Works on plain numbers and on numpy arrays alike, so the bootstrap can pass
    2000 sets of counts at once. Undefined values come back as nan.
    """
    tp, fp, fn, tn = (np.asarray(x, dtype="float64") for x in (tp, fp, fn, tn))
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        recall = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        # F1 is 0, not nan, when there is flood in the truth or the prediction
        # but none of it overlaps, the same as IoU. nan only when both are empty.
        f1 = np.where(tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
        iou = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), np.nan)
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = np.where(den > 0, (tp * tn - fp * fn) / den, np.nan)
    return {"precision": precision, "recall": recall, "f1": f1,
            "iou": iou, "mcc": mcc}


def window_counts(pred, truth, sel, win, n_win):
    """Confusion counts inside sel, split by digitising window.

    Returns an array of shape (n_win + 1, 4): row k holds tp, fp, fn, tn for
    window k, and row 0 holds anything outside every window (should be empty,
    because the reference is nodata outside the windows).
    """
    p = pred[sel].astype(bool)
    t = truth[sel].astype(bool)
    w = win[sel]
    out = np.zeros((n_win + 1, 4), dtype="int64")
    for j, m in enumerate((p & t, p & ~t, ~p & t, ~p & ~t)):
        out[:, j] = np.bincount(w[m], minlength=n_win + 1)[: n_win + 1]
    return out


def score(wc, weights):
    """Scores for one stratum from its per-window counts. None if empty.

    weights: (N_BOOT, n_win) array, how many times each window was drawn in
    each bootstrap resample. The same weights are used for every stratum.
    """
    tp, fp, fn, tn = (int(v) for v in wc.sum(axis=0))
    n = tp + fp + fn + tn
    if n == 0:
        return None
    m = metrics_from_counts(tp, fp, fn, tn)
    r = {
        "pixels": n,
        "flood_pixels_truth": tp + fn,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(float(m["precision"]), 4),
        "recall": round(float(m["recall"]), 4),
        "f1": round(float(m["f1"]), 4),
        "iou": round(float(m["iou"]), 4),
        "csi": round(float(m["iou"]), 4),   # critical success index is the same quantity
        "mcc": round(float(m["mcc"]), 4),
        "overall_accuracy": round((tp + tn) / n, 4),
        "thin": (tp + fn) < MIN_FLOOD_PIXELS,
        "windows_with_flood": int(np.count_nonzero(wc[1:, 0] + wc[1:, 2])),
    }
    # Window bootstrap: each resample draws len(windows) windows with
    # replacement and sums their counts. Percentile interval, 95 per cent.
    # No interval when the reference flood sits in fewer than MIN_WINDOWS_CI
    # windows: resampling one or two windows gives a near-zero width that means
    # nothing.
    bc = weights @ wc[1:]                                            # (N_BOOT, 4)
    bm = metrics_from_counts(bc[:, 0], bc[:, 1], bc[:, 2], bc[:, 3])
    for k in ("f1", "iou", "mcc"):
        v = bm[k][~np.isnan(bm[k])]
        if r["windows_with_flood"] >= MIN_WINDOWS_CI and v.size >= 0.5 * len(bm[k]):
            lo, hi = np.percentile(v, [2.5, 97.5])
            r[f"{k}_lo"], r[f"{k}_hi"] = round(float(lo), 4), round(float(hi), 4)
        else:                       # undefined in most resamples: no interval
            r[f"{k}_lo"] = r[f"{k}_hi"] = float("nan")
    return r


def merge_ordinal(labels, flood):
    """Merge thin classes of an ordered dimension into neighbours.

    labels: class names in order (low to high). flood: reference flood pixels
    per class. Returns a list of groups, each a list of class names. Decided
    from reference counts only.
    """
    groups = [[lab] for lab in labels if lab in flood]
    counts = [flood[lab] for lab in labels if lab in flood]
    while len(groups) > 1 and min(counts) < MIN_FLOOD_PIXELS:
        i = int(np.argmin(counts))
        if i == 0:
            j = 1
        elif i == len(groups) - 1:
            j = i - 1
        else:                       # the neighbour with fewer flood pixels
            j = i - 1 if counts[i - 1] <= counts[i + 1] else i + 1
        a, b = sorted((i, j))
        groups[a:b + 1] = [groups[a] + groups[b]]
        counts[a:b + 1] = [counts[a] + counts[b]]
    return groups


def merge_layover(flood):
    """Layover and shadow join into one class if either is thin. Never into clean."""
    present = [c for c in ("clean", "layover", "shadow") if c in flood]
    distorted = [c for c in ("layover", "shadow") if c in present]
    if len(distorted) == 2 and min(flood["layover"], flood["shadow"]) < MIN_FLOOD_PIXELS:
        return [["clean"]] * ("clean" in present) + [distorted]
    return [[c] for c in present]


def window_raster(path, shape, transform, crs):
    """Burn the digitising windows onto the prediction grid as 1..n, 0 outside."""
    from rasterio.features import rasterize
    from rasterio.crs import CRS
    with open(path, encoding="utf-8") as fh:
        gj = json.load(fh)
    name = gj.get("crs", {}).get("properties", {}).get("name")
    if name and CRS.from_user_input(name) != crs:
        sys.exit(f"\n{os.path.basename(path)} is in {name}, the prediction in {crs}.\n"
                 "They must match.\n")
    shapes = []
    for f in gj["features"]:
        w = f["properties"].get("window")
        if w is None:
            sys.exit(f"\nA feature in {path} has no 'window' number.\n")
        shapes.append((f["geometry"], int(w)))
    win = rasterize(shapes, out_shape=shape, transform=transform, fill=0,
                    dtype="int32")
    return win, max(w for _, w in shapes)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--strata", required=True, help="directory written by 03_terrain_strata.py")
    ap.add_argument("--windows", required=True,
                    help="reference_windows.geojson written by 05_reference_windows.py")
    ap.add_argument("--label", default="unnamed run")
    ap.add_argument("--out", default="outputs/tables")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with rasterio.open(args.pred) as src:
        pred = src.read(1)
        p_transform, p_crs, p_res = src.transform, src.crs, src.res
    truth = read_band(args.truth, pred.shape)
    base_valid = np.isin(pred, [0, 1]) & np.isin(truth, [0, 1])
    print(f"prediction grid {pred.shape[0]} x {pred.shape[1]} at {abs(p_res[0]):.1f} m")

    S = args.strata
    def strat(name, categorical=False):
        return read_onto(os.path.join(S, name), pred.shape, p_transform, p_crs,
                         categorical)
    theta = strat("strata_theta_loc.tif")
    handm = strat("strata_hand_m.tif")
    width = strat("strata_valley_width_m.tif")
    ls = np.nan_to_num(strat("strata_layover_shadow.tif", categorical=True),
                       nan=0).astype("uint8")
    print("strata resampled onto the prediction grid")
    with open(os.path.join(S, "strata_meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)

    dims = {
        "incidence": classify(theta, *BINS["incidence"]),
        "hand": classify(handm, *BINS["hand"]),
        "valley_width": classify(width, *BINS["valley_width"]),
        "layover_shadow": np.choose(np.clip(ls, 0, 2),
                                    np.array(["clean", "layover", "shadow"], dtype=object)),
    }

    # Reach, by the same rule as 03_terrain_strata.py: east of the Devprayag
    # longitude is the gorge, west of it the open valley.
    from rasterio.warp import transform as warp_transform
    split_lon = float(meta.get("reach_split_longitude", 78.5981))
    split_x = warp_transform("EPSG:4326", p_crs, [split_lon], [30.15])[0][0]
    xs = p_transform.c + p_transform.a * (np.arange(pred.shape[1]) + 0.5)
    reach_row = np.where(xs >= split_x, "gorge_above_devprayag",
                         "open_valley_below_devprayag").astype(object)
    reach = np.broadcast_to(reach_row, pred.shape)
    print(f"reach split at x = {split_x:,.0f} m ({split_lon} E)")

    # Digitising windows, the sampling unit for the confidence intervals.
    win, n_win = window_raster(args.windows, pred.shape, p_transform, p_crs)
    outside = int(np.count_nonzero(base_valid & (win == 0)))
    print(f"{n_win} digitising windows read from {os.path.basename(args.windows)}")
    if outside:
        print(f"  WARNING {outside} scored cells lie outside every window. They count in "
              "the scores but not in the confidence intervals.")
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, n_win, size=(N_BOOT, n_win))
    weights = np.stack([np.bincount(d, minlength=n_win) for d in draws])

    REACHES = ("gorge_above_devprayag", "open_valley_below_devprayag")
    ORDER = {"incidence": BINS["incidence"][1], "hand": BINS["hand"][1],
             "valley_width": BINS["valley_width"][1],
             "layover_shadow": ["clean", "layover", "shadow"]}

    # Per-window counts for every class as classified, once. Merged classes are
    # sums of these, so nothing is counted twice.
    wc_all = {"ALL": window_counts(pred, truth, base_valid, win, n_win)}
    for rch in REACHES:
        wc_all[rch] = window_counts(pred, truth, base_valid & (reach == rch), win, n_win)
    wc_cls = {}
    for rch in ("pooled",) + REACHES:
        in_reach = base_valid if rch == "pooled" else base_valid & (reach == rch)
        for dim, cls in dims.items():
            for stratum in ORDER[dim]:
                sel = in_reach & (cls == stratum)
                if np.any(sel):
                    wc_cls[(rch, dim, stratum)] = window_counts(pred, truth, sel, win, n_win)

    def row(wc, table, rch, dim, stratum, merged_from=""):
        s = score(wc, weights)
        if s is not None:
            s.update({"table": table, "reach": rch, "dimension": dim,
                      "stratum": stratum, "merged_from": merged_from})
        return s

    head = [row(wc_all["ALL"], "reported", "pooled", "ALL", "both reaches pooled")]
    head += [row(wc_all[r], "reported", r, "reach", r) for r in REACHES]
    overall = head[0]

    as_classified, reported, merges = [], list(head), []
    for rch in ("pooled",) + REACHES:
        for dim in ORDER:
            present = [c for c in ORDER[dim] if (rch, dim, c) in wc_cls]
            for c in present:
                as_classified.append(row(wc_cls[(rch, dim, c)], "as_classified", rch, dim, c))
            # Merge decided from reference flood counts only (tp + fn).
            flood = {c: int(wc_cls[(rch, dim, c)][:, [0, 2]].sum()) for c in present}
            groups = (merge_layover(flood) if dim == "layover_shadow"
                      else merge_ordinal(ORDER[dim], flood))
            for g in groups:
                wc = sum(wc_cls[(rch, dim, c)] for c in g)
                name = "+".join(g)
                reported.append(row(wc, "reported", rch, dim, name,
                                    merged_from=name if len(g) > 1 else ""))
                if len(g) > 1:
                    merges.append(f"{rch}, {dim}: {' + '.join(f'{c} ({flood[c]})' for c in g)}")
    rows = [r for r in reported + as_classified if r is not None]

    cols = ["table", "reach", "dimension", "stratum", "merged_from", "pixels",
            "flood_pixels_truth", "windows_with_flood", "tp", "fp", "fn", "tn",
            "precision", "recall", "f1", "f1_lo", "f1_hi", "iou", "iou_lo", "iou_hi",
            "csi", "mcc", "mcc_lo", "mcc_hi", "overall_accuracy", "thin"]
    csv_path = os.path.join(args.out, "metrics_by_stratum.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})

    # --- readable summary ---------------------------------------------------
    def ci(r, k):
        lo, hi = r.get(f"{k}_lo"), r.get(f"{k}_hi")
        if lo is None or np.isnan(lo):
            return f"{r[k]}"
        return f"{r[k]} [{lo}, {hi}]"

    def table(rs):
        out = ["| Reach | Dimension | Stratum | Flood px | Windows | Precision | Recall | "
               "F1 [95% CI] | IoU / CSI [95% CI] | MCC [95% CI] | OA |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rs:
            flag = " *" if r.get("thin") else ""
            out.append(f"| {r['reach']} | {r['dimension']} | {r['stratum']}{flag} | "
                       f"{r['flood_pixels_truth']} | {r['windows_with_flood']} | "
                       f"{r['precision']} | {r['recall']} | {ci(r, 'f1')} | {ci(r, 'iou')} | "
                       f"{ci(r, 'mcc')} | {r['overall_accuracy']} |")
        return out

    lines = [f"# Stratified accuracy: {args.label}", "",
             f"Incidence angle source: {meta.get('incidence_angle_source')}",
             f"Orbit pass: {meta.get('orbit_pass')}", "",
             "Critical success index is the same quantity as IoU and is given",
             "under both names because the flood literature uses both.", "",
             "Report the two reaches separately. The pooled rows are there for",
             "checking, not for quoting on their own.", "",
             f"Brackets: 95 per cent interval from resampling the {n_win} digitising "
             f"windows with replacement, {N_BOOT} times (seed {BOOT_SEED}). "
             "Windows: how many windows hold reference flood in that row. "
             f"No interval is given when that is under {MIN_WINDOWS_CI}.", "",
             "## Reported table, thin classes merged", ""]
    lines += [x for x in table([r for r in rows if r["table"] == "reported"])]
    lines += ["", f"`*` still fewer than {MIN_FLOOD_PIXELS} flood pixels in the reference "
                  "after merging. Reported, but too thin to carry an argument.", ""]
    lines += ["Merges (reference flood pixels in brackets), decided from the reference "
              "counts before any score:", ""]
    lines += [f"- {m}" for m in merges] if merges else ["- none"]
    lines += ["", "## As classified, before merging", ""]
    lines += table([r for r in rows if r["table"] == "as_classified"])
    lines += ["", f"`*` fewer than {MIN_FLOOD_PIXELS} flood pixels in the reference.", ""]

    oa, f1 = overall["overall_accuracy"], overall["f1"]
    if not np.isnan(f1) and oa - f1 > 0.15:
        lines += [f"**Read this before quoting the headline.** Overall accuracy is {oa} "
                  f"and F1 is {f1}. The gap is the dry-land effect: most pixels in this "
                  "corridor are not flooded, so overall accuracy is high whatever the "
                  "model does with the water. Quote F1 or IoU, not OA.", ""]

    md_path = os.path.join(args.out, "metrics_summary.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    # Print the reported table only; the full file holds both.
    cut = lines.index("## As classified, before merging")
    print("\n".join(lines[6:cut]))
    print(f"\nwrote {csv_path}\nwrote {md_path}")


if __name__ == "__main__":
    main()
