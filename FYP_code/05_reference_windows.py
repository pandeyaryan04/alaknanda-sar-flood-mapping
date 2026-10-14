r"""
05_reference_windows.py
=======================
Place the reference-extent digitising windows along the corridor, stratified by
terrain, and write them out for QGIS.

WHY THIS EXISTS
---------------
The reference flood extent is being built from windows rather than wall to wall.
That is a sampling design, and a sampling design is only defensible if the
windows were placed by a rule written down in advance rather than by eye after
looking at the imagery.

This script is that rule. It places candidate windows along the river, measures
what terrain each one contains, and selects the set that covers the most
distinct strata. You run it once, look at what it chose, and then digitise
inside those squares and nowhere else.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not look at the flood prediction when choosing. Placing validation
windows where the model already says there is water would make false negatives
invisible: any flood the model missed would fall outside every window and never
be counted against it. The sampling frame here is the river corridor, which is
geography, and every window straddles the channel by construction.

The script does report how many predicted-flood pixels landed in each chosen
window, because you should know what you are walking into before you spend three
hours in QGIS. That column is information, not a selection criterion.
**Re-running the selection to get windows with more water in them would destroy
the design.** If the honest windows contain very little water, that is the
finding, and it is a finding about the 2023 event that the separability figures
already hinted at.

USAGE
-----
  conda activate fyp
  cd /d "<project folder>\04 Code\FYP_code"

  python 05_reference_windows.py ^
    --strata data/strata_2023_v2 ^
    --grid data/flood_2023_v2/flood_2023_otsu.tif ^
    --pred data/flood_2023_altjan/flood_2023altjan_otsu.tif ^
    --n 8 --size 5 ^
    --out data/reference_windows_2023

Then the same for 2021 with strata_2021, its own grid, and --out
data/reference_windows_2021.

  --strata      directory written by 03_terrain_strata.py
  --grid        any raster on the Sentinel-1 grid. Used only for its CRS and
                cell size, so the windows land on whole cells
  --pred        optional. A flood map, reported per window, never used to choose
  --centreline  optional. A GeoJSON line in any CRS. Falls back to the seven
                town anchors if not given, which are approximate and are the
                same seven used in the Earth Engine scripts
  --n           how many windows to select
  --size        window side in km
  --spacing     minimum km between window centres, default = size, so windows
                touch at worst and never overlap

WRITES
------
  reference_windows.geojson   load this in QGIS, digitise inside these squares
  reference_windows.csv       what terrain each window contains
  reference_windows.md        the same table, readable, for the methods section
"""

__stage__ = "Reference extent, window placement"
__reads__ = ["strata_*.tif", "a raster on the S1 grid", "optional centreline"]
__writes__ = ["reference_windows.geojson", "reference_windows.csv",
              "reference_windows.md"]

import argparse
import csv
import json
import os
import sys

import numpy as np

try:
    import rasterio
    from rasterio.transform import rowcol
    from pyproj import Transformer
except ImportError as e:
    sys.exit(f"\nMissing package: {e}.\nActivate the environment first:  "
             f"conda activate fyp\n")

# Same bins as 04_stratified_metrics.py. If you change them there, change them
# here, or the windows will be selected against strata the scoring never uses.
BINS = {
    "incidence":    ([0, 30, 40, 50, 181],        ["lt30", "30_40", "40_50", "gt50"]),
    "hand":         ([0, 5, 15, 50, 1e9],         ["lt5m", "5_15m", "15_50m", "gt50m"]),
    "valley_width": ([0, 100, 300, 1000, 1e9],    ["lt100m", "100_300m", "300_1000m", "gt1000m"]),
}
LAYER = {
    "incidence":    "strata_theta_loc.tif",
    "hand":         "strata_hand_m.tif",
    "valley_width": "strata_valley_width_m.tif",
}

# The seven anchors from 01_export_s1.js, in lon/lat. Approximate town
# positions, which is fine for placing windows along the reach and is NOT fine
# as a study area boundary. Pass --centreline once you have digitised the real
# one.
FALLBACK_ANCHORS = [
    (78.9811, 30.2844),  # Rudraprayag
    (78.9000, 30.2400),
    (78.7833, 30.2222),  # Srinagar (Garhwal)
    (78.6800, 30.1900),
    (78.5981, 30.1469),  # Devprayag
    (78.4300, 30.1300),
    (78.2676, 30.0869),  # Rishikesh
]


def load_centreline(path, dst_crs):
    """Return the centreline as a list of (x, y) in dst_crs."""
    if path:
        with open(path, encoding="utf-8") as fh:
            gj = json.load(fh)
        feats = gj.get("features", [gj])
        coords = []
        for f in feats:
            g = f.get("geometry", f)
            if g["type"] == "LineString":
                coords += g["coordinates"]
            elif g["type"] == "MultiLineString":
                for part in g["coordinates"]:
                    coords += part
        if not coords:
            sys.exit(f"\n{path} holds no LineString. Digitise the centreline as "
                     "a line layer, not a polygon.\n")
        src_crs = gj.get("crs", {}).get("properties", {}).get("name", "EPSG:4326")
    else:
        coords = list(FALLBACK_ANCHORS)
        src_crs = "EPSG:4326"
        print("no --centreline given, using the seven town anchors.\n"
              "  These are approximate. Re-run once the real centreline exists.")

    tr = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return [tr.transform(x, y) for x, y in coords]


def walk(line, step_m):
    """Points every step_m along a polyline, plus the far end."""
    pts, carry = [line[0]], 0.0
    for (x0, y0), (x1, y1) in zip(line[:-1], line[1:]):
        seg = float(np.hypot(x1 - x0, y1 - y0))
        if seg == 0:
            continue
        d = step_m - carry
        while d <= seg:
            t = d / seg
            pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
            d += step_m
        carry = (carry + seg) % step_m
    pts.append(line[-1])
    return pts


def classify(arr, edges, labels):
    out = np.full(arr.shape, "", dtype=object)
    for i, lab in enumerate(labels):
        out[(arr >= edges[i]) & (arr < edges[i + 1])] = lab
    return out


def read_classified(strata_dir):
    """Read every stratum raster once, classified, on its own 30 m grid."""
    dims, ref = {}, None
    for dim, fname in LAYER.items():
        path = os.path.join(strata_dir, fname)
        if not os.path.exists(path):
            sys.exit(f"\nmissing {path}\nRun 03_terrain_strata.py first.\n")
        with rasterio.open(path) as src:
            a = src.read(1).astype("float64")
            if src.nodata is not None:
                a[a == src.nodata] = np.nan
            if ref is None:
                ref = (src.transform, src.crs, a.shape)
        dims[dim] = classify(a, *BINS[dim])

    ls_path = os.path.join(strata_dir, "strata_layover_shadow.tif")
    if os.path.exists(ls_path):
        with rasterio.open(ls_path) as src:
            ls = np.nan_to_num(src.read(1), nan=0).astype("uint8")
        names = np.array(["clean", "layover", "shadow"], dtype=object)
        dims["layover_shadow"] = np.choose(np.clip(ls, 0, 2), names)
    return dims, ref


def compose(dims, transform, shape, cx, cy, half):
    """Stratum counts inside one window. Returns dict and total cell count."""
    r0, c0 = rowcol(transform, cx - half, cy + half)
    r1, c1 = rowcol(transform, cx + half, cy - half)
    r0, r1 = max(0, min(r0, r1)), min(shape[0], max(r0, r1))
    c0, c1 = max(0, min(c0, c1)), min(shape[1], max(c0, c1))
    if r1 <= r0 or c1 <= c0:
        return {}, 0
    out, n = {}, (r1 - r0) * (c1 - c0)
    for dim, cls in dims.items():
        sub = cls[r0:r1, c0:c1]
        for lab in set(sub.ravel().tolist()) - {""}:
            out[(dim, lab)] = int(np.count_nonzero(sub == lab))
    return out, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strata", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--pred", default=None)
    ap.add_argument("--centreline", default=None)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--size", type=float, default=5.0, help="window side, km")
    ap.add_argument("--spacing", type=float, default=None, help="min km apart")
    ap.add_argument("--min-share", type=float, default=0.02,
                    help="ignore a stratum holding less than this share of a "
                         "window, so a handful of stray cells cannot make a "
                         "window look diverse")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    side = args.size * 1000.0
    half = side / 2.0
    spacing = (args.spacing if args.spacing is not None else args.size) * 1000.0

    with rasterio.open(args.grid) as src:
        grid_crs, grid_res, grid_tf = src.crs, src.res, src.transform
    print(f"target grid {grid_crs}, {abs(grid_res[0]):.1f} m cells")

    dims, (s_tf, s_crs, s_shape) = read_classified(args.strata)
    if s_crs != grid_crs:
        sys.exit(f"\nstrata are {s_crs}, the flood grid is {grid_crs}.\n"
                 "They have to match before windows mean anything.\n")

    line = load_centreline(args.centreline, grid_crs)
    cands = walk(line, spacing / 2.0)   # denser than spacing, thinned later
    print(f"{len(cands)} candidate positions along the reach")

    # Snap centres to the flood grid so window edges fall on cell boundaries.
    # A window that straddles half a cell rasterises differently at each edge
    # and the difference is invisible until the metrics disagree by a pixel.
    res = abs(grid_res[0])
    ox, oy = grid_tf.c, grid_tf.f
    snapped = [(ox + round((x - ox) / res) * res,
                oy + round((y - oy) / res) * res) for x, y in cands]

    rows = []
    for i, (cx, cy) in enumerate(snapped):
        comp, n = compose(dims, s_tf, s_shape, cx, cy, half)
        if n == 0:
            continue
        strata = {k for k, v in comp.items() if v / n >= args.min_share}
        rows.append({"i": i, "cx": cx, "cy": cy, "cells": n,
                     "strata": strata, "comp": comp})

    if not rows:
        sys.exit("\nno candidate window overlapped the strata. Check that the "
                 "centreline and the strata cover the same ground.\n")

    # Greedy set cover, with a spacing constraint. At each step take the window
    # that adds the most strata not yet covered; break ties on the one furthest
    # from everything already chosen, which spreads them along the reach instead
    # of clustering them where the terrain happens to be most varied.
    chosen, covered = [], set()
    while len(chosen) < args.n:
        best, best_key = None, None
        for r in rows:
            if any(np.hypot(r["cx"] - c["cx"], r["cy"] - c["cy"]) < spacing
                   for c in chosen):
                continue
            gain = len(r["strata"] - covered)
            far = min([np.hypot(r["cx"] - c["cx"], r["cy"] - c["cy"])
                       for c in chosen], default=0.0)
            key = (gain, far)
            if best_key is None or key > best_key:
                best, best_key = r, key
        if best is None:
            print(f"stopped at {len(chosen)} windows: no position left that is "
                  f"{args.spacing or args.size} km clear of the others")
            break
        chosen.append(best)
        covered |= best["strata"]

    all_strata = set()
    for r in rows:
        all_strata |= r["strata"]
    missing = sorted(all_strata - covered)

    # --- predicted flood per window, reported only ------------------------
    pred_counts = {}
    if args.pred:
        with rasterio.open(args.pred) as src:
            pred = src.read(1)
            p_tf, p_shape = src.transform, pred.shape
        for k, c in enumerate(chosen):
            r0, c0 = rowcol(p_tf, c["cx"] - half, c["cy"] + half)
            r1, c1 = rowcol(p_tf, c["cx"] + half, c["cy"] - half)
            r0, r1 = max(0, min(r0, r1)), min(p_shape[0], max(r0, r1))
            c0, c1 = max(0, min(c0, c1)), min(p_shape[1], max(c0, c1))
            sub = pred[r0:r1, c0:c1]
            pred_counts[k] = int(np.count_nonzero(sub == 1))

    # --- write ------------------------------------------------------------
    feats = []
    for k, c in enumerate(chosen, start=1):
        cx, cy = c["cx"], c["cy"]
        ring = [[cx - half, cy - half], [cx + half, cy - half],
                [cx + half, cy + half], [cx - half, cy + half],
                [cx - half, cy - half]]
        props = {"window": k, "side_km": args.size,
                 "centre_x": round(cx, 1), "centre_y": round(cy, 1)}
        if args.pred:
            props["pred_flood_cells"] = pred_counts[k - 1]
        feats.append({"type": "Feature", "properties": props,
                      "geometry": {"type": "Polygon", "coordinates": [ring]}})

    gj = {"type": "FeatureCollection",
          "crs": {"type": "name",
                  "properties": {"name": str(grid_crs)}},
          "features": feats}
    gj_path = os.path.join(args.out, "reference_windows.geojson")
    with open(gj_path, "w", encoding="utf-8") as fh:
        json.dump(gj, fh, indent=1)

    # Column order is fixed once, from every stratum seen anywhere in the
    # corridor, so the two events produce CSVs with the same columns and can be
    # compared side by side.
    seen = sorted({(d, l) for r in rows for (d, l) in r["comp"]})
    csv_path = os.path.join(args.out, "reference_windows.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["window", "centre_x", "centre_y", "cells",
                    "pred_flood_cells"] + [f"{d}:{l}" for d, l in seen])
        for k, c in enumerate(chosen, start=1):
            n = c["cells"]
            shares = [round(c["comp"].get(key, 0) / n, 4) for key in seen]
            w.writerow([k, round(c["cx"], 1), round(c["cy"], 1), n,
                        pred_counts.get(k - 1, "")] + shares)

    # Corridor area from the strata run, not fixed text (the old line said
    # "about 808 km2", the retired corridor; corrected 7 October 2026).
    try:
        with open(os.path.join(args.strata, "strata_meta.json"),
                  encoding="utf-8") as fh:
            corridor_km2 = f"{json.load(fh)['valid_area_km2']:.0f} km2"
    except (OSError, KeyError, ValueError):
        corridor_km2 = "unknown size (strata_meta.json not read)"

    lines = [f"# Reference windows, {os.path.basename(args.out)}", "",
             f"{len(chosen)} windows of {args.size} x {args.size} km, "
             f"minimum {args.spacing or args.size} km between centres.",
             f"Total digitising area {len(chosen) * args.size ** 2:.0f} km2 "
             f"against a corridor of {corridor_km2}.", "",
             "Placed by terrain only. The flood prediction was not used to "
             "choose them.", "",
             "| Window | Centre E | Centre N | Dominant strata | Predicted "
             "flood cells |", "|---|---|---|---|---|"]
    for k, c in enumerate(chosen, start=1):
        n = c["cells"]
        top = sorted(((v / n, d, l) for (d, l), v in c["comp"].items()),
                     reverse=True)[:3]
        desc = ", ".join(f"{l} {s:.0%}" for s, d, l in top)
        lines.append(f"| {k} | {c['cx']:.0f} | {c['cy']:.0f} | {desc} | "
                     f"{pred_counts.get(k - 1, 'n/a')} |")
    lines += ["", f"Strata covered: {len(covered)} of {len(all_strata)}."]
    if missing:
        lines += ["", "**Not represented by any chosen window:** " +
                  ", ".join(f"{d}:{l}" for d, l in missing) + ".",
                  "", "These strata exist in the corridor but no window holds "
                  "enough of one to count. Either raise --n, or say in the "
                  "methods that the reference does not cover them and drop "
                  "their rows from the stratified table rather than reporting "
                  "them from a handful of cells."]
    md_path = os.path.join(args.out, "reference_windows.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print("\n".join(lines[2:]))
    print(f"\nwrote {gj_path}\nwrote {csv_path}\nwrote {md_path}")
    if args.pred:
        print("\npred_flood_cells is reported so you know what you are walking "
              "into.\nDo NOT re-run with different settings to get windows with "
              "more water in them.\nThat would place validation where the model "
              "already agrees and hide every miss.")


if __name__ == "__main__":
    main()
