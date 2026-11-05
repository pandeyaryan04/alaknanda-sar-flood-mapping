"""Jal-SAR extensions, M11 follow-up: the Beas maps against the NDEM image pair, and a valley scan.

Reads only the outputs of m11_beas.py run (nothing is downloaded, nothing is re-run).

What it does:
  site   the NDEM site at Mandi (Bhiuli). For a 2.2 km box round the site, Sentinel-1 VV is grouped
         into: permanent channel (JRC occurrence of 50 % or more), cells each model marks as flood,
         low ground near the channel that the U-Net leaves dry (HAND under 5 m, within 80 m of the
         channel), and the rest of the box. Writes site_stats.csv and BEAS_NDEM_SITE.png
  scan   the whole valley outline. Each model's flood cells are grouped into patches (cells touching
         at a side or a corner). Each patch gets its size, median HAND, median slope, distance to
         the permanent river, share of layover or shadow and median VV. A patch is marked "check"
         when any of these holds (rules set for this scan, not taken from the literature):
           H  median HAND above 15 m   (well above the valley floor)
           S  median slope above 20 degrees
           D  nearest cell more than 500 m from the permanent river
           L  more than half its cells in layover or shadow
         Writes valley_scan_<model>.csv for all patches, valley_scan_check_<model>.geojson (one point
         per marked patch of 5 cells or more, for QGIS, with empty
         'verdict' and 'note' fields to fill in by eye) and VALLEY_SCAN.md
         The marks only say where to look. Whether a patch is a false alarm is decided by eye.

Outputs in 06 Outputs\\jalsar\\extensions\\m11_beas\\ndem_check\\

Run:  python m11_ndem_site.py site
      python m11_ndem_site.py scan
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform as warp_xy
from rasterio.windows import from_bounds

HERE = Path(__file__).resolve().parent
M11 = HERE.parents[2] / "06 Outputs" / "jalsar" / "extensions" / "m11_beas"
OUT = M11 / "ndem_check"
SCENE = "S1_20230712_orb136.tif"
MODELS = ["unet", "rf", "terramind"]
# the NDEM site box, EPSG:32643 (placed from QGIS cursor positions over the site, 29 October 2026)
SITE = (682600.0, 3508300.0, 684800.0, 3510500.0)
PERM_PCT = 50.0
RULES = {"H": ("hand_med_m", 15.0, "median HAND above 15 m"),
         "S": ("slope_med_deg", 20.0, "median slope above 20 degrees"),
         "D": ("dist_river_m", 500.0, "more than 500 m from the permanent river"),
         "L": ("layshad_share", 0.5, "more than half in layover or shadow")}
MIN_POINT_CELLS = 5


def to_db(a):
    """COPERNICUS/S1_GRD, as m11_beas.py downloads it, is already in dB (median well below 0).
    Only a linear-power array (median above 0) is converted."""
    return a if np.nanmedian(a) < 0 else 10.0 * np.log10(np.clip(a, 1e-5, None))


def read_on(grid_path, src_path, window=None, nearest=True):
    """src_path resampled onto grid_path's grid (or a window of it), NaN for nodata."""
    with rasterio.open(grid_path) as g:
        w = window if window is not None else rasterio.windows.Window(0, 0, g.width, g.height)
        tr, shape, crs = g.window_transform(w), (int(w.height), int(w.width)), g.crs
    with rasterio.open(src_path) as s:
        out = np.full(shape, np.nan, dtype="float32")
        reproject(rasterio.band(s, 1), out, dst_transform=tr, dst_crs=crs,
                  resampling=Resampling.nearest if nearest else Resampling.bilinear,
                  src_nodata=s.nodata, dst_nodata=np.nan)
    return out


def flood(model, window=None):
    with rasterio.open(M11 / f"flood_{model}.tif") as d:
        return d.read(1, window=window)


def site():
    from scipy import ndimage
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    grid = M11 / f"flood_unet.tif"
    with rasterio.open(grid) as g:
        win = from_bounds(*SITE, g.transform).round_offsets().round_lengths()
    with rasterio.open(M11 / SCENE) as s:
        vv = to_db(s.read(1, window=from_bounds(*SITE, s.transform).round_offsets().round_lengths()).astype(float))
    jrc = read_on(grid, M11 / "strata" / "WATER_JRC_occurrence_30m.tif", win)
    hand = read_on(grid, M11 / "strata" / "strata_hand_m.tif", win)
    ls = read_on(grid, M11 / "strata" / "strata_layover_shadow.tif", win)
    maps = {m: flood(m, win) for m in MODELS}
    perm = np.nan_to_num(jrc) >= PERM_PCT
    near = ndimage.binary_dilation(perm, iterations=8)
    groups = {"permanent channel (JRC >= 50 %)": perm}
    for m in MODELS:
        groups[f"marked as flood by {m}"] = maps[m] == 1
    groups["low ground near channel (HAND < 5 m, within 80 m), not perm, not U-Net flood"] = (
        (np.nan_to_num(hand, nan=99) < 5) & near & ~perm & (maps["unet"] != 1))
    groups["rest of box (more than 80 m from channel)"] = ~near
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, gmask in groups.items():
        x = vv[gmask & ~np.isnan(vv)]
        rows.append({"group": k, "cells": int(x.size), "vv_median_db": round(float(np.median(x)), 1),
                     "vv_p25_db": round(float(np.percentile(x, 25)), 1),
                     "share_below_minus15db": round(float(np.mean(x < -15)), 2)})
    rows.append({"group": "layover or shadow cells in box", "cells": int(np.nansum(ls > 0)),
                 "vv_median_db": "", "vv_p25_db": "", "share_below_minus15db": ""})
    with open(OUT / "site_stats.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    for r in rows:
        print(f"  {r['group']:80s} {r['cells']:>7} {r['vv_median_db']:>6}")

    ext = [SITE[0], SITE[2], SITE[1], SITE[3]]
    cy = ListedColormap(["#00e5ff"])
    fig, ax = plt.subplots(1, 4, figsize=(22, 6.4))
    ax[0].imshow(vv, cmap="gray", vmin=-25, vmax=0, extent=ext)
    ax[0].set_title("Sentinel-1 VV, 12 Jul 2023 00:51 UTC\n(darker = smoother surface)")
    for a, m in zip(ax[1:], MODELS):
        a.imshow(vv, cmap="gray", vmin=-25, vmax=0, extent=ext)
        a.imshow(np.ma.masked_where(maps[m] != 1, maps[m]), cmap=cy, extent=ext, interpolation="nearest")
        a.set_title(f"flood_{m}.tif (cyan)")
    for a in ax:
        a.contour(perm, levels=[0.5], colors="yellow", extent=ext, origin="upper", linewidths=1)
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("Beas at Mandi (Bhiuli). Yellow line = JRC permanent water (occurrence >= 50%), removed from all "
                 "flood maps. Box EPSG:32643, E 682600-684800, N 3508300-3510500 (2.2 km square). North up.",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / "BEAS_NDEM_SITE.png", dpi=90)
    print(f"  wrote {OUT / 'site_stats.csv'} and {OUT / 'BEAS_NDEM_SITE.png'}")


def scan():
    from scipy import ndimage

    grid = M11 / "flood_unet.tif"
    with rasterio.open(grid) as g:
        tr, crs, cell = g.transform, g.crs, abs(g.transform.a)
    jrc = read_on(grid, M11 / "strata" / "WATER_JRC_occurrence_30m.tif")
    perm = np.nan_to_num(jrc) >= PERM_PCT
    del jrc
    dist = ndimage.distance_transform_edt(~perm).astype("float32") * cell
    hand = read_on(grid, M11 / "strata" / "strata_hand_m.tif")
    slope = read_on(grid, M11 / "strata" / "strata_slope_deg.tif")
    ls = read_on(grid, M11 / "strata" / "strata_layover_shadow.tif")
    with rasterio.open(M11 / SCENE) as s:
        vv = to_db(s.read(1).astype("float32"))
    OUT.mkdir(parents=True, exist_ok=True)
    eight = np.ones((3, 3), dtype=bool)
    summary = []
    for m in MODELS:
        fm = flood(m) == 1
        lab, n = ndimage.label(fm, structure=eight)
        idx = np.arange(1, n + 1)
        cells = ndimage.sum_labels(fm, lab, idx)
        hmed = ndimage.median(np.nan_to_num(hand, nan=np.nan), lab, idx)
        smed = ndimage.median(slope, lab, idx)
        dmin = ndimage.minimum(dist, lab, idx)
        lsh = ndimage.mean(np.nan_to_num(ls) > 0, lab, idx)
        vmed = ndimage.median(vv, lab, idx)
        cen = ndimage.center_of_mass(fm, lab, idx)
        rows = []
        for i in range(n):
            r_, c_ = cen[i]
            x, y = tr * (c_ + 0.5, r_ + 0.5)
            lon, lat = warp_xy(crs, "EPSG:4326", [x], [y])
            row = {"patch": int(idx[i]), "cells": int(cells[i]), "area_m2": int(cells[i] * cell * cell),
                   "e_32643": round(x, 1), "n_32643": round(y, 1),
                   "lon": round(lon[0], 6), "lat": round(lat[0], 6),
                   "hand_med_m": round(float(hmed[i]), 2), "slope_med_deg": round(float(smed[i]), 1),
                   "dist_river_m": round(float(dmin[i]), 0), "layshad_share": round(float(lsh[i]), 2),
                   "vv_med_db": round(float(vmed[i]), 1)}
            flags = [k for k, (col, lim, _) in RULES.items() if np.isfinite(row[col]) and row[col] > lim]
            row["flags"] = "".join(flags)
            row["check"] = "yes" if flags else ""
            rows.append(row)
        rows.sort(key=lambda r: -r["cells"])
        with open(OUT / f"valley_scan_{m}.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        pts = [r for r in rows if r["check"] and r["cells"] >= MIN_POINT_CELLS]
        gj = {"type": "FeatureCollection", "name": f"valley_scan_check_{m}",
              "features": [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
                            "properties": {**r, "rank": k + 1, "verdict": "", "note": ""}} for k, r in enumerate(pts)]}
        (OUT / f"valley_scan_check_{m}.geojson").write_text(json.dumps(gj), encoding="utf-8")
        tot = sum(r["cells"] for r in rows)
        s = {"model": m, "patches": n, "flood_km2": round(tot * cell * cell / 1e6, 3),
             "check_patches": sum(1 for r in rows if r["check"]),
             "check_km2": round(sum(r["cells"] for r in rows if r["check"]) * cell * cell / 1e6, 3),
             "check_points": len(pts)}
        for k in RULES:
            fl = [r for r in rows if k in r["flags"]]
            s[f"{k}_patches"] = len(fl)
            s[f"{k}_km2"] = round(sum(r["cells"] for r in fl) * cell * cell / 1e6, 3)
        summary.append(s)
        print(f"  {m}: {n} patches, {s['flood_km2']} km2; marked {s['check_patches']} patches, "
              f"{s['check_km2']} km2; points (5+ cells) {len(pts)}")

    md = ["# M11 Beas: valley scan of the flood maps", "",
          "Made by m11_ndem_site.py scan from the m11_beas.py outputs (pass 12 July 2023, orbit 136). "
          "A patch is a group of flood cells touching at a side or a corner. A patch is marked for checking "
          "when any rule below holds. The rules were set for this scan, not taken from the literature. "
          "A mark is not a verdict: whether a patch is a false alarm is decided by eye.", ""]
    md += [f"- {k}: {d}" for k, (_, _, d) in RULES.items()]
    md += ["", "| Model | Patches | Flood km2 | Marked patches | Marked km2 | Points for QGIS (5+ cells) | "
           + " | ".join(f"{k} patches / km2" for k in RULES) + " |",
           "|---" * (6 + len(RULES)) + "|"]
    for s in summary:
        md.append(f"| {s['model']} | {s['patches']:,} | {s['flood_km2']} | {s['check_patches']:,} | {s['check_km2']} | "
                  f"{s['check_points']} | " + " | ".join(f"{s[k + '_patches']} / {s[k + '_km2']}" for k in RULES) + " |")
    md += ["", "Files: valley_scan_<model>.csv (every patch, largest first), valley_scan_check_<model>.geojson "
           "(marked patches of 5 cells or more, ranked by size, with empty 'verdict' and 'note' fields)."]
    (OUT / "VALLEY_SCAN.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"  wrote {OUT / 'VALLEY_SCAN.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["site", "scan"])
    a = ap.parse_args()
    site() if a.what == "site" else scan()


if __name__ == "__main__":
    main()
