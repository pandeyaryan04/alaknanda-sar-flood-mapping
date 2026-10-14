r"""
16_landcover_corridor.py
========================
Land cover inside the corridor, from the two land-cover sources (register
rows 12 and 13). Remakes, on corridor v2, the files first made on the old
5 km corridor (807 km2) on 6 and 9 September 2026:

  LULC50K_corridor_clip.geojson                  Bhuvan LULC 50K polygons, cut to the corridor
  LANDCOVER_LULC50K_area_by_class_corridor.csv   area and share by LULC 50K class
  LANDCOVER_WorldCover_area_by_class_corridor.csv area and share by ESA WorldCover class
      (replaces LANDCOVER_area_by_class_corridor.csv and
       LANDCOVER_class_histogram_corridor.csv, which were Earth Engine exports)

THE METHOD
----------
LULC 50K: the shapefile is cut to the corridor in its own coordinates
(EPSG:4326), then each piece is measured in EPSG:32644. This is the method of
the 9 September note, LULC50K - verification and comparison. Run on the old
corridor it gives the old table to within 0.0001 km2 per class.

WorldCover: cells of LC_ESA_WorldCover_10m.tif whose centre is inside the
corridor, counted by class, times 100 m2. Earth Engine weighted the edge
cells by the share inside, so on the old corridor this gives each class
within 0.05 per cent of the old export, not identical.

USAGE
-----
  conda activate fyp
  cd /d "<project folder>\04 Code\FYP_code"

  1. Prove the method on the old corridor (writes to a check folder):
  python 16_landcover_corridor.py --corridor "..\..\99 Archive\superseded 2026-09-21\corridor_v1_807km2_2026-08-21.geojson" --worldcover "..\..\99 Archive\superseded 2026-09-30\old corridor exports\landcover\LC_ESA_WorldCover_10m.tif" --out-dir "..\..\06 Outputs\remake\landcover_check_v1" --compare-lulc "..\..\05 Data\landcover\LANDCOVER_LULC50K_area_by_class_corridor.csv"

  2. Make the corridor v2 files:
  python 16_landcover_corridor.py

  --corridor       default 05 Data\boundaries\corridor_v2_3km_clipped.geojson
  --worldcover     default 05 Data\landcover\LC_ESA_WorldCover_10m.tif
  --out-dir        default 06 Outputs\remake\landcover
  --compare-lulc   optional: an LULC 50K table to compare class by class
  --force          overwrite outputs if they exist
"""

__stage__ = "Land cover inside the corridor"
__reads__ = ["aoi_UK_LULC50K_1516.shp", "LC_ESA_WorldCover_10m.tif",
             "corridor_v2_3km_clipped.geojson"]
__writes__ = ["LULC50K_corridor_clip.geojson",
              "LANDCOVER_LULC50K_area_by_class_corridor.csv",
              "LANDCOVER_WorldCover_area_by_class_corridor.csv"]

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import geopandas as gpd
    import pandas as pd
    import rasterio
    from rasterio.features import rasterize
except ImportError as e:
    sys.exit(f"\nMissing package: {e}.\nActivate the environment first:  "
             f"conda activate fyp\n")

ROOT = Path(__file__).resolve().parents[2]
LC = ROOT / "05 Data" / "landcover"
LULC_SHP = LC / "lulc50k" / "aoi_UK_LULC50K_1516.shp"
WORLDCOVER = LC / "LC_ESA_WorldCover_10m.tif"
CORRIDOR = ROOT / "05 Data" / "boundaries" / "corridor_v2_3km_clipped.geojson"
CRS_METRIC = "EPSG:32644"
LULC_TOL_KM2 = 0.0001

WC_NAMES = {10: "Tree cover", 20: "Shrubland", 30: "Grassland",
            40: "Cropland", 50: "Built-up", 60: "Bare / sparse vegetation",
            70: "Snow and ice", 80: "Permanent water bodies",
            90: "Herbaceous wetland", 95: "Mangroves", 100: "Moss and lichen"}


def lulc(corridor):
    src = gpd.read_file(LULC_SHP)
    cor = corridor.to_crs(src.crs)
    clip = gpd.clip(src, cor)
    clip = clip[~clip.is_empty].copy()
    clip["area_km2"] = clip.to_crs(CRS_METRIC).area / 1e6
    clip = clip[["DIST_OFF", "LU_Webcode", "DESCR_1", "DESCR_2", "area_km2",
                 "geometry"]].reset_index(drop=True)

    t = (clip.groupby(["LU_Webcode", "DESCR_1", "DESCR_2"], as_index=False)
         ["area_km2"].sum().sort_values("area_km2", ascending=False))
    total = t["area_km2"].sum()
    table = pd.DataFrame({
        "lu_webcode": t["LU_Webcode"].astype(int),
        "level_1": t["DESCR_1"], "level_2": t["DESCR_2"],
        "area_km2": t["area_km2"].round(4),
        "share_pct": (100 * t["area_km2"] / total).round(3)})
    table = pd.concat([table, pd.DataFrame([{
        "lu_webcode": None, "level_1": "TOTAL", "level_2": None,
        "area_km2": round(total, 4), "share_pct": 100.0}])], ignore_index=True)
    table["lu_webcode"] = table["lu_webcode"].astype("Int64")
    return clip, table


def worldcover(corridor, path):
    with rasterio.open(path) as s:
        a = s.read(1)
        tr, crs = s.transform, s.crs
    res = abs(tr.a)
    inside = rasterize([(g, 1) for g in corridor.to_crs(crs).geometry],
                       out_shape=a.shape, transform=tr, fill=0,
                       dtype="uint8", all_touched=False).astype(bool)
    vals, cells = np.unique(a[inside & (a > 0)], return_counts=True)
    blank = int((inside & (a == 0)).sum())
    t = pd.DataFrame({"class": vals.astype(int), "cells": cells})
    t["name"] = t["class"].map(WC_NAMES)
    t["area_km2"] = (t["cells"] * res * res / 1e6).round(4)
    total = t["area_km2"].sum()
    t["share_pct"] = (100 * t["area_km2"] / total).round(3)
    t = t.sort_values("area_km2", ascending=False)[
        ["class", "name", "cells", "area_km2", "share_pct"]]
    t = pd.concat([t, pd.DataFrame([{
        "class": None, "name": "TOTAL", "cells": int(t["cells"].sum()),
        "area_km2": round(total, 4), "share_pct": 100.0}])], ignore_index=True)
    t["class"] = t["class"].astype("Int64")
    return t, blank * res * res / 1e6


def compare_lulc(new, path):
    old = pd.read_csv(path)
    old = old[old["lu_webcode"].notna()].copy()
    old["lu_webcode"] = old["lu_webcode"].astype(int)
    n = new[new["lu_webcode"].notna()].copy()
    n["lu_webcode"] = n["lu_webcode"].astype(int)
    m = old.merge(n, on="lu_webcode", how="outer", suffixes=("_old", "_new"))
    m["diff"] = (m["area_km2_new"].fillna(0) - m["area_km2_old"].fillna(0)).abs()
    worst = float(m["diff"].max())
    print(f"  classes: old {len(old)}, new {len(n)}; largest class difference "
          f"{worst:.4f} km2")
    ok = len(old) == len(n) and worst <= LULC_TOL_KM2
    print("  PASS: same table" if ok else "  DIFF: not the same table")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corridor", default=str(CORRIDOR))
    ap.add_argument("--out-dir", default=str(ROOT / "06 Outputs" / "remake" / "landcover"))
    ap.add_argument("--worldcover", default=str(WORLDCOVER))
    ap.add_argument("--compare-lulc", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    out_dir = Path(a.out_dir)
    if out_dir.resolve() == LC.resolve():
        sys.exit("--out-dir is 05 Data\\landcover, where the files in use are. "
                 "Write somewhere else, then copy the files in by hand (RUN ORDER.md, step 3.8).")
    outs = {"clip": out_dir / "LULC50K_corridor_clip.geojson",
            "lulc": out_dir / "LANDCOVER_LULC50K_area_by_class_corridor.csv",
            "wc": out_dir / "LANDCOVER_WorldCover_area_by_class_corridor.csv"}
    for p in outs.values():
        if p.exists():
            if not a.force:
                sys.exit(f"{p} exists. Add --force to overwrite it.")
            p.unlink()

    corridor = gpd.read_file(a.corridor)
    cor_km2 = corridor.to_crs(CRS_METRIC).area.sum() / 1e6
    print(f"  corridor: {Path(a.corridor).name}, {cor_km2:.2f} km2")

    clip, lt = lulc(corridor)
    print(f"  LULC 50K: {len(clip)} pieces, {lt['area_km2'].iloc[-1]:.4f} km2")
    for _, r in lt.iloc[:-1].iterrows():
        print(f"    {r['level_2']:32s} {r['area_km2']:9.4f} km2  {r['share_pct']:6.3f} %")

    wt, blank_km2 = worldcover(corridor, Path(a.worldcover))
    print(f"  WorldCover: {wt['area_km2'].iloc[-1]:.4f} km2 classified, "
          f"{blank_km2:.4f} km2 of the corridor blank in the file")
    for _, r in wt.iloc[:-1].iterrows():
        print(f"    {r['name']:32s} {r['area_km2']:9.4f} km2  {r['share_pct']:6.3f} %")

    out_dir.mkdir(parents=True, exist_ok=True)
    clip.to_file(outs["clip"], driver="GeoJSON")
    lt.to_csv(outs["lulc"], index=False)
    wt.to_csv(outs["wc"], index=False)
    for p in outs.values():
        print(f"  written: {p}")

    if a.compare_lulc:
        print(f"\nComparison with {Path(a.compare_lulc).name}")
        return 0 if compare_lulc(lt, a.compare_lulc) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
