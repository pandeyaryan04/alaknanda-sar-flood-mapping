r"""
12_permanent_water_mask.py
==========================
Remake the permanent-water mask, row 11 of the data register:

  05 Data\dem\WATER_PERMANENT_JRC50_S2DRY_10m.tif

THE RULE (adopted 2 October 2026)
---------------------------------
  permanent water =  [ dry-season Sentinel-2 water, 2021 OR 2023,
                       closed by 2 pixels, pieces of at least 2 ha ]
                     OR  JRC occurrence >= 50 %  (30 m, bilinear to 10 m)
                     AND inside corridor v2

  dry-season Sentinel-2 water, per year =
      MNDWI > 0  AND  B11 reflectance < 0.03  AND  scene class clear
  scene class clear = not 0 (no data), not 3, 8, 9, 10 (cloud, shadow,
      cirrus), not 11 (snow). Same thresholds as gis_tools\.env.

The closing and the 2 ha filter are applied to the Sentinel-2 water only,
before the JRC layer is added. Output: uint8, 100 = permanent water, 0 = not,
on the 10 m grid of the dry mosaics.

USAGE
-----
  conda activate fyp
  cd /d "<project folder>\04 Code\FYP_code"
  python 12_permanent_water_mask.py

It writes to 06 Outputs\remake\ and compares the result with the file in use.
It never overwrites the file in use. PASS means every cell is identical.

  --out       where to write (default 06 Outputs\remake\<same name>)
  --compare   file to compare with (default the file in use); "none" to skip
  --force     overwrite --out if it exists
"""

__stage__ = "Remake, permanent-water mask"
__reads__ = ["S2_<year>_dry_mosaic.tif", "S2_<year>_dry_mndwi.tif",
             "WATER_JRC_occurrence_30m.tif", "corridor_v2_3km_clipped.geojson"]
__writes__ = ["WATER_PERMANENT_JRC50_S2DRY_10m.tif"]

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from rasterio.warp import Resampling, reproject
    from scipy.ndimage import binary_closing, label
except ImportError as e:
    sys.exit(f"\nMissing package: {e}.\nActivate the environment first:  "
             f"conda activate fyp\n")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "05 Data"
NAME = "WATER_PERMANENT_JRC50_S2DRY_10m.tif"

YEARS = (2021, 2023)
MNDWI_WATER = 0.00
SWIR_WATER = 0.03
S2_SCALE = 10000
SCL_NODATA = 0
SCL_CLOUD = (3, 8, 9, 10)
SCL_SNOW = (11,)
BAND_B11, BAND_SCL = 5, 6          # B2,B3,B4,B8,B11,SCL in the dry mosaic
CLOSE_PX = 2
MIN_HA = 2.0
JRC_MIN = 50
VALUE = 100


def dry_water(year):
    """MNDWI > 0 and B11 < 0.03 and clear, for one dry-season mosaic."""
    s2 = DATA / "sentinel2"
    with rasterio.open(s2 / f"S2_{year}_dry_mndwi.tif") as s:
        mndwi = s.read(1)
        grid = {"transform": s.transform, "crs": s.crs, "shape": s.shape}
    with rasterio.open(s2 / f"S2_{year}_dry_mosaic.tif") as s:
        if s.shape != grid["shape"] or s.transform != grid["transform"]:
            sys.exit(f"S2_{year}_dry_mosaic.tif and its MNDWI file are on "
                     f"different grids.")
        b11 = s.read(BAND_B11).astype("float32")
        if s.nodata is not None:
            b11[b11 == s.nodata] = np.nan
        b11 /= S2_SCALE
        scl = s.read(BAND_SCL)
    clear = ((scl != SCL_NODATA) & ~np.isin(scl, SCL_CLOUD)
             & ~np.isin(scl, SCL_SNOW))
    return (mndwi > MNDWI_WATER) & (b11 < SWIR_WATER) & clear, grid


def clean(water, res_m):
    """Close by CLOSE_PX pixels, keep 8-connected pieces of MIN_HA or more."""
    w = binary_closing(water, structure=np.ones((3, 3), bool),
                       iterations=CLOSE_PX)
    lab, n = label(w, structure=np.ones((3, 3), int))
    if n == 0:
        return w
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    min_cells = int(MIN_HA * 1e4 / (res_m * res_m))
    return np.isin(lab, np.nonzero(sizes >= min_cells)[0])


def jrc_on_grid(grid):
    out = np.zeros(grid["shape"], "float32")
    with rasterio.open(DATA / "dem" / "WATER_JRC_occurrence_30m.tif") as s:
        reproject(rasterio.band(s, 1), out,
                  src_transform=s.transform, src_crs=s.crs,
                  dst_transform=grid["transform"], dst_crs=grid["crs"],
                  resampling=Resampling.bilinear,
                  src_nodata=s.nodata, dst_nodata=0)
    return out >= JRC_MIN


def corridor_on_grid(grid):
    g = gpd.read_file(DATA / "boundaries" / "corridor_v2_3km_clipped.geojson")
    g = g.to_crs(grid["crs"])
    return rasterize([(geom, 1) for geom in g.geometry],
                     out_shape=grid["shape"], transform=grid["transform"],
                     fill=0, dtype="uint8", all_touched=False).astype(bool)


def compare(new, path, res_m):
    with rasterio.open(path) as s:
        old = s.read(1)
        same_grid = s.shape == new.shape
    if not same_grid:
        print(f"  DIFF: {path.name} is {old.shape}, the remake is {new.shape}")
        return False
    diff = int((old != new).sum())
    km2 = lambda a: a.sum() * res_m * res_m / 1e6
    print(f"  file in use : {km2(old > 0):.3f} km2")
    print(f"  remake      : {km2(new > 0):.3f} km2")
    print(f"  cells that differ: {diff}")
    print("  PASS: identical" if diff == 0 else "  DIFF: not identical")
    return diff == 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "06 Outputs" / "remake" / NAME))
    ap.add_argument("--compare", default=str(DATA / "dem" / NAME))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    out = Path(a.out)
    if out.resolve() == (DATA / "dem" / NAME).resolve():
        sys.exit("--out is the file in use. Write somewhere else and compare.")
    if out.exists() and not a.force:
        sys.exit(f"{out} exists. Add --force to overwrite it.")

    wet = None
    grid = None
    for y in YEARS:
        w, g = dry_water(y)
        if grid is None:
            grid = g
        elif g["shape"] != grid["shape"] or g["transform"] != grid["transform"]:
            sys.exit("The 2021 and 2023 dry mosaics are on different grids.")
        res = abs(grid["transform"].a)
        print(f"  {y} dry-season water: {w.sum() * res * res / 1e6:.3f} km2")
        wet = w if wet is None else (wet | w)

    res = abs(grid["transform"].a)
    s2 = clean(wet, res)
    jrc = jrc_on_grid(grid)
    inside = corridor_on_grid(grid)
    mask = (s2 | jrc) & inside
    print(f"  Sentinel-2 water after closing and 2 ha: "
          f"{s2.sum() * res * res / 1e6:.3f} km2")
    print(f"  JRC >= {JRC_MIN} %: {jrc.sum() * res * res / 1e6:.3f} km2")
    print(f"  permanent water, inside corridor v2: "
          f"{mask.sum() * res * res / 1e6:.3f} km2")

    arr = np.where(mask, VALUE, 0).astype("uint8")
    out.parent.mkdir(parents=True, exist_ok=True)
    prof = {"driver": "GTiff", "dtype": "uint8", "count": 1,
            "width": arr.shape[1], "height": arr.shape[0],
            "crs": grid["crs"], "transform": grid["transform"],
            "tiled": True, "blockxsize": 256, "blockysize": 256,
            "compress": "lzw"}
    with rasterio.open(out, "w", **prof) as d:
        d.write(arr, 1)
        d.update_tags(meaning=(
            "100 = permanent water, 0 = not. Union of JRC occurrence >= 50 "
            "(bilinear to 10 m) and dry-season Sentinel-2 water (MNDWI>0, "
            "B11<0.03, SCL clear) from S2_2021_dry and S2_2023_dry mosaics, "
            "closed 2 px, components >= 2 ha. Cut to corridor v2. "
            "Made by 12_permanent_water_mask.py."))
    print(f"  written: {out}")

    if a.compare.lower() == "none":
        return 0
    print("\nComparison with the file in use")
    return 0 if compare(arr, Path(a.compare), res) else 1


if __name__ == "__main__":
    sys.exit(main())
