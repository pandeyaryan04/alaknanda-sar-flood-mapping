r"""
13_cloudgap_holes.py
====================
Remake the cloud-gap polygons the digitisers were told not to draw in, and
that 06_rasterise_reference.py marks as nodata (--holes):

  05 Data\sentinel2\S2_<year>_event_cloudgap.gpkg, layer "holes"

THE RULE (2 October 2026)
-------------------------
One polygon per 4-connected patch of value 1 in

  05 Data\sentinel2\S2_<year>_event_gapmask.tif
      1 = clear in no event scene, inside the corridor  (made by RUN MOSAICS)

No smoothing, no size filter. Columns: source, note.

USAGE
-----
  conda activate fyp
  cd /d "<project folder>\04 Code\FYP_code"
  python 13_cloudgap_holes.py --year 2021
  python 13_cloudgap_holes.py --year 2023

It writes to 06 Outputs\remake\ and compares the result with the file in use.
It never overwrites the file in use. PASS means the same number of polygons
and the same area, and the two sets of polygons cover exactly the same ground.

  --out       where to write (default 06 Outputs\remake\<same name>)
  --compare   file to compare with (default the file in use); "none" to skip
  --force     overwrite --out if it exists
"""

__stage__ = "Remake, cloud-gap holes"
__reads__ = ["S2_<year>_event_gapmask.tif"]
__writes__ = ["S2_<year>_event_cloudgap.gpkg"]

import argparse
import sys
from pathlib import Path

try:
    import geopandas as gpd
    import rasterio
    import shapely
    from rasterio.features import shapes
    from shapely.geometry import shape
except ImportError as e:
    sys.exit(f"\nMissing package: {e}.\nActivate the environment first:  "
             f"conda activate fyp\n")

ROOT = Path(__file__).resolve().parents[2]
S2 = ROOT / "05 Data" / "sentinel2"
LAYER = "holes"
NOTE = "no usable optical image on either event date; do not draw here"
AREA_TOL_M2 = 1.0      # floating-point slack when comparing areas


def build(year):
    src = S2 / f"S2_{year}_event_gapmask.tif"
    with rasterio.open(src) as s:
        a = s.read(1)
        tr, crs = s.transform, s.crs
    polys = [shape(geom) for geom, v in
             shapes(a, mask=(a == 1), transform=tr, connectivity=4) if v == 1]
    return gpd.GeoDataFrame(
        {"source": [f"{src.name}, value 1"] * len(polys),
         "note": [NOTE] * len(polys)},
        geometry=polys, crs=crs)


def compare(new, path):
    old = gpd.read_file(path, layer=LAYER).to_crs(new.crs)
    sym = shapely.union_all(new.geometry.values).symmetric_difference(
        shapely.union_all(old.geometry.values)).area
    print(f"  file in use : {len(old)} polygons, {old.area.sum() / 1e6:.4f} km2")
    print(f"  remake      : {len(new)} polygons, {new.area.sum() / 1e6:.4f} km2")
    print(f"  ground covered by one and not the other: {sym:.1f} m2")
    ok = len(old) == len(new) and sym <= AREA_TOL_M2
    print("  PASS: identical" if ok else "  DIFF: not identical")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", required=True, choices=["2021", "2023"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    name = f"S2_{a.year}_event_cloudgap.gpkg"
    in_use = S2 / name
    out = Path(a.out) if a.out else ROOT / "06 Outputs" / "remake" / name
    cmp_path = a.compare or str(in_use)
    if out.resolve() == in_use.resolve():
        sys.exit("--out is the file in use. Write somewhere else and compare.")
    if out.exists():
        if not a.force:
            sys.exit(f"{out} exists. Add --force to overwrite it.")
        out.unlink()

    gdf = build(a.year)
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out, layer=LAYER, driver="GPKG")
    print(f"  {a.year}: {len(gdf)} polygons, {gdf.area.sum() / 1e6:.4f} km2")
    print(f"  written: {out}")

    if cmp_path.lower() == "none":
        return 0
    print("\nComparison with the file in use")
    return 0 if compare(gdf, Path(cmp_path)) else 1


if __name__ == "__main__":
    sys.exit(main())
