r"""
14_reservoir_outline.py
=======================
Remake the Srinagar reservoir outline, used by gis_tools\t04_reservoir.py
(RESERVOIR_AOI) and in the reservoir-stability check:

  05 Data\boundaries\RESERVOIR_srinagar_aoi.geojson

THE RULE (adopted 2 October 2026)
---------------------------------
  the corridor centreline from km 0 (Rudraprayag) to km 20.6 (the dam),
  buffered 300 m each side, round ends, clipped to corridor v2

Distances are measured along the centreline in EPSG:32644. The buffer uses 16
segments per quarter circle (the shapely default). Stored in EPSG:4326 like
the file in use.

USAGE
-----
  conda activate fyp
  cd /d "<project folder>\04 Code\FYP_code"
  python 14_reservoir_outline.py

It writes to 06 Outputs\remake\ and compares the result with the file in use.
It never overwrites the file in use. PASS means the two outlines cover the
same ground to within 1 m2 (the rest is rounding in the stored coordinates).

  --out       where to write (default 06 Outputs\remake\<same name>)
  --compare   file to compare with (default the file in use); "none" to skip
  --force     overwrite --out if it exists
"""

__stage__ = "Remake, reservoir outline"
__reads__ = ["corridor_centreline.geojson", "corridor_v2_3km_clipped.geojson"]
__writes__ = ["RESERVOIR_srinagar_aoi.geojson"]

import argparse
import sys
from pathlib import Path

try:
    import geopandas as gpd
    import shapely
    from shapely.ops import substring
except ImportError as e:
    sys.exit(f"\nMissing package: {e}.\nActivate the environment first:  "
             f"conda activate fyp\n")

ROOT = Path(__file__).resolve().parents[2]
BOUND = ROOT / "05 Data" / "boundaries"
NAME = "RESERVOIR_srinagar_aoi.geojson"
CRS_METRIC = "EPSG:32644"
CRS_STORE = "EPSG:4326"
KM_FROM, KM_TO = 0.0, 20.6
HALF_WIDTH_M = 300
QUAD_SEGS = 16
TOL_M2 = 1.0


def build():
    line = gpd.read_file(BOUND / "corridor_centreline.geojson").to_crs(CRS_METRIC)
    if len(line) != 1:
        sys.exit(f"corridor_centreline.geojson has {len(line)} features, "
                 f"expected 1.")
    line = line.geometry.iloc[0]
    seg = substring(line, KM_FROM * 1000, KM_TO * 1000)
    poly = seg.buffer(HALF_WIDTH_M, cap_style="round", join_style="round",
                      quad_segs=QUAD_SEGS)
    cor = gpd.read_file(BOUND / "corridor_v2_3km_clipped.geojson").to_crs(CRS_METRIC)
    poly = poly.intersection(shapely.union_all(cor.geometry.values))
    print(f"  centreline {line.length / 1000:.2f} km long; "
          f"segment km {KM_FROM} to {KM_TO}")
    print(f"  outline {poly.area / 1e6:.4f} km2")
    return gpd.GeoDataFrame(
        {"name": [f"Srinagar HEP impoundment, centreline km {KM_FROM:g} to "
                  f"{KM_TO:g} (dam), {HALF_WIDTH_M} m each side, clipped to "
                  f"corridor v2"],
         "made": ["14_reservoir_outline.py, from corridor_centreline.geojson"]},
        geometry=[poly], crs=CRS_METRIC)


def compare(new, path):
    old = gpd.read_file(path).to_crs(CRS_METRIC)
    a = shapely.union_all(old.geometry.values)
    b = new.geometry.iloc[0]
    sym = a.symmetric_difference(b).area
    print(f"  file in use : {a.area / 1e6:.4f} km2")
    print(f"  remake      : {b.area / 1e6:.4f} km2")
    print(f"  ground covered by one and not the other: {sym:.2f} m2")
    ok = sym <= TOL_M2
    print("  PASS: identical" if ok else "  DIFF: not identical")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "06 Outputs" / "remake" / NAME))
    ap.add_argument("--compare", default=str(BOUND / NAME))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    out = Path(a.out)
    if out.resolve() == (BOUND / NAME).resolve():
        sys.exit("--out is the file in use. Write somewhere else and compare.")
    if out.exists():
        if not a.force:
            sys.exit(f"{out} exists. Add --force to overwrite it.")
        out.unlink()

    gdf = build()
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_crs(CRS_STORE).to_file(out, driver="GeoJSON")
    print(f"  written: {out}")

    if a.compare.lower() == "none":
        return 0
    print("\nComparison with the file in use")
    # compare what was written, after the round trip through EPSG:4326
    written = gpd.read_file(out).to_crs(CRS_METRIC)
    return 0 if compare(written, Path(a.compare)) else 1


if __name__ == "__main__":
    sys.exit(main())
