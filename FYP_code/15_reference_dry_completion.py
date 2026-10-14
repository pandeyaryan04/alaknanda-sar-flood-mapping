r"""
15_reference_dry_completion.py
==============================
Remake the one step applied to the hand-drawn reference after it came back:
the February (dry) layer completed by rule, decided on 7 October 2026 after
the quality check of the returned drawings.

Reads the original drawings, untouched, straight from the RETURN zips in
  00 New work\Digitising handoff\received\
and writes two GeoPackages:

  QA_dry_water_not_drawn_<year>.gpkg              layer dry_water_not_drawn
  reference_<year>_event_CORRECTED_dry_added.gpkg layers water_event, water_dry

THE RULE
--------
On the 10 m grid of S2_<year>_dry_mndwi.tif:

  missing dry water =  February MNDWI > 0
                       AND inside a drawn water_event polygon
                       AND NOT inside a drawn water_dry polygon
                       AND inside a digitising window
                       AND inside corridor v2

  polygons are cut along 4-connected cell edges; single cells are dropped
  (pieces of 2 cells, 200 m2, or more are kept)

The completed file is the drawings unchanged, plus the missing pieces added
to water_dry, each with its window number and a note. water_event is copied
unchanged. Polygons are rasterised without all_touched, as in
06_rasterise_reference.py.

USAGE
-----
  conda activate fyp
  cd /d "<project folder>\04 Code\FYP_code"
  python 15_reference_dry_completion.py --year 2021
  python 15_reference_dry_completion.py --year 2023

It writes to 06 Outputs\remake\ and compares with the file in use,
06 Outputs\gis_tools\qgis\<year>_event\reference_<year>_event.gpkg. It never
overwrites the file in use. PASS means both layers cover exactly the same
ground (to 1 m2) and have the same number of polygons. Same ground means
06_rasterise_reference.py gives the same reference_<year>.tif.

  --zip       the RETURN zip (default: RETURN_<year>_reference_drawing.zip)
  --compare   file to compare with (default the file in use); "none" to skip
  --out-dir   default 06 Outputs\remake
  --force     overwrite outputs if they exist
"""

__stage__ = "Remake, February completion of the reference"
__reads__ = ["RETURN_<name>_<year>Event.zip", "S2_<year>_dry_mndwi.tif",
             "reference_windows.geojson", "corridor_v2_3km_clipped.geojson"]
__writes__ = ["QA_dry_water_not_drawn_<year>.gpkg",
              "reference_<year>_event_CORRECTED_dry_added.gpkg"]

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

try:
    import geopandas as gpd
    import pandas as pd
    import rasterio
    import shapely
    from rasterio.features import rasterize, shapes
    from scipy.ndimage import label
    from shapely.geometry import shape
except ImportError as e:
    sys.exit(f"\nMissing package: {e}.\nActivate the environment first:  "
             f"conda activate fyp\n")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "05 Data"
RECEIVED = ROOT / "00 New work" / "Digitising handoff" / "received"
ZIPS = {"2021": RECEIVED / "RETURN_2021_reference_drawing.zip",
        "2023": RECEIVED / "RETURN_2023_reference_drawing.zip"}
WINDOWS = ROOT / "04 Code" / "FYP_code" / "data" / "reference_windows_{y}" / "reference_windows.geojson"
IN_USE = ROOT / "06 Outputs" / "gis_tools" / "qgis" / "{y}_event" / "reference_{y}_event.gpkg"

MNDWI_WATER = 0.0
MIN_CELLS = 2
NOTE = ("added by rule (15_reference_dry_completion.py): February "
        "MNDWI above 0 inside the event outline, not drawn")
TOL_M2 = 1.0


def burn(geoms, shape_, transform, values=None):
    geoms = list(geoms)
    vals = values if values is not None else [1] * len(geoms)
    pairs = [(g, int(v)) for g, v in zip(geoms, vals)
             if g is not None and not g.is_empty]
    if not pairs:
        return np.zeros(shape_, "uint8")
    return rasterize(pairs, out_shape=shape_, transform=transform, fill=0,
                     dtype="uint8", all_touched=False)


def read_return(zip_path, year):
    """The original drawings, from inside the RETURN zip."""
    member = f"RETURN/reference_{year}_event.gpkg"
    with tempfile.TemporaryDirectory(prefix="fyp_return_") as tmp:
        with zipfile.ZipFile(zip_path) as z:
            if member not in z.namelist():
                sys.exit(f"{zip_path.name} has no {member}.")
            z.extract(member, tmp)
        p = Path(tmp) / member
        return (gpd.read_file(p, layer="water_event"),
                gpd.read_file(p, layer="water_dry"))


def missing_dry(year, event, dry):
    with rasterio.open(DATA / "sentinel2" / f"S2_{year}_dry_mndwi.tif") as s:
        mndwi = s.read(1)
        tr, crs, sh = s.transform, s.crs, s.shape
    res = abs(tr.a)
    win = gpd.read_file(str(WINDOWS).format(y=year)).to_crs(crs)
    cor = gpd.read_file(DATA / "boundaries" / "corridor_v2_3km_clipped.geojson").to_crs(crs)

    wid = burn(win.geometry, sh, tr, win["window"])
    x = ((mndwi > MNDWI_WATER)
         & (burn(event.to_crs(crs).geometry, sh, tr) > 0)
         & (burn(dry.to_crs(crs).geometry, sh, tr) == 0)
         & (wid > 0)
         & (burn(cor.geometry, sh, tr) > 0))

    lab, n = label(x)                       # 4-connected
    sizes = np.bincount(lab.ravel())
    small = sizes < MIN_CELLS
    small[0] = False
    x[small[lab]] = False

    coded = np.where(x, wid, 0).astype("uint8")
    rows = [(shape(g), int(v)) for g, v in
            shapes(coded, mask=coded > 0, transform=tr, connectivity=4)]
    gdf = gpd.GeoDataFrame({"window": [v for _, v in rows]},
                           geometry=[g for g, _ in rows], crs=crs)
    gdf["area_m2"] = gdf.area.round().astype("int64")
    gdf = gdf.sort_values(["window", "area_m2"], kind="stable").reset_index(drop=True)
    print(f"  missing February water: {len(gdf)} polygons, "
          f"{gdf.area.sum() / 1e6:.4f} km2 (cells of {res:g} m)")
    for w, a in gdf.groupby("window")["area_m2"].sum().items():
        print(f"    window {w}: {a / 1e6:.3f} km2")
    return gdf[["window", "area_m2", "geometry"]]


def same_ground(a, b, label_):
    ua = shapely.union_all(a.geometry.values)
    ub = shapely.union_all(b.to_crs(a.crs).geometry.values)
    sym = ua.symmetric_difference(ub).area
    ok = len(a) == len(b) and sym <= TOL_M2
    print(f"  {label_:12s} in use {len(b):5d} polygons {ub.area / 1e6:.4f} km2 | "
          f"remake {len(a):5d} polygons {ua.area / 1e6:.4f} km2 | "
          f"differ {sym:.1f} m2  {'PASS' if ok else 'DIFF'}")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", required=True, choices=["2021", "2023"])
    ap.add_argument("--zip", default=None)
    ap.add_argument("--compare", default=None)
    ap.add_argument("--out-dir", default=str(ROOT / "06 Outputs" / "remake"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    y = a.year
    zip_path = Path(a.zip) if a.zip else ZIPS[y]
    in_use = Path(str(IN_USE).format(y=y))
    out_dir = Path(a.out_dir)
    out_qa = out_dir / f"QA_dry_water_not_drawn_{y}.gpkg"
    out_ref = out_dir / f"reference_{y}_event_CORRECTED_dry_added.gpkg"
    if out_dir.resolve() == in_use.parent.resolve():
        sys.exit("--out-dir is the folder of the file in use. Write somewhere else.")
    for p in (out_qa, out_ref):
        if p.exists():
            if not a.force:
                sys.exit(f"{p} exists. Add --force to overwrite it.")
            p.unlink()

    print(f"  originals: {zip_path.name}")
    event, dry = read_return(zip_path, y)
    print(f"  drawn: water_event {len(event)} polygons, water_dry {len(dry)}")
    add = missing_dry(y, event, dry)

    added = gpd.GeoDataFrame({"window": add["window"].values,
                              "note": [NOTE] * len(add)},
                             geometry=add.geometry.values, crs=add.crs).to_crs(dry.crs)
    dry_new = gpd.GeoDataFrame(pd.concat([dry, added], ignore_index=True),
                               crs=dry.crs)

    out_dir.mkdir(parents=True, exist_ok=True)
    add.to_file(out_qa, layer="dry_water_not_drawn", driver="GPKG")
    event.to_file(out_ref, layer="water_event", driver="GPKG")
    dry_new.to_file(out_ref, layer="water_dry", driver="GPKG")
    print(f"  written: {out_qa}")
    print(f"  written: {out_ref}  (water_dry {len(dry)} drawn + {len(added)} added)")

    cmp_path = a.compare or str(in_use)
    if cmp_path.lower() == "none":
        return 0
    print(f"\nComparison with the file in use, {Path(cmp_path).name}")
    ok = True
    for layer, new in (("water_event", event), ("water_dry", dry_new)):
        old = gpd.read_file(cmp_path, layer=layer)
        ok &= same_ground(new, old, layer)
    print("  PASS: same ground in both layers" if ok else "  DIFF: not identical")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
