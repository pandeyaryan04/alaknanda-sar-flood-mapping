r"""
06_rasterise_reference.py
=========================
Turn the hand-digitised polygons into reference_<event>.tif, on exactly the
Sentinel-1 grid, ready for 04_stratified_metrics.py.

WHAT IT DOES
------------
  flood = water on the event date  AND NOT  water on the dry date

inside the digitising windows, and nodata everywhere else. Scoring flood rather
than total water is what 04_stratified_metrics.py expects: the Alaknanda is
there on a dry day too, and so is the Srinagar reservoir, and neither is a flood.

Output is uint8: 1 flood, 0 not flood, 255 nodata. 04_stratified_metrics.py
counts only cells that are 0 or 1, so everything outside the windows, and every
cloud hole, drops out of the denominator instead of being scored as dry land.
That is the point. A cloud hole counted as dry is a false negative that never
happened.

USAGE
-----
  conda activate fyp
  cd /d "<project folder>\04 Code\FYP_code"

  python 06_rasterise_reference.py ^
    --event "..\..\05 Data\sentinel2\reference_2021.gpkg" ^
    --event-layer water_event ^
    --dry-layer water_dry ^
    --windows data/reference_windows_2021/reference_windows.geojson ^
    --grid data/flood_2021_v2/flood_2021_otsu.tif ^
    --out data/reference_2021.tif

  --event        GeoPackage holding both digitised layers. A second file can be
                 given with --dry if you kept them apart
  --holes        optional. Polygons where both dates were cloudy and you did not
                 draw. Marked nodata rather than dry
  --selftest     build a synthetic case, check the arithmetic, write nothing

CHECK THE NUMBERS IT PRINTS
---------------------------
Six bugs have been found in this project and not one of them raised an error.
Every one produced a clean table. This script prints flood area per window and
flags anything above 25 per cent of a window, because a quarter of a 5 km square
under water in a Himalayan gorge is not a result, it is a mistake.
"""

__stage__ = "Reference extent, rasterisation"
__reads__ = ["digitised polygons", "reference_windows.geojson",
             "a raster on the S1 grid"]
__writes__ = ["reference_<event>.tif"]

import argparse
import os
import sys

import numpy as np

try:
    import rasterio
    from rasterio.features import rasterize
    import geopandas as gpd
    from shapely.geometry import box
except ImportError as e:
    sys.exit(f"\nMissing package: {e}.\nActivate the environment first:  "
             f"conda activate fyp\n")

NODATA = 255
WARN_SHARE = 0.25       # flood share of a window above which something is wrong


def burn(gdf, shape, transform):
    """Rasterise polygons to a boolean array on the given grid."""
    if gdf is None or len(gdf) == 0:
        return np.zeros(shape, dtype=bool)
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None
              and not geom.is_empty]
    if not shapes:
        return np.zeros(shape, dtype=bool)
    return rasterize(shapes, out_shape=shape, transform=transform,
                     fill=0, default_value=1, dtype="uint8",
                     all_touched=False).astype(bool)


def load(path, layer, crs):
    if path is None:
        return None
    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if gdf.crs is None:
        sys.exit(f"\n{os.path.basename(path)} layer {layer} has no CRS.\n"
                 "Set it in QGIS before exporting. Guessing here would be worse "
                 "than stopping.\n")
    if str(gdf.crs) != str(crs):
        print(f"  reprojecting {layer or os.path.basename(path)} "
              f"from {gdf.crs} to {crs}")
        gdf = gdf.to_crs(crs)
    return gdf


def selftest():
    """Synthetic case with a known answer, on a 100 x 100 metre-scale grid."""
    from rasterio.transform import from_origin
    shape, tf = (100, 100), from_origin(0, 100, 1, 1)

    win = gpd.GeoDataFrame(geometry=[box(10, 10, 60, 60)], crs="EPSG:32644")
    # event water 30x30 = 900 cells, dry water 10x10 = 100 cells inside it
    ev  = gpd.GeoDataFrame(geometry=[box(20, 20, 50, 50)], crs="EPSG:32644")
    dry = gpd.GeoDataFrame(geometry=[box(20, 20, 30, 30)], crs="EPSG:32644")

    w  = burn(win, shape, tf)
    e  = burn(ev,  shape, tf)
    d  = burn(dry, shape, tf)
    flood = e & ~d & w

    out = np.full(shape, NODATA, dtype="uint8")
    out[w] = 0
    out[flood] = 1

    checks = [
        ("window cells", int(w.sum()), 2500),
        ("event water in window", int((e & w).sum()), 900),
        ("dry water in window", int((d & w).sum()), 100),
        ("flood cells", int(flood.sum()), 800),
        ("nodata cells", int((out == NODATA).sum()), 10000 - 2500),
        ("scored cells", int(np.isin(out, [0, 1]).sum()), 2500),
    ]
    bad = 0
    for name, got, want in checks:
        ok = got == want
        bad += (not ok)
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got} (expected {want})")

    # dry water outside the window must not create flood, and must not be scored
    d2 = burn(gpd.GeoDataFrame(geometry=[box(70, 70, 80, 80)],
                               crs="EPSG:32644"), shape, tf)
    leak = int((d2 & (out != NODATA)).sum())
    ok = leak == 0
    bad += (not ok)
    print(f"  {'ok  ' if ok else 'FAIL'} nothing outside the window is scored: "
          f"{leak} cells (expected 0)")

    print("\nself-test passed" if bad == 0 else f"\n{bad} CHECKS FAILED")
    return 0 if bad == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--event")
    ap.add_argument("--dry", default=None, help="defaults to the --event file")
    ap.add_argument("--event-layer", default="water_event")
    ap.add_argument("--dry-layer", default="water_dry")
    ap.add_argument("--windows")
    ap.add_argument("--holes", default=None)
    ap.add_argument("--holes-layer", default=None)
    ap.add_argument("--grid")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())

    for name in ("event", "windows", "grid", "out"):
        if not getattr(args, name):
            sys.exit(f"\n--{name} is required unless you pass --selftest\n")

    with rasterio.open(args.grid) as src:
        shape, transform, crs = src.shape, src.transform, src.crs
        res = abs(src.res[0])
        profile = src.profile
    cell_km2 = (res * res) / 1e6
    print(f"grid {shape[0]} x {shape[1]} at {res:.1f} m, {crs}")

    win = load(args.windows, None, crs)
    ev  = load(args.event, args.event_layer, crs)
    dry = load(args.dry or args.event, args.dry_layer, crs)
    holes = load(args.holes, args.holes_layer, crs) if args.holes else None

    w = burn(win, shape, transform)
    if not w.any():
        sys.exit("\nthe windows did not land on the grid at all. Check that "
                 "reference_windows.geojson and the flood raster share a CRS "
                 "and overlap.\n")

    e = burn(ev, shape, transform)
    d = burn(dry, shape, transform)
    flood = e & ~d & w

    out = np.full(shape, NODATA, dtype="uint8")
    out[w] = 0
    out[flood] = 1
    if holes is not None:
        h = burn(holes, shape, transform)
        out[h] = NODATA
        print(f"  {int((h & w).sum())} cells inside windows marked nodata as "
              f"cloud holes")

    # --- the numbers, per window -----------------------------------------
    print("\n| Window | Cells | Event water | Dry water | Flood | Flood km2 | "
          "Share |")
    print("|---|---|---|---|---|---|---|")
    problems = []
    for _, row in win.reset_index(drop=True).iterrows():
        m = burn(gpd.GeoDataFrame(geometry=[row.geometry], crs=crs),
                 shape, transform)
        n = int(m.sum())
        if n == 0:
            continue
        f = int((flood & m).sum())
        share = f / n
        wid = row.get("window", "?")
        print(f"| {wid} | {n} | {int((e & m).sum())} | {int((d & m).sum())} | "
              f"{f} | {f * cell_km2:.3f} | {share:.1%} |")
        if share > WARN_SHARE:
            problems.append((wid, share))

    scored = int(np.isin(out, [0, 1]).sum())
    nflood = int((out == 1).sum())
    print(f"\nscored cells      {scored}  ({scored * cell_km2:.1f} km2, "
          f"the digitised windows)")
    print(f"flood cells       {nflood}  ({nflood * cell_km2:.3f} km2)")
    print(f"flood share       {nflood / scored:.2%} of the scored area")

    if nflood == 0:
        print("\nWARNING: no flood cells at all. Either the event and dry "
              "polygons are identical, or the layer names are the wrong way "
              "round, or nothing was drawn.")
    for wid, share in problems:
        print(f"\nWARNING: window {wid} is {share:.0%} flooded. A quarter of a "
              f"5 km square under water in this terrain is a mistake, not a "
              f"result. Check that layer before believing the table.")

    profile.update(dtype="uint8", count=1, nodata=NODATA, compress="lzw")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with rasterio.open(args.out, "w", **profile) as dst:
        dst.write(out, 1)
    print(f"\nwrote {args.out}")
    print("Load it back into QGIS over the imagery and look at it before "
          "scoring anything.")


if __name__ == "__main__":
    main()
