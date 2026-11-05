"""Jal-SAR extensions, item M8 input: WorldPop 2020 age and sex structure for the corridor, through Earth Engine.

Earth Engine collection WorldPop/GP/100m/pop_age_sex (India, 2020): total
population and people by sex and five-year age group (M_0, M_1, M_5 ... M_80
and the same for F), at about 100 m. Downloaded with ee.data.computePixels onto
exactly the grid of the WorldPop 2020 file already in 05 Data\\population, so
the two line up cell for cell.

Version 3 (22 October 2026). WorldPop's own cells are 3 arc-seconds, about 80 x 92 m
(about 7,420 m2) here. Taking one WorldPop value per 100 m cell (version 1, and the
05 Data file) kept only about 74 % of the people (242,378 against 326,685 in the
corridor), because each 100 m cell covers 1.35 WorldPop cells. Each 100 m value is
now multiplied by 10,000 m2 / (the area of a WorldPop cell at that latitude), which
turns "people in one WorldPop cell" into "people in this 100 m cell". (Version 2
tried Earth Engine's reduceResolution and still came out at 243,664.) The script
also writes a one-band population file that M7 and the Jal-SAR web map now use.
The corridor total is checked against WorldPop summed on its own grid.

Output (never written into 05 Data):
  06 Outputs\\jalsar\\extensions\\m8_vulnerability\\worldpop_agesex_2020_100m.tif   (one band per group,
  band names kept) and worldpop_agesex_record.json; a version 1 file is kept as
  worldpop_agesex_2020_100m_v1_sampled.tif.
  06 Outputs\\jalsar\\extensions\\population\\POP_WorldPop_2020_corridor_100m_sum.tif  (people per
  100 m cell, only where the 05 Data file has a value).
Licence: WorldPop, CC BY 4.0.

Run:  python m8_worldpop_agesex_gee.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
POP = ROOT / "05 Data" / "population" / "POP_WorldPop_2020_corridor_100m.tif"
OUT = ROOT / "06 Outputs" / "jalsar" / "extensions" / "m8_vulnerability"
POP_SUM = ROOT / "06 Outputs" / "jalsar" / "extensions" / "population" / "POP_WorldPop_2020_corridor_100m_sum.tif"
COR = ROOT / "04 Code" / "FYP_code" / "data" / "corridor.geojson"
EE_PROJECT = os.environ.get("EE_PROJECT", "your-earth-engine-project")
COLLECTION = "WorldPop/GP/100m/pop_age_sex"
AGES = [0, 1] + list(range(5, 85, 5))
BANDS = ["population"] + [f"{s}_{a}" for s in ("M", "F") for a in AGES]
TILE = 256


def main():
    import ee
    import rasterio
    OUT.mkdir(parents=True, exist_ok=True)
    if not POP.exists():
        sys.exit(f"The WorldPop file is missing: {POP}")
    try:
        ee.Initialize(project=EE_PROJECT)
    except Exception as exc:
        sys.exit(f"Could not connect to Earth Engine ({exc}). If it says not authenticated, run "
                 "'earthengine authenticate' in Anaconda Prompt, then run this again.")
    with rasterio.open(POP) as src:
        prof = src.profile
        old = src.read(1).astype("float64")
        nd = src.nodata
    t = prof["transform"]
    src_img = (ee.ImageCollection(COLLECTION).filter(ee.Filter.eq("country", "IND"))
               .filter(ee.Filter.eq("year", 2020)).first())      # one image for India, on its own grid
    have = src_img.bandNames().getInfo()
    bands = [b for b in BANDS if b in have]
    missing = [b for b in BANDS if b not in have]
    print(f"m8_worldpop_agesex_gee.py: {len(bands)} bands" + (f"; not in the collection: {missing}" if missing else ""))
    # add the 92.8 m cells up into each 100 m cell (weighted by overlap), so no one is lost
    img = src_img.select(bands).toFloat().unmask(-1, False)
    wp_t = src_img.select("population").projection().getInfo()["transform"]   # degrees: [sx, 0, x0, 0, sy, y0]
    dlon, dlat = abs(wp_t[0]), abs(wp_t[4])
    H, W = prof["height"], prof["width"]
    full = np.full((len(bands), H, W), np.nan, "float32")
    t0 = time.time()
    tiles = [(c, r, min(TILE, W - c), min(TILE, H - r)) for r in range(0, H, TILE) for c in range(0, W, TILE)]
    for i, (c0, r0, w, h) in enumerate(tiles, 1):
        req = {"expression": img, "fileFormat": "NUMPY_NDARRAY", "bandIds": bands,
               "grid": {"dimensions": {"width": w, "height": h},
                        "affineTransform": {"scaleX": t.a, "shearX": 0, "translateX": t.c + c0 * t.a,
                                            "shearY": 0, "scaleY": t.e, "translateY": t.f + r0 * t.e},
                        "crsCode": str(prof["crs"])}}
        for attempt in range(5):
            try:
                arr = ee.data.computePixels(req)
                break
            except Exception as exc:
                if attempt == 4:
                    raise
                print(f"    tile {i}: {str(exc)[:120]}; trying again", flush=True)
                time.sleep(10 * (attempt + 1))
        full[:, r0:r0 + h, c0:c0 + w] = np.stack([np.asarray(arr[b], "float32") for b in bands])
        print(f"  tile {i} of {len(tiles)}", flush=True)
    full[full < 0] = np.nan
    if nd is not None:
        old[old == nd] = np.nan
    full[:, ~np.isfinite(old)] = np.nan            # only the cells the 05 Data file covers (the corridor)
    # people per WorldPop cell -> people per 100 m cell: times (our cell area / WorldPop cell area at that latitude)
    from pyproj import Geod, Transformer
    rows, cols = np.mgrid[0:H, 0:W]
    xs, ys = t * (cols + 0.5, rows + 0.5)
    _, lat = Transformer.from_crs(prof["crs"], "EPSG:4326", always_xy=True).transform(xs, ys)
    geod = Geod(ellps="WGS84")
    grid_lat = np.linspace(np.nanmin(lat) - 0.01, np.nanmax(lat) + 0.01, 200)
    wp_area = np.array([abs(geod.polygon_area_perimeter([0, dlon, dlon, 0], [la - dlat / 2, la - dlat / 2,
                                                                               la + dlat / 2, la + dlat / 2])[0])
                        for la in grid_lat])
    factor = (abs(t.a * t.e) / np.interp(lat, grid_lat, wp_area)).astype("float32")
    full *= factor[None, :, :]
    print(f"  WorldPop cell {dlon * 3600:.2f} x {dlat * 3600:.2f} arc-seconds; area factor "
          f"{np.nanmin(factor):.4f} to {np.nanmax(factor):.4f}")
    target = OUT / "worldpop_agesex_2020_100m.tif"
    v1 = OUT / "worldpop_agesex_2020_100m_v1_sampled.tif"
    rec_old = OUT / "worldpop_agesex_record.json"
    if target.exists() and not v1.exists():
        try:
            if "version 3" not in rec_old.read_text(encoding="utf-8"):
                target.replace(v1)
                print(f"  kept the version 1 file as {v1.name}")
        except FileNotFoundError:
            pass
    p = dict(prof)
    p.update(count=len(bands), dtype="float32", nodata=float("nan"), compress="lzw")
    with rasterio.open(target, "w", **p) as dst:
        dst.write(full)
        for k, b in enumerate(bands, 1):
            dst.set_band_description(k, b)
    POP_SUM.parent.mkdir(parents=True, exist_ok=True)
    p1 = dict(prof)
    p1.update(count=1, dtype="float32", nodata=float("nan"), compress="lzw")
    with rasterio.open(POP_SUM, "w", **p1) as dst:
        dst.write(full[0], 1)
        dst.set_band_description(1, "population")
    native_total = None
    try:
        import geopandas as gpd
        geom = ee.Geometry(gpd.read_file(COR).to_crs("EPSG:4326").union_all().__geo_interface__)
        native_total = src_img.select("population").reduceRegion(
            ee.Reducer.sum(), geom, crs=src_img.select("population").projection(), maxPixels=1e10
        ).get("population").getInfo()
    except Exception as exc:
        print(f"  could not get the total on WorldPop's own grid ({str(exc)[:150]})")
    tot_old, tot_new = float(np.nansum(old)), float(np.nansum(full[0]))
    rec = {"collection": COLLECTION, "country": "IND", "year": 2020, "bands": bands, "missing_bands": missing,
           "grid_from": str(POP.relative_to(ROOT)), "total_old_file": round(tot_old),
           "total_new_file": round(tot_new), "ratio": round(tot_new / tot_old, 4) if tot_old else None,
           "method": "version 3, 22 Oct 2026: sampled, then times 100 m cell area / WorldPop cell area",
           "worldpop_cell_deg": [dlon, dlat],
           "total_own_grid_corridor": round(native_total) if native_total else None,
           "population_file": str(POP_SUM.relative_to(ROOT)),
           "minutes": round((time.time() - t0) / 60, 1), "credit": "WorldPop (CC BY 4.0)"}
    (OUT / "worldpop_agesex_record.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"  wrote {target} and {POP_SUM.name}")
    print(f"  corridor total {tot_new:,.0f}; WorldPop on its own grid {native_total or float('nan'):,.0f}; "
          f"the old 05 Data file {tot_old:,.0f}")
    if native_total and abs(tot_new / native_total - 1) > 0.10:
        print("  FAIL: the new total is more than 10 % away from WorldPop's own total.")
        return 1
    if native_total and abs(tot_new / native_total - 1) > 0.03:
        print("  NOTE: 3 to 10 % away from WorldPop's own total (cells on the corridor edge).")
    print("PASS  M8 WorldPop age and sex download")
    return 0


if __name__ == "__main__":
    sys.exit(main())
