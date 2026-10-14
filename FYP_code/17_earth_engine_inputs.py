r"""
17_earth_engine_inputs.py
=========================
Records exactly which Earth Engine images the project's
exports used: the image ID of every Sentinel-1 and Sentinel-2 scene, the
Copernicus DEM tiles, the WorldCover and WorldPop images, and the last-update
time of every Earth Engine dataset the project read.

It downloads nothing and changes nothing. It asks Earth Engine the same
questions 04 Code\gee\10_reexport_corridor_v2.js asked on 1 October 2026
(same dates, orbits, filters and corridor) and writes what comes back to

  04 Code\provenance\EARTH_ENGINE_INPUTS.csv

One row per image or dataset. If Earth Engine has reprocessed a scene since
1 October, the ID it returns today is the current one; the "checked" column
says when it was asked, and the exports' own date (1 October) is in the
register.

USAGE
-----
  conda activate fyp
  cd /d "<project folder>\04 Code\FYP_code"
  python 17_earth_engine_inputs.py

It also recomputes, the same way, the numbers the export script printed to
the Earth Engine console on 1 October (corridor area on the ellipsoid,
WorldPop total, WorldCover table, catchment area), plus the WorldPop total at
WorldPop's own pixel size, so 19_number_audit.py can check them.

PASS: it ends with "scenes with no image: 0" and writes the CSV.
"""

__stage__ = "Provenance, exact Earth Engine inputs"
__reads__ = ["corridor_v2_3km_clipped.geojson", "Earth Engine catalogue"]
__writes__ = ["provenance/EARTH_ENGINE_INPUTS.csv"]

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORRIDOR = ROOT / "05 Data" / "boundaries" / "corridor_v2_3km_clipped.geojson"
OUT = ROOT / "04 Code" / "provenance" / "EARTH_ENGINE_INPUTS.csv"
EE_PROJECT = "gbpuat-fyp-alaknanda"

# The same lists as 10_reexport_corridor_v2.js, blocks A and B
S1 = [
    ("2021_post", "2021-10-21", 63, "DESCENDING"),
    ("2021_pre", "2020-11-19", 63, "DESCENDING"),
    ("2021_pre_alt_dec", "2020-12-25", 63, "DESCENDING"),
    ("2021_pre_alt_jan", "2021-01-18", 63, "DESCENDING"),
    ("2021_pre_alt_feb", "2021-02-11", 63, "DESCENDING"),
    ("2023_post", "2023-08-16", 129, "ASCENDING"),
    ("2023_pre", "2022-11-01", 129, "ASCENDING"),
    ("2023_pre_alt_dec", "2022-12-19", 129, "ASCENDING"),
    ("2023_pre_alt_jan", "2023-01-12", 129, "ASCENDING"),
    ("2023_pre_alt_feb", "2023-02-05", 129, "ASCENDING"),
    ("2023jul_post", "2023-07-19", 63, "DESCENDING"),
    ("2023jul_pre_dec", "2022-12-27", 63, "DESCENDING"),
    ("2023jul_pre_jan", "2023-01-08", 63, "DESCENDING"),
    ("2023jul_pre_feb", "2023-02-13", 63, "DESCENDING"),
]
S2 = [
    ("2021_dry_a", "2021-02-08"), ("2021_dry_b", "2021-02-10"),
    ("2021_pre", "2021-10-16"), ("2021_ref", "2021-10-21"),
    ("2021_ref_alt", "2021-10-23"), ("2023_dry_a", "2023-02-03"),
    ("2023_dry_b", "2023-02-05"), ("2023_ref", "2023-08-19"),
    ("2023_ref_alt", "2023-08-17"),
]
S1_PROPS = ["system:index", "system:time_start", "system:version",
            "relativeOrbitNumber_start", "orbitProperties_pass", "sliceNumber",
            "resolution_meters", "instrumentMode"]
S2_PROPS = ["system:index", "system:time_start", "system:version", "PRODUCT_ID",
            "PROCESSING_BASELINE", "MGRS_TILE", "SENSING_ORBIT_NUMBER",
            "CLOUDY_PIXEL_PERCENTAGE"]
DATASETS = [
    "COPERNICUS/S1_GRD", "COPERNICUS/S2_SR_HARMONIZED", "COPERNICUS/DEM/GLO30",
    "USGS/SRTMGL1_003", "JRC/GSW1_4/GlobalSurfaceWater", "ESA/WorldCover/v200",
    "WorldPop/GP/100m/pop", "UCSB-CHG/CHIRPS/DAILY",
    "WWF/HydroSHEDS/v1/Basins/hybas_7", "FAO/GAUL/2015/level2",
]


def ee_init():
    try:
        import ee
    except ImportError:
        sys.exit("\nThe earthengine-api package is not in this environment. "
                 "Run this in the fyp environment: conda activate fyp\n")
    try:
        ee.Initialize(project=EE_PROJECT)
    except Exception as exc:                      # noqa: BLE001
        sys.exit(f"\nCould not connect to Earth Engine ({exc}).\n"
                 "If it says you are not authenticated, run:\n"
                 "    earthengine authenticate\nthen run this script again.\n")
    return ee


def corridor_geometry(ee):
    g = json.loads(CORRIDOR.read_text(encoding="utf-8"))
    geom = g["features"][0]["geometry"]
    if geom["type"] != "Polygon":
        sys.exit(f"corridor geometry is {geom['type']}, expected Polygon")
    # planar, as in the export script: ee.Geometry.Polygon(coords, null, false)
    return ee.Geometry.Polygon(geom["coordinates"], None, False)


def props_of(ee, col, names):
    """One dict per image of the collection, missing properties as None."""
    fc = ee.FeatureCollection(col.map(
        lambda im: ee.Feature(None, {n.replace(":", "_"): im.get(n) for n in names})))
    out = []
    for f in fc.getInfo()["features"]:
        p = f["properties"]
        out.append({n: p.get(n.replace(":", "_")) for n in names})
    return out


def ms_to_iso(ms):
    if ms is None:
        return ""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    ee = ee_init()
    corridor = corridor_geometry(ee)
    checked = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rows = []
    empty = []

    def add(group, tag, date, asset, image_id, details):
        rows.append({"group": group, "tag": tag, "date": date, "asset": asset,
                     "image_id": image_id,
                     "details": json.dumps(details, sort_keys=True, default=str),
                     "checked": checked})

    print("Sentinel-1, 14 scene dates")
    for tag, date, orbit, pas in S1:
        d0 = ee.Date(date)
        col = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(corridor)
               .filterDate(d0, d0.advance(1, "day"))
               .filter(ee.Filter.eq("instrumentMode", "IW"))
               .filter(ee.Filter.eq("relativeOrbitNumber_start", orbit))
               .filter(ee.Filter.eq("orbitProperties_pass", pas))
               .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
               .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
        items = props_of(ee, col, S1_PROPS)
        print(f"  {tag:18s} {date}  {len(items)} image(s)")
        if not items:
            empty.append(f"S1 {tag}")
        for p in items:
            p["system:time_start"] = ms_to_iso(p.get("system:time_start"))
            add("sentinel1", tag, date, "COPERNICUS/S1_GRD", p["system:index"], p)

    print("Sentinel-2, 9 scene dates")
    for tag, date in S2:
        d0 = ee.Date(date)
        col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(corridor)
               .filterDate(d0, d0.advance(1, "day")))
        items = props_of(ee, col, S2_PROPS)
        print(f"  {tag:18s} {date}  {len(items)} image(s)")
        if not items:
            empty.append(f"S2 {tag}")
        for p in items:
            p["system:time_start"] = ms_to_iso(p.get("system:time_start"))
            add("sentinel2", tag, date, "COPERNICUS/S2_SR_HARMONIZED",
                p["system:index"], p)

    print("Copernicus GLO-30 tiles, WorldCover, WorldPop, JRC, SRTM")
    glo = ee.ImageCollection("COPERNICUS/DEM/GLO30").filterBounds(corridor)
    for i in glo.aggregate_array("system:index").getInfo():
        add("dem", "COP GLO-30 tile", "", "COPERNICUS/DEM/GLO30", i, {})
    wc = ee.ImageCollection("ESA/WorldCover/v200").first()
    add("landcover", "WorldCover", "", "ESA/WorldCover/v200",
        wc.get("system:index").getInfo(), {})
    pop = (ee.ImageCollection("WorldPop/GP/100m/pop")
           .filter(ee.Filter.eq("country", "IND")).filter(ee.Filter.eq("year", 2020)))
    for i in pop.aggregate_array("system:index").getInfo():
        add("population", "WorldPop IND 2020", "2020", "WorldPop/GP/100m/pop", i, {})
    for asset in ("JRC/GSW1_4/GlobalSurfaceWater", "USGS/SRTMGL1_003"):
        v = ee.Image(asset).get("system:version").getInfo()
        add("image", asset, "", asset, asset, {"system:version": v})
    chirps = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
              .filterDate("2023-05-01", "2023-08-31"))
    ids = chirps.aggregate_array("system:index").getInfo()
    add("rainfall", "CHIRPS 2023 window", "2023-05-01 to 2023-08-30",
        "UCSB-CHG/CHIRPS/DAILY", f"{ids[0]} to {ids[-1]}" if ids else "",
        {"images": len(ids)})

    print("Console numbers, recomputed the way 10_reexport_corridor_v2.js did")
    console = {}
    console["corridor_area_km2_ellipsoid"] = ee.Number(corridor.area(1)).divide(1e6)
    pop_img = pop.mosaic()
    console["worldpop_2020_sum_scale100"] = pop_img.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=corridor, scale=100, maxPixels=1e9).get("population")
    native = pop.first().projection().nominalScale()
    console["worldpop_native_scale_m"] = native
    console["worldpop_2020_sum_native_scale"] = pop_img.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=corridor, scale=native, maxPixels=1e10).get("population")
    wc_area = ee.Image.pixelArea().divide(1e6).addBands(wc).reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1, groupName="class"),
        geometry=corridor, scale=10, maxPixels=1e10).get("groups")
    console["worldcover_area_by_class_km2"] = wc_area
    hybas = ee.FeatureCollection("WWF/HydroSHEDS/v1/Basins/hybas_7")
    seed = hybas.filterBounds(ee.Geometry.Point([78.9842, 30.288061])).aggregate_array("HYBAS_ID")

    def grow(i, acc):
        acc = ee.List(acc)
        return acc.cat(hybas.filter(ee.Filter.inList("NEXT_DOWN", acc))
                       .aggregate_array("HYBAS_ID")).distinct()
    up = ee.List(ee.List.sequence(1, 12).iterate(grow, seed))
    catch = hybas.filter(ee.Filter.inList("HYBAS_ID", up)).union(1).geometry()
    console["catchment_area_km2"] = ee.Number(catch.area(100)).divide(1e6)
    values = ee.Dictionary(console).getInfo()
    groups = values.pop("worldcover_area_by_class_km2") or []
    wc_total = sum(g["sum"] for g in groups)
    values["worldcover_total_km2"] = wc_total
    for g in groups:
        values[f"worldcover_class_{g['class']}_km2"] = g["sum"]
    if wc_total:
        values["worldcover_tree_share_pct"] = 100 * sum(
            g["sum"] for g in groups if g["class"] == 10) / wc_total
    for k, v in values.items():
        add("console", k, "", "", "", {"value": v})
        print(f"  {k:36s} {v:,.4f}" if isinstance(v, (int, float)) else f"  {k:36s} {v}")

    print("Dataset update times")
    for asset in DATASETS:
        try:
            info = ee.data.getAsset(asset)
            add("dataset", asset, "", asset, "", {"type": info.get("type"),
                                                  "updateTime": info.get("updateTime")})
            print(f"  {asset:36s} updated {info.get('updateTime')}")
        except Exception as exc:                  # noqa: BLE001
            add("dataset", asset, "", asset, "", {"error": str(exc)[:200]})
            print(f"  {asset:36s} could not read: {str(exc)[:80]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["group", "tag", "date", "asset",
                                           "image_id", "details", "checked"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n  rows written: {len(rows)}")
    print(f"  scenes with no image: {len(empty)}" + (f"  ({', '.join(empty)})" if empty else ""))
    print(f"  written: {OUT}")
    return 1 if empty else 0


if __name__ == "__main__":
    sys.exit(main())
