"""Jal-SAR extensions, item M7: Google Open Buildings (v3) for the corridor, through Earth Engine.

Reads GOOGLE/Research/open-buildings/v3/polygons inside the corridor polygon
(FYP_code\\data\\corridor.geojson), tile by tile (about 2 km squares, split again
if a square holds more than 4,000 buildings), and writes every building with its
confidence score and area. Nothing is filtered here; the overlay (m7_exposure_overlay.py)
uses buildings with confidence 0.70 or more by default, and says so.

Uses the Earth Engine cloud project in the EE_PROJECT setting (the same as
Jal-SAR). Finished tiles are kept, so a re-run carries on where it stopped.

Output: 06 Outputs\\jalsar\\extensions\\exposure\\buildings_openbuildings_v3.gpkg (EPSG:4326)
        and buildings_record.json (counts, time, dataset id).
Licence: Open Buildings is CC BY 4.0 and ODbL; credit "Google Open Buildings".

Run:  python m7_buildings_gee.py
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CODE = ROOT / "04 Code" / "FYP_code"
OUT = ROOT / "06 Outputs" / "jalsar" / "extensions" / "exposure"
TILES = OUT / "buildings_tiles"
DATASET = "GOOGLE/Research/open-buildings/v3/polygons"
EE_PROJECT = os.environ.get("EE_PROJECT", "your-earth-engine-project")
STEP = 0.02                 # degrees, about 2 km
MAX_PER_CALL = 4000


def main():
    import ee
    import geopandas as gpd
    from shapely.geometry import box, shape
    OUT.mkdir(parents=True, exist_ok=True)
    TILES.mkdir(parents=True, exist_ok=True)
    try:
        ee.Initialize(project=EE_PROJECT)
    except Exception as exc:
        sys.exit(f"Could not connect to Earth Engine ({exc}). If it says not authenticated, run "
                 "'earthengine authenticate' in Anaconda Prompt, then run this again.")
    cor = gpd.read_file(CODE / "data" / "corridor.geojson").to_crs("EPSG:4326")
    poly = cor.geometry.union_all() if hasattr(cor.geometry, "union_all") else cor.unary_union
    eepoly = ee.Geometry(json.loads(gpd.GeoSeries([poly]).to_json())["features"][0]["geometry"])
    fc = ee.FeatureCollection(DATASET).filterBounds(eepoly)
    w, s, e, n = poly.bounds
    cells = []
    y = s
    while y < n:
        x = w
        while x < e:
            b = box(x, y, min(x + STEP, e), min(y + STEP, n))
            if b.intersects(poly):
                cells.append(b.intersection(poly).buffer(0))
            x += STEP
        y += STEP
    print(f"m7_buildings_gee.py: {len(cells)} tiles over the corridor", flush=True)
    t0 = time.time()

    def fetch(geom, tag, depth=0):
        p = TILES / f"{tag}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        g = ee.Geometry(json.loads(gpd.GeoSeries([geom]).to_json())["features"][0]["geometry"])
        sub = fc.filterBounds(g)
        for attempt in range(5):
            try:
                k = sub.size().getInfo()
                if k > MAX_PER_CALL and depth < 6:
                    gx0, gy0, gx1, gy1 = geom.bounds
                    mx, my = (gx0 + gx1) / 2, (gy0 + gy1) / 2
                    feats = []
                    for i, q in enumerate((box(gx0, gy0, mx, my), box(mx, gy0, gx1, my),
                                           box(gx0, my, mx, gy1), box(mx, my, gx1, gy1))):
                        part = q.intersection(geom)
                        if not part.is_empty:
                            feats += fetch(part, f"{tag}_{i}", depth + 1)
                    break
                feats = sub.select(["confidence", "area_in_meters", "full_plus_code"]).getInfo()["features"] if k else []
                break
            except Exception as exc:
                if attempt == 4:
                    raise
                print(f"    {tag}: {str(exc)[:150]}; trying again", flush=True)
                time.sleep(10 * (attempt + 1))
        p.write_text(json.dumps(feats), encoding="utf-8")
        return feats

    allf = []
    for i, c in enumerate(cells, 1):
        allf += fetch(c, f"t{i:04d}")
        if i % 20 == 0 or i == len(cells):
            el = time.time() - t0
            print(f"  {i}/{len(cells)} tiles, {len(allf):,} buildings so far, about "
                  f"{el / i * (len(cells) - i) / 60:.0f} min left", flush=True)
    seen, rows, geoms = set(), [], []
    for f in allf:
        pid = f["properties"].get("full_plus_code") or json.dumps(f["geometry"]["coordinates"])[:80]
        if pid in seen:                       # a building on a tile edge comes twice
            continue
        seen.add(pid)
        rows.append({"confidence": f["properties"].get("confidence"),
                     "area_m2": f["properties"].get("area_in_meters"), "plus_code": pid})
        geoms.append(shape(f["geometry"]))
    g = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")
    target = OUT / "buildings_openbuildings_v3.gpkg"
    if target.exists():
        target.unlink()
    g.to_file(target, layer="buildings", driver="GPKG")
    rec = {"dataset": DATASET, "buildings": len(g), "confidence_0_70_or_more": int((g["confidence"] >= 0.70).sum()),
           "tiles": len(cells), "minutes": round((time.time() - t0) / 60, 1),
           "downloaded": time.strftime("%Y-%m-%d %H:%M"), "credit": "Google Open Buildings v3 (CC BY 4.0 and ODbL)"}
    (OUT / "buildings_record.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"  wrote {target}: {len(g):,} buildings ({rec['confidence_0_70_or_more']:,} with confidence 0.70 or more)")
    print("PASS  M7 Open Buildings download")
    return 0


if __name__ == "__main__":
    sys.exit(main())
