"""Jal-SAR extensions: Google Open Buildings checked against OpenStreetMap in a few villages.

Building counts from Open Buildings v3 are quoted in M7 and the bulletins. This checks them where
OpenStreetMap has buildings drawn by people, in a handful of corridor villages.

1. Villages: --n (default 6) villages touching a mapped flood, half in the gorge above Devprayag and
   half in the open valley below it, those with the most Open Buildings first (from
   m8_risk_view.py's risk_villages_all_events.csv; without it, any corridor village).
2. OpenStreetMap buildings inside each village, downloaded once from Overpass into
   06 Outputs\\jalsar\\extensions\\exposure\\osm_buildings_check.geojson (kept; not downloaded again).
3. Per village, inside the village polygon and the corridor:
     Open Buildings count (confidence 0.70 or more, as in M7) and OpenStreetMap count
     share of OpenStreetMap buildings that touch an Open Buildings footprint ("found by Open Buildings")
     share of Open Buildings footprints that touch an OpenStreetMap building ("also in OpenStreetMap")
     footprint area of each
   OpenStreetMap is not complete either: where it has far fewer buildings than Open Buildings, the
   second share says little. The first share is the useful one: of the buildings people drew,
   how many did Open Buildings find.

Writes 06 Outputs\\jalsar\\extensions\\m7_exposure\\buildings_check\\BUILDINGS_CHECK.md (read first),
buildings_check.csv.

Run:  python m7_buildings_osm_check.py
      python m7_buildings_osm_check.py --n 8 --village "Kirtinagar"
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
EXPO = ROOT / "06 Outputs" / "jalsar" / "extensions" / "exposure"
OUT = ROOT / "06 Outputs" / "jalsar" / "extensions" / "m7_exposure" / "buildings_check"
VILLAGES = ROOT / "05 Data" / "boundaries" / "VILLAGES_corridor.gpkg"
RISK = ROOT / "06 Outputs" / "jalsar" / "extensions" / "m8_vulnerability" / "risk_view" / "risk_villages_all_events.csv"
OSM_FILE = EXPO / "osm_buildings_check.geojson"
SPLIT_LON = 78.5981                      # the reach boundary at Devprayag (strata_meta.json)


def fetch_osm(bboxes):
    """OpenStreetMap buildings (ways and multipolygon relations) inside the boxes, as GeoJSON."""
    import requests
    import m7_exposure_download as dl
    parts = "".join(f'way["building"]({s},{w},{n},{e});relation["building"]({s},{w},{n},{e});'
                    for s, w, n, e in bboxes)
    body = f"[out:json][timeout:300];({parts});out geom;"
    for url in dl.OVERPASS:
        for attempt in range(3):
            try:
                r = requests.post(url, data={"data": body}, headers=dl.HEAD, timeout=400)
                if r.status_code == 429 or r.status_code >= 500:
                    raise RuntimeError(f"HTTP {r.status_code}")
                r.raise_for_status()
                osm = r.json()
                gj = dl.to_geojson(osm)
                for el in osm.get("elements", []):           # outer rings of building relations
                    if el["type"] == "relation":
                        for m in el.get("members", []):
                            if m.get("role") == "outer" and m.get("geometry"):
                                pts = [[p["lon"], p["lat"]] for p in m["geometry"]]
                                if len(pts) > 3 and pts[0] == pts[-1]:
                                    gj["features"].append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [pts]},
                                                           "properties": {"osm_type": "relation", "osm_id": el["id"], "building": "yes"}})
                gj["features"] = [f for f in gj["features"] if f["geometry"]["type"] == "Polygon"]
                gj["query"] = body
                gj["from"] = url
                gj["downloaded"] = time.strftime("%Y-%m-%d %H:%M")
                return gj
            except Exception as exc:                          # noqa: BLE001
                print(f"  {url} attempt {attempt + 1} failed ({str(exc)[:160]})")
                time.sleep(20 * (attempt + 1))
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--village", action="append", default=[], help="add this village by name (can repeat)")
    ap.add_argument("--min-confidence", type=float, default=0.70)
    args = ap.parse_args()
    import geopandas as gpd
    bp = EXPO / "buildings_openbuildings_v3.gpkg"
    for p in (bp, VILLAGES):
        if not p.exists():
            print(f"PROBLEM: {p} is missing")
            return 1
    OUT.mkdir(parents=True, exist_ok=True)
    vil = gpd.read_file(VILLAGES)
    vil = vil[vil.geometry.notna() & ~vil.geometry.is_empty].reset_index(drop=True)
    utm = vil.estimate_utm_crs()
    vil = vil.to_crs(utm)
    ob = gpd.read_file(bp, layer="buildings")
    ob = ob[ob["confidence"] >= args.min_confidence].to_crs(utm).reset_index(drop=True)
    print(f"  Open Buildings (confidence {args.min_confidence:.2f}+): {len(ob):,}; villages: {len(vil)}")

    # 1. pick villages
    cand = set()
    if RISK.exists():
        with open(RISK, newline="", encoding="utf-8") as fh:
            cand = {r["vlcode"] for r in csv.DictReader(fh)}
    vil["vlcode_s"] = vil["vlcode"].astype(str)
    pool = vil[vil["vlcode_s"].isin(cand)] if cand else vil
    j = gpd.sjoin(ob[["geometry"]], pool[["geometry"]], predicate="intersects", how="inner")
    n_ob = j.groupby("index_right").size()
    pool = pool.assign(n_ob=n_ob.reindex(pool.index).fillna(0).astype(int))
    pool = pool[pool["n_ob"] > 0]
    lon = pool.to_crs("EPSG:4326").geometry.representative_point().x
    pool = pool.assign(reach=np.where(lon >= SPLIT_LON, "gorge", "open valley"))
    half = max(1, args.n // 2)
    pick = []
    for reach in ("gorge", "open valley"):
        pick += list(pool[pool["reach"] == reach].sort_values("n_ob", ascending=False).index[:half])
    for name in args.village:
        pick += list(vil.index[vil["village"].astype(str).str.lower() == name.lower()])
    pick = list(dict.fromkeys(pick))
    sel = vil.loc[pick]
    print("  villages: " + ", ".join(f"{r['village']} ({pool.loc[i, 'reach'] if i in pool.index else 'asked'})" for i, r in sel.iterrows()))

    # 2. OpenStreetMap buildings, downloaded once
    boxes = [tuple(np.round(b, 5)) for b in sel.to_crs("EPSG:4326").geometry.bounds[["miny", "minx", "maxy", "maxx"]].to_numpy()]
    have = json.loads(OSM_FILE.read_text(encoding="utf-8")) if OSM_FILE.exists() else None
    if have is None or have.get("boxes") != [list(b) for b in boxes]:
        print(f"  downloading OpenStreetMap buildings for {len(boxes)} villages ...")
        gj = fetch_osm(boxes)
        if gj is None:
            print("FAIL  Overpass did not answer. Try again later (it is a free, shared service).")
            return 1
        gj["boxes"] = [list(b) for b in boxes]
        OSM_FILE.write_text(json.dumps(gj), encoding="utf-8")
        have = gj
    osm = gpd.GeoDataFrame.from_features(have["features"], crs="EPSG:4326").to_crs(utm) if have["features"] else \
        gpd.GeoDataFrame(geometry=[], crs=utm)
    print(f"  OpenStreetMap buildings in the boxes: {len(osm):,} ({have.get('downloaded')})")

    # 3. compare inside each village
    rows = []
    for i, v in sel.iterrows():
        g = v.geometry
        o = ob[ob.intersects(g)]
        s = osm[osm.intersects(g)] if len(osm) else osm
        o_hit = o.intersects(s.unary_union) if len(s) else np.zeros(len(o), bool)
        s_hit = s.intersects(o.unary_union) if len(o) and len(s) else np.zeros(len(s), bool)
        rows.append({"village": v.get("village"), "subdistrict": v.get("subdistric"), "district": v.get("district"),
                     "vlcode": v["vlcode_s"], "reach": pool.loc[i, "reach"] if i in pool.index else "asked",
                     "open_buildings": len(o), "osm_buildings": len(s),
                     "osm_found_by_open_buildings": round(float(np.mean(s_hit)), 3) if len(s) else None,
                     "open_buildings_also_in_osm": round(float(np.mean(o_hit)), 3) if len(o) else None,
                     "open_buildings_area_m2": round(float(o.area.sum())), "osm_area_m2": round(float(s.area.sum()))})
    with open(OUT / "buildings_check.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    tot_s = sum(r["osm_buildings"] for r in rows)
    tot_found = sum((r["osm_found_by_open_buildings"] or 0) * r["osm_buildings"] for r in rows)
    pc = lambda x: "n/a" if x is None else f"{x * 100:.0f} %"  # noqa: E731
    md = ["# Open Buildings against OpenStreetMap in a few villages", "",
          f"Open Buildings v3, confidence {args.min_confidence:.2f} or more (as in M7); OpenStreetMap buildings from "
          f"Overpass ({have.get('from')}, {have.get('downloaded')}). Counted inside each village polygon. A building "
          "counts as found when the two footprints touch.", "",
          "| Village | Reach | Open Buildings | OpenStreetMap | OSM buildings found by Open Buildings | Open Buildings also in OSM | Area OB / OSM m2 |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['village']} ({r['subdistrict']}) | {r['reach']} | {r['open_buildings']:,} | {r['osm_buildings']:,} | "
                  f"{pc(r['osm_found_by_open_buildings'])} | {pc(r['open_buildings_also_in_osm'])} | "
                  f"{r['open_buildings_area_m2']:,} / {r['osm_area_m2']:,} |")
    md += ["", (f"Over all {len(rows)} villages, Open Buildings found {tot_found / tot_s * 100:.0f} % of the "
                f"{tot_s:,} buildings drawn in OpenStreetMap." if tot_s else
                "OpenStreetMap has no buildings in these villages, so this check cannot say anything; pick villages "
                "where OpenStreetMap has buildings (--village)."), "",
           "How to read it: the first share is the useful one (of the buildings people drew, how many Open Buildings "
           "found). The second share is low wherever OpenStreetMap is simply incomplete, which is common in Garhwal, "
           "so it does not mean Open Buildings is wrong. Neither source is a full count; the report should say the "
           "building numbers are from Open Buildings and give this check as its limit.", ""]
    (OUT / "BUILDINGS_CHECK.md").write_text("\n".join(md), encoding="utf-8")
    print(f"  wrote {OUT / 'BUILDINGS_CHECK.md'}")
    print("PASS  buildings check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
