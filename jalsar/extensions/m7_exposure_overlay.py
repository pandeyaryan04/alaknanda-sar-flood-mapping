"""Jal-SAR extensions, item M7: what lies inside and near each Jal-SAR event's mapped flood.

For every event in 06 Outputs\\jalsar\\ (its headline flood map, 10 m):
  population  WorldPop 2020 (05 Data\\population, 100 m): people inside the flood
              (each 100 m cell counts by the share of it that is flooded), and
              inside the flood widened by --edge-m (30 m) for edge error
  buildings   Google Open Buildings v3 with confidence >= --min-confidence (0.70):
              inside (any 10 m cell of the building is flooded) and near (within
              --near-m, 100 m)
  power       OpenStreetMap power lines plus the World Bank derived line map:
              km inside and near; OpenStreetMap poles, pylons, substations:
              inside and near
  towers      towers_corridor.geojson, OpenStreetMap only (the public
              OpenCelliD copy on Source Cooperative holds no Indian cells):
              inside and near
Writes, per event, 06 Outputs\\jalsar\\<event>\\extras\\exposure.json and
exposure_features.geojson (only features inside or near the flood), which the
web map shows (jalsar.py webmap --force), and one summary for all events,
06 Outputs\\jalsar\\extensions\\m7_exposure\\M7_EXPOSURE.md.

Inside or near the mapped flood does not mean damaged. Every layer's source,
date and limits are printed in the summary.

Run:  python m7_exposure_overlay.py            (all events)
      python m7_exposure_overlay.py --event 2023-08-16_orbit129
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "FYP_code"))
import models_common as mc  # noqa: E402

ROOT = HERE.parents[2]
EXPO = ROOT / "06 Outputs" / "jalsar" / "extensions" / "exposure"
JAL = ROOT / "06 Outputs" / "jalsar"
POP_OLD = ROOT / "05 Data" / "population" / "POP_WorldPop_2020_corridor_100m.tif"
# people added up from WorldPop's own grid (22 Oct 2026); the 05 Data file kept only about 74 % of them
POP_SUM = ROOT / "06 Outputs" / "jalsar" / "extensions" / "population" / "POP_WorldPop_2020_corridor_100m_sum.tif"
POP = POP_SUM if POP_SUM.exists() else POP_OLD
OUTSUM = ROOT / "06 Outputs" / "jalsar" / "extensions" / "m7_exposure"
MAX_BUILDING_POLYS = 6000          # more than this inside or near: near ones become points


def distance_to_flood(flood, cell):
    from scipy.ndimage import distance_transform_edt
    f = flood == 1
    if not f.any():
        return np.full(flood.shape, np.inf, "float32")
    return (distance_transform_edt(~f) * cell).astype("float32")


def population(flood, prof, edge_m):
    import rasterio
    from rasterio.warp import reproject, Resampling
    if not POP.exists():
        return None
    with rasterio.open(POP) as src:
        pop = src.read(1).astype("float64")
        nd = src.nodata
        pprof = src.profile
    pop[(pop < 0) | ~np.isfinite(pop) | ((pop == nd) if nd is not None else False)] = 0
    out = {}
    from scipy.ndimage import binary_dilation
    k = max(1, int(round(edge_m / abs(prof["transform"].a))))
    f = (flood == 1)
    fb = binary_dilation(f, iterations=k)
    for name, mask in (("inside", f), ("inside_buffer", fb)):
        frac = np.zeros(pop.shape, "float32")
        reproject(mask.astype("float32"), frac, src_transform=prof["transform"], src_crs=prof["crs"],
                  dst_transform=pprof["transform"], dst_crs=pprof["crs"], resampling=Resampling.average)
        out[name] = round(float((pop * np.nan_to_num(frac)).sum()))
    out.update({"buffer_m": edge_m, "corridor_total": round(float(pop.sum())), "source": "WorldPop 2020, 100 m",
                "note": "modelled estimate, not a count"})
    return out


def sample_lines(gdf, flood, dist, prof, near_m, step=5.0):
    """Per line: km inside the flood, km within near distance, min distance; uses segment midpoints."""
    import shapely
    inv = ~prof["transform"]
    res = []
    for g in gdf.geometry:
        tot_in, tot_near, dmin = 0.0, 0.0, np.inf
        if g is None or g.is_empty:
            res.append((0.0, 0.0, np.inf))
            continue
        for part in getattr(g, "geoms", [g]):
            c = shapely.get_coordinates(shapely.segmentize(part, step))
            if len(c) < 2:
                continue
            mid = (c[1:] + c[:-1]) / 2
            ln = np.hypot(*(c[1:] - c[:-1]).T)
            col, row = inv * (mid[:, 0], mid[:, 1])
            col, row = np.floor(col).astype(int), np.floor(row).astype(int)
            ok = (row >= 0) & (row < flood.shape[0]) & (col >= 0) & (col < flood.shape[1])
            if not ok.any():
                continue
            fl = np.zeros(len(mid), bool)
            dd = np.full(len(mid), np.inf, "float32")
            fl[ok] = flood[row[ok], col[ok]] == 1
            dd[ok] = dist[row[ok], col[ok]]
            tot_in += float(ln[fl].sum() / 1000)
            res_near = dd <= near_m
            tot_near += float(ln[res_near & ~fl].sum() / 1000)
            dmin = min(dmin, float(dd.min()))
        res.append((tot_in, tot_near, dmin))
    return res


def sample_points(gdf, flood, dist, prof):
    inv = ~prof["transform"]
    xs = gdf.geometry.x.to_numpy()
    ys = gdf.geometry.y.to_numpy()
    col, row = inv * (xs, ys)
    col, row = np.floor(col).astype(int), np.floor(row).astype(int)
    ok = (row >= 0) & (row < flood.shape[0]) & (col >= 0) & (col < flood.shape[1])
    inside = np.zeros(len(xs), bool)
    d = np.full(len(xs), np.inf, "float32")
    inside[ok] = flood[row[ok], col[ok]] == 1
    d[ok] = dist[row[ok], col[ok]]
    return inside, d


def to_feature(geom_4326, props):
    from shapely.geometry import mapping
    g = mapping(geom_4326)
    return {"type": "Feature", "geometry": json.loads(json.dumps(g)), "properties": props}


def load_vector(path, crs, bbox_4326=None, layer=None):
    import geopandas as gpd
    if not Path(path).exists():
        return None
    kw = {"bbox": bbox_4326} if bbox_4326 else {}
    if layer:
        kw["layer"] = layer
    g = gpd.read_file(path, **kw)
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    return g.to_crs(crs)


def event_exposure(ev_dir, args, cache):
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    flood_p = ev_dir / "flood_headline.tif"
    with rasterio.open(flood_p) as src:
        flood = src.read(1)
        prof = src.profile
    cell = abs(prof["transform"].a)
    dist = distance_to_flood(flood, cell)
    feats, out = [], {"event": ev_dir.name, "near_m": args.near_m, "made": time.strftime("%Y-%m-%d %H:%M"),
                      "flood_km2": round(float((flood == 1).sum() * cell * cell / 1e6), 4), "sources": {}}
    out["population"] = population(flood, prof, args.edge_m)

    # Buildings.
    if "buildings" not in cache:
        p = EXPO / "buildings_openbuildings_v3.gpkg"
        b = load_vector(p, prof["crs"], layer="buildings") if p.exists() else None
        cache["buildings"] = b[b["confidence"] >= args.min_confidence].reset_index(drop=True) if b is not None else None
    b = cache["buildings"]
    if b is not None and len(b):
        ids = rasterize(((g, i + 1) for i, g in enumerate(b.geometry)), out_shape=flood.shape,
                        transform=prof["transform"], fill=0, dtype="int32", all_touched=True)
        n = len(b) + 1
        flooded = np.bincount(ids[flood == 1].ravel(), minlength=n)[1:] > 0
        dmin = np.full(n, np.inf, "float32")
        m = ids > 0
        np.minimum.at(dmin, ids[m], dist[m])
        dmin = dmin[1:]
        near = (~flooded) & (dmin <= args.near_m)
        out["buildings"] = {"inside": int(flooded.sum()), "near": int(near.sum()), "corridor_total": len(b),
                            "min_confidence": args.min_confidence,
                            "area_m2_inside": round(float(b.geometry.area[flooded].sum()), 1)}
        out["sources"]["buildings"] = "Google Open Buildings v3 (CC BY 4.0 and ODbL)"
        sel = np.flatnonzero(flooded | near)
        as_points = len(sel) > MAX_BUILDING_POLYS
        g4326 = b.geometry.iloc[sel].to_crs("EPSG:4326")
        for i, g in zip(sel, g4326):
            fl = bool(flooded[i])
            geom = g if (fl or not as_points) else g.centroid
            feats.append(to_feature(geom, {"layer": "building", "flooded": fl, "near_m": args.near_m,
                                           "label": f"building, {b['area_m2'].iloc[i]:.0f} m2, confidence "
                                                    f"{b['confidence'].iloc[i]:.2f}"}))

    # Power lines: OpenStreetMap and the World Bank map.
    lines = []
    if "lines" not in cache:
        parts = []
        osm = load_vector(EXPO / "osm_power_lines.geojson", prof["crs"])
        if osm is not None and len(osm):
            osm = osm[osm.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
            osm["src"] = "OpenStreetMap"
            osm["label"] = osm.get("power", "line").astype(str) + " " + osm.get("voltage", "").fillna("").astype(str)
            parts.append(osm[["src", "label", "geometry"]])
        wb_p = EXPO / "worldbank_grid.gpkg"
        if wb_p.exists():
            wb = load_vector(wb_p, prof["crs"], bbox_4326=(78.1, 29.9, 79.3, 30.6))
            if wb is not None and len(wb):
                wb = wb[wb.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
                wb["src"] = "World Bank derived map"
                wb["label"] = "predicted line (World Bank, modelled)"
                parts.append(wb[["src", "label", "geometry"]])
        cache["lines"] = gpd.GeoDataFrame(__import__("pandas").concat(parts, ignore_index=True),
                                          crs=prof["crs"]) if parts else None
    L = cache["lines"]
    if L is not None and len(L):
        r = sample_lines(L, flood, dist, prof, args.near_m)
        km_in = {s: 0.0 for s in L["src"].unique()}
        km_near = dict(km_in)
        g4326 = L.geometry.to_crs("EPSG:4326")
        for (kin, knear, dmn), src, lab, g in zip(r, L["src"], L["label"], g4326):
            km_in[src] += kin
            km_near[src] += knear
            if kin > 0 or dmn <= args.near_m:
                feats.append(to_feature(g, {"layer": "power_line", "flooded": kin > 0, "near_m": args.near_m,
                                            "label": f"{lab.strip()} ({src})"}))
        lines = {"line_km_inside": round(sum(km_in.values()), 3), "line_km_near": round(sum(km_near.values()), 3),
                 "line_km_inside_by_source": {k: round(v, 3) for k, v in km_in.items()}}
        out["sources"]["power_lines"] = "OpenStreetMap (ODbL); World Bank derived map of transmission and distribution lines (CC BY 4.0)"
    out["power"] = lines or None

    # Power points.
    pts = load_vector(EXPO / "osm_power_points.geojson", prof["crs"])
    if pts is not None and len(pts):
        pts = pts.copy()
        pts["geometry"] = pts.geometry.centroid
        ins, d = sample_points(pts, flood, dist, prof)
        nr = (~ins) & (d <= args.near_m)
        out["power"] = {**(out["power"] or {}), "points_inside": int(ins.sum()), "points_near": int(nr.sum())}
        g4326 = pts.geometry.to_crs("EPSG:4326")
        for k in np.flatnonzero(ins | nr):
            feats.append(to_feature(g4326.iloc[k], {"layer": "power_point", "flooded": bool(ins[k]),
                                                    "near_m": args.near_m,
                                                    "label": f"power {pts.get('power', 'point').iloc[k]}"}))

    # Towers.
    tw = load_vector(EXPO / "towers_corridor.geojson", prof["crs"])
    if tw is not None and len(tw):
        ins, d = sample_points(tw, flood, dist, prof)
        nr = (~ins) & (d <= args.near_m)
        by = {s: int(((tw["source"] == s) & ins).sum()) for s in tw["source"].unique()}
        out["towers"] = {"inside": int(ins.sum()), "near": int(nr.sum()), "inside_by_source": by,
                         "corridor_total": len(tw)}
        out["sources"]["towers"] = ("OpenStreetMap (ODbL) only; the public OpenCelliD copy on Source Cooperative "
                                    "holds no Indian cells, so OpenCelliD is not used")
        g4326 = tw.geometry.to_crs("EPSG:4326")
        for k in np.flatnonzero(ins | nr):
            p = tw.iloc[k]
            lab = (f"OpenCelliD tower, {int(p.get('cells') or 1)} radio cell(s) {p.get('radios') or ''}"
                   if p["source"] == "opencellid" else f"OpenStreetMap {p.get('kind') or 'tower'}")
            feats.append(to_feature(g4326.iloc[k], {"layer": "tower", "flooded": bool(ins[k]),
                                                    "near_m": args.near_m, "label": lab.strip()}))
    d = ev_dir / "extras"
    d.mkdir(exist_ok=True)
    mc.save_json(d / "exposure.json", out)
    (d / "exposure_features.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}),
                                                 encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", default="all")
    ap.add_argument("--near-m", type=float, default=100.0)
    ap.add_argument("--edge-m", type=float, default=30.0)
    ap.add_argument("--min-confidence", type=float, default=0.70)
    args = ap.parse_args()
    evs = sorted(p for p in JAL.iterdir() if p.is_dir() and (p / "flood_headline.tif").exists()
                 and (args.event == "all" or p.name == args.event))
    if not evs:
        print("PROBLEM: no Jal-SAR event with a headline map found")
        return 1
    have = {n: (EXPO / f).exists() for n, f in (("buildings", "buildings_openbuildings_v3.gpkg"),
                                                ("OSM power lines", "osm_power_lines.geojson"),
                                                ("OSM power points", "osm_power_points.geojson"),
                                                ("World Bank lines", "worldbank_grid.gpkg"),
                                                ("towers", "towers_corridor.geojson"))}
    have["WorldPop"] = POP.exists()
    print("m7_exposure_overlay.py: inputs " + ", ".join(f"{k} {'yes' if v else 'MISSING'}" for k, v in have.items()))
    cache, res = {}, []
    for e in evs:
        t0 = time.time()
        r = event_exposure(e, args, cache)
        res.append(r)
        print(f"  {e.name}: flood {r['flood_km2']} km2; people {(r.get('population') or {}).get('inside')}; "
              f"buildings {(r.get('buildings') or {}).get('inside')}; power line km "
              f"{(r.get('power') or {}).get('line_km_inside')}; towers {(r.get('towers') or {}).get('inside')} "
              f"({time.time() - t0:.0f} s)", flush=True)
    OUTSUM.mkdir(parents=True, exist_ok=True)

    def g(r, a, b):
        v = (r.get(a) or {}).get(b)
        return "n/a" if v is None else (f"{v:,}" if isinstance(v, int) else str(v))
    md = ["# M7: exposure inside and near the mapped flood", "",
          f"Near: within {args.near_m:g} m of the mapped flood. Population inside counts each 100 m WorldPop cell by "
          f"the share of it that is flooded; 'with edge' widens the flood by {args.edge_m:g} m. Buildings: Google Open "
          f"Buildings v3 with confidence {args.min_confidence:.2f} or more.", "",
          "| Event | Flood km2 | People inside | People with edge | Buildings inside | Buildings near | "
          "Power line km inside | Power points inside | Towers inside | Towers near |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        md.append(f"| {r['event']} | {r['flood_km2']} | {g(r, 'population', 'inside')} | "
                  f"{g(r, 'population', 'inside_buffer')} | {g(r, 'buildings', 'inside')} | {g(r, 'buildings', 'near')} | "
                  f"{g(r, 'power', 'line_km_inside')} | {g(r, 'power', 'points_inside')} | {g(r, 'towers', 'inside')} | "
                  f"{g(r, 'towers', 'near')} |")
    md += ["", "Inputs found: " + ", ".join(f"{k}: {'yes' if v else 'missing'}" for k, v in have.items()) + ".", "",
           "Limits: WorldPop is modelled, not counted. Open Buildings is detected from imagery and misses or merges "
           "some buildings under trees and on steep slopes. OpenStreetMap power and telecom data in Garhwal are "
           "incomplete. The World Bank line map is predicted (about 70 % accurate at 1 km by its own page). "
           "Towers are from OpenStreetMap only: the public OpenCelliD copy on Source Cooperative holds no Indian "
           "cells, so any tower nobody has mapped in OpenStreetMap is missing. Inside or near "
           "the mapped flood does not mean damaged.", "",
           "Sources: WorldPop (CC BY 4.0); Google Open Buildings v3; (c) OpenStreetMap contributors (ODbL); World "
           "Bank and ESMAP derived map of electricity transmission and distribution lines (CC BY 4.0)."]
    (OUTSUM / "M7_EXPOSURE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"  wrote {OUTSUM / 'M7_EXPOSURE.md'}")
    print("PASS  M7 exposure overlay")
    return 0


if __name__ == "__main__":
    sys.exit(main())
