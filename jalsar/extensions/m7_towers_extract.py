"""Jal-SAR extensions, item M7: mobile towers for the corridor (OpenCelliD copy plus OpenStreetMap).

1. OpenCelliD: reads the corridor's part of cellid.pmtiles (the copy on Source
   Cooperative, downloaded by m7_exposure_download.py) at its most detailed zoom.
   Each OpenCelliD record is one radio cell, and one tower carries several, so
   cells closer than --merge-m (default 50 m) are merged into one tower, keeping
   how many cells and which radio types (GSM, UMTS, LTE, NR) it holds. OpenCelliD
   positions are crowd-sourced estimates, not surveyed tower positions; that is
   stated wherever the layer is shown.
2. OpenStreetMap: masts and communication towers (osm_telecom.geojson from the
   same download).

Needs the pmtiles and mapbox-vector-tile packages (M INSTALL EXTRA.bat).

Output: 06 Outputs\\jalsar\\extensions\\exposure\\towers_corridor.geojson (EPSG:4326), one point per
tower, with source (opencellid, osm), cells, radios; and towers_record.json.

Run:  python m7_towers_extract.py
"""
import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXPO = ROOT / "06 Outputs" / "jalsar" / "extensions" / "exposure"
S, W, N, E = 29.9, 78.1, 30.6, 79.3


def lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


def tile_px_to_lonlat(z, x, y, px, py, extent):
    n = 2 ** z
    lon = (x + px / extent) / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + py / extent) / n))))
    return lon, lat


def opencellid_points(path, zoom=None):
    from pmtiles.reader import MmapSource, Reader
    import mapbox_vector_tile as mvt
    pts = []
    with open(path, "rb") as fh:
        reader = Reader(MmapSource(fh))
        hdr = reader.header()
        z = int(zoom if zoom is not None else hdr["max_zoom"])
        x0, y0 = lonlat_to_tile(W, N, z)
        x1, y1 = lonlat_to_tile(E, S, z)
        print(f"  OpenCelliD tiles: zoom {z}, x {x0} to {x1}, y {y0} to {y1}", flush=True)
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                data = reader.get(z, x, y)
                if not data:
                    continue
                if data[:2] == b"\x1f\x8b":
                    data = gzip.decompress(data)
                layers = mvt.decode(data, default_options={"y_coord_down": True})
                for lname, layer in layers.items():
                    ext = layer.get("extent", 4096)
                    for f in layer.get("features", []):
                        g = f.get("geometry", {})
                        if g.get("type") != "Point":
                            continue
                        px, py = g["coordinates"][:2]
                        lon, lat = tile_px_to_lonlat(z, x, y, px, py, ext)
                        if W <= lon <= E and S <= lat <= N:
                            pts.append((lon, lat, f.get("properties", {})))
    return pts, z


def merge(points, merge_m):
    """Greedy grid merge: points within merge_m of a tower's first point join it."""
    if not points:
        return []
    lat0 = (S + N) / 2
    mx = 111320 * math.cos(math.radians(lat0))
    xy = np.array([[p[0] * mx, p[1] * 110540] for p in points])
    cell = {}
    towers = []
    for i, (x, y) in enumerate(xy):
        key = (int(x // merge_m), int(y // merge_m))
        hit = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for t in cell.get((key[0] + dx, key[1] + dy), []):
                    if (towers[t]["x"] - x) ** 2 + (towers[t]["y"] - y) ** 2 <= merge_m ** 2:
                        hit = t
                        break
                if hit is not None:
                    break
            if hit is not None:
                break
        props = points[i][2]
        radio = str(props.get("radio") or props.get("Radio") or "").upper()
        if hit is None:
            towers.append({"x": x, "y": y, "lon": points[i][0], "lat": points[i][1], "cells": 1,
                           "radios": {radio} if radio else set()})
            cell.setdefault(key, []).append(len(towers) - 1)
        else:
            towers[hit]["cells"] += 1
            if radio:
                towers[hit]["radios"].add(radio)
    return towers


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merge-m", type=float, default=50.0)
    ap.add_argument("--zoom", type=int, default=None)
    args = ap.parse_args()
    feats, rec = [], {"merge_m": args.merge_m}
    pm = EXPO / "opencellid_cellid.pmtiles"
    if pm.exists():
        pts, z = opencellid_points(pm, args.zoom)
        towers = merge(pts, args.merge_m)
        rec.update(opencellid_cells=len(pts), opencellid_towers=len(towers), opencellid_zoom=z)
        for t in towers:
            feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(t["lon"], 6),
                                                                                         round(t["lat"], 6)]},
                          "properties": {"source": "opencellid", "cells": t["cells"],
                                         "radios": ",".join(sorted(t["radios"]))}})
        print(f"  OpenCelliD: {len(pts):,} radio cells -> {len(towers):,} towers (merged within {args.merge_m:g} m)")
    else:
        print(f"  OpenCelliD file not found ({pm}); OpenStreetMap only")
        rec["opencellid"] = "file not found"
    osm = EXPO / "osm_telecom.geojson"
    n_osm = 0
    if osm.exists():
        for f in json.loads(osm.read_text(encoding="utf-8")).get("features", []):
            g = f["geometry"]
            if g["type"] == "Point":
                c = g["coordinates"]
            elif g["type"] in ("LineString", "Polygon"):
                ring = g["coordinates"] if g["type"] == "LineString" else g["coordinates"][0]
                c = [sum(p[0] for p in ring) / len(ring), sum(p[1] for p in ring) / len(ring)]
            else:
                continue
            p = f["properties"]
            feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": c},
                          "properties": {"source": "osm", "osm_id": p.get("osm_id"),
                                         "kind": p.get("man_made") or p.get("telecom") or "telecom",
                                         "operator": p.get("operator", "")}})
            n_osm += 1
    rec["osm_towers"] = n_osm
    if not feats:
        print("PROBLEM: no tower data found (run the M7 downloads first)")
        return 1
    out = EXPO / "towers_corridor.geojson"
    out.write_text(json.dumps({"type": "FeatureCollection", "features": feats,
                               "attribution": "OpenCelliD (CC BY-SA 4.0), copy on Source Cooperative; "
                                              "(c) OpenStreetMap contributors (ODbL)"}), encoding="utf-8")
    (EXPO / "towers_record.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"  wrote {out}: {len(feats):,} points ({n_osm} from OpenStreetMap)")
    print("PASS  M7 towers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
