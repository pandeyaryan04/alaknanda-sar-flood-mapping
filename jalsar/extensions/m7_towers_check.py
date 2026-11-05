"""Jal-SAR extensions, M7 check (22 Oct 2026): why m7_towers_extract.py found 0 OpenCelliD cells.

Reads only the header, metadata and a few tiles of opencellid_cellid.pmtiles, and prints:
  - the header (zoom range, bounds, compression) and the metadata JSON (layer names, fields)
  - for every zoom level: how many corridor tiles exist, features per layer and geometry type
  - the same for three test places: Dehradun, Delhi and Rishikesh town
Writes nothing except its own printout. Run: python m7_towers_check.py
"""
import gzip
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PM = ROOT / "06 Outputs" / "jalsar" / "extensions" / "exposure" / "opencellid_cellid.pmtiles"
BOX = (29.9, 78.1, 30.6, 79.3)  # S, W, N, E
PLACES = {"Rishikesh": (78.29, 30.11), "Dehradun": (78.03, 30.32), "Delhi": (77.21, 28.61)}


def tile(lon, lat, z):
    n = 2 ** z
    return int((lon + 180) / 360 * n), int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)


def unpack(data):
    if data[:2] == b"\x1f\x8b":
        return gzip.decompress(data)
    return data


def describe(reader, mvt, z, x, y):
    data = reader.get(z, x, y)
    if not data:
        return None
    out = Counter()
    try:
        layers = mvt.decode(unpack(data), default_options={"y_coord_down": True})
    except Exception as exc:
        return {"error": str(exc)[:200], "bytes": len(data), "first_bytes": data[:4].hex()}
    sample = None
    for lname, layer in layers.items():
        for f in layer.get("features", []):
            out[(lname, f.get("geometry", {}).get("type"))] += 1
            if sample is None:
                sample = {"layer": lname, "type": f.get("geometry", {}).get("type"),
                          "props": f.get("properties", {})}
    return {"bytes": len(data), "features": {f"{k[0]}/{k[1]}": v for k, v in out.items()}, "sample": sample}


def main():
    from pmtiles.reader import MmapSource, Reader
    import mapbox_vector_tile as mvt
    print(f"file: {PM} ({PM.stat().st_size / 1e6:.1f} MB)")
    with open(PM, "rb") as fh:
        reader = Reader(MmapSource(fh))
        hdr = reader.header()
        print("header:", json.dumps({k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in hdr.items()}, indent=1))
        try:
            meta = reader.metadata()
            print("metadata:", json.dumps(meta, indent=1)[:3000])
        except Exception as exc:
            print("metadata: could not read", exc)
        zmin, zmax = int(hdr.get("min_zoom", 0)), int(hdr.get("max_zoom", 14))
        S, W, N, E = BOX
        for z in range(zmin, zmax + 1):
            x0, y0 = tile(W, N, z)
            x1, y1 = tile(E, S, z)
            have, feats, first = 0, Counter(), None
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    d = describe(reader, mvt, z, x, y)
                    if d is None:
                        continue
                    have += 1
                    if "error" in d:
                        first = first or d
                        continue
                    feats.update(d["features"])
                    first = first or d
            total = (x1 - x0 + 1) * (y1 - y0 + 1)
            print(f"corridor zoom {z}: {have} of {total} tiles present; features {dict(feats)}")
            if first:
                print(f"   first tile: {json.dumps(first)[:600]}")
        for name, (lon, lat) in PLACES.items():
            for z in sorted({zmin, (zmin + zmax) // 2, zmax}):
                d = describe(reader, mvt, z, *tile(lon, lat, z))
                print(f"{name} zoom {z}: {json.dumps(d)[:400] if d else 'no tile'}")
    print("DONE  m7_towers_check")


if __name__ == "__main__":
    main()
