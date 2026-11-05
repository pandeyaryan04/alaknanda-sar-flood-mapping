"""Jal-SAR extensions, M7 check 2 (22 Oct 2026): does the OpenCelliD copy hold any cells in India at all?

Walks the file's tile index (not the tile data) and counts, per zoom, how many tiles
exist inside: the corridor box, the whole of India (68 to 98 E, 6 to 37 N) and three
control areas (London, Pakistan around Lahore, Nepal around Kathmandu). Then reads the
zoom-14 tiles in the India box (up to 300) and counts cells by country code (mcc;
India is 404 and 405). Read only. Run: python m7_towers_check2.py
"""
import gzip
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PM = ROOT / "06 Outputs" / "jalsar" / "extensions" / "exposure" / "opencellid_cellid.pmtiles"
AREAS = {"corridor": (29.9, 78.1, 30.6, 79.3), "India": (6.0, 68.0, 37.0, 98.0),
         "London": (51.3, -0.5, 51.7, 0.3), "Lahore": (31.3, 74.1, 31.7, 74.5),
         "Kathmandu": (27.6, 85.2, 27.8, 85.5)}


def tile(lon, lat, z):
    n = 2 ** z
    return int((lon + 180) / 360 * n), int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)


def main():
    from pmtiles.reader import MmapSource, Reader
    from pmtiles.tile import deserialize_directory, tileid_to_zxy
    import mapbox_vector_tile as mvt
    with open(PM, "rb") as fh:
        get = MmapSource(fh)
        hdr = Reader(get).header()
        ranges = {}
        for name, (S, W, N, E) in AREAS.items():
            for z in range(0, 15):
                x0, y0 = tile(W, N, z)
                x1, y1 = tile(E, S, z)
                ranges[(name, z)] = (x0, x1, y0, y1)
        count = Counter()
        per_zoom = Counter()
        india14 = []
        stack = [(hdr["root_offset"], hdr["root_length"])]
        n_entries = 0
        while stack:
            off, ln = stack.pop()
            for e in deserialize_directory(get(off, ln)):
                if e.run_length == 0:
                    stack.append((hdr["leaf_directory_offset"] + e.offset, e.length))
                    continue
                for i in range(e.run_length):
                    z, x, y = tileid_to_zxy(e.tile_id + i)
                    n_entries += 1
                    per_zoom[z] += 1
                    for name in AREAS:
                        x0, x1, y0, y1 = ranges[(name, z)]
                        if x0 <= x <= x1 and y0 <= y <= y1:
                            count[(name, z)] += 1
                            if name == "India" and z == 14 and len(india14) < 300:
                                india14.append((x, y, hdr["tile_data_offset"] + e.offset, e.length))
        print(f"tile index walked: {n_entries} tiles (header says {hdr['addressed_tiles_count']})")
        print("tiles per zoom in the whole file:", dict(sorted(per_zoom.items())))
        for name in AREAS:
            print(f"{name:10s}", " ".join(f"z{z}:{count[(name, z)]}" for z in range(0, 15)))
        mcc = Counter()
        for x, y, off, ln in india14:
            data = get(off, ln)
            if data[:2] == b"\x1f\x8b":
                data = gzip.decompress(data)
            for layer in mvt.decode(data).values():
                for f in layer.get("features", []):
                    mcc[f.get("properties", {}).get("mcc")] += 1
        print(f"zoom-14 tiles read inside the India box: {len(india14)}; cells by mcc (top 10): {mcc.most_common(10)}")
    print("DONE  m7_towers_check2")


if __name__ == "__main__":
    main()
