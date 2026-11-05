"""Jal-SAR extensions, item M7: download the exposure layers that need no account.

For the box around the corridor (north 30.6, west 78.1, south 29.9, east 79.3):

1. OpenStreetMap, through the Overpass API (the same route Jal-SAR uses for
   roads), saved as GeoJSON with every tag kept:
     power_lines      power=line, minor_line, cable (ways)
     power_points     power=tower, pole, substation, transformer (nodes)
     telecom          man_made=mast or tower with tower:type=communication,
                      communication:mobile_phone=*, telecom=* (nodes and ways)
     facilities       amenity=hospital, clinic, doctors, school, college
                      (only used if asked for)
   Credit: (c) OpenStreetMap contributors, ODbL.
2. World Bank, "Derived map of global electricity transmission and
   distribution lines" (grid.gpkg, CC BY 4.0), the whole world file; it is
   clipped to the corridor later.
3. OpenCelliD, the copy on Source Cooperative (UN Smart Maps Group,
   cellid.pmtiles, CC BY-SA 4.0), the whole file (about 534 MB); the corridor
   part is read out of it later.

Large files resume where they stopped if the connection drops. Everything goes
into 06 Outputs\\jalsar\\extensions\\exposure\\ with a manifest (source URL, time, size,
SHA-256) and a log.

Run:  python m7_exposure_download.py
"""
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "06 Outputs" / "jalsar" / "extensions" / "exposure"
S, W, N, E = 29.9, 78.1, 30.6, 79.3
OVERPASS = ["https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
            "https://overpass.private.coffee/api/interpreter"]
QUERIES = {
    "power_lines": 'way["power"~"^(line|minor_line|cable)$"]',
    "power_points": 'node["power"~"^(tower|pole|substation|transformer)$"];way["power"="substation"]',
    "telecom": ('node["man_made"="mast"];node["man_made"="tower"]["tower:type"="communication"];'
                'way["man_made"="tower"]["tower:type"="communication"];node["communication:mobile_phone"];'
                'node["telecom"];way["telecom"]'),
    "facilities": ('node["amenity"~"^(hospital|clinic|doctors|school|college)$"];'
                   'way["amenity"~"^(hospital|clinic|doctors|school|college)$"]'),
}
FILES = {
    "worldbank_grid.gpkg": ["https://zenodo.org/records/3628142/files/grid.gpkg?download=1"],
    "opencellid_cellid.pmtiles": ["https://data.source.coop/smartmaps/opencellid/cellid.pmtiles"],
}
HEAD = {"User-Agent": "GBPUAT-final-year-project (Alaknanda flood exposure; contact via the project's GitHub repository)"}


def log(msg):
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(OUT / "exposure_log.txt", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def manifest(name, url, path):
    m = OUT / "exposure_manifest.json"
    data = json.loads(m.read_text(encoding="utf-8")) if m.exists() else {}
    data[name] = {"url": url, "file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path),
                  "downloaded": f"{dt.datetime.now():%Y-%m-%d %H:%M}"}
    m.write_text(json.dumps(data, indent=1), encoding="utf-8")


def to_geojson(osm):
    """Overpass JSON (out geom) -> GeoJSON: nodes as points, ways as lines or polygons."""
    feats = []
    for el in osm.get("elements", []):
        tags = el.get("tags", {})
        props = {"osm_type": el["type"], "osm_id": el["id"], **tags}
        if el["type"] == "node":
            geom = {"type": "Point", "coordinates": [el["lon"], el["lat"]]}
        elif el["type"] == "way" and el.get("geometry"):
            pts = [[p["lon"], p["lat"]] for p in el["geometry"]]
            closed = len(pts) > 3 and pts[0] == pts[-1]
            area = closed and ("building" in tags or "amenity" in tags or tags.get("power") == "substation"
                               or tags.get("man_made") == "tower")
            geom = {"type": "Polygon", "coordinates": [pts]} if area else {"type": "LineString", "coordinates": pts}
        else:
            continue
        feats.append({"type": "Feature", "properties": props, "geometry": geom})
    return {"type": "FeatureCollection", "features": feats,
            "attribution": "(c) OpenStreetMap contributors, ODbL 1.0"}


def overpass(name, q):
    target = OUT / f"osm_{name}.geojson"
    if target.exists():
        log(f"  {target.name}: already here, skipped")
        return True
    bbox = f"({S},{W},{N},{E})"
    body = "[out:json][timeout:300];(" + "".join(part + bbox + ";" for part in q.split(";") if part) + ");out geom;"
    for url in OVERPASS:
        for attempt in range(3):
            try:
                r = requests.post(url, data={"data": body}, headers=HEAD, timeout=400)
                if r.status_code == 429 or r.status_code >= 500:
                    raise RuntimeError(f"HTTP {r.status_code}")
                r.raise_for_status()
                gj = to_geojson(r.json())
                target.write_text(json.dumps(gj), encoding="utf-8")
                kinds = {}
                for f in gj["features"]:
                    kinds[f["geometry"]["type"]] = kinds.get(f["geometry"]["type"], 0) + 1
                log(f"  {target.name}: {len(gj['features'])} features {kinds} from {url}")
                manifest(f"osm_{name}", url + " " + body, target)
                return True
            except Exception as exc:
                log(f"  {name}: {url} attempt {attempt + 1} failed ({str(exc)[:200]})")
                time.sleep(30 * (attempt + 1))
    return False


def download(name, urls):
    target = OUT / name
    if target.exists():
        log(f"  {name}: already here, skipped")
        return True
    part = target.with_name(target.name + ".part")
    for url in urls:
        for attempt in range(6):
            try:
                have = part.stat().st_size if part.exists() else 0
                hdr = dict(HEAD, **({"Range": f"bytes={have}-"} if have else {}))
                with requests.get(url, headers=hdr, stream=True, timeout=120, allow_redirects=True) as r:
                    if r.status_code == 416:                      # the .part already holds the whole file
                        part.replace(target)
                        log(f"  {name}: done, {target.stat().st_size / 1e6:,.1f} MB from {url}")
                        manifest(name, url, target)
                        return True
                    r.raise_for_status()
                    if have and r.status_code != 206:             # server ignored the range: start again
                        have = 0
                    total = int(r.headers.get("Content-Length", 0)) + have
                    t0, last = time.time(), 0
                    with open(part, "ab" if have else "wb") as fh:
                        for chunk in r.iter_content(1 << 20):
                            fh.write(chunk)
                            have += len(chunk)
                            if time.time() - last > 60:
                                last = time.time()
                                rate = (have) / max(time.time() - t0, 1) / 1e6
                                log(f"  {name}: {have / 1e6:,.0f} of {total / 1e6:,.0f} MB ({rate:.2f} MB/s)")
                if total and have < total:                        # connection closed early without an error
                    raise IOError(f"short download, {have} of {total} bytes; will resume")
                part.replace(target)
                log(f"  {name}: done, {target.stat().st_size / 1e6:,.1f} MB from {url}")
                manifest(name, url, target)
                return True
            except Exception as exc:
                log(f"  {name}: {url} attempt {attempt + 1} failed ({str(exc)[:200]}); retrying")
                time.sleep(60)
    return False


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"m7_exposure_download.py start; box S {S}, W {W}, N {N}, E {E}")
    failed = []
    for name, q in QUERIES.items():
        if not overpass(name, q):
            failed.append(f"osm_{name}")
        time.sleep(10)                     # be polite to the public Overpass servers
    for name, urls in FILES.items():
        if not download(name, urls):
            failed.append(name)
    if failed:
        log(f"FINISHED WITH GAPS: {', '.join(failed)}. Run again later; finished files are skipped.")
        return 1
    log("PASS  M7 exposure downloads complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
