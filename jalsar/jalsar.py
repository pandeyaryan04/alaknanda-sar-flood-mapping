r"""
jalsar.py
=========
Jal-SAR ("jal", जल, is water in Hindi; SAR is the radar): Sentinel-1 flood
maps of the Alaknanda corridor with a trust layer. One command per job.

COMMANDS
--------
roads      Download the OpenStreetMap roads of the corridor once, through
           the Overpass API, into 06 Outputs\jalsar\osm\roads_corridor.gpkg.
           Needs the internet. Roads (c) OpenStreetMap contributors, ODbL.
maps       For each event: every model family's flood map, the agreement
           map, the headline map, the trust layer, village and road exposure,
           and summary.json, in 06 Outputs\jalsar\<event>\.
bulletin   For each event: the bulletin as an interactive HTML file and a PDF
           (printed by Microsoft Edge), in 06 Outputs\jalsar\<event>\.
webmap     One web map with every event, in 06 Outputs\jalsar\webmap\.
run        A new date: find the Sentinel-1 pass, download it through Earth
           Engine, strata for its orbit, maps, bulletin, web map. Settings in
           jalsar.env; the command line wins over it.
check-download
           Download 21 October 2021 again and compare it with the scene in
           05 Data (the downloader's acceptance test). Needs Earth Engine.
events     List the events Jal-SAR knows and whether each is done.

USAGE: HOW TO RUN THIS  (copy one line, paste into Anaconda Prompt, press Enter)
================================================================================
    conda activate fyp
    cd /d "<project folder>\04 Code\jalsar"

    python jalsar.py --selftest
    python jalsar.py events
    python jalsar.py roads
    python jalsar.py maps --event all
    python jalsar.py maps --event 2023-08-16 --force
    python jalsar.py bulletin --event all
    python jalsar.py webmap --force
    python jalsar.py check-download
    python jalsar.py run --date 2025-08-14

`maps` takes about 5 to 15 minutes per event on the RTX 3050 (three models
on the whole corridor; estimate); a bulletin about a minute; the web map one
to three minutes; `run` adds a 5 to 20 minute download. Anything already
made is kept unless --force. The last line says PASS.
================================================================================
"""

__stage__ = "Jal-SAR: command line"
__reads__ = ["see jalsar_core.py"]
__writes__ = ["06 Outputs/jalsar/ (maps, bulletins, web map, scenes, events.json)",
              "FYP_code/data/strata_orbit<nnn>_<pass>/ (run, new orbits only)"]

import argparse
import json
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

import jalsar_core as gc

mc = gc.mc
OVERPASS = "https://overpass-api.de/api/interpreter"


def cmd_events(gp, args):
    evs = gc.load_events(gp, only_existing=False)
    print(f"jalsar.py events: {len(evs)} known")
    for e in evs:
        state = ("done" if (gp.event_dir(e.id) / "summary.json").exists() else
                 "scene ready" if e.scene_path().exists() else "scene missing")
        ref = f"reference {e.validated_year}" if e.validated_year else "no reference"
        print(f"  {e.id:22s} {e.label or ''} ({ref}): {state}")
    print("PASS")
    return 0


def overpass_query(bounds):
    w, s, e, n = bounds
    return (f'[out:json][timeout:300];way["highway"]({s:.5f},{w:.5f},{n:.5f},{e:.5f});'
            "out geom tags;")


def roads_from_overpass(data, corridor):
    """GeoDataFrame of road lines from an Overpass JSON answer, cut to the corridor."""
    import geopandas as gpd
    from shapely.geometry import LineString
    rows = []
    for el in data.get("elements", []):
        if el.get("type") != "way" or len(el.get("geometry", [])) < 2:
            continue
        tags = el.get("tags", {})
        rows.append({"osm_id": int(el["id"]), "highway": tags.get("highway", ""),
                     "name": tags.get("name", ""), "ref": tags.get("ref", ""),
                     "geometry": LineString([(p["lon"], p["lat"]) for p in el["geometry"]])})
    g = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326") if rows else \
        gpd.GeoDataFrame({"osm_id": [], "highway": [], "name": [], "ref": []},
                         geometry=gpd.GeoSeries([], crs="EPSG:4326"))
    if len(g) and corridor is not None:
        g = gpd.clip(g, corridor.to_crs("EPSG:4326"))
        g = g[~g.geometry.is_empty].reset_index(drop=True)
    return g


def cmd_roads(gp, args, fetch=None):
    import geopandas as gpd
    print("jalsar.py roads: OpenStreetMap roads of the corridor (Overpass API)")
    if gp.roads.exists() and not args.force:
        print(f"  already there, kept: {gp.roads}  (--force downloads again)")
        print("PASS")
        return 0
    if not gp.corridor.exists():
        mc.fail(f"The corridor polygon is missing:\n  {gp.corridor}")
    corridor = gpd.read_file(gp.corridor)
    q = overpass_query(corridor.to_crs("EPSG:4326").total_bounds)
    print("  asking the Overpass API (1 to 3 minutes) ...", flush=True)
    t0 = time.time()
    if fetch is None:
        req = urllib.request.Request(OVERPASS, data=urllib.parse.urlencode({"data": q}).encode(),
                                     headers={"User-Agent": "Jal-SAR (GBPUAT student project)"})
        try:
            with urllib.request.urlopen(req, timeout=360) as r:
                raw = r.read()
        except Exception as e:                  # noqa: BLE001
            mc.fail(f"The Overpass API did not answer ({e}). Check the internet and try again "
                    "later; the bulletins say 'roads not available' meanwhile.")
    else:
        raw = fetch(q)
    data = json.loads(raw)
    g = roads_from_overpass(data, corridor)
    gp.roads.parent.mkdir(parents=True, exist_ok=True)
    (gp.roads.parent / "overpass_answer.json").write_bytes(raw)
    g.to_file(gp.roads, driver="GPKG")
    by = g["highway"].value_counts().to_dict() if len(g) else {}
    mc.save_json(gp.roads.parent / "roads_record.json",
                 {"query": q, "endpoint": OVERPASS, "ways": int(len(g)), "by_class": by,
                  "seconds": round(time.time() - t0, 1),
                  "credit": "(c) OpenStreetMap contributors, ODbL",
                  "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(f"  {len(g):,} road pieces inside the corridor: "
          + ", ".join(f"{k} {v}" for k, v in list(by.items())[:8]))
    print(f"  wrote {gp.roads}")
    print("PASS")
    return 0


def cmd_maps(gp, args):
    fams = tuple(f.strip() for f in args.families.split(",") if f.strip())
    bad = [f for f in fams if f not in gc.FAMILIES]
    if bad:
        mc.fail(f"Unknown model family: {', '.join(bad)}. Use {', '.join(gc.FAMILIES)}.")
    evs = gc.load_events(gp) if args.event == "all" else [gc.find_event(gp, args.event)]
    if not evs:
        mc.fail("No event has its scene on disk. Check 05 Data\\sentinel1.")
    print(f"jalsar.py maps: {len(evs)} event(s), families {', '.join(fams)}")
    if not gp.roads.exists():
        print("  note: no roads file yet (run 'jalsar.py roads'); the road table is left out")
    for e in evs:
        if not e.scene_path().exists():
            mc.fail(f"The scene of {e.id} is missing:\n  {e.scene}")
        gc.make_event(gp, e, fams, force=args.force, cpu_ok=args.cpu_ok, tile=args.tile,
                      overlap=args.overlap)
    print("PASS")
    return 0


def cmd_bulletin(gp, args):
    import jalsar_bulletin as gb
    evs = ([e for e in gc.load_events(gp, only_existing=False)
            if (gp.event_dir(e.id) / "summary.json").exists()]
           if args.event == "all" else [gc.find_event(gp, args.event)])
    if not evs:
        mc.fail("No event has maps yet. Run 'python jalsar.py maps' first.")
    print(f"jalsar.py bulletin: {len(evs)} event(s)")
    bad = 0
    for e in evs:
        r = gb.make_bulletin(gp, e, force=args.force, pdf=not args.no_pdf)
        bad += bool(r.get("pdf_problem"))
    if bad:
        print(f"  {bad} PDF(s) not made; the HTML bulletins are fine. See the PROBLEM line above.")
        return 1
    print("PASS")
    return 0


def cmd_webmap(gp, args):
    import jalsar_webmap as gw
    print("jalsar.py webmap")
    gw.build_webmap(gp, force=args.force, flood_cell=args.flood_cell)   # over the size limit: a warning
    print("PASS")
    return 0


def cmd_run(gp, args):
    import jalsar_runner as gr
    s = gr.settings(args)
    gr.run(gp, s, force=args.force, cpu_ok=args.cpu_ok, pdf=not args.no_pdf)
    print("PASS")
    return 0


def cmd_check(gp, args):
    import jalsar_runner as gr
    res = gr.check_download(gp, force=args.force)
    print("PASS" if res["same"] else "CHECK DIFFERENT (read CHECK REPORT.md)")
    return 0 if res["same"] else 1


def main(argv=None):
    mc.setup_console()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=None, help="project folder (default: found from here)")
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("events")
    r = sub.add_parser("roads")
    r.add_argument("--force", action="store_true")
    m = sub.add_parser("maps")
    m.add_argument("--event", default="all", help="an event id or date, or all (default)")
    m.add_argument("--families", default="rf,unet,terramind")
    m.add_argument("--force", action="store_true")
    m.add_argument("--cpu-ok", action="store_true")
    m.add_argument("--tile", type=int, default=512)
    m.add_argument("--overlap", type=int, default=64)
    b = sub.add_parser("bulletin")
    b.add_argument("--event", default="all", help="an event id or date, or all (default)")
    b.add_argument("--force", action="store_true")
    b.add_argument("--no-pdf", action="store_true", help="HTML only")
    w = sub.add_parser("webmap")
    w.add_argument("--force", action="store_true")
    w.add_argument("--flood-cell", type=float, default=10.0,
                   help="picture cell size in metres (20 halves the page size)")
    r = sub.add_parser("run")
    r.add_argument("--date", help="YYYY-MM-DD (else EVENT_DATE in jalsar.env)")
    r.add_argument("--search-days", type=int)
    r.add_argument("--min-coverage", type=float)
    r.add_argument("--families", help="e.g. rf,unet,terramind (else MODELS in jalsar.env)")
    r.add_argument("--pre-event", choices=["on", "off"])
    r.add_argument("--force", action="store_true")
    r.add_argument("--cpu-ok", action="store_true")
    r.add_argument("--no-pdf", action="store_true")
    c = sub.add_parser("check-download")
    c.add_argument("--force", action="store_true", help="download again even if already done")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.cmd:
        ap.print_help()
        return 1
    gp = gc.GPaths(args.project)
    return {"events": cmd_events, "roads": cmd_roads, "maps": cmd_maps, "bulletin": cmd_bulletin,
            "webmap": cmd_webmap, "run": cmd_run, "check-download": cmd_check}[args.cmd](gp, args)


def selftest():
    print("jalsar.py self-test")
    import geopandas as gpd
    with tempfile.TemporaryDirectory(prefix="jalsar selftest ", ignore_cleanup_errors=True) as td:
        root = Path(td) / "Final Year Project"
        gp = gc.build_test_project(root)
        gp.roads.unlink()
        # roads: a fake Overpass answer, one way inside the corridor and one outside
        cor = gpd.read_file(gp.corridor).to_crs("EPSG:4326")
        w, s, e, n = cor.total_bounds
        cx, cy = cor.geometry.iloc[0].representative_point().coords[0]
        fake = {"elements": [
            {"type": "way", "id": 1, "tags": {"highway": "primary", "name": "NH 7"},
             "geometry": [{"lon": cx - 0.001, "lat": cy}, {"lon": cx + 0.001, "lat": cy}]},
            {"type": "way", "id": 2, "tags": {"highway": "track"},
             "geometry": [{"lon": e + 1, "lat": n + 1}, {"lon": e + 1.1, "lat": n + 1.1}]}]}
        ns = argparse.Namespace(force=False)
        ok_roads = cmd_roads(gp, ns, fetch=lambda q: json.dumps(fake).encode()) == 0
        g = gpd.read_file(gp.roads)
        roads_cut = len(g) == 1 and g["highway"].iloc[0] == "primary"
        ok_events = main(["--project", str(root), "events"]) == 0
        ok_maps = main(["--project", str(root), "maps", "--event", "2021-10-21", "--cpu-ok",
                        "--tile", "64", "--overlap", "16"]) == 0
        summ = json.loads((gp.event_dir("2021-10-21_orbit063") / "summary.json").read_text())
        has_roads = summ["roads"] is not None and summ["roads"]["credit"].startswith("Roads")
        ok_bull = main(["--project", str(root), "bulletin", "--event", "2021-10-21", "--no-pdf"]) == 0
        ok_web = main(["--project", str(root), "webmap"]) == 0
        made = ((gp.event_dir("2021-10-21_orbit063") / "bulletin_2021-10-21_orbit063.html").exists()
                and (gp.gj / "webmap" / "jalsar_map.html").exists())
        q = overpass_query((78.0, 30.0, 78.9, 30.4))
        ok_query = q.startswith("[out:json]") and "(30.00000,78.00000,30.40000,78.90000)" in q
    checks = [("roads: Overpass answer cut to the corridor", ok_roads and roads_cut),
              ("events listed", ok_events), ("maps for one event", ok_maps),
              ("summary carries the road table and its credit", has_roads),
              ("Overpass bounding box in south, west, north, east order", ok_query),
              ("bulletin and web map from the command line", ok_bull and ok_web and made)]
    for name, c in checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {name}")
    good = all(c for _, c in checks)
    print("SELFTEST PASS" if good else "SELFTEST FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(mc.cli(main))
