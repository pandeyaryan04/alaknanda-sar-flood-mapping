r"""
01_scene_inventory.py
=====================
Sentinel-1 scene inventory and event screening for the Alaknanda corridor.

WHAT THIS DOES
--------------
Before any flood detection is possible, one question has to be answered:
does a usable pair of Sentinel-1 images actually exist for the flood we want
to map? "Usable" has a specific meaning, taken from the project methodology:

  * a POST-event scene acquired within N days of the gauge peak, AND
  * a PRE-event scene from a low-flow period, on the SAME relative orbit and
    the SAME pass direction (ascending or descending).

The orbit and pass must match because a radar looks sideways. Two scenes taken
from different orbits view the same valley wall from different angles, and in
steep terrain that changes the brightness more than the flood does. Comparing
across orbits would measure the geometry, not the water.

This script queries Google Earth Engine, lists every Sentinel-1 scene over the
corridor for each candidate event, works out how much of the corridor each
scene actually covers, and then prints a plain-language verdict on whether each
event passes the screening rule.

It downloads nothing and computes nothing heavy. All the work happens on
Google's servers; the local machine only sends the request and prints the answer.

OUTPUTS (written into ./data/)
------------------------------
  corridor.geojson           the corridor polygon, open this in QGIS
  scene_inventory.csv        every scene found, one row each
  screening_verdict.csv      one row per event x orbit x pass, with PASS/FAIL

HOW TO RUN
----------
  (fyp) C:\\...\\fyp> python 01_scene_inventory.py

See docs/SETUP_RUNBOOK.md for the one-time Earth Engine authentication step.

================================================================================
HOW TO RUN THIS  (copy the lines, paste into Anaconda Prompt, press Enter)
================================================================================
    conda activate fyp
    cd /d "<project folder>\04 Code\FYP_code"
    python 01_scene_inventory.py

A few minutes. Queries Earth Engine metadata only, downloads no imagery.
Writes into data\ : corridor.geojson, scene_inventory.csv, screening_verdict.csv

ALREADY RUN, 21 August 2026. Both events passed at 100% corridor coverage:
  2023-08-14 event, orbit 129 ascending,  post 2023-08-16 (+2 d), pre 2022-11-01
  2021-10-19 event, orbit  63 descending, post 2021-10-21 (+2 d), pre 2020-11-19

Only re-run it if the corridor changes, the candidate events change, or the
EE_PROJECT line below changes.

If Earth Engine says you are not authenticated:
    earthengine authenticate
================================================================================
"""

__stage__ = "Sentinel-1 scene inventory and event screening"
__reads__ = ["Sentinel-1 GRD archive (Earth Engine)", "CWC gauge peak dates"]
__writes__ = ["corridor.geojson", "scene_inventory.csv", "screening_verdict.csv"]

import csv
import os
import sys
from datetime import datetime, timedelta

# ============================================================================
# CONFIGURATION - these are the only lines you should need to edit
# ============================================================================

# Your Earth Engine cloud project. If you make a new one for this project,
# change this string and nothing else.
EE_PROJECT = "gbpuat-fyp-alaknanda"

OUT_DIR = "data"

# Corridor centreline, upstream to downstream.
# APPROXIMATE town coordinates - good enough to decide which satellite scenes
# cover the reach, NOT good enough for the final study-area boundary. Replace
# these with a digitised river centreline in QGIS during week 2.
CORRIDOR_POINTS = [
    (78.9811, 30.2844),   # Rudraprayag        (Alaknanda x Mandakini confluence)
    (78.9000, 30.2400),   # intermediate bend
    (78.7833, 30.2222),   # Srinagar (Garhwal)
    (78.6800, 30.1900),   # intermediate bend
    (78.5981, 30.1469),   # Devprayag          (Alaknanda x Bhagirathi -> Ganga)
    (78.4300, 30.1300),   # intermediate bend
    (78.2676, 30.0869),   # Rishikesh
]

# Half-width of the corridor, in kilometres. 5 km either side of the centreline
# is generous for a valley this narrow and makes the coverage test meaningful.
BUFFER_KM = 5.0

# A scene must cover at least this percentage of the corridor to count.
MIN_COVER_PCT = 60.0

# How many days after the gauge peak a scene may be acquired and still be
# treated as a post-event observation.
POST_WINDOW_DAYS = 5

# Candidate events. Peak dates are from Central Water Commission gauge records
# as recorded in the project methodology. Verify each against the CWC source
# before it goes into the report.
EVENTS = [
    {
        "name": "2023-08-14 Alaknanda high stage",
        "peak": "2023-08-14",
        "note": "Srinagar 537.2 m (danger 536 m); Devprayag new HFL 469.97 m",
        "low_flow": ("2022-11-01", "2023-02-28"),
    },
    {
        "name": "2021-10-19 Alaknanda high stage",
        "peak": "2021-10-19",
        "note": "Devprayag new HFL 467.48 m; Karanprayag above HFL",
        "low_flow": ("2020-11-15", "2021-03-15"),
    },
]

# How far either side of the peak to list scenes. Wider than the post window on
# purpose, so you can see what was nearly available as well as what was.
SEARCH_BEFORE_DAYS = 14
SEARCH_AFTER_DAYS = 18

# ============================================================================
# End of configuration
# ============================================================================

try:
    import ee
except ImportError:
    sys.exit(
        "\nERROR: the earthengine-api package is not installed in this environment.\n"
        "Activate the project environment first:\n"
        "    conda activate fyp\n"
        "If that env does not exist yet, follow docs/SETUP_RUNBOOK.md step 2.\n"
    )


def init_ee():
    """Connect to Earth Engine, with a readable message if it is not set up."""
    try:
        ee.Initialize(project=EE_PROJECT)
    except Exception as exc:
        sys.exit(
            "\nERROR: could not connect to Earth Engine.\n"
            f"  project tried : {EE_PROJECT}\n"
            f"  message       : {exc}\n\n"
            "Two things to check:\n"
            "  1. Have you authenticated on this machine? Run:\n"
            "         earthengine authenticate\n"
            "     then follow the browser prompt.\n"
            "  2. Is EE_PROJECT at the top of this file the right project id?\n"
            "     Find it at https://code.earthengine.google.com (top right).\n"
        )
    print(f"Connected to Earth Engine, project '{EE_PROJECT}'.")


def build_corridor():
    """Buffer the centreline into a corridor polygon."""
    line = ee.Geometry.LineString([list(p) for p in CORRIDOR_POINTS])
    return line.buffer(BUFFER_KM * 1000.0)


def d(s):
    return datetime.strptime(s, "%Y-%m-%d")


def s1_collection(corridor, start, end):
    """Sentinel-1 IW scenes with both VV and VH, overlapping the corridor."""
    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(corridor)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )


def fetch_scenes(col, corridor, corridor_area, tag):
    """Pull scene metadata plus corridor coverage in a single server call."""

    def to_feature(img):
        overlap = (
            img.geometry()
            .intersection(corridor, ee.ErrorMargin(100))
            .area(maxError=100)
        )
        return ee.Feature(
            None,
            {
                "scene_id": img.get("system:index"),
                "datetime_utc": ee.Date(img.get("system:time_start")).format(
                    "YYYY-MM-dd HH:mm"
                ),
                "date": ee.Date(img.get("system:time_start")).format("YYYY-MM-dd"),
                "orbit": img.get("relativeOrbitNumber_start"),
                "pass": img.get("orbitProperties_pass"),
                "platform": img.get("platform_number"),
                "polarisation": ee.List(
                    img.get("transmitterReceiverPolarisation")
                ).join("+"),
                "resolution_m": img.get("resolution_meters"),
                "cover_pct": overlap.divide(corridor_area).multiply(100),
            },
        )

    try:
        info = ee.FeatureCollection(col.map(to_feature)).getInfo()
    except Exception as exc:
        print(f"  ! query failed for {tag}: {exc}")
        return []

    rows = []
    for feat in info.get("features", []):
        p = dict(feat["properties"])
        p["cover_pct"] = round(float(p.get("cover_pct") or 0.0), 1)
        p["window"] = tag
        rows.append(p)
    rows.sort(key=lambda r: r["datetime_utc"])
    return rows


def print_table(rows, title):
    print(f"\n  {title}")
    if not rows:
        print("    (no scenes found)")
        return
    print(
        "    {:<17} {:>6} {:<6} {:<4} {:>8}  {}".format(
            "acquired (UTC)", "orbit", "pass", "sat", "cover", "note"
        )
    )
    print("    " + "-" * 62)
    for r in rows:
        flag = "" if r["cover_pct"] >= MIN_COVER_PCT else "partial coverage"
        print(
            "    {:<17} {:>6} {:<6} {:<4} {:>7.1f}%  {}".format(
                r["datetime_utc"],
                r["orbit"],
                (r["pass"] or "")[:4].lower(),
                str(r["platform"]),
                r["cover_pct"],
                flag,
            )
        )


def screen_event(ev, corridor, corridor_area):
    """List scenes around the peak and in the low-flow window, then judge."""
    peak = d(ev["peak"])
    ev_start = (peak - timedelta(days=SEARCH_BEFORE_DAYS)).strftime("%Y-%m-%d")
    ev_end = (peak + timedelta(days=SEARCH_AFTER_DAYS)).strftime("%Y-%m-%d")

    print("\n" + "=" * 72)
    print(f"EVENT: {ev['name']}")
    print(f"  gauge peak : {ev['peak']}")
    print(f"  context    : {ev['note']}")
    print("=" * 72)

    around = fetch_scenes(
        s1_collection(corridor, ev_start, ev_end), corridor, corridor_area, "event"
    )
    low = fetch_scenes(
        s1_collection(corridor, ev["low_flow"][0], ev["low_flow"][1]),
        corridor,
        corridor_area,
        "low_flow",
    )

    print_table(around, f"Scenes {ev_start} to {ev_end} (around the peak)")
    print_table(low, f"Scenes {ev['low_flow'][0]} to {ev['low_flow'][1]} (low flow)")

    usable_post = [
        r
        for r in around
        if r["cover_pct"] >= MIN_COVER_PCT
        and peak <= d(r["date"]) <= peak + timedelta(days=POST_WINDOW_DAYS)
    ]
    usable_pre = [r for r in low if r["cover_pct"] >= MIN_COVER_PCT]

    tracks = sorted({(r["orbit"], r["pass"]) for r in usable_post})
    verdicts = []

    print(f"\n  SCREENING (post within {POST_WINDOW_DAYS} d of peak, "
          f"coverage >= {MIN_COVER_PCT:.0f}%, matching orbit and pass)")
    if not tracks:
        print("    FAIL - no post-event scene meets the coverage and timing rule.")
        verdicts.append(
            {
                "event": ev["name"],
                "orbit": "",
                "pass": "",
                "post_scene": "",
                "pre_scene": "",
                "verdict": "FAIL",
                "reason": "no qualifying post-event scene",
            }
        )
    for orbit, pas in tracks:
        posts = [r for r in usable_post if (r["orbit"], r["pass"]) == (orbit, pas)]
        pres = [r for r in usable_pre if (r["orbit"], r["pass"]) == (orbit, pas)]
        best_post = min(posts, key=lambda r: abs((d(r["date"]) - peak).days))
        ok = bool(pres)
        best_pre = max(pres, key=lambda r: r["cover_pct"]) if pres else None
        lag = (d(best_post["date"]) - peak).days
        print(
            f"    orbit {orbit} {str(pas).lower()}: "
            f"{'PASS' if ok else 'FAIL'} - "
            f"post {best_post['date']} (+{lag} d, {best_post['cover_pct']:.0f}%)"
            + (
                f", pre {best_pre['date']} ({best_pre['cover_pct']:.0f}%)"
                if ok
                else ", no matching low-flow scene on this track"
            )
        )
        verdicts.append(
            {
                "event": ev["name"],
                "orbit": orbit,
                "pass": pas,
                "post_scene": best_post["scene_id"],
                "post_date": best_post["date"],
                "post_lag_days": lag,
                "post_cover_pct": best_post["cover_pct"],
                "pre_scene": best_pre["scene_id"] if best_pre else "",
                "pre_date": best_pre["date"] if best_pre else "",
                "pre_cover_pct": best_pre["cover_pct"] if best_pre else "",
                "verdict": "PASS" if ok else "FAIL",
                "reason": "" if ok else "no low-flow scene on the same track",
            }
        )

    for r in around + low:
        r["event"] = ev["name"]
    return around + low, verdicts


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    init_ee()

    corridor = build_corridor()
    corridor_area = corridor.area(maxError=100)
    area_km2 = corridor_area.getInfo() / 1e6
    print(
        f"Corridor: {len(CORRIDOR_POINTS)} anchor points, "
        f"{BUFFER_KM:g} km buffer, {area_km2:,.0f} km2."
    )

    geo_path = os.path.join(OUT_DIR, "corridor.geojson")
    with open(geo_path, "w", encoding="utf-8") as fh:
        import json

        json.dump(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "name": "Alaknanda corridor, Rudraprayag to Rishikesh",
                            "buffer_km": BUFFER_KM,
                            "source": "approximate town anchors, replace with digitised centreline",
                        },
                        "geometry": corridor.getInfo(),
                    }
                ],
            },
            fh,
        )
    print(f"  wrote {geo_path}   <- open this in QGIS to check the reach")

    all_scenes, all_verdicts = [], []
    for ev in EVENTS:
        scenes, verdicts = screen_event(ev, corridor, corridor_area)
        all_scenes += scenes
        all_verdicts += verdicts

    print("\n" + "=" * 72)
    print("FILES WRITTEN")
    print("=" * 72)
    write_csv(
        os.path.join(OUT_DIR, "scene_inventory.csv"),
        all_scenes,
        ["event", "window", "scene_id", "datetime_utc", "date", "orbit", "pass",
         "platform", "polarisation", "resolution_m", "cover_pct"],
    )
    write_csv(
        os.path.join(OUT_DIR, "screening_verdict.csv"),
        all_verdicts,
        ["event", "orbit", "pass", "verdict", "post_scene", "post_date",
         "post_lag_days", "post_cover_pct", "pre_scene", "pre_date",
         "pre_cover_pct", "reason"],
    )

    passes = [v for v in all_verdicts if v["verdict"] == "PASS"]
    print("\n" + "=" * 72)
    print("WHAT THIS MEANS")
    print("=" * 72)
    if passes:
        print(f"  {len(passes)} usable orbit/pass combination(s) found.")
        print("  Pick one and record it in the methodology as the chosen event.")
        print("  If more than one passes, prefer the smallest post_lag_days,")
        print("  then the highest post_cover_pct.")
    else:
        print("  No combination passed. This is the fallback branch in the work")
        print("  plan, not a dead end. In order:")
        print("    1. Widen POST_WINDOW_DAYS to 7 and re-run.")
        print("    2. Lower MIN_COVER_PCT to 40 and re-run - a scene covering")
        print("       only the lower reach is still usable if you say so.")
        print("    3. Try the other candidate event.")
        print("    4. Shorten the corridor to below Devprayag and re-run.")
        print("  The two CSVs show which of these applies.")
    print()


if __name__ == "__main__":
    main()
