"""
t07_download.py - the downloads that have an API worth trusting.

Subcommands:

  gee-inventory   corridor-clear percentage per Sentinel-2 date in a window.
                  Finds the partner date for
                  20 July 2023 by measuring it rather than assuming it.
  gee-export      submit the Sentinel-2 exports for chosen dates, with the
                  same bands, scale and CRS as 03_export_remaining.js.
  gee-s1-timing   every Sentinel-1 acquisition near a date, with orbit, pass
                  direction, corridor coverage and lag (used for the
                  June 2021 question).
  asf             submit and poll ASF HyP3 On-Demand RTC at 10 m, then
                  download.
  earthdata       search NASA Earthdata over the corridor and report coverage
                  BEFORE downloading. For the HMA 8 m DEM.
  imd             IMD gridded daily rainfall for a range of years.

Bhuvan, Census of India and the Glacial Lake Atlas have no API worth
trusting. They stay manual and are written out step by step in
`DATA ACQUISITION - full plan, semester 1 and 2`.

Every optional package is imported inside its own subcommand, so a missing
one stops that subcommand with a clear message and leaves the others working.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import common as C

TOOL = "t07_download"

MISSING = """
{pkg} is not installed in this environment.

  conda activate fyp
  pip install {pip}

If that fails or you would rather not add the package, the manual route for
this item is written out step by step in:
  05 Data\\DATA ACQUISITION - full plan, semester 1 and 2, 9 September 2026.md
  section {section}
"""


def need(pkg, pip, section):
    try:
        return __import__(pkg)
    except ImportError:
        raise RuntimeError(MISSING.format(pkg=pkg, pip=pip, section=section))


def corridor_ee(cfg, ee):
    import json

    gj = json.loads(cfg.path("CORRIDOR").read_text(encoding="utf-8"))
    feats = gj.get("features", [gj])
    geom = feats[0].get("geometry", feats[0])
    return ee.Geometry(geom)


def init_ee(cfg):
    ee = need("ee", "earthengine-api", "A1")
    project = cfg.str("GEE_PROJECT")
    try:
        ee.Initialize(project=project)
    except Exception:
        C.say(f"  authenticating Earth Engine for project {project}")
        C.say("  a browser window will open; sign in as the account that owns "
              "the project")
        ee.Authenticate()
        ee.Initialize(project=project)
    C.say(f"  Earth Engine ready, project {project}")
    return ee


# --------------------------------------------------------------------------

def cmd_gee_inventory(cfg, a):
    ee = init_ee(cfg)
    corridor = corridor_ee(cfg, ee)
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(corridor)
           .filterDate(a.start, a.end))
    dates = col.aggregate_array("system:time_start").getInfo()
    seen = sorted({ee.Date(d).format("YYYY-MM-dd").getInfo() for d in dates}) \
        if dates else []
    if not seen:
        C.say("  no Sentinel-2 acquisitions in that window over the corridor")
        return 0

    C.say("")
    C.say("  date         covered %   cloud %   snow %   CLEAR %")
    rows = []
    for d in seen:
        day = col.filterDate(d, ee.Date(d).advance(1, "day"))
        scl = day.select("SCL").mosaic().unmask(0).clip(corridor)
        covered = scl.gt(0).rename("covered")
        cloud = scl.eq(3).Or(scl.gte(8).And(scl.lte(10))).rename("cloud")
        snow = scl.eq(11).rename("snow")
        clear = covered.And(cloud.Not()).And(snow.Not()).rename("clear")
        m = (covered.addBands(cloud).addBands(snow).addBands(clear).toFloat()
             .reduceRegion(reducer=ee.Reducer.mean(), geometry=corridor,
                           scale=100, maxPixels=int(1e9), bestEffort=True)
             .getInfo())
        rows.append((d, 100 * m["covered"], 100 * m["cloud"],
                     100 * m["snow"], 100 * m["clear"]))
        C.say(f"  {d}   {rows[-1][1]:7.2f}   {rows[-1][2]:7.2f}  "
              f"{rows[-1][3]:6.2f}   {rows[-1][4]:7.2f}")

    best = sorted(rows, key=lambda r: -r[4])[:3]
    C.say("")
    C.say("  best three by corridor-clear:")
    for d, _, _, _, c in best:
        C.say(f"    {d}  {c:.2f} %")
    C.say("")
    C.say("  A single date reaches 91 to 93 per cent of this corridor. Pick a")
    C.say("  pair two days apart, which are different relative orbits, and")
    C.say("  check the mosaic with t02 before digitising anything.")
    return 0


def cmd_gee_export(cfg, a):
    ee = init_ee(cfg)
    corridor = corridor_ee(cfg, ee)
    crs = cfg.str("CRS_METRIC")
    drive = cfg.str("GEE_DRIVE_FOLDER")
    n = 0
    for d in a.dates:
        day = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
               .filterBounds(corridor)
               .filterDate(d, ee.Date(d).advance(1, "day"))
               .mosaic().clip(corridor))
        stamp = d.replace("-", "")
        base = f"{a.tag}_{stamp}"

        ee.batch.Export.image.toDrive(
            image=day.select(["B2", "B3", "B4", "B8", "B11", "SCL"]).toInt16(),
            description=f"{base}_10m", folder=drive,
            fileNamePrefix=f"{base}_10m", region=corridor,
            scale=10, crs=crs, maxPixels=int(1e13),
        ).start()

        mndwi = day.normalizedDifference(["B3", "B11"]).rename("MNDWI")
        ndwi = day.normalizedDifference(["B3", "B8"]).rename("NDWI")
        ee.batch.Export.image.toDrive(
            image=mndwi.addBands(ndwi).toFloat(),
            description=f"{base}_indices_10m", folder=drive,
            fileNamePrefix=f"{base}_indices_10m", region=corridor,
            scale=10, crs=crs, maxPixels=int(1e13),
        ).start()
        n += 2
        C.say(f"  queued {base}_10m and {base}_indices_10m")

    C.say("")
    C.say(f"  {n} tasks submitted to Drive folder {drive}.")
    C.say("  Watch them at code.earthengine.google.com, Tasks tab.")
    C.say("  Each file should land at 64 to 79 MB. Much smaller means the")
    C.say("  export was clipped or split into tiles.")
    return 0


def cmd_gee_s1_timing(cfg, a):
    ee = init_ee(cfg)
    corridor = corridor_ee(cfg, ee)
    target = ee.Date(a.date)
    col = (ee.ImageCollection("COPERNICUS/S1_GRD")
           .filterBounds(corridor)
           .filterDate(target.advance(-a.window, "day"),
                       target.advance(a.window + 1, "day"))
           .filter(ee.Filter.eq("instrumentMode", "IW")))
    info = col.getInfo().get("features", [])
    if not info:
        C.say(f"  no Sentinel-1 IW acquisitions within {a.window} days of {a.date}")
        return 0

    C.say("")
    C.say("  date         orbit  pass        coverage %   lag d")
    rows = []
    seen = set()
    for f in info:
        p = f["properties"]
        d = ee.Date(p["system:time_start"]).format("YYYY-MM-dd").getInfo()
        orb = p.get("relativeOrbitNumber_start")
        direc = p.get("orbitProperties_pass", "")
        key = (d, orb, direc)
        if key in seen:
            continue
        seen.add(key)
        day = (col.filterDate(d, ee.Date(d).advance(1, "day"))
               .filter(ee.Filter.eq("relativeOrbitNumber_start", orb)))
        cov = (day.select("VV").mosaic().mask().unmask(0).clip(corridor)
               .reduceRegion(reducer=ee.Reducer.mean(), geometry=corridor,
                             scale=100, maxPixels=int(1e9), bestEffort=True)
               .getInfo())
        pct = 100 * list(cov.values())[0]
        lag = (ee.Date(d).difference(target, "day")).getInfo()
        rows.append((d, orb, direc, pct, lag))
        C.say(f"  {d}   {int(orb):5d}  {direc:10s}  {pct:8.2f}   {lag:+.0f}")

    full = [r for r in rows if r[3] >= 99.0]
    C.say("")
    if full:
        near = min(full, key=lambda r: abs(r[4]))
        C.say(f"  nearest full-coverage acquisition: {near[0]}, orbit "
              f"{int(near[1])}, {near[2]}, lag {near[4]:+.0f} days")
        if abs(near[4]) <= 3:
            C.say("")
            C.say("  STOP. That is within 2 to 3 days, which is comparable to")
            C.say("  the lag both current events have. This is a scope")
            C.say("  decision, not a data result. Do nothing else with it")
            C.say("  until you have decided whether it is in or out.")
        else:
            C.say("")
            C.say(f"  {abs(near[4]):.0f} days is worse than the 2-day lag both")
            C.say("  current events have. That is the one sentence that")
            C.say("  retires the question.")
    else:
        C.say("  no acquisition covers the corridor fully in this window.")
    return 0


PLACEHOLDER = "PUT_SCENE_NAMES_HERE"


def read_scene_list(path):
    """One Sentinel-1 granule per line. Blank lines and # comments ignored."""
    names = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if line:
                names.append(line)
    return names


def check_scenes(names):
    """Refuse before the API does, and say what to do about it.

    On 9 September 2026 the launcher passed the literal string
    PUT_SCENE_NAMES_HERE straight through and HyP3 answered with a schema
    validation error that said nothing useful. This turns that into one clear
    sentence.
    """
    if not names:
        raise RuntimeError("no scene names given. Use --scenes or --scenes-file.")
    bad = [n for n in names
           if n == PLACEHOLDER or n.endswith("...")
           or not n.upper().startswith(("S1A_", "S1B_", "S1C_", "S1D_"))
           or len(n) < 40]
    if bad:
        raise RuntimeError(
            "these are not Sentinel-1 granule names:\n"
            + "".join("    %s\n" % n for n in bad)
            + "\n"
            "  A real one looks like\n"
            "    S1A_IW_GRDH_1SDV_20230816T004411_20230816T004436_049912_060179_1B2C\n"
            "\n"
            "  Get the three post-event names from the scene inventory:\n"
            "    cd \"..\\FYP_code\"\n"
            "    python 01_scene_inventory.py\n"
            "  or from https://search.asf.alaska.edu with the corridor as the AOI.\n"
            "\n"
            "  Then put one per line in\n"
            "    05 Data\\sentinel1_rtc_scenes.txt\n"
            "  and run RUN DOWNLOADS.bat option 3 again."
        )


def cmd_asf(cfg, a):
    sdk = need("hyp3_sdk", "hyp3-sdk", "A4")

    names = list(a.scenes or [])
    if getattr(a, "scenes_file", None):
        if not Path(a.scenes_file).exists():
            raise RuntimeError(
                "scene list not found:\n    %s\n\n"
                "  Create it, one Sentinel-1 granule per line." % a.scenes_file)
        names += read_scene_list(a.scenes_file)
    check_scenes(names)
    a.scenes = names
    C.say("  %d scene%s to submit" % (len(names), "" if len(names) == 1 else "s"))
    for n in names:
        C.say("    %s" % n)

    try:
        hyp3 = sdk.HyP3(prompt=True)
    except Exception as e:
        msg = str(e)
        if "authoriz" in msg.lower() or "approve_app" in msg:
            raise RuntimeError(
                "Earthdata has not authorised the HyP3 application for your "
                "account.\n"
                "  Open this once in a browser while signed in to Earthdata, "
                "approve it,\n  then run this again:\n"
                "    https://urs.earthdata.nasa.gov/approve_app"
                "?client_id=BO_n7nTIlMljdvU6kRRB3g\n\n"
                "  Original error: %s" % msg)
        raise
    out = cfg.path("DATA_DIR") / "sentinel1_rtc"
    out.mkdir(parents=True, exist_ok=True)

    jobs = sdk.Batch()
    for scene in a.scenes:
        jobs += hyp3.submit_rtc_job(
            scene, name=a.name,
            resolution=int(a.resolution),
            radiometry="gamma0",
            scale="power",
            dem_matching=False,
            include_dem=True,
            include_inc_map=True,
        )
        C.say(f"  submitted {scene}")
    C.say("")
    C.say(f"  {len(jobs)} jobs submitted at {a.resolution} m. The free tier is")
    C.say("  limited per month, so the pre-images are deliberately not here:")
    C.say("  the ablation only needs the post-event scenes.")
    C.say("  Waiting. This takes hours; you also get an email.")
    jobs = hyp3.watch(jobs)
    jobs.download_files(out)
    C.say(f"  downloaded into {out}")
    return 0


def cmd_earthdata(cfg, a):
    ea = need("earthaccess", "earthaccess", "B4")
    ea.login()
    bbox = tuple(float(v) for v in a.bbox.split(","))
    results = ea.search_data(short_name=a.short_name, bounding_box=bbox,
                             count=a.limit)
    C.say(f"  {len(results)} granules intersect {bbox}")
    if not results:
        C.say("  Nothing covers the corridor. For the HMA 8 m DEM this is the")
        C.say("  expected outcome in places: coverage is incomplete because")
        C.say("  the product is built from stereo pairs. Stop here rather")
        C.say("  than planning a hydraulic model around it.")
        return 0
    for r in results[:20]:
        C.say(f"    {r}")
    if not a.download:
        C.say("")
        C.say("  This was the coverage check. Add --download to fetch them.")
        return 0
    out = cfg.path("DATA_DIR") / a.subdir
    out.mkdir(parents=True, exist_ok=True)
    ea.download(results, str(out))
    C.say(f"  downloaded into {out}")
    return 0


def cmd_imd(cfg, a):
    imd = need("imdlib", "imdlib", "B5")
    out = cfg.path("DATA_DIR") / "rainfall" / "imd"
    out.mkdir(parents=True, exist_ok=True)
    y0, y1 = (int(v) for v in a.years.split("-"))
    C.say(f"  downloading IMD gridded {a.variable}, {y0} to {y1}")
    data = imd.get_data(a.variable, y0, y1, fn_format="yearwise",
                        file_dir=str(out))
    C.say(f"  written to {out}")
    C.say("")
    C.say("  READ THIS BEFORE USING IT. IMD daily totals run 0830 IST to")
    C.say("  0830 IST, not midnight to midnight. An IMD '19 June' total is")
    C.say("  rain that fell from the morning of 18 June. Get this wrong and")
    C.say("  the rainfall peak sits one day off the flood peak, which looks")
    C.say("  like a physical result and is not.")
    try:
        data.to_netcdf(str(out))
    except Exception:
        pass
    return 0


# --------------------------------------------------------------------------

def selftest(cfg):
    ck = C.Checks(f"{TOOL} self-test")
    ck.require_file("corridor", cfg.path("CORRIDOR"))
    ck.check(".env has GEE_PROJECT", cfg.has("GEE_PROJECT"),
             cfg.str("GEE_PROJECT", ""))
    for pkg, label in (("ee", "earthengine-api"), ("hyp3_sdk", "hyp3-sdk"),
                       ("earthaccess", "earthaccess"), ("imdlib", "imdlib")):
        try:
            __import__(pkg)
            ck.check(f"optional: {label}", True, "installed")
        except ImportError:
            ck.check(f"optional: {label}", True,
                     "not installed; that subcommand will say so")
    return ck.report()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("selftest")

    p = sub.add_parser("gee-inventory")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)

    p = sub.add_parser("gee-export")
    p.add_argument("--dates", nargs="+", required=True)
    p.add_argument("--tag", required=True,
                   help="e.g. S2_2023jul_ref")

    p = sub.add_parser("gee-s1-timing")
    p.add_argument("--date", required=True)
    p.add_argument("--window", type=int, default=10)

    p = sub.add_parser("asf")
    p.add_argument("--scenes", nargs="+", default=None,
                   help="Sentinel-1 granule names, space separated")
    p.add_argument("--scenes-file", dest="scenes_file", default=None,
                   help="a text file with one granule name per line. Blank "
                        "lines and # comments are ignored. Use this rather than "
                        "typing forty-character names into a batch file.")
    p.add_argument("--resolution", default="10")
    p.add_argument("--name", default="fyp-rtc")

    p = sub.add_parser("earthdata")
    p.add_argument("--short-name", required=True)
    p.add_argument("--bbox", default="78.216,30.042,79.033,30.329")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--subdir", default="dem/hma8m")
    p.add_argument("--download", action="store_true")

    p = sub.add_parser("imd")
    p.add_argument("--years", default="2020-2024")
    p.add_argument("--variable", default="rain")

    a = ap.parse_args(argv)
    if not a.cmd:
        ap.print_help()
        return 1

    cfg = C.Cfg.load(a.config)
    if a.cmd == "selftest":
        return 0 if selftest(cfg) else 1

    fn = {
        "gee-inventory": cmd_gee_inventory,
        "gee-export": cmd_gee_export,
        "gee-s1-timing": cmd_gee_s1_timing,
        "asf": cmd_asf,
        "earthdata": cmd_earthdata,
        "imd": cmd_imd,
    }[a.cmd]
    C.rule(a.cmd)
    return fn(cfg, a)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        C.say("")
        C.say("ERROR  " + type(exc).__name__)
        for ln in str(exc).splitlines():
            C.say("       " + ln)
        sys.exit(1)
