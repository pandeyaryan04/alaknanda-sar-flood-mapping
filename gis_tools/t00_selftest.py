"""
t00_selftest.py - run every tool's self-test, plus the checks that span them.

This is what you double-click first, before doing anything real. It writes
nothing and changes nothing.
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from pathlib import Path

import common as C

TOOL = "t00_selftest"

TOOLS = [
    ("t01_centreline", "corridor centreline"),
    ("t02_mosaic", "Sentinel-2 mosaics"),
    ("t03_watermask_audit", "permanent-water layer audit"),
    ("t04_reservoir", "reservoir stability"),
    ("t05_qgis_project", "QGIS project and GeoPackage"),
    ("t06_draft_polygons", "draft polygons"),
    ("t07_download", "downloads"),
]


def cross_checks(cfg):
    """The checks no single tool owns."""
    ck = C.Checks("package-level checks")

    ck.check("python >= 3.9", sys.version_info >= (3, 9), sys.version.split()[0])

    missing = []
    for mod in ("numpy", "scipy", "skimage", "rasterio", "geopandas",
                "shapely", "fiona", "pyproj", "matplotlib", "pandas"):
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(mod)
    ck.check("required packages", not missing,
             "all present" if not missing else "missing: " + ", ".join(missing))

    # openpyxl is not needed here, but its absence is what broke SCREEN GAUGES
    try:
        importlib.import_module("openpyxl")
        ck.check("openpyxl (pandas needs it for .xlsx)", True, "present")
    except Exception:
        ck.check("openpyxl (pandas needs it for .xlsx)", False,
                 "pip install openpyxl - not needed by gis_tools, "
                 "but screen_gauges.py cannot read the gauge files without it")

    # every key the code reads must exist in the .env
    used = set()
    for name, _ in TOOLS:
        p = Path(__file__).resolve().parent / f"{name}.py"
        if not p.exists():
            continue
        import re
        used |= set(re.findall(
            r'cfg\.(?:str|f|i|b|path|csv|ints|lonlat|sites|has|raw)\("([A-Z0-9_]+)"',
            p.read_text(encoding="utf-8")))
    absent = sorted(k for k in used if k not in cfg.as_dict())
    ck.check("every key the code reads exists in .env", not absent,
             "all present" if not absent else "missing: " + ", ".join(absent))

    # UTM zone against the corridor, the one that catches a new project
    try:
        import geopandas as gpd
        g = gpd.read_file(cfg.path("CORRIDOR"))
        if g.crs is None:
            g = g.set_crs(cfg.str("CRS_STORE"))
        c = g.to_crs("EPSG:4326").union_all().centroid \
            if hasattr(g, "union_all") else g.to_crs("EPSG:4326").unary_union.centroid
        zone = C.utm_zone_of(c.x)
        declared = cfg.str("CRS_METRIC").upper().replace("EPSG:", "")
        ck.check("CRS_METRIC is the UTM zone the corridor sits in",
                 declared == f"326{zone:02d}",
                 f"corridor centroid {c.x:.3f} E is zone {zone}; "
                 f"CRS_METRIC is {declared}")
    except Exception as e:
        ck.check("UTM zone against the corridor", False, f"{type(e).__name__}: {e}")

    # disk
    try:
        out = cfg.path("OUT_DIR")
        out.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(out).free / 1e9
        ck.check("OUT_DIR writable, 5 GB free", free > 5, f"{free:.1f} GB free")
    except Exception as e:
        ck.check("OUT_DIR writable", False, f"{type(e).__name__}: {e}")

    return ck.report()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)

    C.say("")
    C.say("gis_tools self-test, package version " + C.__version__)
    try:
        cfg = C.Cfg.load(a.config)
        C.say(f"config: {cfg.source}")
    except Exception as e:
        C.say("")
        C.say("ERROR  " + str(e))
        return 1

    results = {"package": cross_checks(cfg)}

    for name, what in TOOLS:
        try:
            mod = importlib.import_module(name)
        except Exception as e:
            C.rule(f"{name}  ({what})")
            C.say(f"  FAIL  the script itself will not import: "
                  f"{type(e).__name__}: {e}")
            results[name] = False
            continue
        try:
            results[name] = bool(mod.selftest(cfg))
        except Exception as e:
            C.say(f"  FAIL  self-test raised {type(e).__name__}: {e}")
            results[name] = False

    C.rule("summary")
    width = max(len(k) for k in results)
    for k, ok in results.items():
        C.say(f"  {'PASS' if ok else 'FAIL'}  {k.ljust(width)}")
    n_bad = sum(1 for v in results.values() if not v)
    C.say("")
    if n_bad == 0:
        C.say("  Everything passed. Run RUN CENTRELINE.bat next.")
        return 0
    C.say(f"  {n_bad} of {len(results)} failed.")
    C.say("")
    C.say("  A FAIL here is not a crash; it is the package telling you what it")
    C.say("  cannot find. Most of them are a path in .env or a file that has")
    C.say("  not been produced yet. Read the detail column, fix it, run again.")
    C.say("  If a FAIL makes no sense, keep a copy of this whole screen.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
