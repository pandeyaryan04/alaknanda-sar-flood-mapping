r"""
jalsar_runner.py
================
Jal-SAR's runner: give it a date, and it finds the
Sentinel-1 scene, downloads it onto the corridor's 10 m grid, gets the terrain
strata for that orbit, runs the models and writes the maps, the bulletin and
the web map.

Steps, in order:
  1. Search Earth Engine (COPERNICUS/S1_GRD, IW, VV and VH, the same filters
     as 01_scene_inventory.py and gee/10_reexport_corridor_v2.js) from the date
     to SEARCH_DAYS days after. Scenes of one pass (same day, orbit and
     direction) are joined, as the export script joins them. The pass that
     covers most of the corridor is used; below MIN_COVERAGE per cent it stops
     and lists what it found.
  2. Download VV, VH and angle with ee.data.computePixels, in tiles, onto the
     exact grid of the scenes in 05 Data\sentinel1 (EPSG:32644, 10 m, origin
     231660, 3357010, 7915 x 3044 cells), NaN outside the corridor. A
     download that stops part-way carries on from the last finished tile.
     The scene goes to 06 Outputs\jalsar\scenes\ (never into 05 Data).
  3. Terrain strata: orbit 63 descending reuses strata_2021_v3 and orbit 129
     ascending reuses strata_2023_v3. Any other orbit runs
     03_terrain_strata.py once with the new scene's angle band, into
     FYP_code\data\strata_orbitNNN_<pass>\, and reuses it after that.
  4. Optional (PRE_EVENT=on): a dry-season scene of the same orbit (month
     PRE_EVENT_MONTH, February by default, as the project's own baseline) and
     the change-detection Otsu map from 03b_otsu_baseline.py.
  5. Maps, agreement, trust layer and exposure (jalsar_core.make_event),
     the bulletin, and the web map made again with the new event.

Settings are read from jalsar.env next to this file (made from
jalsar.env.example on the first run, and kept out of git); --date and the
other options on the command line win over it.

The 21 October 2021 scene is already in 05 Data. 'check-download' downloads
it again and compares the two cell by cell: the acceptance test of the
downloader, before any new date is trusted.

USAGE  (Anaconda Prompt, fyp active, from 04 Code\jalsar)
-----
    python jalsar_runner.py --selftest
    python jalsar.py check-download
    python jalsar.py run --date 2023-07-19
    python jalsar.py run --date 2025-08-14 --search-days 6 --pre-event on

Needs the internet and Earth Engine on this computer ('earthengine
authenticate' once). A download is about 280 MB of radar and takes about 5
to 20 minutes (estimate); the maps take as long as 'jalsar.py maps'.
"""

__stage__ = "Jal-SAR: runner for a new date"
__reads__ = ["Earth Engine COPERNICUS/S1_GRD", "FYP_code/data/corridor.geojson",
             "05 Data/dem (strata for a new orbit)", "the trained models"]
__writes__ = ["06 Outputs/jalsar/scenes/S1_<date>_orb<nnn>.tif (+ _record.json)",
              "06 Outputs/jalsar/downloads/ (tiles while downloading)",
              "FYP_code/data/strata_orbit<nnn>_<pass>/ (new orbits only)",
              "06 Outputs/jalsar/<event>/ (maps, bulletin)", "06 Outputs/jalsar/events.json"]

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import jalsar_core as gc

mc = gc.mc
rasterio = gc.rasterio
HERE = Path(__file__).resolve().parent
ENV_FILE = HERE / "jalsar.env"                 # yours, not in git (edit freely)
ENV_EXAMPLE = HERE / "jalsar.env.example"     # in git; copied to jalsar.env on first use
EE_PROJECT = "gbpuat-fyp-alaknanda"          # the same project as 01_scene_inventory.py
GRID = {"crs": "EPSG:32644", "x0": 231660.0, "y0": 3357010.0, "cell": 10.0,
        "width": 7915, "height": 3044}       # GRID10 and RECT10 of 10_reexport_corridor_v2.js
BANDS = ("VV", "VH", "angle")
FILL = -9999.0                               # masked cells, turned into NaN on arrival
DEFAULTS = {"EVENT_DATE": "", "SEARCH_DAYS": "3", "MIN_COVERAGE": "99.5",
            "MODELS": "rf,unet,terramind", "PRE_EVENT": "off", "PRE_EVENT_MONTH": "2",
            "TILE_PX": "1024"}
REUSE_STRATA = {(63, "DESCENDING"): 2021, (129, "ASCENDING"): 2023}
SAME_ABS = 1e-3          # check-download: values agree within rounding
SAME_SHARE = 0.99        # ... for at least this share of cells with data in both
MASK_SHARE = 0.999       # ... and the data / no-data pattern agrees on this share


# =============================================================================
# Settings
# =============================================================================

def read_env(path=None):
    """KEY=VALUE lines; '#' starts a comment. Missing file: the defaults.

    With no path: jalsar.env, made from jalsar.env.example the first time,
    so your settings never clash with a 'git pull'.
    """
    out = dict(DEFAULTS)
    if path is None:
        if not ENV_FILE.exists() and ENV_EXAMPLE.exists():
            shutil.copyfile(ENV_EXAMPLE, ENV_FILE)
        path = ENV_FILE
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip().upper()] = v.strip().strip('"')
    return out


def settings(args, env=None):
    env = dict(env or read_env())
    for key, attr in (("EVENT_DATE", "date"), ("SEARCH_DAYS", "search_days"),
                      ("MIN_COVERAGE", "min_coverage"), ("MODELS", "families"),
                      ("PRE_EVENT", "pre_event")):
        v = getattr(args, attr, None)
        if v not in (None, ""):
            env[key] = str(v)
    s = {"event_date": env["EVENT_DATE"], "search_days": int(env["SEARCH_DAYS"]),
         "min_coverage": float(env["MIN_COVERAGE"]),
         "families": tuple(f.strip() for f in env["MODELS"].split(",") if f.strip()),
         "pre_event": env["PRE_EVENT"].lower() in ("on", "yes", "true", "1"),
         "pre_event_month": int(env["PRE_EVENT_MONTH"]), "tile_px": int(env["TILE_PX"])}
    if s["event_date"]:
        try:
            date.fromisoformat(s["event_date"])
        except ValueError:
            mc.fail(f"EVENT_DATE '{s['event_date']}' is not a date like 2023-08-16.")
    bad = [f for f in s["families"] if f not in gc.FAMILIES]
    if bad:
        mc.fail(f"MODELS has unknown names: {', '.join(bad)}. Use {', '.join(gc.FAMILIES)}.")
    if not 0 <= s["min_coverage"] <= 100 or s["search_days"] < 0 or not 1 <= s["pre_event_month"] <= 12:
        mc.fail("Check MIN_COVERAGE (0 to 100), SEARCH_DAYS (0 or more) and PRE_EVENT_MONTH (1 to 12).")
    if not 128 <= s["tile_px"] <= 2048:
        mc.fail("TILE_PX must be between 128 and 2048.")
    return s


def scene_name(d, orbit):
    return f"S1_{d.replace('-', '')}_orb{int(orbit):03d}"


def event_id(d, orbit):
    return f"{d}_orbit{int(orbit):03d}"


# =============================================================================
# Earth Engine
# =============================================================================

def ee_init():
    try:
        import ee
    except ImportError:
        mc.fail("The earthengine-api package is not in this environment. Run: conda activate fyp")
    try:
        ee.Initialize(project=EE_PROJECT)
    except Exception as exc:                      # noqa: BLE001
        mc.fail(f"Could not connect to Earth Engine (project {EE_PROJECT}): {exc}\n"
                "If it says you are not authenticated, run:  earthengine authenticate\n"
                "then run this again.")
    return ee


def corridor_geometry(ee, gp):
    g = json.loads(gp.corridor.read_text(encoding="utf-8"))
    geom = g["features"][0]["geometry"] if "features" in g else g
    if geom["type"] != "Polygon":
        mc.fail(f"The corridor geometry is a {geom['type']}, expected a Polygon:\n  {gp.corridor}")
    return ee.Geometry.Polygon(geom["coordinates"], None, False)   # planar, as in the JS


def s1_collection(ee, corridor, start, end):
    return (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(corridor).filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))


def pass_collection(ee, corridor, day, orbit, orbit_pass):
    """One pass, exactly as s1day() in 10_reexport_corridor_v2.js."""
    d0 = ee.Date(day)
    return (s1_collection(ee, corridor, d0, d0.advance(1, "day"))
            .filter(ee.Filter.eq("relativeOrbitNumber_start", int(orbit)))
            .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass)))


def group_passes(props):
    """Scene properties -> one row per pass (UTC day, relative orbit, direction)."""
    groups = {}
    for p in props:
        t = datetime.fromtimestamp(p["system:time_start"] / 1000, tz=timezone.utc)
        key = (t.strftime("%Y-%m-%d"), int(p["relativeOrbitNumber_start"]), p["orbitProperties_pass"])
        g = groups.setdefault(key, {"date": key[0], "orbit": key[1], "pass": key[2], "ids": [],
                                    "time_utc": t.strftime("%Y-%m-%d %H:%M")})
        g["ids"].append(p["system:index"])
        g["time_utc"] = min(g["time_utc"], t.strftime("%Y-%m-%d %H:%M"))
    return sorted(groups.values(), key=lambda g: (g["date"], g["orbit"]))


def pick_pass(rows, min_cov):
    """The pass with the highest coverage (earliest on a tie), or None below min_cov."""
    ok = [r for r in rows if r.get("coverage_pct") is not None and r["coverage_pct"] >= min_cov]
    if not ok:
        return None
    return sorted(ok, key=lambda r: (-r["coverage_pct"], r["date"]))[0]


def search(ee, corridor, start, end, orbit=None, orbit_pass=None):
    """Every pass in [start, end) with its corridor coverage (per cent)."""
    col = s1_collection(ee, corridor, start, end)
    if orbit is not None:
        col = (col.filter(ee.Filter.eq("relativeOrbitNumber_start", int(orbit)))
               .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass)))
    names = ["system:index", "system:time_start", "relativeOrbitNumber_start",
             "orbitProperties_pass"]
    fc = ee.FeatureCollection(col.map(lambda im: ee.Feature(None, {
        n.replace(":", "_"): im.get(n) for n in names})))
    props = [{n: f["properties"].get(n.replace(":", "_")) for n in names}
             for f in fc.getInfo()["features"]]
    rows = group_passes(props)
    for r in rows:
        day = pass_collection(ee, corridor, r["date"], r["orbit"], r["pass"])
        cov = (day.select("VV").mosaic().mask().unmask(0).clip(corridor)
               .reduceRegion(reducer=ee.Reducer.mean(), geometry=corridor, scale=100,
                             maxPixels=1e9).get("VV").getInfo())
        r["coverage_pct"] = round(float(cov or 0) * 100, 2)
    return rows


def pass_image(ee, corridor, row):
    """VV, VH, angle of one pass: mosaic, clip to the corridor, float; masked cells FILL.

    sameFootprint=False: without it, cells outside the clipped footprint stay
    masked and computePixels returns them as 0 (found by the 8 October check).
    """
    return (pass_collection(ee, corridor, row["date"], row["orbit"], row["pass"]).mosaic()
            .select(list(BANDS)).clip(corridor).toFloat().unmask(FILL, False))


def ee_fetcher(ee, image, grid=GRID, tries=4):
    """fetch(col0, row0, w, h) -> float32 (3, h, w) from computePixels."""
    def fetch(c0, r0, w, h):
        req = {"expression": image, "fileFormat": "NUMPY_NDARRAY", "bandIds": list(BANDS),
               "grid": {"dimensions": {"width": int(w), "height": int(h)},
                        "affineTransform": {"scaleX": grid["cell"], "shearX": 0,
                                            "translateX": grid["x0"] + c0 * grid["cell"],
                                            "shearY": 0, "scaleY": -grid["cell"],
                                            "translateY": grid["y0"] - r0 * grid["cell"]},
                        "crsCode": grid["crs"]}}
        for attempt in range(tries):
            try:
                arr = ee.data.computePixels(req)
                return np.stack([np.asarray(arr[b], dtype="float32") for b in BANDS])
            except Exception as exc:              # noqa: BLE001
                if attempt == tries - 1:
                    raise
                wait = 10 * (attempt + 1) if ("429" in str(exc) or "quota" in str(exc).lower()) else 3
                print(f"    tile ({r0}, {c0}): {str(exc)[:120]}; trying again in {wait} s", flush=True)
                time.sleep(wait)
    return fetch


# =============================================================================
# Download in tiles, with resume
# =============================================================================

def tiles(width, height, size):
    return [(c, r, min(size, width - c), min(size, height - r))
            for r in range(0, height, size) for c in range(0, width, size)]


def download(fetch, tile_dir, out_tif, grid=GRID, tile_px=1024, record=None):
    """Fetch every tile (keeping finished ones), join them, write the scene GeoTIFF."""
    from rasterio.transform import from_origin
    tile_dir = Path(tile_dir)
    tile_dir.mkdir(parents=True, exist_ok=True)
    tl = tiles(grid["width"], grid["height"], tile_px)
    done = 0
    t0 = time.time()
    for i, (c0, r0, w, h) in enumerate(tl, 1):
        p = tile_dir / f"tile_r{r0:05d}_c{c0:05d}.npy"
        if p.exists():
            done += 1
            continue
        a = fetch(c0, r0, w, h)
        if a.shape != (len(BANDS), h, w):
            mc.fail(f"Tile ({r0}, {c0}) came back with shape {a.shape}, expected {(len(BANDS), h, w)}.")
        tmp = p.with_name(p.stem + "_part.npy")
        np.save(tmp, a.astype("float32"))
        os.replace(tmp, p)
        el = time.time() - t0
        left = el / (i - done) * (len(tl) - i) if i > done else 0
        print(f"    tile {i} of {len(tl)} done; about {left / 60:.0f} min left", flush=True)
    if done:
        print(f"    {done} tile(s) were already downloaded and are reused")
    full = np.full((len(BANDS), grid["height"], grid["width"]), np.nan, "float32")
    for c0, r0, w, h in tl:
        full[:, r0:r0 + h, c0:c0 + w] = np.load(tile_dir / f"tile_r{r0:05d}_c{c0:05d}.npy")
    full[full == np.float32(FILL)] = np.nan
    # Second guard: a cell where VV, VH and angle are all exactly 0 is a masked
    # cell returned as 0 (the radar angle is never 0), so it has no data.
    full[:, (full == 0).all(axis=0)] = np.nan
    full[~np.isfinite(full)] = np.nan
    out_tif = Path(out_tif)
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_tif.with_name(out_tif.stem + "_part.tif")
    prof = {"driver": "GTiff", "dtype": "float32", "count": len(BANDS), "crs": grid["crs"],
            "transform": from_origin(grid["x0"], grid["y0"], grid["cell"], grid["cell"]),
            "width": grid["width"], "height": grid["height"], "nodata": float("nan"),
            "compress": "lzw", "tiled": True, "blockxsize": 256, "blockysize": 256,
            "BIGTIFF": "IF_SAFER"}
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(full)
        for i, b in enumerate(BANDS, 1):
            dst.set_band_description(i, b)
    os.replace(tmp, out_tif)
    valid = float(np.isfinite(full[0]).mean())
    if record is not None:
        record.update(valid_share_of_rectangle=round(valid, 4), tiles=len(tl), tile_px=tile_px)
        mc.save_json(out_tif.with_name(out_tif.stem + "_record.json"), record)
    shutil.rmtree(tile_dir, ignore_errors=True)
    return valid


def compare_scenes(a_path, b_path):
    """Cell-by-cell comparison of two scenes: grid, data pattern, values per band."""
    with rasterio.open(a_path) as a, rasterio.open(b_path) as b:
        grid_same = (a.width, a.height, a.crs, a.count) == (b.width, b.height, b.crs, b.count) and \
            all(abs(x - y) < 1e-6 for x, y in zip(a.transform, b.transform))
        out = {"grid_same": bool(grid_same), "bands": {}}
        if not grid_same:
            out["same"] = False
            return out
        for i in range(1, min(a.count, 3) + 1):
            x = mc._to_nan(a.read(i), a.nodata)
            y = mc._to_nan(b.read(i), b.nodata)
            fx, fy = np.isfinite(x), np.isfinite(y)
            both = fx & fy
            d = np.abs(x[both] - y[both])
            name = (a.descriptions[i - 1] or BANDS[i - 1])
            out["bands"][name] = {
                "mask_agree_share": round(float((fx == fy).mean()), 6),
                "data_cells_a": int(fx.sum()), "data_cells_b": int(fy.sum()),
                "within_rounding_share": round(float((d <= SAME_ABS).mean()), 6) if d.size else None,
                "median_abs_diff": float(np.median(d)) if d.size else None,
                "p99_abs_diff": float(np.percentile(d, 99)) if d.size else None}
        out["same"] = all(v["mask_agree_share"] >= MASK_SHARE and v["within_rounding_share"] is not None
                          and v["within_rounding_share"] >= SAME_SHARE for v in out["bands"].values())
    return out


# =============================================================================
# Strata, baseline and the whole run
# =============================================================================

def strata_for(gp, orbit, orbit_pass, scene, build=True):
    """(strata folder, note): reuse 2021 or 2023, an earlier run's, or build new."""
    y = REUSE_STRATA.get((int(orbit), orbit_pass))
    if y:
        return gp.strata(y), (f"reused strata_{y}_v3 (orbit {orbit} {orbit_pass.lower()}, the same "
                              f"track as the {y} event)")
    d = gp.data / f"strata_orbit{int(orbit):03d}_{orbit_pass.lower()}"
    if (d / "strata_layover_shadow.tif").exists():
        return d, f"reused {d.name}, built by Jal-SAR for this orbit on an earlier run"
    if not build:
        return d, "to be built"
    dem = gp.d05 / "dem" / "DEM_COP_GLO30_30m.tif"
    water = gp.d05 / "dem" / "WATER_JRC_occurrence_30m.tif"
    for p in (dem, water, gp.corridor):
        if not p.exists():
            mc.fail(f"The strata for orbit {orbit} need this file, which is missing:\n  {p}")
    print(f"  building the terrain strata for orbit {orbit} {orbit_pass.lower()} "
          "(03_terrain_strata.py, a few minutes)", flush=True)
    cmd = [sys.executable, str(gc.FYP / "03_terrain_strata.py"), "--dem", str(dem),
           "--pass", orbit_pass, "--water", str(water), "--corridor", str(gp.corridor),
           "--angle", str(scene), "--out", str(d)]
    r = subprocess.run(cmd, cwd=str(gc.FYP))
    if r.returncode != 0 or not (d / "strata_layover_shadow.tif").exists():
        mc.fail(f"03_terrain_strata.py did not finish for orbit {orbit}. Scroll up for its message.")
    return d, (f"built by Jal-SAR on {time.strftime('%d %B %Y')} with 03_terrain_strata.py from "
               f"this scene's angle band ({d.name})")


def pre_event_window(event_date, month):
    """The dry-season month before the event: this year's if it is earlier, else last year's."""
    d = date.fromisoformat(event_date)
    y = d.year if month < d.month else d.year - 1
    start = date(y, month, 1)
    end = date(y + (month == 12), month % 12 + 1, 1)
    return start.isoformat(), end.isoformat()


def get_scene(ee, corridor, gp, row, label, tile_px):
    name = scene_name(row["date"], row["orbit"])
    out = gp.gj / "scenes" / f"{name}.tif"
    if out.exists():
        print(f"  {label}: {out.name} is already downloaded, reused")
        return out
    print(f"  {label}: downloading {name} ({row['coverage_pct']:.1f} % of the corridor), "
          f"{len(tiles(GRID['width'], GRID['height'], tile_px))} tiles", flush=True)
    rec = {"scene_ids": row["ids"], "time_utc": row["time_utc"], "orbit": row["orbit"],
           "pass": row["pass"], "coverage_pct": row["coverage_pct"],
           "collection": "COPERNICUS/S1_GRD", "grid": GRID, "bands": list(BANDS),
           "processing": "mosaic of the pass, select VV VH angle, clip to corridor, toFloat "
                         "(as gee/10_reexport_corridor_v2.js); ee.data.computePixels",
           "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    valid = download(ee_fetcher(ee, pass_image(ee, corridor, row)), gp.gj / "downloads" / name,
                     out, tile_px=tile_px, record=rec)
    print(f"    wrote {out} ({valid * 100:.1f} % of the rectangle has data)")
    return out


def otsu_change(gp, event, pre_scene):
    out = gp.event_dir(event.id) / "otsu_change"
    if (out / f"flood_{event.id}_otsu.tif").exists():
        return out / f"flood_{event.id}_otsu.tif"
    cmd = [sys.executable, str(gc.FYP / "03b_otsu_baseline.py"), "--post", event.scene,
           "--pre", str(pre_scene), "--mode", "change", "--water", str(gp.perm_water()),
           "--mask", str(Path(event.strata) / "strata_layover_shadow.tif"), "--tiles", "10",
           "--out", str(out), "--event", event.id]
    r = subprocess.run(cmd, cwd=str(gc.FYP))
    if r.returncode != 0:
        print("  PROBLEM: the change-detection baseline failed (scroll up); the model maps go ahead.")
        return None
    return out / f"flood_{event.id}_otsu.tif"


def print_passes(rows):
    for r in rows:
        print(f"    {r['date']}  orbit {r['orbit']:3d} {r['pass']:10s}  {len(r['ids'])} scene(s)  "
              f"coverage {r['coverage_pct']:6.2f} %")


def run(gp, s, force=False, cpu_ok=False, tile=512, overlap=64, pdf=True):
    import jalsar_bulletin as gb
    import jalsar_webmap as gw
    if not s["event_date"]:
        mc.fail("No date given. Use --date 2025-08-14, or set EVENT_DATE in jalsar.env.")
    t0 = time.time()
    ee = ee_init()
    corridor = corridor_geometry(ee, gp)
    d0 = date.fromisoformat(s["event_date"])
    end = (d0 + timedelta(days=s["search_days"] + 1)).isoformat()
    print(f"jalsar.py run: {s['event_date']} to {end} (exclusive), passes over the corridor:")
    rows = search(ee, corridor, s["event_date"], end)
    if not rows:
        mc.fail("No Sentinel-1 scene in that window. Widen it with --search-days.")
    print_passes(rows)
    row = pick_pass(rows, s["min_coverage"])
    if row is None:
        mc.fail(f"No pass covers at least {s['min_coverage']} % of the corridor (MIN_COVERAGE). "
                "Widen the window with --search-days, or lower MIN_COVERAGE if a partial map is "
                "acceptable.")
    eid = event_id(row["date"], row["orbit"])
    if eid in {e.id for e in gc.builtin_events(gp)}:
        mc.fail(f"{eid} is one of the events already in 05 Data; its maps come from "
                "'jalsar.py maps'. To test the downloader on it, use 'jalsar.py check-download'.")
    print(f"  using {row['date']} orbit {row['orbit']} {row['pass'].lower()} "
          f"({row['coverage_pct']:.2f} %): event {eid}")
    scene = get_scene(ee, corridor, gp, row, "scene", s["tile_px"])
    strata, note = strata_for(gp, row["orbit"], row["pass"], scene)
    ev = gc.Event(eid, row["date"], str(scene), str(strata), row["orbit"], row["pass"], 0,
                  f"{gc_long(row['date'])} (radar only)", ";".join(row["ids"]), row["time_utc"], note)
    gc.register_event(gp, ev)
    run_rec = {"event": eid, "settings": s, "passes_found": rows, "chosen": row,
               "scene": str(scene), "strata": str(strata), "strata_note": note,
               "started_utc": datetime.fromtimestamp(t0, timezone.utc).isoformat(timespec="seconds")}
    if s["pre_event"]:
        a, b = pre_event_window(row["date"], s["pre_event_month"])
        print(f"  pre-event scene: orbit {row['orbit']} {row['pass'].lower()}, {a} to {b}")
        pre = search(ee, corridor, a, b, row["orbit"], row["pass"])
        print_passes(pre)
        prow = pick_pass(pre, s["min_coverage"])
        if prow is None:
            print("  PROBLEM: no dry-season pass of this orbit covers the corridor; no baseline.")
        else:
            pscene = get_scene(ee, corridor, gp, prow, "pre-event scene", s["tile_px"])
            run_rec["pre_event"] = {"chosen": prow, "scene": str(pscene),
                                    "baseline": str(otsu_change(gp, ev, pscene))}
    summ = gc.make_event(gp, ev, s["families"], force=force, cpu_ok=cpu_ok, tile=tile,
                         overlap=overlap)
    b = gb.make_bulletin(gp, ev, force=True, pdf=pdf)
    w = gw.build_webmap(gp, force=True)
    run_rec.update(bulletin=b, webmap=w.get("path"), headline=summ["headline"]["family"],
                   flood_km2=summ["areas"]["total_km2"], minutes=round((time.time() - t0) / 60, 1),
                   finished_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    mc.save_json(gp.event_dir(eid) / "run_record.json", run_rec)
    print(f"  run record: {gp.event_dir(eid) / 'run_record.json'}")
    return run_rec


def gc_long(d):
    x = date.fromisoformat(d)
    return f"{x.day} {x.strftime('%B %Y')}"


def check_download(gp, tile_px=1024, force=False):
    """Download 21 October 2021 again and compare it with the scene in 05 Data."""
    ref = gp.scene(2021, "grd")
    if not ref.exists():
        mc.fail(f"The 2021 scene is missing, so there is nothing to compare with:\n  {ref}")
    out = gp.gj / "checks" / "S1_20211021_orb063_redownload.tif"
    if out.exists() and force:
        out.unlink()
    print("jalsar.py check-download: 21 October 2021, orbit 63 descending, against 05 Data")
    if not out.exists():
        ee = ee_init()
        corridor = corridor_geometry(ee, gp)
        rows = search(ee, corridor, "2021-10-21", "2021-10-22", 63, "DESCENDING")
        print_passes(rows)
        if not rows:
            mc.fail("Earth Engine found no orbit 63 pass on 21 October 2021.")
        row = rows[0]
        rec = {"scene_ids": row["ids"], "time_utc": row["time_utc"], "coverage_pct": row["coverage_pct"]}
        download(ee_fetcher(ee, pass_image(ee, corridor, row)), gp.gj / "downloads" / "check_2021",
                 out, tile_px=tile_px, record=rec)
    res = compare_scenes(out, ref)
    lines = ["# Jal-SAR download check, 21 October 2021", "",
             f"Re-downloaded: `{out}`", f"In use: `{ref}`", "",
             f"Same grid: {'yes' if res['grid_same'] else 'NO'}", "",
             "| Band | Data pattern agrees | Cells within 0.001 | Median difference | 99th percentile |",
             "|---|---|---|---|---|"]
    for b, v in res["bands"].items():
        pct = (lambda x: "n/a" if x is None else f"{x * 100:.3f} %")
        num = (lambda x: "n/a" if x is None else f"{x:.6f}")
        lines.append(f"| {b} | {pct(v['mask_agree_share'])} | {pct(v['within_rounding_share'])} | "
                     f"{num(v['median_abs_diff'])} | {num(v['p99_abs_diff'])} |")
    verdict = ("SAME: the downloader reproduces the scene in use." if res["same"] else
               "DIFFERENT: read the table before trusting new dates. A small share of differing "
               "cells at the corridor edge is a clipping difference; differences everywhere mean "
               "Earth Engine's copy of the scene has changed since the first export.")
    lines += ["", f"Rule: data pattern agrees on at least {MASK_SHARE * 100:.1f} % of cells and "
              f"values agree within {SAME_ABS} on at least {SAME_SHARE * 100:.0f} % of cells with "
              "data in both.", "", f"**{verdict}**"]
    rep = out.with_name("CHECK REPORT.md")
    rep.write_text("\n".join(lines) + "\n", encoding="utf-8")
    mc.save_json(out.with_name("check_result.json"), res)
    for ln in lines[6:]:
        print("  " + ln)
    print(f"  wrote {rep}")
    return res


# =============================================================================
# Self-test (no Earth Engine: a stand-in fetch serves a synthetic scene)
# =============================================================================

def selftest():
    print("jalsar_runner.py self-test")
    import argparse
    from rasterio.transform import from_origin
    with tempfile.TemporaryDirectory(prefix="jalsar runner ", ignore_cleanup_errors=True) as td:
        td = Path(td)
        grid = {"crs": "EPSG:32644", "x0": 231660.0, "y0": 3357010.0, "cell": 10.0,
                "width": 130, "height": 70}
        rng = np.random.default_rng(3)
        truth = rng.normal(-15, 3, (3, 70, 130)).astype("float32")
        truth[:, :10, :20] = np.nan                    # outside the corridor
        truth[:, 40:50, 100:] = np.nan                 # served as 0, not FILL (tile row 32)
        served = {"n": 0, "fail_at": 3}

        def fetch(c0, r0, w, h):
            served["n"] += 1
            if served["n"] == served["fail_at"]:
                raise RuntimeError("connection reset (stand-in)")
            a = truth[:, r0:r0 + h, c0:c0 + w].copy()
            # masked cells: some as FILL, some as 0 (what Earth Engine did on 8 October)
            a[:, ~np.isfinite(a[0])] = np.where(r0 % 64 == 0, FILL, 0.0)
            return a
        out = td / "scenes" / "S1_20250814_orb063.tif"
        try:
            download(fetch, td / "dl", out, grid=grid, tile_px=32)
            first_stopped = False
        except RuntimeError:
            first_stopped = True
        kept = len(list((td / "dl").glob("tile_*.npy")))
        n_before = served["n"]
        download(fetch, td / "dl", out, grid=grid, tile_px=32, record={"test": True})
        n_tiles = len(tiles(130, 70, 32))
        resumed = first_stopped and kept == 2 and served["n"] - n_before == n_tiles - kept
        with rasterio.open(out) as r:
            back = r.read()
            desc = r.descriptions
            tf = r.transform
        same = np.allclose(np.nan_to_num(back, nan=1e9), np.nan_to_num(truth, nan=1e9))
        grid_ok = desc == BANDS and (tf.c, tf.f, tf.a, tf.e) == (231660.0, 3357010.0, 10.0, -10.0)
        cleaned = not (td / "dl").exists() and out.with_name(out.stem + "_record.json").exists()
        # compare_scenes: identical, then a changed copy
        prof = {"driver": "GTiff", "dtype": "float32", "count": 3, "crs": "EPSG:32644",
                "transform": from_origin(231660, 3357010, 10, 10), "width": 130, "height": 70,
                "nodata": float("nan")}
        other = td / "other.tif"
        with rasterio.open(other, "w", **prof) as d:
            d.write(truth + 0.5)
        cmp_ok = compare_scenes(out, out)["same"] and not compare_scenes(out, other)["same"]
        # pass grouping and choice
        ms = lambda s: int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp() * 1000)  # noqa: E731
        props = [{"system:index": "A1", "system:time_start": ms("2025-08-14T00:40:00"),
                  "relativeOrbitNumber_start": 63, "orbitProperties_pass": "DESCENDING"},
                 {"system:index": "A2", "system:time_start": ms("2025-08-14T00:40:25"),
                  "relativeOrbitNumber_start": 63, "orbitProperties_pass": "DESCENDING"},
                 {"system:index": "B1", "system:time_start": ms("2025-08-15T12:50:00"),
                  "relativeOrbitNumber_start": 129, "orbitProperties_pass": "ASCENDING"}]
        rows = group_passes(props)
        rows[0]["coverage_pct"], rows[1]["coverage_pct"] = 100.0, 99.9
        group_ok = len(rows) == 2 and rows[0]["ids"] == ["A1", "A2"] and rows[0]["time_utc"] == "2025-08-14 00:40"
        pick_ok = pick_pass(rows, 99.5)["orbit"] == 63 and pick_pass(rows, 100.1) is None
        # settings: file, then the command line wins
        envf = td / "jalsar.env"
        envf.write_text("# test\nEVENT_DATE=2025-08-14\nSEARCH_DAYS = 5  # wider\nPRE_EVENT=on\n",
                        encoding="utf-8")
        s = settings(argparse.Namespace(date=None, search_days=2, min_coverage=None, families=None,
                                        pre_event=None), read_env(envf))
        set_ok = (s["event_date"], s["search_days"], s["pre_event"], s["families"]) == \
            ("2025-08-14", 2, True, ("rf", "unet", "terramind"))
        pre_ok = pre_event_window("2025-08-14", 2) == ("2025-02-01", "2025-03-01") and \
            pre_event_window("2025-01-20", 2) == ("2024-02-01", "2024-03-01") and \
            pre_event_window("2025-12-20", 12) == ("2024-12-01", "2025-01-01")
        gp = gc.GPaths(td / "Final Year Project")
        s63, n63 = strata_for(gp, 63, "DESCENDING", None)
        s64, n64 = strata_for(gp, 64, "ASCENDING", None, build=False)
        strata_ok = s63 == gp.strata(2021) and "reused" in n63 and s64.name == "strata_orbit064_ascending"
        ids_ok = event_id("2025-08-14", 63) == "2025-08-14_orbit063" and \
            scene_name("2025-08-14", 63) == "S1_20250814_orb063"
    checks = [("download stops part-way, then carries on from the finished tiles", resumed),
              ("scene back exactly, masked cells as NaN", same),
              ("bands VV, VH, angle on the corridor grid", grid_ok),
              ("tiles removed, download record written", cleaned),
              ("scene comparison: same and different", cmp_ok),
              ("scenes of one pass joined", group_ok), ("best-covered pass chosen; below the limit none", pick_ok),
              ("settings file read; command line wins", set_ok),
              ("dry-season window before the event", pre_ok),
              ("strata reused for orbit 63; a new orbit gets its own folder", strata_ok),
              ("event and scene names", ids_ok)]
    for name, c in checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {name}")
    good = all(c for _, c in checks)
    print("SELFTEST PASS" if good else "SELFTEST FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    mc.setup_console()
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print(__doc__)
