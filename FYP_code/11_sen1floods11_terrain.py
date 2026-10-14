r"""
11_sen1floods11_terrain.py
==========================
Terrain inputs (HAND and slope) for every Sen1Floods11 chip, made with exactly
the same code that made the corridor's strata, so a model can be trained on
Sen1Floods11 with terrain and then run on the corridor with terrain.

WHY THIS EXISTS
---------------
Sen1Floods11 chips carry radar and labels but no elevation. In a gorge the
height above the river (HAND) and the slope are the strongest clues to where
water can and cannot be. The corridor already has both, from
03_terrain_strata.py. This script gives every training chip the same two layers.

WHAT IT DOES, PER CHIP
----------------------
1. Reads the chip's position and pixel grid from its S1 file.
2. Asks Earth Engine for Copernicus GLO-30 (band DEM, mosaicked) and JRC Global
   Surface Water 1.4 occurrence, on a 30 m UTM grid covering the chip plus a
   2.5 km margin. These are the same two products, with the same default
   resampling, as gee/10_reexport_corridor_v2.js used for the corridor.
3. Computes slope (Horn), D8 flow, flow accumulation, the drainage network
   (500 cells, or JRC occurrence >= 50 %) and HAND with the functions imported
   from 03_terrain_strata.py. Nothing is reimplemented.
4. Resamples HAND and slope onto the chip's own 10 m grid (bilinear, as
   04_stratified_metrics.py does for the corridor) and writes one file:
       <chip>_Terrain.tif   band 1 hand_m, band 2 slope_deg, float32, nodata NaN
The 30 m download is kept as <chip>_dem_jrc_30m.tif, so a re-run never
downloads twice. A finished chip is skipped, so the script can be stopped and
started again at any time.

KNOWN DIFFERENCE FROM THE CORRIDOR
----------------------------------
The corridor's HAND was computed over the whole corridor; a chip's over a
box of the chip plus 2.5 km each side, about 10 km. Drainage that would
leave the box is cut off at its edge. --selftest proves the code is the same
as 03's, but cannot measure this edge effect, because the corridor DEM itself
stops at the corridor edge.

HOW TO RUN THIS  (copy one line, paste into Anaconda Prompt, press Enter)
================================================================================
    conda activate fyp
    cd /d "<project folder>\04 Code\FYP_code"

    1. Offline check, no internet, about 1 minute:
       python 11_sen1floods11_terrain.py --selftest
    2. Three chips from three events, about 1 minute:
       python 11_sen1floods11_terrain.py --set hand --test
    3. All 446 hand-labelled chips:
       python 11_sen1floods11_terrain.py --set hand
    4. All weak-label chips (about 4,400, a few hours, can run overnight):
       python 11_sen1floods11_terrain.py --set weak

The exact steps are in RUN ORDER.md, step 4.10.
================================================================================
"""

__stage__ = "Model inputs: terrain for Sen1Floods11"
__reads__ = ["Sen1Floods11 S1 chips", "Earth Engine: COPERNICUS/DEM/GLO30, JRC/GSW1_4"]
__writes__ = ["data/sen1floods11_terrain/<set>/<chip>_Terrain.tif", "terrain_log.csv"]

import argparse
import concurrent.futures as cf
import csv
import importlib.util
import os
import sys
import time

import numpy as np

try:
    import rasterio
    from rasterio.warp import reproject, transform_bounds, Resampling
except ImportError as e:
    sys.exit(f"\nMissing package: {e}.\nActivate the environment first:  conda activate fyp\n")

HERE = os.path.dirname(os.path.abspath(__file__))
EE_PROJECT = "gbpuat-fyp-alaknanda"          # the same project as 01_scene_inventory.py
S1F = os.path.join(HERE, "data", "sen1floods11", "v1.1", "data", "flood_events")
SETS = {
    "hand": (os.path.join(S1F, "HandLabeled", "S1Hand"), "_S1Hand.tif"),
    "weak": (os.path.join(S1F, "WeaklyLabeled", "S1Weak"), "_S1Weak.tif"),
}
OUT_ROOT = os.path.join(HERE, "data", "sen1floods11_terrain")
PAD_M = 2500.0          # margin around each chip for the drainage to make sense
RES = 30.0              # metres, the corridor strata resolution
DEM_NODATA = -32768.0
WATER_THRESHOLD = 50.0  # JRC occurrence, per cent: the same default as 03

# --- the strata script, imported rather than copied -----------------------------
_spec = importlib.util.spec_from_file_location(
    "terrain_strata", os.path.join(HERE, "03_terrain_strata.py"))
ts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ts)


# =============================================================================
# Grid helpers
# =============================================================================

def utm_epsg(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def chip_info(path):
    """Grid of one chip, or (None, reason) if the chip cannot be used."""
    try:
        with rasterio.open(path) as src:
            a = src.read(1, masked=True)
            info = {"crs": src.crs, "transform": src.transform,
                    "width": src.width, "height": src.height,
                    "bounds": src.bounds}
    except Exception as exc:                      # noqa: BLE001
        return None, f"unreadable: {exc}"
    finite = np.isfinite(np.asarray(a.filled(np.nan), dtype="float64"))
    if a.mask is not np.ma.nomask:
        finite &= ~np.ma.getmaskarray(a)
    if not finite.any():
        return None, "empty chip (no radar data)"
    return info, None


def dem_grid(info):
    """30 m UTM grid over the chip plus PAD_M, snapped to whole 30 m cells."""
    lon0, lat0, lon1, lat1 = transform_bounds(info["crs"], "EPSG:4326", *info["bounds"])
    epsg = utm_epsg((lon0 + lon1) / 2, (lat0 + lat1) / 2)
    x0, y0, x1, y1 = transform_bounds(info["crs"], f"EPSG:{epsg}", *info["bounds"])
    x0 = np.floor((x0 - PAD_M) / RES) * RES
    y1 = np.ceil((y1 + PAD_M) / RES) * RES
    x1 = np.ceil((x1 + PAD_M) / RES) * RES
    y0 = np.floor((y0 - PAD_M) / RES) * RES
    w, h = int(round((x1 - x0) / RES)), int(round((y1 - y0) / RES))
    transform = rasterio.Affine(RES, 0, x0, 0, -RES, y1)
    return epsg, transform, w, h


# =============================================================================
# The two sources of DEM and water: Earth Engine, or local files for --selftest
# =============================================================================

_EE_READY = False


def ee_init():
    global _EE_READY
    if _EE_READY:
        return
    try:
        import ee
    except ImportError:
        sys.exit("\nThe earthengine-api package is not in this environment. "
                 "Run this in the fyp environment: conda activate fyp\n")
    try:
        ee.Initialize(project=EE_PROJECT)
    except Exception as exc:                      # noqa: BLE001
        sys.exit("\nCould not connect to Earth Engine.\n"
                 f"  project tried : {EE_PROJECT}\n  message       : {exc}\n\n"
                 "If it says you are not authenticated, run:\n"
                 "    earthengine authenticate\n"
                 "then run this script again.\n")
    _EE_READY = True


def fetch_ee(epsg, transform, w, h, tries=4):
    """DEM and JRC occurrence on the given grid, from Earth Engine."""
    import ee
    dem = (ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM").mosaic()
           .unmask(DEM_NODATA).toFloat().rename("dem"))
    occ = (ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
           .unmask(0).toFloat().rename("occ"))
    request = {
        "expression": dem.addBands(occ),
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": w, "height": h},
            "affineTransform": {"scaleX": transform.a, "shearX": 0,
                                "translateX": transform.c, "shearY": 0,
                                "scaleY": transform.e, "translateY": transform.f},
            "crsCode": f"EPSG:{epsg}",
        },
    }
    for attempt in range(tries):
        try:
            arr = ee.data.computePixels(request)
            return (np.asarray(arr["dem"], dtype="float32"),
                    np.asarray(arr["occ"], dtype="float32"))
        except Exception as exc:                  # noqa: BLE001
            if attempt == tries - 1:
                raise
            msg = str(exc)
            time.sleep(5 * (attempt + 1) if "429" in msg or "quota" in msg.lower() else 2)


def fetch_local(epsg, transform, w, h, dem_path, water_path):
    """The same grid, cut from local GeoTIFFs. Used by --selftest only."""
    out = []
    for path, fill in ((dem_path, DEM_NODATA), (water_path, 0.0)):
        dst = np.full((h, w), np.nan, dtype="float64")
        with rasterio.open(path) as src:
            reproject(source=rasterio.band(src, 1), destination=dst,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs=f"EPSG:{epsg}",
                      resampling=Resampling.nearest,
                      src_nodata=src.nodata, dst_nodata=np.nan)
        out.append(np.where(np.isfinite(dst), dst, fill).astype("float32"))
    return out[0], out[1]


# =============================================================================
# Terrain, with 03_terrain_strata.py's own functions
# =============================================================================

def terrain(dem, occ):
    """HAND (m) and slope (deg) on the 30 m grid, exactly as 03 does it."""
    dem = dem.astype("float64")
    dem[dem <= DEM_NODATA + 1] = np.nan
    if not np.isfinite(dem).any():
        raise ValueError("no DEM values in the box")
    dem = np.where(np.isfinite(dem), dem, np.nanmin(dem))   # as 03: fill for routing
    slope, _aspect = ts.slope_aspect(dem, RES, RES)
    fdir = ts.d8_flowdir(dem)
    # accumulation by descending elevation, copied line for line from 03 main()
    order = np.argsort(-dem.ravel())
    acc = np.ones(dem.size, dtype=np.int64)
    flat_dir = fdir.ravel()
    for idx in order:
        t = flat_dir[idx]
        if t >= 0:
            acc[t] += acc[idx]
    stream = (acc >= ts.STREAM_ACCUM_CELLS).reshape(dem.shape)
    stream = stream | (np.nan_to_num(occ, nan=0.0) >= WATER_THRESHOLD)
    h = ts.hand(dem, fdir, stream)
    return h.astype("float32"), np.degrees(slope).astype("float32")


def onto_chip(arr, epsg, transform, info):
    dst = np.full((info["height"], info["width"]), np.nan, dtype="float32")
    reproject(source=arr, destination=dst,
              src_transform=transform, src_crs=f"EPSG:{epsg}",
              dst_transform=info["transform"], dst_crs=info["crs"],
              resampling=Resampling.bilinear,
              src_nodata=np.nan, dst_nodata=np.nan)
    return dst


def write_tif(path, bands, names, crs, transform):
    tmp = path + ".part"
    with rasterio.open(tmp, "w", driver="GTiff", height=bands[0].shape[0],
                       width=bands[0].shape[1], count=len(bands), dtype="float32",
                       crs=crs, transform=transform, nodata=np.nan,
                       compress="deflate", predictor=3) as dst:
        for i, (b, n) in enumerate(zip(bands, names), start=1):
            dst.write(b.astype("float32"), i)
            dst.set_band_description(i, n)
    os.replace(tmp, path)          # never leaves a half-written file behind


# =============================================================================
# One chip, start to finish
# =============================================================================

def get_30m(chip_id, info, out_dir, fetch):
    """The 30 m DEM and water for a chip, from the cache or from the source."""
    epsg, transform, w, h = dem_grid(info)
    cache = os.path.join(out_dir, "dem_jrc_30m", f"{chip_id}_dem_jrc_30m.tif")
    if os.path.exists(cache):
        with rasterio.open(cache) as src:
            return epsg, transform, src.read(1), src.read(2), True
    dem, occ = fetch(epsg, transform, w, h)
    if dem.shape != (h, w):
        raise ValueError(f"download came back {dem.shape}, asked for {(h, w)}")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    write_tif(cache, [dem, occ], ["dem_m", "jrc_occurrence_pct"],
              f"EPSG:{epsg}", transform)
    return epsg, transform, dem, occ, False


def finish_chip(chip_id, info, epsg, transform, dem, occ, out_dir):
    h30, s30 = terrain(dem, occ)
    hand_c = onto_chip(h30, epsg, transform, info)
    slope_c = onto_chip(s30, epsg, transform, info)
    out = os.path.join(out_dir, f"{chip_id}_Terrain.tif")
    write_tif(out, [hand_c, slope_c], ["hand_m", "slope_deg"],
              info["crs"], info["transform"])
    ok = np.isfinite(hand_c)
    return {
        "hand_defined_pct": round(100 * ok.mean(), 2),
        "hand_median_m": round(float(np.nanmedian(hand_c)), 2) if ok.any() else "",
        "hand_lt5m_pct": round(100 * float(np.mean(hand_c[ok] < 5)), 2) if ok.any() else "",
        "slope_median_deg": round(float(np.nanmedian(slope_c)), 2),
        "dem_min_m": round(float(np.nanmin(np.where(dem > DEM_NODATA + 1, dem, np.nan))), 1),
        "dem_max_m": round(float(np.nanmax(dem)), 1),
    }


# =============================================================================
# --selftest: run the whole chain on the corridor, against 03's own HAND
# =============================================================================

def selftest():
    data = os.path.join(HERE, "data")
    dem_path = os.path.join(HERE, "..", "..", "05 Data", "dem", "DEM_COP_GLO30_30m.tif")
    water_path = os.path.join(HERE, "..", "..", "05 Data", "dem", "WATER_JRC_occurrence_30m.tif")
    strata = os.path.join(data, "strata_2021_v3")
    for p in (dem_path, water_path, os.path.join(strata, "strata_hand_m.tif")):
        if not os.path.exists(p):
            sys.exit(f"selftest needs {p}")
    print("SELFTEST. Three fake 5.12 km chips on the corridor, in lat/lon like")
    print("Sen1Floods11, run through the whole chain with the corridor DEM and")
    print("JRC files in place of Earth Engine, then compared with the HAND and")
    print("slope 03_terrain_strata.py made for the whole corridor.\n")
    # chip centres: near Srinagar, near Devprayag, near Rishikesh (lon, lat)
    centres = [(78.78, 30.22), (78.60, 30.15), (78.33, 30.08)]
    fails = 0
    for k, (lon, lat) in enumerate(centres, start=1):
        px = 8.983152841195214e-05          # Sen1Floods11's 10 m in degrees
        half = 256 * px
        info = {"crs": rasterio.crs.CRS.from_epsg(4326),
                "transform": rasterio.Affine(px, 0, lon - half, 0, -px, lat + half),
                "width": 512, "height": 512}
        info["bounds"] = rasterio.coords.BoundingBox(lon - half, lat - half,
                                                     lon + half, lat + half)
        epsg, transform, w, h = dem_grid(info)
        dem, occ = fetch_local(epsg, transform, w, h, dem_path, water_path)
        h30, s30 = terrain(dem, occ)
        hand_c = onto_chip(h30, epsg, transform, info)
        slope_c = onto_chip(s30, epsg, transform, info)
        # 03's corridor rasters onto the same chip grid
        ref = {}
        for name in ("strata_hand_m.tif", "strata_slope_deg.tif"):
            dst = np.full((512, 512), np.nan, dtype="float32")
            with rasterio.open(os.path.join(strata, name)) as src:
                reproject(source=rasterio.band(src, 1), destination=dst,
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=info["transform"], dst_crs=info["crs"],
                          resampling=Resampling.bilinear,
                          src_nodata=src.nodata, dst_nodata=np.nan)
            ref[name] = dst
        both = np.isfinite(hand_c) & np.isfinite(ref["strata_hand_m.tif"])
        sl = np.isfinite(slope_c) & np.isfinite(ref["strata_slope_deg.tif"])
        dslope = np.abs(slope_c[sl] - ref["strata_slope_deg.tif"][sl])
        hc, hr = hand_c[both], ref["strata_hand_m.tif"][both]
        low_r, low_c = hr < 15, hc < 15
        agree15 = float(np.mean(low_r == low_c)) if both.any() else float("nan")
        r = float(np.corrcoef(hc, hr)[0, 1]) if both.sum() > 10 else float("nan")
        print(f"  chip {k} at {lon} E {lat} N, UTM {epsg}, box {w} x {h} cells")
        print(f"    cells compared with the corridor: {int(both.sum()):,}")
        print(f"    slope: median difference {np.median(dslope):.2f} deg "
              f"(should be 0: same DEM, same formula)")
        print(f"    HAND: correlation {r:.3f}; same side of 15 m in {100*agree15:.1f} % "
              f"of cells; median {np.median(hc):.1f} m against {np.median(hr):.1f} m")
        if not (np.median(dslope) < 0.5 and r > 0.8 and agree15 > 0.85):
            fails += 1
            print("    ** outside the expected range")
    print()
    if fails:
        print(f"SELFTEST: {fails} of 3 chips outside the expected range. Check the terrain inputs before going on.")
        return 1
    print("SELFTEST PASSED. The chain gives the same slope and HAND as")
    print("03_terrain_strata.py on the same DEM, so chips and corridor are computed")
    print("the same way. It cannot measure the 10 km box edge effect described at")
    print("the top of this file: the corridor DEM stops at the corridor edge.")
    return 0


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=["hand", "weak"], default="hand",
                    help="which Sen1Floods11 chips: hand-labelled (446) or weak (about 4,400)")
    ap.add_argument("--test", action="store_true",
                    help="only three chips, from three different events, then stop")
    ap.add_argument("--workers", type=int, default=4,
                    help="downloads at the same time (default 4)")
    ap.add_argument("--selftest", action="store_true",
                    help="offline check on the corridor, no internet needed")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    chip_dir, suffix = SETS[args.set]
    if not os.path.isdir(chip_dir):
        sys.exit(f"\nNot found: {chip_dir}\nSen1Floods11 should be in data\\sen1floods11.\n")
    chips = sorted(f for f in os.listdir(chip_dir) if f.endswith(suffix))
    out_dir = os.path.join(OUT_ROOT, args.set + ("_test" if args.test else ""))
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "terrain_log.csv")
    cols = ["chip", "status", "seconds", "downloaded", "hand_defined_pct",
            "hand_median_m", "hand_lt5m_pct", "slope_median_deg", "dem_min_m", "dem_max_m"]
    new_log = not os.path.exists(log_path)
    already_logged = set()
    if not new_log:
        with open(log_path, newline="", encoding="utf-8") as fh:
            already_logged = {(r["chip"], r["status"]) for r in csv.DictReader(fh)}
    logf = open(log_path, "a", newline="", encoding="utf-8")
    log = csv.DictWriter(logf, fieldnames=cols)
    if new_log:
        log.writeheader()

    todo, done, skipped, events = [], 0, 0, set()
    for f in chips:
        chip_id = f[: -len(suffix)]
        event = chip_id.split("_")[0]
        if args.test and (event in events or len(events) == 3):
            continue                       # test: the first usable chip of three events
        if os.path.exists(os.path.join(out_dir, f"{chip_id}_Terrain.tif")):
            done += 1
            events.add(event)
            continue
        info, why = chip_info(os.path.join(chip_dir, f))
        if info is None:
            skipped += 1
            if (chip_id, why) not in already_logged:
                log.writerow({"chip": chip_id, "status": why})
            continue
        events.add(event)
        todo.append((chip_id, info))
    if args.test:
        chips = [c for c in chips if c.split("_")[0] in events]
    logf.flush()
    print(f"{args.set}: {len(chips)} chips. Already done {done}, "
          f"skipped {skipped} (empty or unreadable, see the log), to do {len(todo)}.")
    if not todo:
        print(f"Nothing to do. Output: {out_dir}")
        return 0

    ee_init()
    print(f"Connected to Earth Engine, project '{EE_PROJECT}'. "
          f"Downloading with {args.workers} at a time.\n")

    t_start, n_ok, n_fail = time.time(), 0, 0

    def job(item):
        chip_id, info = item
        t0 = time.time()
        epsg, transform, dem, occ, cached = get_30m(chip_id, info, out_dir, fetch_ee)
        return chip_id, info, epsg, transform, dem, occ, cached, t0

    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(job, it): it[0] for it in todo}
        for i, fut in enumerate(cf.as_completed(futures), start=1):
            chip_id = futures[fut]
            try:
                chip_id, info, epsg, transform, dem, occ, cached, t0 = fut.result()
                stats = finish_chip(chip_id, info, epsg, transform, dem, occ, out_dir)
                row = {"chip": chip_id, "status": "ok",
                       "seconds": round(time.time() - t0, 1),
                       "downloaded": "no, cached" if cached else "yes", **stats}
                n_ok += 1
            except Exception as exc:              # noqa: BLE001
                row = {"chip": chip_id, "status": f"FAILED: {exc}"[:300]}
                n_fail += 1
            log.writerow(row)
            logf.flush()
            elapsed = time.time() - t_start
            left = (len(todo) - i) * elapsed / i
            line = (f"  {i}/{len(todo)}  {chip_id:<26} {row['status'][:40]:<40}")
            if row["status"] == "ok":
                line += (f"  HAND median {row['hand_median_m']} m, "
                         f"slope {row['slope_median_deg']} deg")
            print(line + f"   about {left/60:.0f} min left", flush=True)
    logf.close()

    print(f"\nDone: {n_ok} written, {n_fail} failed, {skipped} skipped as empty.")
    print(f"Output: {out_dir}")
    print(f"Log:    {log_path}")
    if n_fail:
        print("Run the same command again: finished chips are skipped and the "
              "failed ones are tried again. If the same chips fail twice, look "
              "at terrain_log.csv for the reason.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
