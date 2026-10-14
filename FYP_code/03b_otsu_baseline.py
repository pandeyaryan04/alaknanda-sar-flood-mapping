#!/usr/bin/env python3
"""
03b_otsu_baseline.py
Otsu thresholding baseline for SAR flood detection, Alaknanda corridor.

This is the first script in the pipeline that produces a flood map. Everything
downstream (04_stratified_metrics.py) scores what this produces.

WHY 03b AND NOT 05
------------------
The run order is 00, 01, 02, 03, THIS, 04. 04_stratified_metrics.py consumes the
flood map this script writes, so a file called 05 would sit after the script that
needs it and anyone running the folder in numeric order would score a map that
does not exist yet. Rename it to 05_ if you prefer plain numbering, but then put
a line at the top of 04 saying it must not run first.

PIPELINE
--------
  1. read pre and post Sentinel-1 VV
  2. convert to linear power if the input is in dB
  3. speckle filter in LINEAR power (never in dB, see note below)
  4. convert to dB
  5. choose a threshold by Otsu, optionally on bimodal tiles only
  6. threshold to a water mask
  7. subtract permanent water (JRC occurrence)
  8. remove layover and shadow (mask from 03_terrain_strata.py)
  9. write the mask, a diagnostics JSON, and a summary you should actually read

WHY THE FILTER RUNS IN LINEAR POWER
-----------------------------------
Speckle is multiplicative in power. In dB it becomes additive, and the mean of a
set of dB values is not the dB of their mean, so filtering in dB biases every
window low. GEE's COPERNICUS/S1_GRD is already in dB; COPERNICUS/S1_GRD_FLOAT is
linear power. This script detects which it has and says so. Check that line.

HOW TO RUN
----------
Self-test first. Takes about ten seconds and needs no data:

    python 03b_otsu_baseline.py --selftest

Real run, 2023 event, from 04 Code\\FYP_code\\:

    python 03b_otsu_baseline.py --post data\\sentinel1\\s1_2023_post.tif --pre data\\sentinel1\\s1_2023_pre.tif --water data\\dem\\jrc_occurrence.tif --mask data\\strata_2023\\distortion.tif --mask-values 1,2 --out data\\flood_2023 --event 2023

Minimum viable run if you do not have the aux rasters yet. It will warn about
every one of them:

    python 03b_otsu_baseline.py --post data\\sentinel1\\s1_2023_post.tif --out data\\flood_2023 --event 2023

Windows Command Prompt continues lines with ^, not \\. The commands above are
single lines on purpose.

ASSUMPTIONS MADE, BECAUSE THE MACHINE WAS NOT REACHABLE WHEN THIS WAS WRITTEN
----------------------------------------------------------------------------
  * VV is band 1 of the Sentinel-1 export, VH is band 2, angle is band 3. This
    matches 01_export_s1.js as described in the 21 August delta. Override with
    --vv-band if wrong.
  * 03_terrain_strata.py writes a distortion raster where layover and shadow are
    distinct nonzero integer codes. If it writes something else, point --mask at
    whatever raster does carry those flags and set --mask-values to match.
  * The JRC raster is water occurrence as a percentage, 0 to 100.
  * All aux rasters are reprojected onto the post scene's grid by nearest
    neighbour, which is what 03_terrain_strata.py already does for the angle band.

Written 26 August 2026.
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    HAVE_RASTERIO = True
except ImportError:
    HAVE_RASTERIO = False

try:
    from scipy.ndimage import uniform_filter
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


# --------------------------------------------------------------------------
# units
# --------------------------------------------------------------------------

def looks_like_db(a):
    """Sentinel-1 VV in dB sits roughly in -30..+5. In linear power it sits in
    0..~2 and is never negative. Negative values are the giveaway."""
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return False
    return float(np.nanmin(finite)) < -1.0


def db_to_linear(a):
    return np.power(10.0, a / 10.0)


def linear_to_db(a):
    out = np.full(a.shape, np.nan, dtype=np.float64)
    good = np.isfinite(a) & (a > 0)
    out[good] = 10.0 * np.log10(a[good])
    return out


# --------------------------------------------------------------------------
# speckle
# --------------------------------------------------------------------------

def lee_filter(img, size=5, cu=None):
    """Lee (1980) adaptive filter for multiplicative speckle. Input must be
    linear power. cu is the noise coefficient of variation; for Sentinel-1 GRD
    with roughly 4.4 equivalent looks, 1/sqrt(4.4) is about 0.48."""
    if not HAVE_SCIPY:
        raise RuntimeError("scipy is required for the Lee filter. "
                           "conda install scipy, or run with --filter none")
    if cu is None:
        cu = 1.0 / np.sqrt(4.4)

    a = np.where(np.isfinite(img), img, 0.0).astype(np.float64)
    valid = np.isfinite(img).astype(np.float64)

    # local mean over valid cells only, so nodata does not drag windows down
    s = uniform_filter(a, size=size, mode="nearest")
    n = uniform_filter(valid, size=size, mode="nearest")
    n = np.where(n <= 0, np.nan, n)
    mean = s / n

    s2 = uniform_filter(a * a, size=size, mode="nearest")
    var = np.maximum(s2 / n - mean * mean, 0.0)

    ci2 = np.divide(var, mean * mean, out=np.zeros_like(var),
                    where=np.abs(mean) > 1e-12)
    cu2 = cu * cu
    w = np.clip((ci2 - cu2) / np.maximum(ci2, 1e-12), 0.0, 1.0)

    out = mean + w * (img - mean)
    return np.where(np.isfinite(img), out, np.nan)


def boxcar_filter(img, size=5):
    if not HAVE_SCIPY:
        raise RuntimeError("scipy is required for the boxcar filter.")
    a = np.where(np.isfinite(img), img, 0.0).astype(np.float64)
    valid = np.isfinite(img).astype(np.float64)
    s = uniform_filter(a, size=size, mode="nearest")
    n = uniform_filter(valid, size=size, mode="nearest")
    n = np.where(n <= 0, np.nan, n)
    return np.where(np.isfinite(img), s / n, np.nan)


# --------------------------------------------------------------------------
# Otsu
# --------------------------------------------------------------------------

def otsu_threshold(values, bins=256, vmin=None, vmax=None):
    """Otsu (1979) on a 1-D array. Returns (threshold, separability).

    separability is the between-class variance at the chosen threshold divided
    by the total variance. It runs 0 to 1. Below about 0.7 the histogram is not
    convincingly bimodal and the threshold should not be trusted."""
    v = values[np.isfinite(values)]
    if v.size < 100:
        raise ValueError("fewer than 100 finite values, cannot threshold")

    lo = np.percentile(v, 0.5) if vmin is None else vmin
    hi = np.percentile(v, 99.5) if vmax is None else vmax
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError("degenerate histogram range")

    hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
    centres = 0.5 * (edges[:-1] + edges[1:])
    p = hist.astype(np.float64) / max(hist.sum(), 1)

    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    m0 = np.cumsum(p * centres)
    mt = m0[-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        mean0 = m0 / w0
        mean1 = (mt - m0) / w1
    between = w0 * w1 * (mean0 - mean1) ** 2
    between = np.where(np.isfinite(between), between, 0.0)

    k = int(np.argmax(between))
    total_var = float(np.var(v))
    sep = float(between[k] / total_var) if total_var > 0 else 0.0
    return float(centres[k]), sep


def bimodal_tiles(arr, tiles=10, min_std=1.5, min_valid=0.5):
    """Martinis-style tile selection. Split into tiles x tiles, keep tiles whose
    dB standard deviation is high and whose mean is below the scene mean. Those
    are the tiles that straddle a land/water boundary. Otsu on the pooled
    histogram of only those tiles finds the real threshold instead of splitting
    the dry-land mode in half.

    Returns (pooled_values, n_selected, n_total)."""
    h, w = arr.shape
    rs = np.linspace(0, h, tiles + 1).astype(int)
    cs = np.linspace(0, w, tiles + 1).astype(int)

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.array([]), 0, 0
    scene_mean = float(np.mean(finite))

    kept = []
    n_total = 0
    for i in range(tiles):
        for j in range(tiles):
            block = arr[rs[i]:rs[i + 1], cs[j]:cs[j + 1]]
            if block.size == 0:
                continue
            n_total += 1
            v = block[np.isfinite(block)]
            if v.size < 0.0 + min_valid * block.size:
                continue
            if float(np.std(v)) < min_std:
                continue
            if float(np.mean(v)) > scene_mean:
                continue
            kept.append(v)

    if not kept:
        return np.array([]), 0, n_total
    return np.concatenate(kept), len(kept), n_total


# --------------------------------------------------------------------------
# raster io
# --------------------------------------------------------------------------

def read_band(path, band):
    with rasterio.open(path) as src:
        if band > src.count:
            raise ValueError("%s has %d bands, band %d was asked for"
                             % (os.path.basename(path), src.count, band))
        a = src.read(band).astype(np.float64)
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
        prof = src.profile.copy()
    return a, prof


def align_to(path, band, ref_profile):
    """Reproject one band of path onto the reference grid, nearest neighbour."""
    with rasterio.open(path) as src:
        dst = np.full((ref_profile["height"], ref_profile["width"]),
                      np.nan, dtype=np.float64)
        reproject(
            source=rasterio.band(src, band),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            resampling=Resampling.nearest,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )
    return dst


def write_mask(path, arr_uint8, ref_profile):
    prof = ref_profile.copy()
    prof.update(dtype="uint8", count=1, nodata=255, compress="lzw")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr_uint8, 1)


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

def run(args):
    if not HAVE_RASTERIO:
        sys.exit("rasterio is not installed. conda install -c conda-forge rasterio")

    os.makedirs(args.out, exist_ok=True)
    diag = {"event": args.event, "warnings": []}

    def warn(msg):
        diag["warnings"].append(msg)
        print("  WARNING: " + msg)

    print("")
    print("=" * 68)
    print("  Otsu flood baseline, event %s" % args.event)
    print("=" * 68)

    # 1. post scene
    post_raw, prof = read_band(args.post, args.vv_band)
    print("  post scene : %s" % os.path.basename(args.post))
    print("  grid       : %d x %d, %s" % (prof["height"], prof["width"], prof["crs"]))

    in_db = looks_like_db(post_raw)
    diag["input_units"] = "dB" if in_db else "linear power"
    print("  input units: %s" % diag["input_units"])
    post_lin = db_to_linear(post_raw) if in_db else post_raw

    valid = np.isfinite(post_lin)
    diag["valid_cells"] = int(valid.sum())
    diag["total_cells"] = int(post_lin.size)
    frac_valid = diag["valid_cells"] / max(diag["total_cells"], 1)
    px_km2 = abs(prof["transform"][0] * prof["transform"][4]) / 1e6
    corridor_km2 = diag["valid_cells"] * px_km2
    diag["corridor_km2"] = round(corridor_km2, 3)
    print("  valid cells: %.2f%%  =  %.1f km2" % (100.0 * frac_valid, corridor_km2))
    print("               (a low percentage here is expected and correct: the")
    print("                export is clipped to the corridor inside a rectangle)")

    # 2. speckle filter, in linear power
    if args.filter == "lee":
        post_lin = lee_filter(post_lin, size=args.window)
    elif args.filter == "boxcar":
        post_lin = boxcar_filter(post_lin, size=args.window)
    print("  filter     : %s, %dx%d" % (args.filter, args.window, args.window))

    post_db = linear_to_db(post_lin)

    # 3. build the surface Otsu runs on
    if args.mode == "change":
        if not args.pre:
            sys.exit("--mode change needs --pre")
        pre_raw, _ = read_band(args.pre, args.vv_band)
        pre_lin = db_to_linear(pre_raw) if looks_like_db(pre_raw) else pre_raw
        if pre_lin.shape != post_lin.shape:
            pre_lin = align_to(args.pre, args.vv_band, prof)
            if looks_like_db(pre_lin):
                pre_lin = db_to_linear(pre_lin)
            warn("pre scene was on a different grid and has been resampled")
        if args.filter == "lee":
            pre_lin = lee_filter(pre_lin, size=args.window)
        elif args.filter == "boxcar":
            pre_lin = boxcar_filter(pre_lin, size=args.window)
        surface = post_db - linear_to_db(pre_lin)
        print("  mode       : change, post dB minus pre dB")
        print("               new water is strongly negative")
    else:
        surface = post_db
        print("  mode       : single, Otsu on post VV dB")
        if args.pre:
            print("               (--pre supplied but unused in single mode)")

    # 4. threshold
    if args.tiles > 1:
        pooled, n_sel, n_tot = bimodal_tiles(surface, tiles=args.tiles,
                                             min_std=args.tile_min_std)
        print("  tiles      : %d of %d selected as bimodal" % (n_sel, n_tot))
        if n_sel < 3:
            warn("fewer than 3 bimodal tiles found, falling back to the whole scene. "
                 "Either the scene has very little water or --tile-min-std is too high")
            pooled = surface[np.isfinite(surface)]
        diag["tiles_selected"] = n_sel
        diag["tiles_total"] = n_tot
    else:
        pooled = surface[np.isfinite(surface)]
        diag["tiles_selected"] = None

    # Physical search limits. Without these, Otsu on a corridor that is 95%+ dry
    # land splits the LAND distribution in half and returns a threshold around
    # -7 dB, which floods most of the map. Sentinel-1 VV over open water sits
    # near -15 to -25 dB, so the threshold has to be looked for there.
    if args.no_search_limits:
        smin = smax = None
        print("  search     : unconstrained (--no-search-limits)")
    else:
        smin = args.search_min
        smax = args.search_max
        if smin is None:
            smin = -30.0 if args.mode == "single" else -20.0
        if smax is None:
            smax = -10.0 if args.mode == "single" else -2.0
        print("  search     : %.1f to %.1f dB" % (smin, smax))
    diag["search_min"] = smin
    diag["search_max"] = smax

    if args.threshold is not None:
        thr = float(args.threshold)
        sep = float("nan")
        print("  threshold  : %.3f dB (set by hand, Otsu skipped)" % thr)
    else:
        thr, sep = otsu_threshold(pooled, bins=args.bins, vmin=smin, vmax=smax)
        print("  threshold  : %.3f dB (Otsu)" % thr)
        print("  separability: %.3f" % sep)
        if sep < 0.7 and args.no_search_limits:
            warn("separability %.3f is below 0.7 on an unconstrained search. "
                 "The threshold is probably splitting the land distribution "
                 "rather than separating land from water. Drop "
                 "--no-search-limits." % sep)
        elif sep < 0.7:
            print("               (low separability inside a narrow search")
            print("                window is normal when the scene is mostly")
            print("                dry. The area check below is the real test)")
        if args.mode == "single" and thr > -10.0:
            warn("threshold %.2f dB is too high for open water in Sentinel-1 VV, "
                 "which sits near -15 to -25 dB. This map is almost certainly "
                 "land, not flood." % thr)

    diag["threshold_db"] = thr
    diag["separability"] = None if np.isnan(sep) else sep

    water = np.zeros(surface.shape, dtype=bool)
    water[np.isfinite(surface)] = surface[np.isfinite(surface)] < thr
    diag["water_before_masks_pct"] = round(
        100.0 * water.sum() / max(diag["valid_cells"], 1), 4)
    print("  water, raw : %.3f%% of valid cells" % diag["water_before_masks_pct"])

    # 5. permanent water
    if args.water:
        occ = align_to(args.water, 1, prof)
        perm = np.isfinite(occ) & (occ >= args.water_threshold)
        diag["permanent_water_pct"] = round(
            100.0 * perm.sum() / max(diag["valid_cells"], 1), 4)
        removed = int((water & perm).sum())
        water = water & ~perm
        print("  permanent water: %.3f%% of scene, %d cells removed from the flood map"
              % (diag["permanent_water_pct"], removed))
        print("  NOTE: JRC occurrence spans 1984 onward. The Srinagar reservoir was")
        print("        filled in 2015, so a full-record occurrence layer may under-")
        print("        represent it. Check the reservoir looks right on the output.")
    else:
        warn("no --water supplied. The permanent river and the Srinagar reservoir "
             "are still counted as flood. Every accuracy figure from this run is "
             "meaningless until you add it")
        diag["permanent_water_pct"] = None

    # 6. layover and shadow
    if args.mask:
        m = align_to(args.mask, args.mask_band, prof)
        codes = [float(x) for x in args.mask_values.split(",") if x.strip() != ""]
        excl = np.zeros(m.shape, dtype=bool)
        for c in codes:
            excl |= np.isclose(m, c)
        diag["distortion_pct"] = round(
            100.0 * excl.sum() / max(diag["valid_cells"], 1), 4)
        removed = int((water & excl).sum())
        water = water & ~excl
        print("  layover and shadow: %.3f%% of scene, %d cells removed"
              % (diag["distortion_pct"], removed))
        if diag["distortion_pct"] < 1.0:
            warn("distortion covers only %.3f%% of the corridor. The 26 August "
                 "run gave 1.48%% layover and that was already flagged as low for "
                 "a confined gorge. Check the 5 km buffer before trusting this"
                 % diag["distortion_pct"])
    else:
        warn("no --mask supplied. Layover and shadow are not excluded, so the "
             "terrain-stratified claim cannot be made from this run")
        diag["distortion_pct"] = None

    # 7. write
    out_mask = np.full(water.shape, 255, dtype=np.uint8)
    out_mask[valid] = 0
    out_mask[water] = 1
    mask_path = os.path.join(args.out, "flood_%s_otsu.tif" % args.event)
    write_mask(mask_path, out_mask, prof)

    diag["water_final_pct"] = round(
        100.0 * water.sum() / max(diag["valid_cells"], 1), 4)
    diag["water_final_cells"] = int(water.sum())

    px = abs(prof["transform"][0] * prof["transform"][4])
    diag["water_final_km2"] = round(water.sum() * px / 1e6, 3)

    diag_path = os.path.join(args.out, "flood_%s_diagnostics.json" % args.event)
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2)

    print("")
    print("  RESULT")
    print("  flood after all masks : %.4f%% of the corridor"
          % diag["water_final_pct"])
    print("  flood area            : %.3f km2  of %.1f km2 corridor"
          % (diag["water_final_km2"], corridor_km2))

    # A 90 km reach of a Himalayan river cannot flood a quarter of a 5 km buffer.
    if diag["water_final_pct"] > 25.0:
        warn("%.1f%% of the corridor mapped as flood. That is not physically "
             "possible on this reach. The threshold is wrong, not the river."
             % diag["water_final_pct"])
    elif diag["water_final_pct"] > 10.0:
        warn("%.1f%% of the corridor mapped as flood, which is high for this "
             "reach. Look at the map before using the number."
             % diag["water_final_pct"])
    print("")
    print("  written: %s" % mask_path)
    print("  written: %s" % diag_path)
    print("")
    if diag["warnings"]:
        print("  %d warning(s). Read them before using this output for anything."
              % len(diag["warnings"]))
    print("  Next: 04_stratified_metrics.py, which needs a --truth raster that")
    print("  does not exist yet. Decide where it comes from first.")
    print("")
    return 0


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def selftest():
    print("")
    print("=" * 68)
    print("  03b_otsu_baseline.py self-test")
    print("=" * 68)
    ok = True
    rng = np.random.default_rng(20260826)

    def check(name, condition, detail=""):
        nonlocal ok
        print("  [%s] %s%s" % ("PASS" if condition else "FAIL", name,
                               ("  " + detail) if detail else ""))
        if not condition:
            ok = False

    # 1. unit round trip
    lin = np.array([0.001, 0.01, 0.1, 1.0])
    check("dB round trip", np.allclose(db_to_linear(linear_to_db(lin)), lin))
    check("dB detection, dB input", looks_like_db(linear_to_db(lin)))
    check("dB detection, linear input", not looks_like_db(lin))

    # 2. Otsu on a known bimodal mixture
    land = rng.normal(0.10, 0.02, 90000)      # linear power, dry land
    water = rng.normal(0.008, 0.002, 10000)   # linear power, smooth water
    mix = linear_to_db(np.clip(np.concatenate([land, water]), 1e-6, None))
    thr, sep = otsu_threshold(mix)
    frac = float(np.mean(mix < thr))
    check("Otsu threshold sits between the modes",
          -22.0 < thr < -14.0, "thr = %.2f dB" % thr)
    check("Otsu recovers roughly the true water fraction",
          0.07 < frac < 0.13, "recovered %.3f, true 0.100" % frac)
    check("separability is high on a clean mixture",
          sep > 0.7, "sep = %.3f" % sep)

    # 3. Otsu warns on a unimodal scene
    uni = linear_to_db(np.clip(rng.normal(0.10, 0.02, 100000), 1e-6, None))
    _, sep_uni = otsu_threshold(uni)
    check("separability is low on a unimodal scene",
          sep_uni < 0.7, "sep = %.3f" % sep_uni)

    # 4. Lee filter reduces variance over homogeneous ground and keeps the mean
    if HAVE_SCIPY:
        flat = rng.gamma(shape=4.4, scale=0.1 / 4.4, size=(200, 200))
        filt = lee_filter(flat, size=5)
        check("Lee filter reduces speckle variance",
              np.nanstd(filt) < 0.75 * np.nanstd(flat),
              "%.4f -> %.4f" % (np.nanstd(flat), np.nanstd(filt)))
        check("Lee filter preserves the mean to within 2%",
              abs(np.nanmean(filt) - np.nanmean(flat)) / np.nanmean(flat) < 0.02)

        # edge preservation: a step should survive
        step = np.full((200, 200), 0.10)
        step[:, 100:] = 0.01
        step = step * rng.gamma(shape=4.4, scale=1.0 / 4.4, size=(200, 200))
        fs = lee_filter(step, size=5)
        left = np.nanmean(fs[:, 20:80])
        right = np.nanmean(fs[:, 120:180])
        check("Lee filter preserves a land/water step",
              left / right > 5.0, "contrast %.1fx" % (left / right))
    else:
        print("  [SKIP] Lee filter tests, scipy not installed")

    # 5. tile selection prefers tiles containing water
    scene = linear_to_db(np.clip(rng.normal(0.10, 0.02, (300, 300)), 1e-6, None))
    scene[140:160, :] = linear_to_db(
        np.clip(rng.normal(0.008, 0.002, (20, 300)), 1e-6, None))
    pooled, n_sel, n_tot = bimodal_tiles(scene, tiles=10, min_std=1.5)
    check("tile selection finds the water-bearing row",
          0 < n_sel <= n_tot, "%d of %d tiles" % (n_sel, n_tot))
    if n_sel > 0:
        thr_t, _ = otsu_threshold(pooled)
        frac_t = float(np.mean(scene[np.isfinite(scene)] < thr_t))
        true_frac = 20.0 / 300.0
        check("tiled threshold recovers the water fraction better than 2x error",
              0.5 * true_frac < frac_t < 2.0 * true_frac,
              "recovered %.4f, true %.4f" % (frac_t, true_frac))

    # 6. masks subtract rather than add
    w = np.zeros((10, 10), dtype=bool)
    w[2:8, 2:8] = True
    perm = np.zeros((10, 10), dtype=bool)
    perm[4:6, 4:6] = True
    after = w & ~perm
    check("permanent water is subtracted, never added",
          after.sum() == w.sum() - 4 and not (after & perm).any())

    print("")
    print("  %s" % ("ALL CHECKS PASSED" if ok else "SOMETHING FAILED, DO NOT RUN ON REAL DATA"))
    print("")
    return 0 if ok else 1


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Otsu thresholding baseline for SAR flood detection.")
    p.add_argument("--selftest", action="store_true",
                   help="run the built-in checks and exit, no data needed")
    p.add_argument("--post", help="post-event Sentinel-1 GeoTIFF")
    p.add_argument("--pre", help="pre-event Sentinel-1 GeoTIFF, needed for --mode change")
    p.add_argument("--out", default="data/flood", help="output folder")
    p.add_argument("--event", default="unnamed", help="label used in output filenames")
    p.add_argument("--vv-band", type=int, default=1, help="1-based VV band index")
    p.add_argument("--mode", choices=["single", "change"], default="single",
                   help="single: Otsu on post VV dB. change: Otsu on post minus pre dB")
    p.add_argument("--filter", choices=["lee", "boxcar", "none"], default="lee")
    p.add_argument("--window", type=int, default=5, help="filter window, odd")
    p.add_argument("--bins", type=int, default=256, help="histogram bins for Otsu")
    p.add_argument("--tiles", type=int, default=1,
                   help="N for an NxN bimodal tile search. 1 disables it. Try 10 "
                        "when separability is low")
    p.add_argument("--tile-min-std", type=float, default=1.5,
                   help="minimum dB standard deviation for a tile to count as bimodal")
    p.add_argument("--threshold", type=float, default=None,
                   help="set the dB threshold by hand and skip Otsu entirely")
    p.add_argument("--search-min", type=float, default=None, dest="search_min",
                   help="lowest dB Otsu may choose. Default -30 single, -20 change")
    p.add_argument("--search-max", type=float, default=None, dest="search_max",
                   help="highest dB Otsu may choose. Default -10 single, -2 change. "
                        "This is what stops Otsu splitting the land distribution")
    p.add_argument("--no-search-limits", action="store_true", dest="no_search_limits",
                   help="let Otsu search the whole histogram. Only for diagnosis")
    p.add_argument("--water", help="JRC occurrence GeoTIFF, permanent water")
    p.add_argument("--water-threshold", type=float, default=50.0,
                   help="occurrence percent at or above which a cell is permanent water")
    p.add_argument("--mask", help="distortion raster from 03_terrain_strata.py")
    p.add_argument("--mask-band", type=int, default=1)
    p.add_argument("--mask-values", default="1,2",
                   help="comma-separated codes in --mask that mean layover or shadow")
    args = p.parse_args()

    if args.selftest:
        return selftest()
    if not args.post:
        p.error("--post is required unless you are running --selftest")
    if args.window % 2 == 0:
        p.error("--window must be odd")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
