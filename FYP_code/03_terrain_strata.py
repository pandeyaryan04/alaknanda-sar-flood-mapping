r"""
03_terrain_strata.py
====================
Build the terrain strata that every accuracy figure in this project is
reported against.

WHY THIS EXISTS
---------------
A single F1 score for a 100 km mountain corridor hides the thing the project is
actually about. Radar looks sideways, so in a confined valley one wall is
compressed into layover and the other can fall into shadow, and a method that
scores well on the open reach below Devprayag can be close to useless in the
gorge above it. Reporting one averaged number would measure the landscape, not
the method.

The nearest published work that applies a foundation model to water mapping in
the Himalaya (Jiang et al. 2025) reports one aggregate F1 and mIoU per dataset,
with no breakdown by terrain and no radiometric terrain correction. This script
is what makes the stratified reporting possible.

WHAT IT COMPUTES
----------------
  slope, aspect            from the DEM
  local incidence angle    the angle between the radar beam and the surface
                           normal, per pixel. This is the radiometric quantity,
                           the one that decides how bright a slope is
  range slope              the signed slope in the look direction, which is the
                           geometric quantity the distortion criteria use
  layover / shadow mask    layover where the range slope reaches the incidence
                           angle, shadow where it reaches minus the grazing angle
  HAND                     height above nearest drainage
  valley width             local width of the low-HAND corridor
  reach                    gorge above Devprayag, open valley below

Then it writes one categorical raster per stratum plus a CSV of pixel counts,
so 04_stratified_metrics.py can report accuracy inside each one.

INPUTS
------
  A DEM GeoTIFF covering the corridor. Copernicus GLO-30 is the primary;
  SRTM and CartoDEM are the comparison. Pass whichever with --dem.

  The radar geometry. Sentinel-1 GRD in Earth Engine carries an `angle` band
  holding the ellipsoidal incidence angle per pixel. Pass the exported S1
  GeoTIFF with --angle; the angle band is found automatically and resampled
  onto the DEM grid, so the two files do NOT have to be at the same
  resolution. Sentinel-1 is 10 m and the DEM is 30 m, and that is correct:
  terrain derivatives belong at the DEM's own resolution.

  Without --angle the script falls back to a constant nominal incidence angle,
  which is a documented approximation and is recorded in the output metadata.

USAGE
-----
  python 03_terrain_strata.py --dem "..\\..\\05 Data\\dem\\DEM_COP_GLO30_30m.tif" \
                              --pass ASCENDING --out data/strata_2023

USAGE WITH THE ANGLE BAND
-------------------------
  python 03_terrain_strata.py ^
    --dem   "..\..\05 Data\dem\DEM_COP_GLO30_30m.tif" ^
    --angle "..\..\05 Data\sentinel1\S1_2023_post_20230816_orb129.tif" ^
    --pass ASCENDING --out data/strata_2023

01_export_s1.js selects ['VV','VH','angle'], so the angle is band 3 of every S1
file. The script finds it by band description, falls back to the last band, and
warns if the values are outside the 29 to 46 degree range Sentinel-1 IW uses.

================================================================================
HOW TO RUN THIS  (copy one line, paste into Anaconda Prompt, press Enter)
================================================================================
Windows Command Prompt uses ^ to continue a line, NOT \ . Easiest is to not
continue lines at all. Do these two first, every time:

    conda activate fyp
    cd /d "<project folder>\04 Code\FYP_code"

Check the radar geometry is still correct (one second, no data needed):

    python 03_terrain_strata.py --selftest

Build the 2023 strata:

    python 03_terrain_strata.py --dem "..\..\05 Data\dem\DEM_COP_GLO30_30m.tif" --angle "..\..\05 Data\sentinel1\S1_2023_post_20230816_orb129.tif" --water "..\..\05 Data\dem\WATER_JRC_occurrence_30m.tif" --pass ASCENDING --out data/strata_2023

Build the 2021 strata (different orbit, opposite pass):

    python 03_terrain_strata.py --dem "..\..\05 Data\dem\DEM_COP_GLO30_30m.tif" --angle "..\..\05 Data\sentinel1\S1_2021_post_20211021_orb63.tif" --water "..\..\05 Data\dem\WATER_JRC_occurrence_30m.tif" --pass DESCENDING --out data/strata_2021

Three mistakes that cost you an hour last time:
  * run from FYP_code, not from "04 Code"
  * single backslashes in paths, not double
  * no \ at the end of lines. That is Linux. Windows ignores it and breaks.
================================================================================
"""

__stage__ = "Terrain stratification for accuracy reporting"
__reads__ = ["DEM GeoTIFF", "Sentinel-1 incidence angle band (optional)"]
__writes__ = ["strata_*.tif", "strata_summary.csv", "strata_meta.json"]

# Run `python 03_terrain_strata.py --selftest` after touching the geometry.

import argparse
import json
import os
import sys

import numpy as np

try:
    import rasterio
    from rasterio.transform import xy
    from scipy import ndimage
except ImportError as e:
    sys.exit(f"\nMissing package: {e}.\nActivate the environment first:  conda activate fyp\n")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEVPRAYAG_LON = 78.5981          # the reach split
NOMINAL_INCIDENCE_DEG = 38.5     # Sentinel-1 IW mid-swath, used only as fallback

# Ground-to-satellite azimuth, degrees clockwise from north.
# Sentinel-1 is right-looking. Ascending heading is about -13 degrees, so the
# beam points about 77 degrees and the satellite sits at about 257 from the
# ground. Descending is the mirror. These are nominal; override with --azimuth
# if you read the real platform heading out of the scene metadata.
SAT_AZIMUTH = {"ASCENDING": 257.0, "DESCENDING": 103.0}

STREAM_ACCUM_CELLS = 500         # cells draining in before a pixel counts as channel
LOW_HAND_M = 25.0                # what counts as valley floor when measuring width

BINS = {
    "incidence": ([0, 30, 40, 50, 181], ["lt30", "30_40", "40_50", "gt50"]),
    "hand":      ([0, 5, 15, 50, 1e9],  ["lt5m", "5_15m", "15_50m", "gt50m"]),
    "width":     ([0, 100, 300, 1000, 1e9], ["lt100m", "100_300m", "300_1000m", "gt1000m"]),
}

# ---------------------------------------------------------------------------


def read_angle_onto(path, shape, transform, crs, band=None):
    """Read the Sentinel-1 incidence angle and put it on the DEM grid.

    The DEM is exported at 30 m and Sentinel-1 at 10 m, so the two rasters are
    never on the same grid and there is no reason they should be. Incidence
    angle varies smoothly across the swath, a degree or so over kilometres, so
    resampling it is not an approximation worth worrying about. Slope, HAND and
    valley width all have to be computed at the DEM's own resolution, because
    terrain derivatives from a 30 m DEM resampled to 10 m are invented detail.

    Band selection: Sentinel-1 GRD exported from Earth Engine with
    select(['VV','VH','angle']) has the angle in band 3. If the file carries
    band descriptions, the one named 'angle' is used. Otherwise the last band
    is used, and a single-band file uses band 1. Override with --angle-band.
    """
    from rasterio.warp import reproject, Resampling

    with rasterio.open(path) as src:
        if band is not None:
            b = band
        else:
            descs = [d.lower() if d else "" for d in (src.descriptions or [])]
            if "angle" in descs:
                b = descs.index("angle") + 1
            else:
                b = src.count            # last band, which is where GEE puts it
        if b < 1 or b > src.count:
            sys.exit(f"\n{os.path.basename(path)} has {src.count} band(s); "
                     f"band {b} does not exist. Set --angle-band.\n")
        out = np.empty(shape, dtype="float64")
        reproject(source=rasterio.band(src, b),
                  destination=out,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform, dst_crs=crs,
                  resampling=Resampling.bilinear,
                  src_nodata=src.nodata, dst_nodata=np.nan)

    finite = np.isfinite(out)
    if not finite.any():
        sys.exit(f"\nBand {b} of {os.path.basename(path)} is empty over the DEM "
                 "extent. Wrong band, or the two rasters do not overlap.\n")
    lo, hi = np.nanmin(out), np.nanmax(out)
    if not (15.0 <= lo <= 50.0 and 15.0 <= hi <= 50.0):
        print(f"  WARNING: band {b} ranges {lo:.1f} to {hi:.1f}. Sentinel-1 IW "
              f"incidence angles sit between about 29 and 46 degrees. This is "
              f"probably the wrong band. Set --angle-band and re-run.")
    # fill any edge gaps with the scene median rather than leaving NaN
    out[~finite] = np.nanmedian(out)
    return out, b


def slope_aspect(dem, xres, yres):
    """Horn's method. Returns slope and aspect in radians.

    Aspect is the downslope azimuth, clockwise from north.
    """
    dzdx = ndimage.sobel(dem, axis=1) / (8.0 * xres)
    dzdy = ndimage.sobel(dem, axis=0) / (8.0 * yres)
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(-dzdx, dzdy)          # 0 = north, increasing clockwise
    aspect = np.mod(aspect, 2 * np.pi)
    return slope, aspect


def local_incidence(slope, aspect, theta_deg, sat_azimuth_deg):
    """Angle between the radar beam and the local surface normal, in degrees.

    cos(theta_loc) = cos(s)cos(t) + sin(s)sin(t)cos(a - phi)

    where s is slope, a is the downslope aspect, t is the ellipsoidal incidence
    angle and phi is the ground-to-satellite azimuth, both clockwise from north.

    The surface normal tilts towards the downslope direction, which is what
    fixes the sign of the second term. Check it against the two limiting cases:
    a slope descending straight towards the satellite (a = phi) gives
    theta_loc = t - s, so it goes into layover once the slope is steeper than
    the incidence angle; a slope descending directly away (a = phi + 180) gives
    theta_loc = t + s, so it falls into shadow once t + s reaches 90.

    This is the radiometric quantity: it is what drives how bright a slope is,
    and it is the one to stratify accuracy against. It cannot go negative, so
    it is NOT the right thing to test for layover. Use range_slope for that.
    """
    t = np.radians(theta_deg)
    phi = np.radians(sat_azimuth_deg)
    cos_loc = (np.cos(slope) * np.cos(t)
               + np.sin(slope) * np.sin(t) * np.cos(aspect - phi))
    return np.degrees(np.arccos(np.clip(cos_loc, -1.0, 1.0)))


def range_slope(slope, aspect, sat_azimuth_deg):
    """Signed terrain slope in the range direction, in degrees.

    tan(s_r) = tan(s) * cos(a - phi)

    Positive where the ground descends towards the sensor, negative where it
    descends away. This is the projected slope the geometric distortion
    criteria are written in terms of, and unlike the true local incidence
    angle it carries a sign.
    """
    phi = np.radians(sat_azimuth_deg)
    return np.degrees(np.arctan(np.tan(slope) * np.cos(aspect - phi)))


def layover_shadow(s_range_deg, theta_deg):
    """0 = clean, 1 = layover, 2 = shadow.

    The standard geometric criteria for a side-looking sensor:

        layover   s_r >= t              the slope out-runs the beam, so the
                                        top of it returns before the bottom
        shadow    s_r <= -(90 - t)      the back slope is steeper than the
                                        grazing angle, so nothing is lit

    Written this way both tests are exact at the limiting cases, which is what
    the smoke test at the bottom of this file checks.
    """
    m = np.zeros(s_range_deg.shape, dtype=np.uint8)
    m[s_range_deg >= theta_deg] = 1
    m[s_range_deg <= -(90.0 - theta_deg)] = 2
    return m


def d8_flowdir(dem):
    """D8 pointer as an index into a flattened array. -1 means a sink or edge."""
    rows, cols = dem.shape
    padded = np.pad(dem, 1, mode="edge")
    best_drop = np.zeros(dem.shape)
    target = np.full(dem.shape, -1, dtype=np.int64)
    # 8 neighbours, with diagonal distance weighting
    offsets = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
               (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]
    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    for dr, dc, dist in offsets:
        nb = padded[1 + dr: 1 + dr + rows, 1 + dc: 1 + dc + cols]
        drop = (dem - nb) / dist
        nr, nc = rr + dr, cc + dc
        inside = (nr >= 0) & (nr < rows) & (nc >= 0) & (nc < cols)
        better = (drop > best_drop) & inside
        best_drop = np.where(better, drop, best_drop)
        target = np.where(better, (nr * cols + nc), target)
    return target


def hand(dem, flowdir, stream_mask):
    """Height above nearest drainage.

    Walks each cell downstream to the first stream cell and takes the elevation
    difference. Memoised, so it is linear in the number of cells.
    """
    rows, cols = dem.shape
    n = rows * cols
    flat_dem = dem.ravel()
    flat_dir = flowdir.ravel()
    flat_stream = stream_mask.ravel()
    outlet = np.full(n, -1, dtype=np.int64)
    outlet[flat_stream] = np.nonzero(flat_stream)[0]

    for start in range(n):
        if outlet[start] != -1:
            continue
        path = []
        cur = start
        while cur != -1 and outlet[cur] == -1:
            path.append(cur)
            cur = flat_dir[cur]
            if len(path) > n:            # cycle guard
                cur = -1
                break
        if cur != -1:
            resolved = outlet[cur]
        elif path:
            # The trace ran off the edge of the clipped corridor without meeting
            # a mapped stream. That is normal on a buffered strip rather than a
            # whole catchment: the Alaknanda's own accumulation restarts at zero
            # where the clip begins. Use the last cell on the path, which is the
            # local drainage exit, so HAND stays defined instead of going NaN.
            resolved = path[-1]
        else:
            resolved = -1
        for p in path:
            outlet[p] = resolved

    h = np.full(n, np.nan)
    known = outlet >= 0
    h[known] = flat_dem[known] - flat_dem[outlet[known]]
    return np.clip(h.reshape(rows, cols), 0, None)


def valley_width(hand_grid, xres, low_hand=LOW_HAND_M):
    """Local width of the low-HAND corridor, in metres.

    Approximated as twice the distance from each valley-floor pixel to the
    nearest non-valley pixel. That is exact for a straight channel of constant
    width and degrades gracefully at bends and confluences. It is an
    approximation and is labelled as one wherever it is reported.
    """
    floor = hand_grid < low_hand
    dist = ndimage.distance_transform_edt(floor, sampling=abs(xres))
    return 2.0 * dist


def classify(arr, edges, labels):
    out = np.full(arr.shape, "", dtype=object)
    for i, lab in enumerate(labels):
        sel = (arr >= edges[i]) & (arr < edges[i + 1])
        out[sel] = lab
    return out


def selftest():
    """Check the radar geometry against cases that can be worked out by hand.

    Run it with --selftest. It touches no data and takes about a second. If it
    fails, every stratified number downstream of this file is wrong, so run it
    once after any edit to the geometry functions.
    """
    T, PHI = 38.5, 257.0
    cross = np.degrees(np.arccos(np.cos(np.radians(20.0)) * np.cos(np.radians(T))))
    cases = [("descends toward the satellite", PHI, 20.0, T - 20, +20.0),
             ("descends away from it", PHI + 180, 20.0, T + 20, -20.0),
             ("across the beam", PHI + 90, 20.0, cross, 0.0)]
    for name, a, s_deg, exp_tl, exp_sr in cases:
        s = np.radians(np.array([s_deg]))
        asp = np.radians(np.array([a]))
        tl = local_incidence(s, asp, T, PHI)[0]
        sr = range_slope(s, asp, PHI)[0]
        assert abs(tl - exp_tl) < 1e-6, f"{name}: theta_loc {tl} != {exp_tl}"
        assert abs(sr - exp_sr) < 1e-6, f"{name}: s_range {sr} != {exp_sr}"
        print(f"  {name:32s} theta_loc {tl:6.2f}   s_range {sr:6.2f}   ok")

    for s_deg, facing, want in [(38.4, "toward", 0), (38.6, "toward", 1),
                                (51.4, "away", 0), (51.6, "away", 2)]:
        asp = np.radians(np.array([PHI if facing == "toward" else PHI + 180]))
        sr = range_slope(np.radians(np.array([s_deg])), asp, PHI)
        got = int(layover_shadow(sr, T)[0])
        assert got == want, f"slope {s_deg} facing {facing}: got {got}, wanted {want}"
    print("  layover threshold at s_r = t and shadow at s_r = -(90 - t)   ok")

    y, x = np.mgrid[0:120, 0:120].astype(float)
    fracs = []
    for amp in (400.0, 1600.0):
        dem = 2000 + amp * np.abs(np.sin(x / 30.0)) + 0.5 * y
        sl, asp = slope_aspect(dem, 30.0, 30.0)
        ls = layover_shadow(range_slope(sl, asp, PHI), T)
        fracs.append((np.mean(ls == 1), np.mean(ls == 2)))
    assert fracs[1][0] > fracs[0][0] and fracs[1][1] > fracs[0][1], \
        "steeper terrain must produce more layover and more shadow"
    print("  layover and shadow both grow with slope on synthetic terrain   ok")
    print("\nself-test passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="check the radar geometry against hand-worked cases and exit")
    ap.add_argument("--dem", help="DEM GeoTIFF over the corridor")
    ap.add_argument("--angle", help="Sentinel-1 GeoTIFF carrying the 'angle' band. "
                    "It does not need to be on the same grid as the DEM.")
    ap.add_argument("--angle-band", type=int, dest="angle_band",
                    help="which band holds the incidence angle. Auto-detected if omitted.")
    ap.add_argument("--water", help="JRC permanent-water occurrence GeoTIFF. Strongly "
                    "recommended: on a clipped corridor, flow accumulation cannot see "
                    "the river arriving from outside the clip, so the drainage network "
                    "it finds is wrong and HAND with it.")
    ap.add_argument("--water-threshold", type=float, default=50.0, dest="water_threshold",
                    help="JRC occurrence percent above which a cell counts as permanent "
                         "water. Default 50.")
    ap.add_argument("--pass", dest="orbit_pass", choices=["ASCENDING", "DESCENDING"])
    ap.add_argument("--azimuth", type=float,
                    help="override the ground-to-satellite azimuth, degrees from north")
    ap.add_argument("--corridor",
                    help="corridor polygon (GeoJSON/GPKG/SHP). Strongly recommended. "
                         "Without it the strata describe the whole DEM rectangle "
                         "while the flood maps describe the corridor, so the two "
                         "are not on the same ground and every share_pct is wrong.")
    ap.add_argument("--out", default="data/strata")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.dem or not args.orbit_pass:
        ap.error("--dem and --pass are required unless you are running --selftest")

    os.makedirs(args.out, exist_ok=True)

    with rasterio.open(args.dem) as src:
        dem = src.read(1).astype("float64")
        profile = src.profile
        xres, yres = src.res
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
    if nodata is not None:
        dem[dem == nodata] = np.nan

    # `valid` is what decides which cells count. Everything downstream is
    # masked by it: the strata rasters, the summary CSV and every share_pct.
    #
    # Before 9 September 2026 this line read
    #     dem = np.nan_to_num(dem, nan=np.nanmin(dem))
    # which put every nodata cell at the lowest elevation in the raster, so
    # they all landed in the lowest elevation and HAND bands and inflated them.
    valid = np.isfinite(dem)
    nodata_pct = 100.0 * (1.0 - valid.mean())
    if nodata_pct > 0:
        print(f"  {nodata_pct:.2f}% of the DEM is nodata and is excluded from "
              f"every stratum and every statistic")

    if args.corridor:
        import geopandas as gpd
        from rasterio.features import rasterize
        cor = gpd.read_file(args.corridor)
        if crs is not None and cor.crs is not None and cor.crs != crs:
            cor = cor.to_crs(crs)
        inside = rasterize([(g, 1) for g in cor.geometry],
                           out_shape=dem.shape, transform=transform,
                           fill=0, dtype="uint8").astype(bool)
        before = valid.sum()
        valid &= inside
        print(f"  clipped to the corridor: {100.0*valid.sum()/dem.size:.1f}% of the "
              f"raster, {100.0*valid.sum()/max(before,1):.1f}% of the valid DEM")
    else:
        print("  WARNING: no --corridor given. The strata will describe the whole "
              "DEM rectangle while the flood maps describe the corridor. Every "
              "share_pct below is against the wrong denominator.")

    # the flow routing cannot take NaN, so fill for routing only. Nothing is
    # reported from a filled cell, because `valid` excludes it.
    dem = np.where(np.isfinite(dem), dem, np.nanmin(dem))

    print(f"DEM {dem.shape[0]} x {dem.shape[1]} at {xres:.1f} m, "
          f"{np.nanmin(dem[valid]) if valid.any() else float('nan'):.0f} to "
          f"{np.nanmax(dem[valid]) if valid.any() else float('nan'):.0f} m "
          f"over {int(valid.sum()):,} valid cells")

    # --- geometry -----------------------------------------------------------
    slope, aspect = slope_aspect(dem, abs(xres), abs(yres))

    if args.angle:
        theta, band_used = read_angle_onto(args.angle, dem.shape, transform, crs,
                                           args.angle_band)
        angle_source = f"{os.path.basename(args.angle)} band {band_used}, resampled to the DEM grid"
        print(f"  incidence angle from band {band_used} of "
              f"{os.path.basename(args.angle)}: "
              f"{np.nanmin(theta):.1f} to {np.nanmax(theta):.1f} deg")
    else:
        theta = np.full(dem.shape, NOMINAL_INCIDENCE_DEG)
        angle_source = f"constant {NOMINAL_INCIDENCE_DEG} deg (APPROXIMATION)"
        print(f"  no --angle given, using a constant {NOMINAL_INCIDENCE_DEG} deg")

    sat_az = args.azimuth if args.azimuth is not None else SAT_AZIMUTH[args.orbit_pass]
    theta_loc = local_incidence(slope, aspect, theta, sat_az)
    s_range = range_slope(slope, aspect, sat_az)
    ls = layover_shadow(s_range, theta)

    # share_pct is against the valid cells, not the whole raster
    n = int(valid.sum()) or ls.size
    _lsv = ls[valid] if valid.any() else ls
    print(f"  layover {100*np.count_nonzero(_lsv==1)/n:.2f}%   "
          f"shadow {100*np.count_nonzero(_lsv==2)/n:.2f}%   "
          f"clean {100*np.count_nonzero(_lsv==0)/n:.2f}%")

    # --- hydrology ----------------------------------------------------------
    print("  flow directions ...")
    fdir = d8_flowdir(dem)
    print("  drainage and HAND ...")
    # accumulation by descending elevation, which is what makes a single pass valid
    order = np.argsort(-dem.ravel())
    acc = np.ones(dem.size, dtype=np.int64)
    flat_dir = fdir.ravel()
    for idx in order:
        t = flat_dir[idx]
        if t >= 0:
            acc[t] += acc[idx]
    stream = (acc >= STREAM_ACCUM_CELLS).reshape(dem.shape)
    if args.water:
        from rasterio.warp import reproject, Resampling
        with rasterio.open(args.water) as wsrc:
            occ = np.full(dem.shape, np.nan)
            reproject(source=rasterio.band(wsrc, 1), destination=occ,
                      src_transform=wsrc.transform, src_crs=wsrc.crs,
                      dst_transform=transform, dst_crs=crs,
                      resampling=Resampling.bilinear,
                      src_nodata=wsrc.nodata, dst_nodata=np.nan)
        water = np.nan_to_num(occ, nan=0.0) >= args.water_threshold
        print(f"  permanent water from {os.path.basename(args.water)}: "
              f"{100*np.count_nonzero(water)/water.size:.2f}% of cells")
        stream = stream | water
        water_source = f"{os.path.basename(args.water)} at occurrence >= {args.water_threshold}%"
    else:
        water_source = None
        print("  no --water given: drainage from flow accumulation only. "
              "On a clipped corridor this understates the main channel.")
    print(f"  drainage network: {np.count_nonzero(stream):,} cells")
    h = hand(dem, fdir, stream)
    defined = float(np.mean(np.isfinite(h))) * 100
    print(f"  HAND defined for {defined:.1f}% of cells, 0 to {np.nanmax(h):.0f} m")
    if defined < 90:
        print("  WARNING: HAND is undefined over much of the corridor, which will "
              "break the HAND and valley-width strata. Pass --water.")
    w = valley_width(h, abs(xres))

    # --- reach --------------------------------------------------------------
    # The raster is in UTM (EPSG:32644), so xy() returns eastings in metres,
    # not degrees. Comparing those against a longitude put the whole corridor
    # on one side of the split. Project the split point into the raster CRS
    # instead, then compare like with like.
    from rasterio.warp import transform as warp_transform
    rows, cols = dem.shape
    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    xs, _ = xy(transform, rr, cc)
    xs = np.asarray(xs)
    if crs and crs.to_epsg() != 4326:
        split_x = warp_transform("EPSG:4326", crs, [DEVPRAYAG_LON], [30.15])[0][0]
    else:
        split_x = DEVPRAYAG_LON
    reach = np.where(xs >= split_x, "gorge_above_devprayag",
                     "open_valley_below_devprayag").astype(object)
    # xy() returns a flat list, so reach is 1-D over rows*cols
    _flat = valid.ravel()
    reach[~_flat] = ""
    above = (float(np.mean((xs >= split_x)[_flat])) * 100
             if _flat.any() else float("nan"))
    print(f"  reach split at x={split_x:,.0f} in {crs}: "
          f"{above:.1f}% gorge, {100-above:.1f}% open valley")
    if above > 95 or above < 5:
        print("  WARNING: the reach split put almost everything on one side. "
              "Check the corridor extent and DEVPRAYAG_LON.")

    # --- write --------------------------------------------------------------
    NODATA = -9999.0
    prof = dict(profile, dtype="float32", count=1, compress="lzw", nodata=NODATA)
    for name, arr in [("theta_loc", theta_loc), ("slope_deg", np.degrees(slope)),
                      ("range_slope_deg", s_range),
                      ("hand_m", h), ("valley_width_m", w)]:
        out_arr = np.where(valid, np.asarray(arr, dtype="float32"),
                           np.float32(NODATA)).astype("float32")
        with rasterio.open(os.path.join(args.out, f"strata_{name}.tif"), "w", **prof) as dst:
            dst.write(out_arr, 1)
    prof8 = dict(profile, dtype="uint8", count=1, compress="lzw", nodata=255)
    ls_out = np.where(valid, ls, 255).astype("uint8")
    with rasterio.open(os.path.join(args.out, "strata_layover_shadow.tif"), "w", **prof8) as dst:
        dst.write(ls_out, 1)

    inc_cls = classify(theta_loc, *BINS["incidence"])
    hand_cls = classify(h, *BINS["hand"])
    width_cls = classify(w, *BINS["width"])
    ls_cls = np.choose(ls, np.array(["clean", "layover", "shadow"], dtype=object))
    # a cell outside the corridor, or with no DEM under it, belongs to no
    # stratum. The summary loop already skips "".
    for _cls in (inc_cls, hand_cls, width_cls, ls_cls):
        _cls[~valid] = ""

    import csv
    rows_out = []
    for dim, cls in [("incidence", inc_cls), ("hand", hand_cls),
                     ("valley_width", width_cls), ("layover_shadow", ls_cls),
                     ("reach", reach)]:
        vals, counts = np.unique(cls, return_counts=True)
        for v, c in zip(vals, counts):
            if v == "":
                continue
            rows_out.append({"dimension": dim, "stratum": str(v), "pixels": int(c),
                             "area_km2": round(c * abs(xres) * abs(yres) / 1e6, 3),
                             "share_pct": round(100.0 * c / n, 2)})
    csv_path = os.path.join(args.out, "strata_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows_out)

    meta = {
        "dem": os.path.abspath(args.dem),
        "incidence_angle_source": angle_source,
        "orbit_pass": args.orbit_pass,
        "ground_to_satellite_azimuth_deg": sat_az,
        "stream_accumulation_threshold_cells": STREAM_ACCUM_CELLS,
        "valley_floor_hand_threshold_m": LOW_HAND_M,
        "valley_width_method": "twice the distance transform of the low-HAND mask, an approximation",
        "drainage_source": water_source or "flow accumulation only",
        "reach_split_longitude": DEVPRAYAG_LON,
        "corridor": os.path.abspath(args.corridor) if args.corridor else None,
        "valid_cells": int(valid.sum()),
        "valid_area_km2": round(int(valid.sum()) * abs(xres) * abs(yres) / 1e6, 3),
        "nodata_excluded_pct": round(nodata_pct, 3),
        "crs": str(crs),
        "pixel_size_m": [abs(xres), abs(yres)],
        "bins": {k: v[0] for k, v in BINS.items()},
    }
    with open(os.path.join(args.out, "strata_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\nwrote {args.out}/strata_*.tif, strata_summary.csv, strata_meta.json")
    print("Check strata_summary.csv and strata_meta.json.")
    if not args.angle:
        print("\nNOTE: incidence angle was a constant. Add 'angle' to the select "
              "list in 01_export_s1.js and re-run for a real per-pixel value.")


if __name__ == "__main__":
    main()
