r"""
07_dem_comparison.py
====================
Compare the digital elevation models over the corridor, and report how much
the choice of DEM changes what the project can claim.

This is one of the three stated contributions: an honest DEM comparison for
hydraulic modelling in a confined Himalayan valley, where the resulting
uncertainty is reported rather than hidden.

WHAT IT DOES

  1. Puts every DEM on the reference DEM's grid, by nearest-neighbour
     reprojection, so no interpolation invents detail that is not there.
  2. Masks to the corridor. Nothing outside it is compared.
  3. For every pair, reports the difference: mean bias, median, standard
     deviation, robust spread as 1.4826 times the median absolute deviation,
     RMSE, and the 5th and 95th percentiles.
  4. Splits every one of those statistics by reach, gorge above Devprayag
     against open valley below it, because that split is how the whole project
     reports, and by slope band, because DEM error in mountains is a function
     of slope and quoting one corridor-wide number hides that.
  5. Draws the long profile of each DEM down the centreline. In a confined
     valley this is the figure that decides whether a DEM is usable for
     hydraulics: a DEM whose profile climbs is a DEM that will not route water.

HOW TO RUN

  conda activate fyp
  cd "<project folder>\\04 Code\\FYP_code"

  python 07_dem_comparison.py --selftest

  python 07_dem_comparison.py ^
    --dem "GLO30=..\\..\\05 Data\\dem\\DEM_COP_GLO30_30m.tif" ^
    --dem "CartoDEM=..\\..\\05 Data\\dem\\cartodem\\...\\cartodem.tif" ^
    --dem "SRTM=..\\..\\05 Data\\dem\\DEM_SRTM_30m.tif" ^
    --reference GLO30 ^
    --corridor "data\\corridor.geojson" ^
    --centreline "..\\..\\05 Data\\boundaries\\corridor_centreline.geojson" ^
    --out "..\\..\\06 Outputs\\dem_comparison"

  The first --dem given is the reference unless --reference names another.

OUTPUTS

  DEM_pairwise_stats.csv     every statistic, whole corridor and by stratum
  DEM_diff_<A>_minus_<B>.tif difference rasters, corridor-masked
  DEM_long_profile.csv       chainage against elevation for every DEM
  DEM_long_profile.png       the figure
  DEM_comparison_report.md   the numbers written out in sentences
"""
import argparse
import os
import sys

import numpy as np

DEVPRAYAG_LON = 78.5981
SLOPE_BINS = [0, 10, 20, 30, 90]
SLOPE_LABELS = ["0-10 deg", "10-20 deg", "20-30 deg", "over 30 deg"]
NODATA = -9999.0


def say(m=""):
    print(m, flush=True)


def rule(t):
    say(); say("-" * 72); say(t); say("-" * 72)


# --------------------------------------------------------------- statistics
def diff_stats(d):
    """Every number this project quotes about a difference field, in one place."""
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {k: float("nan") for k in
                ("n", "mean", "median", "std", "nmad", "rmse", "p5", "p95",
                 "min", "max")}
    med = float(np.median(d))
    nmad = 1.4826 * float(np.median(np.abs(d - med)))
    return {
        "n": int(d.size),
        "mean": float(np.mean(d)),
        "median": med,
        "std": float(np.std(d)),
        "nmad": nmad,
        "rmse": float(np.sqrt(np.mean(d ** 2))),
        "p5": float(np.percentile(d, 5)),
        "p95": float(np.percentile(d, 95)),
        "min": float(np.min(d)),
        "max": float(np.max(d)),
    }


def slope_deg(dem, xres, yres):
    gy, gx = np.gradient(dem.astype("float64"), abs(yres), abs(xres))
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def classify(a, edges, labels):
    out = np.full(a.shape, "", dtype=object)
    for i, lab in enumerate(labels):
        out[(a >= edges[i]) & (a < edges[i + 1])] = lab
    return out


# ----------------------------------------------------------------- selftest
def selftest():
    rule("07_dem_comparison self-test")
    ok = True

    def check(label, cond, note=""):
        nonlocal ok
        say("  %-5s %-46s %s" % ("PASS" if cond else "FAIL", label, note))
        ok = ok and cond

    check("python >= 3.9", sys.version_info >= (3, 9),
          "%d.%d.%d" % sys.version_info[:3])
    for mod in ("numpy", "rasterio", "geopandas", "shapely", "matplotlib"):
        try:
            __import__(mod)
            check("import " + mod, True)
        except Exception as e:
            check("import " + mod, False, str(e)[:40])

    # a difference field whose answers are known before the code runs
    rng = np.random.default_rng(0)
    d = rng.normal(2.0, 3.0, 200000)
    s = diff_stats(d)
    check("mean bias recovered", abs(s["mean"] - 2.0) < 0.05, "%.3f" % s["mean"])
    check("standard deviation recovered", abs(s["std"] - 3.0) < 0.05, "%.3f" % s["std"])
    check("NMAD agrees with std on a clean normal",
          abs(s["nmad"] - 3.0) < 0.1, "%.3f" % s["nmad"])

    # the point of NMAD: it must survive outliers that wreck std
    d2 = d.copy()
    d2[:2000] = 500.0
    s2 = diff_stats(d2)
    check("std is wrecked by 1 % gross outliers", s2["std"] > 20, "%.1f" % s2["std"])
    check("NMAD is not", abs(s2["nmad"] - 3.0) < 0.2, "%.3f" % s2["nmad"])
    check("RMSE is larger than the bias", s2["rmse"] > abs(s2["mean"]))

    # empty input must not raise
    e = diff_stats(np.array([np.nan, np.nan]))
    check("an empty difference returns NaN, not an exception",
          e["n"] == 0 or np.isnan(e["mean"]))

    # slope on a known plane: 30 m cells, 1 m rise per cell, is 1.9091 degrees
    plane = np.tile(np.arange(50, dtype="float64"), (50, 1))
    sl = slope_deg(plane, 30.0, 30.0)
    want = np.degrees(np.arctan(1.0 / 30.0))
    check("slope of a known plane", abs(float(np.median(sl)) - want) < 1e-6,
          "%.4f deg" % float(np.median(sl)))

    cls = classify(np.array([5.0, 15.0, 25.0, 45.0]), SLOPE_BINS, SLOPE_LABELS)
    check("slope binning", list(cls) == SLOPE_LABELS)

    say()
    say("  %s" % ("all checks passed" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


# --------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", action="append", default=[],
                    help="NAME=path. Repeat for each DEM.")
    ap.add_argument("--reference", help="which NAME is the reference grid. "
                                        "Default: the first --dem given.")
    ap.add_argument("--corridor", help="corridor polygon. Strongly recommended.")
    ap.add_argument("--centreline", help="centreline, for the long profile")
    ap.add_argument("--step", type=float, default=100.0,
                    help="long-profile sampling interval in metres, default 100")
    ap.add_argument("--out", default="../../06 Outputs/dem_comparison")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()
    if len(a.dem) < 2:
        ap.error("give at least two --dem NAME=path arguments")
    if selftest() != 0:
        say(); say("self-test failed. Nothing was written.")
        return 1

    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.features import rasterize
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = []
    for spec in a.dem:
        if "=" not in spec:
            ap.error("--dem must look like NAME=path, got %r" % spec)
        name, path = spec.split("=", 1)
        pairs.append((name.strip(), path.strip()))
    names = [n for n, _ in pairs]
    ref_name = a.reference or names[0]
    if ref_name not in names:
        ap.error("--reference %s is not one of %s" % (ref_name, names))
    ref_path = dict(pairs)[ref_name]

    os.makedirs(a.out, exist_ok=True)

    rule("1. reference grid")
    with rasterio.open(ref_path) as src:
        ref = src.read(1).astype("float64")
        if src.nodata is not None:
            ref[ref == src.nodata] = np.nan
        profile = src.profile
        transform = src.transform
        crs = src.crs
        xres, yres = src.res
        shape = ref.shape
    say("  %s  %s  %d x %d at %.1f m  %s"
        % (ref_name, os.path.basename(ref_path), shape[0], shape[1], abs(xres), crs))

    rule("2. loading and regridding")
    grids = {ref_name: ref}
    for name, path in pairs:
        if name == ref_name:
            continue
        with rasterio.open(path) as src:
            arr = np.full(shape, np.nan, dtype="float64")
            reproject(source=rasterio.band(src, 1), destination=arr,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs=crs,
                      src_nodata=src.nodata, dst_nodata=np.nan,
                      resampling=Resampling.nearest)
        grids[name] = arr
        say("  %-10s %s" % (name, os.path.basename(path)))
    say("  nearest neighbour on purpose: bilinear invents detail a 30 m product "
        "does not have")

    rule("3. corridor mask")
    valid = np.ones(shape, dtype=bool)
    for n_, g in grids.items():
        valid &= np.isfinite(g)
    say("  cells with data in every DEM: %.1f %%" % (100 * valid.mean()))
    if a.corridor:
        cor = gpd.read_file(a.corridor)
        if crs is not None and cor.crs is not None and cor.crs != crs:
            cor = cor.to_crs(crs)
        inside = rasterize([(g, 1) for g in cor.geometry], out_shape=shape,
                           transform=transform, fill=0, dtype="uint8").astype(bool)
        valid &= inside
        say("  inside the corridor as well: %.1f %%" % (100 * valid.mean()))
    else:
        say("  WARNING: no --corridor. Statistics cover the whole raster.")
    n_valid = int(valid.sum())
    say("  %d cells, %.1f km2"
        % (n_valid, n_valid * abs(xres) * abs(yres) / 1e6))
    if n_valid == 0:
        say("  nothing to compare."); return 1

    # strata for the split
    sl = slope_deg(np.where(valid, grids[ref_name], np.nan), xres, yres)
    slope_cls = classify(sl, SLOPE_BINS, SLOPE_LABELS)
    from rasterio.transform import xy as rxy
    from rasterio.warp import transform as warp_transform
    rr, cc = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    xs = np.asarray(rxy(transform, rr, cc)[0]).reshape(shape)
    if crs and crs.to_epsg() != 4326:
        split_x = warp_transform("EPSG:4326", crs, [DEVPRAYAG_LON], [30.15])[0][0]
    else:
        split_x = DEVPRAYAG_LON
    reach_cls = np.where(xs >= split_x, "gorge above Devprayag",
                         "open valley below Devprayag").astype(object)

    rule("4. pairwise differences")
    rows = []
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            d = np.where(valid, grids[na] - grids[nb], np.nan)
            out_tif = os.path.join(a.out, "DEM_diff_%s_minus_%s.tif" % (na, nb))
            prof = dict(profile, dtype="float32", count=1, compress="lzw",
                        nodata=NODATA)
            with rasterio.open(out_tif, "w", **prof) as dst:
                dst.write(np.where(np.isfinite(d), d, NODATA).astype("float32"), 1)

            s = diff_stats(d)
            rows.append(dict(pair="%s - %s" % (na, nb), stratum="whole corridor",
                             **s))
            say("  %-22s mean %+7.2f  median %+7.2f  NMAD %6.2f  RMSE %6.2f  "
                "p5 %+7.2f  p95 %+7.2f m"
                % ("%s - %s" % (na, nb), s["mean"], s["median"], s["nmad"],
                   s["rmse"], s["p5"], s["p95"]))

            for lab in ("gorge above Devprayag", "open valley below Devprayag"):
                m = valid & (reach_cls == lab)
                rows.append(dict(pair="%s - %s" % (na, nb), stratum=lab,
                                 **diff_stats(np.where(m, d, np.nan))))
            for lab in SLOPE_LABELS:
                m = valid & (slope_cls == lab)
                rows.append(dict(pair="%s - %s" % (na, nb), stratum="slope " + lab,
                                 **diff_stats(np.where(m, d, np.nan))))

    import csv
    csv_path = os.path.join(a.out, "DEM_pairwise_stats.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})
    say("  wrote %s" % csv_path)

    # ------------------------------------------------------------ profile
    prof_rows = []
    if a.centreline:
        rule("5. long profile down the centreline")
        line = gpd.read_file(a.centreline)
        if crs is not None and line.crs is not None and line.crs != crs:
            line = line.to_crs(crs)
        geom = line.geometry.iloc[0]
        if geom.geom_type == "MultiLineString":
            from shapely.ops import linemerge
            geom = linemerge(geom)
        total = geom.length
        chain = np.arange(0, total, a.step)
        say("  %.2f km, sampled every %.0f m, %d points"
            % (total / 1000.0, a.step, len(chain)))
        pts = [geom.interpolate(float(c)) for c in chain]
        inv = ~transform
        prof = {}
        for name, g in grids.items():
            vals = []
            for p in pts:
                col, row = inv * (p.x, p.y)
                r_, c_ = int(row), int(col)
                if 0 <= r_ < shape[0] and 0 <= c_ < shape[1]:
                    vals.append(g[r_, c_])
                else:
                    vals.append(np.nan)
            prof[name] = np.array(vals, dtype="float64")

        for i, c in enumerate(chain):
            row = {"chainage_m": round(float(c), 1)}
            for name in names:
                v = prof[name][i]
                row[name + "_m"] = None if not np.isfinite(v) else round(float(v), 2)
            prof_rows.append(row)
        pcsv = os.path.join(a.out, "DEM_long_profile.csv")
        with open(pcsv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(prof_rows[0].keys()))
            w.writeheader(); w.writerows(prof_rows)
        say("  wrote %s" % pcsv)

        # the number that matters for hydraulics
        for name in names:
            z = prof[name]
            z = z[np.isfinite(z)]
            if z.size < 2:
                continue
            runmin = np.minimum.accumulate(z)
            rises = float(np.mean(z > runmin + 5.0)) * 100
            say("  %-10s drop %7.1f m   rises above the running minimum by "
                "more than 5 m at %5.2f %% of points" % (name, z[0] - z[-1], rises))

        fig, ax = plt.subplots(figsize=(11, 4.2))
        for name in names:
            ax.plot(chain / 1000.0, prof[name], lw=1.1, label=name)
        ax.set_xlabel("chainage from the upstream anchor, km")
        ax.set_ylabel("elevation, m")
        ax.set_title("Long profile down the corridor centreline")
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout()
        ppng = os.path.join(a.out, "DEM_long_profile.png")
        fig.savefig(ppng, dpi=150); plt.close(fig)
        say("  wrote %s" % ppng)

    # ------------------------------------------------------------- report
    rule("6. report")
    rep = os.path.join(a.out, "DEM_comparison_report.md")
    whole = [r for r in rows if r["stratum"] == "whole corridor"]

    def f2(v):
        """A stratum with no cells in it prints as a dash, not as a crash."""
        return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) \
            else "%+.2f" % v

    def u2(v):
        return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) \
            else "%.2f" % v

    def i0(v):
        return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) \
            else "%d" % int(v)

    with open(rep, "w", encoding="utf-8") as f:
        f.write("# DEM comparison over the corridor\n\n")
        f.write("Reference grid: **%s**, %d by %d at %.0f m, %s.\n\n"
                % (ref_name, shape[0], shape[1], abs(xres), crs))
        f.write("Compared over %d cells, %.1f km2, where every DEM has data%s.\n\n"
                % (n_valid, n_valid * abs(xres) * abs(yres) / 1e6,
                   " inside the corridor" if a.corridor else ""))
        f.write("Every DEM was put on the reference grid by **nearest "
                "neighbour**. Bilinear resampling would invent detail a 30 m "
                "product does not have.\n\n")
        f.write("## Whole corridor\n\n")
        f.write("| pair | mean bias m | median m | NMAD m | RMSE m | p5 m | p95 m |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in whole:
            f.write("| %s | %s | %s | %s | %s | %s | %s |\n"
                    % (r["pair"], f2(r["mean"]), f2(r["median"]), u2(r["nmad"]),
                       u2(r["rmse"]), f2(r["p5"]), f2(r["p95"])))
        f.write("\n**NMAD, not standard deviation, is the number to quote.** "
                "It is 1.4826 times the median absolute deviation, and unlike "
                "the standard deviation it is not moved by the handful of gross "
                "outliers every DEM difference contains at cliff edges.\n\n")
        f.write("## By reach\n\n")
        f.write("| pair | reach | mean bias m | NMAD m | RMSE m |\n|---|---|---|---|---|\n")
        for r in rows:
            if "Devprayag" in r["stratum"]:
                f.write("| %s | %s | %s | %s | %s |\n"
                        % (r["pair"], r["stratum"], f2(r["mean"]),
                           u2(r["nmad"]), u2(r["rmse"])))
        f.write("\n## By slope\n\n")
        f.write("| pair | slope | mean bias m | NMAD m | RMSE m | n |\n|---|---|---|---|---|---|\n")
        for r in rows:
            if r["stratum"].startswith("slope "):
                f.write("| %s | %s | %s | %s | %s | %s |\n"
                        % (r["pair"], r["stratum"].replace("slope ", ""),
                           f2(r["mean"]), u2(r["nmad"]), u2(r["rmse"]),
                           i0(r["n"])))
        f.write("\nIf NMAD grows with slope, and in mountains it will, then a "
                "single corridor-wide accuracy figure is not defensible and the "
                "slope table is the honest version.\n")
        if prof_rows:
            f.write("\n## Long profile\n\n")
            f.write("`DEM_long_profile.png` and `.csv`. A DEM whose profile "
                    "climbs downstream will not route water, whatever its "
                    "reported vertical accuracy. That is the test that decides "
                    "which DEM the Semester 2 hydraulic model is built on.\n")
    say("  wrote %s" % rep)
    say(); say("  done. read %s" % rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
