"""
t01_centreline.py - derive the corridor centreline.

Replaces the hour of hand-digitising the centreline in QGIS.

Method, in one sentence: a least-cost path from Rudraprayag to Rishikesh
across a cost surface that is cheap inside dry-season water, cheapest along
the middle of it, and expensive up the valley walls; computed a second time
from terrain alone as an independent cross-check.

Outputs are described in the package README. Every acceptance number the
report prints is also in run_manifest.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import common as C
import linelib as L
import route as R

TOOL = "t01_centreline"


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

def dry_scene_pairs(cfg):
    """[(indices_path, bands_path, tag), ...] for the dry-season scenes."""
    s2 = cfg.path("S2_DIR")
    out = []
    for tag in cfg.csv("DRY_SCENES"):
        idx = s2 / f"{tag}_indices_10m.tif"
        ten = s2 / f"{tag}_10m.tif"
        out.append((idx, ten, tag))
    return out


def build_water_mask(cfg, log, man):
    """Union over the dry-season scenes of (MNDWI > t) and (B11 < t) and clear.

    Union rather than intersection, per the 9 September decision: any cell wet
    on any dry date counts. It gives channel continuity and fills each date's
    cloud gaps, and it errs toward including water, which is the safe
    direction when the product is a centreline.
    """
    mndwi_t = cfg.f("MNDWI_WATER")
    swir_t = cfg.f("SWIR_WATER")
    crs = cfg.str("CRS_METRIC")

    union = None
    ref = None
    per_scene = {}

    for idx_p, ten_p, tag in dry_scene_pairs(cfg):
        man.input(idx_p, "dry-season indices")
        man.input(ten_p, "dry-season bands")

        dsi = C.open_checked(idx_p, expect_crs=crs,
                             expect_bands=len(cfg.csv("IDX_BANDS")), log=log)
        dst = C.open_checked(ten_p, expect_crs=crs,
                             expect_bands=len(cfg.csv("S2_BANDS")), log=log)
        try:
            if ref is None:
                ref = {
                    "transform": dsi.transform,
                    "shape": (dsi.height, dsi.width),
                    "res": abs(dsi.transform.a),
                    "crs": str(dsi.crs),
                    "bounds": dsi.bounds,
                }
            if (dsi.height, dsi.width) != ref["shape"]:
                raise C.RasterError(
                    f"{idx_p.name} is {dsi.height}x{dsi.width}, "
                    f"the first dry scene is {ref['shape'][0]}x{ref['shape'][1]}.\n"
                    f"  The dry scenes must share one grid. Re-export the odd "
                    f"one with the same region, scale and crs."
                )

            mndwi = dsi.read(C.band_index(cfg, "IDX_BANDS", "MNDWI")).astype("float32")
            b11 = C.s2_reflectance(dst, cfg, "B11")
            scl = dst.read(C.band_index(cfg, "S2_BANDS", "SCL"))
            _, _, _, clear = C.scl_masks(scl, cfg)

            wet = (mndwi > mndwi_t) & (b11 < swir_t) & clear
            per_scene[tag] = float(wet.mean())
            union = wet if union is None else (union | wet)
            C.say(f"  {tag:32s} wet {100*wet.mean():6.2f} %  clear {100*clear.mean():6.2f} %")
        finally:
            dsi.close()
            dst.close()

    if union is None:
        raise RuntimeError("DRY_SCENES is empty; nothing to build a mask from")

    man.param("mndwi_water", mndwi_t)
    man.param("swir_water", swir_t)
    man.param("dry_scene_wet_fraction", per_scene)
    return union, ref


def clean_mask(water, res_m, min_ha, close_px=2):
    from scipy.ndimage import binary_closing, label

    w = binary_closing(water, structure=np.ones((3, 3), bool),
                       iterations=int(close_px))
    lab, n = label(w, structure=np.ones((3, 3), int))
    if n == 0:
        return w, 0
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    min_cells = max(1, int(min_ha * 1e4 / (res_m * res_m)))
    keep = np.isin(lab, np.nonzero(sizes >= min_cells)[0])
    return keep, int((sizes >= min_cells).sum())


def corridor_mask(cfg, ref, log, man):
    import geopandas as gpd
    from rasterio.features import rasterize

    p = cfg.path("CORRIDOR")
    man.input(p, "corridor polygon")
    g = gpd.read_file(p)
    if g.crs is None:
        g = g.set_crs(cfg.str("CRS_STORE"))
    g = g.to_crs(cfg.str("CRS_METRIC"))
    poly = g.union_all() if hasattr(g, "union_all") else g.unary_union
    m = rasterize(
        [(poly, 1)], out_shape=ref["shape"], transform=ref["transform"],
        fill=0, dtype="uint8",
    ).astype(bool)
    log.info("corridor covers %d cells", int(m.sum()))
    return m, poly


def dem_on_grid(cfg, ref, log, man):
    """Read the routing DEM and warp it onto the Sentinel-2 grid."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    p = cfg.path("DEM_ROUTING")
    man.input(p, "routing DEM")
    if not p.exists():
        raise C.RasterError(f"routing DEM not found: {p}")
    dst = np.full(ref["shape"], np.nan, dtype="float32")
    with rasterio.open(p) as src:
        log.info("routing DEM %s, %s, %d x %d", p.name, src.crs, src.width, src.height)
        C.say(f"  routing DEM {p.name}, {src.crs}, {src.width} x {src.height}")
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=ref["transform"], dst_crs=ref["crs"],
            resampling=Resampling.bilinear,
            src_nodata=src.nodata, dst_nodata=np.nan,
        )
    return dst


# --------------------------------------------------------------------------
# cost surface
# --------------------------------------------------------------------------

def centrality(water, res_m, ref_m):
    """0 at the water's edge, 1 at the middle of a channel ref_m wide or more.

    This is what makes the path a centreline rather than a path that clings to
    one bank, and it is also why a wide reservoir needs no special case: the
    cheapest route through it is its middle.
    """
    from scipy.ndimage import distance_transform_edt

    d = distance_transform_edt(water) * res_m
    return np.clip(d / max(ref_m, res_m), 0.0, 1.0)


def build_costs(cfg, water, dem, inside, res_m):
    land = cfg.f("LAND_COST")
    wat = cfg.f("WATER_COST")
    centre_w = cfg.f("CENTRE_WEIGHT")
    centre_ref = cfg.f("CENTRE_REF_M")
    climb_w = cfg.f("CLIMB_WEIGHT")

    outside = ~inside

    terrain = R.cost_from_terrain(dem, nodata=outside, climb_weight=climb_w)

    c = centrality(water, res_m, centre_ref)
    cost_w = np.where(
        water,
        wat * (1.0 + centre_w * (1.0 - c)),
        land * np.clip(terrain, 1.0, None),
    )
    cost_w[outside] = R.BIG

    return cost_w, terrain


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def selftest(cfg) -> bool:
    ck = C.Checks(f"{TOOL} self-test")
    ck.check("python >= 3.9", sys.version_info >= (3, 9), sys.version.split()[0])
    for mod in ("numpy", "scipy", "skimage", "rasterio", "geopandas",
                "shapely", "pyproj", "matplotlib"):
        try:
            __import__(mod)
            ck.check(f"import {mod}", True)
        except Exception as e:
            ck.check(f"import {mod}", False, type(e).__name__)

    for key in ("CORRIDOR", "S2_DIR", "DEM_ROUTING", "CENTRELINE_OUT",
                "CRS_METRIC", "CRS_STORE", "DRY_SCENES", "S2_BANDS",
                "IDX_BANDS", "ANCHOR_UP", "ANCHOR_DOWN"):
        ck.check(f".env has {key}", cfg.has(key))

    ck.require_file("corridor", cfg.path("CORRIDOR"))
    ck.require_file("routing DEM", cfg.path("DEM_ROUTING"))
    for idx_p, ten_p, tag in dry_scene_pairs(cfg):
        ck.require_file(f"dry {tag} indices", idx_p)
        ck.require_file(f"dry {tag} bands", ten_p)

    # UTM zone sanity: the one that catches you on a new project
    try:
        lon = cfg.lonlat("ANCHOR_UP")[0]
        zone = C.utm_zone_of(lon)
        declared = cfg.str("CRS_METRIC").upper().replace("EPSG:", "")
        want = f"326{zone:02d}"
        ck.check("CRS_METRIC matches the UTM zone of ANCHOR_UP",
                 declared == want, f"anchor zone {zone}, CRS_METRIC {declared}")
    except Exception as e:
        ck.check("UTM zone check", False, str(e))

    return ck.report()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="path to .env")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=None, help="override OUT_DIR")
    a = ap.parse_args(argv)

    cfg = C.Cfg.load(a.config)
    if a.selftest:
        return 0 if selftest(cfg) else 1
    if not selftest(cfg):
        C.say("")
        C.say("self-test failed. Nothing was computed. Fix the FAIL lines above.")
        return 1

    out_dir = Path(a.out or cfg.path("OUT_DIR")) / "centreline"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = C.setup_logger(out_dir, TOOL)
    man = C.Manifest(TOOL, cfg)
    force = a.force or cfg.b("FORCE", "false")

    C.rule("1. dry-season water mask")
    water_raw, ref = build_water_mask(cfg, log, man)
    res = ref["res"]
    water, n_parts = clean_mask(water_raw, res, cfg.f("MIN_WATER_HA"))
    C.say(f"  union {C.km2(water_raw.sum(), res):.2f} km2 -> cleaned "
          f"{C.km2(water.sum(), res):.2f} km2 in {n_parts} parts "
          f"(cells {res:.0f} m)")

    C.rule("2. corridor and terrain")
    inside, poly = corridor_mask(cfg, ref, log, man)
    dem = dem_on_grid(cfg, ref, log, man)
    C.say(f"  corridor {C.km2(inside.sum(), res):.1f} km2")

    water &= inside

    C.rule("3. endpoints")
    from pyproj import Transformer
    tr = Transformer.from_crs(cfg.str("CRS_STORE"), cfg.str("CRS_METRIC"),
                              always_xy=True)
    snap_px = cfg.f("SNAP_RADIUS_M") / res
    seeds = {}
    for name, key in (("up", "ANCHOR_UP"), ("down", "ANCHOR_DOWN")):
        lon, lat = cfg.lonlat(key)
        x, y = tr.transform(lon, lat)
        rc = L.xy_to_rc(x, y, ref["transform"])
        if not (0 <= rc[0] < ref["shape"][0] and 0 <= rc[1] < ref["shape"][1]):
            raise RuntimeError(
                f"{key} at {lon},{lat} falls outside the Sentinel-2 grid. "
                f"Check the anchor, and check CRS_STORE."
            )
        rc2, moved = R.snap_to_water(rc, water, snap_px)
        seeds[name] = rc2
        C.say(f"  {key:12s} {lon:.4f},{lat:.4f}  snapped {moved*res:6.0f} m "
              f"to water" if np.isfinite(moved) else f"  {key} not snapped")
        man.param(f"snap_{name}_m", None if not np.isfinite(moved) else round(moved * res, 1))
        if np.isfinite(moved) and moved * res > cfg.f("SNAP_RADIUS_M"):
            C.say(f"  WARNING  {key} is {moved*res:.0f} m from the nearest water "
                  f"cell, beyond SNAP_RADIUS_M. The route will start on land.")

    C.rule("4. routing")
    cost_w, terrain = build_costs(cfg, water, dem, inside, res)
    path_w = R.route(cost_w, seeds["up"], seeds["down"])
    C.say(f"  water-led route  {len(path_w)} cells")

    cost_t = np.where(inside, terrain, R.BIG)
    path_t = R.route(cost_t, seeds["up"], seeds["down"])
    C.say(f"  terrain-only route {len(path_t)} cells")

    C.rule("5. smoothing")
    raw_xy = L.rc_to_xy(path_w, ref["transform"])
    line = L.simplify(raw_xy, cfg.f("SIMPLIFY_M"))
    line = L.chaikin(line, cfg.i("CHAIKIN_PASSES"))
    line = L.densify(line, cfg.f("VERTEX_MAX_M"))
    ter_xy = L.rc_to_xy(path_t, ref["transform"])
    ter_line = L.densify(L.chaikin(L.simplify(ter_xy, cfg.f("SIMPLIFY_M")), 2),
                         cfg.f("VERTEX_MAX_M"))
    C.say(f"  {len(raw_xy)} raw -> {len(line)} vertices, "
          f"{L.length_m(line)/1000:.2f} km")

    C.rule("6. checks")
    checks = run_checks(cfg, line, ter_line, water, dem, ref, poly, tr, man)

    C.rule("7. writing")
    write_outputs(cfg, out_dir, line, ter_line, water, dem, ref, checks,
                  man, force, poly)

    man.write(out_dir)
    C.say("")
    C.say(f"  done. read {out_dir / 'centreline_report.md'}")
    return 0 if checks["all_pass"] else 2


# --------------------------------------------------------------------------
# checks and reporting
# --------------------------------------------------------------------------

def run_checks(cfg, line, ter_line, water, dem, ref, poly, tr, man):
    res = ref["res"]
    out = {}

    length_km = L.length_m(line) / 1000.0
    lo, hi = cfg.f("LENGTH_MIN_KM"), cfg.f("LENGTH_MAX_KM")
    out["length_km"] = round(length_km, 2)
    out["length_ok"] = bool(lo <= length_km <= hi)

    frac_in = L.inside_fraction(line, poly)
    out["inside_fraction"] = round(float(frac_in), 5)
    out["inside_ok"] = bool(frac_in >= cfg.f("INSIDE_MIN_FRACTION"))

    from shapely.geometry import LineString, Point
    g = LineString(line)
    town_tol = cfg.f("TOWN_TOL_M")
    towns = {}
    for key in cfg.csv("CHECK_TOWNS"):
        if not cfg.has(key):
            continue
        lon, lat = cfg.lonlat(key)
        x, y = tr.transform(lon, lat)
        d = float(g.distance(Point(x, y)))
        towns[key] = round(d, 1)
    out["town_distance_m"] = towns
    out["towns_ok"] = all(d <= town_tol for d in towns.values()) if towns else None

    off, off_pts = L.offsets(line, ter_line, step_m=cfg.f("QC_STEP_M"))
    out["offset_median_m"] = round(float(np.median(off)), 1)
    out["offset_p90_m"] = round(float(np.percentile(off, 90)), 1)
    out["offset_max_m"] = round(float(off.max()), 1)
    out["offset_ok"] = bool(np.median(off) <= cfg.f("OFFSET_MEDIAN_MAX_M"))
    out["_offsets"] = off
    out["_offset_points"] = off_pts

    # elevation profile along the finished line
    import rasterio.transform as rt
    zs = []
    h, w = ref["shape"]
    for x, y in line:
        r, c = rt.rowcol(ref["transform"], x, y)
        zs.append(dem[r, c] if (0 <= r < h and 0 <= c < w) else np.nan)
    zs = np.array(zs, dtype="float64")
    ch = L.chainage(line)
    mono = R.monotonic_report(zs, ch, cfg.f("ELEV_NOISE_M"))
    out["elevation"] = mono
    out["elev_ok"] = bool(
        mono.get("checked") and
        mono.get("fraction_over_noise", 1.0) <= cfg.f("ELEV_BAD_MAX_FRACTION")
    )
    out["_z"] = zs
    out["_chain"] = ch

    wid = L.widths(line, water, ref["transform"], res,
                   max_half_m=cfg.f("MAX_HALF_WIDTH_M"))
    out["_width"] = wid
    finite = wid[np.isfinite(wid)]
    out["width_median_m"] = round(float(np.median(finite)), 1) if finite.size else None
    out["on_water_fraction"] = round(float(np.isfinite(wid).mean()), 4)
    wide = np.isfinite(wid) & (wid > cfg.f("RESERVOIR_WIDTH_M"))
    out["wide_reaches_km"] = _spans(ch, wide)

    out["all_pass"] = all(
        v for k, v in out.items()
        if k.endswith("_ok") and v is not None
    )
    for k, v in out.items():
        if not k.startswith("_"):
            man.param(k, v)
    return out


def _spans(chain_m, flag):
    idx = np.nonzero(flag)[0]
    if idx.size == 0:
        return []
    segs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i != prev + 1:
            segs.append((start, prev))
            start = i
        prev = i
    segs.append((start, prev))
    return [(round(float(chain_m[a] / 1000), 2), round(float(chain_m[b] / 1000), 2))
            for a, b in segs if chain_m[b] - chain_m[a] > 200]


def write_outputs(cfg, out_dir, line, ter_line, water, dem, ref, ck,
                  man, force, poly):
    import pandas as pd

    gj = cfg.path("CENTRELINE_OUT")
    gpkg = Path(str(gj).replace(".geojson", ".gpkg"))
    props = {
        "name": cfg.str("RIVER_NAME", "corridor centreline"),
        "method": "least-cost path, dry-season water mask, gis_tools t01",
        "length_km": ck["length_km"],
        "package_version": C.__version__,
    }
    C.write_line(line, cfg, gj, gpkg, layer="centreline", props=props, force=True)
    man.output(gj); man.output(gpkg)
    C.say(f"  wrote {gj}")
    C.say(f"  wrote {gpkg}")

    if cfg.has("CENTRELINE_COPY"):
        cp = cfg.path("CENTRELINE_COPY")
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(Path(gj).read_text(encoding="utf-8"), encoding="utf-8")
        man.output(cp)
        C.say(f"  wrote {cp}")

    tgj = out_dir / "centreline_terrain_crosscheck.geojson"
    C.write_line(ter_line, cfg, tgj, None, props={"method": "terrain-only"},
                 force=True)
    man.output(tgj)

    n = len(line)
    df = pd.DataFrame({
        "chainage_km": np.round(ck["_chain"] / 1000.0, 4),
        "x_m": [p[0] for p in line],
        "y_m": [p[1] for p in line],
        "elevation_m": np.round(ck["_z"], 2),
        "water_width_m": np.round(ck["_width"], 1),
    })
    off = np.interp(ck["_chain"], L.chainage(ck["_offset_points"]), ck["_offsets"]) \
        if len(ck["_offset_points"]) > 1 else np.full(n, np.nan)
    df["terrain_offset_m"] = np.round(off, 1)
    p = out_dir / "centreline_qc.csv"
    df.to_csv(p, index=False)
    man.output(p)

    _figure(out_dir / "centreline_qc.png", line, ter_line, water, ref, ck, cfg)
    man.output(out_dir / "centreline_qc.png")

    rep = _report_md(cfg, ck, line, out_dir)
    p = out_dir / "centreline_report.md"
    p.write_text(rep, encoding="utf-8")
    man.output(p)
    C.say("")
    for ln in rep.splitlines():
        if ln.startswith(("| ", "**")) or ln.startswith("PASS") or ln.startswith("FAIL"):
            C.say("  " + ln)


def _figure(path, line, ter_line, water, ref, ck, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rasterio.plot import plotting_extent

    ext = plotting_extent(np.zeros(ref["shape"]), ref["transform"])
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 3, height_ratios=[2.0, 1.0, 1.0])

    ax = fig.add_subplot(gs[0, :])
    ax.imshow(water, extent=ext, origin="upper", cmap="Blues", alpha=0.8,
              interpolation="nearest")
    ax.plot([p[0] for p in ter_line], [p[1] for p in ter_line],
            lw=1.0, ls="--", color="0.45", label="terrain-only cross-check")
    ax.plot([p[0] for p in line], [p[1] for p in line],
            lw=1.6, color="crimson", label="centreline")
    ax.set_title(f"centreline, {ck['length_km']:.1f} km "
                 f"(blue = dry-season water union)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)

    axp = fig.add_subplot(gs[1, :])
    axp.plot(ck["_chain"] / 1000.0, ck["_z"], lw=1.2, color="darkgreen")
    axp.set_ylabel("elevation, m")
    axp.set_xlabel("chainage, km")
    axp.set_title("elevation profile: it must fall", fontsize=9)
    axp.grid(alpha=0.3)

    axw = fig.add_subplot(gs[2, 0])
    axw.plot(ck["_chain"] / 1000.0, ck["_width"], lw=1.0, color="steelblue")
    axw.axhline(cfg.f("RESERVOIR_WIDTH_M"), color="crimson", lw=0.8, ls=":")
    axw.set_title("water width, m", fontsize=9)
    axw.set_xlabel("chainage, km")
    axw.grid(alpha=0.3)

    axo = fig.add_subplot(gs[2, 1:])
    if len(ck["_offset_points"]) > 1:
        axo.plot(L.chainage(ck["_offset_points"]) / 1000.0, ck["_offsets"],
                 lw=1.0, color="darkorange")
    axo.axhline(cfg.f("OFFSET_MEDIAN_MAX_M"), color="crimson", lw=0.8, ls=":")
    axo.set_title("offset from the terrain-only route, m", fontsize=9)
    axo.set_xlabel("chainage, km")
    axo.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _report_md(cfg, ck, line, out_dir):
    def flag(ok):
        return "PASS" if ok else "FAIL"

    rows = [
        ("length between "
         f"{cfg.f('LENGTH_MIN_KM'):.0f} and {cfg.f('LENGTH_MAX_KM'):.0f} km",
         f"{ck['length_km']:.2f} km", ck["length_ok"]),
        ("line inside the corridor",
         f"{100*ck['inside_fraction']:.2f} %", ck["inside_ok"]),
        ("passes the check towns",
         ", ".join(f"{k.replace('ANCHOR_','').lower()} {v:.0f} m"
                   for k, v in ck["town_distance_m"].items()) or "none set",
         ck["towns_ok"]),
        ("median offset from the terrain-only route",
         f"{ck['offset_median_m']:.0f} m "
         f"(p90 {ck['offset_p90_m']:.0f}, max {ck['offset_max_m']:.0f})",
         ck["offset_ok"]),
        ("elevation falls downstream",
         f"{100*ck['elevation'].get('fraction_over_noise', 1):.2f} % of vertices "
         f"rise above the running minimum by more than "
         f"{cfg.f('ELEV_NOISE_M'):.0f} m; total drop "
         f"{ck['elevation'].get('drop_m', float('nan')):.0f} m",
         ck["elev_ok"]),
    ]

    lines = [
        "# Centreline report",
        "",
        f"Written by `{TOOL}` v{C.__version__}. Every number here is also in "
        "`run_manifest.json`.",
        "",
        "## The five acceptance checks",
        "",
        "| check | value | |",
        "|---|---|---|",
    ]
    for name, val, ok in rows:
        lines.append(f"| {name} | {val} | **{flag(ok)}** |")

    lines += [
        "",
        f"**Overall: {flag(ck['all_pass'])}.**",
        "",
        "## What the numbers mean",
        "",
        "**The elevation check is the one to trust.** A river does not run "
        "uphill. If the line had jumped into a tributary at a confluence, the "
        "profile would climb, and it would climb by far more than DEM noise.",
        "",
        "**The offset check is a second opinion, not a duplicate.** The "
        "centreline comes from the dry-season water mask, which is optical. "
        "The cross-check route comes from terrain alone, which is not. Where "
        "the two agree the line is almost certainly on the channel. Where they "
        "disagree by more than "
        f"{cfg.f('OFFSET_MEDIAN_MAX_M'):.0f} m, the chainages are listed "
        "below and those are the places worth looking at in QGIS.",
        "",
        "## Where the two routes disagree most",
        "",
    ]
    off = ck["_offsets"]
    pts_ch = L.chainage(ck["_offset_points"])
    bad = np.nonzero(off > cfg.f("OFFSET_MEDIAN_MAX_M"))[0]
    if bad.size == 0:
        lines.append("Nowhere. The two routes stay within "
                     f"{cfg.f('OFFSET_MEDIAN_MAX_M'):.0f} m of each other for "
                     "the whole reach.")
    else:
        lines.append("| chainage km | offset m |")
        lines.append("|---|---|")
        for i in bad[np.argsort(-off[bad])][:12]:
            lines.append(f"| {pts_ch[i]/1000:.2f} | {off[i]:.0f} |")

    lines += [
        "",
        "## Wide-water reaches",
        "",
        "Where the channel is wider than "
        f"{cfg.f('RESERVOIR_WIDTH_M'):.0f} m, which in this corridor mostly "
        "means the Srinagar reservoir. No special case was needed: the cost "
        "surface is cheapest along the middle of any water body, so the route "
        "takes the centre of a wide reach the same way it takes the centre of "
        "a narrow one.",
        "",
    ]
    if ck["wide_reaches_km"]:
        lines.append("| from km | to km |")
        lines.append("|---|---|")
        for a, b in ck["wide_reaches_km"]:
            lines.append(f"| {a:.2f} | {b:.2f} |")
    else:
        lines.append("None found.")

    lines += [
        "",
        "## What to do next",
        "",
        "1. Look at `centreline_qc.png`. The elevation panel should fall from "
        "left to right with no step up.",
        "2. Load `corridor_centreline.gpkg` in QGIS over a dry-season scene "
        "and look at it once.",
        "3. If both look right, place the digitising windows:",
        "",
        "```",
        "conda activate fyp",
        'cd /d "%s"' % cfg.str("CODE_DIR") + r"\FYP_code",
        'python 05_reference_windows.py --strata data/strata_2021 '
        '--grid data/flood_2021_v2/flood_2021_otsu.tif '
        '--pred data/flood_2021_v2/flood_2021_otsu.tif '
        '--centreline "%s" --n 8 --size 5 --out data/reference_windows_2021'
        % cfg.str("CENTRELINE_OUT"),
        "```",
        "",
        "4. Keep `centreline_report.md` and `centreline_qc.png` with the project records.",
        "",
        "## What this did not do",
        "",
        "It did not touch `corridor.geojson`. REBUILD_CORRIDOR is "
        f"{str(cfg.b('REBUILD_CORRIDOR', 'false')).lower()}, which is the "
        "9 September decision: every strata raster, every Earth Engine export "
        "and every accuracy figure already computed is clipped to the existing "
        "corridor, so replacing it is a day of recomputation rather than a "
        "side effect of running a centreline tool.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - top level, message is the product
        C.say("")
        C.say("ERROR  " + type(exc).__name__)
        for ln in str(exc).splitlines():
            C.say("       " + ln)
        C.say("")
        C.say("Nothing further was written. Keep this text and the .log file.")
        sys.exit(1)
