"""
t06_draft_polygons.py - candidate water polygons for the digitising windows.

READ THIS BEFORE RUNNING IT.

The reference extent is the ground truth that Otsu and Random Forest are
scored against. Right now it is hand-digitised, which is the strongest thing
you can say about it. If those polygons became machine-generated from a
spectral threshold, they would stop being independent of the automatic
methods they judge, and "different sensor" only half answers that, because
both are still a threshold on an index chosen by a script.

So this tool writes to `water_event_draft` and `water_dry_draft`, in a
separate GeoPackage from the one you draw in. Moving a polygon from draft to
real is an act you perform, deliberately, one polygon at a time. Nothing here
writes to `water_event` or `water_dry`.

What it is genuinely for: correcting a drawn shape is much faster than
tracing from blank, and the marginal-fraction column tells you which windows
to distrust before you start.

If you keep any of it, the methods sentence is printed at the end and it is
the one you must use.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import common as C

TOOL = "t06_draft_polygons"

SCHEMA = {
    "geometry": "Polygon",
    "properties": {"window": "int", "source": "str:40", "note": "str:160"},
}

METHODS_SENTENCE = (
    "Reference polygons were initialised from an MNDWI and SWIR threshold "
    "and then inspected and corrected by hand at 1:5000 against true-colour "
    "imagery, with the operator making the final water/not-water judgement "
    "in all cases, including gravel bars and terrain-shadowed reaches."
)


def hillshade(dem, res_m, azimuth_deg, elevation_deg):
    """Standard hillshade, 0 to 1. Used only to find terrain shadow."""
    z = np.nan_to_num(dem, nan=float(np.nanmedian(dem)))
    dy, dx = np.gradient(z, res_m)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az = np.radians(360.0 - azimuth_deg + 90.0)
    zen = np.radians(90.0 - elevation_deg)
    hs = (np.cos(zen) * np.cos(slope) +
          np.sin(zen) * np.sin(slope) * np.cos(az - aspect))
    return np.clip(hs, 0.0, 1.0)


def windows_for(cfg, tag):
    key = f"WINDOWS_{tag.upper()}"
    if not cfg.has(key):
        return None, key
    return cfg.path(key), key


def selftest(cfg):
    ck = C.Checks(f"{TOOL} self-test")
    try:
        import fiona  # noqa: F401
        ck.check("import fiona", True)
    except Exception as e:
        ck.check("import fiona", False, type(e).__name__)
    ck.require_file("routing DEM (for the shadow mask)", cfg.path("DEM_ROUTING"))
    n = 0
    for spec in cfg.csv("DRAFT_SETS"):
        tag, _, stem = spec.partition(":")
        wp, key = windows_for(cfg, tag)
        if wp is None:
            ck.check(f"{tag}: .env has {key}", False,
                     "run 05_reference_windows.py first")
            continue
        ck.require_file(f"{tag}: windows", wp)
        ck.require_file(f"{tag}: scene {stem}",
                        cfg.path("S2_DIR") / f"{stem}_10m.tif")
        n += 1
    ck.check("at least one draft set configured", n > 0, f"{n}")
    return ck.report()


def draft_one(cfg, tag, stem, pass_name, dem, ref, man, log):
    """Candidate polygons for every window, for one scene and one pass."""
    import geopandas as gpd
    from rasterio.features import shapes
    from scipy.ndimage import binary_closing, binary_opening
    from shapely.geometry import shape as shp

    s2 = cfg.path("S2_DIR")
    idx_p = s2 / f"{stem}_indices_10m.tif"
    ten_p = s2 / f"{stem}_10m.tif"
    man.input(idx_p)
    man.input(ten_p)

    crs = cfg.str("CRS_METRIC")
    dsi = C.open_checked(idx_p, expect_crs=crs,
                         expect_bands=len(cfg.csv("IDX_BANDS")), log=log)
    dst = C.open_checked(ten_p, expect_crs=crs,
                         expect_bands=len(cfg.csv("S2_BANDS")), log=log)
    try:
        mndwi = dsi.read(C.band_index(cfg, "IDX_BANDS", "MNDWI")).astype("float32")
        b11 = C.s2_reflectance(dst, cfg, "B11")
        scl = dst.read(C.band_index(cfg, "S2_BANDS", "SCL"))
    finally:
        dsi.close()
        dst.close()

    _, cloud, snow, clear = C.scl_masks(scl, cfg)
    res = ref["res"]

    hs = hillshade(dem, res, cfg.f("SUN_AZIMUTH"), cfg.f("SUN_ELEVATION"))
    margin = cfg.f("MARGIN")
    marginal = np.abs(mndwi) <= margin
    shadowy = (hs < cfg.f("HILLSHADE_MIN")) & marginal

    wet = (mndwi > cfg.f("MNDWI_WATER")) & (b11 < cfg.f("SWIR_WATER"))
    wet &= clear
    wet &= ~shadowy

    st = np.ones((3, 3), bool)
    wet = binary_opening(wet, structure=st, iterations=1)
    wet = binary_closing(wet, structure=st, iterations=1)

    wins = gpd.read_file(windows_for(cfg, tag)[0])
    if wins.crs is None:
        wins = wins.set_crs(cfg.str("CRS_STORE"))
    wins = wins.to_crs(crs)
    id_field = next((c for c in ("window", "id", "win", "fid")
                     if c in wins.columns), None)

    min_area = cfg.f("MIN_POLY_HA") * 1e4
    rows, stats = [], []
    from rasterio.features import geometry_mask

    for i, w in wins.reset_index(drop=True).iterrows():
        wid = int(w[id_field]) if id_field else int(i + 1)
        inwin = ~geometry_mask([w.geometry], out_shape=ref["shape"],
                               transform=ref["transform"], invert=False)
        sub = wet & inwin
        n_win = int(inwin.sum())
        stats.append({
            "window": wid,
            "pass": pass_name,
            "scene": stem,
            "candidate_km2": round(C.km2(int(sub.sum()), res), 4),
            "window_km2": round(C.km2(n_win, res), 3),
            "marginal_fraction": round(float((marginal & inwin).sum() / max(n_win, 1)), 4),
            "cloud_fraction": round(float(((cloud | snow) & inwin).sum() / max(n_win, 1)), 4),
            "shadow_rejected_km2": round(C.km2(int((shadowy & inwin).sum()), res), 4),
        })
        if not sub.any():
            continue
        for geom, val in shapes(sub.astype("uint8"), mask=sub,
                                transform=ref["transform"]):
            if val != 1:
                continue
            g = shp(geom)
            if g.area < min_area:
                continue
            g = g.intersection(w.geometry)
            if g.is_empty:
                continue
            rows.append({
                "window": wid,
                "source": "t06 draft",
                "note": f"{pass_name} {stem}; correct before use",
                "geometry": g,
            })

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs) if rows else None
    return gdf, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    cfg = C.Cfg.load(a.config)
    if a.selftest:
        return 0 if selftest(cfg) else 1
    if not selftest(cfg):
        C.say("")
        C.say("self-test failed. Nothing was written.")
        C.say("If the failure is a missing reference_windows file, that is "
              "expected: run t01 for the centreline, then "
              "05_reference_windows.py, then come back.")
        return 1

    out_dir = Path(cfg.path("OUT_DIR")) / "draft_polygons"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = C.setup_logger(out_dir, TOOL)
    man = C.Manifest(TOOL, cfg)
    force = a.force or cfg.b("FORCE", "false")

    man.assume("draft polygons are candidates, not a reference. They are "
               "written to _draft tables and must be corrected by hand.")

    first = cfg.csv("DRAFT_SETS")[0].split(":")[1]
    ds = C.open_checked(cfg.path("S2_DIR") / f"{first}_10m.tif",
                        expect_crs=cfg.str("CRS_METRIC"))
    ref = {"shape": (ds.height, ds.width), "transform": ds.transform,
           "res": abs(ds.transform.a), "crs": str(ds.crs)}
    ds.close()

    from t01_centreline import dem_on_grid
    dem = dem_on_grid(cfg, ref, log, man)

    all_stats = []
    for spec in cfg.csv("DRAFT_SETS"):
        tag, _, stem = spec.partition(":")
        pass_name = "dry" if "dry" in stem.lower() else "event"
        C.rule(f"{tag}  {pass_name}  {stem}")
        gdf, stats = draft_one(cfg, tag, stem, pass_name, dem, ref, man, log)
        all_stats += stats
        gpkg = out_dir / f"draft_{tag}.gpkg"
        layer = f"water_{pass_name}_draft"
        if gdf is None or gdf.empty:
            C.say("  no candidate polygons passed MIN_POLY_HA")
        else:
            if gpkg.exists() and not force:
                C.say(f"  {gpkg.name} exists; appending layer {layer}")
            gdf.to_file(gpkg, layer=layer, driver="GPKG")
            man.output(gpkg, layer)
            C.say(f"  {len(gdf)} polygons -> {gpkg.name} : {layer}")

    import pandas as pd
    df = pd.DataFrame(all_stats)
    p = out_dir / "draft_report.csv"
    df.to_csv(p, index=False)
    man.output(p)

    worst = df.sort_values("marginal_fraction", ascending=False).head(8) \
        if not df.empty else df

    md = [
        "# Draft polygon report", "",
        "**These are candidates, not a reference.** They live in "
        "`water_event_draft` and `water_dry_draft`, in a different GeoPackage "
        "from the one you draw in. Nothing here has been written to "
        "`water_event` or `water_dry`.", "",
        "## Per window", "",
        (df.to_markdown(index=False) if not df.empty else "nothing computed"),
        "",
        "## Which windows to distrust", "",
        "`marginal_fraction` is the share of the window where MNDWI sits "
        f"between -{cfg.f('MARGIN')} and +{cfg.f('MARGIN')}. That band is the "
        "gravel-bar ambiguity of step 6.2 of the digitising reference, "
        "measured. A high value means the draft is guessing and the window is "
        "one to draw from blank rather than correct.", "",
        (worst[["window", "pass", "marginal_fraction", "cloud_fraction"]]
         .to_markdown(index=False) if not df.empty else ""),
        "",
        "`shadow_rejected_km2` is area that looked like water in MNDWI, sat "
        "in terrain shadow on the hillshade, and was thrown out. That is the "
        "6.1 failure mode caught before it reaches a polygon. If the number "
        "is large in a gorge window, check that window by eye: the mask may "
        "be over-eager.", "",
        "## If you keep any of it", "",
        "The methods sentence is:", "",
        "> " + METHODS_SENTENCE, "",
        "Use that wording. \"Generated automatically\" is not defensible when "
        "the thing being validated is also automatic.", "",
        "## What was not automated", "",
        "The water/not-water judgement itself. Steps 6.1, 6.2 and 6.3 of "
        "`docs/REFERENCE - how to digitise.md` stay yours, and so does the gravel-"
        "bar rule you have to write down once and apply to every pass.", "",
    ]
    p = out_dir / "draft_report.md"
    p.write_text("\n".join(md), encoding="utf-8")
    man.output(p)
    man.write(out_dir)

    C.say("")
    C.say("  METHODS SENTENCE, if you keep any of these polygons:")
    for ln in _wrap(METHODS_SENTENCE, 66):
        C.say("    " + ln)
    C.say("")
    C.say(f"  done. read {p}")
    return 0


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        out.append(line)
    return out


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        C.say("")
        C.say("ERROR  " + type(exc).__name__)
        for ln in str(exc).splitlines():
            C.say("       " + ln)
        sys.exit(1)
