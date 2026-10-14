"""
t03_watermask_audit.py - which permanent-water layer to subtract, measured.

Replaces the twenty minutes of panning around QGIS asking "does the
polygon hug the water, or does it cover the whole braided bed including dry
bar?"

That question, measured, is: what fraction of the candidate polygon is
actually water at the end of the dry season, and how does its width compare
with the water's width across the same transect. Both are computed here, for
every candidate layer, against the same dry-season reference.

The verdict is still yours. This produces the table and the pictures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import common as C
import linelib as L
from t01_centreline import build_water_mask, clean_mask, corridor_mask

TOOL = "t03_watermask_audit"


def candidates(cfg):
    """[(name, kind, path), ...]"""
    out = []
    lc = cfg.path("LANDCOVER_DIR")
    for fn in cfg.csv("WATERMASK_LAYERS"):
        p = Path(fn)
        if not p.is_absolute():
            p = lc / fn
        out.append((p.stem, "vector", p))
    if cfg.has("JRC_OCCURRENCE"):
        out.append(("JRC_occurrence", "raster", cfg.path("JRC_OCCURRENCE")))
    return out


def rasterise_vector(cfg, path, ref, log):
    import geopandas as gpd
    from rasterio.features import rasterize

    g = gpd.read_file(path)
    field = cfg.str("LULC_WATER_FIELD", "")
    value = cfg.str("LULC_WATER_VALUE", "")
    if field and field in g.columns and value:
        before = len(g)
        g = g[g[field].astype(str).str.strip() == value.strip()]
        log.info("%s: filtered %s == %r, %d of %d features",
                 path.name, field, value, len(g), before)
        C.say(f"    filter {field} == '{value}': {len(g)} of {before} features")
    if g.empty:
        raise RuntimeError(
            f"{path.name}: no features left after the "
            f"{cfg.str('LULC_WATER_FIELD', '')} filter.\n"
            f"  Check LULC_WATER_FIELD and LULC_WATER_VALUE against the "
            f"attribute table."
        )
    if g.crs is None:
        g = g.set_crs(cfg.str("CRS_STORE"))
    g = g.to_crs(cfg.str("CRS_METRIC"))
    return rasterize([(geom, 1) for geom in g.geometry],
                     out_shape=ref["shape"], transform=ref["transform"],
                     fill=0, dtype="uint8").astype(bool)


def warp_raster(cfg, path, ref, threshold):
    import rasterio
    from rasterio.warp import Resampling, reproject

    dst = np.zeros(ref["shape"], dtype="float32")
    with rasterio.open(path) as src:
        reproject(source=rasterio.band(src, 1), destination=dst,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=ref["transform"], dst_crs=ref["crs"],
                  resampling=Resampling.bilinear,
                  src_nodata=src.nodata, dst_nodata=0)
    return dst >= threshold


def warp_raster_class(path, ref, values):
    """Water as a class code rather than an occurrence score.

    ESA WorldCover marks permanent water as class 80. Bilinear resampling a
    class raster invents classes that do not exist, so this uses nearest.
    """
    import rasterio
    from rasterio.warp import Resampling, reproject

    dst = np.zeros(ref["shape"], dtype="float32")
    with rasterio.open(path) as src:
        reproject(source=rasterio.band(src, 1), destination=dst,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=ref["transform"], dst_crs=ref["crs"],
                  resampling=Resampling.nearest,
                  src_nodata=src.nodata, dst_nodata=0)
    return np.isin(np.rint(dst).astype("int32"), list(values))


def load_centreline(cfg):
    import geopandas as gpd

    p = cfg.path("CENTRELINE_OUT")
    if not p.exists():
        return None
    g = gpd.read_file(p)
    if g.crs is None:
        g = g.set_crs(cfg.str("CRS_STORE"))
    g = g.to_crs(cfg.str("CRS_METRIC"))
    return list(g.geometry.iloc[0].coords)


def selftest(cfg):
    from t01_centreline import dry_scene_pairs

    ck = C.Checks(f"{TOOL} self-test")
    ck.require_file("corridor", cfg.path("CORRIDOR"))
    for name, kind, p in candidates(cfg):
        ck.require_file(f"layer {name}", p)
    for idx_p, ten_p, tag in dry_scene_pairs(cfg):
        ck.require_file(f"dry {tag}", idx_p)
    cl = cfg.path("CENTRELINE_OUT")
    ck.check("centreline present (widths need it)", cl.exists(),
             str(cl) if cl.exists() else "run t01 first; widths get skipped")
    return ck.report()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--extra", action="append", default=[],
                    metavar="NAME=PATH[@SPEC]",
                    help="an extra candidate raster, outside the .env list. "
                         "SPEC is either a number, meaning cells at or above it "
                         "are water, or 'class:80' or 'class:80,90', meaning "
                         "cells equal to those class codes are water. "
                         "Example: --extra "
                         "\"ESA_WorldCover=..\\..\\05 Data\\landcover\\"
                         "LC_ESA_WorldCover_10m.tif@class:80\". Repeatable.")
    ap.add_argument("--union", action="append", default=[],
                    metavar="NAME=A+B",
                    help="score the union of two candidates already listed. "
                         "Example: --union \"JRC_or_ESA=JRC_occurrence+ESA_WorldCover\". "
                         "Repeatable. This is what settles the permanent-water "
                         "question: JRC alone under-masks, ESA is a single-year "
                         "snapshot, and the union is the safe direction because "
                         "under-masking turns river into false flood.")
    a = ap.parse_args(argv)

    cfg = C.Cfg.load(a.config)
    if a.selftest:
        return 0 if selftest(cfg) else 1
    if not selftest(cfg):
        C.say("")
        C.say("self-test failed. Nothing was computed.")
        return 1

    out_dir = Path(cfg.path("OUT_DIR")) / "watermask_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = C.setup_logger(out_dir, TOOL)
    man = C.Manifest(TOOL, cfg)

    C.rule("1. dry-season reference water")
    water_raw, ref = build_water_mask(cfg, log, man)
    res = ref["res"]
    water, _ = clean_mask(water_raw, res, cfg.f("MIN_WATER_HA"))
    inside, poly = corridor_mask(cfg, ref, log, man)
    water &= inside
    C.say(f"  reference water {C.km2(water.sum(), res):.3f} km2 in corridor")

    line = load_centreline(cfg)
    wref = None
    if line is None:
        C.say("  no centreline on disk; width columns will be blank")
    else:
        line = L.resample(line, cfg.f("TRANSECT_SPACING_M"))
        wref = L.widths(line, water, ref["transform"], res,
                        max_half_m=cfg.f("MAX_HALF_WIDTH_M"))

    C.rule("2. candidate layers")
    rows, masks = [], {}

    def score(name, m):
        """Every number this tool reports about one candidate mask."""
        masks[name] = m
        inter = int((m & water).sum())
        union = int((m | water).sum())
        row = {
            "layer": name,
            "area_km2": round(C.km2(int(m.sum()), res), 3),
            "wet_fraction": round(float(inter / max(int(m.sum()), 1)), 4),
            "iou_vs_dry_water": round(inter / union, 4) if union else None,
            "commission_km2": round(C.km2(int((m & ~water).sum()), res), 3),
            "omission_km2": round(C.km2(int((water & ~m).sum()), res), 3),
        }
        if line is not None:
            wl = L.widths(line, m, ref["transform"], res,
                          max_half_m=cfg.f("MAX_HALF_WIDTH_M"))
            both = np.isfinite(wl) & np.isfinite(wref)
            row["median_width_m"] = (
                round(float(np.median(wl[np.isfinite(wl)])), 1)
                if np.isfinite(wl).any() else None)
            row["median_water_width_m"] = (
                round(float(np.median(wref[np.isfinite(wref)])), 1)
                if np.isfinite(wref).any() else None)
            row["width_ratio"] = (
                round(float(np.median(wl[both] / wref[both])), 2)
                if both.sum() else None)
        rows.append(row)
        for k, v in row.items():
            if k != "layer":
                C.say(f"    {k:24s} {v}")

    for name, kind, p in candidates(cfg):
        C.say(f"  {name}")
        man.input(p)
        m = (rasterise_vector(cfg, p, ref, log) if kind == "vector"
             else warp_raster(cfg, p, ref, cfg.f("JRC_MIN_OCCURRENCE")))
        score(name, m & inside)

    for spec in a.extra:
        if "=" not in spec:
            ap.error("--extra wants NAME=PATH[@SPEC], got %r" % spec)
        name, rest = spec.split("=", 1)
        path, _, sp = rest.partition("@")
        path = Path(path)
        if not path.exists():
            C.say(f"  SKIPPED {name}: not found at {path}")
            continue
        C.say(f"  {name}")
        man.input(path)
        if sp.startswith("class:"):
            vals = [int(v) for v in sp[6:].split(",") if v.strip()]
            m = warp_raster_class(path, ref, vals)
        else:
            m = warp_raster(cfg, path, ref, float(sp or 50))
        score(name.strip(), m & inside)

    for spec in a.union:
        if "=" not in spec or "+" not in spec:
            ap.error("--union wants NAME=A+B, got %r" % spec)
        name, rest = spec.split("=", 1)
        parts = [x.strip() for x in rest.split("+") if x.strip()]
        missing = [x for x in parts if x not in masks]
        if missing:
            C.say(f"  SKIPPED {name}: no candidate called {', '.join(missing)}. "
                  f"Available: {', '.join(sorted(masks))}")
            continue
        C.say(f"  {name}  ({' + '.join(parts)})")
        u = masks[parts[0]].copy()
        for x in parts[1:]:
            u |= masks[x]
        score(name.strip(), u)

    import pandas as pd
    df = pd.DataFrame(rows)
    p = out_dir / "watermask_audit.csv"
    df.to_csv(p, index=False)
    man.output(p)

    C.rule("3. panels")
    panels(cfg, out_dir, ref, water, masks, man)

    md = [
        "# Permanent-water layer audit", "",
        "Reference is the dry-season union across "
        f"{', '.join(cfg.csv('DRY_SCENES'))}, using "
        f"MNDWI > {cfg.f('MNDWI_WATER')} and B11 < {cfg.f('SWIR_WATER')} and "
        "SCL clear.", "",
        df.to_markdown(index=False), "",
        "## How to read this", "",
        "**`wet_fraction`** answers the original question directly. A layer "
        "that hugs the water is mostly wet at the end of the dry season. A "
        "layer that maps the whole braided bed, dry bar included, is not.", "",
        "**`width_ratio`** is the same question from the other side: the "
        "layer's width across the channel divided by the water's width across "
        "the same transect. A ratio near 1 hugs the water. A ratio of 3 covers "
        "three times the wetted width.", "",
        "**`commission_km2`** is the part of the layer that is not water. It "
        "is the number that matters for the change runs, because that area is "
        "subtracted from every flood extent whether or not it was ever wet.",
        "",
        "## The decision this feeds", "",
        "Under-masking turns river into false flood, which is the dangerous "
        "direction: the flood is about 2 km2 against 9 to 23 km2 of permanent "
        "water. Over-masking only loses a little real flood at the channel "
        "edge. That asymmetry is why the union of two products is the safe "
        "choice and a single product is not, and it is why the union row "
        "above is the one to read first.", "",
    ]
    p = out_dir / "watermask_audit.md"
    p.write_text("\n".join(md), encoding="utf-8")
    man.output(p)
    man.write(out_dir)
    C.say("")
    C.say(f"  done. read {p}")
    return 0


def panels(cfg, out_dir, ref, water, masks, man):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    import rasterio.transform as rt
    import rasterio.windows
    from pyproj import Transformer

    sites = cfg.sites("PANEL_SITES")
    if not sites:
        C.say("  PANEL_SITES is empty; no panels drawn")
        return
    colours = ["red", "yellow", "lime", "magenta"]
    stem = cfg.csv("DRY_SCENES")[0]
    bp = cfg.path("S2_DIR") / f"{stem}_10m.tif"
    tr = Transformer.from_crs(cfg.str("CRS_STORE"), cfg.str("CRS_METRIC"),
                              always_xy=True)
    half = int(cfg.f("PANEL_HALF_M", "1500") / ref["res"])

    with rasterio.open(bp) as src:
        rgb_idx = [C.band_index(cfg, "S2_BANDS", b) for b in ("B4", "B3", "B2")]
        n = len(sites)
        cols = min(3, n)
        rows_n = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows_n, cols,
                                 figsize=(4.6 * cols, 4.2 * rows_n))
        axes = np.atleast_1d(axes).ravel()
        for ax, (lon, lat) in zip(axes, sites):
            x, y = tr.transform(lon, lat)
            r, c = rt.rowcol(ref["transform"], x, y)
            r0, r1 = max(0, r - half), min(ref["shape"][0], r + half)
            c0, c1 = max(0, c - half), min(ref["shape"][1], c + half)
            if r1 <= r0 or c1 <= c0:
                ax.axis("off")
                continue
            win = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
            rgb = np.stack([src.read(i, window=win) for i in rgb_idx]
                           ).astype("float32") / cfg.f("S2_SCALE")
            lo, hi = np.nanpercentile(rgb, [2, 98])
            rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
            ax.imshow(np.transpose(rgb, (1, 2, 0)))
            sub = water[r0:r1, c0:c1]
            if sub.any() and not sub.all():
                ax.contour(sub, levels=[0.5], colors="cyan", linewidths=1.0)
            for k, (nm, m) in enumerate(masks.items()):
                s = m[r0:r1, c0:c1]
                if s.any() and not s.all():
                    ax.contour(s, levels=[0.5], colors=colours[k % 4],
                               linewidths=1.0)
            ax.set_title(f"{lon:.4f}, {lat:.4f}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in axes[len(sites):]:
            ax.axis("off")
        legend = ["dry-season water (cyan)"] + [
            f"{nm} ({colours[k % 4]})" for k, nm in enumerate(masks)]
        fig.suptitle("  |  ".join(legend), fontsize=9)
        fig.tight_layout()
        p = out_dir / "watermask_panels.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        man.output(p)
        C.say(f"  wrote {p.name}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        C.say("")
        C.say("ERROR  " + type(exc).__name__)
        for ln in str(exc).splitlines():
            C.say("       " + ln)
        sys.exit(1)
