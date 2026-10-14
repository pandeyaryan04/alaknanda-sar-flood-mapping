"""
t04_reservoir.py - is the Srinagar reservoir stable between your scenes?

Why it matters. The reservoir level is set by dam operations, not weather,
and it sits in the middle of the reach. A drawdown between the pre-image and
the post-image looks exactly like a change in flood extent. If the surface
area is stable across the four dates, that is one sentence in the methods and
the question is closed. If it is not, it is a confound and it has to be
handled rather than mentioned.

The area of interest is found automatically: the largest part of the
dry-season water mask that is wider than RESERVOIR_WIDTH_M. Set
RESERVOIR_AOI in .env to a polygon file to override that.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import common as C
from t01_centreline import build_water_mask, clean_mask, corridor_mask

TOOL = "t04_reservoir"


def find_aoi(cfg, water, ref, log):
    """Largest connected water body whose half-width exceeds the threshold."""
    from scipy.ndimage import binary_dilation, distance_transform_edt, label

    res = ref["res"]
    half = cfg.f("RESERVOIR_WIDTH_M") / 2.0
    d = distance_transform_edt(water) * res
    core = d >= half
    if not core.any():
        return None, "no water body wider than RESERVOIR_WIDTH_M was found"
    lab, n = label(core, structure=np.ones((3, 3), int))
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    pick = int(np.argmax(sizes))
    core = lab == pick
    # grow the core back out to the water body it came from
    aoi = core.copy()
    for _ in range(int(cfg.f("RESERVOIR_GROW_M", "600") / res)):
        aoi = binary_dilation(aoi, structure=np.ones((3, 3), bool)) & water
    # then a margin of dry land so a drawdown is visible, not clipped away
    for _ in range(int(cfg.f("RESERVOIR_MARGIN_M", "200") / res)):
        aoi = binary_dilation(aoi, structure=np.ones((3, 3), bool))
    log.info("auto AOI %d cells", int(aoi.sum()))
    return aoi, f"auto: widest water body, grown, {C.km2(int(aoi.sum()), res):.2f} km2"


def load_aoi(cfg, ref):
    import geopandas as gpd
    from rasterio.features import rasterize

    g = gpd.read_file(cfg.path("RESERVOIR_AOI"))
    if g.crs is None:
        g = g.set_crs(cfg.str("CRS_STORE"))
    g = g.to_crs(cfg.str("CRS_METRIC"))
    poly = g.union_all() if hasattr(g, "union_all") else g.unary_union
    return rasterize([(poly, 1)], out_shape=ref["shape"],
                     transform=ref["transform"], fill=0,
                     dtype="uint8").astype(bool)


def scene_water(cfg, stem, ref):
    """The same two-index water test, for one scene."""
    s2 = cfg.path("S2_DIR")
    idx_p = s2 / f"{stem}_indices_10m.tif"
    ten_p = s2 / f"{stem}_10m.tif"
    crs = cfg.str("CRS_METRIC")
    dsi = C.open_checked(idx_p, expect_crs=crs,
                         expect_bands=len(cfg.csv("IDX_BANDS")))
    dst = C.open_checked(ten_p, expect_crs=crs,
                         expect_bands=len(cfg.csv("S2_BANDS")))
    try:
        if (dsi.height, dsi.width) != ref["shape"]:
            raise C.RasterError(
                f"{stem} is on a different grid from the dry scenes. "
                f"Re-export it with the same region, scale and crs."
            )
        mndwi = dsi.read(C.band_index(cfg, "IDX_BANDS", "MNDWI")).astype("float32")
        b11 = C.s2_reflectance(dst, cfg, "B11")
        scl = dst.read(C.band_index(cfg, "S2_BANDS", "SCL"))
        _, _, _, clear = C.scl_masks(scl, cfg)
        wet = (mndwi > cfg.f("MNDWI_WATER")) & (b11 < cfg.f("SWIR_WATER")) & clear
        return wet, clear
    finally:
        dsi.close()
        dst.close()


def selftest(cfg):
    from t01_centreline import dry_scene_pairs

    ck = C.Checks(f"{TOOL} self-test")
    ck.require_file("corridor", cfg.path("CORRIDOR"))
    for idx_p, ten_p, tag in dry_scene_pairs(cfg):
        ck.require_file(f"dry {tag}", idx_p)
    s2 = cfg.path("S2_DIR")
    scenes = cfg.csv("RESERVOIR_SCENES")
    ck.check("RESERVOIR_SCENES has at least two dates", len(scenes) >= 2,
             f"{len(scenes)} listed")
    for stem in scenes:
        ck.require_file(f"scene {stem}", s2 / f"{stem}_10m.tif")
    if cfg.has("RESERVOIR_AOI"):
        ck.require_file("reservoir AOI", cfg.path("RESERVOIR_AOI"))
    else:
        ck.check("reservoir AOI", True, "not set; will be found automatically")
    return ck.report()


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
        C.say("\nself-test failed. Nothing was computed.")
        return 1

    out_dir = Path(cfg.path("OUT_DIR")) / "reservoir"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = C.setup_logger(out_dir, TOOL)
    man = C.Manifest(TOOL, cfg)

    C.rule("1. area of interest")
    water_raw, ref = build_water_mask(cfg, log, man)
    res = ref["res"]
    water, _ = clean_mask(water_raw, res, cfg.f("MIN_WATER_HA"))
    inside, _ = corridor_mask(cfg, ref, log, man)
    water &= inside

    if cfg.has("RESERVOIR_AOI"):
        aoi = load_aoi(cfg, ref)
        how = f"from {cfg.path('RESERVOIR_AOI').name}"
    else:
        aoi, how = find_aoi(cfg, water, ref, log)
        if aoi is None:
            C.say(f"  {how}")
            C.say("  Nothing to measure. Either the corridor has no wide water "
                  "body, or RESERVOIR_WIDTH_M is set too high.")
            return 0
        man.assume("reservoir AOI found automatically, not supplied. "
                   "Check reservoir_aoi.png before quoting the numbers.")
    C.say(f"  AOI {how}, {C.km2(int(aoi.sum()), res):.3f} km2")
    man.param("aoi_km2", round(C.km2(int(aoi.sum()), res), 3))

    C.rule("2. water area per scene")
    rows = []
    masks = {}
    for stem in cfg.csv("RESERVOIR_SCENES"):
        man.input(cfg.path("S2_DIR") / f"{stem}_10m.tif")
        wet, clear = scene_water(cfg, stem, ref)
        in_aoi = wet & aoi
        cov = float((clear & aoi).sum() / max(int(aoi.sum()), 1))
        rows.append({
            "scene": stem,
            "water_km2": round(C.km2(int(in_aoi.sum()), res), 4),
            "aoi_clear_pct": round(100 * cov, 2),
        })
        masks[stem] = in_aoi
        C.say(f"  {stem:34s} {rows[-1]['water_km2']:8.4f} km2   "
              f"clear {rows[-1]['aoi_clear_pct']:6.2f} %")

    import pandas as pd
    df = pd.DataFrame(rows)
    usable = df[df["aoi_clear_pct"] >= 90]
    if len(usable) >= 2:
        lo, hi = usable["water_km2"].min(), usable["water_km2"].max()
        spread = 100.0 * (hi - lo) / max(hi, 1e-9)
    else:
        lo = hi = spread = float("nan")
    tol = cfg.f("RESERVOIR_TOL_PCT")
    stable = bool(np.isfinite(spread) and spread <= tol)

    man.param("min_km2", None if not np.isfinite(lo) else round(float(lo), 4))
    man.param("max_km2", None if not np.isfinite(hi) else round(float(hi), 4))
    man.param("spread_pct", None if not np.isfinite(spread) else round(float(spread), 2))
    man.param("stable", stable)

    C.rule("3. verdict")
    if not np.isfinite(spread):
        C.say("  fewer than two scenes had a clear view of the AOI. "
              "No verdict; add dates to RESERVOIR_SCENES.")
    elif stable:
        C.say(f"  STABLE. spread {spread:.2f} % across "
              f"{len(usable)} usable dates, within RESERVOIR_TOL_PCT {tol}.")
        C.say("  This is a one-sentence methods note.")
    else:
        C.say(f"  NOT STABLE. spread {spread:.2f} % exceeds "
              f"RESERVOIR_TOL_PCT {tol}.")
        C.say("  The reservoir changes between your scenes by more than the "
              "tolerance, so a change-detection result over this reach mixes "
              "dam operation with flood. It has to be excluded or reported, "
              "not mentioned in passing.")

    p = out_dir / "reservoir_area.csv"
    df.to_csv(p, index=False)
    man.output(p)

    _figure(out_dir / "reservoir_aoi.png", aoi, masks, ref, man)

    md = [
        "# Srinagar reservoir, surface area by date", "",
        f"Area of interest: {how}.", "",
        df.to_markdown(index=False), "",
        f"**Spread across dates with a clear view: "
        f"{'n/a' if not np.isfinite(spread) else f'{spread:.2f} %'}. "
        f"Tolerance {tol} %. "
        f"Verdict: {'stable' if stable else 'not stable'}.**", "",
        "Dates where `aoi_clear_pct` is below 90 are excluded from the "
        "spread, because a cloud over the reservoir reads as a drawdown.", "",
        "## Why this is here", "",
        "The reservoir level is set by dam operations rather than by weather. "
        "A fill or drawdown between the pre-image and the post-image looks "
        "exactly like a change in flood extent, and it sits in the middle of "
        "the study reach. Stable means one sentence in the methods. Not "
        "stable means this reach needs handling before its change numbers "
        "mean anything.", "",
    ]
    p = out_dir / "reservoir_report.md"
    p.write_text("\n".join(md), encoding="utf-8")
    man.output(p)
    man.write(out_dir)
    C.say("")
    C.say(f"  done. read {p}")
    return 0


def _figure(path, aoi, masks, ref, man):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rr, cc = np.nonzero(aoi)
    if rr.size == 0:
        return
    r0, r1 = rr.min(), rr.max() + 1
    c0, c1 = cc.min(), cc.max() + 1
    n = len(masks)
    cols = min(4, max(n, 1))
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 3.6 * rows_n))
    axes = np.atleast_1d(axes).ravel()
    for ax, (stem, m) in zip(axes, masks.items()):
        ax.imshow(aoi[r0:r1, c0:c1], cmap="Greys", alpha=0.25,
                  interpolation="nearest")
        ax.imshow(np.ma.masked_where(~m[r0:r1, c0:c1], m[r0:r1, c0:c1]),
                  cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(stem, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("reservoir AOI (grey) and water by date (blue)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    man.output(path)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        C.say("")
        C.say("ERROR  " + type(exc).__name__)
        for ln in str(exc).splitlines():
            C.say("       " + ln)
        sys.exit(1)
