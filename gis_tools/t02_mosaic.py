"""
t02_mosaic.py - SCL-aware Sentinel-2 mosaics, and the cloud-gap number.

Replaces the five merges in QGIS, and the manual check of each pair
("load both in QGIS and confirm the cloud gaps in one are covered
by the other").

The primary date wins wherever it is clear; the partner fills only where the
primary is not. That removes the trap in step 2.2 of the digitising
reference, where the merge order decides the answer and the last input listed
silently wins.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import common as C

TOOL = "t02_mosaic"


def sets(cfg):
    out = []
    for key in cfg.csv("MOSAIC_SETS"):
        if not cfg.has(key):
            C.say(f"  skipping {key}: empty in .env")
            continue
        stems = cfg.csv(key)
        tag = key.replace("PAIR_", "").lower()
        out.append((key, tag, stems))
    return out


def corridor_mask(cfg, ref):
    import geopandas as gpd
    from rasterio.features import rasterize

    g = gpd.read_file(cfg.path("CORRIDOR"))
    if g.crs is None:
        g = g.set_crs(cfg.str("CRS_STORE"))
    g = g.to_crs(cfg.str("CRS_METRIC"))
    poly = g.union_all() if hasattr(g, "union_all") else g.unary_union
    return rasterize([(poly, 1)], out_shape=ref["shape"],
                     transform=ref["transform"], fill=0,
                     dtype="uint8").astype(bool)


def selftest(cfg):
    ck = C.Checks(f"{TOOL} self-test")
    for mod in ("numpy", "rasterio", "geopandas"):
        try:
            __import__(mod); ck.check(f"import {mod}", True)
        except Exception as e:
            ck.check(f"import {mod}", False, type(e).__name__)
    ck.require_file("corridor", cfg.path("CORRIDOR"))
    s2 = cfg.path("S2_DIR")
    ck.check("S2_DIR exists", s2.is_dir(), str(s2))
    n = 0
    for key, tag, stems in sets(cfg):
        for stem in stems:
            ck.require_file(f"{tag}: {stem} bands", s2 / f"{stem}_10m.tif")
            ck.require_file(f"{tag}: {stem} indices",
                            s2 / f"{stem}_indices_10m.tif")
            n += 1
    ck.check("at least one pair configured", n > 0, f"{n} scenes")
    return ck.report()


def mosaic_one(cfg, tag, stems, out_dir, inside, man, force):
    import rasterio

    s2 = cfg.path("S2_DIR")
    nb = len(cfg.csv("S2_BANDS"))
    ni = len(cfg.csv("IDX_BANDS"))
    scl_i = C.band_index(cfg, "S2_BANDS", "SCL")
    crs = cfg.str("CRS_METRIC")

    stack, clears, profiles = [], [], []
    for stem in stems:
        bp = s2 / f"{stem}_10m.tif"
        ip = s2 / f"{stem}_indices_10m.tif"
        man.input(bp); man.input(ip)
        db = C.open_checked(bp, expect_crs=crs, expect_bands=nb)
        di = C.open_checked(ip, expect_crs=crs, expect_bands=ni)
        scl = db.read(scl_i)
        _, _, _, clear = C.scl_masks(scl, cfg)
        stack.append((db.read(), di.read()))
        clears.append(clear)
        profiles.append((db.profile.copy(), di.profile.copy()))
        db.close(); di.close()

    shapes = {a.shape[1:] for a, _ in stack}
    if len(shapes) != 1:
        raise C.RasterError(
            f"{tag}: the scenes are on different grids {shapes}. "
            f"Re-export the odd one with the same region, scale and crs."
        )

    h, w = stack[0][0].shape[1:]
    bands = np.zeros((nb, h, w), dtype=stack[0][0].dtype)
    idx = np.full((ni, h, w), np.nan, dtype="float32")
    filled = np.zeros((h, w), bool)
    won = {}

    for k, ((b, i), clear) in enumerate(zip(stack, clears)):
        take = clear & ~filled
        bands[:, take] = b[:, take]
        idx[:, take] = i[:, take].astype("float32")
        filled |= take
        won[stems[k]] = int((take & inside).sum())

    gap = (~filled) & inside
    res = abs(profiles[0][0]["transform"].a)
    n_in = int(inside.sum())

    row = {
        "set": tag,
        "corridor_cells": n_in,
        "mosaic_clear_pct": round(C.pct(int((filled & inside).sum()), n_in), 3),
        "gap_km2": round(C.km2(int(gap.sum()), res), 4),
    }
    for k, stem in enumerate(stems):
        row[f"clear_pct[{stem}]"] = round(
            C.pct(int((clears[k] & inside).sum()), n_in), 3)
        row[f"won_pct[{stem}]"] = round(C.pct(won[stem], n_in), 3)

    pb, pi = profiles[0]
    pb.update(count=nb, compress="deflate")
    pi.update(count=ni, dtype="float32", compress="deflate")

    outs = {}
    bp = C.guard_output(out_dir / f"S2_{tag}_mosaic.tif", force)
    with rasterio.open(bp, "w", **pb) as d:
        d.write(bands)
        for i, nm in enumerate(cfg.csv("S2_BANDS"), start=1):
            d.set_band_description(i, nm)
    outs["bands"] = bp

    ip = C.guard_output(out_dir / f"S2_{tag}_mndwi.tif", force)
    with rasterio.open(ip, "w", **pi) as d:
        d.write(np.nan_to_num(idx, nan=-9999).astype("float32"))
        for i, nm in enumerate(cfg.csv("IDX_BANDS"), start=1):
            d.set_band_description(i, nm)
    outs["indices"] = ip

    pg = pb.copy(); pg.update(count=1, dtype="uint8", nodata=255)
    gp = C.guard_output(out_dir / f"S2_{tag}_gapmask.tif", force)
    with rasterio.open(gp, "w", **pg) as d:
        d.write(gap.astype("uint8"), 1)
        d.set_band_description(1, "1 = clear in no scene, inside corridor")
    outs["gap"] = gp

    for p in outs.values():
        man.output(p)
    return row, outs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default=None, help="one PAIR_ key")
    a = ap.parse_args(argv)

    cfg = C.Cfg.load(a.config)
    if a.selftest:
        return 0 if selftest(cfg) else 1
    if not selftest(cfg):
        C.say("\nself-test failed. Nothing was written.")
        return 1

    out_dir = Path(cfg.path("S2_DIR"))
    rep_dir = Path(cfg.path("OUT_DIR")) / "mosaic"
    rep_dir.mkdir(parents=True, exist_ok=True)
    log = C.setup_logger(rep_dir, TOOL)
    man = C.Manifest(TOOL, cfg)
    force = a.force or cfg.b("FORCE", "false")

    todo = [s for s in sets(cfg) if (a.only is None or s[0] == a.only)]
    if not todo:
        C.say("nothing to do; check MOSAIC_SETS")
        return 1

    first = cfg.path("S2_DIR") / f"{todo[0][2][0]}_10m.tif"
    ds = C.open_checked(first, expect_crs=cfg.str("CRS_METRIC"))
    ref = {"shape": (ds.height, ds.width), "transform": ds.transform}
    ds.close()
    inside = corridor_mask(cfg, ref)

    rows = []
    for key, tag, stems in todo:
        C.rule(f"{key}  ->  S2_{tag}_mosaic.tif")
        row, outs = mosaic_one(cfg, tag, stems, out_dir, inside, man, force)
        rows.append(row)
        for k, v in row.items():
            if k != "set":
                C.say(f"  {k:28s} {v}")
        thr = cfg.f("MIN_MOSAIC_CLEAR")
        if row["mosaic_clear_pct"] < thr:
            C.say(f"  WARNING  mosaic clear {row['mosaic_clear_pct']:.2f} % is "
                  f"below MIN_MOSAIC_CLEAR {thr}. Gap {row['gap_km2']:.3f} km2 "
                  f"is in S2_{tag}_gapmask.tif. Do not digitise inside it.")

    import pandas as pd
    df = pd.DataFrame(rows)
    p = rep_dir / "mosaic_report.csv"
    df.to_csv(p, index=False)
    man.output(p)

    md = ["# Mosaic report", "",
          "Primary date first in each pair; it wins wherever it is clear.",
          "Percentages are of corridor cells and use the same SCL rule as",
          "`03_export_remaining.js`, so they are directly comparable to the",
          "figures the Earth Engine Console printed.", "",
          df.to_markdown(index=False), "",
          "**Reading it.** `mosaic_clear_pct` at or near 100 means the two",
          "dates cover each other's cloud. `gap_km2` is ground that is clear",
          "in neither and must not be digitised: it is marked in the",
          "`_gapmask.tif` for each set, and `06_rasterise_reference.py --holes`",
          "is what excludes it from scoring.", ""]
    p = rep_dir / "mosaic_report.md"
    p.write_text("\n".join(md), encoding="utf-8")
    man.output(p)
    man.write(rep_dir)
    C.say("")
    C.say(f"  done. read {p}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        C.say("")
        C.say("ERROR  " + type(exc).__name__)
        for ln in str(exc).splitlines():
            C.say("       " + ln)
        sys.exit(1)
