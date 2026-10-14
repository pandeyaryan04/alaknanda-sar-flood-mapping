r"""
jalsar_testdata.py
==================
A small synthetic project with three finished Jal-SAR events, for the
self-tests of the web map and the bulletin. It takes seconds: instead of
training models it uses two stand-in "models" (thresholds on VV), and the
Otsu maps of the synthetic project stand in for Table D, scored by the real
04_stratified_metrics.py.

Nothing here touches the real project. It only ever writes inside the folder
it is given (a temporary folder with a space in its name).

USAGE  (Anaconda Prompt, fyp active, from 04 Code\jalsar)
-----
    python jalsar_testdata.py --selftest
"""

__stage__ = "Jal-SAR: synthetic events for the self-tests"
__reads__ = ["nothing outside the test folder"]
__writes__ = ["<test folder>/Final Year Project/ (a synthetic project)"]

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

import jalsar_core as gc

mc = gc.mc
rasterio = gc.rasterio

STAND_IN = {"rf": ("rf", "hand", 0.61, -17.0), "unet": ("unet", "hand", 0.64, -18.5)}


def _fake_member(gp, family):
    m = STAND_IN.get(family)
    return None if m is None else m[:3]


def _fake_predict(gp, member, inp, cpu_ok=False, tile=512, overlap=64):
    thr = next(v[3] for v in STAND_IN.values() if v[0] == member[0])
    vv, hand = np.nan_to_num(inp["x"][0], nan=0.0), np.nan_to_num(inp["x"][2], nan=0.0)
    rng = np.random.default_rng(int(-thr * 10))
    # water below the VV threshold, plus scattered false alarms on high ground
    prob = ((vv < thr) | ((hand > 15) & (rng.random(vv.shape) < 0.01))).astype("float32")
    fmap, diag = mc.make_flood_map(prob, inp["valid"], 0.5, inp["perm"], inp["distort"])
    diag.update(model=member[0], trainset=member[1], threshold=0.5)
    return fmap, diag


def write_dem(gp):
    """A synthetic 30 m DEM in lon/lat over the scene, hills and a valley."""
    from rasterio.transform import from_origin
    from rasterio.warp import transform_bounds
    with rasterio.open(gp.scene(2021, "grd")) as src:
        w, s, e, n = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    d = 0.0003
    w, s, e, n = w - 10 * d, s - 10 * d, e + 10 * d, n + 10 * d
    cols, rows = int((e - w) / d) + 1, int((n - s) / d) + 1
    yy, xx = np.mgrid[0:rows, 0:cols].astype("float32")
    dem = (600 + 300 * np.abs(yy / rows - 0.5) * 2 + 60 * np.sin(xx / 7) * np.cos(yy / 5)
           ).astype("float32")
    p = gp.d05 / "dem" / "DEM_COP_GLO30_30m.tif"
    p.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(p, "w", driver="GTiff", width=cols, height=rows, count=1,
                       dtype="float32", crs="EPSG:4326", transform=from_origin(w, n, d, d),
                       nodata=-9999.0) as dst:
        dst.write(dem, 1)
    return p


# The synthetic reference is easy, so every class scores high. These stand-in
# precisions make the three trust classes all appear in the test bulletin.
STAND_IN_PRECISION = {("hand", "15_50m"): "0.1", ("hand", "gt50m"): "0.1",
                      ("incidence", "40_50"): "0.3", ("incidence", "gt50"): "0.3"}


def lower_precision(path):
    import csv
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        if r["reach"] == "pooled" and (r["dimension"], r["stratum"]) in STAND_IN_PRECISION:
            r["precision"] = STAND_IN_PRECISION[(r["dimension"], r["stratum"])]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def build_display_project(root):
    """Synthetic project with events 2021, 2023 and July 2023 already mapped."""
    import models_synthetic as syn
    syn.build_project(root)
    gp = gc.GPaths(root)
    gc.add_villages_roads(gp)
    write_dem(gp)
    # Table D and Table A stand-ins: the synthetic Otsu maps, scored by the real scorer
    for y in mc.YEARS:
        for model, src in (("rf", gp.otsu(y)), ("unet", gp.otsu(y, "change"))):
            dst = gp.lowo_map(y, model, "grd")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            if True:                  # false alarms on high ground, so the trust classes differ
                with rasterio.open(dst, "r+") as r:
                    a = r.read(1)
                    hand = mc.reproject_onto(gp.strata(y) / "strata_hand_m.tif", r.profile)
                    rng = np.random.default_rng(y)
                    a[(a == 0) & (hand > 15) & (rng.random(a.shape) < (0.15 if model == "rf" else 0.3))] = 1
                    r.write(a, 1)
            gc.run_scorer(gp, dst, y, gp.tables / f"lowo_{y}_{model}_grd")
            lower_precision(gp.tables / f"lowo_{y}_{model}_grd" / "metrics_by_stratum.csv")
            gc.run_scorer(gp, dst, y, gp.tables / f"{y}_{model}_hand")
    # the radar-only July 2023 event: the 2021 scene stands in for its scene
    jul = gp.d05 / "sentinel1" / "S1_2023jul_post_20230719_orb63.tif"
    shutil.copy2(gp.scene(2021, "grd"), jul)
    keep = gc.family_member, gc.predict_family
    gc.family_member, gc.predict_family = _fake_member, _fake_predict
    try:
        for ev in gc.load_events(gp):
            gc.make_event(gp, ev, families=("rf", "unet"), cpu_ok=True)
    finally:
        gc.family_member, gc.predict_family = keep
    return gp


def selftest():
    print("jalsar_testdata.py self-test")
    with tempfile.TemporaryDirectory(prefix="jalsar testdata ", ignore_cleanup_errors=True) as td:
        gp = build_display_project(Path(td) / "Final Year Project")
        evs = gc.load_events(gp)
        done = [(gp.event_dir(e.id) / "summary.json").exists() for e in evs]
        s = mc.load_json(gp.event_dir(evs[0].id) / "summary.json")
        checks = [("three events mapped", len(evs) == 3 and all(done)),
                  ("validated event carries Tables A and D",
                   bool(s["validation"]["table_a"]) and bool(s["validation"]["table_d"])),
                  ("a headline family chosen by the Table D rule",
                   s["headline"]["family"] in ("rf", "unet") and "provisional" not in s["headline"]),
                  ("the DEM for the hillshade exists",
                   (gp.d05 / "dem" / "DEM_COP_GLO30_30m.tif").exists())]
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
