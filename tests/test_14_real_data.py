"""Findings on the owner's real data (7 October 2026), each reproduced on synthetic data.

1. Sen1Floods11 label files differ from their chips' grids by up to 1.6e-10 pixel
   (rounding in the headers): 445 of 446 hand labels and 1747 of 4384 weak labels.
   The preflight compared grids exactly and failed 22 of 50 sampled chips.
"""
import numpy as np
import rasterio
from rasterio.transform import Affine

import models_common as mc
import models_synthetic as syn
from conftest import copy_project, make_project, run

pf = mc.load_script("20_models_preflight.py")


def _nudge(path, dx_px, a_scale=1.0):
    """Rewrite a raster with its origin moved by dx_px pixels (and pixel size scaled)."""
    with rasterio.open(path) as s:
        arr, t, crs, dt, nod = s.read(), s.transform, s.crs, s.dtypes[0], s.nodata
        names = s.descriptions
    t2 = Affine(t.a * a_scale, t.b, t.c + dx_px * t.a, t.d, t.e, t.f)
    syn._write(path, list(arr), dt, crs, t2, nodata=nod,
               names=list(names) if any(names) else None)


def test_same_grid_tolerates_rounding_only():
    t = Affine(syn.PX, 0, 10, 0, -syn.PX, 20)
    assert mc.same_grid((512, 512), t, "EPSG:4326", (512, 512),
                        Affine(t.a, 0, t.c + 1.6e-10 * t.a, 0, t.e, t.f), "EPSG:4326")
    assert mc.same_grid((512, 512), t, "EPSG:4326", (512, 512),
                        Affine(t.a * (1 + 1e-13), 0, t.c, 0, t.e, t.f), "EPSG:4326")
    assert not mc.same_grid((512, 512), t, "EPSG:4326", (512, 512),
                            Affine(t.a, 0, t.c + 0.3 * t.a, 0, t.e, t.f), "EPSG:4326")
    assert not mc.same_grid((512, 512), t, "EPSG:4326", (512, 512),
                            Affine(t.a * 1.001, 0, t.c, 0, t.e, t.f), "EPSG:4326")   # 0.5 px drift
    assert not mc.same_grid((512, 512), t, "EPSG:4326", (511, 512), t, "EPSG:4326")
    assert not mc.same_grid((512, 512), t, "EPSG:4326", (512, 512), t, "EPSG:32644")


def test_preflight_passes_rounded_label_grids(tmp_path):
    root = make_project(tmp_path / "rounded", prepare=False)
    p = mc.Paths(root)
    for f in list(p.label_dir("hand").glob("*.tif")) + list(p.label_dir("weak").glob("*.tif")):
        _nudge(f, 1.6e-10)
    res = pf.run_checks(p, syn.EXPECTED_SYNTHETIC, machine=False)
    assert not [r for r in res if r["status"] == "FAIL"]
    assert any(r["name"] == "labels: same grid as the chip" and r["status"] == "PASS" for r in res)


def test_preflight_still_fails_a_real_shift(tmp_path):
    root = make_project(tmp_path / "shifted", prepare=False)
    p = mc.Paths(root)
    _nudge(p.label_dir("hand") / "Ghana_1001_LabelHand.tif", 0.3)
    res = pf.run_checks(p, syn.EXPECTED_SYNTHETIC, machine=False)
    bad = [r for r in res if r["status"] == "FAIL"]
    assert len(bad) == 1 and "Ghana_1001 (0.3 px)" in bad[0]["detail"]
    assert "diagnose_label_grid.py" in bad[0]["detail"]


def test_chain_runs_on_rounded_grids(tmp_path):
    """21 and 22 accept labels whose header differs by rounding; 26 and 27 accept
    GRD/RTC and maps that differ by rounding."""
    root = tmp_path / "rounded chain" / "Final Year Project"
    syn.build_project(root)
    p = mc.Paths(root)
    for f in p.label_dir("hand").glob("*.tif"):
        _nudge(f, 1.6e-10)
    _nudge(p.scene(2021, "rtc"), 1e-9)
    assert run("21_prepare_chips.py", ["--project", root]) == 0
    assert run("22_train_rf.py", ["--project", root, "--n-estimators", "5",
                                  "--n-per-class", "100"]) == 0
    assert run("26_corridor_lowo.py", ["--project", root, "--model", "rf",
                                       "--n-estimators", "5"]) == 0
    other = tmp_path / "rounded map.tif"
    with rasterio.open(p.otsu(2021)) as s:
        arr, prof = s.read(1), s.profile
    t = prof["transform"]
    prof["transform"] = Affine(t.a, 0, t.c + 1e-9 * t.a, 0, t.e, t.f)
    with rasterio.open(other, "w", **prof) as d:
        d.write(arr, 1)
    assert run("27_compare_maps.py", ["--a", p.otsu(2021), "--b", other, "--truth",
                                      p.reference(2021), "--strata", p.strata(2021),
                                      "--windows", p.windows(2021),
                                      "--out", tmp_path / "cmp rounded"]) == 0


def test_lowo_refuses_a_shifted_rtc(project, tmp_path):
    root = copy_project(project, tmp_path / "rtc shifted")
    p = mc.Paths(root)
    _nudge(p.scene(2021, "rtc"), 0.5)
    import pytest
    with pytest.raises(SystemExit) as e:
        mc.load_script("26_corridor_lowo.py").main(["--project", str(root), "--model", "rf",
                                                    "--n-estimators", "5"])
    assert "not on the same grid" in str(e.value)


# 2. Neighbouring Sen1Floods11 chips meet only to the last floating-point digit:
#    98 of the 571 weak chips that touch a held-out chip were missed (review, checked
#    on the public bucket's headers). And on the owner's laptop 25_predict_corridor.py
#    aborted with "OMP: Error #15" (two OpenMP runtimes in env fyp).

PX = 8.983152841195215e-05


def test_touching_boxes_found_despite_rounding():
    held = [(0.0, 0.0, 512 * PX, 512 * PX)]
    right = held[0][2]
    eps = mc.GRID_TOL_PX * PX
    cases = {
        "edge, 1 ulp gap": (np.nextafter(right, 1.0), 0.0, right + 512 * PX, 512 * PX),
        "edge, 1.6e-10 px gap": (right + 1.6e-10 * PX, 0.0, right + 512 * PX, 512 * PX),
        "corner only": (right + 1e-12 * PX, 512 * PX + 1e-12 * PX, right + 512 * PX, 1024 * PX),
        "overlap": (100 * PX, 100 * PX, 600 * PX, 600 * PX),
    }
    for name, box in cases.items():
        assert mc.boxes_intersect([box], held, eps).tolist() == [True], name
    assert mc.boxes_intersect([cases["edge, 1.6e-10 px gap"]], held).tolist() == [False]
    apart = (right + 1.0 * PX, 0.0, right + 513 * PX, 512 * PX)          # one pixel away
    assert mc.boxes_intersect([apart], held, eps).tolist() == [False]


def test_21_drops_weak_chip_that_touches_a_valid_chip(tmp_path):
    root = tmp_path / "touching" / "Final Year Project"
    syn.build_project(root, corridor=False)
    p = mc.Paths(root)
    with rasterio.open(p.s1_dir("hand") / "Ghana_2001_S1Hand.tif") as s:        # a VALID chip
        b, t = s.bounds, s.transform
    rng = np.random.default_rng(5)
    for chip, x0 in (("Spain_6001", b.right + 1.6e-10 * t.a),    # touches it, rounding gap
                     ("Spain_6002", b.right + 3 * t.a)):          # three pixels away
        tf = Affine(t.a, 0, x0, 0, t.e, t.f)
        vv, vh, lab, hand, slope = syn._chip_arrays(rng, "good", weak=True)
        syn._write(p.s1_dir("weak") / f"{chip}_S1Weak.tif", [vv, vh], "float32", "EPSG:4326", tf)
        syn._write(p.label_dir("weak") / f"{chip}_S2IndexLabelWeak.tif", [lab], "int16",
                   "EPSG:4326", tf)
        syn._write(p.terrain_dir("weak") / f"{chip}_Terrain.tif", [hand, slope], "float32",
                   "EPSG:4326", tf, nodata=np.nan, names=["hand_m", "slope_deg"])
    assert run("21_prepare_chips.py", ["--project", root]) == 0
    idx = {r["chip"]: r for r in mc.load_chip_index(p)}
    assert idx["Spain_6001"]["reason"] == "overlaps a hand valid, test or Bolivia chip"
    assert idx["Spain_6002"]["kept"] == "yes"


def test_openmp_duplicate_allowed_before_numpy_starts():
    import os
    import sys
    from conftest import FYP, run_cli
    src = (FYP / "models_common.py").read_text(encoding="utf-8")
    assert src.index('os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")') < \
        src.index("import numpy as np")
    assert os.environ.get("KMP_DUPLICATE_LIB_OK") == "TRUE"
    r = run_cli(["-c", "import models_common, os; print(os.environ['KMP_DUPLICATE_LIB_OK'])"])
    assert r.returncode == 0 and r.stdout.strip() == "TRUE", r.stderr
    for bat in FYP.glob("RUN MODELS *.bat"):
        text = bat.read_bytes().decode("ascii")
        assert text.index("set KMP_DUPLICATE_LIB_OK=TRUE") < text.index("python ")
    assert sys.modules["models_common"]
