"""Brief 9.9: leave one window out on the corridor."""
import numpy as np
import rasterio

import models_common as mc
from conftest import copy_project, run

LOWO_FAST_RF = ["--n-estimators", "10", "--n-per-class", "200"]
LOWO_FAST_UNET = ["--steps", "6", "--crop", "32", "--batch-size", "4", "--base-width", "8",
                  "--tile", "64", "--overlap", "16", "--cpu-ok"]
lowo = mc.load_script("26_corridor_lowo.py")


def _run(root, model, radar, year=2021, extra=()):
    fast = LOWO_FAST_RF if model == "rf" else LOWO_FAST_UNET
    return run("26_corridor_lowo.py", ["--project", root, "--year", year, "--model", model,
                                       "--radar", radar] + fast + list(extra))


def test_rf_folds_and_assembly(project, tmp_path):
    root = copy_project(project, tmp_path / "lowo data")
    p = mc.Paths(root)
    assert _run(root, "rf", "grd") == 0
    d = p.lowo_dir(2021, "rf", "grd")
    folds = sorted(f.name for f in d.glob("fold_*"))
    assert folds == ["fold_1", "fold_2", "fold_3", "fold_4"]
    for k in range(1, 5):
        rec = mc.load_json(d / f"fold_{k}" / "run_record.json")
        assert k not in rec["train_windows"] and rec["heldout_window"] == k
        assert rec["train_cells_in_heldout_window"] == 0
        assert rec["norm_stats"]["source"].startswith("corridor windows")
    with rasterio.open(p.lowo_map(2021, "rf", "grd")) as m, \
            rasterio.open(p.scene(2021, "grd")) as g, rasterio.open(p.scene(2021, "rtc")) as r:
        a = m.read(1)
        assert m.dtypes[0] == "uint8" and m.nodata == 255 and m.transform == g.transform
        common = np.isfinite(g.read(1)) & np.isfinite(r.read(1))
    win, _ = mc.load_script("04_stratified_metrics.py").window_raster(
        p.windows(2021), a.shape, m.transform, m.crs)
    inside = (win > 0) & common
    assert (a[~inside] == 255).all() and np.isin(a[inside], (0, 1)).all()
    assert (a == 1).any()


def test_common_mask_same_for_grd_and_rtc(project, tmp_path):
    root = copy_project(project, tmp_path / "lowo both")
    p = mc.Paths(root)
    assert _run(root, "rf", "grd") == 0 and _run(root, "rf", "rtc") == 0
    with rasterio.open(p.lowo_map(2021, "rf", "grd")) as a, \
            rasterio.open(p.lowo_map(2021, "rf", "rtc")) as b:
        assert np.array_equal(a.read(1) == 255, b.read(1) == 255)
    with rasterio.open(p.scene(2021, "rtc")) as r:
        shadow_nan = ~np.isfinite(r.read(1))
    with rasterio.open(p.lowo_map(2021, "rf", "grd")) as a:
        assert (a.read(1)[shadow_nan] == 255).all()


def test_fold_training_mask_never_holds_heldout():
    win = np.array([[1, 1, 2, 2, 0], [3, 3, 4, 4, 0]])
    lab = np.array([[1, 0, 1, 0, 1], [-1, 0, 1, 0, 1]], "int8")
    for k in (1, 2, 3, 4):
        m = lowo.fold_train_mask(win, lab, k)
        assert not (m & (win == k)).any() and not (m & (win == 0)).any()
        assert not (m & (lab < 0)).any()


def test_unet_lowo_and_window_without_water(project, tmp_path):
    """2023 has a window with no water; the U-Net folds must still finish."""
    root = copy_project(project, tmp_path / "lowo unet")
    assert _run(root, "unet", "grd", year=2023) == 0
    d = mc.Paths(root).lowo_dir(2023, "unet", "grd")
    assert len(list(d.glob("fold_*/run_record.json"))) == 4
    assert (d / "run_record.json").exists()


def test_resume_skips_finished_folds(project, tmp_path):
    root = copy_project(project, tmp_path / "lowo resume")
    p = mc.Paths(root)
    assert _run(root, "rf", "grd", extra=["--folds", "2"]) == 0
    first = (p.lowo_dir(2021, "rf", "grd") / "fold_2" / "run_record.json").stat().st_mtime_ns
    assert not p.lowo_map(2021, "rf", "grd").exists()          # only one fold asked
    assert _run(root, "rf", "grd") == 0
    again = (p.lowo_dir(2021, "rf", "grd") / "fold_2" / "run_record.json").stat().st_mtime_ns
    assert first == again and p.lowo_map(2021, "rf", "grd").exists()
