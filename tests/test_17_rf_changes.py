"""The two Random Forest changes of 7 October (evening).

1. hand_weak: weak rows equal hand rows (50/50) by default; --weak-fill gives
   the first run's rule (the weak chips take the rest of the row budget).
2. Table D: up to 20000 cells per class per window, and the threshold chosen
   inside each fold from its training windows only (--threshold inner);
   --n-per-class 1000 --threshold 0.5 gives the first run's settings.
Also the archive helper and RUN RF REDO.bat.
The U-Net and TerraMind changes of the same night are in test_18.
"""

import numpy as np
import pytest
import rasterio

import models_common as mc
from conftest import FYP, RF_FAST, copy_project, run

lowo = mc.load_script("26_corridor_lowo.py")
rf = mc.load_script("22_train_rf.py")
FAST = ["--n-estimators", "10", "--n-per-class", "200", "--inner-estimators", "5", "--n-jobs", "2"]


def _rows(n, labelled, water_share):
    return [{"labelled_pixels": labelled, "water_share": water_share} for _ in range(n)]


def test_quotas_half_and_fill():
    hand, weak = _rows(250, 200_000, 0.1), _rows(3500, 200_000, 0.1)
    cap = 1_300_000
    qh, qw = rf.quotas(hand, weak, 1000, cap, 0.5)
    qh2, qw2 = rf.quotas(hand, weak, 1000, cap, 0.5, weak_fill=True)
    hand_rows, weak_rows = 2 * qh * len(hand), 2 * qw * len(weak)
    assert qh == qh2 == 1000
    assert weak_rows <= hand_rows and weak_rows > 0.95 * hand_rows      # about 50/50
    assert qw2 == (cap - hand_rows) // (2 * len(weak))                   # the first run's rule
    assert qw2 > qw


def test_hand_weak_trains_half_and_half(project, tmp_path):
    root = copy_project(project, tmp_path / "rf half")
    assert run("22_train_rf.py", ["--project", root, "--trainset", "hand_weak"] + RF_FAST) == 0
    meta = mc.load_json(mc.Paths(root).model_dir("rf", "hand_weak") / "model_meta.json")
    assert meta["weak_rows"] <= meta["hand_rows"] * 1.05
    assert meta["weak_rule"] == "hand fraction 0.5 of rows"
    rec = mc.load_json(mc.Paths(root).model_dir("rf", "hand_weak") / "run_record.json")
    assert rec["hand_rows"] == meta["hand_rows"] and rec["weak_rows"] == meta["weak_rows"]


def test_hand_weak_fill_reproduces_first_rule(project, tmp_path):
    root = copy_project(project, tmp_path / "rf fill")
    assert run("22_train_rf.py", ["--project", root, "--trainset", "hand_weak", "--weak-fill"]
               + RF_FAST) == 0
    meta = mc.load_json(mc.Paths(root).model_dir("rf", "hand_weak") / "model_meta.json")
    assert meta["weak_rule"].endswith("(--weak-fill)")


def test_hand_fraction_out_of_range_stops(project, tmp_path, capsys):
    root = copy_project(project, tmp_path / "rf bad fraction")
    assert run("22_train_rf.py", ["--project", root, "--trainset", "hand_weak",
                                  "--hand-fraction", "1"] + RF_FAST) == 1
    assert "--hand-fraction must be between 0 and 1" in capsys.readouterr().out
    assert not (mc.Paths(root).model_dir("rf", "hand_weak") / "model.joblib").exists()


def _fold_probs_and_thresholds(p, year, radar):
    d = p.lowo_dir(year, "rf", radar)
    out = []
    for f in sorted(d.glob("fold_*")):
        rec = mc.load_json(f / "run_record.json")
        k = rec["heldout_window"]
        with rasterio.open(f / f"prob_window_{k}.tif") as src:
            out.append((k, rec, src.read(1)))
    return out


def test_lowo_rf_inner_threshold(project, tmp_path):
    root = copy_project(project, tmp_path / "lowo inner")
    p = mc.Paths(root)
    assert run("26_corridor_lowo.py", ["--project", root, "--model", "rf", "--radar", "grd"]
               + FAST) == 0
    folds = _fold_probs_and_thresholds(p, 2021, "grd")
    assert len(folds) == 4
    win = None
    raw = 0
    for k, rec, prob in folds:
        assert "training window" in rec["threshold_how"]
        assert 0 < rec["threshold"] < 1
        assert rec["inner_windows"] == rec["train_windows"] and k not in rec["inner_windows"]
        assert rec["options"]["n_per_class"] == 200
        if win is None:
            with rasterio.open(p.lowo_map(2021, "rf", "grd")) as m:
                win, _ = mc.load_script("04_stratified_metrics.py").window_raster(
                    p.windows(2021), (m.height, m.width), m.transform, m.crs)
                fmap = m.read(1)
        r0, r1, c0, c1 = rec["box_rows_cols"]
        mine = (win[r0:r1, c0:c1] == k) & (fmap[r0:r1, c0:c1] != 255)
        raw += int((np.nan_to_num(prob, nan=0.0)[mine] >= rec["threshold"]).sum())
    asm = mc.load_json(p.lowo_dir(2021, "rf", "grd") / "run_record.json")
    diag = mc.load_json(p.lowo_dir(2021, "rf", "grd") / "flood_lowo_2021_rf_grd_diagnostics.json")
    assert set(asm["threshold_per_fold"]) == {"1", "2", "3", "4"}
    assert asm["threshold_how"].startswith("chosen inside each fold")
    assert diag["water_raw_cells"] == raw          # each fold cut at its own threshold


def test_lowo_rf_first_run_settings(project, tmp_path):
    root = copy_project(project, tmp_path / "lowo first")
    p = mc.Paths(root)
    assert run("26_corridor_lowo.py", ["--project", root, "--model", "rf", "--radar", "grd",
                                       "--threshold", "0.5"] + FAST) == 0
    for k, rec, _ in _fold_probs_and_thresholds(p, 2021, "grd"):
        assert rec["threshold"] == 0.5 and rec["threshold_how"].startswith("fixed in advance")
        assert "inner_windows" not in rec
    asm = mc.load_json(p.lowo_dir(2021, "rf", "grd") / "run_record.json")
    assert asm["threshold"] == 0.5


def test_threshold_option():
    for model in ("rf", "unet", "tm_s1", "tm_s1t"):
        assert lowo.parse_threshold(None, model) == "inner"
        assert lowo.parse_threshold("inner", model) == "inner"
    assert lowo.parse_threshold("0.3", "tm_s1") == 0.3
    for bad, model in (("1.5", "rf"), ("x", "unet"), ("0", "rf")):
        with pytest.raises(SystemExit):
            lowo.parse_threshold(bad, model)


def test_inner_threshold_with_one_class_uses_half():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 20, 30)).astype("float32")
    win = np.zeros((20, 30), int)
    win[:, :10], win[:, 10:20], win[:, 20:] = 1, 2, 3
    lab = np.zeros((20, 30), "int8")
    train = lowo.fold_train_mask(win, lab, 3)
    stats = mc.compute_norm_stats(x[:, train].T, "t")

    class A:
        seed, n_per_class, n_estimators, inner_estimators = 1, 50, 5, 3
        min_samples_leaf, n_jobs, threshold = 5, 1, "inner"

    rec = {}
    pm = win[:, 20:] == 3
    lowo.rf_fold(A, x, lab, win, 3, train, stats, (0, 20, 20, 30), pm, rec)
    assert rec["threshold"] == 0.5 and "one class" in rec["threshold_note"]
