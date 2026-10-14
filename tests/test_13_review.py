"""Tests added in the review pass, one per fix."""
import csv
import json

import numpy as np
import pytest
import torch

import models_common as mc
import models_synthetic as syn
from conftest import FYP, copy_project, make_project, run, run_cli


def test_oom_probe_finds_the_largest_batch_that_fits():
    """CUDA out of memory at the first batch: the exact --batch-size is found."""
    class Fussy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(1))

        def forward(self, x):
            if x.shape[0] > 2:
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            return x[:, 0] * self.w

    assert mc.find_batch_size(Fussy(), torch.device("cpu"), 32, 8, amp=False) == 2
    assert mc.is_oom(RuntimeError("CUDA out of memory")) and not mc.is_oom(RuntimeError("x"))


def test_test_and_bolivia_labels_never_read_before_evaluation(tmp_path, monkeypatch):
    root = tmp_path / "strict data"
    syn.build_project(root, corridor=False)
    read = []
    real = mc.read_label

    def spy(path, window=None):
        read.append(str(path))
        return real(path, window)

    monkeypatch.setattr(mc, "read_label", spy)
    assert run("21_prepare_chips.py", ["--project", root, "--cache"]) == 0
    p = mc.Paths(root)
    held = set(mc.read_split_csv(p.splits / "flood_test_data.csv")) | \
        set(mc.read_split_csv(p.splits / "flood_bolivia_data.csv"))
    assert read and not [f for f in read if any(f"{c}_LabelHand" in f for c in held)]
    for r in mc.load_chip_index(p):
        if r["split"] in ("test", "bolivia"):
            assert r["cache_x"] == "" and r["labelled_pixels"] == 0
            assert r["kept"] == "yes" or r["reason"].startswith("empty")


def test_preflight_does_not_read_test_labels(tmp_path):
    root = make_project(tmp_path / "pf strict", prepare=False)
    p = mc.Paths(root)
    f = p.label_dir("hand") / "Mekong_3001_LabelHand.tif"          # a TEST chip
    import rasterio
    with rasterio.open(f, "r+") as d:
        a = d.read(1)
        a[0, 0] = 5
        d.write(a, 1)
    pf = mc.load_script("20_models_preflight.py")
    res = pf.run_checks(p, syn.EXPECTED_SYNTHETIC, machine=False)
    assert not [r for r in res if r["status"] == "FAIL"]


def test_cache_reads_are_writable_copies(project, tmp_path):
    root = copy_project(project, tmp_path / "cache data")
    assert run("21_prepare_chips.py", ["--project", root, "--cache", "--skip-existing"]) == 0
    p = mc.Paths(root)
    row = next(r for r in mc.load_chip_index(p) if r["split"] == "train" and r["kept"] == "yes")
    assert row["cache_x"]
    x, y = mc.read_chip(row, p.data, (0, 0, 16, 16))
    assert x.flags.writeable and y.flags.writeable and not isinstance(y, np.memmap)
    x, y = mc.read_chip(row, p.data)
    assert y.flags.writeable and not isinstance(y, np.memmap)


def test_label_on_another_grid_stops_21_with_a_sentence(tmp_path):
    root = tmp_path / "grid data"
    syn.build_project(root, corridor=False)
    p = mc.Paths(root)
    f = p.label_dir("hand") / "Ghana_1001_LabelHand.tif"
    syn._write(f, [np.zeros((32, 64))], "int16", "EPSG:4326",
               syn.Affine(syn.PX, 0, 10, 0, -syn.PX, 20))
    with pytest.raises(SystemExit) as e:
        mc.load_script("21_prepare_chips.py").main(["--project", str(root)])
    assert "PROBLEM" in str(e.value) and "Ghana_1001" in str(e.value)
    assert "20_models_preflight.py" in str(e.value)


def test_threshold_hist_ignores_nan():
    h = mc.ThresholdHist()
    h.add(np.array([np.nan, 0.9, 0.1]), np.array([False, True, False]))
    assert h.nonfinite == 1 and h.counts(0.5) == (1, 0, 0, 1)


def test_json_numpy_scalars(tmp_path):
    mc.save_json(tmp_path / "a.json", {"f": np.float32(0.25), "i": np.int64(3),
                                       "a": np.arange(2)})
    d = json.loads((tmp_path / "a.json").read_text())
    assert d == {"f": 0.25, "i": 3, "a": [0, 1]}


def test_inf_inputs_become_the_mean():
    st = mc.compute_norm_stats(np.random.default_rng(0).normal(size=(500, 4)), "t")
    n = mc.normalise(np.array([np.inf, -np.inf, 1.0, 0.0], "float32").reshape(4, 1, 1), st)
    assert n[0, 0, 0] == 0 and n[1, 0, 0] == 0
    f = mc.rf_features(np.array([[np.inf, -np.inf, 1.0, 0.0]], "float32"), st)
    assert f[0, 0] == np.float32(st["mean"][0]) and f[0, 1] == np.float32(st["mean"][1])


def test_rf_fold_with_one_class_still_trains():
    lowo = mc.load_script("26_corridor_lowo.py")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 20, 30)).astype("float32")
    win = np.zeros((20, 30), int)
    win[:, :10], win[:, 10:20], win[:, 20:] = 1, 2, 3
    lab = np.zeros((20, 30), "int8")                     # no water anywhere
    train = lowo.fold_train_mask(win, lab, 3)
    stats = mc.compute_norm_stats(x[:, train].T, "t")

    class A:
        seed, n_per_class, n_estimators, min_samples_leaf, n_jobs = 1, 50, 5, 5, 1

    rec = {}
    pm = win[:, 20:] == 3
    prob = lowo.rf_fold(A, x, lab, win, 3, train, stats, (0, 20, 20, 30), pm, rec)
    assert rec["classes"] == [0] and "trained anyway" in rec["note"]
    assert (prob[pm] == 0).all()


def test_lowo_fold_without_training_cells_stops_clearly(project, tmp_path):
    root = copy_project(project, tmp_path / "no labels")
    p = mc.Paths(root)
    import geopandas as gpd
    from shapely.geometry import box
    # a cloud hole over every window: no labelled cell anywhere
    g = gpd.GeoDataFrame({"source": ["x"], "note": ["x"]},
                         geometry=[box(-1e7, -1e7, 1e7, 1e7)], crs="EPSG:32644")
    g.to_file(p.holes(2021), layer="holes", driver="GPKG")
    with pytest.raises(SystemExit) as e:
        mc.load_script("26_corridor_lowo.py").main(["--project", str(root), "--model", "rf",
                                                    "--n-estimators", "5"])
    assert "nothing to train on" in str(e.value)


def test_best_pt_is_a_full_checkpoint(trained):
    ck = torch.load(mc.Paths(trained).model_dir("unet", "hand") / "best.pt", weights_only=False)
    for k in ("model", "optimizer", "scaler", "scheduler", "rng", "epoch", "history"):
        assert k in ck, k
    assert ck["epoch"] == ck["best_epoch"]


def test_compare_values_equal_what_04_writes(project, tmp_path):
    """27's numbers for map A are exactly the scorer's numbers for map A."""
    p = mc.Paths(project)
    a = p.otsu(2021, "single")
    out04 = tmp_path / "score a"
    r = run_cli([FYP / "04_stratified_metrics.py", "--pred", a, "--truth", p.reference(2021),
                 "--strata", p.strata(2021), "--windows", p.windows(2021), "--out", out04])
    assert r.returncode == 0, r.stderr
    cmp_out = tmp_path / "cmp"
    assert run("27_compare_maps.py", ["--a", a, "--b", p.otsu(2021, "chfeb"), "--truth",
                                      p.reference(2021), "--strata", p.strata(2021), "--windows",
                                      p.windows(2021), "--out", cmp_out]) == 0
    with open(out04 / "metrics_by_stratum.csv", encoding="utf-8") as fh:
        s04 = next(r for r in csv.DictReader(fh) if r["dimension"] == "ALL")
    with open(cmp_out / "comparison.csv", encoding="utf-8") as fh:
        s27 = next(csv.DictReader(fh))
    # 04 scores all cells where A and truth are 0/1; 27 where A, B and truth are.
    # The Otsu maps share one valid mask, so the cell sets are identical here.
    for k in ("f1", "f1_lo", "f1_hi", "iou", "mcc", "mcc_lo"):
        assert float(s04[k]) == float(s27[f"a_{k}"]), k


def test_ctrl_c_ends_cleanly(capsys):
    def boom():
        raise KeyboardInterrupt
    assert mc.cli(boom) == 130
    assert "STOPPED by Ctrl+C" in capsys.readouterr().out


def test_stop_flag_first_press_saves_second_press_stops():
    unet = mc.load_script("23_train_unet.py")
    f = unet.StopFlag()
    try:
        f._handler(2, None)
        assert f.flag
        with pytest.raises(KeyboardInterrupt):
            f._handler(2, None)
    finally:
        f.restore()


def test_rf_ram_check_before_loading(trained):
    d = mc.Paths(trained).model_dir("rf", "hand")
    meta = mc.load_json(d / "model_meta.json")
    assert meta["forest_nodes"] > 0
    with pytest.raises(SystemExit) as e:
        mc.load_predictor("rf", d, max_ram_gb=0.5)
    assert "--max-ram-gb" in str(e.value)


def test_diagnostics_name_the_model_run_record(trained, tmp_path):
    root = copy_project(trained, tmp_path / "diag data")
    assert run("25_predict_corridor.py", ["--project", root, "--model", "rf", "--year", "2023",
                                          "--force"]) == 0
    p = mc.Paths(root)
    out = p.flood_map(2023, "rf", "hand")
    d = json.loads((out.parent / f"{out.stem}_diagnostics.json").read_text())
    assert d["model_run_record"].endswith("run_record.json")
    rec = json.loads((out.parent / "run_record.json").read_text())
    assert rec["outputs"][1]["sha256"] == mc.sha256(out.parent / f"{out.stem}_diagnostics.json")


def test_no_output_ever_in_05_data(project, tmp_path):
    """Run records list every output: none may be inside 05 Data."""
    root = copy_project(project, tmp_path / "w data")
    assert run("26_corridor_lowo.py", ["--project", root, "--model", "rf",
                                       "--n-estimators", "5"]) == 0
    p = mc.Paths(root)
    before = sorted(f.stat().st_mtime_ns for f in p.d05.rglob("*") if f.is_file())
    rec = mc.load_json(p.lowo_dir(2021, "rf", "grd") / "run_record.json")
    assert all("05 Data" not in o["path"] for o in rec["outputs"])
    after = sorted(f.stat().st_mtime_ns for f in p.d05.rglob("*") if f.is_file())
    assert before == after


def test_rf_prediction_bit_identical_across_calls():
    """sklearn's threaded predict_proba can differ in the 16th digit; ours never does."""
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20000, 4)).astype("float32")
    y = (X[:, 0] + rng.normal(0, 1, 20000) > 0).astype("int8")
    clf = RandomForestClassifier(n_estimators=40, min_samples_leaf=5, random_state=7,
                                 n_jobs=4).fit(X, y)
    a = mc.rf_water_prob(clf, X, 4)
    b = mc.rf_water_prob(clf, X, 3)
    clf.set_params(n_jobs=1)
    c = clf.predict_proba(X)[:, 1].astype("float32")
    assert np.array_equal(a, b) and np.array_equal(a, c)


def test_rf_warm_start_fit_equals_one_fit():
    from sklearn.ensemble import RandomForestClassifier
    rf22 = mc.load_script("22_train_rf.py")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3000, 4)).astype("float32")
    y = (X[:, 0] + rng.normal(0, 1, 3000) > 0).astype("int8")

    class A:
        n_estimators, min_samples_leaf, n_jobs, seed = 25, 5, 2, 7

    a = mc.rf_water_prob(rf22.fit_forest(X, y, A), X)
    b = mc.rf_water_prob(RandomForestClassifier(n_estimators=25, min_samples_leaf=5,
                                                max_features="sqrt", random_state=7).fit(X, y), X)
    assert np.array_equal(a, b)


def test_smoke_evaluation_reads_no_test_labels(trained, tmp_path, monkeypatch):
    root = copy_project(trained, tmp_path / "smoke eval")
    assert run("21_prepare_chips.py", ["--project", root, "--smoke"]) == 0
    assert run("22_train_rf.py", ["--project", root, "--smoke"]) == 0
    read = []
    real = mc.read_label
    monkeypatch.setattr(mc, "read_label", lambda p, w=None: (read.append(str(p)), real(p, w))[1])
    assert run("24_eval_sen1floods11.py", ["--project", root, "--smoke", "--model", "rf"]) == 0
    p = mc.Paths(root)
    held = set(mc.read_split_csv(p.splits / "flood_test_data.csv")) | \
        set(mc.read_split_csv(p.splits / "flood_bolivia_data.csv"))
    assert read and not [f for f in read if any(f"{c}_LabelHand" in f for c in held)]
