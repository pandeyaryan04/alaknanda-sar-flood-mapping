"""Brief 9.4: RF and U-Net train on CPU, save, reload, predict; same seed, same result."""
import numpy as np
import torch

import models_common as mc
from conftest import RF_FAST, UNET_FAST, copy_project, run


def _valid_chip(root):
    p = mc.Paths(root)
    row = mc.split_rows(mc.load_chip_index(p), "valid")[0]
    x, y = mc.read_chip(row, p.data)
    return x, y


def test_rf_saved_reloaded_predicts(trained):
    p = mc.Paths(trained)
    d = p.model_dir("rf", "hand")
    for f in ("model.joblib", "model_meta.json", "norm_stats.json", "run_record.json",
              "train_chips.txt"):
        assert (d / f).exists(), f
    meta = mc.load_json(d / "model_meta.json")
    assert 0 < meta["threshold_valid"] < 1 and "iou_valid_at_0.5" in meta
    pred = mc.load_predictor("rf", d)
    x, y = _valid_chip(trained)
    v = mc.input_valid(x)
    prob = pred.predict_image(x, v)
    assert np.isnan(prob[~v]).all() and np.all((prob[v] >= 0) & (prob[v] <= 1))
    tv = v & (y >= 0)
    iou = mc.metrics_from_confusion(*mc.confusion(prob[tv] >= 0.5, y[tv] == 1))["iou"]
    assert iou > 0.5, iou                      # the synthetic water is easy


def test_rf_same_seed_same_predictions(trained, tmp_path):
    other = copy_project(trained, tmp_path / "copy of data")
    assert run("22_train_rf.py", ["--project", other, "--trainset", "hand", "--force"] + RF_FAST) == 0
    x, _ = _valid_chip(trained)
    v = mc.input_valid(x)
    a = mc.load_predictor("rf", mc.Paths(trained).model_dir("rf", "hand")).predict_image(x, v)
    b = mc.load_predictor("rf", mc.Paths(other).model_dir("rf", "hand")).predict_image(x, v)
    assert np.array_equal(a, b, equal_nan=True)


def test_rf_hand_weak_uses_weak_chips(trained, tmp_path):
    other = copy_project(trained, tmp_path / "hw data")
    assert run("22_train_rf.py", ["--project", other, "--trainset", "hand_weak"] + RF_FAST) == 0
    chips = (mc.Paths(other).model_dir("rf", "hand_weak") / "train_chips.txt").read_text().split()
    assert "Ghana_5001" in chips and "Bolivia_5005" not in chips and "Mekong_5006" not in chips
    assert "Ghana_2001" not in chips and "Mekong_3001" not in chips


def test_unet_saved_reloaded_predicts(trained):
    d = mc.Paths(trained).model_dir("unet", "hand")
    for f in ("best.pt", "last.pt", "history.csv", "model_meta.json", "norm_stats.json",
              "run_record.json", "train_chips.txt"):
        assert (d / f).exists(), f
    rec = mc.load_json(d / "run_record.json")
    assert rec["finished"] is True and rec["epochs_done"] == 2
    pred = mc.load_predictor("unet", d, cpu_ok=True, tile=64, overlap=16)
    x, _ = _valid_chip(trained)
    v = mc.input_valid(x)
    prob = pred.predict_image(x, v)
    assert np.isnan(prob[~v]).all() and np.isfinite(prob[v]).all()


def test_unet_same_seed_same_first_loss(trained, tmp_path):
    losses = []
    for k in range(2):
        other = copy_project(trained, tmp_path / f"copy {k}")
        assert run("23_train_unet.py", ["--project", other, "--trainset", "hand", "--force",
                                        "--stop-after-steps", "1"] + UNET_FAST) == 0
        ck = torch.load(mc.Paths(other).model_dir("unet", "hand") / "last.pt",
                        weights_only=False)
        losses.append(ck["first_step_loss"])
    ref = mc.load_json(mc.Paths(trained).model_dir("unet", "hand") / "run_record.json")
    assert losses[0] == losses[1] == ref["first_step_loss"]


def test_unet_with_spawned_workers(trained, tmp_path):
    """Two DataLoader workers started with spawn, as on Windows: everything must pickle."""
    other = copy_project(trained, tmp_path / "workers data")
    argv = [a for a in UNET_FAST]
    argv[argv.index("--workers") + 1] = "2"
    argv[argv.index("--epochs") + 1] = "1"
    assert run("23_train_unet.py", ["--project", other, "--trainset", "hand_weak"] + argv) == 0
    rec = mc.load_json(mc.Paths(other).model_dir("unet", "hand_weak") / "run_record.json")
    assert rec["finished"] and rec["weak_chips"] == 5


def test_unet_refuses_cpu_without_flag(trained, tmp_path):
    if torch.cuda.is_available():
        return
    other = copy_project(trained, tmp_path / "cpu data")
    argv = [a for a in UNET_FAST if a != "--cpu-ok"]
    assert run("23_train_unet.py", ["--project", other, "--trainset", "hand", "--force"] + argv) == 1


def test_no_overwrite_without_force(trained):
    rc = run("22_train_rf.py", ["--project", trained, "--trainset", "hand"] + RF_FAST)
    assert rc == 1
    rc = run("22_train_rf.py", ["--project", trained, "--trainset", "hand", "--skip-existing"]
             + RF_FAST)
    assert rc == 0
