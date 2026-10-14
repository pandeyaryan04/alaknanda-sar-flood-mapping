"""The U-Net and TerraMind changes of 7 October (night).

1. 23_train_unet.py and 30_train_terramind.py pick the best epoch, and stop
   early, by the valid IoU at each epoch's best threshold (--select-by
   best_threshold); --select-by 0.5 is the first run's rule.
2. 26_corridor_lowo.py: U-Net and TerraMind folds choose their threshold on
   one training window, predicted by an extra model trained on the others.
3. unet_lr_test.py, archive_unet_first_run.py, RUN UNET REDO.bat.
"""
import csv

import numpy as np
import torch

import models_common as mc
from conftest import FYP, UNET_FAST, copy_project, run, run_cli

lowo = mc.load_script("26_corridor_lowo.py")
LOWO_UNET = ["--steps", "4", "--crop", "32", "--batch-size", "4", "--base-width", "8",
             "--tile", "64", "--overlap", "16", "--cpu-ok"]


def _history(d):
    with open(d / "history.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _best_epoch_matches(d, column):
    rows = _history(d)
    vals = [float(r[column]) for r in rows]
    best = 0
    for i, v in enumerate(vals):          # first epoch, then strictly better ones
        if i == 0 or v > vals[best]:
            best = i
    improved = [r["improved"] == "True" for r in rows]
    meta = mc.load_json(d / "model_meta.json")
    return meta["best_epoch"] == best + 1 and improved[0] and \
        all(improved[i] == (i == 0 or vals[i] > max(vals[:i])) for i in range(len(vals)))


def test_unet_best_epoch_by_best_threshold(project, tmp_path):
    root = copy_project(project, tmp_path / "unet select")
    args = [a if a != "2" else "4" for a in UNET_FAST]                  # 4 epochs
    assert run("23_train_unet.py", ["--project", root, "--trainset", "hand"] + args) == 0
    d = mc.Paths(root).model_dir("unet", "hand")
    assert _best_epoch_matches(d, "val_iou_at_best_threshold")
    meta = mc.load_json(d / "model_meta.json")
    assert meta["best_epoch_selected_by"] == "valid IoU at the best threshold of each epoch"
    ck = torch.load(d / "last.pt", map_location="cpu", weights_only=False)
    assert ck["args"]["select_by"] == "best_threshold"


def test_unet_select_by_half_is_the_first_rule(project, tmp_path):
    root = copy_project(project, tmp_path / "unet select half")
    args = [a if a != "2" else "4" for a in UNET_FAST]
    assert run("23_train_unet.py", ["--project", root, "--trainset", "hand", "--select-by", "0.5"]
               + args) == 0
    d = mc.Paths(root).model_dir("unet", "hand")
    assert _best_epoch_matches(d, "val_iou_at_0.5")
    assert mc.load_json(d / "model_meta.json")["best_epoch_selected_by"] == "valid IoU at 0.5"


def test_resume_refuses_a_checkpoint_of_the_first_rule(project, tmp_path, capsys):
    root = copy_project(project, tmp_path / "unet resume old")
    base = ["--project", root, "--trainset", "hand"] + UNET_FAST
    assert run("23_train_unet.py", base + ["--stop-after-epochs", "1"]) == 0
    d = mc.Paths(root).model_dir("unet", "hand")
    ck = torch.load(d / "last.pt", map_location="cpu", weights_only=False)
    del ck["args"]["select_by"]                            # as the first run saved it
    torch.save(ck, d / "last.pt")
    assert run("23_train_unet.py", base + ["--resume"]) == 1
    assert "--select-by was None" in capsys.readouterr().out


def test_model_dir_option(project, tmp_path):
    root = copy_project(project, tmp_path / "unet model dir")
    other = tmp_path / "elsewhere" / "unet test run"
    assert run("23_train_unet.py", ["--project", root, "--trainset", "hand", "--model-dir", other]
               + UNET_FAST) == 0
    assert (other / "model_meta.json").exists()
    assert not mc.Paths(root).model_dir("unet", "hand").exists()


def test_terramind_uses_the_same_rule():
    text = (FYP / "30_train_terramind.py").read_text(encoding="utf-8")
    assert '"select_by")' in text and 'default="best_threshold"' in text
    assert 'score > st["best_score"]' in text


def test_inner_window_choice():
    win = np.array([[1, 1, 2, 2, 3, 3, 4, 4]])
    lab = np.array([[0, 0, 0, 0, 1, 0, 1, 0]], "int8")
    train = lowo.fold_train_mask(win, lab, 1)
    assert lowo.inner_window(win, lab, train, 1) == 3        # 2 holds no water
    train = lowo.fold_train_mask(win, lab, 4)
    assert lowo.inner_window(win, lab, train, 4) == 3        # after 4: none, round to 1..3
    lab0 = np.zeros_like(lab)
    assert lowo.inner_window(win, lab0, lowo.fold_train_mask(win, lab0, 1), 1) is None


def test_lowo_unet_inner_threshold(project, tmp_path):
    root = copy_project(project, tmp_path / "lowo unet inner")
    p = mc.Paths(root)
    assert run("26_corridor_lowo.py", ["--project", root, "--model", "unet", "--radar", "grd"]
               + LOWO_UNET) == 0
    d = p.lowo_dir(2021, "unet", "grd")
    thr = {}
    for k in range(1, 5):
        rec = mc.load_json(d / f"fold_{k}" / "run_record.json")
        j = rec["inner_window"]
        assert j != k and j in rec["train_windows"]
        assert k not in rec["inner_train_windows"] and j not in rec["inner_train_windows"]
        assert "one training window" in rec["threshold_how"]
        assert 0 < rec["threshold"] < 1
        thr[str(k)] = rec["threshold"]
    asm = mc.load_json(d / "run_record.json")
    assert asm["threshold_per_fold"] == thr


def test_lowo_unet_fixed_threshold_still_works(project, tmp_path):
    root = copy_project(project, tmp_path / "lowo unet fixed")
    p = mc.Paths(root)
    assert run("26_corridor_lowo.py", ["--project", root, "--model", "unet", "--radar", "grd",
                                       "--threshold", "0.5"] + LOWO_UNET) == 0
    rec = mc.load_json(p.lowo_dir(2021, "unet", "grd") / "fold_1" / "run_record.json")
    assert rec["threshold"] == 0.5 and "inner_window" not in rec


def test_lr_test_selftest():
    r = run_cli([FYP / "unet_lr_test.py", "--selftest"])
    assert r.returncode == 0 and "SELFTEST PASS" in r.stdout, r.stdout[-3000:] + r.stderr[-2000:]
