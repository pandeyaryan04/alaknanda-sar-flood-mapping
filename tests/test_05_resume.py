"""Brief 9.5: stop after epoch 1, resume, reach epoch 2 with exactly the same state."""
import torch

import models_common as mc
from conftest import UNET_FAST, copy_project, run


def _state(root):
    return torch.load(mc.Paths(root).model_dir("unet", "hand") / "last.pt", weights_only=False)


def _same(a, b):
    sa, sb = a["model"], b["model"]
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_resume_after_epoch_matches_uninterrupted(trained, tmp_path):
    ref = _state(trained)                                    # 2 epochs, no stop
    other = copy_project(trained, tmp_path / "resume data")
    base = ["--project", other, "--trainset", "hand"] + UNET_FAST
    assert run("23_train_unet.py", base + ["--force", "--stop-after-epochs", "1"]) == 0
    mid = _state(other)
    assert mid["epoch"] == 1 and mid["step_in_epoch"] == 0
    rec = mc.load_json(mc.Paths(other).model_dir("unet", "hand") / "run_record.json")
    assert rec["finished"] is False
    assert run("23_train_unet.py", base + ["--resume"]) == 0
    end = _state(other)
    assert end["epoch"] == ref["epoch"] == 2
    assert _same(end, ref)
    assert end["history"] == ref["history"]
    assert torch.equal(torch.as_tensor(end["optimizer"]["state"][0]["exp_avg"]),
                       torch.as_tensor(ref["optimizer"]["state"][0]["exp_avg"]))


def test_resume_mid_epoch_matches_uninterrupted(trained, tmp_path):
    """The Ctrl+C path: stop inside epoch 2, save, resume, same result."""
    ref = _state(trained)
    other = copy_project(trained, tmp_path / "ctrl c data")
    base = ["--project", other, "--trainset", "hand"] + UNET_FAST
    assert run("23_train_unet.py", base + ["--force", "--stop-after-steps", "4"]) == 0
    mid = _state(other)
    assert mid["epoch"] == 1 and mid["step_in_epoch"] == 1
    assert run("23_train_unet.py", base + ["--resume"]) == 0
    end = _state(other)
    assert _same(end, ref) and end["history"] == ref["history"]


def test_finished_run_is_not_retrained(trained):
    assert run("23_train_unet.py", ["--project", trained, "--trainset", "hand", "--resume",
                                    "--skip-existing"] + UNET_FAST) == 0
