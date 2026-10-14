"""Shared fixtures. Synthetic data only, built in folders with spaces in their names."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FYP = REPO / "FYP_code"
if str(FYP) not in sys.path:
    sys.path.insert(0, str(FYP))

import models_common as mc          # noqa: E402
import models_synthetic as syn      # noqa: E402

# Small settings so everything trains in seconds on a CPU.
RF_FAST = ["--n-estimators", "10", "--n-per-class", "200", "--n-jobs", "2"]
UNET_FAST = ["--base-width", "8", "--crop", "32", "--batch-size", "4", "--steps-per-epoch", "3",
             "--epochs", "2", "--tile", "64", "--overlap", "16", "--workers", "0", "--cpu-ok"]
SCRIPTS = ["models_common.py", "models_synthetic.py", "04_stratified_metrics.py",
           "06_rasterise_reference.py", "20_models_preflight.py", "21_prepare_chips.py",
           "22_train_rf.py", "23_train_unet.py", "24_eval_sen1floods11.py",
           "25_predict_corridor.py", "26_corridor_lowo.py", "27_compare_maps.py",
           "28_models_summary.py"]


def run(name, argv):
    """Run a numbered script's main() in this process; return its exit code."""
    return mc.run_script(name, [str(a) for a in argv])


def run_cli(args, cwd=FYP, timeout=900):
    """Run a command in a fresh Python process (as on the owner's machine)."""
    return subprocess.run([sys.executable] + [str(a) for a in args], cwd=str(cwd),
                          capture_output=True, text=True, timeout=timeout)


def make_project(base, name="Final Year Project", chips=True, corridor=True, prepare=True):
    root = Path(base) / name
    syn.build_project(root, chips=chips, corridor=corridor)
    if prepare and chips:
        assert run("21_prepare_chips.py", ["--project", root]) == 0
    return root


@pytest.fixture(scope="session")
def project(tmp_path_factory):
    """Synthetic project, chips prepared. Read-only for the tests that use it."""
    return make_project(tmp_path_factory.mktemp("syn data"))


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Synthetic project with a Random Forest and a U-Net trained on 'hand'."""
    root = make_project(tmp_path_factory.mktemp("trained data"))
    assert run("22_train_rf.py", ["--project", root, "--trainset", "hand"] + RF_FAST) == 0
    assert run("23_train_unet.py", ["--project", root, "--trainset", "hand"] + UNET_FAST) == 0
    return root


@pytest.fixture
def fresh(tmp_path):
    """A new synthetic project for a test that changes things."""
    return make_project(tmp_path / "my data")


def copy_project(src, dst):
    shutil.copytree(src, dst)
    return Path(dst)
