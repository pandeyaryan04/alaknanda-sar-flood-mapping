"""Brief 9.11: self-tests, --help, compiling, .bat files, conventions."""
import ast
import py_compile
import re

import pytest

from conftest import FYP, REPO, run_cli

NEW = ["20_models_preflight.py", "21_prepare_chips.py", "22_train_rf.py", "23_train_unet.py",
       "24_eval_sen1floods11.py", "25_predict_corridor.py", "26_corridor_lowo.py",
       "27_compare_maps.py", "28_models_summary.py"]
MODULES = ["models_common.py", "models_synthetic.py"]
BATS = ["RUN MODELS PREFLIGHT.bat", "RUN MODELS SMOKE.bat", "RUN ALL MODELS.bat"]


@pytest.mark.parametrize("name", NEW + ["models_common.py"])
def test_selftest(name):
    r = run_cli([FYP / name, "--selftest"])
    assert r.returncode == 0 and "SELFTEST PASS" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]


@pytest.mark.parametrize("name", NEW)
def test_help(name):
    r = run_cli([FYP / name, "--help"])
    assert r.returncode == 0 and "usage" in r.stdout.lower()


def test_every_py_compiles(tmp_path):
    for f in list(FYP.glob("*.py")) + list((REPO / "tests").glob("*.py")):
        py_compile.compile(str(f), cfile=str(tmp_path / (f.stem + ".pyc")), doraise=True)


@pytest.mark.parametrize("name", NEW + MODULES)
def test_conventions(name):
    text = (FYP / name).read_text(encoding="utf-8")
    assert all(ord(c) < 128 for c in text), "non-ASCII character"
    tree = ast.parse(text)
    doc = ast.get_docstring(tree)
    assert doc and "USAGE" in doc.upper()
    names = {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets
             if isinstance(t, ast.Name)}
    assert {"__stage__", "__reads__", "__writes__"} <= names
    if name in NEW:
        assert "if __name__ == \"__main__\":" in text
        cmds = [ln.strip() for ln in doc.splitlines() if ln.strip().startswith("python " + name)]
        assert cmds, "no copy-paste command in the docstring"
        assert not any(c.endswith("^") or c.endswith("\\") for c in cmds)


@pytest.mark.parametrize("bat", BATS)
def test_bat_files(bat):
    raw = (FYP / bat).read_bytes()
    assert b"\r\n" in raw and raw.count(b"\n") == raw.count(b"\r\n"), "not CRLF throughout"
    text = raw.decode("ascii")
    for word in ("chcp 65001", "PYTHONUTF8=1", "KMP_DUPLICATE_LIB_OK=TRUE", 'cd /d "%~dp0"',
                 "miniconda3", "conda activate fyp", "FAILS", "pause"):
        assert word in text, word
    for script in set(re.findall(r"python\s+\"?([0-9A-Za-z_]+\.py)", text)):
        assert (FYP / script).exists(), script
