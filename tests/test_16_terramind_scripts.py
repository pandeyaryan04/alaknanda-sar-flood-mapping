"""TerraMind scripts and launchers, and the
whole TerraMind chain through the very commands RUN TERRAMIND SMOKE.bat runs."""
import ast
import re
import shlex
import shutil

import pytest
import rasterio

import models_common as mc
import models_synthetic as syn
from conftest import FYP, SCRIPTS, make_project, run_cli

NEW = ["29_terramind_preflight.py", "30_train_terramind.py"]
BATS = ["RUN TERRAMIND PREFLIGHT.bat", "RUN TERRAMIND SMOKE.bat", "RUN ALL MODELS.bat"]


@pytest.mark.parametrize("name", NEW)
def test_selftest(name):
    r = run_cli([FYP / name, "--selftest"])
    assert r.returncode == 0 and "SELFTEST PASS" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]


@pytest.mark.parametrize("name", NEW)
def test_help(name):
    r = run_cli([FYP / name, "--help"])
    assert r.returncode == 0 and "usage" in r.stdout.lower()


@pytest.mark.parametrize("name", NEW)
def test_conventions(name):
    text = (FYP / name).read_text(encoding="utf-8")
    assert all(ord(c) < 128 for c in text), "non-ASCII character"
    tree = ast.parse(text)
    doc = ast.get_docstring(tree)
    assert doc and "USAGE" in doc.upper()
    names = {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets
             if isinstance(t, ast.Name)}
    assert {"__stage__", "__reads__", "__writes__"} <= names
    assert "if __name__ == \"__main__\":" in text
    cmds = [ln.strip() for ln in doc.splitlines() if ln.strip().startswith("python " + name)]
    assert cmds and not any(c.endswith("^") or c.endswith("\\") for c in cmds)


def test_terratorch_is_imported_only_when_needed():
    """DataLoader workers start by importing the training script (spawn): it must not
    import terratorch at the top, which takes 10 to 30 seconds."""
    r = run_cli(["-c", "import sys, importlib.util as u; s = u.spec_from_file_location('t', "
                       "'30_train_terramind.py'); m = u.module_from_spec(s); "
                       "s.loader.exec_module(m); print('terratorch' in sys.modules)"])
    assert r.returncode == 0 and r.stdout.strip() == "False", r.stderr


@pytest.mark.parametrize("bat", BATS)
def test_bat_files(bat):
    raw = (FYP / bat).read_bytes()
    assert b"\r\n" in raw and raw.count(b"\n") == raw.count(b"\r\n"), "not CRLF throughout"
    text = raw.decode("ascii")
    for word in ("chcp 65001", "PYTHONUTF8=1", "KMP_DUPLICATE_LIB_OK=TRUE", 'cd /d "%~dp0"',
                 "miniconda3", "conda activate fyp", "FAILS", "pause"):
        assert word in text, word
    assert text.index("set KMP_DUPLICATE_LIB_OK=TRUE") < text.index("python ")
    for script in set(re.findall(r"python\s+\"?([0-9A-Za-z_]+\.py)", text)):
        assert (FYP / script).exists(), script


def test_table_bats_cover_every_run():
    a = (FYP / "RUN ALL MODELS.bat").read_text(encoding="ascii")
    for v in mc.TM_VARIANTS:
        for ts in mc.TRAINSETS:
            assert f"call :model {v} {ts}" in a
    assert "--resume --skip-existing" in a and "21_prepare_chips.py --cache --skip-existing" in a
    d = a
    assert "for %%Y in (2021 2023) do for %%M in (rf unet tm_s1t tm_s1) do for %%R in (grd rtc)" in d
    c = a
    for a_, b_ in (("unet_hand_weak", "tm_s1t_hand_weak"), ("rf_hand_weak", "tm_s1t_hand_weak"),
                   ("tm_s1_hand", "tm_s1t_hand"), ("tm_s1_hand_weak", "tm_s1t_hand_weak"),
                   ("otsu", "tm_s1t_hand_weak"), ("otsu_chfeb", "tm_s1t_hand_weak"),
                   ("lowo_tm_s1_grd", "lowo_tm_s1_rtc"), ("lowo_tm_s1t_grd", "lowo_tm_s1t_rtc")):
        assert re.search(rf"call :cmp %1 {a_} \S+ {b_} ", c), (a_, b_)


def test_compare_bat_added_9_october():
    """The hand runs against both Otsu baselines, U-Net against tm_s1t, three Table D pairs."""
    c = (FYP / "RUN ALL MODELS.bat").read_text(encoding="ascii")
    pairs = [(b, m) for b in ("otsu", "otsu_chfeb")
             for m in ("rf_hand", "unet_hand", "tm_s1t_hand", "tm_s1_hand")]
    pairs += [("unet_hand", "tm_s1t_hand"), ("lowo_rf_grd", "lowo_unet_grd"),
              ("lowo_unet_grd", "lowo_tm_s1t_grd"), ("lowo_tm_s1_grd", "lowo_tm_s1t_grd")]
    for a_, b_ in pairs:
        assert len(re.findall(rf"call :cmp %1 {a_} \S+ {b_} ", c)) == 1, (a_, b_)
    outs = re.findall(r"call :cmp %1 (\S+) \S+ (\S+) ", c)
    assert len(outs) == len(set(outs)), "two comparisons would write the same folder"


def bat_commands(bat):
    cmds = []
    for line in (FYP / bat).read_text(encoding="ascii").splitlines():
        s = line.strip()
        if s.lower().startswith("python "):
            cmds.append(shlex.split(s.replace("\\", "/"))[1:])
    return cmds


def test_terramind_smoke_chain(tmp_path):
    root = make_project(tmp_path / "tm smoke run", prepare=False)
    syn.build_terramind_weights(root)
    code = root / "04 Code" / "FYP_code"
    for s in SCRIPTS + NEW:
        shutil.copy2(FYP / s, code / s)
    cmds = bat_commands("RUN TERRAMIND SMOKE.bat")
    assert len(cmds) == 13
    for args in cmds:
        r = run_cli(args, cwd=code)
        assert r.returncode == 0, f"{' '.join(args)}\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}"
    p = mc.Paths(root, smoke=True)
    for v in mc.TM_VARIANTS:
        assert (p.model_dir(v, "hand") / "best.pt").exists()
        assert mc.load_json(p.model_dir(v, "hand") / "model_meta.json")["crop"] == 224
        with rasterio.open(p.flood_map(2021, v, "hand")) as d:
            assert d.dtypes[0] == "uint8" and d.nodata == 255
    assert (p.model_dir("tm_s1t", "hand_weak") / "best.pt").exists()
    assert (p.tables / "2021_tm_s1t_hand" / "metrics_by_stratum.csv").exists()
    assert p.lowo_map(2021, "tm_s1t", "grd").exists() and p.lowo_map(2021, "tm_s1", "rtc").exists()
    assert (p.tables / "compare_2021_tm_s1_hand_vs_tm_s1t_hand" / "comparison.md").exists()
    assert (p.tables / "MODELS_SUMMARY.md").exists()
    # nothing of the smoke run lands in the real trees
    assert not (code / "data" / "models").exists()
    assert not (root / "06 Outputs" / "tables").exists()
    for f in tmp_path.rglob("*.pt"):                # about 280 MB each
        f.unlink()
