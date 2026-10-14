"""Repeat runs (9 October 2026): --run-tag for repeat runs with other seeds,
32_seed_summary.py and RUN SEEDS.bat. The original runs' paths must not change."""
import ast
import re

import pytest

import models_common as mc
from conftest import FYP, UNET_FAST, copy_project, run, run_cli


def test_paths_without_tag_are_unchanged(tmp_path):
    p = mc.Paths(tmp_path / "Final Year Project")
    assert p.model_dir("unet", "hand") == p.models / "unet_hand"
    assert p.model_dir("tm_s1t", "hand_weak") == p.models / "tm_s1t_hand_weak"
    assert p.flood_name(2021, "unet", "hand") == "flood_2021_unet_hand"
    assert p.flood_name(2023, "rf", "hand_weak", "rtc") == "flood_2023_rf_hand_weak_rtc"
    assert p.flood_map(2021, "unet", "hand") == \
        p.data / "flood_2021_unet_hand" / "flood_2021_unet_hand.tif"


def test_paths_with_tag(tmp_path):
    p = mc.Paths(tmp_path / "Final Year Project")
    assert p.model_dir("unet", "hand", "seed2") == p.models / "unet_hand_seed2"
    assert p.flood_name(2021, "tm_s1t", "hand", "grd", "seed3") == "flood_2021_tm_s1t_hand_seed3"
    assert p.flood_map(2023, "unet", "hand", tag="seed2") == \
        p.data / "flood_2023_unet_hand_seed2" / "flood_2023_unet_hand_seed2.tif"


class _A:
    def __init__(self, tag, seed=mc.SEED):
        self.run_tag, self.seed = tag, seed


def test_check_run_tag():
    assert mc.check_run_tag(_A("")) == ""
    assert mc.check_run_tag(_A("seed2", mc.SEED + 1)) == "seed2"
    assert mc.check_run_tag(_A("seed2"), seed_used=False) == "seed2"     # eval, predict
    for bad in ("Seed2", "seed_2", "seed 2", "../x", "a" * 17):
        with pytest.raises(SystemExit):
            mc.check_run_tag(_A(bad, mc.SEED + 1))
    with pytest.raises(SystemExit, match="original seed"):
        mc.check_run_tag(_A("seed2"))                                    # training, same seed


def test_unet_repeat_run_end_to_end(trained, tmp_path):
    """A tagged U-Net run trains, evaluates, maps and calibrates into its own folders
    and leaves the original run and its outputs exactly as they were."""
    root = copy_project(trained, tmp_path / "copy of project")
    paths = mc.Paths(root)
    orig = paths.model_dir("unet", "hand")
    before = {f.name: mc.sha256(f) for f in orig.iterdir() if f.is_file()}
    assert run("23_train_unet.py", ["--project", root, "--trainset", "hand", "--seed",
                                    str(mc.SEED + 1), "--run-tag", "seed2"] + UNET_FAST) == 0
    tagged = paths.model_dir("unet", "hand", "seed2")
    assert (tagged / "model_meta.json").exists()
    rec = mc.load_json(tagged / "run_record.json")
    assert rec["seed"] == mc.SEED + 1 and rec["run_tag"] == "seed2" and rec["finished"]
    assert {f.name: mc.sha256(f) for f in orig.iterdir() if f.is_file()} == before
    # the original seed under a tag is refused, and nothing is written
    assert run("23_train_unet.py", ["--project", root, "--trainset", "hand", "--run-tag",
                                    "seed9"] + UNET_FAST) != 0
    assert not paths.model_dir("unet", "hand", "seed9").exists()
    assert run("24_eval_sen1floods11.py", ["--project", root, "--model", "unet", "--run-tag",
                                           "seed2", "--cpu-ok", "--tile", "64",
                                           "--overlap", "16"]) == 0
    assert (paths.tables / "s1f11_unet_hand_seed2" / "metrics.md").exists()
    assert not (paths.tables / "s1f11_unet_hand").exists()
    assert run("25_predict_corridor.py", ["--project", root, "--model", "unet", "--run-tag",
                                          "seed2", "--year", "2021", "--cpu-ok", "--tile",
                                          "64", "--overlap", "16"]) == 0
    assert paths.flood_map(2021, "unet", "hand", tag="seed2").exists()
    assert not paths.flood_map(2021, "unet", "hand").exists()
    assert run("31_calibration.py", ["--project", root, "--models", "unet", "--trainsets",
                                     "hand", "--run-tag", "seed2", "--cpu-ok", "--tile", "64",
                                     "--overlap", "16", "--per-chip", "500"]) == 0
    assert (paths.tables / "calibration_summary_seed2.md").exists()
    assert not (paths.tables / "calibration_summary.md").exists()
    assert (paths.out / "models" / "calibration" / "unet_hand_seed2" / "calibration.json").exists()


def test_terramind_script_has_run_tag():
    text = (FYP / "30_train_terramind.py").read_text(encoding="utf-8")
    assert "mc.add_run_tag_arg(ap)" in text
    assert "paths.model_dir(args.variant, args.trainset, tag)" in text
    assert "run_tag=tag" in text


def test_seed_summary_selftest():
    r = run_cli([FYP / "32_seed_summary.py", "--selftest"])
    assert r.returncode == 0 and "SELFTEST PASS" in r.stdout, r.stdout[-3000:] + r.stderr[-2000:]


def test_seed_summary_conventions():
    text = (FYP / "32_seed_summary.py").read_text(encoding="utf-8")
    assert all(ord(c) < 128 for c in text)
    tree = ast.parse(text)
    doc = ast.get_docstring(tree)
    assert doc and "USAGE" in doc.upper()
    names = {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets
             if isinstance(t, ast.Name)}
    assert {"__stage__", "__reads__", "__writes__"} <= names


def test_run_seeds_bat():
    raw = (FYP / "RUN SEEDS.bat").read_bytes()
    assert b"\r\n" in raw and raw.count(b"\n") == raw.count(b"\r\n")
    text = raw.decode("ascii")
    for word in ("chcp 65001", "PYTHONUTF8=1", "KMP_DUPLICATE_LIB_OK=TRUE", 'cd /d "%~dp0"',
                 "conda activate fyp", "FAILS", "pause", "lr_test\\lr_choice.txt",
                 "--lr %LR%", "call :model unet seed2 20261008",
                 "call :model tm_s1t seed2 20261008", "call :model unet seed3 20261009",
                 "call :model tm_s1t seed3 20261009", "python 32_seed_summary.py --force",
                 "seeds_compare_", "--run-tag %2", "--resume --skip-existing"):
        assert word in text, word
    for script in set(re.findall(r"python\s+\"?([0-9A-Za-z_]+\.py)", text)):
        assert (FYP / script).exists(), script
    # comparisons of the repeat runs stay out of MODELS_SUMMARY.md (28 reads compare_*)
    assert '06 Outputs\\tables\\compare_' not in text
    # the original seed is never used for a repeat run
    assert str(mc.SEED) not in text.split("rem Every finished step")[1]


def _bat_compare_dirs(bat, prefix):
    """Folder names a launcher's :cmp calls write, for one year and (for RUN SEEDS.bat) one tag."""
    text = (FYP / bat).read_bytes().decode("ascii")
    return text, re.findall(r"call :cmp %1 (\S+) \S+ (\S+) ", text)


@pytest.mark.parametrize("year", [2021, 2023])
def test_seed_summary_reads_the_folders_the_launchers_write(year, tmp_path):
    """Found 9 October: 32 looked for compare_..._unet_vs_tm_s1t, the launchers wrote
    compare_..._unet_hand_vs_tm_s1t_hand, so every comparison showed as missing."""
    ss = mc.load_script("32_seed_summary.py")
    tables = tmp_path
    # RUN SEEDS.bat: labels use %2 for the tag
    _, pairs = _bat_compare_dirs("RUN SEEDS.bat", "seeds_compare")
    for tag in ("seed2", "seed3"):
        written = {f"seeds_compare_{year}_{a.replace('%2', tag)}_vs_{b.replace('%2', tag)}"
                   for a, b in pairs}
        for a, b, _ in ss.COMPARISONS:
            assert ss.compare_dir(tables, year, a, b, tag).name in written, (a, b, tag)
    # the original runs: the comparison list of RUN ALL MODELS.bat
    _, pairs = _bat_compare_dirs("RUN ALL MODELS.bat", "compare")
    written = {f"compare_{year}_{a}_vs_{b}" for a, b in pairs}
    for a, b, _ in ss.COMPARISONS:
        assert ss.compare_dir(tables, year, a, b, "").name in written, (a, b)
