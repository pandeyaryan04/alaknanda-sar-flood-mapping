"""Organising of 9 October 2026: RUN ALL MODELS.bat and the repository front door.

The main check: the settings RUN ALL MODELS.bat would pass to each script give
the same values as the runs that produced the reported results
(provenance/RESULT_RUN_OPTIONS.json, copied from their run_record.json files).
"""
import argparse
import json
import re

import pytest

import models_common as mc
from conftest import FYP, REPO

BAT = FYP / "RUN ALL MODELS.bat"
RECORDED = json.loads((REPO / "provenance" / "RESULT_RUN_OPTIONS.json").read_text(encoding="utf-8"))
UNET_LR = "0.0003"


def bat_text():
    return BAT.read_bytes().decode("ascii")


def test_bat_runs_every_reported_run_in_order():
    t = bat_text()
    assert f"set UNET_LR={UNET_LR}" in t
    for m in ("rf", "unet", "tm_s1t", "tm_s1"):
        for ts in mc.TRAINSETS:
            assert len(re.findall(rf"^call :model {m} {ts}\r$", t, re.M)) == 1, (m, ts)
    loop = "for %%Y in (2021 2023) do for %%M in (rf unet tm_s1t tm_s1) do for %%R in (grd rtc)"
    assert loop in t
    order = [t.index(w) for w in ("python 21_prepare_chips.py", "call :model rf hand\r\n", loop,
                                  "call :year 2021", "python 28_models_summary.py --force",
                                  "python 31_calibration.py")]
    assert order == sorted(order)


def test_bat_unet_learning_rate_only_for_the_unet():
    t = bat_text()
    assert ('if "%1"=="unet" python 23_train_unet.py --trainset %2 --lr %UNET_LR% '
            '--resume --skip-existing') in t
    assert ('if "%2"=="unet" python 26_corridor_lowo.py --year %1 --model %2 --radar %3 '
            '--lr %UNET_LR% --skip-existing') in t
    assert ('if not "%2"=="unet" python 26_corridor_lowo.py --year %1 --model %2 --radar %3 '
            '--skip-existing') in t
    for line in t.splitlines():
        if "--lr" in line and line.lstrip().startswith(("if", "python")):
            assert "unet" in line, line


# The 18 comparisons of the separate launchers that made the reported results, in
# their order. RUN ALL MODELS.bat must run them first, unchanged; the ones added on
# 9 October come after.
OLD_COMPARISONS = [
    ("rf_hand", "unet_hand"), ("rf_hand_weak", "unet_hand_weak"), ("rf_hand", "rf_hand_weak"),
    ("unet_hand", "unet_hand_weak"), ("otsu", "rf_hand_weak"), ("otsu", "unet_hand_weak"),
    ("otsu_chfeb", "rf_hand_weak"), ("otsu_chfeb", "unet_hand_weak"),
    ("lowo_rf_grd", "lowo_rf_rtc"), ("lowo_unet_grd", "lowo_unet_rtc"),
    ("unet_hand_weak", "tm_s1t_hand_weak"), ("rf_hand_weak", "tm_s1t_hand_weak"),
    ("tm_s1_hand", "tm_s1t_hand"), ("tm_s1_hand_weak", "tm_s1t_hand_weak"),
    ("otsu", "tm_s1t_hand_weak"), ("otsu_chfeb", "tm_s1t_hand_weak"),
    ("lowo_tm_s1_grd", "lowo_tm_s1_rtc"), ("lowo_tm_s1t_grd", "lowo_tm_s1t_rtc"),
]


def test_bat_comparisons_are_the_old_launchers():
    pairs = re.findall(r"^call :cmp %1 (\S+) \S+ (\S+) \S+\r$", bat_text(), re.M)
    assert pairs[:len(OLD_COMPARISONS)] == OLD_COMPARISONS
    assert len(set(pairs)) == len(pairs)


class _Parsed(Exception):
    pass


def captured_args(script, argv, monkeypatch):
    """The namespace a script builds from argv, after its own resolve() if it has one."""
    mod = mc.load_script(script)
    box = {}
    real = argparse.ArgumentParser.parse_args

    def grab(self, args=None, namespace=None):
        box["ns"] = real(self, args, namespace)
        raise _Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", grab)
    with pytest.raises(_Parsed):
        mod.main(argv)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", real)
    ns = box["ns"]
    return vars(mod.resolve(ns)) if hasattr(mod, "resolve") else vars(ns)


def bat_argv(run):
    """The arguments RUN ALL MODELS.bat passes for a recorded run name."""
    if run.startswith("lowo_"):
        _, year, rest = run.split("_", 2)
        model, radar = rest.rsplit("_", 1)
        lr = ["--lr", UNET_LR] if model == "unet" else []
        return "26_corridor_lowo.py", ["--year", year, "--model", model, "--radar", radar] + lr \
            + ["--skip-existing"]
    for ts in ("hand_weak", "hand"):
        if run.endswith("_" + ts):
            model = run[:-len(ts) - 1]
            break
    if model == "rf":
        return "22_train_rf.py", ["--trainset", ts, "--skip-existing"]
    if model == "unet":
        return "23_train_unet.py", ["--trainset", ts, "--lr", UNET_LR, "--resume",
                                    "--skip-existing"]
    return "30_train_terramind.py", ["--variant", model, "--trainset", ts, "--resume",
                                     "--skip-existing"]


def test_recorded_runs_cover_every_result():
    runs = set(RECORDED["runs"])
    want = {f"{m}_{ts}" for m in mc.MODELS for ts in mc.TRAINSETS}
    want |= {f"lowo_{y}_{m}_{r}" for y in mc.YEARS for m in mc.MODELS for r in mc.RADARS}
    assert runs == want


@pytest.mark.parametrize("run", sorted(RECORDED["runs"]))
def test_bat_settings_match_the_recorded_run(run, monkeypatch):
    script, argv = bat_argv(run)
    if script == "30_train_terramind.py":
        pytest.importorskip("terratorch")
    got = captured_args(script, argv, monkeypatch)
    rec = RECORDED["runs"][run]["options"]
    compared = 0
    for key, value in rec.items():
        if key not in got or got[key] is None:
            continue                  # filled in later inside main(); the result shows it
        assert got[key] == value, f"{run}: {key} would be {got[key]!r}, the result used {value!r}"
        compared += 1
    assert compared >= 5, (run, compared)


def test_repo_root_keeps_only_the_front_door():
    top = {p.name for p in REPO.iterdir() if p.is_file()}
    assert not [n for n in top if n.startswith(("MODELS_", "SECOND_LAPTOP"))], top
    assert {"README.md", "RUN ORDER.md", "LICENSE"} <= top
