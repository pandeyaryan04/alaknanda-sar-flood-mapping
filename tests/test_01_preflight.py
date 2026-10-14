"""Brief 9.1: the preflight passes on good data and fails, with the right message, on broken data."""
import socket

import models_common as mc
import models_synthetic as syn
from conftest import make_project

pf = mc.load_script("20_models_preflight.py")


def fails(root):
    res = pf.run_checks(mc.Paths(root), syn.EXPECTED_SYNTHETIC, machine=False)
    return [r for r in res if r["status"] == "FAIL"], res


def test_good_data_passes(project):
    bad, res = fails(project)
    assert not bad, bad
    names = " ".join(r["name"] for r in res)
    for word in ("S1Hand", "split", "terrain", "GRD", "RTC", "strata", "permanent",
                 "reference", "windows", "GeoPackage", "holes"):
        assert word in names, f"no check mentions {word}"


def test_report_written(tmp_path):
    root = make_project(tmp_path / "pf data", prepare=False)
    rc = pf.main(["--project", str(root)], expected=syn.EXPECTED_SYNTHETIC)
    assert rc == 0
    rep = root / "06 Outputs" / "models" / f"preflight_{socket.gethostname()}.md"
    assert rep.exists() and "PREFLIGHT PASS" in rep.read_text(encoding="utf-8")
    # a second run never overwrites the first report without --force
    assert pf.main(["--project", str(root)], expected=syn.EXPECTED_SYNTHETIC) == 0
    assert len(list((root / "06 Outputs" / "models").glob("preflight_*.md"))) == 2


def test_wrong_band_count_fails(tmp_path):
    root = make_project(tmp_path / "pf data", prepare=False)
    syn.break_band_count(root)
    bad, _ = fails(root)
    assert any("Ghana_1001" in r["detail"] and "3 band" in r["detail"] for r in bad), bad


def test_label_value_5_fails(tmp_path):
    root = make_project(tmp_path / "pf data", prepare=False)
    syn.break_label_value(root)
    bad, _ = fails(root)
    assert any("India_1002" in r["detail"] and "5" in r["detail"] for r in bad), bad


def test_missing_terrain_fails(tmp_path):
    root = make_project(tmp_path / "pf data", prepare=False)
    syn.break_missing_terrain(root)
    bad, _ = fails(root)
    assert any("Mekong_1003" in r["detail"] and "terrain" in r["name"].lower() for r in bad), bad
    assert any("11_sen1floods11_terrain.py" in r["detail"] for r in bad)


def test_cli_exit_code_on_failure(tmp_path):
    root = make_project(tmp_path / "pf data", prepare=False)
    syn.break_label_value(root)
    assert pf.main(["--project", str(root)], expected=syn.EXPECTED_SYNTHETIC) == 1
