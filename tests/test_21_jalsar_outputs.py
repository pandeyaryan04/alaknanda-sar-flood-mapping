"""Jal-SAR (04 Code/jalsar): web map, bulletin, runner and their launchers."""
import argparse
import ast
import sys

import numpy as np
import pytest

from conftest import REPO, run_cli

GJ = REPO / "jalsar"
sys.path.insert(0, str(GJ))
import jalsar_bulletin as gb      # noqa: E402
import jalsar_core as gc          # noqa: E402
import jalsar_display as gd       # noqa: E402
import jalsar_runner as gr        # noqa: E402

MODULES = ("jalsar_display.py", "jalsar_testdata.py", "jalsar_webmap.py",
           "jalsar_bulletin.py", "jalsar_runner.py")


@pytest.mark.parametrize("name", MODULES)
def test_selftest(name):
    r = run_cli([GJ / name, "--selftest"], cwd=GJ)
    assert r.returncode == 0 and "SELFTEST PASS" in r.stdout, r.stdout[-3000:] + r.stderr[-2000:]


@pytest.mark.parametrize("name", MODULES)
def test_conventions(name):
    tree = ast.parse((GJ / name).read_text(encoding="utf-8"))
    doc = ast.get_docstring(tree)
    assert doc and "USAGE" in doc.upper()
    names = {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets
             if isinstance(t, ast.Name)}
    assert {"__stage__", "__reads__", "__writes__"} <= names


def test_help_lists_new_commands():
    r = run_cli([GJ / "jalsar.py", "--help"], cwd=GJ)
    assert r.returncode == 0
    for cmd in ("bulletin", "webmap", "run", "check-download"):
        assert cmd in r.stdout


def test_template_wording():
    t = (GJ / "templates" / "bulletin.html.j2").read_text(encoding="utf-8")
    assert "—" not in t and "–" not in t
    for must in ("Census", "OpenStreetMap contributors", "What this bulletin is not",
                 "There is no reference map for this date"):
        assert must in t


def test_class_labels():
    assert gb.class_label("incidence", "lt30") == "under 30 degrees"
    assert gb.class_label("hand", "5_15m") == "5 to 15 m"
    assert gb.class_label("valley_width", "gt1000m") == "over 1000 m"


def test_spots_and_insets():
    hb = np.full((600, 900), 255, "uint8")
    hb[10:40, 10:40] = 3            # 900 cells in one 1 km square
    hb[500:520, 800:820] = 1        # 400 cells far away
    sp = gb.spots(hb, 1e-4)
    assert [s["trust"] for s in sp] == ["high", "low"]
    ins = gb.pick_insets(sp, hb.shape)
    assert len(ins) == 2 and ins[0] == (0, 0)


def test_footer_needs_no_browser(tmp_path):
    from pypdf import PdfReader, PdfWriter
    src = tmp_path / "blank one.pdf"
    w = PdfWriter()
    w.add_blank_page(595, 842)
    w.add_blank_page(595, 842)
    with open(src, "wb") as fh:
        w.write(fh)
    n = gb.add_footer(src, tmp_path / "out.pdf")
    txt = PdfReader(str(tmp_path / "out.pdf")).pages[1].extract_text()
    assert n == 2 and "Page 2 of 2" in txt and "Aryan Pandey" in txt


def test_env_example_parses():
    env = gr.read_env(GJ / "jalsar.env.example")
    s = gr.settings(argparse.Namespace(date="2025-08-14", search_days=None, min_coverage=None,
                                       families=None, pre_event=None), env)
    assert s["search_days"] == 3 and s["min_coverage"] == 99.5 and not s["pre_event"]
    assert s["families"] == ("rf", "unet", "terramind")


def test_env_example_in_git_and_env_ignored():
    gi = (REPO / ".gitignore").read_text(encoding="utf-8")
    assert "*.env" in gi
    raw = (GJ / "jalsar.env.example").read_bytes()
    assert raw.count(b"\n") == raw.count(b"\r\n")


def test_bad_settings_stop():
    with pytest.raises(SystemExit):
        gr.settings(argparse.Namespace(date="14-08-2025", search_days=None, min_coverage=None,
                                       families=None, pre_event=None), dict(gr.DEFAULTS))


def test_runner_grid_is_the_export_grid():
    g = gr.GRID
    assert (g["x0"], g["y0"], g["cell"], g["width"], g["height"]) == (231660.0, 3357010.0, 10.0, 7915, 3044)
    assert g["x0"] + g["cell"] * g["width"] == 310810 and g["y0"] - g["cell"] * g["height"] == 3326570
    assert sum(w * h for _, _, w, h in gr.tiles(7915, 3044, 1024)) == 7915 * 3044


def test_runner_never_writes_05_data(tmp_path):
    gp = gc.GPaths(tmp_path / "Final Year Project")
    for p in (gp.gj / "scenes", gp.gj / "downloads", gp.roads):
        assert "05 Data" not in str(p)


@pytest.mark.parametrize("name,words", [
    ("RUN JAL-SAR.bat", ["python jalsar.py run", "set /p D=", "conda activate fyp", "pause"]),
    ("RUN JAL-SAR CHECK DOWNLOAD.bat", ["python jalsar.py check-download", "CHECK REPORT.md"]),
    ("RUN JAL-SAR MAPS.bat", ["python jalsar.py bulletin --event all",
                               "python jalsar.py webmap --force"]),
    ("RUN JAL-SAR MAPS AGAIN.bat", ["python jalsar.py maps --event all --force",
                                     "python jalsar.py bulletin --event all --force",
                                     "python jalsar.py webmap --force", "conda activate fyp",
                                     "pause", "JAL-SAR MAPS AGAIN PASS"]),
])
def test_bats(name, words):
    raw = (GJ / name).read_bytes()
    assert b"\r\n" in raw and raw.count(b"\n") == raw.count(b"\r\n")
    text = raw.decode("ascii")
    for w in ["chcp 65001", "PYTHONUTF8=1", 'cd /d "%~dp0"'] + words:
        assert w in text, w


def test_display_colours():
    assert gd.MAROON == "#6d1a2b"
    assert set(gd.TRUST_COLOURS) == {0, 1, 2, 3}


def test_family_uses_hand_even_if_hand_weak_scores_higher(tmp_path):
    import json
    gp = gc.GPaths(tmp_path / "Final Year Project")
    for model, ts, iou in (("unet", "hand", 0.645), ("unet", "hand_weak", 0.679),
                           ("tm_s1", "hand", 0.60), ("tm_s1t", "hand", 0.62),
                           ("tm_s1t", "hand_weak", 0.70)):
        d = gp.model_dir(model, ts)
        d.mkdir(parents=True)
        (d / "model_meta.json").write_text(json.dumps({"iou_valid_at_threshold": iou}))
    assert gc.family_member(gp, "unet") == ("unet", "hand", 0.645)
    assert gc.family_member(gp, "terramind") == ("tm_s1t", "hand", 0.62)
    assert gc.family_member(gp, "rf") is None
