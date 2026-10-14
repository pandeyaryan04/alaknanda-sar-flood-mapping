"""Jal-SAR (04 Code/jalsar): the core and the command line."""
import ast
import re
import sys
from pathlib import Path

import numpy as np

from conftest import REPO, run_cli

GJ = REPO / "jalsar"
sys.path.insert(0, str(GJ))
import jalsar_core as gc      # noqa: E402


def test_core_selftest():
    r = run_cli([GJ / "jalsar_core.py", "--selftest"], cwd=GJ)
    assert r.returncode == 0 and "SELFTEST PASS" in r.stdout, r.stdout[-3000:] + r.stderr[-2000:]


def test_cli_selftest():
    r = run_cli([GJ / "jalsar.py", "--selftest"], cwd=GJ)
    assert r.returncode == 0 and "SELFTEST PASS" in r.stdout, r.stdout[-3000:] + r.stderr[-2000:]


def test_help():
    r = run_cli([GJ / "jalsar.py", "--help"], cwd=GJ)
    assert r.returncode == 0 and "usage" in r.stdout.lower()


def test_conventions():
    for name in ("jalsar_core.py", "jalsar.py"):
        text = (GJ / name).read_text(encoding="utf-8")
        tree = ast.parse(text)
        assert ast.get_docstring(tree) and "USAGE" in ast.get_docstring(tree).upper()
        names = {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets
                 if isinstance(t, ast.Name)}
        assert {"__stage__", "__reads__", "__writes__"} <= names


def test_trust_rule():
    g = gc.trust_grade
    assert [g(v) for v in (0.9, 0.40, 0.399, 0.20, 0.199, 0.0, float("nan"))] == \
        [3, 3, 2, 2, 1, 1, 1]


def test_agreement():
    valid = np.array([[True, True, True, False]])
    a = np.array([[1, 0, 1, 255]], "uint8")
    b = np.array([[1, 0, 0, 255]], "uint8")
    assert gc.agreement([a, b], valid).tolist() == [[2, 0, 1, 255]]


def test_bat():
    raw = (GJ / "RUN JAL-SAR MAPS.bat").read_bytes()
    assert b"\r\n" in raw and raw.count(b"\n") == raw.count(b"\r\n")
    text = raw.decode("ascii")
    for word in ("chcp 65001", "PYTHONUTF8=1", "KMP_DUPLICATE_LIB_OK=TRUE", 'cd /d "%~dp0"',
                 "conda activate fyp", "FAILS", "pause", "python jalsar.py roads",
                 "python jalsar.py maps --event all"):
        assert word in text, word
