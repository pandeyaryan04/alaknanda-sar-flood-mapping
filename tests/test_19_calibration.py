"""31_calibration.py (calibration of the water probabilities)."""
import ast

import numpy as np

import models_common as mc
from conftest import FYP, run_cli

cal = mc.load_script("31_calibration.py")


def test_selftest():
    r = run_cli([FYP / "31_calibration.py", "--selftest"])
    assert r.returncode == 0 and "SELFTEST PASS" in r.stdout, r.stdout[-3000:] + r.stderr[-2000:]


def test_help():
    r = run_cli([FYP / "31_calibration.py", "--help"])
    assert r.returncode == 0 and "usage" in r.stdout.lower()


def test_conventions():
    text = (FYP / "31_calibration.py").read_text(encoding="utf-8")
    assert all(ord(c) < 128 for c in text)
    tree = ast.parse(text)
    doc = ast.get_docstring(tree)
    assert doc and "USAGE" in doc.upper()
    names = {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets
             if isinstance(t, ast.Name)}
    assert {"__stage__", "__reads__", "__writes__"} <= names


def test_ece_and_rescaling_maths():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 1.5, 100000)
    y = (rng.random(z.size) < cal.sigmoid(z)).astype("float64")
    assert cal.reliability(cal.sigmoid(z), y)["ece"] < 0.02
    p_over = cal.sigmoid(4 * z)                                   # overconfident
    assert cal.reliability(p_over, y)["ece"] > 0.05
    t = cal.fit_temperature(cal.logit(p_over), y)
    assert abs(t - 4) < 0.3
    assert cal.reliability(cal.sigmoid(cal.logit(p_over) / t), y)["ece"] < 0.02
    p_low = cal.sigmoid(z - 3)                                    # probabilities shifted low
    a, b = cal.fit_platt(cal.logit(p_low), y)
    assert abs(a - 1) < 0.05 and abs(b - 3) < 0.1
    thr_low, _ = cal.best_threshold(p_low, y)
    thr_fix, _ = cal.best_threshold(cal.sigmoid(a * cal.logit(p_low) + b), y)
    assert thr_low < thr_fix                                      # the threshold moves up
