"""Brief 9.10: paired comparison: A vs A is 0 with zero width; a better B is positive."""
import csv

import numpy as np
import rasterio

import models_common as mc
from conftest import run


def _rows(out):
    with open(out / "comparison.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _cmp(p, a, b, out):
    return run("27_compare_maps.py", ["--a", a, "--b", b, "--truth", p.reference(2021),
                                      "--strata", p.strata(2021), "--windows", p.windows(2021),
                                      "--label-a", "A", "--label-b", "B", "--out", out])


def test_a_vs_a_is_zero(project, tmp_path):
    p = mc.Paths(project)
    a = p.otsu(2021, "single")
    out = tmp_path / "cmp a a"
    assert _cmp(p, a, a, out) == 0
    rows = _rows(out)
    pooled = next(r for r in rows if r["reach"] == "pooled" and r["dimension"] == "ALL")
    for k in ("f1", "iou", "mcc"):
        assert float(pooled[f"diff_{k}"]) == 0.0
        assert float(pooled[f"diff_{k}_lo"]) == 0.0 == float(pooled[f"diff_{k}_hi"])
    assert (out / "comparison.md").exists()


def test_better_b_is_positive(project, tmp_path):
    p = mc.Paths(project)
    a = p.otsu(2021, "single")
    with rasterio.open(a) as s:
        prof, am = s.profile, s.read(1)
    with rasterio.open(p.reference(2021)) as s:
        t = s.read(1)
    b = am.copy()
    sc = np.isin(t, (0, 1)) & np.isin(am, (0, 1))
    b[sc] = t[sc]                                         # perfect where scored
    bp = tmp_path / "better map.tif"
    with rasterio.open(bp, "w", **prof) as d:
        d.write(b, 1)
    out = tmp_path / "cmp a b"
    assert _cmp(p, a, bp, out) == 0
    pooled = next(r for r in _rows(out) if r["reach"] == "pooled" and r["dimension"] == "ALL")
    assert float(pooled["diff_f1"]) > 0 and float(pooled["b_f1"]) == 1.0
    assert float(pooled["diff_f1_lo"]) > 0
    assert "B is better" in (out / "comparison.md").read_text(encoding="utf-8")


def test_same_bootstrap_as_scorer():
    cmp = mc.load_script("27_compare_maps.py")
    sc = mc.load_script("04_stratified_metrics.py")
    assert cmp.window_counts is sc.window_counts and cmp.metrics_from_counts is sc.metrics_from_counts
    w1 = cmp.boot_weights(8)
    rng = np.random.default_rng(sc.BOOT_SEED)
    draws = rng.integers(0, 8, size=(sc.N_BOOT, 8))
    assert np.array_equal(w1, np.stack([np.bincount(d, minlength=8) for d in draws]))
