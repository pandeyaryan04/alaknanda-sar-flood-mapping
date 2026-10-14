"""Brief 9.7: the flood-map contract (uint8, nodata 255, exact grid, masks)."""
import json

import numpy as np
import rasterio

import models_common as mc
from conftest import run


def test_rf_map_contract(trained):
    p = mc.Paths(trained)
    assert run("25_predict_corridor.py", ["--project", trained, "--model", "rf", "--trainset",
                                          "hand", "--year", "2021"]) == 0
    out = p.flood_map(2021, "rf", "hand")
    _check_map(p, out, 2021, "grd")
    diag = json.loads((out.parent / f"{out.stem}_diagnostics.json").read_text())
    for k in ("valid_cells", "water_raw_cells", "permanent_removed_cells",
              "distortion_removed_cells", "flood_cells", "flood_km2", "threshold",
              "run_record"):
        assert k in diag, k
    assert diag["permanent_removed_cells"] > 0 and diag["distortion_removed_cells"] > 0
    assert diag["flood_cells"] == diag["water_raw_cells"] - diag["permanent_removed_cells"] \
        - diag["distortion_removed_cells"]
    assert (out.parent / "run_record.json").exists()


def test_unet_map_contract_rtc(trained):
    p = mc.Paths(trained)
    assert run("25_predict_corridor.py", ["--project", trained, "--model", "unet", "--trainset",
                                          "hand", "--year", "2023", "--radar", "rtc",
                                          "--cpu-ok", "--tile", "64", "--overlap", "16"]) == 0
    _check_map(p, p.flood_map(2023, "unet", "hand", "rtc"), 2023, "rtc")


def _check_map(p, out, year, radar):
    scene = p.scene(year, radar)
    with rasterio.open(out) as m, rasterio.open(scene) as s:
        assert m.count == 1 and m.dtypes[0] == "uint8" and m.nodata == 255
        assert m.crs == s.crs and m.transform == s.transform and m.shape == s.shape
        assert m.profile.get("compress", "").lower() == "lzw"
        a = m.read(1)
        vv = s.read(1)
    assert set(np.unique(a)) <= {0, 1, 255}
    assert np.array_equal(a == 255, ~np.isfinite(vv))
    prof = {"height": a.shape[0], "width": a.shape[1], "transform": m.transform, "crs": m.crs}
    perm = mc.reproject_onto(p.perm_water(), prof, categorical=True) == 100
    ls = mc.reproject_onto(p.strata(year) / "strata_layover_shadow.tif", prof, categorical=True)
    valid = np.isfinite(vv)
    assert (a[valid & perm] == 0).all() and (valid & perm).any()
    dist = np.isin(ls, (1, 2))
    assert (a[valid & dist] == 0).all() and (valid & dist).any()


def test_make_flood_map_counts():
    prob = np.array([[0.9, 0.9, 0.9, 0.1, np.nan]], "float32")
    valid = np.isfinite(prob)
    perm = np.array([[1, 0, 1, 0, 0]], bool)
    dist = np.array([[0, 1, 1, 0, 0]], bool)
    m, d = mc.make_flood_map(prob, valid, 0.5, perm, dist)
    assert m.tolist() == [[0, 0, 0, 0, 255]]
    assert (d["water_raw_cells"], d["permanent_removed_cells"], d["distortion_removed_cells"],
            d["flood_cells"]) == (3, 2, 1, 0)


def test_threshold_is_inclusive():
    m, _ = mc.make_flood_map(np.array([[0.5]], "float32"), np.ones((1, 1), bool), 0.5,
                             np.zeros((1, 1), bool), np.zeros((1, 1), bool))
    assert m[0, 0] == 1
