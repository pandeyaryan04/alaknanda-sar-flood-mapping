"""Brief 9.3: normalisation from hand train only; NaN handling; masks and label conventions."""
import numpy as np
import rasterio
from rasterio.transform import Affine

import models_common as mc
import models_synthetic as syn
from conftest import run


def test_stats_from_hand_train_only(tmp_path):
    root = tmp_path / "norm data"
    syn.build_project(root, corridor=False)
    p = mc.Paths(root)
    # make every chip that is NOT a hand train chip absurd: +100 dB
    train = set(mc.read_split_csv(p.splits / "flood_train_data.csv"))
    for set_ in ("hand", "weak"):
        for f in p.s1_dir(set_).glob("*.tif"):
            if f.name.replace("_S1Hand.tif", "") in train:
                continue
            with rasterio.open(f, "r+") as ds:
                ds.write(ds.read() + 100.0)
    assert run("21_prepare_chips.py", ["--project", root]) == 0
    s = mc.load_json(p.norm_stats)
    assert s["source"] == "hand train chips" and s["chips"] == 8
    assert s["clip_hi"][0] < 0 and s["clip_hi"][1] < 0       # no +100 dB chip got in
    assert s["clip_lo"][0] < s["mean"][0] < s["clip_hi"][0]


def test_normalise_nan_to_zero_and_clip():
    st = mc.compute_norm_stats(np.c_[np.linspace(-20, 0, 1001), np.linspace(-30, -5, 1001),
                                     np.linspace(0, 50, 1001), np.linspace(0, 40, 1001)], "t")
    x = np.array([np.nan, -1000.0, 1000.0, np.inf], "float32").reshape(4, 1, 1)
    n = mc.normalise(x, st)
    assert n[0, 0, 0] == 0.0 and n[3, 0, 0] == 0.0
    assert np.isclose(n[1, 0, 0], (st["clip_lo"][1] - st["mean"][1]) / st["std"][1], atol=1e-5)
    assert np.isclose(n[2, 0, 0], (st["clip_hi"][2] - st["mean"][2]) / st["std"][2], atol=1e-5)
    f = mc.rf_features(np.array([[np.nan, -1000, 1000, 5]], "float32"), st)
    assert f[0, 0] == np.float32(st["mean"][0]) and f[0, 1] == np.float32(st["clip_lo"][1])


def test_stats_ignore_nan_per_channel():
    s = np.array([[1, 2, np.nan, 4], [3, 4, 10, 6]] * 50, float)
    st = mc.compute_norm_stats(s, "t")
    assert st["finite_per_channel"] == [100, 100, 50, 100]
    assert np.isclose(st["mean"][2], 10.0)


def test_label_conventions(tmp_path):
    f = tmp_path / "lab el.tif"
    a = np.array([[-1, 0, 1, 255, 5]], "int16")
    with rasterio.open(f, "w", driver="GTiff", height=1, width=5, count=1, dtype="int16",
                       crs="EPSG:4326", transform=Affine(1, 0, 0, 0, -1, 1), nodata=255) as d:
        d.write(a, 1)
    assert mc.read_label(f).tolist() == [[-1, 0, 1, -1, -1]]


def test_header_nodata_becomes_nan(tmp_path):
    f = tmp_path / "s1 header.tif"
    a = np.array([[[-9999, -12.0]], [[-20.0, -9999]]], "float32")
    with rasterio.open(f, "w", driver="GTiff", height=1, width=2, count=2, dtype="float32",
                       crs="EPSG:4326", transform=Affine(1, 0, 0, 0, -1, 1), nodata=-9999) as d:
        d.write(a)
    x = mc.read_s1(f)
    assert np.isnan(x[0, 0, 0]) and np.isnan(x[1, 0, 1]) and x[0, 0, 1] == -12.0


def test_masks():
    x = np.zeros((4, 1, 4), "float32")
    x[0, 0, 0] = np.nan          # VV missing
    x[1, 0, 1] = np.nan          # VH missing
    x[2, 0, 3] = np.nan          # HAND missing: still valid
    y = np.array([[1, 0, -1, 1]], "int8")
    assert mc.input_valid(x).tolist() == [[False, False, True, True]]
    assert mc.train_valid(x, y).tolist() == [[False, False, False, True]]


def test_crop_of_a_small_odd_chip(tmp_path):
    """A chip smaller than the crop is padded: inputs 0, labels -1; never 512 assumed."""
    root = tmp_path / "odd chip"
    p = mc.Paths(root)
    tf = Affine(syn.PX, 0, 10, 0, -syn.PX, 20)
    s1 = p.s1_dir("hand") / "Odd_1_S1Hand.tif"
    lab = p.label_dir("hand") / "Odd_1_LabelHand.tif"
    ter = p.terrain_dir("hand") / "Odd_1_Terrain.tif"
    rng = np.random.default_rng(0)
    syn._write(s1, [rng.normal(-10, 1, (50, 70)), rng.normal(-17, 1, (50, 70))], "float32",
               "EPSG:4326", tf)
    syn._write(lab, [np.ones((50, 70))], "int16", "EPSG:4326", tf)
    syn._write(ter, [np.ones((50, 70)), np.ones((50, 70))], "float32", "EPSG:4326", tf,
               nodata=np.nan)
    row = {"chip": "Odd_1", "s1": p.rel_any(s1), "label": p.rel_any(lab),
           "terrain": p.rel_any(ter)}
    st = mc.compute_norm_stats(rng.normal(size=(100, 4)), "t")
    ds = mc.TrainCrops([row], p.data, st, crop=64)
    x, y = ds[(0, 123)]
    assert tuple(x.shape) == (4, 64, 64) and tuple(y.shape) == (64, 64)
    assert int((y == -1).sum()) == 64 * 64 - 50 * 64      # padded rows ignored
    assert np.isfinite(x.numpy()).all()
