r"""
models_synthetic.py
===================
Builds a tiny, fake copy of the project's data, in exactly the formats the
model scripts read, so that the model scripts can test themselves
(--selftest) and the test suite can run without the real data.

Nothing here is used on real data. It writes only inside the folder it is
given, which is always a temporary folder with a space in its name.

What it makes, under <project>:
  04 Code/FYP_code/data/sen1floods11/v1.1/...   18 hand chips, 8 weak chips,
      64 x 64, EPSG:4326, the real folder and file names, the four split CSVs,
      a .DS_Store in the weak label folder, an empty chip in each set, a
      chip whose labels are all -1, a Bolivia weak chip, a weak chip that
      overlaps a test chip, NaN borders, label value -1
  04 Code/FYP_code/data/sen1floods11_terrain/{hand,weak}/  terrain + log
  04 Code/FYP_code/data/corridor.geojson
  04 Code/FYP_code/data/strata_<year>_v3/        30 m grid, nodata -9999,
      layover code 1 and shadow code 2, strata_meta.json
  04 Code/FYP_code/data/reference_windows_<year>/reference_windows.geojson
  04 Code/FYP_code/data/flood_<year>_v3/flood_<year>_otsu.tif and the
      February change-detection map
  04 Code/FYP_code/data/reference_<year>.tif    made by the real
      06_rasterise_reference.py
  05 Data/sentinel1/, sentinel1_rtc/  GRD and RTC post-event scenes
  05 Data/dem/WATER_PERMANENT_JRC50_S2DRY_10m.tif   0 / 100
  05 Data/sentinel2/S2_<year>_event_cloudgap.gpkg   layer holes
  06 Outputs/gis_tools/qgis/<year>_event/reference_<year>_event.gpkg
      layers water_event and water_dry
  04 Code/FYP_code/data/weights/TerraMind_v1_small.pt   only when asked
      (build_terramind_weights): random numbers in the layout of the real
      file, the backbone with its S1GRD and S1RTC inputs plus tensors of
      other parts of TerraMind that the models must leave unused

USAGE
-----
  Not run by hand. Imported by the --selftest of scripts 20 to 28 and by
  the tests in 04 Code\tests.
"""

__stage__ = "Models: synthetic test data"
__reads__ = ["nothing"]
__writes__ = ["a synthetic project in a temporary folder"]

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import geopandas as gpd
    import rasterio
    from rasterio.features import shapes as raster_shapes
    from rasterio.transform import Affine
    from rasterio.warp import transform as warp_transform
    from shapely.geometry import box, shape
except ImportError as e:
    sys.exit(f"\nMissing package: {e}.\nActivate the environment first:  conda activate fyp\n")

HERE = Path(__file__).resolve().parent
PX = 8.983152841195214e-05          # Sen1Floods11's 10 m in degrees
CHIP = 64                           # synthetic chip size (real chips are 512)
CORRIDOR_W, CORRIDOR_H = 180, 120   # synthetic corridor grid, 10 m cells
WINDOW_CELLS = 40                   # synthetic digitising window side, cells

# chip id, split, kind. kind: good, empty, nolabel
HAND_CHIPS = [
    ("Ghana_1001", "train", "good"), ("India_1002", "train", "good"),
    ("Mekong_1003", "train", "good"), ("Nigeria_1004", "train", "good"),
    ("Pakistan_1005", "train", "good"), ("Paraguay_34417", "train", "empty"),
    ("Spain_1007", "train", "nolabel"), ("Sri-Lanka_1008", "train", "good"),
    ("USA_1009", "train", "good"), ("Somalia_1010", "train", "good"),
    ("Ghana_2001", "valid", "good"), ("India_2002", "valid", "good"),
    ("USA_2003", "valid", "good"),
    ("Mekong_3001", "test", "good"), ("Spain_3002", "test", "good"),
    ("Sri-Lanka_3003", "test", "good"),
    ("Bolivia_4001", "bolivia", "good"), ("Bolivia_4002", "bolivia", "good"),
]
# weak chip id, kind, overlaps (a hand chip id whose footprint it shares)
WEAK_CHIPS = [
    ("Ghana_5001", "good", None), ("India_5002", "good", None),
    ("Nigeria_5003", "good", None), ("Pakistan_5004", "good", None),
    ("Bolivia_5005", "good", None), ("Mekong_5006", "good", "Mekong_3001"),
    ("USA_5007", "empty", None), ("Paraguay_5008", "good", "Ghana_1001"),
]

EXPECTED_SYNTHETIC = {
    "s1hand": 18, "labelhand": 18, "s1weak": 8, "labelweak": 8,
    "split_train": 10, "split_valid": 3, "split_test": 3, "split_bolivia": 2,
    "terrain_hand": 17, "terrain_weak": 7, "weak_bolivia": 1,
    "empty_hand": 1, "empty_weak": 1,
    "corridor_width": CORRIDOR_W, "corridor_height": CORRIDOR_H,
    "strata_width": 61, "strata_height": 42, "windows": 4,
}


def _paths(root):
    root = Path(root)
    code = root / "04 Code" / "FYP_code"
    data = code / "data"
    s1f = data / "sen1floods11" / "v1.1"
    ev = s1f / "data" / "flood_events"
    return {
        "root": root, "code": code, "data": data, "s1f": s1f,
        "s1hand": ev / "HandLabeled" / "S1Hand",
        "labelhand": ev / "HandLabeled" / "LabelHand",
        "s1weak": ev / "WeaklyLabeled" / "S1Weak",
        "labelweak": ev / "WeaklyLabeled" / "S2IndexLabelWeak",
        "otsuweak": ev / "WeaklyLabeled" / "S1OtsuLabelWeak",
        "splits": s1f / "splits" / "flood_handlabeled",
        "terrain": data / "sen1floods11_terrain",
        "d05": root / "05 Data", "o06": root / "06 Outputs",
    }


# =============================================================================
# Sen1Floods11 chips
# =============================================================================

def _chip_arrays(rng, kind, weak=False):
    """VV, VH (dB, NaN border), label (int16), hand, slope for one chip."""
    n = CHIP
    r, c = np.mgrid[0:n, 0:n]
    centre = rng.uniform(20, 44) + 0.3 * (c - n / 2) * rng.uniform(-1, 1)
    half = rng.uniform(4, 9)
    dist = np.abs(r - centre)                         # cells from the river axis
    water = dist < half
    vv = np.where(water, -20.0, -9.0) + rng.normal(0, 1.5, (n, n))
    vh = vv - 7.0 + rng.normal(0, 1.0, (n, n))
    label = water.astype("int16")
    # terrain overlaps between the classes, so radar decides, as in reality
    hand = np.clip(np.clip(dist - half, 0, None) * 0.8 + rng.normal(0, 1.5, (n, n)), 0, None)
    slope = np.clip(1.0 + 0.4 * np.clip(dist - half, 0, None) + rng.normal(0, 1.5, (n, n)),
                    0, 60)
    if not weak:
        label[5:11, 40:46] = -1                       # some no-data labels
    if kind == "nolabel":
        label[:] = -1
    # NaN border on the left, as some real chips have
    vv[:, :3] = np.nan
    vh[:, :3] = np.nan
    if kind == "empty":
        vv[:] = np.nan
        vh[:] = np.nan
    hand[60:, 60:] = np.nan                           # a corner without DEM
    slope[60:, 60:] = np.nan
    return (vv.astype("float32"), vh.astype("float32"), label,
            hand.astype("float32"), slope.astype("float32"))


def _write(path, bands, dtype, crs, transform, nodata=None, names=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = {"driver": "GTiff", "height": bands[0].shape[0], "width": bands[0].shape[1],
            "count": len(bands), "dtype": dtype, "crs": crs, "transform": transform}
    if nodata is not None:
        prof["nodata"] = nodata
    with rasterio.open(path, "w", **prof) as dst:
        for i, b in enumerate(bands, start=1):
            dst.write(b.astype(dtype), i)
            if names:
                dst.set_band_description(i, names[i - 1])


def build_chips(root, seed=0):
    p = _paths(root)
    rng = np.random.default_rng(seed)
    origins = {}
    for i, (chip, split, kind) in enumerate(HAND_CHIPS):
        lon, lat = 10.0 + 0.02 * i, 20.0
        origins[chip] = (lon, lat)
        tf = Affine(PX, 0, lon, 0, -PX, lat)
        vv, vh, lab, hand, slope = _chip_arrays(rng, kind)
        _write(p["s1hand"] / f"{chip}_S1Hand.tif", [vv, vh], "float32", "EPSG:4326", tf)
        _write(p["labelhand"] / f"{chip}_LabelHand.tif", [lab], "int16", "EPSG:4326", tf)
        if kind != "empty":
            _write(p["terrain"] / "hand" / f"{chip}_Terrain.tif", [hand, slope],
                   "float32", "EPSG:4326", tf, nodata=np.nan, names=["hand_m", "slope_deg"])
    for i, (chip, kind, overlaps) in enumerate(WEAK_CHIPS):
        if overlaps:
            lon, lat = origins[overlaps]
            lon, lat = lon + 10 * PX, lat - 7 * PX       # shifted, still overlapping
        else:
            lon, lat = 10.0 + 0.02 * i, 21.0
        tf = Affine(PX, 0, lon, 0, -PX, lat)
        vv, vh, lab, hand, slope = _chip_arrays(rng, kind, weak=True)
        _write(p["s1weak"] / f"{chip}_S1Weak.tif", [vv, vh], "float32", "EPSG:4326", tf)
        _write(p["labelweak"] / f"{chip}_S2IndexLabelWeak.tif", [lab], "int16",
               "EPSG:4326", tf)
        # the Otsu weak labels exist in the real data but are never used
        _write(p["otsuweak"] / f"{chip}_S1OtsuLabelWeak.tif", [1 - np.clip(lab, 0, 1)],
               "int16", "EPSG:4326", tf)
        if kind != "empty":
            _write(p["terrain"] / "weak" / f"{chip}_Terrain.tif", [hand, slope],
                   "float32", "EPSG:4326", tf, nodata=np.nan, names=["hand_m", "slope_deg"])
    (p["labelweak"] / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1")

    p["splits"].mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test", "bolivia"):
        rows = [f"{c}_S1Hand.tif,{c}_LabelHand.tif" for c, s, _ in HAND_CHIPS if s == split]
        (p["splits"] / f"flood_{split}_data.csv").write_text("\n".join(rows) + "\n",
                                                             encoding="utf-8")
    cols = ["chip", "status", "seconds", "downloaded", "hand_defined_pct",
            "hand_median_m", "hand_lt5m_pct", "slope_median_deg", "dem_min_m", "dem_max_m"]
    for sub, chips in (("hand", [(c, k) for c, _, k in HAND_CHIPS]),
                       ("weak", [(c, k) for c, k, _ in WEAK_CHIPS])):
        lines = [",".join(cols)]
        for c, k in chips:
            if k == "empty":
                lines.append(f"{c},empty chip (no radar data),,,,,,,,")
            else:
                lines.append(f"{c},ok,1.0,yes,98.0,3.2,40.0,2.5,100.0,300.0")
        (p["terrain"] / sub / "terrain_log.csv").write_text("\n".join(lines) + "\n",
                                                            encoding="utf-8")
    return p


# =============================================================================
# The corridor
# =============================================================================

def corridor_transform():
    """10 m grid that straddles the Devprayag split, like the real corridor."""
    xs, _ = warp_transform("EPSG:4326", "EPSG:32644", [78.5981], [30.15])
    x0 = round((xs[0] - 900) / 30.0) * 30.0
    return Affine(10, 0, x0, 0, -10, 3357010.0)


def _windows(year):
    """Four 40 x 40-cell windows as (row0, col0); numbering differs by year."""
    if year == 2021:
        return {1: (55, 5), 2: (46, 50), 3: (22, 95), 4: (41, 138)}
    # 2023: other numbering, and one window off the river (no water in it)
    return {3: (55, 5), 1: (46, 50), 4: (22, 95), 2: (0, 138)}


def _polys(mask, transform, crs, extra=None):
    geoms = [shape(g) for g, v in raster_shapes(mask.astype("uint8"), mask=mask,
                                                transform=transform) if v == 1]
    gdf = gpd.GeoDataFrame({"window": [0] * len(geoms), "note": ["synthetic"] * len(geoms)},
                           geometry=geoms, crs=crs)
    if extra:
        for k, v in extra.items():
            gdf[k] = v
    return gdf


def build_corridor(root, year, seed=0):
    p = _paths(root)
    rng = np.random.default_rng(seed + year)
    H, W = CORRIDOR_H, CORRIDOR_W
    tf = corridor_transform()
    crs = "EPSG:32644"
    r, c = np.mgrid[0:H, 0:W]
    axis = 60 + 18 * np.sin(c / 25.0)                 # river centre line, rows
    dist = np.abs(r - axis)
    inside = dist < 48                                 # the corridor band
    inside &= ~((r < 6) & (c > 150))                   # a ragged corner
    river = dist < 3                                   # permanent water
    flood = (dist < 7) & (c % 60 < 35)                 # extra water on the event
    event_water = (river | flood) & inside
    ls = np.zeros((H, W), "uint8")
    ls[70:76, 10:25] = 1                               # layover, in a window
    ls[25:32, 100:112] = 2                             # shadow, in a window
    ls[95:99, 120:125] = 1

    vv = np.where(event_water, -19.0, -8.9) + rng.normal(0, 1.6, (H, W))
    vv = np.where(ls == 1, vv + 6.0, vv)
    vh = vv - 7.0 + rng.normal(0, 1.0, (H, W))
    angle = 30.0 + 15.0 * c / W
    vv[~inside] = np.nan
    vh[~inside] = np.nan
    angle = np.where(inside, angle, np.nan)
    d1 = p["d05"] / "sentinel1"
    d2 = p["d05"] / "sentinel1_rtc"
    name = {2021: "2021_post_20211021_orb63", 2023: "2023_post_20230816_orb129"}[year]
    _write(d1 / f"S1_{name}.tif", [vv, vh, angle], "float32", crs, tf,
           names=["VV", "VH", "angle"])
    rvv = vv + rng.normal(0, 0.3, (H, W))
    rvh = vh + rng.normal(0, 0.3, (H, W))
    shadow = ls == 2
    rvv[shadow] = np.nan
    rvh[shadow] = np.nan
    _write(d2 / f"S1RTC_{name}.tif", [rvv, rvh, angle], "float32", crs, tf,
           names=["VV", "VH", "angle"])

    perm = np.where(river & inside, 100, 0).astype("uint8")
    perm_path = p["d05"] / "dem" / "WATER_PERMANENT_JRC50_S2DRY_10m.tif"
    if not perm_path.exists():
        _write(perm_path, [perm], "uint8", crs, tf)

    # strata on a 30 m grid that overhangs the 10 m grid by 20 m, like the real one
    sw, sh = EXPECTED_SYNTHETIC["strata_width"], EXPECTED_SYNTHETIC["strata_height"]
    stf = Affine(30, 0, tf.c, 0, -30, tf.f + 20)
    sr, sc = np.mgrid[0:sh, 0:sw]
    xs = stf.c + 30 * (sc + 0.5)
    ys = stf.f - 30 * (sr + 0.5)
    rr = (tf.f - ys) / 10.0
    cc = (xs - tf.c) / 10.0
    sdist = np.abs(rr - (60 + 18 * np.sin(cc / 25.0)))
    sin = sdist < 48
    hand = np.where(sin, np.clip((sdist - 3) * 0.8, 0, None), -9999).astype("float32")
    slope = np.where(sin, np.clip(1 + 0.4 * np.clip(sdist - 3, 0, None), 0, 70),
                     -9999).astype("float32")
    theta = np.where(sin, 20 + 40 * sc / sw, -9999).astype("float32")
    width = np.where(sin, 50 + 30 * sr, -9999).astype("float32")
    sls = np.zeros((sh, sw), "uint8")
    for code in (1, 2):
        rows, cols = np.nonzero(ls == code)
        sls[np.clip((rows * 10 + 20) // 30, 0, sh - 1), np.clip(cols * 10 // 30, 0, sw - 1)] = code
    sls = np.where(sin, sls, 255).astype("uint8")
    sd = p["data"] / f"strata_{year}_v3"
    for nm, arr in (("hand_m", hand), ("slope_deg", slope), ("theta_loc", theta),
                    ("valley_width_m", width)):
        _write(sd / f"strata_{nm}.tif", [arr], "float32", crs, stf, nodata=-9999.0)
    _write(sd / "strata_layover_shadow.tif", [sls], "uint8", crs, stf, nodata=255)
    meta = {"incidence_angle_source": "synthetic", "orbit_pass": "DESCENDING",
            "reach_split_longitude": 78.5981, "crs": crs}
    (sd / "strata_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # digitising windows, as 05_reference_windows.py writes them
    feats, win_mask = [], np.zeros((H, W), bool)
    for k, (r0, c0) in sorted(_windows(year).items()):
        x0, y1 = tf * (c0, r0)
        x1, y0 = tf * (c0 + WINDOW_CELLS, r0 + WINDOW_CELLS)
        ring = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
        feats.append({"type": "Feature",
                      "properties": {"window": k, "side_km": WINDOW_CELLS * 10 / 1000,
                                     "centre_x": round((x0 + x1) / 2, 1),
                                     "centre_y": round((y0 + y1) / 2, 1),
                                     "pred_flood_cells": 0},
                      "geometry": {"type": "Polygon", "coordinates": [ring]}})
        win_mask[r0:r0 + WINDOW_CELLS, c0:c0 + WINDOW_CELLS] = True
    wd = p["data"] / f"reference_windows_{year}"
    wd.mkdir(parents=True, exist_ok=True)
    gj = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": crs}},
          "features": feats}
    (wd / "reference_windows.geojson").write_text(json.dumps(gj, indent=1), encoding="utf-8")

    # hand-drawn water: event and dry layers, inside the windows only
    gp = p["o06"] / "gis_tools" / "qgis" / f"{year}_event" / f"reference_{year}_event.gpkg"
    gp.parent.mkdir(parents=True, exist_ok=True)
    _polys(event_water & win_mask, tf, crs).to_file(gp, layer="water_event", driver="GPKG")
    _polys(river & inside & win_mask, tf, crs).to_file(gp, layer="water_dry", driver="GPKG")

    # one cloud hole inside window 2's square
    r0, c0 = _windows(year)[2]
    hx0, hy1 = tf * (c0 + 5, r0 + 5)
    hx1, hy0 = tf * (c0 + 15, r0 + 15)
    holes = gpd.GeoDataFrame({"source": ["synthetic gap mask, value 1"], "note": ["no image"]},
                             geometry=[box(hx0, hy0, hx1, hy1)], crs=crs)
    hp = p["d05"] / "sentinel2" / f"S2_{year}_event_cloudgap.gpkg"
    hp.parent.mkdir(parents=True, exist_ok=True)
    holes.to_file(hp, layer="holes", driver="GPKG")

    # two Otsu-like baseline maps on the grid: 255 outside, 0/1 inside
    valid = np.isfinite(vv)
    perm_b = perm == 100
    dis = ls > 0
    for folder, fname, thr in ((f"flood_{year}_v3", f"flood_{year}_otsu.tif", -12.0),
                               (f"flood_{year}_ch_feb_v3", f"flood_{year}chfeb_otsu.tif", -14.0)):
        m = np.full((H, W), 255, "uint8")
        m[valid] = 0
        m[valid & (vv < thr) & ~perm_b & ~dis] = 1
        _write(p["data"] / folder / fname, [m], "uint8", crs, tf, nodata=255)

    # the reference, with the real 06_rasterise_reference.py
    out = p["data"] / f"reference_{year}.tif"
    cmd = [sys.executable, str(HERE / "06_rasterise_reference.py"),
           "--event", str(gp), "--event-layer", "water_event", "--dry-layer", "water_dry",
           "--windows", str(wd / "reference_windows.geojson"),
           "--holes", str(hp), "--holes-layer", "holes",
           "--grid", str(p["data"] / f"flood_{year}_v3" / f"flood_{year}_otsu.tif"),
           "--out", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not out.exists():
        raise RuntimeError("06_rasterise_reference.py failed on the synthetic data:\n"
                           + res.stdout + res.stderr)
    return p


def build_corridor_geojson(root):
    p = _paths(root)
    tf = corridor_transform()
    x0, y1 = tf * (0, 0)
    x1, y0 = tf * (CORRIDOR_W, CORRIDOR_H)
    g = gpd.GeoDataFrame({"name": ["corridor"]}, geometry=[box(x0, y0, x1, y1)],
                         crs="EPSG:32644").to_crs("EPSG:4326")
    p["data"].mkdir(parents=True, exist_ok=True)
    (p["data"] / "corridor.geojson").write_text(g.to_json(), encoding="utf-8")


def build_project(root, seed=0, chips=True, corridor=True, years=(2021, 2023)):
    """Build the whole synthetic project under root. Returns root as a Path."""
    root = Path(root)
    if chips:
        build_chips(root, seed)
    if corridor:
        build_corridor_geojson(root)
        for y in years:
            build_corridor(root, y, seed)
    (root / "06 Outputs").mkdir(parents=True, exist_ok=True)
    return root


# =============================================================================
# TerraMind weights (random numbers, the real file's layout)
# =============================================================================

def build_terramind_weights(root, seed=0):
    """A stand-in for TerraMind_v1_small.pt at the path the models read it from.

    The real file is TerraMind's whole state: the encoder, the patch inputs of
    every modality and the decoder. This one holds the encoder with both S1GRD
    and S1RTC inputs (random, built by terratorch exactly as the models build
    it) and a few tensors of other parts, which the models must not use.
    """
    import torch
    import models_common as mc
    path = _paths(root)["data"] / "weights" / "TerraMind_v1_small.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    reg, _ = mc.tm_import()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        grd = reg.build(mc.TM_BACKBONE, modalities=["S1GRD"], pretrained=False,
                        merge_method=None).state_dict()
        rtc = reg.build(mc.TM_BACKBONE, modalities=["S1RTC"], pretrained=False,
                        merge_method=None).state_dict()
        sd = dict(grd)
        sd.update({k: v for k, v in rtc.items() if k.startswith("encoder_embeddings.")})
        sd["encoder_embeddings.untok_sen2l2a@224.proj.weight"] = torch.randn(384, 12 * 256)
        sd["decoder.0.attn.qkv.weight"] = torch.randn(1152, 384)
        sd["decoder_norm.weight"] = torch.ones(384)
    torch.save(sd, path)
    return path


def break_terramind_weights(root, keep=0.5):
    """Rename most backbone tensors of the stand-in weights, as a wrong file would have them."""
    import torch
    path = _paths(root)["data"] / "weights" / "TerraMind_v1_small.pt"
    sd = torch.load(path, weights_only=True)
    keys = [k for k in sd if k.startswith("encoder.")]
    for k in keys[int(len(keys) * keep):]:
        sd["model." + k] = sd.pop(k)
    torch.save(sd, path)
    return path


# =============================================================================
# Broken variants, for the preflight tests
# =============================================================================

def break_band_count(root):
    """Rewrite one hand S1 chip with 3 bands instead of 2."""
    p = _paths(root)
    f = p["s1hand"] / "Ghana_1001_S1Hand.tif"
    with rasterio.open(f) as s:
        a, tf, crs = s.read(), s.transform, s.crs
    _write(f, [a[0], a[1], a[0]], "float32", crs, tf)
    return f


def break_label_value(root):
    """Put the value 5 into one hand label."""
    p = _paths(root)
    f = p["labelhand"] / "India_1002_LabelHand.tif"
    with rasterio.open(f) as s:
        a, tf, crs = s.read(1), s.transform, s.crs
    a[0:4, 20:24] = 5
    _write(f, [a], "int16", crs, tf)
    return f


def break_missing_terrain(root):
    """Delete the terrain file of a non-empty hand chip."""
    f = _paths(root)["terrain"] / "hand" / "Mekong_1003_Terrain.tif"
    f.unlink()
    return f
