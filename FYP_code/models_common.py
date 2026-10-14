r"""
models_common.py
================
Shared code for the Random Forest, U-Net and TerraMind scripts, 20 to 30:
paths, raster reading, chips and splits, normalisation, masks, the flood-map
contract, tiled prediction, metrics, run records, device and memory helpers,
and the model registry.

Every model script imports this file, so a rule written here holds for all of
them. The rules that matter most:
  - channels are always VV, VH, HAND, SLOPE, in that order
  - a training pixel counts only where VV and VH are finite and the label is
    0 or 1; every other label value (-1, 255, anything) is ignored
  - after normalisation every non-finite input becomes 0 (the mean)
  - a flood map is uint8, 255 where VV is not finite, 0 elsewhere, and 1
    where the model says water and the cell is neither permanent water nor
    layover or shadow (exactly as 03b_otsu_baseline.py does it)

USAGE
-----
  Not run on its own in normal use. To check it:
    python models_common.py --selftest
"""

__stage__ = "Models: shared code"
__reads__ = ["Sen1Floods11 chips", "corridor rasters", "norm_stats.json"]
__writes__ = ["run_record.json", "flood maps (through scripts 25 and 26)"]

import argparse
import csv
import ctypes
import hashlib
import importlib.util
import json
import os
import platform
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# Windows, conda env fyp: PyTorch brings Intel's OpenMP (libiomp5md.dll) and
# conda's llvm-openmp brings libomp.dll. When both start in one process,
# Intel's library aborts with "OMP: Error #15" (found on the owner's laptop,
# 7 October 2026, in 25_predict_corridor.py). This setting lets both run.
# It must be set before either starts; the .bat files set it too.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np                                      # noqa: E402

try:
    import rasterio
    from rasterio.warp import Resampling, reproject
    from rasterio.windows import Window
except ImportError as e:
    sys.exit(f"\nMissing package: {e}.\nActivate the environment first:  conda activate fyp\n")

try:                        # the Random Forest scripts work without PyTorch
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:
    torch = None

HERE = Path(__file__).resolve().parent
CHANNELS = ("VV", "VH", "HAND", "SLOPE")
SEED = 20261007
NODATA = 255
YEARS = (2021, 2023)
TRAINSETS = ("hand", "hand_weak")
MODELS = ("rf", "unet", "tm_s1", "tm_s1t")
TM_VARIANTS = ("tm_s1", "tm_s1t")              # TerraMind-small: radar only; radar + terrain
RADARS = ("grd", "rtc")
TM_MODALITY = {"grd": "S1GRD", "rtc": "S1RTC"}  # TerraMind's own input for each radar product
FLOOD_KM2_HIGH, FLOOD_KM2_LOW = 30.0, 0.5      # brief 3.4: outside this is almost certainly broken
CHUNK = 1_000_000                              # pixels per Random Forest prediction call

SCENES = {
    (2021, "grd"): ("sentinel1", "S1_2021_post_20211021_orb63.tif"),
    (2023, "grd"): ("sentinel1", "S1_2023_post_20230816_orb129.tif"),
    (2021, "rtc"): ("sentinel1_rtc", "S1RTC_2021_post_20211021_orb63.tif"),
    (2023, "rtc"): ("sentinel1_rtc", "S1RTC_2023_post_20230816_orb129.tif"),
}
SETS = {   # set: (folder, S1 folder, label folder, S1 suffix, label suffix)
    "hand": ("HandLabeled", "S1Hand", "LabelHand", "_S1Hand.tif", "_LabelHand.tif"),
    "weak": ("WeaklyLabeled", "S1Weak", "S2IndexLabelWeak", "_S1Weak.tif",
             "_S2IndexLabelWeak.tif"),
}
SPLIT_FILES = {"train": "flood_train_data.csv", "valid": "flood_valid_data.csv",
               "test": "flood_test_data.csv", "bolivia": "flood_bolivia_data.csv"}


# =============================================================================
# Console, failures, output guards
# =============================================================================

def setup_console():
    """Never crash on printing a character the Windows console cannot show."""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def fail(msg):
    """Stop with a sentence that says what is wrong and what to do."""
    raise SystemExit("\nPROBLEM: " + msg.strip() + "\n")


def cli(main):
    """Run a script's main(); Ctrl+C ends it with a plain sentence, not a traceback."""
    setup_console()
    try:
        return main()
    except KeyboardInterrupt:
        print("\nSTOPPED by Ctrl+C. Every file is written whole or not at all, so nothing "
              "half-written is left. Run the same command again to carry on.", flush=True)
        return 130


def guard(marker, args, what):
    """Return True if this step should be skipped because it is already done.

    marker is the file a finished run writes last. If it exists the step is
    finished: skip it with --skip-existing, redo it with --force, otherwise
    stop. Files of an unfinished run (no marker) are simply redone.
    """
    marker = Path(marker)
    if not marker.exists() or getattr(args, "force", False):
        return False
    if getattr(args, "skip_existing", False):
        print(f"SKIP: {what} is already done ({marker.parent.name}\\{marker.name}).")
        return True
    fail(f"{what} already exists:\n  {marker}\n"
         "Nothing was changed. Add --force to redo it and overwrite, or "
         "--skip-existing to leave it as it is and carry on.")


def check_disk(path, need_gb, what):
    free = disk_free_gb(path)
    if free < need_gb:
        fail(f"Only {free:.1f} GB free on the drive of {path}; {what} needs about "
             f"{need_gb:.0f} GB. Free some space and run the same command again.")
    return free


class Progress:
    """Prints a progress line at most every `every` seconds, with an ETA."""

    def __init__(self, total, what, every=120.0):
        self.total, self.what, self.every = max(int(total), 1), what, every
        self.t0 = self.last = time.time()
        self.n = 0

    def tick(self, n=1, extra=""):
        self.n += n
        now = time.time()
        if now - self.last >= self.every or self.n >= self.total:
            self.last = now
            el = now - self.t0
            left = el / self.n * (self.total - self.n) if self.n else 0.0
            print(f"  {self.what}: {self.n}/{self.total}, {el / 60:.1f} min so far, "
                  f"about {left / 60:.1f} min left {extra}".rstrip(), flush=True)


# =============================================================================
# Paths
# =============================================================================

class Paths:
    """Every default path, built from the script's own folder.

    With no project given: FYP_code is this file's folder and the project
    folder (holding 04 Code, 05 Data, 06 Outputs) is two levels up. With a
    project folder given (--project, used by the self-tests), FYP_code is
    <project>\\04 Code\\FYP_code. --smoke sends every output into a separate
    models_smoke tree so a smoke run can never touch a real result.
    """

    def __init__(self, project=None, smoke=False):
        if project is None:
            code, root = HERE, HERE.parents[1]
        else:
            root = Path(project).resolve()
            code = root / "04 Code" / "FYP_code"
        self.root, self.code, self.smoke = root, code, bool(smoke)
        self.data = code / "data"
        self.s1f = self.data / "sen1floods11" / "v1.1"
        self.events = self.s1f / "data" / "flood_events"
        self.splits = self.s1f / "splits" / "flood_handlabeled"
        self.terrain = self.data / "sen1floods11_terrain"
        self.d05 = root / "05 Data"
        self.o06 = root / "06 Outputs"
        self.models = self.data / ("models_smoke" if smoke else "models")
        self.maps = self.models if smoke else self.data
        self.out = self.o06 / "models_smoke" if smoke else self.o06
        self.tables = self.out / "tables"
        self.chip_index = self.models / "chip_index.csv"
        self.norm_stats = self.models / "norm_stats.json"
        self.cache = self.models / "cache"

    def s1_dir(self, set_):
        f, s1, _, _, _ = SETS[set_]
        return self.events / f / s1

    def label_dir(self, set_):
        f, _, lab, _, _ = SETS[set_]
        return self.events / f / lab

    def terrain_dir(self, set_):
        return self.terrain / set_

    def scene(self, year, radar="grd"):
        folder, name = SCENES[(int(year), radar)]
        return self.d05 / folder / name

    def strata(self, year):
        return self.data / f"strata_{year}_v3"

    def perm_water(self):
        return self.d05 / "dem" / "WATER_PERMANENT_JRC50_S2DRY_10m.tif"

    def reference(self, year):
        return self.data / f"reference_{year}.tif"

    def windows(self, year):
        return self.data / f"reference_windows_{year}" / "reference_windows.geojson"

    def ref_gpkg(self, year):
        return (self.o06 / "gis_tools" / "qgis" / f"{year}_event"
                / f"reference_{year}_event.gpkg")

    def holes(self, year):
        return self.d05 / "sentinel2" / f"S2_{year}_event_cloudgap.gpkg"

    def otsu(self, year, kind="single"):
        if kind == "single":
            return self.data / f"flood_{year}_v3" / f"flood_{year}_otsu.tif"
        return self.data / f"flood_{year}_ch_feb_v3" / f"flood_{year}chfeb_otsu.tif"

    def model_dir(self, model, trainset, tag=""):
        return self.models / f"{model}_{trainset}{_tag(tag)}"

    def tm_weights(self):
        """TerraMind-1.0-small's pretrained weights (an input, the same for --smoke)."""
        return self.data / "weights" / "TerraMind_v1_small.pt"

    def flood_name(self, year, model, trainset, radar="grd", tag=""):
        return (f"flood_{year}_{model}_{trainset}{_tag(tag)}"
                + ("" if radar == "grd" else f"_{radar}"))

    def flood_map(self, year, model, trainset, radar="grd", tag=""):
        n = self.flood_name(year, model, trainset, radar, tag)
        return self.maps / n / f"{n}.tif"

    def lowo_dir(self, year, model, radar):
        return self.maps / f"lowo_{year}_{model}_{radar}"

    def lowo_map(self, year, model, radar):
        return self.lowo_dir(year, model, radar) / f"flood_lowo_{year}_{model}_{radar}.tif"

    def rel(self, path):
        """Path relative to FYP_code\\data, with forward slashes (portable)."""
        return Path(path).resolve().relative_to(self.data.resolve()).as_posix()

    def rel_any(self, path):
        """Like rel() but for a file that may not exist yet."""
        return Path(path).relative_to(self.data).as_posix()

    def abs(self, rel):
        return self.data / Path(rel)


RUN_TAG_RE = re.compile(r"^[a-z0-9]{1,16}$")


def _tag(tag):
    """'' -> '' (the original runs, every path unchanged); 'seed2' -> '_seed2'."""
    return f"_{tag}" if tag else ""


def add_run_tag_arg(ap):
    """--run-tag (9 October): a repeat run of a model, e.g. with another seed,
    gets its own folders (unet_hand_seed2, flood_2021_unet_hand_seed2, ...) so the
    original runs, which Tables A and D and Jal-SAR use, are never touched."""
    ap.add_argument("--run-tag", default="",
                    help="name of a repeat run, lower-case letters and digits only (e.g. "
                         "seed2): its outputs go to separate folders ending _<tag>. Left out, "
                         "every path is exactly as before")


def check_run_tag(args, seed_used=True):
    """Validate --run-tag. A tagged training run must not use the original seed,
    because it would only repeat the original run under another name."""
    tag = getattr(args, "run_tag", "") or ""
    if tag and not RUN_TAG_RE.match(tag):
        fail(f"--run-tag must be 1 to 16 lower-case letters or digits (e.g. seed2), "
             f"not '{tag}'.")
    if tag and seed_used and getattr(args, "seed", None) == SEED:
        fail(f"--run-tag {tag} is a repeat run, but --seed is the original seed {SEED}. "
             "Give the repeat run its own --seed.")
    return tag


def add_common_args(ap, smoke=True, seed=True, ram=True):
    ap.add_argument("--project", default=None,
                    help="project folder holding 04 Code, 05 Data, 06 Outputs "
                         "(default: two folders above this script)")
    ap.add_argument("--force", action="store_true",
                    help="redo the step and overwrite its outputs")
    ap.add_argument("--skip-existing", action="store_true",
                    help="if the step is already done, say so and exit without error")
    ap.add_argument("--selftest", action="store_true",
                    help="build a tiny synthetic case in a temp folder, run, check")
    if smoke:
        ap.add_argument("--smoke", action="store_true",
                        help="a few chips, tiny settings, outputs into the models_smoke tree")
    if seed:
        ap.add_argument("--seed", type=int, default=SEED)
    if ram:
        ap.add_argument("--max-ram-gb", type=float, default=12.0,
                        help="RAM the script may plan for (default 12; use 12 on a 16 GB laptop)")


def paths_from_args(args):
    return Paths(args.project, getattr(args, "smoke", False))


def load_script(name):
    """Import a numbered script (its name starts with a digit) by file path."""
    path = HERE / name
    modname = "fyp_" + path.stem.replace(" ", "_")
    if modname in sys.modules:
        return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def run_script(name, argv):
    """Run a numbered script's main(argv) in this process; returns its exit code."""
    try:
        rc = load_script(name).main(argv)
    except SystemExit as e:
        if e.code is None or e.code == 0:
            return 0
        if isinstance(e.code, str):
            print(e.code)
        return 1 if isinstance(e.code, str) else int(e.code)
    return 0 if rc is None else int(rc)


# =============================================================================
# Raster reading
# =============================================================================

def _to_nan(a, nodata):
    a = a.astype("float32", copy=False)
    if nodata is not None and not np.isnan(nodata):
        a[a == np.float32(nodata)] = np.nan
    a[~np.isfinite(a)] = np.nan
    return a


def _pad(a, h, w, fill):
    """Pad the last two axes of a to (h, w) with fill."""
    if a.shape[-2] >= h and a.shape[-1] >= w:
        return a
    out = np.full(a.shape[:-2] + (max(h, a.shape[-2]), max(w, a.shape[-1])), fill, a.dtype)
    out[..., :a.shape[-2], :a.shape[-1]] = a
    return out


def read_s1(path, window=None):
    """VV and VH of a Sentinel-1 chip or scene, float32, NaN where no data."""
    with rasterio.open(path) as src:
        if src.count < 2:
            raise ValueError(f"{Path(path).name} has {src.count} band(s); Sentinel-1 input "
                             "needs band 1 VV and band 2 VH")
        a = src.read([1, 2], window=window)
        return _to_nan(a, src.nodata)


def read_label(path, window=None):
    """Label as int8: 1 water, 0 not water, -1 ignore (any other value)."""
    with rasterio.open(path) as src:
        a = src.read(1, window=window)
    return np.where(np.isin(a, (0, 1)), a, -1).astype("int8")


def read_terrain(path, window=None):
    with rasterio.open(path) as src:
        if src.count < 2:
            raise ValueError(f"{Path(path).name} has {src.count} band(s); terrain needs "
                             "band 1 hand_m and band 2 slope_deg")
        a = src.read([1, 2], window=window)
        return _to_nan(a, src.nodata)


def chip_shape(row, data_root):
    cx = row.get("cache_x") or ""
    if cx and (Path(data_root) / cx).exists():
        return tuple(np.load(Path(data_root) / cx, mmap_mode="r").shape[1:])
    with rasterio.open(Path(data_root) / row["s1"]) as src:
        return src.height, src.width


def read_chip(row, data_root, window=None):
    """Inputs (4, h, w) float32 raw, NaN where missing, and label (h, w) int8.

    row: a chip-index row (paths relative to data_root). window: (r0, c0, h, w)
    or None for the whole chip. Uses the .npy cache when it exists.
    """
    data_root = Path(data_root)
    cx, cy = row.get("cache_x") or "", row.get("cache_y") or ""
    if cx and (data_root / cx).exists() and (data_root / cy).exists():
        xm = np.load(data_root / cx, mmap_mode="r")
        ym = np.load(data_root / cy, mmap_mode="r")
        if window is None:          # np.array copies: no open file is held afterwards
            return np.array(xm, dtype="float32"), np.array(ym, dtype="int8")
        r0, c0, h, w = window
        return (np.array(xm[:, r0:r0 + h, c0:c0 + w], dtype="float32"),
                np.array(ym[r0:r0 + h, c0:c0 + w], dtype="int8"))
    win = None if window is None else Window(window[1], window[0], window[3], window[2])
    s1 = read_s1(data_root / row["s1"], win)
    lab = read_label(data_root / row["label"], win)
    if lab.shape != s1.shape[1:]:
        raise ValueError(f"label of chip {row['chip']} is {lab.shape}, the chip is "
                         f"{s1.shape[1:]}; they must be on the same grid")
    if row.get("terrain"):
        ter = read_terrain(data_root / row["terrain"], win)
        if ter.shape[1:] != s1.shape[1:]:
            raise ValueError(f"terrain of chip {row['chip']} is {ter.shape[1:]}, the chip is "
                             f"{s1.shape[1:]}; they must be on the same grid")
    else:
        ter = np.full((2,) + s1.shape[1:], np.nan, "float32")
    return np.concatenate([s1, ter]).astype("float32"), lab


GRID_TOL_PX = 0.01      # grids closer than this (in pixels, anywhere) are the same grid


def grid_offset_px(shape, ta, tb):
    """Largest distance, in pixels of grid ta, between grids ta and tb over a raster of shape.

    Covers a shift of the origin, a different pixel size (drift at the far
    corner) and rotation terms. Sen1Floods11 label files differ from their
    chips by up to 1.6e-10 pixel (rounding in the file headers), which is 0
    for every practical purpose; an exact comparison would call them different.
    """
    h, w = shape
    px, py = abs(ta.a), abs(ta.e)
    dx, dy = (tb.c - ta.c) / px, (tb.f - ta.f) / py
    drift_x, drift_y = (tb.a - ta.a) * w / px, (tb.e - ta.e) * h / py
    shear = (abs(tb.b - ta.b) * h + abs(tb.d - ta.d) * w) / min(px, py)
    return max(abs(dx), abs(dy), abs(dx + drift_x), abs(dy + drift_y)) + shear


def same_grid(shape_a, ta, crs_a, shape_b, tb, crs_b, tol=GRID_TOL_PX):
    """Same size, same CRS, and no cell centre further apart than tol pixels."""
    return (tuple(shape_a) == tuple(shape_b) and crs_a == crs_b
            and grid_offset_px(shape_a, ta, tb) < tol)


def input_valid(x):
    """VV and VH finite."""
    return np.isfinite(x[0]) & np.isfinite(x[1])


def train_valid(x, y):
    """A pixel counts for training and scoring: VV, VH finite and label 0 or 1."""
    return input_valid(x) & (y >= 0)


# =============================================================================
# Chips and splits
# =============================================================================

def chip_id(fname, suffix):
    return fname[: -len(suffix)] if fname.endswith(suffix) else None


def event_of(chip):
    """Event name: the part before the first underscore (Sri-Lanka keeps its hyphen)."""
    return chip.split("_")[0]


def read_split_csv(path):
    """Chip ids from a Sen1Floods11 split CSV (no header; S1 file, label file)."""
    out = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for rec in csv.reader(fh):
            if not rec or not rec[0].strip():
                continue
            name = Path(rec[0].strip()).name
            cid = chip_id(name, "_S1Hand.tif")
            if cid is None:
                raise ValueError(f"{Path(path).name}: '{name}' does not end in _S1Hand.tif")
            out.append(cid)
    return out


def list_chips(folder, suffix):
    """Chip ids of every file in folder ending in suffix (ignores .DS_Store etc.)."""
    if not Path(folder).is_dir():
        return []
    return sorted(c for c in (chip_id(f.name, suffix) for f in Path(folder).iterdir())
                  if c is not None)


def chip_footprint(path):
    """(minx, miny, maxx, maxy) in EPSG:4326, height, width."""
    from rasterio.warp import transform_bounds
    with rasterio.open(path) as src:
        b = src.bounds
        if src.crs is not None and src.crs.to_epsg() != 4326:
            b = transform_bounds(src.crs, "EPSG:4326", *b)
        return tuple(float(v) for v in b), src.height, src.width


def boxes_intersect(a, b, eps=0.0):
    """Closed boxes (minx, miny, maxx, maxy): True if they share any point.

    a is (n, 4), b is (m, 4); returns (n,) bool, True where a[i] touches or
    overlaps any b[j]. Touching counts, the literal meaning of 'intersects'.
    eps widens every box: neighbouring Sen1Floods11 chips share an edge only
    to the last floating-point digit (the edge of one is origin + 512 x pixel,
    the other's origin is stored separately), so without it 98 of the 571 weak
    chips that touch a held-out chip were missed on the real data.
    """
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros(len(a), bool)
    hit = ((a[:, None, 0] <= b[None, :, 2] + eps) & (a[:, None, 2] >= b[None, :, 0] - eps)
           & (a[:, None, 1] <= b[None, :, 3] + eps) & (a[:, None, 3] >= b[None, :, 1] - eps))
    return hit.any(axis=1)


INDEX_COLS = ["chip", "set", "split", "event", "s1", "label", "terrain", "height", "width",
              "finite_share", "labelled_pixels", "labelled_share", "water_share",
              "terrain_nan_share", "minx", "miny", "maxx", "maxy", "kept", "reason"]


def cache_paths(paths, row):
    """Where 21_prepare_chips.py --cache keeps a chip's .npy copy (relative to data)."""
    base = paths.cache / row["set"] / row["chip"]
    return paths.rel_any(base.with_name(base.name + "_x.npy")), \
        paths.rel_any(base.with_name(base.name + "_y.npy"))


def load_chip_index(paths):
    if not paths.chip_index.exists():
        fail(f"The chip index is missing:\n  {paths.chip_index}\n"
             "Run 21_prepare_chips.py first" + (" with --smoke." if paths.smoke else "."))
    with open(paths.chip_index, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        cx, cy = cache_paths(paths, r)
        have = paths.abs(cx).exists() and paths.abs(cy).exists()
        r["cache_x"], r["cache_y"] = (cx, cy) if have else ("", "")
        for k in ("height", "width", "labelled_pixels"):
            r[k] = int(float(r[k])) if r.get(k) not in (None, "") else 0
        for k in ("finite_share", "labelled_share", "water_share", "terrain_nan_share"):
            r[k] = float(r[k]) if r.get(k) not in (None, "") else float("nan")
    return rows


def training_rows(index, trainset):
    """(hand rows, weak rows) a model may train on. Never valid, test or Bolivia."""
    if trainset not in TRAINSETS:
        fail(f"--trainset must be one of {', '.join(TRAINSETS)}, not '{trainset}'.")
    hand = [r for r in index if r["split"] == "train" and r["kept"] == "yes"]
    weak = ([r for r in index if r["split"] == "weak" and r["kept"] == "yes"]
            if trainset == "hand_weak" else [])
    held = {r["chip"] for r in index if r["split"] in ("valid", "test", "bolivia")}
    leak = [r["chip"] for r in hand + weak if r["chip"] in held or event_of(r["chip"]) == "Bolivia"]
    if leak:
        raise RuntimeError(f"held-out chips in a training list: {leak[:5]}")
    return hand, weak


def split_rows(index, split):
    return [r for r in index if r["split"] == split and r["kept"] == "yes"]


# =============================================================================
# Normalisation
# =============================================================================

def sample_pixels(x, y, n, rng):
    """Up to n random training-valid pixels of a chip, as (k, 4) raw values."""
    idx = np.flatnonzero(train_valid(x, y))
    if idx.size > n:
        idx = np.sort(rng.choice(idx, size=n, replace=False))
    return x.reshape(4, -1)[:, idx].T.copy()


def compute_norm_stats(samples, source, extra=None):
    """Clip bounds (0.5 and 99.5 percentiles), then mean and std of the clipped values.

    samples: (n, 4) raw values; NaN entries are left out channel by channel.
    """
    samples = np.asarray(samples, dtype="float64").reshape(-1, len(CHANNELS))
    lo, hi, mean, std, nfin = [], [], [], [], []
    for c in range(len(CHANNELS)):
        v = samples[:, c]
        v = v[np.isfinite(v)]
        nfin.append(int(v.size))
        if v.size == 0:
            lo.append(0.0), hi.append(0.0), mean.append(0.0), std.append(1.0)
            continue
        a, b = np.percentile(v, [0.5, 99.5])
        cv = np.clip(v, a, b)
        lo.append(float(a)), hi.append(float(b))
        mean.append(float(cv.mean())), std.append(float(max(cv.std(), 1e-6)))
    out = {"channels": list(CHANNELS), "clip_lo": lo, "clip_hi": hi, "mean": mean,
           "std": std, "percentiles": [0.5, 99.5], "source": source,
           "pixels_sampled": int(samples.shape[0]), "finite_per_channel": nfin,
           "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out.update(extra or {})
    return out


def save_json(path, obj):
    """Write JSON atomically: never leaves a half-written file behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_norm_stats(path):
    path = Path(path)
    if not path.exists():
        fail(f"Normalisation statistics are missing:\n  {path}\n"
             "Run 21_prepare_chips.py first.")
    s = load_json(path)
    if s.get("channels") != list(CHANNELS):
        fail(f"{path.name} lists channels {s.get('channels')}, expected {list(CHANNELS)}.")
    return s


def _stat(stats, key, ndim):
    shape = (len(CHANNELS),) + (1,) * (ndim - 1)
    return np.asarray(stats[key], dtype="float32").reshape(shape)


def normalise(x, stats):
    """Channel-first raw inputs -> clipped, standardised, non-finite set to 0."""
    x = np.asarray(x, dtype="float32")
    lo, hi = _stat(stats, "clip_lo", x.ndim), _stat(stats, "clip_hi", x.ndim)
    out = (np.clip(x, lo, hi) - _stat(stats, "mean", x.ndim)) / _stat(stats, "std", x.ndim)
    out[~np.isfinite(x) | ~np.isfinite(out)] = 0.0      # NaN and +-inf inputs -> the mean
    return out.astype("float32", copy=False)


def rf_features(x, stats):
    """Random Forest features, (n, 4): raw values clipped like normalise(); NaN -> mean."""
    x = np.asarray(x, dtype="float32").reshape(-1, len(CHANNELS))
    lo = np.asarray(stats["clip_lo"], "float32")
    hi = np.asarray(stats["clip_hi"], "float32")
    f = np.clip(x, lo, hi)
    bad = ~np.isfinite(x)
    if bad.any():
        f = np.where(bad, np.asarray(stats["mean"], "float32"), f)
    return np.ascontiguousarray(f, dtype="float32")


# =============================================================================
# Corridor inputs and the flood-map contract
# =============================================================================

def reproject_onto(path, profile, categorical=False, band=1):
    """One band of path onto profile's grid. Nearest for classes, bilinear otherwise.

    The source nodata (for example -9999 or 255) becomes NaN, exactly as
    04_stratified_metrics.py reads the strata.
    """
    out = np.full((profile["height"], profile["width"]), np.nan, dtype="float32")
    with rasterio.open(path) as src:
        reproject(source=rasterio.band(src, band), destination=out,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=profile["transform"], dst_crs=profile["crs"],
                  resampling=Resampling.nearest if categorical else Resampling.bilinear,
                  src_nodata=src.nodata, dst_nodata=np.nan)
    return out


def _need(path, what, how):
    if not Path(path).exists():
        fail(f"{what} is missing:\n  {path}\n{how}")


def read_scene(path, what):
    """VV, VH and the grid profile of a corridor scene, with its checks."""
    _need(path, what, "Check the file name and folder against RUN ORDER.md.")
    with rasterio.open(path) as src:
        if src.count < 2:
            fail(f"{Path(path).name} has {src.count} band(s); it needs VV and VH as bands 1 and 2.")
        desc = [d or "" for d in src.descriptions[:2]]
        if any(desc) and [d.upper() for d in desc] != ["VV", "VH"]:
            fail(f"{Path(path).name}: bands 1 and 2 are described as {desc}, expected VV, VH.")
        vv, vh = _to_nan(src.read([1, 2]), src.nodata)
        prof = src.profile.copy()
    return vv, vh, prof


def corridor_inputs(paths, year, radar="grd", window=None):
    """Everything a corridor prediction needs, on the scene's own 10 m grid.

    Returns dict: x (4, H, W) raw inputs, valid (VV finite), profile, perm
    (permanent water, value 100), distort (layover code 1 or shadow code 2,
    nearest neighbour from the 30 m strata), terrain_nan_cells.
    """
    vv, vh, prof = read_scene(paths.scene(year, radar), f"The {radar.upper()} scene for {year}")
    sd = paths.strata(year)
    for n in ("strata_hand_m.tif", "strata_slope_deg.tif", "strata_layover_shadow.tif"):
        _need(sd / n, f"Strata file {n} for {year}",
              "Build the strata first (RUN ORDER.md, step 4.1 or 4.2).")
    _need(paths.perm_water(), "The permanent-water mask",
          "It belongs in 05 Data\\dem (RUN ORDER.md, step 2.4).")
    hand = reproject_onto(sd / "strata_hand_m.tif", prof)
    slope = reproject_onto(sd / "strata_slope_deg.tif", prof)
    ls = reproject_onto(sd / "strata_layover_shadow.tif", prof, categorical=True)
    pw = reproject_onto(paths.perm_water(), prof, categorical=True)
    x = np.stack([vv, vh, hand, slope]).astype("float32")
    valid = np.isfinite(vv)
    if window is not None:                         # --smoke: one small box only
        r0, c0, h, w = window
        keep = np.zeros_like(valid)
        keep[r0:r0 + h, c0:c0 + w] = True
        valid &= keep
    tnan = int(np.count_nonzero(valid & ~(np.isfinite(hand) & np.isfinite(slope))))
    return {"x": x, "valid": valid, "profile": prof, "perm": pw == 100,
            "distort": np.isin(ls, (1, 2)), "terrain_nan_cells": tnan}


def make_flood_map(prob, valid, threshold, perm, distort):
    """The flood map, in the baselines' format, and its diagnostic counts."""
    water = valid & (np.nan_to_num(prob, nan=0.0) >= threshold)
    perm_rm = water & perm
    water2 = water & ~perm
    dist_rm = water2 & distort
    flood = water2 & ~distort
    out = np.full(valid.shape, NODATA, dtype="uint8")
    out[valid] = 0
    out[flood] = 1
    diag = {"valid_cells": int(valid.sum()), "water_raw_cells": int(water.sum()),
            "permanent_removed_cells": int(perm_rm.sum()),
            "distortion_removed_cells": int(dist_rm.sum()),
            "flood_cells": int(flood.sum()), "threshold": float(threshold)}
    return out, diag


def write_flood_map(path, arr, profile):
    """uint8, 1 band, nodata 255, LZW, on the scene's grid (as 03b writes it)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = profile.copy()
    prof.update(driver="GTiff", dtype="uint8", count=1, nodata=NODATA, compress="lzw")
    tmp = path.with_name(path.stem + "_part.tif")
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(arr.astype("uint8"), 1)
    os.replace(tmp, path)


def cell_km2(profile):
    t = profile["transform"]
    return abs(t.a * t.e) / 1e6


def area_warning(flood_km2, smoke=False):
    if smoke:
        return None
    if flood_km2 > FLOOD_KM2_HIGH:
        return (f"WARNING: {flood_km2:.2f} km2 of flood on the corridor. Above "
                f"{FLOOD_KM2_HIGH:.0f} km2 the map is almost certainly broken (the "
                "reference has 2.4 to 3.2 km2). Look at the map before scoring it.")
    if flood_km2 < FLOOD_KM2_LOW:
        return (f"WARNING: only {flood_km2:.3f} km2 of flood on the corridor. Below "
                f"{FLOOD_KM2_LOW} km2 the map is almost certainly broken (the reference "
                "has 2.4 to 3.2 km2). Look at the map before scoring it.")
    return None


# =============================================================================
# Tiled prediction
# =============================================================================

def tile_starts(n, tile, overlap):
    if n <= tile:
        return [0]
    step = max(tile - overlap, 1)
    starts = list(range(0, n - tile + 1, step))
    if starts[-1] != n - tile:
        starts.append(n - tile)
    return starts


def ramp(tile, overlap):
    """Blending weight along one tile axis: linear ramp at each edge, never 0."""
    if overlap <= 0:
        return np.ones(tile)
    k = np.arange(tile) + 0.5
    return np.minimum(1.0, np.minimum(k, tile - k) / overlap)


def predict_tiled(fn, x, valid, tile=512, overlap=64, batch=4):
    """Probability (H, W) float32, NaN where not valid.

    fn maps a (B, C, tile, tile) float32 array to (B, tile, tile)
    probabilities. Tiles overlap and are blended with a linear ramp, so seams
    vanish; tiles without one valid cell are skipped. Returns (prob, stats).
    """
    C, H, W = x.shape
    Hp, Wp = max(H, tile), max(W, tile)
    if (Hp, Wp) != (H, W):
        x = _pad(x, Hp, Wp, 0.0)
        valid = _pad(valid, Hp, Wp, False)
    acc = np.zeros((Hp, Wp), "float64")
    wsum = np.zeros((Hp, Wp), "float64")
    w1 = ramp(tile, overlap)
    w2 = np.outer(w1, w1)
    todo, skipped, done = [], 0, 0

    def flush():
        nonlocal done
        if not todo:
            return
        p = np.asarray(fn(np.stack([x[:, r:r + tile, c:c + tile] for r, c in todo])),
                       dtype="float64")
        for (r, c), pt in zip(todo, p):
            acc[r:r + tile, c:c + tile] += pt * w2
            wsum[r:r + tile, c:c + tile] += w2
        done += len(todo)
        todo.clear()

    for r in tile_starts(Hp, tile, overlap):
        for c in tile_starts(Wp, tile, overlap):
            if not valid[r:r + tile, c:c + tile].any():
                skipped += 1
                continue
            todo.append((r, c))
            if len(todo) >= batch:
                flush()
    flush()
    with np.errstate(invalid="ignore", divide="ignore"):
        prob = np.where(wsum > 0, acc / wsum, np.nan).astype("float32")
    prob[~valid] = np.nan
    return prob[:H, :W], {"tiles_predicted": done, "tiles_skipped": skipped}


# =============================================================================
# Metrics
# =============================================================================

def confusion(pred, truth):
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    tn = int(pred.size - tp - fp - fn)
    return tp, fp, fn, tn


def metrics_from_confusion(tp, fp, fn, tn):
    """IoU (water), F1, precision, recall, overall accuracy. nan when undefined."""
    n = tp + fp + fn + tn
    nan = float("nan")
    return {"iou": tp / (tp + fp + fn) if tp + fp + fn else nan,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else nan,
            "precision": tp / (tp + fp) if tp + fp else nan,
            "recall": tp / (tp + fn) if tp + fn else nan,
            "oa": (tp + tn) / n if n else nan}


class ThresholdHist:
    """Water / not-water counts per probability bin of 0.01, to choose a threshold.

    Bin k holds probabilities p with k/100 <= p < (k+1)/100 (bin 100 holds
    p = 1), so 'p >= threshold' is exactly 'bin >= 100 * threshold'.
    """
    EDGES = np.round(np.arange(101) / 100.0, 2)

    def __init__(self):
        self.water = np.zeros(101, "int64")
        self.dry = np.zeros(101, "int64")
        self.nonfinite = 0

    def add(self, prob, truth):
        prob = np.asarray(prob, "float64").ravel()
        truth = np.asarray(truth, bool).ravel()
        fin = np.isfinite(prob)
        if not fin.all():                      # a NaN must never count as p = 1
            self.nonfinite += int((~fin).sum())
            prob, truth = prob[fin], truth[fin]
        b = np.clip(np.searchsorted(self.EDGES, prob, side="right") - 1, 0, 100)
        self.water += np.bincount(b[truth], minlength=101)
        self.dry += np.bincount(b[~truth], minlength=101)

    def counts(self, threshold):
        k = int(round(threshold * 100))
        tp, fn = int(self.water[k:].sum()), int(self.water[:k].sum())
        fp, tn = int(self.dry[k:].sum()), int(self.dry[:k].sum())
        return tp, fp, fn, tn

    def iou(self, threshold):
        return metrics_from_confusion(*self.counts(threshold))["iou"]

    def best(self):
        """Threshold in 0.01..0.99 with the highest water IoU (ties: nearest 0.5)."""
        best_t, best_v = 0.5, -1.0
        for k in range(1, 100):
            t = self.EDGES[k]
            v = self.iou(t)
            if np.isnan(v):
                continue
            if v > best_v + 1e-12 or (abs(v - best_v) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5)):
                best_t, best_v = float(t), float(v)
        return best_t, (best_v if best_v >= 0 else float("nan"))


# =============================================================================
# Run records
# =============================================================================

def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git_head():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(HERE), capture_output=True,
                           text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def ram_total_gb():
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except ImportError:
        pass
    if os.name == "nt":
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullTotalPhys / 1e9
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return float("nan")


def disk_free_gb(path):
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def _version(mod):
    try:
        from importlib.metadata import version
        return version(mod)
    except Exception:                               # noqa: BLE001
        return "not installed"


def gpu_info():
    if torch is None:
        return {"gpu": "PyTorch not installed"}
    if not torch.cuda.is_available():
        return {"gpu": "none (torch.cuda.is_available() is False)"}
    p = torch.cuda.get_device_properties(0)
    return {"gpu": p.name, "gpu_vram_gb": round(p.total_memory / 1e9, 2),
            "gpu_capability": f"{p.major}.{p.minor}"}


def machine_info():
    info = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "hostname": socket.gethostname(), "os": platform.platform(),
            "cpu": platform.processor() or platform.machine(), "cpu_cores": os.cpu_count(),
            "ram_gb": round(ram_total_gb(), 1), "python": sys.version.split()[0],
            "numpy": np.__version__, "rasterio": rasterio.__version__,
            "sklearn": _version("scikit-learn"), "torch": _version("torch"),
            "git_head": git_head()}
    if torch is not None:
        info.update({"cuda": torch.version.cuda,
                     "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None})
    info.update(gpu_info())
    return info


def run_record(args, paths, **extra):
    """The common part of every run_record.json."""
    opts = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    rec = {"machine": machine_info(), "command": " ".join(sys.argv),
           "argv": list(sys.argv), "options": opts, "project": str(paths.root),
           "smoke": paths.smoke}
    rec.update(extra)
    return rec


def file_entry(path):
    path = Path(path)
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


# =============================================================================
# Devices and memory
# =============================================================================

def pick_device(cpu_ok=False):
    if torch is None:
        fail("PyTorch is not installed in this environment. The U-Net needs it. "
             "Activate the right environment: conda activate fyp")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if cpu_ok:
        print("WARNING: no CUDA GPU found. Running on the CPU because --cpu-ok was given. "
              "This is very slow for real training.")
        return torch.device("cpu")
    fail("No CUDA GPU was found (torch.cuda.is_available() is False).\n"
         "On a laptop with an NVIDIA GPU: check that nvidia-smi works in Anaconda Prompt and "
         "that 00_check_machine.py shows CUDA. On a laptop without one, run only the Random "
         "Forest there. To run the U-Net on the CPU anyway (very slow), add --cpu-ok.")


def rf_bytes_per_row(n_estimators, min_samples_leaf):
    """Upper bound of a fitted forest's RAM per training row, plus fitting buffers.

    A tree's bootstrap sample holds about 63.2 % of the rows as distinct
    samples, every leaf at least min_samples_leaf of them, and a tree has
    fewer than twice as many nodes as leaves. scikit-learn stores 64 bytes per
    node plus 16 bytes of class values (two classes, float64).
    """
    nodes_per_row = 2.0 * 0.632 / max(min_samples_leaf, 1)
    return n_estimators * nodes_per_row * 80.0 + 400.0


def rf_row_cap(max_ram_gb, n_estimators, min_samples_leaf):
    """Training rows that fit in 70 % of max_ram_gb, by the upper bound above."""
    return int(max_ram_gb * 1e9 * 0.7 / rf_bytes_per_row(n_estimators, min_samples_leaf))


# =============================================================================
# U-Net (plain PyTorch, no pretrained weights)
# =============================================================================

if torch is not None:

    def _double_conv(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True))

    class UNet(nn.Module):
        """4 encoder levels + bottleneck, two 3x3 conv-BN-ReLU per block, max-pool
        down, transposed convolution up, one output logit. Input sides must be
        multiples of 16."""

        def __init__(self, in_ch=len(CHANNELS), base=32, levels=4):
            super().__init__()
            w = [base * 2 ** i for i in range(levels + 1)]
            self.downs = nn.ModuleList(_double_conv(in_ch if i == 0 else w[i - 1], w[i])
                                       for i in range(levels))
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = _double_conv(w[levels - 1], w[levels])
            self.ups = nn.ModuleList(nn.ConvTranspose2d(w[i + 1], w[i], 2, stride=2)
                                     for i in reversed(range(levels)))
            self.decs = nn.ModuleList(_double_conv(2 * w[i], w[i])
                                      for i in reversed(range(levels)))
            self.head = nn.Conv2d(base, 1, 1)
            self.widths = w

        def forward(self, x):
            skips = []
            for d in self.downs:
                x = d(x)
                skips.append(x)
                x = self.pool(x)
            x = self.bottleneck(x)
            for up, dec, s in zip(self.ups, self.decs, reversed(skips)):
                x = dec(torch.cat([s, up(x)], dim=1))
            return self.head(x)[:, 0]

    def masked_bce_dice(logits, y, pos_weight=None):
        """BCE-with-logits plus soft Dice on water, both over valid labels only."""
        logits = logits.float()
        m = (y >= 0).float()
        t = (y == 1).float()
        n = m.sum().clamp_min(1.0)
        pw = None if pos_weight is None else torch.tensor(float(pos_weight),
                                                          device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none", pos_weight=pw)
        bce = (bce * m).sum() / n
        p = torch.sigmoid(logits) * m
        dice = 1.0 - (2.0 * (p * t).sum() + 1.0) / (p.sum() + (t * m).sum() + 1.0)
        return bce + dice, bce.detach(), dice.detach()

    def unet_prob_fn(model, device, amp):
        """fn((B, 4, h, w) normalised numpy) -> (B, h, w) water probability."""
        def fn(tiles):
            with torch.no_grad():
                t = torch.from_numpy(np.ascontiguousarray(tiles)).to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                    out = model(t)
                return torch.sigmoid(out.float()).cpu().numpy()
        return fn

    def unet_memory_estimate(base=32, batch=8, crop=256, amp=True):
        """VRAM estimate in GB for training, from parameter and activation counts.

        Activations: every convolution, batch-norm, pooling, up-convolution
        and concatenation output kept for the backward pass, counted with a
        shape-only ('meta') forward pass; 2 bytes each under fp16 autocast,
        times 1.5 for gradients alive during the backward pass. Parameters:
        fp32 weights, gradients and two AdamW moments (16 bytes each). Plus
        0.35 GB for cuDNN workspace and 0.6 GB for the CUDA context.
        """
        with torch.device("meta"):
            model = UNet(len(CHANNELS), base)
            x = torch.empty(1, len(CHANNELS), crop, crop)
        counted = [0]
        kinds = (nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d, nn.ConvTranspose2d)
        hooks = [m.register_forward_hook(lambda _m, _i, o: counted.__setitem__(0, counted[0] + o.numel()))
                 for m in model.modules() if isinstance(m, kinds)]
        model(x)
        for h in hooks:
            h.remove()
        cat = sum(2 * w * (crop // 2 ** i) ** 2 for i, w in enumerate(model.widths[:-1]))
        act = (counted[0] + cat) * batch
        params = sum(p.numel() for p in model.parameters())
        act_gb = act * (2 if amp else 4) * 1.5 / 1e9
        total = params * 16 / 1e9 + act_gb + 0.35 + 0.6
        return {"parameters": int(params), "activation_values": int(act),
                "activation_gb": round(act_gb, 3), "estimate_gb": round(total, 2)}

    # ---- data: everything the DataLoader pickles lives here, at module level --

    def worker_init(worker_id):
        """Seed each worker and let only the main process handle Ctrl+C."""
        s = torch.initial_seed() % 2 ** 32
        np.random.seed(s)
        random.seed(s)
        torch.set_num_threads(1)
        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        except (ValueError, OSError):
            pass

    def first_item(batch):
        return batch[0]

    class TrainCrops:
        """Random crops of training chips with flips and 90-degree turns.

        An item is (chip number, seed): the crop position and the turn come
        from the seed alone, so a run is the same whichever worker loads it.
        Holds only strings and numbers, so it pickles for Windows workers.
        """

        def __init__(self, rows, data_root, stats, crop, augment=True):
            self.rows = [{k: r.get(k, "") for k in ("chip", "s1", "label", "terrain",
                                                     "cache_x", "cache_y")} for r in rows]
            self.root = str(data_root)
            self.stats = {k: stats[k] for k in ("clip_lo", "clip_hi", "mean", "std")}
            self.crop, self.augment = int(crop), bool(augment)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, key):
            i, seed = key
            row = self.rows[int(i)]
            rng = np.random.default_rng(int(seed))
            H, W = chip_shape(row, self.root)
            r0 = int(rng.integers(0, max(H - self.crop, 0) + 1))
            c0 = int(rng.integers(0, max(W - self.crop, 0) + 1))
            h, w = min(self.crop, H), min(self.crop, W)
            x, y = read_chip(row, self.root, (r0, c0, h, w))
            x = normalise(_pad(x, self.crop, self.crop, np.nan), self.stats)
            y = _pad(y, self.crop, self.crop, -1)
            if self.augment:
                k = int(rng.integers(4))
                flip = bool(rng.integers(2))
                x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
                if flip:
                    x, y = x[:, :, ::-1], y[:, ::-1]
            return (torch.from_numpy(np.ascontiguousarray(x, dtype="float32")),
                    torch.from_numpy(np.ascontiguousarray(y, dtype="int8")))

    class FullChips:
        """Whole chips for validation and evaluation: (normalised x, input-valid, label)."""

        def __init__(self, rows, data_root, stats):
            self.rows = [{k: r.get(k, "") for k in ("chip", "s1", "label", "terrain",
                                                     "cache_x", "cache_y")} for r in rows]
            self.root = str(data_root)
            self.stats = {k: stats[k] for k in ("clip_lo", "clip_hi", "mean", "std")}

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            x, y = read_chip(self.rows[i], self.root)
            return normalise(x, self.stats), input_valid(x), y, self.rows[i]["chip"]

    class StepBatchSampler:
        """A fixed number of batches per epoch, the same for a given seed and epoch.

        Each batch takes round(batch * hand_fraction) hand chips and the rest
        weak chips (all hand when there are no weak chips). Each pool is walked
        through in shuffled passes. set_epoch(e, start) lets a resumed run
        continue at batch `start` of epoch e with exactly the batches it would
        have had.
        """

        def __init__(self, n_hand, n_weak, batch, steps, seed, hand_fraction=0.5):
            self.n_hand, self.n_weak = int(n_hand), int(n_weak)
            self.batch, self.steps, self.seed = int(batch), int(steps), int(seed)
            if self.n_weak == 0:
                self.k_hand = self.batch
            else:
                self.k_hand = min(max(int(round(self.batch * hand_fraction)), 1), self.batch - 1)
            self.epoch, self.start = 0, 0

        def set_epoch(self, epoch, start=0):
            self.epoch, self.start = int(epoch), int(start)

        def __len__(self):
            return self.steps - self.start

        @staticmethod
        def _stream(rng, n, k):
            out, need = [], k
            while need > 0:
                p = rng.permutation(n)
                out.append(p[:need])
                need -= len(p[:need])
            return np.concatenate(out)

        def plan(self):
            rng = np.random.default_rng([self.seed, self.epoch])
            kh, kw = self.k_hand, self.batch - self.k_hand
            hand = self._stream(rng, self.n_hand, self.steps * kh).reshape(self.steps, kh)
            weak = (self._stream(rng, self.n_weak, self.steps * kw).reshape(self.steps, kw) + self.n_hand
                    if kw else np.zeros((self.steps, 0), int))
            seeds = rng.integers(0, 2 ** 31 - 1, size=(self.steps, self.batch))
            return np.concatenate([hand, weak], axis=1), seeds

        def __iter__(self):
            idx, seeds = self.plan()
            for s in range(self.start, self.steps):
                yield [(int(i), int(sd)) for i, sd in zip(idx[s], seeds[s])]

    def make_loader(dataset, workers, seed, batch_sampler=None, cuda=False):
        kw = {"num_workers": int(workers), "worker_init_fn": worker_init,
              "generator": torch.Generator().manual_seed(int(seed)), "pin_memory": bool(cuda)}
        if workers > 0:
            # spawn on every system, so a test on Linux behaves like Windows
            kw.update(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2)
        if batch_sampler is not None:
            return torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, **kw)
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                           collate_fn=first_item, **kw)

    def set_deterministic(on):
        """--deterministic: deterministic algorithms, cudnn.benchmark off."""
        if on:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            torch.backends.cudnn.benchmark = True

    def is_oom(exc):
        return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()

    def find_batch_size(model, device, crop, batch, amp):
        """Largest batch size below `batch` whose forward and backward fit on the GPU."""
        bs = batch // 2
        while bs >= 1:
            try:
                x = torch.randn(bs, len(CHANNELS), crop, crop, device=device)
                y = torch.randint(0, 2, (bs, crop, crop), device=device, dtype=torch.int8)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                    out = model(x)
                masked_bce_dice(out, y)[0].backward()
                model.zero_grad(set_to_none=True)
                return bs
            except RuntimeError as e:
                if not is_oom(e):
                    raise
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                bs //= 2
        return 0


# =============================================================================
# TerraMind-small: encoder from terratorch, decoder here
# =============================================================================

TM_BACKBONE = "terramind_v1_small"             # terratorch's backbone registry name
TM_KEYS = {"S1GRD": "untok_sen1grd@224", "S1RTC": "untok_sen1rtc@224"}   # terratorch's names
TM_LAYERS = (2, 5, 8, 11)          # encoder blocks 3, 6, 9 and 12 of 12, counted from 0
TM_WIDTHS = (256, 128, 64, 32)     # decoder channels at 1/16, 1/8, 1/4 and 1/2 of the input
TM_STEM = 32                       # tm_s1t: channels of the HAND and slope stem
TM_CROP = 224                      # TerraMind's pretraining size: 14 x 14 patches of 16
TM_MIN_LOADED = 0.95               # stop if fewer backbone parameter tensors are filled
_TM_SAVED = {}                     # memory estimate: values kept for the backward, per crop


def tm_import():
    """terratorch's backbone registry and its TerraMind module.

    Imported only when a TerraMind model is built: importing terratorch takes
    10 to 30 seconds, and DataLoader workers never need it. Importing it also
    lowers PyTorch's float32 matrix-multiply precision to 'medium' (terratorch
    1.2.11, models/backbones/clay_v15/model.py, line 16); the setting is put
    back, so TerraMind trains with PyTorch's defaults, like the U-Net.
    """
    precision = torch.get_float32_matmul_precision() if torch is not None else None
    try:
        from terratorch.models.backbones.terramind.model import terramind_register
        from terratorch.registry import BACKBONE_REGISTRY
    except ImportError as e:
        fail(f"terratorch could not be imported ({e}). TerraMind needs the fyp environment, "
             "which has terratorch 1.2.11: conda activate fyp. Then run "
             "29_terramind_preflight.py.")
    finally:
        if precision is not None:
            torch.set_float32_matmul_precision(precision)
    return BACKBONE_REGISTRY, terramind_register


def tm_radar_norm(modality):
    """TerraMind's own standardisation of VV and VH (dB) for its S1GRD or S1RTC input.

    Read from terratorch, the values TerraMind was pretrained with:
    (value - mean) / std, without clipping.
    """
    _, reg = tm_import()
    key = TM_KEYS[modality]
    return {"modality": modality, "key": key,
            "mean": [float(v) for v in reg.v1_pretraining_mean[key]],
            "std": [float(v) for v in reg.v1_pretraining_std[key]], "clipping": "none",
            "source": f"terratorch {_version('terratorch')}, terratorch/models/backbones/"
                      "terramind/model/terramind_register.py, v1_pretraining_mean and "
                      f"v1_pretraining_std['{key}']"}


def tm_norm_record(radar, terrain_stats=None, terrain_from=""):
    """The exact normalisation of a TerraMind model, as tm_norm.json and run records hold it.

    radar: from tm_radar_norm. terrain_stats: a norm_stats.json dict (tm_s1t),
    whose HAND and slope entries are used; None for tm_s1.
    """
    rec = {"radar": radar, "terrain": None, "terrain_note": "tm_s1: HAND and slope are not used"}
    if terrain_stats is not None:
        rec["terrain"] = {"channels": ["HAND", "SLOPE"], "from": terrain_from,
                          **{k: [float(v) for v in terrain_stats[k][2:4]]
                             for k in ("clip_lo", "clip_hi", "mean", "std")}}
        rec["terrain_note"] = ("clipped to clip_lo..clip_hi, then (value - mean) / std, the "
                               "project's own rule; missing values become 0")
    return rec


def tm_stats(norm):
    """The four-channel statistics normalise() needs, from a tm_norm_record.

    VV and VH: TerraMind's mean and std and no clipping (bounds of -inf and
    inf), so normalise() gives exactly (value - mean) / std. HAND and slope:
    the project's clip bounds, mean and std for tm_s1t; for tm_s1 they are
    never used, so neutral values stand in.
    """
    inf = float("inf")
    radar, t = norm["radar"], norm["terrain"]
    if t is None:
        t = {"clip_lo": [-inf, -inf], "clip_hi": [inf, inf], "mean": [0.0, 0.0],
             "std": [1.0, 1.0]}
    return {"clip_lo": [-inf, -inf] + list(t["clip_lo"]),
            "clip_hi": [inf, inf] + list(t["clip_hi"]),
            "mean": list(radar["mean"]) + list(t["mean"]),
            "std": list(radar["std"]) + list(t["std"])}


if torch is not None:

    def _up2(cin, cout):
        """Double the resolution: 2 x 2 transposed convolution, batch norm, ReLU."""
        return nn.Sequential(nn.ConvTranspose2d(cin, cout, 2, stride=2, bias=False),
                             nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    class TerraMindNet(nn.Module):
        """TerraMind-small encoder (terratorch) and a progressive up-sampling decoder.

        The encoder sees VV and VH only, through TerraMind's S1GRD or S1RTC
        input. The tokens of encoder blocks 3, 6, 9 and 12 are laid out as
        h/16 x w/16 maps (row by row, as terratorch makes the patches) and
        brought to 1/16 (block 12), 1/8 (block 9), 1/4 (block 6) and 1/2
        (block 3) of the input size. The decoder fuses them from the coarsest
        up, U-Net style (up-sample, concatenate, two 3x3 conv-BN-ReLU),
        up-samples to full size and gives one logit. tm_s1t also passes HAND and
        slope through a stem at full resolution (two 3x3 conv-BN-ReLU, 32
        channels) that is concatenated with the decoder's full-resolution
        features before its last block. Input (B, 4, h, w): VV, VH, HAND, SLOPE,
        normalised; h and w multiples of 16. tm_s1 never uses HAND and SLOPE.
        """

        def __init__(self, variant, modality="S1GRD"):
            super().__init__()
            if variant not in TM_VARIANTS:
                raise ValueError(f"variant must be one of {TM_VARIANTS}, not {variant!r}")
            if modality not in TM_KEYS:
                raise ValueError(f"modality must be one of {tuple(TM_KEYS)}, not {modality!r}")
            registry, _ = tm_import()
            self.variant, self.modality, self.terrain = variant, modality, variant == "tm_s1t"
            self.backbone = registry.build(TM_BACKBONE, modalities=[modality], pretrained=False,
                                           merge_method=None)
            dim = int(self.backbone.out_channels[TM_LAYERS[-1]])
            w16, w8, w4, w2 = TM_WIDTHS
            self.neck = nn.ModuleList([
                nn.Sequential(_up2(dim, w8), _up2(w8, w4), _up2(w4, w2)),     # block 3: 1/2
                nn.Sequential(_up2(dim, w8), _up2(w8, w4)),                    # block 6: 1/4
                _up2(dim, w8),                                                 # block 9: 1/8
                nn.Sequential(nn.Conv2d(dim, w16, 1, bias=False), nn.BatchNorm2d(w16),
                              nn.ReLU(inplace=True))])                         # block 12: 1/16
            self.ups = nn.ModuleList([_up2(w16, w8), _up2(w8, w4), _up2(w4, w2), _up2(w2, w2)])
            self.decs = nn.ModuleList([_double_conv(2 * w8, w8), _double_conv(2 * w4, w4),
                                       _double_conv(2 * w2, w2)])
            self.stem = _double_conv(2, TM_STEM) if self.terrain else None
            self.last = _double_conv(w2 + (TM_STEM if self.terrain else 0), w2)
            self.head = nn.Conv2d(w2, 1, 1)

        def decoder_parameters(self):
            """Everything but the encoder: neck, decoder, terrain stem and head."""
            return [p for n, p in self.named_parameters() if not n.startswith("backbone.")]

        def parameter_counts(self):
            enc = sum(p.numel() for p in self.backbone.parameters())
            return {"backbone": int(enc), "decoder": int(sum(p.numel() for p in
                                                             self.decoder_parameters()))}

        def forward(self, x):
            b, _, h, w = x.shape
            tokens = self.backbone(x[:, :2])               # 12 x (B, h/16 * w/16, 384)
            maps = [tokens[i].transpose(1, 2).reshape(b, -1, h // 16, w // 16)
                    for i in TM_LAYERS]
            f2, f4, f8, f16 = (n(m) for n, m in zip(self.neck, maps))
            y = self.decs[0](torch.cat([f8, self.ups[0](f16)], dim=1))
            y = self.decs[1](torch.cat([f4, self.ups[1](y)], dim=1))
            y = self.decs[2](torch.cat([f2, self.ups[2](y)], dim=1))
            y = self.ups[3](y)
            if self.terrain:
                y = torch.cat([y, self.stem(x[:, 2:4])], dim=1)
            return self.head(self.last(y))[:, 0]

    def terramind_memory_estimate(variant, batch=8, crop=TM_CROP, amp=True):
        """VRAM estimate in GB for training, by the same rule as the U-Net's.

        Activations: every tensor autograd keeps for the backward pass of one
        crop, counted with saved-tensor hooks during a real forward pass on the
        CPU (parameters themselves not counted), times the batch; 2 bytes each
        under fp16 autocast, times 1.5 for gradients alive during the backward
        pass. Parameters: 16 bytes each (fp32 weights, gradients, two AdamW
        moments). Plus 0.35 GB for cuDNN workspace and 0.6 GB for the CUDA
        context. The random state is left as it was.
        """
        key = (variant, int(crop))
        if key not in _TM_SAVED:
            with torch.random.fork_rng(devices=[]):
                net = TerraMindNet(variant)
                own = {p.untyped_storage().data_ptr() for p in net.parameters()}
                seen = {}

                def pack(t):
                    s = t.untyped_storage()
                    if t.is_floating_point() and s.data_ptr() not in own:
                        seen[s.data_ptr()] = s.nbytes() // t.element_size()
                    return t

                with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                    net(torch.zeros(1, len(CHANNELS), crop, crop))
                _TM_SAVED[key] = (sum(seen.values()), net.parameter_counts())
        per_crop, counts = _TM_SAVED[key]
        params = counts["backbone"] + counts["decoder"]
        act = per_crop * batch
        act_gb = act * (2 if amp else 4) * 1.5 / 1e9
        total = params * 16 / 1e9 + act_gb + 0.35 + 0.6
        return {"parameters": int(params), "backbone_parameters": counts["backbone"],
                "decoder_parameters": counts["decoder"], "activation_values": int(act),
                "activation_gb": round(act_gb, 3), "estimate_gb": round(total, 2)}

    def tm_read_weights(path, backbone):
        """The tensors of a TerraMind checkpoint that fill this backbone, and a report.

        Only keys the backbone has, with the same shape, are taken: the file also
        holds TerraMind's decoder and its other inputs. Stops with a sentence if
        fewer than 95 % of the backbone's parameter tensors would be filled.
        """
        path = Path(path)
        if not path.exists():
            fail(f"The TerraMind weights are missing:\n  {path}\nPut TerraMind_v1_small.pt "
                 "(504 MB, from huggingface.co/ibm-esa-geospatial/TerraMind-1.0-small) "
                 "there. The code never downloads it.")
        try:
            sd = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e:                          # noqa: BLE001
            fail(f"{path.name} could not be read as PyTorch weights ({type(e).__name__}: {e}). "
                 "The file is probably damaged or incomplete: download it again.")
        if not isinstance(sd, dict):
            sd = {}
        own = backbone.state_dict()
        names = [n for n, _ in backbone.named_parameters()]
        take, wrong = {}, []
        for k, v in own.items():
            t = sd.get(k)
            if torch.is_tensor(t):
                if tuple(t.shape) == tuple(v.shape):
                    take[k] = t
                else:
                    wrong.append(f"{k}: {list(t.shape)} in the file, {list(v.shape)} here")
        filled = [n for n in names if n in take]
        missing = [n for n in names if n not in take]
        buffers = [k for k in own if k not in set(names)]
        unused = [k for k in sd if k not in own]
        share = len(filled) / max(len(names), 1)
        rep = {"file": str(path), "file_tensors": len(sd),
               "backbone_parameter_tensors": len(names), "filled_parameter_tensors": len(filled),
               "filled_share": round(share, 4),
               "filled_values": int(sum(own[n].numel() for n in filled)),
               "backbone_values": int(sum(own[n].numel() for n in names)),
               "buffers": len(buffers), "buffers_filled": int(sum(k in take for k in buffers)),
               "missing": missing, "wrong_shape": wrong, "unused_tensors": len(unused),
               "unused_examples": unused[:8]}
        if share < TM_MIN_LOADED:
            fail(f"Only {len(filled)} of the {len(names)} parameter tensors of the TerraMind "
                 f"backbone ({share:.0%}) are in {path.name}; at least "
                 f"{TM_MIN_LOADED:.0%} are needed, or the model would start mostly from random "
                 "numbers. Is it the right file (TerraMind_v1_small.pt, 504 MB, from "
                 "huggingface.co/ibm-esa-geospatial/TerraMind-1.0-small)?\n"
                 f"  first keys in the file: {list(sd)[:5]}\n"
                 f"  first keys not found  : {missing[:5]}\n"
                 + (f"  wrong shape           : {wrong[:3]}\n" if wrong else "")
                 + "Download the weights file again.")
        return take, rep

    def tm_apply_weights(backbone, take):
        """Copy the tensors from tm_read_weights into the backbone; the rest stay as they are."""
        state = backbone.state_dict()
        state.update(take)
        backbone.load_state_dict(state, strict=True)

    def tm_weights_lines(rep):
        """The key-match report in plain lines."""
        out = [f"TerraMind weights: {rep['filled_parameter_tensors']} of "
               f"{rep['backbone_parameter_tensors']} backbone parameter tensors filled "
               f"({rep['filled_share']:.1%}, {rep['filled_values']:,} of "
               f"{rep['backbone_values']:,} values); {rep['buffers_filled']} of {rep['buffers']} "
               "fixed buffers",
               f"  missing (left as built): {', '.join(rep['missing']) or 'none'}"]
        if rep["wrong_shape"]:
            out.append(f"  wrong shape, not used: {'; '.join(rep['wrong_shape'])}")
        out.append(f"  {rep['unused_tensors']} of the {rep['file_tensors']} tensors in the file "
                   "belong to other parts of TerraMind (its decoder, other inputs) and are "
                   "not used" + (f", e.g. {', '.join(rep['unused_examples'][:3])}"
                                 if rep["unused_examples"] else ""))
        return out


# =============================================================================
# Saved models and the predictor registry
# =============================================================================

def rf_water_prob(model, f, n_jobs=None):
    """Water probability from a fitted forest, bit-identical on every run.

    scikit-learn's own multi-threaded predict_proba adds the trees up in the
    order the threads finish, so two calls can differ in the 16th digit.
    Here the rows are split into blocks, one per thread, and each block adds
    the trees in their fixed order.
    """
    from concurrent.futures import ThreadPoolExecutor
    classes = list(model.classes_)
    out = np.zeros(len(f), "float32")
    if 1 not in classes or len(f) == 0:
        return out
    col = classes.index(1)
    model.set_params(n_jobs=1)
    n_jobs = max(1, n_jobs or (os.cpu_count() or 2) - 1)
    blocks = [b for b in np.array_split(np.arange(len(f)), n_jobs) if b.size]
    with ThreadPoolExecutor(len(blocks)) as ex:
        parts = list(ex.map(lambda b: model.predict_proba(f[b[0]:b[-1] + 1])[:, col], blocks))
    return np.concatenate(parts).astype("float32")


class RFPredictor:
    """A saved Random Forest: refuses one saved by another sklearn major version."""

    kind = "rf"

    def __init__(self, model_dir, max_ram_gb=None, **_):
        import joblib
        import sklearn
        model_dir = Path(model_dir)
        meta_p = model_dir / "model_meta.json"
        if not meta_p.exists():
            fail(f"No trained Random Forest in\n  {model_dir}\nRun 22_train_rf.py first.")
        self.meta = load_json(meta_p)
        need = self.meta.get("forest_nodes", 0) * 80 / 1e9 + 1.0
        if max_ram_gb is not None and need > max_ram_gb:
            fail(f"This Random Forest needs about {need:.1f} GB of RAM to load, more than "
                 f"--max-ram-gb {max_ram_gb:g}. Use a larger --max-ram-gb if this machine has "
                 "the RAM, or train the forest on this machine with its own --max-ram-gb.")
        saved = str(self.meta.get("sklearn_version", "?"))
        if saved.split(".")[0] != sklearn.__version__.split(".")[0]:
            fail(f"This Random Forest was saved with scikit-learn {saved}; this environment "
                 f"has {sklearn.__version__}. A model cannot be loaded across major versions. "
                 "Train it again here (22_train_rf.py --force), or use the environment it "
                 "was trained in.")
        if saved.split(".")[:2] != sklearn.__version__.split(".")[:2]:
            print(f"WARNING: model saved with scikit-learn {saved}, loading with "
                  f"{sklearn.__version__}. Same major version, so it is allowed.")
        self.model = joblib.load(model_dir / "model.joblib")
        self.stats = load_json(model_dir / "norm_stats.json")
        self.threshold = float(self.meta["threshold_valid"])
        self.model_dir = model_dir

    def prob_features(self, f):
        """Water probability for (n, 4) clipped features, 1 million rows at a time."""
        out = np.empty(len(f), "float32")
        prog = Progress((len(f) + CHUNK - 1) // CHUNK, "million-pixel chunks predicted",
                        every=60) if len(f) > 3 * CHUNK else None
        for i in range(0, len(f), CHUNK):
            out[i:i + CHUNK] = rf_water_prob(self.model, f[i:i + CHUNK])
            if prog:
                prog.tick()
        return out

    def predict_image(self, x, valid):
        prob = np.full(valid.shape, np.nan, "float32")
        if valid.any():
            prob[valid] = self.prob_features(rf_features(x[:, valid].T, self.stats))
        return prob


class UNetPredictor:
    """A saved U-Net (best.pt) with its own normalisation and threshold."""

    kind = "unet"

    def __init__(self, model_dir, device=None, tile=512, overlap=64, batch=4, cpu_ok=False, **_):
        model_dir = Path(model_dir)
        ck = model_dir / "best.pt"
        meta_p = model_dir / "model_meta.json"
        if not ck.exists() or not meta_p.exists():
            fail(f"No finished U-Net in\n  {model_dir}\nRun 23_train_unet.py first "
                 "(or --resume it until it says TRAINING FINISHED).")
        self.meta = load_json(meta_p)
        self.device = device or pick_device(cpu_ok)
        state = torch.load(ck, map_location="cpu", weights_only=False)   # our own file
        self.model = UNet(len(CHANNELS), int(self.meta["base_width"]))
        self.model.load_state_dict(state["model"])
        self.model.to(self.device).eval()
        self.stats = load_json(model_dir / "norm_stats.json")
        self.threshold = float(self.meta["threshold_valid"])
        self.tile, self.overlap, self.batch = int(tile), int(overlap), int(batch)
        self.amp = self.device.type == "cuda"
        self.model_dir = model_dir

    def predict_image(self, x, valid):
        fn = unet_prob_fn(self.model, self.device, self.amp)
        prob, _ = predict_tiled(fn, normalise(x, self.stats), valid, self.tile,
                                self.overlap, self.batch)
        return prob


class TerraMindPredictor:
    """A saved TerraMind model (best.pt) with its own normalisation and threshold.

    It predicts in tiles of its training crop (224 x 224 by default): TerraMind's
    position embeddings were made for that size. A --tile option is not used.
    """

    kind = "terramind"

    def __init__(self, model_dir, device=None, tile=None, overlap=64, batch=4, cpu_ok=False,
                 **_):
        model_dir = Path(model_dir)
        ck = model_dir / "best.pt"
        meta_p = model_dir / "model_meta.json"
        if not ck.exists() or not meta_p.exists():
            fail(f"No finished TerraMind model in\n  {model_dir}\nRun 30_train_terramind.py "
                 "first (or --resume it until it says TRAINING FINISHED).")
        self.meta = load_json(meta_p)
        self.device = device or pick_device(cpu_ok)
        state = torch.load(ck, map_location="cpu", weights_only=False)   # our own file
        self.model = TerraMindNet(self.meta["variant"], self.meta["modality"])
        self.model.load_state_dict(state["model"])
        self.model.to(self.device).eval()
        self.norm = load_json(model_dir / "tm_norm.json")
        self.stats = tm_stats(self.norm)
        self.threshold = float(self.meta["threshold_valid"])
        self.tile, self.overlap, self.batch = int(self.meta["crop"]), int(overlap), int(batch)
        check_tile_args(self.tile, self.overlap)
        self.amp = self.device.type == "cuda"
        self.uses_terrain = self.model.terrain
        self.model_dir = model_dir
        self.record = {"variant": self.meta["variant"], "modality": self.meta["modality"],
                       "tile": self.tile, "terramind_norm": self.norm,
                       "terramind_weights": self.meta.get("terramind_weights"),
                       "terramind_weights_sha256": self.meta.get("terramind_weights_sha256")}
        print(f"  TerraMind {self.meta['variant']} ({self.meta['modality']} input) predicts in "
              f"{self.tile} x {self.tile} tiles, the size it was trained on")

    def predict_image(self, x, valid):
        fn = unet_prob_fn(self.model, self.device, self.amp)
        prob, _ = predict_tiled(fn, normalise(x, self.stats), valid, self.tile, self.overlap,
                                self.batch)
        return prob


PREDICTORS = {"rf": RFPredictor, "unet": UNetPredictor, "tm_s1": TerraMindPredictor,
              "tm_s1t": TerraMindPredictor}


def load_predictor(model, model_dir, **kw):
    if model not in PREDICTORS:
        fail(f"--model must be one of {', '.join(PREDICTORS)}, not '{model}'.")
    return PREDICTORS[model](model_dir, **kw)


def check_tile_args(tile, overlap):
    if tile % 16 or tile < 16:
        fail(f"--tile must be a multiple of 16 (the U-Net halves it four times); got {tile}.")
    if not 0 <= overlap < tile // 2:
        fail(f"--overlap must be at least 0 and less than half the tile; got {overlap}.")


# =============================================================================
# Self-test
# =============================================================================

def selftest():
    print("models_common.py self-test")
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}".rstrip())
        ok &= bool(cond)

    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 300, 700)).astype("float32")
    valid = np.ones((300, 700), bool)
    valid[:, 450:] = False
    prob, st = predict_tiled(lambda t: t[:, 0], x, valid, tile=128, overlap=32, batch=3)
    check("tiler: identity model reproduces the input exactly",
          np.array_equal(prob[valid], x[0][valid]))
    check("tiler: blank tiles skipped", st["tiles_skipped"] > 0, f"{st}")
    pv = np.array([[0.9, 0.9, 0.2], [0.9, np.nan, 0.6]], "float32")
    vm = np.isfinite(pv)
    m, d = make_flood_map(pv, vm, 0.5, np.array([[1, 0, 0], [0, 0, 0]], bool),
                          np.array([[0, 1, 0], [0, 0, 0]], bool))
    check("flood map: permanent water and distortion become 0, invalid 255",
          m.tolist() == [[0, 0, 0], [1, 255, 1]], f"{m.tolist()}")
    stats = compute_norm_stats(np.array([[1, 2, np.nan, 4.0]] * 10), "test")
    n = normalise(np.array([[[np.nan]], [[2.0]], [[5.0]], [[4.0]]], "float32"), stats)
    check("normalise: non-finite inputs become 0", float(n[0, 0, 0]) == 0.0)
    th = ThresholdHist()
    th.add(np.array([0.29, 0.5, 0.7]), np.array([True, True, False]))
    check("threshold bins: p >= t", th.counts(0.29)[0] == 2 and th.counts(0.5)[0] == 1)
    with tempfile.TemporaryDirectory(prefix="fyp selftest ") as td:
        p = Paths(Path(td) / "Final Year Project")
        check("paths: project folder with spaces", " " in str(p.models))
    print("SELFTEST PASS" if ok else "SELFTEST FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest() if a.selftest else ap.print_help() or 0)
