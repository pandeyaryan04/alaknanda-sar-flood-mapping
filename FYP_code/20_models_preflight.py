r"""
20_models_preflight.py
======================
Run this FIRST, before any model script. It checks this machine and every
data file the models need against the formats the model scripts expect,
and stops nothing: it only reads, then prints one line per check

    [PASS] ...    fine
    [WARN] ...    unusual but the models handle it; read it
    [FAIL] ...    must be fixed first; the line says what to do

and a final verdict, PREFLIGHT PASS or PREFLIGHT FAIL. A copy goes to
06 Outputs\models\preflight_<computer name>.md (keep that file if anything
fails). It never overwrites an earlier report: a second run gets
the time added to its name.

What it checks: Python and package versions, the GPU (with a small
mixed-precision test on it), RAM and free disk; Sen1Floods11 file counts
(446 hand, 4,384 weak), every chip having its label and terrain file, the
four split files (252, 89, 90, 15), no chip in two splits, Bolivia only in
the Bolivia split; inside 50 random train, valid and weak chips (never test
or Bolivia): band count, data type, dB values, label values (-1, 0, 1
only), grids; and on the corridor: the GRD
and RTC scenes (bands VV, VH, angle; 7915 x 3044 cells at 10 m in
EPSG:32644), the 30 m strata, the permanent-water mask (0 or 100), the
reference rasters, windows and GeoPackages.

USAGE: HOW TO RUN THIS  (copy one line, paste into Anaconda Prompt, press Enter)
================================================================================
    conda activate fyp
    cd /d "<project folder>\04 Code\FYP_code"

    python 20_models_preflight.py --selftest
    python 20_models_preflight.py

Takes 1 to 3 minutes. Or double-click RUN MODELS PREFLIGHT.bat.
================================================================================
"""

__stage__ = "Models: preflight of machine and data"
__reads__ = ["the machine", "Sen1Floods11", "sen1floods11_terrain", "corridor scenes, strata, "
             "permanent water, reference rasters, windows, GeoPackages"]
__writes__ = ["06 Outputs/models/preflight_<hostname>.md"]

import argparse
import csv
import json
import socket
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import models_common as mc

try:
    import rasterio
except ImportError as e:
    sys.exit(f"\nMissing package: {e}.\nActivate the environment first:  conda activate fyp\n")

EXPECTED = {
    "s1hand": 446, "labelhand": 446, "s1weak": 4384, "labelweak": 4384,
    "split_train": 252, "split_valid": 89, "split_test": 90, "split_bolivia": 15,
    "terrain_hand": 445, "terrain_weak": 4374, "weak_bolivia": 224,
    "empty_hand": 1, "empty_weak": 10,
    "corridor_width": 7915, "corridor_height": 3044,
    "corridor_transform": (10.0, 0.0, 231660.0, 0.0, -10.0, 3357010.0),
    "strata_width": 2639, "strata_height": 1016,
    "strata_transform": (30.0, 0.0, 231660.0, 0.0, -30.0, 3357030.0),
    "windows": 8,
}
EVENTS = {"Ghana", "India", "Mekong", "Nigeria", "Pakistan", "Paraguay", "Somalia", "Spain",
          "Sri-Lanka", "USA"}
PACKAGES = [("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"),
            ("sklearn", "scikit-learn"), ("rasterio", "rasterio"), ("geopandas", "geopandas"),
            ("shapely", "shapely"), ("pyproj", "pyproj"), ("pyogrio", "pyogrio"),
            ("joblib", "joblib"), ("matplotlib", "matplotlib"), ("torch", "torch")]
SAMPLE = 50
LOG_COLS = ["chip", "status", "seconds", "downloaded", "hand_defined_pct", "hand_median_m",
            "hand_lt5m_pct", "slope_median_deg", "dem_min_m", "dem_max_m"]


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, status, name, detail=""):
        self.rows.append({"status": status, "name": name, "detail": detail})
        print(f"  [{status}] {name}" + (f": {detail}" if detail else ""), flush=True)

    def ok(self, cond, name, good, bad, level="FAIL"):
        self.add("PASS" if cond else level, name, good if cond else bad)
        return cond


def _tifs(folder, suffix):
    return mc.list_chips(folder, suffix)


def _grid(path):
    with rasterio.open(path) as s:
        return {"shape": s.shape, "transform": s.transform, "crs": s.crs, "count": s.count,
                "dtypes": s.dtypes, "nodata": s.nodata, "desc": s.descriptions}


def _same_grid(a, b):
    """Same size and CRS, cell centres less than 0.01 pixel apart (rounding is allowed)."""
    return mc.same_grid(a["shape"], a["transform"], a["crs"], b["shape"], b["transform"], b["crs"])


def _list(xs, n=5):
    xs = list(xs)
    return ", ".join(map(str, xs[:n])) + (f" and {len(xs) - n} more" if len(xs) > n else "")


# ============================================================================= machine

def check_machine(c):
    v = sys.version_info
    c.ok(v[:2] == (3, 11), "Python version", f"{sys.version.split()[0]}",
         f"{sys.version.split()[0]}; the fyp environment has 3.11. Run: conda activate fyp", "WARN")
    for mod, pip_name in PACKAGES:
        try:
            __import__(mod)
            c.add("PASS", f"package {pip_name}", mc._version(pip_name))
        except ImportError:
            level = "WARN" if mod == "torch" else "FAIL"
            c.add(level, f"package {pip_name}", "not importable. Activate the environment: "
                  "conda activate fyp" + (" (only the U-Net needs it)" if level == "WARN" else ""))
    ram = mc.ram_total_gb()
    c.ok(ram >= 15.0, "RAM", f"{ram:.1f} GB; the default --max-ram-gb 12 is fine",
         f"{ram:.1f} GB; run the model scripts with --max-ram-gb {max(ram * 0.6, 2):.0f}", "WARN")
    free = mc.disk_free_gb(mc.HERE)
    if free < 30:
        c.add("FAIL", "free disk", f"{free:.0f} GB; the models need about 30 GB (15 GB of it for "
              "the chip cache). Free some space.")
    else:
        c.add("PASS", "free disk", f"{free:.0f} GB")
    if mc.torch is None:
        c.add("WARN", "GPU", "PyTorch missing: only the Random Forest can run here")
        return
    torch = mc.torch
    if not torch.cuda.is_available():
        c.add("WARN", "GPU", "no CUDA GPU (torch.cuda.is_available() is False). Only the Random "
              "Forest can run on this machine; run the U-Net on the RTX 3050 laptop.")
        return
    g = mc.gpu_info()
    c.ok(g["gpu_vram_gb"] >= 3.5, "GPU", f"{g['gpu']}, {g['gpu_vram_gb']} GB, compute "
         f"capability {g['gpu_capability']}, CUDA {torch.version.cuda}",
         f"{g['gpu']} has only {g['gpu_vram_gb']} GB; use --batch-size 4", "WARN")
    try:
        dev = torch.device("cuda")
        m = mc.UNet(4, 8).to(dev)
        x = torch.randn(2, 4, 64, 64, device=dev)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = m(x)
        mc.masked_bce_dice(out, torch.ones(2, 64, 64, dtype=torch.int8, device=dev))[0].backward()
        torch.cuda.synchronize()
        c.add("PASS", "GPU mixed-precision test", "a small U-Net ran forward and backward on the GPU")
    except Exception as e:                          # noqa: BLE001
        c.add("FAIL", "GPU mixed-precision test", f"{e}. Update the NVIDIA driver, restart, "
              "and run this again.")


# ============================================================================= Sen1Floods11

def check_s1f(c, paths, exp, rng):
    if not c.ok(paths.events.is_dir(), "Sen1Floods11 folder", str(paths.s1f),
                f"{paths.events} not found. Sen1Floods11 belongs in FYP_code\\data\\sen1floods11"
                "\\v1.1 (RUN ORDER.md step 1.3, 02_fetch_sen1floods11.py)."):
        return
    names = {}
    for key, set_, which, suffix in (("s1hand", "hand", "s1", "_S1Hand.tif"),
                                     ("labelhand", "hand", "label", "_LabelHand.tif"),
                                     ("s1weak", "weak", "s1", "_S1Weak.tif"),
                                     ("labelweak", "weak", "label", "_S2IndexLabelWeak.tif")):
        folder = paths.s1_dir(set_) if which == "s1" else paths.label_dir(set_)
        ids = _tifs(folder, suffix)
        names[key] = ids
        other = [f.name for f in folder.iterdir() if not f.name.endswith(".tif")] \
            if folder.is_dir() else []
        c.ok(len(ids) == exp[key], f"{folder.name} file count",
             f"{len(ids)}" + (f" (ignored: {_list(other)})" if other else ""),
             f"{len(ids)} .tif files, expected {exp[key]} in {folder}. The download is "
             "incomplete or misplaced: rerun 02_fetch_sen1floods11.py.")
    for s1k, lk, what in (("s1hand", "labelhand", "S1Hand"), ("s1weak", "labelweak", "S1Weak")):
        miss = sorted(set(names[s1k]) - set(names[lk]))
        c.ok(not miss, f"every {what} chip has its label", "yes",
             f"{len(miss)} without a label file, e.g. {_list(miss)}")
    both = sorted(set(names["s1hand"]) & set(names["s1weak"]))
    c.ok(not both, "no chip in both the hand and the weak set", "none",
         f"{len(both)} chip ids in both, e.g. {_list(both)}")
    nb = sum(mc.event_of(x) == "Bolivia" for x in names["s1weak"])
    c.ok(nb == exp["weak_bolivia"], "weak chips of event Bolivia (all left out of training)",
         f"{nb}", f"{nb}, expected {exp['weak_bolivia']}; they are left out either way", "WARN")

    # split files
    split_of = {}
    for split, fname in mc.SPLIT_FILES.items():
        f = paths.splits / fname
        if not c.ok(f.exists(), f"split file {fname}", "found",
                    f"missing: {f}. It comes with Sen1Floods11 (v1.1/splits/flood_handlabeled)."):
            continue
        bad_rows = []
        with open(f, newline="", encoding="utf-8-sig") as fh:
            for rec in csv.reader(fh):
                if rec and (len(rec) != 2 or not rec[0].strip().endswith("_S1Hand.tif")
                            or not rec[1].strip().endswith("_LabelHand.tif")):
                    bad_rows.append(",".join(rec))
        try:
            ids = mc.read_split_csv(f)
        except ValueError as e:
            c.add("FAIL", f"split file {fname}", str(e))
            continue
        c.ok(not bad_rows, f"split file {fname} format", "two columns: S1Hand file, LabelHand file",
             f"{len(bad_rows)} rows not like '<chip>_S1Hand.tif,<chip>_LabelHand.tif', e.g. "
             f"{_list(bad_rows, 2)}")
        c.ok(len(ids) == exp[f"split_{split}"], f"split {split} size", f"{len(ids)} chips",
             f"{len(ids)} chips, expected {exp[f'split_{split}']}")
        missing = [x for x in ids if x not in set(names["s1hand"])]
        c.ok(not missing, f"split {split} chips exist", "all in S1Hand",
             f"{len(missing)} not in S1Hand, e.g. {_list(missing)}")
        dup = [x for x in ids if x in split_of]
        c.ok(not dup, f"split {split} overlaps no other split", "no overlap",
             f"{len(dup)} chips also in another split, e.g. {_list(dup)}")
        for x in ids:
            split_of[x] = split
        ev = {mc.event_of(x) for x in ids}
        if split == "bolivia":
            c.ok(ev <= {"Bolivia"}, "Bolivia split holds only Bolivia", "yes", f"events {sorted(ev)}")
        else:
            c.ok("Bolivia" not in ev, f"no Bolivia chip in {split}", "none",
                 "Bolivia chips found; Bolivia must be held out")
            odd = ev - EVENTS
            c.ok(not odd, f"events in {split}", ", ".join(sorted(ev)),
                 f"unexpected events {sorted(odd)}", "WARN")

    # terrain, every chip
    for set_, key in (("hand", "s1hand"), ("weak", "s1weak")):
        tdir = paths.terrain_dir(set_)
        if not c.ok(tdir.is_dir(), f"terrain folder {set_}", str(tdir),
                    f"{tdir} not found. Run: python 11_sen1floods11_terrain.py --set {set_}"):
            continue
        have = set(_tifs(tdir, "_Terrain.tif"))
        c.ok(len(have) == exp[f"terrain_{set_}"], f"terrain files {set_}", f"{len(have)}",
             f"{len(have)}, expected {exp[f'terrain_{set_}']}", "WARN")
        empty = set()
        log = tdir / "terrain_log.csv"
        if c.ok(log.exists(), f"terrain log {set_}", "found",
                f"{log} missing. Run: python 11_sen1floods11_terrain.py --set {set_}"):
            with open(log, newline="", encoding="utf-8") as fh:
                rd = csv.DictReader(fh)
                cols = rd.fieldnames or []
                rows = list(rd)
            c.ok(cols[:2] == ["chip", "status"] and set(LOG_COLS) <= set(cols),
                 f"terrain log {set_} columns", "as written by 11_sen1floods11_terrain.py",
                 f"columns {cols}", "WARN")
            empty = {r["chip"] for r in rows if r.get("status", "").startswith("empty chip")}
            c.ok(len(empty) == exp[f"empty_{set_}"], f"empty chips ({set_}, no radar)",
                 f"{len(empty)} (left out)", f"{len(empty)}, expected {exp[f'empty_{set_}']}",
                 "WARN")
        miss = sorted(set(names[key]) - have - empty)
        c.ok(not miss, f"terrain for every {set_} chip", "every chip with radar has a terrain file",
             f"{len(miss)} chips have no terrain file, e.g. {_list(miss)}. Run: python "
             f"11_sen1floods11_terrain.py --set {set_} (it continues where it stopped)")

    # internals of a sample of chips; never test or Bolivia, whose labels only
    # 24_eval_sen1floods11.py may read (the final evaluation)
    pool = ([("hand", x) for x in names["s1hand"] if split_of.get(x) in ("train", "valid")]
            + [("weak", x) for x in names["s1weak"] if mc.event_of(x) != "Bolivia"])
    if not pool:
        return
    pick = sorted(rng.choice(len(pool), size=min(SAMPLE, len(pool)), replace=False))
    probs = {k: [] for k in ("bands", "dtype", "crs", "db", "lbands", "lvalues", "lweak",
                             "lgrid", "tbands", "tgrid", "tdesc", "size", "pixel")}
    nod = set()
    for i in pick:
        set_, cid = pool[i]
        _, _, _, s1suf, lsuf = mc.SETS[set_]
        f = paths.s1_dir(set_) / f"{cid}{s1suf}"
        try:
            g = _grid(f)
        except Exception as e:                      # noqa: BLE001
            probs["bands"].append(f"{cid} unreadable ({e})")
            continue
        if g["count"] != 2:
            probs["bands"].append(f"{cid}{s1suf} has {g['count']} band(s), expected 2 (VV, VH)")
            continue
        if g["dtypes"][0] != "float32":
            probs["dtype"].append(f"{cid} is {g['dtypes'][0]}")
        if g["crs"] is None or g["crs"].to_epsg() != 4326:
            probs["crs"].append(f"{cid} is {g['crs']}")
        elif abs(abs(g["transform"].a) - 8.983e-05) > 2e-6:
            probs["pixel"].append(f"{cid} pixel {g['transform'].a:.3g} deg")
        if g["shape"] != (512, 512):
            probs["size"].append(f"{cid} {g['shape']}")
        nod.add("NaN" if g["nodata"] is None or np.isnan(g["nodata"]) else str(g["nodata"]))
        x = mc.read_s1(f)
        fin = x[0][np.isfinite(x[0])]
        if fin.size and fin.min() >= -1:
            probs["db"].append(f"{cid} minimum VV {fin.min():.2f}")
        lf = paths.label_dir(set_) / f"{cid}{lsuf}"
        if lf.exists():
            lg = _grid(lf)
            if lg["count"] != 1:
                probs["lbands"].append(f"{cid} label has {lg['count']} bands")
            if not _same_grid(lg, g):
                off = (f"{mc.grid_offset_px(g['shape'], g['transform'], lg['transform']):.3g} px"
                       if lg["shape"] == g["shape"] else f"size {lg['shape']} vs {g['shape']}")
                probs["lgrid"].append(f"{cid} ({off})")
            with rasterio.open(lf) as s:
                vals = set(np.unique(s.read(1)).tolist())
            extra = vals - {-1, 0, 1, 255} - ({lg["nodata"]} if lg["nodata"] is not None else set())
            if extra:
                (probs["lweak"] if set_ == "weak" else probs["lvalues"]).append(
                    f"{cid}{lsuf} has value(s) {sorted(extra)}")
        tf = paths.terrain_dir(set_) / f"{cid}_Terrain.tif"
        if tf.exists():
            tg = _grid(tf)
            if tg["count"] != 2:
                probs["tbands"].append(f"{cid} terrain has {tg['count']} bands")
            if not _same_grid(tg, g):
                probs["tgrid"].append(cid)            # within 0.01 pixel counts as the same
            d = [x or "" for x in tg["desc"][:2]]
            if any(d) and d != ["hand_m", "slope_deg"]:
                probs["tdesc"].append(f"{cid} {d}")
    n = len(pick)
    c.ok(not probs["bands"], f"S1 chips: 2 bands ({n} sampled)", "VV, VH", _list(probs["bands"]))
    c.ok(not probs["dtype"], "S1 chips: float32", "yes", _list(probs["dtype"]), "WARN")
    c.ok(not probs["crs"], "S1 chips: EPSG:4326", "yes", _list(probs["crs"]), "WARN")
    c.ok(not probs["pixel"], "S1 chips: about 10 m pixels", "yes", _list(probs["pixel"]), "WARN")
    c.ok(not probs["db"], "S1 chips: values in dB", "minimum below -1 dB",
         f"{_list(probs['db'])}: these look like linear power, not dB. The models expect dB "
         "(Earth Engine COPERNICUS/S1_GRD). Check the download.")
    c.add("PASS" if not probs["size"] else "WARN", "S1 chips: 512 x 512",
          "yes" if not probs["size"] else f"other sizes {_list(probs['size'])}; handled "
          "(crop or pad), no action needed")
    c.add("PASS", "S1 chips: no-data value", f"{', '.join(sorted(nod))} (both handled)")
    c.ok(not probs["lbands"], "labels: 1 band", "yes", _list(probs["lbands"]))
    c.ok(not probs["lgrid"], "labels: same grid as the chip", "yes (to 0.01 pixel)",
         f"different grid: {_list(probs['lgrid'])}. Run python diagnose_label_grid.py and read "
         "what it prints.")
    c.ok(not probs["lvalues"], "hand labels: values -1, 0, 1 only", "yes",
         f"{_list(probs['lvalues'])}. Labels must be -1 (no data), 0 or 1. The file is damaged "
         "or not a Sen1Floods11 label: download it again.")
    c.ok(not probs["lweak"], "weak labels: values 0, 1 (and -1)", "yes",
         f"{_list(probs['lweak'])}; every value other than 0 and 1 is ignored in training", "WARN")
    c.ok(not probs["tbands"], "terrain: 2 bands", "hand_m, slope_deg", _list(probs["tbands"]))
    c.ok(not probs["tgrid"], "terrain: same grid as its chip", "yes",
         f"different grid: {_list(probs['tgrid'])}. Remake them with 11_sen1floods11_terrain.py.")
    c.ok(not probs["tdesc"], "terrain: band names", "hand_m, slope_deg", _list(probs["tdesc"]),
         "WARN")


# ============================================================================= corridor

def check_corridor(c, paths, exp):
    c.ok((paths.data / "corridor.geojson").exists(), "corridor.geojson", "found",
         f"{paths.data / 'corridor.geojson'} missing (it is in git: pull the repository)")
    for year in mc.YEARS:
        grd = None
        for radar in mc.RADARS:
            f = paths.scene(year, radar)
            tag = f"{year} {radar.upper()} scene"
            if not c.ok(f.exists(), tag, f.name, f"missing: {f}"):
                continue
            g = _grid(f)
            desc = [d or "" for d in g["desc"][:3]]
            ok = g["count"] >= 3 and (not any(desc) or [d.upper() for d in desc] == ["VV", "VH", "ANGLE"])
            c.ok(ok, f"{tag}: bands", "VV, VH, angle",
                 f"{g['count']} bands described {desc}; expected 3: VV, VH, angle")
            shp = (exp["corridor_height"], exp["corridor_width"])
            gok = g["shape"] == shp and g["crs"] is not None and g["crs"].to_epsg() == 32644
            if "corridor_transform" in exp:
                gok &= tuple(round(v, 6) for v in g["transform"][:6]) == exp["corridor_transform"]
            c.ok(gok, f"{tag}: grid", f"{shp[1]} x {shp[0]} at 10 m, EPSG:32644",
                 f"{g['shape'][1]} x {g['shape'][0]}, {g['crs']}, {tuple(g['transform'][:6])}; "
                 f"expected {shp[1]} x {shp[0]}, EPSG:32644"
                 + (f", {exp['corridor_transform']}" if "corridor_transform" in exp else ""))
            vv = mc.read_s1(f)[0]
            fin = vv[np.isfinite(vv)]
            if not c.ok(fin.size > 0, f"{tag}: radar values", "present", "no finite VV at all"):
                continue
            med = float(np.median(fin))
            c.ok(fin.min() < -1 and -25 < med < 0, f"{tag}: VV in dB",
                 f"median {med:.1f} dB, {100 * fin.size / vv.size:.1f} % of the grid finite",
                 f"median {med:.2f}, minimum {fin.min():.2f}: not dB. The models expect dB.")
            if radar == "grd":
                grd = (g, np.isfinite(vv))
            elif grd is not None:
                if c.ok(_same_grid(g, grd[0]), f"{tag}: same grid as GRD", "yes",
                        "the RTC scene must be on the GRD grid (check_rtc.py)"):
                    lost = int((grd[1] & ~np.isfinite(vv)).sum())
                    c.add("PASS", f"{tag}: no data where GRD has data (radar shadow)",
                          f"{100 * lost / max(grd[1].sum(), 1):.2f} % of the corridor")
        sd = paths.strata(year)
        files = ["strata_hand_m.tif", "strata_slope_deg.tif", "strata_layover_shadow.tif",
                 "strata_theta_loc.tif", "strata_valley_width_m.tif", "strata_meta.json"]
        miss = [n for n in files if not (sd / n).exists()]
        if c.ok(not miss, f"{year} strata files", str(sd.name),
                f"missing in {sd}: {_list(miss)}. Build them (RUN ORDER.md 4.1, 4.2)."):
            g = _grid(sd / "strata_hand_m.tif")
            sok = g["shape"] == (exp["strata_height"], exp["strata_width"])
            if "strata_transform" in exp:
                sok &= tuple(round(v, 6) for v in g["transform"][:6]) == exp["strata_transform"]
            c.ok(sok, f"{year} strata grid", f"{exp['strata_width']} x {exp['strata_height']} at 30 m",
                 f"{g['shape'][1]} x {g['shape'][0]}, {tuple(g['transform'][:6])}")
            for n in ("strata_hand_m.tif", "strata_slope_deg.tif"):
                gn = _grid(sd / n)
                c.ok(gn["dtypes"][0] == "float32" and gn["nodata"] == -9999, f"{year} {n}",
                     "float32, nodata -9999", f"{gn['dtypes'][0]}, nodata {gn['nodata']}")
            with rasterio.open(sd / "strata_layover_shadow.tif") as s:
                vals = set(np.unique(s.read(1)).tolist())
                lsok = s.dtypes[0] == "uint8" and s.nodata == 255
            c.ok(lsok and vals <= {0, 1, 2, 255}, f"{year} layover/shadow codes",
                 f"uint8, values {sorted(vals)} (0 clean, 1 layover, 2 shadow, 255 none)",
                 f"values {sorted(vals)}; expected 0, 1, 2 and 255")
            meta = json.loads((sd / "strata_meta.json").read_text(encoding="utf-8"))
            c.ok("reach_split_longitude" in meta, f"{year} strata_meta.json", "reach split present",
                 "no reach_split_longitude; 04 uses 78.5981", "WARN")
        ref = paths.reference(year)
        if c.ok(ref.exists(), f"{year} reference raster", ref.name, f"missing: {ref} (RUN ORDER 4.8)"):
            g = _grid(ref)
            with rasterio.open(ref) as s:
                a = s.read(1)
            vals = set(np.unique(a).tolist())
            c.ok(g["dtypes"][0] == "uint8" and g["nodata"] == 255 and vals <= {0, 1, 255}
                 and (grd is None or _same_grid(g, grd[0])), f"{year} reference format",
                 f"uint8, 0/1/255, on the GRD grid; flood {int((a == 1).sum()) * 1e-4:.3f} km2",
                 f"{g['dtypes'][0]}, nodata {g['nodata']}, values {sorted(vals)[:6]}, grid "
                 f"{g['shape']}; expected uint8, nodata 255, 0/1/255 on the GRD grid")
        w = paths.windows(year)
        if c.ok(w.exists(), f"{year} digitising windows", w.name, f"missing: {w}"):
            gj = json.loads(w.read_text(encoding="utf-8"))
            nums = [f["properties"].get("window") for f in gj.get("features", [])]
            crs = gj.get("crs", {}).get("properties", {}).get("name", "")
            c.ok(len(nums) == exp["windows"] and all(isinstance(v, int) for v in nums)
                 and "32644" in crs, f"{year} windows content",
                 f"{len(nums)} windows {sorted(nums)}, {crs}",
                 f"{len(nums)} features, window numbers {nums}, CRS '{crs}'; expected "
                 f"{exp['windows']} numbered windows in EPSG:32644")
        _check_gpkg(c, paths.ref_gpkg(year), ("water_event", "water_dry"), f"{year} reference GeoPackage")
        _check_gpkg(c, paths.holes(year), ("holes",), f"{year} cloud holes GeoPackage")
        for kind in ("single", "chfeb"):
            f = paths.otsu(year, kind)
            c.ok(f.exists(), f"{year} Otsu map ({kind}) for the comparisons", f.name,
                 f"missing: {f}. Only the comparisons in RUN ALL MODELS.bat need it.", "WARN")
    f = paths.perm_water()
    if c.ok(f.exists(), "permanent-water mask", f.name, f"missing: {f} (RUN ORDER.md 2.4)"):
        g = _grid(f)
        with rasterio.open(f) as s:
            vals = set(np.unique(s.read(1)).tolist())
        ref_grid = _grid(paths.scene(2021)) if paths.scene(2021).exists() else g
        c.ok(g["dtypes"][0] == "uint8" and vals <= {0, 100} and _same_grid(g, ref_grid),
             "permanent-water mask format", "uint8, values 0 and 100, on the GRD grid",
             f"{g['dtypes'][0]}, values {sorted(vals)[:6]}; expected uint8 0/100 on the GRD grid")


def _check_gpkg(c, path, layers, what):
    if not c.ok(path.exists(), what, path.name, f"missing: {path}"):
        return
    try:
        import pyogrio
        have = {str(n) for n, _ in pyogrio.list_layers(path)}
    except Exception:                               # noqa: BLE001
        import fiona
        have = set(fiona.listlayers(path))
    miss = [x for x in layers if x not in have]
    c.ok(not miss, f"{what} layers", ", ".join(layers),
         f"layers {sorted(have)}; missing {miss}")


# ============================================================================= main

def run_checks(paths, expected, machine=True, seed=mc.SEED):
    c = Checks()
    rng = np.random.default_rng(seed)
    if machine:
        print("Machine")
        check_machine(c)
    print("Sen1Floods11")
    check_s1f(c, paths, expected, rng)
    print("Corridor")
    check_corridor(c, paths, expected)
    return c.rows


def main(argv=None, expected=None):
    mc.setup_console()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mc.add_common_args(ap, smoke=False, seed=False, ram=False)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    paths = mc.paths_from_args(args)
    t0 = time.time()
    print(f"20_models_preflight.py on {socket.gethostname()}, project {paths.root}\n")
    rows = run_checks(paths, expected or EXPECTED, machine=expected is None)
    nf = sum(r["status"] == "FAIL" for r in rows)
    nw = sum(r["status"] == "WARN" for r in rows)
    verdict = ("PREFLIGHT PASS" + (f" ({nw} warnings: read them)" if nw else "")) if not nf else \
        f"PREFLIGHT FAIL: {nf} problem(s). Fix the first FAIL line, then run this again."
    out_dir = paths.o06 / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = out_dir / f"preflight_{socket.gethostname()}.md"
    if rep.exists() and not args.force:
        rep = out_dir / f"preflight_{socket.gethostname()}_{datetime.now():%Y%m%d-%H%M%S}.md"
    lines = [f"# Models preflight, {socket.gethostname()}", "",
             f"{datetime.now():%Y-%m-%d %H:%M}, project `{paths.root}`, "
             f"{time.time() - t0:.0f} s", "", f"**{verdict}**", "",
             "| Result | Check | Detail |", "|---|---|---|"]
    lines += [f"| {r['status']} | {r['name']} | {r['detail'].replace('|', '/')} |" for r in rows]
    rep.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nreport: {rep}")
    print(verdict)
    return 0 if not nf else 1


def selftest():
    import models_synthetic as syn
    print("20_models_preflight.py self-test")
    results = []
    with tempfile.TemporaryDirectory(prefix="fyp selftest ", ignore_cleanup_errors=True) as td:
        for case, breaker, word in (("good", None, None),
                                    ("3 bands", syn.break_band_count, "3 band"),
                                    ("label 5", syn.break_label_value, "[5]"),
                                    ("no terrain", syn.break_missing_terrain, "Mekong_1003")):
            proj = Path(td) / case / "Final Year Project"
            syn.build_project(proj)
            if breaker:
                breaker(proj)
            rows = run_checks(mc.Paths(proj), syn.EXPECTED_SYNTHETIC, machine=False)
            fails = [r for r in rows if r["status"] == "FAIL"]
            if breaker is None:
                results.append(("good data passes", not fails))
            else:
                results.append((f"'{case}' fails with the right message",
                                any(word in r["detail"] for r in fails)))
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    good = all(ok for _, ok in results)
    print("SELFTEST PASS" if good else "SELFTEST FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(mc.cli(main))
