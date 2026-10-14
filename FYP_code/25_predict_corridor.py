r"""
25_predict_corridor.py
======================
Applies a saved model to a corridor post-event scene and writes a flood map
in exactly the format of the Otsu baselines, so 04_stratified_metrics.py
scores it the same way.

WHAT IT DOES
------------
1. Reads VV and VH of the scene (GRD by default, --radar rtc for the RTC
   scene), HAND and slope from data\strata_<year>_v3 (30 m, resampled
   bilinearly onto the 10 m grid), the layover/shadow codes (nearest) and
   the permanent-water mask (value 100).
2. Predicts the water probability for every cell where VV is finite (the
   U-Net in overlapping 512 x 512 tiles, the Random Forest in chunks of one
   million cells).
3. Map: 255 where VV is not finite; 0 elsewhere; 1 where the probability is
   at least the threshold chosen on Sen1Floods11 valid AND the cell is not
   permanent water AND not layover (code 1) or shadow (code 2). Layover and
   shadow cells are 0, as in the baselines. uint8, LZW, the scene's grid.
4. Writes flood_<year>_<model>_<trainset>.tif, a _diagnostics.json with the
   cell counts at each step and the flood area, and run_record.json, into
   data\flood_<year>_<model>_<trainset>\.

A flood area above 30 km2 or below 0.5 km2 prints a WARNING: the reference
holds 2.4 (2021) and 3.2 km2 (2023), so such a map is almost certainly wrong.

USAGE: HOW TO RUN THIS  (copy one line, paste into Anaconda Prompt, press Enter)
================================================================================
    conda activate fyp
    cd /d "<project folder>\04 Code\FYP_code"

    python 25_predict_corridor.py --selftest
    python 25_predict_corridor.py --model unet --trainset hand --year 2021
    python 25_predict_corridor.py --model rf --trainset hand_weak --year 2023
    python 25_predict_corridor.py --model tm_s1t --trainset hand --year 2021

About 1 minute for the U-Net, 1 to 3 minutes for TerraMind and 2 to 10
minutes for a Random Forest. The last line says PASS with the flood area in
km2.
================================================================================
"""

__stage__ = "Models: corridor flood maps (Table A)"
__reads__ = ["data/models/<model>_<trainset>/", "05 Data/sentinel1 or sentinel1_rtc scene",
             "data/strata_<year>_v3/", "05 Data/dem/WATER_PERMANENT_JRC50_S2DRY_10m.tif"]
__writes__ = ["data/flood_<year>_<model>_<trainset>/flood_<year>_<model>_<trainset>.tif",
              "<name>_diagnostics.json", "run_record.json"]

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import models_common as mc


def smoke_window(paths, year, shape, transform):
    """Row/column box of the lowest-numbered digitising window (for --smoke)."""
    from rasterio.windows import from_bounds
    f = paths.windows(year)
    if f.exists():
        gj = json.loads(f.read_text(encoding="utf-8"))
        feats = sorted(gj["features"], key=lambda q: int(q["properties"].get("window", 99)))
        ring = np.array(feats[0]["geometry"]["coordinates"][0], float)
        w = from_bounds(ring[:, 0].min(), ring[:, 1].min(), ring[:, 0].max(), ring[:, 1].max(),
                        transform)
        r0, c0 = max(int(w.row_off), 0), max(int(w.col_off), 0)
        return r0, c0, int(np.ceil(w.height)), int(np.ceil(w.width))
    h, w = shape
    return max(h // 2 - 256, 0), max(w // 2 - 256, 0), 512, 512


def main(argv=None):
    mc.setup_console()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mc.add_common_args(ap, seed=False)
    ap.add_argument("--model", choices=mc.MODELS, default="rf")
    ap.add_argument("--trainset", choices=mc.TRAINSETS, default="hand")
    ap.add_argument("--year", type=int, choices=mc.YEARS, default=2021)
    ap.add_argument("--radar", choices=mc.RADARS, default="grd",
                    help="grd (Table A) or rtc")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the threshold chosen on valid (recorded)")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--batch-tiles", type=int, default=4, help="U-Net or TerraMind tiles per GPU batch")
    ap.add_argument("--cpu-ok", action="store_true", help="allow the U-Net or TerraMind on the CPU")
    mc.add_run_tag_arg(ap)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    mc.check_tile_args(args.tile, args.overlap)
    tag = mc.check_run_tag(args, seed_used=False)
    paths = mc.paths_from_args(args)
    name = paths.flood_name(args.year, args.model, args.trainset, args.radar, tag)
    out_map = paths.flood_map(args.year, args.model, args.trainset, args.radar, tag)
    out_dir = out_map.parent
    print(f"25_predict_corridor.py: {name}{' (smoke)' if args.smoke else ''}")
    if mc.guard(out_dir / "run_record.json", args, f"The flood map {name}"):
        return 0
    t0 = time.time()
    mdir = paths.model_dir(args.model, args.trainset, tag)
    pred = mc.load_predictor(args.model, mdir, cpu_ok=args.cpu_ok or args.smoke, max_ram_gb=args.max_ram_gb,
                             tile=args.tile, overlap=args.overlap, batch=args.batch_tiles)
    window = None
    if args.smoke:
        import rasterio
        with rasterio.open(paths.scene(args.year, args.radar)) as src:
            window = smoke_window(paths, args.year, src.shape, src.transform)
        print(f"  smoke: predicting only rows {window[0]}..{window[0] + window[2]}, "
              f"columns {window[1]}..{window[1] + window[3]}")
    inp = mc.corridor_inputs(paths, args.year, args.radar, window)
    need_gb = inp["x"].nbytes * 3 / 1e9
    if need_gb > args.max_ram_gb:
        mc.fail(f"This scene needs about {need_gb:.1f} GB of RAM, more than --max-ram-gb "
                f"{args.max_ram_gb:g}. Close other programs and raise --max-ram-gb.")
    prof, valid = inp["profile"], inp["valid"]
    km2 = mc.cell_km2(prof)
    print(f"  scene {paths.scene(args.year, args.radar).name}: {prof['height']} x "
          f"{prof['width']} cells, {int(valid.sum()):,} valid ({valid.sum() * km2:.1f} km2)")
    if inp["terrain_nan_cells"] and getattr(pred, "uses_terrain", True):
        print(f"  valid cells without HAND or slope (filled with the mean): "
              f"{inp['terrain_nan_cells']:,}")
    t_pred = time.time()
    prob = pred.predict_image(inp["x"], valid)
    t_pred = time.time() - t_pred
    thr = pred.threshold if args.threshold is None else float(args.threshold)
    how = "chosen on Sen1Floods11 valid" if args.threshold is None else "set with --threshold"
    m, diag = mc.make_flood_map(prob, valid, thr, inp["perm"], inp["distort"])
    mc.write_flood_map(out_map, m, prof)
    flood_km2 = diag["flood_cells"] * km2
    warn = mc.area_warning(flood_km2, args.smoke)
    diag_p = out_dir / f"{name}_diagnostics.json"
    diag.update({"event": args.year, "model": args.model, "trainset": args.trainset,
                 "run_tag": tag,
                 "radar": args.radar, "scene": str(paths.scene(args.year, args.radar)),
                 "model_dir": str(mdir), "threshold_how": how,
                 "model_run_record": str(mdir / "run_record.json"),
                 "flood_km2": round(flood_km2, 4), "corridor_km2": round(valid.sum() * km2, 3),
                 "water_raw_km2": round(diag["water_raw_cells"] * km2, 4),
                 "terrain_nan_cells": inp["terrain_nan_cells"], "smoke_window": window,
                 "warnings": [warn] if warn else [],
                 "run_record": str(out_dir / "run_record.json")})
    mc.save_json(diag_p, diag)
    print(f"  water before masks : {diag['water_raw_cells']:,} cells "
          f"({diag['water_raw_cells'] * km2:.3f} km2), threshold {thr:.2f} ({how})")
    print(f"  permanent water off: {diag['permanent_removed_cells']:,} cells")
    print(f"  layover/shadow off : {diag['distortion_removed_cells']:,} cells")
    print(f"  flood              : {diag['flood_cells']:,} cells = {flood_km2:.3f} km2")
    if warn:
        print("  " + warn)
    rec = mc.run_record(args, paths, step="corridor prediction", model=args.model,
                        trainset=args.trainset, run_tag=tag, year=args.year,
                        radar=args.radar, model_dir=str(mdir),
                        model_run_record=str(mdir / "run_record.json"),
                        norm_file=str(mdir / "norm_stats.json"),
                        norm_sha256=mc.sha256(mdir / "norm_stats.json"),
                        threshold=thr, threshold_how=how, predict_seconds=round(t_pred, 1),
                        seconds=round(time.time() - t0, 1),
                        outputs=[mc.file_entry(out_map), mc.file_entry(diag_p)])
    rec.update(getattr(pred, "record", {}))        # TerraMind: its weights and normalisation
    mc.save_json(out_dir / "run_record.json", rec)
    print(f"  wrote {out_map}")
    print(f"PASS  {name}: {flood_km2:.3f} km2 of flood" + ("  (read the WARNING)" if warn else ""))
    return 0


def selftest():
    import rasterio
    import models_synthetic as syn
    print("25_predict_corridor.py self-test")
    with tempfile.TemporaryDirectory(prefix="fyp selftest ", ignore_cleanup_errors=True) as td:
        proj = Path(td) / "Final Year Project"
        syn.build_project(proj)
        ok = mc.run_script("21_prepare_chips.py", ["--project", str(proj)]) == 0
        ok &= mc.run_script("22_train_rf.py", ["--project", str(proj), "--n-estimators", "10",
                                               "--n-per-class", "200"]) == 0
        ok &= main(["--project", str(proj), "--model", "rf", "--year", "2021"]) == 0
        paths = mc.Paths(proj)
        out = paths.flood_map(2021, "rf", "hand")
        with rasterio.open(out) as d, rasterio.open(paths.scene(2021)) as s:
            m, vv = d.read(1), s.read(1)
            grid = d.transform == s.transform and d.crs == s.crs and d.shape == s.shape
            fmt = d.dtypes[0] == "uint8" and d.nodata == 255
        perm = mc.reproject_onto(paths.perm_water(), {"height": m.shape[0], "width": m.shape[1],
                                                      "transform": d.transform, "crs": d.crs},
                                 categorical=True) == 100
        diag = mc.load_json(out.parent / "flood_2021_rf_hand_diagnostics.json")
    checks = [("ran", ok), ("uint8, nodata 255, exact grid", fmt and grid),
              ("255 exactly where VV is not finite", np.array_equal(m == 255, ~np.isfinite(vv))),
              ("permanent water is 0", (m[perm & np.isfinite(vv)] == 0).all()),
              ("diagnostics add up", diag["flood_cells"] == diag["water_raw_cells"]
               - diag["permanent_removed_cells"] - diag["distortion_removed_cells"])]
    for name, c in checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {name}")
    good = all(c for _, c in checks)
    print("SELFTEST PASS" if good else "SELFTEST FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(mc.cli(main))
