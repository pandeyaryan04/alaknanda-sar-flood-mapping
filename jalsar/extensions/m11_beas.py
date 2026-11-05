"""Jal-SAR extensions, item M11: blind test on a flood the models have never seen, the Beas, July 2023.

The three Jal-SAR model families (the same trained files as Table A; nothing is retrained or tuned)
are run on a Sentinel-1 pass over the Beas valley from Manali to Mandi (Himachal Pradesh) after the
floods of 9 to 11 July 2023. The maps are then compared with the Copernicus GFM flood extent of the
same pass (m11_gfm.py), and checked by eye against the NDEM map of 15 July 2023.

What it does:
  search   lists every Sentinel-1 pass over the Beas valley from 8 to 25 July 2023, with coverage
  run      for one pass (the first on or after 10 July covering at least --min-coverage per cent,
           or the one given by --date, --orbit and --pass):
             1. the valley outline: HydroSHEDS free-flowing rivers with at least 300 km2 upstream,
                kept within 4 km of a line Manali, Kullu, Bhuntar, Aut, Larji, Pandoh, Mandi, and
                widened by --buffer-m (1,500 m) on each side
             2. VV, VH and incidence angle on a 10 m UTM 43N grid (as the corridor)
             3. Copernicus GLO-30 and JRC occurrence on a 30 m grid with a 2.5 km margin; slope,
                HAND, layover and shadow by 03_terrain_strata.py itself, run exactly as Jal-SAR runs it
                for a new orbit (jalsar_runner.strata_for); onto the 10 m grid as models_common does
             4. permanent water = JRC occurrence of at least 50 per cent (the corridor also used
                dry-season Sentinel-2 water; not here, stated in the report)
             5. each family's model (rf_hand, unet_hand and the TerraMind variant Jal-SAR uses)
                at its own threshold, permanent water and radar-blind cells taken out, exactly
                as the corridor maps (models_common.make_flood_map)
           and writes the maps, the inputs and BEAS_RUN.md

Outputs in 06 Outputs\\jalsar\\extensions\\m11_beas\\:
  beas_valley.geojson, passes.json, S1_<date>_orb<nnn>.tif (VV, VH, angle), strata\\ (DEM, JRC and
  the 03_terrain_strata.py files), flood_<family>.tif, BEAS_RUN.md, run_record.json

Run:  python m11_beas.py search
      python m11_beas.py run
      python m11_beas.py run --date 2023-07-11 --orbit 27 --pass DESCENDING
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "FYP_code"))
import jalsar_core as gc  # noqa: E402
import jalsar_runner as jr  # noqa: E402

mc = gc.mc
OUT = HERE.parents[2] / "06 Outputs" / "jalsar" / "extensions" / "m11_beas"
# approximate points on the Beas main stem, lon, lat (only used to keep the right river reaches)
WAYPOINTS = [(77.188, 32.243), (77.142, 32.105), (77.110, 31.958), (77.145, 31.875), (77.202, 31.752),
             (77.215, 31.725), (77.060, 31.667), (76.930, 31.708)]
SEARCH = ("2023-07-08", "2023-07-26")
FIRST_DAY = "2023-07-10"
CRS = "EPSG:32643"
GP = None          # the project paths (set in main; a test can set its own)
RES30, PAD = 30.0, 2500.0


def valley(ee, buffer_m):
    """HydroSHEDS free-flowing rivers (300 km2 or more upstream) near the waypoint line, widened."""
    keep = ee.Geometry.LineString(WAYPOINTS).buffer(4000, 50)
    rivers = (ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers").filterBounds(keep)
              .filter(ee.Filter.gte("UPLAND_SKM", 300)))
    return rivers.geometry(30).intersection(keep, 30).buffer(buffer_m, 30).simplify(20)


def grid_for(geom_info, cell):
    """A grid in CRS, snapped to whole cells, round the outline."""
    from rasterio.warp import transform_geom
    g = transform_geom("EPSG:4326", CRS, geom_info)
    xs, ys = [], []

    def walk(c):
        if isinstance(c[0], (int, float)):
            xs.append(c[0])
            ys.append(c[1])
        else:
            for k in c:
                walk(k)
    walk(g["coordinates"])
    x0, x1 = np.floor(min(xs) / cell) * cell, np.ceil(max(xs) / cell) * cell
    y0, y1 = np.floor(min(ys) / cell) * cell, np.ceil(max(ys) / cell) * cell
    return {"crs": CRS, "x0": float(x0), "y0": float(y1), "cell": float(cell),
            "width": int(round((x1 - x0) / cell)), "height": int(round((y1 - y0) / cell))}


def fetch_dem_occ(ee, grid, tile=1024):
    """Copernicus GLO-30 DEM and JRC occurrence on the 30 m grid, in tiles."""
    dem = (ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM").mosaic().unmask(-32768).toFloat().rename("dem"))
    occ = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0).toFloat().rename("occ")
    img = dem.addBands(occ)
    out = np.zeros((2, grid["height"], grid["width"]), "float32")
    for c0, r0, w, h in jr.tiles(grid["width"], grid["height"], tile):
        req = {"expression": img, "fileFormat": "NUMPY_NDARRAY", "bandIds": ["dem", "occ"],
               "grid": {"dimensions": {"width": w, "height": h},
                        "affineTransform": {"scaleX": grid["cell"], "shearX": 0, "translateX": grid["x0"] + c0 * grid["cell"],
                                            "shearY": 0, "scaleY": -grid["cell"], "translateY": grid["y0"] - r0 * grid["cell"]},
                        "crsCode": grid["crs"]}}
        for attempt in range(4):
            try:
                a = ee.data.computePixels(req)
                break
            except Exception as exc:              # noqa: BLE001
                if attempt == 3:
                    raise
                print(f"    DEM tile ({r0}, {c0}): {str(exc)[:100]}; again in {5 * (attempt + 1)} s")
                time.sleep(5 * (attempt + 1))
        out[0, r0:r0 + h, c0:c0 + w] = np.asarray(a["dem"], "float32")
        out[1, r0:r0 + h, c0:c0 + w] = np.asarray(a["occ"], "float32")
    return out[0], out[1]


def write_tif(path, arr, grid, nodata):
    from rasterio.transform import from_origin
    prof = {"driver": "GTiff", "dtype": "float32", "count": 1, "crs": grid["crs"], "width": grid["width"],
            "height": grid["height"], "transform": from_origin(grid["x0"], grid["y0"], grid["cell"], grid["cell"]),
            "nodata": nodata, "compress": "lzw"}
    with gc.rasterio.open(path, "w", **prof) as d:
        d.write(arr.astype("float32"), 1)


def build_strata(scene, orbit_pass, g30, dem, occ, gpath, out):
    """03_terrain_strata.py on the Beas DEM, run exactly as jalsar_runner.strata_for runs it."""
    import subprocess
    out.mkdir(parents=True, exist_ok=True)
    dem_p, occ_p = out / "DEM_COP_GLO30_30m.tif", out / "WATER_JRC_occurrence_30m.tif"
    write_tif(dem_p, np.where(dem <= -32767, -32768, dem), g30, -32768.0)
    write_tif(occ_p, occ, g30, -1.0)
    cmd = [sys.executable, str(gc.FYP / "03_terrain_strata.py"), "--dem", str(dem_p), "--pass", orbit_pass,
           "--water", str(occ_p), "--corridor", str(gpath), "--angle", str(scene), "--out", str(out)]
    r = subprocess.run(cmd, cwd=str(gc.FYP))
    if r.returncode != 0 or not (out / "strata_layover_shadow.tif").exists():
        mc.fail("03_terrain_strata.py did not finish for the Beas. Scroll up for its message.")


def rasterise(geom_info, prof):
    from rasterio.features import rasterize
    from rasterio.warp import transform_geom
    return rasterize([(transform_geom("EPSG:4326", CRS, geom_info), 1)], out_shape=(prof["height"], prof["width"]),
                     transform=prof["transform"], fill=0, dtype="uint8").astype(bool)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("search", "run"))
    ap.add_argument("--date")
    ap.add_argument("--orbit", type=int)
    ap.add_argument("--pass", dest="orbit_pass", choices=("ASCENDING", "DESCENDING"))
    ap.add_argument("--min-coverage", type=float, default=95.0)
    ap.add_argument("--buffer-m", type=float, default=1500.0)
    ap.add_argument("--cpu-ok", action="store_true")
    args = ap.parse_args()
    global GP
    GP = GP or gc.GPaths()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ee = jr.ee_init()
    gpath = OUT / "beas_valley.geojson"
    if gpath.exists():
        ginfo = json.loads(gpath.read_text(encoding="utf-8"))["features"][0]["geometry"]
    else:
        print("  building the Beas valley outline in Earth Engine ...")
        ginfo = valley(ee, args.buffer_m).getInfo()
        gpath.write_text(json.dumps({"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": ginfo, "properties": {
                "how": "HydroSHEDS free-flowing rivers, upstream area >= 300 km2, within 4 km of the Manali to "
                       f"Mandi waypoints, widened by {args.buffer_m:.0f} m", "waypoints": WAYPOINTS}}]}), encoding="utf-8")
    geom = ee.Geometry(ginfo)
    area_km2 = geom.area(10).getInfo() / 1e6
    print(f"  Beas valley outline: {area_km2:.1f} km2 ({gpath.name})")
    rows = jr.search(ee, geom, *SEARCH)
    print(f"  Sentinel-1 passes over the outline, {SEARCH[0]} to {SEARCH[1]}:")
    jr.print_passes(rows)
    (OUT / "passes.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    if args.cmd == "search":
        print("PASS  M11 Beas search (see passes.json)")
        return 0
    if args.date:
        cand = [r for r in rows if r["date"] == args.date and (args.orbit is None or r["orbit"] == args.orbit)
                and (args.orbit_pass is None or r["pass"] == args.orbit_pass)]
    else:
        cand = [r for r in rows if r["date"] >= FIRST_DAY and r["coverage_pct"] >= args.min_coverage]
    if not cand:
        print(f"FAIL  No pass matches (coverage at least {args.min_coverage} %). Look at the list above and give "
              "--date, --orbit and --pass, or lower --min-coverage.")
        return 1
    row = sorted(cand, key=lambda r: r["date"])[0]
    print(f"  using {row['date']} orbit {row['orbit']} {row['pass'].lower()} ({row['coverage_pct']:.1f} %)")

    # 2. radar
    grid = grid_for(ginfo, 10.0)
    name = jr.scene_name(row["date"], row["orbit"])
    scene = OUT / f"{name}.tif"
    if not scene.exists():
        print(f"  downloading {name}: {grid['width']} x {grid['height']} cells, "
              f"{len(jr.tiles(grid['width'], grid['height'], 1024))} tiles", flush=True)
        rec = {"scene_ids": row["ids"], "time_utc": row["time_utc"], "orbit": row["orbit"], "pass": row["pass"],
               "coverage_pct": row["coverage_pct"], "grid": grid, "collection": "COPERNICUS/S1_GRD",
               "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        jr.download(jr.ee_fetcher(ee, jr.pass_image(ee, geom, row), grid), OUT / "downloads" / name, scene,
                    grid=grid, record=rec)
    vv, vh, prof = mc.read_scene(scene, "The Beas scene")
    inside = rasterise(ginfo, prof)

    # 3-4. terrain (03_terrain_strata.py, as for every corridor pass) and permanent water (JRC 50 %)
    sd = OUT / "strata"
    if not (sd / "strata_layover_shadow.tif").exists():
        g30 = {"crs": CRS, "x0": grid["x0"] - PAD, "y0": grid["y0"] + PAD, "cell": RES30,
               "width": int(np.ceil((grid["width"] * 10 + 2 * PAD) / RES30)),
               "height": int(np.ceil((grid["height"] * 10 + 2 * PAD) / RES30))}
        print(f"  DEM and JRC occurrence on a 30 m grid ({g30['width']} x {g30['height']}) ...", flush=True)
        dem, occ = fetch_dem_occ(ee, g30)
        print("  terrain strata (03_terrain_strata.py, a few minutes) ...", flush=True)
        build_strata(scene, row["pass"], g30, dem, occ, gpath, sd)
    hand = mc.reproject_onto(sd / "strata_hand_m.tif", prof)
    slope = mc.reproject_onto(sd / "strata_slope_deg.tif", prof)
    ls = np.isin(mc.reproject_onto(sd / "strata_layover_shadow.tif", prof, categorical=True), (1, 2))
    perm = np.nan_to_num(mc.reproject_onto(sd / "WATER_JRC_occurrence_30m.tif", prof, categorical=True)) >= 50

    # 5. models
    x = np.stack([vv, vh, hand, slope]).astype("float32")
    valid = np.isfinite(vv) & inside & np.isfinite(hand) & np.isfinite(slope)
    km2 = mc.cell_km2(prof)
    res = {}
    for fam in ("rf", "unet", "terramind"):
        mem = gc.family_member(GP, fam)
        if mem is None:
            print(f"  {fam}: no trained model found, skipped")
            continue
        model, trainset, _ = mem
        print(f"  {gc.FAMILY_NAMES[fam]} ({model}_{trainset}) ...", flush=True)
        try:
            pred = mc.load_predictor(model, GP.model_dir(model, trainset), cpu_ok=args.cpu_ok)
            prob = pred.predict_image(x, valid)
        except SystemExit as exc:           # mc.fail: say why, go on with the other families
            print(f"  PROBLEM: {gc.FAMILY_NAMES[fam]} did not run: {exc}")
            res[fam] = {"model": f"{model}_{trainset}", "failed": str(exc)[:300]}
            continue
        fm, diag = mc.make_flood_map(prob, valid, pred.threshold, perm, ls)
        mc.write_flood_map(OUT / f"flood_{fam}.tif", fm, prof)
        res[fam] = {"model": f"{model}_{trainset}", "threshold": pred.threshold,
                    "flood_km2": round(diag["flood_cells"] * km2, 3), **diag}
        print(f"    flood {res[fam]['flood_km2']} km2")
        del pred
    ns = mc.load_json(GP.model_dir("rf", gc.TRAINSET) / "norm_stats.json")
    lo, hi = np.asarray(ns["clip_lo"], "float32"), np.asarray(ns["clip_hi"], "float32")
    xv = x[:, valid]
    out_any = ((xv < lo[:, None]) | (xv > hi[:, None])).any(axis=0)
    beyond = float(out_any.mean()) if valid.any() else float("nan")
    beyond_slope = float((xv[3] > hi[3]).mean()) if valid.any() else float("nan")
    del xv
    rec = {"script": "m11_beas.py", "pass": row, "grid": grid, "valley_km2": round(area_km2, 2),
           "valid_km2": round(float(valid.sum()) * km2, 2), "radar_blind_km2": round(float((valid & ls).sum()) * km2, 2),
           "permanent_water_km2": round(float((valid & perm).sum()) * km2, 2),
           "share_beyond_any_clip_limit": round(beyond, 4), "share_slope_above_clip": round(beyond_slope, 4), "families": res,
           "minutes": round((time.time() - t0) / 60, 1)}
    mc.save_json(OUT / "run_record.json", rec)
    md = ["# M11: the Beas, July 2023 (blind test, models not retrained)", "",
          f"Pass: {row['date']} orbit {row['orbit']} {row['pass'].lower()}, {row['time_utc']} UTC, "
          f"{row['coverage_pct']:.1f} % of the valley outline. Valley outline {area_km2:.1f} km2 "
          f"(HydroSHEDS rivers of 300 km2 or more upstream between Manali and Mandi, widened by "
          f"{args.buffer_m:.0f} m each side). Cells with radar data: {rec['valid_km2']} km2; radar blind "
          f"{rec['radar_blind_km2']} km2; permanent water (JRC 50 %) {rec['permanent_water_km2']} km2.", "",
          f"Cells outside the training clip limits of any input (norm_stats.json of rf_hand; slope above "
          f"{hi[3]:.2f} degrees on its own: {beyond_slope * 100:.1f} %): {beyond * 100:.1f} %. "
          "The corridor figure is 56.7 %.", "",
          "| Family | Model | Threshold | Flood km2 | Removed as permanent water (cells) | Removed as radar blind (cells) |",
          "|---|---|---|---|---|---|"]
    for fam, r in res.items():
        if "failed" in r:
            md.append(f"| {gc.FAMILY_NAMES[fam]} | {r['model']} | did not run: {r['failed'][:80]} | | | |")
            continue
        md.append(f"| {gc.FAMILY_NAMES[fam]} | {r['model']} | {r['threshold']:.2f} | {r['flood_km2']} | "
                  f"{r['permanent_removed_cells']:,} | {r['distortion_removed_cells']:,} |")
    md += ["", "Next: m11_gfm.py compares these maps with the Copernicus GFM flood extent of the same pass. "
           "There is no hand-drawn reference here, so no accuracy figure: agreement with GFM is agreement with "
           "another model. Check the maps by eye against the NDEM map of 15 July 2023 "
           "(ndem.nrsc.gov.in, hpflood50dsc15072023_vap8.pdf).", "",
           "Limits: permanent water from JRC only (the corridor also used dry-season Sentinel-2); HAND from a "
           "DEM box 2.5 km wider than the valley (03_terrain_strata.py takes the drainage exit at the box edge); the "
           "outline is built from approximate waypoints; the pass may be days after the peak.", ""]
    (OUT / "BEAS_RUN.md").write_text("\n".join(md), encoding="utf-8")
    print(f"  wrote {OUT / 'BEAS_RUN.md'} ({rec['minutes']} min)")
    ok = any("failed" not in r for r in res.values())
    print(("PASS" if ok else "FAIL") + "  M11 Beas run" + ("" if ok else ": no model family ran"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
