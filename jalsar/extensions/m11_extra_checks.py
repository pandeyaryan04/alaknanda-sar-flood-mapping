"""Jal-SAR extensions, M11 follow-up: two extra checks on the Beas U-Net flood patches.

There is no flood-time reference map for the Beas valley. These two checks add evidence to the
by-eye check of the patches marked by m11_ndem_site.py scan. Neither is a score.

  pre       Pre-flood radar. The median of every Sentinel-1 pass of the same orbit (136,
            descending) from 1 April to 15 June 2023, before the monsoon, on the same 10 m grid as
            the 12 July 2023 scene. A patch that was already as dark before the flood is more
            likely a dark surface (shadow, a smooth surface) than flood water.
  optical   Optical images. Every Sentinel-2 (surface reflectance) and Landsat 8 and 9 (level 2)
            day from 10 to 20 July 2023 over the valley, with the share of the valley free of cloud.
            The clearest day, if at least MIN_CLEAR_PCT per cent of the valley is clear, is
            downloaded on the same grid: a true-colour picture for QGIS and a water mask
            (MNDWI above 0 on clear cells).
  merge     For every U-Net flood patch (the same patches and numbers as m11_ndem_site.py scan):
            median VV before and on 12 July, the change, the optical clear and water shares, and a
            plain-words hint for each check. Writes extra_checks_unet.csv (all patches),
            extra_checks_unet.geojson (the 104 marked patches, same ranks as
            valley_scan_check_unet.geojson) and EXTRA_CHECKS.md.

Hints (rules set for this check, not taken from the literature):
  radar     "darker on 12 July"  : change of -3 dB or more (darker by at least 3 dB)
            "dark before too"    : median VV before the flood below -15 dB and change above -3 dB
            "no clear change"    : anything else
  optical   "water seen"         : at least half the patch clear and at least half the clear cells water
            "no water seen"      : at least half the patch clear and under a tenth of the clear cells water
            "cloud or unclear"   : anything else, or no usable image

Outputs: 06 Outputs\\jalsar\\extensions\\m11_beas\\S1_pre_20230401_20230615_orb136.tif (and its record),
         m11_beas\\optical\\ (search list, picture and water mask, if a usable day exists),
         m11_beas\\ndem_check\\extra_checks_unet.csv, .geojson and EXTRA_CHECKS.md

Run:  python m11_extra_checks.py pre
      python m11_extra_checks.py optical
      python m11_extra_checks.py merge
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "FYP_code"))

M11 = HERE.parents[2] / "06 Outputs" / "jalsar" / "extensions" / "m11_beas"
CHECK = M11 / "ndem_check"
OPT = M11 / "optical"
EVENT = "S1_20230712_orb136"
PRE_WINDOW = ("2023-04-01", "2023-06-16")       # end is exclusive: up to 15 June
PRE = f"S1_pre_{PRE_WINDOW[0].replace('-', '')}_20230615_orb136"
ORBIT, ORBIT_PASS = 136, "DESCENDING"
OPT_WINDOW = ("2023-07-10", "2023-07-21")       # end is exclusive: up to 20 July
MIN_CLEAR_PCT = 10.0
DARK_DB, CHANGE_DB = -15.0, -3.0
CLEAR_MIN, WATER_YES, WATER_NO = 0.5, 0.5, 0.1
OPT_BANDS = ("red", "green", "blue", "water", "clear")


def grid_of_event():
    rec = json.loads((M11 / f"{EVENT}_record.json").read_text(encoding="utf-8"))
    return rec["grid"]


def valley_geom(ee):
    g = json.loads((M11 / "beas_valley.geojson").read_text(encoding="utf-8"))
    return ee.Geometry(g["features"][0]["geometry"])


def pre():
    import jalsar_runner as jr
    out = M11 / f"{PRE}.tif"
    if out.exists():
        print(f"  {out.name} is already there; kept")
        return 0
    ee = jr.ee_init()
    geom = valley_geom(ee)
    col = (jr.s1_collection(ee, geom, *PRE_WINDOW)
           .filter(ee.Filter.eq("relativeOrbitNumber_start", ORBIT))
           .filter(ee.Filter.eq("orbitProperties_pass", ORBIT_PASS)))
    ids = col.aggregate_array("system:index").getInfo()
    if not ids:
        print(f"FAIL  no Sentinel-1 pass of orbit {ORBIT} {ORBIT_PASS.lower()} from {PRE_WINDOW[0]} to 15 June 2023")
        return 1
    days = sorted({i.split("_")[4][:8] for i in ids})
    print(f"  {len(ids)} scenes on {len(days)} days: {', '.join(days)}")
    img = col.select(list(jr.BANDS)).median().clip(geom).toFloat().unmask(jr.FILL, False)
    grid = grid_of_event()
    rec = {"what": "median of every pass in the window", "scene_ids": ids, "days": days,
           "window": [PRE_WINDOW[0], "2023-06-15"], "orbit": ORBIT, "pass": ORBIT_PASS, "grid": grid,
           "collection": "COPERNICUS/S1_GRD",
           "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    print(f"  downloading the pre-flood median: {grid['width']} x {grid['height']} cells, "
          f"{len(jr.tiles(grid['width'], grid['height'], 1024))} tiles", flush=True)
    valid = jr.download(jr.ee_fetcher(ee, img, grid), M11 / "downloads" / PRE, out, grid=grid, record=rec)
    print(f"PASS  pre-flood radar: {out.name} ({valid * 100:.1f} % of the rectangle has data)")
    return 0


def _s2(ee, geom, d0, d1):
    col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(geom).filterDate(d0, d1)

    def prep(im):
        scl = im.select("SCL")
        clear = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)).Or(scl.eq(11))
        sr = im.select(["B4", "B3", "B2", "B11"]).divide(10000)
        return sr.rename(["red", "green", "blue", "swir1"]).addBands(clear.rename("clear"))
    return col, prep


def _landsat(ee, geom, d0, d1):
    col = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
           .filterBounds(geom).filterDate(d0, d1))

    def prep(im):
        qa = im.select("QA_PIXEL")
        cloudy = (qa.bitwiseAnd(1 << 1).neq(0).Or(qa.bitwiseAnd(1 << 3).neq(0))
                  .Or(qa.bitwiseAnd(1 << 4).neq(0)))
        sr = im.select(["SR_B4", "SR_B3", "SR_B2", "SR_B6"]).multiply(0.0000275).add(-0.2)
        return sr.rename(["red", "green", "blue", "swir1"]).addBands(cloudy.Not().rename("clear"))
    return col, prep


def _day_image(ee, col, prep, day, geom):
    d0 = ee.Date(day)
    im = col.filterDate(d0, d0.advance(1, "day")).map(prep).mosaic()
    clear = im.select("clear").unmask(0)
    mndwi = im.select("green").subtract(im.select("swir1")).divide(im.select("green").add(im.select("swir1")))
    water = mndwi.gt(0).And(clear).unmask(0)
    return im.select(["red", "green", "blue"]).unmask(0).addBands(water.rename("water")).addBands(
        clear.rename("clear")).clip(geom).toFloat()


def optical():
    import jalsar_runner as jr
    ee = jr.ee_init()
    geom = valley_geom(ee)
    OPT.mkdir(parents=True, exist_ok=True)
    rows = []
    for sensor, maker in (("Sentinel-2", _s2), ("Landsat 8/9", _landsat)):
        col, prep = maker(ee, geom, *OPT_WINDOW)
        times = col.aggregate_array("system:time_start").getInfo()
        days = sorted({datetime.fromtimestamp(t / 1000, tz=timezone.utc).strftime("%Y-%m-%d") for t in times})
        for day in days:
            im = _day_image(ee, col, prep, day, geom)
            st = im.select(["clear"]).reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=60,
                                                   maxPixels=1e9).getInfo()
            pct = round(float(st.get("clear") or 0) * 100, 1)
            rows.append({"sensor": sensor, "date": day, "clear_pct_of_valley": pct})
            print(f"  {sensor:12s} {day}: {pct:5.1f} % of the valley clear")
    (OPT / "optical_search.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    if not rows:
        print("PASS  optical search: no Sentinel-2 or Landsat image of the valley from 10 to 20 July 2023")
        return 0
    best = sorted(rows, key=lambda r: (-r["clear_pct_of_valley"], r["date"]))[0]
    if best["clear_pct_of_valley"] < MIN_CLEAR_PCT:
        print(f"PASS  optical search: the clearest day ({best['sensor']}, {best['date']}) has only "
              f"{best['clear_pct_of_valley']} % of the valley clear (under {MIN_CLEAR_PCT:.0f} %); nothing downloaded")
        return 0
    maker = _s2 if best["sensor"] == "Sentinel-2" else _landsat
    col, prep = maker(ee, geom, *OPT_WINDOW)
    im = _day_image(ee, col, prep, best["date"], geom)
    grid = grid_of_event()
    arr = np.zeros((len(OPT_BANDS), grid["height"], grid["width"]), "float32")
    tl = jr.tiles(grid["width"], grid["height"], 1024)
    print(f"  downloading {best['sensor']} {best['date']}: {len(tl)} tiles", flush=True)
    for i, (c0, r0, w, h) in enumerate(tl, 1):
        req = {"expression": im, "fileFormat": "NUMPY_NDARRAY", "bandIds": list(OPT_BANDS),
               "grid": {"dimensions": {"width": int(w), "height": int(h)},
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
                print(f"    tile ({r0}, {c0}): {str(exc)[:100]}; again in {5 * (attempt + 1)} s")
                time.sleep(5 * (attempt + 1))
        for b, name in enumerate(OPT_BANDS):
            arr[b, r0:r0 + h, c0:c0 + w] = np.asarray(a[name], "float32")
        print(f"    tile {i} of {len(tl)}", flush=True)
    import rasterio
    from rasterio.transform import from_origin
    tr = from_origin(grid["x0"], grid["y0"], grid["cell"], grid["cell"])
    tag = f"{best['sensor'].split()[0].replace('/', '').lower()}_{best['date'].replace('-', '')}"
    rgb = np.clip(arr[:3] / 0.3 * 255, 0, 255).astype("uint8")    # reflectance 0 to 0.3 stretched to 0 to 255
    with rasterio.open(OPT / f"optical_rgb_{tag}.tif", "w", driver="GTiff", dtype="uint8", count=3,
                       width=grid["width"], height=grid["height"], crs=grid["crs"], transform=tr,
                       compress="lzw", photometric="RGB") as d:
        d.write(rgb)
    wc = np.where(arr[4] > 0.5, np.where(arr[3] > 0.5, 1, 0), 255).astype("uint8")
    with rasterio.open(OPT / f"optical_water_{tag}.tif", "w", driver="GTiff", dtype="uint8", count=1,
                       width=grid["width"], height=grid["height"], crs=grid["crs"], transform=tr,
                       compress="lzw", nodata=255) as d:
        d.write(wc, 1)
    (OPT / "optical_used.json").write_text(json.dumps({**best, "files": [f"optical_rgb_{tag}.tif",
                                                                           f"optical_water_{tag}.tif"],
                                                         "water_rule": "MNDWI = (green - SWIR1) / (green + SWIR1) above 0, clear cells only",
                                                         "water_file_values": "1 water, 0 clear and dry, 255 cloud or no image"},
                                                        indent=1), encoding="utf-8")
    print(f"PASS  optical: {best['sensor']} {best['date']} written to {OPT}")
    return 0


def merge():
    import rasterio
    from scipy import ndimage
    import m11_ndem_site as ns

    grid_path = M11 / "flood_unet.tif"
    with rasterio.open(grid_path) as g:
        fm = g.read(1) == 1
        cell = abs(g.transform.a)
    lab, n = ndimage.label(fm, structure=np.ones((3, 3), bool))
    idx = np.arange(1, n + 1)
    out = {"patch": idx}
    with rasterio.open(M11 / f"{EVENT}.tif") as s:
        ev = ns.to_db(s.read(1).astype("float32"))
    out["vv_12jul_db"] = ndimage.median(ev, lab, idx)
    pre_p = M11 / f"{PRE}.tif"
    have_pre = pre_p.exists()
    if have_pre:
        with rasterio.open(pre_p) as s:
            pv = ns.to_db(s.read(1).astype("float32"))
        out["vv_pre_db"] = ndimage.median(pv, lab, idx)
    used = OPT / "optical_used.json"
    opt = json.loads(used.read_text(encoding="utf-8")) if used.exists() else None
    if opt:
        with rasterio.open(OPT / opt["files"][1]) as s:
            w = s.read(1)
        clear = (w != 255)
        out["opt_clear_share"] = ndimage.mean(clear, lab, idx)
        nclear = ndimage.sum_labels(clear, lab, idx)
        nwater = ndimage.sum_labels(w == 1, lab, idx)
        out["opt_water_share"] = np.where(nclear > 0, nwater / np.maximum(nclear, 1), np.nan)

    rows = {}
    for i in range(n):
        r = {"patch": int(idx[i]), "vv_12jul_db": round(float(out["vv_12jul_db"][i]), 1)}
        if have_pre and np.isfinite(out["vv_pre_db"][i]):
            pre_v = float(out["vv_pre_db"][i])
            ch = r["vv_12jul_db"] - pre_v
            r["vv_pre_db"] = round(pre_v, 1)
            r["change_db"] = round(ch, 1)
            r["radar_hint"] = ("darker on 12 July" if ch <= CHANGE_DB else
                               "dark before too" if pre_v < DARK_DB else "no clear change")
        elif have_pre:
            r.update(vv_pre_db="", change_db="", radar_hint="no pre-flood radar data here")
        else:
            r.update(vv_pre_db="", change_db="", radar_hint="no pre-flood radar")
        if opt:
            cs, wsh = float(out["opt_clear_share"][i]), float(out["opt_water_share"][i])
            r["opt_date"] = f"{opt['sensor']} {opt['date']}"
            r["opt_clear_share"] = round(cs, 2)
            r["opt_water_share"] = round(wsh, 2) if np.isfinite(wsh) else ""
            r["optical_hint"] = ("water seen" if cs >= CLEAR_MIN and wsh >= WATER_YES else
                                 "no water seen" if cs >= CLEAR_MIN and wsh < WATER_NO else "cloud or unclear")
        else:
            r.update(opt_date="", opt_clear_share="", opt_water_share="", optical_hint="no usable optical image")
        rows[r["patch"]] = r

    CHECK.mkdir(parents=True, exist_ok=True)
    allrows = sorted(rows.values(), key=lambda r: r["patch"])
    with open(CHECK / "extra_checks_unet.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(allrows[0]))
        wr.writeheader()
        wr.writerows(allrows)
    pts = json.loads((CHECK / "valley_scan_check_unet.geojson").read_text(encoding="utf-8"))
    feats = []
    for ft in pts["features"]:
        p = ft["properties"]
        e = rows[int(p["patch"])]
        keep = {k: p[k] for k in ("rank", "patch", "cells", "area_m2", "flags", "lon", "lat")}
        feats.append({"type": "Feature", "geometry": ft["geometry"], "properties": {**keep, **{
            k: v for k, v in e.items() if k != "patch"}}})
    (CHECK / "extra_checks_unet.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "name": "extra_checks_unet", "features": feats}), encoding="utf-8")

    def tally(rs, key):
        c = {}
        for r in rs:
            c[r[key]] = c.get(r[key], 0) + 1
        return c
    marked = [ft["properties"] for ft in feats]
    md = ["# M11 Beas: extra checks on the U-Net flood patches", "",
          "Made by m11_extra_checks.py from the m11_beas.py outputs. Neither check is a score: each adds evidence "
          "to the by-eye check of the marked patches. The rules for the hints were set for this check, not taken "
          "from the literature.", ""]
    if have_pre:
        prec = json.loads((M11 / f"{PRE}_record.json").read_text(encoding="utf-8"))
        md += [f"Pre-flood radar: median of {len(prec['scene_ids'])} Sentinel-1 scenes of orbit {ORBIT} "
               f"{ORBIT_PASS.lower()}, {len(prec['days'])} days from {PRE_WINDOW[0]} to 2023-06-15 "
               f"({', '.join(prec['days'])}).", ""]
    else:
        md += ["Pre-flood radar: not available (run `m11_extra_checks.py pre`).", ""]
    if opt:
        md += [f"Optical: {opt['sensor']} {opt['date']}, {opt['clear_pct_of_valley']} % of the valley clear. "
               "Water = MNDWI above 0 on clear cells.", ""]
    else:
        srch = OPT / "optical_search.json"
        found = json.loads(srch.read_text(encoding="utf-8")) if srch.exists() else []
        best = max((r["clear_pct_of_valley"] for r in found), default=None)
        md += [f"Optical: no usable image from 10 to 20 July 2023 ({len(found)} sensor-days found; the clearest had "
               f"{best} % of the valley clear, under the {MIN_CLEAR_PCT:.0f} % needed)." if found else
               "Optical: no Sentinel-2 or Landsat image of the valley from 10 to 20 July 2023 was found.", ""]
    md += ["- radar hints: darker on 12 July = at least 3 dB darker than before the flood; dark before too = "
           "median VV before the flood below -15 dB and not 3 dB darker on 12 July; no clear change = anything else",
           "- optical hints: water seen = at least half the patch clear and at least half of that water; no water "
           "seen = at least half clear and under a tenth water; cloud or unclear = anything else", "",
           "| Patches | Count | " + " | ".join(sorted(tally(allrows, "radar_hint"))) + " |",
           "|---" * (2 + len(tally(allrows, "radar_hint"))) + "|"]
    for label, rs in (("all U-Net patches", allrows), ("the 104 marked patches", marked)):
        t = tally(rs, "radar_hint")
        md.append(f"| {label} | {len(rs)} | " + " | ".join(str(t.get(k, 0)) for k in sorted(tally(allrows, "radar_hint"))) + " |")
    md += ["", "| Patches | " + " | ".join(sorted(tally(allrows, "optical_hint"))) + " |",
           "|---" * (1 + len(tally(allrows, "optical_hint"))) + "|"]
    for label, rs in (("all U-Net patches", allrows), ("the 104 marked patches", marked)):
        t = tally(rs, "optical_hint")
        md.append(f"| {label} | " + " | ".join(str(t.get(k, 0)) for k in sorted(tally(allrows, "optical_hint"))) + " |")
    md += ["", "Files: extra_checks_unet.csv (every patch, by patch number), extra_checks_unet.geojson (the marked "
           "patches with their ranks, for QGIS)."]
    (CHECK / "EXTRA_CHECKS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    t = tally(marked, "radar_hint")
    print(f"  marked patches, radar: {t}")
    print(f"  marked patches, optical: {tally(marked, 'optical_hint')}")
    print(f"PASS  merge: {CHECK / 'EXTRA_CHECKS.md'}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["pre", "optical", "merge"])
    a = ap.parse_args()
    return {"pre": pre, "optical": optical, "merge": merge}[a.what]()


if __name__ == "__main__":
    sys.exit(main())
