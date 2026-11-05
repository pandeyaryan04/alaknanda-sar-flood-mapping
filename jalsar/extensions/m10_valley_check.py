"""Jal-SAR extensions, M10: the valley HAND check (25 Oct 2026). Reads only; changes nothing.

Why: in M10 every open-valley radar map had its best HAND threshold at the smallest value tried
(0.25 m, then 0.05 m once the grid was extended down). This shows where the radar flood cells of each reach sit:
HAND percentiles of flood and dry cells, the share at HAND 0, the distance to permanent water, and
CSI, hit rate and false alarm ratio of "HAND below H" for a few H. Same cells as M10's event_layers
(flood map 0/1, trust not radar-blind, permanent water left out, HAND known).

Run:  python m10_valley_check.py      writes 06 Outputs\\jalsar\\extensions\\m10_forecast_extent\\valley_check.json
"""
import json, numpy as np, rasterio
from rasterio.warp import reproject, Resampling, transform as wt
from pathlib import Path
U = str(Path(__file__).resolve().parents[3])
OUTF = Path(U) / "06 Outputs" / "jalsar" / "extensions" / "m10_forecast_extent" / "valley_check.json"
EV = {"2021-10-21_orbit063": "2021", "2023-07-19_orbit063": "2021", "2023-08-16_orbit129": "2023",
      "2025-08-29_orbit129": "2023", "2026-08-03_orbit063": "2021", "2026-08-07_orbit129": "2023"}
def rd(p):
    with rasterio.open(p) as s: return s.read(1), s.profile
def onto(p, prof, cat=False):
    out = np.full((prof["height"], prof["width"]), np.nan, "float32")
    with rasterio.open(p) as s:
        reproject(rasterio.band(s, 1), out, src_transform=s.transform, src_crs=s.crs, dst_transform=prof["transform"],
                  dst_crs=prof["crs"], resampling=Resampling.nearest if cat else Resampling.bilinear,
                  src_nodata=s.nodata, dst_nodata=np.nan)
    return out
H = [0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 15, 25]
res = {}
for e, y in EV.items():
    fl, prof = rd(f"{U}/06 Outputs/jalsar/{e}/flood_headline.tif"); tr, _ = rd(f"{U}/06 Outputs/jalsar/{e}/trust.tif")
    sd = f"{U}/04 Code/FYP_code/data/strata_{y}_v3"
    hand = onto(f"{sd}/strata_hand_m.tif", prof); vw = onto(f"{sd}/strata_valley_width_m.tif", prof)
    perm = onto(f"{U}/05 Data/dem/WATER_PERMANENT_JRC50_S2DRY_10m.tif", prof, True) == 100
    lon = json.load(open(f"{sd}/strata_meta.json"))["reach_split_longitude"]
    sx = wt("EPSG:4326", prof["crs"], [lon], [30.15])[0][0]; t = prof["transform"]
    xs = t.c + t.a * (np.arange(prof["width"]) + 0.5); valley = np.broadcast_to(xs < sx, fl.shape)
    mask = np.isin(fl, (0, 1)) & (tr != 0) & (tr != 255) & ~perm & np.isfinite(hand)
    # distance from permanent water, in cells (10 m)
    from scipy.ndimage import distance_transform_edt
    dist_m = distance_transform_edt(~perm) * abs(t.a)
    r = {}
    for name, sel in (("valley", valley), ("gorge", ~valley)):
        m = mask & sel; f = m & (fl == 1); d = m & (fl == 0)
        hf, hd = hand[f], hand[d]
        r[name] = {"flood_cells": int(f.sum()), "dry_cells": int(d.sum()), "flood_pct_of_valid": round(100 * f.sum() / max(m.sum(), 1), 2),
                   "hand_flood_pctl": {p: round(float(np.percentile(hf, p)), 2) for p in (10, 25, 50, 75, 90)} if f.any() else None,
                   "hand_dry_pctl": {p: round(float(np.percentile(hd, p)), 2) for p in (10, 25, 50, 75, 90)} if d.any() else None,
                   "flood_share_hand_eq0": round(float((hf <= 0.001).mean()), 3) if f.any() else None,
                   "dry_share_hand_eq0": round(float((hd <= 0.001).mean()), 3) if d.any() else None,
                   "flood_dist_to_perm_water_m_median": round(float(np.median(dist_m[f])), 0) if f.any() else None,
                   "flood_within_30m_of_perm_water": round(float((dist_m[f] <= 30).mean()), 3) if f.any() else None,
                   "flood_valley_width_median_m": round(float(np.nanmedian(vw[f])), 0) if f.any() else None,
                   "curve": {}}
        for h in H:
            tp = float((hf < h).sum()); fp = float((hd < h).sum()); fn = len(hf) - tp
            r[name]["curve"][h] = {"csi": round(tp / (tp + fp + fn), 3) if tp + fp + fn else None,
                                   "hit": round(tp / len(hf), 3) if len(hf) else None,
                                   "far": round(fp / (tp + fp), 3) if tp + fp else None}
    res[e] = r
json.dump(res, open(OUTF, "w"), indent=1)
for e, r in res.items():
    print("\n##", e)
    for n in ("valley", "gorge"):
        x = r[n]
        print(f" {n:6s} flood {x['flood_cells']:7d} ({x['flood_pct_of_valid']}% of valid) | HAND flood p10/50/90 "
              f"{x['hand_flood_pctl'][10]}/{x['hand_flood_pctl'][50]}/{x['hand_flood_pctl'][90]} | dry p10/50/90 "
              f"{x['hand_dry_pctl'][10]}/{x['hand_dry_pctl'][50]}/{x['hand_dry_pctl'][90]} | HAND=0: flood {x['flood_share_hand_eq0']} dry {x['dry_share_hand_eq0']} "
              f"| dist to perm water med {x['flood_dist_to_perm_water_m_median']} m, <=30 m {x['flood_within_30m_of_perm_water']}")
        print("        CSI/hit/FAR by H:", "  ".join(f"{h}:{c['csi']}/{c['hit']}/{c['far']}" for h, c in x["curve"].items()))
