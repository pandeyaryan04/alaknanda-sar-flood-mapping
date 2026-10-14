"""Clip check: how many corridor cells had HAND or slope above the training clip limit.

Reads (nothing is changed):
  04 Code\\FYP_code\\data\\models\\norm_stats.json      clip_hi for HAND and SLOPE
  04 Code\\FYP_code\\data\\strata_<year>_v3\\           HAND and slope, 30 m
  06 Outputs\\jalsar\\<event>\\flood_headline.tif       headline flood map, 10 m (the U-Net)
  04 Code\\FYP_code\\data\\reference_<year>.tif        digitised flood, 10 m (1 flood, 0 not, 255 not scored)
Writes:
  06 Outputs\\checks\\clip_check.md
"""

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

# The project folder is two levels up: <project>\04 Code\clip_check\clip_check.py
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "04 Code" / "FYP_code" / "data"
OUTS = ROOT / "06 Outputs"
NORM = DATA / "models" / "norm_stats.json"
EVENTS = [
    ("2021", OUTS / "jalsar" / "2021-10-21_orbit063"),
    ("2023", OUTS / "jalsar" / "2023-08-16_orbit129"),
]


def reproject_to(src_path, profile, resampling, src_nodata, dst_nodata, dtype):
    """Read one band and resample it onto the headline flood grid."""
    with rasterio.open(src_path) as src:
        data = src.read(1)
        src_transform, src_crs = src.transform, src.crs
    dst = np.full((profile["height"], profile["width"]), dst_nodata, dtype=dtype)
    reproject(
        source=data,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=profile["transform"],
        dst_crs=profile["crs"],
        resampling=resampling,
        src_nodata=src_nodata,
        dst_nodata=dst_nodata,
    )
    return dst


def scores(mask, pred, truth):
    """Cell-level precision, recall and F1 for cells inside mask and scored in the reference."""
    m = mask & (truth != 255)
    t = truth[m] == 1
    p = pred[m]
    tp = int(np.sum(p & t))
    fp = int(np.sum(p & ~t))
    fn = int(np.sum(~p & t))
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")
    return int(m.sum()), int(t.sum()), prec, rec, f1


def run_event(year, out_dir, hand_cap, slope_cap):
    lines = [f"## {year}", ""]
    flood_path = out_dir / "flood_headline.tif"
    ref_path = DATA / f"reference_{year}.tif"
    strata = DATA / f"strata_{year}_v3"
    needed = [flood_path, ref_path, strata / "strata_hand_m.tif", strata / "strata_slope_deg.tif"]
    missing = [p for p in needed if not p.exists()]
    if missing:
        return lines + ["NOT RUN, files missing:"] + [f"- {p}" for p in missing] + [""]

    with rasterio.open(flood_path) as src:
        prof = src.profile
        flood = src.read(1)
    pixel_km2 = abs(prof["transform"].a * prof["transform"].e) / 1e6

    vals, counts = np.unique(flood, return_counts=True)
    lines.append("Flood map values (expected 1 = flood, 0 = not flood, nodata otherwise): " +
                 ", ".join(f"{int(v)}: {int(c)}" for v, c in zip(vals, counts)))

    hand = reproject_to(strata / "strata_hand_m.tif", prof, Resampling.bilinear, -9999, np.nan, "float32")
    slope = reproject_to(strata / "strata_slope_deg.tif", prof, Resampling.bilinear, -9999, np.nan, "float32")
    ref = reproject_to(ref_path, prof, Resampling.nearest, 255, 255, "uint8")

    valid = np.isfinite(hand) & np.isfinite(slope)
    cap_h = valid & (hand > hand_cap)
    cap_s = valid & (slope > slope_cap)
    cap = cap_h | cap_s
    flood_bool = flood == 1

    n = int(valid.sum())
    lines += [
        "",
        f"Corridor cells (valid HAND and slope): {n:,}, area {n * pixel_km2:.2f} km2",
        f"Largest HAND in corridor: {np.nanmax(hand[valid]):.1f} m (cap {hand_cap:.2f} m)",
        f"Largest slope in corridor: {np.nanmax(slope[valid]):.1f} degrees (cap {slope_cap:.2f} degrees)",
        "",
        "| Capped input | Cells | Share of corridor | Area km2 |",
        "|---|---|---|---|",
        f"| HAND above cap | {int(cap_h.sum()):,} | {100 * cap_h.sum() / n:.2f} % | {cap_h.sum() * pixel_km2:.2f} |",
        f"| Slope above cap | {int(cap_s.sum()):,} | {100 * cap_s.sum() / n:.2f} % | {cap_s.sum() * pixel_km2:.2f} |",
        f"| Either | {int(cap.sum()):,} | {100 * cap.sum() / n:.2f} % | {cap.sum() * pixel_km2:.2f} |",
        "",
    ]

    flood_total = int((flood_bool & valid).sum())
    flood_capped = int((flood_bool & cap).sum())
    share = 100 * flood_capped / flood_total if flood_total else float("nan")
    lines.append(f"Headline flood cells in corridor: {flood_total:,}; of these in capped cells: "
                 f"{flood_capped:,} ({share:.2f} %)")
    lines.append("")

    lines += [
        "Cell-level scores inside the digitised windows (headline map, not comparable with the published F1):",
        "",
        "| Group | Scored cells | Reference flood cells | Precision | Recall | F1 |",
        "|---|---|---|---|---|---|",
    ]
    for label, mask in [("Capped", valid & cap), ("Not capped", valid & ~cap)]:
        nc, npos, prec, rec, f1 = scores(mask, flood_bool, ref)
        lines.append(f"| {label} | {nc:,} | {npos:,} | {prec:.3f} | {rec:.3f} | {f1:.3f} |")
    lines.append("")
    return lines


def main():
    norm = json.loads(NORM.read_text(encoding="utf-8"))
    hand_cap = norm["clip_hi"][2]
    slope_cap = norm["clip_hi"][3]

    lines = [
        "# Clip check",
        "",
        f"Training clip limits (99.5th percentile of the 251 hand train chips): "
        f"HAND {hand_cap:.2f} m, slope {slope_cap:.2f} degrees.",
        "Cells above a limit were fed the limit value. This check counts them and does not change any output.",
        "",
    ]
    for year, out_dir in EVENTS:
        lines += run_event(year, out_dir, hand_cap, slope_cap)

    text = "\n".join(lines) + "\n"
    out = OUTS / "checks" / "clip_check.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print("Written:", out)


if __name__ == "__main__":
    main()
