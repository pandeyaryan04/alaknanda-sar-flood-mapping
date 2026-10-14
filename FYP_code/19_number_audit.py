r"""
19_number_audit.py
==================
Every number the report and the methods quote, recomputed
from the files, side by side with the value as written.

Each line says where the number comes from: recomputed here from the data,
read from the output file of the script that made it, or (for the Earth
Engine console numbers) from 04 Code\provenance\EARTH_ENGINE_INPUTS.csv,
which 17_earth_engine_inputs.py writes. Run that first.

Writes
  06 Outputs\number_audit\NUMBER_AUDIT.md    the table
  06 Outputs\number_audit\NUMBER_AUDIT.csv   the same, for a spreadsheet

USAGE
-----
  conda activate fyp
  cd /d "<project folder>\04 Code\FYP_code"
  python 19_number_audit.py

PASS: "all <n> numbers match". A line marked DIFF means the written number
does not follow from the files: correct the text, or find out why. A line
marked NO FILE names the file that is missing.

When the report quotes a new number, add a line to CHECKS below.
"""

__stage__ = "Provenance, number audit"
__reads__ = ["the data and script outputs named in each check"]
__writes__ = ["06 Outputs/number_audit/NUMBER_AUDIT.md", "NUMBER_AUDIT.csv"]

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

try:
    import geopandas as gpd
    import pandas as pd
    import rasterio
    import shapely
    from rasterio.features import rasterize
except ImportError as e:
    sys.exit(f"\nMissing package: {e}.\nActivate the environment first:  "
             f"conda activate fyp\n")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
D = ROOT / "05 Data"
FD = HERE / "data"
OUTD = ROOT / "06 Outputs"
CORRIDOR = D / "boundaries" / "corridor_v2_3km_clipped.geojson"
EE_CSV = ROOT / "04 Code" / "provenance" / "EARTH_ENGINE_INPUTS.csv"
OUT = OUTD / "number_audit"
UTM = "EPSG:32644"


class NoFile(Exception):
    pass


def need(p):
    p = Path(p)
    if not p.exists():
        raise NoFile(str(p.relative_to(ROOT)) if ROOT in p.parents else str(p))
    return p


def corridor():
    return gpd.read_file(need(CORRIDOR)).to_crs(UTM)


def km2_cells(mask, transform):
    return float(mask.sum()) * abs(transform.a * transform.e) / 1e6


def raster_km2(path, value=1):
    with rasterio.open(need(path)) as s:
        a = s.read(1)
        return km2_cells(a == value, s.transform)


def load12():
    spec = importlib.util.spec_from_file_location("pwm", need(HERE / "12_permanent_water_mask.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def strata(year, dim, stratum, col):
    t = pd.read_csv(need(FD / f"strata_{year}_v3" / "strata_summary.csv"))
    return float(t[(t.dimension == dim) & (t.stratum == stratum)][col].iloc[0])


def perm_parts():
    m = load12()
    with rasterio.open(need(D / "dem" / "WATER_PERMANENT_JRC50_S2DRY_10m.tif")) as s:
        pw = s.read(1) > 0
        grid = {"transform": s.transform, "crs": s.crs, "shape": s.shape}
    jrc = m.jrc_on_grid(grid) & m.corridor_on_grid(grid)
    t = grid["transform"]
    return {"all": km2_cells(pw, t), "jrc": km2_cells(jrc & pw, t),
            "s2_only": km2_cells(pw & ~jrc, t)}


def change_range(year, which):
    vals = [raster_km2(FD / f"flood_{year}_ch_{m}_v3" / f"flood_{year}ch{m}_otsu.tif")
            for m in ("nov", "dec", "jan", "feb")]
    return min(vals) if which == "min" else max(vals)


def qa_added(year, window=None):
    g = gpd.read_file(need(OUTD / "remake" / f"QA_dry_water_not_drawn_{year}.gpkg"))
    if window is not None:
        g = g[g["window"] == window]
    return g["area_m2"].sum() / 1e6


def window_gap(year, window, what):
    w = gpd.read_file(need(FD / f"reference_windows_{year}" / "reference_windows.geojson")).to_crs(UTM)
    poly = w[w["window"] == window].geometry.iloc[0].intersection(corridor().union_all())
    if what == "inside":
        return poly.area / 1e6
    h = gpd.read_file(need(D / "sentinel2" / f"S2_{year}_event_cloudgap.gpkg")).to_crs(UTM)
    return poly.intersection(h.union_all()).area / 1e6


def reservoir(folder, scene=None, spread=False):
    t = pd.read_csv(need(Path(folder) / "reservoir_area.csv")).set_index("scene")
    if spread:
        a = t.loc[["S2_2021_dry_a_20210208", "S2_2023_dry_a_20230203"], "water_km2"]
        return 100 * (a.max() - a.min()) / a.max()
    return float(t.loc[scene, "water_km2"])


SRINAGAR = OUTD / "gis_tools" / "reservoir"
PASHULOK = ROOT / "99 Archive" / "reservoir check - Pashulok pond Rishikesh - 2026-10-02"


def villages(what):
    v = gpd.read_file(need(D / "boundaries" / "VILLAGES_corridor.gpkg"))
    p = pd.to_numeric(v["total_population_village"], errors="coerce")
    return {"count": len(v), "with_pop": int(p.notna().sum()),
            "without_pop": int(p.isna().sum()), "whole": float(p.sum()),
            "weighted": float((p * v["fraction_in_corridor"]).sum())}[what]


def landslides(what):
    g = gpd.read_file(need(D / "landslide" / "LANDSLIDE_corridor.gpkg"))
    return len(g) if what == "polygons" else g["gid"].nunique()


def rain(first, last):
    t = pd.read_csv(need(D / "rainfall" / "RAIN_CHIRPS_catchment_daily_2023.csv"),
                    parse_dates=["date"]).set_index("date")
    return float(t["rain_mean_mm"][first:last].sum())


def dem(pair, stratum, col):
    t = pd.read_csv(need(OUTD / "dem_comparison" / "DEM_pairwise_stats.csv"))
    return float(t[(t.pair == pair) & (t.stratum == stratum)][col].iloc[0])


_RTC = {}


def rtc(year, what):
    if year not in _RTC:
        stem, g = {2021: ("S1RTC_2021_post_20211021_orb63", "S1_2021_post_20211021_orb63"),
                   2023: ("S1RTC_2023_post_20230816_orb129", "S1_2023_post_20230816_orb129")}[year]
        with rasterio.open(need(D / "sentinel1_rtc" / f"{stem}.tif")) as s:
            rv = s.read(1)
        with rasterio.open(need(D / "sentinel1_rtc" / f"{stem}_incmap_deg.tif")) as s:
            inc = s.read(1)
        with rasterio.open(need(D / "sentinel1_rtc" / f"{stem}_lsmap.tif")) as s:
            ls = s.read(1)
        with rasterio.open(need(D / "sentinel1" / f"{g}.tif")) as s:
            gv, ga = s.read(1), s.read(3)
        with rasterio.open(need(D / "dem" / "WATER_PERMANENT_JRC50_S2DRY_10m.tif")) as s:
            pw = s.read(1)
        inside = np.isfinite(gv)
        both = inside & np.isfinite(rv)
        shadow = inside & (ls != 255) & ((ls & 16) > 0)
        flat = both & np.isfinite(inc) & (np.abs(inc - ga) < 5)

        def sep(a):
            w, l = a[both & (pw >= 50)], a[both & (pw < 50)]
            return (np.median(l) - np.median(w)) / np.sqrt(w.var() + l.var())
        _RTC[year] = {
            "empty_pct": 100 - 100 * both.sum() / inside.sum(),
            "outside_shadow_pct": 100 * (both & ~shadow).sum() / (inside & ~shadow).sum(),
            "r_flat": np.corrcoef(rv[flat], gv[flat])[0, 1],
            "sep_grd": sep(gv), "sep_rtc": sep(rv)}
    return float(_RTC[year][what])


def ee_console(key):
    t = pd.read_csv(need(EE_CSV))
    row = t[(t["group"] == "console") & (t["tag"] == key)]
    if row.empty:
        raise NoFile(f"{EE_CSV.name}: no '{key}' row (run 17_earth_engine_inputs.py)")
    return float(json.loads(row["details"].iloc[0])["value"])


def landcover(name, kind):
    f = {"lulc": "LANDCOVER_LULC50K_area_by_class_corridor.csv",
         "wc": "LANDCOVER_WorldCover_area_by_class_corridor.csv"}[kind]
    t = pd.read_csv(need(D / "landcover" / f))
    col = "level_1" if kind == "lulc" and name == "TOTAL" else ("level_2" if kind == "lulc" else "name")
    row = t[t[col] == name]
    return row


def lulc_total():
    r = landcover("TOTAL", "lulc")
    if r.empty:
        raise NoFile("05 Data/landcover/LANDCOVER_LULC50K_area_by_class_corridor.csv has no TOTAL row")
    v = float(r["area_km2"].iloc[0])
    if v > 700:
        raise NoFile("05 Data/landcover still holds the old-corridor table (807 km2): run 16_landcover_corridor.py and copy its tables in (RUN ORDER.md, step 3.8)")
    return v


# (id, what the text says, value as written, tolerance, function, source)
CHECKS = [
    ("C1", "Corridor v2 area, km2", 606.35, 0.005,
     lambda: corridor().area.sum() / 1e6, "recomputed, corridor_v2_3km_clipped.geojson"),
    ("C2", "Gorge above Devprayag, % of corridor", 52.58, 0.005,
     lambda: strata(2021, "reach", "gorge_above_devprayag", "share_pct"), "03_terrain_strata.py, strata_summary.csv"),
    ("C3", "Gorge above Devprayag, km2", 318.84, 0.005,
     lambda: strata(2021, "reach", "gorge_above_devprayag", "area_km2"), "03_terrain_strata.py, strata_summary.csv"),
    ("C4", "Open valley below Devprayag, km2", 287.52, 0.005,
     lambda: strata(2021, "reach", "open_valley_below_devprayag", "area_km2"), "03_terrain_strata.py, strata_summary.csv"),
    ("C5", "Layover 2021 (descending), %", 2.05, 0.005,
     lambda: strata(2021, "layover_shadow", "layover", "share_pct"), "03_terrain_strata.py"),
    ("C6", "Shadow 2021, %", 0.64, 0.005,
     lambda: strata(2021, "layover_shadow", "shadow", "share_pct"), "03_terrain_strata.py"),
    ("C7", "Layover 2023 (ascending), %", 4.12, 0.005,
     lambda: strata(2023, "layover_shadow", "layover", "share_pct"), "03_terrain_strata.py"),
    ("C8", "Shadow 2023, %", 0.42, 0.005,
     lambda: strata(2023, "layover_shadow", "shadow", "share_pct"), "03_terrain_strata.py"),
    ("P1", "Permanent-water mask, km2", 11.08, 0.005,
     lambda: perm_parts()["all"], "recomputed, WATER_PERMANENT mask"),
    ("P2", "  of which JRC 50 %, km2", 8.65, 0.005,
     lambda: perm_parts()["jrc"], "recomputed, JRC occurrence on the 10 m grid"),
    ("P3", "  of which dry-season river JRC misses, km2", 2.43, 0.005,
     lambda: perm_parts()["s2_only"], "recomputed"),
    ("F1", "Single-image Otsu water 2021, km2", 62.4, 0.05,
     lambda: raster_km2(FD / "flood_2021_v3" / "flood_2021_otsu.tif"), "recomputed, flood map"),
    ("F2", "Single-image Otsu water 2023, km2", 46.3, 0.05,
     lambda: raster_km2(FD / "flood_2023_v3" / "flood_2023_otsu.tif"), "recomputed, flood map"),
    ("F3", "Change detection, smallest of 8 maps, km2", 2.0, 0.05,
     lambda: min(change_range(2021, "min"), change_range(2023, "min")), "recomputed, 8 flood maps"),
    ("F4", "Change detection, largest of 8 maps, km2", 2.7, 0.05,
     lambda: max(change_range(2021, "max"), change_range(2023, "max")), "recomputed, 8 flood maps"),
    ("R1", "Reference flood 2021, km2", 2.401, 0.0005,
     lambda: raster_km2(FD / "reference_2021.tif"), "recomputed, reference_2021.tif"),
    ("R2", "Reference flood 2023, km2", 3.209, 0.0005,
     lambda: raster_km2(FD / "reference_2023.tif"), "recomputed, reference_2023.tif"),
    ("R3", "February water added by rule 2021, km2", 1.57, 0.005,
     lambda: qa_added(2021), "15_reference_dry_completion.py output"),
    ("R4", "  of which window 8 (Srinagar impoundment), km2", 1.12, 0.005,
     lambda: qa_added(2021, 8), "15_reference_dry_completion.py output"),
    ("R5", "February water added by rule 2023, km2", 0.77, 0.005,
     lambda: qa_added(2023), "15_reference_dry_completion.py output"),
    ("W1", "2023 window 1, area inside the corridor, km2", 24.6, 0.05,
     lambda: window_gap(2023, 1, "inside"), "recomputed, windows and corridor"),
    ("W2", "2023 window 1, no clear optical image, km2", 8.98, 0.005,
     lambda: window_gap(2023, 1, "gap"), "recomputed, windows and cloud-gap holes"),
    ("S1", "Srinagar impoundment, Feb 2021, km2", 1.56, 0.005,
     lambda: reservoir(SRINAGAR, "S2_2021_dry_a_20210208"), "t04_reservoir.py, reservoir_area.csv"),
    ("S2", "Srinagar impoundment, Feb 2023, km2", 2.06, 0.005,
     lambda: reservoir(SRINAGAR, "S2_2023_dry_a_20230203"), "t04_reservoir.py"),
    ("S3", "Srinagar spread between the two, %", 24.6, 0.05,
     lambda: reservoir(SRINAGAR, spread=True), "t04 formula on reservoir_area.csv (was worked by hand)"),
    ("S4", "Srinagar impoundment, 21 Oct 2021, km2", 2.40, 0.005,
     lambda: reservoir(SRINAGAR, "S2_2021_ref_20211021"), "t04_reservoir.py"),
    ("S5", "Pashulok pond, Feb 2021, km2", 0.28, 0.005,
     lambda: reservoir(PASHULOK, "S2_2021_dry_a_20210208"), "t04_reservoir.py, first run, archived"),
    ("S6", "Pashulok pond, Feb 2023, km2", 0.53, 0.005,
     lambda: reservoir(PASHULOK, "S2_2023_dry_a_20230203"), "t04_reservoir.py, first run, archived"),
    ("S7", "Pashulok spread, %", 48.0, 0.5,
     lambda: reservoir(PASHULOK, spread=True), "t04 formula"),
    ("V1", "Villages meeting the corridor", 544, 0,
     lambda: villages("count"), "t08_villages.py output"),
    ("V2", "  with census population", 478, 0, lambda: villages("with_pop"), "t08_villages.py output"),
    ("V3", "  with no census population", 66, 0, lambda: villages("without_pop"), "t08_villages.py output"),
    ("V4", "Population, villages counted whole", 281520, 0.5,
     lambda: villages("whole"), "t08_villages.py output"),
    ("V5", "Population, by share of village area inside", 239796, 0.5,
     lambda: villages("weighted"), "recomputed (was worked by hand)"),
    ("V6", "WorldPop 2020 inside the corridor (native ~93 m grid)", 327395, 164,
     lambda: ee_console("worldpop_2020_sum_native_scale"),
     "Earth Engine, 17_earth_engine_inputs.py (within 0.05 %)"),
    ("V7", "Catchment above Rudraprayag, km2", 8638, 1,
     lambda: ee_console("catchment_area_km2"), "Earth Engine, 17_earth_engine_inputs.py"),
    ("V8", "WorldCover total in the Earth Engine table, km2", 606.08, 0.05,
     lambda: ee_console("worldcover_total_km2"), "Earth Engine, 17_earth_engine_inputs.py"),
    ("V9", "WorldCover tree cover, Earth Engine table, %", 77.30, 0.05,
     lambda: ee_console("worldcover_tree_share_pct"), "Earth Engine, 17_earth_engine_inputs.py"),
    ("L1", "Landslide polygons meeting the corridor", 626, 0,
     lambda: landslides("polygons"), "09_landslide_merge.py output"),
    ("L2", "Landslides (distinct gid) meeting the corridor", 625, 0,
     lambda: landslides("slides"), "recomputed (was worked by hand)"),
    ("N1", "CHIRPS 60 days to 18 July 2023, mm", 461.2, 0.05,
     lambda: rain("2023-05-20", "2023-07-18"), "recomputed, CHIRPS catchment CSV"),
    ("N2", "CHIRPS 30 days to 18 July 2023, mm", 391.7, 0.05,
     lambda: rain("2023-06-19", "2023-07-18"), "recomputed"),
    ("N3", "CHIRPS 7 days to 18 July 2023, mm", 149.3, 0.05,
     lambda: rain("2023-07-12", "2023-07-18"), "recomputed"),
    ("D1", "Copernicus minus CartoDEM, mean, m", 46.92, 0.005,
     lambda: dem("GLO30 - CartoDEM", "whole corridor", "mean"), "07_dem_comparison.py output"),
    ("D2", "NMAD CartoDEM against SRTM, m", 10.56, 0.005,
     lambda: dem("CartoDEM - SRTM", "whole corridor", "nmad"), "07_dem_comparison.py output"),
    ("D3", "NMAD Copernicus against SRTM, m", 6.68, 0.005,
     lambda: dem("GLO30 - SRTM", "whole corridor", "nmad"), "07_dem_comparison.py output"),
    ("D4", "NMAD CartoDEM against SRTM, slopes over 30 deg, m", 15.38, 0.005,
     lambda: dem("CartoDEM - SRTM", "slope over 30 deg", "nmad"), "07_dem_comparison.py output"),
    ("D5", "NMAD Copernicus against SRTM, slopes over 30 deg, m", 8.44, 0.005,
     lambda: dem("GLO30 - SRTM", "slope over 30 deg", "nmad"), "07_dem_comparison.py output"),
    ("T1", "RTC empty (radar shadow), 2021, %", 1.18, 0.005, lambda: rtc(2021, "empty_pct"), "recomputed, RTC and GRD grids"),
    ("T2", "RTC empty, 2023, %", 0.53, 0.005, lambda: rtc(2023, "empty_pct"), "recomputed"),
    ("T3", "RTC coverage outside shadow, 2021, %", 99.5, 0.05, lambda: rtc(2021, "outside_shadow_pct"), "recomputed"),
    ("T4", "RTC coverage outside shadow, 2023, %", 99.8, 0.05, lambda: rtc(2023, "outside_shadow_pct"), "recomputed"),
    ("T5", "RTC against GRD, r, flat-facing ground, 2021", 0.80, 0.005, lambda: rtc(2021, "r_flat"), "recomputed (rule of check_rtc.py)"),
    ("T6", "RTC against GRD, r, flat-facing ground, 2023", 0.73, 0.005, lambda: rtc(2023, "r_flat"), "recomputed"),
    ("T7", "Water-land separation GRD 2021", 1.12, 0.005, lambda: rtc(2021, "sep_grd"), "recomputed (was one-off code)"),
    ("T8", "Water-land separation RTC 2021", 1.34, 0.005, lambda: rtc(2021, "sep_rtc"), "recomputed"),
    ("T9", "Water-land separation GRD 2023", 0.92, 0.005, lambda: rtc(2023, "sep_grd"), "recomputed"),
    ("T10", "Water-land separation RTC 2023", 1.20, 0.005, lambda: rtc(2023, "sep_rtc"), "recomputed"),
    ("K1", "LULC 50K total inside corridor v2, km2", 606.39, 0.005, lulc_total, "16_landcover_corridor.py output"),
    ("K2", "LULC 50K evergreen forest share, %", 42.30, 0.005,
     lambda: float(landcover("Evergreen / Semi Evergreen", "lulc")["share_pct"].iloc[0]) if lulc_total() else 0,
     "16_landcover_corridor.py output"),
    ("K3", "WorldCover tree cover, cell count, %", 77.27, 0.005,
     lambda: float(landcover("Tree cover", "wc")["share_pct"].iloc[0]), "16_landcover_corridor.py output"),
]


def main():
    rows, n_ok, n_diff, n_nofile = [], 0, 0, 0
    print(f"{'id':4s} {'what':52s} {'written':>10s} {'from files':>12s}  verdict")
    for cid, what, value, tol, fn, src in CHECKS:
        try:
            got = fn()
            ok = abs(got - value) <= tol + 1e-9
            verdict = "ok" if ok else "DIFF"
            n_ok += ok
            n_diff += not ok
            got_s = f"{got:,.4f}".rstrip("0").rstrip(".")
        except NoFile as e:
            verdict, got_s = f"NO FILE: {e}", ""
            n_nofile += 1
        print(f"{cid:4s} {what:52s} {value:>10,} {got_s:>12s}  {verdict}")
        rows.append({"id": cid, "what": what, "written": value, "from_files": got_s,
                     "tolerance": tol, "verdict": verdict, "source": src})

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "NUMBER_AUDIT.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = ["# Number audit", "", "Written by `04 Code\\FYP_code\\19_number_audit.py`.", "",
          "| id | What | Written | From the files | Verdict | Source |", "|---|---|---|---|---|---|"]
    md += [f"| {r['id']} | {r['what'].strip()} | {r['written']:,} | {r['from_files']} | {r['verdict']} | {r['source']} |"
           for r in rows]
    (OUT / "NUMBER_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\n  {len(rows)} numbers: {n_ok} match, {n_diff} DIFF, {n_nofile} NO FILE")
    print("  all %d numbers match" % len(rows) if n_ok == len(rows) else "  NOT all numbers match")
    print(f"  written: {OUT / 'NUMBER_AUDIT.md'}")
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
