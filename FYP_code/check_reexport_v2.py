"""
check_reexport_v2.py
Checks the corridor v2 re-export before anything on disk is replaced.
Written 30 September 2026.

Reads every file in  05 Data\\_reexport_v2\\  and compares it with the file of
the same name already in 05 Data. For each file it checks:

  1. Same grid: width, height, transform, CRS, band count, band names, dtype.
     If the grid moved, nothing downstream lines up. Hard fail.
  2. Blank share inside corridor v2, and km of the traced river with no data.
     The old files are 22.0 per cent blank and miss 29.7 km of river.
  3. Agreement with the old file where both have data. Same scene, same
     product, so the values should be identical. If they are not, the product
     changed (for example a different Copernicus DEM version) and that has to
     be understood before the swap.

Writes  _reexport_v2\\CHECK REPORT.md  always, and  _reexport_v2\\CHECK_PASSED.txt
only if every file passes. SWAP IN REEXPORT.bat refuses to run without it.

Run from Anaconda Prompt:
    conda activate fyp
    cd /d "<project folder>\\04 Code\\FYP_code"
    python check_reexport_v2.py

Added 2 October 2026:
    python check_reexport_v2.py --accept-new-cop
Use the flag ONLY after deciding to use the new Copernicus DEM. With it, DEM_COP_GLO30_30m.tif disagreeing with the old file is
recorded in the report instead of failing the check. Every other file is
checked exactly as before.

    python check_reexport_v2.py --accept-new-cop --accept-new-s2-2021
The second flag does the same for the four Sentinel-2 files of 10 February and
23 October 2021, ONLY after deciding to accept the reprocessed images.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask

ROOT = Path(__file__).resolve().parents[2]          # Final Year Project
DATA = ROOT / "05 Data"
NEW = DATA / "_reexport_v2"
CORRIDOR = DATA / "boundaries" / "corridor_v2_3km_clipped.geojson"
THALWEG = DATA / "boundaries" / "RIVER_thalweg_cartodem.geojson"

SUBDIR = {"S1_": "sentinel1", "S2_": "sentinel2", "DEM_": "dem", "WATER_": "dem",
          "LC_": "landcover", "POP_": "population", "RAIN_": "rainfall"}

# Blank-share limits inside corridor v2. A single Sentinel-2 date reaches only
# 91 to 93 per cent of the corridor (tile edges), so its limit is looser; the
# pair test below is the one that has to reach full cover.
LIMIT = {"S1_": 0.001, "S2_": 0.10, "DEM_": 0.001, "LC_": 0.001, "POP_": 0.02}
ACCEPT_NEW_COP = "--accept-new-cop" in sys.argv[1:]
ACCEPT_NEW_S2_2021 = "--accept-new-s2-2021" in sys.argv[1:]
# the two Sentinel-2 dates whose values changed in Earth Engine
S2_2021_CHANGED = ("S2_2021_dry_b_20210210_", "S2_2021_ref_alt_20211023_")
S2_PAIRS = [("2021_dry_a", "2021_dry_b"), ("2023_dry_a", "2023_dry_b"),
            ("2021_ref", "2021_ref_alt"), ("2023_ref", "2023_ref_alt")]


def subdir_for(name):
    for p, d in SUBDIR.items():
        if name.startswith(p):
            return d
    return None


def blank(a, dtype):
    if np.issubdtype(np.dtype(dtype), np.floating):
        return ~np.isfinite(a)
    return a == 0


def main():
    if not NEW.is_dir():
        sys.exit("No folder %s. Download the Drive files there first." % NEW)
    cor = gpd.read_file(CORRIDOR).to_crs(32644)
    th = gpd.read_file(THALWEG).to_crs(32644).geometry.iloc[0]
    river_pts = [th.interpolate(d) for d in np.arange(0, th.length, 50.0)]

    rows, fails, s2blank = [], [], {}
    tifs = sorted(NEW.glob("*.tif"))
    if not tifs:
        sys.exit("No .tif files in %s." % NEW)

    for f in tifs:
        sub = subdir_for(f.name)
        old = DATA / sub / f.name if sub else None
        r = {"file": f.name, "verdict": "PASS", "notes": []}
        with rasterio.open(f) as n:
            a = n.read(1)
            inside = ~geometry_mask(cor.geometry, out_shape=(n.height, n.width),
                                    transform=n.transform)
            b = blank(a, n.dtypes[0])
            r["blank_pct"] = 100.0 * b[inside].mean()
            v = np.array([x[0] for x in n.sample([(p.x, p.y) for p in river_pts])],
                         dtype=float)
            vb = ~np.isfinite(v) if np.issubdtype(np.dtype(n.dtypes[0]), np.floating) else v == 0
            r["river_km_blank"] = vb.sum() * 0.05
            if f.name.startswith("S2_") and "_indices" not in f.name:
                s2blank[f.name] = inside & b

            if old is None or not old.exists():
                r["notes"].append("no old file of this name to compare with")
                r["grid"] = "n/a"
                r["agree_pct"] = float("nan")
            else:
                with rasterio.open(old) as o:
                    same = (o.width, o.height, tuple(o.transform)[:6], o.crs,
                            o.count, o.descriptions, o.dtypes) == \
                           (n.width, n.height, tuple(n.transform)[:6], n.crs,
                            n.count, n.descriptions, n.dtypes)
                    r["grid"] = "same" if same else "DIFFERENT"
                    if not same:
                        r["verdict"] = "FAIL"
                        r["notes"].append("grid or bands differ from the old file: old %s %sx%s %s, new %s %sx%s %s"
                                          % (o.descriptions, o.width, o.height, o.dtypes,
                                             n.descriptions, n.width, n.height, n.dtypes))
                        r["agree_pct"] = float("nan")
                    else:
                        ob = o.read(1)
                        both = ~blank(ob, o.dtypes[0]) & ~b
                        if both.sum() == 0:
                            r["agree_pct"] = float("nan")
                        else:
                            tol = 1e-3 if np.issubdtype(np.dtype(n.dtypes[0]), np.floating) else 0
                            eq = np.abs(a[both].astype(float) - ob[both].astype(float)) <= tol
                            r["agree_pct"] = 100.0 * eq.mean()
                            if r["agree_pct"] < 99.9 and ACCEPT_NEW_COP \
                                    and f.name.startswith("DEM_COP_"):
                                d = np.abs(a[both].astype(float) - ob[both].astype(float))
                                r["notes"].append(
                                    "ACCEPTED (--accept-new-cop): new Copernicus DEM differs on %.2f%% "
                                    "of shared cells, median difference there %.2f m, max %.2f m"
                                    % (100.0 - r["agree_pct"], np.median(d[d > tol]), d.max()))
                            elif r["agree_pct"] < 99.9 and ACCEPT_NEW_S2_2021 \
                                    and any(k in f.name for k in S2_2021_CHANGED):
                                r["notes"].append(
                                    "ACCEPTED (--accept-new-s2-2021): Earth Engine now serves different "
                                    "values for this date; differs on %.2f%% of shared cells"
                                    % (100.0 - r["agree_pct"]))
                            elif r["agree_pct"] < 99.9:
                                r["verdict"] = "FAIL"
                                r["notes"].append("values differ from the old file where both have data. "
                                                  "Same scene should mean same values. Find out why before swapping")

        prefix = next((p for p in LIMIT if f.name.startswith(p)), None)
        if prefix and r["blank_pct"] > 100 * LIMIT[prefix]:
            r["verdict"] = "FAIL"
            r["notes"].append("blank share above %.1f per cent" % (100 * LIMIT[prefix]))
        if prefix in ("S1_", "DEM_", "LC_") and r["river_km_blank"] > 0:
            r["verdict"] = "FAIL"
            r["notes"].append("river still has blank stretches")
        if f.name.startswith("WATER_"):
            r["notes"].append("JRC is blank wherever water never occurred, so its blank share is not a test")
        rows.append(r)
        if r["verdict"] == "FAIL":
            fails.append(f.name)
        print("%-52s %-5s blank %6.2f%%  river blank %5.2f km  grid %-9s agree %s"
              % (f.name, r["verdict"], r["blank_pct"], r["river_km_blank"], r["grid"],
                 "n/a" if not np.isfinite(r["agree_pct"]) else "%.3f%%" % r["agree_pct"]))

    pair_lines = []
    for a_tag, b_tag in S2_PAIRS:
        fa = [k for k in s2blank if ("S2_%s_" % a_tag) in k]
        fb = [k for k in s2blank if ("S2_%s_" % b_tag) in k]
        if fa and fb:
            both_blank = s2blank[fa[0]] & s2blank[fb[0]]
            with rasterio.open(NEW / fa[0]) as src:
                inside_n = (~geometry_mask(cor.geometry, out_shape=both_blank.shape,
                                           transform=src.transform)).sum()
            pct = 100.0 * both_blank.sum() / inside_n
            ok = pct <= 0.5
            pair_lines.append("| %s + %s | %.2f | %s |" % (a_tag, b_tag, pct, "PASS" if ok else "FAIL"))
            if not ok:
                fails.append("pair %s + %s" % (a_tag, b_tag))

    rain = NEW / "RAIN_CHIRPS_catchment_daily_2023.csv"
    rain_line = "not downloaded"
    if rain.exists():
        with open(rain, newline="") as fh:
            rr = list(csv.DictReader(fh))
        rain_line = "%d rows, %s to %s" % (len(rr), rr[0]["date"], rr[-1]["date"])
        if rr[0]["date"] != "2023-05-01":
            fails.append("rainfall CSV does not start on 2023-05-01")
            rain_line += "  FAIL: should start 2023-05-01"

    out = ["# Re-export check, corridor v2", "",
           "Files checked: %d. Failures: %d." % (len(rows), len(fails)), "",
           "| File | Verdict | Blank in corridor % | River km blank | Grid | Agrees with old % | Notes |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append("| %s | %s | %.2f | %.2f | %s | %s | %s |" % (
            r["file"], r["verdict"], r["blank_pct"], r["river_km_blank"], r["grid"],
            "n/a" if not np.isfinite(r["agree_pct"]) else "%.3f" % r["agree_pct"],
            "; ".join(r["notes"])))
    out += ["", "## Sentinel-2 pairs, share of corridor blank in BOTH dates", "",
            "| Pair | Blank in both % | Verdict |", "|---|---|---|"] + pair_lines
    out += ["", "## Rainfall", "", rain_line, ""]
    if ACCEPT_NEW_S2_2021:
        out += ["Run with --accept-new-s2-2021: the changed values for 10 February and "
                "23 October 2021 were accepted by decision, not by the check.", ""]
    if ACCEPT_NEW_COP:
        out += ["Run with --accept-new-cop: a Copernicus DEM change was accepted by decision, "
                "not by the check.", ""]
    out.append("**RESULT: %s**" % ("PASS. Run SWAP IN REEXPORT.bat" if not fails
                                   else "FAIL. Do not swap. Read the FAIL lines above"))
    (NEW / "CHECK REPORT.md").write_text("\n".join(out), encoding="utf-8")
    flag = NEW / "CHECK_PASSED.txt"
    if fails:
        if flag.exists():
            flag.unlink()
        print("\nFAIL:", ", ".join(fails))
        print("Report written to", NEW / "CHECK REPORT.md")
        sys.exit(1)
    flag.write_text("All %d files passed.\n" % len(rows), encoding="utf-8")
    print("\nPASS. All %d files. Report written to %s" % (len(rows), NEW / "CHECK REPORT.md"))


if __name__ == "__main__":
    main()
