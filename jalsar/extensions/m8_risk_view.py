"""Jal-SAR extensions: village risk view and WorldPop people per village.

For every Jal-SAR event (headline flood map, 10 m) and every census village touching it:
  people in the village     WorldPop 2020, corrected 100 m file (POP_WorldPop_2020_corridor_100m_sum.tif),
                            the part of the village inside the corridor grid
  people in the flood       WorldPop people on the village's flooded 10 m cells
                            (each 100 m cell's people spread evenly over its 100 cells)
  vulnerability class       M8 main index (vulnerability.csv, quintiles 1 lowest to 5 highest)
  exposure class            from the people in the flood (EXPOSURE_BREAKS below)
  risk class                hazard (flooded) x exposure x vulnerability, by the matrix below

Risk is RELATIVE and for THIS CORRIDOR ONLY: the classes rank villages against each other
for one radar map. They are not a probability of loss or damage. UNDRR's terms: risk comes
from hazard, exposure and vulnerability together.

The matrix and the exposure breaks are choices (assumptions), printed in every report:
                     exposure low   moderate   high
  vulnerability 1-2     low          low       medium
  vulnerability 3       low          medium    high
  vulnerability 4-5     medium       high      high
  A village with fewer than 1 person in the flood is "flooded, no one mapped inside".
  A village with no vulnerability class (M8 left it out) is "no vulnerability class".

Writes, per event, 06 Outputs\\jalsar\\<event>\\extras\\:
  village_people.json   per village code: WorldPop people (village, in flood), classes; read by the
                        bulletin and the web map risk layer
  risk_villages.csv     the same as a table, with the Census 2011 figures beside it
and 06 Outputs\\jalsar\\extensions\\m8_vulnerability\\risk_view\\RISK_VIEW.md (read first) and
risk_villages_all_events.csv.

Run:  python m8_risk_view.py
      python m8_risk_view.py --event 2023-08-16_orbit129
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "FYP_code"))
import models_common as mc  # noqa: E402

ROOT = HERE.parents[2]
JAL = ROOT / "06 Outputs" / "jalsar"
POP = ROOT / "06 Outputs" / "jalsar" / "extensions" / "population" / "POP_WorldPop_2020_corridor_100m_sum.tif"
VILLAGES = ROOT / "05 Data" / "boundaries" / "VILLAGES_corridor.gpkg"
VULN = ROOT / "06 Outputs" / "jalsar" / "extensions" / "m8_vulnerability" / "vulnerability.csv"
OUT = ROOT / "06 Outputs" / "jalsar" / "extensions" / "m8_vulnerability" / "risk_view"

EXPOSURE_BREAKS = (1.0, 10.0, 50.0)       # people in the flood: <1 none, 1-10 low, 10-50 moderate, 50+ high
EXPOSURE_NAMES = ("none", "low", "moderate", "high")
VULN_GROUP = {1: "low", 2: "low", 3: "middle", 4: "high", 5: "high"}
MATRIX = {("low", "low"): "low", ("low", "moderate"): "low", ("low", "high"): "medium",
          ("middle", "low"): "low", ("middle", "moderate"): "medium", ("middle", "high"): "high",
          ("high", "low"): "medium", ("high", "moderate"): "high", ("high", "high"): "high"}
RISK_ORDER = ("high", "medium", "low", "flooded, no one mapped inside", "no vulnerability class")
RISK_COLOURS = {"high": "#b91c1c", "medium": "#f59e0b", "low": "#fde68a",
                "flooded, no one mapped inside": "#e5e7eb", "no vulnerability class": "#9ca3af"}


def _int(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return int(round(x)) if np.isfinite(x) else None


def exposure_class(people):
    return EXPOSURE_NAMES[int(np.searchsorted(EXPOSURE_BREAKS, people, side="right"))]


def risk_class(people, vclass):
    e = exposure_class(people)
    if e == "none":
        return e, "flooded, no one mapped inside"
    if vclass not in VULN_GROUP:
        return e, "no vulnerability class"
    return e, MATRIX[(VULN_GROUP[vclass], e)]


def people_on_grid(prof):
    """WorldPop people per 10 m cell of the flood grid (each 100 m cell's people spread over its area)."""
    import rasterio
    from rasterio.warp import reproject, Resampling
    with rasterio.open(POP) as src:
        pop = src.read(1).astype("float64")
        nd, pprof = src.nodata, src.profile
    pop[(pop < 0) | ~np.isfinite(pop) | ((pop == nd) if nd is not None else False)] = 0
    area_src = abs(pprof["transform"].a * pprof["transform"].e)
    dens = np.zeros((prof["height"], prof["width"]), "float64")
    reproject(pop / area_src, dens, src_transform=pprof["transform"], src_crs=pprof["crs"],
              dst_transform=prof["transform"], dst_crs=prof["crs"], resampling=Resampling.nearest)
    return dens * abs(prof["transform"].a * prof["transform"].e), float(pop.sum())


def load_vuln():
    if not VULN.exists():
        return {}
    with open(VULN, newline="", encoding="utf-8") as fh:
        return {r["vlcode"]: r for r in csv.DictReader(fh)}


def event_risk(ev_dir, vil, vuln):
    import rasterio
    from rasterio.features import rasterize
    with rasterio.open(ev_dir / "flood_headline.tif") as src:
        flood = src.read(1)
        prof = src.profile
    v = vil.to_crs(prof["crs"])
    ids = rasterize(((g, i + 1) for i, g in enumerate(v.geometry)), out_shape=flood.shape,
                    transform=prof["transform"], fill=0, dtype="int32", all_touched=False)
    ppl, pop_total = people_on_grid(prof)
    n = len(v) + 1
    f = flood == 1
    in_village = np.bincount(ids.ravel(), weights=ppl.ravel(), minlength=n)
    in_flood = np.bincount(ids[f].ravel(), weights=ppl[f].ravel(), minlength=n)
    cells_flood = np.bincount(ids[f].ravel(), minlength=n)
    km2 = abs(prof["transform"].a * prof["transform"].e) / 1e6
    rows = []
    for i, r in v.iterrows():
        k = i + 1
        if cells_flood[k] == 0:
            continue
        code = str(r.get("vlcode", ""))
        vr = vuln.get(code) or {}
        try:
            vclass = int(float(vr.get("class") or 0))
        except ValueError:
            vclass = 0
        e, risk = risk_class(in_flood[k], vclass)
        rows.append({"vlcode": code, "village": str(r.get("village", "")), "subdistrict": str(r.get("subdistric", "")),
                     "district": str(r.get("district", "")), "flood_km2": round(float(cells_flood[k] * km2), 4),
                     "worldpop_village": int(round(in_village[k])), "worldpop_in_flood": round(float(in_flood[k]), 1),
                     "vulnerability_class": vclass or None,
                     "vulnerability_name": vr.get("class_name") or "not in the M8 table",
                     "exposure_class": e, "risk_class": risk,
                     "population_2011": _int(r.get("total_population_village")),
                     "households_2011": _int(r.get("total_households")),
                     "lon": float(r.geometry.representative_point().x), "lat": float(r.geometry.representative_point().y)})
    if rows:
        from rasterio.warp import transform as wt
        xs, ys = wt(prof["crs"], "EPSG:4326", [x["lon"] for x in rows], [x["lat"] for x in rows])
        for x, a, b in zip(rows, xs, ys):
            x["lon"], x["lat"] = round(a, 5), round(b, 5)
    rows.sort(key=lambda x: (RISK_ORDER.index(x["risk_class"]), -x["worldpop_in_flood"]))
    tot_in = float(ppl[f].sum())
    summ = {"event": ev_dir.name, "villages": len(rows), "worldpop_in_flood_all": round(tot_in),
            "worldpop_in_flood_villages": round(sum(x["worldpop_in_flood"] for x in rows)),
            "worldpop_villages": int(sum(x["worldpop_village"] for x in rows)),
            "by_risk": {c: sum(x["risk_class"] == c for x in rows) for c in RISK_ORDER},
            "worldpop_file": POP.name, "worldpop_corridor_total": round(pop_total)}
    d = ev_dir / "extras"
    d.mkdir(exist_ok=True)
    mc.save_json(d / "village_people.json", {
        "summary": summ, "matrix": {f"{a}|{b}": c for (a, b), c in MATRIX.items()},
        "exposure_breaks": EXPOSURE_BREAKS, "colours": RISK_COLOURS,
        "note": "Relative risk for this corridor and this radar map only; not a probability of loss.",
        "villages": {x["vlcode"]: x for x in rows}})
    keys = list(rows[0].keys()) if rows else ["vlcode"]
    with open(d / "risk_villages.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    return summ, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", default="all")
    args = ap.parse_args()
    t0 = time.time()
    for p, what in ((POP, "the corrected WorldPop file (m8_worldpop_agesex_gee.py)"), (VILLAGES, "the village boundaries"),
                    (VULN, "the M8 vulnerability table (m8_vulnerability.py)")):
        if not p.exists():
            print(f"PROBLEM: {p} is missing ({what})")
            return 1
    import geopandas as gpd
    vil = gpd.read_file(VILLAGES)
    vil = vil[vil.geometry.notna() & ~vil.geometry.is_empty].reset_index(drop=True)
    vuln = load_vuln()
    evs = sorted(p for p in JAL.iterdir() if p.is_dir() and (p / "flood_headline.tif").exists()
                 and (args.event == "all" or p.name == args.event))
    if not evs:
        print("PROBLEM: no Jal-SAR event with a headline map")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    allrows, sums = [], []
    for e in evs:
        s, rows = event_risk(e, vil, vuln)
        sums.append(s)
        allrows += [{"event": e.name, **x} for x in rows]
        print(f"  {e.name}: {s['villages']} villages; WorldPop in the flood {s['worldpop_in_flood_all']:,}; "
              + ", ".join(f"{k} {v}" for k, v in s["by_risk"].items() if v))
    with open(OUT / "risk_villages_all_events.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(allrows[0].keys()) if allrows else ["event"])
        w.writeheader(); w.writerows(allrows)
    md = ["# Village risk view (relative, this corridor only)", "",
          "Risk here combines three things for each village touching a radar flood map: it is flooded (hazard), "
          "how many people WorldPop puts on its flooded cells (exposure), and its M8 vulnerability class. It ranks "
          "villages against each other for one radar map. It is not a probability of loss or damage, and "
          "'inside the mapped flood' does not mean anyone was harmed.", "",
          f"People: WorldPop 2020, corrected 100 m file ({POP.name}); each 100 m cell's people are spread evenly "
          "over its 10 m cells. Vulnerability: the M8 main index (quintiles; 1 lowest, 5 highest).", "",
          "## The matrix (a choice, check it)", "",
          f"Exposure classes from the people in the flood: under {EXPOSURE_BREAKS[0]:g} none; "
          f"{EXPOSURE_BREAKS[0]:g} to {EXPOSURE_BREAKS[1]:g} low; {EXPOSURE_BREAKS[1]:g} to {EXPOSURE_BREAKS[2]:g} "
          f"moderate; {EXPOSURE_BREAKS[2]:g} or more high.", "",
          "| Vulnerability \\ exposure | low | moderate | high |", "|---|---|---|---|",
          "| classes 1-2 (lowest, low) | low | low | medium |",
          "| class 3 (middle) | low | medium | high |",
          "| classes 4-5 (high, highest) | medium | high | high |", "",
          "## Per event", "",
          "| Event | Villages touching the flood | WorldPop in the flood (whole map) | of it in villages | "
          + " | ".join(RISK_ORDER) + " |", "|---|---|---|---|" + "---|" * len(RISK_ORDER)]
    for s in sums:
        md.append(f"| {s['event']} | {s['villages']} | {s['worldpop_in_flood_all']:,} | {s['worldpop_in_flood_villages']:,} | "
                  + " | ".join(str(s["by_risk"][c]) for c in RISK_ORDER) + " |")
    md += ["", "WorldPop in the flood can differ slightly from M7 (which counts each 100 m cell by the flooded "
           "share of it); both use the same corrected file.", "", "## High-risk villages, all events", "",
           "| Event | Village | Sub-district | People in the flood (WorldPop) | People in the village (WorldPop) | "
           "Vulnerability | Flood km2 |", "|---|---|---|---|---|---|---|"]
    for x in [r for r in allrows if r["risk_class"] == "high"][:60]:
        md.append(f"| {x['event']} | {x['village']} | {x['subdistrict']} | {x['worldpop_in_flood']:,.0f} | "
                  f"{x['worldpop_village']:,} | {x['vulnerability_name']} | {x['flood_km2']:.3f} |")
    md += ["", "Limits: WorldPop is modelled, not counted, and its age and sex shares copy a few administrative "
           "areas (see M8). The vulnerability index is relative and partly from Census 2011. The flood map is one "
           "radar pass. The matrix and breaks are choices; changing them changes the classes, not the numbers.",
           "", "Per event: 06 Outputs\\jalsar\\<event>\\extras\\risk_villages.csv and village_people.json.", ""]
    (OUT / "RISK_VIEW.md").write_text("\n".join(md), encoding="utf-8")
    print(f"  wrote {OUT / 'RISK_VIEW.md'} ({(time.time() - t0) / 60:.1f} min)")
    print("PASS  risk view")
    return 0


if __name__ == "__main__":
    sys.exit(main())
