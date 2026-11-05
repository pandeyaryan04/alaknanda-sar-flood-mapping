"""Smoke test, step 5: which Sentinel-1 passes could be the live Jal-SAR demo date.

Read-only. It downloads no radar image and changes no result. It:
  1. connects to Earth Engine (the same project and filters as the Jal-SAR runner),
  2. lists every Sentinel-1 pass over the corridor from --start (default 1 June 2026) to today,
     with the share of the corridor it covers,
  3. for each pass day, reads the GloFAS 4.0 reanalysis flow at the two reach gauges
     (Rudraprayag (conf) for the gorge, Rishikesh for the valley) and divides it by that gauge's
     2-year flow from m9_forecast\\thresholds.csv (QREL, as in M13). A higher QREL means more
     water in the river that day, so more flood for the demo to show,
  4. marks the dates already run (a folder in 06 Outputs\\jalsar) and the orbits whose terrain
     strata already exist (63 descending, 129 ascending; any other orbit adds a strata build
     to the live run).

Output: the table on screen and 06 Outputs\\jalsar\\extensions\\logs\\smoke\\demo_dates.csv.

Run:  python smoke_demo_dates.py
      python smoke_demo_dates.py --start 2026-07-01
"""
import argparse
import csv
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

GAUGES = {"gorge": "Rudraprayag (conf)", "valley": "Rishikesh"}
VERSION = "version_4_0"            # the reanalysis that runs to October 2026 (5.0 ends 31 July 2026)
STRATA_READY = {(63, "DESCENDING"), (129, "ASCENDING")}
MIN_COVERAGE = 99.5                # the runner's default (jalsar.env MIN_COVERAGE)


def rp2_by_gauge(thr_csv):
    out = {}
    with open(thr_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["version"] == VERSION:
                out[r["station"]] = (float(r["rp2"]), float(r["cell_lat"]), float(r["cell_lon"]))
    return out


def flows(glofas_hist, rp2, days):
    """{(day, reach): QREL} from the GloFAS 4.0 reanalysis files of the years asked for."""
    import m9_glofas_read as gr
    years = sorted({d[:4] for d in days})
    files = [f for f in sorted(glofas_hist.glob("*.nc"))
             if VERSION in f.name and any(y in f.name for y in years)]
    if not files:
        print(f"  no GloFAS {VERSION} reanalysis files for {', '.join(years)}: QREL left empty")
        return {}, None
    h = gr.open_historical(files)
    last = str(h["day"][-1])
    out = {}
    for reach, g in GAUGES.items():
        if g not in rp2:
            continue
        q2, la, lo = rp2[g]
        i, j = int(np.abs(h["lat"] - la).argmin()), int(np.abs(h["lon"] - lo).argmin())
        for d in days:
            k = np.where(h["day"] == np.datetime64(d, "D"))[0]
            if len(k):
                out[(d, reach)] = float(h["q"][k[0], i, j]) / q2
    return out, last


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2026-06-01")
    ap.add_argument("--end", default=None, help="default: tomorrow")
    args = ap.parse_args()
    import jalsar_core as gc
    import jalsar_runner as gr_run
    gp = gc.GPaths(None)
    ext = gp.o06 / "jalsar" / "extensions"
    logs = ext / "logs" / "smoke"
    logs.mkdir(parents=True, exist_ok=True)
    end = args.end or (date.today() + timedelta(days=1)).isoformat()

    print(f"  Earth Engine: connecting (project {gr_run.EE_PROJECT}) ...")
    ee = gr_run.ee_init()
    corridor = gr_run.corridor_geometry(ee, gp)
    print(f"  Earth Engine: PASS. Searching passes {args.start} to {end} (a few seconds per pass) ...")
    rows = gr_run.search(ee, corridor, args.start, end)
    print(f"  {len(rows)} passes found")

    done = {p.name for p in gp.gj.iterdir() if p.is_dir()} if gp.gj.exists() else set()
    rp2 = rp2_by_gauge(ext / "m9_forecast" / "thresholds.csv")
    q, last = flows(ext / "glofas" / "historical", rp2, [r["date"] for r in rows])
    if last:
        print(f"  GloFAS {VERSION} reanalysis read; it runs to {last} (later days have no QREL)")

    out = []
    for r in rows:
        eid = gr_run.event_id(r["date"], r["orbit"])
        qg, qv = q.get((r["date"], "gorge")), q.get((r["date"], "valley"))
        out.append({
            "date": r["date"], "orbit": r["orbit"], "pass": r["pass"], "time_utc": r["time_utc"],
            "coverage_pct": r["coverage_pct"],
            "covers_corridor": "yes" if r["coverage_pct"] >= MIN_COVERAGE else "no",
            "strata_ready": "yes" if (r["orbit"], r["pass"]) in STRATA_READY else "no (adds a strata build)",
            "already_run": "yes" if eid in done else "no",
            "qrel_gorge": "" if qg is None else round(qg, 2),
            "qrel_valley": "" if qv is None else round(qv, 2),
        })
    with open(logs / "demo_dates.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]) if out else ["date"])
        w.writeheader()
        w.writerows(out)

    print()
    print("  date        orbit pass        cover %  covers  strata  run before  QREL gorge  QREL valley")
    for o in out:
        print(f"  {o['date']}  {o['orbit']:5d} {o['pass']:11s} {o['coverage_pct']:7.2f}  {o['covers_corridor']:6s}  "
              f"{o['strata_ready'][:3]:6s}  {o['already_run']:10s}  {str(o['qrel_gorge']):10s}  {o['qrel_valley']}")
    good = [o for o in out if o["covers_corridor"] == "yes" and o["strata_ready"] == "yes"
            and o["already_run"] == "no"]
    good.sort(key=lambda o: -(o["qrel_gorge"] or 0))
    print()
    print("  For reference, QREL on the checked floods: October 2021 gorge 0.64, August 2023 gorge 1.05.")
    if good:
        print("  Usable demo dates (full cover, strata ready, not run before), wettest first:")
        for o in good[:5]:
            print(f"    {o['date']}  orbit {o['orbit']}  QREL gorge {o['qrel_gorge']}  valley {o['qrel_valley']}")
    else:
        print("  No pass meets all three conditions; send the table.")
    print(f"  Saved: {logs / 'demo_dates.csv'}")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
