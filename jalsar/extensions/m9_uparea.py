"""Jal-SAR extensions: the GloFAS upstream area file, downloaded once and checked.

With this file in the glofas folder, each gauge gets the GloFAS river cell whose
upstream area is closest to the gauge's catchment area (m9_glofas_read.match_cells),
instead of the cell with the largest mean flow. That also gives Rudraprayag (A),
above the Mandakini, and Rudraprayag (conf), below it, their own cells (until now
they shared 30.275 N, 78.925 E).

Source: CEMS Auxiliary Data page (ECMWF Confluence, space CEMS, page "Auxiliary Data"),
file uparea_glofas_v4_0.nc, about 87 MB, upstream area in m2 on the GloFAS 4.0 grid
(0.05 degree, 7200 x 3000), for forecasts from 26 July 2023. The page lists no
separate GloFAS 5.0 file; 5.0 runs on the same 0.05 degree grid, so the 4.0 file is
used for both (an assumption, stated in the report).

What it does:
  1. Downloads the file into 06 Outputs\\jalsar\\extensions\\glofas\\ (skipped if it is already
     there and opens). If the download is refused, it says how to get it by hand.
  2. For each gauge, lists the 3 x 3 GloFAS cells around it with their upstream area,
     the gauge's catchment area, and which cell is now chosen; and which cell the old
     rule (largest mean flow) chose.
  3. Writes 06 Outputs\\jalsar\\extensions\\glofas\\UPAREA_CHECK.md.

Run:  python m9_uparea.py
      python m9_uparea.py --check-only      (file already downloaded by hand)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m9_glofas_read as gr  # noqa: E402

URL = ("https://confluence.ecmwf.int/download/attachments/242067380/uparea_glofas_v4_0.nc"
       "?version=2&modificationDate=1668604690076&api=v2")
NAME = "uparea_glofas_v4_0.nc"
PAGE = "https://confluence.ecmwf.int/spaces/CEMS/pages/242067380/Auxiliary+Data"


def download(dest):
    import requests
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, 4):
        try:
            with requests.get(URL, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                got, t0 = 0, time.time()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
                        got += len(chunk)
                        if total:
                            print(f"\r  {got / 1e6:.0f} of {total / 1e6:.0f} MB", end="", flush=True)
                print(f"\n  downloaded in {(time.time() - t0) / 60:.1f} min")
            if tmp.stat().st_size < 10e6:
                head = tmp.read_bytes()[:200]
                raise RuntimeError(f"only {tmp.stat().st_size} bytes arrived (starts {head[:60]!r}); "
                                   "the page may want a login or a browser")
            tmp.replace(dest)
            return True
        except Exception as exc:                      # noqa: BLE001
            print(f"  attempt {attempt} failed: {exc}")
            time.sleep(5 * attempt)
    if tmp.exists():
        tmp.unlink()
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    dest = gr.GLOFAS / NAME
    gr.GLOFAS.mkdir(parents=True, exist_ok=True)
    if not dest.exists() and not args.check_only:
        print(f"Downloading {NAME} (about 87 MB) ...")
        if not download(dest):
            print("FAIL  The download did not work. Get it by hand instead:\n"
                  f"  1. Open {PAGE}\n"
                  f"  2. Find '{NAME}' in the GloFAS v4.0 table and download it.\n"
                  f"  3. Put it in {gr.GLOFAS}\n"
                  "  4. Run this again with --check-only.")
            return 1
    if not dest.exists():
        print(f"FAIL  {dest} is not there yet.")
        return 1
    up = gr.load_uparea()
    if up is None:
        print("FAIL  The file is there but could not be read.")
        return 1
    hfiles, hver = gr.historical_files("version_4_0")
    hist = gr.open_historical(hfiles) if hfiles else None
    if hist is None:
        print("FAIL  No GloFAS reanalysis files to take the grid from.")
        return 1
    sts = gr.stations()
    new = gr.match_cells(hist, sts, uparea=up)
    old = gr.match_cells(hist, sts, uparea=None)
    lat, lon = hist["lat"], hist["lon"]
    md = ["# GloFAS upstream area at the gauges", "",
          f"File: {NAME} (CEMS Auxiliary Data page, {PAGE}); upstream area converted from m2 to km2. "
          "GloFAS 4.0 grid, used for 5.0 as well (the page lists no 5.0 file; same 0.05 degree grid; an assumption).", "",
          "| Gauge | Catchment km2 (station list) | Cell chosen now | Its upstream area km2 | Off by | Cell before (largest mean flow) |",
          "|---|---|---|---|---|---|"]
    detail = []
    for s in sts:
        n, o = new[s["name"]], old[s["name"]]
        ua = float(up(np.array([n["cell_lat"]]), np.array([n["cell_lon"]]))[0, 0])
        off = (ua - s["catchment_km2"]) / s["catchment_km2"] * 100 if np.isfinite(s["catchment_km2"]) else float("nan")
        md.append(f"| {s['name']} | {s['catchment_km2']:.0f} | {n['cell_lat']:.3f} N, {n['cell_lon']:.3f} E | {ua:,.0f} | "
                  f"{off:+.0f} % | {o['cell_lat']:.3f} N, {o['cell_lon']:.3f} E |")
        i0, j0 = int(np.abs(lat - s["lat"]).argmin()), int(np.abs(lon - s["lon"]).argmin())
        ii = list(range(max(0, i0 - 1), min(len(lat), i0 + 2)))
        jj = list(range(max(0, j0 - 1), min(len(lon), j0 + 2)))
        grid = up(lat[ii], lon[jj])
        detail += ["", f"**{s['name']}** ({s['lat']:.4f} N, {s['lon']:.4f} E): upstream area km2 of the cells around it "
                   "(rows north to south as in the file, columns west to east; * chosen)", "",
                   "| | " + " | ".join(f"{lon[j]:.3f} E" for j in jj) + " |", "|---|" + "---|" * len(jj)]
        for a, i in enumerate(ii):
            detail.append(f"| {lat[i]:.3f} N | " + " | ".join(
                f"{grid[a, b]:,.0f}{' *' if (i, j) == (n['i'], n['j']) else ''}" for b, j in enumerate(jj)) + " |")
    shared = len({(v["i"], v["j"]) for k, v in new.items() if k.startswith("Rudraprayag")}) == 1
    md += ["", ("Rudraprayag (A) and (conf) still share one cell: their catchments differ by the Mandakini, "
                "which the 0.05 degree grid may not separate here. Say so in the report."
                if shared else "Rudraprayag (A) and (conf) now have their own cells."), ""] + detail + [
           "", "Next: run M9 M10 FORECAST.bat again. Both M9 and M10 pick the cells up by themselves.", ""]
    (gr.GLOFAS / "UPAREA_CHECK.md").write_text("\n".join(md), encoding="utf-8")
    for s in sts:
        n = new[s["name"]]
        print(f"  {s['name']}: {n['cell_lat']:.3f} N, {n['cell_lon']:.3f} E ({n['rule']})")
    print(f"  wrote {gr.GLOFAS / 'UPAREA_CHECK.md'}")
    print("PASS  upstream area")
    return 0


if __name__ == "__main__":
    sys.exit(main())
