"""Jal-SAR extensions, items M9 and M10: download the GloFAS river discharge data (version 2, 21 Oct 2026).

Version 1 asked for too much at once: the data store refused every request with
"cost limits exceeded" (or "not a valid combination" for the settings it does
not hold). Version 2 asks in small pieces and splits a piece again when it is
still too large. Only settings the data store accepted on the first night are
used:
  forecasts   system_version operational (fallback version_3_1), hydrological
              model lisflood, river_discharge_in_the_last_24_hours
  reanalysis  system_version version_4_0 and version_5_0, lisflood, product
              consolidated (fallback intermediate, for the latest months),
              average_river_discharge_in_the_last_24_hours, timespan time_mean

Two sets, for a small box around the corridor (north 30.6, west 78.1, south 29.9, east 79.3):
1. Forecasts (cems-glofas-forecast), for the forecast watch (M9) hindcast: every
   forecast issued from 16 days before each of the six Jal-SAR radar dates up
   to that date, lead times 1 to 15 days (24 to 360 h). One file per issue day
   and product (control forecast; the 50 ensemble members), split by lead time
   if the data store wants it smaller.
2. Reanalysis (cems-glofas-historical), daily, 1979 to now, for the flood
   thresholds and the stage relation (M10). One file per year, split by month
   if needed. Version 4.0 first (the one the thresholds use), then 5.0.

Uses the key in C:\\Users\\<you>\\.cdsapirc (saved by M0 SAVE KEYS.bat). Each request waits in the
data store's queue; hours in total is normal. Files already downloaded are
skipped, so a re-run carries on where it stopped.

Outputs in 06 Outputs\\jalsar\\extensions\\glofas\\:
  forecast\\glofas_fc_<issue day>_<ctrl|ens>[_L<from>-<to>].nc
  historical\\glofas_hist_<version>_<year>[_<month>].nc
  glofas_manifest.csv, glofas_log.txt, allowed_<dataset>.json

Run:  python m9_glofas_download.py                     (both sets)
      python m9_glofas_download.py --only historical   (or forecast)
      python m9_glofas_download.py --test              (one small forecast request)
      python m9_glofas_download.py --refresh-year 2026 (that year's reanalysis again, see below)

--refresh-year (choice 3a, 25 Oct 2026): a year file that was downloaded while the
year was still running stops at the last month the store held then (on 25 Oct 2026: version
4.0 to 30 June, version 5.0 to 31 July), and a normal run skips it as finished. This asks,
for each version, first for the "intermediate" product of that year (the newest months, which
the consolidated product does not hold yet), then for the consolidated year again. The old
file is moved to historical\\superseded_<date time>\\ first and put back if the new request
fails, is stopped, or brings a smaller file. Pieces the store does not hold are logged.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "06 Outputs" / "jalsar" / "extensions" / "glofas"
AREA = [30.6, 78.1, 29.9, 79.3]                      # N, W, S, E
API = "https://ewds.climate.copernicus.eu/api/retrieve/v1/processes/"
PASSES = ["2021-10-21", "2023-07-19", "2023-08-16", "2025-08-29", "2026-08-03", "2026-08-07"]
DAYS_BEFORE = 16
LEADS = [str(h) for h in range(24, 361, 24)]
PRODUCTS = {"control_forecast": "ctrl", "ensemble_perturbed_forecasts": "ens"}
HIST_VERSIONS = ["version_4_0", "version_5_0"]
FIRST_YEAR = 1979


class TooLarge(Exception):
    pass


class Invalid(Exception):
    pass


def log(msg):
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "glofas_log.txt", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def allowed(dataset):
    """The data store's allowed values per input, or {} if it cannot be read."""
    import requests
    try:
        r = requests.get(API + dataset, timeout=60)
        r.raise_for_status()
        inputs = r.json().get("inputs", {})
        (OUT / f"allowed_{dataset}.json").write_text(json.dumps(inputs, indent=1), encoding="utf-8")
        out = {}
        for k, v in inputs.items():
            sch = v.get("schema", {})
            en = sch.get("enum") or sch.get("items", {}).get("enum")
            if en:
                out[k] = en
        return out
    except Exception as exc:
        log(f"  could not read the allowed values for {dataset} ({exc}); using the built-in names")
        return {}


def manifest_row(row):
    f = OUT / "glofas_manifest.csv"
    new = not f.exists()
    with open(f, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["time", "dataset", "file", "result", "bytes", "sha256", "request"])
        if new:
            w.writeheader()
        w.writerow(row)


def fetch(client, dataset, target, req):
    """One request. Raises TooLarge or Invalid for the two refusals that mean 'change the request'."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    short = {k: (v if len(v) < 6 else f"{v[0]}..{v[-1]} ({len(v)})") if isinstance(v, list) else v
             for k, v in req.items() if k != "area"}
    log(f"  {target.name}: asking {json.dumps(short)}")
    t0 = time.time()
    row = {"time": f"{dt.datetime.now():%Y-%m-%d %H:%M}", "dataset": dataset,
           "file": str(target.relative_to(OUT)), "request": json.dumps(req)}
    try:
        client.retrieve(dataset, req, str(part))
    except Exception as exc:
        msg = str(exc).replace("\n", " ")
        manifest_row({**row, "result": "failed: " + msg[:200], "bytes": "", "sha256": ""})
        if part.exists():
            part.unlink()
        low = msg.lower()
        if "too large" in low or "cost limits" in low:
            log(f"  {target.name}: too large for one request")
            raise TooLarge(msg)
        if "valid combination" in low or "invalid request" in low or "no data" in low or "marsnodata" in low:
            log(f"  {target.name}: not held by the data store with these settings")
            raise Invalid(msg)
        log(f"  {target.name}: failed: {msg[:300]}")
        raise
    part.replace(target)
    size = target.stat().st_size
    log(f"  {target.name}: OK, {size / 1e6:.2f} MB in {(time.time() - t0) / 60:.1f} min")
    manifest_row({**row, "result": "ok", "bytes": size, "sha256": sha256(target)})


def done(target):
    return target.exists() and target.stat().st_size > 0


def get_split(client, dataset, stem, req, split_keys, folder):
    """Ask for req; if too large, split the first splittable key in halves and ask for each half.
    Returns (files, ok). The file names record the split, so a re-run finds them again."""
    target = folder / f"{stem}.nc"
    if done(target):
        return [target], True
    # pieces from an earlier run
    prev = sorted(folder.glob(f"{stem}__*.nc"))
    marker = folder / f"{stem}.split"
    if marker.exists():
        key = marker.read_text(encoding="utf-8").strip()
        return _halves(client, dataset, stem, req, split_keys, folder, key)
    try:
        fetch(client, dataset, target, req)
        return [target], True
    except TooLarge:
        for key in split_keys:
            if len(req.get(key, [])) > 1:
                marker.write_text(key, encoding="utf-8")
                return _halves(client, dataset, stem, req, split_keys, folder, key)
        log(f"  {stem}: still too large and nothing left to split")
        return prev, False


def _halves(client, dataset, stem, req, split_keys, folder, key):
    vals = req[key]
    mid = len(vals) // 2
    files, ok = [], True
    for half in (vals[:mid], vals[mid:]):
        sub = {**req, key: half}
        tag = f"{key[:1]}{half[0]}-{half[-1]}" if len(half) > 1 else f"{key[:1]}{half[0]}"
        f, o = get_split(client, dataset, f"{stem}__{tag}", sub, split_keys, folder)
        files += f
        ok = ok and o
    return files, ok


def forecast_days():
    days = set()
    for p in PASSES:
        d1 = dt.date.fromisoformat(p)
        for n in range(DAYS_BEFORE + 1):
            days.add(d1 - dt.timedelta(days=n))
    return sorted(days)


def forecast_request(sv, product, d, leads, var):
    return {"system_version": [sv], "hydrological_model": ["lisflood"], "product_type": [product],
            "variable": [var], "year": [str(d.year)], "month": [f"{d.month:02d}"], "day": [f"{d.day:02d}"],
            "leadtime_hour": list(leads), "area": AREA, "data_format": "netcdf", "download_format": "unarchived"}


def get_forecast(client, d, product, leads=LEADS, var="river_discharge_in_the_last_24_hours",
                 folder=None, stem=None):
    """One issue day and product; operational first, version_3_1 if operational does not hold it."""
    folder = folder or OUT / "forecast"
    stem = stem or f"glofas_fc_{d.isoformat()}_{PRODUCTS[product]}"
    for sv in ("operational", "version_3_1"):
        try:
            files, ok = get_split(client, "cems-glofas-forecast", stem + ("" if sv == "operational" else "_v31"),
                                  forecast_request(sv, product, d, leads, var), ["leadtime_hour"], folder)
            if ok:
                return files, sv
        except Invalid:
            continue
    return [], None


def hist_request(sv, prod, y, months):
    return {"system_version": [sv], "hydrological_model": ["lisflood"], "product_type": [prod],
            "variable": ["average_river_discharge_in_the_last_24_hours"], "timespan": ["time_mean"],
            "year": [str(y)], "month": months, "day": [f"{d:02d}" for d in range(1, 32)],
            "area": AREA, "data_format": "netcdf", "download_format": "unarchived"}


def refresh_one(client, stem, req, split_keys):
    """Ask for <stem>.nc again. The old file (and any pieces of it) is moved aside first and put back if the
    new one does not arrive, is stopped, or is smaller. Returns 'new', 'kept old' or 'not held'."""
    hdir = OUT / "historical"
    old = [p for p in hdir.glob(f"{stem}.nc")] + list(hdir.glob(f"{stem}__*.nc")) + list(hdir.glob(f"{stem}.split"))
    aside = hdir / f"superseded_{dt.datetime.now():%Y-%m-%d_%H%M}"
    moved = []
    for p in old:
        aside.mkdir(exist_ok=True)
        p.replace(aside / p.name)
        moved.append(aside / p.name)
    old_bytes = sum(p.stat().st_size for p in moved if p.suffix == ".nc")

    def put_back(why):
        for p in hdir.glob(f"{stem}.nc"):
            p.unlink()
        for p in hdir.glob(f"{stem}__*.nc"):
            p.unlink()
        for p in moved:
            p.replace(hdir / p.name)
        log(f"  {stem}: old file kept ({why})")

    result = "kept old"
    try:
        files, ok = get_split(client, "cems-glofas-historical", stem, req, split_keys, hdir)
        new_bytes = sum(f.stat().st_size for f in files if f.exists())
        if not ok or not files:
            if moved:
                put_back("the new request did not complete")
        elif moved and new_bytes < old_bytes:
            put_back(f"the new file is smaller, {new_bytes / 1e6:.3f} MB against {old_bytes / 1e6:.3f} MB")
        else:
            result = "new"
            log(f"  {stem}: new file {new_bytes / 1e6:.3f} MB" + (f" (old {old_bytes / 1e6:.3f} MB, kept in "
                                                                  f"{aside.name})" if moved else ""))
    except Invalid:
        if moved:
            put_back("the store does not hold it with these settings")
        result = "not held"
    except KeyboardInterrupt:                      # Ctrl+C: never leave the year missing
        if moved:
            put_back("stopped with Ctrl+C")
        if aside.exists() and not any(aside.iterdir()):
            aside.rmdir()
        raise
    except Exception as exc:
        log(f"  {stem}: failed: {str(exc)[:200]}")
        if moved:
            put_back("the request failed")
        result = "failed"
    if aside.exists() and not any(aside.iterdir()):
        aside.rmdir()
    return result


def refresh_year(client, y):
    a = allowed("cems-glofas-historical")
    versions = [v for v in HIST_VERSIONS if not a.get("system_version") or v in a["system_version"]]
    months = [f"{m:02d}" for m in range(1, 13) if dt.date(y, m, 1) <= dt.date.today()]
    log(f"refresh {y}: versions {versions}; intermediate first, then consolidated again")
    report = []
    for sv in versions:
        r1 = refresh_one(client, f"glofas_hist_{sv}_{y}_intermediate",
                         hist_request(sv, "intermediate", y, months), ["month", "day"])
        if r1 == "not held":                       # the store may hold only the newest months: one by one
            got = []
            for m in months:
                rm = refresh_one(client, f"glofas_hist_{sv}_{y}_{m}_inte",
                                 hist_request(sv, "intermediate", y, [m]), ["day"])
                if rm == "new":
                    got.append(m)
            r1 = f"months {','.join(got)}" if got else "not held"
        r2 = refresh_one(client, f"glofas_hist_{sv}_{y}", hist_request(sv, "consolidated", y, months),
                         ["month", "day"])
        report.append(f"{sv}: intermediate {r1}; consolidated {r2}")
        log(f"refresh {y} {sv}: intermediate {r1}; consolidated {r2}")
    log(f"refresh {y} done: " + " | ".join(report))
    if not any(("intermediate new" in r) or ("intermediate months" in r) or ("consolidated new" in r) for r in report):
        log(f"refresh {y}: nothing new from the store (the old files are all still in place)")
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", action="store_true", help="one small forecast request only")
    ap.add_argument("--only", choices=("forecast", "historical"))
    ap.add_argument("--refresh-year", type=int, help="ask for this year's reanalysis again (see the top of the file)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        import cdsapi
    except ImportError:
        sys.exit("cdsapi is not installed in this environment (run M0 INSTALL PACKAGES.bat).")
    if not (Path.home() / ".cdsapirc").exists():
        sys.exit(f"No key file at {Path.home() / '.cdsapirc'}. Run M0 SAVE KEYS.bat first.")
    log(f"m9_glofas_download.py (version 2) start; area N,W,S,E = {AREA}")
    client = cdsapi.Client(quiet=True, progress=False)
    failed = []

    if args.test:
        d = forecast_days()[-1]
        files, sv = get_forecast(client, d, "control_forecast", leads=LEADS[:2],
                                 folder=OUT / "test", stem=f"glofas_test_{d.isoformat()}")
        log(f"test: {'PASS' if files else 'FAIL'} ({sv})")
        return 0 if files else 1

    if args.refresh_year:
        return refresh_year(client, args.refresh_year)

    if args.only in (None, "forecast"):
        a = allowed("cems-glofas-forecast")
        var = "river_discharge_in_the_last_24_hours"
        if a.get("variable") and var not in a["variable"]:
            var = [v for v in a["variable"] if "discharge" in v][0]
        days = forecast_days()
        log(f"forecasts: {len(days)} issue days x 2 products; variable {var}")
        for i, d in enumerate(days, 1):
            for product in PRODUCTS:
                try:
                    files, sv = get_forecast(client, d, product, var=var)
                except Exception as exc:
                    files, sv = [], None
                    log(f"  {d} {product}: {str(exc)[:200]}")
                if not files:
                    failed.append(f"forecast {d} {PRODUCTS[product]}")
            if i % 10 == 0:
                log(f"forecasts: {i} of {len(days)} issue days done")

    if args.only in (None, "historical"):
        a = allowed("cems-glofas-historical")
        versions = [v for v in HIST_VERSIONS if not a.get("system_version") or v in a["system_version"]]
        years = [y for y in range(FIRST_YEAR, dt.date.today().year + 1)
                 if not a.get("year") or str(y) in a["year"]]
        log(f"reanalysis: versions {versions}, years {years[0]} to {years[-1]}, one file per year")
        for sv in versions:
            for y in years:
                months = [f"{m:02d}" for m in range(1, 13) if dt.date(y, m, 1) <= dt.date.today()]
                got = False
                for prod in ("consolidated", "intermediate"):
                    req = {"system_version": [sv], "hydrological_model": ["lisflood"], "product_type": [prod],
                           "variable": ["average_river_discharge_in_the_last_24_hours"], "timespan": ["time_mean"],
                           "year": [str(y)], "month": months, "day": [f"{d:02d}" for d in range(1, 32)],
                           "area": AREA, "data_format": "netcdf", "download_format": "unarchived"}
                    stem = f"glofas_hist_{sv}_{y}" + ("" if prod == "consolidated" else "_intermediate")
                    try:
                        files, ok = get_split(client, "cems-glofas-historical", stem, req, ["month", "day"],
                                              OUT / "historical")
                        if ok:
                            got = True
                            break
                    except Invalid:
                        continue
                    except Exception as exc:
                        log(f"  {stem}: {str(exc)[:200]}")
                if not got:                 # a year the store holds only in part: month by month
                    miss = []
                    for m in months:
                        ok_m = False
                        for prod in ("consolidated", "intermediate"):
                            req = {"system_version": [sv], "hydrological_model": ["lisflood"], "product_type": [prod],
                                   "variable": ["average_river_discharge_in_the_last_24_hours"],
                                   "timespan": ["time_mean"], "year": [str(y)], "month": [m],
                                   "day": [f"{d:02d}" for d in range(1, 32)], "area": AREA,
                                   "data_format": "netcdf", "download_format": "unarchived"}
                            try:
                                files, ok_m = get_split(client, "cems-glofas-historical",
                                                        f"glofas_hist_{sv}_{y}_{m}_{prod[:4]}", req, ["day"],
                                                        OUT / "historical")
                                if ok_m:
                                    break
                            except Invalid:
                                continue
                            except Exception as exc:
                                log(f"  {sv} {y}-{m}: {str(exc)[:200]}")
                        if not ok_m:
                            miss.append(m)
                    if miss:
                        failed.append(f"reanalysis {sv} {y} months {','.join(miss)}")

    if failed:
        log(f"FINISHED WITH GAPS: {len(failed)} piece(s) missing: {', '.join(failed[:40])}"
            + (" ..." if len(failed) > 40 else ""))
        log("Run again later; finished files are skipped. Pieces the data store does not hold "
            "(for example the newest reanalysis months) stay missing; that is expected.")
        return 1
    log("PASS  M9 GloFAS downloads complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
