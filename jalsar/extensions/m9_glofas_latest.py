"""Jal-SAR extensions, item M9: download the newest GloFAS forecast for the corridor (for the live watch view).

Asks the data store for the forecast issued today; if that is not out yet,
yesterday; and so on back 6 days. Lead times 1 to 10 days (24 to 240 h), the
control forecast and the 50 ensemble members, the same box and settings as
m9_glofas_download.py (it uses that script's request and logging code, and
changes nothing in it). GloFAS is a daily model: it gives one value per day,
so the watch can say which day, never which hour.

Output: 06 Outputs\\jalsar\\extensions\\glofas\\latest\\glofas_fc_latest_<issue date>_<ctrl|ens>*.nc (and lines
in glofas_manifest.csv and glofas_log.txt). An issue already downloaded is not asked for again.

Run:  python m9_glofas_latest.py
      python m9_glofas_latest.py --date 2026-10-20     (one issue day)
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m9_glofas_download as dl  # noqa: E402

LEADS = [str(h) for h in range(24, 241, 24)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", help="issue day YYYY-MM-DD (default: today, then back up to 6 days)")
    args = ap.parse_args()
    dl.OUT.mkdir(parents=True, exist_ok=True)
    try:
        import cdsapi
    except ImportError:
        sys.exit("cdsapi is not installed in this environment (run M0 INSTALL PACKAGES.bat).")
    if not (Path.home() / ".cdsapirc").exists():
        sys.exit("No key file at ~/.cdsapirc. Run M0 SAVE KEYS.bat first.")
    client = cdsapi.Client(quiet=True, progress=False)
    days = ([dt.date.fromisoformat(args.date)] if args.date
            else [dt.date.today() - dt.timedelta(days=n) for n in range(7)])
    dl.log(f"m9_glofas_latest.py: trying issue days {days[0]} back to {days[-1]}")
    for d in days:
        got = {}
        for product in dl.PRODUCTS:
            try:
                files, sv = dl.get_forecast(client, d, product, leads=LEADS, folder=dl.OUT / "latest",
                                            stem=f"glofas_fc_latest_{d.isoformat()}_{dl.PRODUCTS[product]}")
            except Exception as exc:
                files, sv = [], None
                dl.log(f"  {d} {product}: {str(exc)[:200]}")
            got[product] = files
        if got["control_forecast"] and got["ensemble_perturbed_forecasts"]:
            dl.log(f"PASS  M9 latest forecast: issued {d}")
            print(f"PASS  newest forecast: issued {d} (06 Outputs\\jalsar\\extensions\\glofas\\latest)")
            return 0
    dl.log("PROBLEM: no forecast found for the last 7 issue days (the data store may be behind; try later)")
    print("PROBLEM: no forecast found for the last 7 issue days (try again later)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
