"""Smoke test, step 1: is the laptop ready for the M13 run, the daily forecast and the live demo?

Read-only. It checks, and prints PASS, WARN or FAIL for each line:
  - the Python packages the runs need, and their versions
  - the graphics card (CUDA) and a small sum on it
  - free disk space on the project drive
  - the key files: model weights, terrain strata, M13 inputs, Jal-SAR events, the web map
  - the Earth Engine and GloFAS key files (that they exist; step 5 and 6 test them for real)
  - whether the daily forecast switch is on (Windows Task Scheduler)
Exit code 1 only if a FAIL line was printed.

Run:  python smoke_env.py
"""
import importlib
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                       # ...\Final Year Project
FAILS = []


def say(level, text):
    print(f"  {level:4s}  {text}")
    if level == "FAIL":
        FAILS.append(text)


def packages():
    for name, dist in (("numpy", "numpy"), ("scipy", "scipy"), ("pandas", "pandas"), ("torch", "torch"),
                       ("sklearn", "scikit-learn"), ("rasterio", "rasterio"), ("geopandas", "geopandas"),
                       ("xarray", "xarray"), ("netCDF4", "netCDF4"), ("ee", "earthengine-api"),
                       ("cdsapi", "cdsapi"), ("folium", "folium"), ("jinja2", "jinja2"), ("shap", "shap"),
                       ("terratorch", "terratorch"), ("matplotlib", "matplotlib")):
        try:
            importlib.import_module(name)
            say("PASS", f"{dist} {metadata.version(dist)}")
        except Exception as exc:                 # noqa: BLE001
            say("FAIL", f"{dist} does not import: {str(exc).splitlines()[0][:120]}")


def gpu():
    try:
        import torch
    except Exception:                            # noqa: BLE001
        return
    if not torch.cuda.is_available():
        say("FAIL", "no graphics card seen by torch (CUDA). M13 step 3 and the U-Net maps need it")
        return
    name = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info()
    a = torch.ones((1024, 1024), device="cuda")
    s = float((a @ a).sum().item())
    ok = abs(s - 1024 ** 3) < 1
    say("PASS" if ok else "FAIL", f"graphics card {name}, {free / 1e9:.1f} of {total / 1e9:.1f} GB free, "
        f"test sum {'right' if ok else 'WRONG'}")
    if free < 2.5e9:
        say("WARN", "under 2.5 GB free on the card: close other programs (browsers, games) before M13")


def disk():
    free = shutil.disk_usage(ROOT).free / 1e9
    say("PASS" if free > 10 else "WARN" if free > 3 else "FAIL", f"{free:.0f} GB free on the project drive")


def files():
    fc = ROOT / "04 Code" / "FYP_code" / "data"
    o = ROOT / "06 Outputs" / "jalsar"
    need = {
        "U-Net weights folder": fc / "models" / "unet_hand",
        "Random Forest weights folder": fc / "models" / "rf_hand",
        "TerraMind weights folder": fc / "models" / "tm_s1t_hand",
        "terrain strata, orbit 63": fc / "strata_2021_v3" / "strata_hand_m.tif",
        "terrain strata, orbit 129": fc / "strata_2023_v3" / "strata_hand_m.tif",
        "permanent water": ROOT / "05 Data" / "dem" / "WATER_PERMANENT_JRC50_S2DRY_10m.tif",
        "corridor outline": fc / "corridor.geojson",
        "M13 input: calibration.json": o / "extensions" / "m10_forecast_extent" / "calibration.json",
        "M13 input: thresholds.csv": o / "extensions" / "m9_forecast" / "thresholds.csv",
        "M13 input: hindcast.csv": o / "extensions" / "m9_forecast" / "hindcast.csv",
        "M13 input: forecast_watch.json": o / "extensions" / "m9_forecast" / "forecast_watch.json",
        "Jal-SAR web map": o / "webmap" / "jalsar_map.html",
        "Jal-SAR settings (jalsar.env)": ROOT / "04 Code" / "jalsar" / "jalsar.env",
        "Jal-SAR bulletin template folder": ROOT / "04 Code" / "jalsar" / "templates",
    }
    for label, p in need.items():
        say("PASS" if p.exists() else "FAIL", f"{label}: {'found' if p.exists() else 'MISSING ' + str(p)}")
    events = sorted(p.name for p in o.iterdir() if p.is_dir() and (p / "flood_headline.tif").exists()) \
        if o.exists() else []
    say("PASS" if len(events) >= 6 else "FAIL", f"{len(events)} Jal-SAR events with a flood map "
        f"(M13 needs the 6): {', '.join(events)}")
    m13 = o / "extensions" / "m13_forecast_ml"
    if m13.exists():
        say("WARN", "m13_forecast_ml already has outputs; the 9 to 10 am run carries on from them "
            "(that is fine)")


def keys():
    home = Path.home()
    ee = [home / ".config" / "earthengine" / "credentials",
          home / "AppData" / "Roaming" / "earthengine" / "credentials"]
    say("PASS" if any(p.exists() for p in ee) else "FAIL",
        "Earth Engine key file " + ("found" if any(p.exists() for p in ee)
                                    else "MISSING: run  earthengine authenticate  once"))
    c = home / ".cdsapirc"
    say("PASS" if c.exists() else "FAIL",
        "GloFAS key file .cdsapirc " + ("found" if c.exists() else "MISSING: run M0 SAVE KEYS.bat"))


def daily_switch():
    if sys.platform != "win32":
        return
    r = subprocess.run(["schtasks", "/query", "/tn", "Alaknanda forecast daily"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        say("PASS", "daily forecast switch: ON (FORECAST DAILY OFF.bat turns it off)")
    else:
        say("PASS", "daily forecast switch: OFF (the default; FORECAST DAILY ON.bat turns it on)")


def main():
    print(f"  Python {sys.version.split()[0]}, environment {sys.prefix}")
    for part in (packages, gpu, disk, files, keys, daily_switch):
        try:
            part()
        except Exception as exc:                 # noqa: BLE001
            say("FAIL", f"{part.__name__} check stopped: {exc}")
    print()
    if FAILS:
        print(f"  {len(FAILS)} FAIL line(s) above.")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
