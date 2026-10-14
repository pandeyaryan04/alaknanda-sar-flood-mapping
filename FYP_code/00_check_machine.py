r"""
00_check_machine.py
===================
Run this FIRST. It answers, in one go, every question about the machine that
matters before the models are sized or the setup is finished.

It changes nothing and installs nothing. It reads, prints, and writes one text
file to keep with the project records.

HOW TO RUN
----------
    (fyp) <project folder>\\04 Code\\FYP_code> python 00_check_machine.py

If the environment does not exist yet, plain Python works too - the script
degrades gracefully and tells you what it could not check:

    <project folder>\\04 Code\\FYP_code> python 00_check_machine.py

OUTPUT
------
    machine_report.txt      the report; keep it with the project records

================================================================================
HOW TO RUN THIS  (copy the lines, paste into Anaconda Prompt, press Enter)
================================================================================
    conda activate fyp
    cd /d "<project folder>\04 Code\FYP_code"
    python 00_check_machine.py

Takes about a minute. Installs nothing, changes nothing, downloads nothing except
a small speed-test file. Writes machine_report.txt next to the script.

Re-run it after changing GPU drivers, reinstalling the environment, or moving to
a different machine.
================================================================================
"""

__stage__ = "Machine and environment check"
__reads__ = ["the local machine", "the active conda environment"]
__writes__ = ["machine_report.txt"]

import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime

LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


def head(t):
    say()
    say("=" * 68)
    say(t)
    say("=" * 68)


def run(cmd):
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=45)
        return (out.stdout or "") + (out.stderr or "")
    except Exception as exc:
        return f"<could not run: {exc}>"


# ---------------------------------------------------------------- system
head("SYSTEM")
say(f"checked at      : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
say(f"platform        : {platform.platform()}")
say(f"processor       : {platform.processor()}")
say(f"python          : {sys.version.split()[0]}  ({sys.executable})")
say(f"cpu cores       : {os.cpu_count()}")

if platform.system() == "Windows":
    ed = run('powershell -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).Caption"').strip()
    say(f"windows edition : {ed or '<unknown>'}")
    say("                  (Pro allows incoming Remote Desktop; Home does not.")
    say("                   SSH works on both.)")

try:
    import psutil  # noqa
    say(f"total RAM       : {psutil.virtual_memory().total / 1024**3:.1f} GB")
except ImportError:
    if platform.system() == "Windows":
        r = run('powershell -NoProfile -Command "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"').strip()
        try:
            say(f"total RAM       : {int(r) / 1024**3:.1f} GB")
        except ValueError:
            say("total RAM       : <psutil not installed, could not read>")

# ---------------------------------------------------------------- disk
head("DISK  (need 80 GB free on whichever drive holds the project)")
for part in ["C:\\", "D:\\", "E:\\", "/"]:
    if os.path.exists(part):
        try:
            t, u, f = shutil.disk_usage(part)
            flag = "  <-- ENOUGH" if f / 1024**3 >= 80 else "  <-- tight"
            say(f"  {part:<6} total {t/1024**3:7.1f} GB   free {f/1024**3:7.1f} GB{flag}")
        except Exception as exc:
            say(f"  {part:<6} <could not read: {exc}>")
say()
say(f"  project folder : {os.path.abspath('.')}")

# ---------------------------------------------------------------- gpu
head("GPU  (this decides how the models get sized)")
smi = run("nvidia-smi")
if "<could not run" in smi or "not recognized" in smi.lower() or not smi.strip():
    say("  nvidia-smi did not run.")
    say("  Either the NVIDIA driver is not installed, or it is not on PATH.")
    say("  Try: C:\\Windows\\System32\\nvidia-smi.exe")
else:
    q = run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")
    say(f"  {q.strip()}")
    say()
    say("  full nvidia-smi output:")
    for ln in smi.strip().splitlines():
        say("    " + ln)

# ---------------------------------------------------------------- torch
head("PYTORCH  (only meaningful inside the fyp environment)")
try:
    import torch
    say(f"  torch version   : {torch.__version__}")
    say(f"  CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        say(f"  CUDA version    : {torch.version.cuda}")
        say(f"  device          : {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        vram = props.total_memory / 1024**3
        say(f"  VRAM            : {vram:.2f} GB")
        say(f"  compute capab.  : {props.major}.{props.minor}")
        say()
        if vram < 4.5:
            say("  READ: 4 GB class card. Workable, with these constraints:")
            say("    - U-Net (ResNet-18 encoder), 512px, mixed precision, batch 4")
            say("    - TerraMind-small (20M) full fine-tune, 512px, batch 4 to 8")
            say("    - Prithvi-300M full fine-tune is NOT possible on this card")
            say("    - use gradient accumulation for a larger effective batch")
        else:
            say("  READ: more headroom than assumed. Batch sizes can go up.")
        say()
        say("  live memory test:")
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            say(f"    free now      : {free_b/1024**3:.2f} GB of {total_b/1024**3:.2f} GB")
        except Exception:
            say("    <mem_get_info unavailable on this build>")
        try:
            x = torch.zeros((1, 3, 512, 512), device="cuda")
            y = (x + 1).sum().item()
            say(f"    allocation    : OK (test tensor sum {y:.0f})")
            del x
            torch.cuda.empty_cache()
        except Exception as exc:
            say(f"    allocation    : FAILED - {exc}")
    else:
        say("  Torch is installed but cannot see the GPU.")
        say("  Usually means the CPU-only build was installed. Fix:")
        say("    pip uninstall torch torchvision")
        say("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
except ImportError:
    say("  torch not installed in this environment.")
    say("  This is expected if you have not done runbook step 3 yet.")

# ---------------------------------------------------------------- packages
head("PACKAGES")
for mod, why in [
    ("ee", "Earth Engine"),
    ("geemap", "Earth Engine mapping"),
    ("rasterio", "raster IO"),
    ("geopandas", "vector GIS"),
    ("sklearn", "Random Forest"),
    ("skimage", "Otsu thresholding"),
    ("xgboost", "gradient boosting"),
    ("requests", "downloads"),
    ("pysheds", "HAND and flow routing"),
]:
    try:
        m = __import__(mod)
        v = getattr(m, "__version__", "?")
        say(f"  OK      {mod:<12} {v:<12} {why}")
    except ImportError:
        say(f"  MISSING {mod:<12} {'':<12} {why}")

# ---------------------------------------------------------------- earth engine
head("EARTH ENGINE")
try:
    import ee
    try:
        ee.Initialize(project=os.environ.get("EE_PROJECT", "gbpuat-fyp-alaknanda"))
        n = ee.ImageCollection("COPERNICUS/S1_GRD").filterDate("2023-08-01", "2023-08-20").size().getInfo()
        say(f"  authenticated   : yes")
        say(f"  test query      : OK ({n} Sentinel-1 scenes worldwide, 1-20 Aug 2023)")
    except Exception as exc:
        say(f"  authenticated   : NO")
        say(f"  message         : {str(exc)[:300]}")
        say()
        say("  Fix: run  earthengine authenticate  and follow the browser prompt.")
        say("  If the project id is wrong, find it at https://code.earthengine.google.com")
except ImportError:
    say("  earthengine-api not installed. Runbook step 2.")

# ---------------------------------------------------------------- network
head("NETWORK")
try:
    import time
    import requests
    t0 = time.time()
    r = requests.get(
        "https://storage.googleapis.com/sen1floods11/"
        "v1.1/data/flood_events/HandLabeled/S1Hand/Bolivia_103757_S1Hand.tif",
        timeout=60,
    )
    dt = max(time.time() - t0, 0.01)
    mb = len(r.content) / 1024**2
    say(f"  test download   : {mb:.2f} MB in {dt:.1f} s  =  {mb/dt:.2f} MB/s")
    say(f"  status          : {r.status_code}")
    say()
    est = 1.25 * 1024 / max(mb / dt, 0.01) / 60
    say(f"  at this speed the core training set (~1.25 GB) takes about {est:.0f} min")
    est2 = 25.0 * 1024 / max(mb / dt, 0.01) / 3600
    say(f"  the optional weak-label set (~25 GB) takes about {est2:.1f} hours")
except Exception as exc:
    say(f"  test download failed: {exc}")

# ---------------------------------------------------------------- done
head("DONE")
say("Send machine_report.txt back. Nothing here was installed or changed.")

with open("machine_report.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(LINES))
print("\nwrote machine_report.txt")
