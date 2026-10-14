r"""
18_data_manifest.py
===================
A checksum (SHA-256) and size for every data file the
project uses, so anyone can confirm they hold exactly the same files.

  04 Code\provenance\DATA_MANIFEST.csv     path, bytes, sha256, modified

The manifest goes into the code repository; the data itself does not.

What it covers (paths relative to the project folder):
  05 Data\                                  every source dataset and derived layer
                                            (not 05 Data\00 Notes, which are documents)
  00 New work\Digitising handoff\received\  the original hand-drawn returns
  06 Outputs\gis_tools\qgis\                the reference GeoPackages in use
  06 Outputs\tables\                        the scored baseline tables
  04 Code\FYP_code\data\                    Sen1Floods11, terrain, strata, flood maps,
                                            reference rasters

USAGE
-----
  conda activate fyp
  cd /d "<project folder>\04 Code\FYP_code"

  python 18_data_manifest.py              write the manifest (takes a while: tens of GB)
  python 18_data_manifest.py --verify     compare the files on this machine with it

Close QGIS first: QGIS writes into a GeoPackage when it opens one, which
changes its checksum without changing the data.

--verify prints one line per file that is changed or missing, and per file
present on disk but not in the manifest. PASS: "changed 0, missing 0".
"""

__stage__ = "Provenance, data checksums"
__reads__ = ["05 Data", "the reference drawings", "key outputs"]
__writes__ = ["provenance/DATA_MANIFEST.csv"]

import argparse
import csv
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "04 Code" / "provenance" / "DATA_MANIFEST.csv"
COVER = [
    "05 Data",
    "00 New work/Digitising handoff/received",
    "06 Outputs/gis_tools/qgis",
    "06 Outputs/tables",
    "04 Code/FYP_code/data",
]
SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints"}
# notes are documents that keep being edited, not data
SKIP_PREFIX = ("05 Data/00 Notes/",)
SKIP_SUFFIX = {".gpkg-shm", ".gpkg-wal", ".lock", ".tmp", ".part"}
CHUNK = 8 * 1024 * 1024


def files():
    for top in COVER:
        base = ROOT / top
        if not base.exists():
            print(f"  not found, skipped: {top}")
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and not (set(p.relative_to(ROOT).parts) & SKIP_DIRS) \
                    and p.suffix.lower() not in SKIP_SUFFIX \
                    and not rel(p).startswith(SKIP_PREFIX):
                yield p


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def rel(p):
    return p.relative_to(ROOT).as_posix()


def write():
    t0 = time.time()
    rows, total = [], 0
    for i, p in enumerate(files(), 1):
        st = p.stat()
        rows.append({"path": rel(p), "bytes": st.st_size, "sha256": sha256(p),
                     "modified": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
                     .strftime("%Y-%m-%dT%H:%M:%SZ")})
        total += st.st_size
        if i % 500 == 0:
            print(f"  {i} files, {total / 1e9:.1f} GB, {time.time() - t0:.0f} s")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "bytes", "sha256", "modified"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n  files: {len(rows)}, {total / 1e9:.2f} GB, {time.time() - t0:.0f} s")
    print(f"  written: {OUT}")
    return 0


def verify():
    if not OUT.exists():
        sys.exit(f"No manifest at {OUT}. Run without --verify first.")
    with open(OUT, newline="", encoding="utf-8") as fh:
        want = {r["path"]: r for r in csv.DictReader(fh)}
    seen, changed, missing = set(), [], []
    for p in files():
        r = rel(p)
        seen.add(r)
        if r not in want:
            continue
        if p.stat().st_size != int(want[r]["bytes"]) or sha256(p) != want[r]["sha256"]:
            changed.append(r)
    for r in want:
        if r not in seen:
            missing.append(r)
    extra = sorted(seen - set(want))
    for r in changed:
        print(f"  CHANGED  {r}")
    for r in missing:
        print(f"  MISSING  {r}")
    for r in extra:
        print(f"  NEW      {r}")
    print(f"\n  in manifest {len(want)}; same {len(want) - len(changed) - len(missing)}, "
          f"changed {len(changed)}, missing {len(missing)}; new on disk {len(extra)}")
    ok = not changed and not missing
    print("  PASS" if ok else "  DIFF: see the lines above")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)
    return verify() if a.verify else write()


if __name__ == "__main__":
    sys.exit(main())
