"""
pad_reexport_to_old_grid.py
Puts the corridor v2 re-export back onto the old pixel grid.
Written 2 October 2026.

WHY
Earth Engine did not export the full rectangle the script asked for. Because
every image was clipped to corridor v2, Earth Engine cut each file down to the
corridor's bounding box. The cells sit on the same lattice as the old files
(the offsets are whole cells), but the files are smaller:
    10 m   7251 x 2756 instead of 7915 x 3044
    30 m   2418 x  920 instead of 2639 x 1016
   100 m    726 x  277 instead of  793 x  306
check_reexport_v2.py then failed every file on "grid DIFFERENT".

WHAT IT DOES
For every .tif in 05 Data\\_reexport_v2\\ it:
  1. opens the old file of the same name in 05 Data and checks the new file is
     on the same lattice: same CRS, same cell size, whole-cell offset. If not,
     it stops and changes nothing for that file.
  2. writes a file with the OLD width, height and transform, the new bands,
     band names and data type. Cells the new file does not cover are blank
     (NaN for float bands, 0 for integer bands, the same blanks the old files
     used). New cells outside the old rectangle are dropped; the script counts
     how many of those held data and prints it.
  3. keeps the file exactly as downloaded in _reexport_v2\\_as_downloaded\\.
It also moves any .aux.xml beside a re-exported file into _as_downloaded,
because QGIS statistics for the old shape would be wrong for the new one.

Running it twice is safe. A file already on the old grid is skipped.

The July 2023 Sentinel-2 pair has no old file. For those,
the grid is taken from S2_2023_ref_20230819_10m.tif in 05 Data\\sentinel2\\,
the file in use, which is on the old grid.

Run from Anaconda Prompt:
    conda activate fyp
    cd /d "<project folder>\\04 Code\\FYP_code"
    python pad_reexport_to_old_grid.py
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[2]          # Final Year Project
DATA = ROOT / "05 Data"
NEW = DATA / "_reexport_v2"
KEEP = NEW / "_as_downloaded"
TMP = NEW / "_tmp_padding"

SUBDIR = {"S1_": "sentinel1", "S2_": "sentinel2", "DEM_": "dem", "WATER_": "dem",
          "LC_": "landcover", "POP_": "population"}
# Files with no old namesake, and the old-grid file to copy the grid from
GRID_FROM = {"S2_2023jul_": "S2_2023_ref_20230819_10m.tif"}


def subdir_for(name):
    for p, d in SUBDIR.items():
        if name.startswith(p):
            return d
    return None


def whole(x):
    return abs(x - round(x)) < 1e-6


def main():
    tifs = sorted(NEW.glob("*.tif"))
    if not tifs:
        sys.exit("No .tif files in %s." % NEW)
    KEEP.mkdir(exist_ok=True)
    TMP.mkdir(exist_ok=True)
    done, skipped, stopped = 0, 0, []

    for f in tifs:
        sub = subdir_for(f.name)
        old = DATA / sub / f.name if sub else None
        if old is not None and not old.exists():
            # The July 2023 pair has no old file. Take the grid from a 10 m
            # Sentinel-2 file that is already on the old grid.
            for prefix, ref in GRID_FROM.items():
                if f.name.startswith(prefix):
                    old = DATA / sub / ref
                    print("%-48s grid taken from %s" % (f.name, ref))
        if old is None or not old.exists():
            print("%-48s STOP  no old file of this name to take the grid from" % f.name)
            stopped.append(f.name)
            continue

        with rasterio.open(old) as o, rasterio.open(f) as n:
            ot, nt = o.transform, n.transform
            if (o.width, o.height, tuple(ot)[:6]) == (n.width, n.height, tuple(nt)[:6]):
                print("%-48s skip  already on the old grid" % f.name)
                skipped += 1
                continue
            if o.crs != n.crs or ot.a != nt.a or ot.e != nt.e or ot.b != 0 or nt.b != 0 \
                    or ot.d != 0 or nt.d != 0:
                print("%-48s STOP  CRS or cell size differs from the old file" % f.name)
                stopped.append(f.name)
                continue
            col = (nt.c - ot.c) / ot.a
            row = (nt.f - ot.f) / ot.e
            if not (whole(col) and whole(row)):
                print("%-48s STOP  not on the same lattice (offset %.3f, %.3f cells)"
                      % (f.name, col, row))
                stopped.append(f.name)
                continue
            col, row = int(round(col)), int(round(row))

            dtype = n.dtypes[0]
            is_float = np.issubdtype(np.dtype(dtype), np.floating)
            fill = np.nan if is_float else 0

            # the part of the new file that falls inside the old rectangle
            r0, c0 = max(0, row), max(0, col)
            r1, c1 = min(o.height, row + n.height), min(o.width, col + n.width)
            src_win = Window(c0 - col, r0 - row, c1 - c0, r1 - r0)

            full = n.read(1)
            valid_all = int(np.isfinite(full).sum() if is_float else (full != 0).sum())
            part = n.read(1, window=src_win)
            valid_kept = int(np.isfinite(part).sum() if is_float else (part != 0).sum())
            del full, part

            out = np.full((n.count, o.height, o.width), fill, dtype=dtype)
            out[:, r0:r1, c0:c1] = n.read(window=src_win)

            prof = n.profile.copy()
            prof.update(width=o.width, height=o.height, transform=ot, BIGTIFF="IF_SAFER")
            descriptions = n.descriptions
            tags = n.tags()
            band_tags = [n.tags(i) for i in range(1, n.count + 1)]

        tmp = TMP / f.name
        with rasterio.open(tmp, "w", **prof) as d:
            d.write(out)
            d.update_tags(**tags)
            for i, desc in enumerate(descriptions, start=1):
                if desc:
                    d.set_band_description(i, desc)
                if band_tags[i - 1]:
                    d.update_tags(i, **band_tags[i - 1])
        del out

        shutil.move(str(f), str(KEEP / f.name))
        aux = f.with_name(f.name + ".aux.xml")
        if aux.exists():
            shutil.move(str(aux), str(KEEP / aux.name))
        shutil.move(str(tmp), str(f))
        dropped = valid_all - valid_kept
        print("%-48s done  offset %d cols, %d rows; data cells outside old grid dropped: %d"
              % (f.name, col, row, dropped))
        done += 1

    try:
        TMP.rmdir()
    except OSError:
        pass
    print("\nPadded %d, already done %d, stopped %d." % (done, skipped, len(stopped)))
    if stopped:
        print("STOPPED, nothing changed for:", ", ".join(stopped))
        sys.exit(1)
    print("Next: python check_reexport_v2.py")


if __name__ == "__main__":
    main()
