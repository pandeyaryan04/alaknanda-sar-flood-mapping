r"""
diagnose_label_grid.py
======================
Measures, for every Sen1Floods11 chip, how far the label file's grid (and the
terrain file's grid) is from the radar chip's grid, in pixels.

WHY
---
20_models_preflight.py reported "[FAIL] labels: same grid as the chip" for 22
of 50 sampled chips. That check compares the two grids exactly. A difference
in the 15th digit of a coordinate (rounding when the files were written) is
harmless: the pixels sit on top of each other. A shift of half a pixel or more,
or a different number of rows or columns, is not. This script tells which.

It reads only the file headers (size, position, pixel size, projection).
It never reads a label value, so the test and Bolivia labels stay unread.

WHAT IT PRINTS
--------------
For hand and weak chips, and for labels and terrain files, how many are:
  identical          the same grid, to the last digit
  rounding           same size; differs by less than 0.01 of a pixel anywhere
  shift < 0.5 px     same size; shifted or stretched by up to half a pixel
  shift >= 0.5 px    same size; off by half a pixel or more
  other size         a different number of rows or columns
  other CRS          a different projection
  missing            no such file
then up to 5 examples of each kind with the numbers.

USAGE  (Anaconda Prompt, fyp active, from 04 Code\FYP_code)
-----
    python diagnose_label_grid.py --selftest
    python diagnose_label_grid.py

About 1 to 3 minutes (it opens about 14,500 file headers).
"""

__stage__ = "Models: diagnosis of label and terrain grids against the chips"
__reads__ = ["headers of Sen1Floods11 S1, label and terrain files"]
__writes__ = ["nothing (prints only)"]

import argparse
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import models_common as mc

try:
    import rasterio
except ImportError as e:
    sys.exit(f"\nMissing package: {e}.\nActivate the environment first:  conda activate fyp\n")

KINDS = ["identical", "rounding", "shift < 0.5 px", "shift >= 0.5 px", "other size",
         "other CRS", "missing", "unreadable"]


def header(path):
    with rasterio.open(path) as s:
        return {"shape": s.shape, "t": s.transform, "crs": s.crs}


def compare(a, b):
    """Kind of difference of grid b against grid a, and the numbers behind it."""
    if a["crs"] != b["crs"]:
        return "other CRS", {"crs_chip": str(a["crs"]), "crs_other": str(b["crs"])}
    ta, tb = a["t"], b["t"]
    worst = mc.grid_offset_px(a["shape"], ta, tb)       # the rule the model scripts use
    nums = {"chip_shape": a["shape"], "other_shape": b["shape"],
            "dx_px": (tb.c - ta.c) / abs(ta.a), "dy_px": (tb.f - ta.f) / abs(ta.e),
            "pixel_x_chip": ta.a, "pixel_x_other": tb.a, "pixel_y_chip": ta.e,
            "pixel_y_other": tb.e, "worst_px": worst}
    if a["shape"] != b["shape"]:
        return "other size", nums
    if ta == tb:
        return "identical", nums
    if worst < mc.GRID_TOL_PX:
        return "rounding", nums
    return ("shift < 0.5 px" if worst < 0.5 else "shift >= 0.5 px"), nums


def run(paths, show=5):
    counts = defaultdict(Counter)
    examples = defaultdict(list)
    worst = defaultdict(float)
    for set_ in ("hand", "weak"):
        _, _, _, s1suf, labsuf = mc.SETS[set_]
        chips = mc.list_chips(paths.s1_dir(set_), s1suf)
        if not chips:
            print(f"  no {set_} chips found in {paths.s1_dir(set_)}")
            continue
        prog = mc.Progress(len(chips), f"{set_} chips", every=30)
        for c in chips:
            try:
                a = header(paths.s1_dir(set_) / f"{c}{s1suf}")
            except Exception as e:                      # noqa: BLE001
                counts[(set_, "radar")]["unreadable"] += 1
                examples[(set_, "radar", "unreadable")].append((c, str(e)[:120]))
                prog.tick()
                continue
            for what, f in (("label", paths.label_dir(set_) / f"{c}{labsuf}"),
                            ("terrain", paths.terrain_dir(set_) / f"{c}_Terrain.tif")):
                if not f.exists():
                    counts[(set_, what)]["missing"] += 1
                    continue
                try:
                    kind, nums = compare(a, header(f))
                except Exception as e:                  # noqa: BLE001
                    kind, nums = "unreadable", {"error": str(e)[:120]}
                counts[(set_, what)][kind] += 1
                if "worst_px" in nums and kind != "other size":
                    worst[(set_, what)] = max(worst[(set_, what)], nums["worst_px"])
                if kind != "identical" and len(examples[(set_, what, kind)]) < show:
                    examples[(set_, what, kind)].append((c, nums))
            prog.tick()
    return counts, examples, worst


def report(counts, examples, worst):
    print("\nGrid of each file against its radar chip (counts)")
    print("| set | file | " + " | ".join(KINDS) + " | largest shift, px |")
    print("|---|---|" + "---|" * (len(KINDS) + 1))
    for key in sorted(counts):
        set_, what = key
        row = " | ".join(str(counts[key].get(k, 0)) for k in KINDS)
        print(f"| {set_} | {what} | {row} | {worst.get(key, 0.0):.6g} |")
    print("\nExamples")
    for (set_, what, kind), ex in sorted(examples.items()):
        print(f"  {set_} {what}, {kind}:")
        for c, nums in ex:
            if isinstance(nums, dict) and "dx_px" in nums:
                print(f"    {c}: chip {nums['chip_shape']}, {what} {nums['other_shape']}, "
                      f"origin shift x {nums['dx_px']:+.6g} px, y {nums['dy_px']:+.6g} px, "
                      f"pixel size chip {nums['pixel_x_chip']:.12g} / {nums['pixel_y_chip']:.12g}, "
                      f"{what} {nums['pixel_x_other']:.12g} / {nums['pixel_y_other']:.12g}, "
                      f"worst {nums['worst_px']:.6g} px")
            else:
                print(f"    {c}: {nums}")
    bad = sum(counts[k].get(x, 0) for k in counts
              for x in ("shift < 0.5 px", "shift >= 0.5 px", "other size", "other CRS"))
    print()
    if bad == 0:
        print("RESULT: every label and terrain file sits on its chip's grid (identical or "
              "rounding only). The preflight check was too strict.")
    else:
        print(f"RESULT: {bad} files are shifted, stretched or sized differently. Do not train "
              "yet. The lines above say which files and by how much.")


def main(argv=None):
    mc.setup_console()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    paths = mc.Paths(args.project)
    print(f"diagnose_label_grid.py: {paths.events}")
    counts, examples, worst = run(paths)
    report(counts, examples, worst)
    return 0


def selftest():
    import models_synthetic as syn
    print("diagnose_label_grid.py self-test")
    with tempfile.TemporaryDirectory(prefix="fyp selftest ", ignore_cleanup_errors=True) as td:
        root = Path(td) / "Final Year Project"
        syn.build_project(root, corridor=False)
        p = mc.Paths(root)

        def rewrite(f, dx=0.0, a_scale=1.0, shape=None):
            with rasterio.open(f) as s:
                arr, t, crs, dt = s.read(), s.transform, s.crs, s.dtypes[0]
            if shape:
                arr = arr[:, :shape[0], :shape[1]]
            t = rasterio.Affine(t.a * a_scale, t.b, t.c + dx * t.a, t.d, t.e, t.f)
            syn._write(f, list(arr), dt, crs, t)

        rewrite(p.label_dir("hand") / "Ghana_1001_LabelHand.tif", dx=1e-9)          # rounding
        rewrite(p.label_dir("hand") / "India_1002_LabelHand.tif", dx=0.3)           # sub-pixel
        rewrite(p.label_dir("weak") / "Ghana_5001_S2IndexLabelWeak.tif", dx=1.0)    # 1 pixel
        rewrite(p.label_dir("weak") / "India_5002_S2IndexLabelWeak.tif", shape=(63, 64))
        counts, examples, worst = run(p)
        report(counts, examples, worst)
    hl, wl = counts[("hand", "label")], counts[("weak", "label")]
    checks = [("identical counted", hl["identical"] == 16),
              ("rounding recognised", hl["rounding"] == 1),
              ("sub-pixel shift recognised", hl["shift < 0.5 px"] == 1),
              ("whole-pixel shift recognised", wl["shift >= 0.5 px"] == 1),
              ("other size recognised", wl["other size"] == 1),
              ("terrain all identical", counts[("hand", "terrain")]["identical"] == 17)]
    for name, c in checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {name}")
    good = all(c for _, c in checks)
    print("SELFTEST PASS" if good else "SELFTEST FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(mc.cli(main))
