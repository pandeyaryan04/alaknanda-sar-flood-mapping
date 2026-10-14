r"""
21_prepare_chips.py
===================
Builds the list of Sen1Floods11 chips the models may use, and the
normalisation statistics every model and every corridor prediction share.

WHAT IT DOES
------------
1. Reads the four split files (train 252, valid 89, test 90, Bolivia 15) and
   lists every hand-labelled and weakly labelled chip.
2. Drops, and says why:
     - chips with no radar at all (no finite VV; 1 hand and 10 weak chips)
     - chips with no label pixel that is 0 or 1
     - chips without a terrain file from 11_sen1floods11_terrain.py
     - every weak chip of event Bolivia (Bolivia is the unseen-event test)
     - every weak chip whose footprint touches or overlaps a hand-labelled
       valid, test or Bolivia chip, so no test ground is ever trained on
3. Writes data\models\chip_index.csv: one row per chip with its set, split,
   event, files, radar and label shares, footprint, and kept / dropped + why.
4. Computes the normalisation statistics from the hand TRAIN chips only
   (up to 20,000 random valid pixels per chip, seeded): per channel the 0.5
   and 99.5 percentiles, then mean and standard deviation of the clipped
   values. Writes data\models\norm_stats.json.
5. With --cache, also writes each kept chip as float16 .npy (inputs) and
   int8 .npy (label) into data\models\cache\ (about 10 GB). Test and Bolivia
   chips are never cached, and their labels are not read here at all: only
   24_eval_sen1floods11.py reads them, for the final evaluation.
   Training reads these much faster; it works without them too.

USAGE: HOW TO RUN THIS  (copy one line, paste into Anaconda Prompt, press Enter)
================================================================================
    conda activate fyp
    cd /d "<project folder>\04 Code\FYP_code"

    python 21_prepare_chips.py --selftest
    python 21_prepare_chips.py --cache

It reads every chip once: about 5 to 15 minutes, plus the cache. Check the
printed counts: hand train about 251 kept, weak about 4,100 kept, every
dropped chip with its reason. The last line says PASS.
================================================================================
"""

__stage__ = "Models: chip index and normalisation statistics"
__reads__ = ["data/sen1floods11/v1.1 (chips and splits)",
             "data/sen1floods11_terrain/{hand,weak}/<chip>_Terrain.tif"]
__writes__ = ["data/models/chip_index.csv", "data/models/norm_stats.json",
              "data/models/prepare_record.json", "data/models/cache/ (with --cache)"]

import argparse
import csv
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

import models_common as mc

SAMPLE_PER_CHIP = 20000
SMOKE_SPLIT = {"train": 8, "valid": 4, "test": 4, "bolivia": 2}
SMOKE_WEAK, SMOKE_WEAK_BOLIVIA = 10, 2


def _row(paths, set_, chip, split):
    _, _, _, s1suf, labsuf = mc.SETS[set_]
    s1 = paths.s1_dir(set_) / f"{chip}{s1suf}"
    lab = paths.label_dir(set_) / f"{chip}{labsuf}"
    ter = paths.terrain_dir(set_) / f"{chip}_Terrain.tif"
    return {"chip": chip, "set": set_, "split": split, "event": mc.event_of(chip),
            "s1": paths.rel_any(s1), "label": paths.rel_any(lab),
            "terrain": paths.rel_any(ter), "_s1": s1, "_lab": lab, "_ter": ter,
            "height": "", "width": "", "finite_share": "", "labelled_pixels": "",
            "labelled_share": "", "water_share": "", "terrain_nan_share": "",
            "minx": "", "miny": "", "maxx": "", "maxy": "", "kept": "no", "reason": ""}


def _footprint(row):
    b, h, w = mc.chip_footprint(row["_s1"])
    row.update(minx=b[0], miny=b[1], maxx=b[2], maxy=b[3], height=h, width=w)
    return b


def _read_stats(paths, row, labels=True):
    """Read the chip once; fill the shares; return (x, y) or None if unusable.

    labels=False (test and Bolivia chips): only the radar is read. Their
    labels are read by 24_eval_sen1floods11.py and nowhere else.
    """
    if not row["_lab"].exists():
        row["reason"] = "label file missing"
        return None
    has_ter = row["_ter"].exists()
    r = dict(row, terrain=row["terrain"] if has_ter else "")
    try:
        if labels:
            x, y = mc.read_chip(r, paths.data)
        else:
            x, y = mc.read_s1(row["_s1"]), None
    except Exception as e:                          # noqa: BLE001
        mc.fail(f"Chip {row['chip']} cannot be read: {e}\n"
                "Run 20_models_preflight.py: it checks every file the models need and says "
                "what to do.")
    fin = mc.input_valid(x)
    row["finite_share"] = round(float(fin.mean()), 4)
    n = 1
    if labels:
        tv = fin & (y >= 0)
        n = int(tv.sum())
        row.update(labelled_pixels=n, labelled_share=round(n / y.size, 4),
                   water_share=round(float((y[tv] == 1).mean()), 4) if n else 0.0)
        if n:
            bad = ~(np.isfinite(x[2]) & np.isfinite(x[3]))
            row["terrain_nan_share"] = round(float(bad[tv].mean()), 4)
    if not np.isfinite(x[0]).any():
        row["reason"] = "empty chip (no radar data)"
        return None
    if not has_ter:
        row["reason"] = "no terrain file"
        return None
    if n == 0:
        row["reason"] = "no labelled pixels"
        return None
    row["kept"] = "yes"
    return x, y


def build_index(paths, smoke=False, seed=mc.SEED):
    hand_ids = mc.list_chips(paths.s1_dir("hand"), "_S1Hand.tif")
    weak_ids = mc.list_chips(paths.s1_dir("weak"), "_S1Weak.tif")
    if not hand_ids:
        mc.fail(f"No hand-labelled chips found in\n  {paths.s1_dir('hand')}\n"
                "Sen1Floods11 should be in FYP_code\\data\\sen1floods11\\v1.1 "
                "(RUN ORDER.md, step 1.3). Run 20_models_preflight.py for a full check.")
    split_of, split_lists = {}, {}
    for split, fname in mc.SPLIT_FILES.items():
        f = paths.splits / fname
        if not f.exists():
            mc.fail(f"Split file missing:\n  {f}\nIt comes with Sen1Floods11 (v1.1/splits).")
        ids = mc.read_split_csv(f)
        split_lists[split] = ids
        for c in ids:
            if c in split_of:
                mc.fail(f"Chip {c} is in both the {split_of[c]} and the {split} split files.")
            split_of[c] = split
    missing = [c for c in split_of if c not in set(hand_ids)]
    if missing:
        mc.fail(f"{len(missing)} chips in the split files have no S1Hand file, for example "
                f"{missing[:3]}. The download is incomplete: run 02_fetch_sen1floods11.py.")

    if smoke:
        hand_use = [c for s, k in SMOKE_SPLIT.items() for c in sorted(split_lists[s])[:k]]
        bol = [c for c in weak_ids if mc.event_of(c) == "Bolivia"][:SMOKE_WEAK_BOLIVIA]
        oth = [c for c in weak_ids if mc.event_of(c) != "Bolivia"][:SMOKE_WEAK]
        weak_use = sorted(bol + oth)
    else:
        hand_use, weak_use = hand_ids, weak_ids

    rows = [_row(paths, "hand", c, split_of.get(c, "none")) for c in hand_use]
    weak_rows = [_row(paths, "weak", c, "weak") for c in weak_use]

    # footprints of every held-out hand chip (valid, test, Bolivia), smoke or not
    held = [_row(paths, "hand", c, s) for c, s in split_of.items()
            if s in ("valid", "test", "bolivia")]
    print(f"  reading footprints of {len(held)} held-out hand chips and "
          f"{len(weak_rows)} weak chips")
    held_boxes = np.array([_footprint(r) for r in held]).reshape(-1, 4)
    # 0.01 pixel in degrees: touching chips meet to the last digit only
    held_px = [(r["maxx"] - r["minx"]) / r["width"] for r in held if r["width"]]
    touch_eps = mc.GRID_TOL_PX * (float(np.median(held_px)) if held_px else 0.0)
    for r in rows:
        _footprint(r)

    train_samples, rng0 = [], seed
    prog = mc.Progress(len(rows) + len(weak_rows), "chips read", every=60)
    for i, r in enumerate(rows):
        if r["split"] == "none":
            r["reason"] = "not in any split file"
            prog.tick()
            continue
        got = _read_stats(paths, r, labels=r["split"] not in ("test", "bolivia"))
        if got is not None and r["split"] == "train":
            rng = np.random.default_rng([rng0, i])
            train_samples.append(mc.sample_pixels(got[0], got[1], SAMPLE_PER_CHIP, rng))
        prog.tick()

    weak_boxes = []
    for r in weak_rows:
        if r["event"] == "Bolivia":
            r["reason"] = "weak chip of event Bolivia"
            weak_boxes.append((np.inf, np.inf, -np.inf, -np.inf))
            continue
        weak_boxes.append(_footprint(r))
    hit = mc.boxes_intersect(np.array(weak_boxes).reshape(-1, 4), held_boxes, touch_eps)
    for r, h in zip(weak_rows, hit):
        if r["reason"]:
            prog.tick()
            continue
        if h:
            r["reason"] = "overlaps a hand valid, test or Bolivia chip"
            prog.tick()
            continue
        _read_stats(paths, r)
        prog.tick()
    return rows + weak_rows, train_samples, split_lists


def write_index(paths, rows):
    paths.models.mkdir(parents=True, exist_ok=True)
    tmp = paths.chip_index.with_name(paths.chip_index.name + ".part")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=mc.INDEX_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, paths.chip_index)


def write_cache(paths, index):
    todo = []
    for r in index:
        if r["kept"] != "yes" or r["split"] in ("test", "bolivia"):
            continue                    # test and Bolivia are read only by 24, from the .tif
        cx, cy = mc.cache_paths(paths, r)
        if not (paths.abs(cx).exists() and paths.abs(cy).exists()):
            todo.append((r, cx, cy))
    if not todo:
        print("  cache: every kept chip is already cached")
        return 0
    need = sum(r["height"] * r["width"] * 9 for r, _, _ in todo) / 1e9 + 1.0
    mc.check_disk(paths.cache, need, "the .npy cache")
    prog = mc.Progress(len(todo), "chips cached", every=60)
    for r, cx, cy in todo:
        x, y = mc.read_chip(dict(r, cache_x="", cache_y=""), paths.data)
        for rel, arr in ((cx, x.astype("float16")), (cy, y.astype("int8"))):
            dst = paths.abs(rel)
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".part")
            with open(tmp, "wb") as fh:
                np.save(fh, arr)
            os.replace(tmp, dst)
        prog.tick()
    print(f"  cache: {len(todo)} chips written to {paths.cache}")
    return len(todo)


def report(rows, split_lists):
    print("\n  Chip index")
    print("  | set | split | listed | kept | dropped |")
    print("  |---|---|---|---|---|")
    for set_, split in (("hand", "train"), ("hand", "valid"), ("hand", "test"),
                        ("hand", "bolivia"), ("weak", "weak")):
        rs = [r for r in rows if r["set"] == set_ and r["split"] == split]
        k = sum(r["kept"] == "yes" for r in rs)
        print(f"  | {set_} | {split} | {len(rs)} | {k} | {len(rs) - k} |")
    reasons = Counter((r["set"], r["reason"]) for r in rows if r["kept"] != "yes")
    print("\n  Dropped, by reason:")
    for (set_, why), n in sorted(reasons.items()):
        print(f"    {set_:<5} {n:>6}  {why}")
    if not reasons:
        print("    none")
    return reasons


def main(argv=None):
    mc.setup_console()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mc.add_common_args(ap)
    ap.add_argument("--cache", action="store_true",
                    help="also write float16/int8 .npy copies of every kept chip (faster training)")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    paths = mc.paths_from_args(args)
    t0 = time.time()
    print(f"21_prepare_chips.py{' (smoke)' if args.smoke else ''}")
    print(f"  Sen1Floods11: {paths.s1f}")
    print(f"  writes to   : {paths.models}")

    if mc.guard(paths.norm_stats, args, "The chip index and norm_stats.json"):
        if args.cache:
            write_cache(paths, mc.load_chip_index(paths))
        print("PASS")
        return 0

    rows, samples, split_lists = build_index(paths, args.smoke, args.seed)
    write_index(paths, rows)
    reasons = report(rows, split_lists)

    index = mc.load_chip_index(paths)
    hand, weak = mc.training_rows(index, "hand_weak")
    if not hand:
        mc.fail("No hand-labelled training chip survived. Run 20_models_preflight.py.")
    smp = np.concatenate(samples) if samples else np.zeros((0, 4), "float32")
    tnan = float(np.mean(~np.isfinite(smp[:, 2]) | ~np.isfinite(smp[:, 3]))) if len(smp) else 0.0
    stats = mc.compute_norm_stats(smp, "hand train chips",
                                  {"chips": len(hand), "pixels_per_chip_max": SAMPLE_PER_CHIP,
                                   "seed": args.seed,
                                   "terrain_nan_share_of_sampled_pixels": round(tnan, 5)})
    print("\n  Normalisation, from the hand train chips only "
          f"({len(hand)} chips, {stats['pixels_sampled']:,} pixels sampled):")
    print("    channel   clip_lo    clip_hi     mean      std")
    for i, ch in enumerate(mc.CHANNELS):
        print(f"    {ch:<6} {stats['clip_lo'][i]:>9.3f} {stats['clip_hi'][i]:>10.3f} "
              f"{stats['mean'][i]:>8.3f} {stats['std'][i]:>8.3f}")
    print(f"    sampled training pixels with HAND or slope missing (filled with 0 after "
          f"normalisation): {100 * tnan:.2f} %")
    ok = True
    if stats["clip_lo"][0] > -1:
        print("  WARNING: VV does not look like dB (its 0.5 percentile is above -1). "
              "Sen1Floods11 S1 chips should be in dB. Run 20_models_preflight.py.")
        ok = False

    if args.cache:
        write_cache(paths, index)
    rec = mc.run_record(args, paths, step="prepare chips",
                        counts={f"{s}|{p}": sum(1 for r in rows if r["set"] == s and r["split"] == p
                                                and r["kept"] == "yes")
                                for s, p in {(r["set"], r["split"]) for r in rows}},
                        dropped={f"{s}|{w}": n for (s, w), n in reasons.items()},
                        seconds=round(time.time() - t0, 1),
                        outputs=[mc.file_entry(paths.chip_index)])
    mc.save_json(paths.models / "prepare_record.json", rec)
    mc.save_json(paths.norm_stats, stats)          # written last: marks the step done
    print(f"\n  wrote {paths.chip_index}")
    print(f"  wrote {paths.norm_stats}")
    print(f"  {len(hand)} hand train chips and {len(weak)} weak chips available for training")
    print("PASS" if ok else "FINISHED WITH A WARNING. Read it above.")
    return 0


def selftest():
    import models_synthetic as syn
    print("21_prepare_chips.py self-test")
    with tempfile.TemporaryDirectory(prefix="fyp selftest ", ignore_cleanup_errors=True) as td:
        proj = Path(td) / "Final Year Project"
        syn.build_project(proj, corridor=False)
        rc = main(["--project", str(proj), "--cache"])
        paths = mc.Paths(proj)
        idx = {r["chip"]: r for r in mc.load_chip_index(paths)}
        checks = [
            ("ran", rc == 0),
            ("empty hand chip dropped", idx["Paraguay_34417"]["reason"].startswith("empty")),
            ("chip with all labels -1 dropped", idx["Spain_1007"]["reason"] == "no labelled pixels"),
            ("Bolivia weak chip dropped", "Bolivia" in idx["Bolivia_5005"]["reason"]),
            ("weak chip over a test chip dropped", "overlaps" in idx["Mekong_5006"]["reason"]),
            ("weak chip over a train chip kept", idx["Paraguay_5008"]["kept"] == "yes"),
            ("cache written, never for test or Bolivia",
             all(bool(r["cache_x"]) == (r["split"] not in ("test", "bolivia"))
                 for r in idx.values() if r["kept"] == "yes")),
            ("test labels not read", idx["Mekong_3001"]["labelled_pixels"] == 0
             and idx["Mekong_3001"]["kept"] == "yes"),
            ("norm stats from the 8 kept train chips", mc.load_json(paths.norm_stats)["chips"] == 8),
        ]
    bad = 0
    for name, c in checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {name}")
        bad += not c
    print("SELFTEST PASS" if not bad else "SELFTEST FAIL")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(mc.cli(main))
