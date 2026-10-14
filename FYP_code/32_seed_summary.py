r"""
32_seed_summary.py
==================
Repeat runs (9 October 2026): how much do the two leading models change when
they are trained again with another random seed?

`unet_hand` and `tm_s1t_hand` were each trained once (seed 20261007). This
trains each twice more, with seeds 20261008 (run tag seed2) and 20261009
(seed3), in their own folders, and scores them exactly as Table A. This
script only reads what 24, 04, 27 and 31 wrote and puts the three runs of
each model side by side:

- each run: seed, best epoch, threshold, training hours, Sen1Floods11 IoU
  (valid, test, Bolivia), corridor F1 2021 and 2023 (pooled, with the 95 %
  window interval), calibration error as trained (if 31 was run)
- across the three seeds: mean, standard deviation, lowest and highest
- the paired comparisons for each seed (27): tm_s1t against the U-Net, and
  each against February change detection, with how many seeds give an
  interval that excludes 0

The window intervals (from 04 and 27) cover the spread between the 8
windows. The spread across seeds covers the spread between training runs.
They are different things and both are reported. With three runs the
standard deviation is a rough figure.

Nothing is trained or tuned here, and no map or earlier table changes.

USAGE: HOW TO RUN THIS  (copy one line, paste into Anaconda Prompt, press Enter)
================================================================================
    conda activate fyp
    cd /d "<project folder>\04 Code\FYP_code"

    python 32_seed_summary.py --selftest
    python 32_seed_summary.py --force

Takes a second. --force rewrites the summary (it is made from the other
files, so nothing is lost). The last line says PASS. RUN SEEDS.bat runs it
at the end.
================================================================================
"""

__stage__ = "Models: spread across training seeds"
__reads__ = ["data/models/<model>_hand[_<tag>]/run_record.json",
             "06 Outputs/tables/s1f11_<model>_hand[_<tag>]/metrics.csv",
             "06 Outputs/tables/<year>_<model>_hand[_<tag>]/metrics_by_stratum.csv",
             "06 Outputs/tables/compare_* and seeds_compare_*/comparison.csv",
             "06 Outputs/tables/calibration_summary[_<tag>].csv"]
__writes__ = ["06 Outputs/tables/SEEDS_SUMMARY.md", "06 Outputs/tables/SEEDS_SUMMARY.csv"]

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import models_common as mc

MODELS = ("unet", "tm_s1t")
TAGS = ("", "seed2", "seed3")          # "" is the original run (seed 20261007)
SPLITS = ("valid", "test", "bolivia")
COLS = ["kind", "model", "run", "seed", "metric", "value", "lo", "hi", "n"]
NAMES = {"unet": "U-Net", "tm_s1t": "TerraMind tm_s1t"}
# training settings that must be the same in every run of a model; only the seed may differ
SAME = ("trainset", "epochs", "steps_per_epoch", "batch_size", "crop", "base_width", "lr",
        "lr_encoder", "lr_decoder", "weight_decay", "hand_fraction", "pos_weight",
        "warmup_steps", "select_by", "patience", "variant")


def _csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _sfx(tag):
    return f"_{tag}" if tag else ""


def _run_name(model, tag):
    return f"{model}_hand{_sfx(tag)}"


def compare_label(m, tag):
    """A map's label as the launchers write it: 'otsu_chfeb', or 'unet_hand_seed2'."""
    return m if m.startswith("otsu") else f"{_run_name(m, tag)}"


def compare_dir(tables, year, a, b, tag):
    """The folder 27_compare_maps.py wrote: compare_<year>_<A>_vs_<B> for the original
    runs (RUN MODELS COMPARE.bat, RUN ALL MODELS.bat) and seeds_compare_<year>_<A>_vs_<B>
    for the repeat runs (RUN SEEDS.bat), with A and B labelled as in those launchers."""
    pre = "seeds_compare" if tag else "compare"
    return tables / f"{pre}_{year}_{compare_label(a, tag)}_vs_{compare_label(b, tag)}"


def pooled_compare(tables, year, a, b, tag):
    d = compare_dir(tables, year, a, b, tag)
    f = d / "comparison.csv"
    if not f.exists():
        return None, d
    r = next((q for q in _csv(f) if q["reach"] == "pooled" and q["dimension"] == "ALL"), None)
    if not r:
        return None, d
    return (_f(r["diff_f1"]), _f(r["diff_f1_lo"]), _f(r["diff_f1_hi"])), d


def read_run(paths, model, tag):
    """Everything about one training run. Missing pieces are None and listed."""
    run = _run_name(model, tag)
    out, missing = {"model": model, "tag": tag, "run": run}, []
    rec_p = paths.model_dir(model, "hand", tag) / "run_record.json"
    if rec_p.exists():
        rec = mc.load_json(rec_p)
        out["options"] = rec.get("options", {})
        out.update(seed=rec.get("seed"), best_epoch=rec.get("best_epoch"),
                   threshold=_f(rec.get("threshold")),
                   hours=(_f(rec.get("train_seconds")) or 0) / 3600 or None,
                   finished=bool(rec.get("finished")))
        if not rec.get("finished"):
            missing.append(f"{run}: training not finished")
    else:
        out.update(seed=None, best_epoch=None, threshold=None, hours=None, finished=False,
                   options={})
        missing.append(f"{run}: not trained ({rec_p.parent.name}\\run_record.json)")
    f = paths.tables / f"s1f11_{run}" / "metrics.csv"
    if f.exists():
        for r in _csv(f):
            if r["threshold_kind"] == "valid_chosen":
                out[f"iou_{r['split']}"] = _f(r["iou"])
    else:
        missing.append(f"{run}: Sen1Floods11 evaluation (s1f11_{run})")
    for year in mc.YEARS:
        f = paths.tables / f"{year}_{run}" / "metrics_by_stratum.csv"
        r = None
        if f.exists():
            r = next((q for q in _csv(f) if q.get("table") == "reported"
                      and q.get("reach") == "pooled" and q.get("dimension") == "ALL"), None)
        if r:
            out[f"f1_{year}"] = (_f(r["f1"]), _f(r.get("f1_lo")), _f(r.get("f1_hi")))
        else:
            missing.append(f"{run}: corridor score {year}_{run}")
    f = paths.tables / f"calibration_summary{_sfx(tag)}.csv"
    if f.exists():
        for r in _csv(f):
            if r["model"] == model and r["trainset"] == f"hand{_sfx(tag)}":
                out[f"ece_{r['split']}"] = _f(r["ece_raw"])
    return out, missing


def spread(values):
    v = [x for x in values if x is not None]
    if not v:
        return None
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else None
    return {"n": len(v), "mean": m, "sd": sd, "min": min(v), "max": max(v)}


def _n(x, d=3):
    return "" if x is None else f"{x:.{d}f}"


def _signed(t, d=3):
    if t is None or t[0] is None:
        return "not run"
    v, lo, hi = t
    star = " *" if lo is not None and hi is not None and (lo > 0 or hi < 0) else ""
    if lo is None or hi is None:
        return f"{v:+.{d}f} (no interval)"
    return f"{v:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]{star}"


SPLIT_NAMES = {"valid": "valid", "test": "test", "bolivia": "Bolivia"}
METRICS = ([("iou_" + s, f"Sen1Floods11 IoU, {SPLIT_NAMES[s]}") for s in SPLITS]
           + [(f"f1_{y}", f"Corridor F1, {y}") for y in mc.YEARS]
           + [("ece_" + s, f"Calibration error as trained, {SPLIT_NAMES[s]}")
              for s in SPLITS]
           + [("threshold", "Threshold (chosen on valid)"), ("best_epoch", "Best epoch"),
              ("hours", "Training hours")])

COMPARISONS = [("unet", "tm_s1t", "TerraMind tm_s1t minus U-Net"),
               ("otsu_chfeb", "unet", "U-Net minus Feb change detection"),
               ("otsu_chfeb", "tm_s1t", "TerraMind tm_s1t minus Feb change detection")]


def _val(run, key):
    v = run.get(key)
    return v[0] if isinstance(v, tuple) else v


def build(paths):
    runs, missing = {}, []
    for model in MODELS:
        for tag in TAGS:
            r, m = read_run(paths, model, tag)
            runs[(model, tag)] = r
            missing += m
    for model in MODELS:
        ref = runs[(model, "")].get("options") or {}
        for tag in TAGS[1:]:
            opt = runs[(model, tag)].get("options") or {}
            if not ref or not opt:
                continue
            diff = [f"{k} {ref.get(k)} -> {opt.get(k)}" for k in SAME
                    if k in ref and ref.get(k) != opt.get(k)]
            if diff:
                missing.append(f"WARNING {_run_name(model, tag)} was not trained with the "
                               f"original settings: {'; '.join(diff)}")
            if runs[(model, tag)].get("seed") == runs[(model, "")].get("seed"):
                missing.append(f"WARNING {_run_name(model, tag)} has the original seed")
    comps = {}
    for a, b, _ in COMPARISONS:
        for year in mc.YEARS:
            for tag in TAGS:
                c, d = pooled_compare(paths.tables, year, a, b, tag)
                comps[(a, b, year, tag)] = c
                if c is None:
                    missing.append(f"comparison {d.name}")
    return runs, comps, missing


def write(paths, runs, comps, missing):
    rows = []
    for (model, tag), r in runs.items():
        for key, _ in METRICS:
            v = r.get(key)
            lo = hi = None
            if isinstance(v, tuple):
                v, lo, hi = v
            if v is not None:
                rows.append({"kind": "run", "model": model, "run": r["run"],
                             "seed": r.get("seed"), "metric": key, "value": v, "lo": lo,
                             "hi": hi, "n": 1})
    stats = {}
    for model in MODELS:
        for key, _ in METRICS:
            s = spread([_val(runs[(model, t)], key) for t in TAGS])
            stats[(model, key)] = s
            if s:
                for k in ("mean", "sd", "min", "max"):
                    if s[k] is not None:
                        rows.append({"kind": f"across seeds, {k}", "model": model,
                                     "run": f"{model}_hand, {s['n']} runs", "seed": "",
                                     "metric": key, "value": s[k], "lo": "", "hi": "",
                                     "n": s["n"]})
    for (a, b, year, tag), c in comps.items():
        if c:
            rows.append({"kind": "paired comparison", "model": f"{b} minus {a}",
                         "run": tag or "original", "seed": "", "metric": f"f1 {year}",
                         "value": c[0], "lo": c[1], "hi": c[2], "n": 1})
    paths.tables.mkdir(parents=True, exist_ok=True)
    with open(paths.tables / "SEEDS_SUMMARY.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)

    lines = ["# The two leading models trained with three seeds", "",
             "`unet_hand` and `tm_s1t_hand`, each trained three times on the same data with "
             "seeds 20261007 (the original run, used by Tables A and D and by Jal-SAR), "
             "20261008 (`seed2`) and 20261009 (`seed3`). Everything else is the same as "
             "Table A. Corridor F1 is pooled over the 8 windows; brackets are the 95 % window "
             "intervals from 04 and 27. The spread across seeds is a different thing: it "
             "shows how much a result moves when the model is trained again. With three runs "
             "the standard deviation (SD) is a rough figure.", ""]
    for model in MODELS:
        lines += [f"## {NAMES[model]} (`{model}_hand`)", "",
                  "| | original | seed2 | seed3 | mean | SD | lowest | highest |",
                  "|---|---|---|---|---|---|---|---|"]
        seeds = [runs[(model, t)].get("seed") for t in TAGS]
        lines.append("| Seed | " + " | ".join("" if s is None else str(s) for s in seeds)
                     + " | | | | |")
        for key, label in METRICS:
            cells = []
            for t in TAGS:
                v = runs[(model, t)].get(key)
                if isinstance(v, tuple):
                    cells.append("" if v[0] is None else
                                 (f"{v[0]:.3f}" if v[1] is None else
                                  f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"))
                elif key == "best_epoch":
                    cells.append("" if v is None else str(v))
                else:
                    cells.append(_n(v, 2 if key in ("threshold", "hours") else 3))
            if not any(cells):
                continue
            s = stats[(model, key)]
            d = 2 if key in ("threshold", "hours") else (1 if key == "best_epoch" else 3)
            tail = ["", "", "", ""] if not s else [_n(s["mean"], d), _n(s["sd"], d),
                                                   _n(s["min"], d), _n(s["max"], d)]
            lines.append(f"| {label} | " + " | ".join(cells + tail) + " |")
        lines.append("")

    lines += ["## Paired comparisons for each seed (27_compare_maps.py)", "",
              "F1 difference, pooled, with the paired 95 % window interval. * means the "
              "interval excludes 0. Each seed's TerraMind is compared with the same seed's "
              "U-Net (the pairing by seed number is a convention; the seeds are not linked).",
              "",
              "| Comparison | Year | original | seed2 | seed3 | seeds with an interval "
              "above 0 | below 0 | including 0 |",
              "|---|---|---|---|---|---|---|---|"]
    for a, b, label in COMPARISONS:
        for year in mc.YEARS:
            cs = [comps[(a, b, year, t)] for t in TAGS]
            above = sum(1 for c in cs if c and c[1] is not None and c[1] > 0)
            below = sum(1 for c in cs if c and c[2] is not None and c[2] < 0)
            incl = sum(1 for c in cs if c and c[1] is not None and c[2] is not None
                       and c[1] <= 0 <= c[2])
            lines.append(f"| {label} | {year} | " + " | ".join(_signed(c) for c in cs)
                         + f" | {above} | {below} | {incl} |")
    lines += ["", "## Not there yet, and warnings", ""] + ([f"- {m}" for m in missing]
                                                             or ["- nothing missing"])
    md = paths.tables / "SEEDS_SUMMARY.md"
    tmp = md.with_name(md.name + ".part")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, md)
    return md, len(rows)


def main(argv=None):
    mc.setup_console()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mc.add_common_args(ap, seed=False, ram=False)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    paths = mc.paths_from_args(args)
    md = paths.tables / "SEEDS_SUMMARY.md"
    print(f"32_seed_summary.py{' (smoke)' if args.smoke else ''}: reading {paths.tables}")
    if mc.guard(md, args, "SEEDS_SUMMARY.md"):
        return 0
    runs, comps, missing = build(paths)
    md, n = write(paths, runs, comps, missing)
    print(f"  {n} result rows; {len(missing)} expected results not there yet")
    for m in missing:
        print(f"    {m}" if m.startswith("WARNING") else f"    not there yet: {m}")
    print(f"  wrote {md}")
    print("PASS")
    return 0


# =============================================================================
# Self-test
# =============================================================================

def _fake_tables(paths, f1s, with_seeds=True, lr=0.001):
    """Write the files 24, 04, 27 and the training scripts would write."""
    t = paths.tables
    head = ("table,reach,dimension,stratum,merged_from,pixels,flood_pixels_truth,"
            "windows_with_flood,tp,fp,fn,tn,precision,recall,f1,f1_lo,f1_hi,iou,iou_lo,"
            "iou_hi,csi,mcc,mcc_lo,mcc_hi,overall_accuracy,thin")
    tags = TAGS if with_seeds else ("",)
    for i, tag in enumerate(tags):
        for model in MODELS:
            run = _run_name(model, tag)
            d = paths.model_dir(model, "hand", tag)
            d.mkdir(parents=True, exist_ok=True)
            (d / "run_record.json").write_text(json.dumps(
                {"seed": mc.SEED + i, "finished": True, "best_epoch": 5 + i,
                 "threshold": 0.4, "train_seconds": 3600.0,
                 "options": {"seed": mc.SEED + i, "lr": lr if (model == "unet" and i == 2)
                             else 0.001, "epochs": 100}}), encoding="utf-8")
            (t / f"s1f11_{run}").mkdir(parents=True, exist_ok=True)
            (t / f"s1f11_{run}" / "metrics.csv").write_text(
                "split,threshold_kind,iou\n" + "".join(
                    f"{s},valid_chosen,{0.6 + 0.01 * i}\n{s},default_0.5,0.1\n"
                    for s in SPLITS), encoding="utf-8")
            for year in mc.YEARS:
                f1 = f1s[(model, i, year)]
                (t / f"{year}_{run}").mkdir(parents=True, exist_ok=True)
                (t / f"{year}_{run}" / "metrics_by_stratum.csv").write_text(
                    head + "\n" + f"reported,pooled,ALL,both reaches pooled,,100,10,3,5,5,5,85,"
                    f"0.5,0.5,{f1},{f1 - 0.05},{f1 + 0.05},0.3,0.2,0.4,0.3,0.4,0.3,0.5,0.9,"
                    "False\n" + "reported,gorge,reach,gorge,,50,5,2,1,1,1,1,0.5,0.5,0.9,0.8,"
                    "1.0,0.3,0.2,0.4,0.3,0.4,0.3,0.5,0.9,False\n", encoding="utf-8")
        for a, b, _ in COMPARISONS:
            for year in mc.YEARS:
                d = compare_dir(t, year, a, b, tag)
                d.mkdir(parents=True, exist_ok=True)
                (d / "comparison.csv").write_text(
                    "reach,dimension,diff_f1,diff_f1_lo,diff_f1_hi\n"
                    f"pooled,ALL,{0.01 * (i - 1)},{-0.02},{0.03}\n"
                    "gorge,reach,0.5,0.4,0.6\n", encoding="utf-8")


def selftest():
    print("32_seed_summary.py self-test")
    f1s = {(m, i, y): base + 0.01 * i for m, base in (("unet", 0.40), ("tm_s1t", 0.42))
           for i in range(3) for y in mc.YEARS}
    with tempfile.TemporaryDirectory(prefix="fyp selftest ", ignore_cleanup_errors=True) as td:
        paths = mc.Paths(Path(td) / "Final Year Project")
        _fake_tables(paths, f1s, with_seeds=False)
        ok1 = main(["--project", str(paths.root)]) == 0
        text1 = (paths.tables / "SEEDS_SUMMARY.md").read_text(encoding="utf-8")
        refuses = mc.run_script("32_seed_summary.py", ["--project", str(paths.root)]) == 1
        _fake_tables(paths, f1s, with_seeds=True)
        ok2 = main(["--project", str(paths.root), "--force"]) == 0
        text2 = (paths.tables / "SEEDS_SUMMARY.md").read_text(encoding="utf-8")
        _fake_tables(paths, f1s, with_seeds=True, lr=0.0003)
        main(["--project", str(paths.root), "--force"])
        text3 = (paths.tables / "SEEDS_SUMMARY.md").read_text(encoding="utf-8")
        _fake_tables(paths, f1s, with_seeds=True)
        with open(paths.tables / "SEEDS_SUMMARY.csv", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    mean = [r for r in rows if r["kind"] == "across seeds, mean" and r["model"] == "unet"
            and r["metric"] == "f1_2021"]
    sd = [r for r in rows if r["kind"] == "across seeds, sd" and r["model"] == "unet"
          and r["metric"] == "f1_2021"]
    checks = [
        ("runs with only the originals", ok1),
        ("lists the missing repeat runs", "unet_hand_seed2: not trained" in text1),
        ("refuses to overwrite without --force", refuses),
        ("runs with all three seeds", ok2 and "nothing missing" in text2),
        ("mean of 0.40, 0.41, 0.42 is 0.41",
         len(mean) == 1 and abs(float(mean[0]["value"]) - 0.41) < 1e-9),
        ("sample SD of 0.40, 0.41, 0.42 is 0.01",
         len(sd) == 1 and abs(float(sd[0]["value"]) - 0.01) < 1e-9),
        ("pooled row read, not the gorge row", "0.900" not in text2.split("## TerraMind")[0]),
        ("comparison counts: all three intervals include 0",
         "| TerraMind tm_s1t minus U-Net | 2021 | " in text2 and "| 0 | 0 | 3 |" in text2),
        ("a changed learning rate is flagged",
         "WARNING unet_hand_seed3 was not trained with the original settings: lr 0.001 -> "
         "0.0003" in text3 and "WARNING" not in text2),
        ("seed numbers shown", f"| Seed | {mc.SEED} | {mc.SEED + 1} | {mc.SEED + 2} |" in text2),
    ]
    for name, c in checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {name}")
    good = all(c for _, c in checks)
    print("SELFTEST PASS" if good else "SELFTEST FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(mc.cli(main))
