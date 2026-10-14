r"""
28_models_summary.py
====================
Collects every model result into one table: Table A (models trained on
Sen1Floods11, applied to the corridor GRD scenes, scored by 04), Table D
(leave one window out on the corridor, GRD and RTC), the Sen1Floods11
scores (24) and the paired comparisons (27).

It only reads what the other scripts wrote under 06 Outputs\tables and
lists every expected run that is not there yet, so it can be run at any
time. Writes 06 Outputs\tables\MODELS_SUMMARY.md and MODELS_SUMMARY.csv.

USAGE: HOW TO RUN THIS  (copy one line, paste into Anaconda Prompt, press Enter)
================================================================================
    conda activate fyp
    cd /d "<project folder>\04 Code\FYP_code"

    python 28_models_summary.py --selftest
    python 28_models_summary.py --force

Takes a few seconds. --force rewrites the summary (it is made from the
other files, so nothing is lost). The last line says PASS.
================================================================================
"""

__stage__ = "Models: summary of every result"
__reads__ = ["06 Outputs/tables/<year>_<model>_<trainset>/metrics_by_stratum.csv",
             "06 Outputs/tables/lowo_*/metrics_by_stratum.csv",
             "06 Outputs/tables/s1f11_*/metrics.csv", "06 Outputs/tables/compare_*/comparison.csv"]
__writes__ = ["06 Outputs/tables/MODELS_SUMMARY.md", "06 Outputs/tables/MODELS_SUMMARY.csv"]

import argparse
import csv
import os
import re
import sys
import tempfile
from pathlib import Path

import models_common as mc

COLS = ["table", "run", "year", "model", "variant", "row", "f1", "f1_lo", "f1_hi", "iou",
        "iou_lo", "iou_hi", "mcc", "precision", "recall", "reference_flood_pixels"]


def _csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def scored_rows(table, run, year, model, variant, path):
    out = []
    for r in _csv(path):
        if r.get("table") != "reported" or r.get("dimension") not in ("ALL", "reach"):
            continue
        out.append({"table": table, "run": run, "year": year, "model": model,
                    "variant": variant, "row": r["stratum"],
                    **{k: r.get(k, "") for k in ("f1", "f1_lo", "f1_hi", "iou", "iou_lo",
                                                  "iou_hi", "mcc", "precision", "recall")},
                    "reference_flood_pixels": r.get("flood_pixels_truth", "")})
    return out


def collect(tables):
    rows, missing = [], []
    for year in mc.YEARS:
        for model in mc.MODELS:
            for ts in mc.TRAINSETS:
                run = f"{year}_{model}_{ts}"
                f = tables / run / "metrics_by_stratum.csv"
                if f.exists():
                    rows += scored_rows("A", run, year, model, ts, f)
                else:
                    missing.append(f"Table A {run}")
            for radar in mc.RADARS:
                run = f"lowo_{year}_{model}_{radar}"
                f = tables / run / "metrics_by_stratum.csv"
                if f.exists():
                    rows += scored_rows("D", run, year, model, radar, f)
                else:
                    missing.append(f"Table D {run}")
    for model in mc.MODELS:
        for ts in mc.TRAINSETS:
            run = f"s1f11_{model}_{ts}"
            f = tables / run / "metrics.csv"
            if not f.exists():
                missing.append(f"Sen1Floods11 {run}")
                continue
            for r in _csv(f):
                if r["threshold_kind"] != "valid_chosen":
                    continue
                rows.append({"table": "S1F11", "run": run, "year": "", "model": model,
                             "variant": ts, "row": r["split"], "f1": r["f1"], "iou": r["iou"],
                             "precision": r["precision"], "recall": r["recall"],
                             "reference_flood_pixels": r["water_pixels"]})
    if tables.exists():
        for d in sorted(tables.glob("compare_*")):
            f = d / "comparison.csv"
            if not f.exists():
                continue
            r = next((q for q in _csv(f) if q["reach"] == "pooled" and q["dimension"] == "ALL"),
                     None)
            if r:
                m = re.match(r"compare_(\d{4})_", d.name)
                rows.append({"table": "compare", "run": d.name, "year": m.group(1) if m else "",
                             "row": "pooled, B minus A", "f1": r["diff_f1"],
                             "f1_lo": r["diff_f1_lo"], "f1_hi": r["diff_f1_hi"],
                             "iou": r["diff_iou"], "iou_lo": r["diff_iou_lo"],
                             "iou_hi": r["diff_iou_hi"], "mcc": r["diff_mcc"],
                             "reference_flood_pixels": r["flood_pixels_truth"]})
    return rows, missing


def _ci(r, k):
    lo, hi = r.get(f"{k}_lo", ""), r.get(f"{k}_hi", "")
    if lo in ("", None, "nan") or hi in ("", None, "nan"):
        return f"{r.get(k, '')}"
    return f"{r.get(k, '')} [{lo}, {hi}]"


def main(argv=None):
    mc.setup_console()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mc.add_common_args(ap, seed=False, ram=False)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    paths = mc.paths_from_args(args)
    md = paths.tables / "MODELS_SUMMARY.md"
    print(f"28_models_summary.py{' (smoke)' if args.smoke else ''}: reading {paths.tables}")
    if mc.guard(md, args, "MODELS_SUMMARY.md"):
        return 0
    rows, missing = collect(paths.tables)
    paths.tables.mkdir(parents=True, exist_ok=True)
    with open(paths.tables / "MODELS_SUMMARY.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    lines = ["# Models: summary of every result", "",
             "F1 and IoU are for flood (Table A, D, comparisons) or water (Sen1Floods11). "
             "Brackets: 95 % window-bootstrap intervals from 04 or 27.", ""]
    sections = [("A", "Table A: trained on Sen1Floods11, applied to the corridor GRD scenes"),
                ("D", "Table D: leave one window out on the corridor (GRD and RTC)")]
    for key, title in sections:
        rs = [r for r in rows if r["table"] == key]
        lines += [f"## {title}", ""]
        if not rs:
            lines += ["Nothing yet.", ""]
            continue
        lines += ["| Run | Row | F1 [95% CI] | IoU [95% CI] | MCC | Precision | Recall | "
                  "Ref. flood px |", "|---|---|---|---|---|---|---|---|"]
        for r in rs:
            lines.append(f"| {r['run']} | {r['row']} | {_ci(r, 'f1')} | {_ci(r, 'iou')} | "
                         f"{r['mcc']} | {r['precision']} | {r['recall']} | "
                         f"{r['reference_flood_pixels']} |")
        lines.append("")
    rs = [r for r in rows if r["table"] == "S1F11"]
    lines += ["## Sen1Floods11 (threshold chosen on valid; water class)", ""]
    if rs:
        lines += ["| Run | Split | IoU | F1 | Precision | Recall |", "|---|---|---|---|---|---|"]
        lines += [f"| {r['run']} | {r['row']} | {r['iou']} | {r['f1']} | {r['precision']} | "
                  f"{r['recall']} |" for r in rs]
    else:
        lines.append("Nothing yet.")
    rs = [r for r in rows if r["table"] == "compare"]
    lines += ["", "## Paired comparisons (B minus A, pooled)", ""]
    if rs:
        lines += ["| Comparison | F1 B-A [95% CI] | IoU B-A [95% CI] | MCC B-A |",
                  "|---|---|---|---|"]
        lines += [f"| {r['run']} | {_ci(r, 'f1')} | {_ci(r, 'iou')} | {r['mcc']} |" for r in rs]
    else:
        lines.append("Nothing yet.")
    lines += ["", "## Not run yet", ""] + ([f"- {m}" for m in missing] or ["- nothing missing"])
    tmp = md.with_name(md.name + ".part")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, md)
    print(f"  {len(rows)} result rows; {len(missing)} expected runs not there yet")
    print(f"  wrote {md}")
    print("PASS")
    return 0


def selftest():
    print("28_models_summary.py self-test")
    with tempfile.TemporaryDirectory(prefix="fyp selftest ", ignore_cleanup_errors=True) as td:
        proj = Path(td) / "Final Year Project"
        t = mc.Paths(proj).tables
        (t / "2021_rf_hand").mkdir(parents=True)
        head = ("table,reach,dimension,stratum,merged_from,pixels,flood_pixels_truth,"
                "windows_with_flood,tp,fp,fn,tn,precision,recall,f1,f1_lo,f1_hi,iou,iou_lo,"
                "iou_hi,csi,mcc,mcc_lo,mcc_hi,overall_accuracy,thin")
        body = ("reported,pooled,ALL,both reaches pooled,,100,10,3,5,5,5,85,0.5,0.5,0.5,0.4,0.6,"
                "0.333,0.3,0.4,0.333,0.44,0.3,0.5,0.9,False")
        (t / "2021_rf_hand" / "metrics_by_stratum.csv").write_text(head + "\n" + body + "\n")
        ok = main(["--project", str(proj)]) == 0
        text = (t / "MODELS_SUMMARY.md").read_text(encoding="utf-8")
        again = mc.run_script("28_models_summary.py", ["--project", str(proj)])
    checks = [("ran", ok), ("Table A row present", "| 2021_rf_hand | both reaches pooled | 0.5 [0.4, 0.6]" in text),
              ("missing runs listed", "Table A 2023_unet_hand_weak" in text),
              ("refuses to overwrite without --force", again == 1)]
    for name, c in checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {name}")
    good = all(c for _, c in checks)
    print("SELFTEST PASS" if good else "SELFTEST FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(mc.cli(main))
