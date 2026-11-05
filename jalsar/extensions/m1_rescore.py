"""Jal-SAR extensions, item M1: hit rate, false alarm ratio and CSI for every scored map.

Re-scores every flood map that already has a table in 06 Outputs\\tables, with the
updated 04_stratified_metrics.py, into a separate folder 06 Outputs\\tables_m1.
The original tables are never touched.

Then it checks that every column that existed before came out exactly the same
(so the new script changed nothing it already reported), and writes
06 Outputs\\tables_m1\\HYDRO_SCORES.md and .csv: hit rate, false alarm ratio and
CSI with their 95 per cent window-bootstrap intervals, for Table A, the seed
runs, Table D and the baselines, pooled and by reach; plus the Sen1Floods11 test
scores (no intervals there, as before).

Run from M1 HIT RATE AND CSI.bat (env fyp). Re-running skips maps already done.
"""
import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]          # ...\Final Year Project
CODE = ROOT / "04 Code" / "FYP_code"
TABLES = ROOT / "06 Outputs" / "tables"
OUT = ROOT / "06 Outputs" / "tables_m1"
CHECK = ("2023_unet_hand", "both reaches pooled", {"csi": "0.3017", "hit_rate": "0.4261", "far": "0.4918"})


def pred_for(name):
    """Return (year, flood map path relative to FYP_code, group) or None."""
    m = re.fullmatch(r"(2021|2023)_single", name)
    if m:
        y = m[1]
        return y, f"data/flood_{y}_v3/flood_{y}_otsu.tif", "Baseline"
    m = re.fullmatch(r"(2021|2023)_change_(nov|dec|jan|feb)", name)
    if m:
        y, mo = m[1], m[2]
        return y, f"data/flood_{y}_ch_{mo}_v3/flood_{y}ch{mo}_otsu.tif", "Baseline"
    m = re.fullmatch(r"lowo_(2021|2023)_(\w+)_(grd|rtc)", name)
    if m:
        return m[1], f"data/{name}/flood_{name}.tif", "Table D"
    m = re.fullmatch(r"(2021|2023)_(rf|unet|tm_s1t|tm_s1)_(hand\w*)", name)
    if m:
        group = "Seeds" if "seed" in name else "Table A"
        return m[1], f"data/flood_{name}/flood_{name}.tif", group
    return None


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def label_of(name):
    md = TABLES / name / "metrics_summary.md"
    first = md.read_text(encoding="utf-8").splitlines()[0]
    return first.replace("# Stratified accuracy:", "").strip() or name


def ci(r, k):
    lo, hi = r.get(f"{k}_lo", ""), r.get(f"{k}_hi", "")
    if lo in ("", "nan", None):
        return r[k]
    return f"{r[k]} [{lo}, {hi}]"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = []
    for d in sorted(p for p in TABLES.iterdir() if p.is_dir()):
        info = pred_for(d.name)
        if info and (d / "metrics_by_stratum.csv").exists():
            runs.append((d.name,) + info)
    print(f"  {len(runs)} scored maps found in 06 Outputs\\tables")

    problems = []
    for i, (name, year, pred, group) in enumerate(runs, 1):
        dest = OUT / name
        new_csv = dest / "metrics_by_stratum.csv"
        if not (CODE / pred).exists():
            problems.append(f"{name}: flood map not found ({pred})")
            print(f"  [{i}/{len(runs)}] {name}: PROBLEM flood map not found")
            continue
        if new_csv.exists():
            print(f"  [{i}/{len(runs)}] {name}: already done, skipped")
        else:
            print(f"  [{i}/{len(runs)}] {name}: scoring ...", flush=True)
            dest.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "04_stratified_metrics.py", "--pred", pred,
                   "--truth", f"data/reference_{year}.tif", "--strata", f"data/strata_{year}_v3",
                   "--windows", f"data/reference_windows_{year}/reference_windows.geojson",
                   "--label", label_of(name), "--out", str(dest)]
            with open(dest / "run_log.txt", "w", encoding="utf-8") as log:
                rc = subprocess.run(cmd, cwd=CODE, stdout=log, stderr=subprocess.STDOUT).returncode
            if rc != 0 or not new_csv.exists():
                problems.append(f"{name}: 04 stopped (see tables_m1\\{name}\\run_log.txt)")
                print(f"           PROBLEM: 04 stopped, see tables_m1\\{name}\\run_log.txt")
                continue
        # Every old column must come out exactly the same.
        old = read_rows(TABLES / name / "metrics_by_stratum.csv")
        new = read_rows(new_csv)
        if len(old) != len(new):
            problems.append(f"{name}: {len(old)} rows before, {len(new)} now")
            continue
        diffs = [(k, o[k], n.get(k)) for o, n in zip(old, new) for k in o if o[k] != n.get(k)]
        if diffs:
            k, a, b = diffs[0]
            problems.append(f"{name}: {len(diffs)} old values changed (first: {k} {a} -> {b})")

    # Summary tables.
    groups = {"Table A": [], "Seeds": [], "Table D": [], "Baseline": []}
    flat = []
    for name, year, pred, group in runs:
        f = OUT / name / "metrics_by_stratum.csv"
        if not f.exists():
            continue
        for r in read_rows(f):
            if r["table"] == "reported" and r["dimension"] in ("ALL", "reach"):
                groups[group].append((name, r))
                flat.append({"group": group, "run": name, "row": r["stratum"],
                             **{k: r.get(k, "") for k in ("hit_rate", "hit_rate_lo", "hit_rate_hi",
                                                          "far", "far_lo", "far_hi", "csi", "csi_lo",
                                                          "csi_hi", "f1", "flood_pixels_truth",
                                                          "windows_with_flood", "tp", "fp", "fn")}})

    lines = ["# Hit rate, false alarm ratio and CSI (Stage M1)", "",
             "Hit rate (probability of detection) = TP / (TP + FN), the same as recall.",
             "False alarm ratio = FP / (TP + FP), that is 1 minus precision.",
             "Critical success index (CSI) = TP / (TP + FP + FN), the same as IoU.", "",
             "Brackets: 95 per cent interval from resampling the 8 digitising windows,",
             "2000 times (seed 20261007), the same resamples as every other interval.",
             "No interval where the reference flood sits in fewer than 3 windows.",
             "Report the two reaches separately; pooled rows are for checking.", ""]
    titles = {"Table A": "Table A: trained on Sen1Floods11, applied to the corridor (GRD)",
              "Seeds": "Seed runs (Table A, seeds 2 and 3)",
              "Table D": "Table D: leave one window out on the corridor (GRD and RTC)",
              "Baseline": "Baselines: change detection and single-image Otsu"}
    for g, rows in groups.items():
        if not rows:
            continue
        lines += [f"## {titles[g]}", "",
                  "| Run | Row | Hit rate [95% CI] | False alarm ratio [95% CI] | CSI [95% CI] | F1 | Ref. flood px |",
                  "|---|---|---|---|---|---|---|"]
        for name, r in rows:
            lines.append(f"| {name} | {r['stratum']} | {ci(r, 'hit_rate')} | {ci(r, 'far')} | "
                         f"{ci(r, 'csi')} | {r['f1']} | {r['flood_pixels_truth']} |")
        lines.append("")

    lines += ["## Sen1Floods11 (water, not flood; threshold chosen on the validation chips)", "",
              "| Run | Split | Hit rate | False alarm ratio | CSI |", "|---|---|---|---|---|"]
    for d in sorted(TABLES.glob("s1f11_*")):
        f = d / "metrics.csv"
        if not f.exists():
            continue
        for r in read_rows(f):
            if r["threshold_kind"] != "valid_chosen":
                continue
            tp, fp, fn = (int(r[k]) for k in ("tp", "fp", "fn"))
            hr = round(tp / (tp + fn), 4) if tp + fn else float("nan")
            fa = round(fp / (tp + fp), 4) if tp + fp else float("nan")
            cs = round(tp / (tp + fp + fn), 4) if tp + fp + fn else float("nan")
            lines.append(f"| {d.name} | {r['split']} | {hr} | {fa} | {cs} |")
            flat.append({"group": "Sen1Floods11", "run": d.name, "row": r["split"],
                         "hit_rate": hr, "far": fa, "csi": cs, "tp": tp, "fp": fp, "fn": fn})
    lines.append("")

    (OUT / "HYDRO_SCORES.md").write_text("\n".join(lines), encoding="utf-8")
    cols = ["group", "run", "row", "hit_rate", "hit_rate_lo", "hit_rate_hi", "far", "far_lo",
            "far_hi", "csi", "csi_lo", "csi_hi", "f1", "flood_pixels_truth", "windows_with_flood",
            "tp", "fp", "fn"]
    with open(OUT / "HYDRO_SCORES.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(flat)

    # Hand check against the worked example in the methods explainer.
    run, row, want = CHECK
    got = next((r for g in groups.values() for n, r in g if n == run and r["stratum"] == row), None)
    print()
    if got is None:
        problems.append(f"check row {run} / {row} not found")
    else:
        for k, v in want.items():
            ok = got[k] == v
            print(f"  check {run}, {row}: {k} = {got[k]} (expected {v}) {'OK' if ok else 'WRONG'}")
            if not ok:
                problems.append(f"check {k}: {got[k]}, expected {v}")

    print()
    print(f"  wrote {OUT / 'HYDRO_SCORES.md'}")
    print(f"  wrote {OUT / 'HYDRO_SCORES.csv'}")
    if problems:
        print("\n  PROBLEMS:")
        for p in problems:
            print(f"    - {p}")
        sys.exit(1)
    print(f"\n  all {len(runs)} maps re-scored; every value that existed before is unchanged")


if __name__ == "__main__":
    main()
