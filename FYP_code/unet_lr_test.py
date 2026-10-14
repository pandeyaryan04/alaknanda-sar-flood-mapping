r"""
unet_lr_test.py
===============
A short test of the U-Net's learning rate before the full U-Net runs, on
the Sen1Floods11 valid chips only (never the corridor reference).

WHY
---
In the first run (7 October) the U-Net's valid IoU at its best threshold
stopped improving after about epoch 4 (0.59 to 0.62) while the training loss
kept falling. A lower learning rate may train more steadily. This tests it
instead of guessing.

WHAT IT DOES
------------
1. Trains the U-Net on 'hand' twice, with every setting of the real run
   except the learning rate: 1e-3 (the default) and 3e-4. Each stops after
   30 epochs (--stop-after-epochs, so the learning-rate schedule is the real
   100-epoch one), or earlier if early stopping triggers. Folders:
   data\models\lr_test\unet_hand_lr0.001 and ...\unet_hand_lr0.0003.
2. For each, the best valid IoU at the best threshold over its epochs, and
   how much that IoU moves over its last 10 epochs (standard deviation).
3. The rule, fixed before any result (7 October, 22:00): 3e-4 is chosen
   only if its best valid IoU is at least 0.01 higher than 1e-3's;
   otherwise 1e-3 stays.
4. Writes the chosen value to data\models\lr_test\lr_choice.txt (one number,
   chosen on 8 October 2026; RUN ALL MODELS.bat uses the value it chose,
   0.0003) and a table to
   06 Outputs\models\unet_lr_test.md.

A finished test run is kept; started again it continues where it stopped
(--resume), and once lr_choice.txt exists it only prints the choice.

USAGE  (Anaconda Prompt, fyp active, from 04 Code\FYP_code)
-----
    python unet_lr_test.py --selftest
    python unet_lr_test.py

About 30 to 45 minutes on the RTX 3050 (estimate). The last line says PASS.
"""

__stage__ = "Models: U-Net learning-rate test on the Sen1Floods11 valid chips"
__reads__ = ["data/models/chip_index.csv", "data/models/norm_stats.json", "Sen1Floods11 chips"]
__writes__ = ["data/models/lr_test/", "06 Outputs/models/unet_lr_test.md"]

import argparse
import csv
import math
import sys
import tempfile
from pathlib import Path

import models_common as mc

RATES = (1e-3, 3e-4)
EPOCHS = 30
MARGIN = 0.01


def run_dir(paths, lr):
    return paths.models / "lr_test" / f"unet_hand_lr{lr:g}"


def summary(history_csv):
    """Best valid IoU at the best threshold, its epoch, and the spread over the last 10 epochs."""
    with open(history_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    vals = [float(r["val_iou_at_best_threshold"]) for r in rows]
    vals = [v if math.isfinite(v) else -1.0 for v in vals]
    best = max(vals)
    last = vals[-10:]
    mean = sum(last) / len(last)
    sd = math.sqrt(sum((v - mean) ** 2 for v in last) / len(last))
    return {"epochs": len(rows), "best": best, "best_epoch": vals.index(best) + 1, "sd_last10": sd,
            "iou05_last": float(rows[-1]["val_iou_at_0.5"])}


def choose(results):
    """The rule of step 3: 3e-4 only if at least MARGIN better than 1e-3."""
    return RATES[1] if results[RATES[1]]["best"] >= results[RATES[0]]["best"] + MARGIN else RATES[0]


def main(argv=None):
    mc.setup_console()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=None)
    ap.add_argument("--epochs", type=int, default=EPOCHS, help="epochs per test run (default 30)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--extra", default="", help=argparse.SUPPRESS)   # self-test settings
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    paths = mc.Paths(args.project)
    choice_p = paths.models / "lr_test" / "lr_choice.txt"
    print("unet_lr_test.py: U-Net learning rate, 1e-3 against 3e-4, on the valid chips")
    if choice_p.exists():
        print(f"  already decided: learning rate {choice_p.read_text(encoding='utf-8').strip()} "
              f"({choice_p})")
        print("PASS")
        return 0
    results = {}
    for lr in RATES:
        d = run_dir(paths, lr)
        print(f"\n  ---- learning rate {lr:g}: {args.epochs} epochs into {d}", flush=True)
        argv2 = ["--trainset", "hand", "--lr", f"{lr:g}", "--model-dir", str(d), "--resume",
                 "--stop-after-epochs", str(args.epochs)] + args.extra.split()
        if args.project:
            argv2 = ["--project", args.project] + argv2
        hist = d / "history.csv"
        done = hist.exists() and summary(hist)["epochs"] >= args.epochs
        if not done and not (d / "model_meta.json").exists():
            if mc.run_script("23_train_unet.py", argv2) != 0:
                mc.fail(f"The test run at learning rate {lr:g} stopped. Read the PROBLEM line "
                        "above; start this again and it continues.")
        if not hist.exists():
            mc.fail(f"No history.csv in {d}. Start this again.")
        results[lr] = summary(hist)
    pick = choose(results)
    lines = ["# U-Net learning-rate test", "",
             f"Sen1Floods11 hand training set, {args.epochs} epochs at most per run, valid chips "
             "only. Rule fixed before the results: 3e-4 only if its best valid IoU (at the best "
             f"threshold) is at least {MARGIN} higher than 1e-3's.", "",
             "| Learning rate | Epochs run | Best valid IoU (best threshold) | Best epoch | "
             "Spread, last 10 epochs (sd) | Valid IoU at 0.5, last epoch |",
             "|---|---|---|---|---|---|"]
    for lr in RATES:
        r = results[lr]
        lines.append(f"| {lr:g} | {r['epochs']} | {r['best']:.4f} | {r['best_epoch']} | "
                     f"{r['sd_last10']:.4f} | {r['iou05_last']:.4f} |")
    lines += ["", f"**Chosen: {pick:g}.**", ""]
    report = paths.out / "models" / "unet_lr_test.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    choice_p.parent.mkdir(parents=True, exist_ok=True)
    choice_p.write_text(f"{pick:g}\n", encoding="utf-8")
    print()
    for ln in lines[4:]:
        print("  " + ln)
    print(f"  wrote {report}")
    print(f"PASS  learning rate {pick:g} for the U-Net runs")
    return 0


def selftest():
    import models_synthetic as syn
    print("unet_lr_test.py self-test")
    fast = ("--base-width 8 --crop 32 --batch-size 4 --steps-per-epoch 3 --tile 64 "
            "--overlap 16 --workers 0 --cpu-ok")
    with tempfile.TemporaryDirectory(prefix="fyp selftest ", ignore_cleanup_errors=True) as td:
        proj = Path(td) / "Final Year Project"
        syn.build_project(proj, corridor=False)
        ok = mc.run_script("21_prepare_chips.py", ["--project", str(proj)]) == 0
        ok &= main(["--project", str(proj), "--epochs", "2", "--extra", fast]) == 0
        paths = mc.Paths(proj)
        choice = (paths.models / "lr_test" / "lr_choice.txt").read_text(encoding="utf-8").strip()
        runs = all(summary(run_dir(paths, lr) / "history.csv")["epochs"] == 2 for lr in RATES)
        again = main(["--project", str(proj), "--epochs", "2", "--extra", fast]) == 0
        real = not paths.model_dir("unet", "hand").exists()
    a = {RATES[0]: {"best": 0.60}, RATES[1]: {"best": 0.605}}
    b = {RATES[0]: {"best": 0.60}, RATES[1]: {"best": 0.611}}
    checks = [("both test runs trained 2 epochs", ok and runs),
              ("a choice was written", choice in ("0.001", "0.0003")),
              ("a second start only prints the choice", again),
              ("the real unet_hand folder untouched", real),
              ("rule: +0.005 keeps 1e-3, +0.011 takes 3e-4",
               choose(a) == RATES[0] and choose(b) == RATES[1])]
    for name, c in checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {name}")
    good = all(c for _, c in checks)
    print("SELFTEST PASS" if good else "SELFTEST FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(mc.cli(main))
