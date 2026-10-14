r"""
29_terramind_preflight.py
=========================
Run this FIRST, before any TerraMind step, on the laptop that will train it.
It reads, runs a short test on the GPU, and prints one line per check

    [PASS] ...    fine
    [WARN] ...    unusual but TerraMind can run; read it
    [FAIL] ...    must be fixed first; the line says what to do

and a final verdict, TERRAMIND PREFLIGHT PASS or TERRAMIND PREFLIGHT FAIL. A
copy goes to 06 Outputs\models\terramind_preflight_<computer name>.md (keep
that file if anything fails). It never overwrites an earlier
report: a second run gets the time added to its name.

What it checks:
- terratorch imports in this environment; its version and those of torch,
  timm, einops, huggingface_hub and safetensors against the versions this
  code was built and tested with
- terratorch's TerraMind-small: its name in the backbone registry, the
  S1GRD and S1RTC inputs, and TerraMind's own radar normalisation values
- the weights file data\weights\TerraMind_v1_small.pt: there, its size and
  SHA-256, and how many of the encoder's tensors it fills, for the S1GRD and
  the S1RTC input (FAIL below 95 %)
- the GPU: both variants (tm_s1, tm_s1t) run forward and backward at the
  default batch (8) and crop (224) with mixed precision, with the peak
  memory (the plan is under 3.3 GB of the 4 GB)
- the speed in crops per second, and from it the time each RUN TERRAMIND
  .bat file should take
- the chip index, normalisation file and cache made by 21_prepare_chips.py,
  and free disk

USAGE: HOW TO RUN THIS  (copy one line, paste into Anaconda Prompt, press Enter)
================================================================================
    conda activate fyp
    cd /d "<project folder>\04 Code\FYP_code"

    python 29_terramind_preflight.py --selftest
    python 29_terramind_preflight.py

Takes 2 to 5 minutes. Or double-click RUN TERRAMIND PREFLIGHT.bat.
================================================================================
"""

__stage__ = "Models: TerraMind preflight"
__reads__ = ["the machine", "data/weights/TerraMind_v1_small.pt", "data/models/chip_index.csv",
             "data/models/norm_stats.json"]
__writes__ = ["06 Outputs/models/terramind_preflight_<hostname>.md"]

import argparse
import gc
import socket
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import models_common as mc

if mc.torch is None:
    sys.exit("\nPyTorch is not installed in this environment.\n"
             "Activate the environment first:  conda activate fyp\n")
import torch                                   # noqa: E402

VERSIONS = {"terratorch": "1.2.11", "torch": "2.6.0", "timm": "1.0.28", "einops": "0.8.2",
            "huggingface_hub": "1.28.0", "safetensors": "0.8.0"}
VRAM_LIMIT_GB = 3.3
WEIGHTS_MB = 504                               # the file on the model page (MB or MiB)
# what the .bat files run, for the time estimates (30 and 26 defaults)
TRAIN = {"hand": (100, 250), "hand_weak": (80, 1500)}       # epochs, steps per epoch
BATCH, VALID_CHIPS, TEST_CHIPS = 8, 89, 90 + 15
TILES_PER_CHIP = 9                             # a 512 x 512 chip in 224 tiles, overlap 64
CORRIDOR_TILES = 950                           # 7915 x 3044 cells in 224 tiles, overlap 64
LOWO_STEPS, LOWO_RUNS, LOWO_FOLDS = 1000, 8, 8
SCORE_MIN = 3.0                                # one 04_stratified_metrics.py run, minutes
SMOKE_STEPS, STARTUP_MIN = 13, 0.75            # RUN TERRAMIND SMOKE.bat: steps, start-up each


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, status, name, detail=""):
        self.rows.append({"status": status, "name": name, "detail": detail})
        print(f"  [{status}] {name}" + (f": {detail}" if detail else ""), flush=True)


def _stopped(e):
    """The sentence of a mc.fail() stop, without its PROBLEM: prefix."""
    return str(e.code if isinstance(e, SystemExit) else e).replace("PROBLEM:", "").strip()


# ============================================================================= software

def check_software(c):
    v = sys.version_info
    c.add("PASS" if v[:2] == (3, 11) else "WARN", "Python version",
          sys.version.split()[0] + ("" if v[:2] == (3, 11) else
                                    "; the fyp environment has 3.11. Run: conda activate fyp"))
    t = time.time()
    try:
        reg, tmr = mc.tm_import()
    except SystemExit as e:
        c.add("FAIL", "terratorch imports", _stopped(e))
        return False
    c.add("PASS", "terratorch imports", f"in {time.time() - t:.0f} s")
    for name, want in VERSIONS.items():
        have = mc._version(name)
        same = have.split("+")[0] == want
        c.add("PASS" if same else "WARN", f"package {name}", have if same else
              f"{have}; this code was built and tested with {want}. It may still work: the "
              "smoke run will tell")
    ok = mc.TM_BACKBONE in reg
    c.add("PASS" if ok else "FAIL", "TerraMind-small in terratorch's backbone registry",
          mc.TM_BACKBONE if ok else f"'{mc.TM_BACKBONE}' not found: terratorch is not 1.2.11. "
          "Reinstall terratorch 1.2.11.")
    if not ok:
        return False
    for modality in mc.TM_KEYS.values():
        if modality not in tmr.v1_pretraining_mean:
            c.add("FAIL", "TerraMind radar normalisation", f"no values for {modality} in "
                  "terratorch. Check the terratorch version (1.2.11).")
            return False
    for modality in mc.TM_KEYS:
        r = mc.tm_radar_norm(modality)
        c.add("PASS", f"TerraMind {modality} normalisation",
              f"VV, VH mean {r['mean']}, std {r['std']} (dB)")
    return True


# ============================================================================= weights

def check_weights(c, path):
    if not path.exists():
        c.add("FAIL", "TerraMind weights file", f"missing: {path}. Put TerraMind_v1_small.pt "
              "(504 MB, from huggingface.co/ibm-esa-geospatial/TerraMind-1.0-small) there.")
        return None
    mb = path.stat().st_size / 1e6
    near = 450 < mb < 560
    c.add("PASS" if near else "WARN", "TerraMind weights file",
          f"{path.name}, {mb:.0f} MB" + ("" if near else
                                         f"; the model page's file is about {WEIGHTS_MB} MB"))
    sha = mc.sha256(path)
    c.add("PASS", "TerraMind weights SHA-256", sha)
    for modality in mc.TM_KEYS:
        net = mc.TerraMindNet("tm_s1", modality)
        try:
            _, rep = mc.tm_read_weights(path, net.backbone)
        except SystemExit as e:
            c.add("FAIL", f"weights fill the encoder ({modality} input)", _stopped(e))
            return None
        lines = mc.tm_weights_lines(rep)
        c.add("PASS", f"weights fill the encoder ({modality} input)",
              lines[0].replace("TerraMind weights: ", "") + "; " + "; ".join(
                  s.strip() for s in lines[1:]))
    return sha


# ============================================================================= GPU and speed

def measure(variant, device, batch, crop, steps):
    """Seconds per training step and per prediction tile, and peak GPU memory."""
    amp = device.type == "cuda"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        net = mc.TerraMindNet(variant).to(device)
    opt = torch.optim.AdamW([{"params": list(net.backbone.parameters()), "lr": 5e-5},
                             {"params": net.decoder_parameters(), "lr": 1e-3}],
                            weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    x = torch.randn(batch, len(mc.CHANNELS), crop, crop, device=device)
    y = torch.randint(0, 2, (batch, crop, crop), device=device, dtype=torch.int8)

    def sync():
        if amp:
            torch.cuda.synchronize()

    def step():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            out = net(x)
        loss = mc.masked_bce_dice(out, y)[0]
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        return out

    if amp:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    out = step()
    shape_ok = tuple(out.shape) == (batch, crop, crop)
    for _ in range(2):
        step()
    sync()
    t = time.time()
    for _ in range(steps):
        step()
    sync()
    train_s = (time.time() - t) / steps
    peak = torch.cuda.max_memory_reserved() / 1e9 if amp else float("nan")
    net.eval()
    fn = mc.unet_prob_fn(net, device, amp)
    tiles = np.random.default_rng(0).normal(size=(2, len(mc.CHANNELS), crop, crop)).astype(
        "float32")
    fn(tiles)
    sync()
    t = time.time()
    for _ in range(max(steps // 2, 2)):
        fn(tiles)
    sync()
    tile_s = (time.time() - t) / (max(steps // 2, 2) * 2)
    return {"crops_per_s": batch / train_s, "tile_s": tile_s, "peak_gb": peak,
            "shape_ok": shape_ok}


def check_gpu(c, batch, crop, steps, cpu_ok):
    if not torch.cuda.is_available():
        if not cpu_ok:
            c.add("FAIL", "GPU", "no CUDA GPU (torch.cuda.is_available() is False). TerraMind "
                  "trains only on the RTX 3050 laptop: check that nvidia-smi works in Anaconda "
                  "Prompt, restart, and run this again.")
            return None
        c.add("WARN", "GPU", "none; measuring on the CPU because --cpu-ok was given (the "
              "times below are then the CPU's)")
        device = torch.device("cpu")
    else:
        g = mc.gpu_info()
        c.add("PASS" if g["gpu_vram_gb"] >= 3.5 else "WARN", "GPU",
              f"{g['gpu']}, {g['gpu_vram_gb']} GB, CUDA {torch.version.cuda}")
        device = torch.device("cuda")
        mc.set_deterministic(False)
    speed = {}
    for variant in mc.TM_VARIANTS:
        est = mc.terramind_memory_estimate(variant, batch, crop, device.type == "cuda")
        try:
            m = measure(variant, device, batch, crop, steps)
        except RuntimeError as e:
            if not (device.type == "cuda" and mc.is_oom(e)):
                c.add("FAIL", f"{variant}: forward and backward", f"{e}. Update the NVIDIA "
                      "driver, restart, and run this again; if it repeats, keep this report.")
                return None
            m = None
        if device.type == "cuda":                  # the measured network is gone now
            gc.collect()
            torch.cuda.empty_cache()
        if m is None:
            with torch.random.fork_rng(devices=[]):
                net = mc.TerraMindNet(variant).to(device)
            bs = mc.find_batch_size(net, device, crop, batch, True)
            del net
            torch.cuda.empty_cache()
            c.add("FAIL", f"{variant}: forward and backward at batch {batch}",
                  f"the GPU ran out of memory. --batch-size {bs} fits: add it to the "
                  "30_train_terramind.py and 26_corridor_lowo.py commands, or close the other "
                  "programs that use the GPU and run this again." if bs else
                  "the GPU ran out of memory even at batch 1. Close every other program that "
                  "uses the GPU and run this again.")
            return None
        where = "mixed precision on the GPU" if device.type == "cuda" else "CPU"
        c.add("PASS" if m["shape_ok"] else "FAIL", f"{variant}: forward and backward",
              f"batch {batch}, crop {crop}, {where}; output {crop} x {crop}" if m["shape_ok"]
              else "the output is not the input's size. Keep this report.")
        if device.type == "cuda":
            peak = m["peak_gb"]
            c.add("PASS" if peak <= VRAM_LIMIT_GB else "WARN", f"{variant}: GPU memory",
                  f"{peak:.2f} GB at most (estimate {est['estimate_gb']:.2f} GB, plan under "
                  f"{VRAM_LIMIT_GB} GB)" + ("" if peak <= VRAM_LIMIT_GB else
                                            "; it fits, but close other programs that use the "
                                            "GPU before a long run"))
        c.add("PASS", f"{variant}: speed", f"{m['crops_per_s']:.1f} training crops per second, "
              f"{m['tile_s'] * 1000:.0f} ms per prediction tile")
        speed[variant] = m
    return speed


def estimates(speed, batch=BATCH):
    """Hours per .bat from the measured speed at this batch: (what, hours, note) rows."""
    rows = []
    valid_tiles = VALID_CHIPS * TILES_PER_CHIP
    total_a = 0.0
    for variant in mc.TM_VARIANTS:
        cps, tile_s = speed[variant]["crops_per_s"], speed[variant]["tile_s"]
        for ts, (ep, steps) in TRAIN.items():
            train_h = ep * (steps * batch / cps + valid_tiles * tile_s + 5) / 3600
            other_h = ((VALID_CHIPS + TEST_CHIPS) * TILES_PER_CHIP * tile_s
                       + 2 * (CORRIDOR_TILES * tile_s + 90) + 2 * SCORE_MIN * 60 + 120) / 3600
            rows.append((f"Table A: {variant} {ts} (train at most {ep} epochs, evaluate, 2 maps, "
                         "2 scores)", train_h + other_h,
                         "less if early stopping triggers"))
            total_a += train_h + other_h
    lowo_h = 0.0
    for variant in mc.TM_VARIANTS:
        per_fold = LOWO_STEPS * batch / speed[variant]["crops_per_s"] + 40
        lowo_h += (LOWO_RUNS // 2) * (LOWO_FOLDS * per_fold + 60 + SCORE_MIN * 60) / 3600
    slow = min(s["crops_per_s"] for s in speed.values())
    smoke_min = SMOKE_STEPS * STARTUP_MIN + (3 * 4 * 4 + 2 * 10 * 4) / slow / 60
    return rows, total_a, lowo_h, smoke_min


# ============================================================================= data

def check_data(c, paths):
    for f, what in ((paths.chip_index, "chip index"), (paths.norm_stats, "normalisation file")):
        c.add("PASS" if f.exists() else "WARN", what, str(f) if f.exists() else
              f"not made yet ({f.name}). RUN ALL MODELS.bat makes it in its first step "
              "(21_prepare_chips.py)")
    if paths.chip_index.exists():
        index = mc.load_chip_index(paths)
        n = {s: len(mc.split_rows(index, s)) for s in ("train", "valid", "weak")}
        cached = sum(bool(r["cache_x"]) for r in index if r["kept"] == "yes"
                     and r["split"] in ("train", "valid", "weak"))
        c.add("PASS", "chips for TerraMind", f"hand train {n['train']}, valid {n['valid']}, "
              f"weak {n['weak']} kept")
        c.add("PASS" if cached else "WARN", "chip cache",
              f"{cached} chips cached" if cached else "no cache: training reads the .tif files "
              "and is slower. 21_prepare_chips.py --cache makes it")
    free = mc.disk_free_gb(paths.models)
    c.add("PASS" if free >= 5 else "FAIL", "free disk", f"{free:.0f} GB" if free >= 5 else
          f"{free:.0f} GB; TerraMind needs about 5 GB for checkpoints and maps. Free some space.")


# ============================================================================= main

def run_checks(paths, weights, batch=BATCH, crop=mc.TM_CROP, steps=20, cpu_ok=False):
    c = Checks()
    print("Software")
    if not check_software(c):
        return c.rows, None
    print("Weights")
    check_weights(c, weights)
    print("GPU and speed")
    speed = check_gpu(c, batch, crop, steps, cpu_ok)
    print("Data")
    check_data(c, paths)
    return c.rows, speed


def main(argv=None):
    mc.setup_console()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mc.add_common_args(ap, smoke=False, seed=False, ram=False)
    ap.add_argument("--weights", default=None,
                    help="TerraMind_v1_small.pt (default data\\weights\\TerraMind_v1_small.pt)")
    ap.add_argument("--batch-size", type=int, default=BATCH, help="default 8, as for training")
    ap.add_argument("--crop", type=int, default=mc.TM_CROP, help="default 224, as for training")
    ap.add_argument("--steps", type=int, default=20, help="timed training steps per variant")
    ap.add_argument("--cpu-ok", action="store_true",
                    help="measure on the CPU when there is no GPU (for testing only)")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.crop % 16 or args.crop < 16:
        mc.fail(f"--crop must be a multiple of 16; got {args.crop}.")
    paths = mc.paths_from_args(args)
    weights = Path(args.weights) if args.weights else paths.tm_weights()
    t0 = time.time()
    print(f"29_terramind_preflight.py on {socket.gethostname()}, project {paths.root}\n")
    rows, speed = run_checks(paths, weights, args.batch_size, args.crop, args.steps, args.cpu_ok)
    nf = sum(r["status"] == "FAIL" for r in rows)
    nw = sum(r["status"] == "WARN" for r in rows)
    verdict = ("TERRAMIND PREFLIGHT PASS" + (f" ({nw} warnings: read them)" if nw else "")) \
        if not nf else (f"TERRAMIND PREFLIGHT FAIL: {nf} problem(s). Fix the first FAIL line, "
                        "then run this again.")
    times = []
    if speed:
        rows_a, total_a, lowo_h, smoke_min = estimates(speed, args.batch_size)
        times = ["## Time estimates from the measured speed", "",
                 f"Measured at batch {args.batch_size}, crop {args.crop}. Data loading, the "
                 "laptop's temperature and other programs can make the real runs slower; "
                 "30_train_terramind.py prints the real ETA after its first epoch.", "",
                 "| What | Hours | Note |", "|---|---|---|",
                 f"| RUN TERRAMIND SMOKE.bat | {smoke_min / 60:.2f} | about {smoke_min:.0f} "
                 "minutes, most of it starting the 13 steps |"]
        times += [f"| {w} | {h:.1f} | {n} |" for w, h, n in rows_a]
        times += [f"| RUN ALL MODELS.bat, TerraMind Table A, all four | {total_a:.1f} | at most; "
                  "less with early stopping |",
                  f"| RUN ALL MODELS.bat, TerraMind Table D, 8 runs of 8 folds | {lowo_h:.1f} | both "
                  "variants, both years, GRD and RTC |"]
        print("\n" + "\n".join(times[4:]))
    out_dir = paths.o06 / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    host = socket.gethostname()
    rep = out_dir / f"terramind_preflight_{host}.md"
    if rep.exists() and not args.force:
        rep = out_dir / f"terramind_preflight_{host}_{datetime.now():%Y%m%d-%H%M%S}.md"
    lines = [f"# TerraMind preflight, {host}", "",
             f"{datetime.now():%Y-%m-%d %H:%M}, project `{paths.root}`, "
             f"{time.time() - t0:.0f} s", "", f"**{verdict}**", "",
             "| Result | Check | Detail |", "|---|---|---|"]
    lines += [f"| {r['status']} | {r['name']} | "
              f"{' '.join(r['detail'].replace('|', '/').split())} |" for r in rows]
    lines += [""] + times
    rep.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nreport: {rep}")
    print(verdict)
    return 0 if not nf else 1


def selftest():
    import models_synthetic as syn
    print("29_terramind_preflight.py self-test")
    small = ["--batch-size", "2", "--crop", "64", "--steps", "2", "--cpu-ok"]
    results = []
    with tempfile.TemporaryDirectory(prefix="fyp selftest ", ignore_cleanup_errors=True) as td:
        proj = Path(td) / "Final Year Project"
        syn.build_project(proj, corridor=False)
        syn.build_terramind_weights(proj)
        mc.run_script("21_prepare_chips.py", ["--project", str(proj)])
        rc = main(["--project", str(proj)] + small)
        reps = list((proj / "06 Outputs" / "models").glob("terramind_preflight_*.md"))
        text = reps[0].read_text(encoding="utf-8") if reps else ""
        results.append(("good data and weights pass", rc == 0 and "PREFLIGHT PASS" in text
                        and "Hours" in text))
        syn.break_terramind_weights(proj)
        rows, _ = run_checks(mc.Paths(proj), mc.Paths(proj).tm_weights(), 2, 64, 2, True)
        results.append(("a wrong weights file fails under 95 %",
                        any(r["status"] == "FAIL" and "at least 95%" in r["detail"]
                            for r in rows)))
        mc.Paths(proj).tm_weights().unlink()
        rows, _ = run_checks(mc.Paths(proj), mc.Paths(proj).tm_weights(), 2, 64, 2, True)
        results.append(("a missing weights file fails", any(
            r["status"] == "FAIL" and "missing" in r["detail"] for r in rows)))
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    good = all(ok for _, ok in results)
    print("SELFTEST PASS" if good else "SELFTEST FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(mc.cli(main))
