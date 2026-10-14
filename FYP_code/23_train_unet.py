r"""
23_train_unet.py
================
Trains the U-Net flood model on Sen1Floods11, in plain
PyTorch, from scratch (no pretrained weights).

WHAT IT DOES
------------
- Model: U-Net, 4 encoder levels + bottleneck, two 3x3 conv-BN-ReLU per
  block, base width 32 (7.8 million parameters), one output logit.
- Inputs: VV, VH, HAND, slope, normalised with data\models\norm_stats.json.
- Training: random 256 x 256 crops with flips and 90-degree turns; masked
  BCE + soft Dice on water; AdamW (lr 1e-3, weight decay 1e-4); cosine
  schedule with a short linear warm-up; mixed precision on the GPU.
- An epoch is a fixed number of steps: 250 for 'hand', 1500 for
  'hand_weak'. With hand_weak every batch is half hand chips, half weak.
- After every epoch: water IoU on the 89 hand valid chips (whole chips,
  tiled) at every threshold from 0.01 to 0.99. The best epoch, and early
  stopping after --patience epochs without improvement, go by the IoU at
  that epoch's best threshold, the one the corridor maps then use
  (--select-by best_threshold). The first run of 7 October went by the IoU
  at 0.5, which jumped between epochs as the best threshold drifted down to
  0.01 to 0.06; --select-by 0.5 reproduces it.
- Saves last.pt every epoch and best.pt when valid IoU improves, both with
  the optimiser, scaler, schedule, epoch and random states.
- Stop it any time with Ctrl+C (it saves first), or close the laptop: run
  the same command with --resume and it continues exactly where it stopped.

Defaults stay under about 2.2 GB of GPU memory (estimate printed at the
start). If the GPU still runs out of memory on the first batch, the script
finds a batch size that fits and tells you the exact --batch-size to use.

USAGE: HOW TO RUN THIS  (copy one line, paste into Anaconda Prompt, press Enter)
================================================================================
    conda activate fyp
    cd /d "<project folder>\04 Code\FYP_code"

    python 23_train_unet.py --selftest
    python 23_train_unet.py --trainset hand --resume
    python 23_train_unet.py --trainset hand_weak --resume

A repeat run with another seed goes to its own folder,
data\models\unet_hand_seed2, and never touches the original:

    python 23_train_unet.py --trainset hand --seed 20261008 --run-tag seed2 --resume

--resume continues an unfinished run and starts a new one if there is none.
Estimated time on the RTX 3050 4 GB: hand about 1 to 2 hours, hand_weak
about 6 to 10 hours (an estimate; the real figure is printed after epoch 1).
The last line says TRAINING FINISHED and PASS.
================================================================================
"""

__stage__ = "Models: U-Net training"
__reads__ = ["data/models/chip_index.csv", "data/models/norm_stats.json",
             "Sen1Floods11 chips or data/models/cache"]
__writes__ = ["data/models/unet_<trainset>/last.pt, best.pt, history.csv, model_meta.json, "
              "norm_stats.json, train_chips.txt, run_record.json"]

import argparse
import csv
import math
import os
import random
import shutil
import signal
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import models_common as mc

if mc.torch is None:
    sys.exit("\nPyTorch is not installed in this environment.\n"
             "Activate the environment first:  conda activate fyp\n")
import torch                                   # noqa: E402

DEFAULTS = {   # trainset: (epochs, steps per epoch)
    "hand": (100, 250),
    "hand_weak": (100, 1500),
}
SMOKE = {"epochs": 1, "steps_per_epoch": 4, "crop": 128, "batch_size": 4}
SAME_ON_RESUME = ("trainset", "epochs", "steps_per_epoch", "batch_size", "crop", "base_width",
                  "lr", "weight_decay", "seed", "hand_fraction", "pos_weight", "warmup_steps",
                  "select_by")
HIST_COLS = ["epoch", "train_loss", "train_bce", "train_dice", "val_iou_at_0.5",
             "val_f1_at_0.5", "val_best_threshold", "val_iou_at_best_threshold", "lr", "improved"]


class StopFlag:
    """First Ctrl+C: finish the step, save, stop. Second Ctrl+C: stop at once."""

    def __init__(self):
        self.flag = False
        try:
            self.old = signal.signal(signal.SIGINT, self._handler)
        except ValueError:                      # not the main thread
            self.old = None

    def _handler(self, signum, frame):
        if self.flag:
            raise KeyboardInterrupt
        self.flag = True
        print("\nCtrl+C: finishing this step, saving a checkpoint, then stopping. "
              "(Ctrl+C again stops at once without saving.)", flush=True)

    def restore(self):
        if self.old is not None:
            signal.signal(signal.SIGINT, self.old)


def lr_schedule(warmup, total):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = min((step - warmup) / max(total - warmup, 1), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    return f


def save_atomic(obj, path):
    tmp = path.with_name(path.name + ".part")
    torch.save(obj, tmp)
    for attempt in range(5):                     # Windows: a virus scanner may hold the file
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(1 + attempt)
    os.replace(tmp, path)


def validate(model, loader, device, amp, tile, overlap):
    model.eval()
    hist = mc.ThresholdHist()
    fn = mc.unet_prob_fn(model, device, amp)
    for xn, valid, y, _chip in loader:
        prob, _ = mc.predict_tiled(fn, xn, valid, tile, overlap, batch=2)
        tv = valid & (y >= 0)
        hist.add(prob[tv], y[tv] == 1)
    model.train()
    thr, iou_thr = hist.best()
    m05 = mc.metrics_from_confusion(*hist.counts(0.5))
    return {"iou05": m05["iou"], "f1_05": m05["f1"], "thr": thr, "iou_thr": iou_thr}


def resolve(args):
    ep, st = DEFAULTS[args.trainset]
    if args.smoke:
        for k, v in SMOKE.items():
            if getattr(args, k) is None:
                setattr(args, k, v)
    args.epochs = args.epochs or ep
    args.steps_per_epoch = args.steps_per_epoch or st
    args.batch_size = args.batch_size or 8
    args.crop = args.crop or 256
    if args.crop % 16 or args.crop < 16:
        mc.fail(f"--crop must be a multiple of 16; got {args.crop}.")
    mc.check_tile_args(args.tile, args.overlap)
    if args.batch_size < 1 or args.steps_per_epoch < 1 or args.epochs < 1:
        mc.fail("--batch-size, --steps-per-epoch and --epochs must be at least 1.")
    total = args.epochs * args.steps_per_epoch
    if args.warmup_steps is None:
        args.warmup_steps = max(1, min(500, total // 20))
    return args


def pos_weight_value(spec, rows):
    if spec == "none":
        return None
    if spec == "auto":
        w = sum(r["labelled_pixels"] * r["water_share"] for r in rows)
        n = sum(r["labelled_pixels"] for r in rows)
        return float((n - w) / max(w, 1.0))
    return float(spec)


def write_history(path, history):
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=HIST_COLS + ["seconds"], extrasaction="ignore")
        w.writeheader()
        w.writerows(history)
    os.replace(tmp, path)


def main(argv=None):
    mc.setup_console()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mc.add_common_args(ap)
    ap.add_argument("--trainset", choices=mc.TRAINSETS, default="hand")
    ap.add_argument("--epochs", type=int, default=None, help="default 100 (hand and hand_weak)")
    ap.add_argument("--steps-per-epoch", type=int, default=None,
                    help="batches per epoch; default 250 hand, 1500 hand_weak")
    ap.add_argument("--batch-size", type=int, default=None, help="default 8")
    ap.add_argument("--crop", type=int, default=None, help="training crop side, default 256")
    ap.add_argument("--workers", type=int, default=2, help="data loading processes; 0 works")
    ap.add_argument("--resume", action="store_true",
                    help="continue from last.pt if it exists (starts fresh if not)")
    ap.add_argument("--base-width", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=None,
                    help="linear warm-up steps (default 5 percent of all steps, at most 500)")
    ap.add_argument("--patience", type=int, default=20,
                    help="stop after this many epochs without a better valid IoU")
    ap.add_argument("--select-by", choices=["best_threshold", "0.5"], default="best_threshold",
                    help="the valid IoU that picks the best epoch and drives early stopping: "
                         "at the best threshold of that epoch (default), or at 0.5 (the first "
                         "U-Net run of 7 October)")
    ap.add_argument("--hand-fraction", type=float, default=0.5,
                    help="hand_weak: share of each batch taken from hand chips")
    ap.add_argument("--pos-weight", default="none",
                    help="BCE weight of water: none (default), auto (from the training water "
                         "share) or a number")
    ap.add_argument("--deterministic", action="store_true",
                    help="deterministic algorithms (slower); cudnn.benchmark off")
    ap.add_argument("--cpu-ok", action="store_true", help="allow training on the CPU (very slow)")
    ap.add_argument("--tile", type=int, default=512, help="validation tile side")
    ap.add_argument("--overlap", type=int, default=64, help="validation tile overlap")
    ap.add_argument("--model-dir", default=None,
                    help="folder for this run (default data\\models\\unet_<trainset>); "
                         "unet_lr_test.py uses it for its test runs")
    mc.add_run_tag_arg(ap)
    ap.add_argument("--stop-after-epochs", type=int, default=0,
                    help="stop (resumable) after this many epochs in all; 0 = never")
    ap.add_argument("--stop-after-steps", type=int, default=0,
                    help="stop (resumable) after this many steps in all, as Ctrl+C would; 0 = never")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    args = resolve(args)
    tag = mc.check_run_tag(args)
    if tag and args.model_dir:
        mc.fail("Use --run-tag or --model-dir, not both.")
    paths = mc.paths_from_args(args)
    out = Path(args.model_dir).resolve() if args.model_dir else paths.model_dir("unet",
                                                                               args.trainset, tag)
    rec_p, last_p, best_p = out / "run_record.json", out / "last.pt", out / "best.pt"
    print(f"23_train_unet.py: U-Net, trainset {args.trainset}"
          + (f", repeat run '{tag}' with seed {args.seed}" if tag else "")
          + (' (smoke)' if args.smoke else ''))

    if rec_p.exists() and mc.load_json(rec_p).get("finished"):
        if mc.guard(rec_p, args, f"The U-Net '{args.trainset}'"):
            return 0
    resume = args.resume and last_p.exists() and not args.force
    if last_p.exists() and not resume and not args.force:
        mc.fail(f"An unfinished U-Net run exists in\n  {out}\n"
                "Add --resume to continue it, or --force to start again from scratch.")
    if args.resume and not last_p.exists():
        print("  no checkpoint to resume from: starting a new run")

    device = mc.pick_device(args.cpu_ok or args.smoke)
    amp = device.type == "cuda"
    mc.set_deterministic(args.deterministic)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    index = mc.load_chip_index(paths)
    hand, weak = mc.training_rows(index, args.trainset)
    valid_rows = mc.split_rows(index, "valid")
    if not hand:
        mc.fail("No hand train chips in the chip index. Run 21_prepare_chips.py again.")
    out.mkdir(parents=True, exist_ok=True)
    mc.check_disk(out, 1.0, "U-Net checkpoints")
    if resume:
        stats = mc.load_json(out / "norm_stats.json")
        if paths.norm_stats.exists() and \
                mc.sha256(out / "norm_stats.json") != mc.sha256(paths.norm_stats):
            mc.fail("norm_stats.json changed since this run started (was 21_prepare_chips.py "
                    "run again?). Resuming would mix two normalisations. Start again with --force.")
    else:
        stats = mc.load_norm_stats(paths.norm_stats)
        shutil.copy2(paths.norm_stats, out / "norm_stats.json")
    pos_weight = pos_weight_value(args.pos_weight, hand + weak)
    est = mc.unet_memory_estimate(args.base_width, args.batch_size, args.crop, amp)
    print(f"  device {device}{' (mixed precision)' if amp else ''}; {len(hand)} hand chips"
          + (f" + {len(weak)} weak chips, {args.hand_fraction:.0%} of each batch from hand"
             if weak else "") + f"; {len(valid_rows)} valid chips")
    print(f"  U-Net base width {args.base_width}: {est['parameters']:,} parameters; "
          f"GPU memory estimate at batch {args.batch_size}, crop {args.crop}: "
          f"{est['estimate_gb']:.2f} GB")
    print(f"  {args.epochs} epochs x {args.steps_per_epoch} steps x batch {args.batch_size}; "
          f"warm-up {args.warmup_steps} steps; patience {args.patience}")

    ds = mc.TrainCrops(hand + weak, paths.data, stats, args.crop)
    sampler = mc.StepBatchSampler(len(hand), len(weak), args.batch_size, args.steps_per_epoch,
                                  args.seed, args.hand_fraction)
    loader = mc.make_loader(ds, args.workers, args.seed, batch_sampler=sampler, cuda=amp)
    vloader = mc.make_loader(mc.FullChips(valid_rows, paths.data, stats), min(args.workers, 2),
                             args.seed + 1, cuda=False)

    model = mc.UNet(len(mc.CHANNELS), args.base_width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    total = args.epochs * args.steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule(args.warmup_steps, total))
    st = {"epoch": 0, "step_in_epoch": 0, "global_step": 0, "best_iou": -1.0, "best_epoch": 0,
          "best_threshold": 0.5, "best_iou_threshold": float("nan"), "bad_epochs": 0,
          "best_score": -1.0,
          "history": [], "epoch_seconds": [], "sums": [0.0, 0.0, 0.0, 0], "first_step_loss": None,
          "train_seconds": 0.0, "early_stopped": False}
    if resume:
        ck = torch.load(last_p, map_location="cpu", weights_only=False)     # our own file
        diff = {k: (ck["args"].get(k), getattr(args, k)) for k in SAME_ON_RESUME
                if ck["args"].get(k) != getattr(args, k)}
        if diff:
            mc.fail("--resume needs the same settings as the run it continues. Different: "
                    + "; ".join(f"--{k.replace('_', '-')} was {a}, now {b}"
                                for k, (a, b) in diff.items())
                    + ". Run the same command as before, or --force to start again.")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if ck["scaler"] and amp:            # empty when the run started on a CPU
            scaler.load_state_dict(ck["scaler"])
        sched.load_state_dict(ck["scheduler"])
        for k in st:
            st[k] = ck.get(k, st[k])
        random.setstate(ck["rng"]["python"])
        np.random.set_state(ck["rng"]["numpy"])
        torch.set_rng_state(ck["rng"]["torch"])
        if amp and ck["rng"].get("cuda"):
            torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        print(f"  resumed from last.pt: {st['epoch']} epochs done"
              + (f", {st['step_in_epoch']} steps into the next" if st["step_in_epoch"] else ""))

    def checkpoint():
        rng = {"python": random.getstate(), "numpy": np.random.get_state(),
               "torch": torch.get_rng_state(),
               "cuda": torch.cuda.get_rng_state_all() if amp else None}
        save_atomic(dict(st, model=model.state_dict(), optimizer=opt.state_dict(),
                         scaler=scaler.state_dict(), scheduler=sched.state_dict(), rng=rng,
                         args={k: getattr(args, k) for k in SAME_ON_RESUME}), last_p)

    def record(finished):
        outs = [out / f for f in ("best.pt", "last.pt", "history.csv", "model_meta.json",
                                  "norm_stats.json", "train_chips.txt") if (out / f).exists()]
        rec = mc.run_record(args, paths, model="unet", trainset=args.trainset, seed=args.seed,
                            run_tag=tag,
                            finished=finished, epochs_done=st["epoch"],
                            early_stopped=st["early_stopped"], device=str(device), amp=amp,
                            norm_file=str(paths.norm_stats),
                            norm_sha256=mc.sha256(out / "norm_stats.json"),
                            hand_chips=len(hand), weak_chips=len(weak),
                            train_chip_ids=[r["chip"] for r in hand + weak],
                            valid_chips=len(valid_rows), pos_weight=pos_weight,
                            memory_estimate=est, best_epoch=st["best_epoch"],
                            threshold=st["best_threshold"],
                            threshold_how="highest water IoU on the hand valid chips at the best "
                                          "epoch, 0.01 to 0.99 in 0.01",
                            iou_valid_at_0_5=st["best_iou"],
                            iou_valid_at_threshold=st["best_iou_threshold"],
                            first_step_loss=st["first_step_loss"],
                            history=str(out / "history.csv"),
                            train_seconds=round(st["train_seconds"], 1),
                            outputs=[mc.file_entry(f) for f in outs])
        mc.save_json(rec_p, rec)

    (out / "train_chips.txt").write_text("\n".join(r["chip"] for r in hand + weak) + "\n",
                                         encoding="utf-8")
    stop = StopFlag()
    first_step_here = st["global_step"]
    t_session = time.time()
    epochs_here = 0
    try:
        for epoch in range(st["epoch"], args.epochs):
            sampler.set_epoch(epoch, st["step_in_epoch"])
            model.train()
            t_ep = time.time()
            t_print = time.time()
            for s, (x, y) in enumerate(loader, start=st["step_in_epoch"]):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                try:
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                        logits = model(x)
                    loss, bce, dice = mc.masked_bce_dice(logits, y, pos_weight)
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                except RuntimeError as e:
                    if not (amp and mc.is_oom(e) and st["global_step"] == first_step_here):
                        raise
                    del x, y
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    bs = mc.find_batch_size(model, device, args.crop, args.batch_size, amp)
                    if bs:
                        mc.fail(f"The GPU ran out of memory with --batch-size {args.batch_size}. "
                                f"--batch-size {bs} fits. Run the same command again with "
                                f"--batch-size {bs} added (and --force if it refuses).")
                    mc.fail("The GPU ran out of memory even with --batch-size 1. Close every "
                            "other program that uses the GPU and try again, or add --crop 128.")
                sched.step()
                lv = float(loss.detach())
                if st["first_step_loss"] is None:
                    st["first_step_loss"] = lv
                st["sums"] = [st["sums"][0] + lv, st["sums"][1] + float(bce),
                              st["sums"][2] + float(dice), st["sums"][3] + 1]
                st["global_step"] += 1
                st["step_in_epoch"] = s + 1
                if time.time() - t_print > 120:
                    t_print = time.time()
                    print(f"    epoch {epoch + 1}, step {s + 1}/{args.steps_per_epoch}, "
                          f"loss {st['sums'][0] / st['sums'][3]:.4f}", flush=True)
                if args.stop_after_steps and st["global_step"] >= args.stop_after_steps:
                    stop.flag = True
                if stop.flag and st["step_in_epoch"] < args.steps_per_epoch:
                    st["train_seconds"] += time.time() - t_ep
                    checkpoint()
                    record(False)
                    print(f"STOPPED and saved at epoch {epoch + 1}, step {st['step_in_epoch']}. "
                          "Run the same command with --resume to continue.")
                    return 0

            v = validate(model, vloader, device, amp, args.tile, args.overlap)
            n = max(st["sums"][3], 1)
            iou = v["iou05"] if np.isfinite(v["iou05"]) else -1.0
            score = iou if args.select_by == "0.5" else (
                v["iou_thr"] if np.isfinite(v["iou_thr"]) else -1.0)
            improved = score > st["best_score"] or not best_p.exists()
            row = {"epoch": epoch + 1, "train_loss": round(st["sums"][0] / n, 6),
                   "train_bce": round(st["sums"][1] / n, 6),
                   "train_dice": round(st["sums"][2] / n, 6),
                   "val_iou_at_0.5": round(v["iou05"], 6), "val_f1_at_0.5": round(v["f1_05"], 6),
                   "val_best_threshold": v["thr"],
                   "val_iou_at_best_threshold": round(v["iou_thr"], 6),
                   "lr": sched.get_last_lr()[0], "improved": bool(improved)}
            st["history"].append(row)
            if improved:
                st.update(best_iou=iou, best_epoch=epoch + 1, best_threshold=v["thr"],
                          best_iou_threshold=v["iou_thr"], best_score=score, bad_epochs=0)
            else:
                st["bad_epochs"] += 1
            secs = time.time() - t_ep
            st["epoch_seconds"].append(round(secs, 1))
            st["train_seconds"] += secs
            st.update(epoch=epoch + 1, step_in_epoch=0, sums=[0.0, 0.0, 0.0, 0])
            if st["bad_epochs"] >= args.patience:
                st["early_stopped"] = True
            checkpoint()
            if improved:                    # best.pt: the full checkpoint of the best epoch
                shutil.copy2(last_p, best_p.with_name("best.pt.part"))
                os.replace(best_p.with_name("best.pt.part"), best_p)
            write_history(out / "history.csv",
                          [dict(h, seconds=sec) for h, sec in
                           zip(st["history"], st["epoch_seconds"])])
            epochs_here += 1
            print(f"  epoch {epoch + 1}/{args.epochs}: loss {row['train_loss']:.4f}, valid IoU "
                  f"{v['iou05']:.4f} at 0.5 ({v['iou_thr']:.4f} at {v['thr']:.2f})"
                  f"{'  best so far' if improved else ''}  [{secs / 60:.1f} min]", flush=True)
            if epochs_here == 1 and amp:
                print(f"  GPU memory used at most: "
                      f"{torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
            if epochs_here == 1 and epoch + 1 < args.epochs:
                left = (args.epochs - epoch - 1) * secs
                end = datetime.now() + timedelta(seconds=left)
                print(f"  ETA: about {left / 3600:.1f} hours more for the remaining "
                      f"{args.epochs - epoch - 1} epochs (around {end:%H:%M %d %b}); "
                      "sooner if early stopping triggers.", flush=True)
            if st["early_stopped"]:
                print(f"  early stopping: no better valid IoU "
                      f"({'at 0.5' if args.select_by == '0.5' else 'at the best threshold'}) "
                      f"for {args.patience} epochs")
                break
            if stop.flag or (args.stop_after_epochs and st["epoch"] >= args.stop_after_epochs):
                record(False)
                print(f"STOPPED and saved after epoch {st['epoch']}. Run the same command "
                      "with --resume to continue.")
                return 0
    finally:
        stop.restore()

    if not best_p.exists():
        mc.fail("Training ended without a best checkpoint. Run again with --force.")
    meta = {"model": "unet", "trainset": args.trainset, "base_width": args.base_width,
            "channels": list(mc.CHANNELS), "threshold_default": 0.5,
            "threshold_valid": st["best_threshold"],
            "threshold_how": "highest water IoU on the hand valid chips at the best epoch",
            "iou_valid_at_0.5": st["best_iou"],
            "iou_valid_at_threshold": st["best_iou_threshold"], "best_epoch": st["best_epoch"],
            "best_epoch_selected_by": ("valid IoU at 0.5" if args.select_by == "0.5" else
                                       "valid IoU at the best threshold of each epoch"),
            "torch_version": torch.__version__}
    mc.save_json(out / "model_meta.json", meta)
    record(True)
    print(f"TRAINING FINISHED in {st['train_seconds'] / 3600:.2f} hours "
          f"({time.time() - t_session:.0f} s in this session). Best epoch {st['best_epoch']}: "
          f"valid IoU {st['best_iou']:.4f} at 0.5, {st['best_iou_threshold']:.4f} at threshold "
          f"{st['best_threshold']:.2f}")
    print(f"PASS  U-Net '{args.trainset}' saved in {out}")
    return 0


def selftest():
    import models_synthetic as syn
    print("23_train_unet.py self-test (uses the GPU if there is one)")
    fast = ["--base-width", "8", "--crop", "32", "--batch-size", "4", "--steps-per-epoch", "3",
            "--epochs", "2", "--tile", "64", "--overlap", "16", "--workers", "0", "--cpu-ok"]
    with tempfile.TemporaryDirectory(prefix="fyp selftest ", ignore_cleanup_errors=True) as td:
        proj = Path(td) / "Final Year Project"
        syn.build_project(proj, corridor=False)
        ok = mc.run_script("21_prepare_chips.py", ["--project", str(proj)]) == 0
        ok &= main(["--project", str(proj), "--trainset", "hand_weak"] + fast) == 0
        paths = mc.Paths(proj)
        d = paths.model_dir("unet", "hand_weak")
        rec = mc.load_json(d / "run_record.json")
        p = mc.load_predictor("unet", d, cpu_ok=True, tile=64, overlap=16)
        row = mc.split_rows(mc.load_chip_index(paths), "test")[0]
        x, _ = mc.read_chip(row, paths.data)
        v = mc.input_valid(x)
        prob = p.predict_image(x, v)
        stop_ok = main(["--project", str(proj), "--trainset", "hand", "--stop-after-steps", "4"]
                       + fast) == 0
        res_ok = main(["--project", str(proj), "--trainset", "hand", "--resume"] + fast) == 0
        rec2 = mc.load_json(paths.model_dir("unet", "hand") / "run_record.json")
    checks = [("trained 2 epochs and saved", ok and rec["finished"] and rec["epochs_done"] == 2),
              ("reloaded and predicted", np.isfinite(prob[v]).all() and np.isnan(prob[~v]).all()),
              ("stopped mid-epoch and resumed to the end", stop_ok and res_ok and rec2["finished"])]
    for name, c in checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {name}")
    good = all(c for _, c in checks)
    print("SELFTEST PASS" if good else "SELFTEST FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(mc.cli(main))
