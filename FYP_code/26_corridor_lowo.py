r"""
26_corridor_lowo.py
===================
Table D: leave one digitising window out, on the corridor itself
For one event, one model and one radar product.
Models: rf, unet, and the two TerraMind variants tm_s1 and tm_s1t.

WHAT IT DOES
------------
1. Labels: the hand-drawn 'water_event' layer of that year's reference
   GeoPackage, burnt onto the 10 m grid (all_touched off): 1 inside, 0
   elsewhere inside the 8 windows. Ignored: outside the windows, inside the
   cloud holes, and wherever the radar input is not valid.
2. Common mask: a cell counts only where BOTH the GRD and the RTC VV are
   finite, for the GRD run and the RTC run alike, so the two are scored on
   exactly the same cells.
3. Fold k (one per window): a model is trained from scratch on the other
   windows, with normalisation computed from those windows only, and
   predicts window k. Random Forest: up to N water and N not-water cells per
   training window (N 20000), and a threshold chosen inside the fold (5).
   U-Net: random 128 x 128 crops from the training windows
   for a fixed number of steps set in advance (no early stopping, nothing
   tuned on the held-out window); the held-out window's cells are blanked in
   every training crop. Same seed and settings for GRD and RTC.
   TerraMind: every fold starts again from the pretrained weights, GRD
   through TerraMind's S1GRD input and RTC through its S1RTC input, each
   with TerraMind's own radar normalisation (HAND and slope for tm_s1t from
   the training windows); random 224 x 224 crops, 1000 steps of batch 8
   (the same number of training pixels per fold as the U-Net), learning
   rates 5e-5 (encoder) and 1e-3 (decoder).
4. The 8 held-out predictions are put together into one map (each cell
   predicted by the model that never saw its window), each cut at its own
   fold's threshold (5); permanent water and layover/shadow are removed as in
   every other map, and the map is written with 255 outside the windows and
   outside the common mask.
5. Random Forest threshold ('--threshold inner', its default): inside fold k,
   each of the 7 training windows is left out in turn, a forest of 100 trees
   is trained on the other 6 and predicts every labelled cell of the one left
   out. The threshold with the best pooled water IoU over those 7 predictions
   is the fold's threshold. Window k is never read. The training sample is
   balanced (equal water and not water), so 0.5 would call too much water;
   this is the same rule as Table A, where the threshold is chosen on chips
   the forest never saw. The first RF run of 7 October used
   --n-per-class 1000 --threshold 0.5; both options still reproduce it.
6. U-Net and TerraMind threshold ('--threshold inner', also their default):
   a network costs too much to train 7 extra times per fold, so one extra
   network is trained, with the fold's settings, on the training windows
   except one (the first training window after k, counting round, that holds
   water) and predicts that window; the threshold with the best water IoU
   there is the fold's threshold. Window k is never read. This doubles the
   training time of a network run. The first runs used 0.5
   (--threshold 0.5 reproduces it).

Each finished fold is kept (fold_<k>\), so a stopped run continues where it
was when started again.

USAGE: HOW TO RUN THIS  (copy one line, paste into Anaconda Prompt, press Enter)
================================================================================
    conda activate fyp
    cd /d "<project folder>\04 Code\FYP_code"

    python 26_corridor_lowo.py --selftest
    python 26_corridor_lowo.py --year 2021 --model rf --radar grd
    python 26_corridor_lowo.py --year 2023 --model unet --radar rtc
    python 26_corridor_lowo.py --year 2021 --model tm_s1t --radar grd

Random Forest about 10 to 30 minutes per run (8 folds, each with its
threshold step), U-Net about 20 to 40 minutes per run, TerraMind about 40 to
80 minutes per run on the RTX 3050, both with their threshold step
(estimates).
The last line says PASS.
================================================================================
"""

__stage__ = "Models: leave one window out on the corridor (Table D)"
__reads__ = ["05 Data/sentinel1 and sentinel1_rtc post-event scenes", "data/strata_<year>_v3/",
             "reference_windows.geojson", "reference_<year>_event.gpkg (water_event)",
             "S2_<year>_event_cloudgap.gpkg (holes)", "WATER_PERMANENT_JRC50_S2DRY_10m.tif",
             "data/weights/TerraMind_v1_small.pt (TerraMind only)"]
__writes__ = ["data/lowo_<year>_<model>_<radar>/flood_lowo_<year>_<model>_<radar>.tif",
              "fold_<k>/run_record.json, prob_window_<k>.tif", "run_record.json"]

import argparse
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import models_common as mc

try:
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from rasterio.windows import Window
    from rasterio.windows import transform as window_transform
except ImportError as e:
    sys.exit(f"\nMissing package: {e}.\nActivate the environment first:  conda activate fyp\n")


def burn(path, layer, profile, what):
    if not Path(path).exists():
        mc.fail(f"{what} is missing:\n  {path}\nSee RUN ORDER.md (steps 2.6 and 4.7).")
    try:
        gdf = gpd.read_file(path, layer=layer)
    except Exception as e:                      # noqa: BLE001
        mc.fail(f"Could not read layer '{layer}' of {Path(path).name}: {e}")
    if gdf.crs is None:
        mc.fail(f"{Path(path).name} layer {layer} has no CRS. Set it in QGIS first.")
    gdf = gdf.to_crs(profile["crs"])
    geoms = [(g, 1) for g in gdf.geometry if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros((profile["height"], profile["width"]), bool)
    return rasterize(geoms, out_shape=(profile["height"], profile["width"]),
                     transform=profile["transform"], fill=0, default_value=1, dtype="uint8",
                     all_touched=False).astype(bool)


def load_data(paths, year, radar):
    """Inputs of `radar`, labels, windows and the common GRD/RTC mask."""
    inp = mc.corridor_inputs(paths, year, radar)
    prof = inp["profile"]
    other = "rtc" if radar == "grd" else "grd"
    ovv, _, oprof = mc.read_scene(paths.scene(year, other), f"The {other.upper()} scene for {year}")
    if not mc.same_grid((prof["height"], prof["width"]), prof["transform"], prof["crs"],
                        (oprof["height"], oprof["width"]), oprof["transform"], oprof["crs"]):
        mc.fail(f"The GRD and RTC scenes of {year} are not on the same grid. They must be "
                "for the GRD and RTC comparison. Run 20_models_preflight.py.")
    common = np.isfinite(inp["x"][0]) & np.isfinite(ovv)
    scorer = mc.load_script("04_stratified_metrics.py")
    if not paths.windows(year).exists():
        mc.fail(f"The digitising windows are missing:\n  {paths.windows(year)}")
    win, _ = scorer.window_raster(str(paths.windows(year)), common.shape, prof["transform"],
                                  prof["crs"])
    water = burn(paths.ref_gpkg(year), "water_event", prof, f"The reference GeoPackage for {year}")
    holes = burn(paths.holes(year), "holes", prof, f"The cloud-hole GeoPackage for {year}")
    lab = np.where(water, 1, 0).astype("int8")
    ok = (win > 0) & common & ~holes & np.isfinite(inp["x"][1])
    lab[~ok] = -1
    inp.update(common=common, win=win, lab=lab, holes=holes)
    return inp


def fold_train_mask(win, lab, k):
    """Cells fold k may train on: labelled, inside a window, not window k."""
    return (win > 0) & (win != k) & (lab >= 0)


def bbox(mask, margin=0, shape=None):
    rows, cols = np.nonzero(mask)
    shape = shape or mask.shape
    r0, r1 = max(rows.min() - margin, 0), min(rows.max() + 1 + margin, shape[0])
    c0, c1 = max(cols.min() - margin, 0), min(cols.max() + 1 + margin, shape[1])
    return r0, r1, c0, c1


# ----------------------------------------------------------------------------- RF

def rf_window_samples(args, x, lab, win, k, train):
    """Per training window: up to N water and N not-water cells (features, labels)."""
    out = {}
    for j in sorted(set(np.unique(win[train]).tolist())):
        rng = np.random.default_rng([args.seed, k, j])
        feats, labels = [], []
        for cls in (1, 0):
            idx = np.flatnonzero(train & (win == j) & (lab == cls))
            if idx.size > args.n_per_class:
                idx = np.sort(rng.choice(idx, size=args.n_per_class, replace=False))
            feats.append(x.reshape(4, -1)[:, idx].T)
            labels.append(np.full(idx.size, cls, "int8"))
        out[j] = (np.concatenate(feats), np.concatenate(labels))
    return out


def rf_forest(args, samples, windows, stats, n_estimators):
    from sklearn.ensemble import RandomForestClassifier
    X = mc.rf_features(np.concatenate([samples[j][0] for j in windows]), stats)
    y = np.concatenate([samples[j][1] for j in windows])
    clf = RandomForestClassifier(n_estimators=n_estimators,
                                 min_samples_leaf=args.min_samples_leaf, max_features="sqrt",
                                 n_jobs=args.n_jobs, random_state=args.seed)
    return clf.fit(X, y), y


def rf_inner_threshold(args, x, lab, win, train, samples, stats, rec):
    """Threshold from the training windows only: leave one TRAINING window out.

    For each training window j, a smaller forest (--inner-estimators trees)
    is trained on the samples of the other training windows and predicts
    every labelled cell of j, at its real share of water. The threshold with
    the highest pooled water IoU over these predictions is used for the
    fold. The held-out window of the fold is never read.
    """
    hist = mc.ThresholdHist()
    tw = sorted(samples)
    for j in tw:
        others = [i for i in tw if i != j]
        clf, y = rf_forest(args, samples, others, stats, args.inner_estimators)
        cells = train & (win == j)
        if len(set(y.tolist())) < 2 or not cells.any():
            continue
        f = mc.rf_features(x[:, cells].T, stats)
        prob = np.concatenate([mc.rf_water_prob(clf, f[i:i + mc.CHUNK], args.n_jobs)
                               for i in range(0, len(f), mc.CHUNK)])
        hist.add(prob, lab[cells] == 1)
    if hist.water.sum() == 0 or hist.dry.sum() == 0:
        rec["threshold_note"] = "inner windows hold one class only; threshold 0.5 used"
        print(f"    note: {rec['threshold_note']}")
        return 0.5
    thr, iou = hist.best()
    rec.update(inner_iou_at_threshold=iou, inner_iou_at_0_5=hist.iou(0.5),
               inner_windows=tw, inner_estimators=args.inner_estimators)
    print(f"    threshold {thr:.2f} from the {len(tw)} training windows "
          f"(water IoU {iou:.3f}; {hist.iou(0.5):.3f} at 0.5)", flush=True)
    return float(thr)


def rf_fold(args, x, lab, win, k, train, stats, pred_box, pred_mask, rec):
    samples = rf_window_samples(args, x, lab, win, k, train)
    if getattr(args, "threshold", 0.5) == "inner":
        rec["threshold"] = rf_inner_threshold(args, x, lab, win, train, samples, stats, rec)
        rec["threshold_how"] = ("highest pooled water IoU, leaving out one training window at "
                                "a time (the held-out window is never read), 0.01 to 0.99")
    clf, y = rf_forest(args, samples, sorted(samples), stats, args.n_estimators)
    classes = sorted(set(y.tolist()))
    rec.update(rows=int(len(y)), water_rows=int((y == 1).sum()), classes=classes)
    print(f"    forest: {len(y):,} training cells, {rec['water_rows']:,} water", flush=True)
    if len(classes) < 2:
        rec["note"] = f"only class {classes} in the training windows; trained anyway"
        print(f"    note: {rec['note']}")
    r0, r1, c0, c1 = pred_box
    sub = x[:, r0:r1, c0:c1]
    prob = np.full(pred_mask.shape, np.nan, "float32")
    if pred_mask.any():
        f = mc.rf_features(sub[:, pred_mask].T, stats)
        prob[pred_mask] = np.concatenate([mc.rf_water_prob(clf, f[i:i + mc.CHUNK], args.n_jobs)
                                          for i in range(0, len(f), mc.CHUNK)])
    return prob


# ----------------------------------------------------------------------------- U-Net

def unet_fold(args, x, lab, win, k, train, stats, pred_box, pred_mask, rec, device):
    import torch
    amp = device.type == "cuda"
    # one normalised copy per training window; window k blanked (input 0, label -1)
    pieces, leak = [], 0
    for j in sorted(set(np.unique(win[train]).tolist())):
        r0, r1, c0, c1 = bbox(win == j)
        xs = mc.normalise(x[:, r0:r1, c0:c1], stats)
        ys = np.where(train[r0:r1, c0:c1], lab[r0:r1, c0:c1], -1).astype("int8")
        held = win[r0:r1, c0:c1] == k
        xs[:, held] = 0.0
        leak += int(((ys >= 0) & held).sum())
        pieces.append((xs, ys))
    rec["heldout_labels_in_training_crops"] = leak
    rng = np.random.default_rng([args.seed, k])
    torch.manual_seed(args.seed)
    model = mc.UNet(len(mc.CHANNELS), args.base_width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    warm = max(1, min(100, args.steps // 10))

    def lr_f(s):
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min((s - warm) / max(args.steps - warm, 1), 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_f)
    c = args.crop
    losses = []
    model.train()
    for step in range(args.steps):
        xb = np.zeros((args.batch_size, len(mc.CHANNELS), c, c), "float32")
        yb = np.full((args.batch_size, c, c), -1, "int8")
        for b in range(args.batch_size):
            xs, ys = pieces[int(rng.integers(len(pieces)))]
            H, W = ys.shape
            r = int(rng.integers(0, max(H - c, 0) + 1))
            q = int(rng.integers(0, max(W - c, 0) + 1))
            xc, yc = xs[:, r:r + c, q:q + c], ys[r:r + c, q:q + c]
            kk, flip = int(rng.integers(4)), bool(rng.integers(2))
            xc, yc = np.rot90(xc, kk, axes=(1, 2)), np.rot90(yc, kk)
            if flip:
                xc, yc = xc[:, :, ::-1], yc[:, ::-1]
            xb[b, :, :xc.shape[1], :xc.shape[2]] = xc
            yb[b, :yc.shape[0], :yc.shape[1]] = yc
        xt = torch.from_numpy(xb).to(device)
        yt = torch.from_numpy(yb).to(device)
        try:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                out = model(xt)
            loss = mc.masked_bce_dice(out, yt)[0]
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        except RuntimeError as e:
            if not (amp and step == 0 and mc.is_oom(e)):
                raise
            del xt, yt
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            bs = mc.find_batch_size(model, device, c, args.batch_size, amp)
            mc.fail(f"The GPU ran out of memory with --batch-size {args.batch_size}. "
                    + (f"Run the same command again with --batch-size {bs}." if bs else
                       "Close other programs that use the GPU and try again."))
        sched.step()
        losses.append(float(loss.detach()))
    rec.update(steps=args.steps, first_loss=losses[0], last_loss=float(np.mean(losses[-10:])),
               losses=[round(v, 6) for v in losses])
    model.eval()
    r0, r1, c0, c1 = pred_box
    xn = mc.normalise(x[:, r0:r1, c0:c1], stats)
    prob, _ = mc.predict_tiled(mc.unet_prob_fn(model, device, amp), xn, pred_mask,
                               args.tile, args.overlap, batch=2)
    return prob


# ----------------------------------------------------------------------------- TerraMind

def tm_weights(args, paths):
    """TerraMind's pretrained encoder tensors for this radar's input, read once per run."""
    modality = mc.TM_MODALITY[args.radar]
    path = Path(args.weights) if args.weights else paths.tm_weights()
    ref = mc.TerraMindNet(args.model, modality)
    take, rep = mc.tm_read_weights(path, ref.backbone)
    for line in mc.tm_weights_lines(rep):
        print("  " + line)
    return {"modality": modality, "radar": mc.tm_radar_norm(modality), "take": take,
            "path": str(path), "sha256": mc.sha256(path), "report": rep}


def tm_fold(args, x, lab, win, k, train, stats, pred_box, pred_mask, rec, device, tm):
    """One TerraMind fold: from the pretrained weights, trained on the other windows."""
    import torch
    amp = device.type == "cuda"
    norm = mc.tm_norm_record(tm["radar"], stats if args.model == "tm_s1t" else None,
                             f"corridor windows {rec['train_windows']} (fold {k})")
    tstats = mc.tm_stats(norm)
    # one normalised copy per training window; window k blanked (input 0, label -1)
    pieces, leak = [], 0
    for j in sorted(set(np.unique(win[train]).tolist())):
        r0, r1, c0, c1 = bbox(win == j)
        xs = mc.normalise(x[:, r0:r1, c0:c1], tstats)
        ys = np.where(train[r0:r1, c0:c1], lab[r0:r1, c0:c1], -1).astype("int8")
        held = win[r0:r1, c0:c1] == k
        xs[:, held] = 0.0
        leak += int(((ys >= 0) & held).sum())
        pieces.append((xs, ys))
    rec.update(heldout_labels_in_training_crops=leak, modality=tm["modality"],
               terramind_norm=norm, terramind_weights=tm["path"],
               terramind_weights_sha256=tm["sha256"], terramind_weights_report=tm["report"])
    rng = np.random.default_rng([args.seed, k])
    torch.manual_seed(args.seed)
    model = mc.TerraMindNet(args.model, tm["modality"])
    mc.tm_apply_weights(model.backbone, tm["take"])
    model.to(device)
    opt = torch.optim.AdamW([{"params": list(model.backbone.parameters()), "lr": args.lr_encoder},
                             {"params": model.decoder_parameters(), "lr": args.lr}],
                            weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    warm = max(1, min(100, args.steps // 10))

    def lr_f(s):
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min((s - warm) / max(args.steps - warm, 1), 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_f)
    c = args.crop
    losses = []
    model.train()
    for step in range(args.steps):
        xb = np.zeros((args.batch_size, len(mc.CHANNELS), c, c), "float32")
        yb = np.full((args.batch_size, c, c), -1, "int8")
        for b in range(args.batch_size):
            xs, ys = pieces[int(rng.integers(len(pieces)))]
            H, W = ys.shape
            r = int(rng.integers(0, max(H - c, 0) + 1))
            q = int(rng.integers(0, max(W - c, 0) + 1))
            xc, yc = xs[:, r:r + c, q:q + c], ys[r:r + c, q:q + c]
            kk, flip = int(rng.integers(4)), bool(rng.integers(2))
            xc, yc = np.rot90(xc, kk, axes=(1, 2)), np.rot90(yc, kk)
            if flip:
                xc, yc = xc[:, :, ::-1], yc[:, ::-1]
            xb[b, :, :xc.shape[1], :xc.shape[2]] = xc
            yb[b, :yc.shape[0], :yc.shape[1]] = yc
        xt = torch.from_numpy(xb).to(device)
        yt = torch.from_numpy(yb).to(device)
        try:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                out = model(xt)
            loss = mc.masked_bce_dice(out, yt)[0]
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        except RuntimeError as e:
            if not (amp and step == 0 and mc.is_oom(e)):
                raise
            del xt, yt
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            bs = mc.find_batch_size(model, device, c, args.batch_size, amp)
            mc.fail(f"The GPU ran out of memory with --batch-size {args.batch_size}. "
                    + (f"Run the same command again with --batch-size {bs}." if bs else
                       "Close other programs that use the GPU and try again."))
        sched.step()
        losses.append(float(loss.detach()))
    rec.update(steps=args.steps, first_loss=losses[0], last_loss=float(np.mean(losses[-10:])),
               losses=[round(v, 6) for v in losses])
    model.eval()
    r0, r1, c0, c1 = pred_box
    xn = mc.normalise(x[:, r0:r1, c0:c1], tstats)
    prob, _ = mc.predict_tiled(mc.unet_prob_fn(model, device, amp), xn, pred_mask,
                               c, args.overlap, batch=2)
    return prob


# ----------------------------------------------------------------------------- inner threshold, U-Net and TerraMind

def inner_window(win, lab, train, k):
    """The training window that sets a network fold's threshold: the first one after k
    (counting round, 8 -> 1) that holds labelled water; None if no window holds any."""
    tw = sorted(int(v) for v in np.unique(win[train]))
    order = [w for w in tw if w > k] + [w for w in tw if w < k]
    for j in order:
        if (train & (win == j) & (lab == 1)).any():
            return j
    return None


def nn_inner_threshold(fold_fn, args, x, lab, win, k, train, stats, rec, device, *extra):
    """Threshold for a U-Net or TerraMind fold from its training windows only.

    One extra network, with the fold's settings, is trained on the training
    windows except one (inner_window) and predicts every labelled cell of
    that one; the threshold with the highest water IoU there is the fold's
    threshold. The held-out window k is never read.
    """
    j = inner_window(win, lab, train, k)
    if j is None:
        rec["threshold_note"] = "no training window holds water; threshold 0.5 used"
        print(f"    note: {rec['threshold_note']}")
        return 0.5
    inner_train = train & (win != j)
    box = bbox(win == j, margin=args.overlap)
    cells = (train & (win == j))[box[0]:box[1], box[2]:box[3]]
    sub = {"train_windows": sorted(int(v) for v in np.unique(win[inner_train]))}
    prob = fold_fn(args, x, lab, win, k, inner_train, stats, box, cells, sub, device, *extra)
    truth = lab[box[0]:box[1], box[2]:box[3]][cells] == 1
    hist = mc.ThresholdHist()
    hist.add(prob[cells], truth)
    thr, iou = hist.best()
    rec.update(inner_window=j, inner_train_windows=sub["train_windows"],
               inner_iou_at_threshold=iou, inner_iou_at_0_5=hist.iou(0.5),
               inner_first_loss=sub.get("first_loss"), inner_last_loss=sub.get("last_loss"))
    print(f"    threshold {thr:.2f} from training window {j} (a model trained on windows "
          f"{sub['train_windows']}; water IoU {iou:.3f}, {hist.iou(0.5):.3f} at 0.5)", flush=True)
    return float(thr)


# ----------------------------------------------------------------------------- main

def write_prob(path, prob, profile, box):
    r0, r1, c0, c1 = box
    tf = window_transform(Window(c0, r0, c1 - c0, r1 - r0), profile["transform"])
    tmp = path.with_name(path.stem + "_part.tif")
    with rasterio.open(tmp, "w", driver="GTiff", height=r1 - r0, width=c1 - c0, count=1,
                       dtype="float32", crs=profile["crs"], transform=tf, nodata=np.nan,
                       compress="deflate") as d:
        d.write(prob.astype("float32"), 1)
    os.replace(tmp, path)


def parse_threshold(value, model):
    """'inner' (the default, every model) or a fixed number in (0, 1)."""
    if value is None or str(value).lower() == "inner":
        return "inner"
    try:
        t = float(value)
    except ValueError:
        mc.fail(f"--threshold must be a number or 'inner', not '{value}'.")
    if not 0 < t < 1:
        mc.fail(f"--threshold must be between 0 and 1, not {t}.")
    return t


def main(argv=None):
    mc.setup_console()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mc.add_common_args(ap)
    ap.add_argument("--year", type=int, choices=mc.YEARS, default=2021)
    ap.add_argument("--model", choices=mc.MODELS, default="rf")
    ap.add_argument("--radar", choices=mc.RADARS, default="grd")
    ap.add_argument("--folds", default="all",
                    help="windows to run, e.g. 1,2 (default all; the map is made once all are done)")
    ap.add_argument("--threshold", default=None,
                    help="'inner' (default): the threshold chosen inside each fold from its "
                         "training windows only (section 5 above); or a fixed number. The first "
                         "runs of 7 October used 0.5")
    ap.add_argument("--n-per-class", type=int, default=20000,
                    help="RF: cells per class per window, at most (default 20000; the first RF "
                         "run of 7 October used 1000)")
    ap.add_argument("--inner-estimators", type=int, default=100,
                    help="RF, --threshold inner: trees of the forests that choose the threshold")
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--min-samples-leaf", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--steps", type=int, default=None,
                    help="training steps per fold: U-Net 1500, TerraMind 1000")
    ap.add_argument("--crop", type=int, default=None,
                    help="training crop: U-Net 128, TerraMind 224 (also its prediction tile)")
    ap.add_argument("--batch-size", type=int, default=None, help="U-Net 16, TerraMind 8")
    ap.add_argument("--base-width", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="learning rate of the U-Net and of the TerraMind decoder")
    ap.add_argument("--lr-encoder", type=float, default=5e-5,
                    help="TerraMind: learning rate of the pretrained encoder")
    ap.add_argument("--weights", default=None,
                    help="TerraMind_v1_small.pt (default data\\weights\\TerraMind_v1_small.pt)")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--cpu-ok", action="store_true", help="allow the U-Net or TerraMind on the CPU")
    ap.add_argument("--deterministic", action="store_true",
                    help="U-Net, TerraMind: deterministic algorithms (slower); cudnn.benchmark off")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    tmv = args.model in mc.TM_VARIANTS
    args.threshold = parse_threshold(args.threshold, args.model)
    args.steps = args.steps or (1000 if tmv else 1500)
    args.crop = args.crop or (mc.TM_CROP if tmv else 128)
    args.batch_size = args.batch_size or (8 if tmv else 16)
    if args.smoke:
        args.n_estimators = min(args.n_estimators, 20)
        args.inner_estimators = min(args.inner_estimators, 10)
        args.steps = min(args.steps, 10)
        args.batch_size = min(args.batch_size, 4)
    if args.crop % 16:
        mc.fail(f"--crop must be a multiple of 16; got {args.crop}.")
    mc.check_tile_args(args.tile, args.overlap)
    if tmv and not 0 <= args.overlap < args.crop // 2:
        mc.fail(f"--overlap must be less than half the crop ({args.crop}), TerraMind's "
                f"prediction tile; got {args.overlap}.")
    paths = mc.paths_from_args(args)
    name = f"lowo_{args.year}_{args.model}_{args.radar}"
    out = paths.lowo_dir(args.year, args.model, args.radar)
    print(f"26_corridor_lowo.py: {name}{' (smoke)' if args.smoke else ''}")
    if mc.guard(out / "run_record.json", args, f"Table D run {name}"):
        return 0
    t0 = time.time()
    device, tm = None, None
    if args.model != "rf":
        device = mc.pick_device(args.cpu_ok or args.smoke)
        mc.set_deterministic(args.deterministic)
    d = load_data(paths, args.year, args.radar)
    x, lab, win, prof = d["x"], d["lab"], d["win"], d["profile"]
    ks = sorted(int(v) for v in np.unique(win[win > 0]))
    if len(ks) < 2:
        mc.fail(f"Only {len(ks)} digitising window lands on the grid; leave-one-out needs 2 or more.")
    if args.smoke:
        todo = ks[:1]
    elif args.folds == "all":
        todo = ks
    else:
        todo = [int(v) for v in args.folds.split(",") if v.strip()]
        bad = [k for k in todo if k not in ks]
        if bad:
            mc.fail(f"--folds {bad}: no such window on the grid (windows: {ks}).")
    print(f"  windows {ks}; common GRD/RTC cells in windows: "
          f"{int(((win > 0) & d['common']).sum()):,}; labelled cells "
          f"{int((lab >= 0).sum()):,}, water {int((lab == 1).sum()):,}")
    out.mkdir(parents=True, exist_ok=True)
    for k in todo:
        fdir = out / f"fold_{k}"
        if (fdir / "run_record.json").exists() and not args.force:
            print(f"  fold {k}: already done, kept")
            continue
        tf = time.time()
        train = fold_train_mask(win, lab, k)
        if not train.any():
            mc.fail(f"Fold {k} has no labelled cell in any other window: nothing to train on. "
                    "Check the reference GeoPackage and the windows (20_models_preflight.py).")
        tw = sorted(int(v) for v in np.unique(win[train]))
        stats = mc.compute_norm_stats(x[:, train].T, f"corridor windows {tw} (fold {k})",
                                      {"year": args.year, "radar": args.radar})
        box = bbox(win == k, margin=args.overlap)
        pred_mask = ((win == k) & d["common"])[box[0]:box[1], box[2]:box[3]]
        rec = {"heldout_window": k, "train_windows": tw, "train_cells": int(train.sum()),
               "train_water_cells": int((train & (lab == 1)).sum()),
               "train_cells_in_heldout_window": int((train & (win == k)).sum()),
               "heldout_cells_predicted": int(pred_mask.sum()), "norm_stats": stats}
        print(f"  fold {k}: training on windows {tw} ({rec['train_cells']:,} cells, "
              f"{rec['train_water_cells']:,} water)", flush=True)
        if args.model == "rf":
            prob = rf_fold(args, x, lab, win, k, train, stats, box, pred_mask, rec)
        else:
            if args.model == "unet":
                fold_fn, extra = unet_fold, ()
            else:
                tm = tm or tm_weights(args, paths)
                fold_fn, extra = tm_fold, (tm,)
            if args.threshold == "inner":
                rec["threshold"] = nn_inner_threshold(fold_fn, args, x, lab, win, k, train,
                                                      stats, rec, device, *extra)
                rec["threshold_how"] = (
                    "highest water IoU on one training window, predicted by a model trained "
                    "on the other training windows (the held-out window is never read), "
                    "0.01 to 0.99")
            prob = fold_fn(args, x, lab, win, k, train, stats, box, pred_mask, rec, device,
                           *extra)
        fdir.mkdir(parents=True, exist_ok=True)
        if "losses" in rec:                       # the U-Net's or TerraMind's training curve
            with open(fdir / "history.csv", "w", encoding="utf-8") as fh:
                fh.write("step,loss\n" + "".join(f"{i + 1},{v}\n"
                                                  for i, v in enumerate(rec.pop("losses"))))
        pp = fdir / f"prob_window_{k}.tif"
        write_prob(pp, prob, prof, box)
        rec.update(box_rows_cols=list(map(int, box)), seconds=round(time.time() - tf, 1))
        thr = rec.pop("threshold", args.threshold)
        how = rec.pop("threshold_how", "fixed in advance (no tuning on held-out data)")
        mc.save_json(fdir / "run_record.json",
                     mc.run_record(args, paths, step=f"Table D fold {k}", model=args.model,
                                   year=args.year, radar=args.radar, seed=args.seed,
                                   threshold=thr, threshold_how=how,
                                   outputs=[mc.file_entry(pp)], **rec))
        n_done = todo.index(k) + 1
        left = (time.time() - t0) / n_done * (len(todo) - n_done)
        print(f"    fold {k} done in {rec['seconds'] / 60:.1f} min; about {left / 60:.1f} min "
              "left for this run", flush=True)

    done = [k for k in ks if (out / f"fold_{k}" / "run_record.json").exists()]
    if set(done) != set(ks) and not args.smoke:
        print(f"  folds done: {done} of {ks}. Run again without --folds to finish and make the map.")
        print("PASS (partial)")
        return 0
    # each fold's probabilities are cut at that fold's own threshold (the same
    # number for every fold unless --threshold inner), then joined as 0 / 1
    above = np.full(win.shape, np.nan, "float32")
    thresholds = {}
    for k in done:
        frec = mc.load_json(out / f"fold_{k}" / "run_record.json")
        thresholds[k] = float(frec.get("threshold", 0.5))
        with rasterio.open(out / f"fold_{k}" / f"prob_window_{k}.tif") as src:
            r0, r1, c0, c1 = frec["box_rows_cols"]
            sub = src.read(1)
        m = (win[r0:r1, c0:c1] == k) & np.isfinite(sub)
        above[r0:r1, c0:c1][m] = (sub[m] >= thresholds[k]).astype("float32")
    valid = (win > 0) & d["common"] & np.isin(win, done)
    fmap, diag = mc.make_flood_map(above, valid, 0.5, d["perm"], d["distort"])
    diag["threshold"] = thresholds[done[0]] if len(set(thresholds.values())) == 1 else "per fold"
    diag["threshold_per_fold"] = {str(k): v for k, v in thresholds.items()}
    mp = paths.lowo_map(args.year, args.model, args.radar)
    mc.write_flood_map(mp, fmap, prof)
    km2 = mc.cell_km2(prof)
    diag.update({"event": args.year, "model": args.model, "radar": args.radar,
                 "folds": done, "flood_km2": round(diag["flood_cells"] * km2, 4),
                 "common_cells_in_windows": int(((win > 0) & d["common"]).sum()),
                 "run_record": str(out / "run_record.json")})
    warn = mc.area_warning(diag["flood_km2"], args.smoke)
    diag["warnings"] = [warn] if warn else []
    dp = out / f"flood_lowo_{args.year}_{args.model}_{args.radar}_diagnostics.json"
    mc.save_json(dp, diag)
    extra = {}
    if tmv:
        extra["terramind_weights_sha256"] = sorted({
            mc.load_json(out / f"fold_{k}" / "run_record.json").get("terramind_weights_sha256")
            for k in done})
    mc.save_json(out / "run_record.json",
                 mc.run_record(args, paths, step="Table D assembly", model=args.model, **extra,
                               year=args.year, radar=args.radar, seed=args.seed, folds=done,
                               threshold=diag["threshold"],
                               threshold_per_fold=diag["threshold_per_fold"],
                               threshold_how=("chosen inside each fold from its training windows"
                                              if args.threshold == "inner" else
                                              "fixed in advance (no tuning on held-out data)"),
                               fold_records=[str(out / f"fold_{k}" / "run_record.json")
                                             for k in done],
                               seconds=round(time.time() - t0, 1),
                               outputs=[mc.file_entry(mp), mc.file_entry(dp)]))
    if warn:
        print("  " + warn.replace("on the corridor", "in the windows"))
    print(f"  map: {diag['flood_cells']:,} flood cells ({diag['flood_km2']:.3f} km2) in "
          f"{diag['valid_cells']:,} scored cells; permanent water off "
          f"{diag['permanent_removed_cells']:,}, layover/shadow off "
          f"{diag['distortion_removed_cells']:,}")
    print(f"  wrote {mp}")
    print(f"PASS  {name}")
    return 0


def selftest():
    import models_synthetic as syn
    print("26_corridor_lowo.py self-test")
    with tempfile.TemporaryDirectory(prefix="fyp selftest ", ignore_cleanup_errors=True) as td:
        proj = Path(td) / "Final Year Project"
        syn.build_project(proj, chips=False)
        paths = mc.Paths(proj)
        ok = main(["--project", str(proj), "--model", "rf", "--radar", "grd",
                   "--n-estimators", "10"]) == 0
        ok &= main(["--project", str(proj), "--model", "rf", "--radar", "rtc",
                    "--n-estimators", "10"]) == 0
        ok &= main(["--project", str(proj), "--model", "unet", "--radar", "grd", "--steps", "4",
                    "--crop", "32", "--batch-size", "2", "--base-width", "8", "--tile", "64",
                    "--overlap", "16", "--cpu-ok", "--year", "2023"]) == 0
        recs = [mc.load_json(paths.lowo_dir(2021, "rf", "grd") / f"fold_{k}" / "run_record.json")
                for k in range(1, 5)]
        with rasterio.open(paths.lowo_map(2021, "rf", "grd")) as a, \
                rasterio.open(paths.lowo_map(2021, "rf", "rtc")) as b:
            same = np.array_equal(a.read(1) == 255, b.read(1) == 255)
    checks = [("three runs finished", ok),
              ("held-out window never trained on",
               all(r["train_cells_in_heldout_window"] == 0 and r["heldout_window"]
                   not in r["train_windows"] for r in recs)),
              ("GRD and RTC maps scored on the same cells", same)]
    for name, c in checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {name}")
    good = all(c for _, c in checks)
    print("SELFTEST PASS" if good else "SELFTEST FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(mc.cli(main))
