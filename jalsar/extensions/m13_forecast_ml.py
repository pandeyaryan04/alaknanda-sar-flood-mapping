"""Jal-SAR extensions, M13: forecast flood maps from machine learning models.

The radar models (Random Forest, U-Net, TerraMind) map a flood from a radar image taken during or
after it, so they cannot forecast. These models take no radar at all. For each 10 m cell they see

    HAND    height above the nearest drainage, m (the strata of each orbit)
    SLOPE   slope, degrees
    DIST    distance to the permanent river (JRC plus dry-season Sentinel-2 water), m
    REACH   1 in the gorge above Devprayag, 0 in the open valley below
    QREL    GloFAS river flow at the reach's gauge divided by that gauge's 2-year flow
            (both from the same GloFAS version, so the 4.0 and 5.0 flows can be compared)

and say whether the cell is flooded. QREL is the only input that changes from one flood to the
next, so a forecast flow gives a forecast map.

Two label sets (both trained and compared):
  radar   the 6 Jal-SAR headline radar maps (U-Net). Many cells, but the labels are themselves model
          output, errors included. Left-out test: train on 5 events, test on the 6th.
  hand    the 2 hand-drawn references (October 2021, August 2023), inside the 8 windows of each.
          Clean labels but only 2 floods. Left-out test: train on one, test on the other.
Two models for each: a Random Forest and a small U-Net (5 input channels).

Steps (each keeps what it has finished, so running it again carries on):
  rf       Random Forests: the left-out folds, then the final forests on all events
  unet     U-Nets: the same (needs the GPU; --cpu-ok to allow the CPU, very slow)
  forecast only the maps for the newest forecast (quick; used by the daily run)
  report   scores against the left-out radar maps and against the hand references (Table D windows
           and bootstrap), the same for the M10 HAND rule, how the predicted area changes with the
           flow, a hindcast with the forecasts issued 1, 3, 7 and 14 days before each checked flood,
           and maps for the newest forecast (m9_forecast\\forecast_watch.json)

Needs (already made by M9 M10 FORECAST.bat): m10_forecast_extent\\calibration.json (the GloFAS flow
on each radar day), m9_forecast\\thresholds.csv (2-year flows), m9_forecast\\hindcast.csv and
m9_forecast\\forecast_watch.json.

Outputs in 06 Outputs\\jalsar\\extensions\\m13_forecast_ml\\:
  M13_FORECAST_ML.md (read first), scores_radar.csv, scores_reference.csv, q_sensitivity.csv,
  hindcast_ml.csv, forecast_ml.json, fig_q_sensitivity.png, fig_forecast_<label>_<model>.png,
  forecast_<label>_<model>.tif, folds\\, final\\

Run:  python m13_forecast_ml.py rf
      python m13_forecast_ml.py unet
      python m13_forecast_ml.py report
      python m13_forecast_ml.py --selftest
"""
import argparse
import csv
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

FEATURES = ("HAND", "SLOPE", "DIST", "REACH", "QREL")
CAPS = {"HAND": (0.0, 200.0, 50.0), "SLOPE": (0.0, 70.0, 45.0), "DIST": (0.0, 5000.0, 2000.0),
        "REACH": (0.0, 1.0, 1.0), "QREL": (0.0, 10.0, 1.0)}          # (low, high, divisor for the U-Net)
LABELS = ("radar", "hand")
MODELS = ("rf", "unet")
MODEL_NAMES = {"rf": "Random Forest", "unet": "U-Net"}
LABEL_NAMES = {"radar": "6 radar maps", "hand": "2 hand-drawn references"}
SEED = 20261007
RF_PER_CLASS = 60000          # cells per class per training event (radar labels)
RF_TREES = 150
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)
Q_TEST = (0.5, 1.0, 1.5, 2.0, 3.0)
LEADS = (1, 3, 7, 14)
GORGE = "gorge_above_devprayag"


def _ctx(project=None):
    import jalsar_core as gc
    gp = gc.GPaths(project)
    out = gp.o06 / "jalsar" / "extensions" / "m13_forecast_ml"
    return gc, gp, out


# =============================================================================
# Inputs
# =============================================================================

def read_q_inputs(gp):
    """Flow on each radar day per reach (M10), 2-year flows per gauge and version (M9)."""
    ext = gp.o06 / "jalsar" / "extensions"
    cal_p = ext / "m10_forecast_extent" / "calibration.json"
    thr_p = ext / "m9_forecast" / "thresholds.csv"
    for p in (cal_p, thr_p):
        if not p.exists():
            sys.exit(f"PROBLEM: {p.name} is missing. Run M9 M10 FORECAST.bat first.")
    cal = json.loads(cal_p.read_text(encoding="utf-8"))
    rp2 = {}
    with open(thr_p, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                rp2[(r["version"], r["station"])] = float(r["rp2"])
            except (TypeError, ValueError):
                pass
    qrel = {}
    for x in cal["events"]:
        v, q, g = x.get("glofas_version"), x.get("glofas_q"), x.get("gauge")
        if v and q is not None and np.isfinite(q) and (v, g) in rp2 and rp2[(v, g)] > 0:
            qrel[(x["event"], x["reach"])] = float(q) / rp2[(v, g)]
    return cal, rp2, qrel


class EventData:
    """Static inputs, labels and masks of one event, cut to the cells that matter."""

    def __init__(self, gc, gp, e):
        import jalsar_display as gd
        from scipy import ndimage
        mc = gc.mc
        d = gp.event_dir(e.id)
        flood, prof = gd.read_u8(d / "flood_headline.tif")
        trust, _ = gd.read_u8(d / "trust.tif")
        hand = mc.reproject_onto(e.strata_path() / "strata_hand_m.tif", prof)
        slope = mc.reproject_onto(e.strata_path() / "strata_slope_deg.tif", prof)
        perm = mc.reproject_onto(gp.perm_water(), prof, categorical=True) == 100
        reach, _ = gc.reach_raster(e, prof)
        cell = abs(prof["transform"].a)
        dist = ndimage.distance_transform_edt(~perm) * cell if perm.any() else np.full(flood.shape, 5000.0)
        radar_ok = np.isin(flood, (0, 1)) & (trust != 0) & (trust != mc.NODATA) & ~perm & np.isfinite(hand) \
            & np.isfinite(slope)
        rows = np.where((radar_ok | perm).any(axis=1))[0]
        cols = np.where((radar_ok | perm).any(axis=0))[0]
        if not len(rows):
            raise SystemExit(f"PROBLEM: event {e.id} has no valid cells")
        r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        sl = (slice(r0, r1), slice(c0, c1))
        self.e, self.id, self.prof, self.window = e, e.id, prof, (r0, r1, c0, c1)
        self.full_shape = flood.shape
        self.hand, self.slope, self.dist = hand[sl], slope[sl], dist[sl].astype("float32")
        self.gorge = (np.asarray(reach)[sl] == GORGE)
        self.perm, self.mask = perm[sl], radar_ok[sl]
        self.radar = (flood[sl] == 1)
        self.km2 = cell * cell / 1e6
        self.truth = None           # hand reference, 0/1, 255 elsewhere
        self.win, self.n_win = None, 0
        if e.validated_year:
            sc = gc.scorer
            truth = sc.read_band(str(gp.reference(e.validated_year)), flood.shape)
            win, n_win = sc.window_raster(str(gp.windows(e.validated_year)), flood.shape,
                                          prof["transform"], prof["crs"])
            self.truth, self.win, self.n_win = truth[sl], win[sl], n_win

    def labels(self, which):
        """(y int8: 1 flood, 0 dry, -1 not used) for a label set."""
        y = np.full(self.mask.shape, -1, "int8")
        if which == "radar":
            y[self.mask & self.radar] = 1
            y[self.mask & ~self.radar] = 0
        elif self.truth is not None:
            ok = self.mask & (self.win > 0) & np.isin(self.truth, (0, 1))
            y[ok & (self.truth == 1)] = 1
            y[ok & (self.truth == 0)] = 0
        return y

    def stack(self, qrel_by_reach, scaled=False):
        """(5, h, w) float32 inputs, NaN where not usable. qrel_by_reach: {reach: value}."""
        q = np.where(self.gorge, qrel_by_reach.get(GORGE, np.nan),
                     qrel_by_reach.get("open_valley_below_devprayag", np.nan)).astype("float32")
        x = np.stack([self.hand, self.slope, self.dist, self.gorge.astype("float32"), q]).astype("float32")
        for k, f in enumerate(FEATURES):
            lo, hi, div = CAPS[f]
            x[k] = np.clip(x[k], lo, hi)
            if scaled:
                x[k] = x[k] / div
        x[:, ~self.mask] = np.nan
        return x

    def full(self, arr, fill):
        """A cut array back onto the event's full grid."""
        out = np.full(self.full_shape, fill, dtype=arr.dtype)
        r0, r1, c0, c1 = self.window
        out[r0:r1, c0:c1] = arr
        return out


def load_all(gc, gp, events=None):
    evs = [e for e in gc.load_events(gp) if (gp.event_dir(e.id) / "flood_headline.tif").exists()]
    if events:
        evs = [e for e in evs if e.id in events]
    print(f"  reading {len(evs)} events ...")
    return {e.id: EventData(gc, gp, e) for e in evs}


def qrel_for(qrel, eid):
    return {r: qrel.get((eid, r), np.nan) for r in (GORGE, "open_valley_below_devprayag")}


# =============================================================================
# Scores
# =============================================================================

def counts(pred, truth, sel):
    p, t = pred[sel], truth[sel]
    return int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum()), int((~p & ~t).sum())


def csi_hit_far(tp, fp, fn):
    csi = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
    hit = tp / (tp + fn) if tp + fn else float("nan")
    far = fp / (tp + fp) if tp + fp else float("nan")
    return csi, hit, far


def score_radar(ed, flood):
    """Against the event's own radar map: pooled and per reach."""
    out = {}
    for name, sel in (("both reaches", ed.mask), ("gorge", ed.mask & ed.gorge), ("valley", ed.mask & ~ed.gorge)):
        tp, fp, fn, _ = counts(flood, ed.radar, sel)
        c, h, f = csi_hit_far(tp, fp, fn)
        out[name] = {"csi": c, "hit_rate": h, "far": f, "flood_km2": float(flood[sel].sum()) * ed.km2}
    return out


def score_reference(gc, ed, flood):
    """Against the hand reference with Table D's windows and bootstrap (as M10 scores its extent)."""
    if ed.truth is None:
        return None
    sc = gc.scorer
    pred = np.where(ed.mask | ed.perm, (flood & ~ed.perm).astype("uint8"), gc.mc.NODATA).astype("uint8")
    base = np.isin(pred, (0, 1)) & np.isin(ed.truth, (0, 1))
    rng = np.random.default_rng(sc.BOOT_SEED)
    draws = rng.integers(0, ed.n_win, size=(sc.N_BOOT, ed.n_win))
    weights = np.stack([np.bincount(dw, minlength=ed.n_win) for dw in draws])
    s = sc.score(sc.window_counts(pred, ed.truth, base, ed.win, ed.n_win), weights)
    return {k: s.get(k) for k in ("hit_rate", "hit_rate_lo", "hit_rate_hi", "far", "far_lo", "far_hi",
                                  "csi", "csi_lo", "csi_hi")} if s else None


def best_threshold(prob, y, w=None):
    """Threshold with the highest CSI for labels y (1/0) and probabilities prob (1-D), weighted."""
    w = np.ones_like(prob) if w is None else w
    best, bt = -1.0, 0.5
    for t in THRESHOLDS:
        p = prob >= t
        tp = w[p & (y == 1)].sum()
        fp = w[p & (y == 0)].sum()
        fn = w[~p & (y == 1)].sum()
        c = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        if c > best:
            best, bt = c, float(t)
    return bt, float(best)


# =============================================================================
# Folds
# =============================================================================

def folds(label, data, qrel):
    """(name, train ids, test id) for a label set. The final model has test id None.
    An event needs labels and a GloFAS flow for both reaches."""
    ids = [k for k, ed in data.items() if (ed.labels(label) >= 0).any()
           and all(np.isfinite(v) for v in qrel_for(qrel, k).values())]
    out = [(f"leftout_{t}", [i for i in ids if i != t], t) for t in ids] if len(ids) >= 2 else []
    return out + [("final", ids, None)], ids


# ----------------------------------------------------------------------------- Random Forest

def rf_sample(ed, y, x, rng, per_class):
    rows, ys, ws = [], [], []
    for cls in (1, 0):
        idx = np.flatnonzero((y == cls) & np.isfinite(x).all(axis=0))
        if not len(idx):
            continue
        take = rng.choice(idx, size=min(per_class, len(idx)), replace=False)
        rows.append(x.reshape(x.shape[0], -1)[:, take].T)
        ys.append(np.full(len(take), cls, "int8"))
        ws.append(np.full(len(take), len(idx) / len(take), "float64"))   # back to the true class shares
    return rows, ys, ws


def rf_fit(data, ids, label, qrel, per_class, trees, n_jobs):
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(SEED)
    X, Y, Wt = [], [], []
    for i in ids:
        ed = data[i]
        r, y, w = rf_sample(ed, ed.labels(label), ed.stack(qrel_for(qrel, i)), rng, per_class)
        X += r
        Y += y
        Wt += w
    X, Y, Wt = np.concatenate(X), np.concatenate(Y), np.concatenate(Wt)
    rf = RandomForestClassifier(n_estimators=trees, min_samples_leaf=5, n_jobs=n_jobs, oob_score=True,
                                random_state=SEED)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Some inputs do not have OOB scores")
        rf.fit(X, Y)
    oob = rf.oob_decision_function_[:, 1]
    ok = np.isfinite(oob)
    thr, csi_in = best_threshold(oob[ok], Y[ok], Wt[ok])
    return rf, thr, {"rows": int(len(Y)), "flood_rows": int(Y.sum()), "threshold": thr,
                     "threshold_rule": "best CSI on the out-of-bag predictions, cells weighted back to the true shares",
                     "oob_csi": round(csi_in, 4),
                     "importance": {f: round(float(v), 4) for f, v in zip(FEATURES, rf.feature_importances_)}}


def rf_prob(rf, x, chunk=1_000_000):
    flat = x.reshape(x.shape[0], -1)
    ok = np.isfinite(flat).all(axis=0)
    prob = np.full(flat.shape[1], np.nan, "float32")
    idx = np.flatnonzero(ok)
    for s in range(0, len(idx), chunk):
        part = idx[s:s + chunk]
        prob[part] = rf.predict_proba(flat[:, part].T)[:, 1]
    return prob.reshape(x.shape[1:])


# ----------------------------------------------------------------------------- U-Net

def unet_fit(data, ids, label, qrel, steps, base, batch, crop, device, log):
    import torch
    mc = __import__("models_common")
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    xs, ys = [], []
    for i in ids:
        ed = data[i]
        x = np.nan_to_num(ed.stack(qrel_for(qrel, i), scaled=True), nan=0.0)
        y = ed.labels(label)
        if min(y.shape) < crop:
            pad_h, pad_w = max(0, crop - y.shape[0]), max(0, crop - y.shape[1])
            x = np.pad(x, ((0, 0), (0, pad_h), (0, pad_w)))
            y = np.pad(y, ((0, pad_h), (0, pad_w)), constant_values=-1)
        xs.append(x)
        ys.append(y)
    centres = []
    for k, y in enumerate(ys):
        lab = np.argwhere(y >= 0)
        fl = np.argwhere(y == 1)
        centres.append((lab, fl))
    model = mc.UNet(len(FEATURES), base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    model.train()
    t0 = time.time()
    for step in range(steps):
        bx, by = [], []
        for _ in range(batch):
            k = int(rng.integers(len(xs)))
            lab, fl = centres[k]
            pool = fl if (len(fl) and rng.random() < 0.5) else lab
            cy, cx = pool[int(rng.integers(len(pool)))]
            H, W = ys[k].shape
            r = int(np.clip(cy - crop // 2 + rng.integers(-crop // 4, crop // 4 + 1), 0, H - crop))
            c = int(np.clip(cx - crop // 2 + rng.integers(-crop // 4, crop // 4 + 1), 0, W - crop))
            xx, yy = xs[k][:, r:r + crop, c:c + crop], ys[k][r:r + crop, c:c + crop]
            if rng.random() < 0.5:
                xx, yy = xx[:, :, ::-1], yy[:, ::-1]
            if rng.random() < 0.5:
                xx, yy = xx[:, ::-1, :], yy[::-1, :]
            bx.append(np.ascontiguousarray(xx))
            by.append(np.ascontiguousarray(yy))
        tx = torch.from_numpy(np.stack(bx)).to(device)
        ty = torch.from_numpy(np.stack(by).astype("float32")).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            logits = model(tx)
        loss, bce, dice = mc.masked_bce_dice(logits, ty)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if (step + 1) % max(1, steps // 10) == 0:
            log(f"      step {step + 1}/{steps}  loss {float(loss):.3f}  ({(time.time() - t0) / 60:.1f} min)")
    model.eval()
    return model


def unet_prob(model, ed, qrel_by, device, tile=512, overlap=64):
    mc = __import__("models_common")
    x = ed.stack(qrel_by, scaled=True)
    valid = np.isfinite(x).all(axis=0)
    fn = mc.unet_prob_fn(model, device, device.type == "cuda")
    prob, _ = mc.predict_tiled(fn, np.nan_to_num(x, nan=0.0), valid, tile, overlap, 4)
    return prob


def inner_split(label, data, ids):
    """(train ids, a function that gives the held-out labels) for the U-Net threshold, never the test event.
    radar labels: one training event held out. hand labels (one training event): two of its windows held out."""
    if label == "radar" and len(ids) >= 2:
        hold = ids[-1]
        return [i for i in ids if i != hold], hold, None
    ed = data[ids[0]]
    wins = [w for w in range(1, ed.n_win + 1) if ((ed.win == w) & (ed.labels(label) == 1)).any()]
    held = wins[-2:] if len(wins) >= 3 else wins[-1:]
    return ids, ids[0], held


# =============================================================================
# Running the folds
# =============================================================================

def write_u8(path, arr, prof):
    import rasterio
    p = dict(prof)
    p.update(count=1, dtype="uint8", nodata=255, compress="lzw")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr, 1)


def run_model(kind, args, project=None):
    gc, gp, out = _ctx(project)
    cal, rp2, qrel = read_q_inputs(gp)
    data = load_all(gc, gp)
    for i in data:
        q = qrel_for(qrel, i)
        print(f"  {i}: flow / 2-year flow, gorge {q[GORGE]:.2f}, valley {q['open_valley_below_devprayag']:.2f}")
    fold_dir = out / "folds"
    fin_dir = out / "final"
    fold_dir.mkdir(parents=True, exist_ok=True)
    fin_dir.mkdir(parents=True, exist_ok=True)
    device = None
    if kind == "unet":
        import torch
        mc = gc.mc
        device = mc.pick_device(args.cpu_ok)
        print(f"  device: {device}")
    for label in LABELS:
        flist, ids = folds(label, data, qrel)
        print(f"  labels: {LABEL_NAMES[label]} ({len(ids)} events: {', '.join(ids)})")
        if len(ids) < 2:
            print(f"  NOTE: fewer than 2 events with {label} labels; only a final model")
        for name, train, test in flist:
            tag = f"{label}_{kind}_{name}"
            meta_p = (fold_dir if test else fin_dir) / f"{tag}.json"
            if meta_p.exists() and not args.force:
                print(f"    {tag}: already done")
                continue
            t0 = time.time()
            print(f"    {tag}: training on {', '.join(train)}")
            meta = {"label": label, "model": kind, "train": train, "test": test}
            if kind == "rf":
                rf, thr, info = rf_fit(data, train, label, qrel, args.per_class, args.trees, args.n_jobs)
                meta.update(info)
                if test is None:
                    import joblib
                    joblib.dump(rf, fin_dir / f"{tag}.joblib", compress=3)
                else:
                    ed = data[test]
                    prob = rf_prob(rf, ed.stack(qrel_for(qrel, test)))
            else:
                import torch
                tr_in, hold, held_w = inner_split(label, data, train)
                if tr_in != train or held_w:
                    print(f"      threshold network: held out {hold}" + (f" windows {held_w}" if held_w else ""))
                    m_in = unet_fit(_hide(data, hold, held_w, label), tr_in, label, qrel, args.steps,
                                    args.base, args.batch, args.crop, device, print)
                    edh = data[hold]
                    ph = unet_prob(m_in, edh, qrel_for(qrel, hold), device)
                    yh = edh.labels(label)
                    if held_w:
                        yh = np.where(np.isin(edh.win, held_w), yh, -1)
                    ok = (yh >= 0) & np.isfinite(ph)
                    thr, c_in = best_threshold(ph[ok], yh[ok])
                    meta.update(threshold=thr, threshold_rule=f"best CSI on held-out {hold}"
                                + (f" windows {held_w}" if held_w else ""), threshold_csi=round(c_in, 4))
                    del m_in
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    meta.update(threshold=0.5, threshold_rule="0.5 (no event to hold out)")
                    thr = 0.5
                model = unet_fit(data, train, label, qrel, args.steps, args.base, args.batch, args.crop,
                                 device, print)
                meta.update(steps=args.steps, base_width=args.base, batch=args.batch, crop=args.crop)
                torch.save({"model": model.state_dict(), "meta": meta},
                           (fin_dir if test is None else fold_dir) / f"{tag}.pt")
                if test is not None:
                    prob = unet_prob(model, data[test], qrel_for(qrel, test), device)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            if test is not None:
                ed = data[test]
                flood = (prob >= meta["threshold"]) & ed.mask
                arr = np.where(ed.mask, flood.astype("uint8"), 255).astype("uint8")
                write_u8(fold_dir / f"{tag}.tif", ed.full(arr, 255), ed.prof)
            meta["minutes"] = round((time.time() - t0) / 60, 2)
            gc.mc.save_json(meta_p, meta)
            print(f"      done in {meta['minutes']} min, threshold {meta['threshold']}")
    print(f"PASS  {kind}: folds in {fold_dir}, final models in {fin_dir}")
    return 0


def _hide(data, hold, held_w, label):
    """A view of data where the threshold network cannot see the held-out labels."""
    if not held_w:
        return data
    ed = data[hold]

    class Masked:
        def __getattr__(self, k):
            return getattr(ed, k)

        def labels(self, which):
            y = ed.labels(which)
            return np.where(np.isin(ed.win, held_w), -1, y).astype("int8")

        def stack(self, q, scaled=False):
            return ed.stack(q, scaled)
    out = dict(data)
    out[hold] = Masked()
    return out


# =============================================================================
# Report
# =============================================================================

def load_final(gc, fin_dir, label, kind, device):
    tag = f"{label}_{kind}_final"
    meta_p = fin_dir / f"{tag}.json"
    if not meta_p.exists():
        return None, None
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    if kind == "rf":
        import joblib
        return joblib.load(fin_dir / f"{tag}.joblib"), meta
    import torch
    ck = torch.load(fin_dir / f"{tag}.pt", map_location="cpu", weights_only=False)
    m = gc.mc.UNet(len(FEATURES), int(meta["base_width"]))
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), meta


def predict(kind, model, ed, qrel_by, device):
    return rf_prob(model, ed.stack(qrel_by)) if kind == "rf" else unet_prob(model, ed, qrel_by, device)


def fold_model(gc, out, label, kind, test, data, qrel, device, args):
    """Retrain is not needed: the fold maps are on disk. Returns the left-out flood map (cut grid)."""
    p = out / "folds" / f"{label}_{kind}_leftout_{test}.tif"
    if not p.exists():
        return None
    import rasterio
    with rasterio.open(p) as r:
        a = r.read(1)
    ed = data[test]
    r0, r1, c0, c1 = ed.window
    return (a[r0:r1, c0:c1] == 1) & ed.mask


def m10_maps(gc, gp, data, cal):
    """The M10 HAND rule's left-out extent per event (constant link, as in the M10 report)."""
    import m10_rating_hand as m10
    model = cal.get("model", "constant")
    out = {}
    for i, ed in data.items():
        h_by = {}
        for r in (GORGE, "open_valley_below_devprayag"):
            lo = [x for x in cal.get("loeo", []) if x["event"] == i and x["reach"] == r]
            h_by[r] = lo[0].get(f"h_{model}") if lo else None
        reach = np.where(ed.gorge, GORGE, "open_valley_below_devprayag")
        ext = m10.extent(ed.hand, reach, ed.perm, h_by) == 1
        out[i] = ext & ed.mask
    return out, model


def run_report(args, project=None):
    gc, gp, out = _ctx(project)
    cal, rp2, qrel = read_q_inputs(gp)
    data = load_all(gc, gp)
    # events with no GloFAS flow (for example a Jal-SAR run made after M10 last ran) cannot be scored:
    # the models need the flow as an input. They are listed in the report, not scored.
    skipped = [k for k in data if not all(np.isfinite(v) for v in qrel_for(qrel, k).values())]
    for k in skipped:
        print(f"  {k}: no GloFAS flow in calibration.json (run M9 M10 FORECAST.bat to add it); not scored")
        del data[k]
    device = None
    have_unet = any((out / "final").glob("*_unet_final.json"))
    if have_unet:
        device = gc.mc.pick_device(True)
    fin = out / "final"
    rows_r, rows_t = [], []
    m10_by, m10_model = m10_maps(gc, gp, data, cal)
    maps = {}
    for i, ed in data.items():
        cand = [("M10 HAND rule (" + m10_model + ", left out)", m10_by[i], "m10")]
        for label in LABELS:
            for kind in MODELS:
                f = fold_model(gc, out, label, kind, i, data, qrel, device, args)
                note = "left out"
                if f is None:                   # this event was never a training event: the final model
                    mdl, meta = load_final(gc, fin, label, kind, device) if (kind == "rf" or have_unet) else (None, None)
                    if mdl is None:
                        continue
                    if i in meta["train"]:
                        continue                 # a final model has seen this event: no fair score
                    prob = predict(kind, mdl, ed, qrel_for(qrel, i), device)
                    f = (prob >= meta["threshold"]) & ed.mask
                    note = "not a training event"
                cand.append((f"{MODEL_NAMES[kind]}, {LABEL_NAMES[label]} ({note})", f, f"{label}_{kind}"))
        cand.append(("radar map (headline, for comparison)", ed.radar & ed.mask, "radar"))
        maps[i] = cand
        for name, f, key in cand:
            if key != "radar":
                for reach, s in score_radar(ed, f).items():
                    rows_r.append({"event": i, "map": name, "key": key, "reach": reach,
                                   **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}})
            s = score_reference(gc, ed, f)
            if s:
                rows_t.append({"event": i, "map": name, "key": key, **s})
    _write_csv(out / "scores_radar.csv", rows_r)
    _write_csv(out / "scores_reference.csv", rows_t)

    # how the predicted area changes with the flow (final models, on the newest event's grid)
    ref_id = sorted(data, key=lambda k: data[k].e.date)[-1]
    ed = data[ref_id]
    sens = []
    finals = {}
    for label in LABELS:
        for kind in MODELS:
            if kind == "unet" and not have_unet:
                continue
            mdl, meta = load_final(gc, fin, label, kind, device)
            if mdl is None:
                continue
            finals[(label, kind)] = (mdl, meta)
            for q in Q_TEST:
                prob = predict(kind, mdl, ed, {GORGE: q, "open_valley_below_devprayag": q}, device)
                f = (prob >= meta["threshold"]) & ed.mask
                sens.append({"label": label, "model": kind, "q_over_2yr": q,
                             "gorge_km2": round(float((f & ed.gorge).sum()) * ed.km2, 3),
                             "valley_km2": round(float((f & ~ed.gorge).sum()) * ed.km2, 3)})
    _write_csv(out / "q_sensitivity.csv", sens)
    _fig_sens(out, sens)

    # hindcast: the forecasts issued before each checked flood
    hind = []
    hp = gp.o06 / "jalsar" / "extensions" / "m9_forecast" / "hindcast.csv"
    import m9_glofas_read as gr
    if hp.exists():
        with open(hp, newline="", encoding="utf-8") as fh:
            hrows = list(csv.DictReader(fh))
        for i, edi in data.items():
            if edi.truth is None:
                continue
            for n in LEADS:
                issue = np.datetime64(edi.e.date, "D") - np.timedelta64(n, "D")
                fv = gr.forecast_version(issue)
                tv = fv if fv in ("version_4_0", "version_5_0") else "version_4_0"
                q_by = {}
                for r, g in cal["reach_gauges"].items():
                    rr = [x for x in hrows if x["event"] == i and x["station"] == g]
                    try:
                        q = float(rr[0].get(f"q50_ahead_{n}", "nan")) if rr else float("nan")
                    except ValueError:
                        q = float("nan")
                    q_by[r] = q / rp2[(tv, g)] if (tv, g) in rp2 and np.isfinite(q) else float("nan")
                if not all(np.isfinite(v) for v in q_by.values()):
                    hind.append({"event": i, "days_ahead": n, "note": "no forecast"})
                    continue
                for label in LABELS:
                    for kind in MODELS:
                        mdl, meta = _fold_or_final(gc, out, fin, label, kind, i, device, have_unet, data, qrel, args)
                        if mdl is None:
                            continue
                        prob = predict(kind, mdl, edi, q_by, device)
                        f = (prob >= meta["threshold"]) & edi.mask
                        s = score_reference(gc, edi, f) or {}
                        hind.append({"event": i, "days_ahead": n, "forecast_version": gr.VERSION_NAME.get(fv, fv),
                                     "thresholds_version": gr.VERSION_NAME.get(tv, tv),
                                     "qrel_gorge": round(q_by[GORGE], 3),
                                     "qrel_valley": round(q_by["open_valley_below_devprayag"], 3),
                                     "map": f"{MODEL_NAMES[kind]}, {LABEL_NAMES[label]}", **s})
    _write_csv(out / "hindcast_ml.csv", hind)

    fc_out = forecast_maps(gc, gp, out, cal, ed, ref_id, finals, device)
    gc.mc.save_json(out / "forecast_ml.json", fc_out)
    _write_md(out, data, rows_r, rows_t, sens, hind, fc_out, cal, m10_model, fin, qrel, skipped)
    print(f"PASS  report: {out / 'M13_FORECAST_ML.md'}")
    return 0


def forecast_maps(gc, gp, out, cal, ed, ref_id, finals, device):
    """Maps for the newest forecast (forecast_watch.json) from every final model, on ed's grid."""
    fw = gp.o06 / "jalsar" / "extensions" / "m9_forecast" / "forecast_watch.json"
    fc_out = {}
    if fw.exists() and finals:
        w = json.loads(fw.read_text(encoding="utf-8"))
        q_by, peak_day = {}, {}
        for r, g in cal["reach_gauges"].items():
            pt = next((p for p in w.get("points", []) if p["name"] == g), None)
            if not pt or not pt.get("days") or not pt.get("lower_m3s"):
                continue
            best = max(pt["days"], key=lambda d: d["q50"])
            q_by[r] = best["q50"] / pt["lower_m3s"] if w.get("rule", "rp") == "rp" else float("nan")
            peak_day[r] = best["date"]
        if len(q_by) == 2 and all(np.isfinite(v) for v in q_by.values()):
            fc_out = {"issued": w.get("issued"), "rule": w.get("rule"), "qrel": q_by, "peak_day": peak_day,
                      "grid_event": ref_id, "maps": []}
            for (label, kind), (mdl, meta) in finals.items():
                prob = predict(kind, mdl, ed, q_by, device)
                f = (prob >= meta["threshold"]) & ed.mask
                arr = np.where(ed.mask, f.astype("uint8"), 255).astype("uint8")
                tif = out / f"forecast_{label}_{kind}.tif"
                write_u8(tif, ed.full(arr, 255), ed.prof)
                _fig_map(out / f"fig_forecast_{label}_{kind}.png", ed, f,
                         f"{MODEL_NAMES[kind]} ({LABEL_NAMES[label]}), forecast issued {w.get('issued')}")
                fc_out["maps"].append({"label": label, "model": kind, "tif": str(tif),
                                       "gorge_km2": round(float((f & ed.gorge).sum()) * ed.km2, 3),
                                       "valley_km2": round(float((f & ~ed.gorge).sum()) * ed.km2, 3)})
    return fc_out


def run_forecast(args, project=None):
    """Only the newest-forecast maps (quick; for the daily run)."""
    gc, gp, out = _ctx(project)
    cal, rp2, qrel = read_q_inputs(gp)
    evs = [e for e in gc.load_events(gp) if (gp.event_dir(e.id) / "flood_headline.tif").exists()]
    ref = sorted(evs, key=lambda e: e.date)[-1]
    data = load_all(gc, gp, [ref.id])
    ed = data[ref.id]
    have_unet = any((out / "final").glob("*_unet_final.json"))
    device = gc.mc.pick_device(True) if have_unet else None
    finals = {}
    for label in LABELS:
        for kind in MODELS:
            if kind == "unet" and not have_unet:
                continue
            mdl, meta = load_final(gc, out / "final", label, kind, device)
            if mdl is not None:
                finals[(label, kind)] = (mdl, meta)
    if not finals:
        print("PROBLEM: no final models yet (run the rf and unet steps first)")
        return 1
    fc_out = forecast_maps(gc, gp, out, cal, ed, ref.id, finals, device)
    gc.mc.save_json(out / "forecast_ml.json", fc_out)
    if not fc_out.get("maps"):
        print("NOTE  no newest forecast with both reach gauges; nothing drawn")
        return 0
    for m in fc_out["maps"]:
        print(f"  {MODEL_NAMES[m['model']]}, {LABEL_NAMES[m['label']]}: gorge {m['gorge_km2']} km2, "
              f"valley {m['valley_km2']} km2")
    print(f"PASS  forecast maps for the forecast issued {fc_out['issued']}")
    return 0


def _fold_or_final(gc, out, fin, label, kind, test, device, have_unet, data, qrel, args):
    """For a hindcast on event test: a model that never saw it. The left-out fold's model is not kept on
    disk, so it is trained again here (Random Forest only: quick); the U-Net uses the final model only when
    the event was not one of its training events."""
    if kind == "unet" and not have_unet:
        return None, None
    mdl, meta = load_final(gc, fin, label, kind, device)
    if mdl is not None and test not in meta["train"]:
        return mdl, meta
    if kind == "rf":
        fm = out / "folds" / f"{label}_rf_leftout_{test}.json"
        if not fm.exists():
            return None, None
        m = json.loads(fm.read_text(encoding="utf-8"))
        key = (label, test)
        if key not in _RF_CACHE:
            rf, thr, info = rf_fit(data, m["train"], label, qrel, args.per_class, args.trees, args.n_jobs)
            _RF_CACHE[key] = (rf, {"threshold": m["threshold"], "train": m["train"]})
        return _RF_CACHE[key]
    p = out / "folds" / f"{label}_unet_leftout_{test}.pt"
    if p.exists():
        import torch
        ck = torch.load(p, map_location="cpu", weights_only=False)
        mm = gc.mc.UNet(len(FEATURES), int(ck["meta"]["base_width"]))
        mm.load_state_dict(ck["model"])
        return mm.to(device).eval(), ck["meta"]
    return None, None


_RF_CACHE = {}


def _write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r)) or ["note"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _fig_sens(out, sens):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not sens:
        return
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2), sharex=True)
    for (label, kind) in dict.fromkeys((s["label"], s["model"]) for s in sens):
        ss = [s for s in sens if s["label"] == label and s["model"] == kind]
        for ax, k in zip(axs, ("gorge_km2", "valley_km2")):
            ax.plot([s["q_over_2yr"] for s in ss], [s[k] for s in ss], marker="o",
                    label=f"{MODEL_NAMES[kind]}, {LABEL_NAMES[label]}")
    for ax, t in zip(axs, ("Gorge above Devprayag", "Open valley below Devprayag")):
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("GloFAS flow / 2-year flow")
        ax.set_ylabel("predicted flood, km2")
    axs[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "fig_q_sensitivity.png", dpi=170)
    plt.close(fig)


def _fig_map(path, ed, f, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    img = np.full(ed.mask.shape + (3,), 1.0)
    img[ed.mask] = (0.93, 0.93, 0.93)
    img[ed.perm] = (0.55, 0.7, 0.85)
    img[f] = (0.05, 0.25, 0.6)
    fig, ax = plt.subplots(figsize=(6, 6 * ed.mask.shape[0] / max(ed.mask.shape[1], 1)))
    ax.imshow(img, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _f(v, n=3):
    try:
        return "n/a" if v is None or not np.isfinite(float(v)) else f"{float(v):.{n}f}"
    except (TypeError, ValueError):
        return "n/a"


def _write_md(out, data, rows_r, rows_t, sens, hind, fc_out, cal, m10_model, fin, qrel=None, skipped=()):
    md = ["# M13: forecast flood maps from machine learning models", "",
          "Models that take no radar: HAND, slope, distance to the permanent river, the reach, and the GloFAS "
          "flow at the reach's gauge divided by its 2-year flow (QREL). Trained on two label sets: the "
          f"{LABEL_NAMES['radar']} (Jal-SAR headline maps) and the {LABEL_NAMES['hand']}. Every score of a model "
          "on an event is from a model that never saw that event.", "",
          "## 1. Flow on each radar day (QREL = GloFAS flow / 2-year flow, same GloFAS version)", "",
          "| Event | Gorge flow (m3/s) | Gorge QREL | Valley flow (m3/s) | Valley QREL |", "|---|---|---|---|---|"]
    V = "open_valley_below_devprayag"
    for i, ed in data.items():
        q = qrel_for(qrel or {}, i)
        md.append(f"| {i} | {_f(_q(cal, i, GORGE), 0)} | {_f(q[GORGE], 2)} | {_f(_q(cal, i, V), 0)} | {_f(q[V], 2)} |")
    if skipped:
        md += ["", "Not scored (no GloFAS flow for its date in calibration.json, because the Jal-SAR run was made "
               "after M10 last ran): " + ", ".join(skipped) + ". The models never trained on it."]
    md += ["", "QREL is the only input that changes between floods. With six floods the models have six values to "
           "learn its effect from, so section 4 checks whether the predicted area responds to it at all.", "",
           "## 2. Against the left-out radar map (both reaches)", "",
           "| Event | Map | CSI | Hit rate | False alarm ratio | Flood km2 |", "|---|---|---|---|---|---|"]
    for r in rows_r:
        if r["reach"] == "both reaches":
            md.append(f"| {r['event']} | {r['map']} | {_f(r['csi'])} | {_f(r['hit_rate'])} | {_f(r['far'])} | "
                      f"{_f(r['flood_km2'], 2)} |")
    mean = {}
    for r in rows_r:
        if r["reach"] == "both reaches":
            mean.setdefault(r["key"], []).append(r["csi"])
    md += ["", "Mean CSI over the events: " + ", ".join(f"{k} {_f(np.nanmean(v))}" for k, v in mean.items()) + ".",
           "Per-reach rows are in scores_radar.csv.", "",
           "## 3. Against the hand-drawn reference (Table D windows, 2000 resamples)", "",
           "| Event | Map | Hit rate | False alarm ratio | CSI (95 % range) |", "|---|---|---|---|---|"]
    for r in rows_t:
        md.append(f"| {r['event']} | {r['map']} | {_f(r.get('hit_rate'))} | {_f(r.get('far'))} | "
                  f"{_f(r.get('csi'))} ({_f(r.get('csi_lo'))} to {_f(r.get('csi_hi'))}) |")
    md += ["", "## 4. Does the flow change the map? (final models, on the grid of the newest event)", "",
           "| Label set | Model | Flow / 2-year flow | Gorge km2 | Valley km2 |", "|---|---|---|---|---|"]
    for s in sens:
        md.append(f"| {LABEL_NAMES[s['label']]} | {MODEL_NAMES[s['model']]} | {s['q_over_2yr']} | "
                  f"{_f(s['gorge_km2'], 2)} | {_f(s['valley_km2'], 2)} |")
    md += ["", "If the area hardly changes between 0.5 and 3 times the 2-year flow, the model has learnt the terrain "
           "(where floods happen) but not the flow (how big this one will be). Figure: fig_q_sensitivity.png.", "",
           "## 5. Hindcast: forecasts issued before each checked flood, against the hand reference", "",
           "| Event | Days ahead | Forecast | Map | QREL gorge / valley | CSI (95 % range) |", "|---|---|---|---|---|---|"]
    for h in hind:
        if "note" in h:
            md.append(f"| {h['event']} | {h['days_ahead']} | {h['note']} | | | |")
        else:
            md.append(f"| {h['event']} | {h['days_ahead']} | {h['forecast_version']} (thresholds "
                      f"{h['thresholds_version']}) | {h['map']} | {_f(h['qrel_gorge'], 2)} / {_f(h['qrel_valley'], 2)} | "
                      f"{_f(h.get('csi'))} ({_f(h.get('csi_lo'))} to {_f(h.get('csi_hi'))}) |")
    md += ["", "## 6. The newest forecast", ""]
    if fc_out.get("maps"):
        md += [f"Issued {fc_out['issued']}. Peak middle forecast over the watch days: QREL gorge "
               f"{_f(fc_out['qrel'][GORGE], 2)} (day {fc_out['peak_day'].get(GORGE)}), valley "
               f"{_f(fc_out['qrel']['open_valley_below_devprayag'], 2)} "
               f"(day {fc_out['peak_day'].get('open_valley_below_devprayag')}).", "",
               "| Label set | Model | Gorge km2 | Valley km2 | Map |", "|---|---|---|---|---|"]
        for m in fc_out["maps"]:
            md.append(f"| {LABEL_NAMES[m['label']]} | {MODEL_NAMES[m['model']]} | {_f(m['gorge_km2'], 2)} | "
                      f"{_f(m['valley_km2'], 2)} | {Path(m['tif']).name} |")
    else:
        md.append("No newest forecast with both reach gauges in forecast_watch.json, or no final model yet.")
    thr = []
    for jf in sorted(fin.glob("*_final.json")):
        try:
            t = json.loads(jf.read_text(encoding="utf-8")).get("threshold")
            if t is not None:
                thr.append(f"{jf.stem.replace('_final', '')} {float(t):.2f}")
        except (ValueError, OSError):
            pass
    md += ["", "## 7. Settings", "",
           "Final thresholds: " + (", ".join(thr) if thr else "n/a") + f" (tried {THRESHOLDS[0]:.2f} to "
           f"{THRESHOLDS[-1]:.2f}; a threshold at {THRESHOLDS[-1]:.2f} sits at the top of the range tried).", "",
           f"Inputs and caps: " + "; ".join(f"{k} {v[0]:g} to {v[1]:g}" for k, v in CAPS.items()) + ". "
           "Cells used: the same as M10 (radar map present, not radar blind, not permanent water, HAND known). "
           "Hand labels: inside the reference windows only. Random Forest: " + str(RF_TREES) + " trees, minimum 5 "
           "cells per leaf, up to " + str(RF_PER_CLASS) + " cells per class per event, threshold = best CSI on its "
           "out-of-bag predictions weighted back to the true class shares. U-Net: the project's U-Net with 5 input "
           "channels, random crops (half centred on flood), flips, masked BCE plus Dice, a fixed number of steps "
           "(no early stopping), threshold from a second network trained without one training event (radar labels) "
           "or without two windows of the training event (hand labels). Seed " + str(SEED) + ".",
           "", "## 8. Limits", "",
           "Six floods (two with hand references) are very few to learn a flow response from. The radar labels are "
           "model output; the hand labels cover 16 windows of two floods. GloFAS is a global model (about 5 km, one "
           "value a day). Every radar pass came after the peak, so the labels describe receding water. This is a "
           "study tool, not an official forecast: for warnings, follow CWC and the district administration.", ""]
    (out / "M13_FORECAST_ML.md").write_text("\n".join(md), encoding="utf-8")


def _q(cal, eid, reach):
    return next((x["glofas_q"] for x in cal["events"] if x["event"] == eid and x["reach"] == reach), float("nan"))


# =============================================================================
# Self-test
# =============================================================================

def selftest():
    print("m13_forecast_ml.py self-test (synthetic project, CPU)")
    import jalsar_testdata as td
    import jalsar_core as gc
    with tempfile.TemporaryDirectory(prefix="m13 selftest ", ignore_cleanup_errors=True) as tmp:
        root = Path(tmp) / "Final Year Project"
        gp = td.build_display_project(root)
        evs = [e for e in gc.load_events(gp) if (gp.event_dir(e.id) / "flood_headline.tif").exists()]
        ext = gp.o06 / "jalsar" / "extensions"
        (ext / "m10_forecast_extent").mkdir(parents=True, exist_ok=True)
        (ext / "m9_forecast").mkdir(parents=True, exist_ok=True)
        reaches = (GORGE, "open_valley_below_devprayag")
        gauges = {GORGE: "Rudraprayag (conf)", "open_valley_below_devprayag": "Rishikesh"}
        rows, loeo = [], []
        for k, e in enumerate(evs):
            for r in reaches:
                rows.append({"event": e.id, "reach": r, "gauge": gauges[r], "glofas_version": "version_4_0",
                             "glofas_q": 600.0 + 300 * k})
                loeo.append({"event": e.id, "reach": r, "h_constant": 3.0})
        gc.mc.save_json(ext / "m10_forecast_extent" / "calibration.json",
                        {"events": rows, "loeo": loeo, "model": "constant", "reach_gauges": gauges})
        with open(ext / "m9_forecast" / "thresholds.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["version", "station", "rp2"])
            for v in ("version_4_0", "version_5_0"):
                for g in gauges.values():
                    w.writerow([v, g, 850.0])
        with open(ext / "m9_forecast" / "hindcast.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["event", "station"] + [f"q50_ahead_{n}" for n in range(15)])
            for e in evs:
                for g in gauges.values():
                    w.writerow([e.id, g] + [900.0] * 15)
        gc.mc.save_json(ext / "m9_forecast" / "forecast_watch.json",
                        {"issued": "2026-10-28", "rule": "rp",
                         "points": [{"name": g, "lower_m3s": 850.0,
                                     "days": [{"date": "2026-10-29", "q50": 1200.0}]} for g in gauges.values()]})

        class A:
            force = False
            per_class = 300
            trees = 10
            n_jobs = 2
            cpu_ok = True
            steps = 6
            base = 4
            batch = 2
            crop = 32
        checks = [("rf folds and final", run_model("rf", A, root) == 0)]
        try:
            import torch  # noqa: F401
            checks.append(("unet folds and final", run_model("unet", A, root) == 0))
        except ImportError:
            print("  (torch not installed here: the U-Net part is skipped)")
        checks.append(("report", run_report(A, root) == 0))
        checks.append(("forecast step", run_forecast(A, root) == 0))
        out = gp.o06 / "jalsar" / "extensions" / "m13_forecast_ml"
        md = (out / "M13_FORECAST_ML.md").read_text(encoding="utf-8")
        # an event with no GloFAS flow (a Jal-SAR run made after M10) is listed, not scored
        last = evs[-1].id
        cal_p = ext / "m10_forecast_extent" / "calibration.json"
        cal_all = json.loads(cal_p.read_text(encoding="utf-8"))
        cal_cut = dict(cal_all, events=[r for r in cal_all["events"] if r["event"] != last])
        gc.mc.save_json(cal_p, cal_cut)
        ok_skip = run_report(A, root) == 0
        md2 = (out / "M13_FORECAST_ML.md").read_text(encoding="utf-8")
        sec2 = md2.split("## 2.")[1].split("## 3.")[0]
        checks.append(("event with no flow listed, not scored", ok_skip and "Not scored" in md2 and last not in sec2))
        gc.mc.save_json(cal_p, cal_all)
        run_report(A, root)
        md = (out / "M13_FORECAST_ML.md").read_text(encoding="utf-8")
        checks += [("report has the reference table", "## 3." in md and "Random Forest" in md),
                   ("radar scores written", (out / "scores_radar.csv").stat().st_size > 100),
                   ("forecast map written", any(out.glob("forecast_radar_rf.tif"))),
                   ("hindcast written", (out / "hindcast_ml.csv").stat().st_size > 50)]
        if "--show" in sys.argv:
            print(md)
        fj = json.loads((out / "forecast_ml.json").read_text(encoding="utf-8"))
        checks.append(("forecast json has maps", bool(fj.get("maps"))))
    for name, c in checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {name}")
    good = all(c for _, c in checks)
    print("SELFTEST PASS" if good else "SELFTEST FAIL")
    return 0 if good else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", nargs="?", choices=("rf", "unet", "report", "forecast"))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--show", action="store_true", help="self-test: print the report")
    ap.add_argument("--force", action="store_true", help="train again even where a result exists")
    ap.add_argument("--per-class", type=int, default=RF_PER_CLASS)
    ap.add_argument("--trees", type=int, default=RF_TREES)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--cpu-ok", action="store_true")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.step:
        print(__doc__)
        return 1
    t0 = time.time()
    rc = (run_model(args.step, args) if args.step in MODELS else run_forecast(args) if args.step == "forecast"
          else run_report(args))
    print(f"  ({(time.time() - t0) / 60:.1f} min)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
