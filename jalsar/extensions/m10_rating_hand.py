"""Jal-SAR extensions, item M10 (forecast option B): GloFAS discharge to river stage to a HAND flood extent,
calibrated on the Jal-SAR radar maps.

1. Gauge cells. Each gauge in GAUGE_station_reference.csv gets the GloFAS river
   cell that stands for it (m9_glofas_read.match_cells; the choice is listed).
2. Stage relation. GloFAS reanalysis daily discharge against the GUARDIAN daily
   mean stage (GAUGE_wse_clean.csv, spikes left out), 2020 to July 2024:
   stage = h0 + a * Q^b, fitted by least squares, once for each reanalysis
   version found (4.0 and 5.0), so a forecast can be turned into a
   stage with the relation of its own version. GloFAS discharge is modelled, so
   this is a GloFAS-to-stage relation for the gauge, not a measured rating curve.
   The fit is also tried with GloFAS one day earlier and one day later and all
   three are reported; the same-day fit is used. The CWC warning and danger
   stages are turned into discharges through it.
3. HAND threshold of each radar map. For each Jal-SAR event and each reach (the
   gorge above Devprayag, the open valley below), the HAND value H that makes
   "HAND below H" agree best (highest CSI) with the radar flood map, leaving out
   permanent water and radar-blind cells. H is tried from 0.05 m to 1 m in 0.05 m
   steps, then to 30 m in 0.25 m steps. The GloFAS discharge on the radar day comes
   from --version (default 4.0) and, for a day that version does not reach, from
   the next version, through that version's own stage relation.
4. Link. Each reach has a reference gauge (--gauge-gorge, --gauge-valley). The
   stage at that gauge on the radar day (from step 2, so the same route the
   forecast takes) is compared with H. Three ways of predicting H are tested by
   leaving one event out at a time:
     offset     H = stage - c            (c: median over the other events)
     linear     H = alpha + beta * stage (least squares on the other events)
     constant   H = median of the other events' H (ignores the forecast; the baseline)
   The CSI each one gives on the left-out map is reported next to the best
   possible CSI. --model picks the one used for the forecast extent. The default,
   auto, takes whichever of the three has the highest mean left-out CSI and says
   which it took; you can fix one instead.
5. On the two validated events the HAND extent is also scored against the
   reference with the same windows and bootstrap as Table D (hit rate, false
   alarm ratio, CSI).

Outputs in 06 Outputs\\jalsar\\extensions\\m10_forecast_extent\\:
  calibration.json        everything the forecast extent needs (read by m9_forecast_watch.py)
  M10_RATING_HAND.md      the report (read first)
  rating_<gauge>.png, csi_vs_hand_<event>.png, loeo.csv

Run:  python m10_rating_hand.py
      python m10_rating_hand.py --gauge-gorge "Rudraprayag (A)" --gauge-valley "Rishikesh" --model offset
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import m9_glofas_read as gr  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = gr.ROOT / "06 Outputs" / "jalsar" / "extensions" / "m10_forecast_extent"
# HAND thresholds tried: 0.05 m steps to 1 m, then 0.25 m steps to 30 m (the 0.25 m start was the best
# value for every valley map on 25 Oct 2026, so the best may lie lower; choice 2a)
H_GRID = np.unique(np.round(np.concatenate([np.arange(0.05, 1.0, 0.05), np.arange(1.0, 30.01, 0.25)]), 2))
MODELS = ("offset", "linear", "constant")


def rating_tag(name, version, primary):
    base = name.replace(" ", "_").replace("(", "").replace(")", "")
    return base if version == primary else f"{base}_{version.replace('version_', 'v')}"


def fit_station(s, hist, cell, version, primary):
    """Stage relation of one gauge against one reanalysis version (same-day fit used; a day earlier and
    later also tried) and the plot. Returns (record, daily discharge at the gauge cell)."""
    q = hist["q"][:, cell["i"], cell["j"]]
    st = gr.daily_stage(s["name"])
    rec = {"cell": cell, "warning_stage_m": s["warning_stage_m"], "danger_stage_m": s["danger_stage_m"],
           "version": version}
    if st is None:
        return rec, q
    idx = {d: k for k, d in enumerate(hist["day"])}
    days = st.index.values.astype("datetime64[D]")
    lags = {}
    for lag in (-1, 0, 1):
        k = np.array([idx.get(d - np.timedelta64(lag, "D"), -1) for d in days])
        ok = k >= 0
        lags[lag] = gr.fit_rating(q[k[ok]], st.values[ok])
    r = lags[0]
    rec["fit_lags"] = {str(k): (None if v is None else {"r2": round(v["r2"], 3), "rmse_m": round(v["rmse_m"], 3)})
                       for k, v in lags.items()}
    if not r:
        return rec, q
    rec["rating"] = r
    # the power fit allows 0.05 <= b <= 1.5 (fit_rating); a b on either edge means the power curve does not
    # describe this gauge well. Where b sits on the 0.05 edge, fit_rating takes the log curve instead
    # instead, which is what the power curve tends to as b goes to 0; then the curve is not at a limit.
    pb = r.get("power_b", r.get("b"))
    rec["power_b_at_limit"] = bool(pb is not None and (pb <= 0.0501 or pb >= 1.4999))
    rec["b_at_limit"] = bool(r.get("shape", "power") == "power" and rec["power_b_at_limit"])
    for k in ("warning", "danger"):
        hh = s[f"{k}_stage_m"]
        if np.isfinite(hh):
            qq = float(gr.q_from_stage(r, hh))
            rec[f"q_{k}"] = qq
            # beyond the data: the stage is above the highest stage fitted, or the discharge it gives is
            # above the highest discharge fitted
            rec[f"q_{k}_extrapolated"] = bool(hh > r["h_range"][1] or qq > r["q_range"][1])
    k = np.array([idx.get(d, -1) for d in days])
    ok = k >= 0
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.scatter(q[k[ok]], st.values[ok], s=3, alpha=0.4, color="#334155")
    q_top = min(rec.get("q_danger", 0), 3 * r["q_range"][1])     # a log curve's CWC flow can be huge; keep the plot readable
    xs = np.linspace(max(r["q_range"][0], 1), max(r["q_range"][1], q_top * 1.05), 200)
    ax.plot(xs, gr.stage_from_q(r, xs), color="#b91c1c")
    for k2, col in (("warning_stage_m", "#f59e0b"), ("danger_stage_m", "#dc2626")):
        if np.isfinite(s[k2]):
            ax.axhline(s[k2], ls="--", lw=0.8, color=col)
    ax.set_xlabel(f"GloFAS discharge ({gr.VERSION_NAME.get(version, version)}), m3/s (same day)")
    ax.set_ylabel("GUARDIAN stage, m")
    ax.set_title(f"{s['name']}: {curve_text(r)}, R2 {r['r2']:.2f}, RMSE {r['rmse_m']:.2f} m", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"rating_{rating_tag(s['name'], version, primary)}.png", dpi=180)
    plt.close(fig)
    return rec, q


def shared_cell_note(rating_by_version):
    """A sentence when two gauges sit on one GloFAS cell (without the upstream area file, the two Rudraprayag gauges did)."""
    for rv in rating_by_version.values():
        cells = {}
        for n, rec in rv.items():
            c = rec["cell"]
            cells.setdefault((round(c["cell_lat"], 3), round(c["cell_lon"], 3)), []).append(n)
        same = [v for v in cells.values() if len(v) > 1]
        if same:
            return (" " + "; ".join(" and ".join(v) for v in same) + " share one GloFAS cell, so their stages "
                    "differ only through their stage relations.")
        return " Every gauge has its own GloFAS cell (upstream area file)."
    return ""


def curve_text(r):
    """The fitted curve in words, for the reports and plots."""
    if not r:
        return "no fit"
    if r.get("shape") == "log":
        return f"stage = {r['h0']:.2f} + {r['a']:.3f} ln(Q)"
    return f"stage = {r['h0']:.2f} + {r['a']:.4f} Q^{r['b']:.2f}"


# the gauges tried for each reach in the gauge loop. The reach boundary is Devprayag.
GAUGE_CANDIDATES = {"gorge_above_devprayag": ("Rudraprayag (A)", "Rudraprayag (conf)", "Srinagar"),
                    "open_valley_below_devprayag": ("Devprayag (A)", "Rishikesh")}


def rating_for(cal, gauge, version):
    """The stage relation of a gauge for a reanalysis version; the calibration version's if that one has none."""
    rb = (cal.get("rating_by_version") or {}).get(version, {}).get(gauge, {}).get("rating")
    return rb or cal["rating"].get(gauge, {}).get("rating")


def _core():
    import jalsar_core as gc
    return gc


def event_layers(gc, gp, e):
    """flood (bool), compare mask (bool), hand (float), reach (object), profile for one event."""
    import jalsar_display as gd
    d = gp.event_dir(e.id)
    flood, prof = gd.read_u8(d / "flood_headline.tif")
    trust, _ = gd.read_u8(d / "trust.tif")
    hand = gc.mc.reproject_onto(e.strata_path() / "strata_hand_m.tif", prof)
    perm = gc.mc.reproject_onto(gp.perm_water(), prof, categorical=True) == 100
    reach, _ = gc.reach_raster(e, prof)
    mask = np.isin(flood, (0, 1)) & (trust != 0) & (trust != gc.mc.NODATA) & ~perm & np.isfinite(hand)
    return flood == 1, mask, hand, reach, prof, perm


def csi_curve(flood, mask, hand, sel):
    """CSI, hit rate and FAR of 'HAND < H' against the radar map, for every H in H_GRID."""
    m = mask & sel
    hf, hd = hand[m & flood], hand[m & ~flood]
    tp = np.searchsorted(np.sort(hf), H_GRID, side="left").astype(float)
    fp = np.searchsorted(np.sort(hd), H_GRID, side="left").astype(float)
    fn = len(hf) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        csi = tp / (tp + fp + fn)
        hit = tp / (tp + fn)
        far = fp / (tp + fp)
    return csi, hit, far, int(len(hf))


def predict_h(model, s_train, h_train, s_new):
    s_train, h_train = np.asarray(s_train, float), np.asarray(h_train, float)
    if model == "constant" or len(s_train) == 0:
        return float(np.median(h_train)) if len(h_train) else float("nan")
    if model == "offset":
        return float(s_new - np.median(s_train - h_train))
    if len(s_train) < 3 or np.ptp(s_train) == 0:
        return float(np.median(h_train))
    b, a = np.polyfit(s_train, h_train, 1)
    return float(a + b * s_new)


def extent(hand, reach, perm, h_by_reach):
    """uint8 1 where HAND is below the reach's H (permanent water left out), else 0."""
    out = np.zeros(hand.shape, "uint8")
    for r, h in h_by_reach.items():
        if h is None or not np.isfinite(h) or h <= 0:
            continue
        out[(reach == r) & np.isfinite(hand) & (hand < h) & ~perm] = 1
    return out


def h_for_stage(cal, stage_by_reach, leave_out=None):
    """H per reach from the stage at each reach's gauge, with the calibrated model.
    leave_out: an event id whose own map is left out of the calibration (for an honest hindcast)."""
    out = {}
    for r, s in stage_by_reach.items():
        c = cal["reaches"].get(r)
        if not c or s is None or not np.isfinite(s):
            out[r] = None
            continue
        keep = [k for k, ev in enumerate(c.get("events", [])) if ev != leave_out] or list(range(len(c["stages"])))
        h = predict_h(cal["model"], [c["stages"][k] for k in keep], [c["h_best"][k] for k in keep], s)
        out[r] = float(np.clip(h, 0, H_GRID[-1]))
    return out


def score_extent(gc, gp, e, h_by):
    """Hit rate, false alarm ratio and CSI (with 95 % ranges) of the HAND extent for h_by against the
    reference of a validated event, with Table D's windows and bootstrap."""
    flood, mask, hand, reach, prof, perm = event_layers(gc, gp, e)
    ext = extent(hand, reach, perm, h_by)
    pred = np.where(mask | perm, ext, gc.mc.NODATA).astype("uint8")
    truth = gc.scorer.read_band(str(gp.reference(e.validated_year)), pred.shape)
    win, n_win = gc.scorer.window_raster(str(gp.windows(e.validated_year)), pred.shape,
                                         prof["transform"], prof["crs"])
    base = np.isin(pred, (0, 1)) & np.isin(truth, (0, 1))
    rng = np.random.default_rng(gc.scorer.BOOT_SEED)
    draws = rng.integers(0, n_win, size=(gc.scorer.N_BOOT, n_win))
    weights = np.stack([np.bincount(dw, minlength=n_win) for dw in draws])
    s = gc.scorer.score(gc.scorer.window_counts(pred, truth, base, win, n_win), weights)
    return {k: s.get(k) for k in ("hit_rate", "hit_rate_lo", "hit_rate_hi", "far", "far_lo", "far_hi",
                                  "csi", "csi_lo", "csi_hi")} if s else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gauge-gorge", default="Rudraprayag (conf)")    # chosen with the gauge loop (section 4b)
    ap.add_argument("--gauge-valley", default="Rishikesh")             # chosen with the gauge loop (section 4b)
    # Fixed to constant (25 Oct 2026). On 6 events linear beat constant by 0.001 left-out CSI
    # (0.292 against 0.291), far inside the bootstrap ranges, so the stage is not shown to add anything.
    # --model auto still lets the left-out CSI choose.
    ap.add_argument("--model", choices=MODELS + ("auto",), default="constant")
    ap.add_argument("--version", default="version_4_0")
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    gc = _core()
    gp = gc.GPaths()

    # 1-2. gauge cells and stage relations, one set per reanalysis version, so each forecast uses its own
    # version. The calibration
    # version (--version) gives the gauge cells; the other versions use the same cells (same grid).
    hfiles, hver = gr.historical_files(args.version)
    if not hfiles:
        print("PROBLEM: no GloFAS reanalysis files yet (run m9_glofas_download.py first)")
        return 1
    hists = gr.open_versions(args.version)
    if args.version not in hists:            # no file carries the asked version: the old behaviour
        hists = {hver: gr.open_historical(hfiles)}
        args.version = hver
    hist = hists[args.version]
    sts = gr.stations()
    cells = gr.match_cells(hist, sts, uparea=gr.load_uparea())
    rating_by_version, series_by_version = {}, {}
    for v, h in hists.items():
        cv = cells if gr.same_grid(h, hist) else gr.match_cells(h, sts, uparea=gr.load_uparea())
        rating_by_version[v], series_by_version[v] = {}, {}
        for s in sts:
            rec, q = fit_station(s, h, cv[s["name"]], v, args.version)
            rating_by_version[v][s["name"]], series_by_version[v][s["name"]] = rec, q
    rating = rating_by_version[args.version]
    reach_gauge = {gc.REACHES[0]: args.gauge_gorge, gc.REACHES[1]: args.gauge_valley}
    for r, g in reach_gauge.items():
        if not rating.get(g, {}).get("rating"):
            print(f"PROBLEM: no stage relation for the {r} gauge '{g}' (pick another with --gauge-...)")
            return 1
    day_index_by_version = {v: {d: k for k, d in enumerate(h["day"])} for v, h in hists.items()}

    def q_on(g, d):
        """(discharge, version) at gauge g on day d: the calibration version when it covers the day,
        otherwise the next version that does (so radar days after the end of 4.0 still get a value)."""
        for v in hists:
            k = day_index_by_version[v].get(d)
            if k is not None and np.isfinite(series_by_version[v][g][k]) and \
                    rating_by_version[v].get(g, {}).get("rating"):
                return float(series_by_version[v][g][k]), v
        return float("nan"), None

    # 3. best HAND threshold per event and reach
    evs = [e for e in gc.load_events(gp) if (gp.event_dir(e.id) / "flood_headline.tif").exists()]
    rows, curves, layers_cache = [], {}, {}
    for e in evs:
        flood, mask, hand, reach, prof, perm = event_layers(gc, gp, e)
        d = np.datetime64(e.date, "D")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for r in gc.REACHES:
            csi, hit, far, nflood = csi_curve(flood, mask, hand, reach == r)
            k = int(np.nanargmax(np.nan_to_num(csi, nan=-1)))
            g = reach_gauge[r]
            q, qv = q_on(g, d)
            stage = (float(gr.stage_from_q(rating_by_version[qv][g]["rating"], q))
                     if qv is not None and np.isfinite(q) else float("nan"))
            st_obs = gr.daily_stage(g)
            obs = float(st_obs.get(pd.Timestamp(e.date), np.nan)) if st_obs is not None else float("nan")
            rows.append({"event": e.id, "reach": r, "gauge": g, "glofas_version": qv, "glofas_q": q,
                         "stage_from_glofas": stage,
                         "stage_observed": obs, "h_best": float(H_GRID[k]), "csi_best": float(csi[k]),
                         "flood_cells_radar": nflood, "validated_year": e.validated_year})
            curves[(e.id, r)] = csi
            ax.plot(H_GRID, csi, label=r.replace("_", " "))
        ax.set_xlabel("HAND threshold H, m"); ax.set_ylabel("CSI against the radar map")
        ax.set_title(e.id, fontsize=9); ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(OUT / f"csi_vs_hand_{e.id}.png", dpi=180); plt.close(fig)
        layers_cache[e.id] = e

    # 4. leave one event out
    loeo = []
    for r in gc.REACHES:
        rr = [x for x in rows if x["reach"] == r and np.isfinite(x["stage_from_glofas"])]
        for x in rr:
            others = [y for y in rr if y["event"] != x["event"]]
            s_tr = [y["stage_from_glofas"] for y in others]
            h_tr = [y["h_best"] for y in others]
            out = {"event": x["event"], "reach": r, "h_best": x["h_best"], "csi_best": round(x["csi_best"], 4)}
            for m in MODELS:
                hp = float(np.clip(predict_h(m, s_tr, h_tr, x["stage_from_glofas"]), H_GRID[0], H_GRID[-1]))
                k = int(np.abs(H_GRID - hp).argmin())
                out[f"h_{m}"] = round(hp, 2)
                out[f"csi_{m}"] = round(float(np.nan_to_num(curves[(x["event"], r)][k])), 4)
            loeo.append(out)
    with open(OUT / "loeo.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(loeo[0].keys()) if loeo else ["event"])
        w.writeheader(); w.writerows(loeo)

    mean_csi = {m: (float(np.mean([x[f"csi_{m}"] for x in loeo])) if loeo else None) for m in MODELS + ("best",)}

    # 4b. gauge loop: every candidate gauge of each reach through the same left-out test.
    # The best H of each map does not depend on the gauge; only the stage on the radar day does.
    ev_date = {e.id: np.datetime64(e.date, "D") for e in evs}
    gauge_loop = []
    for r in gc.REACHES:
        base = [x for x in rows if x["reach"] == r]
        for g in GAUGE_CANDIDATES.get(r, ()):
            rg = rating_by_version[args.version].get(g, {}).get("rating")
            if not rg:
                gauge_loop.append({"reach": r, "gauge": g, "note": "no stage relation (no GUARDIAN data)"})
                continue
            pts = []
            for x in base:
                q, qv = q_on(g, ev_date[x["event"]])
                if qv is None or not np.isfinite(q):
                    continue
                pts.append((x["event"], float(gr.stage_from_q(rating_by_version[qv][g]["rating"], q)), x["h_best"]))
            res = {"reach": r, "gauge": g, "in_use": g == reach_gauge[r], "events": len(pts),
                   "curve": curve_text(rg), "r2_stage_fit": round(rg["r2"], 3),
                   "stage_min_m": round(min(p[1] for p in pts), 2) if pts else None,
                   "stage_max_m": round(max(p[1] for p in pts), 2) if pts else None}
            if len(pts) >= 3 and np.ptp([p[1] for p in pts]) > 0:
                res["corr_stage_hbest"] = round(float(np.corrcoef([p[1] for p in pts], [p[2] for p in pts])[0, 1]), 3)
            for m in MODELS:
                cs = []
                for ev, st, _ in pts:
                    oth = [p for p in pts if p[0] != ev]
                    hp = float(np.clip(predict_h(m, [o[1] for o in oth], [o[2] for o in oth], st), H_GRID[0], H_GRID[-1]))
                    cs.append(float(np.nan_to_num(curves[(ev, r)][int(np.abs(H_GRID - hp).argmin())])))
                res[f"loeo_csi_{m}"] = round(float(np.mean(cs)), 4) if cs else None
            gauge_loop.append(res)
    if gauge_loop:
        keys = list(dict.fromkeys(k for x in gauge_loop for k in x))
        with open(OUT / "gauge_loop.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader(); w.writerows(gauge_loop)
    chosen = args.model
    if chosen == "auto":
        cand = [m for m in MODELS if mean_csi[m] is not None]
        chosen = max(cand, key=lambda m: (round(mean_csi[m], 4), m == "constant")) if cand else "constant"

    # 5. validated events against the reference (Table D windows and bootstrap)
    ref_scores = []
    for e in evs:
        if not e.validated_year:
            continue
        flood, mask, hand, reach, prof, perm = event_layers(gc, gp, e)
        h_by = {}
        for r in gc.REACHES:
            lo = [x for x in loeo if x["event"] == e.id and x["reach"] == r]
            h_by[r] = lo[0][f"h_{chosen}"] if lo else None
        ext = extent(hand, reach, perm, h_by)
        pred = np.where(mask | perm, ext, gc.mc.NODATA).astype("uint8")
        truth = gc.scorer.read_band(str(gp.reference(e.validated_year)), pred.shape)
        win, n_win = gc.scorer.window_raster(str(gp.windows(e.validated_year)), pred.shape,
                                             prof["transform"], prof["crs"])
        base = np.isin(pred, (0, 1)) & np.isin(truth, (0, 1))
        rng = np.random.default_rng(gc.scorer.BOOT_SEED)
        draws = rng.integers(0, n_win, size=(gc.scorer.N_BOOT, n_win))
        weights = np.stack([np.bincount(dw, minlength=n_win) for dw in draws])
        for name, arr in (("HAND extent (" + chosen + ", left out)", pred),
                          ("radar map (headline)", np.where(base, flood.astype("uint8"), gc.mc.NODATA))):
            s = gc.scorer.score(gc.scorer.window_counts(arr, truth, base, win, n_win), weights)
            if s:
                ref_scores.append({"event": e.id, "map": name, **{k: s.get(k) for k in
                                   ("hit_rate", "hit_rate_lo", "hit_rate_hi", "far", "far_lo", "far_hi",
                                    "csi", "csi_lo", "csi_hi")}})

    reaches = {}
    for r in gc.REACHES:
        rr = [x for x in rows if x["reach"] == r and np.isfinite(x["stage_from_glofas"])]
        reaches[r] = {"gauge": reach_gauge[r], "stages": [x["stage_from_glofas"] for x in rr],
                      "h_best": [x["h_best"] for x in rr], "events": [x["event"] for x in rr]}
    cal = {"script": "m10_rating_hand.py", "model": chosen, "model_asked": args.model, "glofas_reanalysis": args.version,
           "reanalysis_files": hist["files"], "reach_gauges": reach_gauge, "reaches": reaches,
           "rating": rating, "rating_by_version": rating_by_version,
           "reanalysis_days": {v: [str(h["day"][0]), str(h["day"][-1])] for v, h in hists.items()},
           "events": rows, "loeo": loeo, "loeo_mean_csi": mean_csi, "gauge_loop": gauge_loop,
           "reference_scores": ref_scores,
           "hand_grid_m": {"first": float(H_GRID[0]), "last": float(H_GRID[-1]),
                           "steps": "0.05 m to 1 m, then 0.25 m"},
           "day_convention": "a 24 h value is stamped with the end of its day",
           "minutes": round((time.time() - t0) / 60, 1)}
    gc.mc.save_json(OUT / "calibration.json", cal)

    f2 = lambda v, n=2: "n/a" if v is None or not np.isfinite(v) else f"{v:.{n}f}"  # noqa: E731
    vname = lambda v: gr.VERSION_NAME.get(v, v or "none")  # noqa: E731
    md = ["# M10: GloFAS discharge to stage to HAND flood extent, calibrated on the radar maps", "",
          f"GloFAS reanalysis used for the calibration: {args.version} ({len(hist['files'])} files, "
          f"{str(hist['day'][0])} to {str(hist['day'][-1])}). "
          + " ".join(f"Also read: {v} ({str(h['day'][0])} to {str(h['day'][-1])})." for v, h in hists.items()
                     if v != args.version)
          + " A radar day after the end of the calibration version takes its discharge from the next version, "
          "through that version's own stage relation (column \"GloFAS version\" in section 3). "
          "Day convention: a GloFAS 24 h value is stamped with the end of its day.", "",
          "## 1-2. Gauge cells and stage relations (stage = h0 + a * Q^b)", "",
          "One table per reanalysis version, same gauge cells. The forecast watch (M9) uses the table of the "
          "version that made each forecast: 4.0 for forecasts from 26 July 2023, 5.0 from 15 July 2026.", ""]
    for v, rv in rating_by_version.items():
        md += [f"### {vname(v)} reanalysis ({v}){' (calibration version)' if v == args.version else ''}", "",
               "| Gauge | GloFAS cell | How chosen | Mean Q m3/s | Fit R2 (Q a day earlier / same day / a day later) | RMSE m | Q at warning stage | Q at danger stage |",
               "|---|---|---|---|---|---|---|---|"]
        for n, rec in rv.items():
            c = rec["cell"]
            fl = rec.get("fit_lags", {})
            lag = " / ".join(f2((fl.get(k) or {}).get("r2")) for k in ("1", "0", "-1")) if fl else "no stage data"
            r = rec.get("rating")
            mq = float(np.nanmean(series_by_version[v][n]))
            qw = f"{rec['q_warning']:.0f}{' (beyond the data)' if rec.get('q_warning_extrapolated') else ''}" if "q_warning" in rec else "n/a"
            qd = f"{rec['q_danger']:.0f}{' (beyond the data)' if rec.get('q_danger_extrapolated') else ''}" if "q_danger" in rec else "n/a"
            md.append(f"| {n} | {c['cell_lat']:.3f} N, {c['cell_lon']:.3f} E ({c['offset_km']:.1f} km off) | {c['rule']} | "
                      f"{mq:.0f} | {lag} | {f2(r['rmse_m'] if r else None)} | {qw} | {qd} |")
        at_lim = [n for n, rec in rv.items() if rec.get("b_at_limit")]
        to_log = [n for n, rec in rv.items() if rec.get("power_b_at_limit") and not rec.get("b_at_limit")]
        md += ["", (f"Exponent b at the edge of the allowed range (0.05 to 1.5) at {', '.join(at_lim)}: the power "
                    "curve does not fit these gauges well, and the discharges at the CWC levels in this table are "
                    "not reliable." if at_lim else "No curve used is at the edge of its allowed range.")
               + (f" The power curve's b was on its 0.05 edge at {', '.join(to_log)}, so the log curve is used there."
                  if to_log else ""),
               "Curve used: " + "; ".join(f"{n}: {curve_text(rec['rating'])} (R2 power {f2(rec['rating'].get('r2_power'), 3)}, "
                                         f"log {f2(rec['rating'].get('r2_log'), 3)})"
                                         for n, rec in rv.items() if rec.get("rating")) + ".",
               ("The log curve stage = h0 + a ln(Q) is what the power curve becomes as b goes to 0. It is used only "
                "where the power curve's b sits on its 0.05 edge and the log curve fits at least as well. "
                "Inverting a log curve gives Q = exp((stage - h0) / a), which grows very fast above the fitted range, "
                "so a CWC discharge beyond the data is even less certain with it."
                if any((rec.get("rating") or {}).get("shape") == "log" for rec in rv.values()) else
                "Every gauge uses the power curve."),
               ""]
    n_floor = sum(1 for x in rows if x["h_best"] <= H_GRID[0])
    md += ["A lag of one day is written as the GloFAS day before the stage day. GloFAS discharge is modelled, "
           "so these are GloFAS-to-stage relations, not measured rating curves.", "",
           f"## 3. Best HAND threshold of each radar map (reach gauges: gorge {args.gauge_gorge}, valley {args.gauge_valley})", "",
           f"Thresholds tried: {H_GRID[0]:.2f} m to 1 m in 0.05 m steps, then to {H_GRID[-1]:.0f} m in 0.25 m steps. "
           + (f"**{n_floor} of {len(rows)} maps have their best value at the smallest threshold tried "
              f"({H_GRID[0]:.2f} m)**, so for them the best agreement is with almost no HAND extent at all."
              if n_floor else "No map has its best value at the smallest threshold tried."), "",
           "| Event | Reach | GloFAS version | GloFAS Q m3/s | Stage from GloFAS m | Stage observed m | Best H m | CSI at best H |",
           "|---|---|---|---|---|---|---|---|"]
    for x in rows:
        md.append(f"| {x['event']} | {x['reach'].replace('_', ' ')} | {vname(x['glofas_version']) if x['glofas_version'] else 'n/a'} | "
                  f"{f2(x['glofas_q'], 0)} | {f2(x['stage_from_glofas'])} | "
                  f"{f2(x['stage_observed'])} | {x['h_best']:.2f} | {x['csi_best']:.3f} |")
    md += ["", "## 4. Leaving one event out", "",
           "| Event | Reach | Best H (CSI) | Offset H (CSI) | Linear H (CSI) | Constant H (CSI) |", "|---|---|---|---|---|---|"]
    for x in loeo:
        md.append(f"| {x['event']} | {x['reach'].replace('_', ' ')} | {x['h_best']:.2f} ({x['csi_best']:.3f}) | "
                  + " | ".join(f"{x['h_' + m]:.2f} ({x['csi_' + m]:.3f})" for m in MODELS) + " |")
    md += ["", "Mean CSI over the left-out maps: " + ", ".join(f"{m} {f2(v, 3)}" for m, v in mean_csi.items()) + ".",
           f"The forecast extent uses: **{chosen}**" + (" (auto: the highest mean left-out CSI; a tie goes to constant)"
                                                        if args.model == "auto" else
                                                        " (fixed; --model auto lets the left-out CSI choose)") + ". "
           "If offset and linear do not beat constant, the river stage adds nothing to the extent with this few "
           "events: the forecast extent is then the same whatever the forecast says, and the report must say so.", "",
           "## 4b. Gauge loop: every candidate gauge per reach", "",
           "Each gauge's stage on the radar days, from its own GloFAS-to-stage relation, put through the same "
           "left-out test. The best H of each map does not change with the gauge. With the constant link "
           "the gauge does not change the forecast extent at all; it matters only if offset or linear beat "
           "constant for some gauge." + shared_cell_note(rating_by_version), "",
           "| Reach | Gauge | In use | Curve (R2) | Stage range on radar days m | Correlation stage vs best H | Left-out CSI offset / linear / constant |",
           "|---|---|---|---|---|---|---|"] + [
           (f"| {x['reach'].replace('_', ' ')} | {x['gauge']} | | {x['note']} | | | |" if "note" in x else
            f"| {x['reach'].replace('_', ' ')} | {x['gauge']} | {'yes' if x['in_use'] else ''} | {x['curve']} ({x['r2_stage_fit']:.2f}) | "
            f"{f2(x['stage_min_m'])} to {f2(x['stage_max_m'])} | {f2(x.get('corr_stage_hbest'), 3)} | "
            + " / ".join(f2(x[f'loeo_csi_{m}'], 3) for m in MODELS) + " |") for x in gauge_loop] + [
           "", "Gauges in use: one per reach, chosen from this loop; "
           f"{args.gauge_gorge} and {args.gauge_valley}.", "",
           "## 5. Validated events against the reference (Table D windows, 2000 resamples)", "",
           "| Event | Map | Hit rate | False alarm ratio | CSI |", "|---|---|---|---|---|"]
    for x in ref_scores:
        ci = lambda k: f"{f2(x[k], 3)} ({f2(x[k + '_lo'], 3)} to {f2(x[k + '_hi'], 3)})"  # noqa: E731
        md.append(f"| {x['event']} | {x['map']} | {ci('hit_rate')} | {ci('far')} | {ci('csi')} |")
    md += ["", "## Limits", "",
           f"Only {len(evs)} radar maps, two of them checked against a reference; the calibration is thin and the "
           "left-out test is the honest measure of it. The radar maps are themselves model output. HAND comes from "
           "the 30 m strata, resampled to 10 m. GloFAS is a global model at about 5 km; at a gauge in a gorge its "
           "discharge can be off by a large factor, which is why the stage relation is fitted at each gauge rather "
           "than taken from GloFAS directly.", ""]
    (OUT / "M10_RATING_HAND.md").write_text("\n".join(md), encoding="utf-8")
    print(f"  {len(evs)} events, {len(rows)} event-reach rows; mean left-out CSI " +
          ", ".join(f"{m} {f2(v, 3)}" for m, v in mean_csi.items()))
    print(f"  wrote {OUT / 'calibration.json'} and M10_RATING_HAND.md")
    print("PASS  M10 rating and HAND calibration")
    return 0


if __name__ == "__main__":
    sys.exit(main())
