"""Jal-SAR extensions, item M9 (forecast option A): the GloFAS forecast watch, its hindcast on the six radar
dates, and the forecast picture for the Jal-SAR web map.

1. Thresholds at each gauge's GloFAS cell (the same cell M10 uses):
     rp    the 2, 5 and 20 year flood flows, from a Gumbel fit to the annual
           maxima of the GloFAS reanalysis (how GloFAS sets its own thresholds)
     cwc   the CWC warning and danger stages, turned into discharges by the M10
           stage relation (needs m10_rating_hand.py first)
   --rule picks which pair sets the status (default rp: 2 and 5 year).
   One set per reanalysis version. Each forecast is compared with the
   set of the version that made it, by its issue day (m9_glofas_read.forecast_version):
   4.0 from 26 July 2023 (5.0 once its forecasts are operational). GloFAS 3.1 forecasts (before that)
   have no reanalysis here and use --version (default 4.0), flagged in the report.
2. Status of a day at a gauge, from the 51 forecasts (control and 50 members):
     warning  at least --p (default 50 %) of them above the upper threshold
     watch    at least --p of them above the lower threshold
     none     otherwise
3. Hindcast: for each radar date in the forecast files, the status the forecast
   gave for that day when issued 0 to 14 days before it, next to the GloFAS
   reanalysis for the day and, where GUARDIAN has it (to July 2024), whether the
   gauge passed its CWC warning stage that day.
4. Watch: the newest forecast (m9_glofas_latest.py), days 1 to --days (default 7).
   With no newest forecast, --issued picks any issue day in the hindcast files
   (to show how the watch looked before a past flood), clearly labelled.
5. Picture for the web map (forecast_extent.tif, on the Jal-SAR grid):
     2  flooded in any past Jal-SAR radar map (the past footprints)
     1  M10 forecast extent for the forecast's peak day, drawn only in a reach whose
        gauge is on watch or warning (with nothing on watch, nothing is drawn)

Outputs in 06 Outputs\\jalsar\\extensions\\m9_forecast\\:
  forecast_watch.json   read by Jal-SAR's web map (Forecast layer and panel)
  forecast_extent.tif, thresholds.csv, hindcast.csv, hindcast_extent.csv (M10, issued 1, 3, 7
  and 14 days ahead, scored on the checked events), M9_FORECAST.md (read first),
  fig_watch.png, fig_hindcast.png

Run:  python m9_forecast_watch.py
      python m9_forecast_watch.py --rule cwc --p 0.3
      python m9_forecast_watch.py --issued 2023-08-14          (a past issue, for the demo of how it works)
"""
import argparse
import csv
import json
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

OUT = gr.ROOT / "06 Outputs" / "jalsar" / "extensions" / "m9_forecast"
CAL = gr.ROOT / "06 Outputs" / "jalsar" / "extensions" / "m10_forecast_extent" / "calibration.json"
LEADS_SCORED = (1, 3, 7, 14)
# The forecast extent is drawn in the gorge only. In the open valley the best HAND
# threshold of every radar map was the smallest one tried (VALLEY_HAND_CHECK.md): the radar flood there
# is a narrow strip along the channel, which a fixed HAND threshold cannot scale up for a bigger flood.
# The valley gets a separate, static layer instead ("not a flood forecast"), off by default.
FORECAST_REACHES = ("gorge_above_devprayag",)
VALLEY_REACH = "open_valley_below_devprayag"
VALLEY_STRIP_H = 0.05
STATUS_RANK = {"none": 0, "watch": 1, "warning": 2, "unknown": -1}
SYMBOL = {"none": "-", "watch": "w", "warning": "W", "unknown": "?"}


def status_of(members, lo, hi, p):
    m = members[np.isfinite(members)]
    if len(m) == 0 or not np.isfinite(lo):
        return "unknown", float("nan"), float("nan")
    p_lo, p_hi = float((m > lo).mean()), float((m > hi).mean()) if np.isfinite(hi) else 0.0
    return ("warning" if p_hi >= p else "watch" if p_lo >= p else "none"), p_lo, p_hi


def newest_full(blocks, min_members=20):
    """The newest issue day with at least min_members forecasts (a half-downloaded day is passed over)."""
    best = None
    for b in blocks:
        with np.errstate(all="ignore"):
            n = np.isfinite(b["q"]).any(axis=(2, 3, 4)).sum(axis=1)
        for d, k in zip(b["issue"], n):
            if k >= min_members and (best is None or d > best):
                best = d
    return best if best is not None else max(b["issue"].max() for b in blocks)


def fc_day(issue, lead_h):
    """Calendar day a lead covers (day convention: a value is stamped with the end of its day)."""
    return issue + np.timedelta64(int(lead_h) // 24 - 1, "D")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rule", choices=("rp", "cwc"), default="rp")
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--issued", help="use this issue day from the hindcast files instead of the newest forecast")
    ap.add_argument("--version", default="version_4_0")
    args = ap.parse_args()
    global OUT
    if args.rule == "cwc":            # both rules are run; the CWC run goes beside the main one
        OUT = OUT / "cwc_rule"
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    import jalsar_core as gc
    gp = gc.GPaths()
    cal = json.loads(CAL.read_text(encoding="utf-8")) if CAL.exists() else None

    hfiles, hver = gr.historical_files(args.version)
    if not hfiles:
        print("PROBLEM: no GloFAS reanalysis files (run m9_glofas_download.py first)")
        return 1
    hists = gr.open_versions(args.version)
    if args.version not in hists:            # no file carries the asked version: the old behaviour
        hists = {hver: gr.open_historical(hfiles)}
        args.version = hver
    hist = hists[args.version]
    sts = gr.stations()
    if cal:
        cells = {n: r["cell"] for n, r in cal["rating"].items()}
        missing = [s for s in sts if s["name"] not in cells]
        cells.update(gr.match_cells(hist, missing, uparea=gr.load_uparea()) if missing else {})
    else:
        cells = gr.match_cells(hist, sts, uparea=gr.load_uparea())
    day_idx_v = {v: {d: k for k, d in enumerate(h["day"])} for v, h in hists.items()}
    vname = lambda v: gr.VERSION_NAME.get(v, v)  # noqa: E731

    # 1. thresholds, one set per reanalysis version
    def make_thr(v, h):
        cv = cells if gr.same_grid(h, hist) else gr.match_cells(h, sts, uparea=gr.load_uparea())
        out = {}
        for s in sts:
            c = cv[s["name"]]
            q = h["q"][:, c["i"], c["j"]]
            rp, am, fit = gr.gumbel_thresholds(q, h["day"])
            rec = {"rp": rp or {}, "fit": fit, "cell": c, "version": v}
            rbv = ((cal or {}).get("rating_by_version") or {}).get(v)
            if rbv is None:          # an older calibration.json holds only the calibration version
                rbv = (cal or {}).get("rating", {}) if v == (cal or {}).get("glofas_reanalysis", args.version) else {}
            r = rbv.get(s["name"], {})
            # each CWC discharge carries its own "beyond the data" flag (before 29 Oct only one flag,
            # printed after the danger value, so a beyond-the-data warning flow looked fine)
            rec["cwc"] = {"warning": r.get("q_warning"), "danger": r.get("q_danger"),
                          "warning_beyond": bool(r.get("q_warning_extrapolated")),
                          "danger_beyond": bool(r.get("q_danger_extrapolated")),
                          "beyond_data": bool(r.get("q_warning_extrapolated") or r.get("q_danger_extrapolated")),
                          "fit_at_limit": bool(r.get("b_at_limit")),
                          "curve": (r.get("rating") or {}).get("shape", "power")}
            if args.rule == "rp":
                rec["lo"], rec["hi"] = (rp or {}).get(2, np.nan), (rp or {}).get(5, np.nan)
            else:
                rec["lo"] = rec["cwc"]["warning"] if rec["cwc"]["warning"] is not None else np.nan
                rec["hi"] = rec["cwc"]["danger"] if rec["cwc"]["danger"] is not None else np.nan
            out[s["name"]] = rec
        return out

    thr_v = {v: make_thr(v, h) for v, h in hists.items()}
    thr = thr_v[args.version]

    def thr_of(issue):
        """(threshold version, forecast version, thresholds) for a forecast issued on this day."""
        tv, fv = gr.threshold_version(issue, thr_v, args.version)
        return tv, fv, thr_v[tv]

    if args.rule == "cwc" and all(not np.isfinite(t["lo"]) for t in thr.values()):
        print("PROBLEM: --rule cwc needs m10_rating_hand.py to have run (no CWC discharges found)")
        return 1
    lo_name, hi_name = (("2-year flow", "5-year flow") if args.rule == "rp" else ("CWC warning level", "CWC danger level"))
    with open(OUT / "thresholds.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["version", "station", "cell_lat", "cell_lon", "cell_rule", "rp2", "rp5", "rp20", "years",
                    "q_cwc_warning", "q_cwc_danger", "q_cwc_warning_beyond_data", "q_cwc_danger_beyond_data",
                    "stage_curve", "stage_fit_at_limit", "lower_used", "upper_used"])
        for v, tt in thr_v.items():
            for n, t in tt.items():
                w.writerow([v, n, t["cell"]["cell_lat"], t["cell"]["cell_lon"], t["cell"]["rule"]]
                           + [round(t["rp"].get(k, np.nan), 1) for k in (2, 5, 20)] + [(t["fit"] or {}).get("years")]
                           + [t["cwc"]["warning"], t["cwc"]["danger"], t["cwc"]["warning_beyond"],
                              t["cwc"]["danger_beyond"], t["cwc"]["curve"], t["cwc"]["fit_at_limit"],
                              round(t["lo"], 1), round(t["hi"], 1)])

    # 3. hindcast
    ffiles = sorted((gr.GLOFAS / "forecast").glob("*.nc"))
    fc = gr.open_forecast(ffiles) if ffiles else None
    evs = gc.load_events(gp, only_existing=False)
    hind = []
    cell_cache = {}

    def members(blocks, day, lead_h, s):
        """The 51 forecasts for a gauge, issued on day, at lead_h; None when not downloaded."""
        b, ii = gr.find_issue(blocks, day)
        if b is None:
            return None
        li = np.where(b["lead_h"] == int(lead_h))[0]
        if not len(li):
            return None
        key = (id(b), s["name"])
        if key not in cell_cache:
            cell_cache[key] = gr.block_cell(b, s, hist, thr[s["name"]]["cell"])
        i, j, _ = cell_cache[key]
        return b["q"][ii, :, li[0], i, j]

    if fc is not None:
        stage_cache = {}
        for e in evs:
            d = np.datetime64(e.date, "D")
            tv_run, _, _ = thr_of(d)                   # the version running on the radar day
            for s in sts:
                # the reanalysis of the running version; if it does not reach the day (5.0 ends
                # 31 July 2026), the first other version that does, with that version's own thresholds
                qa, tv_d, note = float("nan"), tv_run, ""
                for v in [tv_run] + [x for x in hists if x != tv_run]:
                    c = thr_v[v][s["name"]]["cell"]
                    k = day_idx_v[v].get(d)
                    q_try = float(hists[v]["q"][k, c["i"], c["j"]]) if k is not None else float("nan")
                    if np.isfinite(q_try):
                        qa, tv_d = q_try, v
                        if v != tv_run:
                            note = (f"{vname(tv_run)} reanalysis ends {hists[tv_run]['day'][-1]}; "
                                    f"{vname(v)} shown with its own thresholds")
                        break
                lo, hi = thr_v[tv_d][s["name"]]["lo"], thr_v[tv_d][s["name"]]["hi"]
                obs_status = ("warning" if qa > hi else "watch" if qa > lo else "none") if np.isfinite(qa) else "unknown"
                if s["name"] not in stage_cache:
                    stage_cache[s["name"]] = gr.daily_stage(s["name"])
                sg = stage_cache[s["name"]]
                stage = float(sg.get(pd.Timestamp(e.date), np.nan)) if sg is not None else float("nan")
                row = {"event": e.id, "date": e.date, "station": s["name"], "reanalysis_version": tv_d,
                       "reanalysis_note": note, "reanalysis_q": round(qa, 1),
                       "reanalysis_status": obs_status, "gauge_stage_m": round(stage, 2) if np.isfinite(stage) else "",
                       "gauge_above_warning": ("" if not np.isfinite(stage) else bool(stage >= s["warning_stage_m"]))}
                pairs = set()
                for n in range(15):
                    tv_n, fv_n, thr_n = thr_of(d - np.timedelta64(n, "D"))
                    m = members(fc, d - np.timedelta64(n, "D"), 24 * (n + 1), s)
                    if m is None or not np.isfinite(m).any():
                        row[f"ahead_{n}"] = "unknown"
                        continue
                    pairs.add((fv_n, tv_n))
                    st_, _, _ = status_of(m, thr_n[s["name"]]["lo"], thr_n[s["name"]]["hi"], args.p)
                    row[f"ahead_{n}"] = st_
                    row[f"q50_ahead_{n}"] = round(float(np.nanmedian(m)), 1)
                row["forecast_versions"] = "; ".join(
                    f"{vname(f)} forecast, {vname(t)} thresholds" for f, t in sorted(pairs))
                hind.append(row)
        if hind:
            keys = list(dict.fromkeys(k for r in hind for k in r))
            with open(OUT / "hindcast.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader(); w.writerows(hind)

    # 3b. hindcast of the forecast extent on the checked events (M10), issued 1, 3, 7 and 14 days ahead
    ext_hind = []
    if cal and fc is not None:
        import m10_rating_hand as m10
        for e in evs:
            if not e.validated_year or not (gp.event_dir(e.id) / "flood_headline.tif").exists():
                continue
            d = np.datetime64(e.date, "D")
            for n in LEADS_SCORED:
                stage_by, qs = {}, {}
                tv_n, _, _ = thr_of(d - np.timedelta64(n, "D"))
                for r, g in cal["reach_gauges"].items():
                    s = next((x for x in sts if x["name"] == g), None)
                    rt = m10.rating_for(cal, g, tv_n)          # the stage relation of the forecast's version
                    m = members(fc, d - np.timedelta64(n, "D"), 24 * (n + 1), s) if s else None
                    q = float(np.nanmedian(m)) if m is not None and np.isfinite(m).any() else float("nan")
                    qs[r] = q
                    stage_by[r] = float(gr.stage_from_q(rt, q)) if rt and np.isfinite(q) else None
                if any(v is None for v in stage_by.values()):
                    ext_hind.append({"event": e.id, "days_ahead": n, "note": "no forecast"})
                    continue
                h_by = m10.h_for_stage(cal, stage_by, leave_out=e.id)
                sc = m10.score_extent(gc, gp, e, h_by) or {}
                ext_hind.append({"event": e.id, "days_ahead": n, **{f"q50_{r}": round(q, 1) for r, q in qs.items()},
                                 **{f"hand_{r}": (None if h is None else round(h, 2)) for r, h in h_by.items()}, **sc})
        if ext_hind:
            keys = list(dict.fromkeys(k for r in ext_hind for k in r))
            with open(OUT / "hindcast_extent.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader(); w.writerows(ext_hind)

    # 4. watch
    latest = sorted((gr.GLOFAS / "latest").glob("*.nc"))
    label = ""
    if args.issued:
        issue = np.datetime64(args.issued, "D")
        wf = fc
        if gr.find_issue(fc, issue)[0] is None:
            print(f"PROBLEM: issue day {args.issued} is not in the forecast files")
            return 1
        label = "past forecast, shown to explain the method"
    elif latest:
        wf = gr.open_forecast(latest)
        issue = newest_full(wf)
        label = "newest forecast"
    elif fc is not None:
        wf = fc
        issue = newest_full(wf)
        label = "no newest forecast downloaded; the last past forecast is shown"
    else:
        print("PROBLEM: no forecast files at all")
        return 1
    wb, _ = gr.find_issue(wf, issue)
    tv_w, fv_w, thr_w = thr_of(issue)
    points, peak = [], {}
    for s in sts:
        t = thr_w[s["name"]]
        days = []
        for L in wb["lead_h"]:
            if int(L) // 24 > args.days:
                continue
            m = members(wf, issue, L, s)
            if m is None or not np.isfinite(m).any():
                continue
            st_, plo, phi = status_of(m, t["lo"], t["hi"], args.p)
            days.append({"date": str(fc_day(issue, L)), "lead_days": int(L) // 24, "q50": round(float(np.nanmedian(m)), 1),
                         "q10": round(float(np.nanpercentile(m, 10)), 1), "q90": round(float(np.nanpercentile(m, 90)), 1),
                         "p_lower": round(plo, 3), "p_upper": round(phi, 3), "status": st_})
        worst = max(days, key=lambda d: (STATUS_RANK[d["status"]], d["p_lower"])) if days else None
        status = worst["status"] if worst else "unknown"
        if status == "none":
            txt = (f"Below the {lo_name} on every day to {days[-1]['date']} (highest chance "
                   f"{max(d['p_lower'] for d in days) * 100:.0f} % on any day).")
        elif status == "unknown":
            txt = "No forecast value for this gauge."
        else:
            first = next(d for d in days if d["status"] == status)
            txt = (f"{status.capitalize()} from {first['date']}: {first['p_lower' if status == 'watch' else 'p_upper'] * 100:.0f} % "
                   f"of the 51 forecasts above the {lo_name if status == 'watch' else hi_name} "
                   f"({t['lo' if status == 'watch' else 'hi']:.0f} m3/s); middle forecast {first['q50']:.0f} m3/s.")
        points.append({"name": s["name"], "lat": s["lat"], "lon": s["lon"], "status": status, "status_text": txt,
                       "days": days, "lower_m3s": round(t["lo"], 1), "upper_m3s": round(t["hi"], 1)})
        peak[s["name"]] = (max(d["q50"] for d in days) if days else float("nan"), status)

    # 5. picture
    ext_path, strip_path, strip_km2 = None, None, None
    head = [e for e in gc.load_events(gp) if (gp.event_dir(e.id) / "flood_headline.tif").exists()]
    drawn = {}
    if head:
        import rasterio
        import jalsar_display as gd
        import m10_rating_hand as m10
        ref = head[0]
        _, prof = gd.read_u8(gp.event_dir(ref.id) / "flood_headline.tif")
        pic = np.zeros((prof["height"], prof["width"]), "uint8")
        for e in head:
            f, p2 = gd.read_u8(gp.event_dir(e.id) / "flood_headline.tif")
            if f.shape == pic.shape:
                pic[f == 1] = 2
        if cal:
            hand = gc.mc.reproject_onto(ref.strata_path() / "strata_hand_m.tif", prof)
            perm = gc.mc.reproject_onto(gp.perm_water(), prof, categorical=True) == 100
            reach, _ = gc.reach_raster(ref, prof)
            stage_by = {}
            for r, g in cal["reach_gauges"].items():
                q, stt = peak.get(g, (np.nan, "unknown"))
                rt = m10.rating_for(cal, g, tv_w)              # the stage relation of the forecast's version
                on = STATUS_RANK.get(stt, -1) >= 1 and r in FORECAST_REACHES    # gorge only
                stage_by[r] = float(gr.stage_from_q(rt, q)) if (rt and on and np.isfinite(q)) else None
                drawn[r] = {"gauge": g, "status": stt, "peak_q50": q, "stage_m": stage_by[r]}
                if r not in FORECAST_REACHES:
                    drawn[r]["note"] = "never drawn: see the valley channel strip layer"
            h_by = m10.h_for_stage(cal, stage_by)
            for r in drawn:
                drawn[r]["hand_m"] = h_by.get(r)
            ext = m10.extent(hand, reach, perm, h_by)
            pic[ext == 1] = 1
            # the valley channel strip: static, the same whatever the forecast says
            strip = ((reach == VALLEY_REACH) & np.isfinite(hand) & (hand < VALLEY_STRIP_H) & ~perm).astype("uint8")
            ps = dict(prof)
            ps.update(count=1, dtype="uint8", nodata=None, compress="lzw")
            strip_path = OUT / "valley_channel_strip.tif"
            with rasterio.open(strip_path, "w", **ps) as dst:
                dst.write(strip, 1)
            strip_km2 = float(strip.sum()) * abs(prof["transform"][0] * prof["transform"][4]) / 1e6 \
                if prof.get("crs") and not getattr(prof["crs"], "is_geographic", False) else None
        p = dict(prof)
        p.update(count=1, dtype="uint8", nodata=None, compress="lzw")
        ext_path = OUT / "forecast_extent.tif"
        with rasterio.open(ext_path, "w", **p) as dst:
            dst.write(pic, 1)

    method = (f"Status from the 51 GloFAS forecasts at each gauge's river cell: watch when at least "
              f"{args.p * 100:.0f} % are above the {lo_name}, warning when at least {args.p * 100:.0f} % are above the {hi_name}.")
    ver_txt = (f"This forecast was made by {vname(fv_w)}; its thresholds are from the {vname(tv_w)} reanalysis"
               + (", the same version" if fv_w == tv_w else
                  f" ({vname(fv_w)} has no reanalysis here, so the thresholds do not match its version)"))
    limits = ("GloFAS is a global model at about 5 km and one value per day, so it says which day, not which hour, "
              "and it can be far off in a gorge. Thresholds are from the GloFAS reanalysis"
              + (" and the CWC levels through a fitted stage relation" if args.rule == "cwc" else "")
              + f". {ver_txt}. Dark blue: where a forecast this high would put water in the gorge above Devprayag "
              "(HAND, calibrated on the radar maps; the open valley is never drawn); pale blue: "
              "where past radar maps found flood. This is a study tool, not an official warning: follow CWC and the "
              "district administration.")
    fcj = {"issued": str(issue), "issued_label": label, "rule": args.rule, "p": args.p, "days": args.days,
           "points": points, "extent_tif": str(ext_path) if ext_path else None, "extent_reaches": drawn,
           "forecast_reaches": list(FORECAST_REACHES),
           "valley_strip_tif": str(strip_path) if strip_path else None, "valley_strip_h_m": VALLEY_STRIP_H,
           "valley_strip_km2": None if strip_km2 is None else round(strip_km2, 3),
           "valley_strip_text": ("Not a flood forecast. Cells of the open valley below Devprayag less than "
                                 f"{VALLEY_STRIP_H:.2f} m above the nearest stream: the narrow strip along the "
                                 "channel where every radar map found its valley flood. In the valley the "
                                 "best HAND threshold was the smallest one tried on all six maps, so a forecast "
                                 "cannot be turned into a valley extent this way; this layer does not change "
                                 "with the forecast."),
           "method_short": method, "limits": limits, "glofas_reanalysis": tv_w,
           "forecast_version": fv_w, "threshold_version": tv_w,
           "made": time.strftime("%Y-%m-%d %H:%M")}
    gc.mc.save_json(OUT / "forecast_watch.json", fcj)

    # figures
    fig, axs = plt.subplots(1, len(points), figsize=(3.2 * len(points), 3), sharey=False, squeeze=False)
    for ax, pt in zip(axs[0], points):
        x = [d["lead_days"] for d in pt["days"]]
        ax.fill_between(x, [d["q10"] for d in pt["days"]], [d["q90"] for d in pt["days"]], color="#93c5fd", alpha=0.6)
        ax.plot(x, [d["q50"] for d in pt["days"]], color="#1d4ed8")
        ax.axhline(pt["lower_m3s"], ls="--", color="#f59e0b", lw=0.9)
        ax.axhline(pt["upper_m3s"], ls="--", color="#dc2626", lw=0.9)
        ax.set_title(f"{pt['name']}: {pt['status']}", fontsize=8); ax.set_xlabel("days ahead")
    axs[0][0].set_ylabel("discharge, m3/s")
    fig.suptitle(f"GloFAS forecast issued {issue} ({label})", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig_watch.png", dpi=180); plt.close(fig)
    if hind:
        lab = [f"{r['date']} {r['station']}" for r in hind]
        grid = np.array([[STATUS_RANK[r.get(f"ahead_{n}", "unknown")] for n in range(15)] for r in hind], float)
        grid[grid < 0] = np.nan
        fig, ax = plt.subplots(figsize=(7, 0.6 + 0.22 * len(hind)))
        from matplotlib.colors import ListedColormap
        ax.imshow(grid, cmap=ListedColormap(["#16a34a", "#f59e0b", "#dc2626"]), vmin=0, vmax=2, aspect="auto")
        ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab, fontsize=6)
        ax.set_xticks(range(15)); ax.set_xlabel("days before the radar pass the forecast was issued")
        fig.tight_layout(); fig.savefig(OUT / "fig_hindcast.png", dpi=180); plt.close(fig)

    read_txt = ", ".join("%s (%s to %s)" % (v, h["day"][0], h["day"][-1]) for v, h in hists.items())
    md = ["# M9: GloFAS forecast watch", "",
          f"Rule: **{args.rule}** ({lo_name} and {hi_name}), at least {args.p * 100:.0f} % of the 51 forecasts. "
          f"Reanalysis versions read: {read_txt}. "
          "Each forecast is compared with the thresholds of the version that made it: "
          + (f"GloFAS 5.0 from {gr.V5_FROM}, " if gr.V5_FROM is not None else "")
          + f"GloFAS 4.0 from {gr.V4_FROM}; GloFAS 3.1 forecasts before that have no reanalysis here and "
          f"use the {args.version} thresholds. Day convention: a 24 h value is stamped with the end of its day.", ""]
    f0 = lambda v: "n/a" if v is None or not np.isfinite(v) else f"{v:.0f}"  # noqa: E731
    for v, tt in thr_v.items():
        md += [f"## Thresholds at each gauge's GloFAS cell, {vname(v)} reanalysis (m3/s)", "",
               "| Gauge | Cell | 2-year | 5-year | 20-year | Years in the fit | CWC warning | CWC danger |",
               "|---|---|---|---|---|---|---|---|"]
        for n, t in tt.items():
            md.append(f"| {n} | {t['cell']['cell_lat']:.3f} N, {t['cell']['cell_lon']:.3f} E | {f0(t['rp'].get(2))} | "
                      f"{f0(t['rp'].get(5))} | {f0(t['rp'].get(20))} | {(t['fit'] or {}).get('years', 'n/a')} | "
                      f"{f0(t['cwc']['warning'])}{' (beyond the data)' if t['cwc']['warning_beyond'] else ''} | "
                      f"{f0(t['cwc']['danger'])}{' (beyond the data)' if t['cwc']['danger_beyond'] else ''}"
                      f"{' (stage fit at its limit, not reliable)' if t['cwc'].get('fit_at_limit') else ''} |")
        if any(t["cwc"]["curve"] == "log" for t in tt.values()):
            md += ["", "The CWC discharges of this version come from a log stage relation (stage = h0 + a ln(Q)), "
                   "because the power curve's exponent sat on its lower edge. Above the fitted range "
                   "a log curve gives very large discharges, so read any value marked beyond the data as unknown."]
        md.append("")
    md += ["## Hindcast: what the forecast said about each radar day", "",
           "Days ahead 14 to 0, left to right (W warning, w watch, - none, ? no forecast). The reanalysis column "
           "is from the version running on the radar day; where that version does not reach the day, from the "
           "other version, with its own thresholds, marked with an asterisk.", "",
           "| Radar day | Gauge | 14 ... 0 days ahead | Forecast and threshold versions | Reanalysis that day | Gauge above CWC warning |",
           "|---|---|---|---|---|---|"]
    for r in hind:
        seq = " ".join(SYMBOL[r.get(f"ahead_{n}", "unknown")] for n in range(14, -1, -1))
        md.append(f"| {r['date']} | {r['station']} | `{seq}` | {r.get('forecast_versions') or 'n/a'} | "
                  f"{r['reanalysis_q']} m3/s, {r['reanalysis_status']} ({vname(r['reanalysis_version'])}"
                  f"{'*' if r.get('reanalysis_note') else ''}) | "
                  f"{'no data' if r['gauge_above_warning'] == '' else ('yes' if r['gauge_above_warning'] else 'no')} |")
    notes = sorted({r["reanalysis_note"] for r in hind if r.get("reanalysis_note")})
    if notes:
        md += ["", "\\* " + "; ".join(notes) + "."]
    if fc:
        md += ["", "## Forecast files read", "", "| Grid | Issue days | From | To | Files | Cells used |", "|---|---|---|---|---|---|"]
        for bl in fc:
            used = sorted({v[2] for k, v in cell_cache.items() if k[0] == id(bl)})
            md.append(f"| {bl['step_deg']:.2f} degree | {len(bl['issue'])} | {bl['issue'][0]} | {bl['issue'][-1]} | "
                      f"{len(bl['files'])} | {'; '.join(used)} |")
        md += ["", "GloFAS 3.1 (0.1 degree; forecasts up to 25 July 2023) has no reanalysis here, so its rows of the "
               f"hindcast use the {vname(args.version)} thresholds, a different model version; read them with that in mind. "
               "The forecast files carry no version number: the version is set from the issue day and the "
               "operational dates. GloFAS 5.0 forecasts are not operational yet: the 15 July 2026 date on the "
               "EWDS historical page is the 5.0 reanalysis, and the 2026 forecasts here match the 4.0 reanalysis."]
    md += ["", f"## Watch: forecast issued {issue} ({label})", "", ver_txt + ".", ""]
    for pt in points:
        md.append(f"- **{pt['name']}**: {pt['status']}. {pt['status_text']}")
    if drawn:
        md += ["", "Forecast extent per reach: " + "; ".join(
            f"{r.replace('_', ' ')} ({v['gauge']}, {v['status']}): "
            + (v["note"] if v.get("note") else "not drawn" if v.get("hand_m") is None else f"HAND below {v['hand_m']:.2f} m")
            for r, v in drawn.items()),
               "", "The forecast extent is drawn in the gorge only. The open valley has a separate, "
               f"static layer, \"Valley: channel strip (HAND {VALLEY_STRIP_H:.2f} m)\" (valley_channel_strip.tif), off by "
               "default and labelled \"not a flood forecast\". Why: in the valley the best HAND threshold of all six "
               "radar maps was the smallest one tried, because the radar flood there is a narrow strip along the "
               "channel (VALLEY_HAND_CHECK.md); a fixed threshold cannot scale that up for a bigger flood."]
    if ext_hind:
        md += ["", "## Hindcast of the forecast extent on the checked events (M10)", "",
               "The forecast's middle value at each reach gauge, issued 1, 3, 7 and 14 days before the radar pass, "
               "turned into a HAND extent with that event's own map left out of the calibration, and scored against "
               "the reference like Table D (95 % ranges in brackets).", "",
               "| Event | Days ahead | Hit rate | False alarm ratio | CSI |", "|---|---|---|---|---|"]
        f3 = lambda x, k: "n/a" if x.get(k) is None or not np.isfinite(x[k]) else f"{x[k]:.3f}"  # noqa: E731
        for x in ext_hind:
            if "note" in x:
                md.append(f"| {x['event']} | {x['days_ahead']} | {x['note']} | | |")
                continue
            md.append(f"| {x['event']} | {x['days_ahead']} | " + " | ".join(
                f"{f3(x, k)} ({f3(x, k + '_lo')} to {f3(x, k + '_hi')})" for k in ("hit_rate", "far", "csi")) + " |")
    md += ["", "## Limits", "", limits, "",
           f"With {len({r['event'] for r in hind})} radar days the hindcast shows how the watch behaves; it is far "
           "too small to give a hit rate for the forecast.", ""]
    (OUT / "M9_FORECAST.md").write_text("\n".join(md), encoding="utf-8")
    print(f"  forecast issued {issue} ({label}); " + ", ".join(f"{p['name']} {p['status']}" for p in points))
    print(f"  wrote {OUT / 'forecast_watch.json'} and M9_FORECAST.md ({(time.time() - t0) / 60:.1f} min)")
    print("PASS  M9 forecast watch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
