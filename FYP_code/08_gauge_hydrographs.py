r"""
08_gauge_hydrographs.py
=======================
Plot the gauge record around each event, with the warning, danger and highest
flood levels drawn on, and write the numbers the report quotes.

Written 9 September 2026. The station reference file carries `warning_level_m`,
`danger_level_m`, `highest_flood_level_m` and `hfl_date` and nothing in the
project had used them.

WHAT IT PRODUCES, per event

  GAUGE_hydrograph_<event>.png   one panel per corridor station, stacked, with
                                 warning and danger drawn as horizontal lines
  GAUGE_event_peaks.csv          peak, time of peak, and how far above warning
                                 and danger each station reached, for every
                                 event, plus the share of the station's own
                                 highest recorded flood

WHY THE SHARE MATTERS

  Four of the five referenced stations record their highest ever flood on
  17 June 2013, the Kedarnath disaster. Reporting each event as a share of that
  gives the study events a scale from the official record rather than from a
  paper.

HOW TO RUN

  conda activate fyp
  cd "<project folder>\\04 Code\\FYP_code"

  python 08_gauge_hydrographs.py --selftest

  python 08_gauge_hydrographs.py ^
    --clean "..\\..\\05 Data\\gauges\\guardian\\GAUGE_wse_clean.csv" ^
    --ref   "..\\..\\05 Data\\gauges\\GAUGE_station_reference.csv" ^
    --out   "..\\..\\06 Outputs\\gauges"

  Events default to the two study events plus June 2021. Override with
  --event NAME=YYYY-MM-DD, repeatable. The window either side is --days.
"""
import argparse
import os
import sys

import numpy as np

DEFAULT_EVENTS = [
    "2021_october=2021-10-19",
    "2023_august=2023-08-14",
    "2021_june=2021-06-19",
]
CORRIDOR = ["Rudraprayag (A)", "Srinagar", "Devprayag (A)", "Rishikesh"]


def say(m=""):
    print(m, flush=True)


def rule(t):
    say(); say("-" * 72); say(t); say("-" * 72)


def peak_stats(series, ref):
    """series: a DataFrame with time and wse for one station in one window."""
    if series.empty:
        return None
    i = series["wse"].idxmax()
    row = series.loc[i]
    out = {"peak_m": float(row["wse"]), "peak_time": row["time"]}
    for k, label in (("warning_level_m", "above_warning_m"),
                     ("danger_level_m", "above_danger_m")):
        v = ref.get(k)
        out[label] = None if v is None or not np.isfinite(v) else float(row["wse"]) - v
    zero = ref.get("zero_gauge_rl_m")
    hfl = ref.get("highest_flood_level_m")
    out["stage_m"] = None if zero is None else float(row["wse"]) - zero
    if zero is not None and hfl is not None and np.isfinite(zero) and np.isfinite(hfl) \
            and (hfl - zero) > 0:
        out["share_of_record_pct"] = 100.0 * (float(row["wse"]) - zero) / (hfl - zero)
    else:
        out["share_of_record_pct"] = None
    return out


def selftest():
    rule("08_gauge_hydrographs self-test")
    ok = True

    def check(label, cond, note=""):
        nonlocal ok
        say("  %-5s %-44s %s" % ("PASS" if cond else "FAIL", label, note))
        ok = ok and cond

    check("python >= 3.9", sys.version_info >= (3, 9),
          "%d.%d.%d" % sys.version_info[:3])
    for mod in ("numpy", "pandas", "matplotlib"):
        try:
            __import__(mod)
            check("import " + mod, True)
        except Exception as e:
            check("import " + mod, False, str(e)[:40])

    import pandas as pd
    t = pd.date_range("2021-10-15", periods=48, freq="6h")
    w = np.full(len(t), 620.0)
    w[20] = 628.60                      # the known peak
    df = pd.DataFrame({"time": t, "wse": w})
    ref = {"zero_gauge_rl_m": 614.0, "warning_level_m": 626.0,
           "danger_level_m": 627.0, "highest_flood_level_m": 634.85}
    s = peak_stats(df, ref)
    check("peak value found", abs(s["peak_m"] - 628.60) < 1e-9, "%.2f" % s["peak_m"])
    check("peak time found", s["peak_time"] == t[20], str(s["peak_time"]))
    check("stage above zero", abs(s["stage_m"] - 14.60) < 1e-9, "%.2f m" % s["stage_m"])
    check("metres above danger", abs(s["above_danger_m"] - 1.60) < 1e-9,
          "%+.2f m" % s["above_danger_m"])
    check("metres above warning", abs(s["above_warning_m"] - 2.60) < 1e-9,
          "%+.2f m" % s["above_warning_m"])
    # 14.60 of a 20.85 m record is 70.02 per cent
    check("share of the station's own record",
          abs(s["share_of_record_pct"] - 70.02) < 0.02,
          "%.2f %%" % s["share_of_record_pct"])

    empty = peak_stats(df.iloc[0:0], ref)
    check("an empty window returns None, not an exception", empty is None)

    partial = peak_stats(df, {"zero_gauge_rl_m": 614.0})
    check("a station with no danger level does not crash",
          partial["above_danger_m"] is None)

    say(); say("  %s" % ("all checks passed" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", help="GAUGE_wse_clean.csv")
    ap.add_argument("--ref", help="GAUGE_station_reference.csv")
    ap.add_argument("--out", default="../../06 Outputs/gauges")
    ap.add_argument("--event", action="append", default=None,
                    metavar="NAME=YYYY-MM-DD",
                    help="peak date of an event. Repeatable. Defaults to the "
                         "three the project has looked at.")
    ap.add_argument("--days", type=int, default=7,
                    help="days either side of the peak to plot, default 7")
    ap.add_argument("--stations", default=",".join(CORRIDOR),
                    help="comma separated, in downstream order")
    ap.add_argument("--keep-spikes", dest="keep_spikes", action="store_true",
                    help="let readings marked spike_flag count as the peak. Off "
                         "by default: the 631.22 m reading at Rudraprayag on "
                         "24 June 2021 is flagged and is not a flood, and it "
                         "sits inside the June 2021 window.")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()
    for need in ("clean", "ref"):
        if not getattr(a, need):
            ap.error("--%s is required unless --selftest is given" % need)
    if selftest() != 0:
        say(); say("self-test failed. Nothing was written.")
        return 1

    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(a.out, exist_ok=True)
    events = a.event or DEFAULT_EVENTS
    stations = [s.strip() for s in a.stations.split(",") if s.strip()]

    rule("1. reading")
    df = pd.read_csv(a.clean, parse_dates=["time"])
    ref = pd.read_csv(a.ref).set_index("station").to_dict("index")
    say("  %d readings, %d stations" % (len(df), df["station"].nunique()))
    if "spike_flag" in df.columns and not a.keep_spikes:
        flagged = df["spike_flag"].astype(str).str.lower().isin(["true", "1"])
        say("  %d readings carry spike_flag and are excluded from the peak "
            "search. They are still drawn." % int(flagged.sum()))
        df["_spike"] = flagged
    else:
        df["_spike"] = False
    missing = [s for s in stations if s not in set(df["station"])]
    if missing:
        say("  WARNING: not in the clean file: %s" % ", ".join(missing))
        stations = [s for s in stations if s not in missing]
    for s in stations:
        if s not in ref:
            say("  note: %s has no reference row, so its panel carries no "
                "warning or danger line" % s)

    rows = []
    for spec in events:
        name, day = spec.split("=", 1)
        peak = pd.Timestamp(day)
        lo, hi = peak - pd.Timedelta(days=a.days), peak + pd.Timedelta(days=a.days)
        rule("2. %s, peak %s" % (name, peak.date()))

        fig, axes = plt.subplots(len(stations), 1, figsize=(10, 2.2 * len(stations)),
                                 sharex=True)
        if len(stations) == 1:
            axes = [axes]
        for ax, st in zip(axes, stations):
            sub = df[(df["station"] == st) & (df["time"] >= lo) & (df["time"] <= hi)]
            sub = sub.sort_values("time")
            r = ref.get(st, {})
            ax.plot(sub["time"], sub["wse"], lw=1.2, color="#1f3864")
            sp = sub[sub["_spike"]]
            if len(sp):
                ax.plot(sp["time"], sp["wse"], "x", ms=5, color="#b03030",
                        label="flagged")
            for key, colour, label in (("warning_level_m", "#c08a00", "warning"),
                                       ("danger_level_m", "#b03030", "danger")):
                v = r.get(key)
                if v is not None and np.isfinite(v):
                    ax.axhline(v, color=colour, lw=0.9, ls="--")
                    ax.annotate(label, (0.995, v), xycoords=("axes fraction", "data"),
                                ha="right", va="bottom", fontsize=7, color=colour)
            ax.set_ylabel(st, fontsize=8)
            ax.grid(alpha=0.25)

            s = peak_stats(sub[~sub["_spike"]], r)
            if s is None:
                say("  %-20s no data in the window" % st)
                continue
            rows.append({"event": name, "station": st,
                         "peak_m": round(s["peak_m"], 2),
                         "peak_time": s["peak_time"],
                         "stage_m": None if s["stage_m"] is None else round(s["stage_m"], 2),
                         "above_warning_m": None if s["above_warning_m"] is None
                                            else round(s["above_warning_m"], 2),
                         "above_danger_m": None if s["above_danger_m"] is None
                                           else round(s["above_danger_m"], 2),
                         "share_of_record_pct": None if s["share_of_record_pct"] is None
                                                else round(s["share_of_record_pct"], 1)})
            say("  %-20s peak %8.2f m at %s   danger %+6s m   %5s %% of its record"
                % (st, s["peak_m"], s["peak_time"],
                   "n/a" if s["above_danger_m"] is None else "%.2f" % s["above_danger_m"],
                   "n/a" if s["share_of_record_pct"] is None
                   else "%.1f" % s["share_of_record_pct"]))

        axes[0].set_title("Water surface elevation, %s event, peak %s"
                          % (name.replace("_", " "), peak.date()))
        axes[-1].set_xlabel("date")
        fig.tight_layout()
        png = os.path.join(a.out, "GAUGE_hydrograph_%s.png" % name)
        fig.savefig(png, dpi=150)
        plt.close(fig)
        say("  wrote %s" % png)

    rule("3. writing the table")
    out_csv = os.path.join(a.out, "GAUGE_event_peaks.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    say("  wrote %s" % out_csv)
    say()
    say("  `share_of_record_pct` is the stage as a share of that station's own")
    say("  highest recorded flood. Four of the five referenced stations set")
    say("  theirs on 17 June 2013, the Kedarnath disaster, so this puts the")
    say("  study events on a scale taken from the official record.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
