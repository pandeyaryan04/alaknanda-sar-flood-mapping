"""
screen_gauges.py  --  quality screen for the GUARDIAN wse station files.

Written 9 September 2026, after nine readings in "Rudraprayag (A)" turned out to be
Srinagar's readings copied in at the same timestamps, and after a 631.22 m spike at
Rudraprayag on 24 June 2021 turned out to be physically impossible because no
downstream station moved.

Three screens, in this order:

  1. BAND      a reading outside [zero_gauge_rl + 1, zero_gauge_rl + 25] cannot be
               a water surface at that station.
  2. LEAK      an out-of-band reading whose value matches another station's reading
               at the same timestamp is that other station's reading, not noise.
  3. SPIKE     a rise of more than --spike metres above the station's own 24 h
               running median, with no OTHER station on the reach rising by at
               least --propagate metres within +/- 24 h, cannot be a flood wave.
               The wave crosses Joshimath to Haridwar in under a day, so a real
               rise shows at the neighbours. Corroboration is taken from any other
               station rather than only from downstream ones, so the screen still
               works at the two ends of the reach.

HOW TO RUN

  conda activate fyp
  cd "<project folder>\\04 Code\\FYP_code"
  python screen_gauges.py --dir "..\\..\\05 Data\\gauges\\guardian" --ref "..\\..\\05 Data\\gauges\\GAUGE_station_reference.csv"

  python screen_gauges.py --selftest

Reading .xlsx needs openpyxl. If it is missing, pandas raises on every file and the
run reports that nothing was read.

Outputs, written next to the input files:
  GAUGE_screen_report.csv   every flagged reading, with the screen that caught it
  GAUGE_wse_clean.csv       all stations, all readings that passed, long format
"""
import argparse, os, sys, glob
import pandas as pd
import numpy as np

BAND_LOW, BAND_HIGH = 1.0, 25.0


def station_name_from_file(path):
    base = os.path.basename(path)
    for suffix in ("2020_2024_July.xlsx", ".xlsx"):
        if base.endswith(suffix):
            return base[: -len(suffix)].strip()
    return base


def load_one(path):
    df = pd.read_excel(path, sheet_name=0)
    cols = {c.lower(): c for c in df.columns if isinstance(c, str)}
    tcol = cols.get("time")
    vcol = cols.get("wse")
    ncol = cols.get("name")
    if tcol is None or vcol is None:
        raise ValueError("%s: expected 'time' and 'WSE' columns, found %s"
                         % (os.path.basename(path), list(df.columns)))
    name = str(df[ncol].dropna().iloc[0]) if ncol in df else station_name_from_file(path)
    out = pd.DataFrame({
        "station": name,
        "time": pd.to_datetime(df[tcol]),
        "wse": pd.to_numeric(df[vcol], errors="coerce"),
    }).dropna(subset=["time", "wse"])
    return out


def load_reference(ref_path):
    if not ref_path or not os.path.exists(ref_path):
        return {}
    r = pd.read_csv(ref_path)
    return dict(zip(r["station"].astype(str), r["zero_gauge_rl_m"].astype(float)))


def infer_zero(series):
    """Fallback when a station has no reference row. The gauge zero sits a little
    below the low-flow water surface, so the 1st percentile minus 2 m is a safe,
    deliberately loose stand-in. It only has to be good enough to catch a value
    that belongs to a station tens or hundreds of metres away."""
    return float(np.nanpercentile(series, 1)) - 2.0


def screen(frames, zeros, spike_m, propagate_m, no_corroborate=()):
    flags = []

    # ---- screen 1: band
    for name, df in frames.items():
        z = zeros[name]
        lo, hi = z + BAND_LOW, z + BAND_HIGH
        bad = df[(df["wse"] < lo) | (df["wse"] > hi)]
        for _, row in bad.iterrows():
            flags.append({"station": name, "time": row["time"], "wse": row["wse"],
                          "screen": "BAND", "detail": "outside %.2f to %.2f" % (lo, hi)})

    # ---- screen 2: leak
    lookup = {n: dict(zip(df["time"], df["wse"])) for n, df in frames.items()}
    for f in flags:
        if f["screen"] != "BAND":
            continue
        for other, m in lookup.items():
            if other == f["station"]:
                continue
            v = m.get(f["time"])
            if v is not None and abs(v - f["wse"]) < 0.005:
                f["screen"] = "LEAK"
                f["detail"] = "identical to %s at the same timestamp" % other
                break

    # ---- screen 3: spike
    # A real flood wave crosses this reach in under a day, so a genuine rise at one
    # station shows at the others within +/- 24 h. A rise that appears nowhere else
    # is an instrument or transcription artefact.
    flagged = {(f["station"], f["time"]) for f in flags}

    def rose(other, t):
        od, zo = frames[other], zeros[other]
        ok = od[(od["wse"] >= zo + BAND_LOW) & (od["wse"] <= zo + BAND_HIGH)]
        near = ok[(ok["time"] >= t - pd.Timedelta("24h")) & (ok["time"] <= t + pd.Timedelta("24h"))]
        quiet = ok[(ok["time"] >= t - pd.Timedelta("7D")) & (ok["time"] <= t - pd.Timedelta("2D"))]
        if near.empty or quiet.empty:
            return False
        return (near["wse"].max() - quiet["wse"].median()) >= propagate_m

    for name, df in frames.items():
        z = zeros[name]
        d = df[(df["wse"] >= z + BAND_LOW) & (df["wse"] <= z + BAND_HIGH)]
        d = d.sort_values("time").drop_duplicates("time").set_index("time")
        if len(d) < 10:
            continue
        excess = d["wse"] - d["wse"].rolling("24h", min_periods=3).median()
        # A regulated station rises by more than propagate_m every single day by
        # design, so leaving one in the corroboration set answers "did anything
        # else rise?" with yes on every date and suppresses every spike at every
        # other station. Found 9 September 2026: the 631.22 m reading at
        # Rudraprayag on 24 June 2021, which this screen was written to catch,
        # was not being flagged for exactly that reason.
        others = [n for n in frames if n != name and n not in no_corroborate]
        for t, ex in excess[excess > spike_m].items():
            if (name, t) in flagged:
                continue
            if any(rose(o, t) for o in others):
                continue
            flags.append({"station": name, "time": t, "wse": float(d["wse"].loc[t]),
                          "screen": "SPIKE",
                          "detail": "%.2f m above its own 24h median, no other station rose %.2f m"
                                    % (float(ex), propagate_m)})
    return flags


def selftest():
    print("selftest: building a synthetic three-station reach")
    t = pd.date_range("2021-01-01", periods=400, freq="h")
    rng = np.random.default_rng(0)
    up = pd.DataFrame({"station": "Rudraprayag (A)", "time": t,
                       "wse": 620.0 + rng.normal(0, 0.05, len(t))})
    mid = pd.DataFrame({"station": "Srinagar", "time": t,
                        "wse": 531.0 + rng.normal(0, 0.05, len(t))})
    dn = pd.DataFrame({"station": "Devprayag (A)", "time": t,
                       "wse": 451.0 + rng.normal(0, 0.05, len(t))})
    # a real event at hour 200: everything rises
    for df, amp in ((up, 8.0), (mid, 5.0), (dn, 6.0)):
        df.loc[198:206, "wse"] += amp
    # a lone spike at hour 300, upstream only
    up.loc[300, "wse"] += 11.0
    # a leak at hour 350: upstream carries the midstream value
    up.loc[350, "wse"] = float(mid.loc[350, "wse"])
    frames = {"Rudraprayag (A)": up, "Srinagar": mid, "Devprayag (A)": dn}
    zeros = {"Rudraprayag (A)": 614.0, "Srinagar": 525.0, "Devprayag (A)": 446.0}
    flags = screen(frames, zeros, spike_m=3.0, propagate_m=1.0)
    # and again with a station that rises every day, which must not hide the spike
    import numpy as _np
    dam = pd.DataFrame({"station": "Dam", "time": t,
                        "wse": 500.0 + 2.0 * _np.sin(_np.arange(len(t)) / 3.8)})
    f2 = dict(frames); f2["Dam"] = dam
    z2 = dict(zeros); z2["Dam"] = 494.0
    flags_dam = screen(f2, z2, 3.0, 1.0, no_corroborate={"Dam"})
    flags_nodam = screen(f2, z2, 3.0, 1.0)
    got = {f["screen"] for f in flags}
    by = {}
    for f in flags:
        by.setdefault(f["screen"], []).append(f)

    ok = True
    def check(label, cond):
        nonlocal ok
        print("   [%s] %s" % ("PASS" if cond else "FAIL", label))
        ok = ok and cond

    check("LEAK screen caught the copied value", "LEAK" in got and len(by.get("LEAK", [])) == 1)
    check("SPIKE screen caught the lone spike", any(
        f["station"] == "Rudraprayag (A)" and f["time"] == t[300] for f in by.get("SPIKE", [])))
    check("real event at hour 200 was NOT flagged as a spike", not any(
        t[196] <= f["time"] <= t[208] for f in by.get("SPIKE", [])))
    check("no station other than Rudraprayag was flagged", all(
        f["station"] == "Rudraprayag (A)" for f in flags))
    check("total flags is exactly 2", len(flags) == 2)
    got_dam = {f["screen"] for f in flags_dam if f["station"] == "Rudraprayag (A)"}
    got_nodam = {f["screen"] for f in flags_nodam if f["station"] == "Rudraprayag (A)"}
    check("a daily-cycling dam does not hide a real spike when excluded",
          "SPIKE" in got_dam)
    check("and it DOES hide it when left in, which is the bug",
          "SPIKE" not in got_nodam)
    print("selftest:", "all checks passed" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", help="folder holding the GUARDIAN .xlsx station files")
    p.add_argument("--ref", help="GAUGE_station_reference.csv, for the zero gauge RLs")
    p.add_argument("--spike", type=float, default=3.0,
                   help="metres above a station's own 24h median before it is a spike candidate")
    p.add_argument("--propagate", type=float, default=1.0,
                   help="metres a downstream station must rise for the event to count as real")
    p.add_argument("--drop-spikes", action="store_true",
                   help="also remove SPIKE rows from the clean file. Off by default: BAND and "
                        "LEAK are certain, SPIKE is a candidate list that wants an eyeball first")
    p.add_argument("--no-corroborate", dest="no_corroborate",
                   default="Koteshwar,Dharasu,Joshimath",
                   help="comma separated stations that may NOT be used to "
                        "corroborate a rise at another station. Pass an empty "
                        "string to disable. Three by default, for two different "
                        "reasons. KOTESHWAR is a dam on a daily peaking cycle: "
                        "it rises by more than --propagate every single day, so "
                        "leaving it in answers 'did anything else rise' with yes "
                        "on every date and suppresses every spike at every other "
                        "station. DHARASU is on the Bhagirathi, a different "
                        "river, and JOSHIMATH is upstream of the corridor, so a "
                        "rise at either says nothing about a rise on the study "
                        "reach. All three stay in the screen as stations in "
                        "their own right; they are only barred from vouching "
                        "for somebody else.")
    p.add_argument("--drop-day", dest="drop_days", action="append", default=[],
                   metavar="STATION=YYYY-MM-DD",
                   help="remove one station-day outright, for a record known to "
                        "be corrupt. Repeatable. Example: "
                        "--drop-day \"Koteshwar=2020-07-14\". Every dropped day "
                        "is listed in the run output and written to the report.")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()

    if a.selftest:
        sys.exit(selftest())
    if not a.dir:
        p.error("--dir is required unless --selftest is given")

    files = sorted(glob.glob(os.path.join(a.dir, "*.xlsx")))
    if not files:
        print("No .xlsx files found in", a.dir)
        sys.exit(1)
    print("Reading %d station files" % len(files))

    frames = {}
    for f in files:
        try:
            df = load_one(f)
        except Exception as e:
            print("  SKIPPED %s: %s" % (os.path.basename(f), e))
            continue
        frames[df["station"].iloc[0]] = df
        print("  %-26s %6d readings  %s to %s" % (
            df["station"].iloc[0], len(df), df["time"].min().date(), df["time"].max().date()))

    if a.drop_days:
        pairs = []
        for spec in a.drop_days:
            if "=" not in spec:
                p.error("--drop-day wants STATION=YYYY-MM-DD, got %r" % spec)
            st, day = spec.split("=", 1)
            pairs.append((st.strip(), pd.Timestamp(day.strip()).date()))
        for st, day in pairs:
            if st not in frames:
                print("  --drop-day: no station called %r. Names are: %s"
                      % (st, ", ".join(sorted(frames))))
                continue
            before = len(frames[st])
            frames[st] = frames[st][frames[st]["time"].dt.date != day]
            print("  dropped %-24s %s   %d readings removed"
                  % (st, day, before - len(frames[st])))

    if not frames:
        print("\nAll %d files were skipped, so there is nothing to screen." % len(files))
        print("Read the SKIPPED lines above for the reason on each file.")
        print("If they say openpyxl, run:  conda activate fyp    then    pip install openpyxl")
        sys.exit(1)

    ref = load_reference(a.ref)
    zeros = {}
    for n, df in frames.items():
        if n in ref:
            zeros[n] = ref[n]
        else:
            zeros[n] = infer_zero(df["wse"])
            print("  no reference row for %-24s using inferred zero %.2f m" % (n, zeros[n]))

    nocorr = {x.strip() for x in (a.no_corroborate or "").split(",") if x.strip()}
    if nocorr:
        present = sorted(nocorr & set(frames))
        if present:
            print("\n  not used to corroborate a rise elsewhere: %s"
                  % ", ".join(present))
    flags = screen(frames, zeros, a.spike, a.propagate, no_corroborate=nocorr)

    print("\n%d readings flagged" % len(flags))
    if flags:
        rep = pd.DataFrame(flags).sort_values(["screen", "station", "time"])
        for s in ["LEAK", "SPIKE", "BAND"]:
            sub = rep[rep["screen"] == s]
            if sub.empty:
                continue
            print("\n  %s  (%d)" % (s, len(sub)))
            for _, r in sub.head(30).iterrows():
                print("    %-24s %s  %8.2f   %s" % (r["station"], r["time"], r["wse"], r["detail"]))
            if len(sub) > 30:
                print("    ... %d more, see the report" % (len(sub) - 30))
        out_rep = os.path.join(a.dir, "GAUGE_screen_report.csv")
        rep.to_csv(out_rep, index=False)
        print("\nwrote", out_rep)

    certain = {"BAND", "LEAK"} | ({"SPIKE"} if a.drop_spikes else set())
    drop = {(f["station"], pd.Timestamp(f["time"])) for f in flags if f["screen"] in certain}
    spikes = {(f["station"], pd.Timestamp(f["time"])) for f in flags if f["screen"] == "SPIKE"}
    clean = []
    for n, df in frames.items():
        keep = df[~df.apply(lambda r: (n, pd.Timestamp(r["time"])) in drop, axis=1)].copy()
        keep["zero_gauge_rl_m"] = zeros[n]
        keep["stage_m"] = keep["wse"] - zeros[n]
        keep["spike_flag"] = keep["time"].apply(lambda t: (n, pd.Timestamp(t)) in spikes)
        clean.append(keep)
    allc = pd.concat(clean).sort_values(["station", "time"])
    out_clean = os.path.join(a.dir, "GAUGE_wse_clean.csv")
    allc.to_csv(out_clean, index=False)
    print("wrote %s  (%d readings kept, %d removed)" % (out_clean, len(allc), len(drop)))
    if spikes and not a.drop_spikes:
        print("      %d SPIKE rows were KEPT and marked in the spike_flag column." % len(spikes))
        print("      Look at them in the report, then re-run with --drop-spikes if you agree.")


if __name__ == "__main__":
    main()
