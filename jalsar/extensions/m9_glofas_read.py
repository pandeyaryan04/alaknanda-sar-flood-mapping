"""Jal-SAR extensions, items M9 and M10: shared reading code for the GloFAS files (no outputs of its own).

What it does:
  open_forecast(paths)     the forecast files as blocks, one per grid (issue day, member, lead, lat, lon)
  open_historical(paths)   the reanalysis files as one daily series per cell (day, lat, lon)
  stations()               the five gauges of GAUGE_station_reference.csv
  match_cells(hist, st)    the GloFAS river cell that stands for each gauge
  gumbel_thresholds(...)   2, 5 and 20 year flood discharges from the annual maxima
  fit_rating(...)          stage = h0 + a * Q^b at a gauge (GloFAS discharge against GUARDIAN stage),
                           or stage = h0 + a * ln(Q) where the power curve's b hits its 0.05 edge

Day convention (an assumption, stated in every report): a GloFAS value
for "the last 24 hours" is stamped with the END of its 24 hours. So the value
for calendar day D (00 to 24 UTC) is the one stamped D+1 00:00, and a forecast
issued at 00 UTC on day I with lead time L hours covers the day I + L/24 - 1.

Run:  python m9_glofas_read.py --check    (prints what it finds in the GloFAS folder)
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
GLOFAS = ROOT / "06 Outputs" / "jalsar" / "extensions" / "glofas"
GAUGES = ROOT / "05 Data" / "gauges"
STATION_CSV = GAUGES / "GAUGE_station_reference.csv"
STAGE_CSV = GAUGES / "guardian" / "GAUGE_wse_clean.csv"
DAY = np.timedelta64(1, "D")

# Which GloFAS version made a forecast, so it is compared with its own version's thresholds.
# The forecast files carry no version number,
# so this goes by the operational dates. v5.0: "Operational version - GloFAS v5.0 released
# 2026-07-15" (EWDS cems-glofas-historical overview page, read 25 Oct 2026). v4.0: from 26 July
# 2023; before that GloFAS 3.1 (0.1 degree grid; the last 0.1 degree file here is 19 July 2023).
V5_FROM = np.datetime64("2026-07-15", "D")
V4_FROM = np.datetime64("2023-07-26", "D")
HIST_VERSIONS = ("version_4_0", "version_5_0")
VERSION_NAME = {"version_3_1": "GloFAS 3.1", "version_4_0": "GloFAS 4.0", "version_5_0": "GloFAS 5.0"}


def forecast_version(issue):
    """The GloFAS version that made a forecast issued on this day."""
    issue = np.datetime64(issue, "D")
    if issue >= V5_FROM:
        return "version_5_0"
    if issue >= V4_FROM:
        return "version_4_0"
    return "version_3_1"


def threshold_version(issue, available, fallback="version_4_0"):
    """(reanalysis version whose thresholds go with this forecast, version that made the forecast).
    GloFAS 3.1 has no reanalysis here, so its forecasts get the fallback version, flagged by the second value."""
    v = forecast_version(issue)
    return (v if v in available else fallback), v


def _need_xarray():
    try:
        import xarray as xr
    except ImportError:
        sys.exit("xarray is missing in this environment.")
    try:
        import netCDF4  # noqa: F401
    except ImportError:
        try:
            import h5netcdf  # noqa: F401
        except ImportError:
            sys.exit("Neither netCDF4 nor h5netcdf is installed, so the GloFAS NetCDF files cannot be "
                     "read. Run M INSTALL EXTRA.bat first.")
    return xr


def _open_one(path):
    xr = _need_xarray()
    path = Path(path)
    if path.suffix in (".grib", ".grib2", ".grb"):
        try:
            return xr.open_dataset(path, engine="cfgrib")
        except Exception as exc:
            sys.exit(f"{path.name} is GRIB, and GRIB cannot be read here ({exc}). The usual "
                     "fix is to download that month again as NetCDF.")
    return xr.open_dataset(path)


def _discharge_var(ds):
    names = list(ds.data_vars)
    for n in ("dis24", "dis", "rdis"):
        if n in names:
            return n
    cand = [n for n in names if "dis" in n.lower()]
    if not cand:
        sys.exit(f"No river discharge variable in the file (variables: {names}).")
    return cand[0]


def _rename(ds):
    m = {}
    for a, b in (("latitude", "lat"), ("longitude", "lon"), ("number", "member"), ("realization", "member"),
                 ("forecast_reference_time", "issue"), ("forecast_period", "lead"), ("step", "lead"),
                 ("leadtime", "lead"), ("valid_time", "valid")):
        if a in ds.dims or a in ds.coords:
            m[a] = b
    return ds.rename(m)


def _lead_hours(v):
    v = np.asarray(v)
    if np.issubdtype(v.dtype, np.timedelta64):
        return (v / np.timedelta64(1, "h")).astype(int)
    return v.astype(int)


def _forecast_part(p):
    """One forecast file as a DataArray (issue, member, lead, lat, lon)."""
    ds = _rename(_open_one(p))
    da = ds[_discharge_var(ds)]
    if "issue" not in da.dims:
        if "time" in da.dims:
            da = da.rename(time="issue")
        elif "issue" in da.coords:
            da = da.expand_dims("issue")
        elif "time" in da.coords:
            da = da.rename(time="issue").expand_dims("issue")
        else:
            sys.exit(f"{Path(p).name}: cannot find the issue time.")
    if "member" not in da.dims:
        m = int(da["member"].values) if "member" in da.coords else 0
        if "member" in da.coords:
            da = da.drop_vars("member")
        da = da.expand_dims(member=[m])
    if "lead" not in da.dims:
        if "valid" in da.dims:
            lead = da["valid"].values - da["issue"].values[0]
            da = da.rename(valid="lead").assign_coords(lead=lead)
        elif "lead" in da.coords:
            da = da.expand_dims("lead")
        else:
            sys.exit(f"{Path(p).name}: cannot find the lead time.")
    for c in [c for c in da.coords if c not in ("issue", "member", "lead", "lat", "lon")]:
        da = da.drop_vars(c)
    da = da.transpose("issue", "member", "lead", "lat", "lon").load()
    ds.close()
    return da


def open_forecast(paths):
    """The forecast files as blocks, one per grid (GloFAS 3.1 is 0.1 degree, 4.0 and later 0.05).

    Each block: {'q': (issue, member, lead, lat, lon), 'issue': datetime64[D], 'member', 'lead_h',
    'lat', 'lon', 'files', 'step_deg'}. Pieces of one issue day (control, members, lead
    ranges) are put together; a missing piece stays NaN."""
    groups = {}
    for p in paths:
        da = _forecast_part(p)
        lat, lon = np.round(da["lat"].values.astype(float), 4), np.round(da["lon"].values.astype(float), 4)
        key = (tuple(lat), tuple(lon))
        groups.setdefault(key, []).append((Path(p).name, da))
    blocks = []
    for (lat, lon), parts in groups.items():
        iss = sorted({d for _, da in parts for d in da["issue"].values.astype("datetime64[D]")})
        mem = sorted({int(m) for _, da in parts for m in da["member"].values})
        lead = sorted({int(h) for _, da in parts for h in _lead_hours(da["lead"].values)})
        q = np.full((len(iss), len(mem), len(lead), len(lat), len(lon)), np.nan)
        ii, mi, li = ({v: k for k, v in enumerate(x)} for x in (iss, mem, lead))
        for _, da in parts:
            a = [ii[d] for d in da["issue"].values.astype("datetime64[D]")]
            b = [mi[int(m)] for m in da["member"].values]
            c = [li[int(h)] for h in _lead_hours(da["lead"].values)]
            q[np.ix_(a, b, c, range(len(lat)), range(len(lon)))] = np.asarray(da.values, "float64")
        blocks.append({"q": q, "issue": np.array(iss, "datetime64[D]"), "member": np.array(mem),
                       "lead_h": np.array(lead), "lat": np.array(lat), "lon": np.array(lon),
                       "files": [n for n, _ in parts],
                       "step_deg": float(abs(lat[1] - lat[0])) if len(lat) > 1 else float("nan")})
    return sorted(blocks, key=lambda b: b["issue"][0]) if blocks else None


def block_cell(block, station, hist=None, hist_cell=None, search=1):
    """The cell of a forecast block that stands for a gauge: the reanalysis cell when the grids are
    the same; otherwise, within +-search cells of the gauge, the cell with the largest mean forecast."""
    lat, lon = block["lat"], block["lon"]
    if hist is not None and hist_cell is not None and len(lat) == len(hist["lat"]) \
            and np.allclose(lat, hist["lat"], atol=1e-3) and np.allclose(lon, hist["lon"], atol=1e-3):
        return hist_cell["i"], hist_cell["j"], "same grid as the reanalysis"
    i0, j0 = int(np.abs(lat - station["lat"]).argmin()), int(np.abs(lon - station["lon"]).argmin())
    with np.errstate(all="ignore"):
        mean = np.nanmean(block["q"], axis=(0, 1, 2))
    cands = [(i, j) for i in range(max(0, i0 - search), min(len(lat), i0 + search + 1))
             for j in range(max(0, j0 - search), min(len(lon), j0 + search + 1))]
    vals = [mean[i, j] if np.isfinite(mean[i, j]) else -1 for i, j in cands]
    i, j = cands[int(np.argmax(vals))]
    return i, j, f"largest mean forecast within one cell ({block['step_deg']:.2f} degree grid)"


def find_issue(blocks, day):
    for b in blocks or []:
        k = np.where(b["issue"] == day)[0]
        if len(k):
            return b, int(k[0])
    return None, None


def open_historical(paths, version=None):
    """{'q': array (day, lat, lon), 'day': datetime64[D] (the calendar day each value covers),
    'lat', 'lon', 'files'}. version picks one system version from the file names (version_4_0)."""
    paths = [Path(p) for p in paths if version is None or version in Path(p).name]
    # consolidated files first, "intermediate" ones (the newest months) after, so that on a day
    # both hold, the consolidated value is kept
    paths = sorted(paths, key=lambda p: (("intermediate" in p.name or p.name.endswith("_inte.nc")), p.name))
    parts = []
    for p in paths:
        ds = _rename(_open_one(p))
        da = ds[_discharge_var(ds)]
        tdim = "valid" if "valid" in da.dims else "time" if "time" in da.dims else None
        if tdim is None:
            sys.exit(f"{p.name}: cannot find the time.")
        da = da.rename({tdim: "t"})
        for extra in [d for d in da.dims if d not in ("t", "lat", "lon")]:
            da = da.isel({extra: 0})
        parts.append(da.transpose("t", "lat", "lon").load())
        ds.close()
    if not parts:
        return None
    xr = _need_xarray()
    da = xr.concat(parts, dim="t")
    stamp = da["t"].values.astype("datetime64[D]")
    order = np.argsort(stamp, kind="stable")                  # equal days keep the file order above
    _, first = np.unique(stamp[order], return_index=True)
    da = da.isel(t=order[first])
    day = da["t"].values.astype("datetime64[D]") - DAY        # day convention: stamp is the end of the day
    return {"q": np.asarray(da.values, "float64"), "day": day, "lat": np.asarray(da["lat"].values, float),
            "lon": np.asarray(da["lon"].values, float), "files": [p.name for p in paths]}


def historical_files(version="version_4_0"):
    fs = sorted((GLOFAS / "historical").glob("*.nc"))
    pick = [f for f in fs if version in f.name]
    return pick or fs, (version if pick else "all versions found")


def open_versions(primary="version_4_0"):
    """{version: open_historical(...)} for every reanalysis version with files, the primary first."""
    out = {}
    for v in [primary] + [x for x in HIST_VERSIONS if x != primary]:
        fs = sorted(f for f in (GLOFAS / "historical").glob("*.nc") if v in f.name)
        if fs:
            out[v] = open_historical(fs)
    return out


def same_grid(a, b):
    return (len(a["lat"]) == len(b["lat"]) and len(a["lon"]) == len(b["lon"])
            and np.allclose(a["lat"], b["lat"], atol=1e-3) and np.allclose(a["lon"], b["lon"], atol=1e-3))


def stations(names=None):
    rows = list(csv.DictReader(open(STATION_CSV, encoding="utf-8-sig")))
    out = []
    for r in rows:
        if names and r["station"] not in names:
            continue
        d = {"name": r["station"], "river": r["river"], "lat": float(r["lat"]), "lon": float(r["lon"]),
             "zero_m": float(r["zero_gauge_rl_m"]), "catchment_km2": float(r["catchment_km2"] or "nan")}
        for k in ("warning_stage_m", "danger_stage_m", "hfl_stage_m"):
            d[k] = float(r[k]) if r.get(k) else float("nan")
        out.append(d)
    return out


def match_cells(hist, sts, search=1, uparea=None):
    """The river cell for each gauge, within +-search cells of the nearest cell.

    With the GloFAS upstream area file (uparea_glofas_v4_0.nc in the glofas folder), the
    cell whose upstream area is closest to the gauge's catchment area is taken.
    Without it, the cell with the largest mean discharge (the main river). Both
    choices are recorded."""
    lat, lon = hist["lat"], hist["lon"]
    mean = np.nanmean(hist["q"], axis=0)
    out = {}
    for s in sts:
        i0, j0 = int(np.abs(lat - s["lat"]).argmin()), int(np.abs(lon - s["lon"]).argmin())
        best, rule = None, ""
        cands = [(i, j) for i in range(max(0, i0 - search), min(len(lat), i0 + search + 1))
                 for j in range(max(0, j0 - search), min(len(lon), j0 + search + 1))]
        if uparea is not None and np.isfinite(s["catchment_km2"]):
            ua = uparea(lat, lon)
            errs = [abs(ua[i, j] - s["catchment_km2"]) / s["catchment_km2"] for i, j in cands]
            k = int(np.nanargmin(errs))
            best, rule = cands[k], f"upstream area closest to {s['catchment_km2']:.0f} km2 ({errs[k] * 100:.0f} % off)"
        if best is None:
            vals = [mean[i, j] if np.isfinite(mean[i, j]) else -1 for i, j in cands]
            best, rule = cands[int(np.argmax(vals))], "largest mean discharge within one cell"
        i, j = best
        out[s["name"]] = {"i": i, "j": j, "cell_lat": float(lat[i]), "cell_lon": float(lon[j]),
                          "mean_q": float(mean[i, j]), "rule": rule,
                          "offset_km": float(np.hypot((lat[i] - s["lat"]) * 110.6,
                                                      (lon[j] - s["lon"]) * 96.2))}
    return out


UPAREA_FILES = ("uparea_glofas_v4_0.nc", "glofas_upArea.nc")   # the 0.05 degree file first
CORRIDOR_BOX = (29.5, 31.0, 77.5, 80.0)                          # lat min, lat max, lon min, lon max


def uparea_path():
    """The GloFAS upstream area file in the glofas folder, or None. uparea_glofas_v4_0.nc (0.05 degree,
    GloFAS 4.0 and later, CEMS Auxiliary Data page) is preferred over the old 0.1 degree glofas_upArea.nc."""
    for n in UPAREA_FILES:
        if (GLOFAS / n).exists():
            return GLOFAS / n
    return None


def load_uparea():
    """at(lat, lon) -> upstream area in km2 on those cell centres, or None without the file.
    Only the corridor box is read, so the 87 MB global file costs little memory."""
    p = uparea_path()
    if p is None:
        return None
    ds = _rename(_open_one(p))
    da = ds[list(ds.data_vars)[0]]
    la0, la1, lo0, lo1 = CORRIDOR_BOX
    lat_desc = bool(da["lat"].values[0] > da["lat"].values[-1])
    da = da.sel(lat=slice(la1, la0) if lat_desc else slice(la0, la1), lon=slice(lo0, lo1))
    vals = np.asarray(da.values, float).squeeze()
    if np.nanmax(vals) > 1e7:                  # given in m2 (the CEMS files are)
        vals = vals / 1e6
    la, lo = np.asarray(da["lat"].values), np.asarray(da["lon"].values)

    def at(lat, lon):
        ii = [int(np.abs(la - x).argmin()) for x in lat]
        jj = [int(np.abs(lo - x).argmin()) for x in lon]
        return vals[np.ix_(ii, jj)]
    return at


def gumbel_thresholds(series, days, periods=(2, 5, 20)):
    """Gumbel (maximum likelihood) on the annual maxima of a daily series; years with
    under 300 days are left out. Returns {T: discharge}, the maxima and the fit."""
    from scipy.stats import gumbel_r
    years = days.astype("datetime64[Y]").astype(int) + 1970
    am = []
    for y in np.unique(years):
        v = series[years == y]
        v = v[np.isfinite(v)]
        if len(v) >= 300:
            am.append(float(v.max()))
    am = np.array(am)
    if len(am) < 10:
        return None, am, None
    loc, scale = gumbel_r.fit(am)
    return ({int(T): float(gumbel_r.ppf(1 - 1 / T, loc, scale)) for T in periods}, am,
            {"loc": float(loc), "scale": float(scale), "years": int(len(am))})


def daily_stage(name):
    """Daily mean stage (m above the gauge zero) at a GUARDIAN gauge, spikes left out."""
    import pandas as pd
    d = pd.read_csv(STAGE_CSV, parse_dates=["time"])
    d = d[(d["station"] == name) & ~d["spike_flag"].astype(bool)]
    if d.empty:
        return None
    s = d.set_index("time")["stage_m"].resample("D").mean().dropna()
    return s


def _fit_power(q, h):
    from scipy.optimize import curve_fit

    def f(x, h0, a, b):
        return h0 + a * np.power(x, b)
    try:
        p, _ = curve_fit(f, q, h, p0=[float(h.min()), 0.3, 0.5], bounds=([-50, 1e-6, 0.05], [50, 100, 1.5]),
                         maxfev=20000)
    except Exception:
        return None
    return {"shape": "power", "h0": float(p[0]), "a": float(p[1]), "b": float(p[2])}


def _fit_log(q, h):
    A = np.vstack([np.ones_like(q), np.log(q)]).T
    co, *_ = np.linalg.lstsq(A, h, rcond=None)
    if not np.isfinite(co).all() or co[1] <= 0:
        return None
    return {"shape": "log", "h0": float(co[0]), "a": float(co[1]), "b": None}


def _score(r, q, h):
    pred = stage_from_q(r, q)
    ss = float(((h - h.mean()) ** 2).sum())
    r["r2"] = 1 - float(((h - pred) ** 2).sum()) / ss if ss > 0 else float("nan")
    r["rmse_m"] = float(np.sqrt(np.mean((h - pred) ** 2)))
    return r


def fit_rating(q, h, shape="auto"):
    """Stage relation at a gauge, least squares.
      power  stage = h0 + a * Q^b   (0.05 <= b <= 1.5)
      log    stage = h0 + a * ln(Q) (the limit of the power curve as b goes to 0)
    shape "auto": the power curve, unless its b sits on the 0.05 edge and the log curve
    fits at least as well (R2 within 0.002), then the log curve. Both R2 values are kept.
    Returns dict with shape, h0, a, b, r2, rmse_m, n, q_range, h_range, r2_power, r2_log."""
    q, h = np.asarray(q, float), np.asarray(h, float)
    ok = np.isfinite(q) & np.isfinite(h) & (q > 0)
    q, h = q[ok], h[ok]
    if len(q) < 30:
        return None
    pw = _fit_power(q, h)
    lg = _fit_log(q, h)
    pw = _score(pw, q, h) if pw else None
    lg = _score(lg, q, h) if lg else None
    if shape == "power" or lg is None:
        r = pw
    elif shape == "log" or pw is None:
        r = lg
    else:
        at_edge = pw["b"] <= 0.0501
        r = lg if (at_edge and lg["r2"] >= pw["r2"] - 0.002) else pw
    if r is None:
        return None
    r = dict(r)
    r.update({"n": int(len(q)), "q_range": [float(q.min()), float(q.max())],
              "h_range": [float(h.min()), float(h.max())],
              "r2_power": None if pw is None else pw["r2"], "r2_log": None if lg is None else lg["r2"],
              "power_b": None if pw is None else pw["b"]})
    return r


def stage_from_q(r, q):
    q = np.maximum(np.asarray(q, float), 0)
    if r.get("shape") == "log":
        with np.errstate(divide="ignore"):
            return r["h0"] + r["a"] * np.log(np.maximum(q, 1e-6))
    return r["h0"] + r["a"] * np.power(q, r["b"])


def q_from_stage(r, h):
    x = (np.asarray(h, float) - r["h0"]) / r["a"]
    if r.get("shape") == "log":
        with np.errstate(over="ignore"):
            return np.exp(np.minimum(x, 700))
    return np.where(x > 0, np.power(np.maximum(x, 1e-12), 1 / r["b"]), 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    fs = sorted((GLOFAS / "forecast").glob("*.nc"))
    hs, ver = historical_files()
    print(f"forecast files: {len(fs)}; reanalysis files: {len(hs)} ({ver})")
    if args.check and hs:
        h = open_historical(hs[:1])
        print(f"  first reanalysis file: {h['day'][0]} to {h['day'][-1]}, grid {len(h['lat'])} x {len(h['lon'])}")
        for v in HIST_VERSIONS:
            vf = sorted(f for f in (GLOFAS / "historical").glob("*.nc") if v in f.name)
            if vf:
                hv = open_historical(vf)
                print(f"  reanalysis {v}: {len(vf)} files, {hv['day'][0]} to {hv['day'][-1]}")
    if args.check and fs:
        for b in open_forecast(fs):
            n_nan = float(np.isnan(b["q"]).mean())
            print(f"  forecast block {b['step_deg']:.2f} degree: {len(b['issue'])} issue days ({b['issue'][0]} to "
                  f"{b['issue'][-1]}), {len(b['member'])} members, leads {b['lead_h'].min()} to "
                  f"{b['lead_h'].max()} h, {n_nan * 100:.0f} % missing, from {len(b['files'])} files")
    if args.check and not (fs and hs):
        print("PROBLEM: forecast or reanalysis files missing")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
