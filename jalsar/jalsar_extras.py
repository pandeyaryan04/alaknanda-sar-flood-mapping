r"""
jalsar_extras.py
================
Extensions to Jal-SAR (review suggestions, 21 October 2026).
Nothing here changes a flood map, the trust rule or an existing number.

M2  Trust explained. The headline map of each validated event is scored
    against the hand-drawn reference inside each trust class: hit rate,
    false alarm ratio and critical success index (CSI), with the same window
    bootstrap as every other interval (04_stratified_metrics.py). Both events
    are pooled (16 windows) and also given one by one. The trust rule itself
    is unchanged (fixed on 8 October 2026, before these scores). Caveat printed with it: the classes were set from the
    same reference windows, so the numbers describe the classes; they are not
    an independent test.

Extra layers for the web map, each shown only when its file exists:
    population      WorldPop 2020, people per 100 m cell (05 Data\population)
    buildings       buildings inside or near the mapped flood (M7)
    power           power lines and poles inside or near the flood (M7)
    towers          mobile and communication towers inside or near the flood (M7)
    vulnerability   the PCA vulnerability index (M8)
    forecast        the GloFAS forecast watch points (M9) and the forecast
                    extent (M10)

The files they read are written by the extension scripts in 04 Code\jalsar\extensions:
    06 Outputs\jalsar\<event>\extras\exposure.json and exposure_features.geojson
    06 Outputs\jalsar\extensions\m8_vulnerability\vulnerability.geojson
    06 Outputs\jalsar\extensions\m9_forecast\forecast_watch.json (and forecast_*.tif)

USAGE  (Anaconda Prompt, fyp active, from 04 Code\jalsar)
-----
    python jalsar.py trustscores          (M2: writes 06 Outputs\jalsar\trust_scores.json and .md)
    python jalsar_extras.py --selftest
"""

__stage__ = "Jal-SAR: extensions"
__reads__ = ["06 Outputs/jalsar/<event>/ (headline map, trust layer)", "reference rasters and windows",
             "extension outputs (exposure, vulnerability, forecast)"]
__writes__ = ["06 Outputs/jalsar/trust_scores.json", "06 Outputs/jalsar/trust_scores.md"]

import html
import json
import math
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import jalsar_core as gc
import jalsar_display as gd

mc = gc.mc
scorer = gc.scorer

TRUST_ORDER = (3, 2, 1, 0)
POP_BINS = [1, 5, 20, 50, 1e9]                 # people per 100 m cell
POP_COLOURS = {1: "#fde0c5", 2: "#facba6", 3: "#f59e72", 4: "#d9534f"}
VULN_COLOURS = {1: "#1a9850", 2: "#91cf60", 3: "#fee08b", 4: "#fc8d59", 5: "#d73027"}
EXPO_COLOURS = {"building": "#b91c1c", "power_line": "#7c3aed", "power_point": "#7c3aed",
                "tower": "#0f766e"}
FORECAST_COLOURS = {"none": "#1f7a3a", "watch": "#e08e0b", "warning": "#c62828", "unknown": "#8a8a8a"}
# relative risk classes (flooded, people exposed, vulnerability class) of m8_risk_view.py
RISK_COLOURS = {"high": "#b91c1c", "medium": "#f59e0b", "low": "#fde68a",
                "flooded, no one mapped inside": "#e5e7eb", "no vulnerability class": "#9ca3af"}
EXPOSURE_LIMITS = ("Inside or near the mapped flood does not mean damaged. Near: within 100 m. People: WorldPop "
                   "2020, a modelled estimate ('with edge' widens the flood by 30 m for edge error). Buildings: "
                   "Google Open Buildings v3, confidence 0.70 or more; it misses or merges some buildings under trees "
                   "and on steep slopes. Power lines: OpenStreetMap and the World Bank's predicted line map (about "
                   "70 % accurate at 1 km). Towers: OpenStreetMap only; the public OpenCelliD copy holds no Indian "
                   "cells, so towers nobody has mapped are missing. Roads: OpenStreetMap.")


# =============================================================================
# M2: trust explained with hit rate, false alarm ratio and CSI
# =============================================================================

def trust_scores_path(gp):
    return gp.gj / "trust_scores.json"


def _event_counts(gp, e):
    """Per trust class, per window counts (n_win + 1, 4) of the headline map, and n_win."""
    d = gp.event_dir(e.id)
    flood, prof = gd.read_u8(d / "flood_headline.tif")
    trust, _ = gd.read_u8(d / "trust.tif")
    truth = scorer.read_band(str(gp.reference(e.validated_year)), flood.shape)
    win, n_win = scorer.window_raster(str(gp.windows(e.validated_year)), flood.shape,
                                      prof["transform"], prof["crs"])
    base = np.isin(flood, (0, 1)) & np.isin(truth, (0, 1))
    out = {}
    for k in TRUST_ORDER:
        out[k] = scorer.window_counts(flood, truth, base & (trust == k), win, n_win)
    out["all"] = scorer.window_counts(flood, truth, base, win, n_win)
    return out, n_win


def _weights(n_win):
    rng = np.random.default_rng(scorer.BOOT_SEED)
    draws = rng.integers(0, n_win, size=(scorer.N_BOOT, n_win))
    return np.stack([np.bincount(dw, minlength=n_win) for dw in draws])


def _score(wc, weights):
    s = scorer.score(wc, weights)
    if s is None:
        return None
    keep = ("pixels", "flood_pixels_truth", "windows_with_flood", "tp", "fp", "fn",
            "hit_rate", "hit_rate_lo", "hit_rate_hi", "far", "far_lo", "far_hi",
            "csi", "csi_lo", "csi_hi", "precision", "f1")
    return {k: (None if isinstance(s.get(k), float) and not math.isfinite(s[k]) else s.get(k))
            for k in keep}


def compute_trust_scores(gp, write=True):
    """Scores of the headline map in each trust class, per validated event and pooled."""
    evs = [e for e in gc.load_events(gp) if e.validated_year
           and (gp.event_dir(e.id) / "trust.tif").exists()
           and gp.reference(e.validated_year).exists() and gp.windows(e.validated_year).exists()]
    if not evs:
        return None
    per_event, stacks = {}, {k: [] for k in TRUST_ORDER + ("all",)}
    head = None
    for e in evs:
        summ = mc.load_json(gp.event_dir(e.id) / "summary.json")
        head = head or summ["headline"]["name"]
        counts, n_win = _event_counts(gp, e)
        w = _weights(n_win)
        per_event[e.id] = {"year": e.validated_year, "windows": n_win,
                           "classes": {gc.TRUST_NAMES[k]: _score(counts[k], w) for k in TRUST_ORDER},
                           "all": _score(counts["all"], w)}
        for k in stacks:
            stacks[k].append(counts[k][1:])
    n_all = sum(len(s) for s in stacks["all"])
    w_all = _weights(n_all)
    pooled = {}
    for k in TRUST_ORDER + ("all",):
        wc = np.vstack([np.zeros((1, 4), "int64")] + stacks[k])
        pooled[gc.TRUST_NAMES[k] if k != "all" else "all"] = _score(wc, w_all)
    tot_ref = (pooled["all"] or {}).get("flood_pixels_truth") or 0
    share = {gc.TRUST_NAMES[k]: (round((pooled[gc.TRUST_NAMES[k]] or {}).get("flood_pixels_truth", 0)
                                       / tot_ref, 4) if tot_ref else None) for k in TRUST_ORDER}
    out = {"made_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "headline_model": head, "events": [e.id for e in evs], "windows_pooled": n_all,
           "pooled": pooled, "per_event": per_event, "reference_flood_share_by_class": share,
           "method": ("headline flood map of each validated event against the hand-drawn reference, "
                      "cells inside each trust class; 95 % interval from resampling the digitising "
                      f"windows ({scorer.N_BOOT} resamples, seed {scorer.BOOT_SEED}); pooled over "
                      f"{len(evs)} events ({n_all} windows)"),
           "caveat": ("The trust classes were set from Table D on the same reference windows, so these "
                      "numbers describe what each class means on the checked events. They are not an "
                      "independent test of the classes."),
           "rule_unchanged": "lowest Table D precision of the cell's three terrain classes; 0.40 and "
                             "0.20 (fixed 8 October 2026)"}
    if write:
        mc.save_json(trust_scores_path(gp), out)
        (gp.gj / "trust_scores.md").write_text(trust_scores_md(out), encoding="utf-8")
    return out


def _ci(s, k, pct=False):
    if not s or s.get(k) is None:
        return "n/a"
    f = (lambda v: f"{v * 100:.0f} %") if pct else (lambda v: f"{v:.2f}")
    lo, hi = s.get(f"{k}_lo"), s.get(f"{k}_hi")
    if lo is None or hi is None:
        return f(s[k])
    return f"{f(s[k])} [{f(lo)} to {f(hi)}]"


def trust_scores_md(ts):
    lines = ["# What each trust class means (Stage M2)", "",
             f"Headline model: {ts['headline_model']}. {ts['method']}.", "",
             "Hit rate: share of the real (hand-drawn) flood the map found. False alarm ratio: share "
             "of the map's flood calls that were wrong. CSI: one score that counts both misses and "
             "false alarms (1 is perfect).", "",
             "| Trust class | Reference flood in this class | Hit rate | False alarm ratio | CSI |",
             "|---|---|---|---|---|"]
    for k in TRUST_ORDER:
        n = gc.TRUST_NAMES[k]
        s = ts["pooled"].get(n)
        sh = ts["reference_flood_share_by_class"].get(n)
        lines.append(f"| {n} | {sh * 100:.1f} % |" if sh is not None else f"| {n} | n/a |")
        lines[-1] += f" {_ci(s, 'hit_rate')} | {_ci(s, 'far')} | {_ci(s, 'csi')} |"
    s = ts["pooled"].get("all")
    lines += [f"| all cells | 100 % | {_ci(s, 'hit_rate')} | {_ci(s, 'far')} | {_ci(s, 'csi')} |", "",
              "Radar blind cells are never mapped as flood (layover or shadow), so their hit rate is 0 "
              "by rule; the share of the real flood that sat in them is what the radar could not see.",
              "", ts["caveat"], "", "## One event at a time", ""]
    for eid, pe in ts["per_event"].items():
        lines += [f"### {eid}", "", "| Trust class | Hit rate | False alarm ratio | CSI |",
                  "|---|---|---|---|"]
        for k in TRUST_ORDER:
            s = pe["classes"].get(gc.TRUST_NAMES[k])
            lines.append(f"| {gc.TRUST_NAMES[k]} | {_ci(s, 'hit_rate')} | {_ci(s, 'far')} | {_ci(s, 'csi')} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def load_trust_scores(gp):
    p = trust_scores_path(gp)
    return gd.clean(mc.load_json(p)) if p.exists() else None


def trust_plain_numbers(ts):
    """One sentence per trust class, in plain words, from the pooled scores (or {})."""
    if not ts:
        return {}
    out = {}
    for k in (3, 2, 1):
        s = ts["pooled"].get(gc.TRUST_NAMES[k])
        if not s or s.get("hit_rate") is None:
            continue
        far = s.get("far")
        wrong = (f"{far * 100:.0f} % of its flood calls there were wrong" if far is not None
                 else "it made no flood calls there")
        out[gc.TRUST_NAMES[k]] = (f"On the checked events the map found {s['hit_rate'] * 100:.0f} % of the "
                                  f"real flood in {gc.TRUST_NAMES[k]}-trust terrain, and {wrong} "
                                  f"(CSI {s['csi']:.2f}).")
    sh = ts["reference_flood_share_by_class"].get("radar blind")
    if sh is not None:
        out["radar blind"] = (f"{sh * 100:.0f} % of the real flood on the checked events lay on radar-blind "
                              "slopes, where the radar could not see it.")
    return out


def _ci_html(s, k, pct=False):
    """Value on one line, its 95 % range small underneath."""
    if not s or s.get(k) is None:
        return "n/a"
    f = (lambda v: f"{v * 100:.0f} %") if pct else (lambda v: f"{v:.2f}")
    lo, hi = s.get(f"{k}_lo"), s.get(f"{k}_hi")
    rng = f"<br><span class='mu'>{f(lo)} to {f(hi)}</span>" if lo is not None and hi is not None else ""
    return f"{f(s[k])}{rng}"


def trust_table_html(ts):
    if not ts:
        return ""
    rows = []
    for k in TRUST_ORDER:
        n = gc.TRUST_NAMES[k]
        s = ts["pooled"].get(n)
        sh = ts["reference_flood_share_by_class"].get(n)
        rows.append(f"<tr><td><span class='sw' style='background:{gd.TRUST_COLOURS[k]}'></span>{n}</td>"
                    f"<td>{'n/a' if sh is None else f'{sh * 100:.0f} %'}</td>"
                    f"<td>{_ci_html(s, 'hit_rate', True)}</td><td>{_ci_html(s, 'far', True)}</td>"
                    f"<td>{_ci_html(s, 'csi')}</td></tr>")
    return ("<table class='tt'><tr><th>Trust</th><th>Real flood here</th><th>Found (hit rate)</th>"
            "<th>Wrong calls (false alarm ratio)</th><th>CSI</th></tr>" + "".join(rows) + "</table>"
            f"<p class='gj-note'>{html.escape(ts['caveat'])} Small numbers under each value: the 95 % range over the "
            f"{ts['windows_pooled']} checked windows.</p>")


# =============================================================================
# Extra layers for the web map
# =============================================================================

def extras_dir(gp, e):
    return gp.event_dir(e.id) / "extras"


def population_path(gp):
    # people added up from WorldPop's own grid (22 Oct 2026); the 05 Data file kept only about 74 % of them
    p = gp.o06 / "jalsar" / "extensions" / "population" / "POP_WorldPop_2020_corridor_100m_sum.tif"
    return p if p.exists() else gp.d05 / "population" / "POP_WorldPop_2020_corridor_100m.tif"


def vulnerability_path(gp):
    return gp.o06 / "jalsar" / "extensions" / "m8_vulnerability" / "vulnerability.geojson"


def forecast_path(gp):
    return gp.o06 / "jalsar" / "extensions" / "m9_forecast" / "forecast_watch.json"


def population_layer(folium, gp, cell_m=100.0):
    """WorldPop people per cell as a 4-class picture, or (None, 0)."""
    p = population_path(gp)
    if not p.exists():
        return None, 0
    with gc.rasterio.open(p) as src:
        a = src.read(1).astype("float64")
        prof = src.profile
        nd = src.nodata
    if nd is not None:
        a[a == nd] = np.nan
    cls = np.full(a.shape, mc.NODATA, "uint8")
    for i in range(len(POP_BINS) - 1):
        cls[(a >= POP_BINS[i]) & (a < POP_BINS[i + 1])] = i + 1
    merc, bounds = gd.to_mercator(cls, prof, cell_m)
    png = gd.palette_png(merc, {k: gd.hex_rgba(c, 170) for k, c in POP_COLOURS.items()})
    fg = folium.FeatureGroup(name="Population (WorldPop 2020)", show=False)
    folium.raster_layers.ImageOverlay(image=gd.data_url(png), bounds=bounds, pixelated=True,
                                      interactive=False, cross_origin=False).add_to(fg)
    return fg, len(png)


def _round_coords(c, nd=5):
    """Coordinates to 5 decimals (about 1 m), which is finer than any input layer here."""
    if isinstance(c, (list, tuple)):
        if c and isinstance(c[0], (int, float)):
            return [round(v, nd) for v in c]
        return [_round_coords(v, nd) for v in c]
    return c


def _tip(p, kind):
    return (p.get("label") or kind.replace("_", " ")) + (
        " | inside the mapped flood" if p.get("flooded") else
        f" | within {p.get('near_m', 0):g} m of the mapped flood")


def _feature_layer(folium, feats, kind, name):
    """All features of one kind as at most two GeoJson blocks (points; lines and shapes).
    One block per kind keeps the page small: a marker per feature, each with its own
    tooltip, made the six-event page 55 MB (22 October 2026)."""
    fg = folium.FeatureGroup(name=name, show=False)
    col = EXPO_COLOURS[kind]
    pts, shapes = [], []
    for f in feats:
        p = f["properties"]
        geom = {"type": f["geometry"]["type"], "coordinates": _round_coords(f["geometry"]["coordinates"])}
        g = {"type": "Feature", "geometry": geom,
             "properties": {"tip": _tip(p, kind), "flooded": bool(p.get("flooded"))}}
        (pts if f["geometry"]["type"] == "Point" else shapes).append(g)

    def tooltip():
        return folium.GeoJsonTooltip(fields=["tip"], labels=False, sticky=True)

    if pts:
        folium.GeoJson({"type": "FeatureCollection", "features": pts},
                       marker=folium.CircleMarker(radius=3, weight=1.5, fill=True, fill_opacity=0.9),
                       style_function=lambda f, c=col: {
                           "color": c, "weight": 1.5, "radius": 4 if f["properties"]["flooded"] else 3,
                           "fillColor": c if f["properties"]["flooded"] else "#ffffff", "fillOpacity": 0.9},
                       tooltip=tooltip()).add_to(fg)
    if shapes:
        folium.GeoJson({"type": "FeatureCollection", "features": shapes},
                       style_function=lambda f, c=col: {
                           "color": c, "weight": 3 if f["properties"]["flooded"] else 1.5, "fillColor": c,
                           "fillOpacity": 0.6 if f["properties"]["flooded"] else 0.15,
                           "opacity": 1 if f["properties"]["flooded"] else 0.6},
                       tooltip=tooltip()).add_to(fg)
    return fg


def exposure(gp, e):
    """(summary dict, feature list) of the event's exposure files, or (None, [])."""
    d = extras_dir(gp, e)
    if not (d / "exposure.json").exists():
        return None, []
    feats = []
    if (d / "exposure_features.geojson").exists():
        feats = json.loads((d / "exposure_features.geojson").read_text(encoding="utf-8")).get("features", [])
    return gd.clean(mc.load_json(d / "exposure.json")), feats


def village_people(gp, e):
    """m8_risk_view.py's per-village WorldPop people and risk classes for the event, or None."""
    p = extras_dir(gp, e) / "village_people.json"
    return gd.clean(mc.load_json(p)) if p.exists() else None


def exposure_rows(xs, roads=None):
    """Rows (what, inside, near, source) of the bulletin's exposure table, or []."""
    if not xs:
        return []
    f0 = lambda v: "n/a" if v is None else f"{v:,.0f}"            # noqa: E731
    f2 = lambda v: "n/a" if v is None else f"{v:,.2f}"            # noqa: E731
    rows = []
    p = xs.get("population") or {}
    if p:
        rows.append({"what": "People (WorldPop 2020)", "inside": f0(p.get("inside")),
                     "near": f"{f0(p.get('inside_buffer'))} with a {p.get('buffer_m', 30):g} m edge",
                     "source": "WorldPop (CC BY 4.0)"})
    b = xs.get("buildings") or {}
    if b:
        rows.append({"what": "Buildings", "inside": f0(b.get("inside")), "near": f0(b.get("near")),
                     "source": "Google Open Buildings v3"})
    if roads:
        rows.append({"what": "Roads (km)", "inside": f2(roads.get("total_flood_km")), "near": "n/a",
                     "source": "OpenStreetMap (ODbL)"})
    pw = xs.get("power") or {}
    if pw:
        rows.append({"what": "Power lines (km)", "inside": f2(pw.get("line_km_inside")), "near": f2(pw.get("line_km_near")),
                     "source": "OpenStreetMap; World Bank derived line map (CC BY 4.0)"})
        if "points_inside" in pw:
            rows.append({"what": "Power poles, pylons and substations", "inside": f0(pw.get("points_inside")),
                         "near": f0(pw.get("points_near")), "source": "OpenStreetMap (ODbL)"})
    t = xs.get("towers") or {}
    if t:
        rows.append({"what": "Mobile and communication towers", "inside": f0(t.get("inside")), "near": f0(t.get("near")),
                     "source": "OpenStreetMap only"})
    return rows


def risk_layer(folium, gp, e):
    """Villages touching the flood, coloured by relative risk class, or None."""
    vp = village_people(gp, e)
    if not vp or not vp.get("villages"):
        return None
    fg = folium.FeatureGroup(name="Risk (relative)", show=False)
    for x in vp["villages"].values():
        col = RISK_COLOURS.get(x.get("risk_class"), "#9ca3af")
        tip = (f"<b>{html.escape(str(x.get('village', '')))}</b> ({html.escape(str(x.get('subdistrict', '')))})<br>"
               f"Risk: {html.escape(str(x.get('risk_class')))} (relative, this corridor only)<br>"
               f"People in the flood: {x.get('worldpop_in_flood') or 0:,.0f}; in the village: "
               f"{x.get('worldpop_village') or 0:,} (WorldPop 2020)<br>"
               f"Vulnerability: {html.escape(str(x.get('vulnerability_name')))}; flood {x.get('flood_km2', 0):.3f} km&sup2;")
        r = 4 + 2.5 * math.log10(1 + (x.get("worldpop_in_flood") or 0))
        folium.CircleMarker([x["lat"], x["lon"]], radius=r, color="#111827", weight=1, fill=True,
                            fill_color=col, fill_opacity=0.9, tooltip=folium.Tooltip(tip, sticky=True)).add_to(fg)
    return fg


def event_extra_layers(folium, gp, e):
    """{key: FeatureGroup} of the event's extension layers that exist."""
    summ, feats = exposure(gp, e)
    out = {}
    rk = risk_layer(folium, gp, e)
    if rk is not None:
        out["risk"] = rk
    if summ is None:
        return out
    groups = {"buildings": ["building"], "power": ["power_line", "power_point"], "towers": ["tower"]}
    names = {"buildings": "Buildings", "power": "Power lines and poles", "towers": "Mobile towers"}
    for key, kinds in groups.items():
        fs = [f for f in feats if f["properties"].get("layer") in kinds]
        if fs:
            out[key] = folium.FeatureGroup(name=names[key], show=False)
            for kind in kinds:
                sub = [f for f in fs if f["properties"].get("layer") == kind]
                if sub:
                    _feature_layer(folium, sub, kind, names[key]).add_to(out[key])
    return out


def vulnerability_layer(folium, gp):
    p = vulnerability_path(gp)
    if not p.exists():
        return None
    gj = json.loads(p.read_text(encoding="utf-8"))
    fg = folium.FeatureGroup(name="Vulnerability index", show=False)

    def style(f):
        c = VULN_COLOURS.get(int(f["properties"].get("class") or 0), "#cccccc")
        return {"color": "#555555", "weight": 0.4, "fillColor": c, "fillOpacity": 0.55}

    fields = [k for k in ("name", "class_name", "score") if gj["features"]
              and k in gj["features"][0]["properties"]]
    folium.GeoJson(gj, style_function=style,
                   tooltip=folium.GeoJsonTooltip(fields=fields) if fields else None).add_to(fg)
    return fg


def load_forecast(gp):
    p = forecast_path(gp)
    return gd.clean(mc.load_json(p)) if p.exists() else None


def forecast_layer(folium, gp, fc):
    """Forecast points coloured by status; the forecast extent picture if it exists."""
    if not fc:
        return None, 0
    fg = folium.FeatureGroup(name="Forecast", show=False)
    size = 0
    ext = fc.get("extent_tif")
    if ext and Path(ext).exists():
        arr, prof = gd.read_u8(ext)
        merc, bounds = gd.to_mercator(arr, prof, 20.0)
        png = gd.palette_png(merc, {1: gd.hex_rgba("#1d4ed8", 150), 2: gd.hex_rgba("#93c5fd", 120)})
        folium.raster_layers.ImageOverlay(image=gd.data_url(png), bounds=bounds, pixelated=True,
                                          interactive=False, cross_origin=False).add_to(fg)
        size += len(png)
    for pt in fc.get("points", []):
        col = FORECAST_COLOURS.get(pt.get("status", "unknown"), "#8a8a8a")
        tip = (f"<b>{html.escape(pt['name'])}</b><br>{html.escape(pt.get('status_text', ''))}"
               f"<br>Forecast issued {html.escape(str(fc.get('issued', '')))}")
        folium.CircleMarker([pt["lat"], pt["lon"]], radius=8, color="#111827", weight=1.5, fill=True,
                            fill_color=col, fill_opacity=0.95,
                            tooltip=folium.Tooltip(tip, sticky=True)).add_to(fg)
    return fg, size


def valley_strip_layer(folium, gp, fc):
    """The open valley channel strip, grey and hatched, off by default. Not a forecast."""
    p = (fc or {}).get("valley_strip_tif")
    if not p or not Path(p).exists():
        return None, 0
    h = fc.get("valley_strip_h_m", 0.05)
    fg = folium.FeatureGroup(name=f"Valley: channel strip (HAND {h:.2f} m)", show=False)
    arr, prof = gd.read_u8(p)
    merc, bounds = gd.to_mercator(arr, prof, 20.0)
    ii, jj = np.indices(merc.shape)
    hatch = ((ii + jj) // 2) % 3 == 0                 # diagonal stripes, so it never reads as a flood colour
    pic = np.where(merc == 1, np.where(hatch, 1, 2), 0).astype("uint8")
    png = gd.palette_png(pic, {1: gd.hex_rgba("#374151", 210), 2: gd.hex_rgba("#9ca3af", 110)})
    folium.raster_layers.ImageOverlay(image=gd.data_url(png), bounds=bounds, pixelated=True,
                                      interactive=False, cross_origin=False).add_to(fg)
    return fg, len(png)


def forecast_html(fc):
    """The forecast block of the side panel (a small table per point), or ''."""
    if not fc:
        return ""
    rows = []
    for pt in fc.get("points", []):
        col = FORECAST_COLOURS.get(pt.get("status", "unknown"), "#8a8a8a")
        days = "".join(
            f"<td title='{html.escape(str(d.get('date')))}: {d.get('q50', 'n/a')} m3/s'"
            f" style='background:{FORECAST_COLOURS.get(d.get('status', 'unknown'), '#8a8a8a')}'></td>"
            for d in pt.get("days", []))
        rows.append(f"<tr><th><span class='sw' style='background:{col}'></span>{html.escape(pt['name'])}</th>"
                    f"{days}</tr><tr><td colspan='99' class='mu'>{html.escape(pt.get('status_text', ''))}"
                    "</td></tr>")
    lab = f" ({html.escape(fc['issued_label'])})" if fc.get("issued_label") else ""
    return (f"<p class='gj-note'>GloFAS river forecast issued <b>{html.escape(str(fc.get('issued', '')))}</b>{lab}; "
            f"one square per day ahead (green: below the flood thresholds; orange: watch; red: warning). "
            f"{html.escape(fc.get('method_short', ''))}</p><table class='fc'>{''.join(rows)}</table>"
            f"<p class='gj-note'>{html.escape(fc.get('limits', ''))}</p>")


def extra_items():
    """(key, name, group, what, why, legend html) of the extra layers, for the panel."""
    pop_lg = "".join(f'<span class="li"><span class="sw" style="background:{c}"></span>'
                     f'{["1 to 5", "5 to 20", "20 to 50", "50 or more"][k - 1]}</span>'
                     for k, c in POP_COLOURS.items())
    vul_lg = "".join(f'<span class="li"><span class="sw" style="background:{c}"></span>'
                     f'{["lowest", "low", "middle", "high", "highest"][k - 1]}</span>'
                     for k, c in VULN_COLOURS.items())

    def pt(c, label, filled=True):
        return (f'<span class="li"><span class="sw" style="background:{c if filled else "#fff"};'
                f'border-color:{c};border-radius:50%"></span>{label}</span>')

    return [
        ("population", "Population", "Exposure",
         "People living in each 100 m square (WorldPop 2020, a modelled estimate, not a count).",
         "Shows where people live, so the flood can be read as people affected, not only area.",
         pop_lg + '<span class="li">people per 100 m square</span>'),
        ("buildings", "Buildings", "Exposure",
         "Building outlines inside the mapped flood (solid) or within a short distance of it (pale).",
         "Each one is a home, shop or school the water may have reached.",
         pt(EXPO_COLOURS["building"], "inside the flood") + pt(EXPO_COLOURS["building"], "near it", False)),
        ("power", "Power lines and poles", "Exposure",
         "Electricity lines and poles crossing or near the mapped flood (OpenStreetMap and the World "
         "Bank's map of lines).",
         "A flooded line or pole can cut power to villages beyond the flood itself.",
         pt(EXPO_COLOURS["power_line"], "inside the flood") + pt(EXPO_COLOURS["power_line"], "near it", False)),
        ("towers", "Mobile towers", "Exposure",
         "Mobile and communication towers inside or near the mapped flood, from OpenStreetMap "
         "only. The public OpenCelliD copy holds no Indian cells, so towers nobody has mapped are "
         "missing.",
         "Towers keep phones working for rescue and warnings.",
         pt(EXPO_COLOURS["tower"], "inside the flood") + pt(EXPO_COLOURS["tower"], "near it", False)),
        ("vulnerability", "Vulnerability index", "Vulnerability",
         "How badly people in each area would be affected, from a principal component analysis of "
         "WorldPop indicators (see the report for the indicators and their weights).",
         "Flood plus exposure plus vulnerability is flood risk, in the UN's (UNDRR) terms.", vul_lg),
        ("risk", "Risk (relative)", "Vulnerability",
         "Each village touching the mapped flood, coloured by a relative risk class: it is flooded, how many "
         "people WorldPop puts on its flooded cells, and its vulnerability class, combined by a simple matrix "
         "(see the report). A bigger circle is more people in the flood.",
         "Flood, exposure and vulnerability together are risk in the UN's (UNDRR) terms. The classes rank "
         "villages in this corridor against each other; they are not a probability of loss.",
         "".join(pt(RISK_COLOURS[k], k) for k in ("high", "medium", "low"))
         + pt(RISK_COLOURS["no vulnerability class"], "no vulnerability class")),
        ("forecast", "Forecast", "Forecast",
         "The GloFAS river forecast at Rudraprayag, Srinagar and Devprayag for the coming days, and "
         "where it points to flooding.",
         "An early look ahead, from published forecasts, before any satellite pass.",
         "".join(pt(FORECAST_COLOURS[k], k) for k in ("none", "watch", "warning"))
         + '<span class="li"><span class="sw" style="background:#1d4ed8"></span>where this forecast would '
           'put water (gorge above Devprayag only)</span><span class="li"><span class="sw" '
           'style="background:#93c5fd"></span>flooded in a past radar map</span>'),
        ("valley_strip", "Valley: channel strip (HAND 0.05 m)", "Forecast",
         "Not a flood forecast. The open valley below Devprayag, only the cells less than 0.05 m above the "
         "nearest stream: the narrow strip along the channel where every radar map found its valley flood. "
         "It is the same whatever the forecast says.",
         "In the valley no forecast could be turned into a flood extent: on all six radar maps the best "
         "height-above-stream threshold was the smallest one tried, so the forecast extent is drawn in the "
         "gorge only. This layer shows where valley flooding has been seen, for context.",
         '<span class="li"><span class="sw" style="background:repeating-linear-gradient(45deg,#374151 0 2px,'
         '#d1d5db 2px 5px)"></span>channel strip, not a forecast</span>'),
    ]


# =============================================================================
# Self-test
# =============================================================================

def selftest():
    print("jalsar_extras.py self-test")
    import io
    import contextlib
    import jalsar_testdata as td_mod
    with tempfile.TemporaryDirectory(prefix="jalsar extras ", ignore_cleanup_errors=True) as td:
        with contextlib.redirect_stdout(io.StringIO()):
            gp = td_mod.build_display_project(Path(td) / "Final Year Project")
        ts = compute_trust_scores(gp)
        ok_events = ts is not None and len(ts["events"]) >= 1
        pooled = ts["pooled"] if ts else {}
        # The per-class counts must add up to the all-cells counts.
        add = all(sum((pooled.get(gc.TRUST_NAMES[k]) or {}).get(x, 0) for k in TRUST_ORDER)
                  == (pooled.get("all") or {}).get(x) for x in ("tp", "fp", "fn")) if ts else False
        blind0 = ((pooled.get("radar blind") or {}).get("tp", 0) == 0) if ts else False
        files = trust_scores_path(gp).exists() and (gp.gj / "trust_scores.md").exists()
        plain = trust_plain_numbers(ts)
        html_ok = "Found (hit rate)" in trust_table_html(ts)
    checks = [("trust scores made for the validated events", ok_events),
              ("class counts add up to the all-cells counts", add),
              ("radar blind cells hold no hits (never mapped)", blind0),
              ("json and md written", files), ("plain sentences made", bool(plain)),
              ("panel table made", html_ok)]
    for n, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}")
    allok = all(ok for _, ok in checks)
    print("SELFTEST PASS" if allok else "SELFTEST FAIL")
    return 0 if allok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    gp = gc.GPaths()
    r = compute_trust_scores(gp)
    print("no validated event with a trust layer yet" if r is None else
          f"wrote {trust_scores_path(gp)} and trust_scores.md")
