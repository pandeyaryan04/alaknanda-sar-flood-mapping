r"""
jalsar_bulletin.py
==================
Jal-SAR's flood bulletin: for one event, an interactive HTML file and a
PDF, with a dark maroon header.

Contents: header (date, orbit, scene); cards; a plain-English summary; a map
of the headline flood coloured by trust over a hillshade, with pan and zoom,
and close-ups of the three places with the most mapped flood; flood area by
reach and trust; model agreement; villages (searchable, paged); roads; the
accuracy measured against the reference maps; how the map is made and its
limits.

The PDF is printed by Microsoft Edge in headless mode (it comes with Windows
11, so nothing is installed) and the footer "Jal-SAR . Aryan Pandey .
G. B. Pant University ... Page X of Y" is added with pypdf. Edge is found
on its own; to use another Chromium browser set JALSAR_BROWSER to its path.

All wording lives in templates\bulletin.html.j2.

USAGE  (Anaconda Prompt, fyp active, from 04 Code\jalsar)
-----
    python jalsar.py bulletin --event all
    python jalsar.py bulletin --event 2023-08-16 --force
    python jalsar.py bulletin --event all --no-pdf
    python jalsar_bulletin.py --selftest

Writes 06 Outputs\jalsar\<event>\bulletin_<event>.html and .pdf. An
existing bulletin is kept unless --force. About 1 minute per event (estimate).
"""

__stage__ = "Jal-SAR: bulletin (HTML and PDF)"
__reads__ = ["06 Outputs/jalsar/<event>/ (summary.json and rasters)",
             "05 Data/dem/DEM_COP_GLO30_30m.tif (hillshade)", "FYP_code/data/corridor.geojson"]
__writes__ = ["06 Outputs/jalsar/<event>/bulletin_<event>.html",
              "06 Outputs/jalsar/<event>/bulletin_<event>.pdf"]

import io
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import numpy as np

import jalsar_core as gc
import jalsar_display as gd

mc = gc.mc
HERE = Path(__file__).resolve().parent
TEMPLATES = HERE / "templates"
INSET_PX = 300            # close-ups are 3 km squares (300 cells of 10 m)
SPOT_PX = 100             # overview circles summarise 1 km squares
SPOT_MIN_CELLS = 10       # a 1 km square needs 0.001 km2 of flood to get a circle
DIM_NAMES = {"incidence": "Local incidence angle", "hand": "Height above nearest drainage",
             "valley_width": "Valley width"}
REACH_ROWS = [(gd.POOLED, "Both reaches"),
              ("gorge_above_devprayag", "Gorge above Devprayag"),
              ("open_valley_below_devprayag", "Open valley below Devprayag")]
EDGE_PATHS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]


def bulletin_paths(gp, event):
    d = gp.event_dir(event.id)
    return d / f"bulletin_{event.id}.html", d / f"bulletin_{event.id}.pdf"


# =============================================================================
# Small helpers
# =============================================================================

def class_label(dim, s):
    """'lt30' -> 'under 30 degrees', '5_15m' -> '5 to 15 m' and so on."""
    unit = " degrees" if dim == "incidence" else " m"
    t = s[:-1] if s.endswith("m") else s
    if t.startswith("lt"):
        return f"under {t[2:]}{unit}"
    if t.startswith("gt"):
        return f"over {t[2:]}{unit}"
    if "_" in t:
        a, b = t.split("_", 1)
        return f"{a} to {b}{unit}"
    return s


def _order(dim, s):
    t = s[:-1] if s.endswith("m") else s
    try:
        return -1 if t.startswith("lt") else 1e9 if t.startswith("gt") else float(t.split("_")[0])
    except ValueError:
        return 0


def block_reduce(a, f, fn):
    """Reduce an array over f x f blocks (padded with 0)."""
    h, w = a.shape
    H, W = math.ceil(h / f) * f, math.ceil(w / f) * f
    p = np.zeros((H, W), a.dtype)
    p[:h, :w] = a
    return fn(p.reshape(H // f, f, W // f, f), axis=(1, 3))


def long_date(s):
    d = date.fromisoformat(s)
    return f"{d.day} {d.strftime('%B %Y')}"


def _crop(a, r0, c0, n, fill):
    out = np.full((n, n), fill, a.dtype)
    sub = a[max(r0, 0):r0 + n, max(c0, 0):c0 + n]
    out[max(-r0, 0):max(-r0, 0) + sub.shape[0], max(-c0, 0):max(-c0, 0) + sub.shape[1]] = sub
    return out


# =============================================================================
# The map figure
# =============================================================================

def corridor_path(gp, profile, tol=30.0):
    import geopandas as gpd
    if not gp.corridor.exists():
        return ""
    g = gpd.read_file(gp.corridor).to_crs(profile["crs"]).geometry.simplify(tol)
    t = profile["transform"]
    parts = []
    for geom in g:
        for poly in getattr(geom, "geoms", [geom]):
            for ring in [poly.exterior] + list(poly.interiors):
                pts = [f"{(x - t.c) / t.a:.1f},{(y - t.f) / t.e:.1f}" for x, y in ring.coords]
                parts.append("M" + " L".join(pts) + " Z")
    return " ".join(parts)


def spots(hb, km2):
    """1 km squares with mapped flood: centre, area and the trust of most of it."""
    f = hb != mc.NODATA
    n = block_reduce(f.astype("int32"), SPOT_PX, np.sum)
    counts = {k: block_reduce((hb == k).astype("int32"), SPOT_PX, np.sum) for k in (1, 2, 3)}
    out = []
    for r, c in zip(*np.nonzero(n >= SPOT_MIN_CELLS)):
        k = max((3, 2, 1), key=lambda k: counts[k][r, c])
        out.append({"row": int(r), "col": int(c), "cells": int(n[r, c]),
                    "km2": float(n[r, c] * km2), "trust": gc.TRUST_NAMES[k],
                    "col_hex": gd.TRUST_COLOURS[k], "high_km2": float(counts[3][r, c] * km2)})
    return sorted(out, key=lambda s: -s["cells"])


def pick_insets(sp, shape, k=3):
    """Up to k non-overlapping 3 km squares around the 1 km squares with most flood."""
    chosen = []
    for s in sp:
        cy, cx = s["row"] * SPOT_PX + SPOT_PX // 2, s["col"] * SPOT_PX + SPOT_PX // 2
        r0 = int(min(max(cy - INSET_PX // 2, 0), max(shape[0] - INSET_PX, 0)))
        c0 = int(min(max(cx - INSET_PX // 2, 0), max(shape[1] - INSET_PX, 0)))
        if any(abs(r0 - a) < INSET_PX * 1.5 and abs(c0 - b) < INSET_PX * 1.5 for a, b in chosen):
            continue
        chosen.append((r0, c0))
        if len(chosen) == k:
            break
    return chosen


def map_figure(gp, event, summ, flood, trust, prof):
    from rasterio.warp import transform as wt
    km2 = mc.cell_km2(prof)
    H, W = flood.shape
    hb = gd.headline_by_trust(flood, trust)
    tcol = {k: gd.hex_rgba(gd.TRUST_COLOURS[k]) for k in (1, 2, 3)}
    blind_col = {1: gd.hex_rgba(gd.TRUST_COLOURS[0], 120)}
    blind = (trust == 0).astype("uint8")
    blind3 = block_reduce(blind, 3, np.max)
    hs, hs_prof = gd.hillshade(gp.d05 / "dem" / "DEM_COP_GLO30_30m.tif", prof, factor=3)
    stroke = max(W, H) / 600
    font = max(W, H) / 70
    t = prof["transform"]
    split = summ.get("reach_split_x")
    split_px = (split - t.c) / t.a if split is not None else None
    if split_px is not None and not (0 < split_px < W):
        split_px = None
    width_km = W * abs(t.a) / 1000
    scale_km = next(v for v in (50, 20, 10, 5, 2, 1, 0.5, 0.2, 0.1) if v <= width_km / 5)
    sp = spots(hb, km2)
    vill = []
    vrows = ((summ.get("villages") or {}).get("villages") or [])
    if vrows:
        xs, ys = wt("EPSG:4326", prof["crs"], [v["lon"] for v in vrows], [v["lat"] for v in vrows])
        for v, x, y in zip(vrows, xs, ys):
            vill.append({"x": round((x - t.c) / t.a, 1), "y": round((y - t.f) / t.e, 1),
                         "r": round(stroke * (1.2 + 3 * min(math.sqrt(v["flood_km2"] / 0.1), 1)), 1),
                         "name": v["village"], "sub": v["subdistrict"],
                         "flood": f"{v['flood_km2']:.3f}",
                         "share": f"{v['share_flooded'] * 100:.1f} %"
                         if v.get("share_flooded") is not None else "n/a",
                         "pop": gd.num(v["population_2011"], 0) if v["population_2011"] is not None
                         else "no census figure"})
    insets, boxes = [], []
    for key, (r0, c0) in zip("ABC", pick_insets(sp, (H, W))):
        sub = _crop(hb, r0, c0, INSET_PX, mc.NODATA)
        fl = sub != mc.NODATA
        a = float(fl.sum() * km2)
        hi = float((sub == 3).sum() * km2)
        near = ""
        if vill:
            cx, cy = c0 + INSET_PX / 2, r0 + INSET_PX / 2
            dist = [(math.hypot(v["x"] - cx, v["y"] - cy), v["name"]) for v in vill]
            dmin, name = min(dist)
            if dmin < INSET_PX:
                near = f"; near {name}"
        item = {"key": key, "w": INSET_PX, "h": INSET_PX,
                "flood": gd.data_url(gd.palette_png(sub, tcol)),
                "blind": gd.data_url(gd.palette_png(_crop(blind, r0, c0, INSET_PX, 0), blind_col)),
                "hillshade": None,
                "caption": f"{a:.3f} km\u00b2 of mapped flood in this 3 km by 3 km square, {hi:.3f} km\u00b2 "
                           f"of it high trust{near}."}
        if hs is not None:
            item["hillshade"] = gd.data_url(gd.jpeg(_crop(hs, r0 // 3, c0 // 3, INSET_PX // 3, 128)),
                                            "image/jpeg")
        insets.append(item)
        boxes.append({"key": key, "x": c0, "y": r0, "w": INSET_PX, "h": INSET_PX})
    fig = {"w": W, "h": H, "stroke": round(stroke, 2), "font": round(font, 1),
           "hillshade": gd.data_url(gd.jpeg(hs), "image/jpeg") if hs is not None else None,
           "flood": gd.data_url(gd.palette_png(hb, tcol)),
           "blind": gd.data_url(gd.palette_png(blind3, blind_col)),
           "corridor": corridor_path(gp, prof),
           "split_x": round(split_px, 1) if split_px is not None else None,
           "spots": [{"x": s["col"] * SPOT_PX + SPOT_PX / 2, "y": s["row"] * SPOT_PX + SPOT_PX / 2,
                      "r": round(stroke * (1.5 + 6 * math.sqrt(s["km2"])), 1), "km2": s["km2"],
                      "trust": s["trust"], "col": s["col_hex"]} for s in sp],
           "villages": vill, "insets": boxes,
           "scale_km": scale_km if scale_km >= 1 else scale_km,
           "scale_px": scale_km * 1000 / abs(t.a)}
    return fig, insets


# =============================================================================
# The bulletin's numbers
# =============================================================================

def context(gp, event, summ, print_mode=False):
    s = gd.clean(summ)
    d = gp.event_dir(event.id)
    flood, prof = gd.read_u8(d / "flood_headline.tif")
    trust, _ = gd.read_u8(d / "trust.tif")
    fig, insets = map_figure(gp, event, s, flood, trust, prof)
    a = s["areas"]
    head = s["headline"]
    fam = head["family"]
    tot = a["total_km2"]
    v = s.get("villages")
    roads = s.get("roads")
    val = s.get("validation")
    reach_rows = []
    for r in gc.REACHES:
        bt = a["by_reach_trust"][r]
        reach_rows.append({"name": gc.REACH_NAMES[r], "high": gd.num(bt["high"]),
                           "med": gd.num(bt["medium"]), "low": gd.num(bt["low"]),
                           "total": gd.num(a["by_reach"][r]), "total_row": False})
    reach_rows.append({"name": "Whole corridor", "high": gd.num(a["by_trust"]["high"]),
                       "med": gd.num(a["by_trust"]["medium"]), "low": gd.num(a["by_trust"]["low"]),
                       "total": gd.num(tot), "total_row": True})
    nf = len(s["families"])
    agree_rows = [{"label": ("both models" if int(k) == nf == 2 else
                             f"all {nf} models" if int(k) == nf and nf > 1 else
                             f"{k} model" + ("s" if int(k) > 1 else "")),
                   "km2": gd.num(km)} for k, km in sorted(s["agreement_km2"].items())]
    family_rows = [{"name": f["name"], "trainset": f["trainset"].replace("_", " + "),
                    "threshold": gd.num(f["threshold"], 2), "flood": gd.num(f["flood_km2"]),
                    "perm": gd.num(f["permanent_removed_km2"]),
                    "distort": gd.num(f["distortion_removed_km2"]), "headline": k == fam}
                   for k, f in s["families"].items()]
    road_rows = []
    if roads:
        for cls, km in roads["by_class_km"].items():
            road_rows.append({"cls": cls, "flood": gd.num(km),
                              "total": gd.num(roads["corridor_km_by_class"].get(cls))})
    val_rows, n_windows = [], 0
    if val:
        for f in s["families"]:
            for key, label in (("table_a", "A"), ("table_d", "D")):
                rows = val[key].get(f) or {}
                for rk, rname in REACH_ROWS:
                    if rk in rows:
                        r = rows[rk]
                        val_rows.append({"model": s["families"][f]["name"], "table": label,
                                         "reach": rname, "f1": gd.f1_text(r),
                                         "p": gd.num(r.get("precision")), "r": gd.num(r.get("recall")),
                                         "headline": f == fam})
        wp = gp.windows(val["year"])
        if wp.exists():
            n_windows = len(mc.load_json(wp).get("features", []))
    nw = {y: len(mc.load_json(gp.windows(y)).get("features", [])) for y in mc.YEARS
          if gp.windows(y).exists()}
    trust_basis = (f"{len(nw)} event{'s' if len(nw) != 1 else ''} and {sum(nw.values())} reference "
                   "windows of this corridor" if nw else "the reference windows of this corridor")
    score_rows = []
    for f, sc in (head.get("scores") or {}).items():
        score_rows.append({"name": gc.FAMILY_NAMES[f], "headline": f == fam,
                           "f2021": gd.num(sc["table_d_f1"].get("2021")),
                           "f2023": gd.num(sc["table_d_f1"].get("2023")),
                           "mean": gd.num(sc["table_d_f1_mean"])})
    trust_rows = []
    for dim in gc.TRUST_DIMS:
        tbl = (s["trust"]["table_d_precision"] or {}).get(dim) or {}
        for cls in sorted(tbl, key=lambda c: _order(dim, c)):
            p = tbl[cls]
            g = gc.trust_grade(float("nan") if p is None else p)
            trust_rows.append({"dim": DIM_NAMES[dim], "cls": class_label(dim, cls),
                               "p": gd.num(p), "grade": gc.TRUST_NAMES[g],
                               "col": gd.TRUST_COLOURS[g]})

    def pooled(key):
        return gd.f1_text(((val or {}).get(key, {}).get(fam) or {}).get(gd.POOLED))

    mean_d = (head.get("scores") or {}).get(fam, {}).get("table_d_f1_mean")
    fmt = {"area_total": gd.num(tot, 2), "area_high": gd.num(a["by_trust"]["high"], 2),
           "area_med": gd.num(a["by_trust"]["medium"], 2), "area_low": gd.num(a["by_trust"]["low"], 2),
           "high_pct": f"{a['by_trust']['high'] / tot * 100:.0f} %" if tot > 0 else "n/a",
           "n_villages": gd.num(v["count"], 0) if v else "n/a",
           "n_villages_min": gd.num(v["count_min_area"], 0) if v and "count_min_area" in v
           else "n/a",
           "population_min": gd.num(v.get("population_min_area"), 0) if v else "n/a",
           "min_km2": gd.num((v or {}).get("min_flood_km2"), 2),
           "min_ha": gd.num((v or {}).get("min_flood_km2", 0) * 100, 0) if v else "n/a",
           "population": gd.num(v["population_2011"], 0) if v else "n/a",
           "households": gd.num(v["households_2011"], 0) if v else "n/a",
           "road_km": gd.num(roads["total_flood_km"], 2) if roads else "n/a",
           "blind": gd.num(s["trust"]["radar_blind_km2"], 2),
           "f1_a": pooled("table_a"), "f1_d": pooled("table_d"),
           "f1_mean_d": gd.num(mean_d) if mean_d is not None else "n/a (Table D not ready)"}
    ev = dict(s["event"])
    ev["date_long"] = long_date(event.date)
    rule = head.get("provisional") or head.get("rule", "")
    return {"event": ev, "print_mode": print_mode, "version": s.get("jalsar_version", gc.VERSION),
            "made": long_date(time.strftime("%Y-%m-%d")),
            "head": {"family": fam, "name": head["name"], "rule": rule},
            "c": {"maroon": gd.MAROON, "trust": gd.TRUST_COLOURS},
            "fmt": fmt, "map": fig, "insets": insets, "reach_rows": reach_rows,
            "agree_rows": agree_rows, "family_rows": family_rows, "villages": v, "roads": roads,
            "road_rows": road_rows, "validation": val, "val_rows": val_rows,
            "n_windows": n_windows or "several", "score_rows": score_rows,
            "trust_rows": trust_rows, "trust_basis": trust_basis, "trust_plain": gd.TRUST_PLAIN,
            "not_this": gd.NOT_THIS,
            "credits": gd.CREDITS, "footer": gd.FOOTER, "files": s.get("files", [])}


def render(ctx):
    import jinja2
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(TEMPLATES)), autoescape=True,
                             undefined=jinja2.StrictUndefined, trim_blocks=True, lstrip_blocks=True)
    return env.get_template("bulletin.html.j2").render(b=ctx)


# =============================================================================
# PDF
# =============================================================================

def find_browser():
    """Microsoft Edge (or JALSAR_BROWSER, or another Chromium browser), or None."""
    env = os.environ.get("JALSAR_BROWSER")
    cands = [env] if env else []
    cands += EDGE_PATHS
    local = os.environ.get("LOCALAPPDATA")
    if local:
        cands.append(str(Path(local) / "Microsoft" / "Edge" / "Application" / "msedge.exe"))
    cands += [shutil.which(n) for n in ("msedge", "microsoft-edge", "chromium", "chromium-browser",
                                        "google-chrome", "chrome")]
    for c in cands:
        if c and Path(c).is_file():
            return c
    return None


def print_pdf(html_path, pdf_path, browser, timeout=240):
    """Headless print of a local HTML file to PDF, with a throw-away browser profile."""
    html_path, pdf_path = Path(html_path), Path(pdf_path)
    with tempfile.TemporaryDirectory(prefix="jalsar browser ", ignore_cleanup_errors=True) as prof:
        cmd = [browser, "--headless", "--disable-gpu", "--no-first-run",
               "--no-default-browser-check", "--disable-extensions", f"--user-data-dir={prof}",
               "--no-pdf-header-footer", "--print-to-pdf-no-header",
               "--virtual-time-budget=20000", f"--print-to-pdf={pdf_path}",
               html_path.resolve().as_uri()]
        if os.name != "nt" and hasattr(os, "geteuid") and os.geteuid() == 0:
            cmd.insert(1, "--no-sandbox")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, f"the browser did not finish in {timeout} s"
    if not pdf_path.exists() or pdf_path.stat().st_size < 1000:
        return False, (r.stderr or r.stdout or "no PDF written").strip()[-600:]
    return True, ""


def add_footer(src, dst, left=gd.FOOTER):
    """'Jal-SAR . name . university' and 'Page X of Y' on every page (pypdf + matplotlib)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import logging
    from pypdf import PdfReader, PdfWriter
    # Edge's PDFs carry a trailer one object short; pypdf reads them fine but warns
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    rd = PdfReader(str(src))
    n = len(rd.pages)
    buf = io.BytesIO()
    with PdfPages(buf) as pp:
        for i, page in enumerate(rd.pages):
            w, h = float(page.mediabox.width), float(page.mediabox.height)
            fig = plt.figure(figsize=(w / 72, h / 72))
            fig.patch.set_visible(False)
            mm_x, mm_y = 72 / 25.4 / w, 72 / 25.4 / h
            fig.add_artist(plt.Line2D([12 * mm_x, 1 - 12 * mm_x], [11.5 * mm_y] * 2,
                                      color=gd.LINE, linewidth=0.6))
            fig.text(12 * mm_x, 7.5 * mm_y, left, fontsize=7.2, color=gd.MUTED, ha="left")
            fig.text(1 - 12 * mm_x, 7.5 * mm_y, f"Page {i + 1} of {n}", fontsize=7.2,
                     color=gd.MUTED, ha="right")
            pp.savefig(fig, transparent=True)
            plt.close(fig)
    over = PdfReader(io.BytesIO(buf.getvalue()))
    wr = PdfWriter(clone_from=str(src))
    for page, ov in zip(wr.pages, over.pages):
        page.merge_page(ov)
    wr.add_metadata({"/Title": "Jal-SAR flood bulletin", "/Author": gd.AUTHOR,
                     "/Creator": f"Jal-SAR {gc.VERSION}"})
    with open(dst, "wb") as fh:
        wr.write(fh)
    return n


# =============================================================================
# One event
# =============================================================================

def make_bulletin(gp, event, force=False, pdf=True):
    html_p, pdf_p = bulletin_paths(gp, event)
    summ_p = gp.event_dir(event.id) / "summary.json"
    if not summ_p.exists():
        mc.fail(f"{event.id} has no maps yet. Run 'python jalsar.py maps --event {event.date}' first.")
    want_pdf = pdf and (force or not pdf_p.exists())
    if html_p.exists() and not force and not want_pdf:
        print(f"  {event.id}: bulletin already there, kept ({html_p.name}); --force makes it again")
        return {"html": str(html_p), "pdf": str(pdf_p) if pdf_p.exists() else None, "kept": True}
    summ = mc.load_json(summ_p)
    t0 = time.time()
    out = {"html": str(html_p), "pdf": None}
    if force or not html_p.exists():
        html_p.write_text(render(context(gp, event, summ)), encoding="utf-8")
        print(f"  {event.id}: wrote {html_p.name} ({html_p.stat().st_size / 1e6:.1f} MB)", flush=True)
    if want_pdf:
        browser = find_browser()
        if browser is None:
            print("  PROBLEM: Microsoft Edge was not found, so no PDF was made (the HTML is fine).\n"
                  "  Set JALSAR_BROWSER to the full path of msedge.exe or chrome.exe and run again.")
            out["pdf_problem"] = "no browser"
            return out
        tmp_html = html_p.with_name(html_p.stem + "_print.html")
        raw = pdf_p.with_name(pdf_p.stem + "_raw.pdf")
        tmp_html.write_text(render(context(gp, event, summ, print_mode=True)), encoding="utf-8")
        try:
            ok, why = print_pdf(tmp_html, raw, browser)
            if not ok:
                print(f"  PROBLEM: the PDF was not printed ({why}). The HTML is fine.")
                out["pdf_problem"] = why
                return out
            n = add_footer(raw, pdf_p)
        finally:
            for p in (tmp_html, raw):
                if p.exists():
                    p.unlink()
        out["pdf"] = str(pdf_p)
        out["pages"] = n
        print(f"  {event.id}: wrote {pdf_p.name} ({n} pages) with {Path(browser).name}", flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def selftest():
    print("jalsar_bulletin.py self-test")
    import contextlib
    import re
    import jalsar_testdata as td_mod
    from pypdf import PdfReader
    with tempfile.TemporaryDirectory(prefix="jalsar bulletin ", ignore_cleanup_errors=True) as td:
        with contextlib.redirect_stdout(io.StringIO()):
            gp = td_mod.build_display_project(Path(td) / "Final Year Project")
        evs = gc.load_events(gp)
        res = {e.id: make_bulletin(gp, e, pdf=True) for e in evs}
        h21 = Path(res["2021-10-21_orbit063"]["html"]).read_text(encoding="utf-8")
        hjul = Path(res["2023-07-19_orbit063"]["html"]).read_text(encoding="utf-8")
        s21 = mc.load_json(gp.event_dir("2021-10-21_orbit063") / "summary.json")
        html_ok = all(x in h21 for x in ("Jal-SAR", gd.MAROON, "Census 2011", "Gaon",
                                          "Table A", "OpenStreetMap contributors",
                                          "What this bulletin is not", "gjZoom"))
        nums_ok = f"{s21['areas']['total_km2']:.2f} km&sup2;" in h21
        jul_ok = "There is no reference map for this date" in hjul and "Table D" in hjul
        no_nan = not re.search(r"\bnan\b|\bNaN\b|None", h21.split("<script>")[0]
                               .replace("no census figure", ""))
        no_dash = "\u2014" not in h21 and "\u2013" not in h21
        v21 = s21["villages"]
        gj6_ok = (f"{v21['count_min_area']} of them have at least 1 ha" in h21
                  and "share of the whole village area" in h21
                  and "before the networks' Table D results existed" in h21
                  and "3 km by 3 km square" in h21 and "largest flood area first" in h21)
        browser = find_browser()
        pdf_ok = text_ok = True
        if browser:
            p = res["2021-10-21_orbit063"].get("pdf")
            pdf_ok = bool(p) and Path(p).exists()
            if pdf_ok:
                rd = PdfReader(p)
                txt = rd.pages[0].extract_text() + rd.pages[-1].extract_text()
                n = len(rd.pages)
                text_ok = (f"Page 1 of {n}" in txt and f"Page {n} of {n}" in txt
                           and "Aryan Pandey" in txt and "Jal-SAR" in txt)
        kept = make_bulletin(gp, evs[0]).get("kept") is True
    checks = [("HTML: wordmark, maroon, census, validation, roads, limits, pan and zoom", html_ok),
              ("summary area matches summary.json", nums_ok),
              ("wording: 1 ha count, whole-village share, Table D date, 3 km by 3 km",
               gj6_ok),
              ("radar-only event says it has no reference", jul_ok),
              ("no NaN or None printed", no_nan), ("no long dashes", no_dash),
              (f"PDF printed ({'browser found' if browser else 'no browser here, skipped'})", pdf_ok),
              ("PDF footer: name, university, Page X of Y", text_ok),
              ("kept without --force", kept)]
    for name, c in checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {name}")
    good = all(c for _, c in checks)
    print("SELFTEST PASS" if good else "SELFTEST FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    mc.setup_console()
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print(__doc__)
