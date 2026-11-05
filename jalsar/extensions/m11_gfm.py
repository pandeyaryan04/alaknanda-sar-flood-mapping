"""Jal-SAR extensions, item M11: Jal-SAR's maps against the Copernicus GFM flood extent.

GFM (Global Flood Monitoring, Copernicus Emergency Management Service) maps flood water on every
Sentinel-1 pass, worldwide, from 2015, with three algorithms combined ("ensemble"). It is another
model, not a surveyed reference, so this is agreement, not accuracy. For 2021 and 2023, where the
project has its hand-drawn reference, GFM itself is also scored against that reference with the
same scorer and window bootstrap as Table D, beside the three models on the same footprint.

The data come from the EODC STAC catalogue (https://stac.eodc.eu/api/v1, collection GFM), no login.
Tiles are Equi7Grid, 20 m. Each tile is put onto the event's own 10 m grid by nearest neighbour.

Commands:
  probe   one search (default: the corridor, 16 August 2023), prints and writes what the catalogue
          returns: item properties, asset names, and the pixel values of one flood-extent tile.
          Run this first. If the asset names or values differ from what run expects, run stops
          with a message rather than guess.
  run     every Jal-SAR event with model maps (the corridor events and, if m11_beas.py has run, the
          Beas), plus the 2021 and 2023 reference scoring

GFM pixel values assumed (checked by probe and again by run): flood extent 0 no flood, 1 flood,
255 no data; exclusion mask 1 = GFM does not classify here (radar shadow, layover, steep or
unsuitable land), 0 = classified. Excluded cells count as "no flood" in GFM, the same way Jal-SAR's
radar-blind cells are taken out of its maps.

Outputs in 06 Outputs\\jalsar\\extensions\\m11_gfm\\:
  GFM_PROBE.md, gfm_<event>.tif (0, 1, 255 on the event grid), agreement.csv, reference_scores.csv,
  fig_<event>.png, GFM_COMPARISON.md, tiles\\ (downloaded GFM tiles, reused on a re-run)

Run:  python m11_gfm.py probe
      python m11_gfm.py probe --date 2023-07-11 --geojson "..\\..\\..\\06 Outputs\\jalsar\\extensions\\m11_beas\\beas_valley.geojson"
      python m11_gfm.py run
"""
import argparse
import csv
import json
import sys
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "FYP_code"))
import jalsar_core as gc  # noqa: E402

mc = gc.mc
rasterio = gc.rasterio
STAC = "https://stac.eodc.eu/api/v1"
COLLECTION = "GFM"
FLOOD_ASSETS = ("ensemble_flood_extent", "ensemble_flood", "flood_extent")
EXCL_ASSETS = ("exclusion_mask", "ensemble_exclusion_mask")
OUT = HERE.parents[2] / "06 Outputs" / "jalsar" / "extensions" / "m11_gfm"
BEAS = HERE.parents[2] / "06 Outputs" / "jalsar" / "extensions" / "m11_beas"
FAMS = ("rf", "unet", "terramind")
PASS_HOURS = {"ASCENDING": (10, 15), "DESCENDING": (0, 3)}   # UTC hours over the two areas
TIME_TOL_MIN = 30          # GFM items within this many minutes of the scene time belong to the pass
GP = None
HTTP = {"get": None, "post": None}   # a test can put its own functions here


def _req(url, data=None, tries=4):
    for k in range(tries):
        try:
            r = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode(),
                                       headers={"Content-Type": "application/json", "Accept": "*/*",
                                                "User-Agent": "jalsar-study"})
            with urllib.request.urlopen(r, timeout=120) as f:
                return f.read()
        except Exception as exc:                  # noqa: BLE001
            if k == tries - 1:
                raise
            print(f"    {url[:80]}: {str(exc)[:100]}; again in {5 * (k + 1)} s", flush=True)
            time.sleep(5 * (k + 1))


def http_post(url, body):
    f = HTTP["post"] or (lambda u, b: json.loads(_req(u, b)))
    return f(url, body)


def http_get(url):
    f = HTTP["get"] or _req
    return f(url)


def search(geom, day):
    """Every GFM item over geom on the UTC day (a list of STAC item dicts)."""
    d0 = date.fromisoformat(day)
    body = {"collections": [COLLECTION], "intersects": geom, "limit": 100,
            "datetime": f"{d0.isoformat()}T00:00:00Z/{(d0 + timedelta(days=1)).isoformat()}T00:00:00Z"}
    items, url = [], f"{STAC}/search"
    while url:
        page = http_post(url, body)
        items += page.get("features", [])
        nxt = [lk for lk in page.get("links", []) if lk.get("rel") == "next"]
        url, body = (nxt[0]["href"], nxt[0].get("body", body)) if nxt else (None, None)
    return items


def item_time(it):
    t = it["properties"].get("datetime") or it["properties"].get("start_datetime")
    return datetime.fromisoformat(t.replace("Z", "+00:00"))


def pick_items(items, time_utc=None, orbit=None, orbit_pass=None):
    """The items of one Sentinel-1 pass: by relative orbit if GFM gives it, else by time, else the
    acquisition (items within TIME_TOL_MIN of each other) with the most tiles, among those at the
    hour of the pass direction (over India ascending passes are near 12:30 UTC, descending near
    00:45 UTC). Returns (items, how)."""
    if not items:
        return [], "no GFM item on that day"
    rel = [it for it in items if it["properties"].get("sat:relative_orbit") is not None]
    if orbit is not None and len(rel) == len(items):
        keep = [it for it in items if int(it["properties"]["sat:relative_orbit"]) == int(orbit)]
        return keep, f"relative orbit {orbit} (item property sat:relative_orbit)"
    if time_utc:
        t0 = datetime.strptime(time_utc[:16], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        keep = [it for it in items if abs((item_time(it) - t0).total_seconds()) <= TIME_TOL_MIN * 60]
        return keep, f"within {TIME_TOL_MIN} min of the scene time {time_utc} UTC"
    groups = []
    for it in sorted(items, key=item_time):
        if groups and (item_time(it) - item_time(groups[-1][-1])).total_seconds() <= TIME_TOL_MIN * 60:
            groups[-1].append(it)
        else:
            groups.append([it])
    if orbit_pass in PASS_HOURS and len(groups) > 1:
        lo, hi = PASS_HOURS[orbit_pass]
        right = [g for g in groups if lo <= item_time(g[0]).hour <= hi]
        if len(right) == 1:
            g = right[0]
            return g, (f"the {orbit_pass.lower()} acquisition at {item_time(g[0]):%H:%M} UTC "
                       f"(the day had {len(groups)} acquisitions)")
        groups = right or groups
    best = max(groups, key=len)
    how = (f"the acquisition at {item_time(best[0]):%H:%M} UTC with the most tiles"
           + (f" (the day had {len(groups)} acquisitions; check against the scene)" if len(groups) > 1 else ""))
    return best, how


def asset_key(it, names):
    for n in names:
        if n in it.get("assets", {}):
            return n
    return None


def get_tile(it, key):
    """The asset GeoTIFF of one item, downloaded once into tiles\\."""
    href = it["assets"][key]["href"]
    p = OUT / "tiles" / f"{it['id']}__{key}.tif"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.stem + "_part.tif")
        tmp.write_bytes(http_get(href))
        tmp.replace(p)
    return p


def onto_grid(paths, prof):
    """Mosaic of GFM tiles on prof's grid, nearest; 255 where no tile has data."""
    from rasterio.warp import reproject, Resampling
    out = np.full((prof["height"], prof["width"]), 255, "uint8")
    seen = set()
    for p in paths:
        tmp = np.full_like(out, 255)
        with rasterio.open(p) as s:
            reproject(rasterio.band(s, 1), tmp, src_transform=s.transform, src_crs=s.crs,
                      dst_transform=prof["transform"], dst_crs=prof["crs"], resampling=Resampling.nearest,
                      src_nodata=s.nodata if s.nodata is not None else 255, dst_nodata=255)
        seen |= set(np.unique(tmp).tolist())
        have = tmp != 255
        out = np.where(have & (out == 255), tmp, np.where(have, np.maximum(out, tmp), out))
    return out, sorted(seen)


def gfm_map(items, prof):
    """(map 0/1/255, record) for one pass on prof's grid."""
    fk = {asset_key(it, FLOOD_ASSETS) for it in items}
    if None in fk:
        mc.fail("A GFM item has none of the flood-extent assets " + ", ".join(FLOOD_ASSETS)
                + ". Run 'python m11_gfm.py probe' and send GFM_PROBE.md.")
    flood, vals = onto_grid([get_tile(it, asset_key(it, FLOOD_ASSETS)) for it in items], prof)
    odd = [v for v in vals if v not in (0, 1, 255)]
    if odd:
        mc.fail(f"The GFM flood extent has values {vals}, expected 0, 1 and 255. Run probe and send GFM_PROBE.md.")
    rec = {"items": [it["id"] for it in items], "flood_values": vals}
    ek = [asset_key(it, EXCL_ASSETS) for it in items]
    if all(ek):
        excl, ev = onto_grid([get_tile(it, k) for it, k in zip(items, ek)], prof)
        rec["exclusion_values"] = ev
        ex = excl == 1
        rec["excluded_cells"] = int((ex & (flood != 255)).sum())
        flood = np.where(ex & (flood != 255), 0, flood).astype("uint8")
    else:
        rec["excluded_cells"] = None
        rec["note"] = "no exclusion mask asset; GFM's own flood extent used as given"
    return flood, rec


def counts(ours, ref, m):
    o, r = ours[m] == 1, ref[m] == 1
    tp, fp, fn = int((o & r).sum()), int((o & ~r).sum()), int((~o & r).sum())
    return tp, fp, fn


def scores(tp, fp, fn):
    nan = float("nan")
    return {"csi": tp / (tp + fp + fn) if tp + fp + fn else nan,
            "hit_rate": tp / (tp + fn) if tp + fn else nan,
            "far": fp / (tp + fp) if tp + fp else nan}


def events_to_do(gp):
    """[(event id, label, date, time_utc, orbit, pass, folder of the model maps, footprint, year, event)]"""
    cor = json.loads(gp.corridor.read_text(encoding="utf-8"))
    cor = cor["features"][0]["geometry"] if "features" in cor else cor
    out = []
    for e in gc.load_events(gp):
        d = gp.event_dir(e.id)
        if any((d / f"flood_{f}.tif").exists() for f in FAMS):
            out.append((e.id, e.label or e.id, e.date, e.scene_time_utc or None, e.orbit, e.orbit_pass, d, cor,
                        e.validated_year, e))
    rr = BEAS / "run_record.json"
    if rr.exists():
        r = mc.load_json(rr)
        g = json.loads((BEAS / "beas_valley.geojson").read_text(encoding="utf-8"))["features"][0]["geometry"]
        pr = r["pass"]
        out.append((f"beas_{pr['date']}_orbit{int(pr['orbit']):03d}", "Beas, July 2023 (blind test)",
                    pr["date"], pr["time_utc"], pr["orbit"], pr["pass"], BEAS, g, 0, None))
    return out


def fmt(v, pct=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v * 100:.1f} %" if pct else f"{v:.3f}"


def quicklook(path, ours, gfm, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    k = max(1, int(np.ceil(max(ours.shape) / 1600)))
    h, w = (ours.shape[0] // k) * k, (ours.shape[1] // k) * k
    o = (ours[:h, :w] == 1).reshape(h // k, k, w // k, k).any(axis=(1, 3))
    g = (gfm[:h, :w] == 1).reshape(h // k, k, w // k, k).any(axis=(1, 3))
    nd = ((ours[:h, :w] == 255) | (gfm[:h, :w] == 255)).reshape(h // k, k, w // k, k).all(axis=(1, 3))
    c = np.zeros(o.shape, "uint8")
    c[nd] = 1
    c[o & ~g] = 2
    c[~o & g] = 3
    c[o & g] = 4
    fig, ax = plt.subplots(figsize=(10, 10 * o.shape[0] / max(o.shape[1], 1) + 0.8))
    ax.imshow(c, cmap=ListedColormap(["#f4f4f4", "#ffffff", "#d95f02", "#1b9e77", "#3b3b98"]),
              vmin=0, vmax=4, interpolation="nearest")
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)
    import matplotlib.patches as mp
    ax.legend(handles=[mp.Patch(color=c_, label=l_) for c_, l_ in (("#3b3b98", "both"), ("#d95f02", "Jal-SAR only"),
                                                                  ("#1b9e77", "GFM only"))],
              loc="lower right", fontsize=8, frameon=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def probe(args):
    gp = GP or gc.GPaths()
    if args.geojson:
        g = json.loads(Path(args.geojson).read_text(encoding="utf-8"))
    else:
        g = json.loads(gp.corridor.read_text(encoding="utf-8"))
    geom = g["features"][0]["geometry"] if "features" in g else g
    print(f"  GFM search, {args.date}, {COLLECTION} at {STAC} ...")
    items = search(geom, args.date)
    md = ["# GFM probe", "", f"Search: collection {COLLECTION}, {args.date}, "
          f"footprint {'given' if args.geojson else 'the corridor'}. Items found: {len(items)}.", ""]
    print(f"  {len(items)} item(s)")
    if not items:
        md.append("No item on that day. Try another date (GFM has one entry per Sentinel-1 pass).")
        (OUT / "GFM_PROBE.md").write_text("\n".join(md), encoding="utf-8")
        print("FAIL  no GFM item; see GFM_PROBE.md")
        return 1
    md += ["| Item | Time (UTC) | Relative orbit | Orbit state |", "|---|---|---|---|"]
    for it in items:
        pr = it["properties"]
        md.append(f"| {it['id']} | {item_time(it):%Y-%m-%d %H:%M} | {pr.get('sat:relative_orbit', '?')} | "
                  f"{pr.get('sat:orbit_state', '?')} |")
    it = items[0]
    md += ["", "Properties of the first item:", "", "```"]
    md += [f"{k}: {str(v)[:120]}" for k, v in it["properties"].items()]
    md += ["```", "", "Assets of the first item:", "", "| Asset | Type | Href (start) |", "|---|---|---|"]
    md += [f"| {k} | {a.get('type', '')} | {a.get('href', '')[:90]} |" for k, a in it.get("assets", {}).items()]
    fk = asset_key(it, FLOOD_ASSETS)
    md.append("")
    if fk is None:
        md.append(f"PROBLEM: none of {FLOOD_ASSETS} is an asset; run cannot go ahead until the names are fixed.")
        ok = False
    else:
        p = get_tile(it, fk)
        with rasterio.open(p) as s:
            a = s.read(1)
            v, n = np.unique(a, return_counts=True)
            md += [f"Flood extent tile ({fk}): {s.width} x {s.height} cells at {abs(s.res[0]):.1f} m, nodata "
                   f"{s.nodata}, CRS {str(s.crs)[:60]}...", "", "| Value | Cells |", "|---|---|"]
            md += [f"| {int(x)} | {int(c):,} |" for x, c in zip(v, n)]
        ok = set(v.tolist()) <= {0, 1, 255}
        if not ok:
            md += ["", "PROBLEM: values other than 0, 1 and 255; run will stop until this is understood."]
    ek = asset_key(it, EXCL_ASSETS)
    md += ["", f"Exclusion mask asset: {ek or 'not found (run will use the flood extent as given)'}"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "GFM_PROBE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"  wrote {OUT / 'GFM_PROBE.md'}")
    print(("PASS" if ok else "FAIL") + "  GFM probe")
    return 0 if ok else 1


def run(args):
    gp = GP or gc.GPaths()
    evs = events_to_do(gp)
    if not evs:
        mc.fail("No event has model maps yet (06 Outputs\\jalsar\\<event>\\flood_rf.tif and so on).")
    rows, refrows, md_ev = [], [], []
    head, _ = gc.headline_family(gp)
    for eid, label, day, tutc, orbit, opass, folder, geom, year, ev in evs:
        print(f"  {eid}: GFM search on {day} ...", flush=True)
        items, how = pick_items(search(geom, day), tutc, orbit, opass)
        if not items:
            md_ev.append(f"- {label} ({eid}): no GFM item for this pass ({how}).")
            print(f"    no GFM item ({how})")
            continue
        maps = {f: folder / f"flood_{f}.tif" for f in FAMS if (folder / f"flood_{f}.tif").exists()}
        with rasterio.open(next(iter(maps.values()))) as s:
            prof = s.profile.copy()
        print(f"    {len(items)} GFM tile(s), {how}", flush=True)
        gfm, rec = gfm_map(items, prof)
        gpath = OUT / f"gfm_{eid}.tif"
        mc.write_flood_map(gpath, gfm, prof)
        km2 = mc.cell_km2(prof)
        if ev is not None:
            reach, _ = gc.reach_raster(ev, prof)
        else:
            reach = None
        both_any = None
        for fam, p in maps.items():
            with rasterio.open(p) as s:
                ours = s.read(1)
            both = np.isin(ours, (0, 1)) & np.isin(gfm, (0, 1))
            both_any = both if both_any is None else both_any
            parts = [("pooled", both)]
            if reach is not None:
                parts += [(r, both & (reach == r)) for r in gc.REACHES]
            for rname, m in parts:
                tp, fp, fn = counts(ours, gfm, m)
                s_ = scores(tp, fp, fn)
                rows.append({"event": eid, "family": fam, "reach": rname, "common_km2": round(float(m.sum()) * km2, 3),
                             "jalsar_km2": round((tp + fp) * km2, 3), "gfm_km2": round((tp + fn) * km2, 3),
                             "both_km2": round(tp * km2, 3), **{k: round(v, 4) for k, v in s_.items()}})
            if fam == (head or "rf") or (head not in maps and fam == next(iter(maps))):
                quicklook(OUT / f"fig_{eid}.png", np.where(both, ours, 255), np.where(both, gfm, 255),
                          f"{label}: {gc.FAMILY_NAMES[fam]} against GFM ({day})")
            # 2021 and 2023: the hand reference, with the Table D scorer, on GFM's footprint
            if year:
                masked = np.where(gfm == 255, 255, ours).astype("uint8")
                mp_ = OUT / f"_masked_{eid}_{fam}.tif"
                mc.write_flood_map(mp_, masked, prof)
                out = OUT / f"ref_{year}_{fam}"
                gc.run_scorer(gp, mp_, year, out)
                refrows += ref_rows(out, year, gc.FAMILY_NAMES[fam])
                mp_.unlink(missing_ok=True)
        if year:
            out = OUT / f"ref_{year}_gfm"
            gc.run_scorer(gp, gpath, year, out)
            refrows += ref_rows(out, year, "GFM")
        md_ev.append(f"- {label} ({eid}): {len(items)} GFM tile(s), {how}; GFM flood "
                     f"{float((gfm == 1).sum()) * km2:.2f} km2; area both cover "
                     f"{float(both_any.sum()) * km2:.1f} km2; GFM excluded "
                     + (f"{rec['excluded_cells'] * km2:.1f} km2" if rec["excluded_cells"] is not None else "n/a")
                     + f". Figure: fig_{eid}.png")
        mc.save_json(OUT / f"gfm_{eid}_record.json", {**rec, "how_picked": how, "event": eid})
    if rows:
        with open(OUT / "agreement.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    if refrows:
        with open(OUT / "reference_scores.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(refrows[0]))
            w.writeheader()
            w.writerows(refrows)
    write_md(rows, refrows, md_ev)
    print(f"  wrote {OUT / 'GFM_COMPARISON.md'}")
    print("PASS  M11 GFM comparison")
    return 0


def ref_rows(out, year, who):
    p = Path(out) / "metrics_by_stratum.csv"
    with open(p, newline="", encoding="utf-8") as fh:
        keep = [r for r in csv.DictReader(fh) if r["table"] == "reported" and r["dimension"] == "ALL"]
    return [{"year": year, "map": who, "reach": r["reach"], "csi": r["csi"], "csi_lo": r["csi_lo"],
             "csi_hi": r["csi_hi"], "hit_rate": r["hit_rate"], "far": r["far"],
             "flood_pixels_truth": r["flood_pixels_truth"]} for r in keep]


def write_md(rows, refrows, md_ev):
    md = ["# M11: Jal-SAR against Copernicus GFM", "",
          "GFM is another model (three algorithms combined), so the first table is agreement, not accuracy. "
          "Hit rate is the share of GFM's flood that Jal-SAR also maps; false alarm ratio is the share of "
          "Jal-SAR's flood that GFM does not map. Only cells where both maps have data count. GFM's excluded "
          "cells count as no flood.", "", "Events:", ""] + md_ev
    md += ["", "| Event | Family | Reach | Both cover (km2) | Jal-SAR flood (km2) | GFM flood (km2) | In both (km2) "
           "| CSI | Hit rate | False alarm ratio |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['event']} | {gc.FAMILY_NAMES[r['family']]} | {r['reach']} | {r['common_km2']} | "
                  f"{r['jalsar_km2']} | {r['gfm_km2']} | {r['both_km2']} | {fmt(r['csi'])} | "
                  f"{fmt(r['hit_rate'], True)} | {fmt(r['far'], True)} |")
    if refrows:
        md += ["", "## 2021 and 2023: GFM and the models against the hand reference", "",
               "Same scorer and window bootstrap as Table D (04_stratified_metrics.py), but only on the cells GFM "
               "covers, so the model figures here are not the Table D figures.", "",
               "| Year | Map | Reach | CSI (95 % interval) | Hit rate | False alarm ratio |", "|---|---|---|---|---|---|"]
        for r in refrows:
            md.append(f"| {r['year']} | {r['map']} | {r['reach']} | {r['csi']} ({r['csi_lo']} to {r['csi_hi']}) | "
                      f"{r['hit_rate']} | {r['far']} |")
    md += ["", "Source: Copernicus Emergency Management Service, Global Flood Monitoring, via the EODC STAC catalogue "
           f"({STAC}, collection {COLLECTION}).", ""]
    (OUT / "GFM_COMPARISON.md").write_text("\n".join(md), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("probe", "run"))
    ap.add_argument("--date", default="2023-08-16")
    ap.add_argument("--geojson")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    return probe(args) if args.cmd == "probe" else run(args)


if __name__ == "__main__":
    sys.exit(main())
