r"""
jalsar_core.py
==============
The shared core of Jal-SAR: everything the
runner, the bulletin and the web map need, and nothing they show.

It reuses FYP_code (models_common.py and 04_stratified_metrics.py) and
copies nothing from it, so a Jal-SAR map of a validated event is the same
map as the Table A map of that event.

WHAT IS IN HERE
---------------
Events          the three events on disk (21 Oct 2021, 16 Aug 2023, and the
                radar-only 19 Jul 2023) plus any the runner adds later
                (06 Outputs\jalsar\events.json)
Model choice    the three model families (Random Forest, U-Net, TerraMind),
                each trained on the hand-labelled set; TerraMind's variant
                is the one with the higher Sen1Floods11 VALID IoU (rule as
                changed on 8 October); the headline family by the Table D
                rule
Prediction      the scene's inputs on its own 10 m grid, each family's flood
                map (same masks as every map in this project), and the
                agreement map (how many families call a cell flood)
Trust layer     radar blind (layover or shadow), then the lowest
                Table D precision of the cell's incidence, HAND and valley
                width classes, graded high (>= 0.40), medium (>= 0.20) or low
Exposure        villages (Census 2011 households and population) and
                OpenStreetMap roads inside the headline flood map
Summary         one summary.json per event, read by the bulletin and the map

USAGE  (Anaconda Prompt, fyp active, from 04 Code\jalsar)
-----
    python jalsar_core.py --selftest

The commands people run are in jalsar.py.
"""

__stage__ = "Jal-SAR: shared core"
__reads__ = ["trained models (FYP_code/data/models)", "corridor scenes and strata",
             "Table D tables (06 Outputs/tables/lowo_*)", "VILLAGES_corridor.gpkg",
             "06 Outputs/jalsar/osm/roads_corridor.gpkg (optional)"]
__writes__ = ["06 Outputs/jalsar/<event>/ (maps, trust layer, summary.json)"]

import csv
import json
import math
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FYP = HERE.parent / "FYP_code"
if str(FYP) not in sys.path:
    sys.path.insert(0, str(FYP))

import models_common as mc          # noqa: E402

try:
    import rasterio
    from rasterio.features import rasterize
    from rasterio.warp import transform as warp_transform
except ImportError as e:
    sys.exit(f"\nMissing package: {e}.\nActivate the environment first:  conda activate fyp\n")

scorer = mc.load_script("04_stratified_metrics.py")

VERSION = "0.1"
FAMILIES = {"rf": ("rf",), "unet": ("unet",), "terramind": ("tm_s1t", "tm_s1")}
FAMILY_NAMES = {"rf": "Random Forest", "unet": "U-Net", "terramind": "TerraMind"}
SPEED_ORDER = ("rf", "unet", "terramind")          # tie rule: the faster family wins
# Rule as changed on 8 October: every
# family uses the hand-labelled training set. Within TerraMind the variant
# (tm_s1t or tm_s1) is still the one with the higher Sen1Floods11 valid IoU.
TRAINSET = "hand"
TRUST_HIGH, TRUST_MED = 0.40, 0.20                 # fixed 8 October
TRUST_NAMES = {0: "radar blind", 1: "low", 2: "medium", 3: "high"}
TRUST_DIMS = ("incidence", "hand", "valley_width")
STRATA_FILES = {"incidence": "strata_theta_loc.tif", "hand": "strata_hand_m.tif",
                "valley_width": "strata_valley_width_m.tif"}
REACHES = ("gorge_above_devprayag", "open_valley_below_devprayag")
REACH_NAMES = {"gorge_above_devprayag": "Gorge above Devprayag",
               "open_valley_below_devprayag": "Open valley below Devprayag"}
ROAD_STEP_M = 5.0
VILLAGE_MIN_KM2 = 0.01      # villages with at least 1 ha of mapped flood (9 October):
                            # counted beside the villages merely touching the flood


# =============================================================================
# Paths and events
# =============================================================================

class GPaths(mc.Paths):
    """The project's paths plus Jal-SAR's own folders."""

    def __init__(self, project=None):
        super().__init__(project)
        if project is None:            # models_common finds the project from FYP_code
            pass
        self.gj = self.o06 / "jalsar"
        self.villages = self.d05 / "boundaries" / "VILLAGES_corridor.gpkg"
        self.roads = self.o06 / "jalsar" / "osm" / "roads_corridor.gpkg"   # never 05 Data
        self.corridor = self.data / "corridor.geojson"

    def event_dir(self, event_id):
        return self.gj / event_id


@dataclass
class Event:
    id: str                 # e.g. 2021-10-21_orbit063
    date: str               # the scene date, YYYY-MM-DD
    scene: str              # path of the GRD scene (VV, VH, angle)
    strata: str             # strata folder for this orbit
    orbit: int
    orbit_pass: str
    validated_year: int = 0  # 2021 or 2023 when a reference exists, else 0
    label: str = ""
    scene_id: str = ""       # Earth Engine scene id(s), for runner events
    scene_time_utc: str = ""  # acquisition time, for runner events
    strata_note: str = ""    # how the strata of this orbit were made

    def scene_path(self):
        return Path(self.scene)

    def strata_path(self):
        return Path(self.strata)


def builtin_events(gp):
    """The events whose scenes are already in 05 Data."""
    s1 = gp.d05 / "sentinel1"
    return [
        Event("2021-10-21_orbit063", "2021-10-21", str(gp.scene(2021, "grd")),
              str(gp.strata(2021)), 63, "DESCENDING", 2021, "October 2021 flood"),
        Event("2023-08-16_orbit129", "2023-08-16", str(gp.scene(2023, "grd")),
              str(gp.strata(2023)), 129, "ASCENDING", 2023, "August 2023 flood"),
        # same relative orbit and pass as 2021, so the same layover/shadow geometry
        Event("2023-07-19_orbit063", "2023-07-19", str(s1 / "S1_2023jul_post_20230719_orb63.tif"),
              str(gp.strata(2021)), 63, "DESCENDING", 0, "July 2023 (radar only, no reference)"),
    ]


def load_events(gp, only_existing=True):
    """Built-in events plus the runner's (06 Outputs/jalsar/events.json)."""
    ev = builtin_events(gp)
    reg = gp.gj / "events.json"
    if reg.exists():
        known = {e.id for e in ev}
        for d in mc.load_json(reg).get("events", []):
            if d["id"] not in known:
                ev.append(Event(**d))
    if only_existing:
        ev = [e for e in ev if e.scene_path().exists()]
    return ev


def register_event(gp, event):
    """Add a runner event to events.json (kept across runs)."""
    reg = gp.gj / "events.json"
    data = mc.load_json(reg) if reg.exists() else {"events": []}
    data["events"] = [d for d in data["events"] if d["id"] != event.id] + [asdict(event)]
    reg.parent.mkdir(parents=True, exist_ok=True)
    mc.save_json(reg, data)


def find_event(gp, key):
    for e in load_events(gp, only_existing=False):
        if key in (e.id, e.date):
            return e
    mc.fail(f"No event '{key}'. Known: "
            + ", ".join(e.id for e in load_events(gp, only_existing=False)))


# =============================================================================
# Model choice
# =============================================================================

def family_member(gp, family):
    """(model, trainset, valid IoU): the 'hand' model of the family; for TerraMind
    the variant with the higher Sen1Floods11 valid IoU. None if none is trained."""
    best = None
    for model in FAMILIES[family]:
        for ts in (TRAINSET,):
            meta_p = gp.model_dir(model, ts) / "model_meta.json"
            if not meta_p.exists():
                continue
            iou = mc.load_json(meta_p).get("iou_valid_at_threshold")
            if iou is None or not math.isfinite(float(iou)):
                continue
            if best is None or float(iou) > best[2]:
                best = (model, ts, float(iou))
    return best


def table_d_row(gp, year, model, reach="pooled", dimension="ALL"):
    """Rows of Table D (GRD) for one model and year, keyed by stratum."""
    p = gp.tables / f"lowo_{year}_{model}_grd" / "metrics_by_stratum.csv"
    if not p.exists():
        return None
    with open(p, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["table"] == "reported"
                and r["reach"] == reach and (dimension is None or r["dimension"] == dimension)]
    return rows


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def headline_family(gp, families=("rf", "unet", "terramind"), years=mc.YEARS):
    """The Table D rule for the headline family. Returns (family, record)."""
    scores = {}
    for fam in families:
        m = family_member(gp, fam)
        if m is None:
            continue
        f1s = []
        for y in years:
            rows = table_d_row(gp, y, m[0])
            if rows:
                f1s.append(_f(rows[0]["f1"]))
        if len(f1s) == len(years) and all(math.isfinite(v) for v in f1s):
            scores[fam] = {"model": m[0], "trainset": m[1], "table_d_f1_mean": float(np.mean(f1s)),
                           "table_d_f1": dict(zip(map(str, years), f1s))}
    if not scores:
        return None, {"rule": "no family has Table D on GRD for every year yet"}
    ranked = sorted(scores, key=lambda f: -scores[f]["table_d_f1_mean"])
    rec = {"scores": scores, "ranked": ranked, "rule": "highest mean pooled Table D F1 (GRD)"}
    if len(ranked) == 1:
        return ranked[0], rec
    a, b = ranked[0], ranked[1]
    tie = paired_tie(gp, scores[b]["model"], scores[a]["model"], years)
    rec["paired_check"] = tie
    if tie.get("tie"):
        pick = min((a, b), key=SPEED_ORDER.index)
        rec["rule"] += (f"; {FAMILY_NAMES[a]} and {FAMILY_NAMES[b]} not distinguishable "
                        "(paired interval includes 0 in a year), so the faster one")
        return pick, rec
    return a, rec


def paired_tie(gp, model_b, model_a, years):
    """Paired Table D comparison of the top two (27_compare_maps.py), per year.

    A tie unless the interval on the F1 difference excludes 0 in every year.
    """
    out = {"years": {}}
    tie = False
    for y in years:
        ma, mb = gp.lowo_map(y, model_a, "grd"), gp.lowo_map(y, model_b, "grd")
        if not (ma.exists() and mb.exists()):
            out["years"][str(y)] = "map missing"
            tie = True
            continue
        od = gp.gj / "selection" / f"compare_{y}_lowo_{model_b}_vs_{model_a}"
        if not (od / "comparison.csv").exists():
            rc = mc.run_script("27_compare_maps.py", [
                "--a", str(mb), "--b", str(ma), "--truth", str(gp.reference(y)),
                "--strata", str(gp.strata(y)), "--windows", str(gp.windows(y)),
                "--label-a", model_b, "--label-b", model_a, "--out", str(od)])
            if rc != 0:
                out["years"][str(y)] = "comparison failed"
                tie = True
                continue
        with open(od / "comparison.csv", newline="", encoding="utf-8") as fh:
            row = next(r for r in csv.DictReader(fh)
                       if r["reach"] == "pooled" and r["dimension"] == "ALL")
        lo, hi = _f(row["diff_f1_lo"]), _f(row["diff_f1_hi"])
        excl = math.isfinite(lo) and math.isfinite(hi) and (lo > 0 or hi < 0)
        out["years"][str(y)] = {"diff_f1": _f(row["diff_f1"]), "lo": lo, "hi": hi,
                                "excludes_0": bool(excl)}
        tie |= not excl
    out["tie"] = bool(tie)
    return out


# =============================================================================
# Inputs and prediction
# =============================================================================

def event_inputs(gp, event):
    """Like models_common.corridor_inputs, for any scene and strata folder."""
    vv, vh, prof = mc.read_scene(event.scene_path(), f"The scene of {event.date}")
    sd = event.strata_path()
    for n in ("strata_hand_m.tif", "strata_slope_deg.tif", "strata_layover_shadow.tif"):
        if not (sd / n).exists():
            mc.fail(f"Strata file {n} for orbit {event.orbit} is missing:\n  {sd / n}\n"
                    "Build the strata for this orbit first (jalsar.py run does it).")
    if not gp.perm_water().exists():
        mc.fail(f"The permanent-water mask is missing:\n  {gp.perm_water()}")
    hand = mc.reproject_onto(sd / "strata_hand_m.tif", prof)
    slope = mc.reproject_onto(sd / "strata_slope_deg.tif", prof)
    ls = mc.reproject_onto(sd / "strata_layover_shadow.tif", prof, categorical=True)
    pw = mc.reproject_onto(gp.perm_water(), prof, categorical=True)
    x = np.stack([vv, vh, hand, slope]).astype("float32")
    return {"x": x, "valid": np.isfinite(vv), "profile": prof, "perm": pw == 100,
            "distort": np.isin(ls, (1, 2))}


def predict_family(gp, member, inp, cpu_ok=False, tile=512, overlap=64):
    model, ts, _ = member
    pred = mc.load_predictor(model, gp.model_dir(model, ts), cpu_ok=cpu_ok, tile=tile,
                             overlap=overlap)
    prob = pred.predict_image(inp["x"], inp["valid"])
    fmap, diag = mc.make_flood_map(prob, inp["valid"], pred.threshold, inp["perm"],
                                   inp["distort"])
    diag.update(model=model, trainset=ts, threshold=pred.threshold)
    return fmap, diag


def agreement(maps, valid):
    """Per cell, how many maps call it flood (0..n); 255 outside valid."""
    n = np.zeros(valid.shape, "uint8")
    for m in maps:
        n += (m == 1).astype("uint8")
    n[~valid] = mc.NODATA
    return n


# =============================================================================
# Trust layer
# =============================================================================

def trust_table(gp, model, years=mc.YEARS):
    """{dimension: {stratum: precision}} from Table D (GRD), averaged over years.

    Thin classes (fewer than 100 reference flood pixels in a year) and classes
    without a precision count as low. Merged rows ("15_50m+gt50m") apply to
    each class they merge.
    """
    table = {d: {} for d in TRUST_DIMS}
    per = {d: {} for d in TRUST_DIMS}
    for y in years:
        rows = table_d_row(gp, y, model, reach="pooled", dimension=None) or []
        for r in rows:
            if r["dimension"] not in TRUST_DIMS:
                continue
            thin = r.get("thin") == "True" or _f(r["flood_pixels_truth"]) < scorer.MIN_FLOOD_PIXELS
            p = _f(r["precision"])
            for s in r["stratum"].split("+"):
                per[r["dimension"]].setdefault(s, []).append(float("nan") if thin else p)
    for d in TRUST_DIMS:
        for s, vals in per[d].items():
            table[d][s] = (float(np.mean(vals)) if vals and all(math.isfinite(v) for v in vals)
                           else float("nan"))
    return table


def trust_grade(p):
    if not math.isfinite(p):
        return 1
    return 3 if p >= TRUST_HIGH else 2 if p >= TRUST_MED else 1


def strata_classes(event, profile):
    """{dimension: object array of class labels} on the scene grid, and the distortion code."""
    sd = event.strata_path()
    out = {}
    for d in TRUST_DIMS:
        arr = mc.reproject_onto(sd / STRATA_FILES[d], profile)
        out[d] = scorer.classify(arr, *scorer.BINS[d])
    ls = mc.reproject_onto(sd / "strata_layover_shadow.tif", profile, categorical=True)
    return out, np.isin(ls, (1, 2))


def trust_layer(gp, event, profile, valid, model):
    """uint8: 0 radar blind, 1 low, 2 medium, 3 high, 255 outside the scene."""
    table = trust_table(gp, model)
    classes, blind = strata_classes(event, profile)
    grade = np.full(valid.shape, 3, "uint8")
    for d in TRUST_DIMS:
        lut = {s: trust_grade(p) for s, p in table[d].items()}
        g = np.ones(valid.shape, "uint8")                  # unknown class: low
        for s, v in lut.items():
            g[classes[d] == s] = v
        grade = np.minimum(grade, g)
    grade[blind] = 0
    grade[~valid] = mc.NODATA
    return grade, table


# =============================================================================
# Areas and exposure
# =============================================================================

def reach_raster(event, profile):
    """Object array of reach names on the grid (the scorer's rule)."""
    meta_p = event.strata_path() / "strata_meta.json"
    lon = mc.load_json(meta_p).get("reach_split_longitude", 78.5981) if meta_p.exists() else 78.5981
    split_x = warp_transform("EPSG:4326", profile["crs"], [lon], [30.15])[0][0]
    t = profile["transform"]
    xs = t.c + t.a * (np.arange(profile["width"]) + 0.5)
    row = np.where(xs >= split_x, REACHES[0], REACHES[1]).astype(object)
    return np.broadcast_to(row, (profile["height"], profile["width"])), float(split_x)


def areas(flood, trust, reach, km2):
    """Flood km2 by reach and by trust class."""
    f = flood == 1
    out = {"total_km2": round(float(f.sum() * km2), 4), "by_reach": {}, "by_trust": {},
           "by_reach_trust": {}}
    for r in REACHES:
        m = f & (reach == r)
        out["by_reach"][r] = round(float(m.sum() * km2), 4)
        out["by_reach_trust"][r] = {TRUST_NAMES[k]: round(float((m & (trust == k)).sum() * km2), 4)
                                    for k in (3, 2, 1)}
    for k in (3, 2, 1):
        out["by_trust"][TRUST_NAMES[k]] = round(float((f & (trust == k)).sum() * km2), 4)
    return out


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def village_exposure(gp, flood, trust, profile, km2, min_km2=VILLAGE_MIN_KM2):
    """Villages whose polygon contains mapped flood, most flood area first.

    share_flooded is the mapped flood over the area of the WHOLE village polygon
    (Survey of India), not over the part of the village inside the corridor
    (9 October): a village that only grazes the corridor no longer looks
    mostly flooded. Two counts are given: every village touching the flood,
    and those with at least min_km2 of mapped flood (1 ha by default).
    """
    import geopandas as gpd
    if not gp.villages.exists():
        return None
    v = gpd.read_file(gp.villages).to_crs(profile["crs"])
    v = v[v.geometry.notna() & ~v.geometry.is_empty].reset_index(drop=True)
    ids = rasterize(((g, i + 1) for i, g in enumerate(v.geometry)),
                    out_shape=flood.shape, transform=profile["transform"], fill=0,
                    dtype="int32", all_touched=False)
    f = flood == 1
    n = len(v) + 1
    flooded = np.bincount(ids[f].ravel(), minlength=n)
    high = np.bincount(ids[f & (trust == 3)].ravel(), minlength=n)
    rows = []
    for i, r in v.iterrows():
        k = i + 1
        if flooded[k] == 0:
            continue
        vkm2 = float(r.geometry.area) / 1e6               # projected CRS, metres
        pop = _num(r.get("total_population_village"))
        hh = _num(r.get("total_households"))
        rows.append({"village": str(r.get("village", "")), "vlcode": str(r.get("vlcode", "")),
                     "block": str(r.get("block", "")), "subdistrict": str(r.get("subdistric", "")),
                     "district": str(r.get("district", "")),
                     "households_2011": int(hh) if hh is not None else None,
                     "population_2011": int(pop) if pop is not None else None,
                     "flood_km2": round(float(flooded[k] * km2), 4),
                     "flood_high_trust_km2": round(float(high[k] * km2), 4),
                     "village_km2": round(vkm2, 4),
                     "share_flooded": round(min(float(flooded[k] * km2) / vkm2, 1.0), 4)
                     if vkm2 > 0 else None,
                     "lon": float(r.geometry.representative_point().x),
                     "lat": float(r.geometry.representative_point().y)})
    if rows:                                     # points for the map, in lon/lat
        xs, ys = warp_transform(profile["crs"], "EPSG:4326", [r["lon"] for r in rows],
                                [r["lat"] for r in rows])
        for r, x, y in zip(rows, xs, ys):
            r["lon"], r["lat"] = round(x, 5), round(y, 5)
    rows.sort(key=lambda r: (-r["flood_km2"], -(r["share_flooded"] or 0)))
    known = [r["population_2011"] for r in rows if r["population_2011"] is not None]
    big = [r for r in rows if r["flood_km2"] >= min_km2 - 1e-9]
    return {"villages": rows, "count": len(rows), "population_2011": int(sum(known)),
            "households_2011": int(sum(r["households_2011"] or 0 for r in rows)),
            "without_census_figure": sum(r["population_2011"] is None for r in rows),
            "min_flood_km2": min_km2,
            "count_min_area": len(big),
            "population_min_area": int(sum(r["population_2011"] or 0 for r in big)),
            "households_min_area": int(sum(r["households_2011"] or 0 for r in big)),
            "share_basis": "mapped flood over the area of the whole village polygon",
            "source": "Survey of India village boundaries with Census 2011 figures "
                      "(VILLAGES_corridor.gpkg)"}


def road_exposure(gp, flood, profile):
    """Flooded road length (km) by OpenStreetMap highway class, and the top named roads."""
    import geopandas as gpd
    import shapely
    if not gp.roads.exists():
        return None
    r = gpd.read_file(gp.roads).to_crs(profile["crs"])
    t = profile["transform"]
    inv = ~t
    by_class, named, total = {}, {}, {}
    for _, row in r.iterrows():
        g = row.geometry
        if g is None or g.is_empty:
            continue
        cls = str(row.get("highway") or "other")
        for part in getattr(g, "geoms", [g]):
            seg = shapely.segmentize(part, ROAD_STEP_M)
            c = shapely.get_coordinates(seg)
            if len(c) < 2:
                continue
            mid = (c[1:] + c[:-1]) / 2
            ln = np.hypot(*(c[1:] - c[:-1]).T)
            col, rowi = inv * (mid[:, 0], mid[:, 1])
            col, rowi = np.floor(col).astype(int), np.floor(rowi).astype(int)
            inside = (rowi >= 0) & (rowi < flood.shape[0]) & (col >= 0) & (col < flood.shape[1])
            fl = np.zeros(len(mid), bool)
            fl[inside] = flood[rowi[inside], col[inside]] == 1
            km = float(ln[fl].sum() / 1000)
            total[cls] = total.get(cls, 0.0) + float(ln[inside].sum() / 1000)
            if km > 0:
                by_class[cls] = by_class.get(cls, 0.0) + km
                nm = str(row.get("name") or row.get("ref") or "").strip()
                if nm:
                    named[(nm, cls)] = named.get((nm, cls), 0.0) + km
    top = sorted(({"name": n, "class": c, "flood_km": round(v, 3)} for (n, c), v in named.items()),
                 key=lambda d: -d["flood_km"])[:25]
    return {"by_class_km": {k: round(v, 3) for k, v in sorted(by_class.items(), key=lambda x: -x[1])},
            "total_flood_km": round(sum(by_class.values()), 3),
            "corridor_km_by_class": {k: round(v, 3) for k, v in total.items()},
            "top_named": top,
            "credit": "Roads (c) OpenStreetMap contributors, ODbL"}


# =============================================================================
# One event, end to end
# =============================================================================

def write_u8(path, arr, profile):
    mc.write_flood_map(path, arr, profile)


def flood_polygons(path, flood, trust, profile):
    """GeoPackage of the headline flood as polygons, one trust class per polygon."""
    import geopandas as gpd
    from rasterio.features import shapes
    from shapely.geometry import shape
    f = flood == 1
    code = np.where(f, np.where(trust == mc.NODATA, 1, trust), 0).astype("uint8")
    rows = [(shape(g), int(v)) for g, v in shapes(code, mask=f, transform=profile["transform"],
                                                  connectivity=8)]
    g = gpd.GeoDataFrame({"trust": [v for _, v in rows],
                          "trust_name": [TRUST_NAMES.get(v, "low") for _, v in rows]},
                         geometry=[p for p, _ in rows], crs=profile["crs"])
    g["area_m2"] = g.geometry.area.round(1)
    path = Path(path)
    if path.exists():
        path.unlink()
    g.to_file(path, layer="flood_headline", driver="GPKG")
    return len(g)


def model_files(gp, model, trainset):
    """SHA-256 of every file in a model's folder (the run record)."""
    d = gp.model_dir(model, trainset)
    if not d.is_dir():
        return {}
    return {p.name: mc.sha256(p) for p in sorted(d.iterdir())
            if p.is_file() and p.suffix in (".json", ".pt", ".joblib", ".pkl")}


def make_event(gp, event, families=("rf", "unet", "terramind"), force=False, cpu_ok=False,
               tile=512, overlap=64):
    """Maps, agreement, trust layer, exposure and summary.json for one event."""
    out = gp.event_dir(event.id)
    summ_p = out / "summary.json"
    if summ_p.exists() and not force:
        print(f"  {event.id}: already done, kept ({summ_p}); --force redoes it")
        return mc.load_json(summ_p)
    t0 = time.time()
    print(f"  {event.id}: {event.label or event.date}", flush=True)
    inp = event_inputs(gp, event)
    prof, valid = inp["profile"], inp["valid"]
    km2 = mc.cell_km2(prof)
    members, maps, diags = {}, {}, {}
    for fam in families:
        m = family_member(gp, fam)
        if m is None:
            print(f"    {FAMILY_NAMES[fam]}: no trained model yet, left out")
            continue
        members[fam] = m
        fmap, diag = predict_family(gp, m, inp, cpu_ok=cpu_ok, tile=tile, overlap=overlap)
        write_u8(out / f"flood_{fam}.tif", fmap, prof)
        maps[fam], diags[fam] = fmap, diag
        print(f"    {FAMILY_NAMES[fam]} ({m[0]} {m[1]}): {diag['flood_cells'] * km2:.3f} km2 "
              f"of flood, threshold {diag['threshold']:.2f}", flush=True)
    if not maps:
        mc.fail("No trained model was found. Train them first (FYP_code\\RUN ALL MODELS.bat).")
    agree = agreement(list(maps.values()), valid)
    write_u8(out / "agreement.tif", agree, prof)
    head, head_rec = headline_family(gp, tuple(maps))
    if head is None:                      # Table D not there yet: provisional, by valid IoU
        head = max(maps, key=lambda f: members[f][2])
        head_rec["provisional"] = (f"{FAMILY_NAMES[head]}, highest Sen1Floods11 valid IoU, "
                                   "until Table D exists for every family")
    headmap = maps[head]
    write_u8(out / "flood_headline.tif", headmap, prof)
    trust, ttable = trust_layer(gp, event, prof, valid, members[head][0])
    write_u8(out / "trust.tif", trust, prof)
    n_poly = flood_polygons(out / "flood_headline.gpkg", headmap, trust, prof)
    reach, split_x = reach_raster(event, prof)
    summ = {
        "jalsar_version": VERSION, "event": asdict(event),
        "made_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "grid": {"crs": str(prof["crs"]), "transform": list(prof["transform"])[:6],
                 "width": prof["width"], "height": prof["height"], "cell_km2": km2},
        "valid_km2": round(float(valid.sum() * km2), 3),
        "families": {f: {"name": FAMILY_NAMES[f], "model": members[f][0],
                         "trainset": members[f][1], "valid_iou": round(members[f][2], 4),
                         "threshold": diags[f]["threshold"],
                         "flood_km2": round(diags[f]["flood_cells"] * km2, 4),
                         "permanent_removed_km2": round(diags[f]["permanent_removed_cells"] * km2, 4),
                         "distortion_removed_km2": round(diags[f]["distortion_removed_cells"] * km2, 4),
                         "map": f"flood_{f}.tif"} for f in maps},
        "headline": {"family": head, "name": FAMILY_NAMES[head], **head_rec},
        "agreement_km2": {str(k): round(float((agree == k).sum() * km2), 4)
                          for k in range(1, len(maps) + 1)},
        "areas": areas(headmap, trust, reach, km2),
        "reach_split_x": split_x,
        "trust": {"limits": {"high": TRUST_HIGH, "medium": TRUST_MED},
                  "model": members[head][0],
                  "table_d_precision": ttable,
                  "radar_blind_km2": round(float((trust == 0).sum() * km2), 4),
                  "rule": "lowest Table D precision (GRD, mean of 2021 and 2023) of the cell's "
                          "incidence, HAND and valley width classes; layover or shadow is radar "
                          "blind"},
        "villages": village_exposure(gp, headmap, trust, prof, km2),
        "roads": road_exposure(gp, headmap, prof),
        "validation": validation(gp, event, head, members),
        "files": ["flood_headline.tif", "flood_headline.gpkg", "agreement.tif", "trust.tif"]
                 + [f"flood_{f}.tif" for f in maps],
        "flood_polygons": n_poly,
        "model_files": {f: model_files(gp, members[f][0], members[f][1]) for f in maps},
        "seconds": round(time.time() - t0, 1),
    }
    mc.save_json(summ_p, summ)
    print(f"    headline {FAMILY_NAMES[head]}: {summ['areas']['total_km2']:.3f} km2; trust "
          f"high {summ['areas']['by_trust']['high']:.3f}, medium "
          f"{summ['areas']['by_trust']['medium']:.3f}, low {summ['areas']['by_trust']['low']:.3f} km2")
    if summ["villages"]:
        vs = summ["villages"]
        print(f"    villages touching the flood: {vs['count']} (population 2011: "
              f"{vs['population_2011']:,}); with at least {vs['min_flood_km2'] * 100:g} ha of "
              f"flood: {vs['count_min_area']} ({vs['population_min_area']:,})")
    print(f"    wrote {out}", flush=True)
    return summ


def validation(gp, event, head, members):
    """Tables A and D (pooled and per reach) for validated events, from the scorer's files."""
    y = event.validated_year
    if not y:
        return None
    out = {"year": y, "table_a": {}, "table_d": {}}
    for f, (model, ts, _) in members.items():
        for key, folder in (("table_a", f"{y}_{model}_{ts}"), ("table_d", f"lowo_{y}_{model}_grd")):
            p = gp.tables / folder / "metrics_by_stratum.csv"
            if not p.exists():
                continue
            with open(p, newline="", encoding="utf-8") as fh:
                rows = [r for r in csv.DictReader(fh) if r["table"] == "reported"
                        and r["dimension"] in ("ALL", "reach")]
            out[key][f] = {r["stratum"]: {k: _num(r[k]) for k in ("f1", "f1_lo", "f1_hi",
                                                                 "precision", "recall")}
                           for r in rows}
    return out


# =============================================================================
# Self-test
# =============================================================================

def add_villages_roads(gp):
    """Four synthetic villages (one without a census figure) and two roads."""
    import geopandas as gpd
    from shapely.geometry import LineString, box
    with rasterio.open(gp.scene(2021, "grd")) as src:
        b = src.bounds
        crs = src.crs
    w, h = b.right - b.left, b.top - b.bottom
    polys = [box(b.left + i * w / 4, b.bottom, b.left + (i + 1) * w / 4, b.top) for i in range(4)]
    polys[3] = box(b.left + 3 * w / 4, b.bottom - 2 * h, b.right + w, b.top)   # mostly outside
    vil = gpd.GeoDataFrame({"village": [f"Gaon {i}" for i in range(4)],
                            "vlcode": [f"0{i}0001" for i in range(4)],
                            "block": "Block", "subdistric": "Devprayag", "district": "Tehri Garhwal",
                            "total_households": [10.0, None, 30.0, 40.0],
                            "total_population_village": [50.0, None, 150.0, 200.0]},
                           geometry=polys, crs=crs).to_crs("EPSG:4326")
    gp.villages.parent.mkdir(parents=True, exist_ok=True)
    vil.to_file(gp.villages, driver="GPKG")
    roads = gpd.GeoDataFrame({"highway": ["primary", "track"], "name": ["NH 7", ""], "ref": ["NH7", ""]},
                             geometry=[LineString([(b.left, b.bottom + h / 2), (b.right, b.bottom + h / 2)]),
                                       LineString([(b.left + w / 2, b.bottom), (b.left + w / 2, b.top)])],
                             crs=crs).to_crs("EPSG:4326")
    gp.roads.parent.mkdir(parents=True, exist_ok=True)
    roads.to_file(gp.roads, driver="GPKG")


def run_scorer(gp, pred, year, out):
    """04_stratified_metrics.py on one map (it reads sys.argv; it is not changed)."""
    argv = ["04", "--pred", str(pred), "--truth", str(gp.reference(year)),
            "--strata", str(gp.strata(year)), "--windows", str(gp.windows(year)),
            "--out", str(out)]
    old = sys.argv
    try:
        sys.argv = argv
        scorer.main()
    finally:
        sys.argv = old


def build_test_project(root):
    """A synthetic project: chips, corridor, both years, a trained RF and U-Net,
    Table D for both, villages and roads."""
    import models_synthetic as syn
    syn.build_project(root)
    assert mc.run_script("21_prepare_chips.py", ["--project", str(root)]) == 0
    assert mc.run_script("22_train_rf.py", ["--project", str(root), "--n-estimators", "10",
                                            "--n-per-class", "200", "--n-jobs", "2"]) == 0
    assert mc.run_script("23_train_unet.py", ["--project", str(root), "--base-width", "8",
                                              "--crop", "32", "--batch-size", "4",
                                              "--steps-per-epoch", "2", "--epochs", "1",
                                              "--tile", "64", "--overlap", "16",
                                              "--workers", "0", "--cpu-ok"]) == 0
    gp = GPaths(root)
    for y in mc.YEARS:
        for m, extra in (("rf", ["--n-estimators", "10", "--n-per-class", "200",
                                 "--inner-estimators", "5"]),
                         ("unet", ["--steps", "4", "--crop", "32", "--batch-size", "2",
                                   "--base-width", "8", "--tile", "64", "--overlap", "16",
                                   "--cpu-ok"])):
            assert mc.run_script("26_corridor_lowo.py", ["--project", str(root), "--year", str(y),
                                                         "--model", m, "--radar", "grd"] + extra) == 0
            run_scorer(gp, gp.lowo_map(y, m, "grd"), y, gp.tables / f"lowo_{y}_{m}_grd")
    add_villages_roads(gp)
    return gp


def selftest():
    print("jalsar_core.py self-test")
    with tempfile.TemporaryDirectory(prefix="jalsar selftest ", ignore_cleanup_errors=True) as td:
        root = Path(td) / "Final Year Project"
        gp = build_test_project(root)
        evs = load_events(gp)
        ok_events = [e.id for e in evs] == ["2021-10-21_orbit063", "2023-08-16_orbit129"]
        e21 = evs[0]
        s = make_event(gp, e21, cpu_ok=True, tile=64, overlap=16)
        # same map as Table A would make: predict with 25_predict_corridor.py and compare
        rc = mc.run_script("25_predict_corridor.py", ["--project", str(root), "--model", "rf",
                                                      "--trainset", "hand", "--year", "2021"])
        with rasterio.open(gp.flood_map(2021, "rf", "hand")) as a, \
                rasterio.open(gp.event_dir(e21.id) / "flood_rf.tif") as b:
            same_map = rc == 0 and np.array_equal(a.read(1), b.read(1))
        with rasterio.open(gp.event_dir(e21.id) / "trust.tif") as t, \
                rasterio.open(gp.event_dir(e21.id) / "agreement.tif") as g, \
                rasterio.open(gp.event_dir(e21.id) / "flood_headline.tif") as hmap:
            tr, ag, hm = t.read(1), g.read(1), hmap.read(1)
        trust_ok = set(np.unique(tr)) <= {0, 1, 2, 3, 255} and (tr[hm == 255] == 255).all()
        agree_ok = set(np.unique(ag)) <= {0, 1, 2, 255}
        area_ok = abs(sum(s["areas"]["by_trust"].values()) - s["areas"]["total_km2"]) < 1e-3 \
            and abs(sum(s["areas"]["by_reach"].values()) - s["areas"]["total_km2"]) < 1e-3
        v = s["villages"]
        vrows = v["villages"] if v else []
        vil_ok = v is not None and all(r["share_flooded"] <= 1 for r in vrows) and \
            all(r["population_2011"] is None for r in vrows if r["village"] == "Gaon 1")
        # share of the WHOLE village (Gaon 3 reaches far outside the scene), sorted by
        # flood area, and the 1 ha count never above the touching count
        vil_share_ok = v is not None and all(
            abs(r["share_flooded"] - min(r["flood_km2"] / r["village_km2"], 1)) < 1e-3
            for r in vrows) and \
            [r["flood_km2"] for r in vrows] == sorted((r["flood_km2"] for r in vrows), reverse=True) \
            and v["count_min_area"] <= v["count"] and \
            any(r["village"] == "Gaon 3" and r["share_flooded"] < 0.34 for r in vrows)
        rd = s["roads"]
        road_ok = rd is not None and set(rd["by_class_km"]) <= {"primary", "track"}
        head_ok = s["headline"]["family"] in ("rf", "unet")
        again = make_event(gp, e21)["made_utc"] == s["made_utc"]
        import geopandas as gpd
        poly = gpd.read_file(gp.event_dir(e21.id) / "flood_headline.gpkg")
        poly_ok = (len(poly) == s["flood_polygons"] and set(poly["trust"]) <= {1, 2, 3}
                   and abs(poly.area.sum() / 1e6 - s["areas"]["total_km2"]) < 1e-3)
        hash_ok = all(len(h) == 64 for f in s["model_files"].values() for h in f.values()) \
            and all(s["model_files"].values())
        g = trust_grade
        grade_ok = (g(0.5), g(0.4), g(0.3), g(0.2), g(0.1), g(float("nan"))) == (3, 3, 2, 2, 1, 1)
    checks = [("events: the two on disk", ok_events),
              ("RF map identical to 25_predict_corridor.py", same_map),
              ("trust layer: 0 to 3 inside, 255 outside", trust_ok),
              ("agreement: 0 to 2 for two families", agree_ok),
              ("areas add up by trust class and by reach", area_ok),
              ("villages: shares <= 1; no census figure kept as none", vil_ok),
              ("villages: share of the whole village, sorted by flood area, 1 ha count",
               vil_share_ok),
              ("roads measured by class", road_ok),
              ("a headline family chosen", head_ok),
              ("trust limits 0.40 and 0.20", grade_ok),
              ("a second run keeps the event", again),
              ("flood polygons: same area, trust classes 1 to 3", poly_ok),
              ("model files recorded with SHA-256", hash_ok)]
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
