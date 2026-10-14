r"""
09_landslide_merge.py
=====================
Merge the three district landslide inventories into one layer, and remove the
polygons that appear twice because they sit on a shared district border.

Written 9 September 2026. Trap 8 in the project context says: before merging
the three district files, check for duplicates on the shared borders. This does
the check and the merge in one pass, and reports exactly what it removed.

WHY DUPLICATES EXIST

  The inventories are distributed per district. A landslide whose polygon
  straddles the Rudraprayag / Tehri Garhwal boundary is digitised once in each
  file, so a naive concatenation counts it twice. In a susceptibility model
  every duplicated polygon is a duplicated training sample, which quietly
  weights the model toward the district borders.

HOW A DUPLICATE IS DECIDED

  Two polygons from different districts are the same landslide when their
  intersection is at least --overlap of the smaller of the two areas. Default
  0.5. The check runs only across districts, never within one, because a file
  is allowed to hold two genuinely adjacent slides.

  Geometry is compared in a metric CRS, not in degrees. Trap 6 in the project
  context is the same mistake made on an area field.

HOW TO RUN

  conda activate fyp
  cd "<project folder>\\04 Code\\FYP_code"

  python 09_landslide_merge.py --selftest

  python 09_landslide_merge.py ^
    --dir "..\\..\\05 Data\\landslide" ^
    --corridor "data\\corridor.geojson" ^
    --out "..\\..\\05 Data\\landslide"

OUTPUTS

  LANDSLIDE_merged.gpkg          all districts, deduplicated
  LANDSLIDE_corridor.gpkg        the subset meeting the corridor
  LANDSLIDE_duplicates.csv       every pair the check removed, with its overlap
  LANDSLIDE_merge_report.md      counts before and after, per district
"""
import argparse
import glob
import os
import sys

METRIC = "EPSG:32644"


def say(m=""):
    print(m, flush=True)


def rule(t):
    say(); say("-" * 72); say(t); say("-" * 72)


def overlap_fraction(a, b):
    """Intersection as a share of the SMALLER polygon.

    Share of the smaller, not of the union: a small slide fully inside a larger
    one is the same slide digitised at two levels of detail, and intersection
    over union would score that pair low and keep both.
    """
    if not a.intersects(b):
        return 0.0
    inter = a.intersection(b).area
    smaller = min(a.area, b.area)
    return 0.0 if smaller <= 0 else inter / smaller


def find_duplicates(gdf, district_col, threshold):
    """Returns a list of (keep_index, drop_index, fraction), across districts only."""
    import numpy as np
    sindex = gdf.sindex
    seen = set()
    out = []
    dropped = set()
    for i, geom in enumerate(gdf.geometry):
        if i in dropped or geom is None or geom.is_empty:
            continue
        for j in sindex.query(geom, predicate="intersects"):
            j = int(j)
            if j <= i or j in dropped:
                continue
            if gdf[district_col].iloc[i] == gdf[district_col].iloc[j]:
                continue                      # same file: allowed to be adjacent
            key = (i, j)
            if key in seen:
                continue
            seen.add(key)
            f = overlap_fraction(geom, gdf.geometry.iloc[j])
            if f >= threshold:
                # keep the larger polygon: it is the more complete delineation
                keep, drop = (i, j) if geom.area >= gdf.geometry.iloc[j].area else (j, i)
                out.append((keep, drop, f))
                dropped.add(drop)
    return out, dropped


def selftest():
    rule("09_landslide_merge self-test")
    ok = True

    def check(label, cond, note=""):
        nonlocal ok
        say("  %-5s %-46s %s" % ("PASS" if cond else "FAIL", label, note))
        ok = ok and cond

    check("python >= 3.9", sys.version_info >= (3, 9),
          "%d.%d.%d" % sys.version_info[:3])
    for mod in ("geopandas", "shapely", "pandas"):
        try:
            __import__(mod)
            check("import " + mod, True)
        except Exception as e:
            check("import " + mod, False, str(e)[:40])

    from shapely.geometry import box
    import geopandas as gpd

    a = box(0, 0, 10, 10)          # area 100, district A
    b = box(1, 1, 11, 11)          # area 100, district B, overlaps a by 81
    c = box(0, 0, 3, 3)            # area 9, district A, inside a. Against b the
                                   # overlap is 4/9 = 0.44, under the threshold
    d = box(50, 50, 60, 60)        # nowhere near anything
    small_inside = box(3, 3, 5, 5) # area 4, fully inside a

    check("overlap of two 81 per cent boxes",
          abs(overlap_fraction(a, b) - 0.81) < 1e-9,
          "%.4f" % overlap_fraction(a, b))
    check("a small polygon inside a large one scores 1.0",
          abs(overlap_fraction(a, small_inside) - 1.0) < 1e-9,
          "%.4f" % overlap_fraction(a, small_inside))
    check("disjoint polygons score 0",
          overlap_fraction(a, d) == 0.0)

    g = gpd.GeoDataFrame({"district": ["A", "B", "A", "A"]},
                         geometry=[a, b, c, d], crs=METRIC)
    dups, dropped = find_duplicates(g, "district", 0.5)
    check("one cross-district duplicate pair found at 0.5",
          len(dups) == 1, "%d" % len(dups))
    if dups:
        keep, drop, f = dups[0]
        check("the larger polygon is the one kept",
              g.geometry.iloc[keep].area >= g.geometry.iloc[drop].area)
    check("a same-district polygon inside another is kept", 2 not in dropped)
    check("an overlap under the threshold is kept", 2 not in dropped)
    check("a disjoint polygon is kept", 3 not in dropped)

    g2 = gpd.GeoDataFrame({"district": ["A", "A"]}, geometry=[a, b], crs=METRIC)
    d2, _ = find_duplicates(g2, "district", 0.5)
    check("two overlapping polygons in the SAME district are not touched",
          len(d2) == 0, "%d" % len(d2))

    say(); say("  %s" % ("all checks passed" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", help="folder holding the district subfolders")
    ap.add_argument("--corridor", help="corridor polygon, for the subset")
    ap.add_argument("--out", help="output folder")
    ap.add_argument("--overlap", type=float, default=0.5,
                    help="share of the smaller polygon that counts as the same "
                         "landslide, default 0.5")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()
    for need in ("dir", "out"):
        if not getattr(a, need):
            ap.error("--%s is required unless --selftest is given" % need)
    if selftest() != 0:
        say(); say("self-test failed. Nothing was written.")
        return 1

    import geopandas as gpd
    import pandas as pd

    os.makedirs(a.out, exist_ok=True)

    rule("1. reading the district files")
    shps = sorted(glob.glob(os.path.join(a.dir, "**", "*.shp"), recursive=True))
    if not shps:
        say("  no .shp found under %s" % a.dir); return 1
    frames = []
    for p in shps:
        g = gpd.read_file(p)
        name = os.path.basename(os.path.dirname(p)) or os.path.basename(p)
        bad = int((~g.geometry.is_valid).sum())
        empty = int(g.geometry.is_empty.sum())
        say("  %-46s %5d polygons  %s  invalid %d  empty %d"
            % (name[:46], len(g), g.crs, bad, empty))
        if bad:
            g["geometry"] = g.geometry.buffer(0)
        g["_district_file"] = name
        frames.append(g.to_crs(METRIC))
    gdf = pd.concat(frames, ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=METRIC)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].reset_index(drop=True)
    say("  %d polygons before deduplication" % len(gdf))

    rule("2. duplicates across district borders")
    dups, dropped = find_duplicates(gdf, "_district_file", a.overlap)
    say("  %d duplicate pairs at an overlap of %.2f or more" % (len(dups), a.overlap))
    rows = []
    for keep, drop, f in dups:
        rows.append({
            "kept_index": keep, "dropped_index": drop,
            "kept_file": gdf["_district_file"].iloc[keep],
            "dropped_file": gdf["_district_file"].iloc[drop],
            "kept_area_m2": round(float(gdf.geometry.iloc[keep].area), 1),
            "dropped_area_m2": round(float(gdf.geometry.iloc[drop].area), 1),
            "overlap_of_smaller": round(f, 4),
        })
        say("    %-30s x %-30s  %.3f"
            % (rows[-1]["kept_file"][:30], rows[-1]["dropped_file"][:30], f))
    dup_csv = os.path.join(a.out, "LANDSLIDE_duplicates.csv")
    pd.DataFrame(rows).to_csv(dup_csv, index=False)
    say("  wrote %s" % dup_csv)

    merged = gdf.drop(index=sorted(dropped)).reset_index(drop=True)
    say("  %d polygons after deduplication" % len(merged))

    rule("3. writing")
    out_all = os.path.join(a.out, "LANDSLIDE_merged.gpkg")
    merged.to_file(out_all, layer="landslides", driver="GPKG")
    say("  wrote %s" % out_all)

    sel = None
    if a.corridor:
        cor = gpd.read_file(a.corridor).to_crs(METRIC)
        poly = cor.union_all() if hasattr(cor, "union_all") else cor.unary_union
        sel = merged[merged.intersects(poly)].copy()
        out_cor = os.path.join(a.out, "LANDSLIDE_corridor.gpkg")
        sel.to_file(out_cor, layer="landslides", driver="GPKG")
        say("  %d of %d meet the corridor" % (len(sel), len(merged)))
        say("  wrote %s" % out_cor)

    rep = os.path.join(a.out, "LANDSLIDE_merge_report.md")
    with open(rep, "w", encoding="utf-8") as f:
        f.write("# Landslide inventory merge\n\n")
        f.write("Source: %d shapefiles under `%s`.\n\n" % (len(shps), a.dir))
        f.write("| stage | polygons |\n|---|---|\n")
        f.write("| read from the district files | %d |\n" % len(gdf))
        f.write("| removed as cross-border duplicates | %d |\n" % len(dropped))
        f.write("| merged | %d |\n" % len(merged))
        if sel is not None:
            f.write("| meeting the corridor | %d |\n" % len(sel))
        f.write("\n## By district file\n\n| file | before | after |\n|---|---|---|\n")
        before = gdf["_district_file"].value_counts()
        after = merged["_district_file"].value_counts()
        for k in before.index:
            f.write("| %s | %d | %d |\n" % (k, int(before[k]), int(after.get(k, 0))))
        f.write("\n**A duplicate is two polygons from different district files "
                "whose intersection is at least %.0f per cent of the smaller of "
                "the two.** The larger polygon is kept, because it is the more "
                "complete delineation of the same slide. Polygons that overlap "
                "within a single district file are left alone: one file is "
                "allowed to hold two genuinely adjacent slides.\n" % (a.overlap * 100))
        f.write("\nEvery removed pair is listed in `LANDSLIDE_duplicates.csv` "
                "with its overlap, so the decision can be checked rather than "
                "trusted.\n")
    say("  wrote %s" % rep)
    say(); say("  done. read %s" % rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
