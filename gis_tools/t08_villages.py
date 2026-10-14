"""
t08_villages.py  --  turn the Survey of India village boundary KMZ into a
usable vector layer, and cut it to the corridor.

Written 9 September 2026, after vb_soi_uk.kmz was downloaded.

WHAT THE FILE IS

  vb_soi_uk.kmz is 16,920 village polygons covering all 13 districts of
  Uttarakhand, in WGS84 lon/lat. Every polygon carries 73 attributes in an
  HTML table inside its <description> element, which is why nothing reads it
  as a table until it has been converted. The fields that matter here:

      village, vlcode, block, subdistric, district, total_urban_rural,
      total_households, total_population_village, total_geographical_area

  Fill rates measured on the whole file: village, vlcode, subdistric,
  district and total_urban_rural are 100 per cent; total_households and
  total_population_village are 93.7 per cent; total_geographical_area is
  98.2 per cent; block is 99.2 per cent.

  That 93.7 per cent matters: the settlement exposure layer can come from
  this file directly. WorldPop becomes a cross-check rather than the source.

DELIBERATELY NOT USING GDAL'S KML DRIVER

  The LIBKML driver is not present in every conda build, and where it is, it
  returns the attributes as one HTML blob rather than as fields. This reads
  the KML with the standard library instead, so it works in any environment
  that has shapely and geopandas.

HOW TO RUN

  conda activate fyp
  cd "<project folder>\\04 Code\\gis_tools"

  python t08_villages.py --selftest

  python t08_villages.py ^
    --kmz "<path to>\\vb_soi_uk.kmz" ^
    --corridor "..\\..\\05 Data\\boundaries\\corridor.geojson" ^
    --out "..\\..\\05 Data\\boundaries"

OUTPUTS, written to --out

  VILLAGES_corridor.gpkg      villages whose polygon meets the corridor
  VILLAGES_corridor.geojson   the same, for anything that will not read gpkg
  VILLAGES_corridor.csv       attributes only, no geometry
  VILLAGES_report.md          counts, fill rates and the district breakdown

  Add --statewide to also write VILLAGES_uttarakhand.gpkg, all 16,920.
"""
import argparse
import csv
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter

KML_NS = "{http://www.opengis.net/kml/2.2}"

KEEP = [
    "village", "vlcode", "gram_panchayat_name", "block", "bkcode",
    "subdistric", "sdcode", "district", "dtcode", "state",
    "total_urban_rural", "total_households", "total_population_village",
    "total_geographical_area",
]
NUMERIC = {"total_households", "total_population_village",
           "total_geographical_area"}

ROW_RE = re.compile(r"<td>(.*?)</td>\s*<td>(.*?)</td>", re.S)


def say(msg=""):
    print(msg, flush=True)


def rule(title):
    say()
    say("-" * 72)
    say(title)
    say("-" * 72)


# ------------------------------------------------------------------ parsing
def open_kml(path):
    """Return a file object for the KML inside a .kmz, or the .kml itself."""
    if path.lower().endswith(".kmz"):
        z = zipfile.ZipFile(path)
        names = [n for n in z.namelist() if n.lower().endswith(".kml")]
        if not names:
            raise RuntimeError("%s contains no .kml" % os.path.basename(path))
        return z.open(names[0])
    return open(path, "rb")


def parse_attrs(desc_text):
    if not desc_text:
        return {}
    out = {}
    for k, v in ROW_RE.findall(desc_text):
        k = re.sub(r"<[^>]+>", "", k).strip()
        v = re.sub(r"<[^>]+>", "", v).strip()
        if k:
            out[k] = v
    return out


def ring(coord_text):
    pts = []
    for tok in coord_text.split():
        parts = tok.split(",")
        if len(parts) >= 2:
            try:
                pts.append((float(parts[0]), float(parts[1])))
            except ValueError:
                pass
    return pts


def placemark_polygon(el):
    """Build a shapely geometry from one Placemark. Returns None if unusable."""
    from shapely.geometry import Polygon, MultiPolygon
    polys = []
    for poly in el.iter(KML_NS + "Polygon"):
        outer = None
        holes = []
        for b in poly.findall(KML_NS + "outerBoundaryIs"):
            c = b.find(".//" + KML_NS + "coordinates")
            if c is not None and c.text:
                outer = ring(c.text)
        for b in poly.findall(KML_NS + "innerBoundaryIs"):
            c = b.find(".//" + KML_NS + "coordinates")
            if c is not None and c.text:
                h = ring(c.text)
                if len(h) >= 4:
                    holes.append(h)
        if outer and len(outer) >= 4:
            try:
                p = Polygon(outer, holes)
                if not p.is_valid:
                    p = p.buffer(0)
                if not p.is_empty:
                    polys.append(p)
            except Exception:
                pass
    if not polys:
        return None
    return polys[0] if len(polys) == 1 else MultiPolygon(
        [g for p in polys for g in (p.geoms if p.geom_type == "MultiPolygon" else [p])])


def read_villages(kml_fp, log_every=4000):
    """Stream the KML. Yields (attrs dict, shapely geometry)."""
    n = 0
    for ev, el in ET.iterparse(kml_fp, events=("end",)):
        if el.tag != KML_NS + "Placemark":
            continue
        n += 1
        d = el.find(KML_NS + "description")
        attrs = parse_attrs(d.text if d is not None else "")
        if not attrs.get("village"):
            nm = el.find(KML_NS + "name")
            if nm is not None and nm.text:
                attrs["village"] = nm.text.strip()
        geom = placemark_polygon(el)
        el.clear()
        if geom is not None:
            yield attrs, geom
        if n % log_every == 0:
            say("    read %d placemarks" % n)


def tidy(attrs):
    rec = {}
    for k in KEEP:
        v = attrs.get(k, "")
        if k in NUMERIC:
            try:
                v = float(str(v).replace(",", "")) if str(v).strip() else None
            except ValueError:
                v = None
        rec[k] = v
    return rec


# ------------------------------------------------------------------ selftest
def selftest():
    rule("t08_villages self-test")
    ok = True

    def check(label, cond, note=""):
        nonlocal ok
        say("  %-5s %-42s %s" % ("PASS" if cond else "FAIL", label, note))
        ok = ok and cond

    check("python >= 3.9", sys.version_info >= (3, 9),
          "%d.%d.%d" % sys.version_info[:3])
    for mod in ("shapely", "geopandas", "pyproj"):
        try:
            __import__(mod)
            check("import " + mod, True)
        except Exception as e:
            check("import " + mod, False, str(e)[:40])
    # geopandas writes through either engine, depending on its version
    engine = None
    for mod in ("pyogrio", "fiona"):
        try:
            __import__(mod)
            engine = mod
            break
        except Exception:
            pass
    check("a vector write engine is present", engine is not None,
          engine or "install pyogrio or fiona")

    # a two-village KML built here, whose answers are known in advance
    sample = ('<?xml version="1.0" encoding="UTF-8"?>'
              '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
              '<Placemark><name>Alpha</name><description><![CDATA['
              '<table><tr><td>village</td><td>Alpha</td></tr>'
              '<tr><td>vlcode</td><td>000001</td></tr>'
              '<tr><td>district</td><td>Rudra Prayag</td></tr>'
              '<tr><td>total_population_village</td><td>250</td></tr>'
              '<tr><td>total_households</td><td>50</td></tr></table>]]></description>'
              '<Polygon><outerBoundaryIs><LinearRing><coordinates>'
              '0,0 0,1 1,1 1,0 0,0'
              '</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>'
              '<Placemark><name>Beta</name><description><![CDATA['
              '<table><tr><td>village</td><td>Beta</td></tr>'
              '<tr><td>vlcode</td><td>000002</td></tr>'
              '<tr><td>district</td><td>Tehri Garhwal</td></tr>'
              '<tr><td>total_population_village</td><td></td></tr></table>]]></description>'
              '<Polygon><outerBoundaryIs><LinearRing><coordinates>'
              '5,5 5,6 6,6 6,5 5,5'
              '</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>'
              '</Document></kml>')
    import io
    rows = list(read_villages(io.BytesIO(sample.encode())))
    check("two placemarks parsed", len(rows) == 2, "%d" % len(rows))
    if len(rows) == 2:
        a = tidy(rows[0][0])
        b = tidy(rows[1][0])
        check("attribute table read from description",
              a["village"] == "Alpha" and a["district"] == "Rudra Prayag")
        check("numeric field coerced", a["total_population_village"] == 250.0,
              repr(a["total_population_village"]))
        check("empty numeric becomes None, not zero",
              b["total_population_village"] is None,
              repr(b["total_population_village"]))
        check("polygon area is 1 square degree",
              abs(rows[0][1].area - 1.0) < 1e-9, "%.6f" % rows[0][1].area)
        check("the two polygons do not touch",
              not rows[0][1].intersects(rows[1][1]))

    say()
    say("  %s" % ("all checks passed" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kmz", help="vb_soi_uk.kmz, or an already extracted .kml")
    ap.add_argument("--corridor", help="corridor polygon, any format geopandas reads")
    ap.add_argument("--out", help="output folder")
    ap.add_argument("--statewide", action="store_true",
                    help="also write all 16,920 villages, not just the corridor")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()
    for need in ("kmz", "corridor", "out"):
        if not getattr(a, need):
            ap.error("--%s is required unless --selftest is given" % need)
    if selftest() != 0:
        say()
        say("self-test failed. Nothing was written. Fix the FAIL lines above.")
        return 1

    import geopandas as gpd
    from shapely.geometry import shape

    os.makedirs(a.out, exist_ok=True)

    rule("1. reading the KMZ")
    say("  %s" % a.kmz)
    recs, geoms = [], []
    with open_kml(a.kmz) as fp:
        for attrs, geom in read_villages(fp):
            recs.append(tidy(attrs))
            geoms.append(geom)
    say("  %d village polygons" % len(recs))
    if not recs:
        say("  nothing was read. Is this the right file?")
        return 1

    gdf = gpd.GeoDataFrame(recs, geometry=geoms, crs="EPSG:4326")

    rule("2. fill rates")
    fills = {}
    for k in KEEP:
        nonnull = gdf[k].notna().sum() if k in NUMERIC else \
            (gdf[k].astype(str).str.strip() != "").sum()
        fills[k] = 100.0 * nonnull / len(gdf)
        say("  %-28s %6.1f %%" % (k, fills[k]))

    rule("3. cutting to the corridor")
    cor = gpd.read_file(a.corridor).to_crs("EPSG:4326")
    corridor = cor.union_all() if hasattr(cor, "union_all") else cor.unary_union
    sel = gdf[gdf.intersects(corridor)].copy()
    say("  %d of %d villages meet the corridor" % (len(sel), len(gdf)))

    # area of each village actually inside the corridor, in km2, in a metric CRS
    m = sel.to_crs("EPSG:32644")
    cm = gpd.GeoSeries([corridor], crs="EPSG:4326").to_crs("EPSG:32644").iloc[0]
    sel["area_km2"] = (m.geometry.area / 1e6).round(4).values
    sel["area_in_corridor_km2"] = (m.geometry.intersection(cm).area / 1e6).round(4).values
    sel["fraction_in_corridor"] = (
        sel["area_in_corridor_km2"] / sel["area_km2"].replace(0, float("nan"))
    ).round(4)

    by_dist = Counter(sel["district"])
    for d, c in by_dist.most_common():
        say("    %-18s %5d" % (d, c))

    pop = sel["total_population_village"].dropna()
    say("  population field present for %d of %d selected villages"
        % (len(pop), len(sel)))
    if len(pop):
        say("  their population sums to %d" % int(pop.sum()))

    rule("4. writing")
    gpkg = os.path.join(a.out, "VILLAGES_corridor.gpkg")
    gj = os.path.join(a.out, "VILLAGES_corridor.geojson")
    csvp = os.path.join(a.out, "VILLAGES_corridor.csv")
    sel.to_file(gpkg, layer="villages", driver="GPKG")
    sel.to_file(gj, driver="GeoJSON")
    sel.drop(columns="geometry").to_csv(csvp, index=False)
    for p in (gpkg, gj, csvp):
        say("  wrote %s" % p)

    if a.statewide:
        sw = os.path.join(a.out, "VILLAGES_uttarakhand.gpkg")
        gdf.to_file(sw, layer="villages", driver="GPKG")
        say("  wrote %s" % sw)

    rep = os.path.join(a.out, "VILLAGES_report.md")
    with open(rep, "w", encoding="utf-8") as f:
        f.write("# Village boundaries, Survey of India, Uttarakhand\n\n")
        f.write("Source file: `%s`\n\n" % os.path.basename(a.kmz))
        f.write("Read %d village polygons across %d districts. "
                "%d of them meet the corridor.\n\n"
                % (len(gdf), gdf['district'].nunique(), len(sel)))
        f.write("## Attribute fill rates, whole state\n\n")
        f.write("| field | filled |\n|---|---|\n")
        for k in KEEP:
            f.write("| `%s` | %.1f %% |\n" % (k, fills[k]))
        f.write("\n## Villages meeting the corridor, by district\n\n")
        f.write("| district | villages |\n|---|---|\n")
        for d, c in by_dist.most_common():
            f.write("| %s | %d |\n" % (d, c))
        if len(pop):
            f.write("\nPopulation is present for %d of the %d selected "
                    "villages and sums to %d.\n" % (len(pop), len(sel), int(pop.sum())))
        f.write("\n**`fraction_in_corridor` is the share of each village "
                "polygon that falls inside the corridor.** A village at 0.05 "
                "is clipped by the corridor edge and its population should "
                "not be counted whole.\n")
    say("  wrote %s" % rep)

    say()
    say("  done. read %s" % rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
