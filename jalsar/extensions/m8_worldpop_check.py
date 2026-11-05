"""Jal-SAR extensions, M7 and M8 check (22 Oct 2026): why do the WorldPop totals disagree?

  05 Data 100 m file (used for M7 people counts and the web map)   245,259 in the record
  new age and sex file (M8, Earth Engine, same grid)               535,762
  v13 context: WorldPop 2020 summed on its own grid in the corridor 327,395

Prints, read only:
 1. each local file: sum inside the corridor polygon, sum over the whole file, valid cells
 2. Earth Engine: WorldPop/GP/100m/pop and pop_age_sex 'population', IND 2020, summed in the
    corridor on their own grid (about 93 m) and resampled onto our 100 m grid
 3. how many different values the age and sex shares take inside the corridor (few values
    means WorldPop copies the shares from a few admin areas)
Run: python m8_worldpop_check.py
"""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
COR = ROOT / "04 Code" / "FYP_code" / "data" / "corridor.geojson"
POP = ROOT / "05 Data" / "population" / "POP_WorldPop_2020_corridor_100m.tif"
AGE = ROOT / "06 Outputs" / "jalsar" / "extensions" / "m8_vulnerability" / "worldpop_agesex_2020_100m.tif"
EE_PROJECT = os.environ.get("EE_PROJECT", "your-earth-engine-project")


def local(path, cor):
    import rasterio
    from rasterio.features import geometry_mask
    with rasterio.open(path) as src:
        a = src.read(1).astype("float64")
        nd = src.nodata
        if nd is not None and not (isinstance(nd, float) and np.isnan(nd)):
            a[a == nd] = np.nan
        a[a < 0] = np.nan
        geoms = [g.__geo_interface__ for g in cor.to_crs(src.crs).geometry]
        inside = ~geometry_mask(geoms, out_shape=a.shape, transform=src.transform, all_touched=False)
        names = list(src.descriptions)
        full = src.read() if src.count > 1 else None
    print(f"{path.name}: crs {src.crs}, cell {src.transform.a:g} m, {a.shape[1]} x {a.shape[0]}")
    print(f"   sum inside corridor {np.nansum(a[inside]):,.0f}; whole file {np.nansum(a):,.0f}; "
          f"valid cells inside {int(np.isfinite(a[inside]).sum()):,}, outside {int(np.isfinite(a[~inside]).sum()):,}")
    return inside, full, names


def shares(full, names, inside):
    idx = {n: i for i, n in enumerate(names)}
    need = ["population", "M_0", "M_1", "F_0", "F_1"]
    if full is None or any(n not in idx for n in need):
        print("   age and sex bands not found; share check skipped")
        return
    pop = full[idx["population"]].astype("float64")
    ok = inside & np.isfinite(pop) & (pop > 1)
    u5 = sum(full[idx[n]] for n in ("M_0", "M_1", "F_0", "F_1"))[ok] / pop[ok]
    fem = sum(full[i] for n, i in idx.items() if n.startswith("F_"))[ok] / pop[ok]
    o65 = sum(full[i] for n, i in idx.items() if n[:2] in ("M_", "F_") and int(n[2:]) >= 65)[ok] / pop[ok]
    for name, v in (("share under 5", u5), ("share of women", fem), ("share 65 and over", o65)):
        r = np.round(v, 4)
        vals, counts = np.unique(r, return_counts=True)
        top = sorted(zip(counts, vals), reverse=True)[:5]
        print(f"   {name}: {len(vals)} different values in {ok.sum():,} cells; most common "
              + ", ".join(f"{val:.4f} ({c:,} cells)" for c, val in top))


def earth_engine(cor):
    import ee
    try:
        ee.Initialize(project=EE_PROJECT)
    except Exception as exc:
        print(f"Earth Engine: could not connect ({exc}); part 2 skipped")
        return
    geom = ee.Geometry(cor.to_crs("EPSG:4326").union_all().__geo_interface__)
    with __import__("rasterio").open(POP) as src:
        crs, t = str(src.crs), src.transform
    for coll, band in (("WorldPop/GP/100m/pop", "population"), ("WorldPop/GP/100m/pop_age_sex", "population")):
        img = (ee.ImageCollection(coll).filter(ee.Filter.eq("country", "IND"))
               .filter(ee.Filter.eq("year", 2020)).mosaic().select(band))
        try:
            proj = ee.ImageCollection(coll).filter(ee.Filter.eq("country", "IND")).first().select(band).projection()
            native = img.reduceRegion(ee.Reducer.sum(), geom, crs=proj, maxPixels=1e10).get(band).getInfo()
            ours = img.reduceRegion(ee.Reducer.sum(), geom, crs=crs, crsTransform=[t.a, 0, t.c, 0, t.e, t.f],
                                    maxPixels=1e10).get(band).getInfo()
            scale = proj.nominalScale().getInfo()
            print(f"{coll}: corridor sum on its own grid ({scale:.1f} m) {native:,.0f}; "
                  f"sampled onto our 100 m grid {ours:,.0f}")
        except Exception as exc:
            print(f"{coll}: failed ({str(exc)[:200]})")


def main():
    import geopandas as gpd
    cor = gpd.read_file(COR)
    print(f"corridor: {COR.name}, {cor.to_crs(32644).area.sum() / 1e6:,.2f} km2")
    for p in (POP, AGE):
        if not p.exists():
            print(f"missing: {p}")
            continue
        inside, full, names = local(p, cor)
        if p == AGE:
            shares(full, names, inside)
    earth_engine(cor)
    print("DONE  m8_worldpop_check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
