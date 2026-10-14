"""Run every tool against the synthetic corridor and check the known answers.

The point of a synthetic case is that the right answer is known in advance,
so a pass means the tool measured the truth rather than merely finishing.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ENV = HERE / "test.env"
OUT = HERE / "out"

# ground truth, from make_synthetic.py
TRUE_WATER_KM2 = 2.27          # channel + reservoir + tributary
TRUE_LENGTH_KM = (14.0, 17.5)  # a 14 km grid with meanders
TRUE_RESERVOIR_WIDTH_M = 600
BED_WIDTH_M = 280              # test polygon, against 80 m of water


def run(script, *args):
    r = subprocess.run([sys.executable, str(PKG / script), "--config", str(ENV),
                        *args], cwd=PKG, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def man(*parts):
    p = OUT.joinpath(*parts) / "run_manifest.json"
    return json.loads(p.read_text(encoding="utf-8"))["parameters"]


def main():
    results = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))

    print("building the synthetic corridor")
    subprocess.run([sys.executable, str(HERE / "make_synthetic.py")], check=True)
    subprocess.run([sys.executable, str(HERE / "make_test_env.py")], check=True)
    if OUT.exists():
        import shutil
        shutil.rmtree(OUT)

    for script, label in [
        ("t01_centreline.py", "t01 centreline"),
        ("t02_mosaic.py", "t02 mosaic"),
        ("t03_watermask_audit.py", "t03 watermask audit"),
        ("t04_reservoir.py", "t04 reservoir"),
        ("t05_qgis_project.py", "t05 qgis project"),
        ("t06_draft_polygons.py", "t06 draft polygons"),
    ]:
        code, out = run(script, "--force")
        check(f"{label} exits clean", code == 0, f"exit {code}")
        if code != 0:
            print(out[-2500:])

    code, out = run("t07_download.py", "selftest")
    check("t07 self-test", code == 0, f"exit {code}")

    # --- the answers we know in advance --------------------------------
    try:
        p = man("centreline")
        lo, hi = TRUE_LENGTH_KM
        check("centreline length is right", lo <= p["length_km"] <= hi,
              f"{p['length_km']} km, expected {lo} to {hi}")
        check("centreline stays inside the corridor",
              p["inside_fraction"] >= 0.999, f"{p['inside_fraction']}")
        check("elevation falls monotonically",
              p["elevation"]["fraction_over_noise"] == 0,
              f"{p['elevation']['fraction_over_noise']} of vertices rise")
        check("two independent routes agree", p["offset_median_m"] < 100,
              f"median {p['offset_median_m']} m")
        check("reservoir found at the right width",
              len(p["wide_reaches_km"]) == 1,
              f"{p['wide_reaches_km']}")
        check("dead-end tributary not entered",
              p["length_km"] <= hi, f"{p['length_km']} km")
        wet = p["dry_scene_wet_fraction"]
        check("terrain shadow rejected in both scenes",
              all(0.035 < v < 0.045 for v in wet.values()),
              f"{ {k: round(v, 4) for k, v in wet.items()} }; "
              f"shadow would add about 0.021")
    except Exception as e:
        check("centreline manifest readable", False, f"{type(e).__name__}: {e}")

    try:
        import pandas as pd
        df = pd.read_csv(OUT / "mosaic" / "mosaic_report.csv")
        r = df.iloc[0]
        check("mosaic fills both cloud gaps", r["mosaic_clear_pct"] >= 99.99,
              f"{r['mosaic_clear_pct']} %")
        check("no gap left", r["gap_km2"] == 0, f"{r['gap_km2']} km2")
        check("primary date wins where it is clear",
              r["won_pct[SY_dry_a]"] > 90, f"{r['won_pct[SY_dry_a]']} %")
    except Exception as e:
        check("mosaic report readable", False, f"{type(e).__name__}: {e}")

    try:
        import pandas as pd
        df = pd.read_csv(OUT / "watermask_audit" / "watermask_audit.csv")
        r = df.iloc[0]
        expect = BED_WIDTH_M / 80.0
        # the polygon is a 140 m buffer of the channel, so it misses the
        # reservoir wings and the tributary. intersection + omission is what
        # must equal the true water area, not the intersection alone.
        inter = r["area_km2"] * r["wet_fraction"]
        check("water accounted for exactly: intersection + omission",
              abs(inter + r["omission_km2"] - TRUE_WATER_KM2) < 0.05,
              f"{inter:.3f} inside + {r['omission_km2']:.3f} outside = "
              f"{inter + r['omission_km2']:.3f} km2, true water {TRUE_WATER_KM2}")
        check("over-wide polygon is caught",
              r["wet_fraction"] < 0.65, f"wet_fraction {r['wet_fraction']}")
        if "width_ratio" in df.columns and pd.notna(r.get("width_ratio")):
            check("width ratio near the built-in 280/80",
                  1.5 <= r["width_ratio"] <= 5.0,
                  f"{r['width_ratio']}, geometric expectation {expect:.1f}")
    except Exception as e:
        check("audit report readable", False, f"{type(e).__name__}: {e}")

    try:
        p = man("reservoir")
        check("reservoir called stable (both dates identical)", p["stable"],
              f"spread {p['spread_pct']} %")
    except Exception as e:
        check("reservoir manifest readable", False, f"{type(e).__name__}: {e}")

    try:
        import fiona
        g = OUT / "qgis" / "sy_dry" / "reference_sy_dry.gpkg"
        layers = set(fiona.listlayers(g))
        check("QGIS GeoPackage has the two real tables",
              {"water_event", "water_dry"} <= layers, str(sorted(layers)))
        with fiona.open(g, layer="water_event") as src:
            check("schema is Polygon with a window field",
                  src.schema["geometry"] == "Polygon"
                  and "window" in src.schema["properties"],
                  str(src.schema))
            check("GeoPackage CRS is the metric one",
                  "32644" in str(src.crs), str(src.crs))
    except Exception as e:
        check("GeoPackage readable", False, f"{type(e).__name__}: {e}")

    try:
        import fiona
        g = OUT / "draft_polygons" / "draft_sy.gpkg"
        layers = set(fiona.listlayers(g))
        check("drafts land in _draft tables only",
              all(l.endswith("_draft") for l in layers) and layers,
              str(sorted(layers)))
        check("drafts never touch water_event or water_dry",
              not ({"water_event", "water_dry"} & layers), str(sorted(layers)))
    except Exception as e:
        check("draft GeoPackage readable", False, f"{type(e).__name__}: {e}")

    print()
    print("-" * 72)
    width = max(len(r[0]) for r in results)
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name.ljust(width)}  {detail}")
    bad = sum(1 for r in results if not r[1])
    print()
    print(f"  {len(results) - bad} passed, {bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
