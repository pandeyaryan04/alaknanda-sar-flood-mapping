"""Rebuild tests/test.env from .env.example, pointed at the synthetic corridor.

Keeping the test config a derivative of the real one means a key added to
.env.example is exercised by the tests rather than silently untested.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
SYN = HERE / "synthetic"
OUT = HERE / "out"


def build():
    a = json.loads((SYN / "anchors.json").read_text())
    syn, out = str(SYN), str(OUT)
    repl = {
        "PROJECT_ROOT": syn, "DATA_DIR": syn, "CODE_DIR": syn,
        "OUT_DIR": out, "S2_DIR": syn, "DEM_DIR": syn,
        "LANDCOVER_DIR": syn,
        "CORRIDOR": f"{syn}/corridor.geojson",
        "DEM_ROUTING": f"{syn}/DEM_synth_30m.tif",
        "CENTRELINE_OUT": f"{out}/synth_centreline.geojson",
        "CENTRELINE_COPY": "",
        "RIVER_NAME": "synthetic test river",
        "DRY_SCENES": "SY_dry_a,SY_dry_b",
        "MOSAIC_SETS": "PAIR_SY_DRY,PAIR_EMPTY",
        "QGIS_EVENTS": "PAIR_SY_DRY",
        "ANCHOR_UP": a["ANCHOR_UP"], "ANCHOR_DOWN": a["ANCHOR_DOWN"],
        "ANCHOR_SRINAGAR": a["ANCHOR_MID"],
        "CHECK_TOWNS": "ANCHOR_SRINAGAR",
        "LENGTH_MIN_KM": "14", "LENGTH_MAX_KM": "18",
        "MIN_WATER_HA": "0.5", "SNAP_RADIUS_M": "400",
        "TOWN_TOL_M": "300", "MAX_HALF_WIDTH_M": "1000",
        "WATERMASK_LAYERS": "BRAIDED_BED.geojson",
        "JRC_OCCURRENCE": "",
        "PANEL_SITES": f"{a['ANCHOR_UP']}; {a['ANCHOR_MID']}; {a['ANCHOR_DOWN']}",
        "PANEL_HALF_M": "800",
        "TRANSECT_SPACING_M": "200",
        "RESERVOIR_SCENES": "SY_dry_a,SY_dry_b",
        "DRAFT_SETS": "sy:SY_dry_a",
    }
    lines = []
    for ln in (PKG / ".env.example").read_text().splitlines():
        if "=" in ln and not ln.strip().startswith("#"):
            k = ln.split("=")[0].strip()
            if k in repl:
                lines.append(f"{k} = {repl[k]}")
                continue
        lines.append(ln)
    lines += [
        "", "# test-only additions",
        "PAIR_SY_DRY = SY_dry_a,SY_dry_b",
        "PAIR_EMPTY =",
        f"WINDOWS_SY = {SYN}/windows_sy.geojson",
    ]
    p = HERE / "test.env"
    p.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {p}")


if __name__ == "__main__":
    build()
