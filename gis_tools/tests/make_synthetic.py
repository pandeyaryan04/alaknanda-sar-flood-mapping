"""Build a small synthetic corridor so t01 can be tested end to end.

A 4 x 14 km valley in EPSG:32644 with:
  - a meandering channel from west to east
  - a wide reservoir in the middle, to check the route takes its centre
  - a dead-end tributary, to check the route does not enter it
  - valley walls, so the terrain-only cross-check has something to follow
  - one scene with a cloud patch, to check the union fills it
"""
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from pyproj import Transformer

OUT = Path(__file__).resolve().parent / "synthetic"
RES = 10.0
W, H = 1400, 400          # cells -> 14 km x 4 km
X0, Y0 = 700000.0, 3360000.0   # UTM 44N, near the real corridor
CRS = "EPSG:32644"
TR = from_origin(X0, Y0, RES, RES)


def channel_centre(col):
    """Row of the channel centre at each column: a sine plus a slow drift."""
    return 200 + 55 * np.sin(col / 130.0) + 18 * np.sin(col / 37.0)


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    cols = np.arange(W)
    rows = np.arange(H)
    CC, RR = np.meshgrid(cols, rows)

    centre = channel_centre(CC)
    dist = np.abs(RR - centre)

    half = np.full((H, W), 4.0)                  # 80 m wide channel
    res_lo, res_hi = 620, 780
    half[:, res_lo:res_hi] = 30.0                # 600 m wide reservoir
    ramp = np.linspace(4, 30, 40)
    half[:, res_lo - 40:res_lo] = ramp[None, :]
    half[:, res_hi:res_hi + 40] = ramp[::-1][None, :]

    water = dist <= half

    # dead-end tributary joining from the north at col 1000
    tc = 1000
    for k in range(120):
        r = int(channel_centre(tc)) - 3 - k
        c = tc + int(0.35 * k)
        if 0 <= r < H and 0 <= c < W:
            water[max(0, r - 3):r + 4, max(0, c - 3):c + 4] = True

    # terrain: falls east, valley cut around the channel, walls either side
    base = 800.0 - 0.030 * CC * RES / 10.0
    valley = 260.0 * np.clip((dist - half) / 160.0, 0, 1) ** 1.4
    dem = (base + valley).astype("float32")
    dem[water] = base[water] - 2.0

    def write(path, arr, dtype, count, nodata=None):
        with rasterio.open(
            path, "w", driver="GTiff", height=H, width=W, count=count,
            dtype=dtype, crs=CRS, transform=TR, nodata=nodata, compress="deflate",
        ) as d:
            if count == 1:
                d.write(arr, 1)
            else:
                for i in range(count):
                    d.write(arr[i], i + 1)

    write(OUT / "DEM_synth_30m.tif", dem, "float32", 1)

    scenes = ["SY_dry_a", "SY_dry_b"]
    for si, tag in enumerate(scenes):
        mndwi = np.where(water, 0.45, -0.35).astype("float32")
        # terrain shadow: dark, high MNDWI, but SWIR is not low -> must be rejected
        shadow = (RR < 60) & (CC > 300) & (CC < 500)
        mndwi[shadow] = 0.25
        ndwi = mndwi * 0.8
        b11 = np.where(water, 0.012, 0.20).astype("float32")
        b11[shadow] = 0.09
        scl = np.full((H, W), 4, dtype="int16")
        if si == 0:                       # cloud over part of the channel
            scl[:, 900:960] = 9
        else:
            scl[:, 400:440] = 9
        bands = np.stack([
            np.full((H, W), 0.09), np.full((H, W), 0.10),
            np.full((H, W), 0.11), np.full((H, W), 0.25), b11,
        ]) * 10000
        bands = np.concatenate([bands, scl[None, :, :]]).astype("int16")
        write(OUT / f"{tag}_10m.tif", bands, "int16", 6)
        write(OUT / f"{tag}_indices_10m.tif",
              np.stack([mndwi, ndwi]), "float32", 2)

    # corridor polygon: the raster extent inset by 3 cells, in EPSG:4326
    tr = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    pad = 3 * RES
    corners_m = [
        (X0 + pad, Y0 - pad),
        (X0 + W * RES - pad, Y0 - pad),
        (X0 + W * RES - pad, Y0 - H * RES + pad),
        (X0 + pad, Y0 - H * RES + pad),
    ]
    ring = [list(tr.transform(x, y)) for x, y in corners_m]
    ring.append(ring[0])
    (OUT / "corridor.geojson").write_text(json.dumps({
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": [{"type": "Feature",
                      "properties": {"name": "synthetic corridor"},
                      "geometry": {"type": "Polygon", "coordinates": [ring]}}],
    }, indent=1), encoding="utf-8")

    # anchors at the two channel ends, deliberately offset by ~200 m so the
    # snap step has something to do
    def anchor(col, drow):
        r = channel_centre(np.array([col]))[0] + drow
        x = X0 + (col + 0.5) * RES
        y = Y0 - (r + 0.5) * RES
        return tr.transform(x, y)

    up = anchor(6, 20)
    down = anchor(W - 7, -20)
    mid = anchor(700, 0)
    info = {
        "ANCHOR_UP": f"{up[0]:.6f},{up[1]:.6f}",
        "ANCHOR_DOWN": f"{down[0]:.6f},{down[1]:.6f}",
        "ANCHOR_MID": f"{mid[0]:.6f},{mid[1]:.6f}",
        "expected_length_km_min": 14.0,
        "expected_length_km_max": 17.5,
    }
    (OUT / "anchors.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info, indent=1))
    print(f"water cells {water.sum()}  ({water.sum()*RES*RES/1e6:.2f} km2)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
