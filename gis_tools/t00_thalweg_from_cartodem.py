"""
t00_thalweg_from_cartodem.py

Finds the true Ganga/Alaknanda channel between the two anchors without using the
corridor polygon at all, then measures how far that channel runs outside the
current corridor.

Why this exists
---------------
t01_centreline.py routes inside corridor.geojson. If the corridor does not
contain the river somewhere, the route has to climb out of the valley and the
result is nonsense. This script answers the prior question: where is the river
actually, independent of the corridor.

Input is the unclipped CartoDEM 1 degree tile, because that is the only DEM on
this machine that is not already clipped to the corridor. The Copernicus GLO-30
file in 05 Data\\dem is clipped, so it cannot be used here.

Method
------
1. Subset the tile to the reach.
2. Fill depressions with priority-flood plus a small epsilon, so every cell has
   a strictly downhill neighbour.
3. D8 flow direction by steepest descent, corrected for the fact that a cell is
   30.9 m tall and 26.7 m wide at this latitude.
4. Flow accumulation by processing cells in order of decreasing filled height.
5. Snap the upstream anchor to the highest accumulation cell nearby, then walk
   downstream until the downstream anchor is reached.
6. Report distance from that path to the corridor polygon.

Run it with the fyp environment active:
    conda activate fyp
    python t00_thalweg_from_cartodem.py

Outputs land in 05 Data\\boundaries.
"""

import json
from pathlib import Path
import math
import heapq
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import xy, rowcol
from shapely.geometry import LineString, Point, shape, mapping, box
from shapely.ops import transform as shp_transform
from pyproj import Transformer

# ----------------------------------------------------------------- settings
# the project folder, two levels above this file (04 Code\gis_tools\), so the
# script runs from wherever the project is copied
ROOT = str(Path(__file__).resolve().parents[2])
DEM_TILE = (ROOT + r"\05 Data\dem\cartodem\P5_PAN_CD_N30_000_E078_000_30m"
                   r"\P5_PAN_CD_N30_000_E078_000_30m"
                   r"\P5_PAN_CD_N30_000_E078_000_DEM_30m.tif")
CORRIDOR = ROOT + r"\04 Code\FYP_code\data\corridor.geojson"
OUTDIR = ROOT + r"\05 Data\boundaries"

BBOX = (78.20, 30.02, 79.00, 30.40)          # left, bottom, right, top
ANCHOR_UP = (78.9811, 30.2844)
ANCHOR_DOWN = (78.2676, 30.0869)
SNAP_RADIUS_M = 1500
EPS = 1e-4
METRIC = "EPSG:32644"

# The box corridor v2 was clipped to on 16 September: the extent of the 30 m
# grid (COP GLO-30, SRTM, JRC, CartoDEM v2), 2639 x 1016 cells from 231660,
# 3357030. It reaches 20 m further east and south than the 10 m Sentinel grid
# (to 310810 and 3326570). This box gives the 606.35 km2 corridor in use;
# the 10 m box gives 605.99 km2. Decision S2, 7 October 2026.
IMAGERY = box(231660.0, 3326550.0, 310830.0, 3357030.0)

fwd = Transformer.from_crs("EPSG:4326", METRIC, always_xy=True)
inv = Transformer.from_crs(METRIC, "EPSG:4326", always_xy=True)
to_m = lambda g: shp_transform(lambda a, b: fwd.transform(a, b), g)
to_ll = lambda g: shp_transform(lambda a, b: inv.transform(a, b), g)


def read_subset():
    with rasterio.open(DEM_TILE) as src:
        win = from_bounds(*BBOX, transform=src.transform)
        z = src.read(1, window=win).astype("float64")
        T = src.window_transform(win)
    return z, T


def fill_depressions(z):
    """Priority-flood with epsilon. Returns a surface with no interior sinks."""
    nr, nc = z.shape
    filled = np.full(z.shape, np.inf)
    closed = np.zeros(z.shape, bool)
    heap = []
    for r in (0, nr - 1):
        for c in range(nc):
            heapq.heappush(heap, (z[r, c], r, c))
            closed[r, c] = True
            filled[r, c] = z[r, c]
    for c in (0, nc - 1):
        for r in range(1, nr - 1):
            heapq.heappush(heap, (z[r, c], r, c))
            closed[r, c] = True
            filled[r, c] = z[r, c]
    nb = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while heap:
        e, r, c = heapq.heappop(heap)
        for dr, dc in nb:
            rr, cc = r + dr, c + dc
            if 0 <= rr < nr and 0 <= cc < nc and not closed[rr, cc]:
                closed[rr, cc] = True
                new = z[rr, cc] if z[rr, cc] > e + EPS else e + EPS
                filled[rr, cc] = new
                heapq.heappush(heap, (new, rr, cc))
    return filled


def d8(filled, T):
    nr, nc = filled.shape
    lat_mid = (xy(T, 0, 0)[1] + xy(T, nr - 1, 0)[1]) / 2.0
    dy = 30.87
    dx = 30.87 * math.cos(math.radians(lat_mid))
    nb = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    dist = [math.hypot(dc * dx, dr * dy) for dr, dc in nb]

    best = np.full((nr, nc), -1, np.int8)
    slope = np.zeros((nr, nc))
    for k, (dr, dc) in enumerate(nb):
        a = np.full((nr, nc), -np.inf)
        r0, r1 = max(0, -dr), min(nr, nr - dr)
        c0, c1 = max(0, -dc), min(nc, nc - dc)
        a[r0:r1, c0:c1] = (filled[r0:r1, c0:c1]
                           - filled[r0 + dr:r1 + dr, c0 + dc:c1 + dc]) / dist[k]
        m = a > slope
        slope[m] = a[m]
        best[m] = k

    rows, cols = np.divmod(np.arange(nr * nc), nc)
    target = np.full(nr * nc, -1, np.int64)
    flat = best.ravel()
    for k, (dr, dc) in enumerate(nb):
        m = flat == k
        tr, tc = rows[m] + dr, cols[m] + dc
        ok = (tr >= 0) & (tr < nr) & (tc >= 0) & (tc < nc)
        target[np.where(m)[0][ok]] = tr[ok] * nc + tc[ok]

    acc = np.ones(nr * nc)
    for i in np.argsort(filled, axis=None)[::-1]:
        t = target[i]
        if t >= 0:
            acc[t] += acc[i]
    return target, acc.reshape(nr, nc), dx, dy


def snap(acc, T, lon, lat, dx, dy):
    nr, nc = acc.shape
    r, c = rowcol(T, lon, lat)
    pr, pc = int(SNAP_RADIUS_M / dy) + 1, int(SNAP_RADIUS_M / dx) + 1
    r0, r1 = max(0, int(r) - pr), min(nr, int(r) + pr + 1)
    c0, c1 = max(0, int(c) - pc), min(nc, int(c) + pc + 1)
    sub = acc[r0:r1, c0:c1]
    k = np.unravel_index(np.argmax(sub), sub.shape)
    return r0 + k[0], c0 + k[1]


def main():
    z, T = read_subset()
    nr, nc = z.shape
    print("subset %d x %d, elevation %.1f to %.1f m" % (nr, nc, z.min(), z.max()))

    filled = fill_depressions(z)
    print("depressions filled, %d cells raised, largest raise %.1f m"
          % (int((filled > z + 1e-9).sum()), float((filled - z).max())))

    target, acc, dx, dy = d8(filled, T)
    sr, sc = snap(acc, T, *ANCHOR_UP, dx=dx, dy=dy)
    er, ec = snap(acc, T, *ANCHOR_DOWN, dx=dx, dy=dy)

    path, i, seen = [], sr * nc + sc, set()
    while i >= 0 and i not in seen:
        seen.add(i)
        r, c = divmod(i, nc)
        path.append((r, c))
        if (r, c) == (er, ec):
            break
        i = target[i]
    reached = path[-1] == (er, ec)
    print("traced %d cells, reached the downstream anchor: %s" % (len(path), reached))

    pts = [xy(T, r, c) for r, c in path]
    zz = np.array([filled[r, c] for r, c in path])
    aa = np.array([acc[r, c] for r, c in path])

    line_m = to_m(LineString(pts))
    simple = line_m.simplify(60)
    n = int(simple.length // 200) + 1
    smooth = LineString([simple.interpolate(k * simple.length / n) for k in range(n + 1)])

    print("length %.2f km, elevation %.1f down to %.1f m, highest point %.1f m"
          % (smooth.length / 1000, zz[0], zz[-1], zz.max()))
    print("elevation never rises: %s" % bool(np.all(np.diff(zz) <= 1e-6)))
    print("accumulation never falls: %s" % bool(np.all(np.diff(aa) >= 0)))

    dev = (78.5976, 30.1469)
    dd = np.hypot((np.array(pts)[:, 0] - dev[0]) * 96.3,
                  (np.array(pts)[:, 1] - dev[1]) * 110.9) * 1000
    j = int(np.argmax(np.diff(aa)))
    print("closest approach to the Devprayag confluence: %.0f m" % dd.min())
    print("largest tributary join at lon %.4f lat %.4f, adds %.0f cells"
          % (pts[j][0], pts[j][1], np.diff(aa)[j]))

    old = to_m(shape(json.load(open(CORRIDOR))["features"][0]["geometry"]))
    outside, off, seg = [], [], []
    prev = None
    for p in smooth.coords:
        pt = Point(p)
        ins = old.contains(pt)
        outside.append(not ins)
        off.append(0.0 if ins else old.exterior.distance(pt))
        if prev is not None:
            seg.append(math.dist(prev, p))
        prev = p
    outside = np.array(outside); off = np.array(off); seg = np.array(seg)
    m = outside[:-1] | outside[1:]
    print("outside the corridor: %.2f km of channel, worst offset %.0f m, mean offset %.0f m"
          % (seg[m].sum() / 1000, off.max(), off[outside].mean()))

    def write(geom, name, props):
        with open(OUTDIR + "\\" + name, "w") as fh:
            json.dump({"type": "FeatureCollection",
                       "features": [{"type": "Feature", "properties": props,
                                     "geometry": mapping(to_ll(geom))}]}, fh)
        print("wrote " + name)

    write(smooth, "RIVER_thalweg_cartodem.geojson",
          {"source": "CartoDEM 30 m, priority-flood + D8 trace",
           "length_km": round(smooth.length / 1000, 2)})
    for b in (3000, 5000):
        full = smooth.buffer(b)
        clip = full.intersection(IMAGERY)
        if clip.geom_type == "MultiPolygon":
            clip = max(clip.geoms, key=lambda g: g.area)
        write(full, "corridor_v2_%dkm.geojson" % (b // 1000),
              {"buffer_m": b, "area_km2": round(full.area / 1e6, 2)})
        write(clip, "corridor_v2_%dkm_clipped.geojson" % (b // 1000),
              {"buffer_m": b, "clipped_to": "existing imagery footprint",
               "area_km2": round(clip.area / 1e6, 2)})
        gap = LineString([p for p in smooth.coords
                          if 78.41 <= inv.transform(p[0], p[1])[0] <= 78.62])
        patched = old.union(gap.buffer(b))
        write(patched, "corridor_v1_plus_patch%dkm.geojson" % (b // 1000),
              {"note": "current corridor plus a %d m buffer over the gap reach" % b,
               "area_km2": round(patched.area / 1e6, 2)})


if __name__ == "__main__":
    main()
