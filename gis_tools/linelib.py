"""
linelib.py - turning a pixel path into a usable centreline, and measuring it.

Everything here works in projected metres. Nothing here opens a file.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# pixels to metres
# --------------------------------------------------------------------------

def rc_to_xy(path_rc, transform):
    """Cell centres of a [(row, col), ...] path, in the raster's CRS."""
    import rasterio.transform as rt

    xs, ys = rt.xy(transform, [r for r, _ in path_rc], [c for _, c in path_rc])
    return list(zip(xs, ys))


def xy_to_rc(x, y, transform):
    import rasterio.transform as rt

    r, c = rt.rowcol(transform, x, y)
    return int(r), int(c)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def chainage(coords):
    """Cumulative distance along a polyline, in the units of the coords."""
    c = np.asarray(coords, dtype="float64")
    if len(c) < 2:
        return np.zeros(len(c))
    step = np.hypot(np.diff(c[:, 0]), np.diff(c[:, 1]))
    return np.concatenate([[0.0], np.cumsum(step)])


def length_m(coords):
    return float(chainage(coords)[-1]) if len(coords) > 1 else 0.0


def simplify(coords, tol_m):
    from shapely.geometry import LineString

    if tol_m <= 0 or len(coords) < 3:
        return list(coords)
    g = LineString(coords).simplify(tol_m, preserve_topology=False)
    return list(g.coords)


def chaikin(coords, iterations=2, keep_ends=True):
    """Corner cutting. Two passes takes the staircase off a pixel path
    without pulling the line off the channel."""
    pts = [tuple(p) for p in coords]
    for _ in range(int(iterations)):
        if len(pts) < 3:
            break
        out = [pts[0]] if keep_ends else []
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            out.append((0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1))
            out.append((0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1))
        if keep_ends:
            out.append(pts[-1])
        pts = out
    return pts


def densify(coords, max_step_m):
    """No gap longer than max_step_m.

    05_reference_windows.py walks the line in fixed steps; long segments make
    window spacing lumpy, so the line is densified before it is written.
    """
    if max_step_m <= 0 or len(coords) < 2:
        return list(coords)
    out = [tuple(coords[0])]
    for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
        d = float(np.hypot(x1 - x0, y1 - y0))
        n = int(np.ceil(d / max_step_m))
        for k in range(1, n + 1):
            t = k / n
            out.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    return out


def resample(coords, step_m):
    """Points at a fixed spacing along the line, plus the far end."""
    ch = chainage(coords)
    total = ch[-1]
    if total <= 0:
        return list(coords)
    want = np.arange(0.0, total, step_m)
    c = np.asarray(coords, dtype="float64")
    xs = np.interp(want, ch, c[:, 0])
    ys = np.interp(want, ch, c[:, 1])
    out = list(zip(xs, ys))
    out.append(tuple(c[-1]))
    return out


def tangents(coords):
    """Unit tangent at each vertex, by central difference."""
    c = np.asarray(coords, dtype="float64")
    d = np.gradient(c, axis=0)
    n = np.hypot(d[:, 0], d[:, 1])
    n[n == 0] = 1.0
    return d / n[:, None]


def normals(coords):
    t = tangents(coords)
    return np.stack([-t[:, 1], t[:, 0]], axis=1)


# --------------------------------------------------------------------------
# comparison between two lines
# --------------------------------------------------------------------------

def offsets(line_a, line_b, step_m=100.0):
    """Distance from each sampled point of A to the nearest point on B.

    Used for the water-route against terrain-route cross-check. Not a Frechet
    distance; this is deliberately the simpler and more readable measure,
    and it is the one the QC table reports.
    """
    from shapely.geometry import LineString, Point

    gb = LineString(line_b)
    pts = resample(line_a, step_m)
    return np.array([gb.distance(Point(p)) for p in pts]), pts


# --------------------------------------------------------------------------
# width, measured across the line
# --------------------------------------------------------------------------

def widths(coords, mask, transform, res_m, max_half_m=2000.0, step_m=None):
    """Water width across the line at each vertex, in metres.

    Walks outward along the normal in both directions until the mask stops
    being true. Returns nan where the vertex is not on water at all.
    """
    import rasterio.transform as rt

    nrm = normals(coords)
    step = step_m or res_m
    nmax = int(max_half_m / step)
    out = np.full(len(coords), np.nan)
    h, w = mask.shape
    for i, ((x, y), (nx, ny)) in enumerate(zip(coords, nrm)):
        r, c = rt.rowcol(transform, x, y)
        if not (0 <= r < h and 0 <= c < w) or not mask[r, c]:
            continue
        total = step
        for sign in (1, -1):
            for k in range(1, nmax + 1):
                xx = x + sign * nx * step * k
                yy = y + sign * ny * step * k
                rr, cc = rt.rowcol(transform, xx, yy)
                if not (0 <= rr < h and 0 <= cc < w) or not mask[rr, cc]:
                    break
                total += step
        out[i] = total
    return out


def inside_fraction(coords, polygon):
    """Share of the line's length that falls inside a shapely polygon."""
    from shapely.geometry import LineString

    g = LineString(coords)
    if g.length == 0:
        return float("nan")
    return float(g.intersection(polygon).length / g.length)
