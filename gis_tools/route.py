"""
route.py - getting a river line out of a DEM and a water mask.

The method here is a least-cost path between two fixed endpoints, not D8
flow accumulation. Reasons, because this differs from the first draft of the
spec and the reason matters:

  1. Accumulation needs the upstream catchment to mean anything. The
     Copernicus DEM on disk is already clipped to the corridor, so its
     accumulation at Rudraprayag would be near zero and any snap based on it
     would be meaningless.
  2. Filled DEMs have flats, and routing across flats is the fiddly part of
     every D8 implementation. A least-cost path never meets the problem.
  3. The Bhagirathi confluence at Devprayag solves itself. The path runs
     between two fixed points, so a tributary that dead-ends is never
     entered, whatever its size.
  4. skimage.graph.MCP_Geometric is a tested C implementation that is already
     in the fyp environment. Fewer lines of mine to be wrong.

Two routes are computed from the same endpoints:

  route_water    cost driven by the dry-season water mask
  route_terrain  cost driven by elevation only, water ignored

They are independent in the sense that matters: one is optical, one is
topographic. Where they agree the answer is almost certainly right, and where
they disagree the QC file says by how much and where.
"""

from __future__ import annotations

import numpy as np
from skimage.graph import MCP_Geometric

BIG = 1e6


def cost_from_water(water, nodata=None, water_cost=0.05, land_cost=1.0,
                    dilate_px=0):
    """Cheap inside water, expensive outside. Nodata is impassable."""
    cost = np.full(water.shape, land_cost, dtype="float64")
    w = water.astype(bool)
    if dilate_px:
        from scipy.ndimage import binary_dilation
        w = binary_dilation(w, iterations=int(dilate_px))
    cost[w] = water_cost
    if nodata is not None:
        cost[nodata] = BIG
    return cost


def cost_from_terrain(dem, nodata=None, climb_weight=40.0, floor_q=0.10):
    """Cheap on the valley floor, expensive up the walls.

    Height above the local minimum, scaled. This is a poor man's HAND and it
    is enough to keep a path in the valley bottom; the project's real HAND
    raster is computed elsewhere and is not needed here.
    """
    from scipy.ndimage import minimum_filter

    z = np.array(dem, dtype="float64")
    valid = np.isfinite(z)
    if nodata is not None:
        valid &= ~nodata
    if not valid.any():
        raise ValueError("terrain cost: no valid DEM cells")
    fill = np.nanpercentile(z[valid], 99)
    z = np.where(valid, z, fill)

    # local floor over a window a few km across at 30 m cells
    win = max(3, int(round(min(z.shape) / 8)) | 1)
    win = min(win, 151)
    floor = minimum_filter(z, size=win, mode="nearest")
    above = np.clip(z - floor, 0, None)

    scale = np.percentile(above[valid], 90)
    scale = scale if scale > 1e-6 else 1.0
    cost = 1.0 + climb_weight * (above / scale)
    if nodata is not None:
        cost[nodata] = BIG
    return cost


def blend(cost_a, cost_b, w_a=0.5):
    return w_a * cost_a + (1.0 - w_a) * cost_b


def route(cost, start_rc, end_rc):
    """Least-cost 8-connected path from start to end. Returns [(row, col)...]."""
    if cost[start_rc] >= BIG or cost[end_rc] >= BIG:
        raise ValueError(
            "route: an endpoint sits on a nodata cell. "
            "Check the anchors are inside the raster and inside the corridor."
        )
    mcp = MCP_Geometric(cost, fully_connected=True)
    mcp.find_costs([list(start_rc)], [list(end_rc)])
    path = mcp.traceback(list(end_rc))
    return [tuple(int(v) for v in rc) for rc in path]


def snap_to_water(rc, water, max_px):
    """Move a seed onto the nearest water cell within max_px. Returns (rc, moved_px)."""
    r0, c0 = rc
    if water[r0, c0]:
        return (r0, c0), 0.0
    rr, cc = np.nonzero(water)
    if rr.size == 0:
        return (r0, c0), float("nan")
    d2 = (rr - r0) ** 2 + (cc - c0) ** 2
    k = int(np.argmin(d2))
    d = float(np.sqrt(d2[k]))
    if d > max_px:
        return (r0, c0), d
    return (int(rr[k]), int(cc[k])), d


def elevation_profile(dem, path_rc):
    return np.array([dem[r, c] for r, c in path_rc], dtype="float64")


def monotonic_report(z, chain_m, noise_m):
    """Where does the profile climb by more than the noise allowance?

    A river does not run uphill. This is the strongest automatic check we
    have that the line has not wandered into a tributary.
    """
    z = np.asarray(z, dtype="float64")
    ok = np.isfinite(z)
    if ok.sum() < 3:
        return {"checked": False, "reason": "no elevation samples"}
    # running minimum from the top; a rise is any excursion above it
    running = np.minimum.accumulate(np.where(ok, z, np.inf))
    rise = np.where(ok, z - running, 0.0)
    bad = rise > noise_m
    idx = np.nonzero(bad)[0]
    segs = []
    if idx.size:
        start = idx[0]
        prev = idx[0]
        for i in idx[1:]:
            if i != prev + 1:
                segs.append((start, prev))
                start = i
            prev = i
        segs.append((start, prev))
    return {
        "checked": True,
        "max_rise_m": float(rise.max()) if rise.size else 0.0,
        "cells_over_noise": int(bad.sum()),
        "fraction_over_noise": float(bad.mean()),
        "segments_km": [
            (round(float(chain_m[a] / 1000.0), 2), round(float(chain_m[b] / 1000.0), 2),
             round(float(rise[a:b + 1].max()), 1))
            for a, b in segs
        ],
        "drop_m": float(np.nanmax(z) - np.nanmin(z)),
    }
