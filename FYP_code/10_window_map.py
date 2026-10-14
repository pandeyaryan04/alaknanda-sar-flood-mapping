r"""
10_window_map.py
================
Draw the sampling design: the corridor, the centreline, the reach split, the
digitising windows, and where the optical imagery has no data.

This is the figure that shows a supervisor the validation has a design. A
reference extent drawn wherever it was convenient has no error bar; eight
windows placed by terrain along a chainage does.

HOW TO RUN

  conda activate fyp
  cd "<project folder>\\04 Code\\FYP_code"

  python 10_window_map.py --selftest

  python 10_window_map.py ^
    --corridor "data\\corridor.geojson" ^
    --centreline "..\\..\\05 Data\\boundaries\\corridor_centreline.geojson" ^
    --windows "data\\reference_windows_2023\\reference_windows.geojson" ^
    --gapmask "..\\..\\05 Data\\sentinel2\\S2_2023_event_gapmask.tif" ^
    --title "Sampling design, 2023 event" ^
    --out "..\\..\\06 Outputs\\figures\\sampling_design_2023.png"

  --gapmask is optional. Where it is given, the area with no usable optical
  imagery is hatched, and any window overlapping it is reported by number,
  because a window inside the gap cannot be digitised at all.

  --villages is optional too, and draws the settlement layer underneath.
"""
import argparse
import os
import sys

DEVPRAYAG_LON = 78.5981
METRIC = "EPSG:32644"


def say(m=""):
    print(m, flush=True)


def rule(t):
    say(); say("-" * 72); say(t); say("-" * 72)


def window_gap_overlap(win_geom, gap_geoms):
    """Share of one window that falls inside the no-data area."""
    if not gap_geoms:
        return 0.0
    inter = 0.0
    for g in gap_geoms:
        if win_geom.intersects(g):
            inter += win_geom.intersection(g).area
    return 0.0 if win_geom.area <= 0 else min(inter / win_geom.area, 1.0)


def selftest():
    rule("10_window_map self-test")
    ok = True

    def check(label, cond, note=""):
        nonlocal ok
        say("  %-5s %-46s %s" % ("PASS" if cond else "FAIL", label, note))
        ok = ok and cond

    check("python >= 3.9", sys.version_info >= (3, 9),
          "%d.%d.%d" % sys.version_info[:3])
    for mod in ("geopandas", "shapely", "matplotlib", "rasterio"):
        try:
            __import__(mod)
            check("import " + mod, True)
        except Exception as e:
            check("import " + mod, False, str(e)[:40])

    from shapely.geometry import box
    w = box(0, 0, 10, 10)
    check("no gap means no overlap", window_gap_overlap(w, []) == 0.0)
    check("a gap covering half the window scores 0.5",
          abs(window_gap_overlap(w, [box(0, 0, 5, 10)]) - 0.5) < 1e-9,
          "%.4f" % window_gap_overlap(w, [box(0, 0, 5, 10)]))
    check("a gap covering all of it scores 1.0",
          abs(window_gap_overlap(w, [box(-1, -1, 11, 11)]) - 1.0) < 1e-9)
    check("two gaps covering a quarter each sum to 0.5",
          abs(window_gap_overlap(w, [box(0, 0, 5, 5), box(5, 5, 10, 10)]) - 0.5) < 1e-9,
          "%.4f" % window_gap_overlap(w, [box(0, 0, 5, 5), box(5, 5, 10, 10)]))
    check("a disjoint gap scores 0",
          window_gap_overlap(w, [box(50, 50, 60, 60)]) == 0.0)

    say(); say("  %s" % ("all checks passed" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corridor")
    ap.add_argument("--centreline")
    ap.add_argument("--windows")
    ap.add_argument("--gapmask", help="optional, a raster where non-zero is no data")
    ap.add_argument("--villages", help="optional settlement layer, drawn underneath")
    ap.add_argument("--title", default="Sampling design")
    ap.add_argument("--out", default="../../06 Outputs/figures/sampling_design.png")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()
    if not a.corridor:
        ap.error("--corridor is required unless --selftest is given")
    if selftest() != 0:
        say(); say("self-test failed. Nothing was written.")
        return 1

    import geopandas as gpd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)

    rule("1. reading")
    cor = gpd.read_file(a.corridor).to_crs(METRIC)
    say("  corridor  %.1f km2" % (cor.area.sum() / 1e6))

    fig, ax = plt.subplots(figsize=(13, 4.0))

    if a.villages and os.path.exists(a.villages):
        vil = gpd.read_file(a.villages).to_crs(METRIC)
        vil.plot(ax=ax, facecolor="#f2f2f2", edgecolor="#dddddd", linewidth=0.3)
        say("  villages  %d" % len(vil))

    cor.plot(ax=ax, facecolor="#eef3fa", edgecolor="#1f3864", linewidth=1.0)

    gaps = []
    if a.gapmask and os.path.exists(a.gapmask):
        import rasterio
        from rasterio.features import shapes
        from shapely.geometry import shape
        with rasterio.open(a.gapmask) as src:
            arr = src.read(1)
            m = (arr != 0) & (arr != src.nodata if src.nodata is not None else arr != 0)
            for geom, val in shapes(m.astype("uint8"), mask=m, transform=src.transform):
                g = shape(geom)
                if src.crs is not None and str(src.crs) != METRIC:
                    gs = gpd.GeoSeries([g], crs=src.crs).to_crs(METRIC)
                    g = gs.iloc[0]
                gaps.append(g)
        if gaps:
            gpd.GeoSeries(gaps, crs=METRIC).plot(
                ax=ax, facecolor="none", edgecolor="#b03030",
                hatch="///", linewidth=0.4)
            say("  no-data area  %.2f km2 in %d parts"
                % (sum(g.area for g in gaps) / 1e6, len(gaps)))

    if a.centreline and os.path.exists(a.centreline):
        cl = gpd.read_file(a.centreline).to_crs(METRIC)
        cl.plot(ax=ax, color="#2e5496", linewidth=1.4)
        say("  centreline  %.2f km" % (cl.length.sum() / 1000.0))

    rows = []
    if a.windows and os.path.exists(a.windows):
        win = gpd.read_file(a.windows).to_crs(METRIC)
        win.plot(ax=ax, facecolor="none", edgecolor="#c07000", linewidth=1.3)
        total = win.area.sum() / 1e6
        say("  windows  %d, %.0f km2 total, %.1f %% of the corridor"
            % (len(win), total, 100 * total / (cor.area.sum() / 1e6)))
        for i, (idx, r) in enumerate(win.iterrows(), start=1):
            lab = str(r.get("window", i))
            c = r.geometry.centroid
            ax.annotate(lab, (c.x, c.y), ha="center", va="center",
                        fontsize=8, color="#7a4600", weight="bold")
            f = window_gap_overlap(r.geometry, gaps)
            rows.append((lab, f))
        bad = [(l, f) for l, f in rows if f > 0.01]
        if bad:
            say("")
            say("  windows overlapping the no-data area:")
            for l, f in sorted(bad, key=lambda x: -x[1]):
                say("    window %-4s %5.1f %% of its area has no optical imagery"
                    % (l, 100 * f))
            say("  A window more than about a quarter covered cannot be "
                "digitised honestly. Move it.")
        else:
            say("  no window overlaps the no-data area")

    # The reach split, drawn only if it lands inside the corridor. axvline
    # sets the x-limits, so an out-of-range value silently stretches the axes
    # until the corridor is a hairline, and that figure looks fine at a glance.
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", METRIC, always_xy=True)
    sx, _ = tr.transform(DEVPRAYAG_LON, 30.15)
    x0, _, x1, _ = cor.total_bounds
    split_drawn = x0 <= sx <= x1
    if split_drawn:
        ax.axvline(sx, color="#444444", linestyle=":", linewidth=1.2)
        y0, y1 = ax.get_ylim()
        ax.annotate("Devprayag: gorge to the right, open valley to the left",
                    (sx, y1), rotation=90, va="top", ha="right",
                    fontsize=7.5, color="#444444")
    else:
        say("  NOTE  the Devprayag split at easting %.0f is outside the "
            "corridor bounds, so it is left off. Check the corridor CRS."
            % sx)

    ax.set_title(a.title)
    ax.set_xlabel("easting, m (%s)" % METRIC)
    ax.set_ylabel("northing, m")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    handles = [
        Patch(facecolor="#eef3fa", edgecolor="#1f3864", label="corridor"),
        Line2D([], [], color="#2e5496", lw=1.4, label="centreline"),
        Patch(facecolor="none", edgecolor="#c07000", label="digitising window"),
    ]
    if split_drawn:
        handles.append(Line2D([], [], color="#444444", ls=":",
                              label="reach split at Devprayag"))
    if gaps:
        handles.append(Patch(facecolor="none", edgecolor="#b03030", hatch="///",
                             label="no usable optical imagery"))
    # Below the axes, not inside them. The corridor is 100 km by 10 km, so an
    # inset legend sits on top of the first two windows and hides the thing the
    # figure exists to show.
    fig.tight_layout()
    # fig.legend, not ax.legend: with aspect="equal" on a 10:1 corridor the
    # axes are a thin band and anything anchored to them lands on the x-axis
    # label. Anchored to the figure it sits in the empty strip underneath.
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=8, frameon=False)
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    rule("2. writing")
    say("  wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
