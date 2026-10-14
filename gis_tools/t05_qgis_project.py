"""
t05_qgis_project.py - set the QGIS project up so you cannot get it wrong.

Replaces steps 1.1 to 4.5 of `docs/REFERENCE - how to digitise.md`: about twenty
minutes per event, and one failure mode that document itself calls silent.

What it writes, per event:

  reference_<event>.gpkg   empty, with water_event and water_dry already
                           created: polygon, EPSG:32644, integer field
                           `window`. Built with fiona, so the schema is right
                           by construction instead of by clicking through the
                           New GeoPackage Layer dialog twice.
  style_*.qml              MNDWI pseudocolour, true colour, SCL palette,
                           window outlines.
  LOAD ME.md               the four drag-and-drop steps and the snapping
                           settings.
  sampling_design_<tag>.png
                           the design figure: corridor, centreline, the reach
                           split at Devprayag, the numbered windows, and the
                           area with no usable optical imagery hatched. It
                           prints which windows overlap that area and by how
                           much, so a window that cannot be digitised is
                           caught here rather than forty minutes into drawing
                           it. Pass --no-figure to skip it.

The figure duplicates `04 Code\FYP_code\10_window_map.py`, which is the
standalone version used to make the report figure. Two copies is a real cost,
and it was taken deliberately: this package reads its own `.env` and imports
nothing from `FYP_code`, and a cross-package import to save sixty lines would
break the moment either folder moved. If you change the overlap rule, change it
in both.

Honest limit. A .qgz is a zipped project XML whose schema shifts between QGIS
versions. This tool does not write one. The GeoPackage and the styles are the
part that saves the time and removes the EPSG mistake; a .qgz that opens
empty on your version would cost more than it saves.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import common as C

TOOL = "t05_qgis_project"

SCHEMA = {
    "geometry": "Polygon",
    "properties": {"window": "int", "note": "str:120"},
}

# Longitude of the Devprayag confluence, where the corridor changes character
# from confined gorge to open valley. Every accuracy figure in the project is
# reported either side of it, so the figure draws it.
DEVPRAYAG_LON = 78.5981

# A window with more of its area than this inside the no-data area cannot be
# digitised honestly and has to move.
GAP_WARN = 0.25

QML_PSEUDO = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28" styleCategories="AllStyleCategories">
  <pipe>
    <rasterrenderer type="singlebandpseudocolor" band="{band}" opacity="{opacity}"
                    classificationMin="{vmin}" classificationMax="{vmax}">
      <rastershader>
        <colorrampshader colorRampType="INTERPOLATED" clip="0">
          <item value="{vmin}" label="{vmin}" color="#ffffff" alpha="255"/>
          <item value="0" label="0" color="#9ecae1" alpha="255"/>
          <item value="{vmax}" label="{vmax}" color="#08306b" alpha="255"/>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
  </pipe>
</qgis>
"""

QML_RGB = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28" styleCategories="AllStyleCategories">
  <pipe>
    <rasterrenderer type="multibandcolor" opacity="1"
                    redBand="{r}" greenBand="{g}" blueBand="{b}">
      <redContrastEnhancement><algorithm>StretchToMinimumMaximum</algorithm></redContrastEnhancement>
      <greenContrastEnhancement><algorithm>StretchToMinimumMaximum</algorithm></greenContrastEnhancement>
      <blueContrastEnhancement><algorithm>StretchToMinimumMaximum</algorithm></blueContrastEnhancement>
    </rasterrenderer>
  </pipe>
</qgis>
"""

QML_OUTLINE = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28" styleCategories="Symbology">
  <renderer-v2 type="singleSymbol">
    <symbols>
      <symbol type="fill" name="0">
        <layer class="SimpleFill">
          <prop k="style" v="no"/>
          <prop k="outline_color" v="{colour}"/>
          <prop k="outline_width" v="{width}"/>
          <prop k="outline_width_unit" v="MM"/>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
</qgis>
"""

LOAD_ME = """# Load me

Written by `{tool}` v{version}. Everything here is already set up; these are
the four steps that are left.

## 1. Project CRS

QGIS, Project, Properties, CRS. Set **{crs}**.

Do this first. Every raster in this project is in that CRS and a mismatch is
silent: layers still draw, and every distance you measure afterwards is
wrong.

## 2. Drag these in, bottom of the list first

```
{layers}
```

## 3. Styles

Each raster has a `.qml` beside it with the same stem. QGIS picks those up
automatically when the file is dragged in. If one does not apply, right-click
the layer, Properties, Symbology, Style, Load Style, and point at the `.qml`.

The MNDWI layer wants about 60 per cent opacity, sitting above the true
colour, so you can read both at once.

## 4. Snapping

Project, Snapping Options:

- turn snapping **on**
- set it to the `reference_windows` layer
- tolerance **10 pixels**
- enable **Snap to segment**

This makes a polygon edge land exactly on the window boundary rather than a
metre off it.

## 5. Before you draw anything

Open `sampling_design_{tag}.png` in this folder. It shows the corridor, the
centreline, the reach split at Devprayag, your numbered windows, and any area
with no usable optical imagery hatched in red.

If a window sits substantially inside the hatching, do not start it. The run
that made this figure printed the percentage for every affected window; above
about a quarter the window has to move, because a boundary drawn where there is
no image is a guess wearing the same colour as a measurement.

## What is already done

`{gpkg}` exists and already contains two empty tables:

| table | geometry | CRS | fields |
|---|---|---|---|
| `water_event` | Polygon | {crs} | `window` integer, `note` text |
| `water_dry` | Polygon | {crs} | `window` integer, `note` text |

Select one, Toggle Editing, Add Polygon Feature, and draw. Type the window
number into `window` for every polygon, so a polygon can never be attributed
to the wrong square later.

## Then

Work at 1:5000. Do the whole event pass first, all windows, then the whole dry
pass. Do not alternate: the difference between the two passes is your answer,
so anything you do differently between them becomes flood.

The rest of the method, including what counts as water and the three things
that will fool you, is in `docs/REFERENCE - how to digitise.md` steps 5 to 9. None
of that has been automated and none of it should be.
"""


def events(cfg):
    out = []
    for key in cfg.csv("QGIS_EVENTS"):
        if not cfg.has(key):
            continue
        tag = key.replace("PAIR_", "").lower()
        out.append((key, tag, cfg.csv(key)))
    return out


def selftest(cfg):
    ck = C.Checks(f"{TOOL} self-test")
    try:
        import fiona  # noqa: F401
        ck.check("import fiona", True)
    except Exception as e:
        ck.check("import fiona", False, type(e).__name__)
    ev = events(cfg)
    ck.check("QGIS_EVENTS resolves to at least one event", len(ev) > 0,
             ", ".join(t for _, t, _ in ev) or "none")
    s2 = cfg.path("S2_DIR")
    for key, tag, stems in ev:
        for stem in stems:
            ck.require_file(f"{tag}: {stem}", s2 / f"{stem}_10m.tif")

    # The overlap rule, on boxes whose answers are known before the run.
    try:
        from shapely.geometry import box
        w = box(0, 0, 10, 10)
        ck.check("no gap means no overlap",
                 window_gap_overlap(w, []) == 0.0)
        ck.check("a gap over half the window scores 0.5",
                 abs(window_gap_overlap(w, [box(0, 0, 5, 10)]) - 0.5) < 1e-9)
        ck.check("a gap over all of it scores 1.0",
                 abs(window_gap_overlap(w, [box(-1, -1, 11, 11)]) - 1.0) < 1e-9)
        ck.check("two quarter gaps sum to 0.5",
                 abs(window_gap_overlap(
                     w, [box(0, 0, 5, 5), box(5, 5, 10, 10)]) - 0.5) < 1e-9)
        ck.check("a disjoint gap scores 0",
                 window_gap_overlap(w, [box(50, 50, 60, 60)]) == 0.0)
    except Exception as e:  # noqa: BLE001
        ck.check("overlap rule", False, type(e).__name__)
    return ck.report()


def window_gap_overlap(win_geom, gap_geoms):
    """Share of one window's area that falls inside the no-data area.

    Mirrors the identical function in 10_window_map.py. Kept as a share of the
    window rather than of the gap: what matters is how much of this square you
    cannot draw in, not how much of the cloud happens to sit in it.
    """
    if not gap_geoms:
        return 0.0
    inter = 0.0
    for g in gap_geoms:
        if win_geom.intersects(g):
            inter += win_geom.intersection(g).area
    return 0.0 if win_geom.area <= 0 else min(inter / win_geom.area, 1.0)


def read_gap_polygons(path, crs):
    """Vectorise the mosaic gap raster. Returns [] if there is no gap file."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import shapes
    from shapely.geometry import shape

    out = []
    with rasterio.open(path) as src:
        arr = src.read(1)
        m = arr != 0
        if src.nodata is not None:
            m &= arr != src.nodata
        if not m.any():
            return out
        for geom, _ in shapes(m.astype("uint8"), mask=m,
                              transform=src.transform):
            g = shape(geom)
            if src.crs is not None and str(src.crs) != crs:
                g = gpd.GeoSeries([g], crs=src.crs).to_crs(crs).iloc[0]
            out.append(g)
    return out


def draw_design(cfg, tag, out_dir, force):
    """Write sampling_design_<tag>.png. Returns the path, or None if skipped.

    This never raises. The GeoPackage and the styles are what this tool is for;
    a missing matplotlib or a centreline that has not been built yet must not
    stop the digitising setup from being written.
    """
    crs = cfg.str("CRS_METRIC")
    try:
        import geopandas as gpd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        from pyproj import Transformer
    except Exception as e:  # noqa: BLE001
        C.say(f"  NOTE  no figure: {type(e).__name__}. "
              f"pip install geopandas matplotlib, or pass --no-figure.")
        return None

    corridor_p = Path(cfg.str("CORRIDOR"))
    if not corridor_p.exists():
        C.say(f"  NOTE  no figure: CORRIDOR does not exist yet, {corridor_p}")
        return None

    cor = gpd.read_file(corridor_p).to_crs(crs)
    fig, ax = plt.subplots(figsize=(13, 4.0))
    cor.plot(ax=ax, facecolor="#eef3fa", edgecolor="#1f3864", linewidth=1.0)

    gaps = []
    gp = Path(cfg.path("S2_DIR")) / f"S2_{tag}_gapmask.tif"
    if gp.exists():
        try:
            gaps = read_gap_polygons(gp, crs)
        except Exception as e:  # noqa: BLE001
            C.say(f"  NOTE  could not read {gp.name}: {type(e).__name__}")
    if gaps:
        gpd.GeoSeries(gaps, crs=crs).plot(ax=ax, facecolor="none",
                                          edgecolor="#b03030", hatch="///",
                                          linewidth=0.4)

    cl = Path(cfg.str("CENTRELINE_OUT")) if cfg.has("CENTRELINE_OUT") else None
    if cl is not None and cl.exists():
        gpd.read_file(cl).to_crs(crs).plot(ax=ax, color="#2e5496", linewidth=1.4)

    wkey = f"WINDOWS_{tag.split('_')[0].upper()}"
    bad = []
    if cfg.has(wkey) and Path(cfg.str(wkey)).exists():
        win = gpd.read_file(cfg.str(wkey)).to_crs(crs)
        win.plot(ax=ax, facecolor="none", edgecolor="#c07000", linewidth=1.3)
        total = win.area.sum() / 1e6
        C.say(f"  {len(win)} windows, {total:.0f} km2, "
              f"{100 * total / (cor.area.sum() / 1e6):.1f} % of the corridor")
        for i, (_, r) in enumerate(win.iterrows(), start=1):
            lab = str(r.get("window", i))
            c = r.geometry.centroid
            ax.annotate(lab, (c.x, c.y), ha="center", va="center",
                        fontsize=8, color="#7a4600", weight="bold")
            f = window_gap_overlap(r.geometry, gaps)
            if f > 0.01:
                bad.append((lab, f))
    else:
        C.say(f"  NOTE  {wkey} not set or not built yet, so the figure has no "
              f"windows on it. Run 05_reference_windows.py first.")

    # Draw the reach split only if it actually lands inside the corridor.
    # axvline sets the x-limits, so an out-of-range value silently stretches
    # the axes until the corridor is a hairline. That is exactly the kind of
    # figure that gets put in a report without anyone noticing.
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    sx, _ = tr.transform(DEVPRAYAG_LON, 30.15)
    x0, _, x1, _ = cor.total_bounds
    split_drawn = x0 <= sx <= x1
    if split_drawn:
        ax.axvline(sx, color="#444444", linestyle=":", linewidth=1.2)
    else:
        C.say(f"  NOTE  the Devprayag split at easting {sx:,.0f} is outside "
              f"the corridor bounds, so it is left off the figure. Check "
              f"CRS_METRIC against the corridor file.")

    ax.set_title(f"Sampling design, {tag.replace('_', ' ')}")
    ax.set_xlabel(f"easting, m ({crs})")
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
        handles.append(Patch(facecolor="none", edgecolor="#b03030",
                             hatch="///", label="no usable optical imagery"))
    # Below the axes, not inside them. The corridor is 100 km by 10 km, so an
    # inset legend covers the first two windows and the figure hides the very
    # thing it is drawn to show.
    fig.tight_layout()
    # fig.legend, not ax.legend: with aspect="equal" on a 10:1 corridor the
    # axes are a thin band and anything anchored to them lands on the x-axis
    # label. Anchored to the figure it sits in the empty strip underneath.
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=8, frameon=False)

    png = C.guard_output(out_dir / f"sampling_design_{tag}.png", force)
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    if bad:
        C.say("")
        for lab, f in sorted(bad, key=lambda x: -x[1]):
            flag = "  MOVE IT" if f > GAP_WARN else ""
            C.say(f"  window {lab:<4} {100 * f:5.1f} % of its area has no "
                  f"optical imagery{flag}")
        C.say(f"  Above {GAP_WARN * 100:.0f} per cent a window cannot be "
              f"digitised honestly. Move it rather than guessing inside it.")
    elif gaps:
        C.say("  no window overlaps the no-data area")
    return png


def make_gpkg(path, crs, force):
    import fiona

    p = C.guard_output(path, force)
    for layer in ("water_event", "water_dry"):
        with fiona.open(p, "w", driver="GPKG", layer=layer, crs=crs,
                        schema=SCHEMA):
            pass
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-figure", dest="no_figure", action="store_true",
                    help="skip sampling_design_<tag>.png")
    a = ap.parse_args(argv)

    cfg = C.Cfg.load(a.config)
    if a.selftest:
        return 0 if selftest(cfg) else 1
    if not selftest(cfg):
        C.say("\nself-test failed. Nothing was written.")
        return 1

    force = a.force or cfg.b("FORCE", "false")
    crs = cfg.str("CRS_METRIC")
    s2 = cfg.path("S2_DIR")
    man = C.Manifest(TOOL, cfg)

    bands = cfg.csv("S2_BANDS")
    r, g, b = (bands.index("B4") + 1, bands.index("B3") + 1,
               bands.index("B2") + 1)

    for key, tag, stems in events(cfg):
        out_dir = Path(cfg.path("OUT_DIR")) / "qgis" / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        C.rule(f"{key}  ->  {out_dir}")

        gpkg = make_gpkg(out_dir / f"reference_{tag}.gpkg", crs, force)
        man.output(gpkg)
        C.say(f"  {gpkg.name}: water_event and water_dry, Polygon, {crs}, "
              f"fields window/note")

        styles = {
            "style_mndwi.qml": QML_PSEUDO.format(band=1, opacity=0.6,
                                                 vmin=-0.3, vmax=0.3),
            "style_truecolour.qml": QML_RGB.format(r=r, g=g, b=b),
            "style_windows.qml": QML_OUTLINE.format(colour="227,26,28,255",
                                                    width="0.6"),
            "style_centreline.qml": QML_OUTLINE.format(colour="33,113,181,255",
                                                       width="0.4"),
        }
        for name, text in styles.items():
            p = out_dir / name
            p.write_text(text, encoding="utf-8")
            man.output(p)
        C.say(f"  {len(styles)} style files")

        layers = []
        mos = s2 / f"S2_{tag}_mosaic.tif"
        mnd = s2 / f"S2_{tag}_mndwi.tif"
        layers.append(f"{mos}          <- true colour, style_truecolour.qml")
        layers.append(f"{mnd}          <- MNDWI, style_mndwi.qml, 60% opacity")
        if cfg.has("CENTRELINE_OUT"):
            layers.append(f"{cfg.str('CENTRELINE_OUT')}   <- style_centreline.qml")
        wkey = f"WINDOWS_{tag.split('_')[0].upper()}"
        if cfg.has(wkey):
            layers.append(f"{cfg.str(wkey)}   <- style_windows.qml")
        layers.append(f"{gpkg}   <- draw into this")

        if not a.no_figure:
            png = draw_design(cfg, tag, out_dir, force)
            if png is not None:
                man.output(png)
                layers.append(f"{png}   <- not a layer, open it and look at it "
                              f"before you start drawing")
                C.say(f"  {png.name}")

        p = out_dir / "LOAD ME.md"
        p.write_text(LOAD_ME.format(
            tool=TOOL, version=C.__version__, crs=crs,
            layers="\n".join(layers), gpkg=gpkg.name, tag=tag),
            encoding="utf-8")
        man.output(p)
        C.say(f"  LOAD ME.md")

        if not mos.exists():
            C.say(f"  NOTE  {mos.name} does not exist yet. Run t02 first, or "
                  f"drag the two scene files in and merge them by hand.")

    man.write(Path(cfg.path("OUT_DIR")) / "qgis")
    C.say("")
    C.say("  done. open the LOAD ME.md in each event folder.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        C.say("")
        C.say("ERROR  " + type(exc).__name__)
        for ln in str(exc).splitlines():
            C.say("       " + ln)
        sys.exit(1)
