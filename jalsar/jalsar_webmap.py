r"""
jalsar_webmap.py
================
Jal-SAR's web map (simplified on 9 October):
one HTML file with every event Jal-SAR has mapped.

The page has one side panel and no other controls:

1. Choose the event (a list). The panel shows that event's numbers.
2. Layers, each with a one-line description of what it is and why it is
   there: the flood map coloured by trust (on), where the three models agree,
   trust across the whole corridor, villages, and the reference windows
   (validated events only). The corridor outline is always shown.
3. Background: terrain shading made from the Copernicus DEM (inside the file,
   works offline), a satellite picture (Esri World Imagery, needs the
   internet), or none.

OpenStreetMap's background tiles are NOT used: their servers refuse pages
opened from a file on disk ("Access blocked", 9 October). Each model's own
map stays in the event folder and in the bulletin; the agreement layer shows
the three together.

The rasters are inside the file (PNG and JPEG pictures in web mercator), so
the file can be copied anywhere. The page needs the internet only for the map
library (Leaflet) and the optional satellite picture. It is checked against a
size limit (25 MB by default) so it opens on a phone.

USAGE  (Anaconda Prompt, fyp active, from 04 Code\jalsar)
-----
    python jalsar.py webmap
    python jalsar.py webmap --force
    python jalsar_webmap.py --selftest

Writes 06 Outputs\jalsar\webmap\jalsar_map.html. An existing map is kept
unless --force. Takes about 1 to 3 minutes for three events (estimate).
"""

__stage__ = "Jal-SAR: web map"
__reads__ = ["06 Outputs/jalsar/<event>/ (summary.json and rasters)",
             "FYP_code/data/corridor.geojson", "reference windows of 2021 and 2023",
             "05 Data/dem/DEM_COP_GLO30_30m.tif (terrain shading)"]
__writes__ = ["06 Outputs/jalsar/webmap/jalsar_map.html"]

import html
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import jalsar_core as gc
import jalsar_display as gd
import jalsar_extras as gx

mc = gc.mc

BG_GREY = 245          # page colour; the terrain picture's empty corners use it too
VILLAGE_FULL_KM2 = 0.1  # a village circle reaches its largest size at this much flood


def webmap_path(gp):
    return gp.gj / "webmap" / "jalsar_map.html"


def _overlay(folium, arr, profile, colours, cell_m, name):
    merc, bounds = gd.to_mercator(arr, profile, cell_m)
    png = gd.palette_png(merc, colours)
    fg = folium.FeatureGroup(name=name, show=False)
    folium.raster_layers.ImageOverlay(image=gd.data_url(png), bounds=bounds, pixelated=True,
                                      interactive=False, cross_origin=False).add_to(fg)
    return fg, len(png)


def terrain_layer(folium, gp, profile, factor=3):
    """Terrain shading from the DEM on the event grid, as a JPEG in its own pane, or None."""
    dem_p = gp.d05 / "dem" / "DEM_COP_GLO30_30m.tif"
    hs, hs_prof = gd.hillshade(dem_p, profile, factor=factor)
    if hs is None:
        return None, 0
    has_dem = np.isfinite(mc.reproject_onto(dem_p, hs_prof))
    hs = np.where(has_dem, np.minimum(hs, mc.NODATA - 1), mc.NODATA).astype("uint8")
    merc, bounds = gd.to_mercator(hs, hs_prof, abs(hs_prof["transform"].a))
    merc = np.where(merc == mc.NODATA, BG_GREY, merc).astype("uint8")
    jpg = gd.jpeg(merc, quality=72)
    fg = folium.FeatureGroup(name="Terrain shading", show=False)
    folium.raster_layers.ImageOverlay(image=gd.data_url(jpg, "image/jpeg"), bounds=bounds,
                                      interactive=False, cross_origin=False,
                                      pane="gjbg").add_to(fg)
    return fg, len(jpg)


def _village_layer(folium, summ):
    v = summ.get("villages") or {}
    fg = folium.FeatureGroup(name="Villages", show=False)
    for r in v.get("villages", []):
        pop = gd.num(r["population_2011"], 0) if r["population_2011"] is not None else "no census figure"
        hh = gd.num(r["households_2011"], 0) if r["households_2011"] is not None else "no census figure"
        share = r.get("share_flooded")
        share_txt = f"{share * 100:.1f} %" if share is not None else "n/a"
        tip = (f"<b>{html.escape(r['village'])}</b><br>{html.escape(r['subdistrict'])}, "
               f"{html.escape(r['district'])}<br>Mapped flood: {r['flood_km2']:.3f} km² "
               f"({share_txt} of the whole village)<br>Population (Census 2011): {pop}<br>"
               f"Households (Census 2011): {hh}")
        size = min((r["flood_km2"] / VILLAGE_FULL_KM2) ** 0.5, 1.0)
        folium.CircleMarker([r["lat"], r["lon"]], radius=3 + 9 * size, color=gd.MAROON,
                            weight=1.5, fill=True, fill_color="#ffffff", fill_opacity=0.8,
                            tooltip=folium.Tooltip(tip, sticky=True)).add_to(fg)
    return fg


def long_date(iso):
    """'2021-10-21' as '21 October 2021'."""
    import datetime as dt
    try:
        d = dt.date.fromisoformat(iso)
    except ValueError:
        return iso
    return f"{d.day} {d:%B %Y}"


def _event_label(e):
    return f"{e.date}: {e.label}" if e.label else e.date


def event_layers(folium, gp, e, summ, flood_cell, trust_cell):
    """The four layers of one event (flood, agreement, trust, villages), and their PNG bytes."""
    d = gp.event_dir(e.id)
    flood, prof = gd.read_u8(d / "flood_headline.tif")
    trust, _ = gd.read_u8(d / "trust.tif")
    head = summ["headline"]["name"]
    layers, size = {}, 0
    cols = {k: gd.hex_rgba(gd.TRUST_COLOURS[k]) for k in (1, 2, 3)}
    layers["flood"], n = _overlay(folium, gd.headline_by_trust(flood, trust), prof, cols,
                                  flood_cell, f"Flood map ({head})")
    size += n
    agree, _ = gd.read_u8(d / "agreement.tif")
    nf = len(summ["families"])
    layers["agree"], n = _overlay(folium, agree, prof,
                                  {k: gd.hex_rgba(gd.AGREE_COLOURS[k]) for k in range(1, nf + 1)},
                                  flood_cell, "Where the models agree")
    size += n
    tcols = {k: gd.hex_rgba(gd.TRUST_COLOURS[k], 120) for k in (0, 1, 2, 3)}
    layers["trust"], n = _overlay(folium, trust, prof, tcols, trust_cell,
                                  "Trust across the whole corridor")
    size += n
    layers["villages"] = _village_layer(folium, summ)
    return layers, size, prof


def window_layers(folium, gp):
    """{year: FeatureGroup} of the hand-drawn reference windows."""
    import geopandas as gpd
    out = {}
    for y in mc.YEARS:
        p = gp.windows(y)
        if not p.exists():
            continue
        w = gpd.read_file(p).to_crs("EPSG:4326")
        fg = folium.FeatureGroup(name=f"Reference windows {y}", show=False)
        folium.GeoJson(w.__geo_interface__, style_function=lambda f: {
            "color": "#111827", "weight": 2, "fill": False}).add_to(fg)
        out[y] = fg
    return out


# =============================================================================
# The panel: wording
# =============================================================================

def _sw(colour, round_=False, border=None):
    r = "border-radius:50%;" if round_ else ""
    b = f"border-color:{border};" if border else ""
    return f'<span class="sw" style="background:{colour};{r}{b}"></span>'


def layer_items(nf, fam_names=None):
    """(key, name, group, on, what, why, legend html) for each layer, in menu order.

    The extension layers (population, buildings, power, towers, vulnerability,
    forecast) are listed too; the page shows only the ones that exist.
    """
    fam_names = fam_names or ["Random Forest", "U-Net", "TerraMind"]
    trust_lg = "".join(f'<span class="li">{_sw(gd.TRUST_COLOURS[k])}{gc.TRUST_NAMES[k]}</span>'
                       for k in (3, 2, 1))
    blind_lg = f'<span class="li">{_sw(gd.TRUST_COLOURS[0])}radar blind</span>'
    agree_lg = "".join(f'<span class="li">{_sw(c)}{"all " if k == nf and nf > 1 else ""}{k} '
                       f'model{"s" if k > 1 else ""}</span>'
                       for k, c in gd.AGREE_COLOURS.items() if k <= nf)
    base = [
        ("flood", "Flood map", "Flood", True,
         "Water the headline model found on this date outside the river's usual channel "
         "(water that is always there is taken out). The colour is how far to trust it.",
         "This is the answer to the main question: where was the flood.", trust_lg),
        ("agree", "Where the models agree", "Flood", False,
         f"How many of the {nf} models ({', '.join(fam_names)}) call each cell flood.",
         "A cell every model calls flood is the most dependable part of the map. A cell only "
         "one model calls flood is worth checking on the ground first.", agree_lg),
        ("trust", "Trust across the whole corridor", "Flood", False,
         "The trust grade of every cell, flooded or not, on this radar pass. Grey cells are "
         "steep slopes the radar cannot see (layover or shadow).",
         "It shows where a flood could have been mapped reliably, and where it could not have "
         "been seen at all, even where no flood is mapped.", trust_lg + blind_lg),
        ("windows", "Reference windows", "Flood", False,
         "The squares where the flood was drawn by hand from optical satellite pictures, and "
         "the models were scored.",
         "It shows where the accuracy figures come from. Only events with a reference have "
         "them.", f'<span class="li">{_sw("#fff", border="#111827")}reference window</span>'),
        ("villages", "Villages", "Exposure", False,
         "A circle for each village with mapped flood inside its boundary; a bigger circle is "
         "more flood area. Hover over a circle for its Census 2011 figures.",
         "It shows who may be affected. Inside or touching the flood does not mean damaged.",
         f'<span class="li">{_sw("#fff", True, gd.MAROON)}village with mapped flood</span>'),
    ]
    extra = [(k, n, g, False, w, y, lg) for k, n, g, w, y, lg in gx.extra_items()]
    order = ["flood", "agree", "trust", "windows", "villages", "population", "buildings", "power",
             "towers", "vulnerability", "risk", "forecast", "valley_strip"]
    allv = {it[0]: it for it in base + extra}
    return [allv[k] for k in order if k in allv]


EVENT_KEYS = ("flood", "agree", "trust", "windows", "villages", "buildings", "power", "towers", "risk")
GLOBAL_KEYS = ("population", "vulnerability", "forecast", "valley_strip")

TRUST_HELP = ("Trust: the headline model's precision on this corridor in the cell's kind of "
              "terrain (incidence angle, height above the stream, valley width), the lowest of "
              "the three. High: at least 4 in 10 of its flood calls there were right on the "
              "checked events. Medium: 2 to 4 in 10. Low: fewer than 2 in 10, or too little "
              "evidence.")


def headline_text(h):
    """Why this model leads the map, in one phrase."""
    ranked = h.get("ranked") or []
    tie = (h.get("paired_check") or {}).get("tie")
    if tie and ranked and ranked[0] != h["family"]:
        return (f"{h['name']} (as accurate as {gc.FAMILY_NAMES[ranked[0]]} on this corridor, "
                "and faster; see About)")
    return f"{h['name']} (the most accurate on this corridor; see About)"


def exposure_rows(x):
    """Facts rows from an event's exposure.json (Stage M7), or []."""
    if not x:
        return []
    rows = []
    pop = x.get("population") or {}
    if pop.get("inside") is not None:
        rows.append(("People (WorldPop)", f"about {gd.num(pop['inside'], 0)} live in the mapped flood"
                     + (f", {gd.num(pop['inside_buffer'], 0)} within {pop['buffer_m']:g} m of it"
                        if pop.get("inside_buffer") is not None else "")))
    parts = []
    b = x.get("buildings") or {}
    if b.get("inside") is not None:
        parts.append(f"{gd.num(b['inside'], 0)} buildings")
    pw = x.get("power") or {}
    if pw.get("line_km_inside") is not None:
        parts.append(f"{pw['line_km_inside']:.2f} km of power line")
    if pw.get("points_inside") is not None:
        parts.append(f"{gd.num(pw['points_inside'], 0)} poles or pylons")
    t = x.get("towers") or {}
    if t.get("inside") is not None:
        parts.append(f"{gd.num(t['inside'], 0)} mobile towers")
    if parts:
        rows.append(("Inside the flood", ", ".join(parts)))
    return rows


def event_facts(e, s, x=None):
    """The event block of the side panel, as HTML."""
    a = s["areas"]
    v = s.get("villages") or {}
    fam = s["headline"]["family"]
    val = s.get("validation")
    tot = a["total_km2"]
    rows = [("Radar pass", f"Sentinel-1, orbit {e.orbit}, {e.orbit_pass.lower()}"),
            ("Headline model", headline_text(s["headline"])),
            ("Mapped flood", f"<b>{tot:.2f} km²</b>: high trust {a['by_trust']['high']:.2f}, "
                             f"medium {a['by_trust']['medium']:.2f}, low {a['by_trust']['low']:.2f}"),
            ("Radar blind", f"{s['trust']['radar_blind_km2']:.2f} km² of steep slopes the radar "
                            "could not see on this pass")]
    if v:
        txt = f"{v.get('count', 0)} touching the flood (population {gd.num(v.get('population_2011'), 0)})"
        if "count_min_area" in v:
            txt += (f"; {v['count_min_area']} with at least {v['min_flood_km2'] * 100:g} ha of "
                    f"flood (population {gd.num(v['population_min_area'], 0)}). Census 2011, "
                    "whole villages")
        rows.append(("Villages", txt))
    rows += exposure_rows(x)
    if val:
        ta = (val["table_a"].get(fam) or {}).get(gd.POOLED)
        td = (val["table_d"].get(fam) or {}).get(gd.POOLED)
        rows.append(("Accuracy", f"F1 {gd.f1_text(ta)} trained on Sen1Floods11; {gd.f1_text(td)} trained "
                                 "on this corridor (1 is a perfect match with the hand-drawn reference)"))
    else:
        rows.append(("Accuracy", "no hand-drawn reference for this date, so not measured"))
    body = "".join(f"<tr><th>{k}</th><td>{v_}</td></tr>" for k, v_ in rows)
    return (f"<div class='ev-title'>{html.escape(long_date(e.date))}"
            f"</div><div class='mu'>{html.escape(e.label)}</div><table class='facts'>{body}</table>")


TOP_H = 48       # height of the top bar, px

PANEL_CSS = f"""
<style>
.leaflet-container{{background:rgb({BG_GREY},{BG_GREY},{BG_GREY})}}
.leaflet-image-layer{{image-rendering:pixelated}}
.leaflet-top{{top:{TOP_H + 4}px}}
.leaflet-right{{transition:right .25s ease}}
@media (min-width:900px){{ .gj-side-open .leaflet-right{{right:380px}} }}
#gj-top{{position:absolute;top:0;left:0;right:0;height:{TOP_H}px;z-index:1100;display:flex;align-items:center;
 gap:10px;padding:0 10px;background:{gd.MAROON};color:#fff;font:13px Arial,Helvetica,sans-serif;
 box-shadow:0 1px 6px rgba(0,0,0,.25)}}
#gj-top .gj-word{{font:bold 20px Georgia,'Times New Roman',serif;letter-spacing:.5px;white-space:nowrap}}
#gj-top select{{font:13px Arial;padding:4px 6px;border-radius:4px;border:0;max-width:44vw;min-width:0}}
.gj-btn{{background:rgba(255,255,255,.12);color:#fff;border:1px solid rgba(255,255,255,.55);border-radius:4px;
 padding:5px 10px;font:bold 12.5px Arial;cursor:pointer;white-space:nowrap}}
.gj-btn:hover,.gj-btn.on{{background:rgba(255,255,255,.28)}}
.gj-sp{{flex:1}}
.gj-dd{{position:relative}}
#gj-menu{{position:absolute;top:{TOP_H - 8}px;left:0;min-width:250px;max-height:calc(100vh - {TOP_H + 20}px);
 overflow:auto;background:#fff;color:{gd.INK};border:1px solid {gd.LINE};border-radius:6px;
 box-shadow:0 6px 18px rgba(0,0,0,.2);padding:6px 10px 10px;display:none}}
#gj-menu.open{{display:block}}
#gj-menu .grp{{font:bold 11px Arial;text-transform:uppercase;letter-spacing:.6px;color:{gd.MAROON};
 margin:9px 0 3px}}
#gj-menu label{{display:flex;align-items:center;gap:6px;padding:3px 0;cursor:pointer}}
#gj-menu label.off{{opacity:.4;cursor:default}}
#gj-side{{position:absolute;top:{TOP_H}px;right:0;bottom:0;width:min(380px,100vw);z-index:1050;background:#fff;
 border-left:1px solid {gd.LINE};box-shadow:-2px 0 10px rgba(0,0,0,.12);font:13px/1.45 Arial,Helvetica,sans-serif;
 color:{gd.INK};transform:translateX(100%);transition:transform .25s ease;display:flex;flex-direction:column}}
#gj-side.open{{transform:none}}
.side-head{{display:flex;align-items:center;padding:8px 12px;border-bottom:1px solid {gd.LINE};
 font:bold 13px Arial;color:{gd.MAROON}}}
.side-head button{{margin-left:auto;border:0;background:none;font:20px Arial;cursor:pointer;color:{gd.MUTED}}}
.side-body{{overflow:auto;padding:4px 14px 16px}}
.side-body h4{{margin:14px 0 6px;font:bold 13px Arial;color:{gd.MAROON};border-bottom:1px solid {gd.LINE}}}
.ev-title{{font-weight:bold;margin-top:6px}}
.facts{{border-collapse:collapse;width:100%;font-size:12px;margin-top:4px}}
.facts th{{text-align:left;font-weight:normal;color:{gd.MUTED};padding:3px 6px 3px 0;vertical-align:top;width:96px}}
.facts td{{padding:3px 0;vertical-align:top}}
.facts tr+tr th,.facts tr+tr td{{border-top:1px solid {gd.LINE}}}
.lx{{padding:6px 0;border-top:1px solid {gd.LINE}}}
.lx .nm{{font-weight:bold}} .lx:not(.on){{display:none}}
.what,.why{{font-size:12px;margin:2px 0 0}} .why{{color:{gd.MUTED}}}
.lgd{{margin:3px 0 0;font-size:11.5px}} .li{{white-space:nowrap;margin-right:10px}}
.sw{{display:inline-block;width:11px;height:11px;margin-right:4px;vertical-align:-1px;border:1px solid rgba(0,0,0,.3)}}
.mu{{color:{gd.MUTED};font-size:11.5px}}
.gj-note{{font-size:11.5px;color:{gd.MUTED};margin:4px 0}}
.tt{{border-collapse:collapse;width:100%;font-size:11.5px;margin:4px 0}}
.tt th{{text-align:left;color:{gd.MUTED};font-weight:normal;padding:2px 4px 2px 0;vertical-align:bottom}}
.tt td{{padding:3px 4px 3px 0;border-top:1px solid {gd.LINE};vertical-align:top}}
.fc{{border-collapse:separate;border-spacing:2px;font-size:11.5px}} .fc td{{width:12px;height:12px}}
.fc th{{text-align:left;font-weight:normal;padding-right:6px;white-space:nowrap}}
details.about>summary{{cursor:pointer;color:{gd.MAROON};font-weight:bold;margin-top:12px}}
@media (max-width:640px){{ #gj-top .gj-word{{font-size:16px}} #gj-top .gj-hide-s{{display:none}} }}
</style>"""


def panel_html(items, layer_list, nf, made, ts=None, fc=None):
    """The top bar (event, layers, background, explain button) and the side panel."""
    opts = "".join(f'<option value="{html.escape(e.id)}">{html.escape(_event_label(e))}</option>'
                   for e, _ in items)
    menu, grp = [], None
    for k, name, g, on, *_ in layer_list:
        if g != grp:
            menu.append(f'<div class="grp">{html.escape(g)}</div>')
            grp = g
        menu.append(f'<label id="lab-{k}"><input type="checkbox" id="chk-{k}"{" checked" if on else ""}>'
                    f'{html.escape(name)}</label>')
    menu.append('<div class="grp">Background</div>'
                '<label><input type="radio" name="gj-bg" value="terrain" checked>Terrain shading (offline)</label>'
                '<label><input type="radio" name="gj-bg" value="satellite">Satellite picture (Esri)</label>'
                '<label><input type="radio" name="gj-bg" value="none">None</label>')
    lays = "".join(
        f'<div class="lx" id="lx-{k}"><div class="nm">{html.escape(name)}</div>'
        f'<div class="what">{html.escape(what)}</div><div class="why">Why: {html.escape(why)}</div>'
        f'<div class="lgd">{lg}</div></div>'
        for k, name, g, on, what, why, lg in layer_list)
    nots = "".join(f"<p class='gj-note'><b>{html.escape(a)}</b> {html.escape(b)}</p>"
                   for a, b in gd.NOT_THIS)
    plain = gx.trust_plain_numbers(ts)
    trust_block = (gx.trust_table_html(ts)
                   + "".join(f"<p class='gj-note'><b>{html.escape(k.capitalize())}.</b> {html.escape(v)}</p>"
                             for k, v in plain.items())) if ts else ""
    fc_block = f"<h4>Forecast</h4>{gx.forecast_html(fc)}" if fc else ""
    return f"""{PANEL_CSS}
<div id="gj-top">
 <span class="gj-word">Jal-SAR</span>
 <select id="gj-event" aria-label="Flood event">{opts}</select>
 <div class="gj-dd"><button class="gj-btn" id="gj-layers-btn" aria-expanded="false">Layers &#9662;</button>
  <div id="gj-menu">{''.join(menu)}</div></div>
 <span class="gj-sp"></span>
 <span class="gj-hide-s mu" style="color:#f3dfe4">Alaknanda corridor, Sentinel-1 radar</span>
 <button class="gj-btn" id="gj-info-btn" aria-expanded="true">Explain &#9432;</button>
</div>
<aside id="gj-side" aria-label="Explanations">
 <div class="side-head">What you are looking at<button id="gj-side-close" aria-label="Close">&times;</button></div>
 <div class="side-body">
  <div id="gj-facts"></div>
  <p class="gj-note"><b>Tip:</b> the flood lies in narrow strips along the river. Zoom in to see it.</p>
  <h4>Layers switched on</h4>
  {lays}
  <p class="gj-note" id="gj-nolayer">Switch a layer on under <b>Layers</b> to read what it shows here.</p>
  <p class="gj-note">Always shown: the dashed line is the corridor, Rudraprayag to Rishikesh.
  Nothing is mapped outside it.</p>
  <h4>How far to trust the map</h4>
  <p class="gj-note">{html.escape(TRUST_HELP)}</p>
  {trust_block}
  {fc_block}
  <details class="about"><summary>About this map and its limits</summary>
  {nots}
  <p class="gj-note"><b>Headline model.</b> Three models map every event. The headline map comes
  from the one with the highest accuracy on this corridor (Table D, 2021 and 2023); when the best
  two cannot be told apart, the faster one is used.</p>
  <p class="gj-note"><b>Each model's own map</b> is in the event folder (flood_rf.tif,
  flood_unet.tif, flood_terramind.tif) and in the bulletin.</p>
  <p class="gj-note"><b>Sources.</b> {html.escape(gd.CREDITS)} Satellite picture: Esri World Imagery.</p>
  <p class="gj-note">{html.escape(gd.FOOTER)}. Jal-SAR {gc.VERSION}, made {made}.</p>
  </details>
 </div>
</aside>"""


def panel_js(map_name, events, first_id, bg, corridor, keys, global_layers):
    """Switches layers from the top bar; fills and opens the side panel.

    events: {id: {layers: {key: js var}, facts: html}}; global_layers: {key: js var}.
    """
    return f"""
window.addEventListener("load", function() {{        // after folium's own script made the map
  var map = {map_name};
  var EV = {json.dumps(events, ensure_ascii=False)};
  var GL = {json.dumps(global_layers)};
  var BG = {json.dumps(bg)};
  var KEYS = {json.dumps(list(keys))};
  function v(name) {{ return name ? window[name] : null; }}
  function layerFor(ev, key) {{ return ev.layers[key] || GL[key] || null; }}
  L.control.zoom({{position: "bottomright"}}).addTo(map);
  var side = document.getElementById("gj-side"), info = document.getElementById("gj-info-btn");
  var menu = document.getElementById("gj-menu"), lbtn = document.getElementById("gj-layers-btn");
  function setSide(open) {{
    side.classList.toggle("open", open); info.classList.toggle("on", open);
    map.getContainer().classList.toggle("gj-side-open", open);
    info.setAttribute("aria-expanded", open);
    try {{ localStorage.setItem("jalsar-side", open ? "1" : "0"); }} catch (err) {{}}
  }}
  var keep = null;
  try {{ keep = localStorage.getItem("jalsar-side"); }} catch (err) {{}}
  setSide(keep === null ? window.innerWidth >= 900 : keep === "1");
  info.addEventListener("click", function() {{ setSide(!side.classList.contains("open")); }});
  document.getElementById("gj-side-close").addEventListener("click", function() {{ setSide(false); }});
  lbtn.addEventListener("click", function(e) {{
    e.stopPropagation(); var o = !menu.classList.contains("open");
    menu.classList.toggle("open", o); lbtn.classList.toggle("on", o); lbtn.setAttribute("aria-expanded", o);
  }});
  menu.addEventListener("click", function(e) {{ e.stopPropagation(); }});
  document.addEventListener("click", function() {{
    menu.classList.remove("open"); lbtn.classList.remove("on"); lbtn.setAttribute("aria-expanded", false);
  }});
  var cor = {f"v({json.dumps(corridor)})" if corridor else "null"};
  if (cor) {{
    cor.addTo(map);
    var right = side.classList.contains("open") && window.innerWidth >= 900 ? 390 : 10;
    map.fitBounds(cor.getBounds(), {{paddingTopLeft: [10, {TOP_H + 10}], paddingBottomRight: [right, 10]}});
  }}
  function show() {{
    var id = document.getElementById("gj-event").value;
    var seen = {{}};
    Object.keys(EV).forEach(function(k) {{
      KEYS.forEach(function(key) {{
        var n = layerFor(EV[k], key);
        if (n && !seen[n]) {{ seen[n] = 1; var l = v(n); if (l && map.hasLayer(l)) map.removeLayer(l); }}
      }});
    }});
    var ev = EV[id], any = false;
    document.getElementById("gj-facts").innerHTML = ev.facts;
    KEYS.forEach(function(key) {{
      var n = layerFor(ev, key), has = !!n;
      var box = document.getElementById("chk-" + key);
      document.getElementById("lab-" + key).className = has ? "" : "off";
      box.disabled = !has;
      var on = has && box.checked;
      document.getElementById("lx-" + key).className = "lx" + (on ? " on" : "");
      if (on) {{ v(n).addTo(map); any = true; }}
    }});
    document.getElementById("gj-nolayer").style.display = any ? "none" : "";
  }}
  function background() {{
    var pick = document.querySelector("input[name=gj-bg]:checked").value;
    Object.keys(BG).forEach(function(k) {{
      var l = v(BG[k]); if (!l) return;
      if (k === pick) {{ if (!map.hasLayer(l)) l.addTo(map); }} else if (map.hasLayer(l)) map.removeLayer(l);
    }});
  }}
  document.getElementById("gj-event").value = {json.dumps(first_id)};
  document.getElementById("gj-event").addEventListener("change", show);
  KEYS.forEach(function(key) {{ document.getElementById("chk-" + key).addEventListener("change", show); }});
  document.querySelectorAll("input[name=gj-bg]").forEach(function(r) {{ r.addEventListener("change", background); }});
  background();
  show();
}});
"""


def build_webmap(gp, out=None, flood_cell=10.0, trust_cell=30.0, max_mb=25.0, force=False):
    import folium
    out = Path(out) if out else webmap_path(gp)
    if out.exists() and not force:
        print(f"  web map already there, kept: {out}  (--force makes it again)")
        return {"path": str(out), "kept": True}
    items = []
    for e in gc.load_events(gp, only_existing=False):
        p = gp.event_dir(e.id) / "summary.json"
        if p.exists():
            items.append((e, gd.clean(mc.load_json(p))))
    if not items:
        mc.fail("No event has been mapped yet. Run 'python jalsar.py maps' first.")
    items.sort(key=lambda x: x[0].date, reverse=True)
    t0 = time.time()
    import geopandas as gpd
    cor = gpd.read_file(gp.corridor).to_crs("EPSG:4326") if gp.corridor.exists() else None
    if cor is not None:
        w, s, e_, n = cor.total_bounds
        centre = [(s + n) / 2, (w + e_) / 2]
    else:
        centre = [30.15, 78.6]
    m = folium.Map(location=centre, zoom_start=11, tiles=None, control_scale=True,
                   zoom_control=False, prefer_canvas=True)
    folium.map.CustomPane("gjbg", z_index=150).add_to(m)
    if cor is not None:
        m.fit_bounds([[s, w], [n, e_]])
    total = 0
    events, prof0 = {}, None
    win = window_layers(folium, gp)
    present = set()
    for e, summ in items:
        print(f"  {e.id}: layers", flush=True)
        layers, size, prof = event_layers(folium, gp, e, summ, flood_cell, trust_cell)
        prof0 = prof0 or prof
        if e.validated_year and e.validated_year in win:
            layers["windows"] = win[e.validated_year]
        layers.update(gx.event_extra_layers(folium, gp, e))
        for fg in layers.values():
            fg.add_to(m)
        present |= set(layers)
        total += size
        xsum, _ = gx.exposure(gp, e)
        events[e.id] = {"layers": {k: fg.get_name() for k, fg in layers.items()},
                        "facts": event_facts(e, summ, xsum)}
    for fg in win.values():
        if fg.get_name() not in {x for ev in events.values() for x in ev["layers"].values()}:
            fg.add_to(m)                        # a reference year with no mapped event
    glob = {}
    pop, n_pop = gx.population_layer(folium, gp)
    if pop is not None:
        pop.add_to(m)
        glob["population"] = pop.get_name()
        total += n_pop
    vul = gx.vulnerability_layer(folium, gp)
    if vul is not None:
        vul.add_to(m)
        glob["vulnerability"] = vul.get_name()
    fc = gx.load_forecast(gp)
    fcl, n_fc = gx.forecast_layer(folium, gp, fc)
    if fcl is not None:
        fcl.add_to(m)
        glob["forecast"] = fcl.get_name()
        total += n_fc
    vsl, n_vs = gx.valley_strip_layer(folium, gp, fc)        # the open valley gets a static strip, not a forecast extent
    if vsl is not None:
        vsl.add_to(m)
        glob["valley_strip"] = vsl.get_name()
        total += n_vs
    present |= set(glob)
    bg = {}
    terr, n_terr = terrain_layer(folium, gp, prof0)
    if terr is not None:
        terr.add_to(m)
        bg["terrain"] = terr.get_name()
        total += n_terr
    sat = folium.TileLayer("Esri.WorldImagery", name="Satellite picture (Esri)", show=False)
    sat.add_to(m)
    bg["satellite"] = sat.get_name()
    corridor = None
    if cor is not None:
        cg = folium.GeoJson(cor.__geo_interface__, name="Corridor", show=False,
                            style_function=lambda f: {"color": gd.MAROON, "weight": 2,
                                                      "fill": False, "dashArray": "6 4"})
        cg.add_to(m)
        corridor = cg.get_name()
    validated = [e for e, _ in items if e.validated_year]
    first = max(validated, key=lambda e: e.date) if validated else items[0][0]
    made = time.strftime("%d %B %Y")
    fams = max((s_["families"] for _, s_ in items), key=len)
    nf, fam_names = len(fams), [f["name"] for f in fams.values()]
    layer_list = [it for it in layer_items(nf, fam_names) if it[0] in present or it[0] == "windows"]
    keys = [it[0] for it in layer_list]
    ts = gx.load_trust_scores(gp)
    m.get_root().header.add_child(folium.Element(
        "<title>Jal-SAR: flood maps of the Alaknanda corridor</title>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"))
    m.get_root().html.add_child(folium.Element(panel_html(items, layer_list, nf, made, ts, fc)))
    m.get_root().script.add_child(folium.Element(panel_js(m.get_name(), events, first.id, bg,
                                                          corridor, keys, glob)))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + "_part.html")
    m.save(str(tmp))
    tmp.replace(out)
    mb = out.stat().st_size / 1e6
    rec = {"path": str(out), "events": [e.id for e, _ in items], "megabytes": round(mb, 2),
           "picture_megabytes": round(total / 1e6, 2), "flood_cell_m": flood_cell,
           "trust_cell_m": trust_cell, "terrain_shading": terr is not None,
           "layers": keys, "trust_scores": ts is not None, "forecast": fc is not None,
           "seconds": round(time.time() - t0, 1),
           "within_limit": mb <= max_mb, "limit_mb": max_mb}
    mc.save_json(out.with_name("webmap_record.json"), rec)
    print(f"  web map: {len(items)} event(s), layers {', '.join(keys)}; {mb:.1f} MB (limit {max_mb:.0f} MB)")
    print(f"  wrote {out}")
    if mb > max_mb:
        print(f"  WARNING: the page is larger than {max_mb:.0f} MB and may be slow on a phone. "
              "Make it again with --flood-cell 20 --force.")
    return rec


def selftest():
    print("jalsar_webmap.py self-test")
    import io
    import contextlib
    import re
    import jalsar_testdata as td_mod
    with tempfile.TemporaryDirectory(prefix="jalsar webmap ", ignore_cleanup_errors=True) as td:
        with contextlib.redirect_stdout(io.StringIO()):
            gp = td_mod.build_display_project(Path(td) / "Final Year Project")
        rec = build_webmap(gp)
        page = Path(rec["path"]).read_text(encoding="utf-8")
        again = build_webmap(gp).get("kept") is True
        n_img = len(re.findall(r"L\.imageOverlay\(", page))
        # 3 events x (flood, agreement, trust) + the terrain shading = 10 pictures
        imgs_ok = n_img == 10 and rec["terrain_shading"]
        events_ok = all(s in page for s in ('value="2021-10-21_orbit063"',
                                            "2023-07-19: July 2023 (radar only, no reference)",
                                            "no hand-drawn reference for this date"))
        no_osm = ("tile.openstreetmap.org" not in page and "opentopomap" not in page.lower()
                  and "L.control.layers" not in page)
        words_ok = all(s in page for s in ("Where the models agree", "Trust across the whole corridor",
                                           "Reference windows", "Why: ", "Terrain shading",
                                           "Satellite picture", "World Imagery",
                                           "Nothing is mapped outside it"))
        panel_ok = all(s in page for s in ("Jal-SAR", "About this map and its limits",
                                           "Census 2011", "OpenStreetMap contributors",
                                           gd.MAROON, '"pane": "gjbg"'))
        js_ok = all(s in page for s in ("var EV = ", "gj-event", "removeLayer", "chk-windows",
                                        "jalsar-side", "gj-layers-btn", "gj-side", "var GL = "))
        no_nan = "NaN" not in page.split("<body>", 1)[-1].split("L.imageOverlay", 1)[0]
        size_ok = rec["within_limit"] and rec["megabytes"] < 25
        villages_ok = "Gaon" in page and "of the whole village" in page
        forced = build_webmap(gp, force=True).get("kept") is None
    checks = [("one picture per layer (10: 3 events x 3, and the terrain)", imgs_ok),
              ("an event list; the radar-only event says it has no reference", events_ok),
              ("no OpenStreetMap or OpenTopoMap tiles, no layer-control boxes", no_osm),
              ("each layer named, with what and why; backgrounds offered", words_ok),
              ("panel: wordmark, limits, census and road credits, background pane", panel_ok),
              ("top bar switches the layers; side panel explains and hides (script present)", js_ok),
              ("no NaN in the page text", no_nan), ("under the size limit", size_ok),
              ("villages with whole-village share in the tooltips", villages_ok),
              ("kept without --force, made again with it", again and forced)]
    for name, c in checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {name}")
    good = all(c for _, c in checks)
    print("SELFTEST PASS" if good else "SELFTEST FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    mc.setup_console()
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print(__doc__)
