r"""
jalsar_display.py
=================
What the bulletin and the web map share: colours, fixed wording, number
formats, and the raster-to-picture steps (palette PNG, web-mercator
reprojection, hillshade). Nothing here reads a model or computes a result.

Colours: dark maroon #6d1a2b for the header and accents.

USAGE  (Anaconda Prompt, fyp active, from 04 Code\jalsar)
-----
    python jalsar_display.py --selftest
"""

__stage__ = "Jal-SAR: shared display helpers"
__reads__ = ["event rasters (06 Outputs/jalsar/<event>/)", "05 Data/dem/DEM_COP_GLO30_30m.tif"]
__writes__ = ["nothing (returns pictures to the bulletin and the web map)"]

import base64
import io
import json
import math
import sys

import numpy as np

import jalsar_core as gc

mc = gc.mc
rasterio = gc.rasterio

MAROON = "#6d1a2b"
INK, MUTED, LINE, BG = "#1a2332", "#5b6570", "#d7dce2", "#f5f6f8"
TRUST_COLOURS = {3: "#1f4e9c", 2: "#e08e0b", 1: "#c62828", 0: "#8a8a8a"}
FAMILY_COLOURS = {"rf": "#2b7bba", "unet": "#7b3fa0", "terramind": "#138a7a"}
AGREE_COLOURS = {1: "#fcd34d", 2: "#f97316", 3: "#7f1d1d"}
AUTHOR = "Aryan Pandey"
INSTITUTE = "G. B. Pant University of Agriculture and Technology, Pantnagar"
FOOTER = f"Jal-SAR · {AUTHOR} · {INSTITUTE}"
POOLED = "both reaches pooled"

# Printed on every bulletin and on the web map.
NOT_THIS = [
    ("Not a warning system.",
     "Jal-SAR maps a flood after a satellite pass, not before. Sentinel-1 passes the "
     "corridor every few days, so a map can come a day or more after the peak."),
    ("Not operational accuracy.",
     "The models agree with hand-drawn reference maps only partly (the accuracy measured "
     "on this corridor is printed beside every map). The map is much less reliable on "
     "steep slopes and high ground: where the slope is above 33.46 degrees or the ground "
     "is more than 212.76 m above the river (56.7 per cent of the corridor), the inputs "
     "are beyond the range the models were trained on, and in the checked windows the "
     "headline map scored F1 0.09 to 0.16 there, against 0.44 to 0.48 elsewhere. Treat "
     "the map as a guide to where to look, not as a survey."),
    ("No water depth, no forecast, no damage estimate.",
     "Villages are listed as inside or touching the mapped flood, not as damaged."),
    ("The corridor only.",
     "The trust layer is measured on this corridor (Rudraprayag to Rishikesh) and nowhere "
     "else."),
]
TRUST_PLAIN = {
    3: "high: in this kind of terrain, at least 4 in 10 flood calls of this model were right "
       "on the validated events",
    2: "medium: 2 to 4 in 10 were right",
    1: "low: fewer than 2 in 10 were right, or there was too little evidence to say",
    0: "radar blind: the slope is in radar layover or shadow, so no flood can be mapped there",
}
CREDITS = ("Contains modified Copernicus Sentinel data, processed in Google Earth Engine. "
           "Village boundaries: Survey of India. Households and population: Census of India "
           "2011. Roads © OpenStreetMap contributors (ODbL). Terrain: Copernicus GLO-30 DEM.")


def hex_rgba(h, alpha=255):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)


def clean(o):
    """JSON-safe copy: NaN and infinity become None."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def to_json(o):
    return json.dumps(clean(o), ensure_ascii=False, allow_nan=False)


def num(v, nd=3, unit=""):
    """A number for a table, or an en-free 'n/a'."""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    if isinstance(v, int) or (isinstance(v, float) and nd == 0):
        return f"{v:,.0f}{unit}"
    return f"{v:,.{nd}f}{unit}"


def f1_text(d):
    """'0.411 (0.35 to 0.47)' from a validation row."""
    if not d or d.get("f1") is None:
        return "n/a"
    s = f"{d['f1']:.3f}"
    if d.get("f1_lo") is not None and d.get("f1_hi") is not None:
        s += f" ({d['f1_lo']:.2f} to {d['f1_hi']:.2f})"
    return s


def read_u8(path):
    with rasterio.open(path) as r:
        return r.read(1), r.profile


# =============================================================================
# Pictures
# =============================================================================

def palette_png(classes, colours, scale=1):
    """PNG bytes of a class raster: colours {value: (r, g, b, a)}; everything else clear.

    A palette PNG with a transparency table, so mostly-empty flood layers stay small.
    """
    from PIL import Image
    a = classes[::scale, ::scale] if scale > 1 else classes
    lut = np.zeros(256, "uint8")
    pal = [(0, 0, 0, 0)]
    for v, c in colours.items():
        lut[int(v)] = len(pal)
        pal.append(c)
    idx = lut[a]
    img = Image.fromarray(idx, mode="P")
    flat = []
    for c in pal:
        flat += list(c[:3])
    img.putpalette(flat + [0] * (768 - len(flat)))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True, transparency=bytes(c[3] for c in pal))
    return buf.getvalue()


def data_url(data, mime="image/png"):
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def to_mercator(arr, profile, cell_m=10.0):
    """Nearest-neighbour copy of a uint8 raster in EPSG:3857, and its lat/lon bounds.

    Leaflet stretches an image overlay in web mercator, so the picture must be
    in web mercator for every cell to land in the right place.
    """
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.warp import transform as wt
    h, w = arr.shape
    t = profile["transform"]
    left, top = t.c, t.f
    right, bottom = left + t.a * w, top + t.e * h
    lat_c = wt(profile["crs"], "EPSG:4326", [(left + right) / 2], [(top + bottom) / 2])[1][0]
    res = cell_m / math.cos(math.radians(lat_c))
    dt, dw, dh = calculate_default_transform(profile["crs"], "EPSG:3857", w, h,
                                             left, bottom, right, top, resolution=res)
    out = np.full((dh, dw), mc.NODATA, "uint8")
    reproject(arr, out, src_transform=t, src_crs=profile["crs"], dst_transform=dt,
              dst_crs="EPSG:3857", resampling=Resampling.nearest,
              src_nodata=mc.NODATA, dst_nodata=mc.NODATA)
    xs = [dt.c, dt.c + dt.a * dw]
    ys = [dt.f + dt.e * dh, dt.f]
    lon, lat = wt("EPSG:3857", "EPSG:4326", xs, ys)
    return out, [[lat[0], lon[0]], [lat[1], lon[1]]]


def hillshade(dem_path, profile, factor=3, azimuth=315.0, altitude=45.0):
    """Hillshade (0..255) of the DEM on the event grid coarsened by `factor`, or None."""
    from rasterio.transform import Affine
    from pathlib import Path
    if not Path(dem_path).exists():
        return None, None
    t = profile["transform"]
    prof = dict(profile)
    prof.update(width=math.ceil(profile["width"] / factor),
                height=math.ceil(profile["height"] / factor),
                transform=Affine(t.a * factor, t.b, t.c, t.d, t.e * factor, t.f))
    dem = mc.reproject_onto(dem_path, prof).astype("float64")
    if not np.isfinite(dem).any():
        return None, prof
    dem = np.where(np.isfinite(dem), dem, np.nanmean(dem))
    cs = abs(prof["transform"].a)
    gy, gx = np.gradient(dem, cs)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = math.radians(azimuth), math.radians(altitude)
    hs = (math.sin(alt) * np.cos(slope)
          + math.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    return (np.clip(hs, 0, 1) * 200 + 40).astype("uint8"), prof


def jpeg(grey, quality=78):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(grey, mode="L").save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def headline_by_trust(flood, trust):
    """Flood cells coded by their trust class (1 low, 2 medium, 3 high); 255 elsewhere."""
    out = np.full(flood.shape, mc.NODATA, "uint8")
    f = flood == 1
    t = np.where(trust == mc.NODATA, 1, trust)
    out[f] = np.maximum(t[f], 1)
    return out


def selftest():
    print("jalsar_display.py self-test")
    from PIL import Image
    a = np.full((40, 50), 255, "uint8")
    a[5:10, 5:10] = 1
    a[20:30, 20:25] = 3
    png = palette_png(a, {1: hex_rgba(TRUST_COLOURS[1]), 3: hex_rgba(TRUST_COLOURS[3], 128)})
    im = Image.open(io.BytesIO(png)).convert("RGBA")
    px = np.asarray(im)
    png_ok = (px[7, 7].tolist() == list(hex_rgba(TRUST_COLOURS[1]))
              and px[25, 22, 3] == 128 and px[0, 0, 3] == 0)
    from rasterio.transform import from_origin
    prof = {"crs": "EPSG:32644", "transform": from_origin(260000, 3340000, 10, 10),
            "width": 50, "height": 40}
    m, b = to_mercator(a, prof)
    merc_ok = (m == 1).sum() > 0 and (m == 3).sum() > (m == 1).sum() and \
        29.9 < b[0][0] < b[1][0] < 30.3 and 78.4 < b[0][1] < b[1][1] < 78.6
    hb = headline_by_trust(np.array([[1, 1, 0, 255]], "uint8"), np.array([[3, 255, 2, 255]], "uint8"))
    hb_ok = hb.tolist() == [[3, 1, 255, 255]]
    js_ok = to_json({"a": float("nan"), "b": [1.0, float("inf")]}) == '{"a": null, "b": [1.0, null]}'
    txt_ok = (f1_text({"f1": 0.4111, "f1_lo": 0.35, "f1_hi": 0.47}) == "0.411 (0.35 to 0.47)"
              and num(None) == "n/a" and num(1234, 0) == "1,234")
    no_dash = all("—" not in s and "–" not in s
                  for s in [x for p in NOT_THIS for x in p] + list(TRUST_PLAIN.values()) + [CREDITS])
    checks = [("palette PNG: colours and transparency", png_ok),
              ("web mercator copy lands near Devprayag", merc_ok),
              ("flood coded by trust, unknown trust counts as low", hb_ok),
              ("JSON without NaN", js_ok), ("number formats", txt_ok),
              ("fixed wording has no long dashes", no_dash)]
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
