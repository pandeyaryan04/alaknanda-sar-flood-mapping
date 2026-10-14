"""
check_rtc.py - check the ASF HyP3 RTC downloads and put them on the project grid.

Run it with CHECK RTC.bat (double-click), or from Anaconda Prompt:

    conda activate fyp
    cd /d "<project folder>\\04 Code\\FYP_code"
    python check_rtc.py

What it does, in order, and it changes nothing it did not make:

1. Opens every .zip in 05 Data\\sentinel1_rtc. A zip that will not open is an
   incomplete download and is reported as BROKEN.
2. Where one scene has two zips, compares their contents using the checksums
   stored inside the zips (no extraction needed).
3. For each scene, extracts only the files the project uses (VV, VH, the
   incidence-angle map, the layover/shadow map, the DEM, README and log) into
   05 Data\\sentinel1_rtc\\<zip name>\\. Extraction checks every byte against the
   zip's checksum, so a damaged file stops here instead of later.
4. Puts VV and VH (in dB), the layover/shadow map and the incidence angle onto
   exactly the same 10 m grid as the GRD scene of the same date, inside the
   same corridor, and writes them as new files beside the zips:
       S1RTC_<event>_post_<date>_orb<nn>.tif     VV, VH, angle (same bands as GRD)
       S1RTC_<event>_post_<date>_orb<nn>_lsmap.tif
       S1RTC_<event>_post_<date>_orb<nn>_incmap_deg.tif
5. Compares RTC with GRD over the corridor and writes the report
   05 Data\\00 Notes\\RTC CHECK - <date>.md

Nothing is deleted. Re-running skips finished extractions and finished grid
files; add --force to rebuild the grid files.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                       # ...\Final Year Project
DATA = ROOT / "05 Data"
RTC_DIR = DATA / "sentinel1_rtc"
GRD_DIR = DATA / "sentinel1"
NOTES = DATA / "00 Notes"
PERM = DATA / "dem" / "WATER_PERMANENT_JRC50_S2DRY_10m.tif"

# acquisition start in the ASF name -> (GRD file on the project grid, output stem)
SCENES = {
    "20211021T004351": ("S1_2021_post_20211021_orb63.tif", "S1RTC_2021_post_20211021_orb63"),
    "20230816T124741": ("S1_2023_post_20230816_orb129.tif", "S1RTC_2023_post_20230816_orb129"),
}
WANTED = ("_VV.tif", "_VH.tif", "_inc_map.tif", "_ls_map.tif", "_dem.tif",
          ".README.md.txt", ".log")
CHUNK = 16 * 1024 * 1024

lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    lines.append(s)


def scene_key(name: str) -> str | None:
    m = re.search(r"_(\d{8}T\d{6})_", name)
    return m.group(1) if m else None


# ---------------------------------------------------------------- zips
def survey_zip(p: Path) -> dict:
    z = {"path": p, "size": p.stat().st_size, "ok": False, "crc": {}, "members": 0,
         "error": ""}
    try:
        with zipfile.ZipFile(p) as zf:
            infos = zf.infolist()
            z["members"] = len(infos)
            for i in infos:
                base = Path(i.filename).name
                for w in WANTED:
                    if base.endswith(w):
                        z["crc"][w] = (i.CRC, i.file_size, i.filename)
            z["ok"] = True
    except (zipfile.BadZipFile, OSError) as e:
        z["error"] = str(e)
    return z


def extract(z: dict) -> Path:
    """Extract the wanted members. Raises zipfile.BadZipFile on a checksum error."""
    out = RTC_DIR / z["path"].stem
    out.mkdir(exist_ok=True)
    with zipfile.ZipFile(z["path"]) as zf:
        for w, (_crc, size, member) in z["crc"].items():
            dest = out / Path(member).name
            if dest.exists() and dest.stat().st_size == size:
                say(f"    already extracted  {dest.name}")
                continue
            part = dest.with_suffix(dest.suffix + ".part")
            try:
                with zf.open(member) as src, open(part, "wb") as dst:
                    shutil.copyfileobj(src, dst, CHUNK)
            except Exception:
                part.unlink(missing_ok=True)
                raise
            part.replace(dest)
            say(f"    extracted          {dest.name}  ({size / 1e6:,.0f} MB)")
    return out


# ---------------------------------------------------------------- grids
def on_grid(src_path: Path, grd, resampling, dtype="float32"):
    with rasterio.open(src_path) as s:
        nod = s.nodata if s.nodata is not None else 0
        with WarpedVRT(s, crs=grd.crs, transform=grd.transform, width=grd.width,
                       height=grd.height, resampling=resampling, src_nodata=nod,
                       nodata=nod) as v:
            a = v.read(1).astype(dtype)
            info = (str(s.crs), s.res, s.width, s.height, s.dtypes[0], s.nodata)
    return a, nod, info


def write(path: Path, grd, bands, descs, dtype, nodata):
    prof = grd.profile.copy()
    prof.update(count=len(bands), dtype=dtype, nodata=nodata, compress="deflate",
                predictor=2 if dtype == "float32" else 1, tiled=True,
                blockxsize=512, blockysize=512)
    tmp = path.with_suffix(".part.tif")
    with rasterio.open(tmp, "w", **prof) as d:
        for i, (b, n) in enumerate(zip(bands, descs), 1):
            d.write(b.astype(dtype), i)
            d.set_band_description(i, n)
    tmp.replace(path)


def nmad(x):
    m = np.median(x)
    return 1.4826 * np.median(np.abs(x - m))


def shift_test(a, b, ok, r=2):
    """Correlation of a and b with b shifted by up to r cells. Best should be (0, 0)."""
    h, w = a.shape
    best = None
    res = {}
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            sa = a[r:h - r, r:w - r]
            sb = b[r + dy:h - r + dy, r + dx:w - r + dx]
            m = ok[r:h - r, r:w - r] & ok[r + dy:h - r + dy, r + dx:w - r + dx]
            if m.sum() < 1000:
                continue
            c = np.corrcoef(sa[m], sb[m])[0, 1]
            res[(dx, dy)] = c
            if best is None or c > res[best]:
                best = (dx, dy)
    return best, res


# ---------------------------------------------------------------- scene
def check_scene(key: str, folder: Path, force: bool) -> dict:
    grd_name, stem = SCENES[key]
    r = {"stem": stem, "verdicts": []}
    files = {w: next(iter(folder.glob("*" + w)), None) for w in WANTED}
    for w in ("_VV.tif", "_VH.tif", "_ls_map.tif", "_inc_map.tif"):
        if files[w] is None:
            r["verdicts"].append(("FAIL", f"{w} missing from the product"))
            return r

    with rasterio.open(GRD_DIR / grd_name) as grd:
        g_vv = grd.read(1)
        g_vh = grd.read(2)
        g_ang = grd.read(3)
        inside = np.isfinite(g_vv)
        n_in = int(inside.sum())

        vv, _, info_vv = on_grid(files["_VV.tif"], grd, Resampling.bilinear)
        vh, _, _ = on_grid(files["_VH.tif"], grd, Resampling.bilinear)
        ls, ls_nod, info_ls = on_grid(files["_ls_map.tif"], grd, Resampling.nearest, "int32")
        inc, inc_nod, info_inc = on_grid(files["_inc_map.tif"], grd, Resampling.bilinear)

        r["src"] = info_vv
        say(f"    source grid        {info_vv[0]}, {info_vv[1][0]:g} m, "
            f"{info_vv[2]} x {info_vv[3]}, {info_vv[4]}, nodata {info_vv[5]}")

        ok_rtc = np.isfinite(vv) & (vv > 0) & np.isfinite(vh) & (vh > 0)
        both = inside & ok_rtc
        cov = 100.0 * both.sum() / n_in
        r["coverage"] = cov
        # RTC leaves radar shadow (ls_map bit 16) and its edges empty by design,
        # so the test is on the cells outside shadow.
        shadow = inside & ((ls & 16) > 0)
        cov_ns = 100.0 * (both & ~shadow).sum() / max((inside & ~shadow).sum(), 1)
        r["verdicts"].append(("PASS" if cov_ns >= 99.0 else "FAIL",
                              f"covers {cov:.3f} % of the corridor cells the GRD has; "
                              f"{cov_ns:.3f} % outside radar shadow"))
        r["verdicts"].append(("PASS" if abs(info_vv[1][0] - 10) < 0.01 else "WARN",
                              f"source pixel {info_vv[1][0]:g} m"))

        with np.errstate(divide="ignore", invalid="ignore"):
            vv_db = np.where(both, 10 * np.log10(vv), np.nan).astype("float32")
            vh_db = np.where(both, 10 * np.log10(vh), np.nan).astype("float32")

        d = vv_db[both] - g_vv[both]
        c = np.corrcoef(vv_db[both], g_vv[both])[0, 1]
        r["vv"] = dict(rtc_med=float(np.median(vv_db[both])), grd_med=float(np.median(g_vv[both])),
                       diff_med=float(np.median(d)), diff_nmad=float(nmad(d)), corr=float(c),
                       rtc_p=np.percentile(vv_db[both], [1, 99]).tolist())
        dvh = vh_db[both] - g_vh[both]
        r["vh"] = dict(diff_med=float(np.median(dvh)), diff_nmad=float(nmad(dvh)),
                       corr=float(np.corrcoef(vh_db[both], g_vh[both])[0, 1]))
        # On slopes RTC is meant to differ from GRD (that is the terrain
        # flattening), so agreement is tested where the ground faces the radar
        # at its ellipsoid angle: local and ellipsoid incidence within 5 degrees.
        incd_all = np.degrees(inc)
        flat = both & np.isfinite(incd_all) & (np.abs(incd_all - g_ang) < 5)
        cf = np.corrcoef(vv_db[flat], g_vv[flat])[0, 1] if flat.sum() > 1000 else float("nan")
        r["verdicts"].append(("PASS" if cf >= 0.6 else "WARN",
                              f"VV correlation with GRD on flat-facing ground {cf:.3f} "
                              f"({100 * flat.sum() / both.sum():.1f} % of cells); "
                              f"all cells {c:.3f} (VH {r['vh']['corr']:.3f}), low on slopes by design"))

        # alignment, on the central third of the rows to keep it quick
        h = grd.height
        sl = slice(h // 3, 2 * h // 3)
        a = np.nan_to_num(vv_db[sl], nan=0.0)
        b = np.nan_to_num(g_vv[sl], nan=0.0)
        best, res = shift_test(b, a, both[sl])
        r["shift"] = (best, res.get(best), res.get((0, 0)))
        r["verdicts"].append(("PASS" if best is not None and max(map(abs, best)) <= 1 else "WARN",
                              f"best alignment with GRD at shift {best} cells "
                              f"(corr {res.get(best):.3f}; at (0, 0) {res.get((0, 0)):.3f})"))

        # water against land, permanent-water mask
        if PERM.exists():
            pw, _, _ = on_grid(PERM, grd, Resampling.nearest)
            water = both & (pw >= 50)
            land = both & (pw < 50)
            r["water"] = dict(n=int(water.sum()),
                              rtc=float(np.median(vv_db[water])), grd=float(np.median(g_vv[water])),
                              rtc_land=float(np.median(vv_db[land])), grd_land=float(np.median(g_vv[land])))
            r["verdicts"].append(("PASS" if r["water"]["rtc_land"] - r["water"]["rtc"] >= 5 else "WARN",
                                  f"permanent water median VV {r['water']['rtc']:.1f} dB RTC, "
                                  f"{r['water']['grd']:.1f} dB GRD; land {r['water']['rtc_land']:.1f} "
                                  f"and {r['water']['grd_land']:.1f} dB"))

        vals, counts = np.unique(ls[inside], return_counts=True)
        r["ls"] = {int(v): 100.0 * n / n_in for v, n in zip(vals, counts)}
        incd = np.where(inside & (inc != inc_nod), np.degrees(inc), np.nan).astype("float32")
        fin = np.isfinite(incd)
        r["inc"] = np.percentile(incd[fin], [1, 50, 99]).tolist() if fin.any() else None

        out_main = RTC_DIR / f"{stem}.tif"
        if out_main.exists() and not force:
            say(f"    exists, kept       {out_main.name} (use --force to rebuild)")
        else:
            ang = np.where(both, g_ang, np.nan)
            write(out_main, grd, [vv_db, vh_db, ang], ["VV", "VH", "angle"], "float32", None)
            lsm = np.where(inside, ls, 255)
            write(RTC_DIR / f"{stem}_lsmap.tif", grd, [lsm], ["ls_map"], "uint8", 255)
            write(RTC_DIR / f"{stem}_incmap_deg.tif", grd, [incd], ["local_incidence_deg"],
                  "float32", None)
            say(f"    wrote              {out_main.name}, _lsmap.tif, _incmap_deg.tif")

    readme = files[".README.md.txt"]
    if readme is not None:
        txt = readme.read_text(encoding="utf-8", errors="replace").splitlines()
        r["readme_ls"] = [t.strip() for t in txt
                          if re.search(r"layover|shadow|ls_map", t, re.I)][:12]
    return r


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild the grid files")
    a = ap.parse_args()

    today = dt.date.today().isoformat()
    say(f"# RTC check, {today}")
    say("")
    say("Written by `04 Code\\FYP_code\\check_rtc.py`. Nothing was deleted.")
    say("")

    zips = sorted(RTC_DIR.glob("*.zip"))
    if not zips:
        say(f"No .zip files in {RTC_DIR}")
        return 1

    say("## 1. The zips")
    say("")
    say("| zip | size, bytes | opens | files inside |")
    say("|---|---|---|---|")
    surveyed = [survey_zip(p) for p in zips]
    for z in surveyed:
        say(f"| `{z['path'].name}` | {z['size']:,} | "
            f"{'yes' if z['ok'] else '**NO, broken: ' + z['error'] + '**'} | {z['members']} |")
    say("")

    by_scene: dict[str, list[dict]] = {}
    for z in surveyed:
        k = scene_key(z["path"].name)
        if k in SCENES:
            by_scene.setdefault(k, []).append(z)

    say("## 2. Duplicates")
    say("")
    for k, zs in by_scene.items():
        good = [z for z in zs if z["ok"]]
        if len(good) > 1:
            ref = good[0]
            for o in good[1:]:
                same = [w for w in ref["crc"] if w in o["crc"] and ref["crc"][w][0] == o["crc"][w][0]]
                diff = [w for w in ref["crc"] if w not in same]
                say(f"- `{ref['path'].name}` and `{o['path'].name}`: identical "
                    f"{', '.join(same) or 'none'}; different {', '.join(diff) or 'none'}")
        else:
            say(f"- scene {k}: {len(good)} usable zip(s) of {len(zs)}")
    say("")

    say("## 3. Each scene")
    say("")
    results = {}
    for k in SCENES:
        say(f"### {SCENES[k][1]}")
        say("")
        cands = sorted((z for z in by_scene.get(k, []) if z["ok"]),
                       key=lambda z: -z["size"])
        if not cands:
            say("- **FAIL** no usable zip for this scene")
            say("")
            continue
        folder = None
        for z in cands:
            say(f"  using `{z['path'].name}`")
            try:
                folder = extract(z)
                results[k] = {"zip": z["path"].name}
                break
            except zipfile.BadZipFile as e:
                say(f"  **BROKEN inside**: {e}. Trying the next zip.")
        if folder is None:
            say("- **FAIL** every zip for this scene is damaged")
            continue
        r = check_scene(k, folder, a.force)
        results[k].update(r)
        say("")
        for v, t in r["verdicts"]:
            say(f"- **{v}** {t}")
        if "vv" in r:
            vv = r["vv"]
            say(f"- VV median RTC {vv['rtc_med']:.2f} dB, GRD {vv['grd_med']:.2f} dB; "
                f"RTC minus GRD median {vv['diff_med']:+.2f} dB, NMAD {vv['diff_nmad']:.2f} dB; "
                f"RTC 1st to 99th percentile {vv['rtc_p'][0]:.1f} to {vv['rtc_p'][1]:.1f} dB")
            vh = r["vh"]
            say(f"- VH RTC minus GRD median {vh['diff_med']:+.2f} dB, NMAD {vh['diff_nmad']:.2f} dB")
        if r.get("ls"):
            say("- layover/shadow map values inside the corridor, % of cells: " +
                ", ".join(f"{v}: {p:.2f}" for v, p in sorted(r["ls"].items())))
        if r.get("inc"):
            say(f"- local incidence angle, 1st / 50th / 99th percentile: "
                f"{r['inc'][0]:.1f} / {r['inc'][1]:.1f} / {r['inc'][2]:.1f} degrees")
        if r.get("readme_ls"):
            say("- what the product README says about the layover/shadow map:")
            for t in r["readme_ls"]:
                say(f"    > {t}")
        say("")

    fails = [(k, v, t) for k, r in results.items() for v, t in r.get("verdicts", []) if v != "PASS"]
    missing = [k for k in SCENES if k not in results]
    say("## 4. Summary")
    say("")
    if not fails and not missing:
        say("**PASS.** Both scenes extracted, on the project grid and consistent with the GRD.")
    else:
        for k in missing:
            say(f"- **FAIL** {SCENES[k][1]}: no usable zip")
        for k, v, t in fails:
            say(f"- **{v}** {SCENES[k][1]}: {t}")
    say("")
    say("The report is `RTC CHECK - " + today + ".md`.")

    NOTES.mkdir(exist_ok=True)
    rep = NOTES / f"RTC CHECK - {today}.md"
    rep.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport: {rep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
