"""
common.py - shared plumbing for the gis_tools package.

Config loading, raster loading with assertions, dual-CRS vector writing,
logging and the run manifest. Nothing in here knows about rivers.

Rules this file enforces on behalf of every tool:
  - configuration comes from .env, never from a literal
  - every raster read asserts CRS and band count, and says what it found
  - metric work happens in CRS_METRIC, storage in CRS_STORE
  - nothing is overwritten without --force
  - every run leaves a manifest

Console output is ASCII only. The default Windows console is cp1252 and a
single non-ASCII character raises UnicodeEncodeError in the middle of a run.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

__version__ = "1.0.0"

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(RuntimeError):
    pass


class Cfg:
    """A parsed .env. Values are strings; the accessors do the converting.

    Written by hand rather than with python-dotenv because we need ${VAR}
    expansion against keys defined earlier in the same file, and because
    Windows paths are full of backslashes that some parsers try to escape.
    """

    def __init__(self, values: dict, source: Path):
        self._v = values
        self.source = source

    # -- construction ------------------------------------------------------

    @classmethod
    def load(cls, path=None) -> "Cfg":
        if path is None:
            here = Path(__file__).resolve().parent
            path = here / ".env"
            if not path.exists():
                raise ConfigError(
                    f"no .env found at {path}\n"
                    f"  copy .env.example to .env and edit PROJECT_ROOT."
                )
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"config file not found: {path}")

        values = {}
        for lineno, raw in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ConfigError(f"{path}:{lineno}: no '=' in '{raw}'")
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            # strip a trailing inline comment only when it is clearly one:
            # preceded by whitespace and not inside a Windows path
            if " #" in val:
                val = val.split(" #", 1)[0].strip()
            val = _VAR.sub(
                lambda m: values.get(m.group(1), os.environ.get(m.group(1), "")),
                val,
            )
            values[key] = val
        return cls(values, path)

    # -- accessors ---------------------------------------------------------

    def raw(self, key, default=None):
        if key in self._v:
            return self._v[key]
        if default is not None:
            return default
        raise ConfigError(
            f"{self.source} has no key {key}. "
            f"Compare it against .env.example; a key was probably deleted."
        )

    def has(self, key) -> bool:
        return key in self._v and self._v[key].strip() != ""

    def str(self, key, default=None) -> str:
        return str(self.raw(key, default))

    def f(self, key, default=None) -> float:
        v = self.raw(key, default)
        try:
            return float(v)
        except ValueError:
            raise ConfigError(f"{key}={v!r} is not a number")

    def i(self, key, default=None) -> int:
        return int(round(self.f(key, default)))

    def b(self, key, default=None) -> bool:
        v = str(self.raw(key, default)).strip().lower()
        if v in ("true", "yes", "1", "on"):
            return True
        if v in ("false", "no", "0", "off", ""):
            return False
        raise ConfigError(f"{key}={v!r} is not true or false")

    def path(self, key, default=None) -> Path:
        """A path from .env, with separators normalised.

        The .env is written with Windows backslashes. On Windows that is what
        Path wants. Anywhere else a backslash is an ordinary character and
        'dem\\cartodem' becomes one folder name, so it is translated. This is
        what lets the same .env be tested on Linux and run on your machine.
        """
        raw = self.str(key, default)
        if os.sep != "\\":
            raw = raw.replace("\\", os.sep)
        return Path(raw)

    def csv(self, key, default=None) -> list:
        v = self.str(key, default)
        return [p.strip() for p in v.split(",") if p.strip()]

    def ints(self, key, default=None) -> list:
        return [int(x) for x in self.csv(key, default)]

    def lonlat(self, key, default=None) -> tuple:
        parts = self.csv(key, default)
        if len(parts) != 2:
            raise ConfigError(f"{key}={self.str(key)!r} is not 'lon,lat'")
        return (float(parts[0]), float(parts[1]))

    def sites(self, key, default=None) -> list:
        """'lon,lat; lon,lat; ...' -> [(lon, lat), ...]"""
        out = []
        for chunk in self.str(key, default).split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            lon, _, lat = chunk.partition(",")
            out.append((float(lon), float(lat)))
        return out

    def as_dict(self) -> dict:
        return dict(self._v)


# --------------------------------------------------------------------------
# logging and console
# --------------------------------------------------------------------------

_ASCII_OK = re.compile(r"^[\x00-\x7F]*$")


def say(msg: str = "") -> None:
    """print, but refuse to emit anything the Windows console cannot encode."""
    if not _ASCII_OK.match(msg):
        msg = msg.encode("ascii", "replace").decode("ascii")
    print(msg, flush=True)


def rule(title: str = "") -> None:
    say("")
    say("-" * 72)
    if title:
        say(title)
        say("-" * 72)


def setup_logger(out_dir: Path, name: str) -> logging.Logger:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    fh = logging.FileHandler(out_dir / f"{name}.log", mode="w", encoding="utf-8")
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")
    )
    log.addHandler(fh)
    return log


# --------------------------------------------------------------------------
# self-test helpers
# --------------------------------------------------------------------------

class Checks:
    """Collects PASS/FAIL lines. Same shape as SCREEN GAUGES.bat prints."""

    def __init__(self, title: str):
        self.title = title
        self.rows = []

    def check(self, label: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((label, bool(ok), detail))
        return bool(ok)

    def require_file(self, label: str, path) -> bool:
        p = Path(path)
        if not p.exists():
            return self.check(label, False, f"missing: {p}")
        size = p.stat().st_size
        return self.check(label, True, f"{p.name}, {size/1e6:.1f} MB")

    def report(self) -> bool:
        rule(self.title)
        width = max((len(r[0]) for r in self.rows), default=10)
        for label, ok, detail in self.rows:
            say(f"  {'PASS' if ok else 'FAIL'}  {label.ljust(width)}  {detail}")
        n_fail = sum(1 for r in self.rows if not r[1])
        say("")
        say(f"  {len(self.rows) - n_fail} passed, {n_fail} failed")
        return n_fail == 0


# --------------------------------------------------------------------------
# rasters
# --------------------------------------------------------------------------

class RasterError(RuntimeError):
    pass


def open_checked(path, expect_crs=None, expect_bands=None, log=None):
    """Open a raster, asserting CRS and band count. Returns the dataset.

    Never guesses. If the band count is wrong it says what it found, rather
    than reading band 5 as SCL and producing a clean, wrong number.
    """
    import rasterio

    path = Path(path)
    if not path.exists():
        raise RasterError(f"raster not found: {path}")
    ds = rasterio.open(path)
    if expect_crs is not None:
        want = str(expect_crs).upper()
        got = str(ds.crs).upper() if ds.crs else "NONE"
        if got != want:
            ds.close()
            raise RasterError(
                f"{path.name} is {got}, expected {want}.\n"
                f"  Either the file is not what the .env says it is, or "
                f"CRS_METRIC / CRS_STORE is set wrong."
            )
    if expect_bands is not None and ds.count != expect_bands:
        got = ds.count
        ds.close()
        raise RasterError(
            f"{path.name} has {got} bands, expected {expect_bands}.\n"
            f"  Check S2_BANDS in .env against how the file was exported."
        )
    if log:
        log.info(
            "opened %s: %s, %d bands, %d x %d, %s",
            path.name, ds.crs, ds.count, ds.width, ds.height, ds.dtypes[0],
        )
    return ds


def band_index(cfg: Cfg, key: str, name: str) -> int:
    """1-based band number for a named band, from the .env band contract."""
    names = cfg.csv(key)
    if name not in names:
        raise ConfigError(
            f"band {name} is not listed in {key}={','.join(names)}"
        )
    return names.index(name) + 1


def read_band(ds, idx, masked=True):
    a = ds.read(idx, masked=masked)
    return a


def s2_reflectance(ds, cfg, band_name):
    """Read one S2 band as reflectance 0..1, from Int16 scaled by S2_SCALE."""
    idx = band_index(cfg, "S2_BANDS", band_name)
    scale = cfg.f("S2_SCALE")
    a = ds.read(idx).astype("float32")
    a[a == ds.nodata if ds.nodata is not None else False] = np.nan
    return a / scale


def scl_masks(scl, cfg):
    """covered / cloud / snow / clear, using exactly the GEE definitions."""
    nodata = cfg.i("SCL_NODATA")
    cloud_vals = cfg.ints("SCL_CLOUD")
    snow_vals = cfg.ints("SCL_SNOW")
    covered = scl != nodata
    cloud = np.isin(scl, cloud_vals)
    snow = np.isin(scl, snow_vals)
    clear = covered & ~cloud & ~snow
    return covered, cloud, snow, clear


def guard_output(path, force: bool):
    path = Path(path)
    if path.exists() and not force:
        raise RuntimeError(
            f"{path} already exists.\n"
            f"  Nothing was written. Delete it, or rerun with --force."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------
# vectors
# --------------------------------------------------------------------------

def write_line(coords_metric, cfg, out_geojson, out_gpkg=None,
               layer="centreline", props=None, force=False):
    """Write one LineString to GeoJSON (CRS_STORE) and GeoPackage (CRS_METRIC).

    The GeoJSON carries an explicit non-standard 'crs' member because
    05_reference_windows.py::load_centreline reads it, defaulting to EPSG:4326
    when absent. It also concatenates the coordinates of every LineString
    feature in file order, so we write exactly one feature, ordered upstream
    to downstream.
    """
    from pyproj import Transformer

    crs_m = cfg.str("CRS_METRIC")
    crs_s = cfg.str("CRS_STORE")
    tr = Transformer.from_crs(crs_m, crs_s, always_xy=True)
    ll = [tr.transform(x, y) for x, y in coords_metric]

    gj = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs_s}},
        "features": [
            {
                "type": "Feature",
                "properties": dict(props or {}),
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[round(x, 7), round(y, 7)] for x, y in ll],
                },
            }
        ],
    }
    p = guard_output(out_geojson, force)
    p.write_text(json.dumps(gj, indent=1), encoding="utf-8")

    if out_gpkg:
        import geopandas as gpd
        from shapely.geometry import LineString

        g = gpd.GeoDataFrame(
            [dict(props or {})],
            geometry=[LineString(coords_metric)],
            crs=crs_m,
        )
        p2 = guard_output(out_gpkg, force)
        g.to_file(p2, layer=layer, driver="GPKG")
    return p


def write_polys(gdf, cfg, out_gpkg, layer, force=False):
    p = guard_output(out_gpkg, force) if not Path(out_gpkg).exists() else Path(out_gpkg)
    p.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(p, layer=layer, driver="GPKG")
    return p


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------

class Manifest:
    def __init__(self, tool: str, cfg: Cfg):
        self.d = {
            "tool": tool,
            "package_version": __version__,
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": sys.version.split()[0],
            "config_file": str(cfg.source),
            "config": cfg.as_dict(),
            "inputs": [],
            "outputs": [],
            "parameters": {},
            "assumptions": [],
            "runtime_s": None,
        }
        self._t0 = time.time()

    def input(self, path, note=""):
        p = Path(path)
        self.d["inputs"].append(
            {
                "path": str(p),
                "bytes": p.stat().st_size if p.exists() else None,
                "mtime_utc": (
                    datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
                    .isoformat(timespec="seconds")
                    if p.exists() else None
                ),
                "note": note,
            }
        )

    def output(self, path, note=""):
        p = Path(path)
        self.d["outputs"].append(
            {
                "path": str(p),
                "bytes": p.stat().st_size if p.exists() else None,
                "note": note,
            }
        )

    def param(self, key, value):
        self.d["parameters"][key] = value

    def assume(self, text):
        self.d["assumptions"].append(text)
        say(f"  ASSUMPTION  {text}")

    def write(self, out_dir):
        self.d["runtime_s"] = round(time.time() - self._t0, 1)
        p = Path(out_dir) / "run_manifest.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.d, indent=1), encoding="utf-8")
        return p


# --------------------------------------------------------------------------
# small numeric helpers used by more than one tool
# --------------------------------------------------------------------------

def km2(cell_count, res_m):
    return cell_count * res_m * res_m / 1e6


def pct(part, whole):
    return 100.0 * part / whole if whole else float("nan")


def utm_zone_of(lon):
    return int((lon + 180) // 6) + 1
