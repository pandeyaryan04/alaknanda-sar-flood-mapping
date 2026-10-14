# gis_tools

Replaces the hand work in QGIS for the Alaknanda corridor project, and does
the scripted downloads.

Nothing here needs QGIS. QGIS is for looking at the results.

---

## What to do the first time

1. Copy the whole `gis_tools` folder into `04 Code\`.
2. Copy `.env.example` to `.env`.
3. Open `.env` in Notepad. Check `PROJECT_ROOT` points at your project folder.
   Change nothing else yet.
4. Double-click **`SELFTEST.bat`**. It writes nothing.
5. Read the PASS/FAIL lines. If anything says FAIL, see "When something fails"
   below.
6. Double-click **`RUN CENTRELINE.bat`**.
7. Open `06 Outputs\gis_tools\centreline\centreline_report.md` and check the
   five acceptance lines.
8. Keep `centreline_report.md` and `centreline_qc.png` with the project
   records.

Steps 6 to 8 replace about an hour of digitising the centreline at 1:25000.

---

## The tools

| double-click | what it replaces | writes |
|---|---|---|
| `SELFTEST.bat` | nothing; run it first | nothing |
| `RUN CENTRELINE.bat` | digitising the centreline, one hour in QGIS | `corridor_centreline.geojson` and `.gpkg`, a QC csv, a QC png, a report |
| `RUN MOSAICS.bat` | five merges in QGIS, plus the cloud-gap check of each pair | `S2_<event>_mosaic.tif`, `_mndwi.tif`, `_gapmask.tif`, a coverage table |
| `RUN WATERMASK AUDIT.bat` | twenty minutes of panning in QGIS | a table, a markdown note, six map panels |
| `RUN RESERVOIR CHECK.bat` | checking the Srinagar reservoir level by eye | area by date, a verdict, a picture |
| `RUN QGIS SETUP.bat` | steps 1.1 to 4.5 of the digitising reference | an empty GeoPackage with the right schema, four `.qml` styles, a `LOAD ME.md` |
| `RUN DRAFT POLYGONS.bat` | part of the digitising itself | candidate polygons in `_draft` tables. **Read "The draft-polygon rule" below before using it** |
| `RUN DOWNLOADS.bat` | Earth Engine, ASF, Earthdata and IMD by hand | submits jobs, downloads, reports coverage |

Run order matters in one place only: the centreline comes first, because
`05_reference_windows.py` needs it, and the draft polygons need the windows
that script places.

```
RUN CENTRELINE.bat
  -> 05_reference_windows.py            (existing script, needs the centreline)
     -> RUN QGIS SETUP.bat
        -> RUN DRAFT POLYGONS.bat       (optional)
RUN MOSAICS.bat                         (independent)
RUN WATERMASK AUDIT.bat                 (wants the centreline for widths)
RUN RESERVOIR CHECK.bat                 (independent)
```

---

## The draft-polygon rule

The reference extent is the ground truth that Otsu, the Random Forest and the
other models are scored against. It is hand-digitised, which is what keeps it
independent of the automatic methods it judges. If the polygons were generated
from an MNDWI threshold they would stop being independent, because both sides
would be a threshold on a spectral index chosen by a script.

So `t06_draft_polygons.py`:

- writes only to `water_event_draft` and `water_dry_draft`, in a separate
  GeoPackage, and never to `water_event` or `water_dry`;
- leaves the move from draft to real to the operator, by hand, so it is always
  known which polygons were drawn and which were proposed;
- reports a marginal fraction per window (the share of the window where MNDWI
  is close to the threshold), so the ambiguous windows, usually gravel bars
  and terrain shadow, can be drawn by hand rather than corrected. The
  water/not-water call stays with the operator.

If any draft polygon is kept, the methods must say so. The sentence the tool
prints is:

> Reference polygons were initialised from an MNDWI and SWIR threshold and
> then inspected and corrected by hand at 1:5000 against true-colour imagery,
> with the operator making the final water/not-water judgement in all cases,
> including gravel bars and terrain-shadowed reaches.

---

## The .env

Every path, threshold, date and coordinate lives in `.env`. There are no
paths or numbers in the Python. A command-line flag can override a value for
a one-off, and when it does the script says so.

Two keys are worth understanding before you change anything:

**`CRS_METRIC`** must be the UTM zone the corridor actually sits in.
EPSG:32644 is UTM 44N, which is right for roughly 78 to 84 east. The
self-test checks this against the corridor centroid and fails if they
disagree. This is the setting that will catch you on a different river.

**`REBUILD_CORRIDOR`** is `false` and should stay false for now. True makes
the centreline tool also rebuild the 5 km buffer, which invalidates every
raster clipped to the current corridor and every accuracy figure computed
from them. That is a day of recomputation, not a side effect.

---

## What every tool guarantees

- A self-test runs before any work, and the `.bat` stops if it fails.
- Nothing existing is overwritten. If an output path exists the script stops
  and tells you; `--force` overrides, and no `.bat` passes `--force`.
- Nothing in `05 Data\` is modified. Derived files get new names.
- Every run writes `run_manifest.json` beside its outputs: the resolved
  `.env`, every input with its size and modification time, every output,
  every threshold applied, and the runtime.
- Every run writes a `.log`.
- Runs are deterministic. Same inputs, same bytes out.
- Every printed number carries its unit.
- Where the code has to assume something, it prints ASSUMPTION and the value.

---

## When something fails

Keep these three things together; they are what is needed to find the cause:

1. the console text, all of it
2. the `.log` file from the output folder
3. `run_manifest.json` if one was written

A FAIL in the self-test is not a crash. It is the package saying what it
cannot find, which is usually a path in `.env` or a file that has not been
produced yet.

---

## Reusing this for an M.Tech

Copy the folder, then in `.env`:

1. `PROJECT_ROOT`, `CORRIDOR`, `DEM_ROUTING`, `S2_DIR`
2. `CRS_METRIC` to the right UTM zone
3. `ANCHOR_UP` and `ANCHOR_DOWN`, and the towns in `CHECK_TOWNS`
4. `MNDWI_WATER` and `SWIR_WATER` if the water is turbid or sediment-laden.
   Himalayan meltwater and a Gangetic plain river are not the same target
5. `RESERVOIR_WIDTH_M`, `MIN_WATER_HA` and `LENGTH_MIN_KM` / `LENGTH_MAX_KM`
   to the new river's scale

No Python edits. If a new catchment needs one, that is a bug worth
reporting.

---

## Tests

`tests/make_synthetic.py` builds a 14 km synthetic corridor with a meandering
channel, a 600 m reservoir, a dead-end tributary, terrain shadow that must be
rejected, and a different cloud patch in each of two dates.
`tests/make_test_env.py` derives `tests/test.env` from `.env.example`, so a
key added to the real config is exercised by the tests rather than silently
untested.

```
python tests/make_synthetic.py
python tests/make_test_env.py
python tests/run_tests.py
```

What the synthetic case proves, because the answers are known in advance:

- the centreline takes the middle of the reservoir rather than wandering
  across it
- it does not enter the dead-end tributary
- terrain shadow is rejected: the cleaned water mask comes out at exactly the
  true water area
- the union of two dates fills each date's cloud gap, and the mosaic reports
  100.00 per cent
- the elevation profile falls monotonically
- the braided-bed test polygon, built 280 m wide against 80 m of water,
  reports the commission area that width difference implies

---

## Files

```
common.py              config, raster loading with assertions, dual-CRS
                       vector writing, logging, the run manifest
route.py               cost surfaces and the least-cost path
linelib.py             chainage, smoothing, densifying, widths, comparison
t00_selftest.py        runs everything else's self-test
t01_centreline.py  ..  t07_download.py
.env.example           copy to .env
tests/                 synthetic corridor and the test runner
```
