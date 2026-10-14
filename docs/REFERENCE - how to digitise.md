# How to digitise the reference extent

Rewritten 10 September 2026. The 5 September version described drawing every
polygon by hand and gave a budget its own arithmetic disproved. Both are fixed.

**One action per step. Do not skip Step 0.**

---

## The budget, corrected

**About eight hours per event.**

Eight windows, two passes per event, roughly twenty-five minutes a pass. That is
400 minutes of drawing, which is six hours and forty minutes, and about eight
hours once the verification passes are counted. The old version of this file said
three hours, which its own per-window figure contradicted.

An earlier estimate of seventy to a hundred hours was wrong by roughly ten times
and is retired. The error was treating this as painting an area. **You trace a
water edge.** A 5 km window holds a few kilometres of bank; the rest of the box is
hillside nobody touches.

Spread it over sittings. Eye fatigue shows up in the polygons and it is visible to
anyone who looks.

**Windows stay at 5 km and there are eight of them.** The 3 km recommendation is
withdrawn. Drawing time scales with the perimeter of water, not the area of the
box, so a smaller box saves hillside nobody was going to trace and costs strata
coverage. Eight at 5 km is what covers all fourteen strata.

---

## What has changed since 5 September: most of the setup is automated

The `gis_tools` package now does the parts of this guide that used to be manual
QGIS work. The steps below describe **what the tools do and what a good result
looks like**, not how to do it by hand.

| Used to be | Now |
|---|---|
| Digitise the centreline by hand in QGIS, about an hour | `RUN CENTRELINE.bat` |
| Raster, Miscellaneous, Merge, four times per event | `RUN MOSAICS.bat` |
| Create GeoPackages, add the `window` field, set snapping | `RUN QGIS SETUP.bat` |
| Draw every polygon from nothing | `RUN DRAFT POLYGONS.bat`, then verify and correct |

**The last row is the important one.** The method is now draft-and-correct. The
tool proposes, you judge.

---

# Step 0. Before you open QGIS

**0.1 The centreline must have passed.** All five acceptance checks in
`centreline_report.md` (gis_tools): both anchors snapped, 90 to 100 km, about 270 m of drop,
under 2 per cent of vertices rising, and the two routes agreeing to a few tens of
metres. Windows placed against approximate town anchors sit off the real river,
and moving them later means redrawing every polygon inside them.

**0.2 The strata must have been rebuilt with `--corridor`.** Otherwise every
share is against 2,408 km² while the flood maps describe 807.

**0.3 Place the windows.**

```
conda activate fyp
cd /d "<project folder>\04 Code\FYP_code"

python 05_reference_windows.py --strata data/strata_2021 --grid data/flood_2021_v2/flood_2021_otsu.tif --pred data/flood_2021_v2/flood_2021_otsu.tif --centreline "..\..\05 Data\boundaries\corridor_centreline.geojson" --n 8 --size 5 --out data/reference_windows_2021
```

Then the same for 2023 with `--strata data/strata_2023_v2`, its own grid, and
`--out data/reference_windows_2023`.

**0.4 Read `reference_windows.md`, and look at the figure.**

Two checks, and the second is new.

- **All fourteen strata represented.** If not, raise `--n` to 9 or 10 and run
  again. Do this now, not after you have drawn polygons inside the old windows.
- **No window more than about a quarter covered by the cloud gap.** Eleven per
  cent of the corridor has no usable 2023 optical imagery, mapped in
  `S2_2023_event_gapmask.tif`. A window mostly inside that gap cannot be
  digitised honestly.

`RUN WINDOW MAP.bat` draws the sampling design and prints the overlap per window.
`RUN QGIS SETUP.bat` also writes the same figure as `sampling_design_<tag>.png`
into each event folder, which is where you will actually be looking when you
start a session. **Open it before you draw anything.**

**Never re-run window placement with different settings until the windows contain
more water.** `--pred` is passed only so the report can tell you how much
predicted water you are walking into. Placing validation where the model already
sees water makes its false negatives invisible, and it is the one error in this
stage that cannot be recovered from later.

**0.5** If a window fails the gap check twice in a row, stop. Moving a window by
hand needs a reason recorded.

---

# Step 1. Build the QGIS projects

```
"<project folder>\04 Code\gis_tools\RUN QGIS SETUP.bat"
```

This writes, per event:

- a `.qgz` project in **EPSG:32644**, with the event mosaic, the dry mosaic, the
  MNDWI rasters, the windows layer and the water mask already loaded and styled
- empty `water_event` and `water_dry` GeoPackage tables with a `window` integer
  field
- snapping already configured to the windows layer
- `sampling_design_<tag>.png`
- a `LOAD ME.md`

`--no-figure` skips the figure if you only want the GeoPackage and the styles.

**Check before going on.** Open the project. Layers should line up. If they do
not, the project CRS is wrong; a CRS mismatch in QGIS is silent, and layers draw
happily in the wrong place relative to each other.

---

# Step 2. Draft the polygons

```
"<project folder>\04 Code\gis_tools\RUN DRAFT POLYGONS.bat"
```

It writes suggested flood polygons into **`water_event_draft`** and
**`water_dry_draft`** and **never** into the real tables.

**That separation is deliberate and it must stay.** A fully automatic reference is
not independent of the automatic methods it is meant to judge. Moving a polygon
into the real table has to be a deliberate act, because your judgement is what
makes the reference usable as ground truth.

---

# Step 3. Verify and correct, one window at a time

This is the work, and it is the part that decides how good the reference is.

**Do the whole event pass first, all eight windows, then the whole dry pass. Do
not alternate.** Switching between the two images every few minutes is how the two
passes end up drawn to different standards, and the difference between them is
your answer.

For each window:

**3.1** Zoom to the window. Set the scale box at the bottom to **1:5000** and work
at that scale. Zoom in to check something, then come back to 1:5000 to draw.

**3.2** Turn on `water_event_draft`. Look at where the draft boundary sits against
the Sentinel-2 image and the MNDWI overlay.

**3.3 Where the draft is right**, select the polygon and copy it into
`water_event`. Type the window number into the `window` field.

**3.4 Where the draft is wrong**, edit the vertices in `water_event`, or redraw
that stretch. Trace the water surface using **MNDWI as the guide and true colour
as the check**.

**3.5 Draw only inside the window.** Where water crosses the window edge, follow
the edge. Snapping does this for you.

**3.6 Where the image is cloudy in both dates**, leave it out and record the
window as partial. **A guessed boundary is worse than a missing one.**

**3.7 Save the layer edits every window.** Not every hour.

**Record how long window 1 takes you.** The estimate is about twenty-five minutes
a pass. If window 1 takes more than an hour, stop, because at that
rate something is wrong with the workflow rather than with the window count.

---

# Step 4. What counts as water, and the three things that will fool you

None of this is mechanical, and it is why the operator is still in the loop.

## 4.1 Terrain shadow is the big one

You are working in a gorge with walls rising hundreds of metres. In the morning,
whole hillsides sit in shadow, and shadowed slopes are dark in every band. MNDWI
can read them as water.

**The tell: shadow follows the topography, water follows the valley floor.** If a
dark patch runs up a hillside, it is shadow. Load
`05 Data\dem\DEM_COP_GLO30_30m.tif` with a hillshade style for the windows where
this keeps happening.

Getting this wrong is not a small error. It is the same failure mode that made
single-date Otsu call 74.66 km² of this corridor flooded, arriving from the
optical side instead of the radar side.

The draft tool rejects terrain shadow in testing — wet fraction came out 3.97 and
4.00 per cent where a shadow patch would have added 2.1 points — but that was
synthetic terrain built to be rejected. Check it on the real thing.

## 4.2 Wet sand and gravel bars

The Alaknanda has wide gravel beds. Saturated gravel reads between land and water
in MNDWI and it is genuinely ambiguous. This is where MNDWI and NDWI disagree,
which is why both were exported. Band 1 of the indices file is MNDWI, band 2 is
NDWI.

**Decide a rule now and apply it to every pass, both events**, and to July as well
if that event qualifies. The suggested rule, already written into the Methodology:
water is water where MNDWI is above zero **and** the true colour shows a smooth
surface without visible texture.

**Write down which way you went.** It goes in the methods and you will be asked
about it.

## 4.3 Cloud edges and cloud shadow

Band 6 of the `_10m` file is SCL, the scene classification. Load it as a separate
paletted layer and turn it on when a window looks odd. **Class 3 is cloud shadow,
8 to 10 are cloud, 11 is snow.**

Where a window is under cloud in **both** dates of the mosaic, do not guess.
Leave it undrawn and record the window number. `06_rasterise_reference.py` has a
`--holes` option that marks those areas nodata so they are excluded from scoring
rather than counted as dry land. **A cloud hole scored as dry is a false negative
that never happened.**

---

# Step 5. The dry pass

**5.1** Switch the event mosaic off and the dry mosaic on.

**5.2** Select `water_dry` and repeat Step 3 for the same eight windows, using
`water_dry_draft` as the starting point.

**5.3 Draw it the same way.** Same scale, same rule about gravel, same person,
same screen, same week. **The two passes are subtracted from each other, so
anything you do differently between them becomes flood.**

---

# Step 6. Build the raster

**6.1** Turn editing off and save both layers.

**6.2**

```
conda activate fyp
cd /d "<project folder>\04 Code\FYP_code"

python 06_rasterise_reference.py --event "..\..\05 Data\sentinel2\reference_2021.gpkg" --event-layer water_event --dry-layer water_dry --windows data/reference_windows_2021/reference_windows.geojson --grid data/flood_2021_v2/flood_2021_otsu.tif --out data/reference_2021.tif
```

**6.3 Load `reference_2021.tif` back into QGIS over the imagery and look at it.**
It should be 1 where you drew flood, 0 elsewhere inside the windows, and nodata
outside them and in the cloud holes.

**6.4 Check the numbers it prints**, against
`04 Code\CHECKS - what good looks like.md` section 7.

- Total reference flood area of the order of **1 to 3 km²**, matching what change
  detection reports for the whole corridor. Somewhat less is expected, because the
  windows cover about a quarter of the corridor.
- Cells outside the windows are nodata, not zero.
- A window showing 0 per cent flooded is either genuine, in which case record it,
  or the polygons never got promoted out of the `_draft` table.

If the flood area is tens of km², something is wrong with the polygons or the
window attribution. Open the GeoPackage and check the `window` field is filled.

---

# Step 7. Score it

**7.1** Only after Step 6.4 looks right.

```
python 04_stratified_metrics.py --pred data/flood_2021_v2/flood_2021_otsu.tif --truth data/reference_2021.tif --strata data/strata_2021 --label "Otsu single-image baseline, 2021 event" --out "..\..\06 Outputs\tables"
```

**7.2 This is the first time `04_stratified_metrics.py` has ever run.** Not once,
not on synthetic data. Treat the first run as a test of the script, not as a
result. **If it produces a clean table immediately, that is when to be most
suspicious.**

**7.3 Two sanity checks before you trust it.**

- **The Otsu baseline should score badly.** Single-date Otsu is a documented
  failure on this corridor. If it scores well against your hand-drawn reference,
  one of the two is wrong.
- **Gorge and valley should differ.** The whole reporting design assumes they do.
  If the two reaches score identically, the reach stratum is not being applied.

**7.4 Keep everything it prints** with the scores.

---

# The two things to write down while you work

**A running note of where you were unsure.** You should be able to point at three
places on each event where you are confident of the boundary and two where you are
not. Those five places are the honest content of the limitations paragraph, and
you will not remember them tomorrow.

**Which rule you settled on for gravel bars**, from 4.2. One sentence. It goes in
the methods.

---

# If something goes wrong

`04 Code\WHEN IT BREAKS - error index.md`, section 8 for QGIS specifically.

The two that come up most here:

- **Layers do not line up.** Project, Properties, CRS, set EPSG:32644. A mismatch
  is silent.
- **Polygon edges land a metre off the window boundary.** Snapping is off. The
  setup tool configures it; a project built by hand does not have it.
