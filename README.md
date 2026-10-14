# Terrain-stratified SAR flood mapping on the Alaknanda corridor

Sentinel-1 flood mapping along about 121 km of the Alaknanda and Ganga in
Uttarakhand, India, from Rudraprayag through Srinagar (Garhwal) and Devprayag to
Rishikesh. Accuracy is reported by terrain class and separately for the
confined gorge above Devprayag and the open valley below it, never as one
averaged figure.

Final year B.Tech Civil Engineering project, G. B. Pant University of
Agriculture and Technology, Pantnagar. Work in progress (semester 1 of 2,
October 2026).

---

## Why the accuracy is split by terrain

Published flood-mapping accuracies change more with the landscape than with the
algorithm. In a Himalayan gorge the radar looks at steep valley walls, so parts
of the valley fall into layover or shadow, and the river is often narrower than
a few pixels. Below Devprayag the valley opens out. A single accuracy figure for
both describes neither, so every figure here is reported against local incidence
angle, layover and shadow, height above nearest drainage (HAND), valley width and
reach.

## Study area and events

| | |
|---|---|
| Corridor | 3 km either side of the traced river, clipped to the imagery footprint: **606.35 km²** (EPSG:32644, UTM 44N) |
| Reach split | at Devprayag (78.5981 E): gorge 52.58 per cent of the corridor, open valley 47.42 per cent |
| Events | 21 October 2021 (orbit 63, descending) and 16 August 2023 (orbit 129, ascending), both two days after the flood peak. 19 July 2023 is used as a radar-only scene |
| Reference | hand-digitised flood extent in eight 5 km windows per event, placed by terrain only, drawn on Sentinel-2 images of the same days against February images of the normal river |

The two events are imaged from opposite sides of the valley, so terrain effects
can be separated from method error.

## What is in this repository

Code and configuration only. No data: the imagery, DEMs and the Sen1Floods11
benchmark are tens of gigabytes and are excluded by `.gitignore`. All data
sources are free and public (list below).

```
FYP_code/          the pipeline: scene screening, terrain strata, flood maps, models, scoring
gis_tools/         standalone tools that replace hand work in QGIS, plus scripted downloads
gee/               Google Earth Engine export scripts
jalsar/            Jal-SAR, a flood-mapping tool built on the trained models
docs/              the setup runbook and the digitising reference
provenance/        package versions, data checksums, and the settings of every reported run
tests/             automated tests on synthetic data
```

`gis_tools` has its own `README.md` and its own `.env` settings file
(`.env.example` is the template). It imports nothing from `FYP_code`.

## Environment

Windows 11, Miniconda, Python 3.11, PyTorch 2.6 with CUDA 12.4, a 4 GB laptop
GPU.

```
cd FYP_code
conda env create -f environment.yml
conda activate fyp
pip install -r requirements.txt
```

PyTorch is installed with pip afterwards (`docs/SETUP_RUNBOOK.md`). The
models, TerraMind included, run in `fyp`, which also holds TerraTorch 1.2.11
(`provenance/pip_freeze_fyp.txt`). The separate environment `fypdl` has a
CPU-only PyTorch and is not used for the models.

## The pipeline, in run order

| Step | Script | What it does |
|---|---|---|
| 0 | `FYP_code/00_check_machine.py` | GPU, CUDA, disk and package check |
| 1 | `FYP_code/01_scene_inventory.py` | Screens Sentinel-1 scenes over the corridor: coverage, orbit, pass |
| 2 | `FYP_code/02_fetch_sen1floods11.py` | Downloads the Sen1Floods11 benchmark from its public bucket |
| 3 | `gee/10_reexport_corridor_v2.js` | Exports Sentinel-1, Sentinel-2, DEMs, water, land cover, population and rainfall for the corridor |
| 4 | `gis_tools` (`RUN CENTRELINE`, `RUN MOSAICS`, `RUN WATERMASK AUDIT`, `RUN RESERVOIR CHECK`) | River centreline, cloud-free Sentinel-2 pairs, permanent-water audit, regulated reservoir |
| 5 | `FYP_code/03_terrain_strata.py` | Local incidence angle, layover and shadow, HAND, valley width, reach |
| 6 | `FYP_code/03b_otsu_baseline.py` (`RUN FLOOD MAPS v3.bat`) | Otsu flood maps: single-image baseline and change detection against four dry-season dates |
| 7 | `FYP_code/05_reference_windows.py`, `10_window_map.py` | Places the digitising windows by terrain only; checks them against cloud gaps |
| 8 | `gis_tools` (`RUN QGIS SETUP`, `RUN DRAFT POLYGONS`) | QGIS project and draft outlines for the hand digitising |
| 9 | `FYP_code/06_rasterise_reference.py` | Turns the hand-drawn polygons into a reference raster: flood = event water and not February water |
| 10 | `FYP_code/04_stratified_metrics.py` | Scores a flood map against the reference by terrain class and reach: precision, recall, F1, IoU, OA, Matthews correlation. Classes with under 100 reference flood pixels are merged with a neighbour, and F1, IoU and MCC carry 95 % intervals from resampling the digitising windows |
| 11 | `FYP_code/11_sen1floods11_terrain.py` | Height above nearest drainage (HAND) and slope for every Sen1Floods11 training chip, from Copernicus GLO-30 and JRC water via Earth Engine, computed with the same functions as the corridor strata so the models see the same terrain inputs in training and on the corridor |
| 12 | `FYP_code/12_permanent_water_mask.py` to `16_landcover_corridor.py` (`RUN REMAKE CHECK.bat`) | Remake, from the source data, the files first made with one-off code: the permanent-water mask, the cloud-gap holes, the Srinagar reservoir outline, the February layer of the reference completed by rule, and land cover inside the corridor. Each writes to `06 Outputs/remake/` and is compared with the file in use |
| 13 | `FYP_code/20_models_preflight.py` (`RUN MODELS PREFLIGHT.bat`) | Checks the machine and every data file the models use against the documented formats, before anything is trained |
| 14 | `FYP_code/21_prepare_chips.py` | Sen1Floods11 chip index: empty chips, Bolivia and weak chips overlapping valid, test or Bolivia ground left out; normalisation statistics from the hand-labelled training chips only |
| 15 | `FYP_code/22_train_rf.py`, `23_train_unet.py` | Random Forest and plain-PyTorch U-Net on VV, VH, HAND and slope, each trained on the hand labels and on hand plus weak labels; threshold chosen on the validation chips. The U-Net learning rate, 0.0003, was chosen by `unet_lr_test.py` against a rule fixed in advance |
| 16 | `FYP_code/24_eval_sen1floods11.py` | Scores a model on the Sen1Floods11 validation, test and unseen-event (Bolivia) chips |
| 17 | `FYP_code/25_predict_corridor.py` | Applies a model to a corridor scene and writes a flood map in exactly the baselines' format, scored with step 10 |
| 18 | `FYP_code/26_corridor_lowo.py` | Leave one digitising window out on the corridor, GRD against RTC on the same cells |
| 19 | `FYP_code/27_compare_maps.py`, `28_models_summary.py` | Paired window-bootstrap comparison of two maps with step 10's own functions; one summary of every model result |
| 19b | `FYP_code/32_seed_summary.py` (`RUN SEEDS.bat`) | The two leading models trained with two more seeds (`--run-tag` in 23, 24, 25, 30 and 31 keeps each repeat run in its own folders); spread across seeds and paired comparisons per seed |
| 20 | `FYP_code/29_terramind_preflight.py` (`RUN TERRAMIND PREFLIGHT.bat`) | Checks TerraTorch, the TerraMind weights (how many of the encoder's tensors they fill), memory and speed on the GPU, with time estimates for the TerraMind steps |
| 21 | `FYP_code/30_train_terramind.py` | Fine-tunes TerraMind-1.0-small on Sen1Floods11 in two variants, radar only (`tm_s1`) and radar plus HAND and slope in the decoder (`tm_s1t`), each on both training sets; then steps 16, 17 and 10 as for the other models |
| 22 | `FYP_code/26_corridor_lowo.py --model tm_s1 / tm_s1t` | Leave one window out with TerraMind, every fold from the pretrained weights, GRD through TerraMind's S1GRD input and RTC through its S1RTC input |
| also | `FYP_code/07_dem_comparison.py`, `08_gauge_hydrographs.py`, `09_landslide_merge.py`, `screen_gauges.py`, `check_rtc.py` | DEM comparison (Copernicus, CartoDEM, SRTM), river-gauge screening and hydrographs, landslide inventory merge, terrain-corrected (RTC) radar check |

Steps 14 to 22, with `31_calibration.py`, run from one launcher,
`FYP_code/RUN ALL MODELS.bat`.

Every script has a copy-paste command at the top. Several have a `--selftest`
mode, because the bugs found in this project so far produced plausible numbers,
not errors.

## Data sources

| Data | Source |
|---|---|
| Sentinel-1 GRD | Google Earth Engine, `COPERNICUS/S1_GRD` |
| Sentinel-1 RTC (two scenes) | ASF HyP3 On Demand, GAMMA, 10 m |
| Sentinel-2 L2A | Google Earth Engine, `COPERNICUS/S2_SR_HARMONIZED` |
| Copernicus GLO-30, SRTM | Google Earth Engine, `COPERNICUS/DEM/GLO30`, `USGS/SRTMGL1_003` |
| CartoDEM V3 R1 | Bhuvan, National Remote Sensing Centre, ISRO |
| JRC Global Surface Water | Google Earth Engine, `JRC/GSW1_4/GlobalSurfaceWater` |
| ESA WorldCover v200, WorldPop 2020 | Google Earth Engine |
| CHIRPS daily rainfall; IMD gridded rainfall | Google Earth Engine; India Meteorological Department via `imdlib` |
| River water level | GUARDIAN (Patidar, Indu and Karmakar, 2024, *Scientific Data* 11, 1155) |
| Landslide polygons | National Geoscience Data Repository (geodataindia.gov.in) |
| Village boundaries | Survey of India, via the National Water Data Portal |
| Sen1Floods11 | Bonafilia et al. (2020); its repository states no licence |
| TerraMind-1.0-small weights | IBM and ESA, `huggingface.co/ibm-esa-geospatial/TerraMind-1.0-small`, Apache 2.0 |

## Status, October 2026

**Done (semester 1):** corridor and centreline; Sentinel-2 mosaics; the
permanent-water mask; terrain strata for both events; the Otsu baselines;
the hand-digitised reference for both events; the RTC scenes; the Random
Forest, U-Net and TerraMind-small models, trained, applied to both events
and scored (Tables A and D), with paired comparisons and a calibration
check; the Jal-SAR tool.

**In progress:** two more training runs, with other random seeds, of the
U-Net and TerraMind with terrain (`RUN SEEDS.bat`), to measure how much a
result moves when a model is trained again.

**Semester 2:** hydraulic modelling (HEC-RAS) and landslide susceptibility.

## Reproducing the results

The data are not in this repository. To reproduce the model results you
need, besides the free sources above, the hand-digitised reference flood
extent, which cannot be recreated from the public data and is available from
the author.

1. Build the environment (`docs/SETUP_RUNBOOK.md`). An NVIDIA GPU with 4 GB
   of memory is enough.
2. Run `FYP_code/RUN MODELS PREFLIGHT.bat` and
   `FYP_code/RUN TERRAMIND PREFLIGHT.bat`. They check every input file.
3. Run `FYP_code/RUN ALL MODELS.bat`. On an RTX 3050 laptop with 4 GB it took
   about 27 hours in all. It can be stopped and started again.
4. Compare `06 Outputs/tables/MODELS_SUMMARY.md` with the published tables.

`provenance/RESULT_RUN_OPTIONS.json` lists the settings of every reported run,
and a test checks that `RUN ALL MODELS.bat` passes the same ones. Training on
a GPU is not exactly repeatable, so a re-run gives close but not identical
numbers; the seed runs above show how close to expect.

## Use of AI tools

Claude, an AI tool made by Anthropic, was used as a tool to help write the
code. I designed the study and ran all of the code on my own computer.

## Licence

MIT. See `LICENSE`. The licence covers the code only; each data source keeps
its own terms (table above).

## Author

Aryan Pandey, Department of Civil Engineering, G. B. Pant University of
Agriculture and Technology, Pantnagar, Uttarakhand. Supervisor:
[Dr. Sandeep Gupta](https://gbpuat.ac.in/colleges/COT/D1/sandeep_gupta_profile.html).
