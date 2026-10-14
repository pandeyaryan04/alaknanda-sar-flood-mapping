# Run order: from the source data to the results

Every step that makes a file in this project, in the order to run them on a
fresh machine. Paths are relative to the project folder (the folder that holds
`04 Code`, `05 Data` and `06 Outputs`). Commands run in Anaconda Prompt with
`conda activate fyp`, from `04 Code\FYP_code` unless the row says
`gis_tools`. A `.bat` file runs from its own folder by double-click.

Written 7 October 2026. When a step is added, add its row here in the same
commit.

## 0. Before anything

| # | What | How | Check |
|---|---|---|---|
| 0.1 | Environments `fyp` and `fypdl` | `docs\SETUP_RUNBOOK.md`; exact versions in `04 Code\provenance\environment_*.yml` and `pip_freeze_*.txt` | `python 00_check_machine.py` |
| 0.2 | Earth Engine access, project `gbpuat-fyp-alaknanda` | `earthengine authenticate` | |
| 0.3 | `gis_tools\.env` | copy `.env.example` to `.env`, set `PROJECT_ROOT` | `gis_tools\SELFTEST.bat` |

## 1. Source data (downloaded, not made)

Where each came from, its version and its citation: `05 Data\00 Notes\DATA
SOURCES - register, 3 October 2026.md`. Exact Earth Engine image IDs:
`04 Code\provenance\EARTH_ENGINE_INPUTS.csv`. Checksums of every file:
`04 Code\provenance\DATA_MANIFEST.csv` (`python 18_data_manifest.py --verify`
compares a machine's files with it).

| # | Data | How it was got | Lands in |
|---|---|---|---|
| 1.1 | Sentinel-1 (14 scenes), Sentinel-2 (9 dates), Copernicus GLO-30, SRTM, JRC occurrence, WorldCover, WorldPop, CHIRPS 2023 | `gee\10_reexport_corridor_v2.js` in the Earth Engine Code Editor, export to Drive, download into `05 Data\_reexport_v2\`, then `python pad_reexport_to_old_grid.py` and `python check_reexport_v2.py --accept-new-cop --accept-new-s2-2021`, then move into place | `05 Data\sentinel1\`, `sentinel2\`, `dem\`, `landcover\`, `population\`, `rainfall\` |
| 1.2 | CHIRPS 2021, catchment and district boundaries | `gee\04_catchment_rainfall.js` (catchment, CHIRPS 2021) and `gee\08_export_context_layers.js` (districts and context layers), the same way | `05 Data\rainfall\`, `boundaries\` |
| 1.3 | Sen1Floods11 | `python 02_fetch_sen1floods11.py` | `FYP_code\data\sen1floods11\` |
| 1.4 | IMD rainfall; HMA 8 m DEM | `gis_tools\RUN DOWNLOADS.bat` (`t07_download.py imd`, `t07_download.py earthdata --short-name HMA_DEM8m_MOS --download`) | `05 Data\rainfall\imd\`, `dem\hma8m\` |
| 1.5 | Sentinel-1 RTC, two scenes | ASF HyP3 On Demand (job `fyp-rtc`); zips into `05 Data\sentinel1_rtc\` | |
| 1.6 | By hand from portals: CartoDEM tiles, LULC 50K, village boundaries, landslide polygons, GUARDIAN gauges, India-WRIS station pages, GADM, atlases | register, Part 1, rows 7, 8, 13, 19 to 27 | `05 Data\` |
| 1.7 | The hand-drawn reference | drawn in QGIS on the Sentinel-2 images (digitising guide in `docs\REFERENCE - how to digitise.md`); returned as `00 New work\Digitising handoff\received\RETURN_*.zip`, kept untouched | |
| 1.8 | TerraMind-1.0-small weights, `TerraMind_v1_small.pt` (504 MB, Apache 2.0) | by hand from `huggingface.co/ibm-esa-geospatial/TerraMind-1.0-small`; never downloaded by the code; its SHA-256 is in every TerraMind run record | `FYP_code\data\weights\` |

## 2. The corridor and the river

| # | Script | Makes | Needs |
|---|---|---|---|
| 2.1 | `gis_tools\t00_thalweg_from_cartodem.py` | `05 Data\boundaries\RIVER_thalweg_cartodem.geojson`, `corridor_v2_3km_clipped.geojson` (606.35 km²); copy the corridor to `FYP_code\data\corridor.geojson` | CartoDEM tile N30 E078 (1.6) |
| 2.2 | `gis_tools\RUN CENTRELINE.bat` (`t01`) | `05 Data\boundaries\corridor_centreline.*` | 1.1, 2.1, CartoDEM v2 clip |
| 2.3 | `gis_tools\RUN MOSAICS.bat` (`t02`) | `05 Data\sentinel2\S2_<year>_<dry or event>_mosaic.tif`, `_mndwi.tif`, `_gapmask.tif` | 1.1 |
| 2.4 | `python 12_permanent_water_mask.py` | `WATER_PERMANENT_JRC50_S2DRY_10m.tif` (writes to `06 Outputs\remake\`; compares with the file in use) | 2.3 |
| 2.5 | `python 14_reservoir_outline.py` | `RESERVOIR_srinagar_aoi.geojson` | 2.2 |
| 2.6 | `python 13_cloudgap_holes.py --year 2021` and `--year 2023` | `S2_<year>_event_cloudgap.gpkg` | 2.3 |

## 3. Context layers and checks

| # | Script | Makes |
|---|---|---|
| 3.1 | `gis_tools\RUN WATERMASK AUDIT.bat` (`t03`) | `06 Outputs\gis_tools\watermask_audit\` |
| 3.2 | `gis_tools\RUN RESERVOIR CHECK.bat` (`t04`) | `06 Outputs\gis_tools\reservoir\` |
| 3.3 | `gis_tools\RUN VILLAGES.bat` (`t08`) | `05 Data\boundaries\VILLAGES_corridor.*` |
| 3.4 | `RUN DEM COMPARISON.bat` (`07_dem_comparison.py`) | `05 Data\dem\DEM_CARTODEM_30m.tif`, `06 Outputs\dem_comparison\` |
| 3.5 | `RUN LANDSLIDE MERGE.bat` (`09_landslide_merge.py`) | `05 Data\landslide\LANDSLIDE_merged.gpkg`, `LANDSLIDE_corridor.gpkg` |
| 3.6 | `SCREEN GAUGES.bat` (`screen_gauges.py`), then `RUN GAUGE HYDROGRAPHS.bat` (`08`) | `05 Data\gauges\guardian\GAUGE_*.csv`, `06 Outputs\gauges\` |
| 3.7 | `CHECK RTC.bat` (`check_rtc.py`) | `05 Data\sentinel1_rtc\S1RTC_*.tif` |
| 3.8 | `python 16_landcover_corridor.py` | land-cover clip and tables (write to `06 Outputs\remake\landcover\`, then copy into `05 Data\landcover\`) |

## 4. Terrain, flood maps, reference, scores

| # | Command | Makes |
|---|---|---|
| 4.1 | `python 03_terrain_strata.py --dem "..\..\05 Data\dem\DEM_COP_GLO30_30m.tif" --pass DESCENDING --water "..\..\05 Data\dem\WATER_JRC_occurrence_30m.tif" --corridor data\corridor.geojson --angle "..\..\05 Data\sentinel1\S1_2021_post_20211021_orb63.tif" --out data\strata_2021_v3` | `data\strata_2021_v3\` |
| 4.2 | the same with `--pass ASCENDING --angle "..\..\05 Data\sentinel1\S1_2023_post_20230816_orb129.tif" --out data\strata_2023_v3` | `data\strata_2023_v3\` |
| 4.3 | `RUN FLOOD MAPS v3.bat` (`03b_otsu_baseline.py`, 13 maps) | `data\flood_*_v3\` |
| 4.4 | `python 05_reference_windows.py --strata data/strata_2021_v3 --grid data/flood_2021_v3/flood_2021_otsu.tif --pred data/flood_2021_v3/flood_2021_otsu.tif --centreline "..\..\05 Data\boundaries\corridor_centreline.geojson" --n 8 --size 5 --out data/reference_windows_2021`, and the same for 2023 | `data\reference_windows_<year>\` (never re-run once drawing has started) |
| 4.5 | `RUN WINDOW MAP.bat` (`10_window_map.py`) | `06 Outputs\figures\sampling_design_<year>.png` |
| 4.6 | `gis_tools\RUN QGIS SETUP.bat`, `RUN DRAFT POLYGONS.bat` (`t05`, `t06`) | the digitising projects and drafts (aids only; nothing scored comes from them) |
| 4.7 | `python 15_reference_dry_completion.py --year 2021` and `--year 2023` | the reference GeoPackages, February layer completed by rule; copy into `06 Outputs\gis_tools\qgis\<year>_event\reference_<year>_event.gpkg` |
| 4.8 | `python 06_rasterise_reference.py --event "..\..\06 Outputs\gis_tools\qgis\2021_event\reference_2021_event.gpkg" --event-layer water_event --dry-layer water_dry --windows data/reference_windows_2021/reference_windows.geojson --holes "..\..\05 Data\sentinel2\S2_2021_event_cloudgap.gpkg" --holes-layer holes --grid data/flood_2021_v3/flood_2021_otsu.tif --out data/reference_2021.tif`, and the same for 2023 | `data\reference_<year>.tif` (2.401 and 3.209 km² flood) |
| 4.9 | `python 04_stratified_metrics.py --pred <flood map> --truth data/reference_<year>.tif --strata data/strata_<year>_v3 --windows data/reference_windows_<year>/reference_windows.geojson --label "<label>" --out "..\..\06 Outputs\tables\<run>"`, once for each flood map scored | `06 Outputs\tables\<run>\` |
| 4.10 | `python 11_sen1floods11_terrain.py --set hand`, then `--set weak` | `data\sen1floods11_terrain\` (model inputs) |
| 4.11 | `RUN MODELS PREFLIGHT.bat` (`python 20_models_preflight.py`) | `06 Outputs\models\preflight_<computer>.md`; must say `PREFLIGHT PASS` |
| 4.12 | `RUN MODELS SMOKE.bat` (every model step, tiny) | `data\models_smoke\`, `06 Outputs\models_smoke\`; must say `SMOKE PASS` |
| 4.13 | `python 21_prepare_chips.py --cache` (first step of `RUN ALL MODELS.bat`) | `data\models\chip_index.csv`, `norm_stats.json`, `cache\` |
| 4.14 | `python 22_train_rf.py --trainset hand` and `--trainset hand_weak` | `data\models\rf_<trainset>\` |
| 4.15 | `python 23_train_unet.py --trainset hand --lr 0.0003 --resume` and `--trainset hand_weak --lr 0.0003 --resume` (0.0003 chosen by `unet_lr_test.py`, 8 October; see 4.26) | `data\models\unet_<trainset>\` |
| 4.16 | `python 24_eval_sen1floods11.py --model <rf or unet> --trainset <trainset>` | `06 Outputs\tables\s1f11_<model>_<trainset>\` |
| 4.17 | `python 25_predict_corridor.py --model <model> --trainset <trainset> --year <year>`, then 4.9 on the map. **`RUN ALL MODELS.bat` does 4.13 to 4.19, 4.23, 4.24 and 4.27 in one go, with the settings of the reported results** (`provenance\RESULT_RUN_OPTIONS.json`) | `data\flood_<year>_<model>_<trainset>\`, `06 Outputs\tables\<year>_<model>_<trainset>\` |
| 4.18 | `python 26_corridor_lowo.py --year <year> --model <model> --radar <grd or rtc>` (the U-Net with `--lr 0.0003`), then 4.9 | `data\lowo_<year>_<model>_<radar>\`, `06 Outputs\tables\lowo_<year>_<model>_<radar>\` |
| 4.19 | `python 27_compare_maps.py ...` and `python 28_models_summary.py --force` | `06 Outputs\tables\compare_*\`, `06 Outputs\tables\MODELS_SUMMARY.md` |
| 4.21 | `RUN TERRAMIND PREFLIGHT.bat` (`python 29_terramind_preflight.py`) | `06 Outputs\models\terramind_preflight_<computer>.md` with time estimates from the measured speed; must say `TERRAMIND PREFLIGHT PASS` |
| 4.22 | `RUN TERRAMIND SMOKE.bat` (every TerraMind step, tiny) | `data\models_smoke\`, `06 Outputs\models_smoke\`; must say `SMOKE PASS` |
| 4.23 | `python 30_train_terramind.py --variant <tm_s1 or tm_s1t> --trainset <hand or hand_weak> --resume`, then 4.16 and 4.17 with `--model <variant>` | `data\models\<variant>_<trainset>\`, `06 Outputs\tables\s1f11_<variant>_<trainset>\`, `data\flood_<year>_<variant>_<trainset>\`, `06 Outputs\tables\<year>_<variant>_<trainset>\` |
| 4.24 | `python 26_corridor_lowo.py --year <year> --model <tm_s1 or tm_s1t> --radar <grd or rtc>`, then 4.9; then 4.19 again | `data\lowo_<year>_<variant>_<radar>\`, `06 Outputs\tables\lowo_<year>_<variant>_<radar>\` |
| 4.26 | `python unet_lr_test.py` (run once, 8 October, before 4.15): the U-Net trained on `hand` for 30 epochs at 0.001 and at 0.0003; 0.0003 is chosen only if its best valid IoU is at least 0.01 higher, a rule fixed in advance | `data\models\lr_test\lr_choice.txt`, `06 Outputs\models\unet_lr_test.md` |
| 4.27 | After every model is trained (4.19 done): `python 31_calibration.py`: temperature and Platt rescaling fitted on the Sen1Floods11 valid chips, ECE before and after on valid, test and Bolivia, reliability diagrams. Changes no map or score | `06 Outputs\tables\calibration_summary.md` and `.csv`; `06 Outputs\models\calibration\<model>_<trainset>\calibration.json`, `reliability.png` |
| 4.28 | Optional, repeat runs (9 October): `RUN SEEDS.bat`. `unet_hand` and `tm_s1t_hand` trained twice more with seeds 20261008 (`--run-tag seed2`) and 20261009 (`--run-tag seed3`), each with the original run's settings, then 4.15 to 4.18 with `--run-tag`, `27_compare_maps.py` for each seed (tm_s1t against the U-Net, each against February change detection), `31_calibration.py --run-tag`, and `python 32_seed_summary.py --force`. The original runs are not touched | `data\models\<model>_hand_<tag>\`, `06 Outputs\tables\s1f11_<model>_hand_<tag>\`, `data\flood_<year>_<model>_hand_<tag>\`, `06 Outputs\tables\<year>_<model>_hand_<tag>\`, `06 Outputs\tables\seeds_compare_*\`, `calibration_summary_<tag>.md`, `06 Outputs\tables\SEEDS_SUMMARY.md` and `.csv` |

## 5. Checking it all

| # | Command | Result |
|---|---|---|
| 5.1 | `RUN REMAKE CHECK.bat` | scripts 12 to 15 against the files in use: `ALL 8 CHECKS PASS` |
| 5.2 | `RUN PROVENANCE.bat` | environments, Earth Engine IDs (`17`), number audit (`19`), checksums (`18`) |
| 5.3 | `python 18_data_manifest.py --verify` | on another machine: `changed 0, missing 0` |
| 5.4 | `python 19_number_audit.py` | every number quoted in the report against the files |

## Things that are not byte-identical on another machine

- `t00` writes coordinates that can differ in the 15th digit between
  machines (floating point). The polygons cover the same ground to within
  10⁻⁸ m; compare geometry, not checksums.
- Earth Engine exports (1.1) depend on what Earth Engine serves on the day.
  Two 2021 Sentinel-2 dates were reprocessed between September and October
  2026 (decision R3e). The image IDs and dates are recorded in
  `EARTH_ENGINE_INPUTS.csv`.
- QGIS writes into a GeoPackage when it opens one, which changes the checksum
  without changing the data. Close QGIS before `--verify`.
- The models (steps 4.13 to 4.18, 4.23 and 4.24) use the seed 20261007 everywhere (`--seed`).
  The chip index, normalisation and Random Forests are identical on every
  machine with the same package versions. The U-Net and TerraMind are
  identical run to run on the CPU; on a GPU, cuDNN may differ in the last
  digits unless `--deterministic` is given (slower). Each run writes `run_record.json` with
  the machine, versions, options and SHA-256 of every output.
