@echo off
REM RUN FLOOD MAPS v3.bat
REM Written 30 September 2026. Pure ASCII on purpose.
REM
REM Rebuilds every flood map on the re-exported corridor v2 imagery, masked
REM with the v3 strata. Replaces RUN CHANGE MATRIX.bat and RUN JULY 2023.bat,
REM which point at strata folders that were archived on 23 September.
REM
REM 13 runs, each into its own new folder ending _v3. Nothing is overwritten.
REM   2 single-date Otsu baselines, 2021 and 2023 (the documented negative
REM     result; 05_reference_windows.py places the windows on the 2021 and 2023 grids)
REM   8 change-detection runs, 2021 and 2023 August against four pre-images
REM   3 change-detection runs, July 2023 against three pre-images.
REM     July is orbit 63 descending like 2021, so it uses the 2021 layover
REM     and shadow mask, as the old July run did
REM
REM Needs, in this order: the re-export swapped in (after check_reexport_v2.py
REM passed), strata_2021_v3 and strata_2023_v3 built (RUN ORDER.md, 4.1 and 4.2).
REM
REM Run from Anaconda Prompt:
REM   conda activate fyp
REM   cd /d "<project folder>\04 Code\FYP_code"
REM   "RUN FLOOD MAPS v3.bat"
REM Expect five to ten minutes per run, about two hours in all.

setlocal
cd /d "%~dp0"
if not exist "03b_otsu_baseline.py" (
  echo Cannot find 03b_otsu_baseline.py next to this file.
  echo Put this .bat in 04 Code\FYP_code\ and run it again.
  pause
  exit /b 1
)
python -c "import rasterio" >nul 2>&1
if errorlevel 1 (
  echo The fyp environment is not active.
  echo Open Anaconda Prompt, run:  conda activate fyp
  echo then cd here and run this file again.
  pause
  exit /b 1
)
if not exist "data\strata_2021_v3\strata_layover_shadow.tif" (
  echo MISSING data\strata_2021_v3\strata_layover_shadow.tif
  echo Do RUN ORDER.md step 4.1 first.
  pause
  exit /b 1
)
if not exist "data\strata_2023_v3\strata_layover_shadow.tif" (
  echo MISSING data\strata_2023_v3\strata_layover_shadow.tif
  echo Do RUN ORDER.md step 4.2 first.
  pause
  exit /b 1
)

set "S1=..\..\05 Data\sentinel1"
REM Permanent water, adopted 2 October 2026: JRC occurrence >= 50
REM plus dry-season Sentinel-2 water. Values 0 or 100, so the default
REM --water-threshold 50 works unchanged. Before 2 October this line pointed at
REM ..\..\05 Data\dem\WATER_JRC_occurrence_30m.tif
set "JRC=..\..\05 Data\dem\WATER_PERMANENT_JRC50_S2DRY_10m.tif"
if not exist "%JRC%" (
  echo MISSING %JRC%
  echo Make it with 12_permanent_water_mask.py (RUN ORDER.md, step 2.4).
  pause
  exit /b 1
)
set "M21=data\strata_2021_v3\strata_layover_shadow.tif"
set "M23=data\strata_2023_v3\strata_layover_shadow.tif"

echo.
echo === Single-date Otsu baselines ===
call :single "%S1%\S1_2021_post_20211021_orb63.tif"  "%M21%" flood_2021_v3 2021
call :single "%S1%\S1_2023_post_20230816_orb129.tif" "%M23%" flood_2023_v3 2023

echo.
echo === 2021 event, post 21 October 2021, orbit 63 descending ===
set "POST=%S1%\S1_2021_post_20211021_orb63.tif"
call :change "%POST%" "%S1%\S1_2021_pre_20201119_orb63.tif"         "%M21%" flood_2021_ch_nov_v3 2021chnov
call :change "%POST%" "%S1%\S1_2021_pre_alt_dec_20201225_orb63.tif" "%M21%" flood_2021_ch_dec_v3 2021chdec
call :change "%POST%" "%S1%\S1_2021_pre_alt_jan_20210118_orb63.tif" "%M21%" flood_2021_ch_jan_v3 2021chjan
call :change "%POST%" "%S1%\S1_2021_pre_alt_feb_20210211_orb63.tif" "%M21%" flood_2021_ch_feb_v3 2021chfeb

echo.
echo === 2023 August event, post 16 August 2023, orbit 129 ascending ===
set "POST=%S1%\S1_2023_post_20230816_orb129.tif"
call :change "%POST%" "%S1%\S1_2023_pre_20221101_orb129.tif"         "%M23%" flood_2023_ch_nov_v3 2023chnov
call :change "%POST%" "%S1%\S1_2023_pre_alt_dec_20221219_orb129.tif" "%M23%" flood_2023_ch_dec_v3 2023chdec
call :change "%POST%" "%S1%\S1_2023_pre_alt_jan_20230112_orb129.tif" "%M23%" flood_2023_ch_jan_v3 2023chjan
call :change "%POST%" "%S1%\S1_2023_pre_alt_feb_20230205_orb129.tif" "%M23%" flood_2023_ch_feb_v3 2023chfeb

echo.
echo === 2023 July event, post 19 July 2023, orbit 63 descending ===
set "POST=%S1%\S1_2023jul_post_20230719_orb63.tif"
call :change "%POST%" "%S1%\S1_2023jul_pre_dec_20221227_orb63.tif" "%M21%" flood_2023jul_ch_dec_v3 2023julchdec
call :change "%POST%" "%S1%\S1_2023jul_pre_jan_20230108_orb63.tif" "%M21%" flood_2023jul_ch_jan_v3 2023julchjan
call :change "%POST%" "%S1%\S1_2023jul_pre_feb_20230213_orb63.tif" "%M21%" flood_2023jul_ch_feb_v3 2023julchfeb

echo.
echo === All 13 runs attempted. Diagnostics files: ===
for /d %%D in (data\flood_*_v3) do @for %%F in ("%%D\*diagnostics.json") do @echo   %%F
echo.
echo Check in every one: corridor_km2 is about 606, NOT 808.
echo Check every diagnostics file. The numbers that matter:
echo threshold_db, separability, water_final_km2.
echo.
pause
exit /b 0

:single
if not exist %1 ( echo   MISSING %1 & goto :eof )
if exist "data\%3" ( echo   data\%3 already exists, skipped. Move it away to re-run. & goto :eof )
echo --- %4 single-date ---
python 03b_otsu_baseline.py --post %1 --mode single --water "%JRC%" --mask %2 --tiles 10 --out data\%3 --event %4
if errorlevel 1 echo   THIS RUN FAILED. The error is above.
goto :eof

:change
if not exist %1 ( echo   MISSING %1 & goto :eof )
if not exist %2 ( echo   MISSING %2 & goto :eof )
if exist "data\%4" ( echo   data\%4 already exists, skipped. Move it away to re-run. & goto :eof )
echo --- %5 change detection ---
python 03b_otsu_baseline.py --post %1 --pre %2 --mode change --water "%JRC%" --mask %3 --tiles 10 --out data\%4 --event %5
if errorlevel 1 echo   THIS RUN FAILED. The error is above.
goto :eof
