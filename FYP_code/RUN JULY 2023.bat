@echo off
REM RUN JULY 2023.bat
REM Change detection for the 19 July 2023 scene against its three winter
REM baselines. Written 5 September 2026. Pure ASCII on purpose.
REM
REM Run this only after the four files from 07_export_july2023.js are in
REM 05 Data\sentinel1\.
REM
REM Run it FROM ANACONDA PROMPT:
REM     conda activate fyp
REM     cd /d "<project folder>\04 Code\FYP_code"
REM     "RUN JULY 2023.bat"
REM
REM ON THE STRATA THIS USES, WHICH IS A DELIBERATE CHOICE
REM   It masks with data\strata_2021\strata_layover_shadow.tif, not a new
REM   strata run for July.
REM
REM   Layover and shadow are a function of the DEM and the viewing geometry.
REM   The July scene is relative orbit 63 descending, which is the same orbit
REM   and the same geometry as the October 2021 scene, over the same DEM. The
REM   mask is therefore the same physical result, and the 2021 run already
REM   measured it at 2.21 per cent layover.
REM
REM   That makes this correct for a comparison run and good enough to report
REM   as one. It is NOT what should go in the final paper: for that, build
REM   dedicated strata from the July scene's own angle band, so the incidence
REM   raster comes from the acquisition being scored. That run is in the
REM   commented block at the bottom of this file. Run it when you have the
REM   time, and re-run the three change runs against it.

setlocal
cd /d "%~dp0"

if not exist "03b_otsu_baseline.py" (
  echo Cannot find 03b_otsu_baseline.py next to this file.
  pause
  exit /b 1
)

python -c "import rasterio" >nul 2>&1
if errorlevel 1 (
  echo Environment not active, trying to activate fyp ...
  call "%USERPROFILE%\miniconda3\Scripts\activate.bat" fyp
  python -c "import rasterio" >nul 2>&1
  if errorlevel 1 (
    echo Open Anaconda Prompt, run: conda activate fyp, then try again.
    pause
    exit /b 1
  )
)

set S1=..\..\05 Data\sentinel1
set JRC=..\..\05 Data\dem\WATER_JRC_occurrence_30m.tif
set POST=%S1%\S1_2023jul_post_20230719_orb63.tif

if not exist "%POST%" (
  echo.
  echo Cannot find %POST%
  echo Run 07_export_july2023.js in Earth Engine first and download the four
  echo files into 05 Data\sentinel1\.
  echo.
  pause
  exit /b 1
)

echo.
echo ==========================================================
echo  19 July 2023, orbit 63 descending, three winter baselines
echo ==========================================================

call :run "%S1%\S1_2023jul_pre_dec_20221227_orb63.tif" flood_2023jul_ch_dec 2023julchdec
call :run "%S1%\S1_2023jul_pre_jan_20230108_orb63.tif" flood_2023jul_ch_jan 2023julchjan
call :run "%S1%\S1_2023jul_pre_feb_20230213_orb63.tif" flood_2023jul_ch_feb 2023julchfeb

echo.
echo ==========================================================
echo  Done. The diagnostics files:
echo ==========================================================
for /d %%D in (data\flood_2023jul_ch_*) do @for %%F in ("%%D\*diagnostics.json") do @echo   %%F
echo.
echo Check all three.
echo.
echo WHAT TO LOOK FOR:
echo   separability near 0.8, like the August December and January runs
echo   flood area LARGER than the 1.8 km2 the August scene gave
echo.
echo   If July shows more water than August, change detection is tracking
echo   the hydrology and the method is doing its job.
echo   If July shows the same or less after three times the antecedent rain,
echo   that is a problem with the method, not with the scene, and it is much
echo   better to find it now.
echo.
pause
exit /b 0

REM --------------------------------------------------------------------------
:run
REM %1 pre image   %2 out folder   %3 event tag
echo.
echo --- %3 ---
if not exist %1 ( echo   MISSING pre image %1 & goto :eof )
python 03b_otsu_baseline.py --post "%POST%" --pre %1 --mode change --water "%JRC%" --mask data\strata_2021\strata_layover_shadow.tif --tiles 10 --out data\%2 --event %3
if errorlevel 1 echo   THIS RUN FAILED. The error is above.
goto :eof

REM --------------------------------------------------------------------------
REM PROPER STRATA FOR THE JULY SCENE, for when there is time.
REM Uncomment and run once, then re-run the three change runs above with
REM --mask data\strata_2023jul\strata_layover_shadow.tif
REM
REM python 03_terrain_strata.py --dem "..\..\05 Data\dem\DEM_COP_GLO30_30m.tif" --angle "..\..\05 Data\sentinel1\S1_2023jul_post_20230719_orb63.tif" --pass DESCENDING --water "..\..\05 Data\dem\WATER_JRC_occurrence_30m.tif" --out data\strata_2023jul
REM
REM Do NOT run that before the corridor centreline is digitised and the DEM
REM clipping patch is in. Building strata against the approximate corridor
REM means building them twice.
