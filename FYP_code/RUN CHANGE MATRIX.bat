@echo off
REM RUN CHANGE MATRIX.bat
REM Runs change-detection Otsu for both events against all four pre-images.
REM Written 5 September 2026. Pure ASCII on purpose.
REM
REM Eight runs. Two events, four baselines each: the original November
REM pre-image plus the December, January and February winter scenes.
REM The November ones are included deliberately. They are the wet baseline we
REM think is wrong, and a comparison table needs the wrong one in it or the
REM claim that winter is better has nothing to stand on.
REM
REM Put this file in 04 Code\FYP_code\ and run it FROM ANACONDA PROMPT:
REM     conda activate fyp
REM     cd /d "<project folder>\04 Code\FYP_code"
REM     "RUN CHANGE MATRIX.bat"
REM
REM Double-clicking it also works, because it tries to activate the
REM environment itself, but the Anaconda Prompt route is the one that will
REM not surprise you.
REM
REM Nothing is overwritten: each run writes into its own folder.
REM Expect roughly five to ten minutes per run.

setlocal

REM --- make sure we are in the right place -----------------------------------
cd /d "%~dp0"
if not exist "03b_otsu_baseline.py" (
  echo.
  echo Cannot find 03b_otsu_baseline.py next to this file.
  echo Put this .bat in 04 Code\FYP_code\ and run it again.
  echo.
  pause
  exit /b 1
)

REM --- activate the environment if it is not active already ------------------
python -c "import rasterio" >nul 2>&1
if errorlevel 1 (
  echo Environment not active, trying to activate fyp ...
  call "%USERPROFILE%\miniconda3\Scripts\activate.bat" fyp
  python -c "import rasterio" >nul 2>&1
  if errorlevel 1 (
    echo.
    echo Could not activate the fyp environment automatically.
    echo Open Anaconda Prompt, run:  conda activate fyp
    echo then cd here and run this file again.
    echo.
    pause
    exit /b 1
  )
)

set S1=..\..\05 Data\sentinel1
set JRC=..\..\05 Data\dem\WATER_JRC_occurrence_30m.tif

echo.
echo ==========================================================
echo  2021 event, post 21 October 2021, orbit 63 descending
echo ==========================================================

call :run 2021 "%S1%\S1_2021_post_20211021_orb63.tif" "%S1%\S1_2021_pre_20201119_orb63.tif"       data\strata_2021 flood_2021_ch_nov 2021chnov
call :run 2021 "%S1%\S1_2021_post_20211021_orb63.tif" "%S1%\S1_2021_pre_alt_dec_20201225_orb63.tif" data\strata_2021 flood_2021_ch_dec 2021chdec
call :run 2021 "%S1%\S1_2021_post_20211021_orb63.tif" "%S1%\S1_2021_pre_alt_jan_20210118_orb63.tif" data\strata_2021 flood_2021_ch_jan 2021chjan
call :run 2021 "%S1%\S1_2021_post_20211021_orb63.tif" "%S1%\S1_2021_pre_alt_feb_20210211_orb63.tif" data\strata_2021 flood_2021_ch_feb 2021chfeb

echo.
echo ==========================================================
echo  2023 event, post 16 August 2023, orbit 129 ascending
echo ==========================================================

call :run 2023 "%S1%\S1_2023_post_20230816_orb129.tif" "%S1%\S1_2023_pre_20221101_orb129.tif"       data\strata_2023_v2 flood_2023_ch_nov 2023chnov
call :run 2023 "%S1%\S1_2023_post_20230816_orb129.tif" "%S1%\S1_2023_pre_alt_dec_20221219_orb129.tif" data\strata_2023_v2 flood_2023_ch_dec 2023chdec
call :run 2023 "%S1%\S1_2023_post_20230816_orb129.tif" "%S1%\S1_2023_pre_alt_jan_20230112_orb129.tif" data\strata_2023_v2 flood_2023_ch_jan 2023chjan
call :run 2023 "%S1%\S1_2023_post_20230816_orb129.tif" "%S1%\S1_2023_pre_alt_feb_20230205_orb129.tif" data\strata_2023_v2 flood_2023_ch_feb 2023chfeb

echo.
echo ==========================================================
echo  All eight runs attempted. The diagnostics JSONs are in:
echo ==========================================================
REM dir will not take a wildcard in the folder part of a path, so walk the
REM folders instead. The earlier version failed here with "The filename,
REM directory name, or volume label syntax is incorrect" AFTER every run had
REM already succeeded, which reads like a crash and is not one.
for /d %%D in (data\flood_20*_ch_*) do @for %%F in ("%%D\*diagnostics.json") do @echo   %%F
echo.
echo Check every one of those files.
echo The three numbers that matter in each: threshold_db, separability,
echo water_final_km2.
echo.
pause
exit /b 0

REM --------------------------------------------------------------------------
:run
REM %1 event year  %2 post  %3 pre  %4 strata dir  %5 out dir  %6 event tag
echo.
echo --- %6 ---
if not exist %2 ( echo   MISSING post image %2 & goto :eof )
if not exist %3 ( echo   MISSING pre image %3 & goto :eof )
python 03b_otsu_baseline.py --post %2 --pre %3 --mode change --water "%JRC%" --mask %4\strata_layover_shadow.tif --tiles 10 --out data\%5 --event %6
if errorlevel 1 echo   THIS RUN FAILED. The error is above.
goto :eof
