@echo off
rem Runs every TerraMind step end to end on YOUR data, but tiny: a few chips,
rem 1 epoch of 4 steps, 1 fold of 10 steps, one window of the corridor.
rem Everything goes into data\models_smoke\ and 06 Outputs\models_smoke\,
rem never into the real results. It overwrites only its own smoke files.
rem Target: under 20 minutes. Double-click it after RUN TERRAMIND PREFLIGHT.bat.

chcp 65001 >nul
set PYTHONUTF8=1
rem two OpenMP libraries in env fyp (see models_common.py): let both run
set KMP_DUPLICATE_LIB_OK=TRUE
cd /d "%~dp0"

echo.
echo TerraMind smoke run (tiny, on your data)
echo.

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" "%USERPROFILE%\miniconda3" >nul 2>&1
call conda activate fyp
if errorlevel 1 (
  echo Could not activate the conda environment 'fyp'.
  pause
  exit /b 1
)

set FAILS=0
echo ==== 1. chip index and normalisation of the smoke run (kept if RUN MODELS SMOKE.bat made them)
python 21_prepare_chips.py --smoke --skip-existing
if errorlevel 1 set /a FAILS+=1
echo.
echo ==== 2. TerraMind training: tm_s1 hand, tm_s1t hand, tm_s1t hand_weak
python 30_train_terramind.py --smoke --variant tm_s1 --trainset hand --force
if errorlevel 1 set /a FAILS+=1
python 30_train_terramind.py --smoke --variant tm_s1t --trainset hand --force
if errorlevel 1 set /a FAILS+=1
python 30_train_terramind.py --smoke --variant tm_s1t --trainset hand_weak --force
if errorlevel 1 set /a FAILS+=1
echo.
echo ==== 3. Sen1Floods11 evaluation (valid chips only in a smoke run)
python 24_eval_sen1floods11.py --smoke --model tm_s1 --trainset hand --force
if errorlevel 1 set /a FAILS+=1
python 24_eval_sen1floods11.py --smoke --model tm_s1t --trainset hand --force
if errorlevel 1 set /a FAILS+=1
echo.
echo ==== 4. corridor maps, 2021, one window only
python 25_predict_corridor.py --smoke --model tm_s1 --trainset hand --year 2021 --force
if errorlevel 1 set /a FAILS+=1
python 25_predict_corridor.py --smoke --model tm_s1t --trainset hand --year 2021 --force
if errorlevel 1 set /a FAILS+=1
echo.
echo ==== 5. scoring with 04_stratified_metrics.py
python 04_stratified_metrics.py --pred "data\models_smoke\flood_2021_tm_s1t_hand\flood_2021_tm_s1t_hand.tif" --truth data\reference_2021.tif --strata data\strata_2021_v3 --windows data\reference_windows_2021\reference_windows.geojson --label "SMOKE tm_s1t hand 2021" --out "..\..\06 Outputs\models_smoke\tables\2021_tm_s1t_hand"
if errorlevel 1 set /a FAILS+=1
echo.
echo ==== 6. leave one window out, one fold: tm_s1t on GRD (S1GRD input), tm_s1 on RTC (S1RTC input)
python 26_corridor_lowo.py --smoke --year 2021 --model tm_s1t --radar grd --force
if errorlevel 1 set /a FAILS+=1
python 26_corridor_lowo.py --smoke --year 2021 --model tm_s1 --radar rtc --force
if errorlevel 1 set /a FAILS+=1
echo.
echo ==== 7. paired comparison and summary
python 27_compare_maps.py --a "data\models_smoke\flood_2021_tm_s1_hand\flood_2021_tm_s1_hand.tif" --b "data\models_smoke\flood_2021_tm_s1t_hand\flood_2021_tm_s1t_hand.tif" --truth data\reference_2021.tif --strata data\strata_2021_v3 --windows data\reference_windows_2021\reference_windows.geojson --label-a "SMOKE tm_s1" --label-b "SMOKE tm_s1t" --out "..\..\06 Outputs\models_smoke\tables\compare_2021_tm_s1_hand_vs_tm_s1t_hand" --force
if errorlevel 1 set /a FAILS+=1
python 28_models_summary.py --smoke --force
if errorlevel 1 set /a FAILS+=1

echo.
echo ============================================================
if "%FAILS%"=="0" (
  echo SMOKE PASS. Every TerraMind step ran on your data.
  echo Next: double-click RUN ALL MODELS.bat
) else (
  echo SMOKE FAIL: %FAILS% step(s^) failed. Scroll up to the first PROBLEM line.
)
echo ============================================================
echo.
pause
