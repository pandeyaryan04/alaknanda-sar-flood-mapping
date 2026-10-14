@echo off
rem Runs every model step end to end on YOUR data, but tiny: a few chips,
rem 1 epoch of a few steps, 1 fold, one window of the corridor.
rem Everything goes into data\models_smoke\ and 06 Outputs\models_smoke\,
rem never into the real results. It overwrites only its own smoke files.
rem Target: under 15 minutes. Double-click it after RUN MODELS PREFLIGHT.bat.

chcp 65001 >nul
set PYTHONUTF8=1
rem two OpenMP libraries in env fyp (see models_common.py): let both run
set KMP_DUPLICATE_LIB_OK=TRUE
cd /d "%~dp0"

echo.
echo Models smoke run (tiny, on your data)
echo.

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" "%USERPROFILE%\miniconda3" >nul 2>&1
call conda activate fyp
if errorlevel 1 (
  echo Could not activate the conda environment 'fyp'.
  pause
  exit /b 1
)

set FAILS=0
echo ==== 1. chip index and normalisation, a few chips
python 21_prepare_chips.py --smoke --force
if errorlevel 1 set /a FAILS+=1
echo.
echo ==== 2. Random Forest, hand and hand_weak
python 22_train_rf.py --smoke --trainset hand --force
if errorlevel 1 set /a FAILS+=1
python 22_train_rf.py --smoke --trainset hand_weak --force
if errorlevel 1 set /a FAILS+=1
echo.
echo ==== 3. U-Net, hand and hand_weak
python 23_train_unet.py --smoke --trainset hand --force
if errorlevel 1 set /a FAILS+=1
python 23_train_unet.py --smoke --trainset hand_weak --force
if errorlevel 1 set /a FAILS+=1
echo.
echo ==== 4. Sen1Floods11 evaluation
python 24_eval_sen1floods11.py --smoke --model rf --trainset hand --force
if errorlevel 1 set /a FAILS+=1
python 24_eval_sen1floods11.py --smoke --model unet --trainset hand --force
if errorlevel 1 set /a FAILS+=1
echo.
echo ==== 5. corridor maps, 2021, one window only
python 25_predict_corridor.py --smoke --model rf --trainset hand --year 2021 --force
if errorlevel 1 set /a FAILS+=1
python 25_predict_corridor.py --smoke --model unet --trainset hand --year 2021 --force
if errorlevel 1 set /a FAILS+=1
echo.
echo ==== 6. scoring with 04_stratified_metrics.py
python 04_stratified_metrics.py --pred "data\models_smoke\flood_2021_rf_hand\flood_2021_rf_hand.tif" --truth data\reference_2021.tif --strata data\strata_2021_v3 --windows data\reference_windows_2021\reference_windows.geojson --label "SMOKE rf hand 2021" --out "..\..\06 Outputs\models_smoke\tables\2021_rf_hand"
if errorlevel 1 set /a FAILS+=1
echo.
echo ==== 7. leave one window out, one fold
python 26_corridor_lowo.py --smoke --year 2021 --model rf --radar grd --force
if errorlevel 1 set /a FAILS+=1
python 26_corridor_lowo.py --smoke --year 2021 --model unet --radar rtc --force
if errorlevel 1 set /a FAILS+=1
echo.
echo ==== 8. paired comparison and summary
python 27_compare_maps.py --a "data\models_smoke\flood_2021_rf_hand\flood_2021_rf_hand.tif" --b "data\models_smoke\flood_2021_unet_hand\flood_2021_unet_hand.tif" --truth data\reference_2021.tif --strata data\strata_2021_v3 --windows data\reference_windows_2021\reference_windows.geojson --label-a "SMOKE rf" --label-b "SMOKE unet" --out "..\..\06 Outputs\models_smoke\tables\compare_2021_rf_hand_vs_unet_hand" --force
if errorlevel 1 set /a FAILS+=1
python 28_models_summary.py --smoke --force
if errorlevel 1 set /a FAILS+=1

echo.
echo ============================================================
if "%FAILS%"=="0" (
  echo SMOKE PASS. Every step ran on your data.
  echo Next: double-click RUN ALL MODELS.bat
) else (
  echo SMOKE FAIL: %FAILS% step(s^) failed. Scroll up to the first PROBLEM line.
)
echo ============================================================
echo.
pause
