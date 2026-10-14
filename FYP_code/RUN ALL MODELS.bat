@echo off
rem Every model result of the project, in one launcher, with the settings
rem that produced the reported results (9 October 2026). It replaces the
rem separate model launchers used while the results were being made.
rem
rem   1. Table A: Random Forest, U-Net and TerraMind-small (tm_s1t radar and
rem      terrain, tm_s1 radar only), each trained on Sen1Floods11 'hand' and
rem      'hand_weak', evaluated on Sen1Floods11, applied to the 2021 and
rem      2023 corridor scenes and scored with 04_stratified_metrics.py
rem   2. Table D: leave one window out on the corridor, every model, both
rem      events, GRD and RTC, scored with 04
rem   3. the paired comparisons (27) and MODELS_SUMMARY.md (28)
rem   4. calibration (31)
rem
rem The U-Net learning rate is fixed at 0.0003. It was chosen on 8 October
rem 2026 by unet_lr_test.py (0.0003 beat 0.001 by
rem 0.0102 against a rule of 0.01 fixed in advance). A reproduction repeats
rem that decision; it does not take it again. Every other setting is the
rem scripts' default. provenance\RESULT_RUN_OPTIONS.json lists the settings
rem of every reported run, and tests\test_23_organise.py checks this file
rem against it.
rem
rem Every finished step is skipped, so if it stops (power cut, Ctrl+C, a
rem closed lid) just double-click it again: training continues from its last
rem checkpoint. Needs an NVIDIA GPU (4 GB is enough). Measured on the RTX
rem 3050 4 GB laptop: about 27 hours in all (U-Net about 4.5 h, TerraMind
rem Table A about 12 h, TerraMind Table D about 10 h, the rest about 1 h).
rem Run RUN MODELS PREFLIGHT.bat and RUN TERRAMIND PREFLIGHT.bat first.

chcp 65001 >nul
set PYTHONUTF8=1
rem two OpenMP libraries in env fyp (see models_common.py): let both run
set KMP_DUPLICATE_LIB_OK=TRUE
cd /d "%~dp0"

set UNET_LR=0.0003

echo.
echo All models: Table A, Table D, comparisons, calibration
echo.

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" "%USERPROFILE%\miniconda3" >nul 2>&1
call conda activate fyp
if errorlevel 1 (
  echo Could not activate the conda environment 'fyp'.
  pause
  exit /b 1
)

set FAILS=0
set MISSING=0
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 (
  echo No CUDA GPU found. The U-Net and TerraMind need an NVIDIA GPU: restart and try again.
  set /a FAILS+=1
  goto :end
)

echo ==== chip index, normalisation and cache
python 21_prepare_chips.py --cache --skip-existing
if errorlevel 1 (
  set /a FAILS+=1
  goto :end
)

echo.
echo ==== 1. Table A
call :model rf hand
call :model rf hand_weak
call :model unet hand
call :model unet hand_weak
call :model tm_s1t hand
call :model tm_s1 hand
call :model tm_s1t hand_weak
call :model tm_s1 hand_weak

echo.
echo ==== 2. Table D
for %%Y in (2021 2023) do for %%M in (rf unet tm_s1t tm_s1) do for %%R in (grd rtc) do call :lowo %%Y %%M %%R

echo.
echo ==== 3. paired comparisons
call :year 2021
call :year 2023
echo.
echo ==== summary of every result
python 28_models_summary.py --force
if errorlevel 1 set /a FAILS+=1

echo.
echo ==== 4. calibration
python 31_calibration.py
if errorlevel 1 set /a FAILS+=1
goto :end

:model
echo.
echo ==== %1 %2: training
if "%1"=="rf" python 22_train_rf.py --trainset %2 --skip-existing
if "%1"=="unet" python 23_train_unet.py --trainset %2 --lr %UNET_LR% --resume --skip-existing
if "%1"=="tm_s1t" python 30_train_terramind.py --variant tm_s1t --trainset %2 --resume --skip-existing
if "%1"=="tm_s1" python 30_train_terramind.py --variant tm_s1 --trainset %2 --resume --skip-existing
if errorlevel 1 (
  set /a FAILS+=1
  echo   training %1 %2 failed, so its other steps are skipped.
  goto :eof
)
if not exist "data\models\%1_%2\model_meta.json" (
  echo   training %1 %2 is not finished. Double-click this file again to continue it.
  set /a FAILS+=1
  goto :eof
)
echo ==== %1 %2: Sen1Floods11 evaluation
python 24_eval_sen1floods11.py --model %1 --trainset %2 --skip-existing
if errorlevel 1 set /a FAILS+=1
call :predict %1 %2 2021
call :predict %1 %2 2023
goto :eof

:predict
echo ==== %1 %2: corridor map %3
python 25_predict_corridor.py --model %1 --trainset %2 --year %3 --skip-existing
if errorlevel 1 (
  set /a FAILS+=1
  goto :eof
)
if exist "..\..\06 Outputs\tables\%3_%1_%2\metrics_by_stratum.csv" (
  echo   score %3_%1_%2 already exists, skipped.
  goto :eof
)
python 04_stratified_metrics.py --pred "data\flood_%3_%1_%2\flood_%3_%1_%2.tif" --truth data\reference_%3.tif --strata data\strata_%3_v3 --windows data\reference_windows_%3\reference_windows.geojson --label "Table A, %1 %2, %3" --out "..\..\06 Outputs\tables\%3_%1_%2"
if errorlevel 1 set /a FAILS+=1
goto :eof

:lowo
echo.
echo ==== Table D, %1, %2, %3
if "%2"=="unet" python 26_corridor_lowo.py --year %1 --model %2 --radar %3 --lr %UNET_LR% --skip-existing
if not "%2"=="unet" python 26_corridor_lowo.py --year %1 --model %2 --radar %3 --skip-existing
if errorlevel 1 (
  set /a FAILS+=1
  goto :eof
)
if exist "..\..\06 Outputs\tables\lowo_%1_%2_%3\metrics_by_stratum.csv" (
  echo   score lowo_%1_%2_%3 already exists, skipped.
  goto :eof
)
python 04_stratified_metrics.py --pred "data\lowo_%1_%2_%3\flood_lowo_%1_%2_%3.tif" --truth data\reference_%1.tif --strata data\strata_%1_v3 --windows data\reference_windows_%1\reference_windows.geojson --label "Table D, %2 %3, %1" --out "..\..\06 Outputs\tables\lowo_%1_%2_%3"
if errorlevel 1 set /a FAILS+=1
goto :eof



:year
call :cmp %1 rf_hand "data\flood_%1_rf_hand\flood_%1_rf_hand.tif" unet_hand "data\flood_%1_unet_hand\flood_%1_unet_hand.tif"
call :cmp %1 rf_hand_weak "data\flood_%1_rf_hand_weak\flood_%1_rf_hand_weak.tif" unet_hand_weak "data\flood_%1_unet_hand_weak\flood_%1_unet_hand_weak.tif"
call :cmp %1 rf_hand "data\flood_%1_rf_hand\flood_%1_rf_hand.tif" rf_hand_weak "data\flood_%1_rf_hand_weak\flood_%1_rf_hand_weak.tif"
call :cmp %1 unet_hand "data\flood_%1_unet_hand\flood_%1_unet_hand.tif" unet_hand_weak "data\flood_%1_unet_hand_weak\flood_%1_unet_hand_weak.tif"
call :cmp %1 otsu "data\flood_%1_v3\flood_%1_otsu.tif" rf_hand_weak "data\flood_%1_rf_hand_weak\flood_%1_rf_hand_weak.tif"
call :cmp %1 otsu "data\flood_%1_v3\flood_%1_otsu.tif" unet_hand_weak "data\flood_%1_unet_hand_weak\flood_%1_unet_hand_weak.tif"
call :cmp %1 otsu_chfeb "data\flood_%1_ch_feb_v3\flood_%1chfeb_otsu.tif" rf_hand_weak "data\flood_%1_rf_hand_weak\flood_%1_rf_hand_weak.tif"
call :cmp %1 otsu_chfeb "data\flood_%1_ch_feb_v3\flood_%1chfeb_otsu.tif" unet_hand_weak "data\flood_%1_unet_hand_weak\flood_%1_unet_hand_weak.tif"
call :cmp %1 lowo_rf_grd "data\lowo_%1_rf_grd\flood_lowo_%1_rf_grd.tif" lowo_rf_rtc "data\lowo_%1_rf_rtc\flood_lowo_%1_rf_rtc.tif"
call :cmp %1 lowo_unet_grd "data\lowo_%1_unet_grd\flood_lowo_%1_unet_grd.tif" lowo_unet_rtc "data\lowo_%1_unet_rtc\flood_lowo_%1_unet_rtc.tif"
call :cmp %1 unet_hand_weak "data\flood_%1_unet_hand_weak\flood_%1_unet_hand_weak.tif" tm_s1t_hand_weak "data\flood_%1_tm_s1t_hand_weak\flood_%1_tm_s1t_hand_weak.tif"
call :cmp %1 rf_hand_weak "data\flood_%1_rf_hand_weak\flood_%1_rf_hand_weak.tif" tm_s1t_hand_weak "data\flood_%1_tm_s1t_hand_weak\flood_%1_tm_s1t_hand_weak.tif"
call :cmp %1 tm_s1_hand "data\flood_%1_tm_s1_hand\flood_%1_tm_s1_hand.tif" tm_s1t_hand "data\flood_%1_tm_s1t_hand\flood_%1_tm_s1t_hand.tif"
call :cmp %1 tm_s1_hand_weak "data\flood_%1_tm_s1_hand_weak\flood_%1_tm_s1_hand_weak.tif" tm_s1t_hand_weak "data\flood_%1_tm_s1t_hand_weak\flood_%1_tm_s1t_hand_weak.tif"
call :cmp %1 otsu "data\flood_%1_v3\flood_%1_otsu.tif" tm_s1t_hand_weak "data\flood_%1_tm_s1t_hand_weak\flood_%1_tm_s1t_hand_weak.tif"
call :cmp %1 otsu_chfeb "data\flood_%1_ch_feb_v3\flood_%1chfeb_otsu.tif" tm_s1t_hand_weak "data\flood_%1_tm_s1t_hand_weak\flood_%1_tm_s1t_hand_weak.tif"
call :cmp %1 lowo_tm_s1_grd "data\lowo_%1_tm_s1_grd\flood_lowo_%1_tm_s1_grd.tif" lowo_tm_s1_rtc "data\lowo_%1_tm_s1_rtc\flood_lowo_%1_tm_s1_rtc.tif"
call :cmp %1 lowo_tm_s1t_grd "data\lowo_%1_tm_s1t_grd\flood_lowo_%1_tm_s1t_grd.tif" lowo_tm_s1t_rtc "data\lowo_%1_tm_s1t_rtc\flood_lowo_%1_tm_s1t_rtc.tif"
rem added 9 October: the hand runs (the ones Jal-SAR uses) against both Otsu baselines,
rem the U-Net against TerraMind (hand), and three Table D pairs
call :cmp %1 otsu "data\flood_%1_v3\flood_%1_otsu.tif" rf_hand "data\flood_%1_rf_hand\flood_%1_rf_hand.tif"
call :cmp %1 otsu "data\flood_%1_v3\flood_%1_otsu.tif" unet_hand "data\flood_%1_unet_hand\flood_%1_unet_hand.tif"
call :cmp %1 otsu "data\flood_%1_v3\flood_%1_otsu.tif" tm_s1t_hand "data\flood_%1_tm_s1t_hand\flood_%1_tm_s1t_hand.tif"
call :cmp %1 otsu "data\flood_%1_v3\flood_%1_otsu.tif" tm_s1_hand "data\flood_%1_tm_s1_hand\flood_%1_tm_s1_hand.tif"
call :cmp %1 otsu_chfeb "data\flood_%1_ch_feb_v3\flood_%1chfeb_otsu.tif" rf_hand "data\flood_%1_rf_hand\flood_%1_rf_hand.tif"
call :cmp %1 otsu_chfeb "data\flood_%1_ch_feb_v3\flood_%1chfeb_otsu.tif" unet_hand "data\flood_%1_unet_hand\flood_%1_unet_hand.tif"
call :cmp %1 otsu_chfeb "data\flood_%1_ch_feb_v3\flood_%1chfeb_otsu.tif" tm_s1t_hand "data\flood_%1_tm_s1t_hand\flood_%1_tm_s1t_hand.tif"
call :cmp %1 otsu_chfeb "data\flood_%1_ch_feb_v3\flood_%1chfeb_otsu.tif" tm_s1_hand "data\flood_%1_tm_s1_hand\flood_%1_tm_s1_hand.tif"
call :cmp %1 unet_hand "data\flood_%1_unet_hand\flood_%1_unet_hand.tif" tm_s1t_hand "data\flood_%1_tm_s1t_hand\flood_%1_tm_s1t_hand.tif"
call :cmp %1 lowo_rf_grd "data\lowo_%1_rf_grd\flood_lowo_%1_rf_grd.tif" lowo_unet_grd "data\lowo_%1_unet_grd\flood_lowo_%1_unet_grd.tif"
call :cmp %1 lowo_unet_grd "data\lowo_%1_unet_grd\flood_lowo_%1_unet_grd.tif" lowo_tm_s1t_grd "data\lowo_%1_tm_s1t_grd\flood_lowo_%1_tm_s1t_grd.tif"
call :cmp %1 lowo_tm_s1_grd "data\lowo_%1_tm_s1_grd\flood_lowo_%1_tm_s1_grd.tif" lowo_tm_s1t_grd "data\lowo_%1_tm_s1t_grd\flood_lowo_%1_tm_s1t_grd.tif"
goto :eof

:cmp
if not exist %3 (
  echo   %1 %2 vs %4: MISSING %3, skipped.
  set /a MISSING+=1
  goto :eof
)
if not exist %5 (
  echo   %1 %2 vs %4: MISSING %5, skipped.
  set /a MISSING+=1
  goto :eof
)
echo.
echo ==== %1: %4 against %2
python 27_compare_maps.py --a %3 --b %5 --truth data\reference_%1.tif --strata data\strata_%1_v3 --windows data\reference_windows_%1\reference_windows.geojson --label-a "%2" --label-b "%4" --out "..\..\06 Outputs\tables\compare_%1_%2_vs_%4" --skip-existing
if errorlevel 1 set /a FAILS+=1
goto :eof

:end
echo.
echo ============================================================
if "%FAILS%"=="0" (
  echo ALL MODELS PASS. Every step is done; %MISSING% comparison(s^) skipped because a map is missing.
  echo The summary is in 06 Outputs\tables\MODELS_SUMMARY.md
) else (
  echo %FAILS% step(s^) did not finish. Scroll up to the first PROBLEM line.
  echo Double-click this file again after fixing it: finished steps are skipped.
)
echo ============================================================
echo.
pause
