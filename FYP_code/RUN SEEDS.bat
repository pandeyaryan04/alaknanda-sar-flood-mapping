@echo off
rem Repeat runs (9 October 2026): the two leading models, unet_hand and
rem tm_s1t_hand, trained twice more with other random seeds, so the paper can
rem say how much a result moves when a model is trained again.
rem   seed2 = seed 20261008, seed3 = seed 20261009 (the original run is 20261007)
rem Each repeat run goes into its own folders ending _seed2 or _seed3
rem (data\models\unet_hand_seed2, data\flood_2021_unet_hand_seed2,
rem 06 Outputs\tables\2021_unet_hand_seed2, ...). The original runs, which
rem Tables A and D and Jal-SAR use, are never touched.
rem For every repeat run: train with exactly the original settings (the U-Net
rem with the learning rate in data\models\lr_test\lr_choice.txt), evaluate on
rem Sen1Floods11, map 2021 and 2023, score with 04. Then, for each seed, the
rem paired comparisons (tm_s1t against the U-Net, each against February
rem change detection), calibration (31), and 32_seed_summary.py, which writes
rem 06 Outputs\tables\SEEDS_SUMMARY.md.
rem Order: unet seed2, tm_s1t seed2, unet seed3, tm_s1t seed3, so if you stop
rem early you still have one extra seed of both models.
rem Every finished step is skipped, so if it stops (power cut, Ctrl+C, lid)
rem just double-click it again: training continues from its last checkpoint.
rem Estimated 5 to 7 hours on the RTX 3050 (an estimate). Needs the NVIDIA GPU.

chcp 65001 >nul
set PYTHONUTF8=1
rem two OpenMP libraries in env fyp (see models_common.py): let both run
set KMP_DUPLICATE_LIB_OK=TRUE
cd /d "%~dp0"

echo.
echo Repeat runs: unet_hand and tm_s1t_hand with two more seeds
echo.

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" "%USERPROFILE%\miniconda3" >nul 2>&1
call conda activate fyp
if errorlevel 1 (
  echo Could not activate the conda environment 'fyp'.
  pause
  exit /b 1
)

set FAILS=0
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 (
  echo No CUDA GPU found. The U-Net and TerraMind need the RTX 3050: restart the laptop and try again.
  set /a FAILS+=1
  goto :end
)

if not exist "data\models\unet_hand\model_meta.json" (
  echo The original unet_hand is missing. S7 repeats it, so it must be there first.
  set /a FAILS+=1
  goto :end
)
if not exist "data\models\tm_s1t_hand\model_meta.json" (
  echo The original tm_s1t_hand is missing. S7 repeats it, so it must be there first.
  set /a FAILS+=1
  goto :end
)

set "LR="
set /p LR=<"data\models\lr_test\lr_choice.txt"
if not defined LR (
  echo data\models\lr_test\lr_choice.txt is missing or empty: the U-Net learning rate of the
  echo original run is not known, so the repeat runs would not be like for like. Run
  echo unet_lr_test.py first.
  set /a FAILS+=1
  goto :end
)
echo   U-Net learning rate (the original run's): %LR%

echo.
echo ==== chip index, normalisation and cache (made already: skipped)
python 21_prepare_chips.py --cache --skip-existing
if errorlevel 1 (
  set /a FAILS+=1
  goto :end
)

call :model unet seed2 20261008
call :model tm_s1t seed2 20261008
call :model unet seed3 20261009
call :model tm_s1t seed3 20261009

for %%T in (seed2 seed3) do for %%Y in (2021 2023) do call :compare %%Y %%T

echo.
echo ==== calibration of the repeat runs (31)
for %%T in (seed2 seed3) do (
  python 31_calibration.py --models unet,tm_s1t --trainsets hand --run-tag %%T
  if errorlevel 1 set /a FAILS+=1
)

echo.
echo ==== summary across the seeds
python 32_seed_summary.py --force
if errorlevel 1 set /a FAILS+=1
goto :end

:model
echo.
echo ==== %1 hand, %2 (seed %3): training
if "%1"=="unet" (
  python 23_train_unet.py --trainset hand --lr %LR% --seed %3 --run-tag %2 --resume --skip-existing
) else (
  python 30_train_terramind.py --variant %1 --trainset hand --seed %3 --run-tag %2 --resume --skip-existing
)
if errorlevel 1 (
  set /a FAILS+=1
  echo   training %1 hand %2 failed, so its other steps are skipped.
  goto :eof
)
if not exist "data\models\%1_hand_%2\model_meta.json" (
  echo   training %1 hand %2 is not finished. Double-click this file again to continue it.
  set /a FAILS+=1
  goto :eof
)
echo ==== %1 hand %2: Sen1Floods11 evaluation
python 24_eval_sen1floods11.py --model %1 --trainset hand --run-tag %2 --skip-existing
if errorlevel 1 set /a FAILS+=1
call :predict %1 %2 2021
call :predict %1 %2 2023
goto :eof

:predict
echo ==== %1 hand %2: corridor map %3
python 25_predict_corridor.py --model %1 --trainset hand --run-tag %2 --year %3 --skip-existing
if errorlevel 1 (
  set /a FAILS+=1
  goto :eof
)
if exist "..\..\06 Outputs\tables\%3_%1_hand_%2\metrics_by_stratum.csv" (
  echo   score %3_%1_hand_%2 already exists, skipped.
  goto :eof
)
python 04_stratified_metrics.py --pred "data\flood_%3_%1_hand_%2\flood_%3_%1_hand_%2.tif" --truth data\reference_%3.tif --strata data\strata_%3_v3 --windows data\reference_windows_%3\reference_windows.geojson --label "Table A, %1 hand %2, %3" --out "..\..\06 Outputs\tables\%3_%1_hand_%2"
if errorlevel 1 set /a FAILS+=1
goto :eof

:compare
call :cmp %1 unet_hand_%2 "data\flood_%1_unet_hand_%2\flood_%1_unet_hand_%2.tif" tm_s1t_hand_%2 "data\flood_%1_tm_s1t_hand_%2\flood_%1_tm_s1t_hand_%2.tif"
call :cmp %1 otsu_chfeb "data\flood_%1_ch_feb_v3\flood_%1chfeb_otsu.tif" unet_hand_%2 "data\flood_%1_unet_hand_%2\flood_%1_unet_hand_%2.tif"
call :cmp %1 otsu_chfeb "data\flood_%1_ch_feb_v3\flood_%1chfeb_otsu.tif" tm_s1t_hand_%2 "data\flood_%1_tm_s1t_hand_%2\flood_%1_tm_s1t_hand_%2.tif"
goto :eof

:cmp
if not exist %3 (
  echo   %1 %2 vs %4: MISSING %3, skipped.
  set /a FAILS+=1
  goto :eof
)
if not exist %5 (
  echo   %1 %2 vs %4: MISSING %5, skipped.
  set /a FAILS+=1
  goto :eof
)
echo.
echo ==== %1: %4 against %2
python 27_compare_maps.py --a %3 --b %5 --truth data\reference_%1.tif --strata data\strata_%1_v3 --windows data\reference_windows_%1\reference_windows.geojson --label-a "%2" --label-b "%4" --out "..\..\06 Outputs\tables\seeds_compare_%1_%2_vs_%4" --skip-existing
if errorlevel 1 set /a FAILS+=1
goto :eof

:end
echo.
echo ============================================================
if "%FAILS%"=="0" (
  echo SEEDS PASS. Both models have three seeds, scored and compared.
  echo The summary is 06 Outputs\tables\SEEDS_SUMMARY.md
) else (
  echo %FAILS% step(s^) did not finish. Scroll up to the first PROBLEM line.
  echo Double-click this file again after fixing it: finished steps are skipped.
)
echo ============================================================
echo.
pause
