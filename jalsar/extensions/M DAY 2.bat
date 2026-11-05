@echo off
setlocal enabledelayedexpansion
title M DAY 2 (M7, M8, M10, M9, Jal-SAR, then M5)
chcp 65001 >nul
set PYTHONUTF8=1
rem two OpenMP libraries in env fyp (see FYP_code\models_common.py): let both run
set KMP_DUPLICATE_LIB_OK=TRUE
set "HERE=%~dp0"
set "ROOT=%~dp0..\..\.."
set "LOGS=%ROOT%\06 Outputs\jalsar\extensions\logs"
set "STATUS=%LOGS%\DAY2_STATUS.txt"
echo.
echo ================================================================
echo  Jal-SAR extensions, day 2. Every step writes a log in 06 Outputs\jalsar\extensions\logs
echo  (day2_*.txt). A failed step is noted and the run goes on; steps
echo  that need it are skipped. Earth Engine steps need the internet.
echo   1 check the GloFAS files read      7 M10 stage relation and HAND
echo   2 M7 buildings (Earth Engine)      8 M9 newest GloFAS forecast
echo   3 M7 towers                        9 M9 forecast watch
echo   4 M7 exposure on every event      10 Jal-SAR trust scores, web map,
echo   5 M8 age and sex (Earth Engine)       bulletins (again)
echo   6 M8 vulnerability index          11 M5 SHAP networks (GPU, smoke then full)
echo  Charger in. Sleep Never, lid Do nothing, while plugged in.
echo ================================================================
if /i not "%CONDA_DEFAULT_ENV%"=="fyp" (
  echo  PROBLEM: the fyp environment is not switched on.
  echo  Open Anaconda Prompt, run conda activate fyp, then drag this file into that window again.
  pause
  exit /b 1
)
if not exist "%LOGS%\DOWNLOADS_STATUS.txt" (
  echo  PROBLEM: the downloads have not finished ^(no logs\DOWNLOADS_STATUS.txt^).
  echo  Wait for the M DOWNLOADS window to say DOWNLOADS DONE, then run this again.
  pause
  exit /b 1
)
python -c "import netCDF4, pmtiles, mapbox_vector_tile" 2>nul
if errorlevel 1 (
  echo  PROBLEM: run M INSTALL EXTRA.bat first.
  pause
  exit /b 1
)
> "%STATUS%" echo started %date% %time%

set "GDONE=no"
if exist "%LOGS%\GLOFAS_STATUS.txt" set "GDONE=yes"
if "!GDONE!"=="yes" findstr /c:"STOPPED" "%LOGS%\GLOFAS_STATUS.txt" >nul && set "GDONE=no"
if "!GDONE!"=="yes" (
  call :step 1 "GloFAS files read" day2_1_glofas_check "%HERE%m9_glofas_read.py" --check
) else (
  set "R1=skipped" & >> "%STATUS%" echo step 1 GloFAS files read: skipped ^(M9 GLOFAS AGAIN has not finished^)
)
call :step 2 "M7 buildings" day2_2_buildings "%HERE%m7_buildings_gee.py"
call :step 3 "M7 towers" day2_3_towers "%HERE%m7_towers_extract.py"
call :step 4 "M7 exposure overlay" day2_4_exposure "%HERE%m7_exposure_overlay.py"
call :step 5 "M8 age and sex" day2_5_agesex "%HERE%m8_worldpop_agesex_gee.py"
if "!R5!"=="PASS" (
  call :step 6 "M8 vulnerability" day2_6_vulnerability "%HERE%m8_vulnerability.py"
) else (
  set "R6=skipped (step 5 failed)" & >> "%STATUS%" echo step 6 M8 vulnerability: skipped
)
set "GOK=no"
if "!GDONE!"=="yes" if "!R1!"=="PASS" set "GOK=yes"
if "!GOK!"=="yes" (
  call :step 7 "M10 stage and HAND" day2_7_m10 "%HERE%m10_rating_hand.py"
  call :step 8 "M9 newest forecast" day2_8_latest "%HERE%m9_glofas_latest.py"
  call :step 9 "M9 forecast watch" day2_9_watch "%HERE%m9_forecast_watch.py"
) else (
  set "R7=skipped" & set "R8=skipped" & set "R9=skipped"
  >> "%STATUS%" echo steps 7 to 9: skipped ^(GloFAS not finished or not readable; when it is, run M9 M10 FORECAST.bat^)
)
echo  %time%  [10] Jal-SAR trust scores, web map and bulletins ... (log: day2_10_jalsar.txt)
pushd "%ROOT%\04 Code\jalsar"
set "R10=PASS"
python -u jalsar.py trustscores > "%LOGS%\day2_10_jalsar.txt" 2>&1 || set "R10=FAIL"
python -u jalsar.py webmap --force >> "%LOGS%\day2_10_jalsar.txt" 2>&1 || set "R10=FAIL"
python -u jalsar.py bulletin --event all --force >> "%LOGS%\day2_10_jalsar.txt" 2>&1 || set "R10=FAIL"
popd
echo        !R10!
>> "%STATUS%" echo step 10 Jal-SAR: !R10!
call :step 11 "M5 smoke" day2_11_m5_smoke "%HERE%m5_shap_networks.py" --smoke
if "!R11!"=="PASS" (
  call :step 12 "M5 full" day2_12_m5_full "%HERE%m5_shap_networks.py"
) else (
  set "R12=skipped" & >> "%STATUS%" echo step 12 M5 full: skipped
)
>> "%STATUS%" echo finished %date% %time%
echo.
echo ================================================================
echo  DAY 2 DONE  %date% %time%
type "%STATUS%"
echo.
echo  Check the folder 06 Outputs\jalsar\extensions\logs and the reports:
echo   jalsar\extensions\m7_exposure\M7_EXPOSURE.md, m8_vulnerability\M8_VULNERABILITY.md,
echo   m10_forecast_extent\M10_RATING_HAND.md, m9_forecast\M9_FORECAST.md,
echo   m5_shap_networks\M5_SHAP_NETWORKS.md
echo ================================================================
echo.
pause
exit /b 0

:step
rem %1 number, %2 name, %3 log name, %4 script, %5 and on: options
set "N=%~1"
set "NAME=%~2"
set "LOG=%LOGS%\%~3.txt"
echo  %time%  [%N%] %NAME% ... (log: %~3.txt)
python -u %4 %5 %6 %7 > "%LOG%" 2>&1
if errorlevel 1 (set "R%N%=FAIL") else (set "R%N%=PASS")
echo        !R%N%!
>> "%STATUS%" echo step %N% %NAME%: !R%N%!
exit /b 0
