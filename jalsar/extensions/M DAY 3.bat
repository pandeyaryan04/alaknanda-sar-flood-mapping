@echo off
setlocal enabledelayedexpansion
title M DAY 3: Jal-SAR extensions, last items
chcp 65001 >nul
set PYTHONUTF8=1
set KMP_DUPLICATE_LIB_OK=TRUE
set "HERE=%~dp0"
set "ROOT=%~dp0..\..\.."
set "LOGS=%ROOT%\06 Outputs\jalsar\extensions\logs"
set "STATUS=%LOGS%\M_DAY3_STATUS.txt"
echo.
echo ================================================================
echo  M DAY 3, 29 October. Everything written today, in order:
echo   1     GloFAS upstream area file (downloads about 87 MB once)
echo   2     GloFAS files read
echo   3 M10 stage relations (log curve), gauge loop, HAND
echo   4 M9  newest forecast
echo   5 M9  forecast watch, GloFAS rule and CWC rule
echo   6     village risk view and WorldPop people per village
echo   7     Open Buildings against OpenStreetMap (needs the internet)
echo   8     why the SLOPE SHAP sign is positive
echo   9     Jal-SAR web map again (risk layer, valley strip layer)
echo  10     Jal-SAR bulletins again (WorldPop villages, exposure table)
echo  About 30 to 60 minutes. A FAIL does not stop the steps after it,
echo  except step 2.
echo ================================================================
if /i not "%CONDA_DEFAULT_ENV%"=="fyp" (
  echo  PROBLEM: the fyp environment is not switched on.
  echo  Open Anaconda Prompt, run conda activate fyp, then drag this file into that window again.
  pause
  exit /b 1
)
if not exist "%LOGS%" mkdir "%LOGS%"
> "%STATUS%" echo started %date% %time%
call :step 1 "upstream area" d3_1_uparea "%HERE%m9_uparea.py"
call :step 2 "GloFAS files read" d3_2_glofas_check "%HERE%m9_glofas_read.py" --check
if not "!R2!"=="PASS" (
  echo  The GloFAS files could not be read. Check logs\d3_2_glofas_check.txt
  >> "%STATUS%" echo stopped: the GloFAS files could not be read
  pause
  exit /b 1
)
call :step 3 "M10 stage, gauge loop and HAND" d3_3_m10 "%HERE%m10_rating_hand.py"
call :step 4 "M9 newest forecast" d3_4_latest "%HERE%m9_glofas_latest.py"
call :step 5 "M9 forecast watch" d3_5_watch "%HERE%m9_forecast_watch.py"
call :step 5b "M9 forecast watch, CWC rule" d3_5b_watch_cwc "%HERE%m9_forecast_watch.py" --rule cwc
call :step 6 "risk view" d3_6_risk "%HERE%m8_risk_view.py"
call :step 7 "buildings check" d3_7_buildings "%HERE%m7_buildings_osm_check.py"
call :step 8 "slope check" d3_8_slope "%HERE%m4_slope_check.py"
pushd "%ROOT%\04 Code\jalsar"
echo  %time%  [9] Jal-SAR web map ... (log: d3_9_webmap.txt)
python -u jalsar.py webmap --force > "%LOGS%\d3_9_webmap.txt" 2>&1
if errorlevel 1 (set "R9=FAIL") else (set "R9=PASS")
echo        !R9!
>> "%STATUS%" echo step 9 web map: !R9!
echo  %time%  [10] Jal-SAR bulletins ... (log: d3_10_bulletins.txt)
python -u jalsar.py bulletin --event all --force > "%LOGS%\d3_10_bulletins.txt" 2>&1
if errorlevel 1 (set "R10=FAIL") else (set "R10=PASS")
echo        !R10!
>> "%STATUS%" echo step 10 bulletins: !R10!
popd
>> "%STATUS%" echo finished %date% %time%
echo.
echo ================================================================
type "%STATUS%"
echo.
echo  Each step's result is in its log file in the logs folder.
echo  If step 1 says FAIL, download the file by hand (logs\d3_1_uparea.txt gives the link),
echo  then run this file again: finished downloads are kept.
echo ================================================================
echo.
pause
exit /b 0

:step
set "N=%~1"
set "NAME=%~2"
set "LOG=%LOGS%\%~3.txt"
echo  %time%  [%N%] %NAME% ... (log: %~3.txt)
python -u %4 %5 %6 %7 > "%LOG%" 2>&1
if errorlevel 1 (set "R%N%=FAIL") else (set "R%N%=PASS")
echo        !R%N%!
>> "%STATUS%" echo step %N% %NAME%: !R%N%!
exit /b 0
