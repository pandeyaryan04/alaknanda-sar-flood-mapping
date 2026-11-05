@echo off
setlocal enabledelayedexpansion
title M9 M10 FORECAST (stage relation, HAND extent, forecast watch, web map)
chcp 65001 >nul
set PYTHONUTF8=1
set KMP_DUPLICATE_LIB_OK=TRUE
set "HERE=%~dp0"
set "ROOT=%~dp0..\..\.."
set "LOGS=%ROOT%\06 Outputs\jalsar\extensions\logs"
set "STATUS=%LOGS%\FORECAST_STATUS.txt"
echo.
echo ================================================================
echo  M9 and M10 from the GloFAS files (M DAY 2 runs these too; use this
echo  file when GloFAS finished after M DAY 2, or to run them again).
echo   1 check the GloFAS files read    3 M9 newest GloFAS forecast
echo   2 M10 stage relation and HAND    4 M9 forecast watch (GloFAS rule, then CWC rule)
echo                                    5 Jal-SAR web map again
echo ================================================================
if /i not "%CONDA_DEFAULT_ENV%"=="fyp" (
  echo  PROBLEM: the fyp environment is not switched on.
  echo  Open Anaconda Prompt, run conda activate fyp, then drag this file into that window again.
  pause
  exit /b 1
)
if not exist "%LOGS%\GLOFAS_STATUS.txt" (
  echo  PROBLEM: the GloFAS download has not finished ^(no logs\GLOFAS_STATUS.txt^).
  echo  Run M9 GLOFAS AGAIN.bat first, or wait for it to say GLOFAS DONE.
  pause
  exit /b 1
)
findstr /c:"STOPPED" "%LOGS%\GLOFAS_STATUS.txt" >nul
if not errorlevel 1 (
  echo  PROBLEM: the GloFAS download was stopped before the end.
  echo  Run M9 GLOFAS AGAIN.bat and let it reach GLOFAS DONE first.
  pause
  exit /b 1
)
> "%STATUS%" echo started %date% %time%
call :step 1 "GloFAS files read" fc_1_glofas_check "%HERE%m9_glofas_read.py" --check
if not "!R1!"=="PASS" (
  echo  The GloFAS files could not be read. Check logs\fc_1_glofas_check.txt
  >> "%STATUS%" echo stopped: the GloFAS files could not be read
  pause
  exit /b 1
)
call :step 2 "M10 stage and HAND" fc_2_m10 "%HERE%m10_rating_hand.py"
call :step 3 "M9 newest forecast" fc_3_latest "%HERE%m9_glofas_latest.py"
call :step 4 "M9 forecast watch" fc_4_watch "%HERE%m9_forecast_watch.py"
call :step 4b "M9 forecast watch, CWC rule" fc_4b_watch_cwc "%HERE%m9_forecast_watch.py" --rule cwc
echo  %time%  [5] Jal-SAR web map ... (log: fc_5_webmap.txt)
pushd "%ROOT%\04 Code\jalsar"
python -u jalsar.py webmap --force > "%LOGS%\fc_5_webmap.txt" 2>&1
if errorlevel 1 (set "R5=FAIL") else (set "R5=PASS")
popd
echo        !R5!
>> "%STATUS%" echo step 5 web map: !R5!
>> "%STATUS%" echo finished %date% %time%
echo.
echo ================================================================
type "%STATUS%"
echo.
echo  Check logs\FORECAST_STATUS.txt and the two reports:
echo   jalsar\extensions\m10_forecast_extent\M10_RATING_HAND.md, jalsar\extensions\m9_forecast\M9_FORECAST.md,
echo   jalsar\extensions\m9_forecast\cwc_rule\M9_FORECAST.md
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
