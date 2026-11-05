@echo off
setlocal enabledelayedexpansion
title Smoke test: is the laptop ready for M13, the daily forecast and the live demo?
chcp 65001 >nul
set PYTHONUTF8=1
set KMP_DUPLICATE_LIB_OK=TRUE
set "HERE=%~dp0"
set "ROOT=%~dp0..\..\.."
set "JS=%ROOT%\04 Code\jalsar"
set "LOGS=%ROOT%\06 Outputs\jalsar\extensions\logs\smoke"
set "STATUS=%LOGS%\SMOKE_STATUS.txt"
echo.
echo ================================================================
echo  Smoke test (about 10 to 25 minutes). It changes no result.
echo   1 packages, graphics card, disk space, key files
echo   2 Jal-SAR self-test on a small made-up project
echo   3 Jal-SAR: list the events it knows
echo   4 M13 self-test on a small made-up project
echo   5 GloFAS files on the laptop can be read
echo   6 Earth Engine works, and which radar dates suit the live demo
echo   7 the GloFAS data store answers (downloads the newest forecast)
echo  Needs the internet for steps 6 and 7. Charger in.
echo  If step 7 is still running after 10 minutes, close the window:
echo  steps 1 to 6 are already saved.
echo ================================================================
if /i not "%CONDA_DEFAULT_ENV%"=="fyp" (
  echo  PROBLEM: the fyp environment is not switched on.
  echo  Open Anaconda Prompt, run conda activate fyp, then drag this file into that window again.
  pause
  exit /b 1
)
if not exist "%LOGS%" mkdir "%LOGS%"
> "%STATUS%" echo started %date% %time%
call :step 1 "Packages, graphics card, files" s_1_env "%HERE%smoke_env.py"
call :step 2 "Jal-SAR self-test" s_2_jalsar_selftest "%JS%\jalsar.py" --selftest
call :step 3 "Jal-SAR events" s_3_jalsar_events "%JS%\jalsar.py" events
call :step 4 "M13 self-test" s_4_m13_selftest "%HERE%m13_forecast_ml.py" --selftest
call :step 5 "GloFAS files read" s_5_glofas_read "%HERE%m9_glofas_read.py" --check
call :step 6 "Earth Engine and demo dates" s_6_demo_dates "%HERE%smoke_demo_dates.py"
call :step 7 "GloFAS data store" s_7_glofas_latest "%HERE%m9_glofas_latest.py"
>> "%STATUS%" echo finished %date% %time%
echo.
echo ================================================================
type "%STATUS%"
echo.
echo  Send these two files (both in 06 Outputs\jalsar\extensions\logs\smoke):
echo    SMOKE_STATUS.txt
echo    demo_dates.csv
echo  and, for any FAIL, that step's log s_^<n^>_*.txt from the same folder.
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
