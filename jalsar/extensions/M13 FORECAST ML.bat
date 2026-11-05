@echo off
setlocal enabledelayedexpansion
title M13 forecast flood maps from machine learning (Random Forest and U-Net)
chcp 65001 >nul
set PYTHONUTF8=1
set KMP_DUPLICATE_LIB_OK=TRUE
set "HERE=%~dp0"
set "ROOT=%~dp0..\..\.."
set "LOGS=%ROOT%\06 Outputs\jalsar\extensions\logs"
set "STATUS=%LOGS%\M13_STATUS.txt"
echo.
echo ================================================================
echo  M13 forecast flood maps from machine learning:
echo   1 self-test on a small made-up project (a few minutes)
echo   2 Random Forests: left-out tests and final forests, both label sets
echo   3 U-Nets: the same, on the graphics card
echo   4 report: scores, flow check, hindcast and the newest forecast
echo  About 1.5 to 3 hours, most of it step 3 (estimate). Finished parts
echo  are kept, so running this file again carries on.
echo  Before it: charger in, sleep Never, no other model or Jal-SAR window open.
echo ================================================================
if /i not "%CONDA_DEFAULT_ENV%"=="fyp" (
  echo  PROBLEM: the fyp environment is not switched on.
  echo  Open Anaconda Prompt, run conda activate fyp, then drag this file into that window again.
  pause
  exit /b 1
)
if not exist "%LOGS%" mkdir "%LOGS%"
> "%STATUS%" echo started %date% %time%
call :step 1 "Self-test" m13_1_selftest "%HERE%m13_forecast_ml.py" --selftest
if not "!R1!"=="PASS" (
  echo  The self-test failed, so nothing else was run. Send the log m13_1_selftest.txt.
  goto :end
)
call :step 2 "Random Forests" m13_2_rf "%HERE%m13_forecast_ml.py" rf
call :step 3 "U-Nets" m13_3_unet "%HERE%m13_forecast_ml.py" unet
call :step 4 "Report" m13_4_report "%HERE%m13_forecast_ml.py" report
:end
>> "%STATUS%" echo finished %date% %time%
echo.
echo ================================================================
type "%STATUS%"
echo.
if "!R4!"=="PASS" (
  echo  M13 PASS. Results: 06 Outputs\jalsar\extensions\m13_forecast_ml\M13_FORECAST_ML.md
  echo  Send that file for checking.
) else (
  echo  M13 FAIL. Send the logs m13_1_selftest.txt to m13_4_report.txt from
  echo  06 Outputs\jalsar\extensions\logs.
)
if "!R3!"=="FAIL" if "!R4!"=="PASS" echo  Step 3 FAILED but step 4 passed: the report was made without the U-Nets.
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
