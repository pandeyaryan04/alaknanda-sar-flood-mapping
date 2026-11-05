@echo off
rem The daily forecast run, started by Windows Task Scheduler (FORECAST DAILY ON.bat).
rem It can also be double-clicked. It never waits for a key, so it can run on its own.
setlocal enabledelayedexpansion
chcp 65001 >nul
set PYTHONUTF8=1
set KMP_DUPLICATE_LIB_OK=TRUE
set "HERE=%~dp0"
set "ROOT=%~dp0..\..\.."
set "LOGS=%ROOT%\06 Outputs\jalsar\extensions\logs\daily"
set "STATUS=%LOGS%\DAILY_STATUS.txt"
if not exist "%LOGS%" mkdir "%LOGS%"
set "CONDA=%USERPROFILE%\miniconda3\condabin\conda.bat"
if not exist "%CONDA%" (
  > "%STATUS%" echo %date% %time% FAIL: conda not found at %CONDA%
  exit /b 1
)
call "%CONDA%" activate fyp
if errorlevel 1 (
  > "%STATUS%" echo %date% %time% FAIL: could not switch on the fyp environment
  exit /b 1
)
> "%STATUS%" echo started %date% %time%
call :step 1 "newest GloFAS forecast" d_1_latest "%HERE%m9_glofas_latest.py"
call :step 2 "forecast watch, GloFAS rule" d_2_watch "%HERE%m9_forecast_watch.py"
call :step 3 "forecast watch, CWC rule" d_3_watch_cwc "%HERE%m9_forecast_watch.py" --rule cwc
call :step 4 "forecast maps from the M13 models" d_4_m13 "%HERE%m13_forecast_ml.py" forecast
pushd "%ROOT%\04 Code\jalsar"
python -u jalsar.py webmap --force > "%LOGS%\d_5_webmap.txt" 2>&1
if errorlevel 1 (set "R5=FAIL") else (set "R5=PASS")
popd
>> "%STATUS%" echo step 5 web map: !R5!
>> "%STATUS%" echo finished %date% %time%
exit /b 0

:step
set "N=%~1"
set "NAME=%~2"
set "LOG=%LOGS%\%~3.txt"
python -u %4 %5 %6 %7 > "%LOG%" 2>&1
if errorlevel 1 (set "R%N%=FAIL") else (set "R%N%=PASS")
>> "%STATUS%" echo step %N% %NAME%: !R%N%!
exit /b 0
