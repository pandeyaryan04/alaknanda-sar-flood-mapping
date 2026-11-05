@echo off
setlocal
title M9 GLOFAS 2026 AGAIN (newest reanalysis months)
chcp 65001 >nul
set PYTHONUTF8=1
set "LOGS=%~dp0..\..\..\06 Outputs\jalsar\extensions\logs"
if not exist "%LOGS%" mkdir "%LOGS%"
echo.
echo ================================================================
echo  GloFAS reanalysis 2026 again (choice 3a, 25 Oct 2026). The 2026
echo  files stop at 30 June (version 4.0) and 31 July (version 5.0).
echo  This asks for the newest months ("intermediate"), then the whole
echo  of 2026 again. Old files are kept and put back if anything fails.
echo  Internet only, about 4 requests: 20 minutes to a few hours.
echo  Do NOT run M9 M10 FORECAST.bat or M9 GLOFAS AGAIN.bat meanwhile.
echo ================================================================
if /i not "%CONDA_DEFAULT_ENV%"=="fyp" (
  echo  PROBLEM: the fyp environment is not switched on.
  echo  Open Anaconda Prompt, run conda activate fyp, then drag this file into that window again.
  pause
  exit /b 1
)
echo  %date% %time%  asking for 2026 ... (log: logs\m9_glofas_2026.txt; detail in glofas\glofas_log.txt)
python -u "%~dp0m9_glofas_download.py" --refresh-year 2026 > "%LOGS%\m9_glofas_2026.txt" 2>&1
set "RC=%errorlevel%"
rem 0 = something new arrived, 1 = nothing new (old files all in place), anything else = stopped early
if "%RC%"=="0" (set "G=PASS, new 2026 data arrived") else if "%RC%"=="1" (set "G=nothing new from the data store, old files kept") else (set "G=STOPPED before the end, old files kept")
> "%LOGS%\GLOFAS_2026_STATUS.txt" (
  echo GloFAS 2026 again: %G%
  echo finished %date% %time%
)
echo.
echo ================================================================
echo  GLOFAS 2026 DONE: %G%
echo  Next: M9 M10 FORECAST.bat (about 15 minutes), in this window or a new one.
echo  Check logs\GLOFAS_2026_STATUS.txt if anything looks wrong.
echo ================================================================
echo.
pause
