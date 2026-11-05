@echo off
setlocal
title M9 GLOFAS AGAIN (download version 2)
chcp 65001 >nul
set PYTHONUTF8=1
set "LOGS=%~dp0..\..\..\06 Outputs\jalsar\extensions\logs"
if not exist "%LOGS%" mkdir "%LOGS%"
echo.
echo ================================================================
echo  GloFAS download, version 2 (small requests). The first night's
echo  requests were all refused as too large. This only downloads; it
echo  uses the internet, not the GPU, so it can run beside M OVERNIGHT.
echo  Requests wait in the data store's queue: many hours is normal.
echo  Finished files are skipped, so you can stop and start it again.
echo ================================================================
if /i not "%CONDA_DEFAULT_ENV%"=="fyp" (
  echo  PROBLEM: the fyp environment is not switched on.
  echo  Open Anaconda Prompt, run conda activate fyp, then drag this file into that window again.
  pause
  exit /b 1
)
if exist "%LOGS%\GLOFAS_STATUS.txt" del "%LOGS%\GLOFAS_STATUS.txt"
echo  %date% %time%  [1/2] one small test request ... (log: logs\m9_glofas_v2.txt)
python -u "%~dp0m9_glofas_download.py" --test > "%LOGS%\m9_glofas_v2.txt" 2>&1
if errorlevel 1 (
  echo  FAIL: even the small test was refused. Check  06 Outputs\jalsar\extensions\logs\m9_glofas_v2.txt
  > "%LOGS%\GLOFAS_STATUS.txt" echo GloFAS v2: test FAIL
  pause
  exit /b 1
)
echo        test PASS
echo  %time%  [2/2] all forecasts and the reanalysis ... (same log; the full detail is in
echo           06 Outputs\jalsar\extensions\glofas\glofas_log.txt)
python -u "%~dp0m9_glofas_download.py" >> "%LOGS%\m9_glofas_v2.txt" 2>&1
set "RC=%errorlevel%"
rem 0 = all downloaded, 1 = finished with gaps; anything else (Ctrl+C, a crash) = stopped early
if "%RC%"=="0" (set "G=PASS") else if "%RC%"=="1" (set "G=finished with gaps") else (set "G=STOPPED before the end, run this file again")
> "%LOGS%\GLOFAS_STATUS.txt" (
  echo GloFAS v2: %G%
  echo finished %date% %time%
)
echo.
echo ================================================================
echo  GLOFAS DONE: %G%
echo  "gaps" is expected for the newest reanalysis month or two. For
echo  anything more, run this file again later (finished files are skipped).
echo ================================================================
echo.
pause
