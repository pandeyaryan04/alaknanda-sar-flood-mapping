@echo off
setlocal
title M DOWNLOADS (GloFAS, OpenStreetMap, World Bank, OpenCelliD)
set "LOGS=%~dp0..\..\..\06 Outputs\jalsar\extensions\logs"
if not exist "%LOGS%" mkdir "%LOGS%"
echo.
echo  Downloads for M7, M9 and M10. This window can stay open all night.
echo  GloFAS requests wait in the data store's queue: hours is normal.
echo.
echo  [1/2] GloFAS forecasts and reanalysis ... (log: 06 Outputs\jalsar\extensions\glofas\glofas_log.txt)
python -u "%~dp0m9_glofas_download.py" > "%LOGS%\m9_glofas.txt" 2>&1
if errorlevel 1 (set "G=FAIL or gaps") else (set "G=PASS")
echo        %G%
echo  [2/2] OpenStreetMap layers, World Bank lines, OpenCelliD copy ... (log: 06 Outputs\jalsar\extensions\exposure\exposure_log.txt)
python -u "%~dp0m7_exposure_download.py" > "%LOGS%\m7_exposure.txt" 2>&1
if errorlevel 1 (set "X=FAIL or gaps") else (set "X=PASS")
echo        %X%
> "%LOGS%\DOWNLOADS_STATUS.txt" (
  echo GloFAS: %G%
  echo Exposure: %X%
)
echo.
echo ================================================================
echo  DOWNLOADS DONE.  GloFAS: %G%   Exposure: %X%
echo  "gaps" means some requests failed; run this file again later,
echo  finished files are skipped.
echo ================================================================
echo.
pause
