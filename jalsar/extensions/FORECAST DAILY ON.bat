@echo off
setlocal
title Switch ON the daily forecast run
set "TASK=Alaknanda forecast daily"
set "RUN=%~dp0FORECAST DAILY RUN.bat"
echo.
echo ================================================================
echo  This sets Windows to run "FORECAST DAILY RUN.bat" once a day:
echo  the newest GloFAS forecast, the forecast watch, the M13 forecast
echo  maps and the Jal-SAR web map. It runs only while you are signed in
echo  and the laptop is on; a missed day is simply skipped.
echo  Results: 06 Outputs\jalsar\extensions\logs\daily\DAILY_STATUS.txt
echo  To stop it: double-click FORECAST DAILY OFF.bat
echo ================================================================
echo.
set "T="
set /p "T=Time of day, 24-hour HH:MM (press Enter for 10:00): "
if "%T%"=="" set "T=10:00"
schtasks /create /tn "%TASK%" /tr "\"%RUN%\"" /sc daily /st %T% /f
if errorlevel 1 (
  echo.
  echo  FAIL: Windows did not accept the task. Check the time was typed as HH:MM.
) else (
  echo.
  echo  DAILY ON: the forecast will run every day at %T%.
)
echo.
pause
