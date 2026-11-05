@echo off
setlocal
title Switch OFF the daily forecast run
set "TASK=Alaknanda forecast daily"
schtasks /query /tn "%TASK%" >nul 2>&1
if errorlevel 1 (
  echo.
  echo  DAILY OFF: there was no daily forecast run set up. Nothing to do.
) else (
  schtasks /delete /tn "%TASK%" /f
  echo.
  echo  DAILY OFF: the daily forecast run is switched off.
)
echo.
pause
