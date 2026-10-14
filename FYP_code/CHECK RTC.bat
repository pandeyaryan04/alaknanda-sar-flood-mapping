@echo off
rem Checks the ASF RTC zips in 05 Data\sentinel1_rtc and puts them on the
rem project grid. Double-click it. It deletes nothing.

chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

echo.
echo RTC check
echo.

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" "%USERPROFILE%\miniconda3" >nul 2>&1
call conda activate fyp
if errorlevel 1 (
  echo Could not activate the conda environment 'fyp'.
  pause
  exit /b 1
)

python check_rtc.py %*

echo.
pause
