@echo off
rem Jal-SAR: the downloader's acceptance test. Downloads the 21 October 2021
rem scene again through Earth Engine into 06 Outputs\jalsar\checks\ and
rem compares it cell by cell with the scene in 05 Data\sentinel1 (nothing in
rem 05 Data is changed). Needs the internet and Earth Engine. About 5 to 20
rem minutes (estimate). The report is CHECK REPORT.md in the same folder.
rem It always downloads afresh (--force), so a fixed downloader is what is checked.

chcp 65001 >nul
set PYTHONUTF8=1
rem two OpenMP libraries in env fyp (see FYP_code\models_common.py): let both run
set KMP_DUPLICATE_LIB_OK=TRUE
cd /d "%~dp0"

echo.
echo Jal-SAR: download check, 21 October 2021
echo.

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" "%USERPROFILE%\miniconda3" >nul 2>&1
call conda activate fyp
if errorlevel 1 (
  echo Could not activate the conda environment 'fyp'.
  pause
  exit /b 1
)

python jalsar.py check-download --force
set RC=%errorlevel%

echo.
echo ============================================================
if "%RC%"=="0" (
  echo JAL-SAR DOWNLOAD CHECK PASS. The report is 06 Outputs\jalsar\checks\CHECK REPORT.md
) else (
  echo JAL-SAR DOWNLOAD CHECK: NOT THE SAME or FAILED. Read CHECK REPORT.md and the lines above.
)
echo ============================================================
echo.
pause
