@echo off
rem Jal-SAR runner: a flood map for a new date. Finds the Sentinel-1 pass
rem through Earth Engine, downloads it into 06 Outputs\jalsar\scenes\,
rem gets the terrain strata for its orbit, runs the three models, and makes
rem the bulletin and the web map. Settings are in jalsar.env.
rem Needs the internet and Earth Engine ("earthengine authenticate" once).
rem Run "RUN JAL-SAR CHECK DOWNLOAD.bat" once first: it tests the download.
rem Takes about 15 to 40 minutes (estimate); a stopped download carries on.

chcp 65001 >nul
set PYTHONUTF8=1
rem two OpenMP libraries in env fyp (see FYP_code\models_common.py): let both run
set KMP_DUPLICATE_LIB_OK=TRUE
cd /d "%~dp0"

echo.
echo Jal-SAR runner
echo.
set "D="
set /p D=Date of interest (YYYY-MM-DD), or just Enter to use EVENT_DATE in jalsar.env: 
echo.

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" "%USERPROFILE%\miniconda3" >nul 2>&1
call conda activate fyp
if errorlevel 1 (
  echo Could not activate the conda environment 'fyp'.
  pause
  exit /b 1
)

if "%D%"=="" (
  python jalsar.py run
) else (
  python jalsar.py run --date %D%
)
set RC=%errorlevel%

echo.
echo ============================================================
if "%RC%"=="0" (
  echo JAL-SAR RUN PASS. The bulletin and run_record.json are in 06 Outputs\jalsar\^<event^>\.
) else (
  echo JAL-SAR RUN FAIL. Scroll up to the PROBLEM line: it says what to do.
)
echo ============================================================
echo.
pause
