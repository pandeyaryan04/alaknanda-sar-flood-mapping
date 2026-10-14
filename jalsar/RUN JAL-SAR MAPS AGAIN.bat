@echo off
rem Jal-SAR: makes every event's maps, bulletins and the web map AGAIN, replacing
rem the ones already there (--force). Use it after a change to Jal-SAR's code
rem (git pull), e.g. the village figures and wording changed on 9 October. The
rem roads are kept. The model maps themselves do not change: the same models and
rem thresholds give the same cells. Nothing in 06 Outputs\tables changes.
rem Run it when nothing else uses the GPU. Takes about 15 to 30 minutes (estimate).

chcp 65001 >nul
set PYTHONUTF8=1
rem two OpenMP libraries in env fyp (see FYP_code\models_common.py): let both run
set KMP_DUPLICATE_LIB_OK=TRUE
cd /d "%~dp0"

echo.
echo Jal-SAR: maps for every event, made again
echo.

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" "%USERPROFILE%\miniconda3" >nul 2>&1
call conda activate fyp
if errorlevel 1 (
  echo Could not activate the conda environment 'fyp'.
  pause
  exit /b 1
)

set FAILS=0
echo ==== 1. maps for every event (again)
python jalsar.py maps --event all --force
if errorlevel 1 set /a FAILS+=1
echo.
echo ==== 2. bulletins, HTML and PDF (again)
python jalsar.py bulletin --event all --force
if errorlevel 1 set /a FAILS+=1
echo.
echo ==== 3. web map
python jalsar.py webmap --force
if errorlevel 1 set /a FAILS+=1

echo.
echo ============================================================
if "%FAILS%"=="0" (
  echo JAL-SAR MAPS AGAIN PASS. Open 06 Outputs\jalsar\webmap\jalsar_map.html and one bulletin PDF.
  echo Each event's figures are in 06 Outputs\jalsar\^<event^>\summary.json.
) else (
  echo JAL-SAR MAPS AGAIN FAIL. Scroll up to the PROBLEM line: it says what to do.
)
echo ============================================================
echo.
pause
