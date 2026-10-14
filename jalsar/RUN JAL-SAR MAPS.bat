@echo off
rem Jal-SAR: flood maps, agreement, trust layer and exposure for every event
rem whose scene is on disk (21 Oct 2021, 16 Aug 2023, 19 Jul 2023), into
rem 06 Outputs\jalsar\<event>\. First fetches the corridor's OpenStreetMap
rem roads once (needs the internet; skipped when already there).
rem Then the bulletin of every event (HTML and PDF) and the web map.
rem Run it after L6, when every model is trained and nothing else uses the GPU.
rem An event or bulletin already done is kept; the web map is always made
rem again so it carries every event. Takes about 20 to 50 minutes (estimate).

chcp 65001 >nul
set PYTHONUTF8=1
rem two OpenMP libraries in env fyp (see FYP_code\models_common.py): let both run
set KMP_DUPLICATE_LIB_OK=TRUE
cd /d "%~dp0"

echo.
echo Jal-SAR: maps for every event
echo.

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" "%USERPROFILE%\miniconda3" >nul 2>&1
call conda activate fyp
if errorlevel 1 (
  echo Could not activate the conda environment 'fyp'.
  pause
  exit /b 1
)

set FAILS=0
echo ==== 1. OpenStreetMap roads (once)
python jalsar.py roads
if errorlevel 1 (
  echo   roads not fetched; the maps go ahead without the road table.
)
echo.
echo ==== 2. maps for every event
python jalsar.py maps --event all
if errorlevel 1 set /a FAILS+=1
echo.
echo ==== 3. bulletins (HTML and PDF)
python jalsar.py bulletin --event all
if errorlevel 1 set /a FAILS+=1
echo.
echo ==== 4. web map
python jalsar.py webmap --force
if errorlevel 1 set /a FAILS+=1

echo.
echo ============================================================
if "%FAILS%"=="0" (
  echo JAL-SAR MAPS PASS. Open 06 Outputs\jalsar\webmap\jalsar_map.html and one bulletin PDF.
  echo Each event's figures are in 06 Outputs\jalsar\^<event^>\summary.json.
) else (
  echo JAL-SAR MAPS FAIL. Scroll up to the PROBLEM line: it says what to do.
)
echo ============================================================
echo.
pause
