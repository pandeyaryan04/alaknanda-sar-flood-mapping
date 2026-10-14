@echo off
rem Checks that TerraMind can run on this laptop: terratorch, the weights file
rem (data\weights\TerraMind_v1_small.pt), a short test on the GPU with the real
rem settings, and the speed, from which it estimates how long every TerraMind
rem step takes. Double-click it. It only reads.
rem Its report goes to 06 Outputs\models\terramind_preflight_<computer>.md
rem Takes 2 to 5 minutes.

chcp 65001 >nul
set PYTHONUTF8=1
rem two OpenMP libraries in env fyp (see models_common.py): let both run
set KMP_DUPLICATE_LIB_OK=TRUE
cd /d "%~dp0"

echo.
echo TerraMind preflight
echo.

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" "%USERPROFILE%\miniconda3" >nul 2>&1
call conda activate fyp
if errorlevel 1 (
  echo Could not activate the conda environment 'fyp'.
  pause
  exit /b 1
)

set FAILS=0
python 29_terramind_preflight.py
if errorlevel 1 set /a FAILS+=1

echo.
echo ============================================================
if "%FAILS%"=="0" (
  echo TERRAMIND PREFLIGHT PASS. Next: double-click RUN TERRAMIND SMOKE.bat
) else (
  echo TERRAMIND PREFLIGHT FAIL. Scroll up to the first [FAIL] line: it says what to do.
  echo Nothing was changed. Fix it and double-click this file again.
)
echo ============================================================
echo.
pause
