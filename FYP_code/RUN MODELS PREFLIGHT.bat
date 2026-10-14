@echo off
rem Checks this machine and every data file the models need. Double-click it.
rem It only reads. Its report goes to 06 Outputs\models\preflight_<computer>.md
rem Takes 1 to 3 minutes.

chcp 65001 >nul
set PYTHONUTF8=1
rem two OpenMP libraries in env fyp (see models_common.py): let both run
set KMP_DUPLICATE_LIB_OK=TRUE
cd /d "%~dp0"

echo.
echo Models preflight
echo.

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" "%USERPROFILE%\miniconda3" >nul 2>&1
call conda activate fyp
if errorlevel 1 (
  echo Could not activate the conda environment 'fyp'.
  pause
  exit /b 1
)

set FAILS=0
python 20_models_preflight.py
if errorlevel 1 set /a FAILS+=1

echo.
echo ============================================================
if "%FAILS%"=="0" (
  echo PREFLIGHT PASS. Next: double-click RUN MODELS SMOKE.bat
) else (
  echo PREFLIGHT FAIL. Scroll up to the first [FAIL] line: it says what to do.
  echo Nothing was changed. Fix it and double-click this file again.
)
echo ============================================================
echo.
pause
