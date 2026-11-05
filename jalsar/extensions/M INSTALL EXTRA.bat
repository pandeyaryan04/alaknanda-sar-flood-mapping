@echo off
setlocal
title M INSTALL EXTRA
echo.
echo ================================================================
echo  Jal-SAR extensions: install netCDF4, pmtiles and mapbox-vector-tile into env fyp
echo  (netCDF4 reads the GloFAS files; the other two read the OpenCelliD
echo  tower file). Every package already installed is held at its version.
echo  Run this only after M OVERNIGHT has finished.
echo ================================================================
if /i not "%CONDA_DEFAULT_ENV%"=="fyp" (
  echo  PROBLEM: the fyp environment is not switched on.
  echo  Open Anaconda Prompt, run conda activate fyp, then drag this file into that window again.
  goto :fail
)
set "HERE=%~dp0"
set "OUT=%~dp0..\..\..\06 Outputs\checks\m_extra_env"
if not exist "%OUT%" mkdir "%OUT%"

echo.
echo  Step 1 of 4: saving the environment as it is now
call conda env export > "%OUT%\fyp_before_env.yml" 2>nul
python -m pip freeze > "%OUT%\fyp_before_pip.txt" 2>nul
python "%HERE%m0_env_check.py" snapshot "%OUT%\versions_before.json"
if errorlevel 1 goto :fail
python "%HERE%m0_env_check.py" constraints "%OUT%\constraints.txt"
if errorlevel 1 goto :fail

echo.
echo  Step 2 of 4: installing (a few minutes)
python -m pip install "netCDF4" "pmtiles" "mapbox-vector-tile" -c "%OUT%\constraints.txt" > "%OUT%\pip_install_log.txt" 2>&1
if errorlevel 1 (
  echo  The last lines of the install log:
  powershell -NoProfile -Command "Get-Content -LiteralPath '%OUT%\pip_install_log.txt' -Tail 25"
  echo.
  echo  FAIL: pip could not add them without changing a package you already have.
  echo  Nothing you had was changed. Check  06 Outputs\checks\m_extra_env\pip_install_log.txt
  goto :fail
)
echo  installed.

echo.
echo  Step 3 of 4: checking that nothing already installed has changed
python "%HERE%m0_env_check.py" snapshot "%OUT%\versions_after.json" >nul
python "%HERE%m0_env_check.py" compare "%OUT%\versions_before.json" "%OUT%\versions_after.json"
if errorlevel 1 (
  echo  FAIL: check the lines above and the folder  06 Outputs\checks\m_extra_env
  goto :fail
)

echo.
echo  Step 4 of 4: test imports
python -c "import netCDF4, pmtiles, mapbox_vector_tile, xarray; print('  netCDF4', netCDF4.__version__, ' xarray', xarray.__version__, ' pmtiles and mapbox_vector_tile OK')"
if errorlevel 1 (
  echo  FAIL: check the lines above.
  goto :fail
)

echo.
echo ================================================================
echo  M INSTALL EXTRA PASS
echo ================================================================
echo.
pause
exit /b 0

:fail
echo.
echo ================================================================
echo  M INSTALL EXTRA FAIL
echo ================================================================
echo.
pause
exit /b 1
