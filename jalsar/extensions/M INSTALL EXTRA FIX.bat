@echo off
setlocal
title M INSTALL EXTRA FIX
echo.
echo ================================================================
echo  Fix for M INSTALL EXTRA (22 Oct 2026).
echo  pip put in mapbox-vector-tile 1.2.1, which is too old for shapely 2
echo  and fails on import. The current 2.2.0 asks for protobuf below 7, so
echo  pip refused it, because your protobuf 7.35.1 is held. This swaps
echo  1.2.1 for 2.2.0 without touching protobuf or anything else.
echo  (2.2.0 was tested with protobuf 7.35.1 and shapely 2.1.2: it reads tiles.)
echo ================================================================
if /i not "%CONDA_DEFAULT_ENV%"=="fyp" (
  echo  PROBLEM: the fyp environment is not switched on.
  echo  Open Anaconda Prompt, run conda activate fyp, then drag this file into that window again.
  goto :fail
)
set "HERE=%~dp0"
set "OUT=%~dp0..\..\..\06 Outputs\checks\m_extra_env"
if not exist "%OUT%\versions_before.json" (
  echo  PROBLEM: %OUT%\versions_before.json is missing. Run M INSTALL EXTRA.bat first.
  goto :fail
)

echo.
echo  Step 1 of 3: swapping mapbox-vector-tile 1.2.1 for 2.2.0
python -m pip uninstall -y mapbox-vector-tile > "%OUT%\pip_fix_log.txt" 2>&1
python -m pip install --no-deps "mapbox-vector-tile==2.2.0" >> "%OUT%\pip_fix_log.txt" 2>&1
if errorlevel 1 (
  powershell -NoProfile -Command "Get-Content -LiteralPath '%OUT%\pip_fix_log.txt' -Tail 20"
  echo  FAIL: check  06 Outputs\checks\m_extra_env\pip_fix_log.txt
  goto :fail
)
echo  done.

echo.
echo  Step 2 of 3: checking against the environment saved before M INSTALL EXTRA
python "%HERE%m0_env_check.py" snapshot "%OUT%\versions_after_fix.json" >nul
python "%HERE%m0_env_check.py" compare "%OUT%\versions_before.json" "%OUT%\versions_after_fix.json"
if errorlevel 1 (
  echo  FAIL: check the lines above.
  goto :fail
)

echo.
echo  Step 3 of 3: test imports and one test tile
python -c "import netCDF4, pmtiles, xarray, mapbox_vector_tile as m; t=m.encode([{'name':'t','features':[{'geometry':'POINT(1 2)','properties':{'a':1}}]}]); d=m.decode(t, default_options={'y_coord_down': True}); assert d['t']['features'][0]['properties']['a']==1; print('  netCDF4', netCDF4.__version__, ' xarray', xarray.__version__, ' mapbox_vector_tile', m.__version__ if hasattr(m,'__version__') else '2.2.0', ' test tile read OK')"
if errorlevel 1 (
  echo  FAIL: check the lines above.
  goto :fail
)

echo.
echo ================================================================
echo  M INSTALL EXTRA FIX PASS
echo  (pip check will say mapbox-vector-tile wants protobuf below 7.
echo  That note is expected and harmless here.)
echo ================================================================
echo.
pause
exit /b 0

:fail
echo.
echo ================================================================
echo  M INSTALL EXTRA FIX FAIL
echo ================================================================
echo.
pause
exit /b 1
