@echo off
rem Remakes the four files that were made without a script
rem and compares each with the file in use. Double-click it.
rem It writes only to 06 Outputs\remake\ and never touches a file in use.

chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

echo.
echo Remake check
echo.

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" "%USERPROFILE%\miniconda3" >nul 2>&1
call conda activate fyp
if errorlevel 1 (
  echo Could not activate the conda environment 'fyp'.
  pause
  exit /b 1
)

set FAILS=0

echo ==== a. permanent-water mask
python 12_permanent_water_mask.py --force
if errorlevel 1 set /a FAILS+=1

echo.
echo ==== b. cloud-gap holes, 2021
python 13_cloudgap_holes.py --year 2021 --force
if errorlevel 1 set /a FAILS+=1
echo.
echo ==== b. cloud-gap holes, 2023
python 13_cloudgap_holes.py --year 2023 --force
if errorlevel 1 set /a FAILS+=1

echo.
echo ==== c. reservoir outline
python 14_reservoir_outline.py --force
if errorlevel 1 set /a FAILS+=1

echo.
echo ==== d. February completion of the reference, 2021
python 15_reference_dry_completion.py --year 2021 --force
if errorlevel 1 set /a FAILS+=1
echo.
echo ==== d. February completion of the reference, 2023
python 15_reference_dry_completion.py --year 2023 --force
if errorlevel 1 set /a FAILS+=1

echo.
echo ==== d. reference rasters rebuilt from the remade files, 2021
python 06_rasterise_reference.py --event "..\..\06 Outputs\remake\reference_2021_event_CORRECTED_dry_added.gpkg" --event-layer water_event --dry-layer water_dry --windows data/reference_windows_2021/reference_windows.geojson --holes "..\..\06 Outputs\remake\S2_2021_event_cloudgap.gpkg" --holes-layer holes --grid data/flood_2021_v3/flood_2021_otsu.tif --out "..\..\06 Outputs\remake\reference_2021.tif"
python -c "import rasterio,sys;a=rasterio.open(r'..\..\06 Outputs\remake\reference_2021.tif').read(1);b=rasterio.open(r'data\reference_2021.tif').read(1);d=int((a!=b).sum());print('  reference_2021.tif, cells that differ from the file in use:',d,'PASS' if d==0 else 'DIFF');sys.exit(1 if d else 0)"
if errorlevel 1 set /a FAILS+=1
echo.
echo ==== d. reference rasters rebuilt from the remade files, 2023
python 06_rasterise_reference.py --event "..\..\06 Outputs\remake\reference_2023_event_CORRECTED_dry_added.gpkg" --event-layer water_event --dry-layer water_dry --windows data/reference_windows_2023/reference_windows.geojson --holes "..\..\06 Outputs\remake\S2_2023_event_cloudgap.gpkg" --holes-layer holes --grid data/flood_2023_v3/flood_2023_otsu.tif --out "..\..\06 Outputs\remake\reference_2023.tif"
python -c "import rasterio,sys;a=rasterio.open(r'..\..\06 Outputs\remake\reference_2023.tif').read(1);b=rasterio.open(r'data\reference_2023.tif').read(1);d=int((a!=b).sum());print('  reference_2023.tif, cells that differ from the file in use:',d,'PASS' if d==0 else 'DIFF');sys.exit(1 if d else 0)"
if errorlevel 1 set /a FAILS+=1

echo.
echo ============================================================
if "%FAILS%"=="0" (
  echo ALL 8 CHECKS PASS. Every remade file matches the file in use.
) else (
  echo %FAILS% of 8 checks did NOT pass. Scroll up for the line marked DIFF.
  echo Nothing in use was changed.
)
echo ============================================================
echo.
pause
