@echo off
setlocal
for %%I in ("%~dp0..\..") do set "PROJ=%%~fI"
set "CODE=%PROJ%\04 Code\gis_tools"
set "DATA=%PROJ%\05 Data\boundaries"
set "KMZ=%DATA%\vb_soi_uk.kmz"
set "COR=%PROJ%\04 Code\FYP_code\data\corridor.geojson"

echo.
echo gis_tools: village boundaries from the Survey of India KMZ
echo.
echo This has already been run once and its outputs are in
echo   05 Data\boundaries\VILLAGES_corridor.gpkg  .geojson  .csv
echo Re-run it only if the corridor changes or you want the statewide layer.
echo.

if not exist "%KMZ%" goto nokmz
if not exist "%COR%" goto nocor

call conda activate fyp
if errorlevel 1 goto noconda

python "%CODE%\t08_villages.py" --selftest
if errorlevel 1 goto selftestfailed

echo.
echo Self-test passed. Reading the KMZ. This takes two to four minutes.
echo.
python "%CODE%\t08_villages.py" --kmz "%KMZ%" --corridor "%COR%" --out "%DATA%"
goto end

:nokmz
echo ERROR: vb_soi_uk.kmz is not at
echo   %KMZ%
echo Run "RUN ME FIRST - purge and file" to move it there from Downloads.
goto end

:nocor
echo ERROR: corridor.geojson is not at
echo   %COR%
goto end

:noconda
echo ERROR: could not activate the fyp environment.
echo Open Anaconda Prompt and run this file from there instead.
goto end

:selftestfailed
echo.
echo The self-test FAILED. Nothing was written. Read the FAIL lines above.

:end
echo.
pause
