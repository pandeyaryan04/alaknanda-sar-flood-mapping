@echo off
setlocal
for %%I in ("%~dp0..\..") do set "P=%%~fI"
set "CODE=%P%\04 Code\FYP_code"
set "DEM=%P%\05 Data\dem"
set "CARTO=%DEM%\DEM_CARTODEM_30m.tif"

echo.
echo DEM comparison: Copernicus GLO-30 against CartoDEM against SRTM
echo.

call conda activate fyp
if errorlevel 1 goto noconda

python "%CODE%\07_dem_comparison.py" --selftest
if errorlevel 1 goto selftestfailed

if exist "%CARTO%" goto compare
echo.
echo CartoDEM is two tiles. Merging them once into
echo   %CARTO%
echo.
python "%CODE%\merge_tiles.py" --out "%CARTO%" ^
  "%DEM%\cartodem\P5_PAN_CD_N30_000_E078_000_30m\P5_PAN_CD_N30_000_E078_000_30m\P5_PAN_CD_N30_000_E078_000_DEM_30m.tif" ^
  "%DEM%\cartodem\P5_PAN_CD_N30_000_E079_000_30m\P5_PAN_CD_N30_000_E079_000_DEM_30m.tif"
if errorlevel 1 goto mergefailed

:compare
echo.
python "%CODE%\07_dem_comparison.py" ^
  --dem "GLO30=%DEM%\DEM_COP_GLO30_30m.tif" ^
  --dem "CartoDEM=%CARTO%" ^
  --dem "SRTM=%DEM%\DEM_SRTM_30m.tif" ^
  --reference GLO30 ^
  --corridor "%CODE%\data\corridor.geojson" ^
  --centreline "%P%\05 Data\boundaries\corridor_centreline.geojson" ^
  --out "%P%\06 Outputs\dem_comparison"
goto end

:noconda
echo ERROR: could not activate the fyp environment.
echo Open Anaconda Prompt and run this file from there instead.
goto end

:selftestfailed
echo.
echo The self-test FAILED. Nothing was written.
goto end

:mergefailed
echo.
echo The CartoDEM merge failed. The error is above.

:end
echo.
pause
