@echo off
setlocal
for %%I in ("%~dp0..\..") do set "P=%%~fI"
echo.
echo Merging the three district landslide inventories, removing cross-border duplicates.
echo.
call conda activate fyp
if errorlevel 1 goto noconda
python "%P%\04 Code\FYP_code\09_landslide_merge.py" --selftest
if errorlevel 1 goto failed
echo.
python "%P%\04 Code\FYP_code\09_landslide_merge.py" ^
  --dir "%P%\05 Data\landslide" ^
  --corridor "%P%\04 Code\FYP_code\data\corridor.geojson" ^
  --out "%P%\05 Data\landslide"
goto end
:noconda
echo ERROR: could not activate the fyp environment. Use Anaconda Prompt.
goto end
:failed
echo.
echo The self-test FAILED. Nothing was written.
:end
echo.
pause
