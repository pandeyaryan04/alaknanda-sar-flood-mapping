@echo off
setlocal
for %%I in ("%~dp0..\..") do set "P=%%~fI"
echo.
echo Gauge hydrographs for the three events, with warning and danger levels.
echo.
call conda activate fyp
if errorlevel 1 goto noconda
python "%P%\04 Code\FYP_code\08_gauge_hydrographs.py" --selftest
if errorlevel 1 goto failed
echo.
python "%P%\04 Code\FYP_code\08_gauge_hydrographs.py" ^
  --clean "%P%\05 Data\gauges\guardian\GAUGE_wse_clean.csv" ^
  --ref "%P%\05 Data\gauges\GAUGE_station_reference.csv" ^
  --out "%P%\06 Outputs\gauges"
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
