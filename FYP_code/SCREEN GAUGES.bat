@echo off
setlocal
for %%I in ("%~dp0..\..") do set "PROJ=%%~fI"
set "CODE=%PROJ%\04 Code\FYP_code"
set "DIR=%PROJ%\05 Data\gauges\guardian"
set "REF=%PROJ%\05 Data\gauges\GAUGE_station_reference.csv"

echo.
echo Screening the GUARDIAN water level files.
echo.

if not exist "%CODE%\screen_gauges.py" goto nocode
if not exist "%DIR%" goto nodir

call conda activate fyp
if errorlevel 1 goto noconda

echo Running the self-test first.
echo.
python "%CODE%\screen_gauges.py" --selftest
if errorlevel 1 goto selftestfailed

echo.
echo Self-test passed. Screening the real files.
echo.
python "%CODE%\screen_gauges.py" --dir "%DIR%" --ref "%REF%"
echo.
echo Done. Read GAUGE_screen_report.csv in the guardian folder.
echo BAND and LEAK rows are already removed from GAUGE_wse_clean.csv.
echo SPIKE rows are kept and marked. Look at them, then re-run with --drop-spikes
echo if you agree they are not real.
goto end

:nocode
echo ERROR: screen_gauges.py is not in
echo   %CODE%
goto end

:nodir
echo ERROR: the guardian folder is not at
echo   %DIR%
goto end

:noconda
echo ERROR: could not activate the fyp environment.
echo Open Anaconda Prompt and run this file from there instead.
goto end

:selftestfailed
echo.
echo The self-test FAILED. Do not trust the output. Read the FAIL lines above.

:end
echo.
pause
