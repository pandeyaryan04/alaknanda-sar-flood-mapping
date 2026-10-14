@echo off
setlocal
for %%I in ("%~dp0..\..") do set "P=%%~fI"
set "C=%P%\04 Code\FYP_code"
echo.
echo Sampling design figure, both events.
echo.
call conda activate fyp
if errorlevel 1 goto noconda
python "%C%\10_window_map.py" --selftest
if errorlevel 1 goto failed

echo.
echo --- 2021 ---
python "%C%\10_window_map.py" ^
  --corridor "%C%\data\corridor.geojson" ^
  --centreline "%P%\05 Data\boundaries\corridor_centreline.geojson" ^
  --windows "%C%\data\reference_windows_2021\reference_windows.geojson" ^
  --villages "%P%\05 Data\boundaries\VILLAGES_corridor.geojson" ^
  --title "Sampling design, 21 October 2021 event" ^
  --out "%P%\06 Outputs\figures\sampling_design_2021.png"

echo.
echo --- 2023, with the cloud gap ---
python "%C%\10_window_map.py" ^
  --corridor "%C%\data\corridor.geojson" ^
  --centreline "%P%\05 Data\boundaries\corridor_centreline.geojson" ^
  --windows "%C%\data\reference_windows_2023\reference_windows.geojson" ^
  --gapmask "%P%\05 Data\sentinel2\S2_2023_event_gapmask.tif" ^
  --villages "%P%\05 Data\boundaries\VILLAGES_corridor.geojson" ^
  --title "Sampling design, 16 August 2023 event" ^
  --out "%P%\06 Outputs\figures\sampling_design_2023.png"
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
