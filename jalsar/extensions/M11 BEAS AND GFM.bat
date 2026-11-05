@echo off
setlocal enabledelayedexpansion
title M11: Beas blind test and Copernicus GFM comparison
chcp 65001 >nul
set PYTHONUTF8=1
set KMP_DUPLICATE_LIB_OK=TRUE
set "HERE=%~dp0"
set "ROOT=%~dp0..\..\.."
set "LOGS=%ROOT%\06 Outputs\jalsar\extensions\logs"
set "STATUS=%LOGS%\M11_STATUS.txt"
echo.
echo ================================================================
echo  M11. Run this AFTER "M DAY 3.bat" has finished.
echo   1 GFM probe: what the Copernicus GFM catalogue returns (internet)
echo   2 Beas: Sentinel-1 passes, 8 to 25 July 2023 (Earth Engine)
echo   3 Beas: download, terrain, the three models (Earth Engine)
echo   4 GFM against every Jal-SAR event and the Beas (internet)
echo  About 1 to 2 hours, most of it step 3. Finished downloads are
echo  kept, so running this file again carries on where it stopped.
echo ================================================================
if /i not "%CONDA_DEFAULT_ENV%"=="fyp" (
  echo  PROBLEM: the fyp environment is not switched on.
  echo  Open Anaconda Prompt, run conda activate fyp, then drag this file into that window again.
  pause
  exit /b 1
)
if not exist "%LOGS%" mkdir "%LOGS%"
> "%STATUS%" echo started %date% %time%
call :step 1 "GFM probe" m11_1_gfm_probe "%HERE%m11_gfm.py" probe
call :step 2 "Beas passes" m11_2_beas_search "%HERE%m11_beas.py" search
call :step 3 "Beas run" m11_3_beas_run "%HERE%m11_beas.py" run
if "!R1!"=="PASS" (
  call :step 4 "GFM comparison" m11_4_gfm_run "%HERE%m11_gfm.py" run
) else (
  echo  Step 4 skipped: the GFM probe did not pass. Check logs\m11_1_gfm_probe.txt
  >> "%STATUS%" echo step 4 GFM comparison: skipped, probe did not pass
)
>> "%STATUS%" echo finished %date% %time%
echo.
echo ================================================================
type "%STATUS%"
echo.
echo  Each step's result is in its log file in the logs folder.
echo  If step 3 says no pass matches, check logs\m11_2_beas_search.txt
echo ================================================================
echo.
pause
exit /b 0

:step
set "N=%~1"
set "NAME=%~2"
set "LOG=%LOGS%\%~3.txt"
echo  %time%  [%N%] %NAME% ... (log: %~3.txt)
python -u %4 %5 %6 %7 > "%LOG%" 2>&1
if errorlevel 1 (set "R%N%=FAIL") else (set "R%N%=PASS")
echo        !R%N%!
>> "%STATUS%" echo step %N% %NAME%: !R%N%!
exit /b 0
