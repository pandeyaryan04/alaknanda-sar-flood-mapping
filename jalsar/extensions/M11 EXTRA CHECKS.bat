@echo off
setlocal enabledelayedexpansion
title M11 extra checks: pre-flood radar and optical images for the Beas
chcp 65001 >nul
set PYTHONUTF8=1
set KMP_DUPLICATE_LIB_OK=TRUE
set "HERE=%~dp0"
set "ROOT=%~dp0..\..\.."
set "LOGS=%ROOT%\06 Outputs\jalsar\extensions\logs"
set "STATUS=%LOGS%\M11_EXTRA_STATUS.txt"
echo.
echo ================================================================
echo  M11 extra checks on the Beas U-Net flood patches:
echo   1 pre-flood radar: Sentinel-1, April to mid-June 2023 (Earth Engine)
echo   2 optical images: Sentinel-2 and Landsat, 10 to 20 July 2023 (Earth Engine)
echo   3 merge: a radar hint and an optical hint for every patch
echo  About 15 to 40 minutes, most of it step 1. A finished download is
echo  kept, so running this file again carries on.
echo ================================================================
if /i not "%CONDA_DEFAULT_ENV%"=="fyp" (
  echo  PROBLEM: the fyp environment is not switched on.
  echo  Open Anaconda Prompt, run conda activate fyp, then drag this file into that window again.
  pause
  exit /b 1
)
if not exist "%LOGS%" mkdir "%LOGS%"
> "%STATUS%" echo started %date% %time%
call :step 1 "Pre-flood radar" m11x_1_pre "%HERE%m11_extra_checks.py" pre
call :step 2 "Optical images" m11x_2_optical "%HERE%m11_extra_checks.py" optical
call :step 3 "Merge" m11x_3_merge "%HERE%m11_extra_checks.py" merge
>> "%STATUS%" echo finished %date% %time%
echo.
echo ================================================================
type "%STATUS%"
echo.
if "!R3!"=="PASS" (
  echo  M11 EXTRA PASS. Results: 06 Outputs\jalsar\extensions\m11_beas\ndem_check\EXTRA_CHECKS.md
) else (
  echo  M11 EXTRA FAIL. Send the logs m11x_1_pre.txt, m11x_2_optical.txt and
  echo  m11x_3_merge.txt from 06 Outputs\jalsar\extensions\logs.
)
echo  Step 1 or 2 FAIL with step 3 PASS: the merge ran without that check.
echo  If a log says Earth Engine is not signed in, type  earthengine authenticate
echo  in this window, then drag this file in again.
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
