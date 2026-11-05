@echo off
setlocal
title M1 HIT RATE AND CSI
echo.
echo ================================================================
echo  Jal-SAR extensions, item M1: hit rate, false alarm ratio and CSI
echo  Re-scores every map into 06 Outputs\tables_m1.
echo  The old tables in 06 Outputs\tables are not touched.
echo ================================================================
if /i not "%CONDA_DEFAULT_ENV%"=="fyp" (
  echo  PROBLEM: the fyp environment is not switched on.
  echo  Open Anaconda Prompt, run conda activate fyp, then drag this file into that window again.
  goto :fail
)
findstr /c:"Stage M1" "%~dp0..\..\FYP_code\04_stratified_metrics.py" >nul
if errorlevel 1 (
  echo  PROBLEM: 04_stratified_metrics.py is still the old version (no hit rate columns).
  goto :fail
)
echo  About 30 to 40 minutes. Re-running skips the maps already done.
echo.
python "%~dp0m1_rescore.py"
if errorlevel 1 goto :fail
echo.
echo ================================================================
echo  M1 PASS
echo  Results: 06 Outputs\tables_m1\HYDRO_SCORES.md
echo ================================================================
echo.
pause
exit /b 0
:fail
echo.
echo ================================================================
echo  M1 FAIL: check the window's text.
echo ================================================================
echo.
pause
exit /b 1
