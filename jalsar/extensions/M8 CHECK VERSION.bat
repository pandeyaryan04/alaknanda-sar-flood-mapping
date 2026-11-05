@echo off
setlocal
title M8 CHECK VERSION (WorldPop-only vulnerability index, the comparison)
chcp 65001 >nul
set PYTHONUTF8=1
set KMP_DUPLICATE_LIB_OK=TRUE
set "LOGS=%~dp0..\..\..\06 Outputs\jalsar\extensions\logs"
echo.
echo ================================================================
echo  Since 22 October the main index includes OpenStreetMap access
echo  and building density. This makes the WorldPop-only comparison, in
echo  06 Outputs\jalsar\extensions\m8_vulnerability\worldpop_only\
echo ================================================================
if /i not "%CONDA_DEFAULT_ENV%"=="fyp" (
  echo  PROBLEM: the fyp environment is not switched on.
  echo  Open Anaconda Prompt, run conda activate fyp, then drag this file into that window again.
  goto :fail
)
python -u "%~dp0m8_vulnerability.py" --worldpop-only > "%LOGS%\m8_check_version.txt" 2>&1
if errorlevel 1 (
  powershell -NoProfile -Command "Get-Content -LiteralPath '%LOGS%\m8_check_version.txt' -Tail 20"
  echo  FAIL: check  06 Outputs\jalsar\extensions\logs\m8_check_version.txt
  goto :fail
)
powershell -NoProfile -Command "Get-Content -LiteralPath '%LOGS%\m8_check_version.txt' -Tail 4"
echo.
echo ================================================================
echo  M8 CHECK VERSION PASS
echo  Result: 06 Outputs\jalsar\extensions\m8_vulnerability\worldpop_only\M8_VULNERABILITY.md
echo ================================================================
echo.
pause
exit /b 0

:fail
echo.
echo ================================================================
echo  M8 CHECK VERSION FAIL
echo ================================================================
echo.
pause
exit /b 1
