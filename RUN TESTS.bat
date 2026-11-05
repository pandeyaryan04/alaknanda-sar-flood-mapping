@echo off
setlocal
title Automated tests: every test in 04 Code\tests
rem Runs the automated tests in 04 Code\tests and saves the full output in
rem 06 Outputs\tests\, so the test count in the report has a dated log.
rem The tests use small made-up data only; they change no result.
rem Run it from Anaconda Prompt with the fyp environment switched on.
rem The time is not measured yet; leave it running until the end box shows.
chcp 65001 >nul
set PYTHONUTF8=1
set KMP_DUPLICATE_LIB_OK=TRUE
cd /d "%~dp0"
if /i not "%CONDA_DEFAULT_ENV%"=="fyp" (
  echo  PROBLEM: the fyp environment is not switched on.
  echo  Open Anaconda Prompt, run conda activate fyp, then drag this file into that window again.
  pause
  exit /b 1
)
python -m pytest --version >nul 2>&1
if errorlevel 1 (
  echo  pytest is not installed in fyp. Installing it now ^(needs the internet^)...
  python -m pip install pytest
  if errorlevel 1 (
    echo  PROBLEM: pytest could not be installed. Check the internet and run this file again.
    pause
    exit /b 1
  )
)
set "OUT=%~dp0..\06 Outputs\tests"
if not exist "%OUT%" mkdir "%OUT%"
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd_HHmm"') do set "STAMP=%%i"
set "LOG=%OUT%\TESTS_%STAMP%.txt"
echo.
echo  Running the tests. The full output goes to:
echo  %LOG%
echo.
> "%LOG%" echo Tests started %date% %time%
python -m pytest tests -q -rfEs >> "%LOG%" 2>&1
set "RC=%errorlevel%"
>> "%LOG%" echo Tests ended %date% %time%, pytest exit code %RC%
echo  Last lines of the log:
powershell -NoProfile -Command "Get-Content -LiteralPath $env:LOG -Tail 25"
echo.
echo ================================================================
if "%RC%"=="0" (
  echo  TESTS PASS. The log file named above is the record of this run.
) else (
  echo  TESTS FAIL ^(exit code %RC%^). Send the log file named above for checking.
)
echo ================================================================
pause
exit /b %RC%
