@echo off
setlocal
title M0 INSTALL PACKAGES
echo.
echo ================================================================
echo  Jal-SAR extensions, step M0: install shap and cdsapi into env fyp
echo  Every package already installed is held at its current version.
echo ================================================================
if /i not "%CONDA_DEFAULT_ENV%"=="fyp" (
  echo  PROBLEM: the fyp environment is not switched on.
  echo  Open Anaconda Prompt, run conda activate fyp, then drag this file into that window again.
  goto :fail
)
set "HERE=%~dp0"
set "OUT=%~dp0..\..\..\06 Outputs\checks\m0_env_2026-10-21"
if not exist "%OUT%" mkdir "%OUT%"

echo.
echo  Step 1 of 4: saving the environment as it is now, so it can be put back
call conda env export > "%OUT%\fyp_before_env.yml" 2>nul
call conda list --explicit > "%OUT%\fyp_before_explicit.txt" 2>nul
python -m pip freeze > "%OUT%\fyp_before_pip.txt" 2>nul
python "%HERE%m0_env_check.py" snapshot "%OUT%\versions_before.json"
if errorlevel 1 goto :fail
python "%HERE%m0_env_check.py" constraints "%OUT%\constraints.txt"
if errorlevel 1 goto :fail

echo.
echo  Step 2 of 4: installing shap and cdsapi (a few minutes)
python -m pip install "shap" "cdsapi>=0.7.7" -c "%OUT%\constraints.txt" > "%OUT%\pip_install_log.txt" 2>&1
if errorlevel 1 (
  echo  The last lines of the install log:
  powershell -NoProfile -Command "Get-Content -LiteralPath '%OUT%\pip_install_log.txt' -Tail 25"
  echo.
  echo  FAIL: pip could not add shap and cdsapi without changing a package you already have.
  echo  Your existing packages were held, so the models are as they were.
  echo  Check the file  06 Outputs\checks\m0_env_2026-10-21\pip_install_log.txt
  goto :fail
)
echo  installed.

echo.
echo  Step 3 of 4: checking that nothing already installed has changed
python "%HERE%m0_env_check.py" snapshot "%OUT%\versions_after.json" >nul
python "%HERE%m0_env_check.py" compare "%OUT%\versions_before.json" "%OUT%\versions_after.json"
if errorlevel 1 (
  echo  FAIL: check the lines above and the folder  06 Outputs\checks\m0_env_2026-10-21
  goto :fail
)

echo.
echo  Step 4 of 4: test imports
python "%HERE%m0_env_check.py" imports
if errorlevel 1 (
  echo  FAIL: check the lines above.
  goto :fail
)

echo.
echo ================================================================
echo  M0 INSTALL PASS
echo  Records saved in 06 Outputs\checks\m0_env_2026-10-21
echo ================================================================
echo.
pause
exit /b 0

:fail
echo.
echo ================================================================
echo  M0 INSTALL FAIL
echo ================================================================
echo.
pause
exit /b 1
