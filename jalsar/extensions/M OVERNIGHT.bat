@echo off
setlocal
title M OVERNIGHT (M4 SHAP, M6 PCA, and the downloads)
set "ROOT=%~dp0..\..\.."
set "LOGS=%ROOT%\06 Outputs\jalsar\extensions\logs"
echo.
echo ================================================================
echo  Jal-SAR extensions overnight run
echo   1. waits until M1 has finished (06 Outputs\tables_m1\HYDRO_SCORES.md)
echo   2. opens a second window for the downloads (M7, M9, M10)
echo   3. here: M4 SHAP Random Forest (smoke, then full)
echo   4. here: M6 PCA (smoke, then full)
echo  Charger in. Sleep Never, lid Do nothing, while plugged in.
echo ================================================================
if /i not "%CONDA_DEFAULT_ENV%"=="fyp" (
  echo  PROBLEM: the fyp environment is not switched on.
  echo  Open Anaconda Prompt, run conda activate fyp, then drag this file into that window again.
  pause
  exit /b 1
)
if not exist "%LOGS%" mkdir "%LOGS%"
set /a WAITED=0
:wait
if exist "%ROOT%\06 Outputs\tables_m1\HYDRO_SCORES.md" goto go
if %WAITED% GEQ 240 (
  echo  M1 has not finished after 4 hours. Starting anyway.
  goto go
)
if %WAITED%==0 echo  Waiting for M1 to finish (checks every minute) ...
timeout /t 60 /nobreak >nul
set /a WAITED+=1
goto wait
:go
timeout /t 30 /nobreak >nul
echo  %date% %time%  starting
start "M DOWNLOADS" cmd /c ""%~dp0M DOWNLOADS.bat""

set "R4S=not run" & set "R4=not run" & set "R6S=not run" & set "R6=not run"
echo  [M4 smoke] SHAP Random Forest, small test ... (log: 06 Outputs\jalsar\extensions\logs\m4_smoke.txt)
python -u "%~dp0m4_shap_rf.py" --smoke > "%LOGS%\m4_smoke.txt" 2>&1
if errorlevel 1 (set "R4S=FAIL" & goto m6) else (set "R4S=PASS")
echo  %time%  [M4 full] about 30 to 90 minutes ... (log: m4_full.txt)
python -u "%~dp0m4_shap_rf.py" > "%LOGS%\m4_full.txt" 2>&1
if errorlevel 1 (set "R4=FAIL") else (set "R4=PASS")
:m6
echo  %time%  [M6 smoke] PCA, small test, about 15 minutes ... (log: m6_smoke.txt)
python -u "%~dp0m6_pca.py" --smoke > "%LOGS%\m6_smoke.txt" 2>&1
if errorlevel 1 (set "R6S=FAIL" & goto done) else (set "R6S=PASS")
echo  %time%  [M6 full] about 1.5 to 3 hours ... (log: m6_full.txt)
python -u "%~dp0m6_pca.py" > "%LOGS%\m6_full.txt" 2>&1
if errorlevel 1 (set "R6=FAIL") else (set "R6=PASS")
:done
> "%LOGS%\OVERNIGHT_STATUS.txt" (
  echo finished %date% %time%
  echo M4 smoke: %R4S%
  echo M4 full:  %R4%
  echo M6 smoke: %R6S%
  echo M6 full:  %R6%
)
echo.
echo ================================================================
echo  OVERNIGHT DONE  %date% %time%
echo   M4 SHAP RF   smoke %R4S%, full %R4%
echo   M6 PCA       smoke %R6S%, full %R6%
echo   Downloads: see the other window, or logs\DOWNLOADS_STATUS.txt
echo  Check the folder 06 Outputs\jalsar\extensions\logs and the two
echo  summaries: jalsar\extensions\m4_shap_rf\M4_SHAP_RF.md, jalsar\extensions\m6_pca\M6_PCA.md
echo ================================================================
echo.
pause
