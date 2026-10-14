@echo off
rem Records the exact environments, the exact Earth Engine inputs, checks every
rem quoted number and writes a checksum of every data file into 04 Code\provenance\. Reads only; changes no data. Close QGIS first.
rem The checksum step reads tens of gigabytes: expect 15 to 40 minutes.

chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
set "PROV=%~dp0..\provenance"
if not exist "%PROV%" mkdir "%PROV%"

echo.
echo Provenance: environments, Earth Engine inputs, number audit, checksums
echo.

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" "%USERPROFILE%\miniconda3" >nul 2>&1
call conda activate fyp
if errorlevel 1 (
  echo Could not activate the conda environment 'fyp'.
  pause
  exit /b 1
)

set FAILS=0

echo ==== V4. Environments
call conda env export -n fyp --no-builds | findstr /v /b "prefix:" > "%PROV%\environment_fyp.yml"
if errorlevel 1 set /a FAILS+=1
call conda list -n fyp --explicit > "%PROV%\conda_explicit_fyp.txt"
call conda run -n fyp pip freeze > "%PROV%\pip_freeze_fyp.txt"
call conda env export -n fypdl --no-builds | findstr /v /b "prefix:" > "%PROV%\environment_fypdl.yml"
if errorlevel 1 set /a FAILS+=1
call conda list -n fypdl --explicit > "%PROV%\conda_explicit_fypdl.txt"
call conda run -n fypdl pip freeze > "%PROV%\pip_freeze_fypdl.txt"
nvidia-smi > "%PROV%\machine_gpu.txt" 2>&1
call conda run -n fypdl python -c "import platform, torch; print('python', platform.python_version()); print('os', platform.platform()); print('torch', torch.__version__); print('cuda', torch.version.cuda); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')" > "%PROV%\machine_fypdl.txt" 2>&1
for %%F in (environment_fyp.yml environment_fypdl.yml pip_freeze_fyp.txt pip_freeze_fypdl.txt machine_gpu.txt machine_fypdl.txt) do (
  if exist "%PROV%\%%F" (echo   written: %%F) else (echo   MISSING: %%F)
)
type "%PROV%\machine_fypdl.txt"

echo.
echo ==== V2. Exact Earth Engine inputs
python 17_earth_engine_inputs.py
if errorlevel 1 set /a FAILS+=1

echo.
echo ==== V7. Number audit
python 19_number_audit.py
if errorlevel 1 set /a FAILS+=1

echo.
echo ==== V3. Checksums of every data file (the long one)
python 18_data_manifest.py
if errorlevel 1 set /a FAILS+=1

echo.
echo ============================================================
if "%FAILS%"=="0" (
  echo ALL 4 PARTS DONE. Files are in 04 Code\provenance\ and 06 Outputs\number_audit\
) else (
  echo %FAILS% part^(s^) did NOT finish. Scroll up for the error.
)
echo ============================================================
echo.
pause
