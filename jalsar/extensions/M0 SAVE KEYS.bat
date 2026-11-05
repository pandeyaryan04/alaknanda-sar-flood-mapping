@echo off
setlocal
title M0 SAVE KEYS
set "CDS=%USERPROFILE%\.cdsapirc"
set "OCI=%USERPROFILE%\.opencellid_token"
set "BAD=0"
echo.
echo ================================================================
echo  Jal-SAR extensions, step M0: save your GloFAS key and OpenCelliD token
echo ================================================================
echo  Both go into your user folder (%USERPROFILE%),
echo  never into the project folder and never onto GitHub.
echo  This window never shows the key or the token.
echo.
echo ---------------- PART 1 of 2: GloFAS key ----------------
echo  1. Press any key. Notepad opens.
echo  2. Paste the two lines from the grey box on
echo     https://ewds.climate.copernicus.eu/how-to-api
echo        url: https://ewds.climate.copernicus.eu/api
echo        key: (your key)
echo  3. Press Ctrl + S, then close Notepad.
if exist "%CDS%" echo  NOTE: the file already exists. Replace everything in it with the two lines.
echo.
pause
if not exist "%CDS%" type nul > "%CDS%"
start "" /wait notepad "%CDS%"
findstr /b /c:"url: https://ewds.climate.copernicus.eu/api" "%CDS%" >nul
if errorlevel 1 (
  echo  FAIL: the url line is missing or different. It must be exactly:
  echo        url: https://ewds.climate.copernicus.eu/api
  set "BAD=1"
) else (
  echo  OK: url line found.
)
findstr /r /b /c:"key: .........." "%CDS%" >nul
if errorlevel 1 (
  echo  FAIL: the key line is missing or too short. It must start with  key:
  set "BAD=1"
) else (
  echo  OK: key line found.
)
echo.
echo ---------------- PART 2 of 2: OpenCelliD token ----------------
echo  1. Press any key. Notepad opens.
echo  2. Paste ONLY the token, on one line, nothing else.
echo  3. Press Ctrl + S, then close Notepad.
if exist "%OCI%" echo  NOTE: the file already exists. Replace everything in it with the token.
echo.
pause
if not exist "%OCI%" type nul > "%OCI%"
start "" /wait notepad "%OCI%"
for %%A in ("%OCI%") do set "SIZE=%%~zA"
if %SIZE% LSS 8 (
  echo  FAIL: the token file is empty or too short.
  set "BAD=1"
) else (
  echo  OK: token saved.
)
echo.
echo ================================================================
if "%BAD%"=="0" (
  echo  M0 KEYS PASS
) else (
  echo  M0 KEYS FAIL: double-click this file again and fix the line above.
)
echo ================================================================
echo.
pause
