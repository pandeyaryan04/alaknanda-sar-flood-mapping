@echo off
rem Clip check: counts corridor cells with HAND or slope above the training
rem limit and scores the headline Jal-SAR map inside and outside them.
rem It only reads. Its report goes to 06 Outputs\checks\clip_check.md
rem Needs the Jal-SAR maps first (RUN JAL-SAR MAPS.bat). Takes about a minute.

chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

echo.
echo Clip check
echo.

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" "%USERPROFILE%\miniconda3" >nul 2>&1
call conda activate fyp
if errorlevel 1 goto noenv

python clip_check.py
if errorlevel 1 goto fail
echo.
echo CLIP CHECK PASS. Report: 06 Outputs\checks\clip_check.md
goto end
:fail
echo.
echo CLIP CHECK FAIL. Scroll up to the error.
goto end
:noenv
echo Could not activate the conda environment fyp.
:end
echo.
pause
