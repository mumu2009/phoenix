@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%~dp0Python314\python.exe"

if not exist "%PY%" (
    echo [ERROR] Python executable not found.
    exit /b 1
)

"%PY%" -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo [STEP] Installing PyInstaller
    "%PY%" -m pip install pyinstaller
    if errorlevel 1 exit /b 1
)

echo [STEP] Building start_079_oneclick.exe
"%PY%" -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --onefile ^
    --windowed ^
    --name start_079_oneclick ^
    --distpath "%~dp0dist" ^
    --workpath "%~dp0build\pyinstaller" ^
    --specpath "%~dp0build\pyinstaller" ^
    "%~dp0tools\start_079_launcher.py"
if errorlevel 1 exit /b 1

echo [SUCCESS] dist\start_079_oneclick.exe
exit /b 0