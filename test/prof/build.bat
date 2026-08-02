@echo off
setlocal enabledelayedexpansion

REM Build prof benchmark executable from test/prof/main.py
set "ROOT=%~dp0..\.."
pushd "%ROOT%" >nul

echo [1/3] Checking Python and dependencies...
py -3 -m pip show cython >nul 2>nul || (
  echo [INFO] Installing Cython...
  py -3 -m pip install cython || goto :fail
)
py -3 -m pip show pyinstaller >nul 2>nul || (
  echo [INFO] Installing PyInstaller...
  py -3 -m pip install pyinstaller || goto :fail
)

echo [2/3] Cython translate (requested format)...
py -3 -m cython --cplus -3 -X language_level=3 test/prof/main.py -o test/prof/main.cpp || goto :fail

echo [3/3] Packaging exe...
py -3 -m PyInstaller --onefile --name prof_bench test/prof/main.py || goto :fail

echo [OK] Build complete: %ROOT%\dist\prof_bench.exe
popd >nul
exit /b 0

:fail
echo [ERR] Build failed.
popd >nul
exit /b 1
