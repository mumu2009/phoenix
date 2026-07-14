@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM Phoenix v6.0 Intelligent Test Suite
REM Covers: compile check + API regression + Python unit tests + frontend tests
REM Skips:  Intelligence / HAI / Perf (require llamacpp + GGUF model)

if not exist "build" mkdir "build"
if not exist "build\tmp" mkdir "build\tmp"
set "TMP=%CD%\build\tmp"
set "TEMP=%CD%\build\tmp"

echo [INFO] Starting intelligent test suite (skipping llamacpp backend tests)
echo [INFO] Report will be written to build\automation_report.json

powershell.exe -ExecutionPolicy Bypass -File "%~dp0test-tools\automation_suite.ps1" ^
    -BaseUrl "http://127.0.0.1:5080" ^
    -AuthBaseUrl "http://127.0.0.1:5081" ^
    -SkipIntelligence ^
    -SkipHai ^
    -SkipPerf

set "EXIT_CODE=%ERRORLEVEL%"

echo.
if %EXIT_CODE% equ 0 (
    echo [OK] All tests passed. Report: build\automation_report.json
) else (
    echo [FAIL] Tests failed (exit %EXIT_CODE%). Report: build\automation_report.json
)

exit /b %EXIT_CODE%
