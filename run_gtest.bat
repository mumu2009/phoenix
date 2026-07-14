@echo off
REM run_gtest.bat - Run GTest test suite
REM Copyright (C) 2026 079 Project

setlocal enabledelayedexpansion

echo ========================================
echo GTest Test Runner
echo ========================================
echo.

REM Check if GTest executable exists
if not exist "gtest_runner.exe" (
    echo ERROR: gtest_runner.exe not found.
    echo Please run compile_gtest.bat first to build the test suite.
    exit /b 1
)

REM Set environment variables
set TMP=build\tmp
set TEMP=build\tmp
if not exist "%TMP%" mkdir "%TMP%"

echo Running GTest test suite...
echo.

REM Run GTest with various options
REM --gtest_filter: Filter tests (e.g., --gtest_filter=EmotionSystem*)
REM --gtest_repeat: Repeat tests N times
REM --gtest_break_on_failure: Stop on first failure
REM --gtest_catch_exceptions: Catch exceptions (default on Windows)

REM Run all tests
gtest_runner.exe --gtest_color=yes

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo All tests passed!
    echo ========================================
) else (
    echo.
    echo ========================================
    echo Some tests failed.
    echo ========================================
)

endlocal
exit /b %ERRORLEVEL%
