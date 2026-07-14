@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM  Memory Tier Benchmark - One-Click Full Launch
REM  1 provider x 4 scenarios x 100 samples x 3 rounds = 1200
REM  Steps: kill stale -> start phoenix+llamacpp -> bench -> stop
REM ============================================================

cd /d "%~dp0"

if not exist "build\tmp" mkdir "build\tmp"
set "TMP=%CD%\build\tmp"
set "TEMP=%CD%\build\tmp"

REM ---- Resolve Python ----
set "PYTHON_EXE=Python314\python.exe"
if not exist "%PYTHON_EXE%" (
  set "PYTHON_EXE=.venv\Scripts\python.exe"
  if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
)

REM ---- Check required files ----
if not exist "phoenix_main.exe" (
  echo [FATAL] phoenix_main.exe not found. Run compile.bat first.
  exit /b 1
)
if not exist "tools\run_memory_tier_benchmark_tui.py" (
  echo [FATAL] tools\run_memory_tier_benchmark_tui.py not found.
  exit /b 2
)
REM ---- STEP 1: Kill stale processes ----
echo [STEP] Killing stale processes...
taskkill /IM phoenix_main.exe /F >nul 2>&1
taskkill /IM bug_shooter.exe /F >nul 2>&1
for /f "tokens=5" %%P in ('netstat -ano 2^>nul ^| findstr ":5080 \|:8082 \|:8083 \|:8084 " ^| findstr "LISTENING"') do (
  taskkill /PID %%P /F >nul 2>&1
)
REM Clear stale pid file so phoenix does not fail on startup
if exist "runtime_store\phoenix_main.pid" del /f "runtime_store\phoenix_main.pid" >nul 2>&1
timeout /t 2 /nobreak >nul

REM ---- STEP 2: Run benchmark (TUI handles launching llama-server + phoenix internally) ----
REM   benchmark uses start_local_phoenix_stack which:
REM     - starts llama-server.exe on port 8083
REM     - starts phoenix_main.exe on port 5080 with --frontend-enabled=false
REM   so the correct phoenix chat URL is http://127.0.0.1:5080/api/chat (not 5081)
REM   auth token "local-dev" matches DEFAULT_PHOENIX_TOKEN
echo [STEP] Starting benchmark (TUI will auto-launch llama-server + phoenix)
echo [INFO] 4 scenarios x 100 samples x 3 rounds = 1200 total
set "PYTHONPATH=%CD%\tools;%PYTHONPATH%"

"%PYTHON_EXE%" -u tools\run_memory_tier_benchmark_tui.py ^
  --providers "phoenix" ^
  --scenarios "short_dialogue,long_dialogue_5_15,ultra_long_dialogue_15_plus,cross_session" ^
  --sample-per-scenario 100 ^
  --rounds 3 ^
  --context-window 4096 ^
  --timeout 90 ^
  --warmup-timeout 120 ^
  --warmup-retries 3 ^
  --fps 1 ^
  --stall-seconds 180 ^
  --phoenix-url "http://127.0.0.1:5080/api/chat" ^
  --phoenix-token "local-dev"

set "EXIT_CODE=%ERRORLEVEL%"

REM ---- STEP 3: Cleanup ----
echo [STEP] Stopping phoenix and llamacpp...
taskkill /IM phoenix_main.exe /F >nul 2>&1
taskkill /IM bug_shooter.exe /F >nul 2>&1
for /f "tokens=5" %%P in ('netstat -ano 2^>nul ^| findstr ":8083 " ^| findstr "LISTENING"') do (
  taskkill /PID %%P /F >nul 2>&1
)

echo.
if %EXIT_CODE% equ 0 (
  echo [OK] Benchmark complete.
  echo [OK] JSON: build\memory_tier_benchmark_v1.json
  echo [OK] MD:   build\memory_tier_benchmark_v1.md
) else (
  echo [FAIL] Benchmark failed (exit %EXIT_CODE%).
)
exit /b %EXIT_CODE%
