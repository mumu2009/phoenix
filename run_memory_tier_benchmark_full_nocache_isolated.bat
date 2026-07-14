@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Run no-cache memory-tier benchmark with provider isolation and 1 FPS TUI.
REM Ollama is intentionally excluded.
REM Total planned conversations: 2 providers x 4 scenarios x 100 samples x 3 rounds = 2400.

cd /d "%~dp0"

set "PYTHON_EXE=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

set "TUI_SCRIPT=tools\run_memory_tier_benchmark_tui.py"
if not exist "%TUI_SCRIPT%" (
  echo [FATAL] Missing TUI runner script: %TUI_SCRIPT%
  exit /b 2
)

set "SAMPLES=100"
set "ROUNDS=3"
set "SCENARIOS=short_dialogue,long_dialogue_5_15,ultra_long_dialogue_15_plus,cross_session"
set "CONTEXT_WINDOW=4096"
set "TIMEOUT=90"
set "WARMUP_TIMEOUT=120"
set "WARMUP_RETRIES=3"
set "FPS=1"
set "STALL_SECONDS=180"

echo [INFO] Starting TUI benchmark runner...
echo [INFO] Providers: phoenix,llama_server
echo [INFO] Scenarios: %SCENARIOS%
echo [INFO] Sample per scenario: %SAMPLES%
echo [INFO] Rounds: %ROUNDS%
echo [INFO] Refresh rate: %FPS% FPS
echo [INFO] Stall threshold: %STALL_SECONDS% s

"%PYTHON_EXE%" -u "%TUI_SCRIPT%" ^
  --providers "phoenix,llama_server" ^
  --scenarios "%SCENARIOS%" ^
  --sample-per-scenario %SAMPLES% ^
  --rounds %ROUNDS% ^
  --context-window %CONTEXT_WINDOW% ^
  --timeout %TIMEOUT% ^
  --warmup-timeout %WARMUP_TIMEOUT% ^
  --warmup-retries %WARMUP_RETRIES% ^
  --fps %FPS% ^
  --stall-seconds %STALL_SECONDS%

exit /b %ERRORLEVEL%
