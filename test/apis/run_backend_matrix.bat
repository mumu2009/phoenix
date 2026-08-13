@echo off
setlocal ENABLEDELAYEDEXPANSION

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\..\"
pushd "%REPO_ROOT%"

set "PY314=%REPO_ROOT%Python314\python.exe"
if not exist "%PY314%" (
    echo [ERROR] Python314 not found: "%PY314%"
    popd
    exit /b 2
)

set "VENV_DIR=%SCRIPT_DIR%.venv"
set "LOG_DIR=%SCRIPT_DIR%logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo [INFO] Using Python: "%PY314%"
"%PY314%" -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    popd
    exit /b 3
)

set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo [ERROR] venv python not found: "%VENV_PY%"
    popd
    exit /b 4
)

echo [INFO] Installing requirements
"%VENV_PY%" -m pip install -r "%SCRIPT_DIR%requirements.txt" > "%LOG_DIR%\backend_matrix_pip.log" 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to install requirements. See logs\backend_matrix_pip.log
    popd
    exit /b 5
)

set "BASE_URL=http://127.0.0.1:5081"
if not "%~1"=="" set "BASE_URL=%~1"

set "PHOENIX_EXE=%REPO_ROOT%phoenix_main.exe"
set "GATEWAY_PORT=5080"
set "STUDY_PORT=5081"
if not defined AI_OLLAMA_MODEL if defined OLLAMA_MODEL set "AI_OLLAMA_MODEL=%OLLAMA_MODEL%"
if not defined AI_OLLAMA_MODEL (
    set "FIRST_OLLAMA_MODEL="
    for /f "skip=1 tokens=1" %%M in ('ollama list 2^>nul') do (
        if not defined FIRST_OLLAMA_MODEL set "FIRST_OLLAMA_MODEL=%%M"
        if /I "%%M"=="llama3.1:8b" set "AI_OLLAMA_MODEL=%%M"
        if not defined AI_OLLAMA_MODEL if /I "%%M"=="llama3.1:7b" set "AI_OLLAMA_MODEL=%%M"
    )
    if not defined AI_OLLAMA_MODEL if defined FIRST_OLLAMA_MODEL set "AI_OLLAMA_MODEL=%FIRST_OLLAMA_MODEL%"
)
if not defined AI_OLLAMA_MODEL set "AI_OLLAMA_MODEL=llama3.1:8b"
if not defined AI_OLLAMA_BASE_URL if defined OLLAMA_HOST set "AI_OLLAMA_BASE_URL=%OLLAMA_HOST%"
if not defined AI_OLLAMA_KEEP_ALIVE if defined OLLAMA_KEEP_ALIVE set "AI_OLLAMA_KEEP_ALIVE=%OLLAMA_KEEP_ALIVE%"
if not defined AI_OLLAMA_TIMEOUT_MS set "AI_OLLAMA_TIMEOUT_MS=180000"

set "PHOENIX_ARGS=--port=%GATEWAY_PORT% --study-port=%STUDY_PORT% --frontend-enabled=true --http-log=false --frontend-http-log=false --using-ollama=false"
if not defined PHOENIX_LIGHT_MODE set "PHOENIX_LIGHT_MODE=true"
if /I "%PHOENIX_LIGHT_MODE%"=="true" set "PHOENIX_ARGS=%PHOENIX_ARGS% --disable-context-module=true --disable-gnn-module=true"
if defined AI_OLLAMA_MODEL set "PHOENIX_ARGS=%PHOENIX_ARGS% --ollama-model=%AI_OLLAMA_MODEL%"
if defined AI_OLLAMA_BASE_URL set "PHOENIX_ARGS=%PHOENIX_ARGS% --ollama-base-url=%AI_OLLAMA_BASE_URL%"
if defined AI_OLLAMA_TIMEOUT_MS set "PHOENIX_ARGS=%PHOENIX_ARGS% --ollama-timeout-ms=%AI_OLLAMA_TIMEOUT_MS%"
if defined AI_OLLAMA_KEEP_ALIVE set "PHOENIX_ARGS=%PHOENIX_ARGS% --ollama-keep-alive=%AI_OLLAMA_KEEP_ALIVE%"
if not exist "%PHOENIX_EXE%" (
    echo [ERROR] phoenix_main.exe not found: "%PHOENIX_EXE%"
    popd
    exit /b 6
)

set "LLAMA_SERVER_EXE=%REPO_ROOT%outsides\llamacpp\build-gcc\bin\llama-server.exe"
set "LLAMA_MODEL=%REPO_ROOT%GGUF_models\blobs\sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29"
set "TINYLLAMA_MODEL=%REPO_ROOT%GGUF_models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

echo [INFO] Cleaning stale phoenix_main.exe processes
taskkill /IM phoenix_main.exe /F >nul 2>&1
taskkill /IM llama-server.exe /F >nul 2>&1
taskkill /IM bitnet-server.exe /F >nul 2>&1
timeout /t 2 /nobreak >nul

if not exist "%LLAMA_SERVER_EXE%" (
    echo [WARN] llama-server.exe not found: "%LLAMA_SERVER_EXE%". Backend matrix will run without llama.cpp backend.
) else (
    echo [INFO] Starting llama-server.exe on 8082 with main model
    start "llama-server" /B "%LLAMA_SERVER_EXE%" -m "%LLAMA_MODEL%" --port 8082 --host 127.0.0.1 --ctx-size 4096 --parallel 1 -n 512 > "%LOG_DIR%\backend_matrix_llama_server.log" 2>&1

    echo [INFO] Starting tinyllama llama-server.exe on 8086
    start "tinyllama-server" /B "%LLAMA_SERVER_EXE%" -m "%TINYLLAMA_MODEL%" --port 8086 --host 127.0.0.1 --ctx-size 2048 --no-warmup > "%LOG_DIR%\backend_matrix_tinyllama_server.log" 2>&1

    echo [INFO] Waiting for llama-server health on 8082
    for /L %%I in (1,1,60) do (
        powershell -NoProfile -ExecutionPolicy Bypass -Command "$u='http://127.0.0.1:8082/health'; try { $r=Invoke-WebRequest -UseBasicParsing -Uri $u -TimeoutSec 2; if ($r.StatusCode -lt 500) { exit 0 } } catch { } exit 1" >nul 2>&1
        if !ERRORLEVEL! EQU 0 goto :llama_ready
        timeout /t 1 /nobreak >nul
    )
    :llama_ready
    echo [INFO] llama-server health check complete
)

echo [INFO] Starting phoenix_main.exe with documented args
echo [INFO] Command: phoenix_main.exe %PHOENIX_ARGS%
start "phoenix_main" /B "%PHOENIX_EXE%" %PHOENIX_ARGS% > "%LOG_DIR%\backend_matrix_phoenix_main.log" 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to start phoenix_main.exe
    popd
    exit /b 7
)

set "PHOENIX_SERVER_KILL_CMD=taskkill /IM phoenix_main.exe /F"
set "PHOENIX_SERVER_EXE=%PHOENIX_EXE%"
set "PHOENIX_SERVER_ARGS=%PHOENIX_ARGS%"
set "PHOENIX_SERVER_START_CMD=cmd /c start "" /B "%PHOENIX_EXE%" %PHOENIX_ARGS%"
set "PHOENIX_SERVER_CWD=%REPO_ROOT%"
set "PHOENIX_SERVER_READY_WAIT_SEC=15"

echo [INFO] Waiting for frontend service readiness on %BASE_URL%/auth/config
set "READY=0"
for /L %%I in (1,1,40) do (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$u='%BASE_URL%/auth/config'; try { $r=Invoke-WebRequest -UseBasicParsing -Uri $u -TimeoutSec 2; if ($r.StatusCode -lt 500) { exit 0 } else { exit 2 } } catch { exit 1 }" >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        set "READY=1"
        goto :ready_ok
    )
    timeout /t 1 /nobreak >nul
)

:ready_ok
if "%READY%"=="0" (
    echo [ERROR] phoenix_main frontend not ready in time. See "%LOG_DIR%\backend_matrix_phoenix_main.log"
    popd
    exit /b 8
)

echo [INFO] Running backend matrix tests against: %BASE_URL%
"%VENV_PY%" "%SCRIPT_DIR%backend_matrix.py" --base-url "%BASE_URL%" --log-file "%LOG_DIR%\backend_matrix.log" --report-file "%LOG_DIR%\backend_matrix_report.json"
set "EXITCODE=%ERRORLEVEL%"

echo [INFO] Logs: "%LOG_DIR%\backend_matrix.log"
echo [INFO] Report: "%LOG_DIR%\backend_matrix_report.json"

echo [INFO] Stopping backend matrix helper processes
taskkill /IM llama-server.exe /F >nul 2>&1
taskkill /IM phoenix_main.exe /F >nul 2>&1

popd
exit /b %EXITCODE%