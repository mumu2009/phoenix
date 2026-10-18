@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

rem User steps:
rem 1. Put this bat in the extracted Odin root folder.
rem 2. Make sure phoenix_main.exe is next to this bat.
rem 3. Make sure GGUF_models is in the same folder.
rem 4. Run this bat and input 079.
rem Note: the repo runtime flag is "llamacpp".

echo.
if defined ODIN_INTERNAL_NAME (
    set "INTERNAL_NAME=%ODIN_INTERNAL_NAME%"
    echo 这个软件的内部名称是什么: %INTERNAL_NAME%
) else (
    set /p "INTERNAL_NAME=这个软件的内部名称是什么: "
)

if /I not "%INTERNAL_NAME%"=="079" (
    echo.
    echo [ERROR] Authorization failed.
    pause
    exit /b 1
)

if /I "%ODIN_ONECLICK_DRY_RUN%"=="1" (
    echo .\phoenix_main.exe --port=5080 --study-port=5081 --frontend-enabled=true --http-log=false --frontend-http-log=false --using-ollama=false --transformer-mode=llamacpp --external-auto-launch=true --llamacpp-base-url=http://127.0.0.1:8082 --gguf-models-dir=GGUF_models
    exit /b 0
)

taskkill /IM phoenix_main.exe /F >nul 2>&1
taskkill /IM bug_shooter.exe /F >nul 2>&1

.\phoenix_main.exe --port=5080 --study-port=5081 --frontend-enabled=true --http-log=false --frontend-http-log=false --using-ollama=false --transformer-mode=llamacpp --external-auto-launch=true --llamacpp-base-url=http://127.0.0.1:8082 --gguf-models-dir=GGUF_models