@echo off
setlocal

cd /d "D:\_phoenix\_079\v6.0Alixander\phoenix"

set "MODEL=GGUF_models\blobs\sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29"
set "PORT=8082"

if not exist "%MODEL%" (
    echo [ERROR] 8B model blob not found: %MODEL%
    exit /b 1
)

echo [INFO] Starting llama-server on port %PORT% with 8B model: %MODEL%
outsides\llamacpp\build-gcc\bin\llama-server.exe -m "%MODEL%" --host 127.0.0.1 --port %PORT% -c 32768 -t 6 --parallel 1
