@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

:: Single concurrency knob: how many model evolutions run at the same time.
:: The per-model compile parallelism is auto-derived so the total number of
:: concurrent BPU compile jobs stays around 8 (safe for the Kali VM).
set "MAX_CONCURRENT=4"
set /a PARALLEL=8/MAX_CONCURRENT

:: Run all four additive residual models on real data, evaluating on a real edge device.
::
:: Audio:   MUSAN 16 kHz WAV files under Kali /home/kali/phoenix/datasets/musan_16k
:: Images:  Tiny-ImageNet-200 under Kali /home/kali/datasets/tiny-imagenet-200
:: Concept: BGE-small-en downloaded from ModelScope to /home/kali/models/bge-small-en
::
:: The edge device is read from config/edge_devices.json (gitignored).
:: Copy config/edge_devices.example.json to config/edge_devices.json and fill it in.
:: For password auth via env, copy config/edge_devices.env.example to
:: config/edge_devices.env and set PHOENIX_EDGE_PASS.

if exist "config\edge_devices.env" (
    for /f "usebackq tokens=1,* delims==" %%a in (`findstr /r /c:"^[^#].*=.*" "config\edge_devices.env"`) do (
        set "%%a=%%b"
    )
)

if not defined PHOENIX_EDGE_PASS (
    echo [WARN] PHOENIX_EDGE_PASS not set.  Add it to config\edge_devices.env or use --x5-pass.
)

python tools\run_all_additive_training.py ^
    --kali-host 192.168.0.100 ^
    --kali-user kali ^
    --kali-pass kali ^
    --edge-device lab_x5 ^
    --models speech_encoder,speech_decoder,vision_encoder,vision_decoder ^
    --real-data ^
    --speech-dataset /home/kali/phoenix/datasets/musan_16k ^
    --vision-image-dir /home/kali/datasets/tiny-imagenet-200 ^
    --bge-dir /home/kali/models/bge-small-en ^
    --work-dir /home/kali/phoenix/additive_work/real ^
    --concept 128 ^
    --pool-size 200 ^
    --max-rounds 1000 ^
    --lambda 10 ^
    --batch-size 500 ^
    --parallel %PARALLEL% ^
    --max-concurrent %MAX_CONCURRENT% ^
    --block-size small ^
    --no-local-build ^
    --wait ^
    --resume ^
    %*

if %errorlevel% neq 0 (
    echo [ERROR] real training batch failed with code %errorlevel%
    exit /b %errorlevel%
)
