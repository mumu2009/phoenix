@echo off
chcp 65001 >nul
cd /d "%~dp0"

:: Same as run_all_real_training.bat but evaluates candidates on the Kali CPU
:: with ONNX Runtime.  The real BPU .bin files are still produced by the
:: hb_mapper on Kali and can be copied to the edge device later.

python tools\run_all_additive_training.py ^
    --kali-host 192.168.0.100 ^
    --kali-user kali ^
    --kali-pass kali ^
    --models speech_encoder,speech_decoder,vision_encoder,vision_decoder ^
    --real-data ^
    --speech-dataset /home/kali/phoenix/datasets/musan_16k ^
    --vision-image-dir /home/kali/datasets/tiny-imagenet-200 ^
    --bge-dir /home/kali/models/bge-small-en ^
    --work-dir /home/kali/phoenix/additive_work/real ^
    --concept 128 ^
    --pool-size 200 ^
    --max-rounds 3 ^
    --lambda 2 ^
    --batch-size 50 ^
    --parallel 1 ^
    --max-concurrent 1 ^
    --block-size small ^
    --eval-local ^
    --no-local-build ^
    --wait ^
    %*

if %errorlevel% neq 0 (
    echo [ERROR] real training batch failed with code %errorlevel%
    exit /b %errorlevel%
)
