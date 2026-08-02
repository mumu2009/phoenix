@echo off
chcp 65001 >nul
cd /d "%~dp0"

:: Run all four additive residual models on real data.
::
:: Audio:   MUSAN 16 kHz WAV files under Kali /home/kali/phoenix/datasets/musan_16k
:: Images:  Tiny-ImageNet-200 under Kali /home/kali/datasets/tiny-imagenet-200
:: Concept: BGE-small-en downloaded from ModelScope to /home/kali/models/bge-small-en
::
:: The text description of each sample is encoded by the LLM into a shared 128-D
:: Unit concept.  No teacher model is required.

python tools\run_all_additive_training.py ^
    --kali-host 192.168.0.100 ^
    --kali-user kali ^
    --kali-pass kali ^
    --x5-host 192.168.0.107 ^
    --x5-user sunrise ^
    --x5-pass sunrise ^
    --models speech_encoder,speech_decoder,vision_encoder,vision_decoder ^
    --real-data ^
    --speech-dataset /home/kali/phoenix/datasets/musan_16k ^
    --vision-image-dir /home/kali/datasets/tiny-imagenet-200 ^
    --bge-dir /home/kali/models/bge-small-en ^
    --work-dir /home/kali/phoenix/additive_work/real ^
    --x5-work /home/sunrise/phoenix/evolve_real ^
    --concept 128 ^
    --pool-size 2000 ^
    --max-rounds 3 ^
    --lambda 2 ^
    --batch-size 50 ^
    --parallel 1 ^
    --max-concurrent 1 ^
    --block-size small ^
    --wait ^
    %*

if %errorlevel% neq 0 (
    echo [ERROR] real training batch failed with code %errorlevel%
    exit /b %errorlevel%
)
