@echo off
chcp 65001 >nul
cd /d "%~dp0"

:: Default validation run with synthetic data for all 4 models.
:: Change the arguments below for real data or longer training.

python tools\run_all_additive_training.py ^
    --kali-host 192.168.0.100 ^
    --kali-user kali ^
    --kali-pass kali ^
    --x5-host 192.168.0.107 ^
    --x5-user sunrise ^
    --x5-pass sunrise ^
    --models speech_encoder,speech_decoder,vision_encoder,vision_decoder ^
    --max-rounds 3 ^
    --lambda 2 ^
    --batch-size 1000 ^
    --parallel 1 ^
    --max-concurrent 2 ^
    --synthetic ^
    --block-size small ^
    --wait ^
    %*

if %errorlevel% neq 0 (
    echo [ERROR] training batch failed with code %errorlevel%
    exit /b %errorlevel%
)
