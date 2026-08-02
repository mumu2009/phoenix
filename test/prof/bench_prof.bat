@echo off
setlocal enabledelayedexpansion

set "ROOT=%~dp0..\.."
pushd "%ROOT%" >nul

set "USE_EXE=0"
set "EXE=%ROOT%\dist\prof_bench.exe"
if "%USE_EXE%"=="1" (
  if not exist "%EXE%" (
    echo [INFO] prof_bench.exe not found, building first...
    call test\prof\build.bat || goto :fail
  )
)

REM ==== Required endpoints ====
set "SYSTEM_URL=http://127.0.0.1:5080/api/chat"
set "OLLAMA_URL=http://127.0.0.1:11434/api/chat"
set "CASES_FILE=test/intelligence/cases.baseline.json"
set "EXTERNAL_PROMPTS_FILE=questionaire.txt"
set "EXTERNAL_ANSWERS_FILE=answer_20260217-175925-v2.0Multi.txt"
set "EXTERNAL_INDEX_FILE=doc/external_dataset_index.json"
set "EXTERNAL_LIMIT=0"
set "EXTERNAL_THRESHOLD=5"
set "QUESTIONNAIRE_FILE="
set "QUESTIONNAIRE_LIMIT=0"

REM ==== Optional auth ====
REM If 5080 requires JWT, fill these with real user credentials.
set "SYSTEM_TOKEN=local-dev"
set "AUTH_USERNAME="
set "AUTH_PASSWORD="
set "LOGIN_URL=http://127.0.0.1:5081/auth/login"
set "REQUIRE_LOGIN=0"

REM ==== Benchmark params ====
set "ROUNDS=2"
set "CONCURRENCY=2"
set "TIMEOUT=60"
set "MAX_TOKENS=128"
set "PREFERRED_MODELS=llama3.1:8b,qwen2.5:7b,qwen2.5:14b,tinyllama:latest,gpt-oss:20b"
set "OLLAMA_MODEL="

set "STAMP=%DATE:~0,4%%DATE:~5,2%%DATE:~8,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%"
set "STAMP=%STAMP: =0%"
set "OUT_DIR=test/prof/reports/%STAMP%"
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

echo [INFO] Output dir: %OUT_DIR%

set "OUT_FILE=%OUT_DIR%/benchmark_report.md"
set "JSON_FILE=%OUT_DIR%/benchmark_report.json"

echo.
echo [RUN] integrated benchmark with auto-managed Ollama

if "%USE_EXE%"=="1" (
  echo [WARN] prof_bench.exe path is not updated for the new Python-only integrated benchmark.
  goto :fail
) else (
  if "%REQUIRE_LOGIN%"=="1" (
    py -3 test/prof/main.py --system-url "%SYSTEM_URL%" --system-token "%SYSTEM_TOKEN%" --ollama-url "%OLLAMA_URL%" --ollama-model "%OLLAMA_MODEL%" --preferred-models "%PREFERRED_MODELS%" --cases-file "%CASES_FILE%" --external-prompts-file "%EXTERNAL_PROMPTS_FILE%" --external-answers-file "%EXTERNAL_ANSWERS_FILE%" --external-index-file "%EXTERNAL_INDEX_FILE%" --external-limit %EXTERNAL_LIMIT% --external-threshold %EXTERNAL_THRESHOLD% --questionnaire-file "%QUESTIONNAIRE_FILE%" --questionnaire-limit %QUESTIONNAIRE_LIMIT% --rounds %ROUNDS% --concurrency %CONCURRENCY% --timeout %TIMEOUT% --max-tokens %MAX_TOKENS% --auth-username "%AUTH_USERNAME%" --auth-password "%AUTH_PASSWORD%" --login-url "%LOGIN_URL%" --require-login --auto-manage-system --auto-manage-ollama --output "%OUT_FILE%" --json-output "%JSON_FILE%" || goto :fail
  ) else (
    py -3 test/prof/main.py --system-url "%SYSTEM_URL%" --system-token "%SYSTEM_TOKEN%" --ollama-url "%OLLAMA_URL%" --ollama-model "%OLLAMA_MODEL%" --preferred-models "%PREFERRED_MODELS%" --cases-file "%CASES_FILE%" --external-prompts-file "%EXTERNAL_PROMPTS_FILE%" --external-answers-file "%EXTERNAL_ANSWERS_FILE%" --external-index-file "%EXTERNAL_INDEX_FILE%" --external-limit %EXTERNAL_LIMIT% --external-threshold %EXTERNAL_THRESHOLD% --questionnaire-file "%QUESTIONNAIRE_FILE%" --questionnaire-limit %QUESTIONNAIRE_LIMIT% --rounds %ROUNDS% --concurrency %CONCURRENCY% --timeout %TIMEOUT% --max-tokens %MAX_TOKENS% --auth-username "%AUTH_USERNAME%" --auth-password "%AUTH_PASSWORD%" --login-url "%LOGIN_URL%" --auto-manage-system --auto-manage-ollama --output "%OUT_FILE%" --json-output "%JSON_FILE%" || goto :fail
  )
)

echo.
echo [OK] Integrated benchmark finished. Reports in: %OUT_DIR%
popd >nul
exit /b 0

:fail
echo [ERR] Batch benchmark failed.
popd >nul
exit /b 1
