@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "PHOENIX_EXE=%ROOT%phoenix_main.exe"
if not exist "%PHOENIX_EXE%" set "PHOENIX_EXE=%ROOT%build\phoenix_main.exe"

set "PYTHON_EXE=%ROOT%Python314\python.exe"
if not exist "%PYTHON_EXE%" (
  echo [ERROR] missing local Python runtime: %PYTHON_EXE%
  exit /b 1
)

if /I "%FORCE_COMPILE%"=="1" (
  echo [STEP] force compile project
  call "%ROOT%compile.bat"
  if errorlevel 1 exit /b 1
) else if /I "%SKIP_COMPILE%"=="1" (
  echo [STEP] skip compile by SKIP_COMPILE=1
) else if exist "%PHOENIX_EXE%" (
  echo [STEP] reuse existing executable: %PHOENIX_EXE%
) else (
  echo [STEP] compile project
  call "%ROOT%compile.bat"
  if errorlevel 1 exit /b 1
)

if not exist "%PHOENIX_EXE%" (
  echo [ERROR] phoenix_main.exe not found. Place phoenix_main.exe in repo root or build\, or run with FORCE_COMPILE=1 on a machine that has the build toolchain.
  exit /b 1
)

if not defined PLAN_FILE set "PLAN_FILE=test\prof\offline_matrix_plan.json"
if not defined OUTPUT_DIR set "OUTPUT_DIR=build\offline_matrix"
if not defined OLLAMA_MODEL set "OLLAMA_MODEL=llama3.1:8b"
if not defined TESTS_DATASET_LIMIT set "TESTS_DATASET_LIMIT=24"
if not defined ROUNDS set "ROUNDS=1"
if not defined CONCURRENCY set "CONCURRENCY=1"
if not defined MAX_TOKENS set "MAX_TOKENS=160"
if not defined TIMEOUT_SEC set "TIMEOUT_SEC=120"
if not defined OLLAMA_WARMUP_TIMEOUT set "OLLAMA_WARMUP_TIMEOUT=240"
if /I "%STRICT_EXIT%"=="1" (set "STRICT_FLAG=--strict-exit") else (set "STRICT_FLAG=")

set "SUMMARY_MD=%OUTPUT_DIR%\offline_matrix_summary.md"
set "SUMMARY_JSON=%OUTPUT_DIR%\offline_matrix_summary.json"
set "SUMMARY_CSV=%OUTPUT_DIR%\offline_matrix_summary.csv"
set "SUMMARY_ALIAS_MD=%OUTPUT_DIR%_summary.md"
set "SUMMARY_ALIAS_JSON=%OUTPUT_DIR%_summary.json"
set "SUMMARY_ALIAS_CSV=%OUTPUT_DIR%_summary.csv"

echo [STEP] run offline benchmark matrix
"%PYTHON_EXE%" "test\prof\offline_matrix.py" ^
  --plan-file "%PLAN_FILE%" ^
  --output-dir "%OUTPUT_DIR%" ^
  --ollama-model "%OLLAMA_MODEL%" ^
  --tests-dataset-limit %TESTS_DATASET_LIMIT% ^
  --rounds %ROUNDS% ^
  --concurrency %CONCURRENCY% ^
  --max-tokens %MAX_TOKENS% ^
  --timeout %TIMEOUT_SEC% ^
  --ollama-warmup-timeout %OLLAMA_WARMUP_TIMEOUT% ^
  %STRICT_FLAG%

if errorlevel 1 (
  if exist "%SUMMARY_MD%" echo [INFO] summary markdown: %SUMMARY_MD%
  if exist "%SUMMARY_JSON%" echo [INFO] summary json: %SUMMARY_JSON%
  if exist "%SUMMARY_CSV%" echo [INFO] summary csv: %SUMMARY_CSV%
  if exist "%SUMMARY_ALIAS_MD%" echo [INFO] root summary markdown: %SUMMARY_ALIAS_MD%
  if exist "%SUMMARY_ALIAS_JSON%" echo [INFO] root summary json: %SUMMARY_ALIAS_JSON%
  if exist "%SUMMARY_ALIAS_CSV%" echo [INFO] root summary csv: %SUMMARY_ALIAS_CSV%
  echo [ERR] offline benchmark matrix finished with regressions or orchestration errors.
  exit /b 1
)

if exist "%SUMMARY_MD%" echo [OK] summary markdown: %SUMMARY_MD%
if exist "%SUMMARY_ALIAS_MD%" echo [OK] root summary markdown: %SUMMARY_ALIAS_MD%
if exist "%SUMMARY_ALIAS_CSV%" echo [OK] root summary csv: %SUMMARY_ALIAS_CSV%
echo [OK] offline benchmark matrix completed.
exit /b 0