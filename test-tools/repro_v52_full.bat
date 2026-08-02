@echo off
setlocal enabledelayedexpansion

set "ROOT=%~dp0.."
pushd "%ROOT%" >nul
set "PYTHON=%ROOT%\.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
	echo [ERR] python not found: %PYTHON%
	popd >nul
	exit /b 1
)

set "STAMP=%DATE:~0,4%%DATE:~5,2%%DATE:~8,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%"
set "STAMP=%STAMP: =0%"
set "OUT_MD=build\perfReport_%STAMP%.md"
set "OUT_JSON=build\perfReport_%STAMP%.json"
set "OUT_PDF=build\perfReport_%STAMP%.pdf"
set "INTEL_MD=build\intelligence_eval_report_%STAMP%.md"
set "INTEL_JSON=build\intelligence_eval_report_%STAMP%.json"

echo [STEP] compile.bat
call compile.bat || goto :fail

echo [STEP] start phoenix service
start "phoenix_main" /B "%ROOT%\phoenix_main.exe"
set "PHOENIX_READY="
for /L %%I in (1,1,60) do (
	netstat -ano | findstr /R /C:":5080 .*LISTENING" >nul && (
		set "PHOENIX_READY=1"
		goto :phoenix_ready
	)
	timeout /t 1 /nobreak >nul
)

:phoenix_ready
if not defined PHOENIX_READY (
	echo [ERR] phoenix service failed to listen on 5080
	goto :fail
)

echo [STEP] intelligence evaluation
"%PYTHON%" test/intelligence/main.py --system-url http://127.0.0.1:5080/api/chat --system-token local-dev --cases-file test/intelligence/cases.baseline.json --output-md "%INTEL_MD%" --output-json "%INTEL_JSON%" || goto :fail

echo [STEP] full external benchmark
"%PYTHON%" test/prof/main.py --system-url http://127.0.0.1:5080/api/chat --system-token local-dev --ollama-url http://127.0.0.1:11434/api/chat --cases-file test/intelligence/cases.quick.json --external-prompts-file questionaire.txt --external-answers-file answer_20260217-175925-v2.0Multi.txt --external-index-file doc/external_dataset_index.json --external-limit 0 --questionnaire-file "" --questionnaire-limit 0 --preferred-models "llama3.1:8b" --rounds 1 --concurrency 1 --timeout 60 --max-tokens 96 --auto-manage-system --auto-manage-ollama --output "%OUT_MD%" --json-output "%OUT_JSON%" || goto :fail

echo [STEP] export pdf
"%PYTHON%" test/prof/export_pdf.py --benchmark-json "%OUT_JSON%" "%INTEL_JSON%" --output "%OUT_PDF%" --title "Odin v5.2 Repro Report %STAMP%" || goto :fail

echo [OK] done: %OUT_PDF%
popd >nul
exit /b 0

:fail
echo [ERR] repro_v52_full failed
popd >nul
exit /b 1