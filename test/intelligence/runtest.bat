@echo off
setlocal

set "ROOT=%~dp0..\.."
pushd "%ROOT%" >nul

set "PY=%ROOT%\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%ROOT%\Python314\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" test\intelligence\main.py --system-url http://127.0.0.1:5080/api/chat --system-token local-dev --output-json build\intelligence_eval_report.json --output-md build\intelligence_eval_report.md --max-tokens 192 %*
set "EXITCODE=%ERRORLEVEL%"

popd >nul
exit /b %EXITCODE%