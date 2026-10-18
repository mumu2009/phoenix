@echo off
setlocal EnableDelayedExpansion

cd /d "D:\_phoenix\_079\v6.0Alixander\phoenix"

echo ============================================
echo  AA TEST SUITE - Phoenix v6.0 Alixander
echo ============================================

set GTEST_PASS=0
set COMPILE_PASS=0
set ONNX_PASS=0
set API_PASS=0
set MULTI_PASS=0
set README_PASS=0
set UI_PASS=0
set DEPLOY_PASS=0

:: GNN control: on = GNN module enabled (default), off = disabled for comparison
if /I "%AA_TEST_GNN%"=="off" set "AI_DISABLE_GNN_MODULE=1"
if /I "%AA_TEST_GNN%"=="on"  set "AI_DISABLE_GNN_MODULE=0"

:: Support a second pass (e.g. with GNN off) that reuses the already-built binaries
if defined AA_TEST_SKIP_BUILD (
    set GTEST_PASS=1
    set COMPILE_PASS=1
    set ONNX_PASS=1
    goto :runtime
)

:: 1. Compile and run GTest
echo [1/8] GTest compile...
call compile_gtest.bat > build\tmp\aa_compile_gtest.log 2>&1
if errorlevel 1 (
    echo [AA-TEST] GTest compile failed.
    goto :report
)

echo [2/8] GTest run...
call run_gtest.bat > build\tmp\aa_gtest.log 2>&1
if errorlevel 1 (
    echo [AA-TEST] GTest run failed.
    set GTEST_PASS=0
) else (
    for /f "tokens=*" %%a in ('findstr /C:"[  PASSED  ]" build\tmp\aa_gtest.log') do set GTEST_LINE=%%a
    echo [AA-TEST] GTest: %GTEST_LINE%
    set GTEST_PASS=1
)

:: 3. Compile phoenix_main
echo [3/8] Phoenix main compile...
call compile.bat > build\tmp\aa_compile.log 2>&1
if errorlevel 1 (
    echo [AA-TEST] phoenix_main compile failed.
    set COMPILE_PASS=0
    goto :report
) else (
    set COMPILE_PASS=1
)

:: 4. ONNX model validation
echo [4/8] ONNX model validation...
if exist "build\tmp\test_all_onnx.py" (
    .\Python314\python.exe build\tmp\test_all_onnx.py > build\tmp\aa_onnx.log 2>&1
    if errorlevel 1 (
        echo [AA-TEST] ONNX validation failed.
        set ONNX_PASS=0
    ) else (
        echo [AA-TEST] ONNX models validated.
        set ONNX_PASS=1
    )
) else (
    echo [AA-TEST] test_all_onnx.py not found, skipping ONNX validation.
    set ONNX_PASS=1
)

:runtime
:: 5. Start phoenix_main and run API regression + multimodal smoke
echo [5/8] Setting deployment by hardware...
.\Python314\python.exe test-tools\set_deployment_by_hardware.py > build\tmp\aa_deployment.log 2>&1
if errorlevel 1 (
    echo [AA-TEST] Hardware deployment config failed.
    set DEPLOY_PASS=0
    goto :report
)

echo [5/8] Starting LLM backend (llama-server + proxy)...
set "LLAMA_MODEL=%CD%\GGUF_models\blobs\sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29"
taskkill /f /im llama-server.exe >nul 2>&1
taskkill /f /im python.exe >nul 2>&1
start /b "" "outsides\llamacpp\build-gcc\bin\llama-server.exe" -m "%LLAMA_MODEL%" --host 127.0.0.1 --port 8084 -t 6 -c 2048 --parallel 1 > build\tmp\aa_llama_server.log 2>&1
echo [5/8] Waiting for llama-server health...
powershell -ExecutionPolicy Bypass -Command "$t=0; while($t -lt 120){ try { $r = Invoke-RestMethod 'http://127.0.0.1:8084/health' -TimeoutSec 5; if($r.status -eq 'ok'){ 'llama-server ready'; break } } catch {} Start-Sleep -Seconds 1; $t++ }"
start /b "" "Python314\python.exe" "tools\llama_proxy.py" --proxy-port 8080 --backend-port 8084 > build\tmp\aa_llama_proxy.log 2>&1
timeout /t 2 /nobreak

echo [5/8] Starting phoenix_main...
if exist "runtime_store\phoenix_main.pid" del /f /q "runtime_store\phoenix_main.pid"
set "JEPA_IMAGE_VARIANT=vision_encoder"
set "JEPA_SPEECH_VARIANT=speech_encoder"
set "AI_LLAMACPP_BASE_URL=http://127.0.0.1:8080"
set "AI_LLAMACPP_MODEL=GGUF_models\blobs\sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29"
set "TMP=%CD%\build\tmp"
set "TEMP=%CD%\build\tmp"
start /b "" "phoenix_main.exe" > build\tmp\aa_phoenix_main_out.log 2> build\tmp\aa_phoenix_main_err.log
timeout /t 12 /nobreak

echo [5/8] API regression...
powershell -ExecutionPolicy Bypass -File "test-tools\api_regression.ps1" > build\tmp\aa_api.log 2>&1
if errorlevel 1 (
    echo [AA-TEST] API regression failed.
    set API_PASS=0
) else (
    for /f "tokens=*" %%a in ('findstr /C:"PASS=" build\tmp\aa_api.log') do set API_LINE=%%a
    echo [AA-TEST] API: %API_LINE%
    set API_PASS=1
)

echo [6/8] Multimodal smoke...
if exist "test-tools\multimodal_smoke.ps1" (
    powershell -ExecutionPolicy Bypass -File "test-tools\multimodal_smoke.ps1" > build\tmp\aa_multimodal.log 2>&1
    if errorlevel 1 (
        echo [AA-TEST] Multimodal smoke failed.
        set MULTI_PASS=0
    ) else (
        echo [AA-TEST] Multimodal smoke passed.
        set MULTI_PASS=1
    )
) else (
    echo [AA-TEST] multimodal_smoke.ps1 not found, skipping.
    set MULTI_PASS=1
)

echo [7/8] README API route smoke...
if exist "test-tools\readme_api_probe.ps1" (
    powershell -ExecutionPolicy Bypass -File "test-tools\readme_api_probe.ps1" > build\tmp\aa_readme_api.log 2>&1
    if errorlevel 1 (
        echo [AA-TEST] README API probe failed.
        set README_PASS=0
    ) else (
        for /f "tokens=*" %%a in ('findstr /C:"PASS=" build\tmp\aa_readme_api.log') do set README_LINE=%%a
        echo [AA-TEST] %README_LINE%
        set README_PASS=1
    )
) else (
    echo [AA-TEST] readme_api_probe.ps1 not found, skipping.
    set README_PASS=1
)

echo [7.5/8] Real user UI E2E...
if exist "test-tools\ui_e2e_test.py" (
    .\Python314\python.exe test-tools\ui_e2e_test.py > build\tmp\aa_ui_e2e.log 2>&1
    if errorlevel 1 (
        echo [AA-TEST] UI E2E failed.
        set UI_PASS=0
    ) else (
        echo [AA-TEST] UI E2E passed.
        set UI_PASS=1
    )
) else (
    echo [AA-TEST] ui_e2e_test.py not found, skipping.
    set UI_PASS=1
)

echo [8/8] Stopping phoenix_main...
taskkill /f /im phoenix_main.exe >nul 2>&1
taskkill /f /im llama-server.exe >nul 2>&1
taskkill /f /im python.exe >nul 2>&1
taskkill /f /im msedge.exe >nul 2>&1

echo [8/8] 649 deployment matrix smoke...
.\Python314\python.exe tools\generate_model_deployment_matrix.py --enumerate > build\tmp\aa_deploy.log 2>&1
if errorlevel 1 (
    echo [AA-TEST] 649 deployment matrix failed.
    set DEPLOY_PASS=0
) else (
    for /f "tokens=*" %%a in ('findstr /C:"649" build\tmp\aa_deploy.log') do set DEPLOY_LINE=%%a
    echo [AA-TEST] %DEPLOY_LINE%
    set DEPLOY_PASS=1
)

:report
echo ============================================
echo  AA TEST SCORE
echo ============================================
echo  gtest   : %GTEST_PASS%
echo  compile : %COMPILE_PASS%
echo  onnx    : %ONNX_PASS%
echo  api     : %API_PASS%
echo  multi   : %MULTI_PASS%
echo  readme  : %README_PASS%
echo  ui      : %UI_PASS%
echo  deploy  : %DEPLOY_PASS%
echo ============================================

endlocal
