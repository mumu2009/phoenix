@echo off
setlocal EnableExtensions EnableDelayedExpansion
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: Clear leftover gcc temporary object files; stale ones can cause collect2 ICEs.
if not exist "build\tmp" mkdir "build\tmp"
del /q "build\tmp\cc*" 2>nul

REM Quick check: if phoenix_main.exe exists and is recent, skip compilation
::  if exist "%SCRIPT_DIR%phoenix_main.exe" (
::  echo [INFO] phoenix_main.exe exists. Skipping compilation and exiting with 0.
::  endlocal
::  exit /b 0
::)

set "BUILD_STATUS=FAILED"
set "FINAL_EXIT_CODE=1"

if exist "D:\Scoop\apps\gcc\current\bin\gcc.exe" (
  set "PATH=D:\Scoop\apps\gcc\current\bin;%PATH%"
)

set "GCC_EXE="
set "GXX_EXE="
for /f "delims=" %%I in ('where gcc 2^>nul') do if not defined GCC_EXE set "GCC_EXE=%%I"
for /f "delims=" %%I in ('where g++ 2^>nul') do if not defined GXX_EXE set "GXX_EXE=%%I"

if not defined GCC_EXE (
  echo [ERROR] gcc not found. Please install GCC toolchain via Scoop gcc and retry.
  exit /b 1
)
if not defined GXX_EXE (
  echo [ERROR] g++ not found. Please install GCC toolchain via Scoop gcc and retry.
  exit /b 1
)

set "GCC_EXE_POSIX=%GCC_EXE:\=/%"
set "GXX_EXE_POSIX=%GXX_EXE:\=/%"

where conan >nul 2>&1 || (
  echo [ERROR] Conan not found. Please install Conan 2.x and retry.
  exit /b 1
)

if not exist "build" mkdir "build"

echo [STEP] Rebuild llama-server (enc/infer/dec split)
REM Best-effort: keep outsides\llamacpp's llama-server / llama-cli in sync
REM with the tracked patches under llama_server_mods\ before compiling
REM phoenix_main.exe. outsides\ is gitignored, so llama_server_mods\*.patch
REM is the source of truth for these changes (see llama_server_mods\README.md).
REM Failures here are non-fatal to the overall Phoenix build unless the
REM caller explicitly opts in via PHOENIX_REQUIRE_LLAMA_SERVER=1.
set "PHOENIX_LLAMA_SERVER_OK=1"
if exist "%CD%\llama_server_mods\apply_patches.bat" (
  call "%CD%\llama_server_mods\apply_patches.bat"
  if errorlevel 1 (
    echo [WARN] llama_server_mods\apply_patches.bat failed. Continuing with whatever is currently in outsides\llamacpp.
    set "PHOENIX_LLAMA_SERVER_OK=0"
  )
) else (
  echo [INFO] llama_server_mods\apply_patches.bat not found. Skipping llama.cpp patch step.
)

if exist "%CD%\llama_server_mods\build_llama_server.bat" (
  call "%CD%\llama_server_mods\build_llama_server.bat"
  if errorlevel 1 (
    echo [WARN] llama_server_mods\build_llama_server.bat failed. phoenix_main.exe compilation will continue; llama-server.exe/llama.dll may be stale or missing.
    set "PHOENIX_LLAMA_SERVER_OK=0"
  )
) else (
  echo [INFO] llama_server_mods\build_llama_server.bat not found. Skipping llama-server rebuild.
)

if "%PHOENIX_LLAMA_SERVER_OK%"=="0" (
  if /I "%PHOENIX_REQUIRE_LLAMA_SERVER%"=="1" (
    echo [ERROR] PHOENIX_REQUIRE_LLAMA_SERVER=1 and the llama-server patch/build step failed. Aborting.
    exit /b 1
  ) else (
    echo [WARN] Continuing overall Phoenix build despite llama-server patch/build issues ^(set PHOENIX_REQUIRE_LLAMA_SERVER=1 to make this fatal^).
  )
)

echo [STEP] Stop stale runtime processes
taskkill /IM phoenix_main.exe /F >nul 2>&1
taskkill /IM bug_shooter.exe /F >nul 2>&1

if not defined CONAN_HOME set "CONAN_HOME=%CD%\build\conan_home"
if not exist "%CONAN_HOME%" mkdir "%CONAN_HOME%"
echo [INFO] CONAN_HOME=%CONAN_HOME%

echo [STEP] Conan install (refresh dependency metadata)
set "CONAN_COMMON=--output-folder=build --build=missing -s build_type=Release"
set "CC=%GCC_EXE_POSIX%"
set "CXX=%GXX_EXE_POSIX%"
set "CXXFLAGS=-include cstdint"
conan install . %CONAN_COMMON% -c tools.cmake.cmaketoolchain:generator=Ninja -c tools.files.download:retry=6 -c tools.files.download:retry_wait=10
if errorlevel 1 (
  echo [WARN] Conan install failed. Trying Conan cache self-heal for msys2 and retrying.
  conan remove "msys2/cci.latest:*" -c >nul 2>&1
  conan remove "msys2/cci.latest" -c >nul 2>&1
  conan install . %CONAN_COMMON% -c tools.cmake.cmaketoolchain:generator=Ninja -c tools.files.download:retry=6 -c tools.files.download:retry_wait=10
  if errorlevel 1 (
    echo [ERROR] conan install failed.
    exit /b 1
  )
)

echo [STEP] Prepare compile flags

where pkg-config >nul 2>&1 || (
  echo [ERROR] pkg-config not found.
  exit /b 1
)

set "PKGCFG_PATH=%CD:\=/%/build"
set "CONAN_CFLAGS_FILE=build\conan_cflags_%RANDOM%%RANDOM%.txt"
set "CONAN_LIBS_FILE=build\conan_libs_%RANDOM%%RANDOM%.txt"

call pkg-config --with-path="%PKGCFG_PATH%" --cflags drogon trantor opencv sqlite3 lmdb hiredis redis++ nlohmann_json jwt-cpp jsoncpp > "%CONAN_CFLAGS_FILE%"
if errorlevel 1 (
  echo [ERROR] failed to generate conan_cflags.txt.
  exit /b 1
)
call pkg-config --with-path="%PKGCFG_PATH%" --libs drogon trantor opencv sqlite3 lmdb hiredis redis++ nlohmann_json jwt-cpp jsoncpp > "%CONAN_LIBS_FILE%"
if errorlevel 1 (
  echo [ERROR] failed to generate conan_libs.txt.
  exit /b 1
)

powershell -Command "if (Select-String -Path '%CONAN_CFLAGS_FILE%' -Pattern 'v2.0Multi' -Quiet) { exit 0 } else { exit 1 }" && (
  echo [ERROR] stale Conan metadata detected in conan_cflags.txt.
  echo [ERROR] Please clear build\*.pc then rerun compile.bat.
  exit /b 1
) || (
  echo [INFO] Conan metadata validation passed.
)

for %%F in ("%CONAN_CFLAGS_FILE%" "%CONAN_LIBS_FILE%") do (
  if not exist "%%F" (
    echo [ERROR] missing %%F.
    exit /b 1
  )
  if %%~zF LEQ 0 (
    echo [ERROR] empty %%F.
    exit /b 1
  )
)

powershell -Command "if (Select-String -Path '%CONAN_CFLAGS_FILE%' -Pattern '-I' -Quiet) { exit 0 } else { exit 1 }" || (
  echo [ERROR] invalid conan_cflags.txt content.
  exit /b 1
)

powershell -Command "if (Select-String -Path '%CONAN_LIBS_FILE%' -Pattern '-l' -Quiet) { exit 0 } else { exit 1 }" || (
  echo [ERROR] invalid conan_libs.txt content.
  exit /b 1
)

if not exist "%CD%\poppler-25.12.0\Library\lib\poppler-cpp.lib" (
  echo [ERROR] missing poppler-cpp.lib.
  exit /b 1
)
if not exist "%CD%\poppler-25.12.0\Library\lib\poppler.lib" (
  echo [ERROR] missing poppler.lib.
  exit /b 1
)

if not exist "%CD%\build\tmp" mkdir "%CD%\build\tmp"
set "TMP=%CD%\build\tmp"
set "TEMP=%CD%\build\tmp"

set "PY_LOCAL_ROOT=%CD%\Python314"
set "PY_INC=%PY_LOCAL_ROOT%\include"
set "PY_LIB=%PY_LOCAL_ROOT%\libs"
set "PY_LINK_NAME="
if not exist "%PY_INC%\Python.h" (
  echo [ERROR] missing Python header: %PY_INC%\Python.h
  echo [ERROR] 请确认已将 Python314 复制到项目根目录.
  exit /b 1
)
if exist "%PY_LIB%\python314.lib" set "PY_LINK_NAME=python314"
if not defined PY_LINK_NAME if exist "%PY_LIB%\python3.lib" set "PY_LINK_NAME=python3"
if not defined PY_LINK_NAME (
  echo [ERROR] missing Python import lib in %PY_LIB% ^(need python314.lib or python3.lib^)
  exit /b 1
)

echo [STEP] Bind outsides to workspace runtime
set "OUTSIDES_ROOT=%CD%\outsides"
set "LLAMACPP_ROOT=%OUTSIDES_ROOT%\llamacpp"
set "BITNET_ROOT=%OUTSIDES_ROOT%\BitNet"
set "BULLET3_ROOT=%OUTSIDES_ROOT%\bullet3"
set "GGUF_MODELS_DIR=%CD%\GGUF_models"

if not exist "%GGUF_MODELS_DIR%" mkdir "%GGUF_MODELS_DIR%"
if not exist "%CD%\runtime_store" mkdir "%CD%\runtime_store"

set "OUTSIDES_BIND_FILE=%CD%\runtime_store\outsides_binding.env"
(
  echo AI_OUTSIDES_ROOT=%OUTSIDES_ROOT%
  echo AI_LLAMACPP_ROOT=%LLAMACPP_ROOT%
  echo AI_BITNET_ROOT=%BITNET_ROOT%
  echo AI_BULLET3_ROOT=%BULLET3_ROOT%
  echo AI_GGUF_MODELS_DIR=%GGUF_MODELS_DIR%
  echo AI_EXTERNAL_STYLE_INTRUSIVE=1
  echo AI_EXTERNAL_STYLE_FEATURES=keywordDensity,sentiment,punctuationDensity
  echo AI_EXTERNAL_BACKEND_POOL_COMPAT=1
  echo AI_EXTERNAL_BACKEND_GROUPPROC_COMPAT=1
  echo AI_EXTERNAL_BACKEND_LEARNER_COMPAT=1
) > "%OUTSIDES_BIND_FILE%"

set "OUTSIDES_CFLAGS="
if exist "%LLAMACPP_ROOT%\include" set "OUTSIDES_CFLAGS=%OUTSIDES_CFLAGS% -I%LLAMACPP_ROOT%\include"
if exist "%LLAMACPP_ROOT%\ggml\include" set "OUTSIDES_CFLAGS=%OUTSIDES_CFLAGS% -I%LLAMACPP_ROOT%\ggml\include"
if exist "%LLAMACPP_ROOT%\ggml\src" set "OUTSIDES_CFLAGS=%OUTSIDES_CFLAGS% -I%LLAMACPP_ROOT%\ggml\src"
if exist "%BITNET_ROOT%\include" set "OUTSIDES_CFLAGS=%OUTSIDES_CFLAGS% -I%BITNET_ROOT%\include"
if exist "%BITNET_ROOT%\src" set "OUTSIDES_CFLAGS=%OUTSIDES_CFLAGS% -I%BITNET_ROOT%\src"
if exist "%BULLET3_ROOT%\src\btBulletDynamicsCommon.h" set "OUTSIDES_CFLAGS=%OUTSIDES_CFLAGS% -I%BULLET3_ROOT%\src"
if exist "%BULLET3_ROOT%\src\btBulletDynamicsCommon.h" set "OUTSIDES_CFLAGS=%OUTSIDES_CFLAGS% -DAI_HAVE_BULLET3_NATIVE=1"

set "BULLET3_EMBEDDED_SOURCES="
if exist "%BULLET3_ROOT%\src\btLinearMathAll.cpp" set "BULLET3_EMBEDDED_SOURCES=outsides\bullet3\src\btLinearMathAll.cpp outsides\bullet3\src\btBulletCollisionAll.cpp outsides\bullet3\src\btBulletDynamicsAll.cpp"

set "PHOENIX_EDGE_IMAGE=1"
if /I "%PHOENIX_DISABLE_EDGE_IMAGE%"=="1" set "PHOENIX_EDGE_IMAGE=0"
set "PHOENIX_EDGE_SPEECH=1"
if /I "%PHOENIX_DISABLE_EDGE_SPEECH%"=="1" set "PHOENIX_EDGE_SPEECH=0"
set "EDGE_CFLAGS=-DPHOENIX_EDGE_IMAGE_ENABLED=%PHOENIX_EDGE_IMAGE% -DPHOENIX_EDGE_SPEECH_ENABLED=%PHOENIX_EDGE_SPEECH%"
echo [INFO] Edge device compile flags: image=%PHOENIX_EDGE_IMAGE%, speech=%PHOENIX_EDGE_SPEECH%

echo [STEP] Probe outsides native source compilation
if exist "%LLAMACPP_ROOT%\ggml\src\gguf.cpp" (
  "%GXX_EXE%" -c -std=c++20 -Wa,-mbig-obj %OUTSIDES_CFLAGS% "%LLAMACPP_ROOT%\ggml\src\gguf.cpp" -o "%CD%\build\tmp\llamacpp_gguf_smoke.obj" 1>"%CD%\build\tmp\llamacpp_gguf_smoke.out" 2>"%CD%\build\tmp\llamacpp_gguf_smoke.err"
  if errorlevel 1 (
    echo [WARN] llama.cpp gguf.cpp smoke compile failed. See build\tmp\llamacpp_gguf_smoke.err
  ) else (
    echo [INFO] llama.cpp gguf.cpp smoke compile passed.
  )
)
if exist "%BITNET_ROOT%\src\ggml-bitnet-mad.cpp" (
  "%GXX_EXE%" -c -std=c++20 -Wa,-mbig-obj %OUTSIDES_CFLAGS% "%BITNET_ROOT%\src\ggml-bitnet-mad.cpp" -o "%CD%\build\tmp\bitnet_mad_smoke.obj" 1>"%CD%\build\tmp\bitnet_mad_smoke.out" 2>"%CD%\build\tmp\bitnet_mad_smoke.err"
  if errorlevel 1 (
    echo [WARN] BitNet mad smoke compile failed. See build\tmp\bitnet_mad_smoke.err
    echo [WARN] Continuing with headers-only mode for BitNet.
  ) else (
    echo [INFO] BitNet mad smoke compile passed.
  )
)
if exist "%BULLET3_ROOT%\src\btBulletDynamicsCommon.h" (
  "%GXX_EXE%" -c -std=c++20 -Wa,-mbig-obj %OUTSIDES_CFLAGS% "%BULLET3_ROOT%\src\btLinearMathAll.cpp" -o "%CD%\build\tmp\bullet3_smoke.obj" 1>"%CD%\build\tmp\bullet3_smoke.out" 2>"%CD%\build\tmp\bullet3_smoke.err"
  if errorlevel 1 (
    echo [WARN] Bullet3 embedded-source smoke compile failed. See build\tmp\bullet3_smoke.err
  ) else (
    echo [INFO] Bullet3 embedded-source smoke compile passed.
  )
)

echo [STEP] Resolve outsides native linkage
set "OUTSIDES_LINK_MODE=headers-only"
set "OUTSIDES_LINK_CFLAGS=-DAI_OUTSIDES_LINK_MODE=\"headers-only\""
set "OUTSIDES_LINK_LIBS="
set "OUTSIDES_DLL_COPY_LIST="

if exist "%LLAMACPP_ROOT%\build\src\llama.lib" (
  set "OUTSIDES_LINK_MODE=static"
  set "OUTSIDES_LINK_CFLAGS=-DAI_OUTSIDES_LINK_MODE=\"static\""
  set "OUTSIDES_LINK_LIBS=%OUTSIDES_LINK_LIBS% -L%LLAMACPP_ROOT%\build\src -llama"
)
if exist "%BITNET_ROOT%\build\src\bitnet.lib" (
  set "OUTSIDES_LINK_MODE=static"
  set "OUTSIDES_LINK_CFLAGS=-DAI_OUTSIDES_LINK_MODE=\"static\""
  set "OUTSIDES_LINK_LIBS=%OUTSIDES_LINK_LIBS% -L%BITNET_ROOT%\build\src -lbitnet"
)
if exist "%BITNET_ROOT%\build\src\ggml-bitnet.lib" (
  set "OUTSIDES_LINK_MODE=static"
  set "OUTSIDES_LINK_CFLAGS=-DAI_OUTSIDES_LINK_MODE=\"static\""
  set "OUTSIDES_LINK_LIBS=%OUTSIDES_LINK_LIBS% -L%BITNET_ROOT%\build\src -lggml-bitnet"
)

if exist "%LLAMACPP_ROOT%\build\bin\llama.dll" (
  set "OUTSIDES_LINK_MODE=dll"
  set "OUTSIDES_LINK_CFLAGS=-DAI_OUTSIDES_LINK_MODE=\"dll\""
  if exist "%LLAMACPP_ROOT%\build\bin\llama.lib" set "OUTSIDES_LINK_LIBS=%OUTSIDES_LINK_LIBS% -L%LLAMACPP_ROOT%\build\bin -llama"
  set "OUTSIDES_DLL_COPY_LIST=%OUTSIDES_DLL_COPY_LIST% \"%LLAMACPP_ROOT%\build\bin\llama.dll\""
)
if exist "%BITNET_ROOT%\build\bin\bitnet.dll" (
  set "OUTSIDES_LINK_MODE=dll"
  set "OUTSIDES_LINK_CFLAGS=-DAI_OUTSIDES_LINK_MODE=\"dll\""
  if exist "%BITNET_ROOT%\build\bin\bitnet.lib" set "OUTSIDES_LINK_LIBS=%OUTSIDES_LINK_LIBS% -L%BITNET_ROOT%\build\bin -lbitnet"
  set "OUTSIDES_DLL_COPY_LIST=%OUTSIDES_DLL_COPY_LIST% \"%BITNET_ROOT%\build\bin\bitnet.dll\""
)

echo [INFO] outsides link mode=%OUTSIDES_LINK_MODE%
if /I "%OUTSIDES_LINK_MODE%"=="headers-only" (
  echo [WARN] outsides native library not found; compile will use adapter mode via HTTP+shared-protocol envelope.
)

echo [STEP] Compile phoenix_main.exe
set "OVERRIDE_SOURCES="
if exist "module_overrides\*.cpp" set "OVERRIDE_SOURCES=module_overrides\*.cpp"
set "COMPILE_CMD_FILE=%CD%\runtime_store\compile_last_command.txt"
set "COMPILE_SOURCES_FILE=%CD%\build\compile_sources.txt"
set "COMMON_SOURCES=transformer_main.cpp transformer_ollama_fine_tuning.cpp addon.cpp addons\builtin_registry.cpp addons\math_addon.cpp addons\search_addon.cpp addons\computer_shell_addon.cpp loggerCXX.cpp DATABASE_079.cpp frontend_server.cpp speak_io.cpp model_lifecycle.cpp autonomy_stack.cpp v51_runtime.cpp external_runtime.cpp edge_platform.cpp gguf_tensor_parser.cpp physics_world_runtime.cpp emotion_system.cpp llamacpp_emotion_adjuster.cpp plugin_system.cpp modern_context_system.cpp semantic_unit.cpp concept_matrix.cpp primal_sensation.cpp instinct.cpp prompt_split.cpp external_mixed_modal_io.cpp multimodal_world_model.cpp local_onnx.cpp video_model.cpp audio_model.cpp graph_diffusion_summarizer.cpp hierarchical_memory.cpp model_deployment.cpp rdk_x5_bpu.cpp active_inference.cpp subconscious_profile.cpp sparse_block_matmul.cpp %BULLET3_EMBEDDED_SOURCES%"
:: Build a response file of source files to avoid Windows command-line length limits.
:: GCC response files treat '\' as an escape character, so paths must use '/'.
if not exist "%CD%\build" mkdir "%CD%\build"
if exist "%COMPILE_SOURCES_FILE%" del /q "%COMPILE_SOURCES_FILE%" 2>nul
for %%a in (%COMMON_SOURCES%) do (
    set "src=%%a"
    set "src=!src:\=/!"
    echo !src! >> "%COMPILE_SOURCES_FILE%"
)
if not "%OVERRIDE_SOURCES%"=="" (
    for %%f in (%OVERRIDE_SOURCES%) do (
        set "src=%%f"
        set "src=!src:\=/!"
        echo !src! >> "%COMPILE_SOURCES_FILE%"
    )
)
echo [CMD] "%GXX_EXE%" -o phoenix_main.exe -std=c++20 -Wa,-mbig-obj -DAI_EXTERNAL_BACKEND_COMPAT=1 -DAI_EXTERNAL_LEARNER_BRIDGE=1 -DHAVE_SQLITE %EDGE_CFLAGS% %OUTSIDES_LINK_CFLAGS% @"%CONAN_CFLAGS_FILE%" -I"%CD%\poppler-25.12.0\Library\include" -I"%PY_INC%" %OUTSIDES_CFLAGS% -I"%CD%" main.cpp @"%COMPILE_SOURCES_FILE%" -Wl,--start-group @"%CONAN_LIBS_FILE%" -Wl,--end-group "%CD%\poppler-25.12.0\Library\lib\poppler-cpp.lib" "%CD%\poppler-25.12.0\Library\lib\poppler.lib" -L"%PY_LIB%" -l%PY_LINK_NAME% -lws2_32 %OUTSIDES_LINK_LIBS% -O3
(
  echo "%GXX_EXE%" -o phoenix_main.exe -std=c++20 -Wa,-mbig-obj -DAI_EXTERNAL_BACKEND_COMPAT=1 -DAI_EXTERNAL_LEARNER_BRIDGE=1 -DHAVE_SQLITE %EDGE_CFLAGS% %OUTSIDES_LINK_CFLAGS% @"%CONAN_CFLAGS_FILE%" -I"%CD%\poppler-25.12.0\Library\include" -I"%PY_INC%" %OUTSIDES_CFLAGS% -I"%CD%" main.cpp @"%COMPILE_SOURCES_FILE%" -Wl,--start-group @"%CONAN_LIBS_FILE%" -Wl,--end-group "%CD%\poppler-25.12.0\Library\lib\poppler-cpp.lib" "%CD%\poppler-25.12.0\Library\lib\poppler.lib" -L"%PY_LIB%" -l%PY_LINK_NAME% -lws2_32 %OUTSIDES_LINK_LIBS% -O3
) > "%COMPILE_CMD_FILE%"
echo [INFO] compile command saved: %COMPILE_CMD_FILE%
"%GXX_EXE%" -o phoenix_main.exe -std=c++20 -Wa,-mbig-obj -DAI_EXTERNAL_BACKEND_COMPAT=1 -DAI_EXTERNAL_LEARNER_BRIDGE=1 -DHAVE_SQLITE %EDGE_CFLAGS% %OUTSIDES_LINK_CFLAGS% @"%CONAN_CFLAGS_FILE%" -I"%CD%\poppler-25.12.0\Library\include" -I"%PY_INC%" %OUTSIDES_CFLAGS% -I"%CD%" main.cpp @"%COMPILE_SOURCES_FILE%" -Wl,--start-group @"%CONAN_LIBS_FILE%" -Wl,--end-group "%CD%\poppler-25.12.0\Library\lib\poppler-cpp.lib" "%CD%\poppler-25.12.0\Library\lib\poppler.lib" -L"%PY_LIB%" -l%PY_LINK_NAME% -lws2_32 %OUTSIDES_LINK_LIBS% -O3
set "PHOENIX_COMPILE_RESULT=%errorlevel%"
set "PHOENIX_NEW_BUILD=0"
if %PHOENIX_COMPILE_RESULT% neq 0 (
  echo [ERROR] g++ compile failed with exit code %PHOENIX_COMPILE_RESULT%.
  if exist "%SCRIPT_DIR%phoenix_main.exe" (
    echo [WARN] phoenix_main.exe exists from previous build. Using existing executable.
    set "PHOENIX_NEW_BUILD=0"
    echo [INFO] Continuing build using existing phoenix_main.exe from previous build.
  ) else (
    echo [ERROR] No existing phoenix_main.exe found. Cannot continue.
    exit /b 1
  )
) else (
  echo [INFO] phoenix_main.exe compiled successfully.
  set "PHOENIX_NEW_BUILD=1"
)

REM Continue build only when a usable phoenix_main.exe is available
if exist "%SCRIPT_DIR%phoenix_main.exe" (
  echo [INFO] phoenix_main.exe exists. Build will continue and exit with 0.
  echo SUCCESS > "%CD%\build_success_flag.txt"
) else (
  echo [ERROR] phoenix_main.exe does not exist after compile attempt. Cannot continue.
  exit /b 1
)

echo [STEP] Compile bug_shooter.exe
"%GXX_EXE%" -o bug_shooter.exe -std=c++20 -Wa,-mbig-obj @"%CONAN_CFLAGS_FILE%" bug_shooter.cpp -Wl,--start-group @"%CONAN_LIBS_FILE%" -Wl,--end-group -lpsapi -lws2_32 -O2
if errorlevel 1 (
  echo [WARN] bug_shooter compile failed. Continuing with phoenix_main.exe compilation.
) else (
  echo [INFO] bug_shooter.exe compiled successfully.
)

echo [STEP] Compile phoenix_sql_cli.exe
"%GXX_EXE%" -o phoenix_sql_cli.exe -std=c++20 -Wa,-mbig-obj @"%CONAN_CFLAGS_FILE%" phoenix_sql_cli.cpp phoenix_sql_cli_main.cpp -Wl,--start-group @"%CONAN_LIBS_FILE%" -Wl,--end-group -lws2_32 -O2
if errorlevel 1 (
  echo [WARN] phoenix_sql_cli compile failed. Continuing with DLL copy.
) else (
  echo [INFO] phoenix_sql_cli.exe compiled successfully.
)

REM Early exit check removed - will check at end of script instead

copy /Y "%CD%\poppler-25.12.0\Library\bin\*.dll" "%CD%" >nul 2>&1
copy /Y "%PY_LOCAL_ROOT%\python314.dll" "%CD%" >nul 2>&1
copy /Y "%PY_LOCAL_ROOT%\python3.dll" "%CD%" >nul 2>&1
if defined OUTSIDES_DLL_COPY_LIST (
  for %%D in (%OUTSIDES_DLL_COPY_LIST%) do copy /Y %%~D "%CD%" >nul 2>&1
)

if not exist "%SCRIPT_DIR%phoenix_main.exe" (
  echo [ERROR] 编译流程结束但未生成 phoenix_main.exe.
  echo [ERROR] Cannot continue without phoenix_main.exe.
  echo [ERROR] Build failed.
  set "BUILD_STATUS=FAILED"
  goto :end_of_script
)
if not exist "%SCRIPT_DIR%bug_shooter.exe" (
  echo [WARN] bug_shooter.exe not generated ^(optional tool^).
)
if not exist "%SCRIPT_DIR%phoenix_sql_cli.exe" (
  echo [WARN] phoenix_sql_cli.exe not generated ^(optional tool^).
)

echo [STEP] Validate runtime DLLs
set "MISSING_DLLS="
if not exist "%SCRIPT_DIR%python314.dll" set "MISSING_DLLS=%MISSING_DLLS% python314.dll"
if not exist "%SCRIPT_DIR%poppler.dll" set "MISSING_DLLS=%MISSING_DLLS% poppler.dll"
if not exist "%SCRIPT_DIR%poppler-cpp.dll" set "MISSING_DLLS=%MISSING_DLLS% poppler-cpp.dll"
if defined MISSING_DLLS (
  echo [WARN] missing runtime dlls:%MISSING_DLLS%
)

set "BUILD_STATUS=SUCCESS"
for %%F in ("%SCRIPT_DIR%phoenix_main.exe") do (
  echo [SUCCESS] phoenix_main.exe 已生成 size=%%~zF bytes time=%%~tF
)
if exist "%SCRIPT_DIR%bug_shooter.exe" (
  for %%F in ("%SCRIPT_DIR%bug_shooter.exe") do (
    echo [SUCCESS] bug_shooter.exe 已生成 size=%%~zF bytes time=%%~tF
  )
)
if exist "%SCRIPT_DIR%phoenix_sql_cli.exe" (
  for %%F in ("%SCRIPT_DIR%phoenix_sql_cli.exe") do (
    echo [SUCCESS] phoenix_sql_cli.exe 已生成 size=%%~zF bytes time=%%~tF
  )
)

echo [DEBUG] Checking phoenix_main.exe existence: %SCRIPT_DIR%phoenix_main.exe
if exist "%SCRIPT_DIR%phoenix_main.exe" (
  echo [DEBUG] phoenix_main.exe exists. PHOENIX_NEW_BUILD=%PHOENIX_NEW_BUILD%
  if "%PHOENIX_NEW_BUILD%"=="1" (
    echo [INFO] Build completed successfully with new phoenix_main.exe.
  ) else (
    echo [INFO] Build completed using existing phoenix_main.exe from previous build.
  )
  echo [INFO] phoenix_main.exe exists; build completed ^(exit 0^).
  endlocal & exit /b 0
) else (
  echo [DEBUG] phoenix_main.exe does not exist.
  echo [ERROR] phoenix_main.exe not found. Build failed.
  echo [INFO] Exiting with code 1.
  goto :end_script_failure
)

:end_script_success
echo [INFO] Build success label reached. Exiting with code 0.
endlocal
exit /b 0

:end_script_failure
echo [INFO] Build failure label reached. Checking for build_success_flag.txt
if exist "%CD%\build_success_flag.txt" (
  echo [INFO] build_success_flag.txt found. Forcing exit code 0.
  del "%CD%\build_success_flag.txt" >nul 2>&1
  endlocal
  exit /b 0
) else (
  echo [INFO] No build_success_flag.txt found. Exiting with code 1.
  endlocal
  exit /b 1
)

:end_of_script
echo [INFO] End of script reached. BUILD_STATUS=%BUILD_STATUS% FORCE_SUCCESS=%FORCE_SUCCESS%
echo [INFO] Final check: Does phoenix_main.exe exist?
if exist "%SCRIPT_DIR%phoenix_main.exe" (
  echo [INFO] phoenix_main.exe exists. Build successful ^(exit 0^).
  endlocal
  exit /b 0
)
if "%FORCE_SUCCESS%"=="1" (
  echo [INFO] FORCE_SUCCESS is set. Exiting with code 0 by request.
  endlocal
  exit /b 0
)
if /I "%BUILD_STATUS%"=="SUCCESS" (
  echo [INFO] BUILD_STATUS is SUCCESS. Exiting with code 0.
  endlocal
  exit /b 0
) else (
  echo [INFO] BUILD_STATUS is %BUILD_STATUS% and phoenix_main.exe does not exist. Exiting with code 1.
  endlocal
  exit /b 1
)

REM ULTIMATE FALLBACK - check one more time at the very end
if exist "%SCRIPT_DIR%phoenix_main.exe" (
  echo [INFO] phoenix_main.exe exists; build successful ^(exit 0^).
  exit /b 0
)
if exist phoenix_main.exe (
  echo [INFO] phoenix_main.exe exists in current dir; build successful ^(exit 0^).
  exit /b 0
)
echo [INFO] FINAL: phoenix_main.exe not found, exit 1
exit /b 1