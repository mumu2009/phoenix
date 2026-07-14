@echo off
setlocal EnableExtensions
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo [STEP] Compile GTest Tests

if not exist "D:\Scoop\apps\gcc\current\bin\gcc.exe" (
  set "PATH=D:\Scoop\apps\gcc\current\bin;%PATH%"
)

set "GCC_EXE="
set "GXX_EXE="
for /f "delims=" %%I in ('where gcc 2^>nul') do if not defined GCC_EXE set "GCC_EXE=%%I"
for /f "delims=" %%I in ('where g++ 2^>nul') do if not defined GXX_EXE set "GXX_EXE=%%I"

if not defined GCC_EXE (
  echo [ERROR] gcc not found.
  exit /b 1
)
if not defined GXX_EXE (
  echo [ERROR] g++ not found.
  exit /b 1
)

where conan >nul 2>&1 || (
  echo [ERROR] Conan not found.
  exit /b 1
)

if not exist "build" mkdir "build"

if not defined CONAN_HOME set "CONAN_HOME=%CD%\build\conan_home"
if not exist "%CONAN_HOME%" mkdir "%CONAN_HOME%"

echo [STEP] Conan install for GTest
set "CONAN_COMMON=--output-folder=build --build=missing -s build_type=Release"
set "CC=%GCC_EXE:\=/%"
set "CXX=%GXX_EXE:\=/%"
set "CXXFLAGS=-include cstdint"
conan install . %CONAN_COMMON% -c tools.cmake.cmaketoolchain:generator=Ninja
if errorlevel 1 (
  echo [ERROR] Conan install failed.
  exit /b 1
)

where pkg-config >nul 2>&1 || (
  echo [ERROR] pkg-config not found.
  exit /b 1
)

set "PKGCFG_PATH=%CD:\=/%/build"
set "CONAN_CFLAGS_FILE=build\conan_cflags_gtest.txt"
set "CONAN_LIBS_FILE=build\conan_libs_gtest.txt"

call pkg-config --with-path="%PKGCFG_PATH%" --cflags drogon trantor opencv sqlite3 lmdb hiredis redis++ nlohmann_json jwt-cpp jsoncpp gtest > "%CONAN_CFLAGS_FILE%"
if errorlevel 1 (
  echo [ERROR] failed to generate conan_cflags_gtest.txt.
  exit /b 1
)
call pkg-config --with-path="%PKGCFG_PATH%" --libs drogon trantor opencv sqlite3 lmdb hiredis redis++ nlohmann_json jwt-cpp jsoncpp gtest > "%CONAN_LIBS_FILE%"
if errorlevel 1 (
  echo [ERROR] failed to generate conan_libs_gtest.txt.
  exit /b 1
)

if not exist "%CD%\build\tmp" mkdir "%CD%\build\tmp"
set "TMP=%CD%\build\tmp"
set "TEMP=%CD%\build\tmp"

set "PY_LOCAL_ROOT=%CD%\Python314"
set "PY_INC=%PY_LOCAL_ROOT%\include"
set "PY_LIB=%PY_LOCAL_ROOT%\libs"
set "PY_LINK_NAME="
if exist "%PY_LIB%\python314.lib" set "PY_LINK_NAME=python314"
if not defined PY_LINK_NAME if exist "%PY_LIB%\python3.lib" set "PY_LINK_NAME=python3"

set "OUTSIDES_ROOT=%CD%\outsides"
set "LLAMACPP_ROOT=%OUTSIDES_ROOT%\llamacpp"
set "BITNET_ROOT=%OUTSIDES_ROOT%\BitNet"
set "BULLET3_ROOT=%OUTSIDES_ROOT%\bullet3"

set "OUTSIDES_CFLAGS="
if exist "%LLAMACPP_ROOT%\include" set "OUTSIDES_CFLAGS=%OUTSIDES_CFLAGS% -I%LLAMACPP_ROOT%\include"
if exist "%BITNET_ROOT%\include" set "OUTSIDES_CFLAGS=%OUTSIDES_CFLAGS% -I%BITNET_ROOT%\include"
if exist "%BULLET3_ROOT%\src\btBulletDynamicsCommon.h" set "OUTSIDES_CFLAGS=%OUTSIDES_CFLAGS% -I%BULLET3_ROOT%\src"

set "BULLET3_EMBEDDED_SOURCES="
if exist "%BULLET3_ROOT%\src\btLinearMathAll.cpp" set "BULLET3_EMBEDDED_SOURCES=outsides\bullet3\src\btLinearMathAll.cpp outsides\bullet3\src\btBulletCollisionAll.cpp outsides\bullet3\src\btBulletDynamicsAll.cpp"

set "COMMON_SOURCES=transformer_main.cpp transformer_ollama_fine_tuning.cpp addon.cpp addons\builtin_registry.cpp addons\math_addon.cpp addons\search_addon.cpp addons\computer_shell_addon.cpp loggerCXX.cpp DATABASE_079.cpp frontend_server.cpp speak_io.cpp model_lifecycle.cpp autonomy_stack.cpp v51_runtime.cpp external_runtime.cpp edge_platform.cpp gguf_tensor_parser.cpp physics_world_runtime.cpp emotion_system.cpp llamacpp_emotion_adjuster.cpp plugin_system.cpp modern_context_system.cpp module_overrides\adversarial_learner_advanced.cpp module_overrides\gnn_ga_learner_advanced.cpp module_overrides\reinforcement_learner_advanced.cpp %BULLET3_EMBEDDED_SOURCES%"

echo [STEP] Compile GTest test runner
set "GTEST_SOURCES=tests\gtest\gtest_main.cpp tests\gtest\unit\emotion_system\test_emotion_tensor.cpp tests\gtest\unit\emotion_system\test_emotion_state.cpp tests\gtest\unit\emotion_system\test_emotion_analyzer.cpp tests\gtest\unit\emotion_system\test_emotion_system.cpp tests\gtest\unit\emotion_system\test_emotion_weight.cpp tests\gtest\unit\emotion_system\test_emotion_plugin.cpp tests\gtest\unit\plugin_system\test_plugin_lifecycle.cpp tests\gtest\unit\plugin_system\test_plugin_manager.cpp tests\gtest\unit\plugin_system\test_plugin_operations.cpp tests\gtest\unit\context_system\test_context_manager.cpp tests\gtest\unit\context_system\test_context_window.cpp tests\gtest\unit\context_system\test_transformer_model.cpp tests\gtest\unit\context_system\test_summary_model.cpp tests\gtest\unit\context_system\test_adaptive_controller.cpp tests\gtest\unit\context_system\test_context_service_integration.cpp tests\gtest\subsystem\test_emotion_plugin_integration.cpp tests\gtest\subsystem\test_context_plugin_integration.cpp tests\gtest\subsystem\test_full_system_integration.cpp tests\gtest\module\test_edge_platform.cpp tests\gtest\module\test_database_079.cpp tests\gtest\integration\test_end_to_end.cpp tests\gtest\unit\runtime\test_v51_runtime.cpp tests\gtest\unit\runtime\test_external_runtime.cpp tests\gtest\unit\lifecycle\test_model_lifecycle.cpp tests\gtest\unit\autonomy\test_autonomy_stack.cpp tests\gtest\unit\frontend\test_frontend_server.cpp tests\gtest\unit\physics\test_physics_world_runtime.cpp tests\gtest\unit\emotion\test_llamacpp_emotion_adjuster.cpp tests\gtest\unit\addon\test_addon_system.cpp tests\gtest\unit\logger\test_loggercxx.cpp tests\gtest\unit\speak\test_speak_io.cpp tests\gtest\unit\gguf\test_gguf_tensor_parser.cpp tests\gtest\unit\module_overrides\test_adversarial_learner_advanced.cpp tests\gtest\unit\module_overrides\test_gnn_ga_learner_advanced.cpp tests\gtest\unit\module_overrides\test_reinforcement_learner_advanced.cpp"

"%GXX_EXE%" -o gtest_runner.exe -std=c++20 -Wa,-mbig-obj -DAI_EXTERNAL_BACKEND_COMPAT=1 -DAI_EXTERNAL_LEARNER_BRIDGE=1 @"%CONAN_CFLAGS_FILE%" -I"%CD%" -I"%CD%\poppler-25.12.0\Library\include" -I"%PY_INC%" %OUTSIDES_CFLAGS% -I"%CD%\tests\gtest" %GTEST_SOURCES% %COMMON_SOURCES% -Wl,--start-group @"%CONAN_LIBS_FILE%" -Wl,--end-group "%CD%\poppler-25.12.0\Library\lib\poppler-cpp.lib" "%CD%\poppler-25.12.0\Library\lib\poppler.lib" -L"%PY_LIB%" -l%PY_LINK_NAME% -lws2_32 -O3 -g

if errorlevel 1 (
  echo [ERROR] GTest compilation failed.
  exit /b 1
)

echo [SUCCESS] gtest_runner.exe compiled successfully.
endlocal
exit /b 0
