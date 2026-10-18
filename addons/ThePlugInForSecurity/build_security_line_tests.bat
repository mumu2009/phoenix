@echo off
setlocal EnableExtensions
cd /d "%~dp0..\.."

set "GXX=g++"
where g++ >nul 2>&1
if errorlevel 1 (
  if exist "C:\msys64\mingw64\bin\g++.exe" set "GXX=C:\msys64\mingw64\bin\g++.exe"
)
where "%GXX%" >nul 2>&1
if errorlevel 1 if not exist "%GXX%" (
  echo [ERROR] g++ not found
  exit /b 1
)

set "GTEST_INC=build\conan_home\p\b\gtesta603b40da79d1\p\include"
set "GTEST_LIB=build\conan_home\p\b\gtesta603b40da79d1\p\lib"
set "JSON_INC=build\conan_home\p\nlohm1bed1ddc0a2fa\p\include"
if not exist "%GTEST_INC%\gtest\gtest.h" (
  echo [ERROR] gtest headers missing at %GTEST_INC%
  exit /b 1
)
if not exist "%JSON_INC%\nlohmann\json.hpp" (
  echo [ERROR] nlohmann json missing at %JSON_INC%
  exit /b 1
)
if not exist "build" mkdir "build"

echo [STEP] security line tests
"%GXX%" -std=c++20 -O2 -Wa,-mbig-obj -DPHOENIX_SECURITY_LINE_MAIN -DNOMINMAX -I. -I"%JSON_INC%" -I"%GTEST_INC%" -o build\security_line_tests.exe ^
  test\gtest\unit\plugin\test_hard_embed.cpp ^
  addons\ThePlugInForSecurity\security_core.cpp ^
  addons\ThePlugInForSecurity\meme_tensor.cpp ^
  addons\ThePlugInForSecurity\meme_semantic_slicer.cpp ^
  addons\ThePlugInForSecurity\meme_annealer.cpp ^
  memetic_existence.cpp ^
  graph_diffusion_summarizer.cpp ^
  emergency_stop.cpp ^
  instance_registry.cpp ^
  -L"%GTEST_LIB%" -lgtest
if errorlevel 1 (
  echo [ERROR] security line compile failed
  exit /b 1
)
echo [SUCCESS] build\security_line_tests.exe
exit /b 0
