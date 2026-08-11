@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM ---------------------------------------------------------------------
REM build_llama_server.bat
REM
REM (Re-)builds llama-server.exe / llama-cli.exe (and the underlying
REM `llama` static library) from outsides\llamacpp\build-gcc using the
REM ninja generator that is already configured there.
REM
REM Usage:
REM   llama_server_mods\build_llama_server.bat
REM   llama_server_mods\build_llama_server.bat --force   (skip the "is a
REM       rebuild even necessary" timestamp check and always invoke ninja)
REM
REM Exit code: 0 on success. On failure this script prints an error and
REM returns 1; callers (e.g. compile.bat) decide whether that should be
REM fatal for the overall Phoenix build.
REM ---------------------------------------------------------------------

set "SCRIPT_DIR=%~dp0"
set "PHOENIX_ROOT=%SCRIPT_DIR%.."
for %%I in ("%PHOENIX_ROOT%") do set "PHOENIX_ROOT=%%~fI"

set "LLAMACPP_ROOT=%PHOENIX_ROOT%\outsides\llamacpp"
set "BUILD_DIR=%LLAMACPP_ROOT%\build-gcc"
set "STAMP_FILE=%BUILD_DIR%\.phoenix_llama_server_build_stamp"

set "FORCE_BUILD=0"
if /I "%~1"=="--force" set "FORCE_BUILD=1"
if /I "%PHOENIX_LLAMA_SERVER_FORCE_REBUILD%"=="1" set "FORCE_BUILD=1"

echo [build_llama_server] LLAMACPP_ROOT=%LLAMACPP_ROOT%
echo [build_llama_server] BUILD_DIR=%BUILD_DIR%

if not exist "%LLAMACPP_ROOT%" (
  echo [build_llama_server][ERROR] %LLAMACPP_ROOT% does not exist.
  exit /b 1
)

REM --- Locate a usable cmake/ninja toolchain -------------------------------
where ninja >nul 2>&1
if errorlevel 1 (
  echo [build_llama_server][ERROR] ninja not found on PATH.
  exit /b 1
)

REM Pin the C/C++ compiler explicitly (mirroring compile.bat's own GCC
REM detection) instead of letting `cmake` pick whatever compiler happens to
REM be first on PATH. Machines in this project also have LLVM/clang
REM installed via Scoop, and clang's default Windows target expects the MSVC
REM libs (kernel32.lib, msvcrtd.lib, ...) which are not set up here -- if
REM cmake silently picks clang, configure succeeds but the very first
REM compiler-sanity-check link fails. GCC (mingw-w64) is the toolchain this
REM build-gcc directory is named after and was originally configured with.
set "PHX_GCC_EXE="
set "PHX_GXX_EXE="
if exist "D:\Scoop\apps\gcc\current\bin\gcc.exe" set "PHX_GCC_EXE=D:\Scoop\apps\gcc\current\bin\gcc.exe"
if exist "D:\Scoop\apps\gcc\current\bin\g++.exe" set "PHX_GXX_EXE=D:\Scoop\apps\gcc\current\bin\g++.exe"
if not defined PHX_GCC_EXE for /f "delims=" %%I in ('where gcc 2^>nul') do if not defined PHX_GCC_EXE set "PHX_GCC_EXE=%%I"
if not defined PHX_GXX_EXE for /f "delims=" %%I in ('where g++ 2^>nul') do if not defined PHX_GXX_EXE set "PHX_GXX_EXE=%%I"
if not defined PHX_GCC_EXE (
  echo [build_llama_server][ERROR] gcc not found ^(checked D:\Scoop\apps\gcc\current\bin and PATH^).
  exit /b 1
)
if not defined PHX_GXX_EXE (
  echo [build_llama_server][ERROR] g++ not found ^(checked D:\Scoop\apps\gcc\current\bin and PATH^).
  exit /b 1
)
echo [build_llama_server] Using CC=%PHX_GCC_EXE% CXX=%PHX_GXX_EXE%

REM --- Detect a build-gcc directory configured against a *different*
REM     llamacpp checkout path (e.g. left over from copying/renaming the
REM     Phoenix project directory) and reconfigure it in place if so.
REM     Without this check, ninja would happily rebuild/relink against
REM     whatever unrelated source tree the stale CMakeCache.txt points at,
REM     silently ignoring the patches this script is supposed to build.
set "NEED_CONFIGURE=0"
if not exist "%BUILD_DIR%\build.ninja" set "NEED_CONFIGURE=1"
if not exist "%BUILD_DIR%\CMakeCache.txt" set "NEED_CONFIGURE=1"

if "%NEED_CONFIGURE%"=="0" (
  findstr /I /C:"CMAKE_HOME_DIRECTORY" "%BUILD_DIR%\CMakeCache.txt" > "%TEMP%\phx_cmake_home.txt" 2>nul
  set "CMAKE_HOME_LINE="
  for /f "usebackq delims=" %%L in ("%TEMP%\phx_cmake_home.txt") do set "CMAKE_HOME_LINE=%%L"
  echo !CMAKE_HOME_LINE! | findstr /I /C:"%LLAMACPP_ROOT:\=/%" >nul
  if errorlevel 1 (
    echo [build_llama_server][WARN] %BUILD_DIR% was configured against a different source tree ^(!CMAKE_HOME_LINE!^).
    echo [build_llama_server][WARN] Reconfiguring in place for %LLAMACPP_ROOT%.
    set "NEED_CONFIGURE=1"
  )
)

if "%NEED_CONFIGURE%"=="1" (
  where cmake >nul 2>&1
  if errorlevel 1 (
    echo [build_llama_server][ERROR] cmake not found on PATH and %BUILD_DIR% needs to be ^(re^)configured.
    exit /b 1
  )
  if exist "%BUILD_DIR%\CMakeCache.txt" del /q "%BUILD_DIR%\CMakeCache.txt"
  if exist "%BUILD_DIR%\CMakeFiles" rmdir /s /q "%BUILD_DIR%\CMakeFiles"
  if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
  echo [build_llama_server] Configuring %BUILD_DIR% with CMake for %LLAMACPP_ROOT% ...
  cmake -S "%LLAMACPP_ROOT%" -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_C_COMPILER="%PHX_GCC_EXE%" -DCMAKE_CXX_COMPILER="%PHX_GXX_EXE%"
  if not "!errorlevel!"=="0" (
    echo [build_llama_server][ERROR] cmake configure failed with exit code !errorlevel!.
    exit /b 1
  )
  REM Reconfiguring necessarily invalidates any previous build outputs/stamp.
  set "FORCE_BUILD=1"
  if exist "%STAMP_FILE%" del /q "%STAMP_FILE%"
)

REM --- Decide whether a rebuild is actually needed -------------------------
REM Compare the newest mtime among the C++ source files this Phoenix patch
REM stack touches (plus the patches themselves) against a stamp file we
REM write after a successful build. This is intentionally coarse (it does
REM not try to replicate ninja's own dependency graph) -- ninja itself will
REM still no-op quickly if nothing actually changed, so this check exists
REM mainly to let compile.bat skip invoking cmake/ninja entirely on the
REM common "nothing changed since last full Phoenix build" path.
set "NEED_BUILD=%FORCE_BUILD%"

set "SERVER_EXE=%BUILD_DIR%\bin\llama-server.exe"
if not exist "%SERVER_EXE%" set "NEED_BUILD=1"

if "%NEED_BUILD%"=="0" (
  if not exist "%STAMP_FILE%" (
    set "NEED_BUILD=1"
  ) else (
    for %%F in ("%SERVER_EXE%" "%STAMP_FILE%") do if not exist %%F set "NEED_BUILD=1"
    powershell -NoProfile -Command ^
      "$stamp = (Get-Item -LiteralPath '%STAMP_FILE%').LastWriteTimeUtc;" ^
      "$paths = @('%LLAMACPP_ROOT%\include', '%LLAMACPP_ROOT%\src', '%LLAMACPP_ROOT%\examples\server', '%SCRIPT_DIR%existing_mods.patch', '%SCRIPT_DIR%enc_dec_separation.patch');" ^
      "$newer = $false;" ^
      "foreach ($p in $paths) { if (Test-Path $p) { Get-ChildItem -LiteralPath $p -Recurse -File -ErrorAction SilentlyContinue | ForEach-Object { if ($_.LastWriteTimeUtc -gt $stamp) { $newer = $true } } } }" ^
      "if ($newer) { exit 1 } else { exit 0 }"
    if errorlevel 1 set "NEED_BUILD=1"
  )
)

if "%NEED_BUILD%"=="0" (
  echo [build_llama_server] Sources unchanged since last build ^(stamp: %STAMP_FILE%^); skipping ninja invocation.
  exit /b 0
)

echo [build_llama_server] Building llama / llama-server via ninja...
pushd "%BUILD_DIR%"
ninja llama llama-server llama-cli
set "NINJA_RC=%errorlevel%"
popd

if not "%NINJA_RC%"=="0" (
  echo [build_llama_server][ERROR] ninja build failed with exit code %NINJA_RC%.
  exit /b 1
)

if not exist "%SERVER_EXE%" (
  echo [build_llama_server][ERROR] Build reported success but %SERVER_EXE% is missing.
  exit /b 1
)

echo. > "%STAMP_FILE%"
echo [build_llama_server] Build OK: %SERVER_EXE%
exit /b 0
