@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM ---------------------------------------------------------------------
REM apply_patches.bat
REM
REM Applies the tracked Phoenix modifications on top of a clean
REM outsides\llamacpp checkout:
REM   1. llama_server_mods\existing_mods.patch        (style adapter API)
REM   2. llama_server_mods\enc_dec_separation.patch    (enc/infer/dec split)
REM
REM Why this exists: outsides\ is listed in .gitignore (it holds vendored
REM third-party checkouts, including outsides\llamacpp), so any change we
REM make directly inside outsides\llamacpp would NOT be tracked by git and
REM would be lost/overwritten if that checkout is ever re-cloned or wiped.
REM Instead, the actual modifications are kept as .patch files under the
REM tracked llama_server_mods\ directory, and this script (re-)applies them
REM to whatever checkout currently lives at outsides\llamacpp.
REM
REM This script is idempotent: if the patches are already applied it will
REM detect that and skip re-applying them. If the working tree has
REM diverged (patches neither applied nor cleanly applicable), it will try
REM to reset outsides\llamacpp to a clean state (git checkout/clean) before
REM re-applying, since it is a disposable/gitignored vendor checkout.
REM
REM Exit code: 0 on success (including "already applied"), 1 on failure.
REM ---------------------------------------------------------------------

set "SCRIPT_DIR=%~dp0"
set "PHOENIX_ROOT=%SCRIPT_DIR%.."
for %%I in ("%PHOENIX_ROOT%") do set "PHOENIX_ROOT=%%~fI"

set "LLAMACPP_ROOT=%PHOENIX_ROOT%\outsides\llamacpp"
set "PATCH_DIR=%SCRIPT_DIR%"
if "%PATCH_DIR:~-1%"=="\" set "PATCH_DIR=%PATCH_DIR:~0,-1%"

set "PATCH_1=%PATCH_DIR%\existing_mods.patch"
set "PATCH_2=%PATCH_DIR%\enc_dec_separation.patch"

echo [apply_patches] LLAMACPP_ROOT=%LLAMACPP_ROOT%

if not exist "%LLAMACPP_ROOT%\.git" (
  echo [apply_patches][ERROR] %LLAMACPP_ROOT% is not a git checkout. Cannot apply patches.
  exit /b 1
)
if not exist "%PATCH_1%" (
  echo [apply_patches][ERROR] Missing %PATCH_1%
  exit /b 1
)

pushd "%LLAMACPP_ROOT%" || exit /b 1

where git >nul 2>&1
if errorlevel 1 (
  echo [apply_patches][ERROR] git not found on PATH.
  popd
  exit /b 1
)

REM --- Fast path: is the full patch stack already applied? -----------------
REM enc_dec_separation.patch is generated on top of existing_mods.patch, so
REM if it (the *last* patch in the stack) reverse-applies cleanly, both are
REM known to already be in place. We only need to check the last patch that
REM is actually present on disk.
set "LAST_PATCH=%PATCH_1%"
if exist "%PATCH_2%" set "LAST_PATCH=%PATCH_2%"

git apply --reverse --check "%LAST_PATCH%" >nul 2>&1
if not errorlevel 1 (
  echo [apply_patches] Patch stack already applied ^(checked against %LAST_PATCH%^). Nothing to do.
  popd
  exit /b 0
)

REM --- Otherwise: reset to a clean checkout and re-apply both patches. ---
echo [apply_patches] Resetting outsides\llamacpp working tree to a clean checkout...
git reset --hard HEAD >nul 2>&1
git clean -fd -- include src examples/server >nul 2>&1

echo [apply_patches] Applying existing_mods.patch...
git apply --whitespace=nowarn "%PATCH_1%"
if errorlevel 1 (
  echo [apply_patches][ERROR] Failed to apply existing_mods.patch. The vendored llama.cpp checkout may have drifted from the version this patch was written against.
  popd
  exit /b 1
)

if exist "%PATCH_2%" (
  echo [apply_patches] Applying enc_dec_separation.patch...
  git apply --whitespace=nowarn "%PATCH_2%"
  if errorlevel 1 (
    echo [apply_patches][ERROR] Failed to apply enc_dec_separation.patch.
    popd
    exit /b 1
  )
) else (
  echo [apply_patches][WARN] enc_dec_separation.patch not found under %PATCH_DIR%. Skipping enc/infer/dec split.
)

echo [apply_patches] All patches applied successfully.
popd
exit /b 0
