# tools/setup_gdb_local.ps1
#
# Provision a usable MinGW gdb WITHOUT installing it system-wide and WITHOUT
# touching the MSYS2 package database.  Why: gdb is not part of the default
# MSYS2 mingw-w64 toolchain install on this machine, and `pacman -S gdb` would
# add/upgrade shared runtime DLLs under C:\msys64 that the project's conan
# prebuilts + the live gateway depend on.  Instead we download the packages to
# the pacman cache only (`pacman -Sw`) and unpack them into
# build\tmp\gdb, then run gdb from there with an explicit PATH.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\setup_gdb_local.ps1
#   powershell ... -File tools\setup_gdb_local.ps1 -Force   # re-extract
#
# Result: build\tmp\gdb\mingw64\bin\gdb.exe (GDB 17.2 for mingw-w64)
[CmdletBinding()]
param(
    [string]$MsysRoot = 'C:\msys64',
    [string]$Dest = '',
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $Dest) { $Dest = Join-Path $ProjectRoot 'build\tmp\gdb' }
$pacman = Join-Path $MsysRoot 'usr\bin\pacman.exe'
$bsdtar = Join-Path $MsysRoot 'usr\bin\bsdtar.exe'
$cache = Join-Path $MsysRoot 'var\cache\pacman\pkg'
$gdbExe = Join-Path $Dest 'mingw64\bin\gdb.exe'

if (-not (Test-Path $pacman)) { throw "pacman not found: $pacman (is MSYS2 installed at $MsysRoot?)" }
if (-not (Test-Path $bsdtar)) { throw "bsdtar not found: $bsdtar" }

if ((Test-Path $gdbExe) -and -not $Force) {
    Write-Host "[gdb] already provisioned: $gdbExe"
    & $gdbExe --version | Select-Object -First 1
    exit 0
}

Write-Host "[gdb] downloading mingw-w64 gdb + dependencies to the pacman cache (no install)..."
& $pacman -Sw --noconfirm --needed mingw-w64-x86_64-gdb 2>&1 | ForEach-Object { Write-Host "  $_" }
if ($LASTEXITCODE -ne 0) { throw "pacman -Sw failed with exit $LASTEXITCODE" }

# Print the full dependency closure as archive file names, then unpack each one.
$pkgFiles = & $pacman -Sp --print-format '%n-%v-%a.pkg.tar.zst' mingw-w64-x86_64-gdb 2>&1
if ($LASTEXITCODE -ne 0 -or -not $pkgFiles) { throw "pacman -Sp failed to resolve the gdb dependency closure" }

New-Item -ItemType Directory -Force $Dest | Out-Null
$extracted = 0
foreach ($pkgFile in $pkgFiles) {
    $pkgFile = $pkgFile.Trim()
    if (-not $pkgFile) { continue }
    $pkgPath = Join-Path $cache $pkgFile
    if (-not (Test-Path $pkgPath)) {
        Write-Host "[gdb] extracting extra package: $pkgFile"
        & $pacman -Sw --noconfirm --needed ($pkgFile -replace '-\d+[^-]*-any\.pkg\.tar\.zst$', '') 2>&1 | Out-Null
    }
    if (-not (Test-Path $pkgPath)) { throw "package not in cache: $pkgPath" }
    & $bsdtar -xf $pkgPath -C $Dest
    if ($LASTEXITCODE -ne 0) { throw "bsdtar failed for $pkgPath (exit $LASTEXITCODE)" }
    $extracted++
}
Write-Host "[gdb] extracted $extracted packages into $Dest"

if (-not (Test-Path $gdbExe)) { throw "gdb.exe missing after extraction: $gdbExe" }
Write-Host "[gdb] OK:"
& $gdbExe --version | Select-Object -First 2
Write-Host ''
Write-Host "Run gdb with the project-local DLLs on PATH, e.g.:"
Write-Host "  `$env:PATH = '$Dest\mingw64\bin;C:\msys64\mingw64\bin;' + `$env:PATH"
Write-Host "  & '$gdbExe' -batch -nx -ex 'break main' -ex run -ex bt --args phoenix_main.next.exe"
Write-Host ''
Write-Host "Or use the wrapper: tools\gdb_probe.ps1 -Exe phoenix_main.next.exe -Break main"
