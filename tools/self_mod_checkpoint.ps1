# self_mod_checkpoint.ps1 - mandatory pre-self-modification checkpoint.
#
# Creates a rollback point BEFORE any automatic code change:
#   1. refuse when a build is running (compile.bat / compile_gtest.bat / cc1plus)
#   2. git add -A  (including the gitignored tests/ tree the agent edits)
#   3. commit with a timestamped message
#   4. tag selfmod/<timestamp> so rollback is one command away
#
# Usage:  powershell -File tools\self_mod_checkpoint.ps1 [-Note "why"]
param([string]$Note = "self-modification checkpoint")
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

if (Get-Process cc1plus, g++ -ErrorAction SilentlyContinue) {
  Write-Error "a compiler is running - do not checkpoint mid-build"
}
if (Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" |
      Where-Object { $_.CommandLine -match 'compile(_gtest)?\.bat' }) {
  Write-Error "compile.bat is running - do not checkpoint mid-build"
}

# stale lock left by an interrupted git is fatal for the next add
if (Test-Path .git\index.lock) {
  if (-not (Get-Process git -ErrorAction SilentlyContinue)) {
    Remove-Item .git\index.lock -Force
    Write-Host "[checkpoint] removed stale .git/index.lock"
  }
}

git add -A
git add -f tests/gtest/unit 2>$null
$dirty = (git status --porcelain | Measure-Object -Line).Lines
if ($dirty -eq 0) { Write-Host "[checkpoint] nothing to commit; tree already clean" }
else {
  $ts = Get-Date -Format 'yyyyMMdd-HHmmss'
  git -c user.name="phoenix-agent" -c user.email="agent@localhost" `
      commit -q -m "selfmod checkpoint $ts - $Note"
  Write-Host "[checkpoint] committed $dirty path(s)"
}
$tag = "selfmod/" + (Get-Date -Format 'yyyyMMdd-HHmmss')
git tag -f $tag | Out-Null
Write-Host "[checkpoint] tag=$tag"
Write-Host "[checkpoint] rollback: git reset --hard $tag"
$tag
