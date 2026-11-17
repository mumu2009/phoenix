#Requires -Version 7.0
<#
===============================================================================
 vendor_patch_verify.ps1 - prove the Phoenix llama.cpp vendor tree is
 reproducible from a clean upstream checkout.
===============================================================================
 What it does
 ------------
   1. creates a temporary PRISTINE worktree of outsides/llamacpp at the commit
      recorded in the hash truth file (default: 733dd8df),
   2. applies the canonical Phoenix vendor patch set,
   3. compares sha256 per file against the recorded truth and checks that the
      applied tree differs from pristine in exactly the recorded file set,
   4. prints a PASS/FAIL table + writes build/tmp/vendor_patch_verify/last_result.json,
      then removes the temporary worktree.

 The truth file is llama_server_mods/vendor_733dd8df.sha256.json (tracked), with a
 fallback to build/tmp/llama_patch_repair/current_state.json (the raw captured
 vendor state used while the patch was regenerated).

 Usage
 -----
   pwsh -NoProfile -File tools/vendor_patch_verify.ps1
   pwsh -NoProfile -File tools/vendor_patch_verify.ps1 -KeepTemp
   pwsh -NoProfile -File tools/vendor_patch_verify.ps1 -Patch llama_server_mods/other.patch

 Exit codes: 0 = PASS, 1 = FAIL (hash/set mismatch or apply error), 2 = setup error.
===============================================================================
#>
[CmdletBinding()]
param(
  [string]$VendorRoot = '',
  [string]$Patch = '',
  [string]$CurrentState = '',
  [switch]$KeepTemp
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
if (-not $VendorRoot) { $VendorRoot = Join-Path $repo 'outsides\llamacpp' }
if (-not $Patch) { $Patch = Join-Path $repo 'llama_server_mods\phoenix_vendor_733dd8df.patch' }
if (-not $CurrentState) {
  $tracked = Join-Path $repo 'llama_server_mods\vendor_733dd8df.sha256.json'
  $scratch = Join-Path $repo 'build\tmp\llama_patch_repair\current_state.json'
  if (Test-Path -LiteralPath $tracked) { $CurrentState = $tracked }
  elseif (Test-Path -LiteralPath $scratch) { $CurrentState = $scratch }
  else { $CurrentState = $tracked }
}

function Fail-Setup([string]$Message) {
  Write-Host ("ERROR: " + $Message) -ForegroundColor Red
  exit 2
}

if (-not (Test-Path -LiteralPath $VendorRoot)) { Fail-Setup "vendor root not found: $VendorRoot" }
if (-not (Test-Path -LiteralPath (Join-Path $VendorRoot '.git'))) { Fail-Setup "vendor root is not a git repo: $VendorRoot" }
if (-not (Test-Path -LiteralPath $Patch)) { Fail-Setup "patch not found: $Patch" }
if (-not (Test-Path -LiteralPath $CurrentState)) { Fail-Setup "hash truth file not found: $CurrentState" }
$VendorRoot = (Resolve-Path -LiteralPath $VendorRoot).Path
$Patch = (Resolve-Path -LiteralPath $Patch).Path
$CurrentState = (Resolve-Path -LiteralPath $CurrentState).Path

$state = Get-Content -LiteralPath $CurrentState -Raw | ConvertFrom-Json
$head = $state.head
if (-not $head) { Fail-Setup "hash truth file has no 'head': $CurrentState" }
$fileEntries = @()
if ($null -ne $state.files) { $fileEntries = @($state.files) }
elseif ($null -ne $state.modifiedFiles) { $fileEntries = @($state.modifiedFiles) }
if ($fileEntries.Count -eq 0) { Fail-Setup "hash truth file lists no files: $CurrentState" }
$expected = @{}
foreach ($entry in $fileEntries) {
  $rel = ([string]$entry.path) -replace '\\', '/'
  $expected[$rel] = [pscustomobject]@{ sha256 = ([string]$entry.sha256).ToLower(); bytes = [int]$entry.bytes }
}

$patchItem = Get-Item -LiteralPath $Patch
$patchHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Patch).Hash.ToLower()
$vendorHead = (& git -C $VendorRoot rev-parse HEAD 2>$null)
if ($LASTEXITCODE -ne 0) { Fail-Setup "git rev-parse failed in $VendorRoot" }
$vendorHead = $vendorHead.Trim()

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$verifyRoot = Join-Path $repo 'build\tmp\vendor_patch_verify'
New-Item -ItemType Directory -Force -Path $verifyRoot | Out-Null
$temp = Join-Path $verifyRoot ("run-{0}-{1}" -f $stamp, ([guid]::NewGuid().ToString('N').Substring(0, 6)))
$resultPath = Join-Path $verifyRoot 'last_result.json'

Write-Host ''
Write-Host '=== vendor_patch_verify ==='
Write-Host ("repo            : {0}" -f $repo)
Write-Host ("vendor          : {0}" -f (Resolve-Path -LiteralPath $VendorRoot).Path)
Write-Host ("live vendor HEAD: {0}" -f $vendorHead)
Write-Host ("target commit   : {0}" -f $head)
if ($vendorHead -ne $head) {
  Write-Host ("WARN            : live vendor HEAD differs from the recorded commit; verifying against the recorded commit anyway") -ForegroundColor Yellow
}
Write-Host ("patch           : {0} ({1} bytes, sha256 {2})" -f $Patch, $patchItem.Length, $patchHash)
Write-Host ("hash truth      : {0}" -f $CurrentState)

$sw = [System.Diagnostics.Stopwatch]::StartNew()
$worktreeAdded = $false
$rows = New-Object System.Collections.Generic.List[object]
$failures = New-Object System.Collections.Generic.List[string]
$statusList = @()

try {
  & git -C $VendorRoot worktree add --detach $temp $head 2>&1 | Out-Null
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $temp)) {
    throw "git worktree add failed (exit $LASTEXITCODE)"
  }
  $worktreeAdded = $true

  & git -C $temp apply --check --whitespace=nowarn $Patch 2>&1 | Out-Null
  if ($LASTEXITCODE -ne 0) { throw "git apply --check failed for $Patch (patch does not apply to $head)" }

  & git -C $temp apply --whitespace=nowarn $Patch 2>&1 | Out-Null
  if ($LASTEXITCODE -ne 0) { throw "git apply failed for $Patch" }

  $statusRaw = & git -C $temp status --porcelain 2>$null
  $statusList = @($statusRaw | Where-Object { $_.Trim() } | ForEach-Object {
      $line = $_.Trim()
      if ($line.Length -gt 2) { ($line.Substring(2).Trim() -replace '\\', '/') } else { '' }
    } | Where-Object { $_ })
  $expectedPaths = @($expected.Keys | Sort-Object)
  $actualPaths = @($statusList | Sort-Object -Unique)
  $missing = @($expectedPaths | Where-Object { $actualPaths -notcontains $_ })
  $extra = @($actualPaths | Where-Object { $expectedPaths -notcontains $_ })
  if ($missing.Count -gt 0) { $failures.Add("files not modified by patch: $($missing -join ', ')") }
  if ($extra.Count -gt 0) { $failures.Add("unexpected extra changes: $($extra -join ', ')") }

  foreach ($rel in $expectedPaths) {
    $full = Join-Path $temp $rel
    if (-not (Test-Path -LiteralPath $full)) {
      $rows.Add([pscustomobject]@{ file = $rel; expected = $expected[$rel].sha256.Substring(0, 16); actual = 'MISSING'; bytes = 0; match = $false })
      $failures.Add("missing file after patch: $rel")
      continue
    }
    $h = (Get-FileHash -Algorithm SHA256 -LiteralPath $full).Hash.ToLower()
    $ok = ($h -eq $expected[$rel].sha256)
    if (-not $ok) { $failures.Add("sha256 mismatch: $rel") }
    $rows.Add([pscustomobject]@{
        file     = $rel
        expected = $expected[$rel].sha256.Substring(0, 16)
        actual   = $h.Substring(0, 16)
        bytes    = (Get-Item -LiteralPath $full).Length
        match    = $ok
      })
  }
} catch {
  $failures.Add($_.Exception.Message)
} finally {
  $sw.Stop()
  if ($worktreeAdded) {
    if ($KeepTemp) {
      Write-Host ("temp worktree kept: {0}" -f $temp)
    } else {
      & git -C $VendorRoot worktree remove --force $temp 2>&1 | Out-Null
      & git -C $VendorRoot worktree prune 2>&1 | Out-Null
    }
  }
}

Write-Host ''
if ($rows.Count -gt 0) {
  Write-Host ('{0,-44} {1,16} {2,16} {3,10} {4}' -f 'file', 'expected(16)', 'actual(16)', 'bytes', 'match')
  Write-Host ('-' * 104)
  foreach ($row in $rows) {
    Write-Host ('{0,-44} {1,16} {2,16} {3,10} {4}' -f $row.file, $row.expected, $row.actual, $row.bytes, $row.match)
  }
}
$pass = ($failures.Count -eq 0)
Write-Host ''
if ($pass) {
  Write-Host ("RESULT: PASS - pristine {0} + {1} reproduces the recorded vendor tree byte-for-byte ({2} files, {3:N1}s)" -f $head.Substring(0, [math]::Min(10, $head.Length)), (Split-Path -Leaf $Patch), $expected.Count, $sw.Elapsed.TotalSeconds) -ForegroundColor Green
} else {
  Write-Host "RESULT: FAIL" -ForegroundColor Red
  foreach ($failure in $failures) { Write-Host ("  - " + $failure) -ForegroundColor Red }
}

$result = [pscustomobject]@{
  ts             = (Get-Date).ToString('o')
  pass           = $pass
  vendorRoot     = (Resolve-Path -LiteralPath $VendorRoot).Path
  liveVendorHead = $vendorHead
  targetHead     = $head
  patch          = $Patch
  patchBytes     = $patchItem.Length
  patchSha256    = $patchHash
  stateFile      = $CurrentState
  elapsedSec     = [math]::Round($sw.Elapsed.TotalSeconds, 2)
  files          = $rows
  failures       = $failures
  tempWorktree   = if ($KeepTemp -and $worktreeAdded) { $temp } else { $null }
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($resultPath, ($result | ConvertTo-Json -Depth 8), $utf8)
Write-Host ("evidence: {0}" -f $resultPath)
if (-not $pass) { exit 1 }
exit 0
