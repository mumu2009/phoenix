<#
.SYNOPSIS
    Turn a WER minidump (build\tmp\dumps\*.dmp) into a readable markdown report.

.DESCRIPTION
    Wraps tools/dump_report.py, which:
      (a) identifies the faulting thread + exception code/address from the dump's
          exception stream and prints that thread's RIP/RSP/RBP,
      (b) extracts plausible return addresses from that thread's stack,
      (c) validates each against the executable's PE64 .pdata unwind table and
          resolves it against the executable's own COFF symbol table (nm -C -n),
      (d) lists the nearest N code symbols below every validated frame.

    Reports are written to build\tmp\dumps\<dumpname>.report.md.
    phoenix_main.map is NOT used by default: the current copy was produced by a
    failed _GLIBCXX_DEBUG link and does not match the shipped binary (see
    doc/v10.1/debug_protocol.md).  Use -Symbols map only for a matching map.

.PARAMETER Dump
    Path to one .dmp (relative to the project root or absolute). If a bare file
    name is given it is looked up in build\tmp\dumps.

.PARAMETER Latest
    Analyse the newest .dmp in build\tmp\dumps.

.PARAMETER All
    Analyse every .dmp in build\tmp\dumps (oldest first).

.PARAMETER Below
    How many nearest symbols below each frame to print (default 3).

.PARAMETER MaxFrames
    Max validated frames to print per dump (default 40).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\dump_report.ps1 -Latest

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\dump_report.ps1 `
        -Dump build\tmp\dumps\phoenix_main.exe.44428.dmp -Below 5

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\dump_report.ps1 -All
#>
[CmdletBinding(DefaultParameterSetName = 'One')]
param(
    [Parameter(ParameterSetName = 'One', Position = 0)]
    [string]$Dump,

    [Parameter(ParameterSetName = 'Latest')]
    [switch]$Latest,

    [Parameter(ParameterSetName = 'All')]
    [switch]$All,

    [string]$Exe = 'phoenix_main.exe',
    [ValidateSet('auto', 'exe', 'map')]
    [string]$Symbols = 'auto',
    [string]$Map,
    [string]$Nm,
    [int]$MaxFrames = 40,
    [int]$Below = 3,
    [string]$OutDir = 'build\tmp\dumps',
    [string]$Python
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot
try {
    $dumpDir = Join-Path $ProjectRoot 'build\tmp\dumps'
    if (-not (Test-Path $dumpDir)) { throw "dump directory not found: $dumpDir" }
    $outDirAbs = if ([IO.Path]::IsPathRooted($OutDir)) { $OutDir } else { Join-Path $ProjectRoot $OutDir }
    if (-not (Test-Path $outDirAbs)) { New-Item -ItemType Directory -Force $outDirAbs | Out-Null }

    if (-not $Python) {
        $candidates = @('python')
        if ($env:PHOENIX_PYTHON) { $candidates = @($env:PHOENIX_PYTHON) + $candidates }
        $candidates += 'D:\cloudworkfile\tools\python311\python.exe'
        foreach ($cand in $candidates) {
            try {
                $ver = & $cand -c "import sys,minidump; print(sys.version.split()[0])" 2>$null
                if ($LASTEXITCODE -eq 0 -and $ver) { $Python = $cand; break }
            } catch { }
        }
        if (-not $Python) {
            throw "No Python with the 'minidump' package found. Install: python -m pip install minidump"
        }
    }

    $dumps = @()
    if ($All) {
        $dumps = @(Get-ChildItem $dumpDir -Filter '*.dmp' -File | Sort-Object LastWriteTime)
    } elseif ($Latest) {
        $dumps = @(Get-ChildItem $dumpDir -Filter '*.dmp' -File | Sort-Object LastWriteTime -Descending | Select-Object -First 1)
    } elseif ($Dump) {
        $resolved = if (Test-Path $Dump) { (Resolve-Path $Dump).Path }
                    elseif (Test-Path (Join-Path $dumpDir $Dump)) { (Resolve-Path (Join-Path $dumpDir $Dump)).Path }
                    else { $null }
        if (-not $resolved) { throw "dump not found: $Dump (looked in cwd and $dumpDir)" }
        $dumps = @(Get-Item $resolved)
    } else {
        throw "Nothing selected. Pass -Dump <file>, or -Latest, or -All."
    }
    if ($dumps.Count -eq 0) { throw "no .dmp files found in $dumpDir" }

    $reportScript = Join-Path $PSScriptRoot 'dump_report.py'
    if (-not (Test-Path $reportScript)) { throw "missing $reportScript" }

    $results = @()
    $failed = 0
    foreach ($d in $dumps) {
        $outFile = Join-Path $outDirAbs ($d.BaseName + '.report.md')
        $pyArgs = @($reportScript, $d.FullName, '--exe', $Exe, '--symbols', $Symbols,
                    '--max-frames', "$MaxFrames", '--below', "$Below", '--out', $outFile)
        if ($Map) { $pyArgs += @('--map', $Map) }
        if ($Nm) { $pyArgs += @('--nm', $Nm) }
        Write-Host ("[dump_report] {0} ({1:N1} MiB) ..." -f $d.Name, ($d.Length / 1MB))
        & $Python @pyArgs 2>&1 | ForEach-Object { Write-Host "  $_" }
        if ($LASTEXITCODE -ne 0) {
            $failed++
            Write-Host ("[dump_report] FAILED {0} (exit {1})" -f $d.Name, $LASTEXITCODE) -ForegroundColor Red
            continue
        }
        # pull a compact summary out of the generated markdown
        $text = Get-Content $outFile -Raw
        $exc = if ($text -match 'Code: \*\*(0x[0-9A-F]+)\s+([A-Z0-9_]+)\*\*') { "$($Matches[1]) $($Matches[2])" } else { 'n/a' }
        $tid = if ($text -match 'TID: (\d+)') { $Matches[1] } else { '?' }
        $frames = if ($text -match '(\d+) lie inside a real function range') { $Matches[1] } else { '0' }
        $results += [pscustomobject]@{
            Dump     = $d.Name
            SizeMiB  = [math]::Round($d.Length / 1MB, 1)
            Exception = $exc
            ThreadId = $tid
            Frames   = [int]$frames
            Report   = $outFile
        }
    }

    if ($results.Count -gt 0) {
        Write-Host ""
        Write-Host "=== dump reports ===" -ForegroundColor Cyan
        $results | Format-Table Dump, SizeMiB, Exception, ThreadId, Frames -AutoSize
        Write-Host "reports: $outDirAbs"
        Write-Host "symbol source: exe COFF via nm (phoenix_main.map is stale; see doc/v10.1/debug_protocol.md)"
    }
    if ($failed -gt 0) { exit 1 }
} finally {
    Pop-Location
}
