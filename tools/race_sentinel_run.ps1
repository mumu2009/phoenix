#requires -Version 7.0
<#
.SYNOPSIS
    RaceSentinel-MinGW Phase 1 driver (doc/v10.1/methods_backlog.md M1).

.DESCRIPTION
    Builds a *sidecar* gtest runner for tests/gtest/unit/race/test_race_sentinel.cpp
    (standalone: no product sources, no compile.bat, no compile_gtest.bat), then runs
    the RaceSentinel-MinGW experiment:

      * positive controls : 4 race patterns x N seeds  -> detection rate / latency
      * negative controls : same 4 patterns, properly locked -> false positives
      * overhead          : detector armed vs suspended on the locked workloads

    and writes build/tmp/race_sentinel/report.json + report.md (machine + human
    readable), plus every raw log/part file next to them.

    Isolation / safety:
      * never runs compile.bat or compile_gtest.bat (they share build/tmp with the
        live sidecar builds and compile.bat taskkills the live gateway/llama-servers);
      * the g++ invocation gets its own TMP/TEMP under build/tmp/race_sentinel/tmp,
        so it cannot race with build/tmp/cc* cleanup of a concurrent compile;
      * writes only under build/tmp/race_sentinel/.

.PARAMETER SeedCount
    Seeds per pattern for both positive and negative matrices (default 20).

.PARAMETER PositiveWindowMs
    Per-run observation window for a positive control (default 500 ms; the run
    stops early as soon as a race is reported).

.PARAMETER NegativeWindowMs
    Per-run observation window for a negative control (default 2500 ms). Kept well
    below 60 s so each individual control run stays short.

.PARAMETER NegativeParallel
    Number of concurrent negative-control processes (default 4). Aggregate
    exposure = NegativeParallel streams x runs x window.

.PARAMETER Sample
    Detector 1/N sampling rate (RACE_SENTINEL_SAMPLE, default 16).

.PARAMETER SoakMinutes
    Optional: after the normal matrix, keep running negative controls until the
    aggregate negative exposure reaches SoakMinutes minutes of instrumented
    workload time. 95% certification of "<=1 false positive per 30 min" with zero
    observed FPs needs >= 90 min of aggregate exposure (rule of three).

.PARAMETER SkipBuild
    Reuse build/tmp/race_sentinel/race_sentinel_tests.exe if it exists.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\race_sentinel_run.ps1

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\race_sentinel_run.ps1 -SoakMinutes 90
#>
[CmdletBinding()]
param(
    [int]$SeedCount = 20,
    [int]$PositiveWindowMs = 500,
    [int]$NegativeWindowMs = 2500,
    [int]$NegativeParallel = 4,
    [int]$Sample = 16,
    [int]$OverheadReps = 3,
    [int]$OverheadOps = 200000,
    [int]$SoakMinutes = 0,
    [switch]$SkipBuild,
    [string]$Gcc = 'C:\msys64\mingw64\bin\g++.exe',
    [string]$OutDir = 'build\tmp\race_sentinel'
)

$ErrorActionPreference = 'Stop'
$Inv = [System.Globalization.CultureInfo]::InvariantCulture

function Fmt([double]$v, [string]$f) { return $v.ToString($f, $Inv) }

function Get-Median([double[]]$values) {
    if (-not $values -or $values.Count -eq 0) { return $null }
    $s = @($values | Sort-Object)
    return $s[[int][math]::Floor(($s.Count - 1) / 2)]
}

function Get-Percentile([double[]]$values, [double]$p) {
    if (-not $values -or $values.Count -eq 0) { return $null }
    $s = @($values | Sort-Object)
    $idx = [int][math]::Ceiling($p * $s.Count) - 1
    if ($idx -lt 0) { $idx = 0 }
    if ($idx -ge $s.Count) { $idx = $s.Count - 1 }
    return $s[$idx]
}

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
$Root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrEmpty($Root)) { $Root = (Get-Location).Path }
Set-Location $Root

if (-not [System.IO.Path]::IsPathRooted($OutDir)) { $OutDir = Join-Path $Root $OutDir }
$TmpDir = Join-Path $OutDir 'tmp'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
New-Item -ItemType Directory -Force -Path $TmpDir | Out-Null

$Exe = Join-Path $OutDir 'race_sentinel_tests.exe'
$BuildLog = Join-Path $OutDir 'build.log'
$TestCpp = Join-Path $Root 'tests\gtest\unit\race\test_race_sentinel.cpp'
$MainCpp = Join-Path $Root 'tests\gtest\gtest_main.cpp'

if ($Root -match '\s') {
    Write-Warning "project path contains spaces; the sidecar process launcher assumes it does not ($Root)"
}

Write-Host "=== RaceSentinel-MinGW Phase 1 experiment ==="
$scriptStart = Get-Date
Write-Host "root      : $Root"
Write-Host "out       : $OutDir"
Write-Host "sampling  : 1/$Sample (RACE_SENTINEL=1)"
Write-Host "seeds     : $SeedCount per pattern (4 patterns => $($SeedCount * 4) positive runs)"
Write-Host "negative  : $NegativeParallel parallel streams x $SeedCount seeds x 4 patterns x $NegativeWindowMs ms"

# ---------------------------------------------------------------------------
# Locate the conan gtest package (headers + static lib) without conan/build.bat
# ---------------------------------------------------------------------------
function Find-GtestInclude {
    $cflags = Join-Path $Root 'build\conan_cflags_gtest.txt'
    if (Test-Path $cflags) {
        $tokens = (Get-Content $cflags -Raw) -split '\s+'
        $hit = $tokens | Where-Object { $_ -like '-I*' -and $_ -match 'gtest' -and $_ -match 'p[/\\]include$' } | Select-Object -Last 1
        if ($hit) {
            $inc = $hit.Substring(2) -replace '/', '\'
            if (Test-Path (Join-Path $inc 'gtest\gtest.h')) { return $inc }
        }
    }
    $pkgRoot = Join-Path $Root 'build\conan_home\p'
    if (Test-Path $pkgRoot) {
        $hits = Get-ChildItem $pkgRoot -Recurse -Filter gtest.h -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match '\\p\\include\\gtest\\gtest\.h$' }
        if ($hits) { return (Split-Path (Split-Path $hits[0].FullName -Parent) -Parent) }
    }
    return $null
}

# ---------------------------------------------------------------------------
# Build the sidecar gtest runner
# ---------------------------------------------------------------------------
$BuildInfo = [ordered]@{ ok = $false; command = ''; log = $BuildLog; exe = $Exe; gtest_include = ''; gtest_lib = '' }

function Invoke-Build {
    param([string]$Inc, [string]$Lib)
    $gxx = $Gcc
    if (-not (Test-Path $gxx)) {
        $cmdObj = Get-Command g++ -ErrorAction SilentlyContinue
        if ($cmdObj) { $gxx = $cmdObj.Source } else { throw "g++ not found (looked at '$Gcc' and PATH)." }
    }
    $buildArgs = @(
        '-o', $Exe,
        '-std=c++20', '-O2', '-g1', '-Wall', '-Wextra', '-Wno-unused-parameter',
        '-I', (Join-Path $Root 'tests\gtest'),
        '-I', $Inc,
        $TestCpp, $MainCpp,
        '-L', $Lib, '-lgtest', '-lpthread'
    )
    $display = "$gxx " + ($buildArgs -join ' ')
    $oldTmp = $env:TMP
    $oldTemp = $env:TEMP
    try {
        $env:TMP = $TmpDir
        $env:TEMP = $TmpDir
        "# RaceSentinel sidecar build $(Get-Date -Format s)" | Set-Content -Path $BuildLog -Encoding utf8
        "# command: $display" | Add-Content -Path $BuildLog -Encoding utf8
        & $gxx @buildArgs *>> $BuildLog
        $code = $LASTEXITCODE
    } finally {
        $env:TMP = $oldTmp
        $env:TEMP = $oldTemp
    }
    $BuildInfo.ok = ($code -eq 0 -and (Test-Path $Exe))
    $BuildInfo.command = $display
    return $BuildInfo.ok
}

$gtestInc = Find-GtestInclude
if (-not $gtestInc) {
    throw "Could not locate the conan gtest package under build\conan_home (headers). Run compile_gtest.bat once when no compile.bat is running, then retry."
}
$gtestPkg = Split-Path $gtestInc -Parent
$gtestLib = Join-Path $gtestPkg 'lib'
if (-not (Test-Path (Join-Path $gtestLib 'libgtest.a'))) {
    throw "Could not locate libgtest.a under '$gtestLib'."
}
$BuildInfo.gtest_include = $gtestInc
$BuildInfo.gtest_lib = $gtestLib

if ($SkipBuild -and (Test-Path $Exe)) {
    Write-Host "[build] skipped (-SkipBuild), reusing $Exe"
    $BuildInfo.ok = $true
} else {
    Write-Host "[build] compiling sidecar gtest runner ..."
    $ok = Invoke-Build -Inc $gtestInc -Lib $gtestLib
    if (-not $ok) {
        Get-Content $BuildLog -ErrorAction SilentlyContinue | Select-Object -Last 30 | ForEach-Object { Write-Host $_ }
        throw "sidecar gtest build failed (see $BuildLog)."
    }
    Write-Host "[build] ok"
}

# ---------------------------------------------------------------------------
# Process runner
# ---------------------------------------------------------------------------
$BaseEnv = @{
    'RACE_SENTINEL'        = '1'
    'RACE_SENTINEL_SAMPLE' = "$Sample"
}

function Merge-Env {
    param([hashtable]$Base, [hashtable]$Extra)
    $merged = @{}
    foreach ($k in $Base.Keys) { $merged[$k] = $Base[$k] }
    foreach ($k in $Extra.Keys) { $merged[$k] = $Extra[$k] }
    return $merged
}

function Invoke-RaceRun {
    param(
        [string]$Name,
        [hashtable]$ExtraEnv,
        [int]$TimeoutSec
    )
    $log = Join-Path $OutDir ("{0}.log" -f $Name)
    $errLog = Join-Path $OutDir ("{0}.err" -f $Name)
    $envAll = Merge-Env $BaseEnv $ExtraEnv
    $gtArgs = @('--gtest_filter=RaceSentinelExperiment.Matrix')
    $started = Get-Date
    $proc = Start-Process -FilePath $Exe -ArgumentList $gtArgs `
        -RedirectStandardOutput $log -RedirectStandardError $errLog `
        -NoNewWindow -PassThru -Environment $envAll
    $timedOut = $false
    if (-not $proc.WaitForExit($TimeoutSec * 1000)) {
        $timedOut = $true
        try { Stop-Process -Id $proc.Id -Force } catch { }
    }
    $exitCode = if ($timedOut) { -1 } else { $proc.ExitCode }
    $elapsed = ((Get-Date) - $started).TotalSeconds

    $data = $null
    $outPath = $envAll['RACE_SENTINEL_OUT']
    if ($outPath -and (Test-Path $outPath)) {
        try {
            $raw = Get-Content $outPath -Raw
            if ($raw -and $raw.Trim().Length -gt 0) { $data = $raw | ConvertFrom-Json }
        } catch {
            $data = $null
        }
    }
    return [pscustomobject]@{
        name     = $Name
        ok       = ($exitCode -eq 0 -and -not $timedOut -and $null -ne $data)
        exit     = $exitCode
        timedOut = $timedOut
        elapsed  = $elapsed
        log      = $log
        data     = $data
    }
}

function New-PartPath([string]$Name) { return (Join-Path $OutDir ("part_{0}.json" -f $Name)) }

# ---------------------------------------------------------------------------
# 1) Correctness tests (positives must be flagged, negatives must be silent)
# ---------------------------------------------------------------------------
Write-Host "[1/4] correctness tests (8 controls + 12 unit tests) ..."
$corrLog = Join-Path $OutDir 'correctness.log'
$corrErr = Join-Path $OutDir 'correctness.err'
$corrProc = Start-Process -FilePath $Exe -ArgumentList @('--gtest_filter=RaceSentinel*') `
    -RedirectStandardOutput $corrLog -RedirectStandardError $corrErr `
    -NoNewWindow -PassThru -Environment (Merge-Env $BaseEnv @{})
$corrTimedOut = -not $corrProc.WaitForExit(180000)
if ($corrTimedOut) { try { Stop-Process -Id $corrProc.Id -Force } catch { } }
$corrExit = if ($corrTimedOut) { -1 } else { $corrProc.ExitCode }
$corrText = if (Test-Path $corrLog) { Get-Content $corrLog -Raw } else { '' }
$corrPassed = 0
$corrFailed = 0
$corrSkipped = 0
if ($corrText -match '\[  PASSED  \]\s+(\d+)\s+tests?') { $corrPassed = [int]$Matches[1] }
if ($corrText -match '\[  FAILED  \]\s+(\d+)\s+tests?') { $corrFailed = [int]$Matches[1] }
if ($corrText -match '\[  SKIPPED \]\s+(\d+)\s+tests?') { $corrSkipped = [int]$Matches[1] }
$corrOk = ($corrExit -eq 0 -and $corrFailed -eq 0 -and $corrPassed -ge 18)
Write-Host ("[1/4] correctness: exit={0} passed={1} failed={2} skipped={3} ok={4}" -f $corrExit, $corrPassed, $corrFailed, $corrSkipped, $corrOk)
if (-not $corrOk) {
    $corrText -split "`r?`n" | Where-Object { $_ -match 'FAILED|\[  RUN|error' } | Select-Object -Last 20 | ForEach-Object { Write-Host $_ }
}

# ---------------------------------------------------------------------------
# 2) Overhead measurement (detector armed vs suspended, same locked workloads)
# ---------------------------------------------------------------------------
Write-Host "[2/4] overhead measurement ..."
$ovhRun = Invoke-RaceRun -Name 'overhead' -ExtraEnv @{
    'RACE_SENTINEL_MODE'  = 'overhead'
    'RACE_SENTINEL_REPS'  = "$OverheadReps"
    'RACE_SENTINEL_OPS'   = "$OverheadOps"
    'RACE_SENTINEL_OUT'   = (New-PartPath 'overhead')
} -TimeoutSec 600
if (-not $ovhRun.ok) { throw "overhead run failed (exit=$($ovhRun.exit), timedOut=$($ovhRun.timedOut)); see $($ovhRun.log)" }
$ovhPatterns = @($ovhRun.data.patterns)
$ovhRatios = @($ovhPatterns | ForEach-Object { [double]$_.ratio })
$ovhGeomean = if ($ovhRatios.Count -gt 0) {
    $logs = $ovhRatios | Where-Object { $_ -gt 0 } | ForEach-Object { [math]::Log($_) }
    [math]::Exp(($logs | Measure-Object -Average).Average)
} else { 0.0 }
$ovhMedian = Get-Median $ovhRatios
$ovhPass = ($ovhGeomean -gt 0 -and $ovhGeomean -le 5.0)
Write-Host ("[2/4] overhead: geomean={0}x median={1}x pass={2}" -f (Fmt $ovhGeomean '0.00'), (Fmt $ovhMedian '0.00'), $ovhPass)

# ---------------------------------------------------------------------------
# 3) Positive matrix (detection rate over 4 patterns x SeedCount seeds)
# ---------------------------------------------------------------------------
Write-Host "[3/4] positive controls ($($SeedCount * 4) runs) ..."
$posRun = Invoke-RaceRun -Name 'positive' -ExtraEnv @{
    'RACE_SENTINEL_MODE'       = 'positive'
    'RACE_SENTINEL_SEED_FIRST' = '0'
    'RACE_SENTINEL_SEED_COUNT' = "$SeedCount"
    'RACE_SENTINEL_WINDOW_MS'  = "$PositiveWindowMs"
    'RACE_SENTINEL_OUT'        = (New-PartPath 'positive')
} -TimeoutSec 300
if (-not $posRun.ok) { throw "positive matrix failed (exit=$($posRun.exit), timedOut=$($posRun.timedOut)); see $($posRun.log)" }
$posRuns = @($posRun.data.runs)

$patternIds = @('A', 'B', 'C', 'D')
$patternNames = @{}
foreach ($r in $posRuns) { $patternNames[$r.pattern] = $r.pattern_name }

$posPer = foreach ($pat in $patternIds) {
    $rs = @($posRuns | Where-Object { $_.pattern -eq $pat })
    $det = @($rs | Where-Object { $_.detected }).Count
    $lat = @($rs | Where-Object { $_.detected } | ForEach-Object { [double]$_.first_report_ms })
    $sampled = (@($rs | Measure-Object sampled -Sum).Sum)
    [pscustomobject]@{
        pattern       = $pat
        name          = $patternNames[$pat]
        runs          = $rs.Count
        detected      = $det
        detection_rate = if ($rs.Count -gt 0) { $det / $rs.Count } else { 0 }
        latency_median_ms = if ($lat.Count -gt 0) { Get-Median $lat } else { $null }
        latency_p90_ms    = if ($lat.Count -gt 0) { Get-Percentile $lat 0.9 } else { $null }
        latency_max_ms    = if ($lat.Count -gt 0) { ($lat | Measure-Object -Maximum).Maximum } else { $null }
        sampled_accesses  = $sampled
        sample_reports    = @($rs | Where-Object { $_.sample_report } | ForEach-Object { $_.sample_report } | Select-Object -First 3)
        pass          = ($det -ge [math]::Ceiling(0.9 * $rs.Count))
    }
}
$posTotalRuns = @($posRuns).Count
$posDetected = @($posRuns | Where-Object { $_.detected }).Count
$posRate = if ($posTotalRuns -gt 0) { $posDetected / $posTotalRuns } else { 0 }
$posLatAll = @($posRuns | Where-Object { $_.detected } | ForEach-Object { [double]$_.first_report_ms })
$posDetPass = ($posRate -ge 0.9) -and (-not ($posPer | Where-Object { -not $_.pass }))
$posMissing = @($posRuns | Where-Object { -not $_.detected })
Write-Host ("[3/4] positive: detected {0}/{1} ({2}%) pass={3}" -f $posDetected, $posTotalRuns, (Fmt ($posRate * 100) '0.0'), $posDetPass)

# ---------------------------------------------------------------------------
# 4) Negative matrix in parallel streams (false positives)
# ---------------------------------------------------------------------------
Write-Host "[4/4] negative controls ($NegativeParallel parallel streams) ..."
$negParallel = [math]::Max(1, $NegativeParallel)
$perStream = [int][math]::Floor($SeedCount / $negParallel)
$remainder = $SeedCount - ($perStream * $negParallel)
$negJobs = @()
$sFirst = 0
for ($i = 0; $i -lt $negParallel; ++$i) {
    $cnt = $perStream + $(if ($i -lt $remainder) { 1 } else { 0 })
    if ($cnt -le 0) { continue }
    $expected = ($cnt * 4 * $NegativeWindowMs) / 1000.0
    $timeout = [int][math]::Ceiling($expected + 240)
    $name = "negative_s$($i + 1)"
    $job = [pscustomobject]@{
        name      = $name
        extraEnv  = @{
            'RACE_SENTINEL_MODE'       = 'negative'
            'RACE_SENTINEL_SEED_FIRST' = "$sFirst"
            'RACE_SENTINEL_SEED_COUNT' = "$cnt"
            'RACE_SENTINEL_WINDOW_MS'  = "$NegativeWindowMs"
            'RACE_SENTINEL_OUT'        = (New-PartPath $name)
        }
        timeout   = $timeout
    }
    $negJobs += $job
    $sFirst += $cnt
}
$negResults = @()
foreach ($job in $negJobs) {
    $log = Join-Path $OutDir ("{0}.log" -f $job.name)
    $errLog = Join-Path $OutDir ("{0}.err" -f $job.name)
    $envAll = Merge-Env $BaseEnv $job.extraEnv
    $p = Start-Process -FilePath $Exe -ArgumentList @('--gtest_filter=RaceSentinelExperiment.Matrix') `
        -RedirectStandardOutput $log -RedirectStandardError $errLog `
        -NoNewWindow -PassThru -Environment $envAll
    $negResults += [pscustomobject]@{ job = $job; proc = $p; log = $log }
}
$negRuns = @()
foreach ($nr in $negResults) {
    if (-not $nr.proc.WaitForExit($nr.job.timeout * 1000)) {
        try { Stop-Process -Id $nr.proc.Id -Force } catch { }
        throw "negative stream $($nr.job.name) timed out; see $($nr.log)"
    }
    if ($nr.proc.ExitCode -ne 0) { throw "negative stream $($nr.job.name) failed (exit=$($nr.proc.ExitCode)); see $($nr.log)" }
    $outPath = $nr.job.extraEnv['RACE_SENTINEL_OUT']
    if (-not (Test-Path $outPath)) { throw "negative stream $($nr.job.name) produced no JSON" }
    $data = Get-Content $outPath -Raw | ConvertFrom-Json
    $negRuns += @($data.runs)
}
$negRuns = @($negRuns)

# Optional long soak for false-positive certification.
if ($SoakMinutes -gt 0) {
    $targetExposure = $SoakMinutes * 60.0
    $exposure = 0.0
    foreach ($r in $negRuns) { $exposure += [double]$r.wall_ms / 1000.0 }
    $round = 0
    $seedsPerChunk = [int][math]::Max(1, [math]::Floor(60.0 / (($NegativeWindowMs / 1000.0) * 4.0)))
    Write-Host ("[soak] extra negative soak: target {0} min aggregate exposure (current {1:N1} min), chunk = {2} seeds x 4 patterns" -f $SoakMinutes, ($exposure / 60.0), $seedsPerChunk)
    while ($exposure -lt $targetExposure) {
        $round++
        $jobs = @()
        for ($i = 0; $i -lt $negParallel; ++$i) {
            $name = "soak_r$($round)_s$($i + 1)"
            $jobs += [pscustomobject]@{
                name     = $name
                extraEnv = @{
                    'RACE_SENTINEL_MODE'       = 'negative'
                    'RACE_SENTINEL_SEED_FIRST' = "$(($round * $negParallel + $i) * $seedsPerChunk)"
                    'RACE_SENTINEL_SEED_COUNT' = "$seedsPerChunk"
                    'RACE_SENTINEL_WINDOW_MS'  = "$NegativeWindowMs"
                    'RACE_SENTINEL_OUT'        = (New-PartPath $name)
                }
                timeout  = [int][math]::Ceiling($seedsPerChunk * 4 * $NegativeWindowMs / 1000.0 + 240)
            }
        }
        $procs = @()
        foreach ($job in $jobs) {
            $log = Join-Path $OutDir ("{0}.log" -f $job.name)
            $errLog = Join-Path $OutDir ("{0}.err" -f $job.name)
            $procs += [pscustomobject]@{
                job  = $job
                proc = (Start-Process -FilePath $Exe -ArgumentList @('--gtest_filter=RaceSentinelExperiment.Matrix') `
                            -RedirectStandardOutput $log -RedirectStandardError $errLog `
                            -NoNewWindow -PassThru -Environment (Merge-Env $BaseEnv $job.extraEnv))
                log  = $log
            }
        }
        foreach ($pr in $procs) {
            if (-not $pr.proc.WaitForExit($pr.job.timeout * 1000)) {
                try { Stop-Process -Id $pr.proc.Id -Force } catch { }
                throw "soak stream $($pr.job.name) timed out; see $($pr.log)"
            }
            if ($pr.proc.ExitCode -ne 0) { throw "soak stream $($pr.job.name) failed; see $($pr.log)" }
            $data = Get-Content $pr.job.extraEnv['RACE_SENTINEL_OUT'] -Raw | ConvertFrom-Json
            $negRuns += @($data.runs)
            foreach ($r in @($data.runs)) { $exposure += [double]$r.wall_ms / 1000.0 }
        }
        Write-Host ("[soak] round {0}: aggregate exposure {1:N1} min / {2} min" -f $round, ($exposure / 60.0), $SoakMinutes)
    }
    $negRuns = @($negRuns)
}

$negTotalRuns = @($negRuns).Count
$negTotalReports = (@($negRuns | Measure-Object reports -Sum).Sum)
if ($null -eq $negTotalReports) { $negTotalReports = 0 }
$negRunsWithReports = @($negRuns | Where-Object { $_.reports -gt 0 }).Count
$negExposure = 0.0
foreach ($r in $negRuns) { $negExposure += [double]$r.wall_ms / 1000.0 }
$negFpPoint = if ($negExposure -gt 0) { $negTotalReports * 1800.0 / $negExposure } else { 0.0 }
$negFpUpper = if ($negExposure -gt 0) { 3.0 * 1800.0 / $negExposure } else { $null }

$negPer = foreach ($pat in $patternIds) {
    $rs = @($negRuns | Where-Object { $_.pattern -eq $pat })
    $reports = (@($rs | Measure-Object reports -Sum).Sum)
    if ($null -eq $reports) { $reports = 0 }
    $exp = 0.0
    foreach ($r in $rs) { $exp += [double]$r.wall_ms / 1000.0 }
    [pscustomobject]@{
        pattern       = $pat
        name          = $patternNames[$pat]
        runs          = $rs.Count
        runs_with_reports = @($rs | Where-Object { $_.reports -gt 0 }).Count
        reports       = $reports
        exposure_seconds = $exp
        sampled_accesses = (@($rs | Measure-Object sampled -Sum).Sum)
    }
}
$negPass = ($negTotalReports -eq 0)
Write-Host ("[4/4] negative: runs={0} reports={1} runs_with_reports={2} exposure={3:N1}s" -f $negTotalRuns, $negTotalReports, $negRunsWithReports, $negExposure)

# ---------------------------------------------------------------------------
# Aggregate + write report.json / report.md
# ---------------------------------------------------------------------------
$gitHead = ''
try { $gitHead = (& git rev-parse HEAD 2>$null) } catch { $gitHead = '' }
$osInfo = [System.Runtime.InteropServices.RuntimeInformation]::OSDescription
$coreCount = [Environment]::ProcessorCount
$gxxVersion = ''
try { $gxxVersion = ((& $Gcc --version 2>$null) | Select-Object -First 1) } catch { $gxxVersion = '' }
$generated = (Get-Date).ToString('yyyy-MM-ddTHH:mm:ssK')

$verdict = if ($corrOk -and $posDetPass -and $negPass -and $ovhPass -and ($negFpPoint -le 1.0)) { 'PASS' } else { 'FAIL' }
$experimentWallSeconds = [math]::Round(((Get-Date) - $scriptStart).TotalSeconds, 1)

$report = [ordered]@{
    experiment      = 'RaceSentinel-MinGW Phase 1 (doc/v10.1/methods_backlog.md M1)'
    generated       = $generated
    experiment_wall_seconds = $experimentWallSeconds
    host            = [ordered]@{
        os              = $osInfo
        cpu_count       = $coreCount
        project_root    = $Root
        git_head        = "$gitHead".Trim()
        gxx             = $gxxVersion
    }
    detector        = [ordered]@{
        header      = 'tests/gtest/support/race_track.hpp'
        arming      = 'RACE_SENTINEL=1'
        algorithm   = 'Eraser lockset + sampling (LiteRace style) + coarse spawn/join epochs'
        sampling    = $Sample
        source_test = 'tests/gtest/unit/race/test_race_sentinel.cpp'
    }
    build           = [ordered]@{
        ok      = $BuildInfo.ok
        exe     = $Exe
        command = $BuildInfo.command
        log     = $BuildLog
        gtest_include = $BuildInfo.gtest_include
        gtest_lib     = $BuildInfo.gtest_lib
    }
    correctness     = [ordered]@{
        exit    = $corrExit
        passed  = $corrPassed
        failed  = $corrFailed
        skipped = $corrSkipped
        ok      = $corrOk
        log     = $corrLog
    }
    config          = [ordered]@{
        seed_count          = $SeedCount
        positive_window_ms  = $PositiveWindowMs
        negative_window_ms  = $NegativeWindowMs
        negative_parallel   = $negParallel
        overhead_reps       = $OverheadReps
        overhead_ops        = $OverheadOps
        soak_minutes        = $SoakMinutes
    }
    positive = [ordered]@{
        runs            = $posTotalRuns
        detected        = $posDetected
        detection_rate  = $posRate
        per_pattern     = @($posPer)
        latency_median_ms = if ($posLatAll.Count -gt 0) { Get-Median $posLatAll } else { $null }
        latency_p90_ms    = if ($posLatAll.Count -gt 0) { Get-Percentile $posLatAll 0.9 } else { $null }
        latency_max_ms    = if ($posLatAll.Count -gt 0) { ($posLatAll | Measure-Object -Maximum).Maximum } else { $null }
        missed_runs     = @($posMissing)
        pass            = $posDetPass
        pass_rule       = 'detection >= 90% overall and per pattern (>=18/20 seeds)'
        log             = $posRun.log
    }
    negative = [ordered]@{
        runs                  = $negTotalRuns
        false_positive_reports = $negTotalReports
        runs_with_reports     = $negRunsWithReports
        exposure_seconds      = $negExposure
        fp_per_30min_observed = $negFpPoint
        fp_per_30min_rule_of_three_upper95 = $negFpUpper
        per_pattern           = @($negPer)
        pass                  = $negPass
        pass_rule             = 'observed false positives = 0 (point estimate <= 1 per 30 min)'
        certification_note    = 'rule-of-three 95% upper bound needs >= 90 min aggregate exposure with 0 FPs; run -SoakMinutes 90 for that'
    }
    overhead = [ordered]@{
        per_pattern      = @($ovhPatterns)
        median_multiplier = $ovhMedian
        geomean_multiplier = $ovhGeomean
        pass             = $ovhPass
        pass_rule        = 'overhead <= 5x uninstrumented on the locked workloads'
        log              = $ovhRun.log
    }
    criteria = [ordered]@{
        correctness_ok       = $corrOk
        detection_ge_90pct   = $posDetPass
        fp_le_1_per_30min    = $negPass -and ($negFpPoint -le 1.0)
        overhead_le_5x       = $ovhPass
    }
    verdict = $verdict
    reports_dir = $OutDir
    repeat_command = 'powershell -NoProfile -ExecutionPolicy Bypass -File tools\race_sentinel_run.ps1'
}

$ReportJson = Join-Path $OutDir 'report.json'
$report | ConvertTo-Json -Depth 12 | Set-Content -Path $ReportJson -Encoding utf8

# ---- report.md ------------------------------------------------------------
$sb = [System.Text.StringBuilder]::new()
function Add-Line([string]$text) { [void]$sb.AppendLine($text) }

Add-Line '# RaceSentinel-MinGW -- Phase 1 experiment report'
Add-Line ''
Add-Line ('Generated: `{0}`  (experiment wall time {1} s)' -f $generated, $experimentWallSeconds)
Add-Line ''
Add-Line ('**Verdict: {0}**' -f $verdict)
Add-Line ''
Add-Line '| Criterion | Measured | Pass rule | Result |'
Add-Line '|---|---|---|---|'
Add-Line ('| Positive-control detection rate | {0}/{1} = {2}% | >= 90% overall and per pattern | {3} |' -f $posDetected, $posTotalRuns, (Fmt ($posRate * 100) '0.0'), $(if ($posDetPass) { 'PASS' } else { 'FAIL' }))
Add-Line ('| False positives (negative controls) | {0} reports over {1:N1} s exposure | <= 1 per 30 min (observed 0) | {2} |' -f $negTotalReports, $negExposure, $(if ($negPass) { 'PASS' } else { 'FAIL' }))
Add-Line ('| Overhead multiplier | geomean {0}x / median {1}x | <= 5x uninstrumented | {2} |' -f (Fmt $ovhGeomean '0.00'), (Fmt $ovhMedian '0.00'), $(if ($ovhPass) { 'PASS' } else { 'FAIL' }))
Add-Line ('| Harness correctness tests | {0} passed / {1} failed / {2} skipped | all pass | {3} |' -f $corrPassed, $corrFailed, $corrSkipped, $(if ($corrOk) { 'PASS' } else { 'FAIL' }))
Add-Line ''
Add-Line '## 1. Positive controls (races that MUST be detected)'
Add-Line ''
Add-Line '| Pattern | Reproduced race | Detected | Rate | median first report | p90 | max |'
Add-Line '|---|---|---|---|---|---|---|'
foreach ($p in $posPer) {
    $med = if ($null -ne $p.latency_median_ms) { (Fmt $p.latency_median_ms '0.00') + ' ms' } else { '-' }
    $p90 = if ($null -ne $p.latency_p90_ms) { (Fmt $p.latency_p90_ms '0.00') + ' ms' } else { '-' }
    $mx = if ($null -ne $p.latency_max_ms) { (Fmt $p.latency_max_ms '0.00') + ' ms' } else { '-' }
    Add-Line ('| {0} | {1} | {2}/{3} | {4}% | {5} | {6} | {7} |' -f $p.pattern, $p.name, $p.detected, $p.runs, (Fmt ($p.detection_rate * 100) '0.0'), $med, $p90, $mx)
}
Add-Line ''
Add-Line ('Aggregate: **{0}/{1} runs detected ({2}%)**, detector sampling 1/{3}.' -f $posDetected, $posTotalRuns, (Fmt ($posRate * 100) '0.0'), $Sample)
Add-Line ''
Add-Line 'Example reports (previous access -> current access):'
foreach ($p in $posPer) {
    if ($p.sample_reports.Count -gt 0) {
        Add-Line ('- pattern {0}: `{1}`' -f $p.pattern, ($p.sample_reports -join '` / `'))
    }
}
Add-Line ''
Add-Line '## 2. Negative controls (properly locked; must NOT be flagged)'
Add-Line ''
Add-Line '| Pattern | Runs | Runs with any report | Reports | Instrumented window | Sampled accesses |'
Add-Line '|---|---|---|---|---|---|'
foreach ($p in $negPer) {
    Add-Line ('| {0} | {1} | {2} | {3} | {4:N1} s | {5} |' -f $p.pattern, $p.runs, $p.runs_with_reports, $p.reports, $p.exposure_seconds, $p.sampled_accesses)
}
Add-Line ''
Add-Line ('Totals: {0} runs, **{1} false-positive reports**, {2:N1} s of instrumented negative-control exposure.' -f $negTotalRuns, $negTotalReports, $negExposure)
Add-Line ''
Add-Line 'False positives per 30 min:'
Add-Line ''
Add-Line ('- observed point estimate: **{0}** (0 reports / {1:N1} s) -> passes "<= 1 per 30 min".' -f (Fmt $negFpPoint '0.00'), $negExposure)
Add-Line ('- rule-of-three 95% upper bound: **{0}** per 30 min (3 / exposure * 1800). Strict 95% certification of <= 1/30 min needs >= 90 min of aggregate exposure with 0 observed FPs: `-SoakMinutes 90` (approx. {1} min wall at {2} parallel streams).' -f (Fmt $negFpUpper '0.00'), [math]::Ceiling(90.0 * 60.0 / ($negParallel * 60.0)), $negParallel)
Add-Line ''
Add-Line '## 3. Overhead (detector armed vs suspended, same locked workloads)'
Add-Line ''
Add-Line '| Pattern | Uninstrumented (median) | Instrumented (median) | Multiplier |'
Add-Line '|---|---|---|---|'
foreach ($p in $ovhPatterns) {
    Add-Line ('| {0} | {1} ms | {2} ms | {3}x |' -f $p.pattern, (Fmt ([double]$p.off_ms) '0.0'), (Fmt ([double]$p.on_ms) '0.0'), (Fmt ([double]$p.ratio) '0.00'))
}
Add-Line ''
Add-Line ('Aggregate: geomean **{0}x**, median {1}x, budget <= 5x.' -f (Fmt $ovhGeomean '0.00'), (Fmt $ovhMedian '0.00'))
Add-Line ''
Add-Line '## 4. Method and limits (short)'
Add-Line ''
Add-Line '- Detector: `tests/gtest/support/race_track.hpp` -- container decorator recording, per element and per container structural slot, last writer / last reader thread id + coarse happens-before epoch + the test-visible lockset. A conflict is reported when two accesses touch the same slot, at least one is a write, threads differ, no happens-before edge exists, and the lockset intersection is empty (Eraser 1997).'
Add-Line ('- Sampling: only every {0}th container operation is instrumented (LiteRace 2009 idea); slot state persists between sampled events so hot races are still seen.' -f $Sample)
Add-Line '- Locks are visible only through the test-only `race_sentinel::TrackedMutex`; product mutexes are not instrumented in Phase 1 (no product source edits).'
Add-Line '- Happens-before is a coarse global epoch advanced at thread-group spawn and thread join; it is blind to condition variables, atomics used as hand-offs, one-way flag publication and different spawn groups. Blind spots cause false negatives, not false positives on properly locked code.'
Add-Line '- The positive controls are harness-safe reproductions of the four already-fixed patterns (pre-reserved vector, SSO string, bounded map insertion, never-freeing arena allocator for the unordered_map) so the experiment measures detection instead of crashing on heap corruption.'
Add-Line ('- Logs and raw parts: `{0}` (build.log, correctness.log, positive.log, negative_s*.log, overhead.log, part_*.json).' -f $OutDir)
Add-Line ''
Add-Line '## 5. Repeat run'
Add-Line ''
Add-Line '```powershell'
Add-Line ('cd ' + $Root)
Add-Line 'powershell -NoProfile -ExecutionPolicy Bypass -File tools\race_sentinel_run.ps1'
Add-Line '```'
Add-Line ''
Add-Line '- Longer false-positive soak (-SoakMinutes 90, ~23 min wall at 4 parallel streams):'
Add-Line '```powershell'
Add-Line 'powershell -NoProfile -ExecutionPolicy Bypass -File tools\race_sentinel_run.ps1 -SoakMinutes 90'
Add-Line '```'
Add-Line '- The sidecar build uses only `build/conan_home` gtest + `tests/gtest/**`; it never runs compile.bat / compile_gtest.bat and never touches the live gateway or llama-servers.'

$ReportMd = Join-Path $OutDir 'report.md'
$sb.ToString() | Set-Content -Path $ReportMd -Encoding utf8

Write-Host ''
Write-Host '=== summary ==='
Write-Host ('positive detection : {0}/{1} ({2}%)  pass={3}' -f $posDetected, $posTotalRuns, (Fmt ($posRate * 100) '0.0'), $posDetPass)
Write-Host ('false positives    : {0} reports / {1:N1} s exposure -> observed {2} per 30 min (rule-of-three upper95 {3})  pass={4}' -f $negTotalReports, $negExposure, (Fmt $negFpPoint '0.00'), (Fmt $negFpUpper '0.00'), $negPass)
Write-Host ('overhead           : geomean {0}x  pass={1}' -f (Fmt $ovhGeomean '0.00'), $ovhPass)
Write-Host ('correctness        : {0} passed / {1} failed  pass={2}' -f $corrPassed, $corrFailed, $corrOk)
Write-Host ('VERDICT            : {0}' -f $verdict)
Write-Host ''
Write-Host "report.json : $ReportJson"
Write-Host "report.md   : $ReportMd"
Write-Host ''

if ($verdict -eq 'PASS') { exit 0 } else { exit 1 }
