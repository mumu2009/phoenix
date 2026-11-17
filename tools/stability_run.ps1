#requires -Version 7.0
<#
  stability_run.ps1 - v10.1 repeatable stability run with a self-contained
  evidence bundle.

  One command drives a run of N minutes (default 180 = the 3h protocol) and
  produces runtime_store\stability\<ts>[-label]\ with:
    harness.log          this harness' own trace
    samples.jsonl        one JSON object per sample (full metric set)
    samples.csv          flat CSV view of the same samples
    lifecycle.json       assign -> work -> report/complete transcript
    summary.json         machine-readable verdict + metrics + criteria
    summary.md           human-readable metrics table, verdicts, criteria
    forensics\           WER crash records / dump paths / resolved stacks
    lifecycle\           sidecar logs, sidecar autonomy snapshot, deliverable

  Design rules (see doc/v10.1/stability_protocol.md):
    * The LIVE system is only OBSERVED (health/status GETs, log reads,
      process/WER queries).  This script never stops, restarts or
      reconfigures the live gateway or llama-server processes.
    * The mission lifecycle runs on a SIDECAR gateway instance: a copy of
      phoenix_main.exe under a new name, own working directory, own port
      (5100-5199), own runtime_store (AI_BASE_DIR) and its own pid file, so
      the live supervisor/gateway_loop never reaps it by name.  If the
      sidecar cannot be started, -LifecycleMode auto falls back to driving
      the lifecycle on the live gateway and LABELS that in the summary.
    * Every process started here is killed in a finally block; the sidecar
      work dir is removed.  A cleanup.json is written first so
      -CleanupOnly -BundleDir <bundle> can clean up after an interrupt.
    * Criteria are explicit and versioned; a run shorter than 3 hours of
      observed exposure is explicitly labelled SELF-TEST / below S1 and is
      never reported as "stable".

  EXAMPLES
    # real 3h protocol run
    pwsh -NoProfile -File tools\stability_run.ps1 -Minutes 180 -Label v10.1-<build>

    # 8 minute harness dry run
    pwsh -NoProfile -File tools\stability_run.ps1 -Minutes 8 -IntervalSec 30 -Label dryrun

    # drive the lifecycle on the live gateway (labelled fallback)
    pwsh -NoProfile -File tools\stability_run.ps1 -Minutes 180 -LifecycleMode live

    # clean up after an interrupted run
    pwsh -NoProfile -File tools\stability_run.ps1 -CleanupOnly -BundleDir runtime_store\stability\<ts>
#>
[CmdletBinding()]
param(
  [int]$Minutes = 180,
  [int]$IntervalSec = 60,
  [string]$Label = '',
  [string]$RepoRoot = 'E:\079\v6.0Alixander\phoenix',
  [string]$GatewayUrl = '',
  [string]$AuthToken = 'local-verify',
  [ValidateSet('auto', 'sidecar', 'live', 'none')]
  [string]$LifecycleMode = 'auto',
  [int]$SidecarStartupTimeoutSec = 180,
  [int]$LifecycleTimeoutSec = 900,
  [int]$MissionStatusEverySec = 300,
  [int]$HealthTimeoutSec = 15,
  [string]$OutRoot = '',
  [switch]$CleanupOnly,
  [string]$BundleDir = ''
)

$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'

# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
function NowIso { (Get-Date).ToString('o') }
function NowClock { (Get-Date).ToString('HH:mm:ss') }

function Get-Prop($obj, [string]$key, $default = $null) {
  if ($null -eq $obj) { return $default }
  if ($obj -is [System.Collections.IDictionary]) {
    if ($obj.Contains($key)) { return $obj[$key] }
    return $default
  }
  $p = $obj.PSObject.Properties[$key]
  if ($p) { return $p.Value }
  return $default
}

function CsvQuote([object]$v) {
  if ($null -eq $v) { return '' }
  $s = [string]$v
  if ($s -match '[",\r\n]') { return '"' + ($s -replace '"', '""') + '"' }
  return $s
}

function Test-TcpListen([int]$port) {
  $c = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
  return [bool]$c
}

function Find-FreePort([int]$start = 5100, [int]$end = 5199) {
  for ($p = $start; $p -le $end; $p++) {
    if (-not (Test-TcpListen $p)) { return $p }
  }
  throw "no free sidecar port in $start..$end"
}

function Resolve-ProcessPath([int]$procId) {
  if ($procId -le 0) { return '' }
  try {
    $p = Get-Process -Id $procId -ErrorAction Stop
    return $p.Path
  } catch { return '' }
}

function Get-AuthHeaders {
  return @{ Authorization = "Bearer $AuthToken" }
}

function Invoke-Get($url, [int]$timeoutSec, [switch]$NoThrow) {
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $r = [ordered]@{ ok = $false; status = 0; body = $null; raw = ''; ms = 0; error = '' }
  try {
    $resp = Invoke-WebRequest -Uri $url -Headers (Get-AuthHeaders) -TimeoutSec $timeoutSec -UseBasicParsing -ErrorAction Stop
    $sw.Stop()
    $r.status = [int]$resp.StatusCode
    $r.raw = [string]$resp.Content
    try { $r.body = $r.raw | ConvertFrom-Json -ErrorAction Stop } catch { $r.body = $null }
    $r.ok = ($r.status -ge 200 -and $r.status -lt 300 -and $null -ne $r.body)
    $r.ms = [int]$sw.ElapsedMilliseconds
  } catch {
    $sw.Stop()
    $r.ms = [int]$sw.ElapsedMilliseconds
    $r.error = $_.Exception.Message
    if ($_.ErrorDetails -and $_.ErrorDetails.Message) {
      $r.raw = [string]$_.ErrorDetails.Message
      try { $r.body = $r.raw | ConvertFrom-Json -ErrorAction Stop } catch { }
    }
  }
  return [pscustomobject]$r
}

function Invoke-PostJson($url, $bodyObj, [int]$timeoutSec) {
  $json = $bodyObj | ConvertTo-Json -Depth 8 -Compress
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $r = [ordered]@{ ok = $false; status = 0; body = $null; raw = ''; ms = 0; error = '' }
  try {
    $resp = Invoke-WebRequest -Uri $url -Method POST -Headers (Get-AuthHeaders) -ContentType 'application/json' `
      -Body $json -TimeoutSec $timeoutSec -UseBasicParsing -ErrorAction Stop
    $sw.Stop()
    $r.status = [int]$resp.StatusCode
    $r.raw = [string]$resp.Content
    try { $r.body = $r.raw | ConvertFrom-Json -ErrorAction Stop } catch { }
    $r.ok = ($r.status -ge 200 -and $r.status -lt 300)
    $r.ms = [int]$sw.ElapsedMilliseconds
  } catch {
    $sw.Stop()
    $r.ms = [int]$sw.ElapsedMilliseconds
    $r.error = $_.Exception.Message
    if ($_.ErrorDetails -and $_.ErrorDetails.Message) {
      $r.raw = [string]$_.ErrorDetails.Message
      try { $r.body = $r.raw | ConvertFrom-Json -ErrorAction Stop } catch { }
    }
    if ($null -ne $r.body -and $r.body.ok -eq $true) { $r.ok = $true }
  }
  return [pscustomobject]$r
}

function Write-JsonLine([string]$path, $obj) {
  try {
    $line = $obj | ConvertTo-Json -Depth 12 -Compress
    Add-Content -Path $path -Value $line -Encoding utf8
  } catch { }
}

# --------------------------------------------------------------------------
# harness log
# --------------------------------------------------------------------------
function Log([string]$msg, [string]$level = 'INFO') {
  $line = "{0} [{1}] {2}" -f (NowIso), $level, $msg
  switch ($level) {
    'WARN' { Write-Host $line -ForegroundColor Yellow }
    'ERROR' { Write-Host $line -ForegroundColor Red }
    default { Write-Host $line }
  }
  if ($script:G -and $script:G.LogPath) {
    Add-Content -Path $script:G.LogPath -Value $line -Encoding utf8 -ErrorAction SilentlyContinue
  }
}

# --------------------------------------------------------------------------
# GPU
# --------------------------------------------------------------------------
function Get-GpuStats {
  $out = [ordered]@{ ok = $false; count = 0; utilMean = $null; utilMax = $null; memUsedMB = $null; memTotalMB = $null; tempC = $null; error = '' }
  $cmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
  if (-not $cmd) { $out.error = 'nvidia-smi not found'; return [pscustomobject]$out }
  $utils = @()
  for ($i = 0; $i -lt 3; $i++) {
    try {
      $line = (& $cmd.Source --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
      if ($line) {
        $parts = ([string]$line).Split(',') | ForEach-Object { $_.Trim() }
        if ($parts.Count -ge 4) {
          $out.count += 1
          $utils += [double]$parts[0]
          $out.memUsedMB = [int]$parts[1]
          $out.memTotalMB = [int]$parts[2]
          $out.tempC = [int]$parts[3]
        }
      }
    } catch { $out.error = $_.Exception.Message }
    if ($i -lt 2) { Start-Sleep -Milliseconds 350 }
  }
  if ($utils.Count -gt 0) {
    $out.ok = $true
    $out.utilMean = [math]::Round((($utils | Measure-Object -Average).Average), 1)
    $out.utilMax = [math]::Round((($utils | Measure-Object -Maximum).Maximum), 1)
  } elseif (-not $out.error) { $out.error = 'no parsable nvidia-smi output' }
  return [pscustomobject]$out
}

# --------------------------------------------------------------------------
# live gateway identity: the listener owner only counts when it is a REAL
# gateway process; old 0-thread 32KB zombie images must not satisfy checks.
# --------------------------------------------------------------------------
function Get-ProcCpuMs([int]$procId) {
  if ($procId -le 0) { return $null }
  try { $p = Get-Process -Id $procId -ErrorAction Stop; return [int64]$p.TotalProcessorTime.TotalMilliseconds } catch { return $null }
}

function Get-GatewayIdentity([int]$port) {
  $id = [ordered]@{
    port = $port; listenerPid = 0; pid = 0; rssMB = $null; threads = $null
    cpuMs = $null; exe = ''; listenerZombie = $false; zombies = @()
  }
  # zombie census: images with no threads / ~0 RSS are leftover dead handles
  $all = @(Get-Process phoenix_main -ErrorAction SilentlyContinue)
  $zombies = @()
  foreach ($p in $all) {
    $isZombie = ($p.WorkingSet64 -lt 100MB -and $p.Threads.Count -eq 0)
    if ($isZombie) { $zombies += [int]$p.Id }
  }
  $id.zombies = $zombies
  $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($conn) {
    $id.listenerPid = [int]$conn.OwningProcess
    $lp = Get-Process -Id $id.listenerPid -ErrorAction SilentlyContinue
    if ($lp) {
      $id.rssMB = [int]($lp.WorkingSet64 / 1MB)
      if ($lp.WorkingSet64 -ge 100MB -and $lp.ProcessName -eq 'phoenix_main') {
        $id.pid = [int]$lp.Id; $id.threads = $lp.Threads.Count; $id.exe = $lp.Path
        $id.cpuMs = [int64]$lp.TotalProcessorTime.TotalMilliseconds
        return [pscustomobject]$id
      }
      # listener held by a zombie/dead image: fall back to the real gateway
      $id.listenerZombie = $true
    }
  }
  $real = $all | Where-Object { $_.WorkingSet64 -ge 100MB -and $_.Threads.Count -gt 0 } | Sort-Object WorkingSet64 -Descending | Select-Object -First 1
  if ($real) {
    $id.pid = [int]$real.Id; $id.threads = $real.Threads.Count; $id.rssMB = [int]($real.WorkingSet64 / 1MB)
    $id.cpuMs = [int64]$real.TotalProcessorTime.TotalMilliseconds; $id.exe = $real.Path
  }
  return [pscustomobject]$id
}

function Get-LlamaRssMB {
  $procs = @(Get-Process llama-server -ErrorAction SilentlyContinue)
  $sum = 0
  foreach ($p in $procs) { $sum += [int]($p.WorkingSet64 / 1MB) }
  return [pscustomobject]@{ count = $procs.Count; rssMB = $sum }
}

# --------------------------------------------------------------------------
# appended-log tail reader (offset tracked per file; handles truncation when
# gateway_loop restarts the gateway and the redirected stdout is recreated)
# --------------------------------------------------------------------------
function Read-AppendedLines([string]$path, [string]$key) {
  $lines = @()
  if (-not (Test-Path $path)) { return $lines }
  $len = (Get-Item $path).Length
  $off = 0
  if ($script:G.LogOffsets.Contains($key)) { $off = [int64]$script:G.LogOffsets[$key] }
  if ($len -lt $off) {
    $script:G.LogRotations += [ordered]@{ key = $key; at = (NowIso); fromBytes = $off; toBytes = $len }
    $off = 0
  }
  if ($len -eq $off) { $script:G.LogOffsets[$key] = $len; return $lines }
  $fs = $null
  try {
    $fs = [System.IO.File]::Open($path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
    [void]$fs.Seek($off, [System.IO.SeekOrigin]::Begin)
    $count = [int]($len - $off)
    if ($count -gt 33554432) { $count = 33554432 }   # safety cap: 32MB per read
    $buf = New-Object byte[] $count
    $read = $fs.Read($buf, 0, $count)
    if ($read -gt 0) {
      $text = [System.Text.Encoding]::UTF8.GetString($buf, 0, $read)
      $lines = @($text -split "`r?`n" | Where-Object { $_ -ne '' })
      $script:G.LogOffsets[$key] = $off + $read
    } else {
      $script:G.LogOffsets[$key] = $len
    }
  } catch {
    $script:G.LogOffsets[$key] = $off
  } finally {
    if ($fs) { $fs.Dispose() }
  }
  return $lines
}

function Update-LogCounters {
  $runDir = Join-Path $script:G.Repo 'runtime_store\long_run'
  $gwLines = Read-AppendedLines (Join-Path $runDir 'gateway.log') 'gateway.log'
  foreach ($l in $gwLines) {
    if ($l -match 'llama-slot-busy') { $script:G.Counters.slotBusy++ }
    if ($l -match 'llamacpp FAIL') { $script:G.Counters.missionLlmFail++ }
  }
  $errLines = Read-AppendedLines (Join-Path $runDir 'gateway.err.log') 'gateway.err.log'
  foreach ($l in $errLines) {
    if ($l -match 'llama-slot-busy') { $script:G.Counters.slotBusy++ }
  }
  $loopLines = Read-AppendedLines (Join-Path $runDir 'gateway_loop.log') 'gateway_loop.log'
  foreach ($l in $loopLines) {
    if ($l -match 'gateway started pid=(\d+)') {
      $script:G.Counters.wrapperStarted++
      $script:G.RestartEvents += [ordered]@{ at = (NowIso); source = 'gateway_loop.log'; kind = 'gateway_started'; detail = $l.Trim() }
    } elseif ($l -match 'gateway pid=(\d+) exited') {
      $script:G.Counters.wrapperExited++
      $script:G.RestartEvents += [ordered]@{ at = (NowIso); source = 'gateway_loop.log'; kind = 'gateway_exited'; detail = $l.Trim() }
    } elseif ($l -match 'unresponsive|never healthy') {
      $script:G.Counters.wrapperUnresponsive++
      $script:G.RestartEvents += [ordered]@{ at = (NowIso); source = 'gateway_loop.log'; kind = 'watchdog_kill'; detail = $l.Trim() }
    }
  }
  $supLines = Read-AppendedLines (Join-Path $runDir 'supervisor.log') 'supervisor.log'
  foreach ($l in $supLines) {
    if ($l -match 'gateway down -> restart') { $script:G.Counters.supGatewayRestarts++ }
    if ($l -match '(8B|coder) down -> restart') { $script:G.Counters.supLlamaRestarts++ }
  }
  $trLines = Read-AppendedLines (Join-Path $runDir 'training.log') 'training.log'
  foreach ($l in $trLines) {
    if ($l -match 'phase=train') { $script:G.Counters.trainRounds++ }
    if ($l -match 'phase=test|mmlu') { $script:G.Counters.mmluRuns++ }
  }
}

# --------------------------------------------------------------------------
# persisted autonomy state (read-only): fresh within ~15s, carries all
# RUNNING missions + the last 64 finished; used for cheap mission counters
# and goalsRun.  Falls back to the previous good snapshot on parse errors.
# --------------------------------------------------------------------------
function Read-AutonomySnapshot {
  $path = Join-Path $script:G.Repo 'runtime_store\autonomy_state.json'
  if (-not (Test-Path $path)) { return [pscustomobject]@{ ok = $false; error = 'autonomy_state.json missing'; missions = @(); goalsRun = $null; iteration = $null; mtime = $null } }
  for ($try = 1; $try -le 3; $try++) {
    try {
      $raw = Get-Content -Path $path -Raw -ErrorAction Stop
      if (-not $raw -or $raw.Length -lt 10) { throw 'empty' }
      $j = $raw | ConvertFrom-Json -AsHashtable -ErrorAction Stop
      $missions = @()
      if ($j.Contains('missions') -and $null -ne $j['missions']) { $missions = @($j['missions']) }
      $goalsRun = $null; $iteration = $null
      if ($j.Contains('studyPopulation') -and $j['studyPopulation']) { $goalsRun = Get-Prop $j['studyPopulation'] 'goalsRun' }
      if ($j.Contains('iteration')) { $iteration = $j['iteration'] }
      return [pscustomobject]@{
        ok = $true; error = ''; missions = $missions; goalsRun = $goalsRun
        iteration = $iteration; mtime = (Get-Item $path).LastWriteTime
      }
    } catch {
      Start-Sleep -Milliseconds 150
    }
  }
  return [pscustomobject]@{ ok = $false; error = 'unreadable/stale json'; missions = @(); goalsRun = $null; iteration = $null; mtime = $null }
}

function Get-PersistedMissionCounts($snap) {
  $running = 0; $finishedRecent = 0
  foreach ($m in @($snap.missions)) {
    $st = Get-Prop (Get-Prop (Get-Prop $m 'state') 'mission') 'state'
    if ($st -eq 1) { $running++ } else { $finishedRecent++ }
  }
  return [pscustomobject]@{ running = $running; finishedRecent = $finishedRecent }
}

function Find-PersistedMission($snap, [string]$missionId) {
  foreach ($m in @($snap.missions)) {
    if ([string](Get-Prop $m 'id') -eq $missionId) {
      $stats = Get-Prop $m 'state'
      $mission = Get-Prop $stats 'mission'
      $dl = [string](Get-Prop $mission 'deliverable' '')
      return [pscustomobject]@{
        id = $missionId
        state = Get-Prop $mission 'state'
        deliverableLen = $dl.Length
        deliverable = $dl
        pressure = Get-Prop $stats 'pressure' 0
        startMs = Get-Prop $mission 'startMs'
        endMs = Get-Prop $mission 'endMs'
      }
    }
  }
  return $null
}

# --------------------------------------------------------------------------
# authoritative mission counters (heavier: /api/mission/status serializes the
# whole mission container; ~1.5-3s + 11MB).  Sampled at t0, every
# MissionStatusEverySec and at the end, never every minute.
# --------------------------------------------------------------------------
function Get-AuthoritativeMissionCounts([string]$baseUrl) {
  $r = Invoke-Get "$baseUrl/api/mission/status" 90
  if (-not $r.ok -or $null -eq $r.body) {
    return [pscustomobject]@{ ok = $false; error = $r.error; running = $null; completed = $null; failed = $null; idle = $null; total = $null; ms = $r.ms; missions = @() }
  }
  $missions = @()
  $res = Get-Prop $r.body 'result'
  if ($res) { $missions = @(Get-Prop $res 'missions' @()) }
  $c = @{ 0 = 0; 1 = 0; 2 = 0; 3 = 0 }
  foreach ($m in $missions) {
    $st = Get-Prop (Get-Prop (Get-Prop $m 'stats') 'mission') 'state'
    if ($null -ne $st -and $c.ContainsKey([int]$st)) { $c[[int]$st]++ }
  }
  return [pscustomobject]@{
    ok = $true; error = ''; running = $c[1]; completed = $c[2]; failed = $c[3]; idle = $c[0]
    total = $missions.Count; ms = $r.ms; missions = $missions
  }
}

# --------------------------------------------------------------------------
# WER crash detection + forensics hook
# --------------------------------------------------------------------------
function Get-WatchedAppNames {
  $names = @('phoenix_main.exe', 'llama-server.exe')
  if ($script:G.Sidecar -and $script:G.Sidecar.exeName) { $names += $script:G.Sidecar.exeName }
  return $names
}

function Parse-CrashEvent($evt) {
  $msg = [string]$evt.Message
  $out = [ordered]@{
    recordId = [int]$evt.RecordId
    time = $evt.TimeCreated.ToString('o')
    timeLocal = $evt.TimeCreated.ToString('yyyy-MM-dd HH:mm:ss')
    provider = [string]$evt.ProviderName
    appName = ''; appPath = ''; module = ''; exceptionCode = ''; faultingPid = $null; faultingPidHex = ''
    message = $msg
  }
  $patterns = @(
    @{ k = 'appName'; re = '(?:错误应用程序名称|Faulting application name)\s*[:：]\s*([^，,\r\n]+)' },
    @{ k = 'appPath'; re = '(?:错误应用程序路径|Faulting application path)\s*[:：]\s*([^\r\n]+)' },
    @{ k = 'module'; re = '(?:错误模块名称|Faulting module name)\s*[:：]\s*([^，,\r\n]+)' },
    @{ k = 'exceptionCode'; re = '(?:异常代码|Exception code)\s*[:：]\s*([^\r\n]+)' },
    @{ k = 'faultingPidHex'; re = '(?:错误进程 ID|Faulting process id)\s*[:：]\s*([^\r\n]+)' }
  )
  foreach ($p in $patterns) {
    $m = [regex]::Match($msg, $p.re)
    if ($m.Success) { $out[$p.k] = $m.Groups[1].Value.Trim() }
  }
  if ($out.faultingPidHex -match '0x([0-9a-fA-F]+)') {
    try { $out.faultingPid = [int][Convert]::ToInt64($Matches[1], 16) } catch { }
  }
  return [pscustomobject]$out
}

function Find-CrashDump($crash) {
  $search = [ordered]@{ searched = @(); found = ''; reportWer = '' }
  $apps = @($crash.appName)
  if ($crash.appName -and $crash.appName -ne '') { } else { $apps = Get-WatchedAppNames }
  $pidDec = $crash.faultingPid
  $since = if ($script:G.RunStart) { $script:G.RunStart.AddMinutes(-5) } else { (Get-Date).AddMinutes(-30) }
  $dirs = @(
    (Join-Path $env:LOCALAPPDATA 'CrashDumps'),
    (Join-Path $env:ProgramData 'Microsoft\Windows\WER\ReportQueue'),
    (Join-Path $env:ProgramData 'Microsoft\Windows\WER\ReportArchive'),
    (Join-Path $env:LOCALAPPDATA 'Microsoft\Windows\WER\ReportQueue'),
    (Join-Path $env:LOCALAPPDATA 'Microsoft\Windows\WER\ReportArchive'),
    # project-local dump locations (WER LocalDumps target + the gateway's own
    # MiniDumpWriteDump path used by its unhandled-exception filter)
    (Join-Path $script:G.Repo 'build\tmp\dumps'),
    (Join-Path $script:G.Repo 'runtime_store\dumps'),
    (Join-Path $script:G.Repo 'runtime_store\logs')
  )
  $candidates = @()
  # the in-process handler writes runtime_store/logs/crash.dmp directly; it is
  # usually 0 bytes (MiniDumpWriteDump fails in the corrupted process) but a
  # non-empty file is a usable dump.
  $ownDump = Join-Path $script:G.Repo 'runtime_store\logs\crash.dmp'
  if (Test-Path $ownDump) {
    $oi = Get-Item $ownDump
    $search.searched += $ownDump
    if ($oi.Length -gt 0 -and $oi.LastWriteTime -ge $since) {
      $candidates += [pscustomobject]@{ path = $oi.FullName; mtime = $oi.LastWriteTime; pid = $null }
    }
  }
  foreach ($d in $dirs) {
    if (-not (Test-Path $d)) { continue }
    $search.searched += $d
    foreach ($app in $apps) {
      if (-not $app) { continue }
      try {
        Get-ChildItem -Path $d -Filter "$app.*.dmp" -File -ErrorAction SilentlyContinue |
          Where-Object { $_.LastWriteTime -ge $since } |
          ForEach-Object { $candidates += [pscustomobject]@{ path = $_.FullName; mtime = $_.LastWriteTime; pid = $null } }
      } catch { }
      try {
        Get-ChildItem -Path $d -Directory -ErrorAction SilentlyContinue |
          Where-Object { $_.Name -like "*$app*" -and $_.LastWriteTime -ge $since } |
          ForEach-Object {
            $dir = $_
            Get-ChildItem -Path $dir.FullName -Recurse -Filter '*.dmp' -File -ErrorAction SilentlyContinue | ForEach-Object {
              $candidates += [pscustomobject]@{ path = $_.FullName; mtime = $_.LastWriteTime; pid = $null }
            }
            $wer = Join-Path $dir.FullName 'Report.wer'
            if (Test-Path $wer) { $search.reportWer = $wer }
          }
      } catch { }
    }
  }
  # prefer a dump whose filename embeds the faulting pid
  $preferred = $null
  foreach ($c in $candidates) {
    if ($pidDec -and $c.path -match "\.$pidDec\.dmp$") { $preferred = $c; break }
  }
  if (-not $preferred -and $candidates.Count -gt 0) { $preferred = $candidates | Sort-Object mtime -Descending | Select-Object -First 1 }
  if ($preferred) { $search.found = $preferred.path }
  return [pscustomobject]$search
}

function Get-PythonExe {
  foreach ($cand in @((Join-Path $script:G.Repo 'Python314\python.exe'), 'python', 'py')) {
    if ($cand -match '\\' -or $cand -match '^[A-Za-z]:') {
      if (Test-Path $cand) { return $cand }
    } else {
      $c = Get-Command $cand -ErrorAction SilentlyContinue
      if ($c) { return $c.Source }
    }
  }
  return $null
}

function Invoke-Forensics($crash) {
  $dir = Join-Path $script:G.Bundle 'forensics'
  New-Item -ItemType Directory -Force $dir | Out-Null
  $stamp = ([datetime]$crash.time).ToString('yyyyMMdd-HHmmss')
  $base = Join-Path $dir ("crash-$stamp-pid" + $(if ($crash.faultingPid) { $crash.faultingPid } else { 'na' }))
  New-Item -ItemType Directory -Force $base | Out-Null
  Set-Content -Path (Join-Path $base 'event.txt') -Value $crash.message -Encoding utf8
  $rec = [ordered]@{
    time = $crash.time; appName = $crash.appName; appPath = $crash.appPath
    module = $crash.module; exceptionCode = $crash.exceptionCode; faultingPid = $crash.faultingPid
    recordId = $crash.recordId; kind = $(if ($crash.PSObject.Properties['kind']) { $crash.kind } else { 'wer_application_error' })
    severity = $(if ($crash.PSObject.Properties['severity']) { $crash.severity } else { 'crash' })
    dumpPath = ''; dumpSearched = @(); werReport = ''; resolveReport = ''; resolveExit = $null
    resolved = $false; gap = ''
  }
  $d = Find-CrashDump $crash
  $rec.dumpSearched = $d.searched
  $rec.dumpPath = $d.found
  if ($d.reportWer) {
    try { Copy-Item -Path $d.reportWer -Destination (Join-Path $base 'Report.wer') -Force; $rec.werReport = (Join-Path $base 'Report.wer') } catch { }
  }
  if (-not $rec.dumpPath) {
    if ($rec.kind -eq 'fatal_runtime_seh') {
      # The gateway installs SetUnhandledExceptionFilter and returns
      # EXCEPTION_EXECUTE_HANDLER, so an ordinary SEH fault (e.g. 0xC0000005)
      # is swallowed: WER never records it and no minidump is written.
      # runtime_store/logs/fatal_runtime.log is then the ONLY evidence.
      $rec.gap = "fatal_runtime.log unhandled exception (SEH swallowed by the in-process filter): no WER event and no minidump by design; crash.dmp is usually 0-byte. Evidence line recorded in event.txt. Enable LocalDumps only helps fail-fast codes (e.g. 0xC0000374)."
    } else {
      $rec.gap = "no minidump found for $($crash.appName) pid=$($crash.faultingPid) (WER LocalDumps not configured for this binary?) - searched: " + ($d.searched -join '; ')
    }
    Log "forensics: $($rec.gap)" 'WARN'
  } else {
    $resolver = Join-Path $script:G.Repo 'tools\dump_stack_resolve.py'
    $map = Join-Path $script:G.Repo 'phoenix_main.map'
    $py = Get-PythonExe
    if (-not (Test-Path $resolver)) {
      $rec.gap = "dump found at $($rec.dumpPath) but tools\dump_stack_resolve.py is missing"
    } elseif (-not (Test-Path $map)) {
      $rec.gap = "dump found at $($rec.dumpPath) but phoenix_main.map is missing (rebuild with -Wl,-Map=phoenix_main.map)"
    } elseif (-not $py) {
      $rec.gap = "dump found at $($rec.dumpPath) but no usable python interpreter (Python314\python.exe/python/py) - cannot run dump_stack_resolve.py"
    } else {
      $resolveOut = Join-Path $base 'resolve.txt'
      Log "forensics: resolving $($rec.dumpPath) with $py"
      try {
        $output = & $py $resolver $rec.dumpPath --map $map 2>&1 | Out-String
        $rec.resolveExit = $LASTEXITCODE
        Set-Content -Path $resolveOut -Value $output -Encoding utf8
        $rec.resolveReport = $resolveOut
        $rec.resolved = ($LASTEXITCODE -eq 0)
        if (-not $rec.resolved) { $rec.gap = "dump_stack_resolve.py exited $LASTEXITCODE; see $resolveOut" }
      } catch {
        $rec.gap = "dump_stack_resolve.py failed: $($_.Exception.Message)"
        Set-Content -Path $resolveOut -Value ([string]$_) -Encoding utf8
        $rec.resolveReport = $resolveOut
      }
    }
  }
  Set-Content -Path (Join-Path $base 'forensics.json') -Value ($rec | ConvertTo-Json -Depth 6) -Encoding utf8
  $script:G.Crashes += $rec
  if ($rec.gap) { $script:G.ForensicsGaps += $rec.gap }
  return $rec
}

function Scan-NewCrashes {
  $found = @()
  $apps = Get-WatchedAppNames
  foreach ($eventId in @(1000)) {
    try {
      $evts = Get-WinEvent -FilterHashtable @{ LogName = 'Application'; Id = $eventId; StartTime = $script:G.RunStart } -ErrorAction Stop
    } catch { $evts = @() }
    foreach ($e in @($evts)) {
      if ($script:G.CrashIds.Contains([int]$e.RecordId)) { continue }
      $msg = [string]$e.Message
      $hit = $false
      foreach ($app in $apps) {
        if ($app -and $msg -like "*$app*") { $hit = $true; break }
      }
      if (-not $hit) { continue }
      $script:G.CrashIds[[int]$e.RecordId] = $true
      $found += (Parse-CrashEvent $e)
    }
  }
  return $found
}

# The gateway installs its own unhandled-exception filter that returns
# EXCEPTION_EXECUTE_HANDLER: ordinary SEH faults (typically 0xC0000005) are
# swallowed and WER never sees them.  The ONLY evidence is a line in
# runtime_store/logs/fatal_runtime.log (plus a usually-0-byte crash.dmp), and
# the process exits with the exception code.  A 0xC0000374 fail-fast bypasses
# the filter and DOES produce WER event 1000 + a full dump.  Scan both.
function Scan-FatalRuntime {
  $found = @()
  $sources = @()
  $liveLog = Join-Path $script:G.Repo 'runtime_store\logs\fatal_runtime.log'
  $sources += [pscustomobject]@{ path = $liveLog; key = 'fatal_runtime.log'; app = 'phoenix_main.exe' }
  if ($script:G.Sidecar -and $script:G.Sidecar.dir) {
    $scLog = Join-Path $script:G.Sidecar.dir 'runtime_store\logs\fatal_runtime.log'
    $sources += [pscustomobject]@{ path = $scLog; key = 'sidecar_fatal_runtime.log'; app = $script:G.Sidecar.exeName }
  }
  $idx = 0
  foreach ($src in $sources) {
    $lines = Read-AppendedLines $src.path $src.key
    foreach ($ln in $lines) {
      if ($ln -notmatch 'unhandled exception') { continue }
      if ($ln -notmatch 'code=0x([0-9a-fA-F]+)') { continue }
      $idx++
      $key = "fatal|$($src.key)|$($ln)"
      if ($script:G.FatalRuntimeSeen.ContainsKey($key)) { continue }
      $script:G.FatalRuntimeSeen[$key] = $true
      $code = '0x' + $Matches[1]
      $addr = ''; $offset = ''; $rip = ''; $target = ''
      if ($ln -match 'addr=(\S+)') { $addr = $Matches[1] }
      if ($ln -match 'offset=(\S+)') { $offset = $Matches[1] }
      if ($ln -match 'rip=(\S+)') { $rip = $Matches[1] }
      if ($ln -match 'target=(\S+)') { $target = $Matches[1] }
      # fatal_runtime.log lines start with "yyyy-MM-dd HH:mm:ss | ..."
      $ts = (Get-Date)
      if ($ln -match '^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)') { try { $ts = [datetime]::ParseExact($Matches[1], 'yyyy-MM-dd HH:mm:ss', $null) } catch { } }
      $found += [pscustomobject]@{
        recordId = "fatal-$($src.key)-$($ts.ToString('HHmmss'))-$idx"
        time = $ts.ToString('o'); timeLocal = $ts.ToString('yyyy-MM-dd HH:mm:ss')
        provider = 'runtime_store/logs/fatal_runtime.log'; appName = $src.app; appPath = ''
        module = 'in-process SEH filter (setUnhandledExceptionFilter)'
        exceptionCode = $code; faultingPid = $null; faultingPidHex = ''
        message = $ln.Trim(); kind = 'fatal_runtime_seh'; severity = 'crash'
        addr = $addr; offset = $offset; rip = $rip; target = $target
      }
    }
  }
  return $found
}
function Save-CleanupRecord {
  $p = Join-Path $script:G.Bundle 'cleanup.json'
  $rec = [ordered]@{
    runId = $script:G.RunId
    bundle = $script:G.Bundle
    sidecarPid = $script:G.Sidecar.pid
    sidecarExe = $script:G.Sidecar.exe
    sidecarExeName = $script:G.Sidecar.exeName
    sidecarDir = $script:G.Sidecar.dir
    sidecarPort = $script:G.Sidecar.port
    liveGatewayUrl = $script:G.LiveUrl
    note = 'Do not stop the live gateway; this cleanup record only covers processes started by stability_run.ps1'
  }
  try { Set-Content -Path $p -Value ($rec | ConvertTo-Json -Depth 5) -Encoding utf8 } catch { }
}

function Stop-Sidecar([string]$reason = '') {
  $S = $script:G.Sidecar
  if (-not $S -or -not $S.pid) { return }
  Log "sidecar: stopping pid=$($S.pid) ($reason)"
  # preserve evidence before the work dir is removed
  try {
    $st = Join-Path $S.dir 'runtime_store\autonomy_state.json'
    if (Test-Path $st) { Copy-Item $st (Join-Path $script:G.Bundle 'lifecycle\sidecar_autonomy_state.json') -Force -ErrorAction SilentlyContinue }
    if ($script:G.Lifecycle.missionId) {
      $ws = Join-Path $S.dir ("runtime_store\mission_workspace\" + $script:G.Lifecycle.missionId)
      if (Test-Path $ws) {
        $dst = Join-Path $script:G.Bundle 'lifecycle\sidecar_mission_workspace'
        Copy-Item $ws $dst -Recurse -Force -ErrorAction SilentlyContinue
      }
    }
  } catch { }
  try { & taskkill.exe /T /F /PID $S.pid 2>&1 | Out-Null } catch { }
  Start-Sleep -Seconds 2
  # verify every image under our sidecar dir is gone
  $left = @()
  try {
    $left = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
      $_.ExecutablePath -and ($_.ExecutablePath -like ($S.dir + '\*'))
    })
  } catch { }
  foreach ($lp in $left) {
    try { Stop-Process -Id ([int]$lp.ProcessId) -Force -ErrorAction SilentlyContinue } catch { }
  }
  Start-Sleep -Milliseconds 500
  $S.pid = 0
  $S.stoppedAt = NowIso
  $script:G.Sidecar = $S
  try { Remove-Item -Path $S.dir -Recurse -Force -ErrorAction SilentlyContinue } catch { }
  $script:G.Cleanup.sidecarStopped = $true
  $script:G.Cleanup.sidecarDirRemoved = -not (Test-Path $S.dir)
  $script:G.Cleanup.cleanupAt = NowIso
  Save-CleanupRecord
  Log "sidecar: stopped (dirRemoved=$($script:G.Cleanup.sidecarDirRemoved))"
}

function Start-Sidecar {
  $S = $script:G.Sidecar
  $S.dir = Join-Path $script:G.Bundle 'sidecar'
  if (Test-Path $S.dir) { Remove-Item $S.dir -Recurse -Force -ErrorAction SilentlyContinue }
  New-Item -ItemType Directory -Force (Join-Path $S.dir 'runtime_store') | Out-Null
  $S.exeName = "phoenix_stab_$($script:G.SafeLabel)_$((Get-Date).ToString('HHmmss')).exe"
  $S.exe = Join-Path $S.dir $S.exeName
  Copy-Item (Join-Path $script:G.Repo 'phoenix_main.exe') $S.exe -Force
  $stamp = Join-Path $script:G.Repo 'runtime_store\corpus_pretrain_stamp.json'
  if (Test-Path $stamp) { Copy-Item $stamp (Join-Path $S.dir 'runtime_store\corpus_pretrain_stamp.json') -Force }
  $S.port = Find-FreePort 5100 5199
  $S.url = "http://127.0.0.1:$($S.port)"
  $S.outLog = Join-Path $script:G.Bundle 'lifecycle\sidecar.out.log'
  $S.errLog = Join-Path $script:G.Bundle 'lifecycle\sidecar.err.log'
  Log "sidecar: starting $($S.exeName) port=$($S.port) cwd=$($S.dir)"
  $env:CONTROLLER_PORT = "$($S.port)"
  $env:AI_RESEARCH_PROFILE = 'full_components'
  $env:AI_ROBOTS_AUTOLOAD = 'false'
  $env:AI_FRONTEND_ENABLED = 'false'
  $env:AI_BASE_DIR = (Join-Path $S.dir 'runtime_store')
  $oldPath = $env:PATH
  $env:PATH = "$($script:G.Repo);$env:PATH"
  try {
    $proc = Start-Process -FilePath $S.exe -WorkingDirectory $S.dir -WindowStyle Hidden -PassThru `
      -RedirectStandardOutput $S.outLog -RedirectStandardError $S.errLog
  } finally {
    $env:PATH = $oldPath
    Remove-Item Env:CONTROLLER_PORT, Env:AI_RESEARCH_PROFILE, Env:AI_ROBOTS_AUTOLOAD, Env:AI_FRONTEND_ENABLED, Env:AI_BASE_DIR -ErrorAction SilentlyContinue
  }
  $S.pid = [int]$proc.Id
  $S.startedAt = NowIso
  $script:G.Sidecar = $S
  Save-CleanupRecord
  # wait for health
  $deadline = (Get-Date).AddSeconds($SidecarStartupTimeoutSec)
  $healthy = $false
  while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 2
    $p = Get-Process -Id $S.pid -ErrorAction SilentlyContinue
    if (-not $p) {
      $S.exitCode = $null
      Log "sidecar: exited during startup" 'ERROR'
      break
    }
    $h = Invoke-Get "$($S.url)/api/health" 4
    if ($h.ok) { $healthy = $true; break }
  }
  $S.healthy = $healthy
  $S.rssMBAtStart = $null
  if ($healthy) {
    try { $S.rssMBAtStart = [int]((Get-Process -Id $S.pid).WorkingSet64 / 1MB) } catch { }
    Log "sidecar: healthy pid=$($S.pid) port=$($S.port) rss=$($S.rssMBAtStart)MB after $([int]((Get-Date) - [datetime]$S.startedAt).TotalSeconds)s"
  } else {
    Log "sidecar: NOT healthy within ${SidecarStartupTimeoutSec}s" 'ERROR'
  }
  $script:G.Sidecar = $S
  return $healthy
}

# --------------------------------------------------------------------------
# lifecycle driver (assign -> work -> report/complete)
# --------------------------------------------------------------------------
function Add-LifecycleStep([string]$stage, [string]$detail, $extra = $null) {
  $step = [ordered]@{
    at = NowIso; elapsedSec = $script:G.Lifecycle.elapsedSec
    stage = $stage; detail = $detail; extra = $extra
  }
  $script:G.Lifecycle.steps += $step
  Write-JsonLine (Join-Path $script:G.Bundle 'lifecycle.jsonl') $step
  Log "lifecycle: $stage - $detail"
}

function Get-WorkspaceDeliverable([string]$missionId) {
  if (-not $missionId) { return '' }
  $roots = @()
  if ($script:G.Sidecar -and $script:G.Sidecar.dir) { $roots += (Join-Path $script:G.Sidecar.dir 'runtime_store\mission_workspace') }
  $roots += (Join-Path $script:G.Repo 'runtime_store\mission_workspace')
  foreach ($root in $roots) {
    $f = Join-Path $root ($missionId + '\deliverable.md')
    if (Test-Path $f) {
      try {
        $txt = Get-Content -Path $f -Raw -ErrorAction Stop
        if ($txt) { return [string]$txt }
      } catch { }
    }
  }
  return ''
}

function Get-TrackedMission {
  $L = $script:G.Lifecycle
  if (-not $L.missionId) { return $null }
  $wsText = Get-WorkspaceDeliverable $L.missionId
  if ($L.driver -eq 'sidecar') {
    $r = Invoke-Get "$($L.baseUrl)/api/mission/status" 20
    if (-not $r.ok) { return $null }
    $res = Get-Prop $r.body 'result'
    $missions = @(Get-Prop $res 'missions' @())
    foreach ($m in $missions) {
      if ([string](Get-Prop $m 'id') -eq $L.missionId) {
        $stats = Get-Prop $m 'stats'
        $mission = Get-Prop $stats 'mission'
        $dl = [string](Get-Prop $mission 'deliverable' '')
        # the in-memory deliverable can lag the workspace file (file-edit
        # replies are written to deliverable.md first); use the longer one.
        if ($wsText.Length -gt $dl.Length) { $dl = $wsText }
        return [pscustomobject]@{
          id = $L.missionId; state = Get-Prop $mission 'state'
          deliverableLen = $dl.Length; deliverable = $dl
          pressure = Get-Prop $stats 'pressure'
          source = 'sidecar/api/mission/status+workspace'
        }
      }
    }
    return $null
  } else {
    $snap = Read-AutonomySnapshot
    if ($snap.ok) {
      $m = Find-PersistedMission $snap $L.missionId
      if ($m -and $m.state -ne $null) {
        $dl = $m.deliverable
        if ($wsText.Length -gt $dl.Length) { $dl = $wsText }
        return [pscustomobject]@{ id = $m.id; state = $m.state; deliverableLen = $dl.Length; deliverable = $dl; pressure = $m.pressure; source = 'persisted autonomy_state+workspace' }
      }
    }
    # the persisted snapshot keeps only the last-64 FINISHED missions, keyed by
    # map order (id-sorted), so a finished probe can be pruned out of the file.
    # Fall back to the authoritative in-memory container (throttled).
    return (Find-MissionAuthoritative $L.baseUrl $L.missionId)
  }
}

function Find-MissionAuthoritative([string]$baseUrl, [string]$missionId) {
  $now = Get-Date
  if ($script:G.LastAuthMissionLookup -and (($now - [datetime]$script:G.LastAuthMissionLookup).TotalSeconds -lt 15)) {
    return $script:G.LastAuthMissionResult
  }
  $script:G.LastAuthMissionLookup = $now
  $ac = Get-AuthoritativeMissionCounts $baseUrl
  if (-not $ac.ok) { $script:G.LastAuthMissionResult = $null; return $null }
  $script:G.LastAuthoritative = $ac
  $result = $null
  foreach ($m in $ac.missions) {
    if ([string](Get-Prop $m 'id') -eq $missionId) {
      $stats = Get-Prop $m 'stats'
      $mission = Get-Prop $stats 'mission'
      $dl = [string](Get-Prop $mission 'deliverable' '')
      $wsText = Get-WorkspaceDeliverable $missionId
      if ($wsText.Length -gt $dl.Length) { $dl = $wsText }
      $result = [pscustomobject]@{
        id = $missionId; state = Get-Prop $mission 'state'
        deliverableLen = $dl.Length; deliverable = $dl
        pressure = Get-Prop $stats 'pressure'; source = 'live/api/mission/status (authoritative)'
      }
      break
    }
  }
  $script:G.LastAuthMissionResult = $result
  return $result
}

function Start-Lifecycle {
  $L = $script:G.Lifecycle
  $L.mode = $LifecycleMode
  if ($LifecycleMode -eq 'none') {
    $L.stage = 'skipped'
    Add-LifecycleStep 'skipped' 'LifecycleMode=none: no lifecycle driven'
    return
  }
  $driver = $null
  if ($LifecycleMode -eq 'auto' -or $LifecycleMode -eq 'sidecar') {
    $ok = Start-Sidecar
    if ($ok) { $driver = 'sidecar' } else {
      Stop-Sidecar 'startup failed'
      if ($LifecycleMode -eq 'sidecar') {
        $L.stage = 'error'; $L.error = 'sidecar failed to become healthy'
        Add-LifecycleStep 'error' "sidecar failed to start within ${SidecarStartupTimeoutSec}s (LifecycleMode=sidecar: no fallback)"
        $script:G.ForensicsGaps += 'lifecycle sidecar unavailable (explicit sidecar mode)'
        return
      }
      Add-LifecycleStep 'fallback' 'sidecar unavailable -> driving lifecycle on the LIVE gateway (labelled fallback)'
      $L.fallbackReason = 'sidecar failed to become healthy'
      $driver = 'live'
    }
  } elseif ($LifecycleMode -eq 'live') {
    $driver = 'live'
  }
  $L.driver = $driver
  if ($driver -eq 'sidecar') { $L.baseUrl = $script:G.Sidecar.url } else { $L.baseUrl = $script:G.LiveUrl }
  # slow the sidecar mission loop down: the default full_components profile
  # ticks every 3s which would add avoidable load to the shared llama-server.
  if ($driver -eq 'sidecar') {
    $cfg = Invoke-PostJson "$($L.baseUrl)/api/cognition/autonomy/loop" @{ action = 'configure'; intervalSec = 20; deliberateMaxTokens = 128 } 15
    Log "lifecycle: sidecar loop configure ok=$($cfg.ok) interval=20s"
  }
  $L.assignedAt = Get-Date
  if ($driver -eq 'live') {
    # The live supervisor parses ids as tr-<int>-<n>; keep that shape so its
    # backpressure/retirement sort never throws on our probe id.
    $n = [int]((Get-Date).ToString('ddHHmmss'))
    $L.missionId = "tr-$n-1"
  } else {
    $L.missionId = "stab-$($script:G.RunId)-m1"
  }
  $goal = "STABILITY-PROBE $($script:G.RunId): in one short paragraph (about 120 words), explain what a software watchdog does and why a restart loop needs backoff. Keep it on topic."
  $body = @{ id = $L.missionId; goal = $goal; pressureMode = 'linear' }
  $r = Invoke-PostJson "$($L.baseUrl)/api/mission/assign" $body 60
  if ($r.ok) {
    $L.stage = 'assigned'
    $L.assignMs = $r.ms
    Add-LifecycleStep 'assign' "driver=$driver url=$($L.baseUrl) id=$($L.missionId) httpMs=$($r.ms)" @{ goal = $goal }
  } else {
    $L.stage = 'error'; $L.error = "assign failed: $($r.error) $($r.raw)"
    Add-LifecycleStep 'error' "assign failed httpMs=$($r.ms) err=$($r.error)" @{ raw = $r.raw }
    $script:G.ForensicsGaps += "lifecycle assign failed on $driver driver"
  }
}

function Complete-Lifecycle([string]$detail) {
  $L = $script:G.Lifecycle
  $L.completedAt = Get-Date
  $L.stage = 'completed'
  $L.finalState = 2
  Add-LifecycleStep 'completed' $detail
  if ($L.mode -eq 'sidecar' -or $L.driver -eq 'sidecar') {
    Stop-Sidecar 'lifecycle complete'
  }
}

function Step-Lifecycle([int]$seq) {
  $L = $script:G.Lifecycle
  if ($L.stage -notin @('assigned', 'working', 'reported')) { return }
  $L.elapsedSec = [int]((Get-Date) - $L.assignedAt).TotalSeconds
  $m = Get-TrackedMission
  if ($m -eq $null) {
    if ($L.elapsedSec -gt 120 -and $L.stage -eq 'assigned') {
      $L.stage = 'error'; $L.error = 'mission never appeared in status/persisted state'
      Add-LifecycleStep 'error' "mission did not appear within 120s (driver=$($L.driver))"
    }
    return
  }
  if ($m.state -eq 2) {
    $L.autoCompleted = $true
    $L.deliverableChars = $m.deliverableLen
    $L.deliverableExcerpt = if ($m.deliverable.Length -gt 600) { $m.deliverable.Substring(0, 600) } else { $m.deliverable }
    Complete-Lifecycle "mission auto-completed by in-process worker (state=2, deliverable=$($m.deliverableLen) chars)"
    return
  }
  if ($m.state -eq 1) {
    if (-not $L.runningSeen) {
      $L.runningSeen = $true
      Add-LifecycleStep 'running' "state=Running deliverable=$($m.deliverableLen) pressure=$($m.pressure) source=$($m.source)"
    }
    if ($m.deliverableLen -gt 0 -and -not $L.workSeen) {
      $L.workSeen = $true
      $L.deliverableChars = $m.deliverableLen
      $L.deliverableExcerpt = if ($m.deliverable.Length -gt 600) { $m.deliverable.Substring(0, 600) } else { $m.deliverable }
      $L.pressureAtWork = $m.pressure
      Add-LifecycleStep 'work' "deliverable grew to $($m.deliverableLen) chars while Running (pressure=$($m.pressure)) - deliberator produced work"
    }
  }
  if ($L.stage -eq 'reported') {
    if ($L.elapsedSec -gt ($LifecycleTimeoutSec + 120)) {
      $L.stage = 'error'; $L.error = 'mission reported complete but state never became Completed'
      Add-LifecycleStep 'error' "state stuck at $($m.state) after report for $($L.elapsedSec)s"
      Stop-Sidecar 'lifecycle verification failed'
    }
    return
  }
  $shouldReport = $L.workSeen -or ($L.elapsedSec -ge $LifecycleTimeoutSec)
  if ($shouldReport) {
    $why = if ($L.workSeen) { 'work-observed' } else { 'work-timeout' }
    $repBody = @{ missionId = $L.missionId; goalAchieved = $true; summary = "stability lifecycle probe ($why) run=$($script:G.RunId)" }
    $r = Invoke-PostJson "$($L.baseUrl)/api/mission/report" $repBody 60
    if ($r.ok) {
      $L.stage = 'reported'; $L.reportMs = $r.ms; $L.reportedAt = Get-Date
      Add-LifecycleStep 'report' "goalAchieved=true ($why) httpMs=$($r.ms)"
      # the sidecar flips the record to Completed synchronously; verify now so
      # the transcript shows report -> completed back to back.
      for ($v = 0; $v -lt 3; $v++) {
        Start-Sleep -Milliseconds 700
        $m2 = Get-TrackedMission
        if ($m2 -and $m2.state -eq 2) {
          $L.deliverableChars = $m2.deliverableLen
          $L.deliverableExcerpt = if ($m2.deliverable.Length -gt 600) { $m2.deliverable.Substring(0, 600) } else { $m2.deliverable }
          Complete-Lifecycle "report accepted; state verified Completed (state=2, deliverable=$($m2.deliverableLen) chars)"
          break
        }
      }
    } else {
      $L.stage = 'error'; $L.error = "report failed: $($r.error) $($r.raw)"
      Add-LifecycleStep 'error' "report failed httpMs=$($r.ms) err=$($r.error)" @{ raw = $r.raw }
      Stop-Sidecar 'lifecycle report failed'
    }
  }
}

function Finalize-Lifecycle {
  $L = $script:G.Lifecycle
  if ($L.stage -in @('assigned', 'working', 'reported')) {
    # last chance: force a report and poll for the Completed transition
    $mb = @{ missionId = $L.missionId; goalAchieved = $true; summary = "stability lifecycle probe (end-of-run finalize) run=$($script:G.RunId)" }
    $r = Invoke-PostJson "$($L.baseUrl)/api/mission/report" $mb 60
    if ($r.ok) { $L.stage = 'reported'; $L.reportedAt = Get-Date; Add-LifecycleStep 'report' 'end-of-run finalize report goalAchieved=true' }
    $m = $null
    for ($v = 0; $v -lt 5; $v++) {
      Start-Sleep -Seconds 2
      $m = Get-TrackedMission
      if ($m -and $m.state -eq 2) { break }
    }
    if ($m -and $m.state -eq 2) {
      $L.autoCompleted = $false
      $L.deliverableChars = $m.deliverableLen
      $L.deliverableExcerpt = if ($m.deliverable.Length -gt 600) { $m.deliverable.Substring(0, 600) } else { $m.deliverable }
      Complete-Lifecycle 'mission Completed after end-of-run finalize report'
    } else {
      $L.stage = 'error'; $L.error = 'lifecycle did not reach Completed by end of run'
      Add-LifecycleStep 'error' "lifecycle incomplete at run end: stage=$($L.stage) state=$(if ($m) { $m.state } else { 'unseen' })"
      $script:G.ForensicsGaps += 'lifecycle did not reach Completed before end of run'
      Stop-Sidecar 'end of run (lifecycle incomplete)'
    }
  } elseif ($L.stage -eq 'completed') {
    # already stopped; record final deliverable values if empty
  } elseif ($L.stage -eq 'pending' -and $L.mode -ne 'none') {
    Stop-Sidecar 'end of run (lifecycle never started)'
  }
  if ($L.driver -eq 'sidecar') { Stop-Sidecar 'end of run' }
}

# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------
function Get-Sample([int]$seq) {
  $G = $script:G
  $now = Get-Date
  $s = [ordered]@{
    seq = $seq; ts = $now.ToString('o'); elapsedSec = [int]($now - $G.SamplingStart).TotalSeconds
    up = 0; httpMs = $null; httpErr = ''
    gpuOk = $false; gpuUtilMean = $null; gpuUtilMax = $null; gpuMemUsedMB = $null; gpuTempC = $null
    gwPid = 0; gwRssMB = $null; gwThreads = $null; gwCpuPct = $null; gwTick = $false; gwListenerZombie = $false
    zombieCount = 0
    llamaProcCount = 0; llamaRssMB = 0
    crashesTotal = 0; werCrashesTotal = 0; fatalRuntimeCrashesTotal = 0; restartsTotal = 0; pidChangesTotal = 0
    slotBusyTotal = 0; slotBusyDelta = 0; logRotations = 0
    missionRunning = $null; missionFinishedRecent = $null; missionCompletedExact = $null; missionCountsSource = ''
    expertAgeMin = $null; expertChanged = $false
    goalsRun = $null; goalsRunDelta = $null
    sidecarActive = $false; sidecarUp = $null
    lifecycleStage = (Get-Prop $G.Lifecycle 'stage')
    notes = ''
  }
  $notes = @()

  # --- gateway liveness + latency -----------------------------------------
  $h = Invoke-Get "$($G.LiveUrl)/api/health" $HealthTimeoutSec
  $s.httpMs = $h.ms
  if ($h.ok) { $s.up = 1 } else { $s.httpErr = $h.error; $notes += 'health-down' }

  # --- identity / RSS / CPU / zombies ------------------------------------
  $ident = Get-GatewayIdentity $G.LivePort
  $s.gwPid = $ident.pid
  $s.gwRssMB = $ident.rssMB
  $s.gwThreads = $ident.threads
  $s.gwListenerZombie = $ident.listenerZombie
  $s.zombieCount = @($ident.zombies).Count
  $G.LastZombies = @($ident.zombies)
  if ($ident.listenerZombie) { $notes += 'listener-owned-by-zombie' }
  if ($ident.pid -ne 0) {
    if ($G.PrevGwPid -ne 0 -and $G.PrevGwPid -ne $ident.pid) {
      $G.PidChanges++
      $G.RestartEvents += [ordered]@{ at = (NowIso); source = 'sample'; kind = 'gateway_pid_change'; detail = "pid $($G.PrevGwPid) -> $($ident.pid)" }
      $notes += "pid-changed"
    }
    $G.PrevGwPid = $ident.pid
    $prevCpu = $null
    if ($G.PrevGwCpu.ContainsKey([string]$ident.pid)) { $prevCpu = $G.PrevGwCpu[[string]$ident.pid] }
    if ($null -ne $prevCpu -and $null -ne $ident.cpuMs) {
      $deltaMs = [double]($ident.cpuMs - $prevCpu)
      $wallMs = [double]($now - $G.PrevSampleAt).TotalMilliseconds
      if ($wallMs -gt 0) { $s.gwCpuPct = [math]::Round(100.0 * $deltaMs / $wallMs, 2) }
      $s.gwTick = ($deltaMs -gt 50)
    }
    $G.PrevGwCpu[[string]$ident.pid] = $ident.cpuMs
  } else {
    $notes += 'gateway-process-not-found'
  }

  # --- llama-server rss ---------------------------------------------------
  $llama = Get-LlamaRssMB
  $s.llamaProcCount = $llama.count
  $s.llamaRssMB = $llama.rssMB

  # --- GPU ----------------------------------------------------------------
  $gpu = Get-GpuStats
  $s.gpuOk = $gpu.ok
  $s.gpuUtilMean = $gpu.utilMean; $s.gpuUtilMax = $gpu.utilMax
  $s.gpuMemUsedMB = $gpu.memUsedMB; $s.gpuTempC = $gpu.tempC
  if (-not $gpu.ok) { $notes += 'gpu-unavailable' }

  # --- appended logs ------------------------------------------------------
  $beforeSlot = $G.Counters.slotBusy
  Update-LogCounters
  $s.slotBusyTotal = $G.Counters.slotBusy
  $s.slotBusyDelta = $G.Counters.slotBusy - $beforeSlot
  $s.logRotations = $G.LogRotations.Count

  # --- crashes / restarts -------------------------------------------------
  # Two evidence channels, because the gateway SEH filter swallows ordinary
  # faults before WER (see Scan-FatalRuntime):
  #   (a) WER Application Error Id=1000 -> fail-fast codes (e.g. heap corruption)
  #   (b) runtime_store/logs/fatal_runtime.log "unhandled exception" -> SEH faults
  $newCrashes = @()
  $newCrashes += Scan-NewCrashes
  $newCrashes += Scan-FatalRuntime
  foreach ($c in $newCrashes) {
    $kindLabel = if ($c.PSObject.Properties['kind']) { $c.kind } else { 'wer' }
    Log "CRASH DETECTED ($kindLabel): $($c.appName) code=$($c.exceptionCode) at $($c.timeLocal)" 'ERROR'
    $notes += 'CRASH'
    Invoke-Forensics $c | Out-Null
  }
  $s.crashesTotal = $G.Crashes.Count
  $s.werCrashesTotal = @($G.Crashes | Where-Object { $_.kind -eq 'wer_application_error' }).Count
  $s.fatalRuntimeCrashesTotal = @($G.Crashes | Where-Object { $_.kind -eq 'fatal_runtime_seh' }).Count
  $s.restartsTotal = $G.Counters.wrapperStarted
  $s.pidChangesTotal = $G.PidChanges

  # --- mission counters (persisted each sample, authoritative every N) ----
  $snap = Read-AutonomySnapshot
  if ($snap.ok) {
    $mc = Get-PersistedMissionCounts $snap
    $s.missionRunning = $mc.running
    $s.missionFinishedRecent = $mc.finishedRecent
    $s.missionCountsSource = 'persisted(+64 finished cap)'
    if ($null -ne $snap.goalsRun) {
      if ($null -eq $G.GoalsRunStart) { $G.GoalsRunStart = [int64]$snap.goalsRun }
      $s.goalsRun = [int64]$snap.goalsRun
      $s.goalsRunDelta = [int64]$snap.goalsRun - [int64]$G.GoalsRunStart
    }
    # ticking also counts if the persisted mission iteration advanced
    if ($null -ne $snap.iteration) {
      if ($null -ne $G.LastIteration -and $snap.iteration -ne $G.LastIteration) { $s.gwTick = $true }
      $G.LastIteration = $snap.iteration
    }
  } else {
    $notes += 'autonomy_state-unreadable'
    if ($G.LastSnapshot -and $G.LastSnapshot.ok) {
      $mc = Get-PersistedMissionCounts $G.LastSnapshot
      $s.missionRunning = $mc.running; $s.missionFinishedRecent = $mc.finishedRecent
      $s.missionCountsSource = 'cached'
    }
  }
  if ($snap.ok) { $G.LastSnapshot = $snap }
  if ((Get-Date) -ge $G.NextAuthoritativeAt) {
    $ac = Get-AuthoritativeMissionCounts $G.LiveUrl
    if ($ac.ok) {
      $s.missionCompletedExact = $ac.completed
      $G.LastAuthoritative = $ac
      $G.NextAuthoritativeAt = (Get-Date).AddSeconds($MissionStatusEverySec)
      $s.missionCountsSource = 'authoritative+persisted'
      if ($ac.ms -gt 5000) { $notes += "authoritative-status-slow($($ac.ms)ms)" }
    } else {
      $notes += 'authoritative-status-failed'
      $G.NextAuthoritativeAt = (Get-Date).AddSeconds([math]::Min(60, $MissionStatusEverySec))
    }
  } elseif ($G.LastAuthoritative) {
    $s.missionCompletedExact = $G.LastAuthoritative.completed
  }

  # --- expert weights / population ---------------------------------------
  $w = Join-Path $G.Repo 'runtime_store\experts\study_tiny\weights.json'
  if (Test-Path $w) {
    $wi = Get-Item $w
    $s.expertAgeMin = [int]((Get-Date) - $wi.LastWriteTime).TotalMinutes
    $mt = $wi.LastWriteTime.ToString('o')
    if ($null -eq $G.ExpertMtime) { $G.ExpertMtime = $mt }
    elseif ($G.ExpertMtime -ne $mt) { $G.ExpertChanges++; $G.ExpertMtime = $mt; $s.expertChanged = $true }
  }

  # --- sidecar liveness ---------------------------------------------------
  if ($G.Sidecar.pid) {
    $p = Get-Process -Id $G.Sidecar.pid -ErrorAction SilentlyContinue
    $s.sidecarActive = [bool]$p
    if ($p) { $hh = Invoke-Get "$($G.Sidecar.url)/api/health" 5; $s.sidecarUp = $hh.ok; if (-not $hh.ok) { $notes += 'sidecar-health-down' } }
    else { $s.sidecarUp = $false; $notes += 'sidecar-process-gone' }
  }

  # --- exposure bookkeeping (up AND ticking, split by PID) ----------------
  if ($s.gwPid -gt 0) {
    $key = [string]$s.gwPid
    if (-not $G.Exposure.ContainsKey($key)) {
      $G.Exposure[$key] = [ordered]@{ pid = $s.gwPid; upTickingSamples = 0; upSamples = 0; samples = 0; maxWindowSec = 0; currentWindowSec = 0; windowStart = $null }
    }
    $e = $G.Exposure[$key]
    $e.samples++
    if ($s.up -eq 1) { $e.upSamples++ }
    $upTicking = ($s.up -eq 1 -and $s.gwTick)
    if ($upTicking) {
      $e.upTickingSamples++
      if ($null -eq $e.windowStart) { $e.windowStart = $now }
      $e.currentWindowSec = ($now - $e.windowStart).TotalSeconds
      if ($e.currentWindowSec -gt $e.maxWindowSec) { $e.maxWindowSec = $e.currentWindowSec }
    } else {
      $e.windowStart = $null
      $e.currentWindowSec = 0
    }
  }
  if ($s.up -eq 1) { $G.UpCount++ } else { $G.DownCount++ }

  # --- consecutive miss window -------------------------------------------
  if ($s.up -eq 0) {
    $G.MissStreak++
    if ($G.MissStreak -gt $G.MaxConsecutiveMiss) { $G.MaxConsecutiveMiss = $G.MissStreak }
  } else {
    $G.MissStreak = 0
  }

  $s.notes = ($notes -join ';')
  return [pscustomobject]$s
}

# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------
function Get-Percentile($values, [double]$p) {
  if ($null -eq $values -or @($values).Count -eq 0) { return $null }
  $arr = @($values | Sort-Object)
  $idx = [int][math]::Ceiling($p * ($arr.Count - 1))
  return $arr[$idx]
}

function Build-Criteria {
  $G = $script:G
  $L = $G.Lifecycle
  $samples = @($G.Samples)
  $n = $samples.Count
  $upCount = $G.UpCount
  $upRate = if ($n -gt 0) { [math]::Round($upCount / $n, 4) } else { 0 }
  # exposure: sum of (up AND ticking) sample intervals, split by PID -> hours
  $exposureSec = 0.0
  $maxWindowSec = 0.0
  foreach ($k in $G.Exposure.Keys) {
    $e = $G.Exposure[$k]
    $exposureSec += ($e.upTickingSamples * $IntervalSec)
    if ($e.maxWindowSec -gt $maxWindowSec) { $maxWindowSec = $e.maxWindowSec }
  }
  $exposureHours = [math]::Round($exposureSec / 3600.0, 3)
  $crashes = $G.Crashes.Count
  # rule of three (95% one-sided, zero events): lambda_upper = 3 / exposure.
  # For >=1 observed event use the Poisson (chi-square) upper bound approx.
  $lambdaUpper = $null
  if ($exposureHours -gt 0) {
    if ($crashes -eq 0) { $lambdaUpper = [math]::Round(3.0 / $exposureHours, 3) }
    else { $lambdaUpper = [math]::Round(((($crashes + 1) + 1.96 * [math]::Sqrt($crashes + 1))) / $exposureHours, 3) }
  }
  $sRung = 'BELOW-S1'
  $sLabel = 'SELF-TEST / below S1 - no stability claim (exposure < 3h)'
  if ($crashes -gt 0) {
    $sRung = 'CRASH-OBSERVED'
    $sLabel = "FAIL: crash observed (no zero-event bound applies)"
  } elseif ($exposureHours -ge 30) {
    $sRung = 'S3'; $sLabel = 'S3: 30h+ zero-crash / lambda < 0.1/h - may relax the coarse lock'
  } elseif ($exposureHours -ge 12) {
    $sRung = 'S2'; $sLabel = 'S2: 12h zero-crash = significant reduction'
  } elseif ($exposureHours -ge 3) {
    $sRung = 'S1'; $sLabel = 'S1: 3h zero-crash = no new problem found'
  }
  $expected = [int][math]::Floor($Minutes * 60 / $IntervalSec) + 1
  $coverage = if ($expected -gt 0) { [math]::Round($n / $expected, 4) } else { 0 }
  $lifecycleCompleted = ($L.stage -eq 'completed' -and $L.finalState -eq 2)
  $lifecycleSteps = @($L.steps | ForEach-Object { $_.stage })
  $stepsOk = (@('assign', 'running', 'work', 'report', 'completed') | Where-Object { $lifecycleSteps -notcontains $_ }).Count -eq 0
  $gpuOkSamples = @($samples | Where-Object { $_.gpuOk }).Count
  $werCrashCount = @($script:G.Crashes | Where-Object { $_.kind -eq 'wer_application_error' }).Count
  $fatalRuntimeCount = @($script:G.Crashes | Where-Object { $_.kind -eq 'fatal_runtime_seh' }).Count
  $checks = @()
  $checks += [ordered]@{
    id = 'C1_no_crash'; ok = ($crashes -eq 0); value = "crashEvents=$crashes (wer=$werCrashCount fatalRuntime=$fatalRuntimeCount)"; threshold = '0 crash events: WER Id=1000 + fatal_runtime.log unhandled exceptions'
    note = 'WER Application Error Id=1000 AND runtime_store/logs/fatal_runtime.log unhandled-exception lines for phoenix_main/llama-server/sidecar'
  }
  $checks += [ordered]@{
    id = 'C2_up_rate'; ok = ($upRate -ge 0.99); value = "$($upCount)/$n = $upRate"; threshold = '>= 0.99'
    note = "live gateway /api/health within ${HealthTimeoutSec}s; zombie images ignored (RSS<100MB+0 threads do not count)"
  }
  $checks += [ordered]@{
    id = 'C3_consecutive_misses'; ok = ($G.MaxConsecutiveMiss -le 2); value = "$($G.MaxConsecutiveMiss)"; threshold = '<= 2'
    note = 'at 60s sampling a 3-miss run is >=3 min without a serving gateway'
  }
  $checks += [ordered]@{
    id = 'C4_mission_completed'; ok = [bool]$lifecycleCompleted; value = "finalState=$($L.finalState) stage=$($L.stage) driver=$($L.driver)"; threshold = '>= 1 mission reaching Completed (state=2)'
    note = 'lifecycle driver mission; report/complete is part of the exercised lifecycle'
  }
  $checks += [ordered]@{
    id = 'C5_lifecycle_steps'; ok = $stepsOk; value = ($lifecycleSteps -join '>'); threshold = 'assign>running>work>report>completed all observed'
    note = 'work evidence = a non-empty deliverable was produced while the mission was Running'
  }
  $checks += [ordered]@{
    id = 'C6_sample_coverage'; ok = ($coverage -ge 0.99); value = "$n/$expected = $coverage"; threshold = '>= 0.99'
    note = 'timer drift / harness stalls would reduce coverage'
  }
  $checks += [ordered]@{
    id = 'C7_gpu_reported'; ok = ($gpuOkSamples -gt 0); value = "$gpuOkSamples/$n samples"; threshold = 'GPU mean+max reported (no util threshold)'
    note = 'GPU utilisation is workload-dependent; a threshold would encode the workload, not stability'
  }
  $checks += [ordered]@{
    id = 'C8_exposure_reported'; ok = ($exposureHours -gt 0); value = "exposureHours=$exposureHours lambdaUpper95=$lambdaUpper/h rung=$sRung"; threshold = 'up AND ticking hours split by PID must be reported'
    note = 'wall-clock is not exposure; restarts reset the clock per PID'
  }
  $pass = (@($checks | Where-Object { -not $_.ok }).Count -eq 0)
  $verdict = 'FAIL'
  if ($crashes -gt 0) { $verdict = 'FAIL' }
  elseif ($pass -and $exposureHours -ge 3) { $verdict = "PASS ($sRung)" }
  elseif ($pass) { $verdict = 'SELF-TEST (below S1: all checks green, no stability claim)' }
  return [pscustomobject]@{
    checks = $checks; pass = $pass; verdict = $verdict
    upRate = $upRate; upCount = $upCount; sampleCount = $n; expectedSamples = $expected; coverage = $coverage
    exposureHours = $exposureHours; lambdaUpper = $lambdaUpper; sRung = $sRung; sLabel = $sLabel
    maxWindowSec = [math]::Round($maxWindowSec, 0); crashes = $crashes
    lifecycleCompleted = $lifecycleCompleted; stepsOk = $stepsOk; gpuOkSamples = $gpuOkSamples
  }
}

function Write-Summary {
  $G = $script:G
  if ($G.SummaryWritten) { return }
  $G.SummaryWritten = $true
  $crit = Build-Criteria
  $L = $G.Lifecycle
  $samples = @($G.Samples)
  $n = $samples.Count
  $endForCalc = if ($G.EndTime) { $G.EndTime } else { Get-Date }
  $actualMinutes = [math]::Round(($endForCalc - $G.SamplingStart).TotalMinutes, 2)
  $httpOk = @($samples | Where-Object { $_.up -eq 1 -and $null -ne $_.httpMs } | ForEach-Object { $_.httpMs })
  $gpuVals = @($samples | Where-Object { $_.gpuOk } | ForEach-Object { $_.gpuUtilMean })
  $gwRss = @($samples | Where-Object { $null -ne $_.gwRssMB } | ForEach-Object { $_.gwRssMB })
  $zombieMax = 0; $zombieMin = $null
  foreach ($s in $samples) {
    if ($null -ne $s.zombieCount) {
      if ($s.zombieCount -gt $zombieMax) { $zombieMax = $s.zombieCount }
      if ($null -eq $zombieMin -or $s.zombieCount -lt $zombieMin) { $zombieMin = $s.zombieCount }
    }
  }
  # MMLU context
  $mmluNew = @()
  $mmluPath = Join-Path $G.Repo 'runtime_store\long_run\mmlu_history.jsonl'
  if (Test-Path $mmluPath) {
    $all = @(Get-Content $mmluPath -ErrorAction SilentlyContinue)
    $newLines = @($all | Select-Object -Skip ([int]$G.MmluBaselineLines))
    foreach ($ln in $newLines) {
      try { $rec = $ln | ConvertFrom-Json; $mmluNew += [pscustomobject]@{ ts = $rec.ts; correct = $rec.correct; total = $rec.total; score = $rec.score } } catch { }
    }
  }
  $mmluScores = @($mmluNew | ForEach-Object { [double]$_.score })
  $mmluMean = if ($mmluScores.Count -gt 0) { [math]::Round((($mmluScores | Measure-Object -Average).Average), 4) } else { $null }

  # ---- summary.json ----
  $json = [ordered]@{
    schema = 'phoenix.stability.v1'
    protocol = 'doc/v10.1/stability_protocol.md'
    generatedAt = NowIso
    runId = $G.RunId; label = $G.Label; status = $G.Status
    command = $G.Command
    host = $env:COMPUTERNAME; user = $env:USERNAME
    run = [ordered]@{
      startedAt = $G.RunStart.ToString('o'); endedAt = $G.EndTime.ToString('o')
      plannedMinutes = $Minutes; actualMinutes = $actualMinutes
      intervalSec = $IntervalSec; samples = $n; expectedSamples = $crit.expectedSamples
      liveGatewayUrl = $G.LiveUrl; liveGatewayPort = $G.LivePort
      liveGatewayPidAtStart = $G.LivePidStart; liveGatewayPidAtEnd = $G.PidChanges
      lifecycleMode = $LifecycleMode; lifecycleDriver = $L.driver; lifecycleStage = $L.stage
      sidecarPort = $G.Sidecar.port; sidecarPid = $G.Sidecar.pid; sidecarHealthy = $G.Sidecar.healthy
    }
    build = $G.Build
    criteria = $crit.checks
    verdict = [ordered]@{
      overall = $crit.verdict; pass = $crit.pass; sRung = $crit.sRung; sLabel = $crit.sLabel
      stabilityClaimEligible = ($crit.exposureHours -ge 3)
      lambdaUpper95PerHour = $crit.lambdaUpper
      exposureHours = $crit.exposureHours
      maxContinuousWindowSec = $crit.maxWindowSec
      note = $crit.sLabel
    }
    exposure = [ordered]@{
      definition = 'sum over samples of intervalSec where gateway health=up AND ticking, split by gateway PID; restart resets the window'
      tickingDefinition = 'gateway process CPU time advanced >50ms over the interval OR supervisor training rounds advanced OR persisted autonomy iteration advanced'
      totalHours = $crit.exposureHours
      maxContinuousWindowSec = $crit.maxWindowSec
      byPid = @($G.Exposure.Values)
      restartCount = $G.Counters.wrapperStarted
      pidChanges = $G.PidChanges
    }
    metrics = [ordered]@{
      upRate = $crit.upRate; upSamples = $crit.upCount; downSamples = $G.DownCount
      maxConsecutiveMiss = $G.MaxConsecutiveMiss
      httpMs = [ordered]@{
        min = if ($httpOk.Count) { ($httpOk | Measure-Object -Minimum).Minimum } else { $null }
        p50 = Get-Percentile $httpOk 0.5; p95 = Get-Percentile $httpOk 0.95
        max = if ($httpOk.Count) { ($httpOk | Measure-Object -Maximum).Maximum } else { $null }
        samples = $httpOk.Count
      }
      gpu = [ordered]@{
        reported = ($gpuVals.Count -gt 0); meanUtil = if ($gpuVals.Count) { [math]::Round((($gpuVals | Measure-Object -Average).Average), 1) } else { $null }
        maxUtil = if ($gpuVals.Count) { [math]::Round((($gpuVals | Measure-Object -Maximum).Maximum), 1) } else { $null }
        memUsedMBMax = (@($samples | Where-Object { $_.gpuMemUsedMB } | ForEach-Object { $_.gpuMemUsedMB }) | Measure-Object -Maximum).Maximum
        tempCMax = (@($samples | Where-Object { $_.gpuTempC } | ForEach-Object { $_.gpuTempC }) | Measure-Object -Maximum).Maximum
        samples = $gpuVals.Count
      }
      gatewayRssMB = [ordered]@{
        first = if ($gwRss.Count) { $gwRss[0] } else { $null }
        last = if ($gwRss.Count) { $gwRss[-1] } else { $null }
        mean = if ($gwRss.Count) { [math]::Round((($gwRss | Measure-Object -Average).Average), 1) } else { $null }
        max = if ($gwRss.Count) { ($gwRss | Measure-Object -Maximum).Maximum } else { $null }
      }
      llamaRssMBLast = if ($n -gt 0) { $samples[-1].llamaRssMB } else { $null }
      zombieImages = [ordered]@{ min = $zombieMin; max = $zombieMax; last = @($G.LastZombies); note = 'WS<100MB and 0 threads; excluded from gateway identity/up-rate' }
      crashEvents = $G.Crashes.Count
      crashEventsWer = @($G.Crashes | Where-Object { $_.kind -eq 'wer_application_error' }).Count
      crashEventsFatalRuntime = @($G.Crashes | Where-Object { $_.kind -eq 'fatal_runtime_seh' }).Count
      restartEvents = $G.RestartEvents.Count; wrapperRestarts = $G.Counters.wrapperStarted; pidChanges = $G.PidChanges
      slotBusyEvents = $G.Counters.slotBusy
      missionLlmFailures = $G.Counters.missionLlmFail
      supervisorRestarts = [ordered]@{ gateway = $G.Counters.supGatewayRestarts; llama = $G.Counters.supLlamaRestarts }
      missionCounts = [ordered]@{
        persistedRunningAtEnd = if ($n -gt 0) { $samples[-1].missionRunning } else { $null }
        persistedRecentlyFinishedAtEnd = if ($n -gt 0) { $samples[-1].missionFinishedRecent } else { $null }
        authoritativeCompletedAtEnd = $crit.lifecycleCompleted
        # counts only: the raw /api/mission/status container holds ~10k missions
        # and must never be embedded in summary.json (it would be tens of MB)
        authoritativeLast = if ($G.LastAuthoritative) {
          [ordered]@{
            running = $G.LastAuthoritative.running; completed = $G.LastAuthoritative.completed
            failed = $G.LastAuthoritative.failed; idle = $G.LastAuthoritative.idle
            total = $G.LastAuthoritative.total; httpMs = $G.LastAuthoritative.ms; sampledAt = NowIso
          }
        } else { $null }
      }
      expertWeights = [ordered]@{ path = 'runtime_store/experts/study_tiny/weights.json'; lastMtime = $G.ExpertMtime; updatesDuringRun = $G.ExpertChanges }
      population = [ordered]@{ goalsRunStart = $G.GoalsRunStart; goalsRunEnd = $G.GoalsRunEnd; goalsRunDelta = $G.GoalsRunDelta }
    }
    lifecycle = [ordered]@{
      mode = $L.mode; driver = $L.driver; missionId = $L.missionId; baseUrl = $L.baseUrl
      assignedAt = if ($L.assignedAt) { ([datetime]$L.assignedAt).ToString('o') } else { $null }
      reportedAt = if ($L.reportedAt) { ([datetime]$L.reportedAt).ToString('o') } else { $null }
      completedAt = if ($L.completedAt) { ([datetime]$L.completedAt).ToString('o') } else { $null }
      finalState = $L.finalState; autoCompleted = $L.autoCompleted
      workEvidence = [ordered]@{ seen = $L.workSeen; deliverableChars = $L.deliverableChars; pressureAtWork = $L.pressureAtWork; excerpt = $L.deliverableExcerpt }
      fallbackReason = $L.fallbackReason; error = $L.error
      steps = $L.steps
    }
    forensics = [ordered]@{
      hooks = 'WER Application Error scan each sample; dump lookup in CrashDumps + WER ReportQueue/Archive; tools/dump_stack_resolve.py --map phoenix_main.map'
      dumpResolver = 'tools/dump_stack_resolve.py'; map = 'phoenix_main.map'
      crashes = @($G.Crashes); gaps = @($G.ForensicsGaps)
    }
    learningContext = [ordered]@{
      note = 'MMLU/other learning metrics are CONTEXT ONLY (noisy, LLM-limited); never a pass/fail criterion'
      mmluHistoryPath = 'runtime_store/long_run/mmlu_history.jsonl'
      newRecords = $mmluNew.Count; meanScore = $mmluMean
      maxScore = if ($mmluScores.Count) { ($mmluScores | Measure-Object -Maximum).Maximum } else { $null }
      minScore = if ($mmluScores.Count) { ($mmluScores | Measure-Object -Minimum).Minimum } else { $null }
      records = $mmluNew
    }
    harness = [ordered]@{
      version = 'stability_run.ps1 v1'
      cleanup = $G.Cleanup
      liveUntouched = $G.LiveUntouched
      errors = @($G.Errors)
    }
    artifacts = [ordered]@{
      harnessLog = $G.LogPath; samplesJsonl = $G.JsonlPath; samplesCsv = (Join-Path $G.Bundle 'samples.csv')
      lifecycleJsonl = (Join-Path $G.Bundle 'lifecycle.jsonl'); forensicsDir = (Join-Path $G.Bundle 'forensics')
      summaryJson = (Join-Path $G.Bundle 'summary.json'); summaryMd = (Join-Path $G.Bundle 'summary.md')
      cleanupJson = (Join-Path $G.Bundle 'cleanup.json')
    }
  }
  Set-Content -Path (Join-Path $G.Bundle 'summary.json') -Value ($json | ConvertTo-Json -Depth 14) -Encoding utf8

  # ---- summary.md ----
  $md = @()
  $md += "# Stability run - v10.1 protocol"
  $md += ''
  $md += "- **Run id**: ``$($G.RunId)``  label: ``$($G.Label)``"
  $md += "- **Started / ended**: $($G.RunStart.ToString('yyyy-MM-dd HH:mm:ss')) -> $(if ($G.EndTime) { $G.EndTime.ToString('yyyy-MM-dd HH:mm:ss') } else { 'running' })"
  $md += "- **Planned / actual**: $Minutes min planned, $($actualMinutes) min sampled, interval ${IntervalSec}s, samples $n/$($crit.expectedSamples)"
  $md += "- **Live gateway (observed read-only)**: ``$($G.LiveUrl)`` pid $($G.LivePidStart) -> $($G.PidAtEnd)"
  $md += "- **Lifecycle driver**: $($L.driver) ($($L.mode) mode) - sidecar itself: port $($G.Sidecar.port), healthy=$($G.Sidecar.healthy)"
  $md += "- **Build**: $($G.Build.phoenixVersion) exe $($G.Build.phoenixExe.sizeBytes) bytes mtime $($G.Build.phoenixExe.mtimeLocal) sha256 ``$($G.Build.phoenixExe.sha256.Substring(0,16))...``"
  $md += ''
  $md += "## Verdict"
  $md += ''
  $md += "**$($crit.verdict)**"
  $md += ''
  $md += "- Stability claim rung: **$($crit.sRung)** - $($crit.sLabel)"
  $md += "- Observed exposure (up AND ticking, per PID, restarts reset the clock): **$($crit.exposureHours) h**"
  $md += "- Max continuous serving window: **$([int]$crit.maxWindowSec) s**; restarts: **$($G.Counters.wrapperStarted)** (wrapper) / pid changes: **$($G.PidChanges)**"
  $md += "- Zero-crash 95% upper bound on crash rate (rule of three / Hanley 1983): **lambda < $($crit.lambdaUpper) / h**"
  $md += ''
  $md += "| exposure | zero-crash 95% lambda upper | rung | meaning |"
  $md += "|---|---|---|---|"
  $md += "| 30 min | < 6 / h | below S1 | not a stability result |"
  $md += "| 3 h | < 1 / h | S1 | no new problem found |"
  $md += "| 12 h | < 0.25 / h | S2 | significant reduction |"
  $md += "| 30 h+ | < 0.1 / h | S3 | may relax the coarse lock |"
  $md += ''
  $md += "> A short run can only validate the harness; it never proves stability. This run's own lambda bound and rung are printed above."
  $md += ''
  $md += "## Metrics"
  $md += ''
  $md += "| metric | value | note |"
  $md += "|---|---|---|"
  $md += "| samples / expected | $n / $($crit.expectedSamples) (coverage $($crit.coverage)) | 60s-by-default sampling; coverage catches harness stalls |"
  $md += "| gateway up-rate | $($crit.upCount)/$n = $($crit.upRate) | /api/health within ${HealthTimeoutSec}s; zombie images ignored |"
  $md += "| max consecutive misses | $($G.MaxConsecutiveMiss) | threshold <= 2 |"
  $md += "| HTTP health latency p50 / p95 / max | $(Get-Percentile $httpOk 0.5) / $(Get-Percentile $httpOk 0.95) / $(if ($httpOk.Count) { ($httpOk | Measure-Object -Maximum).Maximum }) ms | includes gateway worker stalls |"
  $md += "| GPU util mean / max | $(if ($gpuVals.Count) { [math]::Round((($gpuVals | Measure-Object -Average).Average),1) } else { 'n/a' }) % / $(if ($gpuVals.Count) { ($gpuVals | Measure-Object -Maximum).Maximum } else { 'n/a' }) % | reported, **no threshold** (workload-dependent) |"
  $md += "| GPU mem max / temp max | $(if ($samples | Where-Object { $_.gpuMemUsedMB }) { (@($samples | Where-Object { $_.gpuMemUsedMB } | ForEach-Object { $_.gpuMemUsedMB }) | Measure-Object -Maximum).Maximum } else { 'n/a' }) MB / $(if ($samples | Where-Object { $_.gpuTempC }) { (@($samples | Where-Object { $_.gpuTempC } | ForEach-Object { $_.gpuTempC }) | Measure-Object -Maximum).Maximum } else { 'n/a' }) C | context |"
  $md += "| gateway RSS first / last / max | $(if ($gwRss.Count) { $gwRss[0] }) / $(if ($gwRss.Count) { $gwRss[-1] }) / $(if ($gwRss.Count) { ($gwRss | Measure-Object -Maximum).Maximum }) MB | growth is a watch item, not a criterion |"
  $md += "| llama-server RSS (last) | $($samples[-1].llamaRssMB) MB over $($samples[-1].llamaProcCount) procs | 8B + coder + granite |"
  $md += "| zombie phoenix_main images | min $zombieMin / max $zombieMax | 0-thread ~32KB images; excluded from identity/up-rate |"
  $md += "| crash events (WER + fatal_runtime) | $($G.Crashes.Count) (wer=$(($G.Crashes | Where-Object { $_.kind -eq 'wer_application_error' }).Count) fatalRuntime=$(($G.Crashes | Where-Object { $_.kind -eq 'fatal_runtime_seh' }).Count)) | 0 required; two channels (see Forensics hook) |"
  $md += "| wrapper restarts / pid changes | $($G.Counters.wrapperStarted) / $($G.PidChanges) | restarts reset the exposure clock |"
  $md += "| llama slot-wait failures (`llama-slot-busy`) | $($G.Counters.slotBusy) | informational; expected under slot contention |"
  $md += "| mission LLM failures | $($G.Counters.missionLlmFail) | informational |"
  $md += "| missions: persisted running / recently finished | $(if ($n) { $samples[-1].missionRunning }) / $(if ($n) { $samples[-1].missionFinishedRecent }) | persisted snapshot + last-64-finished cap |"
  $md += "| missions: authoritative completed total | $(if ($G.LastAuthoritative) { $G.LastAuthoritative.completed } else { 'n/a' }) | whole in-memory container |"
  $md += "| expert weights age / updates | $(if ($samples[-1].expertAgeMin -ne $null) { $samples[-1].expertAgeMin }) min / $($G.ExpertChanges) | runtime_store/experts/study_tiny/weights.json |"
  $md += "| population goalsRun delta | $($G.GoalsRunDelta) | studyPopulation.evolve cadence |"
  $md += ''
  $md += "## Pass/fail criteria (explicit)"
  $md += ''
  $md += "| # | criterion | threshold | observed | verdict |"
  $md += "|---|---|---|---|---|"
  foreach ($c in $crit.checks) {
    $md += "| $($c.id) | $($c.note) | $($c.threshold) | $($c.value) | $(if ($c.ok) { 'PASS' } else { 'FAIL' }) |"
  }
  $md += ''
  $md += "### Why these criteria"
  $md += ''
  $md += "- **C1 0 crashes**: a crash is the only unambiguous stability failure. Two channels are required because the gateway's in-process SEH filter swallows ordinary access violations: WER Application Error Id=1000 (fail-fast codes such as 0xC0000374) AND ``runtime_store/logs/fatal_runtime.log`` unhandled-exception lines (0xC0000005 etc.). Zero events bounds the rate (lambda 95% upper = 3/exposure) - it never proves absence."
  $md += "- **C2 up-rate >= 99%**: the gateway must serve almost every 60s probe. 99% allows ~1 missed sample per 100 (~1.8 in a 3h run) for a legitimate stall, while a dead gateway shows immediately. Identity is the :port listener only when RSS>=100MB; the ~7 zero-thread 32KB zombie images are excluded."
  $md += "- **C3 no >2 consecutive misses**: the gateway_loop only kills an unresponsive gateway after 5 consecutive unhealthy minutes, so a 3-miss (>=3 min at default interval) run would be relying on the wrapper to mask a dead gateway."
  $md += "- **C4 >=1 mission Completed**: proves the mission state machine reaches Completed through the harness-driven lifecycle (assign -> work -> report/complete). If it cannot, the summary states why."
  $md += "- **C5 lifecycle steps observed**: the protocol is specifically about a full lifecycle, so assign, Running, produced-work, report and Completed must all appear in lifecycle.jsonl."
  $md += "- **C6 sample coverage >= 99%**: guards against the harness itself stalling (the evidence must be dense enough to trust the other numbers)."
  $md += "- **C7 GPU mean+max reported, no threshold**: GPU utilisation is dominated by concurrent training/mission workload; a threshold would encode the workload, not the platform's stability. C7 only requires the metric to be collected."
  $md += "- **C8 exposure reported**: exposure = sum of (up AND ticking) hours split by PID, so a flapping/restarting gateway cannot buy hours in wall-clock terms. The rule-of-three lambda bound and S1/S2/S3 rung follow from it."
  $md += ''
  $md += "## Lifecycle transcript (assign -> work -> report/complete)"
  $md += ''
  if ($L.missionId) {
    $md += "- mission id: ``$($L.missionId)``, driver: **$($L.driver)**"
    if ($L.fallbackReason) { $md += "- FALLBACK: $($L.fallbackReason) - lifecycle was driven on the LIVE gateway and that is stated explicitly" }
    if ($L.workSeen) { $md += "- work evidence: deliverable grew to **$($L.deliverableChars)** chars while Running (pressure at work $($L.pressureAtWork))" } else { $md += "- work evidence: NONE observed (LLM slot starvation or slow model); report/complete still exercised" }
    if ($L.autoCompleted) { $md += "- completed by the in-process worker (`{done:true}` path), not by the harness report" }
  } else { $md += "- no lifecycle was driven ($(if ($L.error) { $L.error } else { 'LifecycleMode=' + $LifecycleMode }))" }
  $md += ''
  $md += "| # | at | elapsed s | stage | detail |"
  $md += "|---|---|---|---|---|"
  $i = 0
  foreach ($st in $L.steps) {
    $i++
    $md += "| $i | $($st.at) | $($st.elapsedSec) | $($st.stage) | $($st.detail) |"
  }
  $md += ''
  $md += "- final observed mission state: **$($L.finalState)** (2 = Completed)"
  $md += ''
  $md += "Raw lifecycle transcript (``lifecycle.jsonl``, verbatim - one JSON object per recorded step):"
  $md += ''
  $md += '```json'
  foreach ($st in $L.steps) { $md += ($st | ConvertTo-Json -Compress -Depth 6) }
  $md += '```'
  if ($L.deliverableExcerpt) {
    $md += ''
    $md += "Deliverable excerpt:"
    $md += ''
    $md += '```'
    $md += $L.deliverableExcerpt
    $md += '```'
  }
  $md += ''
  $md += "## Forensics hook"
  $md += ''
  $md += "Two crash-evidence channels are scanned every sample: WER ``Application`` Id=1000 (fail-fast codes such as 0xC0000374) and ``runtime_store/logs/fatal_runtime.log`` unhandled-exception lines (ordinary SEH faults such as 0xC0000005 are swallowed by the gateway's own in-process filter and never reach WER)."
  $md += ''
  if ($G.Crashes.Count -eq 0) {
    $md += "No crash events on either channel during this run, so the dump resolver was not invoked."
  } else {
    $md += "| crash (channel) | app / pid | exception | dump path | resolve report |"
    $md += "|---|---|---|---|---|"
    foreach ($c in $G.Crashes) {
      $md += "| $($c.time) ($($c.kind)) | $($c.appName) / $($c.faultingPid) | $($c.exceptionCode) | $(if ($c.dumpPath) { $c.dumpPath } else { '(none)' }) | $(if ($c.resolveReport) { $c.resolveReport } else { '(none)' }) |"
    }
  }
  if ($G.ForensicsGaps.Count -gt 0) {
    $md += ''
    $md += "Recorded gaps (explicit, not silently ignored):"
    $md += ''
    foreach ($gp in ($G.ForensicsGaps | Select-Object -Unique)) { $md += "- $gp" }
  }
  $md += ''
  $md += "## Learning metrics (context only - NOT a pass/fail criterion)"
  $md += ''
  $md += "MMLU subset records appended during this run: **$($mmluNew.Count)**; mean score $(if ($null -ne $mmluMean) { $mmluMean } else { 'n/a' }) (min $(if ($mmluScores.Count) { ($mmluScores | Measure-Object -Minimum).Minimum } else { 'n/a' }), max $(if ($mmluScores.Count) { ($mmluScores | Measure-Object -Maximum).Maximum } else { 'n/a' }))."
  $md += "These are 40-question samples and noisy; they are reported for context only and must never drive a build verdict."
  if ($mmluNew.Count -gt 0) {
    $md += ''
    $md += "| ts | correct | total | score |"
    $md += "|---|---|---|---|"
    foreach ($r in $mmluNew) { $md += "| $($r.ts) | $($r.correct) | $($r.total) | $($r.score) |" }
  }
  $md += ''
  $md += "## Artifacts"
  $md += ''
  $md += "- samples: ``samples.jsonl`` / ``samples.csv`` ($n rows)"
  $md += "- lifecycle: ``lifecycle.jsonl``; sidecar logs ``lifecycle/sidecar.{out,err}.log``"
  $md += "- forensics: ``forensics/``"
  $md += "- machine-readable: ``summary.json``; cleanup record: ``cleanup.json``"
  $md += ''
  $md += "## Reproduce"
  $md += ''
  $md += '```'
  $md += $G.Command
  $md += '```'
  $md += ''
  $md += "Compare two builds by running the same command with different ``-Label`` and diffing ``summary.json`` -> ``verdict``, ``metrics`` and ``lifecycle`` (see doc/v10.1/stability_protocol.md section 6)."
  $md += ''
  $md += "## Harness self-check"
  $md += ''
  $md += "- live gateway untouched: $($G.LiveUntouched)"
  $md += "- sidecar stopped: $($G.Cleanup.sidecarStopped); sidecar dir removed: $($G.Cleanup.sidecarDirRemoved)"
  $md += "- errors: $(if ($G.Errors.Count) { ($G.Errors -join ' | ') } else { 'none' })"
  $md += ''
  Set-Content -Path (Join-Path $G.Bundle 'summary.md') -Value ($md -join "`r`n") -Encoding utf8
  Log "summary written: $(Join-Path $G.Bundle 'summary.md')"
}

# ==========================================================================
# main
# ==========================================================================
$script:G = $null

# ---- CleanupOnly mode ----------------------------------------------------
if ($CleanupOnly) {
  if (-not $BundleDir) { Write-Host '-CleanupOnly requires -BundleDir <bundle>' -ForegroundColor Red; exit 2 }
  $recPath = Join-Path $BundleDir 'cleanup.json'
  if (-not (Test-Path $recPath)) { Write-Host "no cleanup.json in $BundleDir" -ForegroundColor Red; exit 2 }
  $rec = Get-Content $recPath -Raw | ConvertFrom-Json
  Write-Host "cleanup record: pid=$($rec.sidecarPid) dir=$($rec.sidecarDir)"
  if ($rec.sidecarPid) {
    $p = Get-Process -Id ([int]$rec.sidecarPid) -ErrorAction SilentlyContinue
    if ($p) {
      $pp = $p.Path
      if ($pp -and $rec.sidecarDir -and $pp -like ($rec.sidecarDir + '\*')) {
        & taskkill.exe /T /F /PID ([int]$rec.sidecarPid) 2>&1 | Out-Null
        Write-Host "killed sidecar pid $($rec.sidecarPid)"
      } else {
        Write-Host "pid $($rec.sidecarPid) does not point into $($rec.sidecarDir); refusing to kill" -ForegroundColor Yellow
      }
    } else { Write-Host "sidecar pid $($rec.sidecarPid) already gone" }
  }
  if ($rec.sidecarDir -and (Test-Path $rec.sidecarDir)) {
    Remove-Item -Path $rec.sidecarDir -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "removed $($rec.sidecarDir)"
  }
  exit 0
}

if ($Minutes -lt 1) { Write-Host '-Minutes must be >= 1' -ForegroundColor Red; exit 2 }
if ($IntervalSec -lt 5) { Write-Host '-IntervalSec must be >= 5' -ForegroundColor Red; exit 2 }

$runStart = Get-Date
$safeLabel = if ($Label) { ($Label -replace '[^A-Za-z0-9\._-]', '_') } else { 'run' }
$runId = (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + $safeLabel
$bundle = if ($OutRoot) { Join-Path $OutRoot $runId } else { Join-Path $RepoRoot "runtime_store\stability\$runId" }
New-Item -ItemType Directory -Force $bundle | Out-Null
New-Item -ItemType Directory -Force (Join-Path $bundle 'lifecycle') | Out-Null
New-Item -ItemType Directory -Force (Join-Path $bundle 'forensics') | Out-Null

$G = @{
  Repo = (Resolve-Path $RepoRoot).Path
  Bundle = $bundle
  RunId = $runId
  Label = $(if ($Label) { $Label } else { $safeLabel })
  SafeLabel = $safeLabel
  RunStart = $runStart
  EndTime = $null
  Status = 'starting'
  LogPath = (Join-Path $bundle 'harness.log')
  JsonlPath = (Join-Path $bundle 'samples.jsonl')
  Samples = [System.Collections.Generic.List[object]]::new()
  Errors = @()
  Crashes = @(); CrashIds = @{}; FatalRuntimeSeen = @{}; RestartEvents = @(); ForensicsGaps = @()
  LogOffsets = @{}; LogRotations = @(); Counters = [ordered]@{
    slotBusy = 0; missionLlmFail = 0; wrapperStarted = 0; wrapperExited = 0; wrapperUnresponsive = 0
    supGatewayRestarts = 0; supLlamaRestarts = 0; trainRounds = 0; mmluRuns = 0
  }
  Exposure = @{}; PrevGwCpu = @{}; PrevGwPid = 0
  UpCount = 0; DownCount = 0; MissStreak = 0; MaxConsecutiveMiss = 0
  PidChanges = 0
  LastZombies = @()
  LastSnapshot = $null; LastAuthoritative = $null; NextAuthoritativeAt = (Get-Date)
  LastAuthMissionLookup = $null; LastAuthMissionResult = $null
  ExpertMtime = $null; ExpertChanges = 0; GoalsRunStart = $null; GoalsRunEnd = $null; GoalsRunDelta = 0
  MmluBaselineLines = 0
  Build = [ordered]@{}
  Lifecycle = [ordered]@{
    mode = $LifecycleMode; driver = $null; baseUrl = ''; missionId = $null; stage = 'pending'
    assignedAt = $null; reportedAt = $null; completedAt = $null; elapsedSec = 0
    runningSeen = $false; workSeen = $false; deliverableChars = 0; deliverableExcerpt = ''
    pressureAtWork = $null; autoCompleted = $false; finalState = $null; reportMs = 0; assignMs = 0
    fallbackReason = ''; error = ''; steps = [System.Collections.Generic.List[object]]::new()
  }
  Sidecar = [ordered]@{ pid = 0; dir = ''; exe = ''; exeName = ''; port = 0; url = ''; outLog = ''; errLog = ''; healthy = $false; startedAt = $null; stoppedAt = $null; rssMBAtStart = $null }
  Cleanup = [ordered]@{ sidecarStopped = $false; sidecarDirRemoved = $false; cleanupAt = $null }
  LiveUntouched = 'not checked'
  LiveUrl = ''; LivePort = 0; LivePidStart = 0; PidAtEnd = 0; PrevSampleAt = $null
  SamplingStart = $null
  SummaryWritten = $false
  Command = ''
}
$script:G = $G
$G.Command = 'pwsh -NoProfile -File tools\stability_run.ps1 ' + (
  ($PSBoundParameters.GetEnumerator() | ForEach-Object {
    if ($_.Value -is [switch]) { if ($_.Value) { "-$($_.Key)" } else { '' } }
    elseif ($_.Value -is [string] -and $_.Value -eq '') { '' }
    else { "-$($_.Key) $($_.Value)" }
  } | Where-Object { $_ }) -join ' '
)

try {
  Log "stability_run.ps1 start runId=$runId bundle=$bundle"
  Log "command: $($G.Command)"
  $G.Status = 'running'

  # ---- build manifest ---------------------------------------------------
  $exe = Join-Path $G.Repo 'phoenix_main.exe'
  if (Test-Path $exe) {
    $ei = Get-Item $exe
    $sha = (Get-FileHash -Path $exe -Algorithm SHA256).Hash
    $verLine = ''
    $ver = Join-Path $G.Repo 'version.hpp'
    if (Test-Path $ver) { $verLine = ((Select-String -Path $ver -Pattern 'PHOENIX_VERSION_STRING\[\]' | Select-Object -First 1).Line -replace '.*=\s*', '' -replace ';.*', '').Trim() }
    $G.Build = [ordered]@{
      phoenixVersion = $verLine
      versionHppMtime = if (Test-Path $ver) { (Get-Item $ver).LastWriteTime.ToString('o') } else { $null }
      phoenixExe = [ordered]@{ path = $exe; sizeBytes = $ei.Length; mtimeLocal = $ei.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'); mtime = $ei.LastWriteTime.ToString('o'); sha256 = $sha }
      map = if (Test-Path (Join-Path $G.Repo 'phoenix_main.map')) {
        $mi = Get-Item (Join-Path $G.Repo 'phoenix_main.map'); [ordered]@{ path = 'phoenix_main.map'; sizeBytes = $mi.Length; mtime = $mi.LastWriteTime.ToString('o') }
      } else { $null }
      os = [System.Environment]::OSVersion.VersionString
      psVersion = $PSVersionTable.PSVersion.ToString()
    }
    $llamaProcs = @(Get-Process llama-server -ErrorAction SilentlyContinue)
    $G.Build.llamaServers = @($llamaProcs | ForEach-Object { [ordered]@{ pid = $_.Id; path = $_.Path } })
    Log "build: $($G.Build.phoenixVersion) sha256=$($sha.Substring(0,16))..."
  } else { $G.Errors += 'phoenix_main.exe not found'; Log 'phoenix_main.exe not found' 'WARN' }

  # ---- locate live gateway ---------------------------------------------
  if ($GatewayUrl) { $G.LiveUrl = $GatewayUrl.TrimEnd('/') } else {
    $portFile = Join-Path $G.Repo 'runtime_store\long_run\port.txt'
    if (Test-Path $portFile) {
      $p = (Get-Content $portFile -Raw).Trim()
      if ($p -match '^\d+$') { $G.LiveUrl = "http://127.0.0.1:$p" }
    }
    if (-not $G.LiveUrl) {
      foreach ($cand in 5097, 5096, 5098, 5080) {
        $hh = Invoke-Get "http://127.0.0.1:$cand/api/health" 3
        if ($hh.ok) { $G.LiveUrl = "http://127.0.0.1:$cand"; break }
      }
    }
  }
  if (-not $G.LiveUrl) { throw 'cannot locate a live gateway (no port.txt, no health on 5097/5096/5098/5080); use -GatewayUrl' }
  $G.LivePort = ([uri]$G.LiveUrl).Port
  $h0 = Invoke-Get "$($G.LiveUrl)/api/health" $HealthTimeoutSec
  if (-not $h0.ok) { throw "live gateway $($G.LiveUrl) is not healthy before the run: $($h0.error)" }
  $ident0 = Get-GatewayIdentity $G.LivePort
  $G.LivePidStart = $ident0.pid
  Log "live gateway: $($G.LiveUrl) pid=$($G.LivePidStart) rss=$($ident0.rssMB)MB zombies=$(@($ident0.zombies).Count) listenerZombie=$($ident0.listenerZombie)"

  # ---- baseline snapshots ----------------------------------------------
  $snap0 = Read-AutonomySnapshot
  $G.GoalsRunStart = if ($snap0.ok) { $snap0.goalsRun } else { $null }
  $G.GoalsRunEnd = $G.GoalsRunStart
  $w = Join-Path $G.Repo 'runtime_store\experts\study_tiny\weights.json'
  if (Test-Path $w) { $G.ExpertMtime = (Get-Item $w).LastWriteTime.ToString('o') }
  $mmluPath = Join-Path $G.Repo 'runtime_store\long_run\mmlu_history.jsonl'
  if (Test-Path $mmluPath) { $G.MmluBaselineLines = @(Get-Content $mmluPath -ErrorAction SilentlyContinue).Count }
  $runDir = Join-Path $G.Repo 'runtime_store\long_run'
  foreach ($f in 'gateway.log', 'gateway.err.log', 'gateway_loop.log', 'supervisor.log', 'training.log') {
    $fp = Join-Path $runDir $f
    if (Test-Path $fp) { $G.LogOffsets[$f] = (Get-Item $fp).Length } else { $G.LogOffsets[$f] = 0 }
  }
  # crash channel (b): the gateway's SEH-swallowed faults are only visible here
  $fatalLog = Join-Path $G.Repo 'runtime_store\logs\fatal_runtime.log'
  $G.LogOffsets['fatal_runtime.log'] = if (Test-Path $fatalLog) { (Get-Item $fatalLog).Length } else { 0 }
  $G.FatalOffsetBaseline = $G.LogOffsets['fatal_runtime.log']
  Log "baseline log offsets: gateway.log=$($G.LogOffsets['gateway.log']) fatal_runtime.log=$($G.LogOffsets['fatal_runtime.log'])"
  $ac0 = Get-AuthoritativeMissionCounts $G.LiveUrl
  if ($ac0.ok) {
    $G.LastAuthoritative = $ac0
    $G.NextAuthoritativeAt = (Get-Date).AddSeconds($MissionStatusEverySec)
    Log "baseline missions: running=$($ac0.running) completed=$($ac0.completed) failed=$($ac0.failed) total=$($ac0.total) ($($ac0.ms)ms)"
  } else { Log "baseline authoritative mission status failed: $($ac0.error)" 'WARN' }
  Save-CleanupRecord

  # ---- lifecycle --------------------------------------------------------
  Log "lifecycle: mode=$LifecycleMode"
  Start-Lifecycle

  # ---- sampling loop ---------------------------------------------------
  # Schedule is drift-corrected: samples are taken at t0 + (seq-1)*interval,
  # and if a lifecycle HTTP call delays a sample the next one is taken
  # immediately. The loop runs until the EXPECTED sample count is collected
  # (hard-capped at planned + grace) so the evidence density stays honest even
  # when the gateway makes one call slow.
  $G.SamplingStart = Get-Date
  $G.PrevSampleAt = $G.SamplingStart
  $expectedSamples = [int][math]::Floor($Minutes * 60 / $IntervalSec) + 1
  $hardStop = $G.SamplingStart.AddMinutes($Minutes).AddSeconds([math]::Max(120, $IntervalSec * 2))
  Log "sampling: interval=${IntervalSec}s minutes=$Minutes expected=$expectedSamples samples (hard stop $($hardStop.ToString('HH:mm:ss')))"
  $seq = 0
  while ($G.Samples.Count -lt $expectedSamples -and (Get-Date) -lt $hardStop) {
    $seq++
    $expectedAt = $G.SamplingStart.AddSeconds(($seq - 1) * $IntervalSec)
    $waitMs = [int](($expectedAt - (Get-Date)).TotalMilliseconds)
    if ($waitMs -gt 0) { Start-Sleep -Milliseconds $waitMs }
    try {
      $sample = Get-Sample $seq
      $G.Samples.Add($sample)
      Write-JsonLine $G.JsonlPath $sample
      $extra = ''
      if ($sample.notes) { $extra = " notes=$($sample.notes)" }
      Log ("sample {0,-4} up={1} http={2}ms gpu={3}/{4}% gwPid={5} rss={6}MB zombie={7} crash={8} restart={9} slot={10} life={11}{12}" -f `
          $sample.seq, $sample.up, $sample.httpMs, $sample.gpuUtilMean, $sample.gpuUtilMax, $sample.gwPid, $sample.gwRssMB, `
          $sample.zombieCount, $sample.crashesTotal, $sample.restartsTotal, $sample.slotBusyTotal, $sample.lifecycleStage, $extra)
      Step-Lifecycle $seq
    } catch {
      $G.Errors += "sample $($seq): $($_.Exception.Message)"
      Log "sample $($seq) failed: $($_.Exception.Message)" 'ERROR'
    }
    $G.PrevSampleAt = Get-Date
  }
  Log "sampling complete: $($G.Samples.Count)/$expectedSamples samples"

  # ---- finalize --------------------------------------------------------
  Finalize-Lifecycle
  # one last crash scan (events between the last sample and now), both channels
  $finalCrashes = @()
  $finalCrashes += Scan-NewCrashes
  $finalCrashes += Scan-FatalRuntime
  foreach ($c in $finalCrashes) {
    Log "CRASH DETECTED at finalize: $($c.appName) code=$($c.exceptionCode)" 'ERROR'
    Invoke-Forensics $c | Out-Null
  }
  $acEnd = Get-AuthoritativeMissionCounts $G.LiveUrl
  if ($acEnd.ok) { $G.LastAuthoritative = $acEnd; Log "final missions: running=$($acEnd.running) completed=$($acEnd.completed) failed=$($acEnd.failed) total=$($acEnd.total)" }
  $snapEnd = Read-AutonomySnapshot
  if ($snapEnd.ok) { $G.GoalsRunEnd = $snapEnd.goalsRun; if ($null -ne $snapEnd.goalsRun -and $null -ne $G.GoalsRunStart) { $G.GoalsRunDelta = [int64]$snapEnd.goalsRun - [int64]$G.GoalsRunStart } }
  $G.EndTime = Get-Date
  $hEnd = Invoke-Get "$($G.LiveUrl)/api/health" $HealthTimeoutSec
  $identEnd = Get-GatewayIdentity $G.LivePort
  $G.PidAtEnd = $identEnd.pid
  $G.LiveUntouched = if ($identEnd.pid -eq $G.LivePidStart) { 'yes (same live gateway pid before/after)' } else { "pid changed $($G.LivePidStart) -> $($identEnd.pid) (recorded as restart; harness did not stop it)" }
  Log "end: live gateway health=$($hEnd.ok) pid=$($G.PidAtEnd) untouched=$($G.LiveUntouched)"
  $G.Status = 'completed'
} catch {
  $G.Status = 'error'
  $G.Errors += $_.Exception.Message
  Log "FATAL: $($_.Exception.Message)" 'ERROR'
  Log ($_.ScriptStackTrace) 'ERROR'
} finally {
  if (-not $G.EndTime) { $G.EndTime = Get-Date }
  try { Stop-Sidecar 'finally' } catch { }
  try {
    # final zombie / sidecar leftovers check
    $left = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.ExecutablePath -and $G.Sidecar.dir -and $_.ExecutablePath -like ($G.Sidecar.dir + '\*') })
    if ($left.Count -gt 0) { $G.Errors += "leftover sidecar processes: $($left.Count)"; Log "leftover sidecar processes: $($left.Count)" 'ERROR' }
    if ($G.Samples.Count -gt 0) {
      try {
        $flat = @($G.Samples | ForEach-Object {
          [pscustomobject]@{
            seq = $_.seq; ts = $_.ts; elapsedSec = $_.elapsedSec; up = $_.up; httpMs = $_.httpMs
            gpuUtilMean = $_.gpuUtilMean; gpuUtilMax = $_.gpuUtilMax; gpuMemUsedMB = $_.gpuMemUsedMB; gpuTempC = $_.gpuTempC
            gwPid = $_.gwPid; gwRssMB = $_.gwRssMB; gwThreads = $_.gwThreads; gwCpuPct = $_.gwCpuPct; gwTick = $_.gwTick
            zombieCount = $_.zombieCount; llamaProcCount = $_.llamaProcCount; llamaRssMB = $_.llamaRssMB
            crashesTotal = $_.crashesTotal; werCrashesTotal = $_.werCrashesTotal; fatalRuntimeCrashesTotal = $_.fatalRuntimeCrashesTotal; restartsTotal = $_.restartsTotal; pidChangesTotal = $_.pidChangesTotal
            slotBusyTotal = $_.slotBusyTotal; slotBusyDelta = $_.slotBusyDelta; logRotations = $_.logRotations
            missionRunning = $_.missionRunning; missionFinishedRecent = $_.missionFinishedRecent
            missionCompletedExact = $_.missionCompletedExact; missionCountsSource = $_.missionCountsSource
            expertAgeMin = $_.expertAgeMin; expertChanged = $_.expertChanged; goalsRun = $_.goalsRun; goalsRunDelta = $_.goalsRunDelta
            sidecarActive = $_.sidecarActive; sidecarUp = $_.sidecarUp; lifecycleStage = $_.lifecycleStage; notes = $_.notes
          }
        })
        ($flat | ConvertTo-Csv -NoTypeInformation) | Set-Content -Path (Join-Path $G.Bundle 'samples.csv') -Encoding utf8
      } catch { Log "samples.csv failed: $($_.Exception.Message)" 'WARN' }
    }
    Write-Summary
    Save-CleanupRecord
    $verdictLine = ''
    if (Test-Path (Join-Path $G.Bundle 'summary.json')) {
      try { $sj = Get-Content (Join-Path $G.Bundle 'summary.json') -Raw | ConvertFrom-Json; $verdictLine = "verdict=$($sj.verdict.overall) exposure=$($sj.verdict.exposureHours)h lambdaUpper=$($sj.verdict.lambdaUpper95PerHour)/h rung=$($sj.verdict.sRung)" } catch { }
    }
    Log "DONE status=$($G.Status) $verdictLine bundle=$($G.Bundle)"
    Write-Host ''
    Write-Host "evidence bundle: $($G.Bundle)" -ForegroundColor Green
    Write-Host "  summary.md    : $(Join-Path $G.Bundle 'summary.md')" -ForegroundColor Green
    Write-Host "  summary.json  : $(Join-Path $G.Bundle 'summary.json')" -ForegroundColor Green
    Write-Host "  $verdictLine" -ForegroundColor Green
  } catch {
    Write-Host "summary finalize failed: $($_.Exception.Message)" -ForegroundColor Red
  }
}
