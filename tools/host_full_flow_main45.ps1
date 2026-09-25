# 45-min product main flow: Chat (/api/chat) + two concurrent Missions.
# Restarts phoenix_main only. Keeps existing llama-server. Does not curl /health.
# Does not overwrite host_full_flow_questionaire.json or frozen GNN snaps.
param(
  [double]$Minutes = 45,
  [int]$MaxTokens = 128,
  [int]$ChatTimeoutSec = 180
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
. (Join-Path $PSScriptRoot 'experiment_ha.ps1')

$Phoenix = Split-Path -Parent $PSScriptRoot
Set-Location $Phoenix
$QuestionFile = Join-Path $Phoenix 'questionaire.txt'
$GoalFile = Join-Path $Phoenix 'test\intelligence\questionPrompt.txt'
$Exe = Join-Path $Phoenix 'phoenix_main.exe'
$Inc = Join-Path $Phoenix 'main_hub_parts\065_class_runtimestate.inc'
$Store = Join-Path $Phoenix 'runtime_store'
$Journal = Join-Path $Store 'host_full_flow_main45.jsonl'
$Result = Join-Path $Store 'host_full_flow_main45.json'
$GwLog = Join-Path $Store 'host_full_flow_main45_gateway.out.log'
$GwErr = Join-Path $Store 'host_full_flow_main45_gateway.err.log'
$FeLog = Join-Path $Store 'host_full_flow_main45_frontend.out.log'
$FeErr = Join-Path $Store 'host_full_flow_main45_frontend.err.log'
$Model = Join-Path $Phoenix 'GGUF_models\blobs\sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29'
$ChatApi = 'http://127.0.0.1:5080/api/chat'
$AssignApi = 'http://127.0.0.1:5080/api/mission/assign'
$StatusApi = 'http://127.0.0.1:5080/api/mission/status'
$FileApi = 'http://127.0.0.1:5080/api/mission/file'
$Token = 'local-dev'
$MidHelios = 'main45-helios'
$MidOps = 'main45-ops'

if (-not (Test-Path -LiteralPath $QuestionFile)) { throw "missing $QuestionFile" }
if (-not (Test-Path -LiteralPath $GoalFile)) { throw "missing $GoalFile" }
if (-not (Test-Path -LiteralPath $Exe)) { throw "missing $Exe" }
if (-not (Get-Process -Name llama-server -ErrorAction SilentlyContinue)) {
  throw 'llama-server is not running; do not start a second one from this script'
}

$exeTime = (Get-Item -LiteralPath $Exe).LastWriteTimeUtc
$incTime = (Get-Item -LiteralPath $Inc).LastWriteTimeUtc
if ($exeTime -lt $incTime) {
  throw ("phoenix_main.exe is older than 065 ({0:o} < {1:o}); rebuild first" -f $exeTime, $incTime)
}

$questions = Get-Content -LiteralPath $QuestionFile -Encoding UTF8 |
  ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' }
if ($questions.Count -lt 1) { throw 'empty questionnaire' }
$heliosGoal = [IO.File]::ReadAllText($GoalFile, [Text.Encoding]::UTF8).Trim()
if ([string]::IsNullOrWhiteSpace($heliosGoal)) { throw 'empty Helios goal' }
$opsGoal = @'
Write an 800-word Mars weather-station daily operations checklist in Markdown.
Cover power, comms delay, dust, and a fault-recovery loop. No fiction. No exam.
'@

function Invoke-JsonPost([string]$url, $payload, [int]$sec) {
  $body = $payload | ConvertTo-Json -Compress -Depth 8
  $bytes = [Text.Encoding]::UTF8.GetBytes($body)
  $req = [Net.HttpWebRequest]::Create($url)
  $req.Method = 'POST'
  $req.ContentType = 'application/json; charset=utf-8'
  $req.Headers.Add('Authorization', "Bearer $Token")
  $req.KeepAlive = $false
  $req.Timeout = $sec * 1000
  $req.ReadWriteTimeout = $sec * 1000
  $req.ContentLength = $bytes.Length
  $s = $req.GetRequestStream()
  $s.Write($bytes, 0, $bytes.Length)
  $s.Close()
  $resp = $req.GetResponse()
  $reader = New-Object IO.StreamReader($resp.GetResponseStream(), [Text.Encoding]::UTF8)
  try { $raw = $reader.ReadToEnd() } finally { $reader.Close(); $resp.Close() }
  return ($raw | ConvertFrom-Json)
}

function Invoke-JsonGet([string]$url, [int]$sec) {
  $req = [Net.HttpWebRequest]::Create($url)
  $req.Method = 'GET'
  $req.Headers.Add('Authorization', "Bearer $Token")
  $req.KeepAlive = $false
  $req.Timeout = $sec * 1000
  $resp = $req.GetResponse()
  $reader = New-Object IO.StreamReader($resp.GetResponseStream(), [Text.Encoding]::UTF8)
  try { $raw = $reader.ReadToEnd() } finally { $reader.Close(); $resp.Close() }
  return ($raw | ConvertFrom-Json)
}

function Get-Reply($obj) {
  if ($obj.result -and $obj.result.reply) { return [string]$obj.result.reply }
  if ($obj.reply) { return [string]$obj.reply }
  if ($obj.error) { return ("[ERROR] " + $obj.error) }
  return ''
}

function Get-ResultProp($obj, [string]$name) {
  if ($obj.result -and ($obj.result.PSObject.Properties.Name -contains $name)) {
    return $obj.result.$name
  }
  if ($obj.PSObject.Properties.Name -contains $name) { return $obj.$name }
  return $null
}

function Wait-Port([string]$url, [int]$sec) {
  $deadline = [DateTime]::UtcNow.AddSeconds($sec)
  while ([DateTime]::UtcNow -lt $deadline) {
    try {
      $r = Invoke-WebRequest -Uri $url -TimeoutSec 3 -UseBasicParsing
      if ($r.StatusCode -ge 200) { return $true }
    } catch {
      if ($_.Exception.Response) { return $true }
    }
    Start-Sleep -Seconds 1
  }
  return $false
}

function Read-MissionFile([string]$mid, [string]$path) {
  try {
    $r = Invoke-JsonPost $FileApi @{ action = 'read'; path = $path; missionId = $mid } 30
    if ($r.content) { return [string]$r.content }
    if ($r.result -and $r.result.content) { return [string]$r.result.content }
  } catch {}
  return ''
}

function Get-MissionSnap {
  $st = Invoke-JsonGet $StatusApi 30
  $arr = @()
  if ($st.result -and $st.result.missions) { $arr = @($st.result.missions) }
  $map = @{}
  foreach ($m in $arr) {
    $id = [string]$m.id
    $stats = $m.stats
    $msn = $null
    if ($stats -and $stats.mission) { $msn = $stats.mission }
    $deliv = ''
    if ($msn -and $msn.deliverable) { $deliv = [string]$msn.deliverable }
    $fileBody = Read-MissionFile $id 'deliverable.md'
    if ($fileBody.Length -gt $deliv.Length) { $deliv = $fileBody }
    $map[$id] = [ordered]@{
      id = $id
      state = $(if ($msn) { $msn.state } else { $null })
      goalChars = $(if ($msn -and $msn.goal) { ([string]$msn.goal).Length } else { 0 })
      delivChars = $deliv.Length
      pressure = $(if ($stats) { $stats.pressure } else { $null })
    }
  }
  return @{ raw = $st; map = $map; defaultId = $(if ($st.result) { [string]$st.result.defaultMissionId } else { '' }) }
}

Write-Host ("==== main45: chat + 2 missions, {0} min, keep llama ====" -f $Minutes)
Get-Process -Name phoenix_main -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2
if (Get-Process -Name llama-server -ErrorAction SilentlyContinue) {
  Write-Host 'llama-server still up'
} else {
  throw 'llama-server disappeared; stop'
}

foreach ($f in @($GwLog, $GwErr, $FeLog, $FeErr, $Journal)) {
  if (Test-Path -LiteralPath $f) { Remove-Item -LiteralPath $f -Force }
}

$gwArgs = @(
  '--disable-watchdog=1',
  '--gateway-host=127.0.0.1',
  '--port=5080',
  '--frontend-enabled=false',
  '--transformer-mode=llamacpp',
  '--llamacpp-base-url=http://127.0.0.1:8082',
  '--llamacpp-model=' + $Model,
  '--disable-gnn-module=false',
  '--ai-count=1',
  '--concept-matrix-enabled=false',
  '--base-dir=runtime_store',
  '--log-mode=info',
  '--pid-file=runtime_store/phoenix_gateway_main45.pid'
)
$feArgs = @(
  '--frontend-only=1',
  '--frontend-host=127.0.0.1',
  '--frontend-port=5081',
  '--base-dir=runtime_store'
)
$gw = Start-Process -FilePath $Exe -ArgumentList $gwArgs -RedirectStandardOutput $GwLog -RedirectStandardError $GwErr -PassThru -WindowStyle Hidden
$fe = Start-Process -FilePath $Exe -ArgumentList $feArgs -RedirectStandardOutput $FeLog -RedirectStandardError $FeErr -PassThru -WindowStyle Hidden
Write-Host ("gateway pid={0} frontend pid={1}" -f $gw.Id, $fe.Id)
if (-not (Wait-Port 'http://127.0.0.1:5080/' 90)) { throw 'gateway :5080 not ready' }
Write-Host 'gateway ready'

$seeds = @(
  'Photosynthesis transfers carbon when leaf chloroplasts bind carbon dioxide into sugar during the Calvin cycle. Light reactions supply ATP and NADPH so carbon fixation can proceed.',
  'Rainbows form when sunlight enters raindrops, refracts, reflects internally, then exits at a different angle so the spectrum spreads across the sky.',
  'A trolley approaching three trapped people can be diverted onto a side track that kills one researcher. Consequentialist calculus weighs immediate deaths against later lives that research might save.',
  'Consciousness is the felt inner viewpoint of a process. Substrate independence claims the same pattern of causal relations would keep that viewpoint if the process ran on another medium.'
)
$ingestUrl = 'http://127.0.0.1:5080/api/corpus/ingest'
$onlineUrl = 'http://127.0.0.1:5080/api/corpus/online'
$seedOk = 0
$seedNodes = 0
foreach ($s in $seeds) {
  $ing = Invoke-JsonPost $ingestUrl @{ text = $s; source = 'main45-seed' } 60
  if (-not $ing.ok) { throw ("seed ingest failed: " + ($ing | ConvertTo-Json -Compress)) }
  $seedOk++
  if ($ing.results) {
    foreach ($r in @($ing.results)) {
      if ($null -ne $r.memes -and ($r.memes -is [int] -or $r.memes -is [long] -or $r.memes -is [double])) {
        $seedNodes += [int]$r.memes
      } elseif ($r.memes) {
        $seedNodes += @($r.memes).Count
      }
    }
  }
}
Write-Host ("seed ingest docs={0} memeHints={1}" -f $seedOk, $seedNodes)
$probe = Invoke-JsonPost $onlineUrl @{ query = 'how does photosynthesis transfer carbon' } 60
$probeN = 0
if ($probe.result -and $probe.result.suggestions) { $probeN = @($probe.result.suggestions).Count }
Write-Host ("online probe suggestions={0} ok={1}" -f $probeN, [bool]$probe.ok)
if (-not $probe.ok -or $probeN -lt 1) {
  Write-Host 'GNN online probe produced no memes after seed; stop dirty main45'
  exit 4
}

$a1 = Invoke-JsonPost $AssignApi @{
  id = $MidHelios
  goal = $heliosGoal
  deadlineSec = 3600
  ctxSize = 4096
  contextPack = 'full_and_summary'
  includeGnnSummary = $true
  pressureMode = 'asymptotic'
  enabled = $true
} 120
$a2 = Invoke-JsonPost $AssignApi @{
  id = $MidOps
  goal = $opsGoal
  deadlineSec = 3600
  ctxSize = 4096
  contextPack = 'full_and_summary'
  includeGnnSummary = $true
  pressureMode = 'asymptotic'
  enabled = $true
} 120
Write-Host ("assign helios ok={0} id={1}" -f [bool]$a1.ok, $MidHelios)
Write-Host ("assign ops ok={0} id={1}" -f [bool]$a2.ok, $MidOps)
if (-not $a1.ok -or -not $a2.ok) { throw 'mission assign failed; stop dirty main45' }

$snap0 = Get-MissionSnap
Write-Host ("missions after assign count={0} default={1}" -f $snap0.map.Count, $snap0.defaultId)
if (-not $snap0.map.ContainsKey($MidHelios) -or -not $snap0.map.ContainsKey($MidOps)) {
  throw 'both mission ids not present after assign'
}

$doc = [ordered]@{
  startedAtUtc = [DateTime]::UtcNow.ToString('o')
  method = '45min chat + two concurrent missions (Helios + ops) on post-065 phoenix_main'
  exe = $Exe
  exeTime = $exeTime.ToString('o')
  minutes = $Minutes
  seedOk = $seedOk
  seedNodes = $seedNodes
  probeSuggestions = $probeN
  missionIds = @($MidHelios, $MidOps)
  rows = @()
  missionPolls = @()
}
$ok = 0
$empty = 0
$gnnHits = 0
$gnnDisabled = 0
$started = [DateTime]::UtcNow
$deadline = $started.AddMinutes($Minutes)
$n = 0
$emptyStreak = 0
$qIdx = 0

do {
  $n++
  $q = [string]$questions[$qIdx]
  $qIdx = ($qIdx + 1) % $questions.Count
  $sid = 'main45-{0}-{1}' -f $n, ([Guid]::NewGuid().ToString('N').Substring(0, 6))
  Write-Host ("==== Q{0} chars={1} remainMin={2:N1} ====" -f $n, $q.Length, ($deadline - [DateTime]::UtcNow).TotalMinutes)
  $row = [ordered]@{
    n = $n
    sessionId = $sid
    qChars = $q.Length
    qHead = $(if ($q.Length -gt 80) { $q.Substring(0, 80) } else { $q })
  }
  try {
    $obj = Invoke-JsonPost $ChatApi @{
      text = $q
      sessionId = $sid
      enableGraphSelector = $true
      maxTokens = $MaxTokens
      token = $Token
    } $ChatTimeoutSec
    $reply = Get-Reply $obj
    $row.status = 'ok'
    $row.replyChars = $reply.Length
    $row.replyHead = $(if ($reply.Length -gt 120) { $reply.Substring(0, 120) } else { $reply })
    $prov = Get-ResultProp $obj 'provider'
    $row.provider = $(if ($prov -and $prov.id) { [string]$prov.id } else { $null })
    $row.gnnAlign = Get-ResultProp $obj 'gnnAlign'
    $row.gnnStage2 = Get-ResultProp $obj 'gnnStage2'
    $row.gnnModuleDisabled = Get-ResultProp $obj 'gnnModuleDisabled'
    $tr = Get-ResultProp $obj 'gnnTrace'
    if ($tr) {
      $row.gnnTraceId = $(if ($tr.id) { [string]$tr.id } else { $null })
      $src = @()
      if ($tr.seeds) {
        foreach ($s in @($tr.seeds) | Select-Object -First 4) {
          $src += ('{0}<-{1}' -f $s.meme, $s.from)
        }
      }
      $row.gnnTraceSrc = ($src -join ';')
    }
    $memes = Get-ResultProp $obj 'memes'
    $row.memeCount = $(if ($memes) { @($memes).Count } else { 0 })
    $gr = Get-ResultProp $obj 'graphReply'
    $row.graphReplyChars = $(if ($gr) { ([string]$gr).Length } else { 0 })
    $row.hasGraph = ($row.memeCount -gt 0 -or $row.graphReplyChars -gt 0 -or ($row.gnnStage2 -and [string]$row.gnnStage2))
    if ($row.gnnModuleDisabled -eq $true) { $gnnDisabled++ }
    if ($row.hasGraph) { $gnnHits++ }
    if ([string]::IsNullOrWhiteSpace($reply) -or $reply -match 'disconnected') {
      $empty++
      $emptyStreak++
      $row.empty = $true
      Write-Host ("  EMPTY reply streak={0} totalEmpty={1}" -f $emptyStreak, $empty)
    } else {
      $ok++
      $emptyStreak = 0
      $row.empty = $false
      Write-Host ("  replyChars={0} hasGraph={1} id={2}" -f $reply.Length, $row.hasGraph, $row.gnnTraceId)
    }
  } catch {
    $empty++
    $emptyStreak++
    $row.status = 'error'
    $row.empty = $true
    $row.error = [string]$_.Exception.Message
    Write-Host ("  ERROR {0}" -f $row.error)
  }
  $doc.rows += $row
  Add-Jsonl $Journal $row

  try {
    $ms = Get-MissionSnap
    $poll = [ordered]@{
      atUtc = [DateTime]::UtcNow.ToString('o')
      afterQ = $n
      defaultId = $ms.defaultId
      count = $ms.map.Count
      helios = $ms.map[$MidHelios]
      ops = $ms.map[$MidOps]
    }
    $doc.missionPolls += $poll
    Write-Host ("  missions n={0} helios={1}c state={2} ops={3}c state={4}" -f `
      $ms.map.Count, `
      $(if ($ms.map[$MidHelios]) { $ms.map[$MidHelios].delivChars } else { -1 }), `
      $(if ($ms.map[$MidHelios]) { $ms.map[$MidHelios].state } else { '?' }), `
      $(if ($ms.map[$MidOps]) { $ms.map[$MidOps].delivChars } else { -1 }), `
      $(if ($ms.map[$MidOps]) { $ms.map[$MidOps].state } else { '?' }))
  } catch {
    Write-Host ("  mission poll ERROR {0}" -f $_.Exception.Message)
  }

  $doc.ok = $ok
  $doc.empty = $empty
  $doc.gnnHits = $gnnHits
  $doc.gnnDisabled = $gnnDisabled
  $doc.asked = $n
  $doc.elapsedMin = ([DateTime]::UtcNow - $started).TotalMinutes
  Save-JsonAtomic $Result $doc
  if ($gnnDisabled -gt 0) {
    Write-Host 'GNN disabled mid-run; stop dirty main45'
    exit 5
  }
} while ([DateTime]::UtcNow -lt $deadline)

$final = Get-MissionSnap
$doc.finishedAtUtc = [DateTime]::UtcNow.ToString('o')
$doc.ok = $ok
$doc.empty = $empty
$doc.gnnHits = $gnnHits
$doc.gnnDisabled = $gnnDisabled
$doc.asked = $n
$doc.elapsedMin = ([DateTime]::UtcNow - $started).TotalMinutes
$doc.finalMissions = @{
  count = $final.map.Count
  defaultId = $final.defaultId
  helios = $final.map[$MidHelios]
  ops = $final.map[$MidOps]
}
Save-JsonAtomic $Result $doc
Write-Host ("Done chat ok={0} empty={1} gnnHits={2} gnnDisabled={3} asked={4} elapsedMin={5:N1}" -f $ok, $empty, $gnnHits, $gnnDisabled, $n, $doc.elapsedMin)
Write-Host ("Done missions count={0} helios={1} ops={2} {3}" -f $final.map.Count, $doc.finalMissions.helios.delivChars, $doc.finalMissions.ops.delivChars, $Result)
if ($gnnDisabled -gt 0) { exit 5 }
if ($gnnHits -lt 1) { exit 6 }
if ($final.map.Count -lt 2) { exit 7 }
$need = [Math]::Ceiling([Math]::Max(1, $n) * 0.5)
if ($ok -lt $need) { exit 2 }
exit 0
