# Tensor/RoPE ordered-carrier serial. Prompt is the carrier text only:
# raw /completion, no chat template, no system/task/persona wrapper.
# Detect = GNN 1/df + graph-prior lift. Does not overwrite prior serial JSON.
# PowerShell only. Does not call llama /health.
param(
  [string]$OutName = 'gnn_sig_serial_tensor3',
  [string]$SnapFile = '',
  [string]$ScreenFile = '',
  [int]$Rounds = 6,
  [int]$JudgeAfter = 3
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
. (Join-Path $PSScriptRoot 'experiment_ha.ps1')

$Phoenix = Split-Path -Parent $PSScriptRoot
$Exe = Join-Path $Phoenix 'build\meme_barrier_instrument.exe'
if (-not $ScreenFile) { $ScreenFile = Join-Path $Phoenix 'runtime_store\gnn_sig_screen_merge.json' }
if (-not $SnapFile) { $SnapFile = Join-Path $Phoenix 'runtime_store\gnn_instrument_merge.txt' }
$Corpus = Join-Path $Phoenix 'robots\wikitext-103-all.txt'
if (-not (Test-Path -LiteralPath $Corpus)) {
  $Corpus = Join-Path $Phoenix 'robots\wikitext-101-all.txt'
}
$ResultFile = Join-Path $Phoenix ("runtime_store\{0}.json" -f $OutName)
$Journal = Join-Path $Phoenix ("runtime_store\{0}.jsonl" -f $OutName)
$Api = 'http://127.0.0.1:8082/completion'
$TimeoutMs = 300000
$Units = 256

if (-not (Test-Path -LiteralPath $Exe)) { throw "missing $Exe" }
if (-not (Test-Path -LiteralPath $SnapFile)) { throw "missing frozen snap $SnapFile" }
if (-not (Test-Path -LiteralPath $ScreenFile)) { throw "missing screen $ScreenFile" }
if (-not (Test-Path -LiteralPath $Corpus)) { throw "missing corpus" }

function Invoke-LlamaChat([string]$userText) {
  if ([string]::IsNullOrWhiteSpace($userText)) {
    throw 'empty carrier (do not continue dirty serial)'
  }
  $payload = @{
    prompt = $userText
    n_predict = 96
    stream = $false
    cache_prompt = $false
    temperature = 0.2
    top_p = 0.9
  }
  $body = $payload | ConvertTo-Json -Compress -Depth 6
  $bytes = [Text.Encoding]::UTF8.GetBytes($body)
  $req = [Net.HttpWebRequest]::Create($Api)
  $req.Method = 'POST'
  $req.ContentType = 'application/json; charset=utf-8'
  $req.KeepAlive = $false
  $req.Timeout = $TimeoutMs
  $req.ReadWriteTimeout = $TimeoutMs
  $req.ContentLength = $bytes.Length
  $s = $req.GetRequestStream()
  $s.Write($bytes, 0, $bytes.Length)
  $s.Close()
  $resp = $req.GetResponse()
  $reader = New-Object IO.StreamReader($resp.GetResponseStream(), [Text.Encoding]::UTF8)
  try {
    $raw = $reader.ReadToEnd()
  } finally {
    $reader.Close()
    $resp.Close()
  }
  if ([string]::IsNullOrWhiteSpace($raw)) {
    throw 'empty llama http body; stop dirty serial'
  }
  $obj = $raw | ConvertFrom-Json
  $text = ''
  if ($obj.content) { $text = [string]$obj.content }
  if ($obj.choices -and $obj.choices[0].message.content) {
    $text = [string]$obj.choices[0].message.content
  }
  return $text
}

function Get-Detect([string]$memeId, [string]$text) {
  $out = $text | & $Exe detect --snap $SnapFile --meme $memeId
  if ($LASTEXITCODE -ne 0) { throw ("detect exit " + $LASTEXITCODE + ": " + $out) }
  $m = [regex]::Match([string]$out, 'present=(\d) activation=([0-9eE.+-]+) peak=([0-9eE.+-]+) rank=(-?\d+) lift=([0-9eE.+-]+) null=([0-9eE.+-]+)')
  if (-not $m.Success) { throw ("detect parse failed: " + $out) }
  return [ordered]@{
    present = ($m.Groups[1].Value -eq '1')
    activation = [double]$m.Groups[2].Value
    peak = [double]$m.Groups[3].Value
    rank = [int]$m.Groups[4].Value
    lift = [double]$m.Groups[5].Value
    nullScore = [double]$m.Groups[6].Value
  }
}

function Get-Compose([string]$memeId) {
  $raw = & $Exe compose --snap $SnapFile --corpus $Corpus --units $Units --meme $memeId --require-exclusive
  if ($LASTEXITCODE -ne 0) { throw ("compose exit " + $LASTEXITCODE + ": " + $raw) }
  return ($raw | ConvertFrom-Json)
}

function Test-Echo([string]$prompt, [string]$output) {
  return Test-SerialEcho $prompt $output
}

$screen = Get-Content -LiteralPath $ScreenFile -Raw -Encoding UTF8 | ConvertFrom-Json
$memes = @($screen.memes)
if ($memes.Count -lt 1) { throw 'empty screen' }

Write-Host '==== tensor-sentence compose + empty chat serial on frozen instrument ===='
Write-Host ("snap={0} memes={1}" -f $SnapFile, $memes.Count)
$done = Read-JsonlCompleted $Journal

$doc = [ordered]@{
  startedAtUtc = [DateTime]::UtcNow.ToString('o')
  method = 'readme 统计+识别: tensor/RoPE ordered carrier; raw /completion of carrier only (no chat template); detect = GNN 1/df + graph-prior lift'
  note = 'not word-bag /completion; do not ingest output; do not overwrite gnn_sig_serial.json, gnn_sig_serial_lift.json, or gnn_sig_serial_rag*.json'
  snap = $SnapFile
  screen = $ScreenFile
  corpus = $Corpus
  units = $Units
  rounds = $Rounds
  groups = @()
}

foreach ($m in $memes) {
  Write-Host ("==== {0} {1} sig={2} ====" -f $m.pole, $m.id, $m.significance)
  $composed = Get-Compose $m.id
  $seedHit = Get-Detect $m.id ([string]$composed.text)
  Write-Host ("  compose present={0} lift={1:N2} rank={2} units={3} chars={4}" -f $seedHit.present, $seedHit.lift, $seedHit.rank, $composed.units, ([string]$composed.text).Length)
  if (-not $seedHit.present) {
    Write-Host ("  skip {0}: composed carrier does not present" -f $m.id)
    $doc.groups += [ordered]@{
      id = $m.id
      pole = $m.pole
      significance = $m.significance
      skipped = 'compose-not-present'
      compose = $composed
    }
    Save-JsonAtomic $ResultFile $doc
    continue
  }
  $row = [ordered]@{
    id = $m.id
    pole = $m.pole
    rankMost = $m.rankMost
    rankLeast = $m.rankLeast
    significance = $m.significance
    wordCount = $m.wordCount
    seedBag = $m.carrier
    seedRag = [string]$composed.text
    compose = $composed
    rounds = @()
    presentRounds = 0
    echoRounds = 0
  }
  $current = [string]$composed.text
  for ($r = 1; $r -le $Rounds; $r++) {
    $key = '{0}|{1}|{2}' -f $m.pole, $m.id, $r
    if ($done.ContainsKey($key)) {
      $prev = $done[$key]
      $row.rounds += $prev
      if ($prev.detect.present) { $row.presentRounds++ }
      if ($prev.echo) { $row.echoRounds++ }
      $current = [string]$prev.output
      Write-Host ("  resume round {0} present={1}" -f $r, $prev.detect.present)
      if ($prev.judge -and -not $prev.detect.present) { break }
      continue
    }
    $inHit = Get-Detect $m.id $current
    Write-Host ("  round {0} inPresent={1} lift={2:N2} rank={3} inChars={4}" -f $r, $inHit.present, $inHit.lift, $inHit.rank, $current.Length)
    $output = $null
    $det = $null
    $echo = $false
    $tries = 0
    for ($try = 0; $try -lt 3; $try++) {
      $tries++
      $rawOut = Invoke-LlamaChat $current
      if ([string]::IsNullOrWhiteSpace($rawOut)) {
        throw ("empty llama output at {0} round {1} try {2}; stop dirty serial" -f $m.id, $r, $try)
      }
      $exam = Test-SerialExamDump $rawOut
      $cand = Get-SerialContinuation $rawOut
      $echo = Test-Echo $current $cand
      if ([string]::IsNullOrWhiteSpace($cand) -or $echo -or $exam) {
        $output = $rawOut
        $det = Get-Detect $m.id $(if ($cand) { $cand } else { $rawOut })
        Write-Host ("  round {0} try {1} dirty exam={2} echo={3} cleanChars={4}; retry same present seed" -f $r, $try, $exam, $echo, $cand.Length)
        continue
      }
      $candDet = Get-Detect $m.id $cand
      $output = $cand
      $det = $candDet
      if ($candDet.present) { break }
      Write-Host ("  round {0} try {1} miss lift={2:N2}; retry same present seed" -f $r, $try, $candDet.lift)
    }
    $echo = Test-Echo $current $output
    $exam = Test-SerialExamDump $output
    $clean = Get-SerialContinuation $output
    $dirty = $echo -or $exam -or [string]::IsNullOrWhiteSpace($clean)
    $judge = ($r -ge $JudgeAfter)
    if (-not $dirty -and $det.present) { $row.presentRounds++ }
    if ($echo) { $row.echoRounds++ }
    $rec = [ordered]@{
      mode = $m.pole
      id = $m.id
      round = $r
      prompt = $current
      output = $output
      detectIn = $inHit
      detect = $det
      echo = $echo
      exam = $exam
      dirty = $dirty
      judge = $judge
      tries = $tries
    }
    $row.rounds += $rec
    Add-Jsonl $Journal $rec
    Write-Host ("  round {0} outPresent={1} lift={2:N2} rank={3} echo={4} exam={5} dirty={6} judge={7} tries={8} outChars={9}" -f $r, $det.present, $det.lift, $det.rank, $echo, $exam, $dirty, $judge, $tries, $output.Length)
    if ($dirty) {
      Write-Host ("  stop {0}: do not chain echo/exam text" -f $m.id)
      $doc.groups += $row
      Save-JsonAtomic $ResultFile $doc
      break
    }
    if ($judge -and -not $det.present) {
      Write-Host ("  stop {0}: judged absent after round {1}" -f $m.id, $r)
      $doc.groups += $row
      Save-JsonAtomic $ResultFile $doc
      break
    }
    $current = $clean
    $snapDoc = [ordered]@{}
    foreach ($k in $doc.Keys) { $snapDoc[$k] = $doc[$k] }
    $snapDoc.groups = @($doc.groups + $row)
    Save-JsonAtomic $ResultFile $snapDoc
  }
  if ($row.rounds.Count -gt 0 -and $doc.groups[-1] -ne $row) {
    $already = $false
    foreach ($g in $doc.groups) { if ($g.id -eq $row.id -and $g.pole -eq $row.pole) { $already = $true } }
    if (-not $already) {
      $doc.groups += $row
      Save-JsonAtomic $ResultFile $doc
    }
  }
}

$doc.finishedAtUtc = [DateTime]::UtcNow.ToString('o')
Save-JsonAtomic $ResultFile $doc
Write-Host ("Done " + $ResultFile)
