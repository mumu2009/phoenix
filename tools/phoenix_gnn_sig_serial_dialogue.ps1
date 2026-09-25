# Dialogue-class 隔代: wipe GNN, ingest last output only.
# Recovery = live graph grew the same ordered (word, α) typical set
# (not the same meme_p_* hash). Next utterance is live tensor-sentence.
# Llama still sees the short carrier only (raw /completion, no chat template).
# Does not overwrite frozen snap or continuation serial JSON. PowerShell only.
param(
  [string]$OutName = 'gnn_sig_serial_dialogue',
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
if (-not (Get-Process -Name llama-server -ErrorAction SilentlyContinue)) {
  throw 'llama-server is not running; do not start a second one'
}

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
  try { $raw = $reader.ReadToEnd() } finally { $reader.Close(); $resp.Close() }
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

function Get-WipeIngest([string]$memeId, [string]$text) {
  $raw = $text | & $Exe wipe-ingest --snap $SnapFile --units $Units --meme $memeId
  if ($LASTEXITCODE -ne 0) { throw ("wipe-ingest exit " + $LASTEXITCODE + ": " + $raw) }
  return ($raw | ConvertFrom-Json)
}

function Test-Echo([string]$prompt, [string]$output) {
  return Test-SerialEcho $prompt $output
}

$screen = Get-Content -LiteralPath $ScreenFile -Raw -Encoding UTF8 | ConvertFrom-Json
$memes = @($screen.memes)
if ($memes.Count -lt 1) { throw 'empty screen' }

Write-Host '==== dialogue serial: wipe GNN, ingest output only, recompose next utterance ===='
Write-Host ("snap={0} memes={1}" -f $SnapFile, $memes.Count)
$done = Read-JsonlCompleted $Journal

$doc = [ordered]@{
  startedAtUtc = [DateTime]::UtcNow.ToString('o')
  method = 'readme 单智能体隔代再摄入: empty GNN + ingest last output; recover same (word, α) typical set, not bag-hash id; llama raw /completion of short carrier only'
  note = 'not continuation-chain of llama essays; do not overwrite gnn_sig_serial_tensor3.json or frozen snap'
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
  Write-Host ("  compose present={0} lift={1:N2} rank={2} chars={3}" -f $seedHit.present, $seedHit.lift, $seedHit.rank, ([string]$composed.text).Length)
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
    wipeEmptyRounds = 0
    recomposePresentRounds = 0
    alphaSameRounds = 0
  }
  $current = [string]$composed.text
  for ($r = 1; $r -le $Rounds; $r++) {
    $key = '{0}|{1}|{2}' -f $m.pole, $m.id, $r
    if ($done.ContainsKey($key)) {
      $prev = $done[$key]
      $row.rounds += $prev
      if ($prev.detect.present) { $row.presentRounds++ }
      if ($prev.echo) { $row.echoRounds++ }
      if ($prev.wipe -and $prev.wipe.emptyBefore) { $row.wipeEmptyRounds++ }
      if ($prev.wipe -and $prev.wipe.recomposePresent) { $row.recomposePresentRounds++ }
      if ($prev.wipe -and $prev.wipe.alphaSame) { $row.alphaSameRounds++ }
      $current = [string]$prev.nextUtterance
      Write-Host ("  resume round {0} present={1} alphaSame={2}" -f $r, $prev.detect.present, $prev.wipe.alphaSame)
      if ($prev.judge -and -not $prev.wipe.alphaSame) { break }
      continue
    }
    $inHit = Get-Detect $m.id $current
    Write-Host ("  round {0} inPresent={1} lift={2:N2} inChars={3}" -f $r, $inHit.present, $inHit.lift, $current.Length)
    if ($r -eq 1 -and -not $inHit.present) {
      Write-Host ("  stop {0}: seed carrier lost id before generate" -f $m.id)
      break
    }
    $rawOut = Invoke-LlamaChat $current
    if ([string]::IsNullOrWhiteSpace($rawOut)) {
      throw ("empty llama output at {0} round {1}; stop dirty serial" -f $m.id, $r)
    }
    $output = Get-SerialContinuation $rawOut
    $exam = Test-SerialExamDump $output
    $echo = Test-Echo $current $output
    if ([string]::IsNullOrWhiteSpace($output) -or $exam) {
      Write-Host ("  stop {0}: do not ingest empty/exam dump exam={1}" -f $m.id, $exam)
      $det = Get-Detect $m.id $(if ($output) { $output } else { $rawOut })
      $rec = [ordered]@{
        mode = $m.pole
        id = $m.id
        round = $r
        prompt = $current
        output = $rawOut
        detectIn = $inHit
        detect = $det
        echo = $echo
        exam = $exam
        dirty = $true
        nextUtterance = ''
      }
      $row.rounds += $rec
      $doc.groups += $row
      Add-Jsonl $Journal $rec
      Save-JsonAtomic $ResultFile $doc
      break
    }
    # echo text is ingested too: alphaSame on the wiped graph decides
    # whether the copy carried the meme's (word, α) typical set.
    $det = Get-Detect $m.id $output
    $wipe = Get-WipeIngest $m.id $output
    if (-not $wipe.emptyBefore) {
      throw ("live GNN was not empty before ingest at {0} round {1}; stop" -f $m.id, $r)
    }
    $judge = ($r -ge $JudgeAfter)
    $next = [string]$wipe.recomposeText
    $recomposeFallback = $false
    $collapsed = [bool]$wipe.collapsed
    if ($collapsed) {
      # intercept: a collapsed output (token loop / single-token dominance,
      # the "aalborg" basin that ate two memes at round 5) must not steer
      # the next generation. Only the live graph's own recomposition may
      # carry the serial forward; without it the chain stops here.
      if ([string]::IsNullOrWhiteSpace($next)) {
        Write-Host ("  stop {0}: collapsed output (maxRun={1} top1={2:N2}) and no healthy recompose carrier; intercepted" -f $m.id, $wipe.outMaxRun, $wipe.outTop1Frac)
        $rec = [ordered]@{
          mode = $m.pole
          id = $m.id
          round = $r
          prompt = $current
          output = $output
          detectIn = $inHit
          detect = $det
          wipe = $wipe
          echo = $echo
          exam = $false
          dirty = $false
          judge = $judge
          collapsed = $true
          intercepted = $true
          recomposeFallback = $false
          nextUtterance = ''
        }
        $row.rounds += $rec
        $doc.groups += $row
        Add-Jsonl $Journal $rec
        Save-JsonAtomic $ResultFile $doc
        break
      }
      Write-Host ("  intercept {0}: collapsed output (maxRun={1} top1={2:N2}); continue on recompose carrier only" -f $m.id, $wipe.outMaxRun, $wipe.outTop1Frac)
    } elseif ([string]::IsNullOrWhiteSpace($next)) {
      # live graph has not grown a re-composable carrier yet; keep
      # generating on the raw output. alphaSame still judges.
      $next = $output
      $recomposeFallback = $true
    }
    if ($det.present) { $row.presentRounds++ }
    if ($echo) { $row.echoRounds++ }
    $row.wipeEmptyRounds++
    if ($wipe.recomposePresent) { $row.recomposePresentRounds++ }
    if ($wipe.alphaSame) { $row.alphaSameRounds++ }
    $rec = [ordered]@{
      mode = $m.pole
      id = $m.id
      round = $r
      prompt = $current
      output = $output
      detectIn = $inHit
      detect = $det
      wipe = $wipe
      echo = $echo
      exam = $false
      dirty = $false
      judge = $judge
      collapsed = $collapsed
      decayWarn = [bool]$wipe.decayWarn
      recomposeFallback = $recomposeFallback
      nextUtterance = $next
    }
    $row.rounds += $rec
    Add-Jsonl $Journal $rec
    Write-Host ("  round {0} outPresent={1} lift={2:N2} echo={3} judge={4} wipeEmpty={5} liveNodes={6} overlap={7} alphaSame={8} typical={9}/{10} sha1Recompose={11} fallback={12} collapsed={13} decayWarn={14} nextChars={15}" -f $r, $det.present, $det.lift, $echo, $judge, $wipe.emptyBefore, $wipe.liveNodes, $wipe.overlapContent, $wipe.alphaSame, $wipe.typicalHit, $wipe.typicalNeed, $wipe.recomposePresent, $recomposeFallback, $collapsed, $wipe.decayWarn, $next.Length)
    if ($judge -and -not $wipe.alphaSame) {
      Write-Host ("  stop {0}: judged - wiped GNN did not grow the same (word, α) typical set" -f $m.id)
      $doc.groups += $row
      Save-JsonAtomic $ResultFile $doc
      break
    }
    $current = $next
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
