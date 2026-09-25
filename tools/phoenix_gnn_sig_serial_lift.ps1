# Empty-session serial on the frozen phrase-meme screen.
# Detect = GNN 1/df + graph-prior lift only. Does not overwrite
# gnn_instrument.txt or the cosine-era gnn_sig_serial.json.
# PowerShell only. Does not call llama /health.

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
. (Join-Path $PSScriptRoot 'experiment_ha.ps1')

$Phoenix = Split-Path -Parent $PSScriptRoot
$Exe = Join-Path $Phoenix 'build\meme_barrier_instrument.exe'
$ScreenFile = Join-Path $Phoenix 'runtime_store\gnn_sig_screen.json'
$SnapFile = Join-Path $Phoenix 'runtime_store\gnn_instrument.txt'
$ResultFile = Join-Path $Phoenix 'runtime_store\gnn_sig_serial_lift.json'
$Journal = Join-Path $Phoenix 'runtime_store\gnn_sig_serial_lift.jsonl'
$Api = 'http://127.0.0.1:8082/completion'
$Rounds = 6
$TimeoutMs = 7200000

if (-not (Test-Path -LiteralPath $Exe)) { throw "missing $Exe" }
if (-not (Test-Path -LiteralPath $SnapFile)) { throw "missing frozen snap $SnapFile" }
if (-not (Test-Path -LiteralPath $ScreenFile)) { throw "missing screen $ScreenFile" }

function Invoke-LlamaCompletion([string]$prompt) {
  if ([string]::IsNullOrWhiteSpace($prompt)) {
    throw 'empty prompt (do not continue dirty serial)'
  }
  $payload = @{
    prompt = $prompt
    n_predict = 256
    stream = $true
    cache_prompt = $false
    temperature = 0.35
    top_p = 0.9
  }
  $body = $payload | ConvertTo-Json -Compress -Depth 6
  $bytes = [Text.Encoding]::UTF8.GetBytes($body)
  $req = [Net.HttpWebRequest]::Create($Api)
  $req.Method = 'POST'
  $req.ContentType = 'application/json; charset=utf-8'
  $req.Timeout = $TimeoutMs
  $req.ReadWriteTimeout = $TimeoutMs
  $req.ContentLength = $bytes.Length
  $s = $req.GetRequestStream()
  $s.Write($bytes, 0, $bytes.Length)
  $s.Close()
  $resp = $req.GetResponse()
  $reader = New-Object IO.StreamReader($resp.GetResponseStream(), [Text.Encoding]::UTF8)
  $chunks = New-Object Text.StringBuilder
  try {
    while ($null -ne ($line = $reader.ReadLine())) {
      if ([string]::IsNullOrWhiteSpace($line)) { continue }
      if ($line.StartsWith('data:')) { $line = $line.Substring(5).Trim() }
      if ($line -eq '[DONE]') { break }
      $obj = $line | ConvertFrom-Json
      if ($obj.content) { [void]$chunks.Append([string]$obj.content) }
      if ($obj.stop -eq $true) { break }
    }
  } finally {
    $reader.Close()
    $resp.Close()
  }
  return $chunks.ToString()
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

$screen = Get-Content -LiteralPath $ScreenFile -Raw -Encoding UTF8 | ConvertFrom-Json
$memes = @($screen.memes)
if ($memes.Count -lt 1) { throw 'empty screen' }

Write-Host '==== GNN+lift empty-session serial on frozen instrument ===='
Write-Host ("snap={0} memes={1} nodes={2}" -f $SnapFile, $memes.Count, $screen.nodes)

$doc = [ordered]@{
  startedAtUtc = [DateTime]::UtcNow.ToString('o')
  method = 'readme 统计+识别: frozen analyzeGraph screen; detect = GNN 1/df + graph-prior lift (no cosine)'
  note = 'empty HTTP session each round (cache_prompt=false); feed previous output as next prompt; do not ingest output into instrument; do not overwrite cosine-era gnn_sig_serial.json'
  snap = $SnapFile
  screen = $ScreenFile
  units = $screen.nodes
  groups = @()
}

foreach ($m in $memes) {
  Write-Host ("==== {0} {1} sig={2} ====" -f $m.pole, $m.id, $m.significance)
  $row = [ordered]@{
    id = $m.id
    pole = $m.pole
    rankMost = $m.rankMost
    rankLeast = $m.rankLeast
    significance = $m.significance
    wordCount = $m.wordCount
    seedCarrier = $m.carrier
    rounds = @()
    presentRounds = 0
  }
  $current = [string]$m.carrier
  for ($r = 1; $r -le $Rounds; $r++) {
    $seedHit = Get-Detect $m.id $current
    Write-Host ("  round {0} inPresent={1} lift={2:N2} rank={3} inChars={4}" -f $r, $seedHit.present, $seedHit.lift, $seedHit.rank, $current.Length)
    $output = Invoke-LlamaCompletion $current
    if ([string]::IsNullOrWhiteSpace($output)) {
      throw ("empty llama output at {0} round {1}; stop dirty serial" -f $m.id, $r)
    }
    $det = Get-Detect $m.id $output
    if ($det.present) { $row.presentRounds++ }
    $rec = [ordered]@{
      mode = $m.pole
      id = $m.id
      round = $r
      prompt = $current
      output = $output
      detectIn = $seedHit
      detect = $det
    }
    $row.rounds += $rec
    Add-Jsonl $Journal $rec
    Write-Host ("  round {0} outPresent={1} lift={2:N2} rank={3} outChars={4}" -f $r, $det.present, $det.lift, $det.rank, $output.Length)
    $current = $output
    $snapDoc = [ordered]@{}
    foreach ($k in $doc.Keys) { $snapDoc[$k] = $doc[$k] }
    $snapDoc.groups = @($doc.groups + $row)
    Save-JsonAtomic $ResultFile $snapDoc
  }
  $doc.groups += $row
  Save-JsonAtomic $ResultFile $doc
}

$doc.finishedAtUtc = [DateTime]::UtcNow.ToString('o')
Save-JsonAtomic $ResultFile $doc
Write-Host ("Done " + $ResultFile)
