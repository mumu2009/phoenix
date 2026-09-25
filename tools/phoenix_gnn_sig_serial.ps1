# Official-readme significance serial: phrase memes, full-graph analyzeGraph.
# Does not overwrite gnn_meme_serial.json. PowerShell only.

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
. (Join-Path $PSScriptRoot 'experiment_ha.ps1')

$Phoenix = Split-Path -Parent $PSScriptRoot
$Exe = Join-Path $Phoenix 'build\meme_barrier_instrument.exe'
$Corpus = Join-Path $Phoenix 'robots\wikitext-103-all.txt'
if (-not (Test-Path -LiteralPath $Corpus)) {
  $Corpus = Join-Path $Phoenix 'robots\wikitext-101-all.txt'
}
$ScreenFile = Join-Path $Phoenix 'runtime_store\gnn_sig_screen.json'
$SnapFile = Join-Path $Phoenix 'runtime_store\gnn_instrument.txt'
$ResultFile = Join-Path $Phoenix 'runtime_store\gnn_sig_serial.json'
$Journal = Join-Path $Phoenix 'runtime_store\gnn_sig_serial.jsonl'
$Api = 'http://127.0.0.1:8082/completion'
$Rounds = 6
$TimeoutMs = 7200000
$Units = 256

if (-not (Test-Path -LiteralPath $Exe)) { throw "missing $Exe" }
if (-not (Test-Path -LiteralPath $Corpus)) { throw "missing wikitext" }

function Invoke-LlamaCompletion([string]$prompt) {
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

Write-Host '==== official analyzeGraph phrase most/least (no node clip) ===='
$screenRaw = & $Exe screen --corpus $Corpus --units $Units --snap $SnapFile
[IO.File]::WriteAllText($ScreenFile, $screenRaw, (New-Object System.Text.UTF8Encoding $false))
$screen = $screenRaw | ConvertFrom-Json
Write-Host ("analyzed={0} nodes={1} kept={2}" -f $screen.analyzed, $screen.nodes, @($screen.memes).Count)

$done = Read-JsonlCompleted $Journal
$doc = [ordered]@{
  startedAtUtc = [DateTime]::UtcNow.ToString('o')
  method = 'readme 统计+识别: full-graph resolvent; detect = GNN diffusion only (1/df seeds, no cosine)'
  note = 'stopwords kept as low-weight mapping; track meme id; maxMemeWords split on ingest'
  corpus = $Corpus
  units = $Units
  groups = @()
}

foreach ($m in @($screen.memes)) {
  Write-Host ("==== {0} {1} sig={2} ====" -f $m.pole, $m.id, $m.significance)
  $row = [ordered]@{
    id = $m.id
    pole = $m.pole
    rankMost = $m.rankMost
    rankLeast = $m.rankLeast
    significance = $m.significance
    wordCount = $m.wordCount
    carrierTruncated = $m.carrierTruncated
    seedCarrier = $m.carrier
    rounds = @()
    presentRounds = 0
  }
  $current = [string]$m.carrier
  for ($r = 1; $r -le $Rounds; $r++) {
    $key = '{0}|{1}|{2}' -f $m.pole, $m.id, $r
    if ($done.ContainsKey($key)) {
      $prev = $done[$key]
      $row.rounds += $prev
      if ($prev.detect.present) { $row.presentRounds++ }
      $current = [string]$prev.output
      Write-Host ("  resume round {0} present={1}" -f $r, $prev.detect.present)
      continue
    }
    $seedHit = Get-Detect $m.id $current
    Write-Host ("  round {0} inPresent={1} act={2} rank={3} inChars={4}" -f $r, $seedHit.present, $seedHit.activation, $seedHit.rank, $current.Length)
    $output = Invoke-LlamaCompletion $current
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
    $done[$key] = $rec
    Write-Host ("  round {0} outPresent={1} act={2} rank={3} outChars={4}" -f $r, $det.present, $det.activation, $det.rank, $output.Length)
    $current = $output
    if ([string]::IsNullOrWhiteSpace($current)) { break }
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
