# 1h host soak for GNN identify: 1/df seeds + lift vs high-df baseline.
# Does not call llama. Does not overwrite the default frozen snap unless asked.
# PowerShell only.
param(
  [string]$SnapName = 'gnn_instrument.txt',
  [string]$ScreenName = 'gnn_sig_screen.json',
  [string]$JournalName = 'gnn_identify_soak.jsonl',
  [double]$Hours = 1
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
. (Join-Path $PSScriptRoot 'experiment_ha.ps1')

$Phoenix = Split-Path -Parent $PSScriptRoot
$Exe = Join-Path $Phoenix 'build\meme_barrier_instrument.exe'
$Snap = Join-Path $Phoenix ('runtime_store\{0}' -f $SnapName)
$ScreenFile = Join-Path $Phoenix ('runtime_store\{0}' -f $ScreenName)
$Journal = Join-Path $Phoenix ('runtime_store\{0}' -f $JournalName)
$Corpus = Join-Path $Phoenix 'robots\wikitext-103-all.txt'
if (-not (Test-Path -LiteralPath $Corpus)) {
  $Corpus = Join-Path $Phoenix 'robots\wikitext-101-all.txt'
}
$TmpSnap = Join-Path $Phoenix 'runtime_store\gnn_soak_tmp.txt'
$SleepSec = 20

if (-not (Test-Path -LiteralPath $Exe)) { throw "missing $Exe" }
if (-not (Test-Path -LiteralPath $Snap)) { throw "missing frozen snap $Snap" }

function Get-Detect([string]$snap, [string]$memeId, [string]$text) {
  $out = $text | & $Exe detect --snap $snap --meme $memeId
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

$stops = 'the of and to a in on for is was with as by at from that this it be or are'
$unrelated = 'quantum chess tournament pairing sheet omega7 bakery inventory flour yeast'
$deadline = [DateTime]::UtcNow.AddHours($Hours)
$started = [DateTime]::UtcNow
$cycles = 0
$fails = 0
Write-Host ("soak start {0:o} until {1:o}" -f $started, $deadline)

while ([DateTime]::UtcNow -lt $deadline) {
  $cycles++
  $snapNow = $Snap
  $tmpScreen = Join-Path $Phoenix 'runtime_store\gnn_soak_tmp_screen.json'
  if (($cycles % 15) -eq 0) {
    $scr = & $Exe screen --corpus $Corpus --units 48 --snap $TmpSnap
    if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $TmpSnap)) {
      [IO.File]::WriteAllText($tmpScreen, [string]$scr, (New-Object System.Text.UTF8Encoding $false))
      $snapNow = $TmpSnap
    }
  }
  $ok = $true
  $rows = @()
  $use = $memes
  if ($snapNow -eq $TmpSnap) {
    $fresh = Get-Content -LiteralPath (Join-Path $Phoenix 'runtime_store\gnn_soak_tmp_screen.json') -Raw -Encoding UTF8 -ErrorAction SilentlyContinue | ConvertFrom-Json
    if ($fresh -and $fresh.memes) { $use = @($fresh.memes) }
    else { $snapNow = $Snap }
  }
  foreach ($m in $use) {
    $own = Get-Detect $snapNow $m.id ([string]$m.carrier)
    $st = Get-Detect $snapNow $m.id $stops
    $un = Get-Detect $snapNow $m.id $unrelated
    $rowOk = ($own.present -eq $true) -and ($st.present -eq $false) -and ($un.present -eq $false)
    if (-not $rowOk) { $ok = $false }
    $rows += [ordered]@{
      id = $m.id
      pole = $m.pole
      own = $own
      stops = $st
      unrelated = $un
      ok = $rowOk
    }
  }
  if (-not $ok) { $fails++ }
  Add-Jsonl $Journal ([ordered]@{
    utc = [DateTime]::UtcNow.ToString('o')
    cycle = $cycles
    ok = $ok
    fails = $fails
    snap = $snapNow
    rows = $rows
  })
  Write-Host ("cycle={0} ok={1} fails={2} elapsedMin={3:N1}" -f $cycles, $ok, $fails, ([DateTime]::UtcNow - $started).TotalMinutes)
  if (-not $ok) {
    Write-Host 'identify soak failed; stop (do not continue dirty run)'
    break
  }
  if ([DateTime]::UtcNow -ge $deadline) { break }
  Start-Sleep -Seconds $SleepSec
}

Write-Host ("Done cycles={0} fails={1} journal={2}" -f $cycles, $fails, $Journal)
if ($fails -gt 0) { exit 2 }
exit 0
