# Shared backup / atomic save / dual-write. Dot-source from experiment scripts.
$script:HaPhoenix = Split-Path -Parent $PSScriptRoot
$script:HaStore = Join-Path $script:HaPhoenix 'runtime_store'
$script:HaMirror = '<ollama-models>\experiment_work'

function Backup-ExperimentSnapshot([string]$label) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    if ($label) { $stamp = $stamp + '_' + $label }
    $a = Join-Path $script:HaStore ("experiment_backups\" + $stamp)
    $b = Join-Path '<ollama-models>\experiment_backups' $stamp
    New-Item -ItemType Directory -Force -Path $a | Out-Null
    New-Item -ItemType Directory -Force -Path $b | Out-Null
    return @{ a = $a; b = $b }
}

function Save-JsonAtomic([string]$path, $doc) {
    $dir = Split-Path -Parent $path
    if (-not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
    $tmp = $path + '.tmp'
    $prev = $path + '.prev'
    $json = $doc | ConvertTo-Json -Depth 14
    $utf8 = New-Object System.Text.UTF8Encoding $false
    [IO.File]::WriteAllText($tmp, $json, $utf8)
    if (Test-Path -LiteralPath $path) {
        Copy-Item -LiteralPath $path -Destination $prev -Force
    }
    Move-Item -LiteralPath $tmp -Destination $path -Force
    $name = Split-Path -Leaf $path
    if (-not (Test-Path -LiteralPath $script:HaMirror)) {
        New-Item -ItemType Directory -Force -Path $script:HaMirror | Out-Null
    }
    Copy-Item -LiteralPath $path -Destination (Join-Path $script:HaMirror $name) -Force
}

function Add-Jsonl([string]$path, $obj) {
    $dir = Split-Path -Parent $path
    if (-not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
    $line = ($obj | ConvertTo-Json -Depth 12 -Compress)
    Add-Content -LiteralPath $path -Value $line -Encoding UTF8
    $name = Split-Path -Leaf $path
    if (-not (Test-Path -LiteralPath $script:HaMirror)) {
        New-Item -ItemType Directory -Force -Path $script:HaMirror | Out-Null
    }
    Copy-Item -LiteralPath $path -Destination (Join-Path $script:HaMirror $name) -Force
}

function Test-SerialExamDump([string]$text) {
    if ([string]::IsNullOrWhiteSpace($text)) { return $false }
    $t = $text
    if ($t -match '(?i)what (is|can|does) (the main subject|be inferred|the author)') { return $true }
    if ($t -match '(?i)what is the name of') { return $true }
    if ($t -match '(?i)who (is|was|sang) the') { return $true }
    if ($t -match '(?i)answer rationale|the best answer is|this question requires') { return $true }
    if ($t -match '(?i)\bAnswer:\s*[A-D]\b') { return $true }
    if ($t -match '(?m)^\s*[A-D]\s*\)\s') { return $true }
    if ($t -match '(?i)step-by-step solution|final answer is:') { return $true }
    return $false
}

function Get-SerialContinuation([string]$text) {
    if ([string]::IsNullOrWhiteSpace($text)) { return '' }
    $cut = $text
    $markers = @(
        '(?i)what is the main subject',
        '(?i)what is the name of',
        '(?i)who (is|was|sang) the',
        '(?i)what can be inferred',
        '(?i)answer rationale',
        '(?i)this question requires',
        '(?i)the best answer is',
        '(?i)step-by-step solution',
        '(?i)the final answer is',
        '(?m)^\s*[A-D]\s*\)\s'
    )
    $idx = $cut.Length
    foreach ($m in $markers) {
        $hit = [regex]::Match($cut, $m)
        if ($hit.Success -and $hit.Index -ge 0 -and $hit.Index -lt $idx) {
            $idx = $hit.Index
        }
    }
    if ($idx -lt $cut.Length) {
        $cut = $cut.Substring(0, $idx)
    }
    return ($cut -replace '\s+', ' ').Trim()
}

function Test-SerialEcho([string]$prompt, [string]$output) {
    if ([string]::IsNullOrWhiteSpace($prompt) -or [string]::IsNullOrWhiteSpace($output)) {
        return $false
    }
    $p = ($prompt -replace '\s+', ' ').Trim()
    $o = ($output -replace '\s+', ' ').Trim()
    if ($p.Length -lt 24 -or $o.Length -lt 24) {
        return ($o.Contains($p) -or $p.Contains($o))
    }
    foreach ($win in @(80, 32)) {
        $w = [Math]::Min($win, $p.Length)
        if ($w -lt 16) { continue }
        for ($i = 0; $i -le $p.Length - $w; $i += 8) {
            if ($o.Contains($p.Substring($i, $w))) { return $true }
        }
    }
    $pre = [Math]::Min(48, $p.Length)
    if ($pre -ge 24 -and $o.Contains($p.Substring(0, $pre))) { return $true }
    $suf = [Math]::Min(48, $p.Length)
    if ($suf -ge 24 -and $o.Contains($p.Substring($p.Length - $suf, $suf))) { return $true }
    if ($o.Length -ge 72) {
        $span = 24
        $seen = @{}
        for ($i = 0; $i -le $o.Length - $span; $i += 8) {
            $k = $o.Substring($i, $span)
            if ($seen.ContainsKey($k)) { $seen[$k]++ } else { $seen[$k] = 1 }
            if ($seen[$k] -ge 3) { return $true }
        }
    }
    return $false
}

function Read-JsonlCompleted([string]$path) {
    $done = @{}
    if (-not (Test-Path -LiteralPath $path)) { return $done }
    foreach ($line in Get-Content -LiteralPath $path -Encoding UTF8) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $o = $line | ConvertFrom-Json
        $key = '{0}|{1}|{2}' -f $o.mode, $o.id, $o.round
        $done[$key] = $o
    }
    return $done
}
