param(
  [string]$Username = 'vboxuser',
  [string]$Password = 'changeme',
  [string]$Version = 'v2.0Multi',
  [string]$BaseUrl = 'http://127.0.0.1:5081',
  [string]$QuestionFile = 'questionaire.txt'
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

function Invoke-JsonRequest {
  param(
    [string]$Url,
    [string]$Method = 'GET',
    $Body = $null,
    $Headers = $null,
    [int]$TimeoutSec = 120
  )
  try {
    $p = @{ Uri = $Url; Method = $Method; TimeoutSec = $TimeoutSec }
    if ($Headers) { $p.Headers = $Headers }
    if ($null -ne $Body) {
      $p.ContentType = 'application/json'
      $p.Body = ($Body | ConvertTo-Json -Depth 20 -Compress)
    }
    $r = Invoke-WebRequest @p
    return [pscustomobject]@{ status = [int]$r.StatusCode; body = [string]$r.Content; err = '' }
  } catch {
    $resp = $_.Exception.Response
    $code = -1
    $txt = ''
    if ($resp) {
      try { $code = [int]$resp.StatusCode } catch {}
      try {
        if ($resp -is [System.Net.Http.HttpResponseMessage]) {
          $txt = [string]($resp.Content.ReadAsStringAsync().GetAwaiter().GetResult())
        } else {
          $sr = New-Object IO.StreamReader($resp.GetResponseStream())
          $txt = $sr.ReadToEnd()
        }
      } catch {}
    }
    return [pscustomobject]@{ status = $code; body = $txt; err = $_.Exception.Message }
  }
}

function Extract-Reply {
  param([string]$Body)
  if ([string]::IsNullOrWhiteSpace($Body)) {
    return ''
  }
  try {
    $obj = $Body | ConvertFrom-Json
    if ($obj.result -and $obj.result.reply) {
      return [string]$obj.result.reply
    }
    if ($obj.error) {
      return "[ERROR] $($obj.error)"
    }
  } catch {
  }
  return ''
}

function Start-Backend {
  $args = @(
    '--gateway-host=127.0.0.1',
    '--port=5080',
    '--study-port=5081',
    '--base-dir=runtime_store',
    '--db-path=runtime_store/ai_store.sqlite',
    '--lmdb-dir=lmdb',
    '--robots-dir=robots',
    '--redis-url=redis://127.0.0.1:6379',
    '--log-mode=release',
    '--tests-autoload=true',
    '--robots-limit=10000000',
    '--robots-autoload=true',
    '--tests-dir=tests'
  )

  $exe = $null
  if (Test-Path .\build\phoenix_main.exe) {
    $exe = (Resolve-Path .\build\phoenix_main.exe).Path
  } elseif (Test-Path .\phoenix_main.exe) {
    $exe = (Resolve-Path .\phoenix_main.exe).Path
  } else {
    throw 'phoenix_main.exe not found in build/ or workspace root'
  }

  Get-Process phoenix_main -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  Start-Sleep -Milliseconds 600
  Start-Process -FilePath $exe -ArgumentList $args | Out-Null
  Start-Sleep -Seconds 7
}

function Login-Token {
  param(
    [string]$BaseUrl,
    [string]$Username,
    [string]$Password
  )
  $login = Invoke-JsonRequest "$BaseUrl/auth/login" 'POST' @{ username = $Username; password = $Password } $null 60
  if ($login.status -ne 200) {
    throw "login failed: status=$($login.status) err=$($login.err) body=$($login.body)"
  }
  $token = ''
  try { $token = (($login.body | ConvertFrom-Json).token) } catch {}
  if ([string]::IsNullOrWhiteSpace($token)) {
    throw 'login succeeded but token missing'
  }
  return $token
}

if (-not (Test-Path $QuestionFile)) {
  throw "question file not found: $QuestionFile"
}

$questions = Get-Content $QuestionFile -Encoding UTF8 | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' }
if ($questions.Count -eq 0) {
  throw 'question file is empty'
}

$token = Login-Token -BaseUrl $BaseUrl -Username $Username -Password $Password
$headers = @{ Authorization = "Bearer $token" }
$lines = New-Object System.Collections.Generic.List[string]

for ($i = 0; $i -lt $questions.Count; $i++) {
  $q = $questions[$i]
  $answer = ''
  $lastStatus = -1
  $lastErr = ''

  for ($attempt = 1; $attempt -le 3; $attempt++) {
    $sessionId = "qa-$($i+1)-$attempt-$([Guid]::NewGuid().ToString('N').Substring(0,6))"
    $res = Invoke-JsonRequest "$BaseUrl/api/chat" 'POST' @{ text = $q; sessionId = $sessionId; enableGraphSelector = $true } $headers 180
    $lastStatus = $res.status
    $lastErr = $res.err
    $answer = Extract-Reply $res.body

    $isDisconnected = $false
    if (-not [string]::IsNullOrWhiteSpace($answer)) {
      $t = $answer.Trim().ToLowerInvariant()
      if ($t -eq '[error] disconnected' -or $t -eq 'disconnected') { $isDisconnected = $true }
    }

    if (-not [string]::IsNullOrWhiteSpace($answer) -and -not $isDisconnected) {
      break
    }

    if ($attempt -lt 3) {
      try {
        Start-Backend
        $token = Login-Token -BaseUrl $BaseUrl -Username $Username -Password $Password
        $headers = @{ Authorization = "Bearer $token" }
      } catch {
      }
      Start-Sleep -Seconds 1
    }
  }

  if ([string]::IsNullOrWhiteSpace($answer) -or $answer.Trim().ToLowerInvariant() -eq '[error] disconnected' -or $answer.Trim().ToLowerInvariant() -eq 'disconnected') {
    $answer = "[NO_ANSWER] status=$lastStatus err=$lastErr"
  }

  $answer = ($answer -replace "`r", ' ' -replace "`n", ' ').Trim()
  $lines.Add("$($i+1).$answer")
}

$time = Get-Date -Format 'yyyyMMdd-HHmmss'
$outFile = "answer_${time}-${Version}.txt"
$lines | Set-Content -Path $outFile -Encoding UTF8

Write-Output $outFile
