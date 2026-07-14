param(
  [int]$SamplePerScenario = 100,
  [int]$TimeoutSeconds = 90,
  [int]$WarmupTimeoutSeconds = 120,
  [int]$WarmupRetries = 3,
  [string]$PhoenixPort = '5080',
  [string]$StudyPort = '5081',
  [string]$LlamaServerPort = '8083',
  [string]$LlamaServerHost = '127.0.0.1',
  [string]$LlamaServerModelPath = 'GGUF_models\blobs\sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29',
  [string]$LlamaServerModel = 'llama3.1:8b',
  [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

Set-Location (Split-Path -Parent $PSScriptRoot)

function Wait-TcpPort {
  param(
    [string]$HostName,
    [int]$Port,
    [int]$TimeoutSeconds = 60
  )

  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    try {
      $client = New-Object System.Net.Sockets.TcpClient
      $iar = $client.BeginConnect($HostName, $Port, $null, $null)
      $ok = $iar.AsyncWaitHandle.WaitOne(1000)
      if ($ok -and $client.Connected) {
        $client.EndConnect($iar)
        $client.Close()
        return $true
      }
      $client.Close()
    } catch {
    }
    Start-Sleep -Milliseconds 500
  }
  return $false
}

function Stop-ManagedProcesses {
  param([int[]]$ProcessIds)
  foreach ($pid in $ProcessIds) {
    if ($pid -gt 0) {
      Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
    }
  }
}

Get-CimInstance Win32_Process |
  Where-Object { $_.Name -ieq 'python.exe' -and $_.CommandLine -like '*memory_tier_benchmark_v1.py*' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Get-Process ollama,llama-server,phoenix_main -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }

$llamaServerArgs = @(
  '-m', $LlamaServerModelPath,
  '-c', '4096',
  '-t', '8',
  '-ngl', '0',
  '--port', $LlamaServerPort,
  '--host', $LlamaServerHost
)

$phoenixArgs = @(
  '--port=' + $PhoenixPort,
  '--study-port=' + $StudyPort,
  '--frontend-enabled=false',
  '--http-log=false',
  '--frontend-http-log=false',
  '--using-ollama=false',
  '--transformer-mode=llamacpp',
  '--llamacpp-base-url=http://' + $LlamaServerHost + ':' + $LlamaServerPort,
  '--external-auto-launch=false',
  '--bug-shooter=false',
  '--gguf-models-dir=GGUF_models'
)

$benchmarkArgs = @(
  '-u', 'tools\memory_tier_benchmark_v1.py',
  '--sample-per-scenario', $SamplePerScenario,
  '--providers', 'phoenix',
  '--timeout', $TimeoutSeconds,
  '--warmup-timeout', $WarmupTimeoutSeconds,
  '--warmup-retries', $WarmupRetries,
  '--json-output', 'build\memory_tier_benchmark_v1.json',
  '--md-output', 'build\memory_tier_benchmark_v1.md'
)

if ($DryRun) {
  Write-Host "llama-server: outsides\llamacpp\build-gcc\bin\llama-server.exe $($llamaServerArgs -join ' ')"
  Write-Host ".\phoenix_main.exe $($phoenixArgs -join ' ')"
  Write-Host "C:\Python314\python.exe $($benchmarkArgs -join ' ')"
  exit 0
}

$env:OLLAMA_LOAD_TIMEOUT = '20m'
$llamaProc = $null
$phoenixProc = $null

try {
  Write-Host '[phase] starting llama-server'
  $llamaProc = Start-Process -FilePath 'outsides\llamacpp\build-gcc\bin\llama-server.exe' -ArgumentList $llamaServerArgs -PassThru
  if (-not (Wait-TcpPort -HostName $LlamaServerHost -Port ([int]$LlamaServerPort) -TimeoutSeconds 60)) {
    throw ("llama-server failed to listen on {0}:{1}" -f $LlamaServerHost, $LlamaServerPort)
  }

  Write-Host '[phase] warming llama-server'
  $warmPayload = @{
    model = $LlamaServerModel
    messages = @(@{ role = 'user'; content = 'Reply exactly with OK.' })
    stream = $false
    options = @{ num_ctx = 4096; num_predict = 1; temperature = 0 }
    keep_alive = '30m'
  } | ConvertTo-Json -Depth 6 -Compress
  & curl.exe --max-time 300 -sS -X POST ('http://{0}:{1}/v1/chat/completions' -f $LlamaServerHost, $LlamaServerPort) -H 'Content-Type: application/json' -d "$warmPayload" | Out-Null

  Write-Host '[phase] starting phoenix_main'
  $phoenixProc = Start-Process -FilePath '.\phoenix_main.exe' -ArgumentList $phoenixArgs -PassThru
  if (-not (Wait-TcpPort -HostName '127.0.0.1' -Port ([int]$PhoenixPort) -TimeoutSeconds 60)) {
    throw "phoenix_main failed to listen on 127.0.0.1:$PhoenixPort"
  }

  Write-Host '[phase] running memory benchmark'
  & 'C:\Python314\python.exe' @benchmarkArgs
  $exitCode = $LASTEXITCODE
  if ($exitCode -ne 0) {
    exit $exitCode
  }
} finally {
  $cleanupIds = @()
  if ($phoenixProc) { $cleanupIds += $phoenixProc.Id }
  if ($llamaProc) { $cleanupIds += $llamaProc.Id }
  Stop-ManagedProcesses -ProcessIds $cleanupIds
}
