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
      # Keep retrying until timeout.
    }
    Start-Sleep -Milliseconds 500
  }
  return $false
}

# Clean old processes.
Get-CimInstance Win32_Process | Where-Object { $_.Name -ieq 'python.exe' -and $_.CommandLine -like '*investor_benchmark_v3_tri.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Get-Process ollama,llama-server,phoenix_main -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.Id -Force }

# Phase 1: Ollama + Phoenix (llama-server disabled).
$env:OLLAMA_LOAD_TIMEOUT = '15m'
Write-Host '[phase1] starting ollama service'
$ollamaProc = Start-Process -FilePath 'ollama' -ArgumentList 'serve' -PassThru
Start-Sleep -Seconds 3
Write-Host '[phase1] starting phoenix_main'
$phoenixProc = Start-Process -FilePath '.\phoenix_main.exe' -ArgumentList @(
  '--port=5080',
  '--study-port=5081',
  '--frontend-enabled=false',
  '--http-log=false',
  '--frontend-http-log=false',
  '--using-ollama=true',
  '--ollama-model=llama3.1:8b',
  '--ollama-base-url=http://127.0.0.1:11434',
  '--external-auto-launch=false',
  '--bug-shooter=false'
) -PassThru

if (-not (Wait-TcpPort -HostName '127.0.0.1' -Port 5080 -TimeoutSeconds 60)) {
  throw 'phoenix_main failed to listen on 127.0.0.1:5080'
}

Write-Host '[phase1] prewarming ollama (max 90s)'
try {
  $warmPayload = @{
    model = 'llama3.1:8b'
    messages = @(@{ role = 'user'; content = 'warmup' })
    stream = $false
    options = @{ num_ctx = 4096; num_predict = 1 }
  } | ConvertTo-Json -Depth 6
  & curl.exe --max-time 90 -sS -X POST 'http://127.0.0.1:11434/api/chat' -H 'Content-Type: application/json' -d $warmPayload | Out-Null
} catch {
  Write-Host "[warn] prewarm_ollama_failed: $($_.Exception.Message)"
}

Write-Host '[phase1] running benchmark script'
& 'C:\Python314\python.exe' -u tools\investor_benchmark_v3_tri.py `
  --instruction-samples 50 `
  --window-samples 50 `
  --context-window 4096 `
  --timeout 600 `
  --llama-server-url 'http://127.0.0.1:1/v1/chat/completions' `
  --json-output build\investor_advantage_report_v3_phase_ollama_phoenix.json `
  --md-output build\investor_advantage_report_v3_phase_ollama_phoenix.md

Stop-Process -Id $ollamaProc.Id -Force -ErrorAction SilentlyContinue
Stop-Process -Id $phoenixProc.Id -Force -ErrorAction SilentlyContinue

# Phase 2: llama-server only (Ollama/Phoenix disabled).
Write-Host '[phase2] starting llama-server'
$llamaProc = Start-Process -FilePath 'outsides\llamacpp\build-gcc\bin\llama-server.exe' -ArgumentList @(
  '-m','GGUF_models\blobs\sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29',
  '-c','4096',
  '-t','8',
  '-ngl','0',
  '--port','8083',
  '--host','127.0.0.1'
) -PassThru
Start-Sleep -Seconds 8

Write-Host '[phase2] running benchmark script'
& 'C:\Python314\python.exe' -u tools\investor_benchmark_v3_tri.py `
  --instruction-samples 50 `
  --window-samples 50 `
  --context-window 4096 `
  --timeout 120 `
  --ollama-url 'http://127.0.0.1:1/api/chat' `
  --phoenix-transformer-chat-url 'http://127.0.0.1:1/api/transformer/chat' `
  --phoenix-runtime-url 'http://127.0.0.1:1/api/runtime/features' `
  --json-output build\investor_advantage_report_v3_phase_llama_server.json `
  --md-output build\investor_advantage_report_v3_phase_llama_server.md

Stop-Process -Id $llamaProc.Id -Force -ErrorAction SilentlyContinue

# Merge: take Ollama/Phoenix from phase1 and llama-server from phase2.
Write-Host '[merge] combining phase outputs'
$mergeCode = @'
import json
from pathlib import Path

root = Path(r"D:\_phoenix\_079\v6.0Alixander\v6.0Alixander")
phase1 = json.loads((root / "build/investor_advantage_report_v3_phase_ollama_phoenix.json").read_text(encoding="utf-8"))
phase2 = json.loads((root / "build/investor_advantage_report_v3_phase_llama_server.json").read_text(encoding="utf-8"))

for section in ("instruction", "windowMemory"):
    phase1["benchmarks"][section]["llamaServer"] = phase2["benchmarks"][section]["llamaServer"]
    phase1["benchmarks"][section]["llamaServerLatency"] = phase2["benchmarks"][section]["llamaServerLatency"]
    phase1["benchmarks"][section]["llamaServerHttpErrors"] = phase2["benchmarks"][section]["llamaServerHttpErrors"]

if "raw" in phase1 and "raw" in phase2:
    for section in ("instruction", "windowMemory"):
        p2_rows = phase2["raw"].get(section, [])
        p2_map = {r.get("example_id"): r.get("llama_server", {}) for r in p2_rows if isinstance(r, dict)}
        for row in phase1["raw"].get(section, []):
            if isinstance(row, dict):
                row["llama_server"] = p2_map.get(row.get("example_id"), row.get("llama_server", {}))

out_json = root / "build/investor_advantage_report_v3_tri.json"
out_json.write_text(json.dumps(phase1, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[OK] merged -> {out_json}")
'@

$mergeTmp = 'build\\_merge_tri.py'
Set-Content -Path $mergeTmp -Value $mergeCode -Encoding UTF8
& 'C:\Python314\python.exe' $mergeTmp
