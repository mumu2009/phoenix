# C-Eval benchmark with/without GNN comparison.
# Assumes phoenix_main is already compiled (run compile.bat first if needed).
# Uses Phoenix /api/chat (the backend endpoint the web UI also calls), so it
# tests the augmented llama-3.1, not the raw llama-server.
param(
    [string]$Root = "D:\_phoenix\_079\v6.0Alixander\phoenix",
    [int]$PerSubject = 1
)
$ErrorActionPreference = "Stop"
Push-Location -LiteralPath $Root

function Stop-StaleBackend() {
    Get-Process -Name "phoenix_main" -ErrorAction SilentlyContinue | Stop-Process -Force
    Get-Process -Name "llama-server" -ErrorAction SilentlyContinue | Stop-Process -Force
    Get-Process -Name "python" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*llama_proxy.py*" -or $_.CommandLine -like "*aa_ceval*" } |
        Stop-Process -Force
    Start-Sleep -Seconds 2
}

function Invoke-CeValBench([string]$gnn) {
    $log = "build\tmp\aa_${gnn}.log"
    Write-Host ""
    Write-Host "============================================"
    Write-Host " AA C-Eval benchmark - GNN $gnn (AI_DISABLE_GNN_MODULE = $(if ($gnn -eq 'off') { '1' } else { '0/unset' }))"
    Write-Host "============================================"
    Stop-StaleBackend
    & cmd /c "Python314\python.exe test-tools\aa_ceval_api_bench.py --gnn $gnn --per-subject $PerSubject" | Tee-Object -FilePath $log
    return $log
}

$onLog = Invoke-CeValBench -gnn "on"
$offLog = Invoke-CeValBench -gnn "off"

Write-Host ""
Write-Host "============================================"
Write-Host " GNN comparison logs"
Write-Host "============================================"
Write-Host "  GNN ON : $onLog"
Write-Host "  GNN OFF: $offLog"

Pop-Location
