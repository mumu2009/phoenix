$ErrorActionPreference = "Stop"
$root = "d:\_phoenix\_079\v6.0Alixander\v6.0Alixander"

$simArgs = @(
    "-u", "tools\auto_tune_phoenix_params.py",
    "--tune-mode", "sim",
    "--vary-key", "context.similarityThreshold",
    "--vary-values", "0.54,0.57,0.60,0.63,0.66,0.69,0.72",
    "--samples", "10",
    "--max-evals", "3",
    "--run-timeout", "7200",
    "--latency-budget", "300000"
)
$sim = Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList $simArgs -WorkingDirectory $root -RedirectStandardOutput (Join-Path $root "build\auto_tune_sim.log") -RedirectStandardError (Join-Path $root "build\auto_tune_sim.err") -PassThru
$sim.WaitForExit()
if ($sim.ExitCode -ne 0) {
    Write-Error "similarityThreshold tuning failed with exit code $($sim.ExitCode)"
    exit $sim.ExitCode
}

$sim_best = (Get-Content (Join-Path $root "config\phoenix.json") | ConvertFrom-Json).context.similarityThreshold

$otherArgs = @(
    "-u", "tools\auto_tune_phoenix_params.py",
    "--tune-mode", "other",
    "--fix-similarity", $sim_best,
    "--samples", "10",
    "--max-evals", "5",
    "--run-timeout", "7200",
    "--latency-budget", "300000"
)
$other = Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList $otherArgs -WorkingDirectory $root -RedirectStandardOutput (Join-Path $root "build\auto_tune_other.log") -RedirectStandardError (Join-Path $root "build\auto_tune_other.err") -PassThru
$other.WaitForExit()
if ($other.ExitCode -ne 0) {
    Write-Error "other tuning failed with exit code $($other.ExitCode)"
    exit $other.ExitCode
}
