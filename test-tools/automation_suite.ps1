param(
    [string]$BaseUrl = "http://127.0.0.1:5080",
    [string]$AuthBaseUrl = "http://127.0.0.1:5081",
    [string]$RegressionOllamaModel = "llama3.1:8b",
    [string]$IntelligenceOllamaModel = "llama3.1:8b",
    [switch]$SkipUnit,
    [switch]$SkipFrontend,
    [switch]$SkipIntelligence,
    [switch]$SkipHai,
    [switch]$SkipPerf,
    [int]$IntelligenceMaxTokens = 192,
    [int]$IntelligenceTimeoutSec = 120,
    [double]$IntelligenceMinScoreAvg = 0,
    [double]$HaiMinScore = 12,
    [double]$HaiMinCoverage = 60,
    [int]$PerfRounds = 1,
    [int]$PerfQualityQuestionnaireLimit = 20,
    [int]$PerfTimeoutSec = 120,
    [int]$PerfQuestionnaireLimit = 2
)

$ErrorActionPreference = "Stop"

$report = [ordered]@{
    startedAt = (Get-Date).ToString("s")
    compile = @{ ok = $false; detail = "" }
    backendRegression = @{ ok = $false; detail = "" }
    unit = @{ ok = $false; detail = "skipped" }
    frontend = @{ ok = $false; detail = "skipped" }
    intelligence = @{ ok = $false; detail = "skipped" }
    hai = @{ ok = $false; detail = "skipped" }
    perf = @{ ok = $false; detail = "skipped" }
}

$serverProc = $null
$authProc = $null
$workspaceTemp = Join-Path (Resolve-Path ".\runtime_store").Path "tmp"
$workspaceNpmCache = Join-Path (Resolve-Path ".\runtime_store").Path "npm-cache"

if (-not (Test-Path $workspaceTemp)) {
    New-Item -ItemType Directory -Path $workspaceTemp -Force | Out-Null
}
if (-not (Test-Path $workspaceNpmCache)) {
    New-Item -ItemType Directory -Path $workspaceNpmCache -Force | Out-Null
}

$env:TEMP = $workspaceTemp
$env:TMP = $workspaceTemp
$env:NPM_CONFIG_CACHE = $workspaceNpmCache
$pythonExe = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

function Wait-HttpReady {
    param(
        [string]$Url,
        [int]$TimeoutSec = 20
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            Invoke-WebRequest -Uri $Url -TimeoutSec 5 | Out-Null
            return
        }
        catch {
            if ($_.Exception.Response) {
                return
            }
            Start-Sleep -Milliseconds 500
        }
    }

    throw "service not ready: $Url"
}

function Start-BackendForTest {
    param(
        [string]$BaseUrl,
        [string]$OllamaModel
    )
    $uri = [Uri]$BaseUrl
    $bindHost = if ([string]::IsNullOrWhiteSpace($uri.Host)) { "127.0.0.1" } else { $uri.Host }
    $port = if ($uri.Port -gt 0) { $uri.Port } else { 5080 }
    $studyPort = if ($port -eq 5081) { 5082 } else { 5081 }
    $args = @(
        "--gateway-host=$bindHost",
        "--port=$port",
        "--study-port=$studyPort",
        "--base-dir=runtime_store",
        "--db-path=runtime_store/ai_store.sqlite",
        "--lmdb-dir=lmdb",
        "--robots-dir=robots",
        "--redis-url=redis://127.0.0.1:6379",
        "--log-mode=release",
        "--tests-autoload=true",
        "--robots-limit=10000000",
        "--robots-autoload=true",
        "--tests-dir=tests"
    )

    if (-not (Test-Path ".\runtime_store\auth")) {
        New-Item -ItemType Directory -Path ".\runtime_store\auth" -Force | Out-Null
    }
    $env:AUTH_DB = "runtime_store/auth/users.test.json"
    $env:AUTH_OUTBOX_DIR = "runtime_store/auth/outbox"
    $env:AUTH_ALLOW_REGISTER = "true"
    $env:AUTH_REQUIRE_EMAIL_VERIFY = "false"
    $env:AUTH_LOCAL_TOKEN_FALLBACK = "true"
    $env:AUTH_DEV_RETURN_TOKEN = "true"
    $env:AI_OLLAMA_TIMEOUT_MS = "360000"
    if (-not [string]::IsNullOrWhiteSpace($OllamaModel)) {
        $env:AI_OLLAMA_MODEL = $OllamaModel
    }

    $proc = Start-Process -FilePath ".\phoenix_main.exe" -ArgumentList $args -PassThru -WindowStyle Hidden
    Wait-HttpReady -Url "$BaseUrl/api/model/lifecycle"
    return $proc
}

function Set-BackendOllamaModel {
    param(
        [string]$BaseUrl,
        [string]$OllamaModel,
        [int]$TimeoutSec = 480
    )

    if ([string]::IsNullOrWhiteSpace($OllamaModel)) {
        return
    }

    $headers = @{ Authorization = "Bearer local-dev" }
    $current = Invoke-RestMethod -Method Get -Uri "$BaseUrl/api/runtime/features" -Headers $headers -TimeoutSec 30
    $currentModel = ""
    if ($current -and $current.features -and $current.features.pipeline) {
        $currentModel = [string]$current.features.pipeline.ollamaModel
    }
    if ($currentModel -eq $OllamaModel) {
        return
    }

    $body = @{ usingOllama = $true; transformerMode = "ollama"; ollamaModel = $OllamaModel } | ConvertTo-Json -Depth 8 -Compress
    $resp = Invoke-RestMethod -Method Patch -Uri "$BaseUrl/api/runtime/features" -Headers $headers -ContentType "application/json" -Body $body -TimeoutSec 30
    if (-not $resp.ok) {
        throw "runtime feature patch failed: $($resp | ConvertTo-Json -Depth 8)"
    }

    $warmupBody = @{ text = "warmup"; maxTokens = 8 } | ConvertTo-Json -Depth 4 -Compress
    Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/chat" -Headers $headers -ContentType "application/json" -Body $warmupBody -TimeoutSec $TimeoutSec | Out-Null
}

function Start-AuthFrontendForTest {
    param([string]$AuthBaseUrl)

    $uri = [Uri]$AuthBaseUrl
    $bindHost = if ([string]::IsNullOrWhiteSpace($uri.Host)) { "127.0.0.1" } else { $uri.Host }
    $port = if ($uri.Port -gt 0) { $uri.Port } else { 5081 }
    $args = @(
        "--frontend-only=1",
        "--frontend-host=$bindHost",
        "--frontend-port=$port",
        "--study-port=$port",
        "--base-dir=runtime_store",
        "--robots-dir=robots",
        "--tests-dir=tests"
    )

    $proc = Start-Process -FilePath ".\phoenix_main.exe" -ArgumentList $args -PassThru -WindowStyle Hidden
    Wait-HttpReady -Url "$AuthBaseUrl/auth/config"
    return $proc
}

try {
    Write-Host "[STEP] compile"
    & ./compile.bat | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "compile failed" }
    $report.compile.ok = $true
    $report.compile.detail = "compile.bat passed"

    Write-Host "[STEP] backend api regression"
    $serverProc = Start-BackendForTest -BaseUrl $BaseUrl -OllamaModel $RegressionOllamaModel
    $authProc = Start-AuthFrontendForTest -AuthBaseUrl $AuthBaseUrl
    & ./test-tools/api_regression.ps1 -BaseUrl $BaseUrl -AuthBaseUrl $AuthBaseUrl -RequestTimeoutSec 90 -StepTimeoutSec 180 -WarmupTimeoutSec 240 | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "api regression failed" }
    $report.backendRegression.ok = $true
    $report.backendRegression.detail = "api_regression.ps1 passed"

    if ($RegressionOllamaModel -ne $IntelligenceOllamaModel) {
        Write-Host "[STEP] switch backend intelligence model"
        Set-BackendOllamaModel -BaseUrl $BaseUrl -OllamaModel $IntelligenceOllamaModel -TimeoutSec 480
    }
    else {
        Write-Host "[STEP] backend intelligence model unchanged"
    }

    if (-not $SkipUnit) {
        Write-Host "[STEP] python unit tests"
        & $pythonExe -m unittest discover -s test -p "test_*.py" | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "python unit tests failed" }
        $report.unit.ok = $true
        $report.unit.detail = "python unittest passed"
    }

    if (-not $SkipFrontend) {
        Write-Host "[STEP] frontend tests"
        Push-Location ./079project_frontend
        try {
            npm test -- --watchAll=false | Out-Host
            if ($LASTEXITCODE -ne 0) { throw "frontend test failed" }
            $report.frontend.ok = $true
            $report.frontend.detail = "npm test passed"
        }
        finally {
            Pop-Location
        }
    }

    if (-not $SkipIntelligence) {
        Write-Host "[STEP] intelligence evaluation"
        & $pythonExe test/intelligence/main.py --system-url "$BaseUrl/api/chat" --system-token local-dev --ollama-url "http://127.0.0.1:11434/api/chat" --ollama-model "$IntelligenceOllamaModel" --cases-file test/intelligence/cases.quick.json --output-json build/intelligence_eval_report.json --output-md build/intelligence_eval_report.md --timeout "$IntelligenceTimeoutSec" --max-tokens "$IntelligenceMaxTokens" --min-score-avg "$IntelligenceMinScoreAvg" | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "intelligence evaluation failed" }
        $report.intelligence.ok = $true
        $report.intelligence.detail = "test/intelligence/main.py quick suite passed"
    }

    if (-not $SkipHai) {
        Write-Host "[STEP] HAI evaluation"
        & $pythonExe test/intelligence/main.py --system-url "$BaseUrl/api/chat" --system-token local-dev --ollama-url "http://127.0.0.1:11434/api/chat" --ollama-model "$IntelligenceOllamaModel" --cases-file test/intelligence/cases.baseline.json --output-json build/hai_eval_report.json --output-md build/hai_eval_report.md --timeout "$IntelligenceTimeoutSec" --max-tokens "$IntelligenceMaxTokens" --min-hai-score "$HaiMinScore" --min-hai-coverage "$HaiMinCoverage" | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "HAI evaluation failed" }
        $report.hai.ok = $true
        $report.hai.detail = "test/intelligence/main.py HAI suite passed"
    }

    if (-not $SkipPerf) {
        Write-Host "[STEP] performance benchmark"
        $questionnaireFiles = @()
        if (Test-Path ".\questionaire.txt") {
            $questionnaireFiles += (Resolve-Path ".\questionaire.txt").Path
        }

        $perfArgs = @(
            "test/prof/main.py",
            "--system-url", "$BaseUrl/api/chat",
            "--ollama-model", "$IntelligenceOllamaModel",
            "--preferred-models", "$IntelligenceOllamaModel,llama3.1:latest,tinyllama:latest,qwen2.5:7b,qwen2.5:14b,gpt-oss:20b",
            "--rounds", "$PerfRounds",
            "--concurrency", "1",
            "--timeout", "$PerfTimeoutSec",
            "--checkpoint-every", "100",
            "--no-standard-benchmarks",
            "--no-auto-discover-tests-datasets",
            "--no-auto-manage-ollama",
            "--external-limit", "$PerfQualityQuestionnaireLimit",
            "--ollama-warmup-timeout", "240",
            "--output", "build/perf_smoke_report.md",
            "--json-output", "build/perf_smoke_report.json"
        )
        if ($questionnaireFiles.Count -gt 0) {
            $perfArgs += "--questionnaire-files"
            $perfArgs += $questionnaireFiles
        }
        $perfArgs += "--questionnaire-limit"
        $perfArgs += "$PerfQuestionnaireLimit"

        & $pythonExe @perfArgs | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "performance benchmark failed" }
        $report.perf.ok = $true
        $report.perf.detail = "test/prof/main.py smoke suite passed"
    }
}
catch {
    $report.error = $_.Exception.Message
}
finally {
    if ($serverProc -and -not $serverProc.HasExited) {
        try { Stop-Process -Id $serverProc.Id -Force -ErrorAction SilentlyContinue } catch {}
    }
    if ($authProc -and -not $authProc.HasExited) {
        try { Stop-Process -Id $authProc.Id -Force -ErrorAction SilentlyContinue } catch {}
    }
    $report.finishedAt = (Get-Date).ToString("s")
    $reportPath = "build/automation_report.json"
    $report | ConvertTo-Json -Depth 8 | Out-File -FilePath $reportPath -Encoding utf8
    Write-Host "[REPORT] $reportPath"
    Write-Host ($report | ConvertTo-Json -Depth 8)
}

if (-not $report.compile.ok -or -not $report.backendRegression.ok) {
    exit 1
}
if (-not $SkipUnit -and -not $report.unit.ok) {
    exit 1
}
if (-not $SkipFrontend -and -not $report.frontend.ok) {
    exit 1
}
if (-not $SkipIntelligence -and -not $report.intelligence.ok) {
    exit 1
}
if (-not $SkipHai -and -not $report.hai.ok) {
    exit 1
}
if (-not $SkipPerf -and -not $report.perf.ok) {
    exit 1
}
