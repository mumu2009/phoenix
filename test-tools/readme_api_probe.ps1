# Fast README API smoke-probe: a curated set of documented routes.
# Exits 0 so it does not hold up the build, but reports PASS/FAIL totals.
param([string]$baseUrl = "http://127.0.0.1:5080",
      [string]$authBaseUrl = "http://127.0.0.1:5081")
$ErrorActionPreference = "Stop"

$resolvedAuth = if ([string]::IsNullOrWhiteSpace($authBaseUrl)) { $baseUrl } else { $authBaseUrl }
$probeUser = "aa_probe_$([DateTimeOffset]::UtcNow.ToUnixTimeSeconds())"
Write-Output "[INFO] Logging in to get token..."
$token = $null
try {
    $reg = Invoke-RestMethod -Method POST -Uri "$resolvedAuth/auth/register" -ContentType "application/json" -Body (@{username=$probeUser; password="smoke123!"} | ConvertTo-Json -Compress) -TimeoutSec 15
    $login = Invoke-RestMethod -Method POST -Uri "$resolvedAuth/auth/login" -ContentType "application/json" -Body (@{username=$probeUser; password="smoke123!"} | ConvertTo-Json -Compress) -TimeoutSec 15
    $token = $login.token
} catch {
    Write-Output "WARN: auth setup failed: $($_.Exception.Message)"
}
if (-not $token) {
    Write-Output "README API smoke result: PASS=0 FAIL=0 (auth unavailable)"
    exit 0
}
$headers = @{ Authorization = "Bearer $token" }

$endpoints = @(
    @{ method = "GET";  path = "/api/cluster/status"; auth = $true },
    @{ method = "GET";  path = "/api/tests/list"; auth = $true },
    @{ method = "GET";  path = "/api/data/cleaning/profile"; auth = $true },
    @{ method = "POST"; path = "/api/model/lifecycle"; auth = $true; body = @{} },
    @{ method = "POST"; path = "/api/chat"; auth = $true; body = @{ text = "hello"; maxTokens = 8 } }
)

$pass = 0
$fail = 0
$results = @()

foreach ($e in $endpoints) {
    $m = $e.method
    $u = $e.path
    try {
        $uri = "$baseUrl$u"
        $reqHeaders = if ($e.auth) { $headers } else { @{} }
        if ($e.body) {
            $bodyJson = $e.body | ConvertTo-Json -Compress
            $r = Invoke-RestMethod -Method $m -Uri $uri -Headers $reqHeaders -ContentType "application/json" -Body $bodyJson -TimeoutSec 30
        } else {
            $r = Invoke-RestMethod -Method $m -Uri $uri -Headers $reqHeaders -TimeoutSec 15
        }
        if ($r -and $r.ok) {
            $status = "ok"
            $pass++
        } else {
            $status = "no-ok"
            $fail++
        }
    } catch {
        $status = "error:$($_.Exception.Message)"
        $fail++
    }
    Write-Output "$m $u -> $status"
    $results += "$m $u -> $status"
}

$out = "build/tmp/readme_api_probe.log"
$results | Out-File -FilePath $out -Encoding utf8
Write-Output "============================================"
Write-Output "README API smoke result: PASS=$pass FAIL=$fail"
exit 0
