param(
    [string]$BaseUrl = "http://127.0.0.1:5080",
    [string]$Token = "local-dev"
)

$ErrorActionPreference = "Stop"
$headers = @{ Authorization = "Bearer $Token" }

function PostJson([string]$path, [hashtable]$body) {
    $json = $body | ConvertTo-Json -Depth 12 -Compress
    return Invoke-RestMethod -Method Post -Uri "$BaseUrl$path" -Headers $headers -ContentType "application/json" -Body $json
}

function GetJson([string]$path) {
    return Invoke-RestMethod -Method Get -Uri "$BaseUrl$path" -Headers $headers
}

$results = @()

try {
    $r1 = GetJson "/api/cluster/status"
    $results += @{ name = "/api/cluster/status"; ok = [bool]$r1.ok }

    $r2 = PostJson "/api/cluster/route" @{ maxTokens = 128; preferLowLatency = $true }
    $results += @{ name = "/api/cluster/route"; ok = [bool]$r2.ok }

    $nodeId = if ($r2.result -and $r2.result.nodeId) { [string]$r2.result.nodeId } else { "core-node-1" }
    $r3 = PostJson "/api/cluster/feedback" @{ nodeId = $nodeId; latencyMs = 120; success = $true }
    $results += @{ name = "/api/cluster/feedback"; ok = [bool]$r3.ok }

    $r4 = GetJson "/api/data/governance"
    $results += @{ name = "/api/data/governance"; ok = [bool]$r4.ok }

    $r5 = PostJson "/api/data/collect" @{ sources = @("tests", "robots", "external-index") }
    $results += @{ name = "/api/data/collect"; ok = [bool]$r5.ok }

    $r6 = PostJson "/api/data/cleaning/profile" @{ enabled = $true; maxChars = 3072; removeControlChars = $true; normalizeSpace = $true; dropIllegalUtf8 = $true }
    $results += @{ name = "/api/data/cleaning/profile"; ok = [bool]$r6.ok }

    $r7 = PostJson "/api/transformer/modernize" @{ profile = "sota-balanced" }
    $results += @{ name = "/api/transformer/modernize"; ok = [bool]$r7.ok }

    $r8 = GetJson "/api/provider/capabilities"
    $results += @{ name = "/api/provider/capabilities"; ok = [bool]$r8.ok }

    foreach ($r in $results) {
        if ($r.ok) {
            Write-Host "[PASS] $($r.name)"
        } else {
            Write-Host "[FAIL] $($r.name)"
            exit 1
        }
    }

    Write-Host "[DONE] direct endpoint checks passed"
    exit 0
}
catch {
    Write-Host "[FAIL] direct endpoint checks error"
    Write-Host $_
    exit 1
}
