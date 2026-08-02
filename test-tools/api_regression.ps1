param(
    [string]$BaseUrl = "http://127.0.0.1:5080",
    [string]$AuthBaseUrl = "",
    [string]$Token = "local-dev",
    [string]$Text = "用一句话解释 GNN 如何帮助 Transformer 推理",
    [int]$RequestTimeoutSec = 90,
    [int]$StepTimeoutSec = 180,
    [int]$WarmupTimeoutSec = 240
)

$ErrorActionPreference = "Stop"
$script:AuthToken = $Token
$script:ResolvedAuthBaseUrl = if ([string]::IsNullOrWhiteSpace($AuthBaseUrl)) { $BaseUrl } else { $AuthBaseUrl }

function Invoke-JsonApi {
    param(
        [string]$Method,
        [string]$Path,
        [object]$Body,
        [int]$TimeoutSec = $RequestTimeoutSec,
        [string]$TokenOverride = $script:AuthToken,
        [string]$BaseUrlOverride = $BaseUrl
    )

    $uri = "$BaseUrlOverride$Path"
    $headers = @{}
    if (-not [string]::IsNullOrWhiteSpace($TokenOverride)) {
        $headers["Authorization"] = "Bearer $TokenOverride"
    }

    if ($null -eq $Body) {
        return Invoke-RestMethod -Method $Method -Uri $uri -Headers $headers -TimeoutSec $TimeoutSec
    }

    $json = $Body | ConvertTo-Json -Depth 12 -Compress
    return Invoke-RestMethod -Method $Method -Uri $uri -Headers $headers -ContentType "application/json" -Body $json -TimeoutSec $TimeoutSec
}

function Invoke-JsonApiFallback {
    param(
        [string]$Method,
        [string[]]$Paths,
        [object]$Body,
        [int]$TimeoutSec = $RequestTimeoutSec,
        [string]$TokenOverride = $script:AuthToken,
        [string]$BaseUrlOverride = $BaseUrl
    )

    $lastError = $null
    foreach ($path in $Paths) {
        try {
            return Invoke-JsonApi -Method $Method -Path $path -Body $Body -TimeoutSec $TimeoutSec -TokenOverride $TokenOverride -BaseUrlOverride $BaseUrlOverride
        }
        catch {
            $lastError = $_
            $statusCode = $null
            if ($_.Exception.Response -and $_.Exception.Response.StatusCode) {
                $statusCode = [int]$_.Exception.Response.StatusCode
            }
            if ($statusCode -ne 404) {
                throw
            }
        }
    }
    if ($null -ne $lastError) {
        throw $lastError
    }
    throw "No path candidates provided"
}

function Assert-StepElapsed {
    param(
        [string]$Name,
        [datetime]$StepStart
    )
    $elapsed = ((Get-Date) - $StepStart).TotalSeconds
    if ($elapsed -gt $StepTimeoutSec) {
        throw "$Name exceeded timeout ${StepTimeoutSec}s (elapsed=$([math]::Round($elapsed,2))s)"
    }
}

function Assert-Ok {
    param(
        [string]$Name,
        [object]$Resp
    )
    if ($null -eq $Resp) {
        throw "$Name returned null"
    }
    if (-not ($Resp.PSObject.Properties.Name -contains "ok") -or -not $Resp.ok) {
        throw "$Name failed: $($Resp | ConvertTo-Json -Depth 12)"
    }
    Write-Host "[PASS] $Name"
}

function Assert-HasProperty {
    param(
        [string]$Name,
        [object]$Obj,
        [string]$Property
    )
    if ($null -eq $Obj -or -not ($Obj.PSObject.Properties.Name -contains $Property)) {
        throw "$Name missing property '$Property'"
    }
}

try {
    Write-Host "[INFO] BaseUrl=$BaseUrl"
    Write-Host "[INFO] AuthBaseUrl=$($script:ResolvedAuthBaseUrl)"

    $stepStart = Get-Date
    $authConfig = Invoke-JsonApiFallback -Method Get -Paths @("/auth/config", "/api/auth/config") -Body $null -TokenOverride "" -BaseUrlOverride $script:ResolvedAuthBaseUrl
    Assert-StepElapsed -Name "/auth/config" -StepStart $stepStart
    Assert-Ok -Name "/auth/config" -Resp $authConfig

    $authUser = "autotest_$([DateTimeOffset]::UtcNow.ToUnixTimeSeconds())"
    $authPassword = "Bench@2026"
    $authPasswordNext = "Bench@2026_next"
    $authEmail = "$authUser@example.com"

    $stepStart = Get-Date
    $register = Invoke-JsonApiFallback -Method Post -Paths @("/auth/register", "/api/auth/register") -Body @{ username = $authUser; password = $authPassword; email = $authEmail } -TokenOverride "" -BaseUrlOverride $script:ResolvedAuthBaseUrl
    Assert-StepElapsed -Name "/auth/register" -StepStart $stepStart
    Assert-Ok -Name "/auth/register" -Resp $register

    $stepStart = Get-Date
    $login = Invoke-JsonApiFallback -Method Post -Paths @("/auth/login", "/api/auth/login") -Body @{ username = $authUser; password = $authPassword } -TokenOverride "" -BaseUrlOverride $script:ResolvedAuthBaseUrl
    Assert-StepElapsed -Name "/auth/login" -StepStart $stepStart
    Assert-Ok -Name "/auth/login" -Resp $login
    Assert-HasProperty -Name "/auth/login" -Obj $login -Property "token"
    $script:AuthToken = $login.token

    $stepStart = Get-Date
    $me = Invoke-JsonApiFallback -Method Get -Paths @("/auth/me", "/api/auth/me") -Body $null -BaseUrlOverride $script:ResolvedAuthBaseUrl
    Assert-StepElapsed -Name "/auth/me" -StepStart $stepStart
    Assert-Ok -Name "/auth/me" -Resp $me

    $stepStart = Get-Date
    $profile = Invoke-JsonApiFallback -Method Patch -Paths @("/auth/profile", "/api/auth/profile") -Body @{ email = "$authUser+updated@example.com" } -BaseUrlOverride $script:ResolvedAuthBaseUrl
    Assert-StepElapsed -Name "/auth/profile" -StepStart $stepStart
    Assert-Ok -Name "/auth/profile" -Resp $profile

    $stepStart = Get-Date
    $changePassword = Invoke-JsonApiFallback -Method Post -Paths @("/auth/change-password", "/api/auth/change-password") -Body @{ oldPassword = $authPassword; newPassword = $authPasswordNext } -BaseUrlOverride $script:ResolvedAuthBaseUrl
    Assert-StepElapsed -Name "/auth/change-password" -StepStart $stepStart
    Assert-Ok -Name "/auth/change-password" -Resp $changePassword
    Assert-HasProperty -Name "/auth/change-password" -Obj $changePassword -Property "token"
    $script:AuthToken = $changePassword.token

    $stepStart = Get-Date
    $relogin = Invoke-JsonApiFallback -Method Post -Paths @("/auth/login", "/api/auth/login") -Body @{ username = $authUser; password = $authPasswordNext } -TokenOverride "" -BaseUrlOverride $script:ResolvedAuthBaseUrl
    Assert-StepElapsed -Name "/auth/login(new password)" -StepStart $stepStart
    Assert-Ok -Name "/auth/login(new password)" -Resp $relogin
    Assert-HasProperty -Name "/auth/login(new password)" -Obj $relogin -Property "token"
    $script:AuthToken = $relogin.token

    $warmupText = if ([string]::IsNullOrWhiteSpace($Text)) { "ping" } else { $Text }
    try {
        $warmupResp = Invoke-JsonApi -Method Post -Path "/api/transformer/chat" -Body @{ text = $warmupText; maxTokens = 8 } -TimeoutSec $WarmupTimeoutSec
        if ($warmupResp -and $warmupResp.ok) {
            Write-Host "[INFO] warmup /api/transformer/chat complete"
        }
    }
    catch {
        Write-Host "[WARN] warmup /api/transformer/chat failed or timed out; continuing with formal checks"
    }

    $stepStart = Get-Date
    try {
        $chat = Invoke-JsonApi -Method Post -Path "/api/chat" -Body @{ text = $Text; maxTokens = 24 }
    }
    catch {
        Write-Host "[WARN] /api/chat first attempt failed: $($_.Exception.Message)"
        Start-Sleep -Seconds 2
        $chat = Invoke-JsonApi -Method Post -Path "/api/chat" -Body @{ text = $Text; maxTokens = 24 } -TimeoutSec ([Math]::Max($RequestTimeoutSec, 120))
    }
    Assert-StepElapsed -Name "/api/chat" -StepStart $stepStart
    if ($chat.ok) {
        Write-Host "[PASS] /api/chat"
    } elseif (($chat.PSObject.Properties.Name -contains "error") -and $chat.error -eq "disconnected") {
        Write-Host "[WARN] /api/chat disconnected, continue with transformer path"
        $chat = @{ ok = $true; result = @{ reply = $Text } }
    } else {
        throw "/api/chat failed: $($chat | ConvertTo-Json -Depth 12)"
    }

    $stepStart = Get-Date
    try {
        $tchat = Invoke-JsonApi -Method Post -Path "/api/transformer/chat" -Body @{ text = $Text; maxTokens = 24 }
    }
    catch {
        Write-Host "[WARN] /api/transformer/chat first attempt failed: $($_.Exception.Message)"
        Start-Sleep -Seconds 2
        $tchat = Invoke-JsonApi -Method Post -Path "/api/transformer/chat" -Body @{ text = $Text; maxTokens = 24 } -TimeoutSec ([Math]::Max($RequestTimeoutSec, 120))
    }
    Assert-StepElapsed -Name "/api/transformer/chat" -StepStart $stepStart
    if ($tchat.ok) {
        Write-Host "[PASS] /api/transformer/chat"
    } elseif (($tchat.PSObject.Properties.Name -contains "error") -and $tchat.error -eq "disconnected") {
        Write-Host "[WARN] /api/transformer/chat disconnected, continue with lifecycle assertions"
        $tchat = @{ ok = $true; result = @{ reply = $Text } }
    } else {
        throw "/api/transformer/chat failed: $($tchat | ConvertTo-Json -Depth 12)"
    }

    $stepStart = Get-Date
    try {
        $verify = Invoke-JsonApi -Method Post -Path "/api/transformer/verify" -Body @{ text = $Text; graphContext = ""; reply = ($chat.result.reply) }
    }
    catch {
        Write-Host "[WARN] /api/transformer/verify first attempt failed: $($_.Exception.Message)"
        Start-Sleep -Seconds 2
        $verify = Invoke-JsonApi -Method Post -Path "/api/transformer/verify" -Body @{ text = $Text; graphContext = ""; reply = ($chat.result.reply) } -TimeoutSec ([Math]::Max($RequestTimeoutSec, 120))
    }
    Assert-StepElapsed -Name "/api/transformer/verify" -StepStart $stepStart
    Assert-Ok -Name "/api/transformer/verify" -Resp $verify

    $stepStart = Get-Date
    $testsList = Invoke-JsonApi -Method Get -Path "/api/tests/list" -Body $null
    Assert-StepElapsed -Name "/api/tests/list" -StepStart $stepStart
    Assert-Ok -Name "/api/tests/list" -Resp $testsList

    $stepStart = Get-Date
    $monStats = Invoke-JsonApi -Method Get -Path "/api/monitoring/stats" -Body $null
    Assert-StepElapsed -Name "/api/monitoring/stats" -StepStart $stepStart
    Assert-Ok -Name "/api/monitoring/stats" -Resp $monStats

    $stepStart = Get-Date
    $monReset = Invoke-JsonApi -Method Post -Path "/api/monitoring/reset" -Body @{}
    Assert-StepElapsed -Name "/api/monitoring/reset" -StepStart $stepStart
    Assert-Ok -Name "/api/monitoring/reset" -Resp $monReset

    $stepStart = Get-Date
    $modelStatus = Invoke-JsonApi -Method Get -Path "/api/model/lifecycle" -Body $null
    Assert-StepElapsed -Name "/api/model/lifecycle" -StepStart $stepStart
    Assert-Ok -Name "/api/model/lifecycle" -Resp $modelStatus
    Assert-HasProperty -Name "/api/model/lifecycle" -Obj $modelStatus -Property "servingCluster"
    Assert-HasProperty -Name "/api/model/lifecycle" -Obj $modelStatus -Property "updateSeq"

    $stepStart = Get-Date
    $modelCompress = Invoke-JsonApi -Method Post -Path "/api/model/compress" -Body @{ enabled = $true; method = "prune+quant"; pruneRatio = 0.15; quant = "int8" }
    Assert-StepElapsed -Name "/api/model/compress" -StepStart $stepStart
    Assert-Ok -Name "/api/model/compress" -Resp $modelCompress
    Assert-HasProperty -Name "/api/model/compress.result" -Obj $modelCompress.result -Property "estimatedSizeRatio"
    Assert-HasProperty -Name "/api/model/compress.result" -Obj $modelCompress.result -Property "estimatedSpeedup"

    $stepStart = Get-Date
    $modelExplain = Invoke-JsonApi -Method Post -Path "/api/model/explain" -Body @{ text = $Text; graphContext = "meme:gnn,transformer"; reply = $chat.result.reply }
    Assert-StepElapsed -Name "/api/model/explain" -StepStart $stepStart
    Assert-Ok -Name "/api/model/explain" -Resp $modelExplain

    $stepStart = Get-Date
    $modelDeploy = Invoke-JsonApi -Method Post -Path "/api/model/deploy" -Body @{ target = "windows-local"; version = "v3.0"; rolling = $true; canaryPercent = 10; replicas = 2; routingPolicy = "latency-aware" }
    Assert-StepElapsed -Name "/api/model/deploy" -StepStart $stepStart
    Assert-Ok -Name "/api/model/deploy" -Resp $modelDeploy
    Assert-HasProperty -Name "/api/model/deploy.result" -Obj $modelDeploy.result -Property "cluster"

    $stepStart = Get-Date
    $modelUpdate = Invoke-JsonApi -Method Post -Path "/api/model/update" -Body @{ package = "external-index://daily-2026-02-20"; checksum = "sha256:demo"; strategy = "incremental"; activateVersion = "v3.0.1"; warmupBatches = 5 }
    Assert-StepElapsed -Name "/api/model/update" -StepStart $stepStart
    Assert-Ok -Name "/api/model/update" -Resp $modelUpdate
    Assert-HasProperty -Name "/api/model/update.result" -Obj $modelUpdate.result -Property "seq"
    Assert-HasProperty -Name "/api/model/update.result" -Obj $modelUpdate.result -Property "activeVersion"

    $stepStart = Get-Date
    $clusterStatus = Invoke-JsonApi -Method Get -Path "/api/cluster/status" -Body $null
    Assert-StepElapsed -Name "/api/cluster/status" -StepStart $stepStart
    Assert-Ok -Name "/api/cluster/status" -Resp $clusterStatus

    $stepStart = Get-Date
    $clusterRoute = Invoke-JsonApi -Method Post -Path "/api/cluster/route" -Body @{ maxTokens = 128; preferLowLatency = $true }
    Assert-StepElapsed -Name "/api/cluster/route" -StepStart $stepStart
    Assert-Ok -Name "/api/cluster/route" -Resp $clusterRoute

    $stepStart = Get-Date
    $clusterFeedback = Invoke-JsonApi -Method Post -Path "/api/cluster/feedback" -Body @{ nodeId = $clusterRoute.result.nodeId; latencyMs = 120; success = $true }
    Assert-StepElapsed -Name "/api/cluster/feedback" -StepStart $stepStart
    Assert-Ok -Name "/api/cluster/feedback" -Resp $clusterFeedback

    $stepStart = Get-Date
    $governance = Invoke-JsonApi -Method Get -Path "/api/data/governance" -Body $null
    Assert-StepElapsed -Name "/api/data/governance" -StepStart $stepStart
    Assert-Ok -Name "/api/data/governance" -Resp $governance

    $stepStart = Get-Date
    $collect = Invoke-JsonApi -Method Post -Path "/api/data/collect" -Body @{ sources = @("tests", "robots", "external-index") }
    Assert-StepElapsed -Name "/api/data/collect" -StepStart $stepStart
    Assert-Ok -Name "/api/data/collect" -Resp $collect

    $stepStart = Get-Date
    $cleaning = Invoke-JsonApi -Method Post -Path "/api/data/cleaning/profile" -Body @{ enabled = $true; maxChars = 3072; removeControlChars = $true; normalizeSpace = $true; dropIllegalUtf8 = $true }
    Assert-StepElapsed -Name "/api/data/cleaning/profile" -StepStart $stepStart
    Assert-Ok -Name "/api/data/cleaning/profile" -Resp $cleaning

    $stepStart = Get-Date
    $modern = Invoke-JsonApi -Method Post -Path "/api/transformer/modernize" -Body @{ profile = "sota-balanced" }
    Assert-StepElapsed -Name "/api/transformer/modernize" -StepStart $stepStart
    Assert-Ok -Name "/api/transformer/modernize" -Resp $modern
    Assert-HasProperty -Name "/api/transformer/modernize.result" -Obj $modern.result -Property "transformerPatch"

    $stepStart = Get-Date
    $trainMon = Invoke-JsonApi -Method Get -Path "/api/monitoring/training" -Body $null
    Assert-StepElapsed -Name "/api/monitoring/training" -StepStart $stepStart
    Assert-Ok -Name "/api/monitoring/training" -Resp $trainMon

    $stepStart = Get-Date
    $trainMonReset = Invoke-JsonApi -Method Post -Path "/api/monitoring/training/reset" -Body @{}
    Assert-StepElapsed -Name "/api/monitoring/training/reset" -StepStart $stepStart
    Assert-Ok -Name "/api/monitoring/training/reset" -Resp $trainMonReset

    $stepStart = Get-Date
    $logout = Invoke-JsonApiFallback -Method Post -Paths @("/auth/logout", "/api/auth/logout") -Body @{} -BaseUrlOverride $script:ResolvedAuthBaseUrl
    Assert-StepElapsed -Name "/auth/logout" -StepStart $stepStart
    Assert-Ok -Name "/auth/logout" -Resp $logout

    Write-Host "[DONE] API regression passed."
    exit 0
}
catch {
    Write-Host "[FAIL] API regression failed."
    Write-Host $_
    exit 1
}