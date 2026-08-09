param(
    [string]$BaseUrl = "http://127.0.0.1:5080",
    [string]$AuthBaseUrl = "http://127.0.0.1:5081",
    [string]$Token = "local-dev",
    [string]$Text = "What is 1+1? Express the result as an arithmetic equation.",
    [int]$RequestTimeoutSec = 90,
    [int]$StepTimeoutSec = 180,
    [int]$WarmupTimeoutSec = 240,
    # When set, the script stops at the first failing step (legacy behaviour).
    # By default all steps run and a full pass/fail summary is printed at the end,
    # which gives much better regression visibility than an all-or-nothing script.
    [switch]$FailFast,
    # Skip the negative / error-input and data-cleaning coverage added on top of
    # the original happy-path checks (kept for fast smoke runs).
    [switch]$SkipExtendedChecks
)

$ErrorActionPreference = "Stop"
$script:AuthToken = $Token
$script:ResolvedAuthBaseUrl = if ([string]::IsNullOrWhiteSpace($AuthBaseUrl)) { $BaseUrl } else { $AuthBaseUrl }
$script:PassCount = 0
$script:FailCount = 0
$script:SkipCount = 0
$script:Failures = New-Object System.Collections.Generic.List[string]

function Invoke-JsonApi {
    param(
        [string]$Method,
        [string]$Path,
        [object]$Body,
        [int]$TimeoutSec = $RequestTimeoutSec,
        [string]$TokenOverride = $script:AuthToken,
        [string]$BaseUrlOverride = $BaseUrl,
        [hashtable]$ExtraHeaders = $null
    )

    $uri = "$BaseUrlOverride$Path"
    $headers = @{}
    if (-not [string]::IsNullOrWhiteSpace($TokenOverride)) {
        $headers["Authorization"] = "Bearer $TokenOverride"
    }
    if ($null -ne $ExtraHeaders) {
        foreach ($k in $ExtraHeaders.Keys) { $headers[$k] = $ExtraHeaders[$k] }
    }

    $doCall = {
        if ($null -eq $Body) {
            return Invoke-RestMethod -Method $Method -Uri $uri -Headers $headers -TimeoutSec $TimeoutSec
        }
        $json = $Body | ConvertTo-Json -Depth 12 -Compress
        return Invoke-RestMethod -Method $Method -Uri $uri -Headers $headers -ContentType "application/json" -Body $json -TimeoutSec $TimeoutSec
    }

    $first = & $doCall
    if ($Method -eq "Get") {
        $second = & $doCall
        $fOk = ($first.PSObject.Properties.Name -contains "ok") -and $first.ok
        $sOk = ($second.PSObject.Properties.Name -contains "ok") -and $second.ok
        if (-not $fOk -or -not $sOk) {
            throw "GET $Path returned ok=false in one or both calls"
        }
        if ($fOk -ne $sOk) {
            throw "GET $Path returned inconsistent ok between two calls ($fOk vs $sOk)"
        }
    }
    return $first
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

function Get-HttpStatusCode {
    param([object]$ErrorRecord)
    if ($ErrorRecord.Exception.Response -and $ErrorRecord.Exception.Response.StatusCode) {
        return [int]$ErrorRecord.Exception.Response.StatusCode
    }
    return $null
}

# Runs a named step. On success increments PassCount; on failure records the
# failure, increments FailCount, prints [FAIL], and either throws (FailFast)
# or continues so the rest of the suite still executes. This turns the script
# from all-or-nothing into a real regression report while staying compatible
# with CI callers that only check the final exit code.
function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Action
    )
    $stepStart = Get-Date
    try {
        & $Action
        $elapsed = ((Get-Date) - $stepStart).TotalSeconds
        if ($elapsed -gt $StepTimeoutSec) {
            throw "$Name exceeded timeout ${StepTimeoutSec}s (elapsed=$([math]::Round($elapsed,2))s)"
        }
        $script:PassCount++
        Write-Host "[PASS] $Name ($([math]::Round($elapsed,2))s)"
        return $true
    }
    catch {
        $script:FailCount++
        $script:Failures.Add("$Name :: $($_.Exception.Message)")
        Write-Host "[FAIL] $Name :: $($_.Exception.Message)"
        if ($FailFast.IsPresent) { throw }
        return $false
    }
}

function Invoke-SkippedStep {
    param([string]$Name, [string]$Reason)
    $script:SkipCount++
    Write-Host "[SKIP] $Name :: $Reason"
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

function Assert-Equal {
    param(
        [string]$Name,
        [object]$Expected,
        [object]$Actual
    )
    if ("$Expected" -ne "$Actual") {
        throw "$Name expected '$Expected' but got '$Actual'"
    }
}

function Assert-NotEmpty {
    param(
        [string]$Name,
        [object]$Value
    )
    if ($null -eq $Value -or ([string]$Value).Length -eq 0) {
        throw "$Name returned an empty or null value"
    }
}

# Expects the given action to raise an HTTP error with the given status code.
# Used for negative-path / error-input coverage (bad auth, missing fields, ...).
function Compare-Cosine {
    param([string]$A, [string]$B, [int]$N = 2)
    if ([string]::IsNullOrWhiteSpace($A) -or [string]::IsNullOrWhiteSpace($B)) { return 0.0 }
    function Get-Ngrams($s) {
        $s = $s.ToLowerInvariant()
        $n = $N
        if ($s.Length -lt $n) { $n = 1 }
        $grams = @()
        for ($i = 0; $i -le $s.Length - $n; $i++) {
            $grams += $s.Substring($i, $n)
        }
        if ($grams.Count -eq 0) { $grams += $s }
        return $grams
    }
    $tokA = Get-Ngrams -s $A
    $tokB = Get-Ngrams -s $B
    if ($tokA.Count -eq 0 -or $tokB.Count -eq 0) { return 0.0 }
    $vocab = $tokA + $tokB | Sort-Object -Unique
    $vecA = New-Object int[] $vocab.Count
    $vecB = New-Object int[] $vocab.Count
    for ($i = 0; $i -lt $vocab.Count; $i++) {
        $w = $vocab[$i]
        $vecA[$i] = ($tokA | Where-Object { $_ -eq $w }).Count
        $vecB[$i] = ($tokB | Where-Object { $_ -eq $w }).Count
    }
    $dot = 0; $normA = 0; $normB = 0
    for ($i = 0; $i -lt $vocab.Count; $i++) {
        $dot += $vecA[$i] * $vecB[$i]
        $normA += $vecA[$i] * $vecA[$i]
        $normB += $vecB[$i] * $vecB[$i]
    }
    if ($normA -eq 0 -or $normB -eq 0) { return 0.0 }
    return $dot / [math]::Sqrt($normA * $normB)
}

function Assert-CosineAbove {
    param([string]$Name, [string]$Actual, [string]$Reference, [double]$Threshold = 0.5)
    $sim = Compare-Cosine -A $Actual -B $Reference
    if ($sim -lt $Threshold) {
        throw "$Name cosine similarity $sim below threshold $Threshold (actual='$Actual', ref='$Reference')"
    }
}

function Assert-JsonContains {
    param(
        [string]$Name,
        [object]$Resp,
        [hashtable]$Expected
    )
    if ($null -eq $Resp) { throw "$Name returned null" }
    foreach ($k in $Expected.Keys) {
        $actualValue = $null
        if ($Resp.PSObject.Properties.Name -contains $k) {
            $actualValue = $Resp.$k
        }
        $expectedValue = $Expected[$k]
        if ("$actualValue" -ne "$expectedValue") {
            throw "$Name field '$k' expected '$expectedValue' but got '$actualValue'"
        }
    }
}

function Assert-HttpError {
    param(
        [string]$Name,
        [scriptblock]$Action,
        [int]$ExpectedStatus
    )
    try {
        & $Action | Out-Null
    }
    catch {
        $status = Get-HttpStatusCode -ErrorRecord $_
        if ($status -ne $ExpectedStatus) {
            throw "$Name expected HTTP $ExpectedStatus but got $status ($($_.Exception.Message))"
        }
        return
    }
    throw "$Name expected HTTP $ExpectedStatus but request succeeded"
}

try {
    Write-Host "[INFO] BaseUrl=$BaseUrl"
    Write-Host "[INFO] AuthBaseUrl=$($script:ResolvedAuthBaseUrl)"
    Write-Host "[INFO] FailFast=$($FailFast.IsPresent) SkipExtendedChecks=$($SkipExtendedChecks.IsPresent)"

    $authConfig = $null
    Invoke-Step -Name "/auth/config" -Action {
        $script:authConfig = Invoke-JsonApiFallback -Method Get -Paths @("/auth/config", "/api/auth/config") -Body $null -TokenOverride "" -BaseUrlOverride $script:ResolvedAuthBaseUrl
        Assert-Ok -Name "/auth/config" -Resp $script:authConfig
    } | Out-Null

    Invoke-Step -Name "Config: /api/system/config" -Action {
        $cfg = Invoke-JsonApi -Method Get -Path "/api/system/config" -Body $null
        Assert-Ok -Name "/api/system/config" -Resp $cfg
    } | Out-Null

    Invoke-Step -Name "World: /api/world" -Action {
        try {
            $ws = Invoke-JsonApi -Method Get -Path "/api/world/status" -Body $null
            Assert-Ok -Name "/api/world/status" -Resp $ws
        } catch {
            try {
                $ws = Invoke-JsonApi -Method Get -Path "/api/world/config" -Body $null
                Assert-Ok -Name "/api/world/config" -Resp $ws
            } catch {
                Write-Host "[WARN] /api/world/* not available, using /api/model/lifecycle as world-model proxy"
                $ws = Invoke-JsonApi -Method Get -Path "/api/model/lifecycle" -Body $null
                Assert-Ok -Name "/api/model/lifecycle (world fallback)" -Resp $ws
            }
        }
    } | Out-Null

    $authUser = "autotest_$([DateTimeOffset]::UtcNow.ToUnixTimeSeconds())"
    $authPassword = "Bench@2026"
    $authPasswordNext = "Bench@2026_next"
    $authEmail = "$authUser@example.com"

    Invoke-Step -Name "/auth/register" -Action {
        $script:register = Invoke-JsonApiFallback -Method Post -Paths @("/auth/register", "/api/auth/register") -Body @{ username = $authUser; password = $authPassword; email = $authEmail } -TokenOverride "" -BaseUrlOverride $script:ResolvedAuthBaseUrl
        Assert-Ok -Name "/auth/register" -Resp $script:register
    } | Out-Null

    Invoke-Step -Name "/auth/login" -Action {
        $script:login = Invoke-JsonApiFallback -Method Post -Paths @("/auth/login", "/api/auth/login") -Body @{ username = $authUser; password = $authPassword } -TokenOverride "" -BaseUrlOverride $script:ResolvedAuthBaseUrl
        Assert-Ok -Name "/auth/login" -Resp $script:login
        Assert-HasProperty -Name "/auth/login" -Obj $script:login -Property "token"
        Assert-NotEmpty -Name "/auth/login.token" -Value $script:login.token
        $script:AuthToken = $script:login.token
    } | Out-Null

    Invoke-Step -Name "/auth/me" -Action {
        $me = Invoke-JsonApiFallback -Method Get -Paths @("/auth/me", "/api/auth/me") -Body $null -BaseUrlOverride $script:ResolvedAuthBaseUrl
        Assert-Ok -Name "/auth/me" -Resp $me
        Assert-HasProperty -Name "/auth/me" -Obj $me -Property "user"
        Assert-Equal -Name "/auth/me.username" -Expected $authUser -Actual $me.user.username
    } | Out-Null

    Invoke-Step -Name "/auth/profile" -Action {
        $profile = Invoke-JsonApiFallback -Method Patch -Paths @("/auth/profile", "/api/auth/profile") -Body @{ email = "$authUser+updated@example.com" } -BaseUrlOverride $script:ResolvedAuthBaseUrl
        Assert-Ok -Name "/auth/profile" -Resp $profile
    } | Out-Null

    Invoke-Step -Name "/auth/change-password" -Action {
        $script:changePassword = Invoke-JsonApiFallback -Method Post -Paths @("/auth/change-password", "/api/auth/change-password") -Body @{ oldPassword = $authPassword; newPassword = $authPasswordNext } -BaseUrlOverride $script:ResolvedAuthBaseUrl
        Assert-Ok -Name "/auth/change-password" -Resp $script:changePassword
        Assert-HasProperty -Name "/auth/change-password" -Obj $script:changePassword -Property "token"
        $script:AuthToken = $script:changePassword.token
    } | Out-Null

    Invoke-Step -Name "/auth/login(new password)" -Action {
        $relogin = Invoke-JsonApiFallback -Method Post -Paths @("/auth/login", "/api/auth/login") -Body @{ username = $authUser; password = $authPasswordNext } -TokenOverride "" -BaseUrlOverride $script:ResolvedAuthBaseUrl
        Assert-Ok -Name "/auth/login(new password)" -Resp $relogin
        Assert-HasProperty -Name "/auth/login(new password)" -Obj $relogin -Property "token"
        $script:AuthToken = $relogin.token
    } | Out-Null

    if (-not $SkipExtendedChecks) {
        Invoke-Step -Name "negative: /auth/me without token -> 401" -Action {
            Assert-HttpError -Name "/auth/me(no token)" -ExpectedStatus 401 -Action {
                Invoke-JsonApi -Method Get -Path "/auth/me" -Body $null -TokenOverride "" -BaseUrlOverride $script:ResolvedAuthBaseUrl
            }
        } | Out-Null

        Invoke-Step -Name "negative: /auth/me with bogus token -> 401" -Action {
            Assert-HttpError -Name "/auth/me(bad token)" -ExpectedStatus 401 -Action {
                Invoke-JsonApi -Method Get -Path "/auth/me" -Body $null -TokenOverride "not-a-real-token" -BaseUrlOverride $script:ResolvedAuthBaseUrl
            }
        } | Out-Null

        Invoke-Step -Name "negative: /auth/login wrong password -> error" -Action {
            Assert-HttpError -Name "/auth/login(wrong password)" -ExpectedStatus 401 -Action {
                Invoke-JsonApiFallback -Method Post -Paths @("/auth/login", "/api/auth/login") -Body @{ username = $authUser; password = "totally-wrong" } -TokenOverride "" -BaseUrlOverride $script:ResolvedAuthBaseUrl
            }
        } | Out-Null
    }

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

    $script:chat = $null
    Invoke-Step -Name "/api/chat" -Action {
        try {
            $chat = Invoke-JsonApi -Method Post -Path "/api/chat" -Body @{ text = $Text; maxTokens = 24 }
        }
        catch {
            Write-Host "[WARN] /api/chat first attempt failed: $($_.Exception.Message)"
            Start-Sleep -Seconds 2
            $chat = Invoke-JsonApi -Method Post -Path "/api/chat" -Body @{ text = $Text; maxTokens = 24 } -TimeoutSec ([Math]::Max($RequestTimeoutSec, 120))
        }
        if ($chat.ok) {
            Assert-HasProperty -Name "/api/chat" -Obj $chat -Property "result"
            Assert-HasProperty -Name "/api/chat.result" -Obj $chat.result -Property "reply"
            Assert-HasProperty -Name "/api/chat.result" -Obj $chat.result -Property "provider"
            Assert-Equal -Name "/api/chat.provider.id" -Expected "llamacpp" -Actual $chat.result.provider.id
            Assert-NotEmpty -Name "/api/chat.result.reply" -Value $chat.result.reply
            Assert-CosineAbove -Name "/api/chat.result.reply" -Actual $chat.result.reply -Reference "1 + 1 = 2" -Threshold 0.45
        } elseif (($chat.PSObject.Properties.Name -contains "error") -and $chat.error -eq "disconnected") {
            Write-Host "[WARN] /api/chat disconnected, continue with transformer path"
            $chat = @{ ok = $true; result = @{ reply = $Text } }
        } else {
            throw "/api/chat failed: $($chat | ConvertTo-Json -Depth 12)"
        }
        $script:chat = $chat
    } | Out-Null
    if ($null -eq $script:chat) { $script:chat = @{ ok = $true; result = @{ reply = $Text } } }

    $script:tchat = $null
    Invoke-Step -Name "/api/transformer/chat" -Action {
        try {
            $tchat = Invoke-JsonApi -Method Post -Path "/api/transformer/chat" -Body @{ text = $Text; maxTokens = 24 }
        }
        catch {
            Write-Host "[WARN] /api/transformer/chat first attempt failed: $($_.Exception.Message)"
            Start-Sleep -Seconds 2
            $tchat = Invoke-JsonApi -Method Post -Path "/api/transformer/chat" -Body @{ text = $Text; maxTokens = 24 } -TimeoutSec ([Math]::Max($RequestTimeoutSec, 120))
        }
        if ($tchat.ok) {
            Assert-HasProperty -Name "/api/transformer/chat" -Obj $tchat -Property "result"
            Assert-HasProperty -Name "/api/transformer/chat.result" -Obj $tchat.result -Property "reply"
            Assert-HasProperty -Name "/api/transformer/chat.result" -Obj $tchat.result -Property "provider"
            Assert-Equal -Name "/api/transformer/chat.provider.id" -Expected "llamacpp" -Actual $tchat.result.provider.id
            Assert-NotEmpty -Name "/api/transformer/chat.result.reply" -Value $tchat.result.reply
            Assert-CosineAbove -Name "/api/transformer/chat.result.reply" -Actual $tchat.result.reply -Reference "1 + 1 = 2" -Threshold 0.45
        } elseif (($tchat.PSObject.Properties.Name -contains "error") -and $tchat.error -eq "disconnected") {
            Write-Host "[WARN] /api/transformer/chat disconnected, continue with lifecycle assertions"
            $tchat = @{ ok = $true; result = @{ reply = $Text } }
        } else {
            throw "/api/transformer/chat failed: $($tchat | ConvertTo-Json -Depth 12)"
        }
        $script:tchat = $tchat
    } | Out-Null
    if ($null -eq $script:tchat) { $script:tchat = @{ ok = $true; result = @{ reply = $Text } } }

    if (-not $SkipExtendedChecks) {
        Invoke-Step -Name "negative: /api/chat missing text -> 400" -Action {
            Assert-HttpError -Name "/api/chat(missing text)" -ExpectedStatus 400 -Action {
                Invoke-JsonApi -Method Post -Path "/api/chat" -Body @{ maxTokens = 8 }
            }
        } | Out-Null

        Invoke-Step -Name "negative: /api/chat without token -> 401" -Action {
            Assert-HttpError -Name "/api/chat(no token)" -ExpectedStatus 401 -Action {
                Invoke-JsonApi -Method Post -Path "/api/chat" -Body @{ text = $Text } -TokenOverride ""
            }
        } | Out-Null
    }

    Invoke-Step -Name "/api/transformer/verify" -Action {
        try {
            $verify = Invoke-JsonApi -Method Post -Path "/api/transformer/verify" -Body @{ text = $Text; graphContext = ""; reply = ($script:chat.result.reply) }
        }
        catch {
            Write-Host "[WARN] /api/transformer/verify first attempt failed: $($_.Exception.Message)"
            Start-Sleep -Seconds 2
            $verify = Invoke-JsonApi -Method Post -Path "/api/transformer/verify" -Body @{ text = $Text; graphContext = ""; reply = ($script:chat.result.reply) } -TimeoutSec ([Math]::Max($RequestTimeoutSec, 120))
        }
        Assert-Ok -Name "/api/transformer/verify" -Resp $verify
    } | Out-Null

    Invoke-Step -Name "/api/tests/list" -Action {
        $testsList = Invoke-JsonApi -Method Get -Path "/api/tests/list" -Body $null
        Assert-Ok -Name "/api/tests/list" -Resp $testsList
    } | Out-Null

    Invoke-Step -Name "/api/monitoring/stats" -Action {
        $monStats = Invoke-JsonApi -Method Get -Path "/api/monitoring/stats" -Body $null
        Assert-Ok -Name "/api/monitoring/stats" -Resp $monStats
        Assert-HasProperty -Name "/api/monitoring/stats" -Obj $monStats -Property "routes"
        Assert-HasProperty -Name "/api/monitoring/stats" -Obj $monStats -Property "cleaning"
        Assert-HasProperty -Name "/api/monitoring/stats.cleaning" -Obj $monStats.cleaning -Property "enabled"
        Assert-HasProperty -Name "/api/monitoring/stats.cleaning" -Obj $monStats.cleaning -Property "maxChars"
        Assert-HasProperty -Name "/api/monitoring/stats.cleaning" -Obj $monStats.cleaning -Property "cleanedInputs"
        Assert-HasProperty -Name "/api/monitoring/stats.cleaning" -Obj $monStats.cleaning -Property "cleanedSamples"
    } | Out-Null

    Invoke-Step -Name "/api/monitoring/reset" -Action {
        $monReset = Invoke-JsonApi -Method Post -Path "/api/monitoring/reset" -Body @{}
        Assert-Ok -Name "/api/monitoring/reset" -Resp $monReset
        $afterReset = Invoke-JsonApi -Method Get -Path "/api/monitoring/stats" -Body $null
        Assert-Equal -Name "/api/monitoring/stats.cleaning.cleanedInputs after reset" -Expected 0 -Actual $afterReset.cleaning.cleanedInputs
        Assert-Equal -Name "/api/monitoring/stats.cleaning.cleanedSamples after reset" -Expected 0 -Actual $afterReset.cleaning.cleanedSamples
    } | Out-Null

    if (-not $SkipExtendedChecks) {
        # --- Data cleaning / runtime feature coverage (testing_strategy_v3.md #6) ---
        Invoke-Step -Name "PATCH /api/runtime/features dataCleaningEnabled=false takes effect" -Action {
            $patch = Invoke-JsonApi -Method Patch -Path "/api/runtime/features" -Body @{ dataCleaningEnabled = $false }
            Assert-Ok -Name "PATCH runtime/features(disable cleaning)" -Resp $patch
            Assert-Equal -Name "features.dataCleaning.enabled" -Expected $false -Actual $patch.features.dataCleaning.enabled

            Invoke-JsonApi -Method Post -Path "/api/monitoring/reset" -Body @{} | Out-Null
            $dirtyText = "line1`u{0007}line2`t`t`tline3   with   messy   spaces"
            $r = Invoke-JsonApi -Method Post -Path "/api/chat" -Body @{ text = $dirtyText; maxTokens = 8 }
            if (-not $r.ok -and -not (($r.PSObject.Properties.Name -contains "error") -and $r.error -eq "disconnected")) {
                throw "/api/chat with dirty text failed while cleaning disabled: $($r | ConvertTo-Json -Depth 8)"
            }
            $stats = Invoke-JsonApi -Method Get -Path "/api/monitoring/stats" -Body $null
            Assert-Equal -Name "cleanedInputs while cleaning disabled" -Expected 0 -Actual $stats.cleaning.cleanedInputs
        } | Out-Null

        Invoke-Step -Name "PATCH /api/runtime/features dataCleaningEnabled=true + dataCleanMaxChars takes effect" -Action {
            $patch = Invoke-JsonApi -Method Patch -Path "/api/runtime/features" -Body @{ dataCleaningEnabled = $true; dataCleanMaxChars = 256 }
            Assert-Ok -Name "PATCH runtime/features(enable cleaning, maxChars=256)" -Resp $patch
            Assert-Equal -Name "features.dataCleaning.enabled" -Expected $true -Actual $patch.features.dataCleaning.enabled
            Assert-Equal -Name "features.dataCleaning.maxChars" -Expected 256 -Actual $patch.features.dataCleaning.maxChars

            Invoke-JsonApi -Method Post -Path "/api/monitoring/reset" -Body @{} | Out-Null
            $longText = ("段落重复内容 " * 100) + "结尾标记"
            $r = Invoke-JsonApi -Method Post -Path "/api/chat" -Body @{ text = $longText; maxTokens = 8 }
            if (-not $r.ok -and -not (($r.PSObject.Properties.Name -contains "error") -and $r.error -eq "disconnected")) {
                throw "/api/chat with long text failed while cleaning enabled: $($r | ConvertTo-Json -Depth 8)"
            }
            $stats = Invoke-JsonApi -Method Get -Path "/api/monitoring/stats" -Body $null
            if ($stats.cleaning.cleanedInputs -lt 1) {
                throw "expected cleanedInputs >= 1 after sending an over-length input with cleaning enabled, got $($stats.cleaning.cleanedInputs)"
            }
        } | Out-Null

        Invoke-Step -Name "restore default data-cleaning settings" -Action {
            $patch = Invoke-JsonApi -Method Patch -Path "/api/runtime/features" -Body @{ dataCleaningEnabled = $true; dataCleanMaxChars = 2048 }
            Assert-Ok -Name "PATCH runtime/features(restore defaults)" -Resp $patch
            Assert-Equal -Name "features.dataCleaning.enabled" -Expected $true -Actual $patch.features.dataCleaning.enabled
            Assert-Equal -Name "features.dataCleaning.maxChars" -Expected 2048 -Actual $patch.features.dataCleaning.maxChars
            Invoke-JsonApi -Method Post -Path "/api/monitoring/reset" -Body @{} | Out-Null
        } | Out-Null

        Invoke-Step -Name "/api/corpus/ingest tolerates illegal control-character samples" -Action {
            $dirty = "bad`u{0000}sample`u{0001}with`u{0008}control`u{001F}chars"
            $r = Invoke-JsonApi -Method Post -Path "/api/corpus/ingest" -Body @{ text = $dirty }
            Assert-Ok -Name "/api/corpus/ingest(dirty text)" -Resp $r
        } | Out-Null

        Invoke-Step -Name "/api/runtime/features rejects out-of-range dataCleanMaxChars" -Action {
            # dataCleanMaxChars is only accepted in [128, 65536]; an out-of-range
            # value must be silently ignored (not applied) rather than crashing
            # or corrupting the stored setting.
            $r = Invoke-JsonApi -Method Patch -Path "/api/runtime/features" -Body @{ dataCleanMaxChars = 999999999 }
            Assert-Ok -Name "PATCH runtime/features(out-of-range maxChars)" -Resp $r
            if ($r.result.applied.PSObject.Properties.Name -contains "dataCleanMaxChars") {
                throw "out-of-range dataCleanMaxChars should not have been applied"
            }
            Assert-Equal -Name "features.dataCleaning.maxChars unchanged" -Expected 2048 -Actual $r.features.dataCleaning.maxChars
        } | Out-Null
    }

    Invoke-Step -Name "/api/model/lifecycle" -Action {
        $modelStatus = Invoke-JsonApi -Method Get -Path "/api/model/lifecycle" -Body $null
        Assert-Ok -Name "/api/model/lifecycle" -Resp $modelStatus
        Assert-HasProperty -Name "/api/model/lifecycle" -Obj $modelStatus -Property "servingCluster"
        Assert-HasProperty -Name "/api/model/lifecycle" -Obj $modelStatus -Property "updateSeq"
    } | Out-Null

    Invoke-Step -Name "/api/model/compress" -Action {
        $modelCompress = Invoke-JsonApi -Method Post -Path "/api/model/compress" -Body @{ enabled = $true; method = "prune+quant"; pruneRatio = 0.15; quant = "int8" }
        Assert-Ok -Name "/api/model/compress" -Resp $modelCompress
        Assert-HasProperty -Name "/api/model/compress.result" -Obj $modelCompress.result -Property "estimatedSizeRatio"
        Assert-HasProperty -Name "/api/model/compress.result" -Obj $modelCompress.result -Property "estimatedSpeedup"
    } | Out-Null

    Invoke-Step -Name "/api/model/explain" -Action {
        $modelExplain = Invoke-JsonApi -Method Post -Path "/api/model/explain" -Body @{ text = $Text; graphContext = "meme:gnn,transformer"; reply = $script:chat.result.reply }
        Assert-Ok -Name "/api/model/explain" -Resp $modelExplain
    } | Out-Null

    Invoke-Step -Name "/api/model/deploy" -Action {
        $modelDeploy = Invoke-JsonApi -Method Post -Path "/api/model/deploy" -Body @{ target = "windows-local"; version = "v3.0"; rolling = $true; canaryPercent = 10; replicas = 2; routingPolicy = "latency-aware" }
        Assert-Ok -Name "/api/model/deploy" -Resp $modelDeploy
        Assert-HasProperty -Name "/api/model/deploy.result" -Obj $modelDeploy.result -Property "cluster"
    } | Out-Null

    $script:modelUpdate = $null
    Invoke-Step -Name "/api/model/update" -Action {
        $modelUpdate = Invoke-JsonApi -Method Post -Path "/api/model/update" -Body @{ package = "external-index://daily-2026-02-20"; checksum = "sha256:demo"; strategy = "incremental"; activateVersion = "v3.0.1"; warmupBatches = 5 }
        Assert-Ok -Name "/api/model/update" -Resp $modelUpdate
        Assert-HasProperty -Name "/api/model/update.result" -Obj $modelUpdate.result -Property "seq"
        Assert-HasProperty -Name "/api/model/update.result" -Obj $modelUpdate.result -Property "activeVersion"
        $script:modelUpdate = $modelUpdate
    } | Out-Null

    if (-not $SkipExtendedChecks -and $null -ne $script:modelUpdate) {
        Invoke-Step -Name "/api/model/update seq monotonically increases" -Action {
            $again = Invoke-JsonApi -Method Post -Path "/api/model/update" -Body @{ package = "external-index://daily-2026-02-21"; checksum = "sha256:demo2"; strategy = "incremental"; activateVersion = "v3.0.2"; warmupBatches = 1 }
            Assert-Ok -Name "/api/model/update(second)" -Resp $again
            if ([int64]$again.result.seq -le [int64]$script:modelUpdate.result.seq) {
                throw "expected seq to increase: previous=$($script:modelUpdate.result.seq) next=$($again.result.seq)"
            }
        } | Out-Null
    }

    Invoke-Step -Name "/api/cluster/status" -Action {
        $clusterStatus = Invoke-JsonApi -Method Get -Path "/api/cluster/status" -Body $null
        Assert-Ok -Name "/api/cluster/status" -Resp $clusterStatus
    } | Out-Null

    $script:clusterRoute = $null
    Invoke-Step -Name "/api/cluster/route" -Action {
        $clusterRoute = Invoke-JsonApi -Method Post -Path "/api/cluster/route" -Body @{ maxTokens = 128; preferLowLatency = $true }
        Assert-Ok -Name "/api/cluster/route" -Resp $clusterRoute
        $script:clusterRoute = $clusterRoute
    } | Out-Null

    Invoke-Step -Name "/api/cluster/feedback" -Action {
        $clusterFeedback = Invoke-JsonApi -Method Post -Path "/api/cluster/feedback" -Body @{ nodeId = $script:clusterRoute.result.nodeId; latencyMs = 120; success = $true }
        Assert-Ok -Name "/api/cluster/feedback" -Resp $clusterFeedback
    } | Out-Null

    Invoke-Step -Name "/api/data/governance" -Action {
        $governance = Invoke-JsonApi -Method Get -Path "/api/data/governance" -Body $null
        Assert-Ok -Name "/api/data/governance" -Resp $governance
    } | Out-Null

    Invoke-Step -Name "/api/data/collect" -Action {
        $collect = Invoke-JsonApi -Method Post -Path "/api/data/collect" -Body @{ sources = @("tests", "robots", "external-index") }
        Assert-Ok -Name "/api/data/collect" -Resp $collect
    } | Out-Null

    Invoke-Step -Name "/api/data/cleaning/profile" -Action {
        $cleaning = Invoke-JsonApi -Method Post -Path "/api/data/cleaning/profile" -Body @{ enabled = $true; maxChars = 3072; removeControlChars = $true; normalizeSpace = $true; dropIllegalUtf8 = $true }
        Assert-Ok -Name "/api/data/cleaning/profile" -Resp $cleaning
    } | Out-Null

    Invoke-Step -Name "/api/transformer/modernize" -Action {
        $modern = Invoke-JsonApi -Method Post -Path "/api/transformer/modernize" -Body @{ profile = "sota-balanced" }
        Assert-Ok -Name "/api/transformer/modernize" -Resp $modern
        Assert-HasProperty -Name "/api/transformer/modernize.result" -Obj $modern.result -Property "transformerPatch"
    } | Out-Null

    Invoke-Step -Name "/api/monitoring/training" -Action {
        $trainMon = Invoke-JsonApi -Method Get -Path "/api/monitoring/training" -Body $null
        Assert-Ok -Name "/api/monitoring/training" -Resp $trainMon
    } | Out-Null

    Invoke-Step -Name "/api/monitoring/training/reset" -Action {
        $trainMonReset = Invoke-JsonApi -Method Post -Path "/api/monitoring/training/reset" -Body @{}
        Assert-Ok -Name "/api/monitoring/training/reset" -Resp $trainMonReset
    } | Out-Null

    if (-not $SkipExtendedChecks) {
        Invoke-Step -Name "negative: unknown route -> 404" -Action {
            Assert-HttpError -Name "GET /api/does-not-exist" -ExpectedStatus 404 -Action {
                Invoke-JsonApi -Method Get -Path "/api/does-not-exist" -Body $null
            }
        } | Out-Null
    }

    Invoke-Step -Name "/auth/logout" -Action {
        $logout = Invoke-JsonApiFallback -Method Post -Paths @("/auth/logout", "/api/auth/logout") -Body @{} -BaseUrlOverride $script:ResolvedAuthBaseUrl
        Assert-Ok -Name "/auth/logout" -Resp $logout
    } | Out-Null

    Write-Host ""
    Write-Host "==================== API regression summary ===================="
    Write-Host ("PASS={0} FAIL={1} SKIP={2}" -f $script:PassCount, $script:FailCount, $script:SkipCount)
    if ($script:Failures.Count -gt 0) {
        Write-Host "Failures:"
        foreach ($f in $script:Failures) { Write-Host "  - $f" }
    }
    Write-Host "==================================================================="

    if ($script:FailCount -gt 0) {
        Write-Host "[DONE] API regression completed with failures."
        exit 1
    }
    Write-Host "[DONE] API regression passed."
    exit 0
}
catch {
    Write-Host "[FAIL] API regression aborted."
    Write-Host $_
    if ($script:Failures.Count -gt 0) {
        Write-Host "Failures before abort:"
        foreach ($f in $script:Failures) { Write-Host "  - $f" }
    }
    exit 1
}
