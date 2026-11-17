#Requires -Version 7.0
<#
===============================================================================
 selfmod_tree.ps1 - branch-TREE self-modification harness for Phoenix (v10.3)
===============================================================================
 The AI that improves Phoenix creates ONE git branch per code change, forming a
 TREE of branches:

     main -> selfmod/<ts>-n0001 -> selfmod/<ts>-n0002 -> ...
              \-> selfmod/<ts>-n0003

 The tree is explored with a chess-engine style bias:
   * best-first expansion: expand the highest-scoring open leaf next,
   * prune: keep the top-k frontier by score, drop the rest,
   * select: in the end keep the single best root->leaf path.

 THIS HARNESS ONLY PROPOSES.  `select` prints the exact git commands a human
 would run to adopt the best path; it never merges, rebases, resets or pushes.

 Commands
 --------
   init                       create runtime_store/selfmod/tree.json (main root)
   selfcheck                  read-only preflight (gateway, git, toolchain)
   status                     show the tree, scores, live gateway, worktrees
   expand <parent>            create a CHILD worktree+branch, apply patch, build
                              sidecar exe   (-Patch <file> -Goal ... -NoBuild)
   build <node>               (re)run the sidecar build for an existing node
   score <node>               targeted gtest + wide regression + soak on a
                              rotated port; compute the 0.5/0.3/0.2 score
   prune                      keep top-k frontier by score (default k=3),
                              remove the rest (worktree + branch)
   select                     print best root->leaf path + adoption commands
   cleanup -Force             remove every worktree/branch this harness made
   teardown <node> -Force     remove one node's worktree/branch

 Examples
 --------
   pwsh -NoProfile -File tools/selfmod_tree.ps1 init
   pwsh -NoProfile -File tools/selfmod_tree.ps1 expand main -Patch x.diff -Goal "tune X"
   pwsh -NoProfile -File tools/selfmod_tree.ps1 build n0001
   pwsh -NoProfile -File tools/selfmod_tree.ps1 score n0001 -SoakMinutes 25
   pwsh -NoProfile -File tools/selfmod_tree.ps1 prune -K 3
   pwsh -NoProfile -File tools/selfmod_tree.ps1 select
   pwsh -NoProfile -File tools/selfmod_tree.ps1 cleanup -Force

 Hard rails (inherited from doc/v10.1/self_modification_protocol.md)
 ----------------------------------------------------------------
   1. NEVER runs compile.bat (that script kills the live gateway).  The node
      build is a sidecar compile to a per-node exe name.
   2. NEVER stops the live gateway / llama-servers.  Soak runs a *second*
      process on a rotated free port with its own pid-file + AI_BASE_DIR.
   3. NEVER git push / reset --hard / rebase / rewrite history, and never
      deletes branches it did not create (selfmod/<...> only, tracked in
      runtime_store/selfmod/tree.json).
   4. Path whitelist enforced on the patch; deny paths rejected.
   5. Fail-closed: build/test/soak failures never touch main.  Main is only
      ever changed by a human running the commands printed by `select`.
===============================================================================
#>
[CmdletBinding()]
param(
  [Parameter(Position = 0)]
  [ValidateSet('init', 'selfcheck', 'status', 'expand', 'build', 'score', 'prune', 'select', 'cleanup', 'teardown')]
  [string]$Command = 'status',

  # expand: PARENT node id ("main" or an existing node).  build/score/teardown: node id.
  [Parameter(Position = 1)]
  [string]$Node = 'main',

  [string]$Id = '',                       # explicit child id (default: auto nNNNN)
  [string]$Goal = '',
  [string]$Slug = '',
  [string]$Patch = '',                    # unified diff file for expand
  [string]$Base = '',                     # optional explicit base ref for expand (default parent branch)
  [switch]$NoBuild,                       # expand without compiling
  [switch]$BuildGtest,                    # score/build: also build a per-node gtest_runner
  [int]$BuildTimeoutMin = 90,
  [int]$GtestBuildTimeoutMin = 120,
  [double]$SoakMinutes = 25,              # real score uses 20-30 min; dry runs may shorten
  [int]$SoakSeconds = 0,                  # >0 overrides SoakMinutes (abbreviated soak)
  [int]$IntervalSec = 10,
  [int]$StartupGraceSec = 300,
  [int]$TargetedTimeoutSec = 900,
  [int]$WideTimeoutSec = 3600,
  [int]$K = 3,
  [double]$MinScore = 0.75,
  [switch]$RequireNodeRunner,             # refuse the shared runner even for non-product diffs
  [string]$TargetMetricCommand = '',      # optional: prints a number = target metric
  [int]$PortBase = 5196,
  [switch]$DryRun,                        # prune/cleanup/teardown: plan only
  [switch]$Force,                         # cleanup/teardown/init: required to delete
  [string]$RepoRoot = ''
)

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# constants / paths
# ---------------------------------------------------------------------------
if ($RepoRoot) {
  $script:RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
} else {
  $script:RepoRoot = Split-Path -Parent $PSScriptRoot
}
if (-not (Test-Path -LiteralPath (Join-Path $script:RepoRoot 'main.cpp'))) {
  throw "RepoRoot does not look like the Phoenix repo: $($script:RepoRoot)"
}

$script:SelfmodRoot = Join-Path $script:RepoRoot 'build\tmp\selfmod'
$script:StateDir = Join-Path $script:RepoRoot 'runtime_store\selfmod'
$script:TreePath = Join-Path $script:StateDir 'tree.json'
$script:FailurePath = Join-Path $script:StateDir 'last_failure.json'
$script:LockPath = Join-Path $script:StateDir '.lock'
$script:ArtifactsRoot = Join-Path $script:StateDir 'nodes'

$script:Weights = [ordered]@{ pass = 0.5; soak = 0.3; target = 0.2 }
$script:MaxPatchBytes = 204800
$script:SchemaVersion = 1

$script:TargetedFilter = 'LogicEngine.*:PluginSemantics.*:SummonSelect.*:ExploreWidth.*:PopulationFeatures.*:RetrievalDiversity.*:StudyConfigStateSplit.*:AgiActionDispatchTest.*:CognitionHardeningTest.*'
$script:WideFilter = '*Autonomy*:*Mission*:*Study*:*Population*:*Summon*:*Memetic*:*TinyExpert*:*Expert*'

# Hard rails: path whitelist / deny list (protocol section 2).
$script:AllowPrefixes = @(
  'main_hub_parts/',
  'addons/',
  'doc/',
  'tools/',
  'config/phoenix.json'
)
$script:DenyPrefixes = @(
  '.git/',
  'build/',
  'outsides/',
  'GGUF_models/',
  'runtime_store/',
  'Python314/',
  'poppler-',
  'models/',
  'datasets/',
  'node_modules/',
  '079project_frontend/node_modules/',
  'hacktests/',
  'catastrophe/'
)
$script:DenyExtensions = @('.exe', '.dll', '.lib', '.obj', '.o', '.a', '.pdb', '.so',
  '.dylib', '.bin', '.gguf', '.onnx', '.pt', '.pth', '.safetensors', '.ckpt')
$script:ProductExtensions = @('.cpp', '.cc', '.cxx', '.c', '.h', '.hpp', '.hh', '.hxx', '.inc', '.ipp')

# ---------------------------------------------------------------------------
# tiny helpers
# ---------------------------------------------------------------------------
function Write-Log {
  param([string]$Message)
  Write-Host ('[{0}] {1}' -f (Get-Date -Format 'HH:mm:ss'), $Message)
}

function Get-Prop {
  param($Object, [string]$Name, $Default = $null)
  if ($null -eq $Object) { return $Default }
  $p = $Object.PSObject.Properties[$Name]
  if ($null -eq $p) { return $Default }
  return $p.Value
}

function Set-Prop {
  param($Object, [string]$Name, $Value)
  if ($null -eq $Object) { return }
  if ($Object.PSObject.Properties[$Name]) {
    $Object.$Name = $Value
  } else {
    $Object | Add-Member -NotePropertyName $Name -NotePropertyValue $Value
  }
}

function Invoke-Git {
  param(
    [Parameter(Mandatory)][string[]]$Arguments,
    [string]$WorkDir = '',
    [switch]$AllowFail
  )
  if (-not $WorkDir) { $WorkDir = $script:RepoRoot }
  $global:LASTEXITCODE = 0
  $out = & git -C $WorkDir @Arguments 2>&1
  $code = $LASTEXITCODE
  $text = ($out | ForEach-Object { "$_" }) -join "`n"
  if ($code -ne 0 -and -not $AllowFail) {
    throw ("git {0} (in {1}) failed with exit {2}:`n{3}" -f ($Arguments -join ' '), $WorkDir, $code, $text)
  }
  return [pscustomobject]@{ ExitCode = $code; Output = $text }
}

function Get-NodeArtifacts {
  param([string]$NodeId)
  $dir = Join-Path $script:ArtifactsRoot $NodeId
  if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
  return $dir
}

function Get-NodeWorktree {
  param([string]$NodeId)
  return (Join-Path $script:SelfmodRoot $NodeId)
}

function Get-NodeExe {
  param([object]$N)
  return (Join-Path $N.worktree ("build\tmp\selfmod\phoenix_main.{0}.exe" -f $N.id))
}

function Get-NodeBuildLog {
  param([object]$N)
  return (Join-Path $N.worktree ("build\tmp\selfmod\compile_{0}.log" -f $N.id))
}

# ---------------------------------------------------------------------------
# live gateway guard (nothing here may stop it)
# ---------------------------------------------------------------------------
function Get-LiveGateway {
  $portFile = Join-Path $script:RepoRoot 'runtime_store\long_run\port.txt'
  $gwPort = 0
  if (Test-Path -LiteralPath $portFile) {
    $raw = Get-Content -LiteralPath $portFile -Raw -ErrorAction SilentlyContinue
    if ($raw) { [void][int]::TryParse($raw.Trim(), [ref]$gwPort) }
  }
  $livePid = 0
  if ($gwPort -gt 0) {
    $conn = Get-NetTCPConnection -LocalPort $gwPort -State Listen -ErrorAction SilentlyContinue |
      Select-Object -First 1
    if ($conn) { $livePid = [int]$conn.OwningProcess }
  }
  return [pscustomobject]@{ Port = $gwPort; Pid = $livePid; Alive = ($gwPort -gt 0 -and $livePid -gt 0) }
}

function Test-Health {
  param([int]$Port, [int]$TimeoutSec = 5)
  if ($Port -le 0) { return $false }
  try {
    $r = Invoke-RestMethod -Uri ("http://127.0.0.1:{0}/api/health" -f $Port) -TimeoutSec $TimeoutSec -ErrorAction Stop
    return [bool](Get-Prop $r 'ok' $false)
  } catch {
    return $false
  }
}

function Stop-OwnedProcessTree {
  param([int]$OwnerPid)
  if ($OwnerPid -le 0) { return }
  $gw = Get-LiveGateway
  if ($OwnerPid -eq $gw.Pid) {
    throw ("SAFETY: refusing to stop pid {0} - it is the live gateway" -f $OwnerPid)
  }
  $all = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
  $toKill = New-Object System.Collections.Generic.List[int]
  $queue = New-Object System.Collections.Generic.Queue[int]
  $queue.Enqueue($OwnerPid)
  while ($queue.Count -gt 0) {
    $parentId = $queue.Dequeue()
    foreach ($childProc in @($all | Where-Object { $_.ParentProcessId -eq $parentId })) {
      $childProcId = [int]$childProc.ProcessId
      if ($childProcId -eq $gw.Pid) { throw 'SAFETY: child tree contains the live gateway pid' }
      $queue.Enqueue($childProcId)
      $toKill.Add($childProcId)
    }
  }
  for ($i = $toKill.Count - 1; $i -ge 0; $i--) {
    Stop-Process -Id $toKill[$i] -Force -ErrorAction SilentlyContinue
  }
  Stop-Process -Id $OwnerPid -Force -ErrorAction SilentlyContinue
}

function Assert-GatewayIntact {
  param([object]$Before, [string]$When)
  $after = Get-LiveGateway
  if ($Before.Pid -gt 0 -and $after.Pid -ne $Before.Pid) {
    Write-Log ("WARNING: live gateway pid changed ({0} -> {1}) during {2}; investigate immediately" -f $Before.Pid, $after.Pid, $When)
    return $false
  }
  if (-not $after.Alive) {
    Write-Log ("WARNING: live gateway not answering on port {0} {1}" -f $after.Port, $When)
    return $false
  }
  return $true
}

function Assert-NoForbiddenBuild {
  # The harness never calls compile.bat; this only refuses to start a node build
  # while a *main-tree* build is already running (shared compiler caches / RAM).
  $running = Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'compile(_gtest)?\.bat' }
  if ($running) {
    throw 'a main-tree build (compile.bat / compile_gtest.bat) is already running - wait for it to finish'
  }
}

# ---------------------------------------------------------------------------
# state file (tree.json) + short critical-section lock
# ---------------------------------------------------------------------------
function Enter-StateLock {
  if (-not (Test-Path -LiteralPath $script:StateDir)) {
    New-Item -ItemType Directory -Force -Path $script:StateDir | Out-Null
  }
  for ($attempt = 0; $attempt -lt 100; $attempt++) {
    try {
      $fs = [System.IO.File]::Open($script:LockPath, [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
      $sw = New-Object System.IO.StreamWriter($fs)
      $sw.WriteLine([string]$PID)
      $sw.Flush()
      $script:StateLockFs = $fs
      $script:StateLockSw = $sw
      return
    } catch {
      if (Test-Path -LiteralPath $script:LockPath) {
        $oldPid = 0
        $raw = Get-Content -LiteralPath $script:LockPath -Raw -ErrorAction SilentlyContinue
        if ($raw) { [void][int]::TryParse($raw.Trim(), [ref]$oldPid) }
        $alive = $false
        if ($oldPid -gt 0) { $alive = [bool](Get-Process -Id $oldPid -ErrorAction SilentlyContinue) }
        if (-not $alive) {
          Remove-Item -LiteralPath $script:LockPath -Force -ErrorAction SilentlyContinue
          continue
        }
      }
      Start-Sleep -Milliseconds 150
    }
  }
  throw "could not acquire $($script:LockPath) (another selfmod_tree.ps1 is running?)"
}

function Exit-StateLock {
  if ($script:StateLockSw) { try { $script:StateLockSw.Dispose() } catch { } ; $script:StateLockSw = $null }
  if ($script:StateLockFs) { try { $script:StateLockFs.Dispose() } catch { } ; $script:StateLockFs = $null }
  Remove-Item -LiteralPath $script:LockPath -Force -ErrorAction SilentlyContinue
}

function Read-Tree {
  if (-not (Test-Path -LiteralPath $script:TreePath)) {
    throw "tree.json not found - run: pwsh -NoProfile -File tools/selfmod_tree.ps1 init"
  }
  return (Get-Content -LiteralPath $script:TreePath -Raw | ConvertFrom-Json)
}

function Save-Tree {
  param($Tree)
  $Tree.updatedAt = (Get-Date).ToString('o')
  $json = $Tree | ConvertTo-Json -Depth 30
  $utf8 = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText($script:TreePath, $json, $utf8)
}

function Get-NodeById {
  param($Tree, [string]$NodeId)
  return ($Tree.nodes | Where-Object { $_.id -eq $NodeId } | Select-Object -First 1)
}

function Update-NodeState {
  param([string]$NodeId, [scriptblock]$Mutator)
  Enter-StateLock
  try {
    $tree = Read-Tree
    $n = Get-NodeById $tree $NodeId
    if ($null -eq $n) { throw "node not found: $NodeId" }
    & $Mutator $n $tree
    Save-Tree $tree
  } finally {
    Exit-StateLock
  }
}

function Add-Evidence {
  param([object]$N, [string]$Kind, [string]$Path, [string]$Summary)
  $list = New-Object System.Collections.Generic.List[object]
  foreach ($e in @($N.evidence)) { if ($null -ne $e) { $list.Add($e) } }
  $list.Add([pscustomobject]@{
      kind    = $Kind
      path    = $Path
      ts      = (Get-Date).ToString('o')
      summary = $Summary
    })
  $N.evidence = $list.ToArray()
}

function Write-Failure {
  param([string]$NodeId, [string]$Stage, [string]$Message)
  $payload = [pscustomobject]@{
    ts     = (Get-Date).ToString('o')
    node   = $NodeId
    stage  = $Stage
    error  = $Message
    note   = 'main tree untouched; node is fail-closed and prunable'
  }
  $utf8 = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText($script:FailurePath, ($payload | ConvertTo-Json -Depth 6), $utf8)
}

# ---------------------------------------------------------------------------
# patch validation (hard rails)
# ---------------------------------------------------------------------------
function Get-PatchPaths {
  param([string]$PatchPath)
  if (-not (Test-Path -LiteralPath $PatchPath)) { throw "patch file not found: $PatchPath" }
  $full = (Resolve-Path -LiteralPath $PatchPath).Path
  $size = (Get-Item -LiteralPath $full).Length
  if ($size -eq 0) { throw 'patch file is empty' }
  if ($size -gt $script:MaxPatchBytes) {
    throw ("patch is {0} bytes > maxPatchBytes {1}" -f $size, $script:MaxPatchBytes)
  }
  $lines = @(Get-Content -LiteralPath $full)
  $paths = New-Object System.Collections.Generic.List[string]
  for ($i = 0; $i -lt ($lines.Count - 1); $i++) {
    $a = $lines[$i]
    $b = $lines[$i + 1]
    if ($a -match '^---\s+(.+)$' -and $b -match '^\+\+\+\s+(.+)$') {
      $pa = ($Matches[1]).Trim()
      $pb = $null
      if ($b -match '^\+\+\+\s+(.+)$') { $pb = ($Matches[1]).Trim() }
      foreach ($raw in @($pa, $pb)) {
        $p = $raw.Trim('"')
        if ($p -eq '/dev/null' -or $p -eq '') { continue }
        $p = $p -replace '^[ab]/', ''
        $p = ($p -split "`t")[0].Trim()
        if (-not $paths.Contains($p)) { $paths.Add($p) }
      }
      $i++
      continue
    }
    if ($a -match '^rename (?:from|to)\s+(.+)$') {
      $p = ($Matches[1]).Trim().Trim('"') -replace '^[ab]/', ''
      if (-not $paths.Contains($p)) { $paths.Add($p) }
    }
    if ($a -match '^copy (?:from|to)\s+(.+)$') {
      $p = ($Matches[1]).Trim().Trim('"') -replace '^[ab]/', ''
      if (-not $paths.Contains($p)) { $paths.Add($p) }
    }
  }
  if ($paths.Count -eq 0) { throw 'no file paths found in patch headers (expects a unified git diff)' }
  return [pscustomobject]@{ FullPath = $full; Size = $size; Paths = $paths.ToArray() }
}

function Get-PathViolation {
  param([string]$Path)
  $norm = $Path -replace '\\', '/'
  if ($norm -match '(^|/)\.\.(/|$)' -or $norm.StartsWith('/') -or $norm -match '^[A-Za-z]:') {
    return 'absolute or parent-relative path'
  }
  foreach ($deny in $script:DenyPrefixes) {
    if ($norm -ieq $deny -or $norm.StartsWith($deny, [System.StringComparison]::OrdinalIgnoreCase)) {
      return "deny path: $deny"
    }
  }
  $ext = [System.IO.Path]::GetExtension($norm).ToLowerInvariant()
  if ($script:DenyExtensions -contains $ext) { return "deny binary extension: $ext" }
  foreach ($pre in $script:AllowPrefixes) {
    if ($norm.StartsWith($pre, [System.StringComparison]::OrdinalIgnoreCase)) { return $null }
  }
  if ($ext -eq '.hpp') { return $null }
  $base = [System.IO.Path]::GetFileName($norm)
  if ($base -like 'autonomy_stack.*') { return $null }
  return 'not in selfModify.allowPaths whitelist'
}

function Assert-PatchAllowed {
  param([object]$PatchInfo)
  $errors = New-Object System.Collections.Generic.List[string]
  foreach ($p in @($PatchInfo.Paths)) {
    $why = Get-PathViolation $p
    if ($why) { $errors.Add("$p -> $why") }
  }
  if ($errors.Count -gt 0) {
    throw ("patch rejected by hard rails:`n  " + ($errors -join "`n  "))
  }
}

function Test-ProductDiff {
  param([object]$Diff)
  foreach ($f in @(Get-Prop $Diff 'files' @())) {
    $ext = [System.IO.Path]::GetExtension([string]$f).ToLowerInvariant()
    if ($script:ProductExtensions -contains $ext) { return $true }
  }
  return $false
}

function Get-DiffStat {
  param([object]$N)
  $baseRef = Get-Prop $N 'baseBranch' 'main'
  $numstat = & git -C $N.worktree diff --numstat $baseRef 2>$null
  $files = New-Object System.Collections.Generic.List[string]
  $added = 0
  $removed = 0
  foreach ($line in @($numstat)) {
    if (-not $line) { continue }
    $parts = "$line".Split("`t")
    if ($parts.Count -lt 3) { continue }
    if ($parts[0] -match '^\d+$') { $added += [int]$parts[0] }
    if ($parts[1] -match '^\d+$') { $removed += [int]$parts[1] }
    $files.Add($parts[2])
  }
  return [pscustomobject]@{
    files       = $files.ToArray()
    added       = $added
    removed     = $removed
    total       = ($added + $removed)
    productDiff = (Test-ProductDiff ([pscustomobject]@{ files = $files.ToArray() }))
  }
}

# ---------------------------------------------------------------------------
# build: sidecar compile of a node worktree to its OWN exe (never compile.bat)
# ---------------------------------------------------------------------------
function Resolve-ConanFile {
  param([string]$Pattern, [switch]$MainChannel)
  $candidates = @(Get-ChildItem -LiteralPath (Join-Path $script:RepoRoot 'build') -Filter $Pattern -File -ErrorAction SilentlyContinue)
  if ($MainChannel) { $candidates = @($candidates | Where-Object { $_.Name -notmatch 'gtest' }) }
  $candidates = @($candidates | Sort-Object LastWriteTime -Descending)
  if ($candidates.Count -eq 0) { return $null }
  return $candidates[0].FullName
}

function Get-NodeSourcesFile {
  param([object]$N)
  $mainSources = Join-Path $script:RepoRoot 'build\compile_sources.txt'
  if (-not (Test-Path -LiteralPath $mainSources)) {
    throw "missing $mainSources - build the baseline once (compile.bat was already run historically)"
  }
  return $mainSources
}

function New-NodeBuildScript {
  param([object]$N)
  $wt = $N.worktree
  $tmpDir = Join-Path $wt 'build\tmp\selfmod'
  New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
  $gxx = 'C:\msys64\mingw64\bin\g++.exe'
  if (-not (Test-Path -LiteralPath $gxx)) { throw "g++ not found at $gxx" }
  $cflags = Resolve-ConanFile 'conan_cflags_*.txt' -MainChannel
  $libs = Resolve-ConanFile 'conan_libs_*.txt' -MainChannel
  $mainRoot = $script:RepoRoot
  $sources = Get-NodeSourcesFile $N
  $exeRel = "build\tmp\selfmod\phoenix_main.$($N.id).exe"
  $mapRel = "build\tmp\selfmod\phoenix_main.$($N.id).map"
  $logRel = "build\tmp\selfmod\compile_$($N.id).log"

  $sb = New-Object System.Text.StringBuilder
  [void]$sb.AppendLine('@echo off')
  [void]$sb.AppendLine('setlocal EnableExtensions')
  [void]$sb.AppendLine(('cd /d "{0}"' -f $wt))
  [void]$sb.AppendLine(('set "TMP={0}\build\tmp\selfmod\tmp"' -f $wt))
  [void]$sb.AppendLine('set "TEMP=%TMP%"')
  [void]$sb.AppendLine('if not exist "%TMP%" mkdir "%TMP%"')
  [void]$sb.AppendLine('if not exist "build\tmp\selfmod" mkdir "build\tmp\selfmod"')
  $cmd = New-Object System.Text.StringBuilder
  [void]$cmd.Append(('"{0}" -o "{1}" -std=c++20 -Wa,-mbig-obj ' -f $gxx, $exeRel))
  [void]$cmd.Append('-DAI_EXTERNAL_BACKEND_COMPAT=1 -DAI_EXTERNAL_LEARNER_BRIDGE=1 -DHAVE_SQLITE ')
  [void]$cmd.Append('-DPHOENIX_EDGE_IMAGE_ENABLED=1 -DPHOENIX_EDGE_SPEECH_ENABLED=1 ')
  [void]$cmd.Append('-DAI_OUTSIDES_LINK_MODE=\"headers-only\" ')
  [void]$cmd.Append(('@"{0}" ' -f $cflags))
  [void]$cmd.Append(('-I"{0}\poppler-25.12.0\Library\include" ' -f $mainRoot))
  [void]$cmd.Append(('-I"{0}\Python314\include" ' -f $mainRoot))
  [void]$cmd.Append(('-I"{0}\outsides\llamacpp\include" ' -f $mainRoot))
  [void]$cmd.Append(('-I"{0}\outsides\llamacpp\ggml\include" ' -f $mainRoot))
  [void]$cmd.Append(('-I"{0}\outsides\llamacpp\ggml\src" ' -f $mainRoot))
  [void]$cmd.Append(('-I"{0}\outsides\BitNet\include" ' -f $mainRoot))
  [void]$cmd.Append(('-I"{0}\outsides\BitNet\src" ' -f $mainRoot))
  [void]$cmd.Append(('-I"{0}" ' -f $wt))
  [void]$cmd.Append(('-I"{0}" ' -f $mainRoot))
  [void]$cmd.Append('main.cpp ')
  [void]$cmd.Append(('@"{0}" ' -f $sources))
  [void]$cmd.Append(('-Wl,--start-group @"{0}" -Wl,--end-group ' -f $libs))
  [void]$cmd.Append(('"{0}\poppler-25.12.0\Library\lib\poppler-cpp.lib" ' -f $mainRoot))
  [void]$cmd.Append(('"{0}\poppler-25.12.0\Library\lib\poppler.lib" ' -f $mainRoot))
  [void]$cmd.Append(('-L"{0}\Python314\libs" -lpython314 -lws2_32 ' -f $mainRoot))
  [void]$cmd.Append('-Wl,--wrap,__emutls_get_address ')
  [void]$cmd.Append(('-Wl,-Map={0} ' -f $mapRel))
  [void]$cmd.Append('-O3 ')
  [void]$cmd.Append(('> "{0}" 2>&1' -f $logRel))
  [void]$sb.AppendLine($cmd.ToString())
  [void]$sb.AppendLine(('set "RC=%errorlevel%"'))
  [void]$sb.AppendLine(('echo EXIT=%RC% >> "{0}"' -f $logRel))
  [void]$sb.AppendLine('endlocal & exit /b %RC%')

  $batPath = Join-Path $tmpDir ("compile_node_{0}.bat" -f $N.id)
  $utf8 = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText($batPath, $sb.ToString(), $utf8)
  return [pscustomobject]@{ Bat = $batPath; Exe = (Join-Path $wt $exeRel); Log = (Join-Path $wt $logRel) }
}

function Get-GtestCommonSources {
  # Parse COMMON_SOURCES out of the tracked compile_gtest.bat so the node gtest
  # runner compiles exactly the same product translation units.
  $line = Select-String -LiteralPath (Join-Path $script:RepoRoot 'compile_gtest.bat') -Pattern 'set "COMMON_SOURCES=(.*)"' |
    Select-Object -First 1
  if (-not $line) { throw 'could not parse COMMON_SOURCES from compile_gtest.bat' }
  $raw = $line.Matches[0].Groups[1].Value.Trim()
  $tokens = $raw -split '\s+' | Where-Object { $_ }
  $out = New-Object System.Collections.Generic.List[string]
  foreach ($t in $tokens) {
    if ($t -like '*BULLET3_EMBEDDED_SOURCES*') {
      $bullet = Join-Path $script:RepoRoot 'outsides\bullet3\src'
      foreach ($bf in @('btLinearMathAll.cpp', 'btBulletCollisionAll.cpp', 'btBulletDynamicsAll.cpp')) {
        $full = Join-Path $bullet $bf
        if (Test-Path -LiteralPath $full) { $out.Add(($full -replace '\\', '/')) }
      }
    } elseif ($t -like '*outsides*') {
      $out.Add((Join-Path $script:RepoRoot $t) -replace '\\', '/')
    } else {
      $out.Add(($t -replace '\\', '/'))
    }
  }
  return $out.ToArray()
}

function New-NodeGtestBuildScript {
  param([object]$N)
  $wt = $N.worktree
  $tmpDir = Join-Path $wt 'build\tmp\selfmod'
  New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
  $gxx = 'C:\msys64\mingw64\bin\g++.exe'
  $cflags = Resolve-ConanFile 'conan_cflags_gtest.txt'
  $libs = Resolve-ConanFile 'conan_libs_gtest.txt'
  if (-not $cflags -or -not $libs) { throw 'conan gtest cflags/libs missing in build/ - run compile_gtest.bat once on main' }
  $mainRoot = $script:RepoRoot

  $sourcesFile = Join-Path $tmpDir ("gtest_sources_{0}.txt" -f $N.id)
  $srcList = New-Object System.Collections.Generic.List[string]
  foreach ($rel in @('tests\gtest', 'test\gtest')) {
    $dir = Join-Path $wt $rel
    if (Test-Path -LiteralPath $dir) {
      Get-ChildItem -LiteralPath $dir -Recurse -Filter *.cpp -File |
        ForEach-Object { $srcList.Add(($_.FullName -replace '\\', '/')) }
    }
  }
  foreach ($s in Get-GtestCommonSources) { $srcList.Add($s) }
  $utf8 = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllLines($sourcesFile, $srcList.ToArray(), $utf8)

  $logRel = "build\tmp\selfmod\compile_gtest_$($N.id).log"
  $exeRel = "gtest_runner.$($N.id).exe"
  $sb = New-Object System.Text.StringBuilder
  [void]$sb.AppendLine('@echo off')
  [void]$sb.AppendLine('setlocal EnableExtensions')
  [void]$sb.AppendLine(('cd /d "{0}"' -f $wt))
  [void]$sb.AppendLine(('set "TMP={0}\build\tmp\selfmod\tmp"' -f $wt))
  [void]$sb.AppendLine('set "TEMP=%TMP%"')
  [void]$sb.AppendLine('if not exist "%TMP%" mkdir "%TMP%"')
  [void]$sb.AppendLine('if not exist "build\tmp\selfmod" mkdir "build\tmp\selfmod"')
  $cmd = New-Object System.Text.StringBuilder
  [void]$cmd.Append(('"{0}" -o "{1}" -std=c++20 -Wa,-mbig-obj ' -f $gxx, $exeRel))
  [void]$cmd.Append('-DAI_EXTERNAL_BACKEND_COMPAT=1 -DAI_EXTERNAL_LEARNER_BRIDGE=1 -DHAVE_SQLITE ')
  [void]$cmd.Append('-DPHOENIX_EDGE_IMAGE_ENABLED=1 -DPHOENIX_EDGE_SPEECH_ENABLED=1 ')
  [void]$cmd.Append(('@"{0}" ' -f $cflags))
  [void]$cmd.Append(('-I"{0}" ' -f $wt))
  [void]$cmd.Append(('-I"{0}" ' -f $mainRoot))
  [void]$cmd.Append(('-I"{0}\poppler-25.12.0\Library\include" ' -f $mainRoot))
  [void]$cmd.Append(('-I"{0}\Python314\include" ' -f $mainRoot))
  [void]$cmd.Append(('-I"{0}\outsides\llamacpp\include" ' -f $mainRoot))
  [void]$cmd.Append(('-I"{0}\outsides\BitNet\include" ' -f $mainRoot))
  [void]$cmd.Append(('-I"{0}\tests\gtest" ' -f $wt))
  [void]$cmd.Append(('-I"{0}\test\gtest" ' -f $wt))
  [void]$cmd.Append(('@"{0}" ' -f $sourcesFile))
  [void]$cmd.Append(('-Wl,--start-group @"{0}" -Wl,--end-group ' -f $libs))
  [void]$cmd.Append(('"{0}\poppler-25.12.0\Library\lib\poppler-cpp.lib" ' -f $mainRoot))
  [void]$cmd.Append(('"{0}\poppler-25.12.0\Library\lib\poppler.lib" ' -f $mainRoot))
  [void]$cmd.Append(('-L"{0}\Python314\libs" -lpython314 -lws2_32 ' -f $mainRoot))
  [void]$cmd.Append('-Wl,--wrap,__emutls_get_address ')
  [void]$cmd.Append('-O0 -g1 -Wl,--no-keep-memory ')
  [void]$cmd.Append(('> "{0}" 2>&1' -f $logRel))
  [void]$sb.AppendLine($cmd.ToString())
  [void]$sb.AppendLine(('set "RC=%errorlevel%"'))
  [void]$sb.AppendLine(('echo EXIT=%RC% >> "{0}"' -f $logRel))
  [void]$sb.AppendLine('endlocal & exit /b %RC%')
  $batPath = Join-Path $tmpDir ("compile_gtest_node_{0}.bat" -f $N.id)
  [System.IO.File]::WriteAllText($batPath, $sb.ToString(), $utf8)
  return [pscustomobject]@{
    Bat = $batPath
    Exe = (Join-Path $wt $exeRel)
    Log = (Join-Path $wt $logRel)
  }
}

function Invoke-BatchWithTimeout {
  param([string]$BatPath, [string]$WorkDir, [int]$TimeoutSec, [string]$Label)
  $null = $env:ComSpec  # read automatic var (no-op); keeps analyzer calm
  $proc = Start-Process -FilePath $env:ComSpec -ArgumentList @('/c', ('"{0}"' -f $BatPath)) `
    -WorkingDirectory $WorkDir -PassThru -WindowStyle Hidden
  $exited = $proc.WaitForExit([int]($TimeoutSec * 1000))
  if (-not $exited) {
    Write-Log ("$Label timed out after $TimeoutSec s - terminating its process tree")
    Stop-OwnedProcessTree -OwnerPid $proc.Id
    try { $proc.WaitForExit(5000) | Out-Null } catch { }
    return [pscustomobject]@{ ExitCode = -1; TimedOut = $true; DurationSec = $TimeoutSec }
  }
  $proc.Refresh()
  return [pscustomobject]@{ ExitCode = $proc.ExitCode; TimedOut = $false; DurationSec = 0 }
}

function Invoke-NodeBuild {
  param([object]$N, [switch]$Quiet)
  if ($N.id -eq 'main') { throw 'cannot build the main node (baseline exe is built by the normal toolchain)' }
  $wt = $N.worktree
  if (-not (Test-Path -LiteralPath $wt)) { throw "worktree missing for node $($N.id): $wt" }
  $gwBefore = Get-LiveGateway
  $art = Get-NodeArtifacts $N.id
  $spec = New-NodeBuildScript $N
  Write-Log ("building node {0} (sidecar exe phoenix_main.{0}.exe, never compile.bat)" -f $N.id)
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $res = Invoke-BatchWithTimeout -BatPath $spec.Bat -WorkDir $wt -TimeoutSec ([math]::Max(60, $BuildTimeoutMin * 60)) -Label 'node build'
  $sw.Stop()

  $logCopy = Join-Path $art ("build_{0}.log" -f $N.id)
  if (Test-Path -LiteralPath $spec.Log) { Copy-Item -LiteralPath $spec.Log -Destination $logCopy -Force }
  $tail = ''
  if (Test-Path -LiteralPath $spec.Log) { $tail = (Get-Content -LiteralPath $spec.Log -Tail 8) -join "`n" }

  $built = ($res.ExitCode -eq 0) -and (Test-Path -LiteralPath $spec.Exe)
  $status = if ($built) { 'built' } else { 'build_failed' }
  $duration = [math]::Round($sw.Elapsed.TotalMinutes, 2)

  Update-NodeState -NodeId $N.id -Mutator {
    param($node, $tree)
    $node.status = $status
    $node.built = [pscustomobject]@{
      ok          = $built
      exe         = $spec.Exe
      exitCode    = $res.ExitCode
      timedOut    = $res.TimedOut
      durationMin = $duration
      log         = $logCopy
      ts          = (Get-Date).ToString('o')
    }
    Add-Evidence -N $node -Kind 'build' -Path $logCopy -Summary ("exit={0} ok={1} {2} min" -f $res.ExitCode, $built, $duration)
  }

  if (-not $built) {
    Write-Failure -NodeId $N.id -Stage 'build' -Message ("node build failed (exit {0}); log: {1}`n{2}" -f $res.ExitCode, $logCopy, $tail)
    Write-Log ("node {0} build FAILED -> status build_failed (main untouched; node can be pruned)" -f $N.id)
    if (-not $Quiet) { Write-Host $tail }
  } else {
    Write-Log ("node {0} BUILD OK ({1} min) -> {2}" -f $N.id, $duration, $spec.Exe)
  }
  Assert-GatewayIntact -Before $gwBefore -When "build of $($N.id)" | Out-Null
  return $built
}

function Invoke-NodeGtestBuild {
  param([object]$N)
  if ($N.id -eq 'main') { throw 'cannot build gtest for main node' }
  $wt = $N.worktree
  Copy-SharedInputs -N $N -IncludeTests
  $spec = New-NodeGtestBuildScript $N
  $gwBefore = Get-LiveGateway
  Write-Log ("building per-node gtest_runner.{0}.exe (this is the expensive path, ~30-60 min)" -f $N.id)
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $res = Invoke-BatchWithTimeout -BatPath $spec.Bat -WorkDir $wt -TimeoutSec ([math]::Max(60, $GtestBuildTimeoutMin * 60)) -Label 'node gtest build'
  $sw.Stop()
  $art = Get-NodeArtifacts $N.id
  $logCopy = Join-Path $art ("build_gtest_{0}.log" -f $N.id)
  if (Test-Path -LiteralPath $spec.Log) { Copy-Item -LiteralPath $spec.Log -Destination $logCopy -Force }
  $ok = ($res.ExitCode -eq 0) -and (Test-Path -LiteralPath $spec.Exe)
  Update-NodeState -NodeId $N.id -Mutator {
    param($node, $tree)
    Set-Prop $node 'gtestBuild' ([pscustomobject]@{
      ok          = $ok
      exe         = $spec.Exe
      exitCode    = $res.ExitCode
      durationMin = [math]::Round($sw.Elapsed.TotalMinutes, 2)
      log         = $logCopy
      ts          = (Get-Date).ToString('o')
    })
    Add-Evidence -N $node -Kind 'gtest-build' -Path $logCopy -Summary ("exit={0} ok={1}" -f $res.ExitCode, $ok)
  }
  if (-not $ok) {
    Write-Failure -NodeId $N.id -Stage 'gtest-build' -Message "node gtest build failed; see $logCopy"
    throw "node gtest build failed; see $logCopy"
  }
  Assert-GatewayIntact -Before $gwBefore -When "gtest build of $($N.id)" | Out-Null
  return $spec.Exe
}

function Copy-SharedInputs {
  param([object]$N, [switch]$IncludeTests)
  $wt = $N.worktree
  # Runtime config is gitignored; inherit it so the sidecar node behaves like main.
  $cfgSrc = Join-Path $script:RepoRoot 'config\phoenix.json'
  if ((-not $NoInheritConfig) -and (Test-Path -LiteralPath $cfgSrc)) {
    $cfgDstDir = Join-Path $wt 'config'
    New-Item -ItemType Directory -Force -Path $cfgDstDir | Out-Null
    Copy-Item -LiteralPath $cfgSrc -Destination (Join-Path $cfgDstDir 'phoenix.json') -Force
  }
  if ($IncludeTests) {
    # tests/ is gitignored, so a fresh worktree only has the 25 tracked files.
    # Copy the rest from the main tree so the node runner runs the same suites.
    foreach ($rel in @('tests', 'test')) {
      $srcDir = Join-Path $script:RepoRoot $rel
      if (-not (Test-Path -LiteralPath $srcDir)) { continue }
      $dstDir = Join-Path $wt $rel
      & robocopy $srcDir $dstDir /E /XC /XN /XO /NFL /NDL /NJH /NJS /NP | Out-Null
      if ($LASTEXITCODE -ge 8) { throw "robocopy $rel failed with $LASTEXITCODE" }
    }
  }
}

# ---------------------------------------------------------------------------
# score: gtest (targeted + wide) + soak on a rotated port
# ---------------------------------------------------------------------------
function Resolve-GtestRunner {
  param([object]$N)
  $nodeRunner = Join-Path $N.worktree ("gtest_runner.{0}.exe" -f $N.id)
  if (Test-Path -LiteralPath $nodeRunner) {
    return [pscustomobject]@{ Path = $nodeRunner; WorkDir = $N.worktree; Provenance = 'node' }
  }
  $product = [bool](Get-Prop (Get-Prop $N 'diff' $null) 'productDiff' $true)
  if (-not $product -and -not $RequireNodeRunner) {
    # A diff that touches no compiled translation unit cannot change the binary,
    # so the main-tree runner is a *sound* oracle for this node.
    $shared = Join-Path $script:RepoRoot 'gtest_runner.exe'
    if (Test-Path -LiteralPath $shared) {
      return [pscustomobject]@{ Path = $shared; WorkDir = $script:RepoRoot; Provenance = 'shared-main(non-product diff)' }
    }
  }
  return $null
}

function Invoke-GtestRun {
  param([object]$Runner, [string]$Filter, [string]$LogPath, [int]$TimeoutSec, [string]$Label)
  $errLog = "$LogPath.err"
  $oldPythonHome = [Environment]::GetEnvironmentVariable('PYTHONHOME', 'Process')
  try {
    [Environment]::SetEnvironmentVariable('PYTHONHOME', (Join-Path $script:RepoRoot 'Python314'), 'Process')
    $proc = Start-Process -FilePath $Runner.Path -ArgumentList @(('--gtest_filter={0}' -f $Filter), '--gtest_color=no') `
      -WorkingDirectory $Runner.WorkDir -RedirectStandardOutput $LogPath -RedirectStandardError $errLog `
      -PassThru -WindowStyle Hidden
    $exited = $proc.WaitForExit([int]($TimeoutSec * 1000))
    if (-not $exited) {
      Write-Log ("$Label gtest run timed out after $TimeoutSec s - killing own process tree")
      Stop-OwnedProcessTree -OwnerPid $proc.Id
    }
    $proc.Refresh()
    $exit = if ($exited) { $proc.ExitCode } else { -1 }
  } finally {
    [Environment]::SetEnvironmentVariable('PYTHONHOME', $oldPythonHome, 'Process')
  }

  $txt = ''
  if (Test-Path -LiteralPath $LogPath) { $txt = Get-Content -LiteralPath $LogPath -Raw -ErrorAction SilentlyContinue }
  $passed = 0
  $failed = 0
  $total = 0
  if ($txt -match '\[  PASSED  \] (\d+) tests?\.') { $passed = [int]$Matches[1] }
  if ($txt -match '\[  FAILED  \] (\d+) tests?, listed below') { $failed = [int]$Matches[1] }
  elseif ($txt -match '\[  FAILED  \] (\d+) tests?\.') { $failed = [int]$Matches[1] }
  if ($txt -match '\[==========\] (\d+) tests? from') { $total = [int]$Matches[1] }
  if ($total -eq 0) { $total = $passed + $failed }
  if ($failed -gt 0 -and $total -eq 0) { $total = $passed + $failed }

  return [pscustomobject]@{
    Label    = $Label
    ExitCode = $exit
    Passed   = $passed
    Failed   = $failed
    Total    = $total
    Log      = $LogPath
    ErrLog   = $errLog
  }
}

function Get-FreePort {
  $reserved = @(5080, 5081, 5082, 8082, 8086, 8092, 8093, 8100, 5096, 5097)
  $live = Get-LiveGateway
  if ($live.Port -gt 0) { $reserved += $live.Port }
  for ($candidate = $PortBase; $candidate -lt ($PortBase + 100); $candidate++) {
    if ($reserved -contains $candidate) { continue }
    $busy = Get-NetTCPConnection -LocalPort $candidate -State Listen -ErrorAction SilentlyContinue
    if ($busy) { continue }
    return $candidate
  }
  throw "no free soak port in [$PortBase, $($PortBase + 100))"
}

function Invoke-NodeSoak {
  param([object]$N, [int]$DurationSec, [int]$IntervalSec, [int]$GraceSec)
  $exe = Get-NodeExe $N
  if (-not (Test-Path -LiteralPath $exe)) { throw "node exe missing for $($N.id): $exe" }
  $art = Get-NodeArtifacts $N.id
  $soakPort = Get-FreePort
  $runtimeDir = Join-Path $art 'runtime'
  New-Item -ItemType Directory -Force -Path (Join-Path $runtimeDir 'logs') | Out-Null
  $stdout = Join-Path $art ("soak_stdout_{0}.log" -f $N.id)
  $stderr = Join-Path $art ("soak_stderr_{0}.log" -f $N.id)
  $csv = Join-Path $art ("soak_{0}.csv" -f $N.id)
  Remove-Item -LiteralPath $stdout, $stderr, $csv -Force -ErrorAction SilentlyContinue

  $gwBefore = Get-LiveGateway
  # Shared read-only corpora/models stay in the main tree; the node gateway reads
  # them through env overrides (relative config paths otherwise resolve against
  # the worktree and fail validation). Never junction/delete them.
  $sharedEnv = [ordered]@{
    AI_GGUF_MODELS_DIR = (Join-Path $script:RepoRoot 'GGUF_models')
    AI_ROBOTS_DIR      = (Join-Path $script:RepoRoot 'robots')
    AI_LEMMA_CSV       = (Join-Path $script:RepoRoot 'lemma.csv')
  }
  $envNames = @('CONTROLLER_PORT', 'AI_GATEWAY_HOST', 'AI_BASE_DIR', 'AI_FRONTEND_ENABLED',
    'AI_HTTP_LOG', 'AI_PID_FILE', 'AI_SELFMOD_NODE', 'PATH')
  foreach ($k in $sharedEnv.Keys) { if (Test-Path -LiteralPath $sharedEnv[$k]) { $envNames += $k } }
  $savedEnv = @{}
  foreach ($name in $envNames) {
    $savedEnv[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
  }
  # The worktree exe needs the main tree's runtime DLLs (python314.dll, poppler.dll,
  # mingw runtime); scope that to the soak child only, never to the session.
  $dllDirs = @($script:RepoRoot, (Join-Path $script:RepoRoot 'Python314'),
    (Join-Path $script:RepoRoot 'poppler-25.12.0\Library\bin'), 'C:\msys64\mingw64\bin') |
    Where-Object { $_ -and (Test-Path -LiteralPath $_) }
  [Environment]::SetEnvironmentVariable('PATH', (($dllDirs -join ';') + ';' + $savedEnv['PATH']), 'Process')
  [Environment]::SetEnvironmentVariable('CONTROLLER_PORT', [string]$soakPort, 'Process')
  [Environment]::SetEnvironmentVariable('AI_GATEWAY_HOST', '127.0.0.1', 'Process')
  [Environment]::SetEnvironmentVariable('AI_BASE_DIR', $runtimeDir, 'Process')
  [Environment]::SetEnvironmentVariable('AI_FRONTEND_ENABLED', 'false', 'Process')
  [Environment]::SetEnvironmentVariable('AI_HTTP_LOG', 'false', 'Process')
  [Environment]::SetEnvironmentVariable('AI_PID_FILE', (Join-Path $runtimeDir 'phoenix_main.pid'), 'Process')
  [Environment]::SetEnvironmentVariable('AI_SELFMOD_NODE', [string]$N.id, 'Process')
  foreach ($k in $sharedEnv.Keys) {
    if (Test-Path -LiteralPath $sharedEnv[$k]) { [Environment]::SetEnvironmentVariable($k, $sharedEnv[$k], 'Process') }
  }
  # runtime config was copied into the worktree at expand time; make sure it is there
  $cfgLocal = Join-Path $N.worktree 'config\phoenix.json'
  $cfgMain = Join-Path $script:RepoRoot 'config\phoenix.json'
  if ((-not (Test-Path -LiteralPath $cfgLocal)) -and (Test-Path -LiteralPath $cfgMain)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $cfgLocal) | Out-Null
    Copy-Item -LiteralPath $cfgMain -Destination $cfgLocal -Force
  }
  $proc = $null
  $windowStart = $null
  try {
    Write-Log ("soak {0}: launching sidecar gateway on rotated port {1} (live gateway port {2} untouched)" -f $N.id, $soakPort, $gwBefore.Port)
    $proc = Start-Process -FilePath $exe -WorkingDirectory $N.worktree `
      -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -WindowStyle Hidden
    $row = 'ts,elapsedSec,up,httpMs,rssMB'
    Add-Content -LiteralPath $csv -Value $row -Encoding utf8
    $healthy = $false
    $crashed = $false
    $down = 0
    $samples = 0
    $latencies = New-Object System.Collections.Generic.List[int]
    $firstHealthyElapsed = -1
    $soakStart = [System.Diagnostics.Stopwatch]::StartNew()
    $deadline = $GraceSec
    while ($soakStart.Elapsed.TotalSeconds -lt $deadline) {
      if ($proc.HasExited) { $crashed = $true; break }
      $sw = [System.Diagnostics.Stopwatch]::StartNew()
      $ok = Test-Health -Port $soakPort -TimeoutSec 5
      $sw.Stop()
      if ($ok) {
        $healthy = $true
        $firstHealthyElapsed = [math]::Round($soakStart.Elapsed.TotalSeconds, 1)
        break
      }
      Start-Sleep -Seconds ([math]::Max(1, [math]::Min(5, $IntervalSec)))
    }

    if ($healthy -and -not $crashed) {
      $windowStart = [System.Diagnostics.Stopwatch]::StartNew()
      while ($windowStart.Elapsed.TotalSeconds -lt $DurationSec) {
        if ($proc.HasExited) { $crashed = $true; break }
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $ok = Test-Health -Port $soakPort -TimeoutSec ([math]::Max(2, [math]::Min(10, $IntervalSec)))
        $sw.Stop()
        $samples++
        if (-not $ok) { $down++ } else { $latencies.Add([int]$sw.ElapsedMilliseconds) }
        $rssMb = -1
        try {
          $proc.Refresh()
          if (-not $proc.HasExited) { $rssMb = [math]::Round($proc.WorkingSet64 / 1MB, 1) }
        } catch { }
        $elapsed = [math]::Round($windowStart.Elapsed.TotalSeconds, 1)
        Add-Content -LiteralPath $csv -Value ("{0},{1},{2},{3},{4}" -f (Get-Date -Format 'HH:mm:ss'), $elapsed, [int]$ok, $sw.ElapsedMilliseconds, $rssMb) -Encoding utf8
        Start-Sleep -Seconds $IntervalSec
      }
    }
    if ($windowStart) { $windowStart.Stop() }
  } finally {
    if ($proc) {
      if (-not $proc.HasExited) {
        Write-Log ("soak {0}: stopping own sidecar pid {1}" -f $N.id, $proc.Id)
        Stop-OwnedProcessTree -OwnerPid $proc.Id
        try { $proc.WaitForExit(10000) | Out-Null } catch { }
      }
      $proc.Dispose()
    }
    foreach ($name in $envNames) {
      [Environment]::SetEnvironmentVariable($name, $savedEnv[$name], 'Process')
    }
  }

  $fatalPatterns = 'std::bad_alloc|GGML_ASSERT|Access violation|Segmentation fault|terminate called|std::terminate|fatal_runtime'
  $fatalHits = 0
  if (Test-Path -LiteralPath $stderr) {
    $fatalHits = @(Select-String -LiteralPath $stderr -Pattern $fatalPatterns -ErrorAction SilentlyContinue).Count
  }
  $startupExitCode = $null
  $startupErrTail = ''
  if (-not $healthy) {
    try {
      if ($proc -and $proc.HasExited) { $startupExitCode = $proc.ExitCode }
    } catch { }
    if (Test-Path -LiteralPath $stderr) {
      $startupErrTail = ((Get-Content -LiteralPath $stderr -Tail 6 -ErrorAction SilentlyContinue) -join "`n")
    }
    if (-not $startupErrTail -and (Test-Path -LiteralPath $stdout)) {
      $startupErrTail = ((Get-Content -LiteralPath $stdout -Tail 6 -ErrorAction SilentlyContinue) -join "`n")
    }
  }
  $latAvg = 0
  if ($latencies.Count -gt 0) { $latAvg = [math]::Round(($latencies | Measure-Object -Average).Average, 1) }

  $stability = 0.0
  if ($healthy) {
    $crashBad = if ($crashed) { 1 } else { 0 }
    $fatalBad = if ($fatalHits -gt 0) { 1 } else { 0 }
    $bad = $down + $crashBad + $fatalBad
    $denom = [math]::Max(1, $samples)
    $stability = [math]::Max(0.0, 1.0 - [math]::Min(1.0, ($bad / $denom)))
    $stability = [math]::Round($stability, 4)
  }

  $closedWith = if ($crashed) { 'crash' } else { 'clean-stop-by-harness' }
  $result = [pscustomobject]@{
    port             = $soakPort
    healthy          = $healthy
    firstHealthySec  = $firstHealthyElapsed
    crashed          = $crashed
    fatalHits        = $fatalHits
    samples          = $samples
    downSamples      = $down
    latencyAvgMs     = $latAvg
    stability        = $stability
    durationSec      = $DurationSec
    startupExitCode  = $startupExitCode
    startupErrTail   = $startupErrTail
    csv              = $csv
    stdout           = $stdout
    stderr           = $stderr
    closedWith       = $closedWith
  }
  Assert-GatewayIntact -Before $gwBefore -When "soak of $($N.id)" | Out-Null
  return $result
}

function Measure-TargetMetric {
  param([object]$N)
  if (-not $TargetMetricCommand) { return $null }
  $cmd = $TargetMetricCommand
  $wd = if ($N.id -eq 'main') { $script:RepoRoot } else { $N.worktree }
  try {
    $raw = & pwsh -NoProfile -NonInteractive -Command $cmd 2>$null
    $text = ($raw | ForEach-Object { "$_" }) -join "`n"
    $nums = [regex]::Matches($text, '-?\d+(?:\.\d+)?')
    if ($nums.Count -eq 0) { return $null }
    return [double]$nums[$nums.Count - 1].Value
  } catch {
    return $null
  }
}

function Invoke-Score {
  param([object]$N, [switch]$SkipSoak)
  if ((Get-Prop $N 'status' '') -eq 'pruned') { throw "node $($N.id) is already pruned" }
  $gwBefore = Get-LiveGateway
  $art = Get-NodeArtifacts $N.id
  $runner = Resolve-GtestRunner $N
  if (-not $runner) {
    $msg = "no usable gtest runner for $($N.id): the diff touches compiled sources, so the main-tree gtest_runner.exe is not a sound oracle. Re-run `score $($N.id) -BuildGtest` (per-node runner build, slow) or a human re-review."
    Write-Failure -NodeId $N.id -Stage 'score-runner' -Message $msg
    Update-NodeState -NodeId $N.id -Mutator {
      param($node, $tree)
      $node.status = 'unverified'
      Add-Evidence -N $node -Kind 'score' -Path $art -Summary 'UNVERIFIED: no sound gtest runner'
    }
    throw $msg
  }
  Write-Log ("scoring {0}: runner={1} provenance={2}" -f $N.id, $runner.Path, $runner.Provenance)

  $targetedLog = Join-Path $art ("gtest_targeted_{0}.log" -f $N.id)
  $wideLog = Join-Path $art ("gtest_wide_{0}.log" -f $N.id)
  $targeted = Invoke-GtestRun -Runner $runner -Filter $script:TargetedFilter -LogPath $targetedLog -TimeoutSec $TargetedTimeoutSec -Label 'targeted'
  $wide = Invoke-GtestRun -Runner $runner -Filter $script:WideFilter -LogPath $wideLog -TimeoutSec $WideTimeoutSec -Label 'wide'

  $passTotal = $targeted.Total + $wide.Total
  $passCount = $targeted.Passed + $wide.Passed
  $passRatio = if ($passTotal -gt 0) { $passCount / $passTotal } else { 0.0 }
  $passRatio = [math]::Round([math]::Max(0.0, [math]::Min(1.0, $passRatio)), 4)

  $soak = $null
  $soakStability = 0.0
  if (-not $SkipSoak) {
    $duration = if ($SoakSeconds -gt 0) { $SoakSeconds } else { [int]($SoakMinutes * 60) }
    $soak = Invoke-NodeSoak -N $N -DurationSec $duration -IntervalSec $IntervalSec -GraceSec $StartupGraceSec
    $soakStability = [double]$soak.stability
  } else {
    $soak = [pscustomobject]@{ stability = 0.0; sampleCount = 0; note = 'soak skipped' }
  }

  $targetDelta = 0.0
  if ($TargetMetricCommand) {
    $baseline = Get-Prop (Get-Prop (Read-Tree) 'baseline' $null) 'targetMetric' $null
    if ($null -eq $baseline) {
      $mainNode = Get-NodeById (Read-Tree) 'main'
      $baseline = Measure-TargetMetric -N $mainNode
      if ($null -ne $baseline) {
        Enter-StateLock
        try {
          $t = Read-Tree
          if (-not $t.baseline) { $t | Add-Member -NotePropertyName baseline -NotePropertyValue ([pscustomobject]@{ nodeId = 'main'; targetMetric = $null }) }
          $t.baseline.targetMetric = $baseline
          Save-Tree $t
        } finally { Exit-StateLock }
      }
    }
    $nodeMetric = Measure-TargetMetric -N $N
    if ($null -ne $baseline -and $null -ne $nodeMetric) {
      $denom = [math]::Max([math]::Abs([double]$baseline), 1e-9)
      $targetDelta = ($nodeMetric - $baseline) / $denom
      $targetDelta = [math]::Round([math]::Max(-1.0, [math]::Min(1.0, $targetDelta)), 4)
    }
  }

  $total = 0.5 * $passRatio + 0.3 * $soakStability + 0.2 * $targetDelta
  $total = [math]::Round([math]::Max(-1.0, [math]::Min(1.0, $total)), 4)

  $score = [pscustomobject]@{
    ts           = (Get-Date).ToString('o')
    total        = $total
    pass         = $passRatio
    soak         = $soakStability
    targetDelta  = $targetDelta
    weights      = [pscustomobject]@{ pass = 0.5; soak = 0.3; target = 0.2 }
    runner       = [pscustomobject]@{ path = $runner.Path; provenance = $runner.Provenance }
    tests        = [pscustomobject]@{
      targeted = $targeted
      wide     = $wide
      passed   = $passCount
      total    = $passTotal
    }
    soakDetail   = $soak
    targetMetric = $TargetMetricCommand
  }
  $scorePath = Join-Path $art ("score_{0}.json" -f $N.id)
  $utf8 = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText($scorePath, ($score | ConvertTo-Json -Depth 20), $utf8)

  Update-NodeState -NodeId $N.id -Mutator {
    param($node, $tree)
    $node.status = 'scored'
    $node.score = $score
    Add-Evidence -N $node -Kind 'gtest' -Path $targetedLog -Summary ("targeted {0}/{1} exit={2}" -f $targeted.Passed, $targeted.Total, $targeted.ExitCode)
    Add-Evidence -N $node -Kind 'gtest' -Path $wideLog -Summary ("wide {0}/{1} exit={2}" -f $wide.Passed, $wide.Total, $wide.ExitCode)
    if ($soak -and (Get-Prop $soak 'csv' $null)) {
      Add-Evidence -N $node -Kind 'soak' -Path $soak.csv -Summary ("stability={0} samples={1} down={2} port={3}" -f $soakStability, $soak.samples, $soak.downSamples, $soak.port)
    }
    Add-Evidence -N $node -Kind 'score' -Path $scorePath -Summary ("score={0} (pass={1} soak={2} target={3})" -f $total, $passRatio, $soakStability, $targetDelta)
  }

  Write-Log ("score {0}: total={1} = 0.5*{2} + 0.3*{3} + 0.2*{4}" -f $N.id, $total, $passRatio, $soakStability, $targetDelta)
  Write-Host ("  tests: targeted {0}/{1}, wide {2}/{3}, runner={4}" -f $targeted.Passed, $targeted.Total, $wide.Passed, $wide.Total, $runner.Provenance)
  if ($soak) {
    Write-Host ("  soak : port={0} healthy={1} samples={2} down={3} crash={4} stability={5}" -f (Get-Prop $soak 'port' 0), (Get-Prop $soak 'healthy' $false), (Get-Prop $soak 'samples' 0), (Get-Prop $soak 'downSamples' 0), (Get-Prop $soak 'crashed' $false), $soakStability)
  }
  Assert-GatewayIntact -Before $gwBefore -When "score of $($N.id)" | Out-Null
  return $score
}

# ---------------------------------------------------------------------------
# expand
# ---------------------------------------------------------------------------
function New-NodeId {
  param([object]$Tree)
  $max = 0
  foreach ($n in @($Tree.nodes)) {
    if ((Get-Prop $n 'id' '') -match '^n(\d+)$') {
      $v = [int]$Matches[1]
      if ($v -gt $max) { $max = $v }
    }
  }
  return ('n{0:d4}' -f ($max + 1))
}

function Invoke-Expand {
  param([string]$ParentId)
  $parentId = if ($ParentId) { $ParentId } else { 'main' }
  if (-not $Patch) { throw 'expand needs -Patch <unified diff file>' }
  $tree0 = Read-Tree
  $parent = Get-NodeById $tree0 $parentId
  if ($null -eq $parent) { throw "parent node not found: $parentId" }
  if ((Get-Prop $parent 'status' '') -in @('pruned', 'cleaned')) { throw "parent node $parentId is $(Get-Prop $parent 'status' '') - cannot expand it" }

  $patchInfo = Get-PatchPaths $Patch
  Assert-PatchAllowed $patchInfo
  $product = Test-ProductDiff ([pscustomobject]@{ files = $patchInfo.Paths })

  # allocate id + reserve tree entry under the lock
  $nodeId = $Id
  $branch = ''
  $wt = ''
  Enter-StateLock
  try {
    $tree = Read-Tree
    if (-not $nodeId) { $nodeId = New-NodeId $tree }
    if ($nodeId -notmatch '^[a-z0-9][a-z0-9_-]{1,40}$' -or $nodeId -eq 'main') {
      throw "invalid node id '$nodeId' (lowercase letters/digits/-/_, not 'main')"
    }
    if (Get-NodeById $tree $nodeId) { throw "node id already exists: $nodeId" }
    $slugBase = $Slug
    if (-not $slugBase) {
      $first = @($patchInfo.Paths)[0]
      $slugBase = [System.IO.Path]::GetFileNameWithoutExtension($first)
    }
    $slugSafe = ($slugBase.ToLowerInvariant() -replace '[^a-z0-9]+', '-').Trim('-')
    if (-not $slugSafe) { $slugSafe = 'change' }
    if ($slugSafe.Length -gt 28) { $slugSafe = $slugSafe.Substring(0, 28).Trim('-') }
    $stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss')
    $branch = 'selfmod/{0}-{1}-{2}' -f $stamp, $nodeId, $slugSafe
    $wt = Get-NodeWorktree $nodeId
    if (Test-Path -LiteralPath $wt) { throw "worktree path already exists: $wt" }

    $goalText = if ($Goal) { $Goal } else { "apply $([System.IO.Path]::GetFileName($patchInfo.FullPath))" }
    $node = [pscustomobject]@{
      id         = $nodeId
      parent     = $parentId
      goal       = $goalText
      status     = 'applied'
      createdBy  = 'selfmod_tree.ps1'
      createdAt  = (Get-Date).ToString('o')
      branch     = $branch
      baseBranch = (Get-Prop $parent 'branch' 'main')
      baseCommit = ''
      commit     = ''
      worktree   = $wt
      patch      = [pscustomobject]@{ path = $patchInfo.FullPath; bytes = $patchInfo.Size; paths = $patchInfo.Paths; productDiff = $product }
      diff       = [pscustomobject]@{ files = $patchInfo.Paths; added = 0; removed = 0; total = 0; productDiff = $product }
      score      = $null
      built      = $null
      evidence   = @()
    }
    Add-Evidence -N $node -Kind 'patch' -Path $patchInfo.FullPath -Summary ("{0} bytes, {1} file(s), productDiff={2}" -f $patchInfo.Size, @($patchInfo.Paths).Count, $product)
    $tree.nodes = @($tree.nodes) + $node
    Save-Tree $tree
  } finally {
    Exit-StateLock
  }

  # create the worktree+branch (harness is the only creator of selfmod/* branches)
  Write-Log ("creating worktree {0} on branch {1} (parent={2})" -f $wt, $branch, $parentId)
  try {
    Invoke-Git -Arguments @('worktree', 'add', '-b', $branch, $wt, (Get-Prop $parent 'branch' 'main')) | Out-Null
  } catch {
    Update-NodeState -NodeId $nodeId -Mutator {
      param($node, $tree) $node.status = 'expand_failed'
    }
    Write-Failure -NodeId $nodeId -Stage 'worktree-add' -Message "$_"
    throw
  }

  # apply the patch inside the worktree only
  try {
    $localPatch = Join-Path $wt ("build\tmp\selfmod\patch.{0}.diff" -f $nodeId)
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $localPatch) | Out-Null
    Copy-Item -LiteralPath $patchInfo.FullPath -Destination $localPatch -Force
    Invoke-Git -WorkDir $wt -Arguments @('apply', '--check', '--whitespace=nowarn', $localPatch) | Out-Null
    Invoke-Git -WorkDir $wt -Arguments @('apply', '--whitespace=nowarn', $localPatch) | Out-Null
  } catch {
    Write-Log ("patch did not apply -> removing worktree+branch for {0}" -f $nodeId)
    Invoke-Git -Arguments @('worktree', 'remove', '--force', $wt) -AllowFail | Out-Null
    Invoke-Git -Arguments @('branch', '-D', $branch) -AllowFail | Out-Null
    Update-NodeState -NodeId $nodeId -Mutator {
      param($node, $tree) $node.status = 'patch_failed'
    }
    Write-Failure -NodeId $nodeId -Stage 'patch-apply' -Message "$_"
    throw
  }

  # one commit per node branch (protocol: one candidate = one commit)
  Invoke-Git -WorkDir $wt -Arguments @('add', '-A') | Out-Null
  $staged = (Invoke-Git -WorkDir $wt -Arguments @('diff', '--cached', '--name-only')).Output
  foreach ($p in @($patchInfo.Paths)) {
    $rel = $p -replace '\\', '/'
    if (-not (($staged -split "`n") -contains $rel)) {
      $full = Join-Path $wt $rel
      if (Test-Path -LiteralPath $full) {
        # tracked-ignored files (e.g. new files under tools/) need -f to enter the commit
        Invoke-Git -WorkDir $wt -Arguments @('add', '-f', '--', $rel) -AllowFail | Out-Null
      }
    }
  }
  $commitMsg = "selfmod($nodeId): $goalText"
  Invoke-Git -WorkDir $wt -Arguments @('-c', 'user.name=phoenix-selfmod', '-c', 'user.email=selfmod@localhost',
    'commit', '-q', '-m', $commitMsg) | Out-Null
  $commitSha = (Invoke-Git -WorkDir $wt -Arguments @('rev-parse', 'HEAD')).Output.Trim()

  Copy-SharedInputs -N (Get-NodeById (Read-Tree) $nodeId)
  $stat = Get-DiffStat -N (Get-NodeById (Read-Tree) $nodeId)
  Update-NodeState -NodeId $nodeId -Mutator {
    param($node, $tree)
    $node.commit = $commitSha
    $node.baseCommit = (Invoke-Git -Arguments @('rev-parse', 'main')).Output.Trim()
    $node.diff = [pscustomobject]@{ files = $stat.files; added = $stat.added; removed = $stat.removed; total = $stat.total; productDiff = $stat.productDiff }
    Add-Evidence -N $node -Kind 'commit' -Path $wt -Summary ("commit $commitSha on $branch")
  }
  Write-Log ("node {0} created: branch={1} commit={2} diff=+{3}/-{4} files={5}" -f $nodeId, $branch, $commitSha.Substring(0, [math]::Min(10, $commitSha.Length)), $stat.added, $stat.removed, @($stat.files).Count)

  if (-not $NoBuild) {
    $nodeObj = Get-NodeById (Read-Tree) $nodeId
    $ok = Invoke-NodeBuild -N $nodeObj
    if (-not $ok) { throw "node $nodeId build failed; see runtime_store/selfmod/nodes/$nodeId/build_$nodeId.log" }
  } else {
    Write-Log ("node {0}: -NoBuild set; run: selfmod_tree.ps1 build {0}" -f $nodeId)
  }
  Write-Output $nodeId
}

# ---------------------------------------------------------------------------
# prune / select / cleanup
# ---------------------------------------------------------------------------
function Get-Children {
  param($Tree, [string]$NodeId)
  return @($Tree.nodes | Where-Object { (Get-Prop $_ 'parent' '') -eq $NodeId -and (Get-Prop $_ 'status' '') -notin @('pruned', 'cleaned') })
}

function Get-Frontier {
  param($Tree)
  $open = @($Tree.nodes | Where-Object {
      $nid = Get-Prop $_ 'id' ''
      $nid -ne 'main' -and (Get-Prop $_ 'status' '') -notin @('pruned', 'cleaned', 'failed', 'patch_failed', 'expand_failed', 'build_failed', 'unverified')
    })
  $frontier = New-Object System.Collections.Generic.List[object]
  foreach ($n in $open) {
    if (@(Get-Children $Tree $n.id).Count -eq 0) { $frontier.Add($n) }
  }
  return $frontier.ToArray()
}

function Get-ScoreValue {
  param([object]$N)
  $s = Get-Prop $N 'score' $null
  if ($null -eq $s) { return [double]::NegativeInfinity }
  $v = Get-Prop $s 'total' $null
  if ($null -eq $v) { return [double]::NegativeInfinity }
  return [double]$v
}

function Invoke-Prune {
  param([int]$Keep)
  if ($Keep -lt 1) { $Keep = 1 }
  $tree = Read-Tree
  $frontier = Get-Frontier $tree
  $scored = @($frontier | Where-Object { $null -ne (Get-Prop $_ 'score' $null) })
  $unscored = @($frontier | Where-Object { $null -eq (Get-Prop $_ 'score' $null) })
  if ($unscored.Count -gt 0) {
    Write-Log ("prune: {0} frontier node(s) have no score yet (kept, not ranked): {1}" -f $unscored.Count, (($unscored | ForEach-Object { $_.id }) -join ', '))
  }
  if ($scored.Count -eq 0) {
    Write-Log 'prune: no scored frontier nodes; nothing to do'
    return
  }
  $ranked = @($scored | Sort-Object @{ Expression = { Get-ScoreValue $_ }; Descending = $true },
    @{ Expression = { [int](Get-Prop (Get-Prop $_ 'diff' $null) 'total' 999999) }; Descending = $false },
    @{ Expression = { $_.id }; Descending = $false })
  $keepSet = New-Object 'System.Collections.Generic.HashSet[string]'
  foreach ($n in ($ranked | Select-Object -First $Keep)) { [void]$keepSet.Add([string]$n.id) }
  $drop = New-Object System.Collections.Generic.List[object]
  foreach ($n in $ranked) {
    $scoreVal = Get-ScoreValue $n
    if (-not $keepSet.Contains([string]$n.id)) { $drop.Add($n); continue }
    if ($scoreVal -lt $MinScore) { $drop.Add($n) }
  }

  Write-Host ''
  Write-Host ('{0,-8} {1,8} {2,8} {3,-42} {4}' -f 'node', 'score', 'diffln', 'goal', 'decision')
  Write-Host ('-' * 100)
  foreach ($n in $ranked) {
    $decision = if ($drop.id -contains $n.id) { 'PRUNE' } else { 'KEEP' }
    Write-Host ('{0,-8} {1,8} {2,8} {3,-42} {4}' -f $n.id, (Get-ScoreValue $n), (Get-Prop (Get-Prop $n 'diff' $null) 'total' 0), ((Get-Prop $n 'goal' '') -replace "`n", ' ').Substring(0, [math]::Min(42, ((Get-Prop $n 'goal' '') -replace "`n", ' ').Length)), $decision)
  }
  Write-Host ''

  if ($DryRun) {
    Write-Log ('prune: DRY RUN - would drop {0} node(s): {1}' -f $drop.Count, (($drop | ForEach-Object { $_.id }) -join ', '))
    return
  }

  foreach ($n in $drop) {
    $reason = if (-not $keepSet.Contains([string]$n.id)) { "outside top-$Keep frontier" } else { "score $(Get-ScoreValue $n) < minScore $MinScore" }
    Write-Log ("pruning {0} ({1}) - removing worktree {2} and branch {3}" -f $n.id, $reason, $n.worktree, $n.branch)
    Invoke-Git -Arguments @('worktree', 'remove', '--force', $n.worktree) -AllowFail | Out-Null
    if ((Get-Prop $n 'branch' '') -like 'selfmod/*') {
      Invoke-Git -Arguments @('branch', '-D', $n.branch) -AllowFail | Out-Null
    }
    Invoke-Git -Arguments @('worktree', 'prune') -AllowFail | Out-Null
    Update-NodeState -NodeId $n.id -Mutator {
      param($node, $tree)
      $node.status = 'pruned'
      Set-Prop $node 'prunedAt' ((Get-Date).ToString('o'))
      Set-Prop $node 'pruneReason' $reason
      Add-Evidence -N $node -Kind 'prune' -Path '' -Summary $reason
    }
  }
  Write-Log ("prune done: kept {0} frontier node(s)" -f $keepSet.Count)
}

function Get-NodePath {
  param($Tree, [string]$NodeId)
  $path = New-Object System.Collections.Generic.List[object]
  $cur = Get-NodeById $Tree $NodeId
  $guard = 0
  while ($null -ne $cur -and $guard -lt 100) {
    $path.Insert(0, $cur)
    $parentId = Get-Prop $cur 'parent' ''
    if (-not $parentId) { break }
    $cur = Get-NodeById $Tree $parentId
    $guard++
  }
  return $path.ToArray()
}

function Invoke-Select {
  $tree = Read-Tree
  $candidates = @($tree.nodes | Where-Object {
      $nid = Get-Prop $_ 'id' ''
      $nid -ne 'main' -and (Get-Prop $_ 'status' '') -notin @('pruned', 'cleaned')
    })
  $leaves = New-Object System.Collections.Generic.List[object]
  foreach ($n in $candidates) {
    if (@(Get-Children $tree $n.id).Count -eq 0 -and $null -ne (Get-Prop $n 'score' $null)) { $leaves.Add($n) }
  }
  if ($leaves.Count -eq 0) {
    Write-Log 'select: no scored leaf candidates; run expand/score first'
    return
  }
  $best = @($leaves | Sort-Object @{ Expression = { Get-ScoreValue $_ }; Descending = $true },
    @{ Expression = { [int](Get-Prop (Get-Prop $_ 'diff' $null) 'total' 999999) }; Descending = $false },
    @{ Expression = { $_.id }; Descending = $false })[0]
  $path = Get-NodePath $tree $best.id

  Write-Host ''
  Write-Host '=== best path (root -> leaf) ==='
  Write-Host ('{0,-8} {1,-8} {2,8} {3,8} {4,-46} {5}' -f 'node', 'parent', 'score', 'diffln', 'goal', 'status')
  Write-Host ('-' * 118)
  foreach ($n in $path) {
    $scoreVal = Get-ScoreValue $n
    $scoreText = if ([double]::IsNegativeInfinity($scoreVal)) { 'baseline' } else { "$scoreVal" }
    $goalText = ((Get-Prop $n 'goal' '') -replace "`n", ' ')
    Write-Host ('{0,-8} {1,-8} {2,8} {3,8} {4,-46} {5}' -f $n.id, (Get-Prop $n 'parent' ''), $scoreText, (Get-Prop (Get-Prop $n 'diff' $null) 'total' 0), $goalText.Substring(0, [math]::Min(46, $goalText.Length)), (Get-Prop $n 'status' ''))
  }
  Write-Host ''

  $repo = $script:RepoRoot
  $branch = Get-Prop $best 'branch' ''
  $commit = Get-Prop $best 'commit' ''
  $scoreVal = Get-ScoreValue $best
  $ts = (Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss')
  Write-Host '=== evidence ==='
  foreach ($n in $path) {
    if ($n.id -eq 'main') { continue }
    Write-Host ("  {0}: {1}" -f $n.id, (Get-Prop $n 'branch' ''))
    foreach ($e in @(Get-Prop $n 'evidence' @())) {
      if ($null -eq $e) { continue }
      Write-Host ("    - [{0}] {1}" -f (Get-Prop $e 'kind' ''), (Get-Prop $e 'summary' ''))
    }
  }
  Write-Host ''
  Write-Host '=== ADOPTION COMMANDS (printed only - the harness never runs merge/push/reset) ==='
  Write-Host ("# 1) review the candidate (no writes)")
  Write-Host ("git -C `"{0}`" log --oneline main..{1}" -f $repo, $branch)
  Write-Host ("git -C `"{0}`" diff --stat main...{1}" -f $repo, $branch)
  Write-Host ("git -C `"{0}`" diff main...{1}" -f $repo, $branch)
  Write-Host ''
  Write-Host ("# 2) adopt on main (choose ONE; both are local-only, no push)")
  Write-Host ("git -C `"{0}`" checkout main" -f $repo)
  Write-Host ("git -C `"{0}`" merge --no-ff `"{1}`" -m `"selfmod: adopt {2} (score={3})`"" -f $repo, $branch, $best.id, $scoreVal)
  Write-Host ("#   or replay the single candidate commit:")
  Write-Host ("git -C `"{0}`" cherry-pick {1}" -f $repo, $commit)
  Write-Host ''
  Write-Host ("# 3) re-verify on main (sidecar gtest build; never compile.bat while the long run is up)")
  Write-Host ("cd /d `"{0}`"" -f $repo)
  Write-Host 'cmd /c compile_gtest.bat'
  Write-Host ("gtest_runner.exe --gtest_filter=`"{0}`"" -f $script:TargetedFilter)
  Write-Host ("gtest_runner.exe --gtest_filter=`"{0}`"" -f $script:WideFilter)
  Write-Host ''
  Write-Host ("# 4) tag the adopted state")
  Write-Host ("git -C `"{0}`" tag selfmod/{1}-merged-{2}" -f $repo, $ts, $best.id)
  Write-Host ''
  Write-Host ("# 5) rollback if a human decides the merge was wrong (local only; not run by the harness)")
  Write-Host ("git -C `"{0}`" reset --hard selfmod/<previous-good-tag>" -f $repo)
  Write-Host ''

  Enter-StateLock
  try {
    $t = Read-Tree
    $t.selection = [pscustomobject]@{
      ts         = (Get-Date).ToString('o')
      node       = $best.id
      branch     = $branch
      commit     = $commit
      score      = $scoreVal
      path       = @($path | ForEach-Object { $_.id })
      adopted    = $false
      adoptedBy  = $null
    }
    Save-Tree $t
  } finally { Exit-StateLock }
  Write-Log ("select: best node = {0} (score {1}); adoption is a HUMAN step (not executed)" -f $best.id, $scoreVal)
}

function Invoke-Cleanup {
  param([switch]$All)
  if (-not $Force) { throw 'cleanup deletes worktrees/branches; re-run with -Force' }
  $tree = Read-Tree
  $targets = @($tree.nodes | Where-Object {
      $nid = Get-Prop $_ 'id' ''
      $nid -ne 'main' -and (Get-Prop $_ 'createdBy' '') -eq 'selfmod_tree.ps1' -and (Get-Prop $_ 'status' '') -notin @('pruned', 'cleaned')
    })
  if ($targets.Count -eq 0) {
    Write-Log 'cleanup: nothing to remove'
  }
  foreach ($n in $targets) {
    $wt = Get-Prop $n 'worktree' ''
    $branch = Get-Prop $n 'branch' ''
    if ($DryRun) {
      Write-Log ("cleanup DRY RUN: would remove {0} (worktree {1}, branch {2})" -f $n.id, $wt, $branch)
      continue
    }
    Write-Log ("cleanup: removing node {0} worktree={1} branch={2}" -f $n.id, $wt, $branch)
    if ($wt) { Invoke-Git -Arguments @('worktree', 'remove', '--force', $wt) -AllowFail | Out-Null }
    if ($branch -like 'selfmod/*') { Invoke-Git -Arguments @('branch', '-D', $branch) -AllowFail | Out-Null }
    Update-NodeState -NodeId $n.id -Mutator {
      param($node, $tree)
      $node.status = 'cleaned'
      Set-Prop $node 'cleanedAt' ((Get-Date).ToString('o'))
      Add-Evidence -N $node -Kind 'cleanup' -Path '' -Summary 'worktree+branch removed by harness cleanup'
    }
  }
  if (-not $DryRun) {
    Invoke-Git -Arguments @('worktree', 'prune') -AllowFail | Out-Null
    # remove now-empty per-node dirs, keep evidence files under runtime_store
    Get-ChildItem -LiteralPath $script:SelfmodRoot -Directory -ErrorAction SilentlyContinue | ForEach-Object {
      $hasWorktree = Test-Path -LiteralPath (Join-Path $_.FullName 'main.cpp')
      if (-not $hasWorktree) {
        $items = @(Get-ChildItem -LiteralPath $_.FullName -Recurse -File -ErrorAction SilentlyContinue)
        if ($items.Count -eq 0) { Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue }
      }
    }
  }
  Write-Host ''
  Write-Host '=== cleanup proof ==='
  Write-Host 'git worktree list:'
  Write-Host ((Invoke-Git -Arguments @('worktree', 'list')).Output)
  Write-Host "selfmod branches left: $((@(Invoke-Git -Arguments @('branch', '--list', 'selfmod/*') -AllowFail).Output -join '; ').Trim())"
  Write-Host 'git status --short:'
  $statusOut = (Invoke-Git -Arguments @('status', '--short')).Output
  if (-not $statusOut.Trim()) { Write-Host '(clean)' } else { Write-Host $statusOut }
  Assert-GatewayIntact -Before (Get-LiveGateway) -When 'cleanup' | Out-Null
}

function Invoke-Teardown {
  param([string]$NodeId)
  if (-not $Force) { throw 'teardown deletes a worktree/branch; re-run with -Force' }
  if (-not $NodeId -or $NodeId -eq 'main') { throw 'teardown needs a non-main node id' }
  $tree = Read-Tree
  $n = Get-NodeById $tree $NodeId
  if ($null -eq $n) { throw "node not found: $NodeId" }
  if ((Get-Prop $n 'createdBy' '') -ne 'selfmod_tree.ps1') { throw "node $NodeId was not created by this harness - refusing" }
  if ($DryRun) { Write-Log ("teardown DRY RUN: would remove node {0}" -f $NodeId); return }
  Write-Log ("teardown: removing node {0}" -f $NodeId)
  if ((Get-Prop $n 'worktree' '')) { Invoke-Git -Arguments @('worktree', 'remove', '--force', $n.worktree) -AllowFail | Out-Null }
  if ((Get-Prop $n 'branch' '') -like 'selfmod/*') { Invoke-Git -Arguments @('branch', '-D', $n.branch) -AllowFail | Out-Null }
  Invoke-Git -Arguments @('worktree', 'prune') -AllowFail | Out-Null
  Update-NodeState -NodeId $NodeId -Mutator {
    param($node, $tree)
    $node.status = 'cleaned'
    Set-Prop $node 'cleanedAt' ((Get-Date).ToString('o'))
  }
  Write-Log 'teardown done'
}

# ---------------------------------------------------------------------------
# init / status / selfcheck
# ---------------------------------------------------------------------------
function Invoke-Init {
  if ((Test-Path -LiteralPath $script:TreePath) -and -not $Force) {
    throw "tree.json already exists - use -Force to recreate (this forgets all nodes)"
  }
  New-Item -ItemType Directory -Force -Path $script:StateDir, $script:ArtifactsRoot, $script:SelfmodRoot | Out-Null
  $head = (Invoke-Git -Arguments @('rev-parse', 'HEAD')).Output.Trim()
  $tree = [pscustomobject]@{
    schema    = $script:SchemaVersion
    repo      = $script:RepoRoot
    createdAt = (Get-Date).ToString('o')
    updatedAt = (Get-Date).ToString('o')
    policy    = [pscustomobject]@{
      k             = $K
      minScore      = $MinScore
      weights       = [pscustomobject]@{ pass = 0.5; soak = 0.3; target = 0.2 }
      soakMinutes   = $SoakMinutes
      maxPatchBytes = $script:MaxPatchBytes
      allowPaths    = $script:AllowPrefixes
      denyPaths     = $script:DenyPrefixes
      hardRails     = @(
        'never run compile.bat (kills the live gateway)',
        'never stop gateway/llama-servers; soak uses own pid + AI_BASE_DIR on a rotated port',
        'never git push / reset --hard / rebase / rewrite history',
        'only delete selfmod/* branches created by this harness',
        'fail-closed: failures never touch main'
      )
    }
    baseline  = [pscustomobject]@{ nodeId = 'main'; targetMetric = $null; commit = $head }
    nodes     = @([pscustomobject]@{
        id        = 'main'
        parent    = $null
        goal      = 'baseline: live main tree'
        status    = 'baseline'
        branch    = 'main'
        worktree  = $script:RepoRoot
        commit    = $head
        score     = $null
        diff      = [pscustomobject]@{ files = @(); added = 0; removed = 0; total = 0; productDiff = $false }
        evidence  = @()
        createdBy = 'baseline'
        createdAt = (Get-Date).ToString('o')
      })
    selection = $null
  }
  Save-Tree $tree
  Write-Log ("initialized {0}" -f $script:TreePath)
  Write-Log ("baseline main @ {0}" -f $head.Substring(0, [math]::Min(10, $head.Length)))
}

function Invoke-Status {
  $tree = Read-Tree
  $gw = Get-LiveGateway
  $gwOk = if ($gw.Alive) { Test-Health -Port $gw.Port -TimeoutSec 3 } else { $false }
  Write-Host ''
  Write-Host ("repo        : {0}" -f $script:RepoRoot)
  Write-Host ("HEAD        : {0} ({1})" -f (Invoke-Git -Arguments @('rev-parse', 'HEAD')).Output.Trim().Substring(0, 10), (Invoke-Git -Arguments @('rev-parse', '--abbrev-ref', 'HEAD')).Output.Trim())
  Write-Host ("live gateway : port={0} pid={1} health={2}" -f $gw.Port, $gw.Pid, $gwOk)
  Write-Host ("state file  : {0}" -f $script:TreePath)
  Write-Host ''
  Write-Host ('{0,-8} {1,-8} {2,-13} {3,8} {4,7} {5,-42} {6}' -f 'node', 'parent', 'status', 'score', 'diffln', 'goal', 'branch')
  Write-Host ('-' * 126)
  foreach ($n in @($tree.nodes)) {
    $scoreVal = Get-ScoreValue $n
    $scoreText = if ([double]::IsNegativeInfinity($scoreVal)) { '-' } else { "$scoreVal" }
    $goalText = ((Get-Prop $n 'goal' '') -replace "`n", ' ')
    Write-Host ('{0,-8} {1,-8} {2,-13} {3,8} {4,7} {5,-42} {6}' -f $n.id, (Get-Prop $n 'parent' ''), (Get-Prop $n 'status' ''), $scoreText, (Get-Prop (Get-Prop $n 'diff' $null) 'total' 0), $goalText.Substring(0, [math]::Min(42, $goalText.Length)), (Get-Prop $n 'branch' ''))
  }
  Write-Host ''
  Write-Host 'worktrees:'
  Write-Host ((Invoke-Git -Arguments @('worktree', 'list')).Output)
  Write-Host ("selfmod branches: {0}" -f (((Invoke-Git -Arguments @('branch', '--list', 'selfmod/*') -AllowFail).Output) -join ', ').Trim())
}

function Invoke-SelfCheck {
  $ok = $true
  Write-Host ''
  Write-Host '=== selfmod_tree selfcheck (read-only) ==='
  if (-not (Test-Path -LiteralPath $script:TreePath)) { Write-Host 'state file  : MISSING (run init)'; $ok = $false } else { Write-Host ("state file  : OK {0}" -f $script:TreePath) }
  $gw = Get-LiveGateway
  $portFile = Join-Path $script:RepoRoot 'runtime_store\long_run\port.txt'
  if (Test-Path -LiteralPath $portFile) {
    $healthy = Test-Health -Port $gw.Port -TimeoutSec 5
    Write-Host ("live gateway : port={0} pid={1} health={2}" -f $gw.Port, $gw.Pid, $healthy)
    if (-not $healthy) { Write-Host '               (warning: live long run not answering - do not disturb it anyway)' }
  } else {
    Write-Host 'live gateway : no runtime_store/long_run/port.txt (no long run?)'
  }
  $head = (Invoke-Git -Arguments @('rev-parse', '--abbrev-ref', 'HEAD')).Output.Trim()
  $dirty = (Invoke-Git -Arguments @('status', '--short')).Output.Trim()
  Write-Host ("git          : branch={0} dirty={1}" -f $head, [bool]$dirty)
  if ($head -ne 'main') { Write-Host '               (note: main tree is not on main; branch-tree work is unaffected)' }
  Write-Host ("worktrees    : {0}" -f (((Invoke-Git -Arguments @('worktree', 'list')).Output) -split "`n").Count)
  $gxx = 'C:\msys64\mingw64\bin\g++.exe'
  Write-Host ("g++          : {0}" -f (Test-Path -LiteralPath $gxx))
  $cflags = Resolve-ConanFile 'conan_cflags_*.txt' -MainChannel
  $libs = Resolve-ConanFile 'conan_libs_*.txt' -MainChannel
  $sources = Join-Path $script:RepoRoot 'build\compile_sources.txt'
  Write-Host ("conan cflags : {0}" -f $cflags)
  Write-Host ("conan libs   : {0}" -f $libs)
  Write-Host ("source list  : {0}" -f (Test-Path -LiteralPath $sources))
  if (-not $cflags -or -not $libs -or -not (Test-Path -LiteralPath $sources)) { $ok = $false }
  $mainRunner = Join-Path $script:RepoRoot 'gtest_runner.exe'
  Write-Host ("gtest runner : {0}" -f (Test-Path -LiteralPath $mainRunner))
  $compiling = Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'compile(_gtest)?\.bat' }
  $buildState = if ($compiling) { 'RUNNING (wait before expanding)' } else { 'idle' }
  Write-Host ("main build   : {0}" -f $buildState)
  if ($compiling) { $ok = $false }
  $forbidden = @('push', 'reset', 'rebase')
  Write-Host ("hard rails   : sidecar build only; selfmod/* branch scope; no {0} in harness" -f ($forbidden -join '/'))
  Write-Host ''
  if (-not $ok) { Write-Host 'SELFCHECK: FAIL' } else { Write-Host 'SELFCHECK: PASS' }
}

# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
try {
  switch ($Command) {
    'init' { Invoke-Init }
    'selfcheck' { Invoke-SelfCheck }
    'status' { Invoke-Status }
    'expand' { Invoke-Expand -ParentId $Node }
    'build' {
      if ($Node -eq 'main') { throw 'build needs a non-main node id' }
      $tree = Read-Tree
      $n = Get-NodeById $tree $Node
      if ($null -eq $n) { throw "node not found: $Node" }
      if ($BuildGtest) { Invoke-NodeGtestBuild -N $n | Out-Null }
      $ok = Invoke-NodeBuild -N $n
      if (-not $ok) { exit 2 }
    }
    'score' {
      if ($Node -eq 'main') { throw 'score needs a non-main node id' }
      $tree = Read-Tree
      $n = Get-NodeById $tree $Node
      if ($null -eq $n) { throw "node not found: $Node" }
      if ($BuildGtest -and -not (Test-Path -LiteralPath (Join-Path $n.worktree ("gtest_runner.{0}.exe" -f $n.id)))) {
        Invoke-NodeGtestBuild -N $n | Out-Null
      }
      Invoke-Score -N $n | Out-Null
    }
    'prune' { Invoke-Prune -Keep $K }
    'select' { Invoke-Select }
    'cleanup' { Invoke-Cleanup -All }
    'teardown' { Invoke-Teardown -NodeId $Node }
    default { throw "unknown command: $Command" }
  }
} catch {
  Write-Host ''
  Write-Host ("ERROR: {0}" -f $_.Exception.Message) -ForegroundColor Red
  exit 1
}
