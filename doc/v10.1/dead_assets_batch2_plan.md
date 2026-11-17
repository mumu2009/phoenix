# Dead-assets Batch 2 — executable plan (not executed)

Status: **planned only** — Batch 1 was executed on 2026-11-17/18 (see
`build/tmp/dead_assets_executed.json`). Batch 2 rows need a controlled window
because the live gateway (pid 23920, :5097) or the external self-mod harness owns
them. Nothing in this file has been run.

Source of truth: `build/tmp/dead_assets.json` rows H1, H2, H3, G1, C6 (optional),
H8 (decision-only). Live gateway at planning time: pid **23920**, `:5097` healthy;
frontend child **13032**; 3 llama-servers; self-mod candidate harness active.

Legend: reclaim estimates use the sizes measured 2026-11-17 ~23:30.

---

## 0. Order that minimises gateway downtime

| Phase | When | Rows | Gateway impact |
|---|---|---|---|
| **P0** | any time, no window needed | H2 (stale logs), H3 (state backups) | none |
| **P0.5** | after Lead stops the self-mod candidate harness | G1 (mission_workspace prune) | none (but do not race writes) |
| **P1** | **single controlled swap window**, bundled with the next binary change | H1 (`runtime_debug.log` 35 GB rotation) + permanent logger rotation | one restart only — the same ~3 s binary swap, plus health gate |
| **P2** | later, separate approval | C6 resnet18 `.bin` + config edit (optional, 11.9 MB); H8 review | none (C6 needs config edit) |

Rationale: H1 is the only 10s-of-GB item and it is an **open Windows file** — it can
only be renamed/truncated while the logger is closed, i.e. at a gateway restart.
That restart should be the **same restart that installs the next binary change**
(`phoenix_main.next.exe` → `phoenix_main.exe`), so the gateway pays one downtime, not
two. P0 rows are file-only and should be finished *before* the window so the window
contains nothing but rename+swap+start. The P1 restart also naturally resets the
in-memory `missions_` map (~9,758 entries, ~>100 MB), which is a bonus of G1.

---

## P0-1 — H2: delete stale `runtime_store/logs/*` except the two kept files

**Expected reclaim:** ~1.9 MB (37 files; `llama_8082.err.log` 1.1 MB,
`runtime_release.log` 7.2 MB, rest tiny).

**Preconditions:** none. Only requirement: keep `fatal_runtime.log` (crash channel
used by `tools/stability_run.ps1`) and leave `runtime_debug.log` for P1.

```powershell
$root = 'E:\079\v6.0Alixander\phoenix'
$logs = Join-Path $root 'runtime_store\logs'
$keep = @('fatal_runtime.log','runtime_debug.log')
# dry run first
Get-ChildItem -LiteralPath $logs -File | Where-Object { $keep -notcontains $_.Name } |
  Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
# delete (after eyeballing the list)
Get-ChildItem -LiteralPath $logs -File | Where-Object { $keep -notcontains $_.Name } |
  Remove-Item -Force
```

**Risk:** low — historical session logs only; they are ≤2026-10-18. If any is cited
by a handoff doc, copy it to `build/tmp/old_logs/` first (a few MB).

---

## P0-2 — H3: prune `autonomy_state*.json` backups (106 files, 41 MB)

**Expected reclaim:** ~40 MB.

**Preconditions:** none; the live state file is written by the gateway but the
backups are not open. Keep **live `autonomy_state.json` + the 2 newest** snapshots.

```powershell
$root = 'E:\079\v6.0Alixander\phoenix'
$dir  = Join-Path $root 'runtime_store'
$live = 'autonomy_state.json'
$all  = Get-ChildItem -LiteralPath $dir -File -Filter 'autonomy_state*' |
        Where-Object { $_.Name -ne $live } | Sort-Object LastWriteTime -Descending
$keep = $all | Select-Object -First 2
# dry run
$all | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize
$keep | Select-Object Name,LastWriteTime | Format-Table -AutoSize
# delete everything except the 2 newest
$all | Where-Object { $keep.FullName -notcontains $_.FullName } | Remove-Item -Force
```

**Risk:** low — loses old rollback points; 2 newest keep the useful ones. Never touch
the live `autonomy_state.json` while the gateway runs (it rewrites on persist).

---

## P0.5 — G1: prune `runtime_store/mission_workspace/` (38,001 dirs / 8.7 MB)

**Expected reclaim:** 8.7 MB bytes + ~38k directory entries (listing pressure is the
real win). The in-memory mission history (~>100 MB) is only released by the P1
restart, not by this step.

**Preconditions:**
1. Lead has **stopped the self-mod candidate harness** (the external client that
   POSTs `/api/mission/assign` + `/api/mission/report`; see audit §3). New mission
   workspace dirs are being created every few seconds while it runs (newest
   `tr-10065-29` at 23:27:20 during the audit).
2. Ideally no **running** mission (status showed running=6 earlier); if some are
   running, their ids are in `autonomy_state.json` and are kept by the rule below;
   a running mission with a *new* id could get its fresh dir deleted only if
   modified >24 h ago — impossible — so the 24 h guard covers the race.
3. Dry run first; it prints what would be removed.

```powershell
$root  = 'E:\079\v6.0Alixander\phoenix'
$ws    = Join-Path $root 'runtime_store\mission_workspace'
$state = Get-Content -Raw (Join-Path $root 'runtime_store\autonomy_state.json') | ConvertFrom-Json
$keepIds = @{}
foreach ($m in $state.missions) { $keepIds[[string]$m.id] = $true }   # 2 running + 64 recent finished
$cut = (Get-Date).AddHours(-24)
$dirs = Get-ChildItem -LiteralPath $ws -Directory
$del  = $dirs | Where-Object { -not $keepIds.ContainsKey($_.Name) -and $_.LastWriteTime -lt $cut }
# dry run
"total=$($dirs.Count) keepById=$($keepIds.Count) toDelete=$($del.Count)"
$del | Select-Object -First 20 Name,LastWriteTime | Format-Table -AutoSize
# execute only after the dry run looks right
$del | Remove-Item -Recurse -Force
```

**Keep:** all dirs whose id is in `autonomy_state.json.missions[].id`, all dirs
modified in the last 24 h, `runtime_store/selfmod/tree.json` (outside the workspace,
untouched).
**Risk:** medium-low — deletes per-mission deliverable history of old SELF-MOD
candidate experiments (already judged Failed, low value). Do **not** run while the
harness is writing; if a dir is being written, `Remove-Item` may partially fail —
re-run later. No gateway downtime.

---

## P1 — H1: rotate the 35 GB `runtime_debug.log` in the next binary-swap window

**Expected reclaim:** ~36.7 GB at execution (35,011.10 MiB on 2026-11-17; the file
grows ~50 MB/h, so capture the exact size in the window).

**Why the window:** `loggerCXX.cpp:91–93` opens
`runtime_store/logs/runtime_debug.log` with `std::ios::out | std::ios::app` and never
rotates. On Windows an open file can be renamed only if the writer opened it with
`FILE_SHARE_DELETE` — .NET/PowerShell `Move-Item` on a file held by the C++ logger is
unreliable. The safe sequence is: **stop gateway → process exited (handle closed) →
rename → start new binary**. This is the same stop/start as the next binary swap, so
the incremental downtime is ~0 s beyond the swap itself.

**Preconditions (Lead):**
1. Next binary change is ready (`phoenix_main.next.exe` built and validated).
2. Swap procedure/runbook approved (this is the only gateway restart in the plan).
3. Optional same-window source fix: add bounded rotation to `loggerCXX.cpp`
   (e.g. `maxLogBytes=256 MiB`, on exceed rename to `runtime_debug.log.1` and reopen)
   so this 35 GB cannot regrow. This is a **source change**, separate approval.

**Exact sequence (run as one block in the window):**

```powershell
$root = 'E:\079\v6.0Alixander\phoenix'
$logs = Join-Path $root 'runtime_store\logs'
$log  = Join-Path $logs 'runtime_debug.log'
$old  = Join-Path $logs ('runtime_debug.log.' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.old')

# 0) evidence before stopping: size + a cheap head/tail sample (35 GB tail read is slow,
#    keep it bounded; fatal_runtime.log remains the crash-forensics channel)
$fi = Get-Item -LiteralPath $log
"log bytes before = $($fi.Length)  lastWrite = $($fi.LastWriteTime)"
Get-Content -LiteralPath $log -TotalCount 200 | Set-Content -Encoding UTF8 (Join-Path $root 'build\tmp\runtime_debug_head_sample.txt')
Get-Content -LiteralPath $log -Tail 200       | Set-Content -Encoding UTF8 (Join-Path $root 'build\tmp\runtime_debug_tail_sample.txt')

# 1) stop gateway + frontend child in the approved window (NEVER during normal operation)
Stop-Process -Id 13032 -Force -ErrorAction SilentlyContinue   # frontend-only child
Stop-Process -Id 23920 -Force -ErrorAction SilentlyContinue   # gateway (listener :5097)
# wait until the port is free and the processes are gone (max ~5 s)
$t0 = Get-Date
while ((Get-NetTCPConnection -LocalPort 5097 -State Listen -ErrorAction SilentlyContinue) -and ((Get-Date) - $t0).TotalSeconds -lt 5) { Start-Sleep -Milliseconds 200 }

# 2) rotate the log (file is now closed) + pre-create the new one
Move-Item -LiteralPath $log -Destination $old -Force
New-Item -ItemType File -Path $log -Force | Out-Null

# 3) install the new binary (same window)
Copy-Item -LiteralPath (Join-Path $root 'phoenix_main.next.exe') -Destination (Join-Path $root 'phoenix_main.exe') -Force

# 4) start the gateway exactly as the normal runbook does, then health-gate
#    (the runbook start command: phoenix_main.exe; it reopens runtime_debug.log in append mode)
Start-Process -FilePath (Join-Path $root 'phoenix_main.exe') -WorkingDirectory $root
$deadline=(Get-Date).AddSeconds(60); $ok=$false
while((Get-Date) -lt $deadline){ try { $h=Invoke-WebRequest 'http://127.0.0.1:5097/api/health' -UseBasicParsing -TimeoutSec 3; if($h.StatusCode -eq 200){$ok=$true;break} } catch {}; Start-Sleep -Milliseconds 500 }
"health ok = $ok"

# 5) prove the new log is being written, then delete the old snapshot
Start-Sleep -Seconds 60
Get-Item -LiteralPath $log | Select-Object FullName,Length,LastWriteTime
if (Test-Path -LiteralPath $log -and (Get-Item -LiteralPath $log).Length -gt 0) {
    Remove-Item -LiteralPath $old -Force
    "old log deleted; reclaimed ~$($fi.Length) bytes"
} else { "NEW LOG EMPTY/ABSENT -> keep $old, DO NOT DELETE; investigate logger" }
```

**Risk / rollback:**
- *Risk:* losing 35 GB of Debug history. Mitigation: 200-line head+tail samples kept
  in `build/tmp/`; `fatal_runtime.log` (crash channel) is untouched; the log had no
  reader in code.
- *Risk:* gateway fails to start after swap → roll back binary
  (`Copy-Item phoenix_main.prev.exe phoenix_main.exe`) and rename the old log back
  (`Move-Item $old $log -Force`). Health gate is the go/no-go.
- The rename+new-file step must happen only while the process is fully exited;
  otherwise Windows may return "file in use". If `Move-Item` fails, abort the swap and
  restart the old binary — no partial state (the new binary copy happens after the
  successful rename).
- Optional permanent guard: `loggerCXX.cpp` rotation (source change) in the same
  build; otherwise this row regrows at ~50 MB/h.

---

## P2 (optional, separate approval) — C6 resnet18 `.bin` + config edit (11.9 MB)

**Expected reclaim:** 11,932,146 B. **Preconditions:** decide the ResNet18 BPU
archive is dead (today the image BPU encode path is a stub — `VideoHbdnnModel::encode`
returns "BPU image encoder preprocessing is not implemented in this build" — so the
`.bin` files are archive-only). Must edit config in the same change or leave a
dangling key:

- delete `runtime_store/models/ijepa/resnet18_224/model_encoder.bin` (8,325,447 B)
- delete `runtime_store/models/ijepa/resnet18_224/model_decoder.bin` (3,606,699 B)
- edit `config/phoenix.json:778–779` and `config/phoenix.example.json:672–673`:
  either remove `jepa.image.horizonModel` / `horizonDecoderModel` (the BPU override
  falls back to `resolveBpuModelPath` search) or point them at another kept `.bin`.
- keep `model.manifest.json` (provenance).
**Risk:** low functional, medium config-hygiene — if the edit is forgotten the keys
point at missing files (only matters on a BPU box; host runs `cpu`). Value is only
12 MB → **recommend deferring** unless the config edit is already scheduled.

---

## P2 (decision only) — H8 `runtime_store/tmp/` 2.87 GB

Ownership of 858 one-off probe scripts/results is unclear (earlier sessions, some
probe harnesses may still point here). Not scheduled. When decided: separate
inventory, keep anything referenced by `tools/stability_run.ps1` /
`tools/selfmod_tree.ps1` / `build/tmp/*`, delete the rest.

---

## Expected totals

| Row | Reclaim | Gateway downtime |
|---|---|---|
| H1 runtime_debug.log | ~36.7 GB | shares the next binary swap (~3 s + health gate) |
| H2 stale logs | ~1.9 MB | 0 |
| H3 autonomy_state backups | ~40 MB | 0 |
| G1 mission_workspace | 8.7 MB + 38k dirs | 0 |
| C6 resnet `.bin` (optional) | 11.9 MB | 0 |
| **Total scheduled** | **~36.75 GB** | one restart, already needed for the binary swap |
