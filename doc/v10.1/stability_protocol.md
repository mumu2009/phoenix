# v10.1 stability protocol — repeatable 3-hour run with structured evidence

Status: active (harness `tools/stability_run.ps1`, protocol revision **1**, 2026-11-17).
Owner: whoever is on stability duty for the current build.

This document defines **how to run a stability measurement, what the verdicts
mean, and how to compare two builds**. The output of every run is a
self-contained directory under `runtime_store/stability/<ts>[-label]/`.

---

## 1. Why this exists: a 30-minute soak is not a stability result

A long run with zero crashes does **not** prove the absence of crashes; it
bounds the crash **rate** from above. With zero observed events, the 95 %
one-sided upper bound is the **rule of three** (Hanley & Lipp 1983):

```
lambda_95upper = 3 / exposure_hours          [crashes per hour, zero events]
```

| exposure | zero-crash 95 % lambda upper | rung | what it means |
|---|---|---|---|
| 30 min | < 6 / h | **below S1** | not a stability result; harness self-test only |
| 3 h | < 1 / h | **S1** | "no new problem found" — cannot claim "stable" |
| 12 h | < 0.25 / h | **S2** | significant reduction vs the S1 baseline |
| 30 h+ | < 0.1 / h | **S3** | may relax the coarse lock |

**Verdict discipline**

- A run shorter than **3 h of observed exposure** is labelled
  `SELF-TEST (below S1)` — all checks can be green, but the summary never says
  "stable". This is deliberate: a 8-minute dry run validates the harness, not
  the build.
- The lambda bound and the S-rung are printed in every `summary.md` and are
  computed from **exposure**, never from wall-clock.
- Zero crashes only bounds the rate. The S-rungs are comparison statements
  ("no new problem found", "significant reduction"), not proofs.

### Exposure is not wall-clock

Exposure = **sum over samples of `intervalSec` where the gateway answered
(`up`) AND was ticking**, split by **gateway PID**. A gateway restart closes the
current window and starts a new per-PID window, so a flapping gateway cannot
accumulate hours across restarts. `ticking` is defined as any of:

1. the gateway process' total CPU time advanced by > 50 ms during the interval,
2. the supervisor's training rounds advanced (`training.log` `phase=train`),
3. the persisted autonomy `iteration` counter advanced.

A process that listens on the port but is wedged (0 CPU growth) therefore does
**not** buy exposure.

The summary reports: exposure hours, per-PID sample counts, restart count,
PID changes and the **max continuous serving window**.

---

## 2. What one run does

`tools/stability_run.ps1` performs, in order:

1. **Preflight** — locate the live gateway (`runtime_store/long_run/port.txt`,
   fallback probe 5097/5096/5098/5080), verify `/api/health`, snapshot the build
   manifest (`phoenix_main.exe` sha256/mtime/size, `version.hpp`,
   `phoenix_main.map`, llama-server paths), baseline mission counters, baseline
   log offsets, baseline MMLU line count, baseline expert-weights mtime.
2. **Lifecycle driver setup** — start the **sidecar** gateway (or fall back to
   the live gateway, labelled) and assign one probe mission.
3. **Sampling loop** — every `-IntervalSec` (default 60 s) until the expected
   sample count is collected (hard-capped at planned + grace). See §3.
4. **Lifecycle state machine** — advanced on each sample: assign → Running →
   work evidence → report → verified Completed. See §4.
5. **Finalize** — final crash scan + forensics, final authoritative mission
   counters, population/expert deltas, end-of-run health + PID, then
   `summary.json` + `summary.md`, cleanup record, sidecar teardown.

The live gateway/servers are **only observed**. The harness never stops,
restarts or reconfigures them (hard rule, also printed in `cleanup.json`).

---

## 3. Metrics collected every sample

| metric | source | notes |
|---|---|---|
| gateway up? + HTTP latency | `GET /api/health` with `-HealthTimeoutSec` (default 15 s) | identity rule below |
| GPU utilisation mean/max, mem, temp | 3× `nvidia-smi --query-gpu=utilization.gpu,...` ~350 ms apart | mean/max reported, **no threshold** (workload-dependent) |
| gateway RSS / threads / CPU % | `Get-Process` on the true gateway PID; CPU % from TotalProcessorTime delta | OS truth, no HTTP cost |
| zombie images | `phoenix_main` processes with RSS < 100 MB **and** 0 threads | excluded from identity and up-rate |
| llama-server RSS | sum of `Get-Process llama-server` working sets | 8B + coder + granite |
| crash events | WER `Application` log, `Id=1000`, `StartTime >= runStart`, message mentions the app | per-event forensics, §5 |
| restart events | listener PID change + appended `gateway_loop.log` lines (`gateway started`, `exited`, `unresponsive`) + `supervisor.log` restart lines | restarts reset exposure |
| llama slot-wait failures | appended `gateway.log` lines containing `llama-slot-busy` (and the same in `gateway.err.log`) | informational, expected under slot contention |
| mission state | persisted `runtime_store/autonomy_state.json` (**cheap**, fresh ~15 s, all RUNNING + last 64 finished) + authoritative `/api/mission/status` every `-MissionStatusEverySec` (default 300 s, **heavy** ~1.5-3 s) | the heavy call is deliberately not per-sample |
| expert weights mtime | `runtime_store/experts/study_tiny/weights.json` | age + update count |
| population `goalsRun` | `studyPopulation.goalsRun` in `autonomy_state.json` | delta over run |
| MMLU history | `runtime_store/long_run/mmlu_history.jsonl` lines appended during the run | **context only**, §6 |
| lifecycle state | sidecar `/api/mission/status` (+ workspace `deliverable.md`) or persisted state for live fallback | §4 |

**Gateway identity rule (zombie-safe).** The live gateway is the owner of the
`127.0.0.1:<port>` listener **only if that process has RSS ≥ 100 MB**. The
process table routinely carries ~7 zombie `phoenix_main` images (0 threads,
~32 KB RSS from old crashes; they may even hold a leaked listening socket). If
the listener owner fails the RSS test the harness marks `listenerZombie` and
uses the max-RSS `phoenix_main` instead. The zombie count is recorded each
sample. A legitimate frontend child (16 MB but 17 threads, active CPU) is **not**
classified as a zombie.

---

## 4. The mission lifecycle (sidecar by default)

Requirement: the run must cover at least one full mission lifecycle —
**assign → work → report/complete**.

Cheap/Safe evidence is produced on a **sidecar gateway instance**, not on the
live one, so the live mission set and state file are not touched:

- one copy of `phoenix_main.exe` under a **new name**
  (`phoenix_stab_<label>_<hhmmss>.exe`) in `<bundle>/sidecar/` — this avoids the
  live `supervisor.ps1` / `gateway_loop.ps1`, which reap by the name
  `phoenix_main`;
- its own working directory and its own `runtime_store` (`AI_BASE_DIR`), so its
  `runtime_store/phoenix_main.pid` single-instance guard is isolated and relative
  stores (`lmdb/`, `runtime_store/mission_workspace/`) are isolated too; the
  repo root is still discoverable upward, so the real `config/phoenix.json` is
  used;
- its own port in **5100–5199** (`CONTROLLER_PORT`), `AI_RESEARCH_PROFILE=full_components`,
  `AI_ROBOTS_AUTOLOAD=false`, `AI_FRONTEND_ENABLED=false`,
  `AI_BASE_DIR=<sidecar>/runtime_store`, repo root prepended to `PATH` so the
  root DLLs resolve;
- `runtime_store/corpus_pretrain_stamp.json` is copied in so the sidecar skips
  the corpus-pretrain startup work;
- the mission loop interval is configured to **20 s** (`POST
  /api/cognition/autonomy/loop {action:configure, intervalSec:20}`) so the
  sidecar does not add avoidable load to the shared llama-server;
- it talks to the same local llama-server as the live gateway (that is the point
  of a *sidecar*: same platform, separate gateway).

Lifecycle driver steps and evidence:

1. `POST /api/mission/assign` `{id:"stab-<run>-m1", goal:"STABILITY-PROBE ...", pressureMode:"linear"}`.
2. Poll `/api/mission/status`; record when `state=1` (Running).
3. **work evidence**: a non-empty `deliverable` — the in-memory field *or* the
   on-disk `mission_workspace/<id>/deliverable.md` (file-edit replies are written
   to the file first, so the file is the authoritative work signal).
4. `POST /api/mission/report` `{missionId, goalAchieved:true}`.
5. Poll until `state=2` (**Completed**) and record the verification.

If work never appears (e.g. LLM slot starvation), the harness still exercises
report → Completed after `-LifecycleTimeoutSec` (default 900 s) and **states in
the summary that work evidence was not observed**. It never fabricates the work
step.

**Sidecar cleanup.** After the lifecycle completes (or at end of run), the
harness `taskkill /T /F`s the sidecar PID, verifies no process image under the
sidecar dir remains, copies the sidecar `autonomy_state.json`, mission workspace
and stdout/stderr logs into `<bundle>/lifecycle/`, then removes the sidecar work
dir. `<bundle>/cleanup.json` records the PID/exe/dir first so
`tools/stability_run.ps1 -CleanupOnly -BundleDir <bundle>` can clean up after an
interrupt (Ctrl-C may bypass the finally block).

**Live fallback (labelled).** With `-LifecycleMode auto` (default) the harness
falls back to assigning on the **live** gateway if the sidecar cannot become
healthy within `-SidecarStartupTimeoutSec`. `-LifecycleMode live` forces that
mode. The summary then states `Lifecycle driver: live ... FALLBACK: ...` and the
mission id keeps the supervisor-compatible `tr-<int>-<n>` shape (the supervisor
parses ids as `tr-<int>-<n>`; a non-numeric id could break its retirement sort).
Use fallback only when sidecar startup is impossible; it perturbs the live
mission set.

`-LifecycleMode none` is for observation-only runs and fails C4/C5 by design.

---

## 5. Pass/fail criteria (explicit, revision 1)

| # | criterion | threshold | why |
|---|---|---|---|
| **C1** | crash events during the run | **0** | a crash is the only unambiguous stability failure; zero events still only bounds the rate (§1). **Two evidence channels** are required — see §7: WER `Application` Id=1000 (fail-fast codes such as `0xC0000374`) **and** `runtime_store/logs/fatal_runtime.log` unhandled-exception lines (ordinary SEH faults such as `0xC0000005` are swallowed by the gateway's own filter and never reach WER) |
| **C2** | gateway up-rate | **≥ 0.99** | ~1.8 missed samples per 180 allowed for a legitimate stall; a dead gateway shows immediately. Zombie images are ignored (identity rule) |
| **C3** | longest run of consecutive up-misses | **≤ 2 samples** | `gateway_loop.ps1` only kills an unresponsive gateway after 5 consecutive unhealthy minutes; a ≥3-miss run at 60 s would be relying on the wrapper to mask a dead gateway |
| **C4** | missions reaching **Completed** | **≥ 1** | proves the state machine reaches Completed through a real assign→work→report path; if impossible the summary says why |
| **C5** | lifecycle steps recorded | `assign > running > work > report > completed` all present | the protocol is specifically about a full lifecycle, not just a state flag |
| **C6** | sample coverage | **≥ 0.99** collected/expected | guards against the harness itself stalling; bad density invalidates the other numbers |
| **C7** | GPU mean/max reported | metric present; **no util threshold** | GPU utilisation is dominated by the concurrent training/mission workload; a threshold would encode the workload, not stability. GPU mem/temp are context |
| **C8** | exposure + lambda bound reported | exposure hours (up AND ticking, per PID), restarts, max window, `lambda_95upper` | wall-clock is not exposure; a flapping gateway must not buy hours |

Deliberately **not** pass/fail criteria:

- **gateway RSS growth** (reported first/last/max) — allocator/state growth is
  expected; a large jump is a manual triage trigger, not an automatic fail.
- **llama-slot-busy count** (reported) — expected under slot contention and the
  low-priority mission path is designed to yield.
- **MMLU / learning metrics** (context only, §6).
- **population goalsRun / expert mtime deltas** (context only).

**Overall verdict**

- `FAIL` — any of C1–C8 fails, or a crash was observed.
- `PASS (S1|S2|S3)` — all checks pass **and** exposure ≥ 3 h.
- `SELF-TEST (below S1: all checks green, no stability claim)` — all checks pass
  but exposure < 3 h (e.g. the 8-minute dry run). This is the correct label for
  a 30-minute soak as well.

---

## 6. Reading `summary.md` / `summary.json`

`summary.md` contains, in order: run identity + build manifest, **Verdict**
(overall, S-rung, exposure, max window, restarts, lambda upper bound), the
S-ladder table, the metrics table, the criteria table with PASS/FAIL, "why
these criteria", the **lifecycle transcript** (table + verbatim `lifecycle.jsonl`
objects + deliverable excerpt), the forensics section, learning context, the
reproduce command and the harness self-check (live untouched / sidecar cleanup /
errors).

`summary.json` (schema `phoenix.stability.v1`) mirrors the same data; the fields
used by tooling/other agents are:

```
verdict.overall, verdict.pass, verdict.sRung, verdict.exposureHours,
verdict.lambdaUpper95PerHour, verdict.maxContinuousWindowSec, verdict.sLabel
criteria[]              # id/ok/value/threshold/note for C1..C8
exposure.byPid[]        # per-PID up&ticking sample counts + max window
metrics.*               # upRate, httpMs, gpu, rss, crashes, restarts, slotBusy, missions, expert, population
lifecycle.*             # driver, missionId, steps[], workEvidence, finalState
forensics.*             # crashes[] (dump path, resolve report), gaps[]
build.*                 # phoenix exe sha256/mtime, map, llama paths
```

**Learning metrics are context only.** MMLU scores are 40-question samples and
noisy (see `runtime_store/long_run/mmlu_history.jsonl`); a run may append zero
records. They never drive a verdict.

---

## 7. Debugger / forensics hook

### 7.1 Two crash-evidence channels (important)

The gateway installs its **own unhandled-exception filter**
(`main_hub_parts/000_section_before_config.inc`, `onUnhandledException`, armed
by `SetUnhandledExceptionFilter` in `116_section_tail.inc`) and returns
`EXCEPTION_EXECUTE_HANDLER`. Consequences:

- **Ordinary SEH faults** (typically `0xC0000005` access violations) are
  **swallowed**: WER never sees them, so there is **no Application event 1000
  and no WER minidump**. The only evidence is a line in
  `runtime_store/logs/fatal_runtime.log`
  (`windows unhandled exception code=0x... addr=... rip=... offset=...`) plus a
  usually-0-byte `runtime_store/logs/crash.dmp` (MiniDumpWriteDump fails in the
  corrupted process). The process exits with the exception code.
- **Fail-fast codes** (`0xC0000374` `STATUS_HEAP_CORRUPTION`, raised inside
  `ntdll`) bypass the filter: WER **does** record event 1000 and a full dump
  (`build\tmp\dumps\phoenix_main.exe.<pid>.dmp` in this tree, or the configured
  LocalDumps folder).

Therefore C1 counts **both** channels, and the harness scans **both** every
sample (appended-log offsets, so pre-run history is not re-counted). It also
checks `runtime_store/logs/crash.dmp` as a dump candidate (only a non-empty file
written after run start is usable — the usual 0-byte file is recorded through the
`fatal_runtime.log` channel) and tracks gateway PID-change (restart) events, so
a "silent" `0xC0000005` death cannot pass as clean.

### 7.2 Hook behaviour

On **any** crash during a run the harness, inside that sample:

1. records the event (channel, app, path, module, exception code, faulting PID,
   time; `fatal_runtime.log` lines keep the raw evidence line),
2. locates a minidump, in order: `%LOCALAPPDATA%\CrashDumps\<app>.<pid>.dmp`
   (pid preferred), `%ProgramData%\Microsoft\Windows\WER\{ReportQueue,ReportArchive}`,
   `%LOCALAPPDATA%\Microsoft\Windows\WER\{ReportQueue,ReportArchive}`
   (`*<app>*` dirs + `*.dmp` + `Report.wer`), then project-local
   `build\tmp\dumps`, `runtime_store\dumps` and a non-empty
   `runtime_store\logs\crash.dmp`, newest first,
3. copies `Report.wer` and runs
   `python tools/dump_stack_resolve.py <dump> --map phoenix_main.map`
   (Python314 first, then `python`/`py`), saving `resolve.txt`, `event.txt`,
   `forensics.json` under `<bundle>/forensics/crash-<ts>-pid<pid>/`,
4. summarises dump path + resolved frames in `summary.md` / `summary.json`.

If anything is missing the run **records the gap explicitly** in
`forensics.gaps` (and `summary.md`): no dump for a WER event (LocalDumps not
configured), the designed no-dump case for a SEH-swallowed
`fatal_runtime.log` fault, no `phoenix_main.map`, no Python interpreter, or
resolver non-zero exit. Gaps are never silently ignored.

To make *fail-fast* minidumps reliable, configure LocalDumps (admin, one-time;
the harness does not change the registry itself):

```
reg add "HKLM\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps\phoenix_main.exe" /v DumpFolder /t REG_EXPAND_SZ /d "E:\079\v6.0Alixander\phoenix\runtime_store\dumps" /f
reg add "HKLM\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps\phoenix_main.exe" /v DumpType /t REG_DWORD /d 2 /f
```

The same keys apply to `llama-server.exe` if its crashes matter. `DumpType=2`
is a full dump (large, but `dump_stack_resolve.py` only needs the stack).

`tools/dump_stack_resolve.py` reads the faulting thread's return-address chain
from the minidump and resolves it against the GNU ld map — this is how the
0xC0000374 (heap-corruption in `ntdll`) crashes become actionable functions.

---

## 8. How to run it

### 3-hour protocol run (the real thing)

```
cd E:\079\v6.0Alixander\phoenix
pwsh -NoProfile -ExecutionPolicy Bypass -File tools\stability_run.ps1 -Minutes 180 -Label v10.1-<build>
```

The live gateway + 3 llama-servers may already be up; the harness runs beside
them. It starts one sidecar gateway for the lifecycle (a few minutes) and stops
it automatically. Expect ~3 h 5 min total (finalize + authoritative status
calls).

### 8-minute harness dry run (validates the harness, never the build)

```
pwsh -NoProfile -ExecutionPolicy Bypass -File tools\stability_run.ps1 -Minutes 8 -IntervalSec 30 -Label dryrun
```

Expect `SELF-TEST (below S1)` with all checks green and a full lifecycle
transcript.

### Useful parameters

| parameter | default | use |
|---|---|---|
| `-Minutes` | 180 | run length |
| `-IntervalSec` | 60 | sample interval (>=5) |
| `-Label` | run | tag in the bundle dir and in `build`/report |
| `-GatewayUrl` | auto | override live gateway discovery |
| `-LifecycleMode` | auto | `auto` = sidecar with labelled live fallback; `sidecar`; `live`; `none` |
| `-HealthTimeoutSec` | 15 | health probe timeout (defines "up") |
| `-LifecycleTimeoutSec` | 900 | max wait for work evidence before report |
| `-MissionStatusEverySec` | 300 | cadence of the heavy authoritative mission-status call |
| `-SidecarStartupTimeoutSec` | 180 | sidecar health deadline |
| `-OutRoot` | `runtime_store/stability` | bundle parent dir |
| `-CleanupOnly -BundleDir <bundle>` | — | kill the recorded sidecar PID and remove its dir after an interrupt |

### Hard rules

- never stop the live gateway/`llama-server` processes (the harness doesn't);
- do not run `compile.bat` during a run (it kills `phoenix_main`/`llama-server`
  by design);
- no `git branch`/`commit`/`push` from the harness; the Lead commits the bundle
  paths + `tools/` + doc changes;
- never leave the sidecar running: check `cleanup.json` /
  `Get-Process` (the harness verifies and would report `leftover sidecar`).

---

## 9. Comparing two builds

Run the **same command** for both builds with different `-Label` (and ideally on
the same host/GPU with the same live workload):

```
pwsh ... -Minutes 180 -Label v10.1-buildA
# rebuild / checkout buildB (Lead-managed), keep the live system up
pwsh ... -Minutes 180 -Label v10.1-buildB
```

Then compare `summary.json`:

```
$a = Get-Content runtime_store/stability/<A>/summary.json -Raw | ConvertFrom-Json
$b = Get-Content runtime_store/stability/<B>/summary.json -Raw | ConvertFrom-Json
foreach ($k in 'sRung','lambdaUpper95PerHour','exposureHours','pass') {
  "verdict.$k : A=$($a.verdict.$k)  B=$($b.verdict.$k)"
}
foreach ($m in 'upRate','maxConsecutiveMiss','slotBusyEvents','crashEvents') {
  "metrics.$m : A=$($a.metrics.$m)  B=$($b.metrics.$m)"
}
"http p95 : A=$($a.metrics.httpMs.p95)ms B=$($b.metrics.httpMs.p95)ms"
```

Comparison rules:

1. **Verdict first**: a build that fails any criterion is worse regardless of
   the numbers. Compare `verdict.pass`, then `criteria[].ok`.
2. **Rate, not count**: compare `verdict.lambdaUpper95PerHour` at comparable
   exposure. A 3 h run gives λ<1/h (S1); 12 h gives λ<0.25/h (S2). Do not
   compare a 3 h bound against a 12 h bound as if they were the same statistic.
3. **Exposure must be comparable**: if `exposureHours` differ materially, the
   lambda bounds are not directly comparable — extend/shorter the run or compare
   only overlapping windows.
4. **Deltas to review**, in this order: `metrics.upRate`, `metrics.maxConsecutiveMiss`,
   `metrics.httpMs.p95/max`, `metrics.gatewayRssMB` (first→last growth),
   `metrics.slotBusyEvents` per hour, `metrics.restartEvents`,
   `metrics.crashesTotal`.
5. **Lifecycle**: `lifecycle.finalState`, `lifecycle.workEvidence.deliverableChars`
   and the lifecycle duration are build-independent but very noisy (LLM
   latency); use them only to confirm the lifecycle completed, not to rank
   builds.
6. **Noise floor**: at 60 s sampling, a ±1 sample difference in up-rate is
   rounding; treat a single missed sample (0.9944 at 180 samples) as noise, two
   or more as signal. MMLU deltas are never build evidence.

A comparison table you can paste into a report:

| build | run | exposure | λ95upper | rung | crashes | up-rate | p95 ms | restart | slot-busy | lifecycle |
|---|---|---|---|---|---|---|---|---|---|---|
| A | `<bundle>` | 3.0 h | 1.0/h | S1 | 0 | 1.000 | … | 0 | … | Completed |
| B | `<bundle>` | 3.0 h | 1.0/h | S1 | 0 | 0.994 | … | 0 | … | Completed |

Promotion guidance: a build may only claim an S2 rung after a **12 h** exposure
run with zero crashes; S3 requires 30 h+ (or λ<0.1/h). Any build with a WER crash
at any exposure is not promotable until the crash is resolved (see the bundle's
`forensics/`).

---

## 10. Troubleshooting

| symptom | first checks |
|---|---|
| `SELF-TEST` not all-green | read the criteria table; C1 crash → `forensics/`; C2/C3 → `metrics.upRate` + `gateway_loop.log`; C6 → `harness.log` stalls |
| gateway restarted but no WER event | expected for SEH-swallowed `0xC0000005`: look for the `fatal_runtime.log` line in `forensics/crash-*/event.txt` (`kind=fatal_runtime_seh`, no dump by design) |
| no `PASS` despite green checks | exposure < 3 h — expected; run the 3 h protocol |
| lifecycle stuck `assigned` | sidecar `/api/mission/status` unreachable (check `lifecycle/sidecar.out.log`), or the LLM slot is saturated (work appears at the next tick; `LifecycleTimeoutSec` catches it) |
| sidecar fails to start | `lifecycle/sidecar.err.log`; most common: port 5100-5199 all busy, or DLL not found (PATH must include the repo root) |
| `listenerZombie` note | the listener socket leaked to a dead image; the harness fell back to the max-RSS gateway — record it, it is a known artifact of past crashes |
| no GPU numbers | `nvidia-smi` not on PATH / driver restart; C7 fails and the summary says why |
| forensics `gap` | no dump (enable LocalDumps, §7) / no map (rebuild) / no Python / resolver exit |
| interrupted run | `pwsh tools\stability_run.ps1 -CleanupOnly -BundleDir runtime_store\stability\<ts>` |

---

## 11. Bundle layout

```
runtime_store/stability/<ts>[-label]/
  summary.md            human-readable: verdict, S-rung, lambda bound, metrics, criteria, transcript, forensics
  summary.json          schema phoenix.stability.v1 (same content, machine-readable)
  samples.jsonl         one JSON object per sample (full metric set)
  samples.csv           flat CSV view
  harness.log           harness trace (each sample line + lifecycle steps + errors)
  lifecycle.jsonl       verbatim lifecycle transcript (assign/running/work/report/completed)
  lifecycle/            sidecar logs, sidecar autonomy snapshot, deliverable file
  forensics/            crash event text, Report.wer, resolve.txt, forensics.json
  cleanup.json          sidecar pid/exe/dir + live-gateway note for -CleanupOnly
```

The bundle is self-contained: logs needed to explain a verdict are copied in; the
only external references are absolute live-system paths (dump paths, map) which
are recorded explicitly.

---

## 12. Changing the protocol

The criteria, metric definitions and the S-rung ladder are **versioned**:
`summary.json.schema = phoenix.stability.v1` and `harness.version` record the
revision. If you change a threshold, a metric definition or the rung cut-offs,
bump the protocol revision here, bump `harness.version` in
`tools/stability_run.ps1`, and state the change in the run report so old and new
bundles are never compared silently.
