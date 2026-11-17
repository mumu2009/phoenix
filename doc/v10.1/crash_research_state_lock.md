# Research-state lock: the 0xC0000374 heap-corruption fix (2026-11-17/18)

Status: **implemented, verified, awaiting commit** (author: crash-fix teammate; owner: Lead).
Companions: `crash_forensics_2026-11-15.md` (earlier race round),
`crash_hunt_pageheap.md` (PageHeap method), `debug_protocol.md` (tools).

## 1. Symptom

`phoenix_main.exe` died with `0xC0000374` (STATUS_HEAP_CORRUPTION, faulting module ntdll,
offset 0xff489) four times on 2026-11-17 (00:55, 08:12, 08:14, 08:15). Because the exception is
raised inside the heap checker, the exception address says nothing; the faulting thread's stack
carries the return-address chain of the code that was allocating/freeing when the corrupted block
was touched.

## 2. Root cause

`OnlineResearcher::fallback()` re-entered `RuntimeState`'s mutable research state **without
`RuntimeState::stateMu_`**. The raw helpers it calls assume the caller holds that mutex:

* `main_hub_parts/065_class_runtimestate.inc:1931` — `pattern_.rebuild(result.windowInfo);`
  inside `RuntimeState::runPropagation()` (the call site is at `runPropagation+0x632`).
* `main_hub_parts/045_class_graphtensorbridge.inc:150` — `multi_ = json{...}` inside
  `GraphTensorBridge::buildMultiOrderCSR()`; this is the **concurrent writer** seen on the other
  threads (frame offset `buildMultiOrderCSR+0x1f8d`).
* `main_hub_parts/047_class_dimreducer.inc:113` — `DimReducer::project2D()` json array build
  (`PatternMatrix::rebuild+0xcec`).
* `PatternMatrix::rebuild+0x60` resolved to an inlined `std::__new_allocator` heap allocation: the
  heap operation that tripped the checker on the faulting thread.

`PatternMatrix::state_`, `GraphTensorBridge::multi_`/`embeddings_` and the nlohmann json trees they
own are shared mutable members with **no internal mutex**. Every other caller of
`mapWordsToMemesOrdered()` / `runPropagation()` holds `stateMu_` (`processInput` 065:513,
`learnFromDialog` 065:1578, `mapWordsToMemes` 065:508). `OnlineResearcher` is a `friend` of
`RuntimeState` (065) and used that access without the lock.

### Trigger

`config/phoenix.json`: `autonomyLoop.parallelMissions=true`, `autonomyLoop.maxMissionsPerTick=8`,
`autonomyLoop.missionDelibLock=false` (the coarse backstop was demoted to a perf switch after the
2026-11-15 round). Up to 8 mission threads per tick run the deliberator (111) in parallel:
`collectAddonOffers -> AddonManager::collectOffers -> SearchAddon::handle ->
invokeAddonOnlineLookup -> RuntimeState::onlineLookup -> OnlineResearcher::lookup -> fallback ->
mapWordsToMemesOrdered + runPropagation -> pattern_.rebuild`. Nothing serialised them.

## 3. Evidence

### 3.1 Crash dumps, re-resolved with a matching symbol map
`phoenix_main.map` in the tree was **stale** (the failed `-D_GLIBCXX_DEBUG` sidecar link at 20:13
overwrote it; debugger-tooling measured map main `0x141f7d800` vs exe main `0x14190c000` and 5628
`std::__debug` symbols). The map was regenerated from the crashing binary's COFF symbol table:

```
nm -C --defined-only phoenix_main.exe   # -> build\tmp\phx_nm.map (addresses re-based on 0x140000000)
```

Provenance: crashing `phoenix_main.exe` sha256
`5508D3754628947D3B943D206E370ED10E87C9F6219A2CEAAEF4E569B25483D8` (2026-11-16 19:56:15), PE
checksum `0x2ce3127`. All three dumps record module
`...\phoenix_main.exe base=0x7ff7eedb0000 size=0x2111000 checksum=0x2ce3127` — exact match, and no
source file was newer than the exe, so the map matches the crashing build. **Worked example of the
authoritative pair**: map file `build\tmp\phoenix_main.map.1794925428` (sha256
`7DEDBAE5D9CB67E2975C950EA1AF8FA35A07D8B2F7EF19B3161DDF42FF0DEF96`) generated from exe sha256
`5508D3754628947D3B943D206E370ED10E87C9F6219A2CEAAEF4E569B25483D8` (PE checksum `0x2ce3127`,
dump module match above).

**Rule**: `phoenix_main.map` is not a stable artifact — any `-Wl,-Map=phoenix_main.map` sidecar
build rewrites it for a *different* binary (this is exactly what happened at 20:13: the failed
`-D_GLIBCXX_DEBUG` link). The authoritative map for a dump is the one generated with
`nm -C --defined-only` from the exact crashing exe and validated by the (base, size, PE checksum)
recorded in the minidump's module entry. Prefer `tools/dump_report.ps1`, which reads the exe's own
COFF symbol table directly and warns when `--symbols map` is stale.

`build/tmp/all_thread_stacks.py` scans **every** thread's stack (not only the faulting one):

| dump (time) | threads simultaneously inside `PatternMatrix::rebuild` |
|---|---|
| `phoenix_main.exe.44428.dmp` (08:12:37) | 2 — t48740 faulting at `rebuild+0x60`; t41032 `rebuild+0xcec -> buildMultiOrderCSR+0x1f8d -> std::_Rb_tree<...,nlohmann::json,...>::_M_copy+0xcd` |
| `phoenix_main.exe.46512.dmp` (08:14:27) | 3 — t18596 faulting, t24304, t44224 |
| `phoenix_main.exe.16804.dmp` (08:15:39) | 3 — t44704 faulting, t47752, t48288 |

Every chain passes through `OnlineResearcher::fallback`.

### 3.2 Repro (PageHeap, renamed binary, isolated state)
`build/tmp/repro_phx_pageheap.ps1` copies the binary to `phoenix_repro.exe` (the supervisor kills
only `phoenix_main`), enables IFEO PageHeap (`GlobalFlag=0x02000000`) and WER LocalDumps for the new
name only, isolates `cwd=build\tmp\phx_repro` and `AI_BASE_DIR=build\tmp\phx_repro\runtime_store`
(`main.baseDir` is otherwise absolute and would write into the live runtime_store), port 5101,
`full_components`, robots off, then assigns 10 research missions.

* old binary: crashed twice, 4.5 / 9.5 min, exit `-1073741819` (0xC0000005), fatal-log offsets
  `0x6d887`/`0x6d90d` = the same `std::_Rb_tree<...,nlohmann::json,...>::_M_copy` json copy with a
  wild read target; working set 9.2-10.1 GB (PageHeap inflated).
* patched sidecar: **20 min, 0 crashes, 0 WER dumps**, no fatal-log entry.
* gdb verification of the patched `-g1` build (contains the fix: fallback calls
  `pthread_mutex_lock/unlock`): `build/tmp/gdb/mingw64/bin/gdb.exe`, cwd `build\tmp\phx_gdb`,
  port 5102, 10 real research missions, 15 min of real LLM work -> 94 threads, **no fault**
  (operator SIGINT); transcript `build\tmp\gdb\threads_2026-11-17.log`.

Diagnostic methods and the exact commands: `debug_protocol.md`. For dumps prefer
`tools/dump_report.ps1` (probes the exe's own COFF table) over `phoenix_main.map`.

## 4. Patch shape

The mutex is the one the class already uses for this state; the fix is a small RAII entry point for
the outside collaborator, no new lock and no semantics change.

1. `autonomy_stack.hpp` — header-only
   `phoenix::state_guard::ResearchStateGuard` and
   `template <class Fn> auto withResearchStateLock(std::mutex&, Fn&&) -> decltype(fn())`.
   Callback form: the guard is released on every exit path, including exceptions thrown by a
   propagation (otherwise one bad propagation would wedge every chat/mission/study path).
2. `main_hub_parts/065_class_runtimestate.inc` — `std::mutex &researchStateMutex()` accessor
   (returns the existing `stateMu_`; the class already declares `friend class OnlineResearcher`).
3. `main_hub_parts/068_section_before_crawlitem.inc` — the whole `OnlineResearcher::fallback()`
   body now runs inside `withResearchStateLock(runtime_->researchStateMutex(), ...)`. Same json and
   same result. It also excludes `processInput()`'s propagation (the other half of the race) and
   makes it impossible for a future caller of the raw helpers to repeat the mistake through this
   path.
4. `tests/gtest/unit/autonomy/test_research_state_guard.cpp` — pins: exclusion against a plain
   `lock_guard` on the same mutex (the production sharing contract), N concurrent guarded callbacks
   never overlap, lock released when the callback throws, ownership reporting. `RuntimeState` lives
   in the `main.cpp` unity TU and is not linked into `gtest_runner`, so the guard is the reachable
   seam.

The temporary `PHOENIX_RESEARCH_TIMING` instrumentation used for section 5 was removed before the
final build (the guard is the only addition).

## 5. Perf budget and verdict

Instrumented clean run (patched binary, real LLM, 10 parallel research missions, 22 min, otherwise
idle machine). Because the pre-fix body is byte-identical work, "before" is the measured body time
and "after" is body + measured lock wait:

| metric | before | after |
|---|---|---|
| fallback call wall (p50 / p95) | 971 ms / 1612 ms (`body_us`) | 3480 ms / 6578 ms (`body+wait`) |
| added lock wait (p50 / p95) | 0 | 2013 ms / 5942 ms |
| mission tick elapsed (n=29) | — | p50 41.83 s / p95 76.58 s |
| production baseline ticks (old binary, 1542 ticks) | p50 29.18 s / p95 64.97 s | — |

The per-tick **wall** cost is not `sum(wait)` (waits overlap); it is
`sum(body) - max(body)` for a burst (pre-fix bodies ran in parallel):
+4.84 / +5.95 / +5.77 / +5.20 / +5.63 s on the 5 burst ticks = **+7.3% .. +15.3% per affected
tick**, mean burst 4.56 s; over all 29 ticks `27.4 s / 1283.9 s = 2.13%`. Below the 20% gate ->
**keep the whole-body lock**.

`llama-slot-busy` is not attributable to the lock: production before was 18/1542 ticks = 11.7/1000;
the isolated 10-mission harness shows heavy contention for both binaries (old 5 events/8 min,
patched 126/22 min) because 10 parallel missions share one llama slot. No llama call is made inside
the guarded region (`fallback()` is local-only; `lookup()`'s remote fetch precedes it).

### Future option (documented, deliberately NOT implemented now)

If research bursts ever dominate tick time, the narrow shape is a **dedicated mutex around only
`pattern_.rebuild()`** (new member in `RuntimeState`/065; the shared writer lives in
`PatternMatrix` 049 and `GraphTensorBridge` 045), leaving `mapWordsToMemesOrdered` and
`tensor_.iteratePropagation` parallel (they are internally safe: `MemeGraph`/`KVMStore` lock per
call, `TensorEngine::iteratePropagation` is stateless). Measure with the same burst-wall method and
re-run the PageHeap 20-min repro plus the 15-min gdb verification before switching.

## 6. Swallowed-crash caveat (important for every future crash hunt)

`main_hub_parts/000_section_before_config.inc:664` `onUnhandledException` (installed by
`SetUnhandledExceptionFilter` at `main_hub_parts/116_section_tail.inc:8669`, returns
`EXCEPTION_EXECUTE_HANDLER` at 000:715) **swallows ordinary SEH faults**, notably `0xC0000005`
access violations: it writes `<baseDir>/logs/fatal_runtime.log` and attempts a minidump at
`runtime_store/logs/crash.dmp` (000:703, cwd-relative; usually fails in a corrupted process ->
0-byte file) and the process exits with the exception code. There is **no WER event and no WER
dump** in that case. `installFatalHooks` (000:764-778) additionally converts
`std::terminate`/`std::bad_alloc` (`_Exit(134)`) and SIGABRT/SIGSEGV/SIGILL/SIGFPE/SIGTERM
(`_Exit(128+sig)`) into silent exits.

`0xC0000374` (and other fail-fast codes such as `0xC0000409`) are **not** swallowed: fail-fast
bypasses SEH, which is why WER produced the four dumps and why `fatal_runtime.log` has no
unhandled-exception line at 08:12/08:14/08:15.

**Do not read "no WER dump" as "no crash."** For any harness: treat a nonzero exit code plus
`runtime_store/logs/fatal_runtime.log` entries as crash evidence; WER only catches fail-fast
crashes. (Reported to the 3 h stability harness; filter behaviour intentionally unchanged.)

## 7. Verification record

Final gate was run on a **frozen tree** (snapshot 2026-11-18 00:10:52; tracked `git diff` sha256
`6E8A0D6951ADC1A61A45E317FF1C2E98B5D6501AA448816B4816874D875FAF70`). No build input changed while
the compiler ran: every `main.cpp`/`main_hub_parts`/`autonomy_stack.*`/`scoped_trainable_memory.hpp`
file keeps its 22:48-23:31 mtime; the only files touched between 00:10 and the link were the build
outputs (`phoenix_main.next.exe`, `phoenix_main.map`), two other directions' docs
(`conan_abi_gcc16.md`, `race_sentinel.md`) and `lmdb/emotion_states.db` generated by another
agent's test run - all non-compiled.

* frozen-tree sidecar `build\tmp\compile_next.bat` -> **EXIT=0**;
  `phoenix_main.next.exe` sha256
  `A92DC93F3020B88700D5B802F899A853FD320C156DEE11F9DFF9AF75E34245D8`, 47,062,929 bytes
  (2026-11-18 00:32:11). (The earlier patch-state build was
  `3313EF9EF05527C9DCB925A7E0D7A00A7164AAF5B2AB07FAB0F71F7042C3E979`.) The sidecar rewrites
  `phoenix_main.map` for the *new* binary - see the map-provenance rule in section 3.1.
* `-g1` sidecar `build\tmp\phoenix_main.next.g1.exe` (DWARF line tables; contains the fix) used for
  the gdb run and the frame->line mapping.
* `compile_gtest.bat` -> `[SUCCESS] gtest_runner.exe compiled successfully` (2026-11-18 00:06).
* `gtest_runner.exe --gtest_filter=*ResearchStateGuard*` -> **4/4 PASSED**.
* regression subset `*ResearchStateGuard*:*Mcp*:*StudyConfig*:*LogicEngine*:*PluginSemantics*:*MissionLifecycle*:*AutonomyLoop*:*Addon*:*PromptSplit*`
  -> **250 tests / 38 suites, all PASSED** (122 s), re-run after the frozen build.
* frozen-build sanity: `build\tmp\repro_phx_pageheap.ps1 -Minutes 15 -MissionCount 8 -NoPageHeap`
  on exactly `phoenix_main.next.exe` sha256 `A92DC93F...`, isolated `cwd=build\tmp\phx_repro` +
  `AI_BASE_DIR=...\phx_repro\runtime_store`, port 5101 -> **15 min, no crash**, no
  `fatal_runtime.log` exception, 0 WER dumps, 22 mission ticks, p50 tick 36.19 s;
  82 `llama-slot-busy` events (workload: 8 parallel missions on one llama slot - see section 5).
* earlier stress proof on the same patch: 20-min PageHeap run of the patch-state sidecar = 0
  crashes (old binary: 2 crashes in 4.5/9.5 min); 15-min gdb run of the `-g1` build with 10 real
  missions = no fault.
* PageHeap cleanup after every repro run: IFEO `phoenix_repro.exe` key and WER LocalDumps
  `phoenix_repro.exe` key removed, `phoenix_main.exe` IFEO has no `GlobalFlag` (a leftover PageHeap
  would break/OOM production restarts). Production gateway pid 23920 untouched throughout.

Evidence bundle: `build\tmp\crash_fix_evidence.md` (numbers and artifact paths),
`build\tmp\repro_harness.log`, `build\tmp\phx_repro\stdout_*.log`,
`build\tmp\gdb\threads_2026-11-17.log`, `build\tmp\all_thread_stacks.py`,
`build\tmp\frame_to_line.py`, `build\tmp\perf_wall.py`.
