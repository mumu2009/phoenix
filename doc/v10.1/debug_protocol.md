# Phoenix debug protocol (v10.1) — which tool, exact commands, never-do list

Scope: crash/assert/hang/heap-corruption debugging for this project on Windows
with the MinGW g++ 16.1.0 toolchain.  Written 2026-11-17 against the live state
of this box.  Companion document: `doc/v10.1/debugger_transcript.md` (proof that
gdb works and what it can/cannot see on a MinGW binary without a PDB).

> **The one rule that overrides everything:** the production gateway on port
> 5097 (plus its 3 llama-servers) is a **live long run**.  Nothing in this
> document may stop, pause or relaunch it.  `compile.bat` does
> `taskkill phoenix_main.exe` (and llama-server) and must not be used while the
> run is live — use the sidecar scripts under `build\tmp\`.

---

## 1. Tool availability on this machine (measured 2026-11-17)

| Tool | Present? | Version | Where / how to get it | What it can do with a **MinGW** binary (no PDB) |
|---|---|---|---|---|
| **gdb** | **YES (project-local)** | **17.2** | `build\tmp\gdb\mingw64\bin\gdb.exe`; provisioned by `tools/setup_gdb_local.ps1` (not installed system-wide) | Symbol/address breakpoints, multi-frame backtraces, threads/registers/disassembly; source lines & locals **only with a `-g` build** |
| gdbserver | YES (project-local) | 17.2 | `build\tmp\gdb\mingw64\bin\gdbserver.exe` | Remote/attach debugging of a **sidecar**; pauses target (never use on the live gateway) |
| cdb | **NO** | — | Install "Debugging Tools for Windows" via Windows SDK installer (feature *Debugging Tools for Windows*), or `winget install Microsoft.WinDbg`; cdb lives under `...\Windows Kits\10\Debuggers\x64\` | Needs a PDB.  A MinGW DWARF build gives it **nothing**: only exports + `.pdata`-based native unwinding.  `-g -gcodeview` was tested and is **not a working path on this toolchain**: g++ 16.1.0 assembles it only at `-O0/-O1` (`-O2/-O3` fail with `can't resolve .text.unlikely - .LcvlineN`) and even then the exe still contains DWARF, not `.debug$S` CodeView.  Treat cdb/WinDbg as needing separate investigation; gdb is the proven debugger |
| windbg | **NO** | available in winget as `Microsoft.WinDbg` 1.2611.4001.0 | `winget install Microsoft.WinDbg` | Same PDB limitation as cdb |
| procdump | **NO** | — | Sysinternals (`procdump -ma -e -w`), or use the WER LocalDumps that is already configured here | Dump capture only; no symbols needed |
| gflags | **NO** | — | Ships with Debugging Tools for Windows; equivalent registry recipe in §6 | PageHeap / IFEO control (no symbols needed) |
| Application Verifier (`appverif.exe`) | **YES** | 10.0.26100.7705 | `C:\Windows\System32\appverif.exe` (GUI) | Heap/Basic checks; no symbols needed; needs a short-lived sidecar |
| WER LocalDumps | **YES, configured** | werfault 10.0.19041.4522 | `HKLM\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps\phoenix_main.exe` → `DumpFolder=E:\079\v6.0Alixander\phoenix\build\tmp\dumps`, `DumpType=2` (full memory), `DumpCount=10` | Automatic full dumps on any crash; no symbols needed |
| Python + `minidump` | **YES** | 3.11.9 | `python` (D:\cloudworkfile\tools\python311), `minidump` package installed | Offline dump parsing — the project's own `tools/dump_report.ps1` |
| nm / objdump / addr2line / readelf | **YES** | GNU Binutils 2.46.1 | `C:\msys64\mingw64\bin` | Symbol tables, section/`.pdata` inspection, address→line for `-g` builds |
| VS 2022 Build Tools | YES (MSVC 14.44) but **no debugger front-end / no Debugging Tools** | 17.14 | — | Not used on MinGW binaries |
| DbgHelp / DbgEng DLLs | YES (system) | 10.0.19041 | `C:\Windows\System32` | Library-level only; no shell.  Not used by the workflow below |

Notes:

* `gdb` is **not** part of this machine's stock MSYS2 toolchain.  `pacman -S`-ing
  it would add/upgrade DLLs under `C:\msys64\mingw64`, which the conan prebuilts
  and the live gateway depend on.  `tools/setup_gdb_local.ps1` downloads
  (`pacman -Sw`) and unpacks the package + dependencies into `build\tmp\gdb`
  instead — no package DB change, no system DLL touched.
* MinGW does not emit PDBs by default.  gdb does **not** need them: it uses the
  COFF symbol table and the PE64 `.pdata`/`.xdata` unwind tables.  cdb/WinDbg
  would need CodeView/PDB; the advertised `-gcodeview` flag was measured to be
  broken/ineffective in this toolchain (see table above), so cdb is not a
  practical debugger here until that is investigated.  Use gdb.

`gdb --version` (measured): `GNU gdb (GDB) 17.2`, configured as
`x86_64-w64-mingw32`.

---

## 2. Decision table — pick the tool from the symptom, not from habit

| Symptom / exception | First move | Then, if not conclusive |
|---|---|---|
| Process died and a `.dmp` exists in `build\tmp\dumps` (e.g. `0xC0000374` heap corruption, `0xC0000005` AV in a sidecar) | `powershell -File tools\dump_report.ps1 -Latest` | If the faulting stack points at a project function: reproduce in a `-g1` sidecar under gdb, break at that function |
| `0xC0000374` (`STATUS_HEAP_CORRUPTION`) — crashes inside `ntdll`, corrupt heap detected **later** than the bad write | dump report first (return-address chain of the faulting thread) | Enable PageHeap or AppVerifier **for a short isolation window** (§6), capture the now-immediate `0xC0000005`, kill page heap, re-run the report |
| `assert()` / abort / `0xC0000409` / plain `SIGABRT` | Build the `-g1` sidecar (`build\tmp\compile_next_g1.bat`), run under gdb, `handle SIGABRT stop`, `bt` | Add `catch throw` if an exception path is suspected |
| Hang / deadlock / 100% CPU / no response | gdb attach to the **sidecar** PID (never the live gateway): `gdb -p <pid>` → `thread apply all bt` | `gdbserver --attach` for a persistent remote session; Process Explorer stacks if symbols are missing |
| Crash only under concurrency (races) | PageHeap (`0xC0000005` at the true write) + dumps; tracepoints/logs | The race-detector branch (`RaceSentinel` teammate) is the structural fix route |
| Suspected out-of-bounds write in a container | PageHeap/AppVerifier short window + dump report | Add invariant checks (debug-only, product source change — coordinate with the Lead) |
| C++ exception escaping as `terminate` | gdb: `catch throw`, `handle SIGABRT stop`, `bt`; inspect exception object | dump report if it crashed before gdb could catch |
| Need source lines/variables in the real program | `-g1` sidecar (§4.1): line tables.  `-g` full: locals/types | Do not add `-g` to the production build just for this |

Golden rule: **dumps tell you *where the process was*; gdb/PageHeap tell you
*why*.**  Use dumps first (zero risk to the live run), then gdb on a sidecar.

---

## 3. Playbook A — WER minidump → report in seconds (zero risk)

Dumps are already configured (full memory, 10 kept).  After any crash:

```powershell
cd E:\079\v6.0Alixander\phoenix

# newest dump only
powershell -NoProfile -ExecutionPolicy Bypass -File tools\dump_report.ps1 -Latest

# every dump, one markdown report each (10 dumps took 6.8 s total)
powershell -NoProfile -ExecutionPolicy Bypass -File tools\dump_report.ps1 -All

# one specific dump, more nearest-symbols context
powershell -NoProfile -ExecutionPolicy Bypass -File tools\dump_report.ps1 `
    -Dump build\tmp\dumps\phoenix_main.exe.44428.dmp -Below 5 -MaxFrames 60
```

Output: `build\tmp\dumps\<dumpname>.report.md` containing

1. dump module base + **binary identity check** (dump timestamp vs exe timestamp
   — catches "dump is from an older build" immediately),
2. **exception** code/name/address/parameters (`0xC0000374` heap corruption,
   `0xC0000005` access direction+address), resolved to `module+offset`,
3. **faulting thread** TID, stack range, RIP/RSP/RBP,
4. **frames validated against `.pdata`** (real return addresses) resolved to the
   function that contains them, ASLR-independent RVA, delta, plus
5. the **nearest N code symbols below** each frame (default 3), and
6. rejected stack words (image pointers outside any function) for transparency.

The report console summary prints dump / exception / TID / validated frames.

`tools/dump_report.py` can be called directly (add `--quiet`, `--out`,
`--symbols map` if you *know* the map matches the binary; it is stale today —
see §5).

Validated baseline from tonight (all 10 dumps in `build\tmp\dumps`, 2026-11-16/17):

| dump | size | exception | faulting TID | validated frames |
|---|---|---|---|---|
| phoenix_main.exe.23060.dmp | 1400 MiB | 0xC0000374 | 30104 | 49 |
| phoenix_main.exe.28096.dmp | 1370 MiB | 0xC0000374 | 13856 | 45 |
| phoenix_main.exe.36740.dmp | 716 MiB | 0xC0000374 | 25180 | 46 |
| phoenix_main.exe.9868.dmp | 1613 MiB | 0xC0000374 | 39516 | 49 |
| phoenix_main.exe.30368.dmp | 715 MiB | 0xC0000374 | 38956 | 45 |
| phoenix_main.exe.45076.dmp | 1088 MiB | 0xC0000374 | 40360 | 44 |
| phoenix_main.exe.17456.dmp | 1447 MiB | 0xC0000374 | 20928 | 49 |
| phoenix_main.exe.44428.dmp | 1542 MiB | 0xC0000374 | 48740 | 46 |
| phoenix_main.exe.46512.dmp | 746 MiB | 0xC0000374 | 18596 | 45 |
| phoenix_main.exe.16804.dmp | 738 MiB | 0xC0000374 | 44704 | 45 |

Recurring validated functions across the reports (heuristic stack scan, not a
proof of causality): `OnlineResearcher::lookup/fromCache/fallback`,
`RuntimeState::runPropagation/ingestDocument`, `TensorEngine::iteratePropagation`,
`PatternMatrix::rebuild`, `JsonFileStore::flush`.  That points the crash work at
the online-research/graph path — consistent with
`doc/v10.1/crash_forensics_2026-11-15.md`.  The `relu(double)` +
`std::_Function_handler<...>` frames are `std::function` call sites, not real
symbol boundaries.

Companion scanner: `build\tmp\all_thread_stacks.py` (from the crash-fix work)
scans **every** thread's stack in a dump, not just the faulting one; it is what
caught 2–3 threads simultaneously inside `PatternMatrix::rebuild`, i.e. the
corruption is likely a multi-thread container race rather than a single bad
write.  Use it when the faulting-thread report is not enough; it is a
build\tmp scratch tool (may move).

---

## 4. Playbook B — gdb on a sidecar (function break + backtrace)

### 4.1 Prepare gdb once

```powershell
cd E:\079\v6.0Alixander\phoenix
powershell -NoProfile -ExecutionPolicy Bypass -File tools\setup_gdb_local.ps1
```

### 4.2 The safe sidecar run

The sidecar build scripts write **only** to `build\tmp\` / a new exe name and do
not run `taskkill`:

| script | output | debug info |
|---|---|---|
| `build\tmp\compile_next.bat` | `phoenix_main.next.exe` | none (COFF symbols only) |
| `build\tmp\compile_next_g1.bat` | `build\tmp\phoenix_main.next.g1.exe` | `-g1` line tables — **use this for source lines** |

Measured on this box (2026-11-17, under CPU contention): normal sidecar 21.9 min
→ 47 MB exe; `-g1` sidecar 23.5 min → 145 MB exe (about +7% wall time, 3.1×
size).  Both are background-job friendly and do not stop the live run.
`-g1` maps project code to the real source fragments, e.g.
`main_hub_parts/049_class_patternmatrix.inc:14` in `PatternMatrix::rebuild`.  For
a full `-g` (locals/types) expect a much larger exe and longer link.

One-shot transcript run (stops at `main` **before the program body executes**, so
no port is bound and the live run is untouched):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\gdb_probe.ps1 `
    -Exe phoenix_main.next.exe -Break main `
    -Post 'bt','info registers rip rsp rbp','x/4i $pc'
```

Break on a project function (semicolon/namespace requires quoting; the exact
symbol survives in the COFF table even without `-g`):

```powershell
gdb -batch -nx `
    -ex "set pagination off" `
    -ex "break 'phoenix::io::(anonymous namespace)::sampleRateFromMetadata'" `
    -ex "run" -ex "bt" -ex "info registers rip rsp rbp" -ex "quit" `
    --args phoenix_main.next.exe --disable-watchdog
```

For source-line breakpoints use the `-g1` sidecar.  Note the real source lines
live in the `main_hub_parts/*.inc` fragments, not `main.cpp` (main.cpp is only a
hub of `#include`s):

```powershell
# break where main starts (line 8661 of the hub fragment) and inside a
# crash-path function (the recurring PatternMatrix::rebuild frame)
gdb -batch -nx `
    -ex "set pagination off" -ex "set breakpoint pending on" `
    -ex "break main_hub_parts/116_section_tail.inc:8661" `
    -ex "break main_hub_parts/049_class_patternmatrix.inc:13" `
    -ex "run" -ex "bt" -ex "list" -ex "info line *\$pc" `
    build\tmp\phoenix_main.next.g1.exe
```

Measured on this box: line tables for the real program map exactly like that
(`main() at main_hub_parts/116_section_tail.inc:8661`,
`PatternMatrix::rebuild at main_hub_parts/049_class_patternmatrix.inc:14`), and
`info source` reports `GNU C++20 16.1.0 ... -g1 -O3 -std=c++20`, DWARF 5.

Useful gdb commands for this project:

```
set pagination off          # no --More-- in batch
set confirm off
set breakpoint pending on   # templates/anonymous namespaces
break main                  # CRT-safe first stop
bt / bt 30                  # backtrace
thread apply all bt         # hangs: all threads (attached sidecar only)
info registers rip rsp rbp
x/8i $pc
catch throw                 # stop on C++ throw
handle SIGABRT stop         # assertions/abort
info line <func>            # shows "No line number information" on non-g builds
```

### 4.3 Assertions and aborts

MinGW `assert()` ends in `abort()`.  With a `-g1` sidecar:

```powershell
gdb -batch -nx -ex "handle SIGABRT stop" -ex "run" -ex "bt 30" -ex "frame 1" -ex "info locals" -ex "list" `
    --args build\tmp\phoenix_main.next.g1.exe --disable-watchdog
```

Without `-g` you still get the function chain (and `addr2line -e <exe> -f -C 0x<rva>`
maps RVAs to `file:line` for `-g1` binaries only).

### 4.4 Attaching (sidecars only — never the live gateway)

```powershell
# Sidecar PID only. Attaching PAUSES the process.
gdb -p <sidecar-pid> -ex "thread apply all bt" -ex "detach" -ex "quit"

# persistent/remote variant
build\tmp\gdb\mingw64\bin\gdbserver.exe --attach 127.0.0.1:5099 <sidecar-pid>
gdb -ex "target remote 127.0.0.1:5099"   # in another shell
```

Do **not** attach to PID 23920 (`phoenix_main` on :5097) or any llama-server:
the gateway has a watchdog and the long run would be destroyed.

---

## 5. Symbol hygiene (a real trap in this repo)

`phoenix_main.map` at the repo root is **not authoritative**.  History measured
today: at 20:13 it was overwritten by the failed `-D_GLIBCXX_DEBUG` sidecar link
and contained 5,628 symbols with the `std::__debug` ABI (that version produced
plausible-looking but wrong functions — the old `dump_stack_resolve.py` did
exactly this).  At 22:27 the Lead's normal `compile_next.bat` sidecar build ran
again and **rewrote it again** (`compile_next.bat` passes
`-Wl,-Map=phoenix_main.map`), so today's copy is normal-ABI but comes from a
different link than the 11-16/17 dumps: probe says `main=0x14190c040` vs the
dumps' exe `main=0x14190c000` → **MISMATCH** for those dumps.

Two consequences:

* any sidecar build with `-Map=phoenix_main.map` silently replaces the map — do
  not treat it as a stable artifact;
* always let the probe decide; the report tool now prints the
  `mainCRTStartup`/`main` comparison and the `std::__debug` count for
  `--symbols map`.

Correct order of truth:

1. **The executable's own COFF symbol table** — `nm -C -n phoenix_main.exe`
   (42,045 unique code symbols, 44,646 symbol lines).  `tools/dump_report.py`
   caches it at `build\tmp\dumps\phoenix_main.exe.nm.txt` and regenerates when
   the exe is newer than the cache.  This is always the right source for a given
   exe, including `phoenix_main.next.exe`.
2. **PE64 `.pdata`** (40,958 `RUNTIME_FUNCTION` ranges) — exact "is this
   address inside a function" test, used by the report tool.
3. A `.map` file — only if its probe says MATCH for the exe that made the dump.
   The report tool prints that probe when you pass `--symbols map`, plus a
   `std::__debug` ABI check (a debug-ABI map is an automatic MISMATCH).

Always check the report's **binary identity line** (dump module timestamp vs
on-disk exe timestamp).  A MISMATCH means every symbol in the report is suspect;
the fix is to analyse the exe that produced the dump (or rebuild to match).

After a new build, delete the cached symbol file or just let the mtime check
regenerate it (`build\tmp\dumps\phoenix_main.exe.nm.txt`).

If you ever need addresses ↔ file:line without gdb:
`addr2line -e <exe> -f -C 0x<preferred_va>` (works for `-g1`/`-g` builds; add
the PE preferred base 0x140000000 to the dump RVA).

---

## 6. PageHeap / IFEO — short, supervised windows only

Use only when a `0xC0000374` dump report is inconclusive and you can isolate the
process (`doc/v10.1/crash_hunt_pageheap.md` has the isolation recipe: copy
`config\` to `build\tmp\phx_repro`, start with `--gateway-port 5101`,
`--pid-file` override, `--disable-watchdog`, rename or isolate runtime_store).

Page heap flips "detected 25 minutes later" into "faults at the write", with an
immediate `0xC0000005` dump whose faulting thread is the writer.

Correct way (gflags; preferred because it is the documented, reversible tool):

```powershell
gflags.exe /p /enable phoenix_main_repro.exe /full   # renamed copy, not the live exe
# ... run the isolation sidecar, capture the dump ...
gflags.exe /p /disable phoenix_main_repro.exe
gflags.exe /p                          # must NOT list the image afterwards
```

Registry equivalent (if gflags is not installed) — this is the exact-flag
version documented for this repo:

```powershell
$k = 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Image File Execution Options\phoenix_main_repro.exe'
New-Item -Path $k -Force | Out-Null
Set-ItemProperty -Path $k -Name GlobalFlag -Value 0x02000000 -Type DWord   # FLG_HEAP_PAGE_ALLOCS
# cleanup (ALWAYS, immediately after the window):
Remove-ItemProperty -Path $k -Name GlobalFlag -ErrorAction SilentlyContinue
Remove-Item $k -Recurse -Force -ErrorAction SilentlyContinue               # key only
```

Verification that it is gone (must print `GlobalFlag: ABSENT`):

```powershell
$k = 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Image File Execution Options\phoenix_main_repro.exe'
$v = Get-ItemProperty -Path $k -Name GlobalFlag -ErrorAction SilentlyContinue
if ($null -eq $v) { 'GlobalFlag: ABSENT (good)' } else { "GlobalFlag: 0x$('{0:X}' -f $v.GlobalFlag) (LEFT ON - remove now!)" }
```

State at the time of writing (2026-11-17 22:45): IFEO key
`phoenix_main.exe` exists but carries **no** `GlobalFlag`/`PageHeapFlags`.
`phoenix_repro.exe` **does** carry `GlobalFlag=0x02000000` — that is the
crash-fix PageHeap repro window, i.e. another owner's *active* experiment.
Rules that follow from this:

* page-heap state is **time-sensitive** — re-check it before drawing
  conclusions, never assume the doc's snapshot is current;
* only the experiment owner removes their key; if you find a flag left behind
  after an experiment has clearly ended, report it and verify the cleanup
  (§6 snippet) rather than silently deleting an active window;
* the run/prod exe must never carry the flag: if `phoenix_main.exe` ever shows
  `GlobalFlag` without an owner's explicit window, treat it as a leak and
  disable/verify immediately.

Application Verifier alternative (no registry): run `appverif.exe`, add the
sidecar exe, enable **Heaps: Basic** (and "Use LFH guard pages" if offered),
run the isolated sidecar, then remove the app from the list.  It does not need
symbols; combine with WER dumps.

---

## 7. NEVER do this

1. **Never run `compile.bat` while the long run is live.**  It
   `taskkill`s `phoenix_main.exe`/`llama-server.exe` and overwrites
   `phoenix_main.exe` + `phoenix_main.map`.  Use `build\tmp\compile_next.bat`
   or `build\tmp\compile_next_g1.bat` (sidecar names/outputs).
2. **Never attach gdb/gdbserver/procdump-stop to the live gateway (PID on
   :5097) or its llama-servers.**  Attaching stops the process; the watchdog may
   then kill/restart it and the run is lost.  Debug sidecars only.
3. **Never leave PageHeap/IFEO `GlobalFlag` enabled** after an experiment (and
   never enable it on the production exe name for a long window — the 1.6 GB
   process balloons).  Clean up and verify with the snippet in §6.
4. **Never try `-D_GLIBCXX_DEBUG` again.**  It changes `std::{string,vector,map}`
   layout and symbol names; the conan prebuilts (drogon/opencv/sqlite3/...) are
   normal-ABI and the link fails with hundreds of `undefined reference`s.
   Proven in `doc/v10.1/crash_hunt_pageheap.md` §1.
5. **Never trust `phoenix_main.map` as a stable artifact.**  It was a
   std::__debug-ABI artifact of the failed debug link (20:13) and is rewritten
   by every sidecar build that passes `-Wl,-Map=phoenix_main.map` (the 22:27
   sidecar build replaced it with a normal-ABI map that still does not match the
   dumps' exe: probe `main` 0x14190c040 vs 0x14190c000).  Default to the exe's
   `nm` symbol table and only use `--symbols map` after the probe says MATCH.
6. **Never expect cdb/WinDbg to symbolise a MinGW DWARF binary.**  They need
   PDB/CodeView.  `-gcodeview` was measured here and does not produce a usable
   CodeView binary (`-O2/-O3` assembler errors; `-O0/-O1` output stays DWARF).
   Keep using gdb for MinGW builds.
7. **Never kill Phoenix/llama processes** (`taskkill /IM phoenix_main.exe`,
   `Stop-Process` loops, `killall`).  Even "stale-looking" 32 KB `phoenix_main`
   PIDs can be parts of the live environment.
8. **Never start a sidecar gateway in the project root with the live config.**
   It will fight over `runtime_store`, pid files and SQLite/Redis state, and can
   corrupt the long run's data.  Use the isolated `build\tmp\phx_repro` recipe
   with a different port and `--pid-file`.
9. **Never leave debug builds/tools in the product tree.**  gdb lives in
   `build\tmp\gdb`, the `-g1` exe in `build\tmp`; nothing under the repo root
   changes (except the Lead's own sidecar names).
10. **Never edit product sources to debug.**  Debug-only instrumentation goes in
    `tools/`, `build/tmp/` or `doc/`; product changes are a separate, reviewed
    step (and need a rebuild + long-run restart, i.e. a Lead decision).
11. **Never run two full sidecar builds against the same output paths** or run
    `compile.bat` concurrently with a sidecar build — outputs/map collide.
    Distinct output names (`next`, `next.g1`) are safe.

---

## 8. Repeatable commands (copy/paste)

```powershell
cd E:\079\v6.0Alixander\phoenix

# --- dump analysis (zero risk) ---
powershell -NoProfile -ExecutionPolicy Bypass -File tools\dump_report.ps1 -Latest
powershell -NoProfile -ExecutionPolicy Bypass -File tools\dump_report.ps1 -All

# --- provision gdb locally (idempotent) ---
powershell -NoProfile -ExecutionPolicy Bypass -File tools\setup_gdb_local.ps1

# --- gdb on a sidecar, transcript to build\tmp\gdb_transcript.log ---
powershell -NoProfile -ExecutionPolicy Bypass -File tools\gdb_probe.ps1 `
    -Exe phoenix_main.next.exe -Break main `
    -Post 'bt','info registers rip rsp rbp','x/4i $pc'

# --- -g1 sidecar with line tables (background, ~20-30 min) ---
cmd /c build\tmp\compile_next_g1.bat
# then: gdb build\tmp\phoenix_main.next.g1.exe, break file:line

# --- WER dump settings sanity check ---
$k = 'HKLM:\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps\phoenix_main.exe'
Get-ItemProperty $k | Select-Object DumpFolder,DumpType,DumpCount
```

---

## 9. File index (what this protocol owns)

| path | purpose |
|---|---|
| `tools/dump_report.py` | minidump → markdown: faulting thread, exception, `.pdata`-validated frames, nearest-N symbols below each frame |
| `tools/dump_report.ps1` | wrapper: `-Dump`/`-Latest`/`-All`, per-dump reports under `build\tmp\dumps\` |
| `tools/dump_stack_resolve.py` | first-generation raw stack scanner; still importable (map parser) but prefer `dump_report.py` |
| `tools/setup_gdb_local.ps1` | project-local gdb 17.2 provisioning (no system install) |
| `tools/gdb_probe.ps1` | batch gdb session + transcript, sidecars only |
| `build\tmp\compile_next_g1.bat` | proposed `-g1` sidecar build (`build\tmp\phoenix_main.next.g1.exe`) |
| `doc/v10.1/debugger_transcript.md` | proof and exact gdb transcripts |
| `doc/v10.1/debug_protocol.md` | this document |
| `tools/audit_check.py` | static checker R1 versions / R2 dead code+data / R3 lesson gates (see `doc/v10.1/audit_checker.md`) |
| `doc/v10.1/audit_checker.md` | rule catalogue, baseline counts, how to add rules |
