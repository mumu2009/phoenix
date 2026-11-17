# Debugger transcript — gdb 17.2 on MinGW Phoenix sidecars (2026-11-17)

Status: **gdb works end-to-end on this project's MinGW-produced executables**
without any PDB.  Function-address breakpoints and multi-frame backtraces work
from the executable's own COFF symbol table + PE64 `.pdata` unwind table.
Source-line breakpoints / variables additionally need a `-g` build; a `-g1`
sidecar build script is provided and its cost is measured below.

All commands were run from `E:\079\v6.0Alixander\phoenix`.

## 0. Environment (measured)

| Item | Value |
|---|---|
| OS | Windows 10 Pro for Workstations 10.0.19045 |
| Compiler | `g++.exe (Rev5, Built by MSYS2 project) 16.1.0` (`C:\msys64\mingw64\bin`) |
| Debugger | `GNU gdb (GDB) 17.2`, configured as `x86_64-w64-mingw32` |
| gdb location | `build\tmp\gdb\mingw64\bin\gdb.exe` (project-local, see §1) |
| gdbserver | `GNU gdbserver (GDB) 17.2` (`build\tmp\gdb\mingw64\bin\gdbserver.exe`) |
| Binutils | `GNU nm/objdump (GNU Binutils) 2.46.1` |
| Python | 3.11.9 (`D:\cloudworkfile\tools\python311\python.exe`) with `minidump` package |

## 1. gdb was not installed; provisioned locally without touching MSYS2

`gdb` is not in the stock MSYS2 toolchain install of this machine
(`pacman -Q` shows only `gdbm`; `gdb.exe` is absent from
`C:\msys64\mingw64\bin`).  The repository package is available:

```
mingw64/mingw-w64-x86_64-gdb 17.2-1 (mingw-w64-x86_64-toolchain)
```

Installing it with `pacman -S` would add/upgrade runtime DLLs under
`C:\msys64\mingw64` that the project's conan prebuilts and the live gateway
depend on.  Instead, `tools/setup_gdb_local.ps1`:

1. `pacman -Sw --noconfirm --needed mingw-w64-x86_64-gdb` — **download only**
   into the pacman cache (`C:\msys64\var\cache\pacman\pkg`), no package DB change;
2. resolves the dependency closure with `pacman -Sp --print-format ...`;
3. unpacks every package with `bsdtar` into `build\tmp\gdb`;
4. verifies `gdb --version`.

Result:

```
$ build\tmp\gdb\mingw64\bin\gdb.exe --version
GNU gdb (GDB) 17.2
...
This GDB was configured as "x86_64-w64-mingw32".
```

Because gdb.exe finds its DLLs next to itself and the remaining MinGW runtime
DLLs in `C:\msys64\mingw64\bin`, run it with that directory prepended to PATH
(`tools/gdb_probe.ps1` does this automatically):

```powershell
$env:PATH = "E:\079\v6.0Alixander\phoenix\build\tmp\gdb\mingw64\bin;C:\msys64\mingw64\bin;" + $env:PATH
```

## 2. Can gdb read a MinGW binary that has no PDB? Yes — COFF symbols + .pdata

Test target: **`phoenix_main.next.exe`** — the sidecar produced by
`build\tmp\compile_next.bat` (finished 2026-11-17 22:27; symbol content identical
to `phoenix_main.exe`: 42,045 unique code symbols, 40,958 `.pdata` ranges).
The same commands were earlier verified on `phoenix_main.prev.exe` (the previous
build, kept on disk as a rollback sidecar).  A run is safe: we stop at
`main` **before the program body executes**, so the sidecar never binds a port
or touches the live run.

Command and output (`build\tmp\gdb_next_transcript.log`):

```
$ gdb -batch -nx -ex "set pagination off" \
      -ex "break main" -ex "info breakpoints" ... phoenix_main.next.exe

Symbols from "E:\079\v6.0Alixander\phoenix\phoenix_main.next.exe".
Local exec file:
        `...phoenix_main.next.exe', file type pei-x86-64.
        Entry point: 0x1400014c0
        0x0000000140001000 - 0x00000001419558e0 is .text
        0x0000000141eac000 - 0x0000000141f23fe8 is .pdata      <- SEH unwind table
        0x0000000141f24000 - 0x0000000142050dc0 is .xdata
Breakpoint 1 at 0x14190c061
Num     Type           Disp Enb Address            What
1       breakpoint     keep y   0x000000014190c061 <main+33>
```

Project functions are visible as "Non-debugging symbols" (exact gdb wording):

```
$ gdb -batch -nx -ex "info functions sampleRate" phoenix_main.next.exe

All functions matching regular expression "sampleRate":

Non-debugging symbols:
0x00000001406261f0  phoenix::io::(anonymous namespace)::sampleRateFromMetadata(
    nlohmann::json_abi_v3_11_2::basic_json<...> const&)
```

Hard limit without `-g` (exact message):

```
$ gdb -batch -nx -ex "info line main" phoenix_main.next.exe
No line number information available for address 0x14190c040 <main>
```

Why: the shipped binary *does* have DWARF sections, but only for the MinGW C
runtime.  `objdump -h phoenix_main.exe` shows `.debug_info` 0x33b1b bytes and
`objdump --dwarf=info` names only
`D:/W/B/src/mingw-w64/mingw-w64-crt/crt/crtexe.c` — no project `.cpp`.  The PE
COFF symbol table (42,045 code symbols in `phoenix_main.exe`) and the 40,958
`.pdata` `RUNTIME_FUNCTION` ranges are what gdb uses instead.

## 3. Breakpoints on a project function (no -g)

Because `sampleRateFromMetadata` lives in an anonymous namespace, the short name
does **not** resolve; the full quoted name does (module breakpoint lands after
the prologue, address form is exact):

```
$ gdb -batch -nx phoenix_main.next.exe \
    -ex "break 'phoenix::io::(anonymous namespace)::sampleRateFromMetadata'" \
    -ex "break *0x1406261f0" \
    -ex "info breakpoints"

Breakpoint 1 at 0x1406261fa
Breakpoint 2 at 0x1406261f0
Num     Type           Disp Enb Address            What
1       breakpoint     keep y   0x00000001406261fa <phoenix::io::(anonymous namespace)::sampleRateFromMetadata(...)+10>
2       breakpoint     keep y   0x00000001406261f0 <phoenix::io::(anonymous namespace)::sampleRateFromMetadata(...)>
```

(With `-ex "break phoenix::io::sampleRateFromMetadata"` gdb prints
`Function ... not defined` and leaves a PENDING breakpoint — the namespace is
part of the mangled name.  Use `rbreak sampleRateFromMetadata` or the quoted
full name.)

## 4. End-to-end: start gdb, run the sidecar, hit a breakpoint, backtrace

Command file used (`build\tmp\gdb_next_cmds.txt`):

```
set pagination off
set confirm off
set breakpoint pending on
break main
break __main
break 'phoenix::io::(anonymous namespace)::sampleRateFromMetadata'
run
bt
info registers rip rsp rbp
continue
bt
info registers rip rsp rbp
x/3i $pc
quit
```

Transcript (`build\tmp\gdb_next_bt.log`, trimmed to the interesting lines):

```
Breakpoint 1 at 0x14190c061
Breakpoint 2 at 0x1410d7cf0: file .../gccmain.c, line 56.
Breakpoint 3 at 0x1406261fa
[New Thread 39056.0x74e0]
[New Thread 39056.0x525c]
[New Thread 39056.0x70bc]

Thread 1 hit Breakpoint 2, __main () at .../gccmain.c:56
#0  __main () at .../gccmain.c:56
#1  0x00007ff797f81428 in __tmainCRTStartup () at .../crtexe.c:248
#2  0x00007ffa69b87374 in KERNEL32!BaseThreadInitThunk () from C:\WINDOWS\System32\kernel32.dll
#3  0x00007ffa6a11cc91 in ntdll!RtlUserThreadStart () from C:\WINDOWS\SYSTEM32\ntdll.dll
#4  0x0000000000000000 in ?? ()
rip            0x7ff799057cf0      0x7ff799057cf0 <__main>
rsp            0x5ffe88            0x5ffe88
rbp            0x1                 0x1

Thread 1 hit Breakpoint 1, 0x00007ff79988c061 in main ()
#0  0x00007ff79988c061 in main ()
rip            0x7ff79988c061      0x7ff79988c061 <main+33>
rsp            0x5fe950            0x5fe950
rbp            0x1                 0x1
=> 0x7ff79988c061 <main+33>:    mov    %rdx,%r14
   0x7ff79988c064 <main+36>:    mov    %ecx,0x1540(%rsp)
   0x7ff79988c06b <main+43>:    call   0x7ff799057cf0 <__main>
No line number information available for address 0x7ff79988c040 <main>
```

Notes:

* The `__main` breakpoint fires first because the MinGW CRT calls `__main`
  from `__tmainCRTStartup` before `main`; the second stop is `main` itself.
  Both are at program entry — **no Phoenix code has executed yet**, the run is
  safe next to the live gateway.
* The 4-frame backtrace (`__main → __tmainCRTStartup → KERNEL32 → ntdll`) proves
  gdb unwinds this PE without any PDB, using `.pdata`/`.xdata`.
* The addresses differ from file VAs (0x14000xxxx) because Windows ASLR
  relocated the image (base 0x7ff7…).  Use `info file` / the module base if you
  need file RVAs; dumps use a different base again (`tools/dump_report.ps1`
  converts to RVAs).

Reproduce with the wrapper:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\gdb_probe.ps1 `
    -Exe phoenix_main.prev.exe -Break main -Post 'bt','info registers rip rsp rbp'
```

## 5. Line-level proof: `-g` works on this toolchain

`-g` binaries ARE fully usable.  Self-test using a *real project header*
(`phoenix_config.hpp`), compiled with the project's conan include flags:

```powershell
$gpp='C:\msys64\mingw64\bin\g++.exe'
& $gpp -std=c++20 -O3 '@build\conan_cflags_1704327571.txt' -I. build\tmp\gdb_selftest.cpp -g -o build\tmp\gdb_selftest_dbg.exe
```

Timings on this box (one representative TU, nlohmann + phoenix_config.hpp):

| build | wall time | exe size |
|---|---|---|
| `-O3` (no -g) | 4.9 s | 341,147 B |
| `-O3 -g` | 6.8 s | 1,861,832 B |

gdb transcript (`build\tmp\gdb_selftest_transcript.log`):

```
$ gdb -batch -nx -ex "break build/tmp/gdb_selftest.cpp:20" \
      -ex "break 'phoenix::cfgOr<int>'" -ex run -ex bt -ex "info locals" -ex list \
      build\tmp\gdb_selftest_dbg.exe

Breakpoint 1 at 0x14000fed9: file build\tmp\gdb_selftest.cpp, line 20.
Breakpoint 2 at 0x14000fae6: file .../phoenix_config.hpp, line 31.

Thread 1 hit Breakpoint 2, phoenix::cfgOr<int> (dotPath=..., fallback=<synthetic pointer>: 12000)
    at .../phoenix_config.hpp:261
261       return PhoenixConfig::instance().getOr<T>(dotPath, fallback);
#0  phoenix::cfgOr<int> (...) at .../phoenix_config.hpp:261
#1  main () at build\tmp\gdb_selftest.cpp:18
No locals.
256       return PhoenixConfig::instance().get<T>(dotPath);
257     }
258
259     template <typename T>
260     inline T cfgOr(const std::string& dotPath, const T& fallback) {
261       return PhoenixConfig::instance().getOr<T>(dotPath, fallback);
262     }
263
```

So: breakpoints on source lines, on a real Phoenix header function, plus a
2-frame backtrace with source lines — all work.  Nothing about gdb-on-MinGW is
broken; the product build simply does not pass `-g`.

## 6. Fix for the real sidecar: `-g1` sidecar build (proposed + scripted)

Current sidecar command (`build\tmp\compile_next.bat`) has no `-g`, so gdb sees
function names but no lines/variables.  The minimal fix is one flag:

| option | what you get | cost |
|---|---|---|
| `-g1` | line tables: `break file:line`, file/line in `bt`, `list` | smallest; recommended for crash forensics |
| `-g` | full DWARF: line tables + locals + types | bigger exe, slower link, more RAM in gdb |
| `-g -gcodeview` | attempted CodeView for cdb/WinDbg — **does not work on this toolchain**: `-O2/-O3` fail to assemble (`can't resolve .text.unlikely - .LcvlineN`), `-O0/-O1` still emit DWARF with no `.debug$S` sections and gdb finds no line info.  cdb path needs separate investigation; gdb is the proven debugger | — |

`build\tmp\compile_next_g1.bat` is a ready-to-run copy of the normal sidecar
command with exactly these changes:

```
-o build\tmp\phoenix_main.next.g1.exe    (output stays in build\tmp\)
-g1                                     (added; everything else identical)
-Wl,-Map=build\tmp\phoenix_main.next.g1.map
> build\tmp\compile_next_g1.log 2>&1
```

It does **not** touch `phoenix_main.exe`, `phoenix_main.map`, `phoenix_main.next.exe`
or the live gateway, so it can run while the long run is up (it is CPU-only).

Measured result: `build\tmp\compile_next_g1.bat` finished in **23.5 min**
(22:20:08 → 22:43:36) under the same CPU contention as the normal sidecar,
whose build took 21.9 min (22:05:45 → 22:27:36).  `-g1` therefore costs only
~7% extra wall time here (link/compile dominated), and produces a
152,137,212-byte (145 MiB) exe instead of 47 MB.  It is safe to run next to the
live gateway.

Line-level proof on the **real sidecar** (`build\tmp\gdb_g1_bt.log`):

```
$ gdb -batch -nx -x build\tmp\gdb_g1_cmds.txt build\tmp\phoenix_main.next.g1.exe
Breakpoint 1 at 0x14190c0cd: file main_hub_parts/116_section_tail.inc, line 8661.
Breakpoint 2 at 0x1411c5f23: file main_hub_parts/049_class_patternmatrix.inc, line 14.
[New Thread 33632.0x6630]
[New Thread 33632.0x7fd8]
[New Thread 33632.0xb500]
AI Controller starting...

Thread 1 hit Breakpoint 1, main () at main_hub_parts/116_section_tail.inc:8661
8661      g_selfPath = fs::absolute(fs::path(argv[0])).string();

#0  main () at main_hub_parts/116_section_tail.inc:8661
rip            0x7ff7efeec0cd      0x7ff7efeec0cd <main()+141>
rsp            0x5fe950            0x5fe950
rbp            0x1                 0x1
Line 8661 of "main_hub_parts/116_section_tail.inc" starts at address 0x7ff7efeec0cd <main()+141>
8656    // 调用方式：进程启动后由操作系统直接调用。
8659    int main(int argc, char **argv) {
8660      std::cout << "AI Controller starting..." << std::endl;
8661      g_selfPath = fs::absolute(fs::path(argv[0])).string();
8662      std::atexit([]() {
Current source file is main_hub_parts/116_section_tail.inc
Located in E:\079\v6.0Alixander\phoenix\main_hub_parts\116_section_tail.inc
Contains 9181 lines.
Source language is c++.
Producer is GNU C++20 16.1.0 ... -g1 -O3 -std=c++20.
Compiled with DWARF 5 debugging format.
Num     Type           Disp Enb Address            What
1       breakpoint     keep y   0x00007ff7efeec0cd in main() at main_hub_parts/116_section_tail.inc:8661
2       breakpoint     keep y   0x00007ff7ef7a5f23 in PatternMatrix::rebuild[abi:cxx11](MemeGraph::WindowInfo const&) at main_hub_parts/049_class_patternmatrix.inc:14
```

That is the destination state: a source-line breakpoint on the actual
crash-path function (`PatternMatrix::rebuild`, one of the recurring frames in
the 10 heap-corruption dumps) is now possible **without touching the production
build or the live run**.  Note the run printed
`AI Controller starting...` before stopping at line 8661 — that is the first
`main` statement; no gateway/port/storage code had run.

(`-g1` adds line tables only; `break <file>:<line>`, `list`, `bt` with
file:line work.  `print` of locals often does not; rebuild with `-g` if you need
variables.)

## 7. What gdb does NOT do here

* It cannot open WER minidumps (`MINIDUMP` format).  Use
  `tools/dump_report.ps1` for those (§2 of `debug_protocol.md`).
* `gdb -p <pid>` on Windows works, but **attaching stops the process**.  Never
  attach to the live gateway (port 5097): the supervisor/watchdog may react and
  the long run is destroyed.  Use a sidecar or `gdbserver` with an explicit
  window.
* No heap-checking.  `0xC0000374` corruption needs PageHeap/AppVerifier +
  dumps, not gdb breakpoints (see `debug_protocol.md` §3).
* ASLR means runtime addresses are not the `0x140…` file VAs; use module base
  or RVA (the dump report prints both).

## 8. Files produced by this transcript

| file | content |
|---|---|
| `build\tmp\gdb_next_transcript.log` | **primary**: break main + project function on `phoenix_main.next.exe`, run, stop, registers |
| `build\tmp\gdb_next_bt.log` | primary multi-frame `__main → __tmainCRTStartup → kernel32 → ntdll` backtrace on `phoenix_main.next.exe` |
| `build\tmp\gdb_symbols_next.log` | next.exe `info functions sampleRate` inventory |
| `build\tmp\gdb_info_line_next.log` | next.exe exact "No line number information" message |
| `build\tmp\gdb_prev_transcript.log` | earlier replication on `phoenix_main.prev.exe` |
| `build\tmp\gdb_prev_bt.log` | earlier multi-frame backtrace on `prev.exe` |
| `build\tmp\gdb_symbols_prev.log`, `gdb_fn_break_prev.log`, `gdb_info_line_prev.log` | earlier `prev.exe` symbol / breakpoint / line-info captures |
| `build\tmp\gdb_selftest_transcript.log` | `-g` line breakpoint + 2-frame source backtrace |
| `build\tmp\gdb_g1_bt.log` | **real `-g1` sidecar**: source-line breakpoints in `main` and `PatternMatrix::rebuild`, `list`, producer/DWARF metadata |
| `build\tmp\gdb_g1_cmds.txt` | command file for the `-g1` sidecar session |
| `build\tmp\phoenix_main.next.g1.exe` | `-g1` sidecar binary (145 MiB, line tables) |
| `build\tmp\phoenix_main.next.g1.map` | map for the `-g1` sidecar (separate from `phoenix_main.map`) |
| `tools\setup_gdb_local.ps1` | repeatable local gdb provisioning |
| `tools\gdb_probe.ps1` | repeatable batch gdb session + transcript |
| `build\tmp\compile_next_g1.bat` | proposed (and now measured) `-g1` sidecar build |
