# Conan ABI re-pin to g++ 16.1.0 — isolated rebuild gate report

Status: **IN PROGRESS / evidence being collected** (this document is written while the
isolated rebuild runs; result tables are filled in at the end of each step).

Goal: make the prebuilt C++ dependency binaries match the compiler that actually
builds Phoenix (`g++ 16.1.0`).  Today `build/conan_home/profiles/default` pins
`compiler.version=13 / libcxx=libstdc++11 / cppstd=gnu17`, while the app links
with gcc 16 — the same ABI-skew class that made the `-D_GLIBCXX_DEBUG` sidecar
link fail.  This gate rebuilds the dependency graph in a **separate conan home**
and proves the app + gtest sidecars build, link and pass tests against it.

## 0. Non-negotiable constraints (all honoured)

| constraint | how it is honoured |
|---|---|
| live gateway (pid 23920, :5097) + 3 llama-servers keep running | no `taskkill`, no `compile.bat`, no service stop; every command is compile/install only |
| do not mutate `build/conan_home` in place | the old home is only **copied** to `build/conan_home_16`; every conan command sets `CONAN_HOME=build\conan_home_16`; production home mtime/contents unchanged |
| no `compile.bat` | sidecar build generated from `runtime_store/compile_last_command.txt` with rewritten `-o`/map/conan flag files |
| no production binary swap | sidecar exe is `build\tmp\phoenix_main.gcc16.exe`; `phoenix_main.exe` untouched |
| isolated build dir | conan generators go to `build/conan16`; logs/scripts under `build/tmp/` |

## 1. Why (the version-consistency incident)

`tools/audit_check.py` R1.1 baseline: conan profile says gcc **13**, actual
compiler is **16.1.0**, and 24 of 30 installed package binaries were produced by
the gcc-13 ABI profile.  The repo already paid for ABI skew once: the
`_GLIBCXX_DEBUG` sidecar link failed because `std::string/vector/map` layout and
symbols changed on one side of the link (`doc/v10.1/crash_hunt_pageheap.md` §1).
Mixing gcc-13-built libstdc++ consumers with a gcc-16-built app is the same class
of risk, even where the current libstdc++ ABI happens to be compatible.

## 2. Inventory — what is consumed, what is prebuilt with gcc 13

Source of truth: `build/conan_cflags_1704327571.txt` +
`build/conan_libs_2028731685.txt` from the last production link, and
`conan list '*' --format=json` against the old home
(archived at `build/tmp/conan_old_inventory.json`,
flat table `build/tmp/conan_old_table.tsv`).

* 31 recipes in the project cache; **30 package binaries**.
* **24 package binaries carry `compiler.version=13`** (prebuilt gcc-13 ABI).
* 6 entries have no compiler binary: header-only `eigen`, `jwt-cpp`,
  `nlohmann_json`; tools/build helpers `cmake`, `nasm`, `msys2`.
* The saved compile command references 22 distinct `build/conan_home/p/...`
  package dirs via `-I`/`-L`; the link line names 68 libraries
  (`drogon, trantor, redis++_static, opencv_*455, sqlite3, lmdb, hiredis,
  jsoncpp, ssl, crypto, z, png16, tiff, webp, jpeg, openjp2, OpenEXR-3_2,
  Imath-3_1, ade, protobuf, quirc, cares, deflate, lzma, ...`).

| package | old binary ABI | class | consumed | action |
|---|---|---|---|---|
| `ade/0.1.2d` | gcc 13 | C++ | transitive | rebuild with gcc 16 (package_id changes) |
| `c-ares/1.25.0` | gcc 13 | C | transitive | rebuild with gcc 16 (package_id changes) |
| `cmake/4.3.3` | no compiler binary (header-only/tool) | tool | build tool | reuse tool binary (no C++ ABI) |
| `drogon/1.9.5` | gcc 13 | C++ | direct | rebuild with gcc 16 (package_id changes) |
| `eigen/3.4.0` | no compiler binary (header-only/tool) | header-only | build tool | no rebuild needed (headers) |
| `gtest/1.14.0` | gcc 13 | C++ | direct | rebuild with gcc 16 (package_id changes) |
| `hiredis/1.2.0` | gcc 13 | C | direct | rebuild with gcc 16 (package_id changes) |
| `imath/3.1.9` | gcc 13 | C++ | transitive | rebuild with gcc 16 (package_id changes) |
| `jsoncpp/1.9.5` | gcc 13 | C++ | direct | rebuild with gcc 16 (package_id changes) |
| `jwt-cpp/0.7.0` | no compiler binary (header-only/tool) | header-only | direct | no rebuild needed (headers) |
| `libdeflate/1.19` | gcc 13 | C | transitive | rebuild with gcc 16 (package_id changes) |
| `libjpeg/9e` | gcc 13 | C | transitive | rebuild with gcc 16 (package_id changes) |
| `libpng/1.6.58` | gcc 13 | C | transitive | rebuild with gcc 16 (package_id changes) |
| `libtiff/4.6.0` | gcc 13 | C | transitive | rebuild with gcc 16 (package_id changes) |
| `libwebp/1.3.2` | gcc 13 | C | transitive | rebuild with gcc 16 (package_id changes) |
| `lmdb/0.9.29` | gcc 13 | C | direct | rebuild with gcc 16 (package_id changes) |
| `msys2/cci.latest` | no compiler binary (header-only/tool) | tool | build tool | reuse tool binary (no C++ ABI) |
| `nasm/2.16.01` | no compiler binary (header-only/tool) | tool | build tool | reuse tool binary (no C++ ABI) |
| `nlohmann_json/3.11.2` | no compiler binary (header-only/tool) | header-only | direct | no rebuild needed (headers) |
| `opencv/4.5.5` | gcc 13 | C++ | direct | rebuild with gcc 16 (package_id changes) |
| `openexr/3.2.3` | gcc 13 | C++ | transitive | rebuild with gcc 16 (package_id changes) |
| `openjpeg/2.5.2` | gcc 13 | C | transitive | rebuild with gcc 16 (package_id changes) |
| `openssl/3.6.3` | gcc 13 | C | transitive | rebuild with gcc 16 (package_id changes) |
| `protobuf/3.21.12` | gcc 13 | C++ | transitive | rebuild with gcc 16 (package_id changes) |
| `quirc/1.2` | gcc 13 | C | transitive | rebuild with gcc 16 (package_id changes) |
| `redis-plus-plus/1.3.7` | gcc 13 | C++ | direct | rebuild with gcc 16 (package_id changes) |
| `sqlite3/3.45.1` | gcc 13 | C | direct | rebuild with gcc 16 (package_id changes) |
| `trantor/1.5.19` | gcc 13 | C++ | direct | rebuild with gcc 16 (package_id changes) |
| `xz_utils/5.8.3` | gcc 13 | C | transitive | rebuild with gcc 16 (package_id changes) |
| `zlib/1.3.2` | gcc 13 | C | transitive | rebuild with gcc 16 (package_id changes) |

Classification: C++ packages (`ade, drogon, gtest, imath, jsoncpp, opencv,
openexr, protobuf, redis-plus-plus, trantor`) are the true ABI-sensitive set;
C libraries are rebuilt too because the conan package id changes with the
compiler setting, which is what keeps the whole graph consistent.

## 3. Environment measured

```
g++ -v        : gcc version 16.1.0 (Rev5, Built by MSYS2 project)
gcc --version : gcc.exe (Rev5, Built by MSYS2 project) 16.1.0
conan --version: Conan version 2.33.0
old profile   : compiler.version=13 libcxx=libstdc++11 cppstd=gnu17 Release x86_64
new profile   : compiler.version=16 libcxx=libstdc++11 cppstd=gnu17 Release x86_64
```

`conan profile show` with `CONAN_HOME=build\conan_home_16` prints
`compiler.version=16` for both host and build profile (verified before the
install started).  Conan 2.33's `settings.yml` already lists gcc `16`/`16.1`,
so no settings.yml edit was needed.

## 4. Exact commands

```powershell
cd E:\079\v6.0Alixander\phoenix

# 1) copy the old home (never edit it) - 2.72 GB, ~1 min
robocopy build\conan_home build\conan_home_16 /E /NFL /NDL /NJH /NJS /NP /R:1 /W:1

# 2) point the isolated profile at gcc 16 (file build\conan_home_16\profiles\default)
#    compiler.version=16 ; everything else identical to production

# 3) rebuild the dependency graph into the isolated home + isolated output dir
pwsh -NoProfile -ExecutionPolicy Bypass -File build\tmp\conan16_install.ps1
#    = CONAN_HOME=build\conan_home_16
#      CC=C:/msys64/mingw64/bin/gcc.exe
#      CXX=C:/msys64/mingw64/bin/g++.exe
#      CXXFLAGS=-include cstdint
#      conan install . --output-folder=build/conan16 --build=missing -s build_type=Release
#        -c tools.cmake.cmaketoolchain:generator=Ninja
#        -c tools.files.download:retry=6 -c tools.files.download:retry_wait=10

# 4) sidecar app build (rewrites the saved production compile command)
pwsh -NoProfile -ExecutionPolicy Bypass -File build\tmp\build_sidecar_gcc16.ps1 -Run

# 5) gtest sidecar build + tests
pwsh -NoProfile -ExecutionPolicy Bypass -File build\tmp\build_gtest_gcc16.ps1 -Run
build\tmp\gtest_runner.gcc16.exe --gtest_color=no
```

The generated commands are saved verbatim for audit:
`build/tmp/conan16_install.ps1`, `build/tmp/build_sidecar_gcc16.ps1`
(→ `build/tmp/build_sidecar_gcc16.bat`, `build/tmp/sidecar_gcc16_cmd.txt`),
`build/tmp/build_gtest_gcc16.ps1` (→ `build/tmp/build_gtest_gcc16.bat`,
`build/tmp/gtest_gcc16_cmd.txt`).

## 5. Rebuild result — per package

| # | package | attempt | old binary (gcc13) | new binary (gcc16) | ~build s |
|---|---|---|---|---|---|
| ade/0.1.2d | `ade/0.1.2d` | 1st | yes (`09460280a113`) | `4f59c90f4f7a` | - |
| c-ares/1.25.0 | `c-ares/1.25.0` | 1st | yes (`81ecd7261373`) | `868dd9105b4b` | 137.7 |
| gtest/1.14.0 | `gtest/1.14.0` | 1st | yes (`e123e874bb91`) | `4defe68e780c` | 26.8 |
| hiredis/1.2.0 | `hiredis/1.2.0` | 1st | yes (`d62f3295c652`) | `06029292d51a` | 4.3 |
| imath/3.1.9 | `imath/3.1.9` | 1st | yes (`8a37376be017`) | `15b49845773b` | 8.1 |
| jsoncpp/1.9.5 | `jsoncpp/1.9.5` | 1st | yes (`8a37376be017`) | `15b49845773b` | 12.8 |
| libdeflate/1.19 | `libdeflate/1.19` | 1st | yes (`74fce2c720f8`) | `bd9ff2223869` | 12.9 |
| lmdb/0.9.29 | `lmdb/0.9.29` | 1st | yes (`0befdba977ab`) | `42327d3cc2f3` | 8.6 |
| openjpeg/2.5.2 | `openjpeg/2.5.2` | 1st | yes (`74fce2c720f8`) | `bd9ff2223869` | 28.6 |
| quirc/1.2 | `quirc/1.2` | 1st | yes (`1e27c22e55dc`) | `04b6f7792ce3` | 4.6 |
| sqlite3/3.45.1 | `sqlite3/3.45.1` | 1st | yes (`c4cc315a662f`) | `7b777fc40343` | 69.7 |
| zlib/1.3.2 | `zlib/1.3.2` | 1st | yes (`74fce2c720f8`) | `bd9ff2223869` | 9.0 |
| libjpeg/9e | `libjpeg/9e` | 1st | yes (`74fce2c720f8`) | `bd9ff2223869` | 99.9 |
| libpng/1.6.58 | `libpng/1.6.58` | 1st | yes (`cf51190da154`) | `0fda861a5415` | 8.6 |
| libwebp/1.3.2 | `libwebp/1.3.2` | 1st | yes (`9ce816a4df2d`) | `0cc6af3e8d1f` | 19.3 |
| openexr/3.2.3 | `openexr/3.2.3` | 1st | yes (`55e9c5588fb1`) | `26072782bc19` | 27.8 |
| protobuf/3.21.12 | `protobuf/3.21.12` | 1st | yes (`287230967a96`) | `c8ba7f2c2761` | 37.2 |
| redis-plus-plus/1.3.7 | `redis-plus-plus/1.3.7` | 1st | yes (`6a29f2e3a5cf`) | `761a00aad261` | 9.2 |
| xz_utils/5.8.3 | `xz_utils/5.8.3` | 1st | yes (`83e49ad515a2`) | `715f492b1873` | 33.4 |
| openssl/3.6.3 | `openssl/3.6.3` | 1st | yes (`42eca769e97c`) | `07788dc0721f` | 264.9 |
| libtiff/4.6.0 | `libtiff/4.6.0` | 2nd | yes (`c2fb56dea68e`) | `dd84fcd3012c` | - |
| opencv/4.5.5 | `opencv/4.5.5` | 2nd | yes (`cf82c3b00438`) | `a5b752a12bcd` | 460.1 |
| trantor/1.5.19 | `trantor/1.5.19` | 2nd | yes (`def3613996ce`) | `91a372c1df98` | 13.5 |
| drogon/1.9.5 | `drogon/1.9.5` | 2nd | yes (`ecaee9b3088b`) | `1ff7b0de73ea` | 31.7 |

Attempt 1 = 20 packages, 905.1 s (failed on libtiff: Anaconda WebP in PATH). Attempt 2 resumed from cache: libtiff/trantor/drogon/opencv, 547.0 s total. Per-package seconds are completion-time deltas within an attempt (first row of each attempt is '-' because the attempt start time is not a package completion).

Build logs: `build/tmp/conan16_install_attempt1.log` (exit 1, 905.1 s, stopped at
libtiff) and `build/tmp/conan16_install.log` (exit 0, 547.0 s; finished with
`CONAN16_INSTALL_EXIT=0 ELAPSED_SEC=547`).

Deltas measured from conan package folder completion times in
`build/conan_home_16` (consecutive completion times within an attempt ≈ build
wall time; includes configure + compile).  Attempt 2 reused the 20 packages built
in attempt 1 from the isolated home's cache, so only 4 packages were compiled in
attempt 2; the table's `attempt` column shows which pass built a package.

## 6. Sidecar application build

The production compile line was rewritten mechanically:

| production | sidecar |
|---|---|
| `-o phoenix_main.exe` | `-o build\tmp\phoenix_main.gcc16.exe` |
| `@"build\conan_cflags_1704327571.txt"` | `@build\conan_cflags_gcc16.txt` (pkg-config against `build/conan16`, verified to reference `conan_home_16`) |
| `@"build\conan_libs_2028731685.txt"` | `@build\conan_libs_gcc16.txt` |
| `-Wl,-Map=phoenix_main.map` | `-Wl,-Map=build\tmp\phoenix_main.gcc16.map` |

Command actually executed: `build/tmp/sidecar_gcc16_cmd.txt`
(log `build/tmp/compile_gcc16.log`, wall time **1270.5 s ≈ 21.2 min**).

| artifact | value |
|---|---|
| sidecar binary | `build\tmp\phoenix_main.gcc16.exe` |
| size | 48,434,880 bytes |
| sha256 | `02B93DC43A26CE9E4564E51B68FE30B0D020CBA2ED9960312A6A8A48DFBFF18E` |
| link map | `build\tmp\phoenix_main.gcc16.map` (23,943,069 B) |
| conan flags | `build/conan_cflags_gcc16.txt` / `build/conan_libs_gcc16.txt` — **all paths under `build/conan_home_16`** (verified: 0 references to `build/conan_home/p/` remain in the command or log) |
| build log tail | `EXIT=0` |
| production binary | `phoenix_main.exe` unchanged: 47,062,417 B, sha256 `5508D3754628947D3B943D206E370ED10E87C9F6219A2CEAAEF4E569B25483D8`, mtime 2026-11-16 19:56:15 |

Load proof (new-ABI imports resolve): the sidecar was started under gdb and stopped
at `main` before any app logic ran — breakpoint hit, `main+33`, all DLL imports
resolved (transcript `build/tmp/gdb_sidecar_gcc16.log`, exit 0).  No missing-DLL
or ABI loader error.

Conan install totals: attempt 1 = 905.1 s (20 packages, stopped on libtiff), attempt 2
= 547.0 s (resumed from cache: libtiff/trantor/drogon/opencv) → **1452.1 s ≈ 24.2 min**
for 24 package rebuilds.  Root cause of the attempt-1 failure and the fix:
libtiff's `find_package(WebP)` picked `E:/Anaconda/Library/WebP/cmake/WebPConfig.cmake`
because Anaconda's `Library/bin` was in `PATH` and CMake treats PATH entries as
Windows prefixes; that Anaconda config references a missing
`D:/Anaconda/Library/lib/libsharpyuv.lib`.  The install script now strips
`anaconda|conda` from PATH for the isolated build only (no system change).

## 7. Test evidence (new conan home)

{{TEST_RESULT}}

## 8. Checker integration

* **R1.4 (vendor patch reproducibility)** was updated to the authoritative vendor
  truth in the same change: R1.4a calls `tools/vendor_patch_verify.ps1`
  (canonical `phoenix_vendor_733dd8df.patch` → `vendor_733dd8df.sha256.json`),
  R1.4b only requires non-superseded `*.patch` files to apply, R1.4c requires the
  dirty-file set to equal the recorded 10 files.  New verdict: **PASS** (verify
  ~4 s, 10/10 byte-identical, dirty set 10/10), catalogue updated in
  `doc/v10.1/audit_checker.md`.
* **R1.1 (conan ABI) stays FAIL on purpose until the Lead performs the swap**:
  `build/conan_home/profiles/default` is still gcc 13.  No mutation was made to
  the production home.  After the swap procedure below, R1.1 is expected to PASS
  (the checker resolves the actual g++ and the production profile).  The staged
  isolated home is not invisible to the rule by design — production truth is what
  R1.1 checks.

## 9. Swap procedure for the Lead (safe, reversible)

Preconditions: choose a maintenance window; the live run will be restarted by
this procedure, so coordinate it with the user/long-run owner.

```powershell
cd E:\079\v6.0Alixander\phoenix

# 0) evidence snapshot (already produced by this gate)
#    build\tmp\phoenix_main.gcc16.exe + .map + sha256 in build\tmp\evidence\

# 1) stop services cleanly (NOT part of this build task - Lead owns this window)
#    e.g. the project's normal shutdown path / supervisor stop, never taskkill mid-flight

# 2) move the old ABI home aside (rollback point)
Move-Item build\conan_home      build\conan_home_gcc13_backup
Move-Item build\conan_home_16   build\conan_home
#    the moved home already contains profiles\default with compiler.version=16
#    and all gcc16 package binaries; old gcc13 binaries are still inside (harmless)

# 3) rebuild production binaries with the normal script (uses CONAN_HOME=build\conan_home)
#    compile.bat runs conan install --build=missing; with the gcc16 profile it will
#    find the gcc16 binaries already present -> no long rebuild, only link.
compile.bat
#    then rebuild tests if needed: compile_gtest.bat

# 4) restart the gateway exactly as the long run expects, then verify:
#    - GET /api/system/status (port 5097) shows the expected version
#    - `python tools\audit_check.py --only R1` -> R1.1 PASS expected

# 5) rollback (if anything is wrong):
#    stop services, then:
#    Move-Item build\conan_home build\conan_home_gcc16_attempt
#    Move-Item build\conan_home_gcc13_backup build\conan_home
#    re-run the normal build; the old home was never modified.
```

Notes:

* The staged home is a **superset** (old gcc13 binaries copied + new gcc16
  binaries built).  Disk cost ≈ +2.7 GB old copy + net new package binaries.
  Optional later cleanup: `conan remove` the unused gcc13 package ids once the
  new build is green.
* Do not `git add` `build/` artifacts; they are local evidence only.

## 10. Artifacts

| path | content |
|---|---|
| `build/conan_home_16/` | isolated gcc16 conan home (copy + rebuilt binaries) |
| `build/conan_home_16/profiles/default` | gcc16 profile |
| `build/conan16/` | conan install output (generators/`.pc` files for the new graph) |
| `build/conan_cflags_gcc16.txt`, `build/conan_libs_gcc16.txt` | pkg-config flags used by the app sidecar |
| `build/tmp/phoenix_main.gcc16.exe` (+ `.map`) | app sidecar binary |
| `build/tmp/gtest_runner.gcc16.exe` | gtest sidecar binary |
| `build/tmp/conan16_install.log`, `compile_gcc16.log`, `compile_gtest_gcc16.log` | build logs |
| `build/tmp/conan_old_inventory.json`, `conan_old_table.tsv`, `conan_inventory.md` | inventory evidence |
| this document | gate report |

