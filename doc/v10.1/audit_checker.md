# Audit checker (R1 versions / R2 dead code+data / R3 lesson gates)

`tools/audit_check.py` is a **read-only, stdlib-only** static checker that turns
our real incidents into rules.  It exists so the project stops re-learning the
same lessons: version drift (R1), dead code/data (R2), and repeat-offence bug
patterns (R3) are checked in one command and produce one machine-readable JSON +
one human report.

Design contract:

* **Read-only.** It never edits product sources, never runs `compile.bat`, never
  touches the live gateway (`:5097`), and never deletes anything.
* **Stdlib only.** Python 3.11+ (the project's `python` works); no pip packages.
* **Windows-friendly.** Uses `pathlib`/`subprocess` with timeouts and tolerant
  encodings; git/nvcc/g++/conan/nvidia-smi probes are optional
  (`--no-subprocess` keeps the tool fully static).
* **Evidence, not verdicts.** Every finding carries file:line evidence and a
  rationale naming the incident the rule encodes.  Heuristic rules are marked
  and can be false positives; a human decides.
* **R2 does not own deletion.** `deadcode-audit` owns the 20.9 GB asset
  classification and any deletion plan.  R2 only *consumes*
  `build/tmp/dead_assets.json` when it appears; otherwise its own checks are
  advisory only.

## How to run

```powershell
cd E:\079\v6.0Alixander\phoenix

python tools\audit_check.py                 # all groups, writes both files
python tools\audit_check.py --only R1,R3    # subset
python tools\audit_check.py --no-subprocess # skip git/nvcc/g++/conan/nvidia-smi
python tools\audit_check.py --fail-on warn  # stricter CI gate
python tools\audit_check.py --json out.json --md out.md --top 10
```

Outputs (defaults):

| file | content |
|---|---|
| `build/tmp/audit_check.json` | tool/version metadata, per-group summary, every rule (id/group/severity/rationale/status/stats/findings+evidence), top findings |
| `build/tmp/audit_check.md` | summary table, top findings, per-rule detail with evidence and rerun command |

Exit code: `0` = no FAIL; `1` = at least one FAIL (or WARN when `--fail-on warn`).
A rule crash is contained and reported as that rule's FAIL, so one broken rule
cannot hide the rest.

## Rule catalogue (id → what it checks → incident it encodes)

### R1 VERSION CONSISTENCY — the user's explicit worry: "don't die of version problems"

| id | severity | checks | incident / rationale |
|---|---|---|---|
| R1.1 | error | actual `g++` version (resolved from PATH/MSYS2) vs `build/conan_home/profiles/default` `compiler.version/libcxx/cppstd`; samples 400 `conaninfo.txt` for corroboration | Conan prebuilts (drogon/opencv/sqlite3/...) are linked into the same exe. A different compiler/ABI is the same failure class as the `_GLIBCXX_DEBUG` link failure that killed the debug sidecar (`debug_protocol.md` §7.4) |
| R1.2 | warn | `conanfile.txt` requirements vs `conan list '*'` cache refs (+ CMakeDeps version files) | `conanfile.txt` is the dependency contract; a silently different resolved version is a rebuild change nobody reviewed |
| R1.3 | error | `nvcc --version` toolkit vs `nvidia-smi` CUDA UMD version (and GPU arch note) | Toolkit newer than the driver compiles kernels that fail at runtime ("no kernel image available"); CUDA >=13 dropped sm_50/60/70, GTX 1660 is sm_75 |
| R1.4 | error | **R1.4a** `tools/vendor_patch_verify.ps1` → canonical `llama_server_mods/phoenix_vendor_733dd8df.patch` applied to pristine `733dd8df` must reproduce `vendor_733dd8df.sha256.json` byte-for-byte; **R1.4b** every root `*.patch` NOT in `SUPERSEDED.md` must apply cleanly (canonical patch and superseded per-feature patches exempt); **R1.4c** live `git status --porcelain` file set must equal the recorded 10-file set | The old per-feature patch chain drifted (2 corrupt, 5 conflicting, 1 not applied) and could not reproduce the live tree — same failure class as the patch-defect incident where the enc-dec split path leaked template tokens / `GGML_ASSERT`'ed under load. Authoritative truth: `doc/v10.1/vendor_patch_stack.md` + hash file + canonical patch |
| R1.5 | error | `Python314/include/patchlevel.h` `PY_VERSION` vs `python314.dll` version resource vs `Python314/python.exe --version` vs `python314.lib` | 3.14.3 headers with a different DLL patch level corrupt PyObject layouts; this box also has a second Python 3.11 on PATH |
| R1.6 | warn | `local_onnx.cpp` bridge mode (Python subprocess, no C API) + embedded `onnxruntime` package version queried through `Python314/python.exe` | A missing/old onnxruntime silently degrades every audio/video model to the deterministic fallback |
| R1.7 | warn | `version.hpp` (10.0.0 "Gareth") vs newest `doc/v*` shelf, README title, manifest, doc index | `/api/system/status` reports `version.hpp`; humans/scripts read README/doc. Divergence is the "which build am I running?" trap |
| R1.8 | error | persisted `runtime_store/autonomy_state.json` enabled-like keys vs `config/phoenix.json` enabled switches; maps state spellings (`agiEnabled`, `studyExpert.enabled`) to config dot-paths | v8.3 incident: exportState/importState round-trips contained enabled switches, so a restored runtime_store silently re-enables/disables modules after restart (we found 4+; current baseline: 7) |

### R2 DEAD CODE / DATA — heuristic evidence, never auto-delete

| id | severity | checks | notes |
|---|---|---|---|
| R2.0 | info | consumes `build/tmp/dead_assets.json` from `deadcode-audit` when present: counts verdicts, flags paths that no longer exist | owner: deadcode-audit; this rule is the integration point (schema below) |
| R2.1 | warn | root `*.cpp` not reachable from `build/compile_sources.txt`, `compile.bat COMMON_SOURCES`, `main.cpp`, `compile_gtest.bat` or the RDK script; root `*.hpp` never `#include`d by any product source | heuristic: deliberate stubs (RDK), separate binaries and generated lists are excluded/marked; verify with deadcode-audit before deleting |
| R2.2 | info | config dot-paths whose leaf identifier never appears in product sources (identifier-set scan of `.cpp/.hpp/.inc/.addons/.module_overrides`) | heuristic: dynamic paths/env-only reads can be false positives; the v8.0 config audit found hundreds of dead keys |
| R2.3 | info | large (>=32 MiB) `runtime_store` model/data files whose basename appears in no config, manifest.json or nearby `*.json`; bounded depth/time because runtime_store is ~200 GB | advisory only; deadcode-audit owns deletion |
| R2.4 | info | stale `phoenix_main*.exe` / `gtest_*.exe` / `build/tmp/phoenix_main*.exe` (zero-byte, >7d older than current, prev/dbg/edge_disabled variants) | prevents "which binary produced this dump/behaviour" mix-ups (see `debug_protocol.md`) |

### R3 LESSON GATES — patterns from our own postmortems

| id | severity | checks | incident it encodes |
|---|---|---|---|
| R3.1 | error | mutable `static std::container` declarations (function/lambda/namespace scope), excluding `static const/constexpr/thread_local` and static *functions* returning containers | 0xC0000374 postmortem (`crash_forensics_2026-11-15.md`): `explorePaths`/`namedRealPath`/`sectionDropStreak` were function-level mutable statics mutated concurrently (push_back/sort/operator[] while another mission thread iterated) → use-after-free → heap metadata corruption detected minutes later in ntdll. Fix pattern: immutable magic static or mutex-wrapped scope counter |
| R3.2 | error | `mcpActionMap_` / `cache_` / `sessions_`-style container **uses** (not declarations) with no lock in the enclosing function; exempts explicit "Caller holds ..." comments and caller-locked helpers; brace-matched function regions, not a line window | Same postmortem class: a member written under a mutex in one method and read unlocked elsewhere; small windows fail only under concurrent missions |
| R3.3 | error | assignments to enabled-like fields inside real `fromJson/importState/loadState/restoreState/deserialize` function bodies (comment/string safe, brace-scoped body) | v8.3 incident: state restore overwrote config switches (the R1.8 persistent form of the same bug) |
| R3.4 | warn | `#if 0` blocks, commented-out `#include` lines (main.cpp hub), `if (false)`/`if (0)` branches | Disabled code is invisible to build checks and reviewers; main.cpp has hub includes "disabled due to pre-existing compilation errors", which are latent feature gaps |
| R3.5 | warn | lock site with a network/tool/LLM call inside the same brace scope (`generate(`, `tcpExchange`, `invokeAddonTool`, `callMcpTool`, `subprocess`, HTTP...) | Coarse locks around IO serialised the gateway; the unified-pipeline refactor had to move file IO outside mutexes. Holding a lock over an LLM/tool call risks priority inversion/deadlock |

## Baseline (2026-11-17 23:35, current tree)

Command: `python tools\audit_check.py` — 18 rules, ~20 s (R1.4 runs the 4 s vendor verify).

| group | status | rules | PASS | WARN | FAIL | findings |
|---|---|---|---|---|---|---|
| R1 | **FAIL** | 8 | 5 | 1 | 2 | 3 |
| R2 | **WARN** | 5 | 2 | 3 | 0 | 4 |
| R3 | **FAIL** | 5 | 1 | 2 | 2 | 5 |
| total | — | 18 | 8 | 6 | 4 | 12 |

Per-rule baseline (status / findings):

```
R1.1 FAIL 1   R1.2 PASS 0   R1.3 PASS 0   R1.4 PASS 0
R1.5 PASS 0   R1.6 PASS 0   R1.7 WARN 1   R1.8 FAIL 1
R2.0 PASS 0   R2.1 WARN 2   R2.2 WARN 1   R2.3 PASS 0   R2.4 WARN 1
R3.1 FAIL 1   R3.2 PASS 0   R3.3 FAIL 1   R3.4 WARN 2   R3.5 WARN 1
```

R1.4 now PASSes on the authoritative vendor truth: verify script result PASS in
~4 s, 10/10 recorded files byte-identical, canonical patch 66,112 B sha256
`95ae6a367aed65a89e7628f288f3d1a8805789550eba197555525c97e98a6ceb`, dirty set ==
recorded set.  The 12 superseded per-feature patches are exempt (they stay until
the Lead approves deletion; `regenerated/` is outside the gate).

Top findings at baseline (from `build/tmp/audit_check.json`):

1. **R1.1 (error)** conan packages pinned to gcc 13, compiling with g++ 16.1.0 —
   `build/conan_home/profiles/default` says `compiler.version=13
   libcxx=libstdc++11 cppstd=gnu17`; actual `g++.exe (Rev5…) 16.1.0`.
   (An isolated `build/conan_home_16` gcc16 home is being staged; R1.1 flips to
   PASS only when the Lead swaps the production home — see
   `doc/v10.1/conan_abi_gcc16.md`.)
2. **R1.8 (error)** 7 persisted state switches mirror config switches:
   `agiEnabled→agi.enabled`, `missionEnabled→mission.enabled`,
   `selfMind.enabled→selfMind.enabled`, `studyExpert.enabled→study.expert.enabled`,
   `studyPopulation.enabled→study.population.enabled`,
   `studyUnitLearn.enabled→study.unitLearn.enabled`,
   `subconsciousEnabled→subconscious.enabled`.
3. **R3.1 (error)** 36 mutable static containers, e.g.
   `external_mixed_modal_io.cpp:107 static std::unordered_map<int,
   std::unique_ptr<VideoModel>> cache;`, `frontend_server.cpp:204
   gLocalAuthSessions`, `memetic_existence.cpp:1019 unitsCache`,
   `main_hub_parts/060_section_before_lemmaworkerpool.inc:7 gLemmaCache`.
4. **R3.3 (error)** 1 enabled-like switch restored in a real `fromJson` body:
   `mcp_client.cpp:47 c.enabled = j.value("enabled", true);`.
5. **R1.7 (warn)** `version.hpp` says 10.0.0 but newest doc shelf is `v10.1`.

Other baseline facts: R1.3 PASS (nvcc 13.5 == driver UMD 13.5, driver 621.24,
GTX 1660 sm_75 ok), R1.5 PASS (Python headers 3.14.3 == embedded python.exe
3.14.3, major.minor == DLL), R1.6 PASS (onnxruntime present; Python bridge),
R1.2 PASS (all 9 requirements match the conan cache), R3.2 PASS
(mcpActionMap_ 4 guarded/0 unguarded, cache_ 36/0, sessions_ 38/0 —
caller-locked helpers exempted), R3.3 has 1 real hit
(`mcp_client.cpp:47 c.enabled = j.value("enabled", true);`), R3.5 has 1 real hit
(`frontend_server.cpp:4316` lock over `model->generate(...)`).

## How to add a rule

1. Pick the next free id in a group; write the rationale as a sentence naming the
   real incident (this is mandatory — a rule without an incident is just an
   opinion).
2. Add a decorated function near the group's other rules:

```python
@rule("R1.9", "R1", "warn",
      "Short title",
      "Which past incident this encodes and why it matters.")
def r19_something(ctx: Ctx):
    findings, stats = [], {}
    for p in product_source_files(ctx):
        for i, line in enumerate(read_text(p).splitlines(), 1):
            if "bad pattern" in line:
                findings.append(dict(
                    severity="warn",                       # error -> rule FAIL
                    message="what is wrong and how many",
                    evidence=[evidence(ctx, p, i, line)]))
    stats["hits"] = len(findings)
    return findings, stats
```

3. Conventions: return `(findings, stats)`; each finding is
   `{severity, message, evidence[]}`; `stats` is any JSON-able dict (use
   `{"skipped": "..."}` to make the rule SKIP cleanly).  Rule status is derived:
   no findings → PASS; findings → WARN; any error finding (or an `error`-severity
   rule with findings) → FAIL.  Dedicated subprocess probes must go through
   `run(ctx, [...], timeout=...)` so `--no-subprocess` keeps working.
4. Do not write outside `build/tmp/audit_check.{json,md}`; the checker is
   read-only by contract.  Never put auto-deletion in a rule.
5. Run it, then update the baseline table in this document (counts drift is
   expected; a *new* FAIL is the signal).

## R2 × deadcode-audit contract

`deadcode-audit` owns asset classification and deletion planning.  The
integration point is `build/tmp/dead_assets.json`, consumed by R2.0 if present.
Minimal shape (extra fields ignored):

```json
{
  "generated_at": "2026-11-17T23:00:00",
  "assets": [
    {"path": "runtime_store/...", "kind": "model|dataset|binary|dir|source",
     "size_bytes": 123, "verdict": "keep|delete|review|unknown", "reason": "..."}
  ],
  "summary": {"delete": 0, "keep": 0}
}
```

R2.0 reports verdict counts, flags asset paths that no longer exist (stale
audit), and never executes a deletion.  If the file is not published yet R2.0
PASSes with `state: not published yet` and R2's own rules stay advisory.

## Limitations / false-positive policy

* R2.1/R2.2/R2.3/R3.2/R3.5 are heuristics.  They are deliberately worded with
  "heuristic" and carry evidence so a human can overrule quickly.  Never delete
  code/data just because a rule fired — confirm reachability/runtime use first.
* R1.4a shells out to `tools/vendor_patch_verify.ps1` (creates a temporary pristine
  worktree, ~4 s; use `pwsh`).  Without subprocess it falls back to the cached
  `build/tmp/vendor_patch_verify/last_result.json` and says so.  R1.4b only
  judges patches NOT listed in `SUPERSEDED.md`; the superseded old chain is
  intentionally exempt until the Lead approves deletion.  R1.4c compares the
  live dirty set to the tracked 10-file set — an intentional vendor change must
  be re-captured into the canonical patch + hash truth, not hand-waved.
* R1.2/R1.6/R1.3/R1.5 depend on local tools (conan/nvcc/nvidia-smi/embedded
  Python).  `--no-subprocess` turns those probes off and reports SKIP/PASS with
  missing stats instead of lying.
* R3.2 treats "all callers hold the lock" as guarded; that is the documented
  pattern in this codebase (`pruneCache` ← `remember` under `cacheMu_`,
  `evictIfNeeded` ← `put` under `mutex_`).  A future refactor that breaks the
  caller contract must also update the comment/call graph, and the rule will
  then still reason from the code as it is.

## Files

| path | role |
|---|---|
| `tools/audit_check.py` | the checker (this catalogue's implementation) |
| `build/tmp/audit_check.json` | baseline machine output |
| `build/tmp/audit_check.md` | baseline human report |
| `build/tmp/dead_assets.json` | produced by `deadcode-audit`, consumed by R2.0 |
| `doc/v10.1/debug_protocol.md` | incident/debug side of the same lessons (R3.1/R3.3/R3.5 map to it) |
