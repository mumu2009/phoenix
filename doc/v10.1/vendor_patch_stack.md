# llama.cpp Vendor Patch Stack — Reproducibility Repair (v10.3)

Status: repaired and verified.  No runtime rebuild was performed; the live gateway
(pid 23920, port 5097) and all llama-servers were left running untouched.

Related: `llama_server_mods/SUPERSEDED.md` (old patch ledger), `llama_server_mods/vendor_733dd8df.sha256.json`
(machine-readable truth), `tools/vendor_patch_verify.ps1` (acceptance gate),
`tools/audit_check.py` rule R1.4, `doc/v10.1/audit_checker.md`.

---

## 1. Problem (R1.4 baseline)

The vendored llama.cpp checkout is `outsides/llamacpp` at
`733dd8dfc0b4446d653977723c26d2c895b4687d` with **10 modified files**.  A clean
checkout plus the `llama_server_mods/*.patch` stack did **not** reproduce it:

| R1.4 finding | detail |
|---|---|
| 2 patches unreadable | `phx_chunked_encode.patch: corrupt patch at line 87`; `phx_generate_cancel.patch: corrupt patch at line 58` (git apply exit 128 both directions) |
| 5 patches no longer apply | `enc_dec_separation`, `existing_mods`, `phoenix_combined`, `phx_generate_unit_query_sides`, `phx_unit_loop` |
| 1 applies but is not applied | `phx_generate_kv_clear` |
| dirty vendor tree | 10 files, so "clean vendor + apply-all" ≠ current binary |

Two independent probes confirmed the mismatch:

| probe from pristine 733dd8df | files matching the live tree |
|---|---|
| `phoenix_combined.patch` alone | **5 / 10** |
| `existing_mods.patch` + `enc_dec_separation.patch` (old base chain) | **3 / 10** |

Root cause: the vendor tree was hand-evolved past every patch (the patches are
older snapshots), features were merged/refactored (the old patch base had a
~6000-line `server.cpp`; the current one is ~5200 lines), and two patch files are
structurally corrupt.

---

## 2. New canonical patch set

| item | value |
|---|---|
| canonical patch | `llama_server_mods/phoenix_vendor_733dd8df.patch` |
| size / sha256 | **66,112 bytes** / `95ae6a367aed65a89e7628f288f3d1a8805789550eba197555525c97e98a6ceb` |
| base | `733dd8dfc0b4446d653977723c26d2c895b4687d` (llama.cpp merge commit) |
| diff | **+1200 / −11** across **10 files** |
| definition | exact `git diff` of the 10 live modified files against 733dd8df (generated, not hand-written) |
| truth file | `llama_server_mods/vendor_733dd8df.sha256.json` (per-file sha256 + bytes + base commit) |
| acceptance | pristine 733dd8df + canonical patch → **byte-identical** to the live tree (see §3/§6) |

Optional feature-forward patches regenerated from the two corrupt files live in
`llama_server_mods/regenerated/` and are **not** part of the truth set:

| file | size | base it applies to |
|---|---:|---|
| `regenerated/phx_chunked_encode.patch` | 3,620 B | canonical-patched tree |
| `regenerated/phx_generate_cancel.patch` | 3,232 B | canonical-patched tree |

`regenerated/repair_corrupt_patches.py` recomputes `@@` line counts from the diff
body (chunked); `regenerated/rebuild_cancel_feature.py` re-anchors the cancel hooks
onto the current generate loops (the old `for (step ...; ++step)` loops were
refactored into the jacobi/speculative loops).  Neither adds the feature to the
current-binary truth.

---

## 3. Truth hash table (acceptance target)

Raw worktree bytes; the vendor checkout uses `core.autocrlf=true` (CRLF), and the
patch mechanism preserves that exactly.

| file | bytes | sha256 |
|---|---:|---|
| `examples/server/server.cpp` | 216,874 | `dd16e6ff792868847f1512bc4278dd70fe94a0ca6ee04a29268e797114257361` |
| `ggml/src/ggml-cpu/ggml-cpu.c` | 509,945 | `c8f94210c8d8d8ee5954af9997243ecf9966f3e355d340bf048082ccc501603d` |
| `include/llama.h` | 73,785 | `ae28e7a3f8e347776f8e447911ba911133dfa52a03b82ce74228576937ed76ba` |
| `src/llama-context.cpp` | 77,310 | `a7181c439fdcb30bfcd0ddfa5c453e519739eb2a8dcfea7a4886a027d5785476` |
| `src/llama-context.h` | 6,145 | `c338d4969efde9999a309f150ee079fca274ec46870c8687219f4e4eb4ccfcc7` |
| `src/llama-mmap.cpp` | 19,571 | `91ba78fd353dcfee58a734447c048f889075e068357da2acc766404fdf20410e` |
| `src/llama-model.cpp` | 225,953 | `c33b73d99d9c31b4b9f5b4348e26753b42cdbc4225b5c722c8b391877c0f78c3` |
| `src/llama-vocab.cpp` | 130,376 | `55eb6b51e22b15ff562379f482b4caac870a28030ee8527e154bcbace2c247d7` |
| `src/llama.cpp` | 409,879 | `c4f950c45dade625fdb30a70308079459b5094e22eedaec32bc9f45f59d866ab` |
| `src/unicode.cpp` | 33,960 | `6a57cb51a7dbae1da64fdbfb1d2442cf51b773a0bd47c153273359a2643e2637` |

Bug contexts worth remembering: `src/llama-context.cpp` is the `phx_run_stage`
enc/infer/dec split; `examples/server/server.cpp` carries `/phx/*` + pack/jacobi;
`ggml-cpu.c` / `llama-model.cpp` carry the forced-MoE-expert knob; `llama-vocab.cpp`
the deepseek-llm BMP regex fix; `unicode.cpp` the MinGW `wstring_from_utf8` fix.

---

## 4. How the canonical patch was regenerated

All work happened in scratch worktrees under `build/tmp/llama_patch_repair/`; the
live vendor tree was never edited.

1. **Capture the truth** — `git -C outsides/llamacpp rev-parse HEAD`,
   `git status --porcelain` (10 files), and sha256 per modified file, saved as
   `build/tmp/llama_patch_repair/current_state.json` (and the tracked copy
   `llama_server_mods/vendor_733dd8df.sha256.json`).
2. **Pristine reference** — `git -C outsides/llamacpp worktree add --detach
   build/tmp/llama_patch_repair/vendor_base 733dd8df` (detached scratch worktree;
   no branch in the vendor repo).
3. **Overlay + diff** — copy the 10 live files over the pristine worktree and run
   `git diff`; the result is the canonical patch.  Nothing is hand-written.
4. **Corrupt-file regeneration** — the two corrupt patches were repaired
   programmatically from their own diff bodies:
   - `phx_chunked_encode` had one complete hunk with wrong `@@` counts → counts
     recomputed;
   - `phx_generate_cancel` was truncated and written against the old (pre-refactor)
     `server.cpp` → rebased onto the canonical base with the same intended
     additions (epoch member, max_tokens floor 16, abort hooks, `/phx/cancel`).
5. **Acceptance** — a second pristine worktree + `git apply` canonical patch →
   sha256 comparison → **PASS** (10/10 files).

---

## 5. Rebuilding the vendor tree from a clean checkout

Use the non-destructive verifier first (it never touches the live vendor tree):

```powershell
cd E:\079\v6.0Alixander\phoenix
pwsh -NoProfile -File tools/vendor_patch_verify.ps1     # PASS/FAIL + hash table
```

To actually reset and repatch the real checkout (only after human approval — the
vendor tree currently backs the live `/phx` server):

```powershell
cd E:\079\v6.0Alixander\phoenix
git -C outsides/llamacpp checkout -- .          # discard the 10 live edits
git -C outsides/llamacpp clean -fd -- include src examples/server ggml/src
git -C outsides/llamacpp apply --whitespace=nowarn llama_server_mods/phoenix_vendor_733dd8df.patch
pwsh -NoProfile -File tools/vendor_patch_verify.ps1     # must print PASS
```

Important current-state caveat: `llama_server_mods/apply_patches.bat` /
`apply_patches.sh` still apply the **old** patch chain (they were not touched in
this repair).  Until they are repointed at the canonical patch (follow-up in
`SUPERSEDED.md`), a clean-vendor rebuild must use the `git apply` line above; the
`PHOENIX_LLAMA_SPLIT=1` compile path would otherwise re-apply stale patches and
produce a different tree.  No `compile.bat` was run in this repair, and the live
`llama-server.exe` binaries were not rebuilt or restarted.

---

## 6. Verification tool (`tools/vendor_patch_verify.ps1`)

The script (a) creates a temporary pristine worktree of the vendor checkout at the
commit recorded in the truth file, (b) applies the canonical patch, (c) compares
sha256 per file and checks that exactly the recorded file set changed, prints a
PASS/FAIL table and writes `build/tmp/vendor_patch_verify/last_result.json`, then
removes the temporary worktree.

```
RESULT: PASS - pristine 733dd8dfc0 + phoenix_vendor_733dd8df.patch reproduces the
recorded vendor tree byte-for-byte (10 files, 2.0s)
```

Arguments: `-VendorRoot`, `-Patch`, `-CurrentState`, `-KeepTemp`.  Exit codes:
`0` PASS, `1` FAIL, `2` setup error.  This is the reproducibility gate for R1.4 —
the checker owner can call it and/or fold its JSON into `audit_check.py`; R1.4's
current "every `*.patch` must apply" logic should be updated to treat
`phoenix_vendor_733dd8df.patch` + `vendor_733dd8df.sha256.json` as authoritative
and to ignore the superseded ledger until those files are deleted.

---

## 7. Rollback

- Nothing runtime-facing was changed: no `compile.bat`, no `phoenix_main.exe`
  rebuild, live gateway (pid 23920 / port 5097) and llama-servers were not touched.
- To undo a canonical-patch application on the vendor source:
  `git -C outsides/llamacpp apply -R llama_server_mods/phoenix_vendor_733dd8df.patch`
  (returns the vendor source to pristine `733dd8df`; no push/reset needed) — or
  `git -C outsides/llamacpp checkout -- <10 files>` if only the working tree drifted.
- The old patch files (including the two corrupt ones) are preserved byte-for-byte
  until the Lead approves deletion; `SUPERSEDED.md` lists what replaces them.
- Reproducibility of the rollback is checkable with
  `git -C outsides/llamacpp status --porcelain` (empty) and the sha256 truth file.

---

## 8. 中文摘要

**问题**：R1.4 发现 `outsides/llamacpp` 无法从干净上游 + 补丁栈重建：2 个补丁文件结构损坏，
5 个已无法 apply，1 个可 apply 但未应用，vendor 树有 10 个改动文件；旧补丁链最多只能复现
3–5 / 10 个文件。

**修复**：在 `build/tmp/llama_patch_repair/` 的临时 worktree 里，把 live 树的 10 个文件覆盖到
干净的 `733dd8df` 上，用 `git diff` 生成**唯一权威补丁**
`llama_server_mods/phoenix_vendor_733dd8df.patch`（66,112 字节，+1200/−11），并记录逐文件
sha256 真值（`vendor_733dd8df.sha256.json`）。验收：干净树 + 该补丁 == live 树，10/10 文件
字节级一致，PASS。

两个损坏补丁用脚本从其自身 diff 内容重建（不手工编辑），放在 `regenerated/`，它们属于
未来的可选特性（chunked encode、`/phx/cancel`），**不属于**当前二进制真值；旧的 14 个补丁
全部登记为 superseded，等 Lead 批准后再删除，`apply_patches.*` 也要在批准后改为应用新补丁。

**未改动运行时**：没有跑 `compile.bat`、没有重建/重启 `phoenix_main.exe` 或 llama-server；
回滚可用 `git apply -R` 或 `git checkout --`，无需 reset/push。
