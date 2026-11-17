# llama_server_mods — SUPERSEDED patch ledger (v10.3, 2026-11-17)

The Phoenix llama.cpp vendor tree (`outsides/llamacpp`) is now reproducible from a
clean upstream checkout with **one** canonical patch:

    llama_server_mods/phoenix_vendor_733dd8df.patch

That patch is the exact `git diff` of the 10 live modified files against
`733dd8dfc0b4446d653977723c26d2c895b4687d` (66,112 bytes, +1200/−11) and is
verified byte-for-byte by `tools/vendor_patch_verify.ps1` against
`llama_server_mods/vendor_733dd8df.sha256.json`.

Every per-feature `*.patch` in this directory is therefore **superseded** by the
canonical patch.  They are kept only until the Lead approves deletion; do not add
them back into any clean-vendor workflow.

## Why the old stack cannot be used

`tools/audit_check.py` rule R1.4 (as seen on the live dirty vendor tree) classified
the stack as: 6 applied, 5 conflicting, 2 unreadable (corrupt), 1 applicable but
not applied.  Two independent probes show it cannot reproduce the live tree:

| probe (from pristine 733dd8df) | files matching live truth |
|---|---|
| `phoenix_combined.patch` alone | **5 / 10** |
| `existing_mods.patch` + `enc_dec_separation.patch` (old base chain) | **3 / 10** |

Root cause: the vendor tree was hand-evolved past the patches (10 dirty files,
newer than every patch file), split features were merged/refactored, and two patch
files were structurally corrupt.

## Per-file ledger

Status legend: `applied` = reverse-applies on the live tree (content is in the
live tree); `conflicting` = applies neither way; `corrupt` = git cannot parse it;
`not-applied` = applies forward but is absent from the live tree.

| file | bytes | R1.4 status | in live tree? | replacement |
|---|---:|---|---|---|
| `deepseek_llm_regex_bmp.patch` | 1,889 | applied | yes (`llama-vocab.cpp`) | canonical patch |
| `enc_dec_separation.patch` | 37,087 | conflicting | yes (newer, refactored) | canonical patch |
| `existing_mods.patch` | 9,103 | conflicting | yes (newer, refactored) | canonical patch |
| `force_moe_expert.patch` | 2,167 | applied | yes (`ggml-cpu.c`) | canonical patch |
| `force_moe_expert_count.patch` | 2,571 | applied | yes (`llama-model.cpp`) | canonical patch |
| `phoenix_combined.patch` | 50,813 | conflicting | partial (5/10 files) — stale snapshot | canonical patch |
| `phx_chunked_encode.patch` | 3,500 | **corrupt** (bad `@@` counts) | **no** (chunked encode never shipped) | `regenerated/phx_chunked_encode.patch` (valid, optional feature-forward) |
| `phx_generate_cancel.patch` | 3,144 | **corrupt** (truncated hunk + stale anchors) | **no** (`/phx/cancel` never shipped) | `regenerated/phx_generate_cancel.patch` (rebased, optional feature-forward) |
| `phx_generate_kv_clear.patch` | 1,154 | not-applied | **no** (fresh-KV guard never shipped) | none — needs rewrite against the canonical base if it is wanted |
| `phx_generate_unit_query_sides.patch` | 5,015 | conflicting | **no** | none — feature was refactored away; rewrite if wanted |
| `phx_pack_jacobi.patch` | 15,797 | applied | yes (`server.cpp` jacobi path) | canonical patch |
| `phx_unit_loop.patch` | 8,081 | conflicting | **no** (`feedback_mode` absent) | none — feature was refactored away; rewrite if wanted |
| `prefetch_virtual_memory.patch` | 1,740 | applied | yes (`llama-mmap.cpp`) | canonical patch |
| `unicode_wstring_from_utf8.patch` | 1,629 | applied | yes (`unicode.cpp`) | canonical patch |

Also note: `apply_patches.sh` references two patches that do not exist on disk
(`phx_generate_unit_query_packets.patch`, `phx_generate_prompt_first_penalties.patch`);
it skips them only because their in-tree markers are absent.  This is another
symptom of the drifted stack — the canonical patch makes those references moot.

## Regenerated corrupt patches

The two corrupt files were repaired **programmatically from their own diff bodies**
(no hand-editing), and rebased onto the canonical base where their old anchors no
longer exist.  They live in `regenerated/` and are **not part of the reproducibility
truth** — they add features that the current binary does not have:

    llama_server_mods/regenerated/phx_chunked_encode.patch     (3,620 B)  applies on canonical base
    llama_server_mods/regenerated/phx_generate_cancel.patch    (3,232 B)  applies on canonical base
    llama_server_mods/regenerated/repair_corrupt_patches.py    (recomputes @@ counts / strips prose)
    llama_server_mods/regenerated/rebuild_cancel_feature.py    (re-anchors cancel hooks on current loops)

Applying them changes source behaviour (chunked encode + `/phx/cancel` endpoint), so
they must not be used in the "reproduce current binary" workflow.

## Follow-up after Lead approval

1. Delete the 12 superseded root-level `*.patch` files listed above (keep the
   canonical patch and `regenerated/`).
2. Repoint `llama_server_mods/apply_patches.bat` / `apply_patches.sh` at the
   canonical patch (currently they still apply the old chain and will fail or
   produce a different tree).  Until then, clean-vendor rebuilds must use
   `git apply llama_server_mods/phoenix_vendor_733dd8df.patch`.
3. Re-run `tools/vendor_patch_verify.ps1` and (checker owner) update R1.4 to treat
   the canonical patch + `vendor_733dd8df.sha256.json` as the authoritative
   reproducibility gate instead of "every patch must apply".
