#!/bin/bash
# apply_patches.sh - apply Phoenix llama-server patches under outsides/llamacpp.
# Linux/RDK counterpart of apply_patches.bat.  Do NOT edit outsides/llamacpp
# by hand; put changes in llama_server_mods/*.patch and re-run this script.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_ROOT="${LLAMA_ROOT:-$ROOT/outsides/llamacpp}"
MODS="$ROOT/llama_server_mods"

if [ ! -d "$LLAMA_ROOT" ]; then
  echo "[ERROR] llama.cpp checkout missing: $LLAMA_ROOT"
  exit 1
fi

already_applied() {
  # If reverse-check succeeds, the patch is already present.
  git -C "$LLAMA_ROOT" apply --reverse --check "$1" >/dev/null 2>&1
}

apply_one() {
  local patch="$1"
  if [ ! -f "$patch" ]; then
    echo "[WARN] missing patch: $patch"
    return 0
  fi
  if already_applied "$patch"; then
    echo "[OK] already applied: $(basename "$patch")"
    return 0
  fi
  echo "[APPLY] $(basename "$patch")"
  if ! git -C "$LLAMA_ROOT" apply "$patch"; then
    echo "[ERROR] failed to apply $patch"
    return 1
  fi
}

# Never reset a tree that already has the unit-loop handler (hand-edited
# or previously patched) — reset would wipe those changes.
if grep -q 'feedback_mode' "$LLAMA_ROOT/examples/server/server.cpp" 2>/dev/null; then
  echo "[OK] unit-loop already present; skip base-stack reset"
else
# Match apply_patches.bat: enc_dec_separation is generated on top of
# existing_mods, so if the last patch reverse-applies cleanly the full
# stack is already in place.
LAST_PATCH="$MODS/existing_mods.patch"
if [ -f "$MODS/enc_dec_separation.patch" ]; then
  LAST_PATCH="$MODS/enc_dec_separation.patch"
fi
if already_applied "$LAST_PATCH"; then
  echo "[OK] base patch stack already applied (checked against $(basename "$LAST_PATCH"))"
else
  # Otherwise reset to a clean checkout and re-apply both patches.
  if git -C "$LLAMA_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "[RESET] outsides/llamacpp -> clean checkout before patch apply"
    git -C "$LLAMA_ROOT" reset --hard HEAD >/dev/null 2>&1 || true
    git -C "$LLAMA_ROOT" clean -fd -- include src examples/server >/dev/null 2>&1 || true
  fi
  apply_one "$MODS/existing_mods.patch" || exit 1
  apply_one "$MODS/enc_dec_separation.patch" || exit 1
fi
fi

# Additive fix on top of enc/dec stack (safe to re-run).
if grep -q 'llama_kv_cache_clear' "$LLAMA_ROOT/examples/server/server.cpp" 2>/dev/null; then
  echo "[OK] kv-clear already present in server.cpp"
else
  apply_one "$MODS/phx_generate_kv_clear.patch" || exit 1
fi
# Unit-loop / hidden-vs-dec_enc feedback.  If the tree already has the
# handler (hand-edited or previously patched), skip reset-destroying it.
if grep -q 'feedback_mode' "$LLAMA_ROOT/examples/server/server.cpp" 2>/dev/null; then
  echo "[OK] unit-loop already present in server.cpp"
else
  apply_one "$MODS/phx_unit_loop.patch" || exit 1
fi
# Encode context/gnn as unit-query packets (E space, prepended).
if grep -q 'ingest_unit_packets' "$LLAMA_ROOT/examples/server/server.cpp" 2>/dev/null; then
  echo "[OK] unit-query packets already present in server.cpp"
else
  apply_one "$MODS/phx_generate_unit_query_sides.patch" || exit 1
  apply_one "$MODS/phx_generate_unit_query_packets.patch" || exit 1
fi
# Prompt-first unit fuse + sampler penalties (in-tree marker).
if grep -q 'Prompt (chat template) stays at position 0' "$LLAMA_ROOT/examples/server/server.cpp" 2>/dev/null; then
  echo "[OK] prompt-first unit fuse already present in server.cpp"
elif [ -f "$MODS/phx_generate_prompt_first_penalties.patch" ]; then
  apply_one "$MODS/phx_generate_prompt_first_penalties.patch" || exit 1
fi
# File tail after units + prompt-window repeat penalty (in-tree marker).
if grep -q 'resume_suffix' "$LLAMA_ROOT/examples/server/server.cpp" 2>/dev/null; then
  echo "[OK] resume_suffix + full-window penalty already present in server.cpp"
fi
if grep -q 'content-path unit embeddings' "$LLAMA_ROOT/examples/server/server.cpp" 2>/dev/null; then
  echo "[OK] content-path unit embeddings already present in server.cpp"
fi
if grep -q 'paragraph I/O enc' "$LLAMA_ROOT/examples/server/server.cpp" 2>/dev/null ||
   grep -q 'I/O enc/dec (not the internal dec-enc pair)' "$LLAMA_ROOT/examples/server/server.cpp" 2>/dev/null; then
  echo "[OK] paragraph I/O enc already present in server.cpp"
fi
if grep -q 'phx_merge_ngram_rows' "$LLAMA_ROOT/examples/server/server.cpp" 2>/dev/null; then
  echo "[OK] n-gram unit merge already present in server.cpp"
fi
if grep -q 'leftover_mix' "$LLAMA_ROOT/examples/server/server.cpp" 2>/dev/null; then
  echo "[OK] leftover+RAG hidden-path mix already present in server.cpp"
fi
# Forced single MoE expert (LLAMA_FORCE_MOE_EXPERT) in the ggml argsort kernel.
if grep -q 'LLAMA_FORCE_MOE_EXPERT' "$LLAMA_ROOT/ggml/src/ggml-cpu/ggml-cpu.c" 2>/dev/null; then
  echo "[OK] force-moe-expert already present in ggml-cpu.c"
else
  apply_one "$MODS/force_moe_expert.patch" || exit 1
fi
if grep -q 'PHOENIX_PREFETCH_VIRTUAL_MEMORY' "$LLAMA_ROOT/src/llama-mmap.cpp" 2>/dev/null; then
  echo "[OK] prefetch virtual memory already present in llama-mmap.cpp"
else
  apply_one "$MODS/prefetch_virtual_memory.patch" || exit 1
fi
if grep -q 'PHOENIX_FORCE_MOE_N_EXPERT_USED' "$LLAMA_ROOT/src/llama-model.cpp" 2>/dev/null; then
  echo "[OK] force-moe-expert count already present in llama-model.cpp"
else
  apply_one "$MODS/force_moe_expert_count.patch" || exit 1
fi
# MinGW UTF-8 -> UTF-16 fix for unicode_wstring_from_utf8 (deepseek2 vocab load).
if grep -q 'PHOENIX_UNICODE_WSTRING_FROM_UTF8' "$LLAMA_ROOT/src/unicode.cpp" 2>/dev/null; then
  echo "[OK] unicode_wstring_from_utf8 fix already present in unicode.cpp"
else
  apply_one "$MODS/unicode_wstring_from_utf8.patch" || exit 1
fi
if grep -q 'PHOENIX_PHX_PACK_JACOBI' "$LLAMA_ROOT/examples/server/server.cpp" 2>/dev/null; then
  echo "[OK] phx pack/jacobi already present in server.cpp"
else
  apply_one "$MODS/phx_pack_jacobi.patch" || exit 1
fi
# deepseek-llm pre-tokenizer regex: drop >BMP ranges (16-bit wchar_t wregex).
if grep -q 'PHOENIX_DEEPSEEK_LLM_REGEX_BMP' "$LLAMA_ROOT/src/llama-vocab.cpp" 2>/dev/null; then
  echo "[OK] deepseek-llm regex BMP fix already present in llama-vocab.cpp"
else
  apply_one "$MODS/deepseek_llm_regex_bmp.patch" || exit 1
fi
echo "[DONE] llama_server_mods patches applied (chunked encode maintained in-tree)"
