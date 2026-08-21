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

# Prefer a clean tree before first apply if patches are not yet present.
if ! already_applied "$MODS/existing_mods.patch" && \
   ! already_applied "$MODS/enc_dec_separation.patch"; then
  if git -C "$LLAMA_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "[RESET] outsides/llamacpp -> clean checkout before patch apply"
    git -C "$LLAMA_ROOT" reset --hard HEAD >/dev/null 2>&1 || true
    git -C "$LLAMA_ROOT" clean -fd >/dev/null 2>&1 || true
  fi
fi

apply_one "$MODS/existing_mods.patch" || exit 1
apply_one "$MODS/enc_dec_separation.patch" || exit 1
echo "[DONE] llama_server_mods patches applied"
