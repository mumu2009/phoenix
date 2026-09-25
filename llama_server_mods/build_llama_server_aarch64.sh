#!/usr/bin/env bash
# Cross-compile patched (split) llama-server for RDK X5 aarch64 on the host.
# Output: build/rdk-aarch64/llama-server
set -euo pipefail

root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODS="${root}/llama_server_mods"
OUT_DIR="${root}/build/rdk-aarch64"
SYSROOT="${RDK_SYSROOT:-${root}/build/rdk-sysroot}"
LLAMACPP_ROOT="${root}/outsides/llamacpp"
BUILD_DIR="${LLAMACPP_ROOT}/build-aarch64-cross"
TOOLCHAIN="${root}/tools/cmake/toolchain-aarch64-linux-gnu.cmake"

mkdir -p "$OUT_DIR"

: "${CROSS_PREFIX:=}"
if [[ -z "$CROSS_PREFIX" ]]; then
  if command -v aarch64-linux-gnu-gcc >/dev/null 2>&1; then
    CROSS_PREFIX=aarch64-linux-gnu
  elif command -v aarch64-none-linux-gnu-gcc >/dev/null 2>&1; then
    CROSS_PREFIX=aarch64-none-linux-gnu
  else
    CROSS_PREFIX=aarch64-linux-gnu
  fi
fi
export CROSS_PREFIX

if ! command -v "${CROSS_PREFIX}-gcc" >/dev/null 2>&1; then
  echo "[llama-cross] FATAL: ${CROSS_PREFIX}-gcc not on PATH"
  exit 1
fi
if [[ ! -d "$SYSROOT/usr" ]]; then
  echo "[llama-cross] FATAL: RDK_SYSROOT missing; run tools/rdk_sysroot_fetch.sh"
  exit 1
fi

export RDK_SYSROOT="$SYSROOT"
bash "$MODS/apply_patches.sh" || { echo "patch apply failed"; exit 1; }

if ! command -v cmake >/dev/null 2>&1 || ! command -v ninja >/dev/null 2>&1; then
  echo "[llama-cross] FATAL: cmake and ninja required"
  exit 1
fi

link_flags="-B${SYSROOT}/usr/lib/aarch64-linux-gnu -Wl,-rpath-link,${SYSROOT}/usr/lib/aarch64-linux-gnu -Wl,-rpath-link,${SYSROOT}/usr/local/lib"
compat_obj="${MODS}/glibc235_compat.o"
"${CROSS_PREFIX}-gcc" --sysroot="$SYSROOT" -isystem "${SYSROOT}/usr/include/aarch64-linux-gnu" \
  -c "${MODS}/glibc235_compat.c" -o "$compat_obj"
# cygpath -m on /d/... can emit a broken "D;D:\<toolchain>\...\d\<repo>\..."
# path. Prefer an explicit drive-letter form the GNU linker accepts.
compat_obj_link="$compat_obj"
if [[ "$compat_obj_link" =~ ^/([a-zA-Z])/(.*)$ ]]; then
  compat_obj_link="${BASH_REMATCH[1]^}:/${BASH_REMATCH[2]}"
fi
link_flags="$link_flags $compat_obj_link -static-libstdc++ -static-libgcc -lgomp -lpthread -ldl -lm -lc"

cmake -S "$LLAMACPP_ROOT" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DGGML_NATIVE=OFF \
  -DCMAKE_EXE_LINKER_FLAGS="$link_flags" \
  -DCMAKE_SHARED_LINKER_FLAGS="$link_flags"

ninja -C "$BUILD_DIR" llama-server
cp -f "$BUILD_DIR/bin/llama-server" "$OUT_DIR/llama-server"
chmod +x "$OUT_DIR/llama-server"
echo "[llama-cross] OK: $OUT_DIR/llama-server"
