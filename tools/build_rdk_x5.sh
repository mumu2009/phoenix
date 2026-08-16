#!/usr/bin/env bash
set -euo pipefail

# Load local pkg-config overrides for Drogon/Trantor
# shellcheck source=build_rdk_x5_env.sh
source "$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/build_rdk_x5_env.sh"

root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${HB_DNN_ROOT:=/usr}"
: "${CXX:=g++}"

for required in "$HB_DNN_ROOT/include/dnn/hb_dnn.h" "$HB_DNN_ROOT/lib/libdnn.so"; do
    if [[ ! -e "$required" ]]; then
        printf 'missing RDK X5 hbDNN dependency: %s\n' "$required" >&2
        exit 1
    fi
done

mapfile -t sources < <(python3 - "$root" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
names = '''transformer_main_x5_stub.cpp addon.cpp addons/builtin_registry.cpp addons/math_addon.cpp addons/search_addon.cpp addons/computer_shell_addon.cpp loggerCXX.cpp DATABASE_079.cpp frontend_server.cpp speak_io.cpp model_lifecycle.cpp autonomy_stack.cpp v51_runtime.cpp external_runtime.cpp edge_platform.cpp gguf_tensor_parser.cpp physics_world_runtime.cpp emotion_system.cpp llamacpp_emotion_adjuster.cpp plugin_system.cpp modern_context_system.cpp semantic_unit.cpp primal_sensation.cpp instinct.cpp prompt_split.cpp external_mixed_modal_io.cpp video_model.cpp audio_model.cpp model_deployment.cpp rdk_x5_bpu.cpp graph_diffusion_summarizer.cpp hierarchical_memory.cpp active_inference.cpp subconscious_profile.cpp sparse_block_matmul.cpp agi_action_registry.cpp mission_lifecycle.cpp'''.split()
for name in names:
    print(root / name)
for source in sorted((root / 'module_overrides').glob('*.cpp')):
    print(source)
PY
)

pkg_config_packages=(drogon opencv4 sqlite3 lmdb hiredis redis++ jsoncpp)
"$CXX" -std=c++20 -O3 -DNDEBUG -pthread \
    -I"$HB_DNN_ROOT/include" \
    -I/usr/include/eigen3 \
    "${sources[@]}" "$root/main.cpp" \
    $(pkg-config --cflags --libs "${pkg_config_packages[@]}") \
    -L"$HB_DNN_ROOT/lib" -ldnn -lhbrt_bayes_aarch64 -o "$root/phoenix_main"
