import re
from pathlib import Path

root = Path(r"d:\_phoenix\_079\v6.0Alixander\phoenix")
files = [
    "model_lifecycle.cpp",
    "autonomy_stack.cpp",
    "physics_world_runtime.cpp",
    "frontend_server.cpp",
    r"addons\computer_shell_addon.cpp",
    r"addons\math_addon.cpp",
    r"addons\search_addon.cpp",
]

# Patterns: (json variable, .value( or .contains()
# We target variables commonly used to hold JSON objects in these files.
variables = ["payload", "scene", "manifest", "fileJson", "bodySummary", "physicsScene", "request", "configJson"]

for rel in files:
    path = root / rel
    if not path.exists():
        print(f"skip {rel}")
        continue
    text = path.read_text(encoding="utf-8", errors="ignore")
    orig = text

    # Add include for json_safe.hpp if needed
    if 'safeJsonValue' in text and '"json_safe.hpp"' not in text:
        pass

    # Replace var.value( with safeJsonValue(var, for target variables
    for var in variables:
        # avoid double-patching safeJsonValue(var, "...", var.value(...))
        text = re.sub(
            rf"(?<!safeJsonValue\({var},\s*\"[^\"]+\",\s*)\b{var}\.value\(",
            rf"safeJsonValue({var}, ",
            text,
        )
        # Replace var.contains( with safeJsonContains(var,
        text = re.sub(
            rf"(?<!safeJsonContains\({var},\s*)\b{var}\.contains\(",
            rf"safeJsonContains({var}, ",
            text,
        )

    # Add include if any safeJsonValue/safeJsonContains introduced
    if text != orig and '"json_safe.hpp"' not in text:
        # Insert after first #include line that includes a standard header or nlohmann/json
        lines = text.splitlines(keepends=True)
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("#include"):
                insert_idx = i + 1
        lines.insert(insert_idx, '#include "json_safe.hpp"\n')
        text = "".join(lines)

    if text != orig:
        path.write_text(text, encoding="utf-8")
        print(f"patched {rel}")
