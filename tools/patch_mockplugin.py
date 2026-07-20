import pathlib
import re
import sys

files = [
    "tests/gtest/unit/plugin_system/test_plugin_lifecycle.cpp",
    "tests/gtest/unit/plugin_system/test_plugin_manager.cpp",
    "tests/gtest/unit/plugin_system/test_plugin_operations.cpp",
]

for p in files:
    path = pathlib.Path(p)
    s = path.read_text(encoding="utf-8").replace("\r\n", "\n")

    # Remove the temporary destructor override we previously added (if present)
    s = s.replace(
        "    bool onUnload() override { return true; }\n    ~MockPlugin() override;\n\n    bool hasCapability(PluginCapability cap) const override {",
        "    bool onUnload() override { return true; }\n\n    bool hasCapability(PluginCapability cap) const override {",
        1,
    )
    s = s.replace(
        "};\n\nMockPlugin::~MockPlugin() = default;\n\nclass PluginOperationsTest : public HacktestBase {",
        "};\n\nclass PluginOperationsTest : public HacktestBase {",
        1,
    )

    # Wrap the per-file MockPlugin class in an unnamed namespace to avoid ODR violations
    s = s.replace(
        "// Mock Plugin for testing\nclass MockPlugin : public Plugin {",
        "// Mock Plugin for testing\nnamespace {\nclass MockPlugin : public Plugin {",
        1,
    )

    m = re.search(r"class (Plugin\w+Test)\s*:\s*public\s+HacktestBase\s*\{", s)
    if m:
        s = s[: m.start()] + "}\n\n" + s[m.start() :]
    else:
        print(f"WARN: no fixture class found in {p}")
        sys.exit(1)

    path.write_text(s, encoding="utf-8", newline="\n")
    print(f"patched {p}")
