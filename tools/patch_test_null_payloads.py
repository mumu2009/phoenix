import re
from pathlib import Path

root = Path(r"d:\_phoenix\_079\v6.0Alixander\phoenix\tests\gtest")
for path in root.rglob("*.cpp"):
    text = path.read_text(encoding="utf-8", errors="ignore")
    orig = text
    # Replace empty-braces and nullptr initialization of nlohmann::json variables with empty object.
    text = re.sub(
        r"(nlohmann::json\s+\w+\s*=\s*)\{\}",
        r"\1nlohmann::json::object()",
        text,
    )
    text = re.sub(
        r"(nlohmann::json\s+\w+\s*=\s*)nullptr",
        r"\1nlohmann::json::object()",
        text,
    )
    if text != orig:
        path.write_text(text, encoding="utf-8")
        print("patched", path.relative_to(root.parent))

print("done")
