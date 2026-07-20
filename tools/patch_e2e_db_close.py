import re
from pathlib import Path

path = Path(r"d:\_phoenix\_079\v6.0Alixander\phoenix\tests\gtest\integration\test_end_to_end.cpp")
text = path.read_text(encoding="utf-8")

# Insert database.close() before std::filesystem::remove(testDb); in this test file.
# This prevents Database079 file-handle leaks from causing remove() failures.
new_text = re.sub(
    r'^(\s*)std::filesystem::remove\(testDb\);',
    r'\1database.close();\n\1std::filesystem::remove(testDb);',
    text,
    flags=re.MULTILINE,
)

if new_text != text:
    path.write_text(new_text, encoding="utf-8")
    print("patched test_end_to_end.cpp")
else:
    print("no changes needed")
