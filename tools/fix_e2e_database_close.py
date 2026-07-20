import re
from pathlib import Path

path = Path(r"d:\_phoenix\_079\v6.0Alixander\phoenix\tests\gtest\integration\test_end_to_end.cpp")
lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

out = []
i = 0
n = len(lines)

while i < n:
    line = lines[i]
    if re.match(r"^TEST\(EndToEndSessionFlow,", line):
        # collect block until the matching '}' at column 0
        block = [line]
        i += 1
        while i < n:
            block.append(lines[i])
            if lines[i].strip() == "}":
                i += 1
                break
            i += 1
        block_text = "".join(block)
        if "Database079 database(testDb," in block_text:
            marker = "std::filesystem::remove(testDb);"
            pos = block_text.rfind(marker)
            if pos != -1:
                block_text = block_text[:pos] + "    database.close();\n" + block_text[pos:]
        out.append(block_text)
    else:
        out.append(line)
        i += 1

path.write_text("".join(out), encoding="utf-8")
print("inserted database.close() before final removes")
