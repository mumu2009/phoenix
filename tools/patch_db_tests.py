import re

path = r"d:\_phoenix\_079\v6.0Alixander\phoenix\tests\gtest\module\test_database_079.cpp"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

out = []
in_test = False
test_brace_depth = 0
top_level_db_vars = set()

def is_test_start(line):
    return re.match(r"^TEST(_F)?\s*\(", line.lstrip()) is not None

for i, raw in enumerate(lines):
    line = raw.rstrip('\n')
    stripped = line.lstrip()

    if is_test_start(line):
        in_test = True
        test_brace_depth = 0
        top_level_db_vars = set()
        out.append(raw)
        # The function body '{' may be on the same line
        if '{' in line and not ('}' in line and line.index('}') < line.index('{')):
            test_brace_depth += 1
        continue

    if in_test:
        # Track brace depth within this TEST body
        for ch in line:
            if ch == '{':
                test_brace_depth += 1
            elif ch == '}':
                test_brace_depth -= 1

        # Detect top-level Database079 declarations (indent 4 spaces, depth == 1)
        m = re.match(r"^    Database079\s+(\w+)\(", line)
        if m and test_brace_depth == 1:
            top_level_db_vars.add(m.group(1))

        # Detect top-level remove calls for .db files (depth == 1)
        if test_brace_depth == 1 and re.match(r"^    std::filesystem::remove\(\w+\);", line):
            if top_level_db_vars:
                close_calls = ' '.join(f'{v}.close();' for v in sorted(top_level_db_vars))
                out.append(f"    {close_calls}\n")

        # test body ends when depth returns to 0 (after processing the closing brace)
        if test_brace_depth <= 0:
            in_test = False
            top_level_db_vars = set()

    out.append(raw)

with open(path, "w", encoding="utf-8") as f:
    f.writelines(out)

print("patched database tests")
