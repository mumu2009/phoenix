import re
from pathlib import Path

path = Path(r"d:\_phoenix\_079\v6.0Alixander\phoenix\tests\gtest\integration\test_end_to_end.cpp")
text = path.read_text(encoding="utf-8")

def process_block(block):
    lines = block.splitlines(keepends=True)
    # Indent any dangling lines (if column-0, add 4 spaces)
    lines = ["    " + l if l and l[0].isalnum() and not l.startswith("    ") else l for l in lines]
    has_database = any("Database079 database(testDb," in ln for ln in lines)
    remove_indices = [i for i, ln in enumerate(lines) if "std::filesystem::remove(testDb);" in ln]
    # drop all database.close(); first
    lines = [ln for ln in lines if ln.strip() != "database.close();"]
    if has_database and remove_indices:
        # final remove is the last one (by original order)
        # determine index in the filtered list that corresponds to the last remove
        # easier: find last remaining remove line and insert before it
        last_remove_idx = -1
        for i, ln in enumerate(lines):
            if "std::filesystem::remove(testDb);" in ln:
                last_remove_idx = i
        if last_remove_idx != -1:
            indent = "    "
            # if remove line already has more indent, use that
            m = re.match(r"^(\s*)", lines[last_remove_idx])
            if m:
                indent = m.group(1)
            lines.insert(last_remove_idx, f"{indent}database.close();\n")
    return "".join(lines)

# Split into EndToEndSessionFlow blocks and other text
parts = re.split(r'(TEST\(EndToEndSessionFlow,[^\n]*\n)', text)
out = []
i = 0
while i < len(parts):
    out.append(parts[i])
    i += 1
    if i < len(parts) and parts[i-1].startswith("TEST(EndToEndSessionFlow,"):
        # parts[i] is body up to next TEST or end
        # find end of function: matching '}\n'
        body = parts[i]
        # split at first top-level '}\n'
        m = re.search(r'\n\}\n', body)
        if m:
            block = body[:m.end()-1]
            rest = body[m.end()-1:]
            out.append(process_block(block))
            out.append(rest)
        else:
            out.append(process_block(body))
        i += 1

path.write_text("".join(out), encoding="utf-8")
print("fixed EndToEndSessionFlow blocks")
