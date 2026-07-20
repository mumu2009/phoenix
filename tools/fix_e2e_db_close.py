from pathlib import Path

path = Path(r"d:\_phoenix\_079\v6.0Alixander\phoenix\tests\gtest\integration\test_end_to_end.cpp")
lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

out = []
in_session_test = False
database_declared = False

for line in lines:
    stripped = line.strip()
    if stripped.startswith("TEST(EndToEndSessionFlow,"):
        in_session_test = True
        database_declared = False
    elif in_session_test and stripped.startswith("TEST("):
        in_session_test = False

    if in_session_test:
        if "Database079 database" in stripped and stripped.endswith(")") and ";" in stripped:
            database_declared = True
        if stripped == "database.close();" and not database_declared:
            continue

    out.append(line)

path.write_text("".join(out), encoding="utf-8")
print("fixed invalid database.close() calls")
