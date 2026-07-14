import sys

p = r"d:\_phoenix\_079\v6.0Alixander\v6.0Alixander\main_hub_parts\116_section_tail.inc"
with open(p, "rb") as f:
    s = f.read().decode("utf-8")

old = 'oss << "【对话上下文提示 w=" << std::fixed << std::setprecision(2)\n        << hint.weight;\n    if (!hint.mode.empty())\n      oss << " mode=" << hint.mode;\n    oss << "】" << hint.text;'
new = 'oss << "[Context hint w=" << std::fixed << std::setprecision(2)\n        << hint.weight;\n    if (!hint.mode.empty())\n      oss << " mode=" << hint.mode;\n    oss << "]" << hint.text;'

if old not in s:
    print("OLD string not found; nothing changed.")
    sys.exit(0)

s = s.replace(old, new)
with open(p, "wb") as f:
    f.write(s.encode("utf-8"))
print("Replaced Chinese context wrapper in", p)
