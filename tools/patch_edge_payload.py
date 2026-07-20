import re
import sys

path = r"d:\_phoenix\_079\v6.0Alixander\phoenix\edge_platform.cpp"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# Replace payload.value( with safeJsonValue(payload,  (no-op if already safeJsonValue(payload,)
# Avoid replacing safeJsonValue(payload, "...", payload.value(...)) by only matching payload.value(
text = re.sub(r'(?<![\w.])payload\.value\(', 'safeJsonValue(payload, ', text)

# Replace payload.contains( with payload.is_object() && payload.contains(  but not double-prefix
text = re.sub(r'(?<!payload\.is_object\(\) && )payload\.contains\(', 'payload.is_object() && payload.contains(', text)

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print("patched")
