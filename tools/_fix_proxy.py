import pathlib

target = pathlib.Path("D:/_phoenix/_079/v6.0Alixander/v6.0Alixander/tools/llama_proxy.py")
content = target.read_text(encoding="utf-8")

start = content.find("        def build_chat_prompt(messages):")
end = content.find("        if msgs:")
assert start != -1 and end != -1

new_func = '''        def build_chat_prompt(messages):
            parts = ["<|begin_of_text|>"]
            for m in messages:
                if not isinstance(m, dict):
                    continue
                role = m.get("role", "user")
                content = m.get("content", "")
                if role == "system":
                    parts.append("<|start_header_id|>system<|end_header_id|>\\n\\n" + content + "<|eot_id|>")
                elif role == "assistant":
                    parts.append("<|start_header_id|>assistant<|end_header_id|>\\n\\n" + content + "<|eot_id|>")
                else:
                    parts.append("<|start_header_id|>user<|end_header_id|>\\n\\n" + content + "<|eot_id|>")
            parts.append("<|start_header_id|>assistant<|end_header_id|>\\n\\n")
            return "".join(parts)

'''

target.write_text(content[:start] + new_func + content[end:], encoding="utf-8")
print("fixed")
