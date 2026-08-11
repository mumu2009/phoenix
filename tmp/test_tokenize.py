import json
import urllib.request


def post(url, data):
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


prompt = post("http://127.0.0.1:8084/apply-template", {
    "messages": [{"role": "user", "content": "What is 1+1?"}],
    "add_generation_prompt": True,
})["prompt"]
print("PROMPT:", repr(prompt))

for add in (True, False):
    tok = post("http://127.0.0.1:8084/tokenize", {
        "content": prompt,
        "add_special": add,
        "with_pieces": True,
    })
    print("add_special=", add)
    print(json.dumps(tok, ensure_ascii=False, indent=2)[:500])
