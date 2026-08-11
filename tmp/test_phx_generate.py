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

prompt = post("http://127.0.0.1:8082/apply-template", {
    "messages": [{"role": "user", "content": "What is 1+1?"}],
    "add_generation_prompt": True,
})["prompt"]
print("PROMPT:", repr(prompt))

result = post("http://127.0.0.1:8082/phx/generate", {
    "content": prompt,
    "max_tokens": 20,
    "temperature": 0.0,
    "decode_text": True,
    "return_hidden": False,
})
print("n_tokens:", result.get("n_tokens"))
print("tokens:", result.get("tokens"))
print("text:", result.get("text"))
