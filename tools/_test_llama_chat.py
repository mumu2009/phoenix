import json
import time
from urllib.request import urlopen, Request

URL = "http://127.0.0.1:8084/api/chat"
TIMEOUT = 120


def chat(messages, max_tokens=64):
    payload = {
        "model": "llama",
        "messages": messages,
        "stream": False,
        "options": {"num_predict": max_tokens},
    }
    body = json.dumps(payload).encode()
    req = Request(URL, data=body, method="POST",
                  headers={"Content-Type": "application/json", "Connection": "close"})
    try:
        with urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
            return data.get("message", {}).get("content", "")
    except Exception as e:
        return str(e)


key = "senj_no_valkyria_3_unrecorded_chronicles"
messages = [
    {"role": "user", "content": f"Remember the following token: MEMTAG={key}. Reply ACK."},
    {"role": "assistant", "content": "ACK"},
    {"role": "user", "content": "Say OK."},
    {"role": "assistant", "content": "OK"},
    {"role": "user", "content": "Say OK."},
    {"role": "assistant", "content": "OK"},
    {"role": "user", "content": "What is the MEMTAG value? Reply with the exact value only."},
]
print("=== Standard chat with history ===")
print(chat(messages, 64))

# Also test with system prompt
messages2 = [
    {"role": "system", "content": "You are a helpful assistant. Always follow instructions exactly."},
    {"role": "user", "content": f"Remember the following token: MEMTAG={key}. Reply ACK."},
    {"role": "assistant", "content": "ACK"},
    {"role": "user", "content": "What is the MEMTAG value? Reply with the exact value only."},
]
print("\n=== With system prompt ===")
print(chat(messages2, 64))

# Test with simpler completion endpoint
URL2 = "http://127.0.0.1:8084/completion"
prompt = f"User: Remember the following token: MEMTAG={key}. Reply ACK.\nAssistant: ACK\nUser: What is the MEMTAG value? Reply with the exact value only.\nAssistant:"
req2 = Request(URL2, data=json.dumps({"prompt": prompt, "n_predict": 64, "stream": False}).encode(),
               method="POST", headers={"Content-Type": "application/json", "Connection": "close"})
try:
    with urlopen(req2, timeout=TIMEOUT) as resp:
        data2 = json.loads(resp.read().decode("utf-8", errors="replace"))
        print("\n=== Completion endpoint ===")
        print(data2.get("content", ""))
except Exception as e:
    print(f"completion error: {e}")
