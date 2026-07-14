import sys
import time
import json
import re
from pathlib import Path
from urllib.request import urlopen, Request
sys.path.insert(0, str(Path(__file__).resolve().parent))

URL = "http://127.0.0.1:5080/api/chat"
TOKEN = "local-dev"
TIMEOUT = 120


def call(text: str, session_id: str, context_hint: str = "", max_tokens: int = 24) -> tuple[int, str]:
    payload = {
        "text": text,
        "sessionId": session_id,
        "maxTokens": max_tokens,
        "benchmarkSinglePass": True,
        "disableReasoningCritic": True,
        "disableExternalStyleTrainStep": True,
        "enableAddonToolContract": False,
        "enableGraphSelector": False,
    }
    if context_hint:
        payload["contextHint"] = context_hint
        payload["contextWeight"] = 0.95
        payload["contextMode"] = "short"
    body = json.dumps(payload).encode()
    req = Request(URL, data=body, method="POST",
                  headers={"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}", "Connection": "close"})
    try:
        with urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
            reply = ""
            if isinstance(data, dict):
                result = data.get("result")
                if isinstance(result, dict) and isinstance(result.get("reply"), str):
                    reply = result["reply"].strip()
                elif isinstance(data.get("reply"), str):
                    reply = data["reply"].strip()
            return resp.status, reply
    except Exception as e:
        return 0, str(e)


def test_numeric(sid: str, key: str):
    print(f"\n=== Numeric key={key} (sid={sid}) ===")
    store_prompt = f"Remember the following token: MEMTAG={key}. Reply ACK."
    recall_prompt = "What is the MEMTAG value? Reply with the exact value only."
    
    s1, r1 = call(store_prompt, sid, max_tokens=8)
    print(f"store: {r1[:80]!r}")
    
    for j in range(3):
        s, r = call("OK.", sid, max_tokens=4)
        print(f"filler {j}: {r[:40]!r}")
    
    s3, r3 = call(recall_prompt, sid, max_tokens=32)
    print(f"recall: {r3!r}")
    ok = key in r3
    print(f"contains={ok}")
    return ok


def test_short_string(sid: str, key: str):
    print(f"\n=== Short string key={key} (sid={sid}) ===")
    store_prompt = f"Remember the following token: MEMTAG={key}. Reply ACK."
    recall_prompt = "What is the MEMTAG value? Reply with the exact value only."
    
    s1, r1 = call(store_prompt, sid, max_tokens=8)
    print(f"store: {r1[:80]!r}")
    
    for j in range(3):
        call("OK.", sid, max_tokens=4)
    
    s3, r3 = call(recall_prompt, sid, max_tokens=32)
    print(f"recall: {r3!r}")
    ok = key in r3
    print(f"contains={ok}")
    return ok


if __name__ == "__main__":
    t = int(time.time())
    r1 = test_numeric(f"num-{t}", "12345678")
    r2 = test_numeric(f"num2-{t}", "987654321")
    r3 = test_short_string(f"short-{t}", "bluecat")
    print(f"\nSUMMARY: num1={r1}, num2={r2}, short={r3}")
