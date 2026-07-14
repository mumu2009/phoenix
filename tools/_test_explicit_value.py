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


def semantic_similarity(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    a = re.sub(r"[^\w]", "", a.lower())
    b = re.sub(r"[^\w]", "", b.lower())
    return SequenceMatcher(None, a, b).ratio()


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


def test_explicit_in_prompt(sid: str):
    print(f"\n=== Explicit value in prompt (sid={sid}) ===")
    key = "senj_no_valkyria_3_unrecorded_chronicles"
    
    # Store with example
    store_prompt = f"The value of MEMTAG is exactly {key}. Later you will be asked to repeat it. Confirm you understand by replying: {key}"
    s1, r1 = call(store_prompt, sid, max_tokens=32)
    print(f"store: {r1[:80]!r}")
    
    # fillers
    for j in range(3):
        s, r = call("Say OK.", sid, max_tokens=4)
        print(f"filler {j}: {r[:40]!r}")
    
    # Recall with example
    recall_prompt = f"The MEMTAG value is {key}. What is the MEMTAG value? Reply with exactly: {key}"
    s3, r3 = call(recall_prompt, sid, max_tokens=64)
    print(f"recall: {r3!r}")
    sim = semantic_similarity(r3, key)
    print(f"sim={sim:.4f} PASS={sim>=0.70}")
    return sim >= 0.70


def test_extract_after_keyword(sid: str):
    print(f"\n=== Extract from keyword (sid={sid}) ===")
    key = "senj_no_valkyria_3_unrecorded_chronicles"
    
    store_prompt = f"MEMTAG={key}"
    s1, r1 = call(store_prompt, sid, max_tokens=8)
    print(f"store: {r1[:80]!r}")
    
    for j in range(3):
        call("OK.", sid, max_tokens=4)
    
    recall_prompt = f"Complete: MEMTAG="
    s3, r3 = call(recall_prompt, sid, max_tokens=64)
    print(f"recall: {r3!r}")
    sim = semantic_similarity(r3, key)
    print(f"sim={sim:.4f} PASS={sim>=0.70}")
    return sim >= 0.70


if __name__ == "__main__":
    t = int(time.time())
    ok1 = test_explicit_in_prompt(f"explicit-{t}")
    ok2 = test_extract_after_keyword(f"keyword-{t}")
    print(f"\nSUMMARY: explicit={ok1}, keyword={ok2}")
