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


def test_minimal_hint(sid: str):
    key = "senj_no_valkyria_3_unrecorded_chronicles"
    store_prompt = f"Remember the following token for this conversation. Reply exactly with ACK. MEMTAG: {key}"
    recall_prompt = "What is the MEMTAG value from earlier in this conversation? Reply with the exact value only."
    
    # run store and fillers
    s1, r1 = call(store_prompt, sid, max_tokens=8)
    print(f"store: {r1[:80]!r}")
    for j in range(4):
        s, r = call("Filler turn. Reply exactly with OK.", sid, max_tokens=4)
        print(f"filler {j}: {r[:60]!r}")
    
    # minimal hint: only the store user prompt
    hint = f"User: {store_prompt}"
    print(f"\nHINT:\n{hint}\n")
    s3, r3 = call(recall_prompt, sid, context_hint=hint, max_tokens=64)
    print(f"recall: {r3!r}")
    sim = semantic_similarity(r3, key)
    print(f"sim={sim:.4f} PASS={sim>=0.70}")
    return sim >= 0.70


def test_recall_with_value_in_prompt(sid: str):
    key = "senj_no_valkyria_3_unrecorded_chronicles"
    store_prompt = f"Remember the following token for this conversation. Reply exactly with ACK. MEMTAG: {key}"
    
    # run store and fillers
    call(store_prompt, sid, max_tokens=8)
    for j in range(4):
        call("Filler turn. Reply exactly with OK.", sid, max_tokens=4)
    
    # directly provide the value in the prompt
    recall_prompt = f"Earlier in this conversation, MEMTAG was set to {key}. What is the MEMTAG value? Reply with the exact value only."
    s3, r3 = call(recall_prompt, sid, max_tokens=64)
    print(f"\nrecall_with_value: {r3!r}")
    sim = semantic_similarity(r3, key)
    print(f"sim={sim:.4f} PASS={sim>=0.70}")
    return sim >= 0.70


if __name__ == "__main__":
    t = int(time.time())
    print("=== MINIMAL HINT ===")
    ok1 = test_minimal_hint(f"debug-min-{t}")
    print("\n=== VALUE IN PROMPT ===")
    ok2 = test_recall_with_value_in_prompt(f"debug-val-{t}")
    print(f"\nSUMMARY: minimal={ok1}, value_in_prompt={ok2}")
