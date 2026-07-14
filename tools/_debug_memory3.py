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


def test_store_then_recall(sid: str):
    key = "senj_no_valkyria_3_unrecorded_chronicles"
    store_prompt = f"Remember the following token for this conversation. Reply exactly with ACK. MEMTAG: {key}"
    recall_prompt = "What is the MEMTAG value from earlier in this conversation? Reply with the exact value only."
    
    s1, r1 = call(store_prompt, sid, max_tokens=8)
    print(f"store: status={s1} reply={r1!r}")
    
    # Do not use contextHint; instead embed the store turn directly in the recall text.
    combined_prompt = f"User: {store_prompt}\n{recall_prompt}"
    print(f"COMBINED: {combined_prompt[:160]!r}")
    s2, r2 = call(combined_prompt, sid, max_tokens=64)
    print(f"recall: status={s2} reply={r2!r}")
    sim = semantic_similarity(r2, key)
    print(f"sim={sim:.4f} PASS={sim>=0.70}")
    return sim >= 0.70


def test_single_prompt(sid: str):
    key = "senj_no_valkyria_3_unrecorded_chronicles"
    prompt = f"Remember MEMTAG={key}. What is the MEMTAG value? Reply with the exact value only."
    s, r = call(prompt, sid, max_tokens=64)
    print(f"single: status={s} reply={r!r}")
    sim = semantic_similarity(r, key)
    print(f"sim={sim:.4f} PASS={sim>=0.70}")
    return sim >= 0.70


if __name__ == "__main__":
    t = int(time.time())
    print("=== STORE + MINIMAL HINT ===")
    ok1 = test_store_then_recall(f"debug-min2-{t}")
    print("\n=== SINGLE PROMPT ===")
    ok2 = test_single_prompt(f"debug-single-{t}")
    print(f"\nSUMMARY: store_recall={ok1}, single={ok2}")
