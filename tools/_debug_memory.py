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


def benchmark_like_long(sid: str):
    fact = "senj_no_valkyria_3_unrecorded_chronicles is the next game in the series"
    key = "senj_no_valkyria_3_unrecorded_chronicles"
    expected = key
    
    store_prompt = f"Remember the following token for this conversation. Reply exactly with ACK. MEMTAG: {key}"
    recall_prompt = "What is the MEMTAG value from earlier in this conversation? Reply with the exact value only."
    
    history = []
    s1, r1 = call(store_prompt, sid, max_tokens=8)
    print(f"store: status={s1} reply={r1!r}")
    history.append(f"User: {store_prompt}")
    history.append(f"Assistant: {r1}")
    
    fillers = [
        "Filler turn 1. Reply exactly with OK.",
        "Filler turn 2. Reply exactly with OK.",
        "Filler turn 3. Reply exactly with OK.",
        "Filler turn 4. Reply exactly with OK.",
    ]
    for f in fillers:
        s, r = call(f, sid, max_tokens=4)
        print(f"filler: status={s} reply={r!r}")
        history.append(f"User: {f}")
        history.append(f"Assistant: {r}")
    
    hint = "\n".join(history)
    print(f"\nCONTEXT HINT:\n{hint}\n")
    s3, r3 = call(recall_prompt, sid, context_hint=hint, max_tokens=128)
    print(f"recall: status={s3} reply={r3!r}")
    sim = semantic_similarity(r3, expected)
    print(f"similarity={sim:.4f} PASS={sim >= 0.70}")
    return sim >= 0.70


def benchmark_like_long_no_context(sid: str):
    """Same but no contextHint - to confirm session memory does not work"""
    fact = "senj_no_valkyria_3_unrecorded_chronicles is the next game in the series"
    key = "senj_no_valkyria_3_unrecorded_chronicles"
    
    store_prompt = f"Remember the following token for this conversation. Reply exactly with ACK. MEMTAG: {key}"
    recall_prompt = "What is the MEMTAG value from earlier in this conversation? Reply with the exact value only."
    
    s1, r1 = call(store_prompt, sid, max_tokens=8)
    print(f"store: status={s1} reply={r1!r}")
    
    fillers = ["Filler turn 1. Reply exactly with OK.", "Filler turn 2. Reply exactly with OK.", "Filler turn 3. Reply exactly with OK.", "Filler turn 4. Reply exactly with OK."]
    for f in fillers:
        s, r = call(f, sid, max_tokens=4)
        print(f"filler: status={s} reply={r!r}")
    
    s3, r3 = call(recall_prompt, sid, max_tokens=128)
    print(f"recall (no context): status={s3} reply={r3!r}")
    sim = semantic_similarity(r3, key)
    print(f"similarity={sim:.4f} PASS={sim >= 0.70}")
    return sim >= 0.70


if __name__ == "__main__":
    t = int(time.time())
    print("\n=== WITH CONTEXT HINT ===")
    ok1 = benchmark_like_long(f"debug-long-hint-{t}")
    print("\n=== WITHOUT CONTEXT HINT ===")
    ok2 = benchmark_like_long_no_context(f"debug-long-nohint-{t}")
    print(f"\nSUMMARY: hint={ok1}, nohint={ok2}")
