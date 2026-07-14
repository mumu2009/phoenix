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


def test_various_recall_prompts(sid: str):
    key = "senj_no_valkyria_3_unrecorded_chronicles"
    store_prompt = f"MEMTAG: {key}"
    
    # store
    s1, r1 = call(store_prompt, sid, max_tokens=8)
    print(f"store: {r1!r}")
    
    # recall with different prompts
    prompts = [
        f"What is the MEMTAG value? Reply with only the value, no extra text. MEMTAG=",
        f"Complete the sentence: MEMTAG is set to ",
        f"The MEMTAG value is: ",
        f"Question: What is MEMTAG? Answer: {key}\nQuestion: What is MEMTAG? Answer:",
    ]
    
    for i, prompt in enumerate(prompts):
        hint = f"User: {store_prompt}"
        s, r = call(prompt, sid, context_hint=hint, max_tokens=64)
        sim = semantic_similarity(r, key)
        print(f"\nPrompt {i}: {prompt[:60]!r}")
        print(f"Reply: {r!r}")
        print(f"sim={sim:.4f} PASS={sim>=0.70}")


if __name__ == "__main__":
    t = int(time.time())
    test_various_recall_prompts(f"debug-var-{t}")
