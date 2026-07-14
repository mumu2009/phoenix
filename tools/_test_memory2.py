import sys
import time
import json
from pathlib import Path
from urllib.request import urlopen, Request
sys.path.insert(0, str(Path(__file__).resolve().parent))

URL = "http://127.0.0.1:5080/api/chat"
TOKEN = "local-dev"
TIMEOUT = 90


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


def extract_value(reply: str, prefix: str) -> str:
    # 尝试从回复中提取 key
    if prefix in reply:
        idx = reply.index(prefix)
        rest = reply[idx + len(prefix):].strip()
        # 取第一个 token
        parts = rest.split()
        if parts:
            return parts[0].strip(".,;:'\"\n")
    return reply


def test_explicit_recall(sid: str):
    print(f"\n=== Test explicit recall (sid={sid}) ===")
    key = "memtag_42_value"
    store_prompt = f"The secret value is MEMTAG={key}. Confirm by repeating it exactly, then say ACK."
    s1, r1 = call(store_prompt, sid, max_tokens=48)
    print(f"store -> {s1}: {r1[:120]}")
    extracted = extract_value(r1, "MEMTAG=")
    print(f"extracted from store: {extracted}")
    
    s2, r2 = call("Please say OK.", sid, max_tokens=8)
    print(f"filler -> {s2}: {r2[:80]}")
    
    recall_prompt = "What was the secret value of MEMTAG? Reply with the exact value only."
    s3, r3 = call(recall_prompt, sid, max_tokens=48)
    print(f"recall -> {s3}: {r3[:120]}")
    ok = key in r3
    print(f"PASS={ok} (expected {key})")
    return ok


def test_with_context_hint(sid: str):
    print(f"\n=== Test with context_hint (sid={sid}) ===")
    key = "memtag_99_secret"
    store_prompt = f"The secret value is MEMTAG={key}. Confirm by repeating it exactly, then say ACK."
    s1, r1 = call(store_prompt, sid, max_tokens=48)
    print(f"store -> {s1}: {r1[:120]}")
    
    s2, r2 = call("Please say OK.", sid, max_tokens=8)
    print(f"filler -> {s2}: {r2[:80]}")
    
    hint = f"User: {store_prompt}\nAssistant: {r1}\nUser: Please say OK.\nAssistant: {r2}"
    recall_prompt = "What was the secret value of MEMTAG? Reply with the exact value only."
    s3, r3 = call(recall_prompt, sid, context_hint=hint, max_tokens=48)
    print(f"recall -> {s3}: {r3[:120]}")
    ok = key in r3
    print(f"PASS={ok} (expected {key})")
    return ok


def test_baseline_direct(sid: str):
    print(f"\n=== Test baseline: single prompt contains both store and recall (sid={sid}) ===")
    key = "memtag_direct_test"
    prompt = f"The secret value is MEMTAG={key}.\nQuestion: What is the value of MEMTAG? Reply with the exact value only."
    s, r = call(prompt, sid, max_tokens=48)
    print(f"direct -> {s}: {r[:120]}")
    ok = key in r
    print(f"PASS={ok} (expected {key})")
    return ok


if __name__ == "__main__":
    t = int(time.time())
    results = []
    results.append(("explicit_recall", test_explicit_recall(f"memtest-explicit-{t}")))
    results.append(("with_context_hint", test_with_context_hint(f"memtest-hint3-{t}")))
    results.append(("baseline_direct", test_baseline_direct(f"memtest-direct-{t}")))
    print("\n=== SUMMARY ===")
    for name, ok in results:
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
