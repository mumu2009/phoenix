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


def test_session_only(sid: str):
    print(f"\n=== Test session_only (sid={sid}) ===")
    key = "alpha_123_beta"
    s1, r1 = call(f"Remember the following token. Reply exactly with ACK. MEMTAG: {key}", sid, max_tokens=8)
    print(f"store -> {s1}: {r1[:80]}")
    s2, r2 = call("Filler turn. Reply exactly with OK.", sid, max_tokens=4)
    print(f"filler -> {s2}: {r2[:80]}")
    s3, r3 = call("What is the MEMTAG value from earlier? Reply exact value only.", sid, max_tokens=24)
    print(f"recall -> {s3}: {r3[:80]}")
    ok = key in r3
    print(f"PASS={ok} (expected {key})")
    return ok


def test_context_hint(sid: str):
    print(f"\n=== Test context_hint (sid={sid}) ===")
    key = "gamma_456_delta"
    history = []
    s1, r1 = call(f"Remember the following token. Reply exactly with ACK. MEMTAG: {key}", sid, max_tokens=8)
    print(f"store -> {s1}: {r1[:80]}")
    history.append(f"User: Remember the following token. Reply exactly with ACK. MEMTAG: {key}")
    history.append(f"Assistant: {r1}")
    s2, r2 = call("Filler turn. Reply exactly with OK.", sid, max_tokens=4)
    print(f"filler -> {s2}: {r2[:80]}")
    history.append("User: Filler turn. Reply exactly with OK.")
    history.append(f"Assistant: {r2}")
    hint = "\n".join(history)
    s3, r3 = call("What is the MEMTAG value from earlier? Reply exact value only.", sid, context_hint=hint, max_tokens=24)
    print(f"recall -> {s3}: {r3[:80]}")
    ok = key in r3
    print(f"PASS={ok} (expected {key})")
    return ok


def test_context_hint_concat(sid: str):
    print(f"\n=== Test context_hint with explicit format (sid={sid}) ===")
    key = "epsilon_789_zeta"
    # 把 prompt 改成更明确的格式
    s1, r1 = call(f"[MEMTAG]={key}\nConfirm you received the MEMTAG by replying ACK.", sid, max_tokens=8)
    print(f"store -> {s1}: {r1[:80]}")
    s2, r2 = call("Please say OK.", sid, max_tokens=4)
    print(f"filler -> {s2}: {r2[:80]}")
    hint = f"User: [MEMTAG]={key}\nConfirm you received the MEMTAG by replying ACK.\nAssistant: {r1}\nUser: Please say OK.\nAssistant: {r2}"
    s3, r3 = call("What is the exact MEMTAG value? Reply only the value.", sid, context_hint=hint, max_tokens=24)
    print(f"recall -> {s3}: {r3[:80]}")
    ok = key in r3
    print(f"PASS={ok} (expected {key})")
    return ok


if __name__ == "__main__":
    t = int(time.time())
    results = []
    results.append(("session_only", test_session_only(f"memtest-session-{t}")))
    results.append(("context_hint", test_context_hint(f"memtest-hint-{t}")))
    results.append(("context_hint_concat", test_context_hint_concat(f"memtest-hint2-{t}")))
    print("\n=== SUMMARY ===")
    for name, ok in results:
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
