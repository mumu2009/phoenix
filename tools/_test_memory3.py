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


def test_with_context_hint(sid: str):
    print(f"\n=== Test with contextHint (sid={sid}) ===")
    key = "gamma_456_delta"
    store_prompt = f"Remember the following token for this conversation. Reply exactly with ACK. MEMTAG: {key}"
    s1, r1 = call(store_prompt, sid, max_tokens=8)
    print(f"store -> {s1}: {r1[:80]}")
    s2, r2 = call("Filler turn. Reply exactly with OK.", sid, max_tokens=4)
    print(f"filler -> {s2}: {r2[:80]}")
    hint = f"User: {store_prompt}\nAssistant: {r1}\nUser: Filler turn. Reply exactly with OK.\nAssistant: {r2}"
    recall_prompt = "What is the MEMTAG value from earlier? Reply with the exact value only."
    s3, r3 = call(recall_prompt, sid, context_hint=hint, max_tokens=48)
    print(f"recall -> {s3}: {r3[:120]}")
    ok = key in r3
    print(f"PASS={ok} (expected {key})")
    return ok


def test_cross_session_style(sid_a: str, sid_b: str):
    print(f"\n=== Test cross-session style (sid_a={sid_a}, sid_b={sid_b}) ===")
    key = "token_789_xyz"
    profile = "profile_1234"
    store_prompt = f"Remember this label and its value for future reference. Reply ACK only. LABEL: {profile}; TOKEN: {key}"
    s1, r1 = call(store_prompt, sid_a, max_tokens=8)
    print(f"store -> {s1}: {r1[:80]}")
    hint = f"User: {store_prompt}\nAssistant: {r1}\nUser: LABEL: {profile}; TOKEN: {key}"
    recall_prompt = f"What TOKEN value was stored under LABEL {profile}? Reply with the exact value only."
    s2, r2 = call(recall_prompt, sid_b, context_hint=hint, max_tokens=48)
    print(f"recall -> {s2}: {r2[:120]}")
    ok = key in r2
    print(f"PASS={ok} (expected {key})")
    return ok


if __name__ == "__main__":
    t = int(time.time())
    results = []
    results.append(("context_hint", test_with_context_hint(f"memtest-hint-{t}")))
    results.append(("cross_session", test_cross_session_style(f"memtest-a-{t}", f"memtest-b-{t}")))
    print("\n=== SUMMARY ===")
    for name, ok in results:
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
