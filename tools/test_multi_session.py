from urllib.request import urlopen, Request
import json
import sys
import time

URL = "http://127.0.0.1:5080/api/chat"
TOKEN = "local-dev"
N = 15


def ask(sid, text):
    body = json.dumps({
        "text": text,
        "sessionId": sid,
        "maxTokens": 32,
        "benchmarkSinglePass": True,
        "disableReasoningCritic": True,
        "disableExternalStyleTrainStep": True,
        "enableAddonToolContract": False,
        "enableGraphSelector": False,
    }).encode()
    try:
        r = urlopen(
            Request(URL, data=body, method="POST",
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"}),
            timeout=90,
        )
        d = json.loads(r.read())
        reply = d.get("result", {}).get("reply", "")
        idx = sid.split("-")[-1]
        print(f"i={idx:>3} ok={bool(reply)} reply={reply[:30]!r}", flush=True)
        return True
    except Exception as e:
        idx = sid.split("-")[-1]
        print(f"i={idx:>3} ERR={e}", flush=True)
        return False


print(f"--- testing {N} different sessionIds ---")
for i in range(N):
    ok = ask(f"memv1-short-phoenix-{i}", f"What is {i}+{i}?")
    if not ok:
        print(f"CRASHED at i={i}")
        sys.exit(1)
    time.sleep(0.2)
print("ALL PASSED")
