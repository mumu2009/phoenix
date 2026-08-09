import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "test-tools"))
import ui_e2e_test as ui

procs = ui.start_phoenix()
if procs is None:
    raise RuntimeError("phoenix already running or failed")

for _ in range(30):
    try:
        with urllib.request.urlopen("http://127.0.0.1:5080/api/world/status", timeout=3):
            break
    except Exception:
        time.sleep(1)

LONG_PROMPT = (
    "Question: In agricultural production, the most important irreplaceable "
    "basic means of production used as labor objects is\\n"
    "A. agricultural tools\\n"
    "B. land\\n"
    "C. labor\\n"
    "D. capital\\n"
    "Please output only the correct option letter (A/B/C/D), no explanation."
)

for text in [LONG_PROMPT]:
    body = json.dumps({"text": text, "token": "local-dev", "maxTokens": 16}).encode()
    req = urllib.request.Request("http://127.0.0.1:5080/api/chat", data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer local-dev")
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            print("OK", r.read().decode())
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode())

for p in reversed(procs):
    if p:
        p.terminate()
        try:
            p.wait(timeout=5)
        except Exception:
            p.kill()
