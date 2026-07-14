import json
from urllib.request import urlopen

session_id = "session-test-1782477069"
url = f"http://127.0.0.1:5081/context/status?sessionId={session_id}"
try:
    with urlopen(url) as r:
        data = json.loads(r.read())
        print(json.dumps(data, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"error: {e}")
