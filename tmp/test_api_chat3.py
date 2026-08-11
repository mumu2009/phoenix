import urllib.request, json
req = urllib.request.Request(
    'http://127.0.0.1:5080/api/chat',
    data=json.dumps({"text": "What is 2+2? Express the result as an arithmetic equation.", "maxTokens": 24}).encode(),
    headers={'Content-Type': 'application/json', 'Authorization': 'Bearer local-dev'},
    method='POST'
)
try:
    print(urllib.request.urlopen(req, timeout=90).read().decode())
except Exception as e:
    print('error', e)
