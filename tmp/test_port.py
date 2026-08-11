import urllib.request, json
req = urllib.request.Request(
    'http://127.0.0.1:5080/api/chat',
    data=json.dumps({"message": "What is 1+1? Express the result as an arithmetic equation."}).encode(),
    headers={'Content-Type': 'application/json', 'Authorization': 'Bearer local-dev'},
    method='POST'
)
try:
    print(urllib.request.urlopen(req, timeout=30).read().decode()[:200])
except Exception as e:
    print('error', e)
