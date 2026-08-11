import json, urllib.request

base = 'http://127.0.0.1:8082'

def post(path, data, timeout=120):
    req = urllib.request.Request(base + path, data=json.dumps(data).encode(), headers={'Content-Type':'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

tokens = [16, 10]
enc = post('/phx/enc', {'tokens': tokens, 'add_special': False})
infer = post('/phx/infer', {'hidden': enc['hidden'], 'positions': [0, 1]})
for i in range(len(tokens)):
    dec = post('/phx/dec', {'hidden': [infer['hidden'][i]], 'with_logits': True})
    print(i, 'input', tokens[i], '->', dec['tokens'][0]['token'], repr(dec['tokens'][0]['piece']))
