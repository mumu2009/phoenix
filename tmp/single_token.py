import json, urllib.request

base = 'http://127.0.0.1:8082'

def post(path, data, timeout=120):
    req = urllib.request.Request(base + path, data=json.dumps(data).encode(), headers={'Content-Type':'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

tok = 16
enc = post('/phx/enc', {'tokens': [tok], 'add_special': False})
infer = post('/phx/infer', {'hidden': enc['hidden'], 'positions': [0]})
dec = post('/phx/dec', {'hidden': [infer['hidden'][0]], 'with_logits': True})
print('enc top', enc['tokens'])
print('dec infer top', dec['tokens'][0]['token'], repr(dec['tokens'][0]['piece']))

# standard
comp = post('/v1/completions', {'prompt': [tok], 'temperature': 0, 'max_tokens': 1, 'seed': 1})
print('standard next token', comp['choices'][0]['text'], 'finish', comp['choices'][0]['finish_reason'])
