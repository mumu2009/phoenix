import json, urllib.request

base = 'http://127.0.0.1:8082'

def post(path, data, timeout=120):
    req = urllib.request.Request(base + path, data=json.dumps(data).encode(), headers={'Content-Type':'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

prompt = post('/apply-template', {'messages':[{'role':'user','content':'What is 1+1?'}], 'add_generation_prompt': True})['prompt']
enc = post('/phx/enc', {'content': prompt, 'add_special': True})
n = enc['n_tokens']
positions = list(range(n))
infer = post('/phx/infer', {'hidden': enc['hidden'], 'positions': positions})
print('infer n_tokens:', infer['n_tokens'])

for idx in [0, 1, 2, n//2, n-1]:
    dec = post('/phx/dec', {'hidden': [infer['hidden'][idx]], 'with_logits': True})
    print(idx, dec['tokens'][0]['token'], repr(dec['tokens'][0]['piece']))
