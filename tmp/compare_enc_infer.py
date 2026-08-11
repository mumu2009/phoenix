import json, urllib.request, math

base = 'http://127.0.0.1:8082'

def post(path, data, timeout=120):
    req = urllib.request.Request(base + path, data=json.dumps(data).encode(), headers={'Content-Type':'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

prompt = post('/apply-template', {'messages':[{'role':'user','content':'What is 1+1?'}], 'add_generation_prompt': True})['prompt']
enc = post('/phx/enc', {'content': prompt, 'add_special': True})
positions = list(range(enc['n_tokens']))
infer = post('/phx/infer', {'hidden': enc['hidden'], 'positions': positions})

print('n_tokens', enc['n_tokens'])
print('enc[0][:5]', enc['hidden'][0][:5])
print('infer[0][:5]', infer['hidden'][0][:5])

d = sum((a-b)**2 for a,b in zip(enc['hidden'][0], infer['hidden'][0]))
print('MSE row0', d)
