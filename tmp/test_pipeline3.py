import json, urllib.request

base = 'http://127.0.0.1:8082'

def post(path, data, timeout=120):
    req = urllib.request.Request(base + path, data=json.dumps(data).encode(), headers={'Content-Type':'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

prompt = post('/apply-template', {'messages':[{'role':'user','content':'What is 1+1?'}], 'add_generation_prompt': True})['prompt']
print('prompt:', repr(prompt))

# encode with add_special=true
enc = post('/phx/enc', {'content': prompt, 'add_special': True})
print('enc n_tokens:', enc['n_tokens'])

# infer
n = enc['n_tokens']
positions = list(range(n))
infer = post('/phx/infer', {'hidden': enc['hidden'], 'positions': positions})
print('infer n_tokens:', infer['n_tokens'])

# dec last hidden
last = infer['hidden'][-1]
dec = post('/phx/dec', {'hidden': [last], 'with_logits': True})
print('dec top token:', dec['tokens'][0]['token'], 'piece:', repr(dec['tokens'][0]['piece']))
