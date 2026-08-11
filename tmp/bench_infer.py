import json, urllib.request, time

base = 'http://127.0.0.1:8082'

def post(path, data, timeout=300):
    req = urllib.request.Request(base + path, data=json.dumps(data).encode(), headers={'Content-Type':'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

# encode prompt
text = "What is 1+1? Express the result as an arithmetic equation."
prompt = post('/apply-template', {'messages':[{'role':'user','content': text}], 'add_generation_prompt': True})['prompt']
t0 = time.time()
enc = post('/phx/enc', {'content': prompt, 'add_special': True})
t1 = time.time()
print('enc n_tokens', enc['n_tokens'], 'time', t1-t0)
n = enc['n_tokens']

# infer prompt
t0 = time.time()
infer = post('/phx/infer', {'hidden': enc['hidden'], 'positions': list(range(n))})
t1 = time.time()
print('infer prompt time', t1-t0)
last = infer['hidden'][-1]

# decode first token
t0 = time.time()
dec = post('/phx/dec', {'hidden': [last], 'with_logits': True})
t1 = time.time()
print('dec time', t1-t0)

# encode a single token and infer it
tok = 16
t0 = time.time()
enc_tok = post('/phx/enc', {'tokens': [tok], 'add_special': False})
t1 = time.time()
print('enc single tok time', t1-t0)
t0 = time.time()
infer_tok = post('/phx/infer', {'hidden': enc_tok['hidden'], 'positions': [n]})
t1 = time.time()
print('infer single tok time', t1-t0)
