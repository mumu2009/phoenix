import json, urllib.request

base = 'http://127.0.0.1:8082'

def post(path, data, timeout=120):
    req = urllib.request.Request(base + path, data=json.dumps(data).encode(), headers={'Content-Type':'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

text = "What is 1+1? Express the result as an arithmetic equation."
prompt = post('/apply-template', {'messages':[{'role':'user','content': text}], 'add_generation_prompt': True})['prompt']
enc = post('/phx/enc', {'content': prompt, 'add_special': True})
infer = post('/phx/infer', {'hidden': enc['hidden'], 'positions': list(range(enc['n_tokens']))})
dec = post('/phx/dec', {'hidden': [infer['hidden'][-1]], 'with_logits': True})
print('keys:', list(dec.keys()))
print('n_tokens:', dec.get('n_tokens'))
print('tokens[0]:', dec['tokens'][0])
print('logits type:', type(dec['logits']))
print('logits[0] type:', type(dec['logits'][0]) if len(dec['logits'])>0 else 'empty')
if dec['logits']:
    print('logits[0] is list?', isinstance(dec['logits'][0], list))
    if isinstance(dec['logits'][0], list):
        print('logits[0] len:', len(dec['logits'][0]))
