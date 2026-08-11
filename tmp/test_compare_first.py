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
print('split first token:', dec['tokens'][0]['token'], repr(dec['tokens'][0]['piece']))

comp = post('/v1/chat/completions', {
    'messages': [{'role':'user','content': text}],
    'temperature': 0,
    'max_tokens': 1,
    'seed': 1,
}, timeout=120)
print('std first token:', repr(comp['choices'][0]['message']['content']))
