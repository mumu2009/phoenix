import json, urllib.request, sys, time

base = 'http://127.0.0.1:8082'

def post(path, data, timeout=120):
    req = urllib.request.Request(base + path, data=json.dumps(data).encode(), headers={'Content-Type':'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

# 1. apply template
prompt = post('/apply-template', {'messages':[{'role':'user','content':'What is 1+1?'}], 'add_generation_prompt': True})['prompt']
print('prompt:', prompt)

# 2. encode prompt
enc = post('/phx/enc', {'content': prompt, 'add_special': False})
print('n_tokens:', enc['n_tokens'])

# 3. infer full prompt
n = enc['n_tokens']
positions = list(range(n))
infer = post('/phx/infer', {'hidden': enc['hidden'], 'positions': positions})
print('infer n_tokens:', infer['n_tokens'])

# 4. decode last hidden
last = infer['hidden'][-1]
dec = post('/phx/dec', {'hidden': [last], 'with_logits': True})
print('dec n_tokens:', dec.get('n_tokens'))
print('dec top token:', dec['tokens'][0]['token'], 'piece:', repr(dec['tokens'][0]['piece']))

# 5. compare with /v1/chat/completions for same prompt
comp = post('/v1/chat/completions', {
    'messages':[{'role':'user','content':'What is 1+1?'}],
    'temperature':0,
    'max_tokens':8,
    'seed':1,
    'stop':['<|eot_id|>']
}, timeout=180)
print('completions content:', comp['choices'][0]['message']['content'])
