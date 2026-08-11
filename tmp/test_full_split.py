import json, urllib.request

base = 'http://127.0.0.1:8082'

def post(path, data, timeout=120):
    req = urllib.request.Request(base + path, data=json.dumps(data).encode(), headers={'Content-Type':'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

# full prompt
prompt = post('/apply-template', {'messages':[{'role':'user','content':'What is 1+1?'}], 'add_generation_prompt': True})['prompt']
enc = post('/phx/enc', {'content': prompt, 'add_special': True})
n = enc['n_tokens']
positions = list(range(n))
infer = post('/phx/infer', {'hidden': enc['hidden'], 'positions': positions})

generated = []
state = infer['hidden']
last_pos = n - 1
for step in range(8):
    last = [state[-1]]
    dec = post('/phx/dec', {'hidden': last, 'with_logits': True})
    tok = dec['tokens'][0]['token']
    piece = dec['tokens'][0]['piece']
    if tok in [128001, 128009]:
        break
    generated.append(piece)
    # advance one token: infer with the new token hidden (which is just the hidden of the generated token?)
    # But to get hidden for the next token, we need the embedding of tok, then infer at last_pos+1.
    enc_tok = post('/phx/enc', {'tokens': [tok], 'add_special': False})
    # set positions for next token
    pos = last_pos + step + 1
    infer_tok = post('/phx/infer', {'hidden': enc_tok['hidden'], 'positions': [pos]})
    state = infer_tok['hidden']

print('generated:', repr(''.join(generated)))
