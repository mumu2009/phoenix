import json, urllib.request

base = 'http://127.0.0.1:8082'

def post(path, data, timeout=120):
    req = urllib.request.Request(base + path, data=json.dumps(data).encode(), headers={'Content-Type':'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

text = "What is 1+1? Express the result as an arithmetic equation."
prompt = post('/apply-template', {'messages':[{'role':'user','content': text}], 'add_generation_prompt': True})['prompt']

enc = post('/phx/enc', {'content': prompt, 'add_special': True})
n = enc['n_tokens']
positions = list(range(n))
infer = post('/phx/infer', {'hidden': enc['hidden'], 'positions': positions})
last = infer['hidden'][-1]

max_tokens = 24
generated = []
eos = 128009
for step in range(max_tokens):
    dec = post('/phx/dec', {'hidden': [last], 'with_logits': True})
    logits = dec['logits'][0] if isinstance(dec['logits'][0], list) else dec['logits']
    tok = max(range(len(logits)), key=lambda i: logits[i])
    if tok == eos:
        break
    generated.append(tok)
    enc_tok = post('/phx/enc', {'tokens': [tok], 'add_special': False})
    pos = n + step
    infer_tok = post('/phx/infer', {'hidden': enc_tok['hidden'], 'positions': [pos]})
    last = infer_tok['hidden'][0]

reply = post('/detokenize', {'tokens': generated})['content']
print('generated tokens:', generated)
print('reply:', repr(reply))
