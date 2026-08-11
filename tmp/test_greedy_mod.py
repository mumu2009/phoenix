import json, urllib.request

base = 'http://127.0.0.1:8082'

def post(path, data, timeout=120):
    req = urllib.request.Request(base + path, data=json.dumps(data).encode(), headers={'Content-Type':'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

for mod in ["", "high-arousal; directive; tense; optimistic", "Benefit=0.5 Harm=0.2 Net=0.3"]:
    text = "What is 1+1? Express the result as an arithmetic equation."
    prompt_text = text
    if mod:
        prompt_text = "Context:\n" + mod + "\n\nUser:\n" + text
    prompt = post('/apply-template', {'messages':[{'role':'user','content': prompt_text}], 'add_generation_prompt': True})['prompt']
    enc = post('/phx/enc', {'content': prompt, 'add_special': True})
    n = enc['n_tokens']
    infer = post('/phx/infer', {'hidden': enc['hidden'], 'positions': list(range(n))})
    last = infer['hidden'][-1]

    out = []
    pos = n
    eos = 128009
    for step in range(24):
        dec = post('/phx/dec', {'hidden': [last], 'with_logits': True})
        logits = dec['logits'][0]
        tok = max(range(len(logits)), key=lambda i: logits[i])
        if tok == eos:
            break
        out.append(tok)
        enc_tok = post('/phx/enc', {'tokens': [tok], 'add_special': False})
        infer_tok = post('/phx/infer', {'hidden': enc_tok['hidden'], 'positions': [pos]})
        last = infer_tok['hidden'][0]
        pos += 1
    print('mod', repr(mod), '->', repr(post('/detokenize', {'tokens': out})['content']))
