import json, urllib.request, math, random

base = 'http://127.0.0.1:8082'

def post(path, data, timeout=120):
    req = urllib.request.Request(base + path, data=json.dumps(data).encode(), headers={'Content-Type':'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def sample(logits, temperature, top_p, rng):
    m = max(logits)
    exp = [math.exp((l - m) / temperature) for l in logits]
    total = sum(exp)
    probs = sorted([(e / total, i) for i, e in enumerate(exp)], reverse=True)
    cum = 0.0
    cutoff = len(probs)
    for i, (p, _) in enumerate(probs):
        cum += p
        if cum >= top_p:
            cutoff = i + 1
            break
    renorm = sum(p for p, _ in probs[:cutoff])
    r = rng.random() * renorm
    acc = 0.0
    for p, i in probs[:cutoff]:
        acc += p
        if r <= acc:
            return i
    return probs[cutoff-1][1]

text = "What is 1+1? Express the result as an arithmetic equation."
mod = "<emotion: high-arousal; directive; tense; optimistic>"
prompt_text = mod + "\n\n" + text
prompt = post('/apply-template', {'messages':[{'role':'user','content': prompt_text}], 'add_generation_prompt': True})['prompt']
enc = post('/phx/enc', {'content': prompt, 'add_special': True})
n = enc['n_tokens']
infer = post('/phx/infer', {'hidden': enc['hidden'], 'positions': list(range(n))})
last = infer['hidden'][-1]

for seed in range(30):
    rng = random.Random(seed)
    out = []
    pos = n
    eos = 128009
    cur = last
    for step in range(24):
        dec = post('/phx/dec', {'hidden': [cur], 'with_logits': True})
        tok = sample(dec['logits'][0], 1.5, 1.0, rng)
        if tok == eos:
            break
        out.append(tok)
        enc_tok = post('/phx/enc', {'tokens': [tok], 'add_special': False})
        infer_tok = post('/phx/infer', {'hidden': enc_tok['hidden'], 'positions': [pos]})
        cur = infer_tok['hidden'][0]
        pos += 1
    reply = post('/detokenize', {'tokens': out})['content']
    if 'To be' in reply or reply == 'To be' or reply == 'The':
        print('seed', seed, 'reply:', repr(reply))
