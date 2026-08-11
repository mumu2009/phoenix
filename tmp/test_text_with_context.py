import json, urllib.request

base = 'http://127.0.0.1:8082'

def post(path, data, timeout=120):
    req = urllib.request.Request(base + path, data=json.dumps(data).encode(), headers={'Content-Type':'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

text = "What is 1+1? Express the result as an arithmetic equation."
for ctx in ["", "high-arousal; directive; tense; optimistic", "Benefit=0.5 Harm=0.2 Net=0.3"]:
    prompt = text
    if ctx:
        prompt = "Context:\n" + ctx + "\n\nUser:\n" + text
    comp = post('/v1/chat/completions', {
        'messages': [{'role':'user','content': prompt}],
        'temperature': 0,
        'top_p': 0.9,
        'max_tokens': 24,
    }, timeout=120)
    print('ctx', repr(ctx), '->', repr(comp['choices'][0]['message']['content']))
