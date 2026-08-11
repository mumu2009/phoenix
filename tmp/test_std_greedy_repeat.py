import json, urllib.request

base = 'http://127.0.0.1:8082'

def post(path, data, timeout=120):
    req = urllib.request.Request(base + path, data=json.dumps(data).encode(), headers={'Content-Type':'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

text = "What is 1+1? Express the result as an arithmetic equation."
for temp in [0, 0.1, 0.5, 0.7]:
    for rep in [1.0, 1.1, 1.2]:
        comp = post('/v1/chat/completions', {
            'messages': [{'role':'user','content': text}],
            'temperature': temp,
            'top_p': 0.9,
            'max_tokens': 24,
            'repeat_penalty': rep,
        }, timeout=120)
        print(f'temp={temp} rep={rep} -> {repr(comp["choices"][0]["message"]["content"])}')
