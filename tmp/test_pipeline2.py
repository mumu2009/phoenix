import json, urllib.request

base = 'http://127.0.0.1:8082'

def post(path, data, timeout=120):
    req = urllib.request.Request(base + path, data=json.dumps(data).encode(), headers={'Content-Type':'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

# apply template
prompt = post('/apply-template', {'messages':[{'role':'user','content':'What is 1+1?'}], 'add_generation_prompt': True})['prompt']
print('prompt repr:', repr(prompt))

# tokenize with pieces
tok = post('/tokenize', {'content': prompt, 'add_special': False, 'with_pieces': True})
print('tokenize n:', len(tok['tokens']))
for t in tok['tokens']:
    print(t['id'], repr(t['piece']))
