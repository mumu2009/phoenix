import urllib.request
req = urllib.request.Request('http://127.0.0.1:5080/api/monitoring/reset', headers={'Authorization': 'Bearer local-dev'}, method='POST')
print(urllib.request.urlopen(req, timeout=30).read().decode())
