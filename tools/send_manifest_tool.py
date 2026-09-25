import os

import paramiko

REMOTE_HOST = os.environ.get("PHOENIX_REMOTE_HOST", "192.168.1.100")
REMOTE_USER = os.environ.get("PHOENIX_REMOTE_USER", "user")
REMOTE_PASS = os.environ.get("PHOENIX_REMOTE_PASS", "")
REMOTE_BASE = os.environ.get("PHOENIX_REMOTE_BASE", f"/home/{REMOTE_USER}/phoenix")

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(REMOTE_HOST, username=REMOTE_USER, password=REMOTE_PASS, timeout=10)
sftp = c.open_sftp()
# ensure remote dirs
for d in [f'{REMOTE_BASE}/tools', f'{REMOTE_BASE}/runtime_store/checkpoints']:
    try:
        sftp.mkdir(d)
    except IOError:
        pass
here = os.path.dirname(os.path.abspath(__file__))
sftp.put(os.path.join(here, 'create_musan_manifest.py'), f'{REMOTE_BASE}/tools/create_musan_manifest.py')
sftp.close()
_, o, _ = c.exec_command(f'cd {REMOTE_BASE} && python3 tools/create_musan_manifest.py')
print(o.read().decode().strip())
c.close()
