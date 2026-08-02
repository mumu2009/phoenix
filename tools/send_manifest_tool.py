import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.0.100', username='kali', password='kali', timeout=10)
sftp = c.open_sftp()
# ensure remote dirs
for d in ['/home/kali/phoenix/tools', '/home/kali/phoenix/runtime_store/checkpoints']:
    try:
        sftp.mkdir(d)
    except IOError:
        pass
sftp.put(r'D:\_phoenix\_079\v6.0Alixander\phoenix\tools\create_musan_manifest.py', '/home/kali/phoenix/tools/create_musan_manifest.py')
sftp.close()
_, o, _ = c.exec_command('cd /home/kali/phoenix && python3 tools/create_musan_manifest.py')
print(o.read().decode().strip())
c.close()
