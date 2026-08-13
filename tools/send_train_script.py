import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.1.100', username='kali', password='kali', timeout=10)
sftp = c.open_sftp()
try:
    sftp.mkdir('/home/kali/phoenix/tools')
except IOError:
    pass
sftp.put(r'<repo>\tools\train_audio.py', '/home/kali/phoenix/tools/train_audio.py')
sftp.put(r'<repo>\tools\create_musan_manifest.py', '/home/kali/phoenix/tools/create_musan_manifest.py')
sftp.close()
c.close()
print('sent')
