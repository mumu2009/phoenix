import paramiko
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('192.168.0.100', username='kali', password='kali', timeout=20)
stdin, stdout, stderr = client.exec_command('find /home/kali -name encoder_head.json 2>/dev/null')
out = stdout.read().decode()
print(out)
client.close()
