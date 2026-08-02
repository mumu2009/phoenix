import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.0.100', username='kali', password='kali', timeout=10)
cmds = [
    'du -sh /home/kali/phoenix/datasets/musan_16k',
    'find /home/kali/phoenix/datasets/musan_16k -type f -size -1k | wc -l',
    'find /home/kali/phoenix/datasets/musan_16k -type f -size +1M | wc -l',
    'find /home/kali/phoenix/datasets/musan_16k -type f | wc -l',
    'df -h /home/kali',
]
for cmd in cmds:
    _, o, e = c.exec_command(cmd)
    print(f'--- {cmd} ---')
    print(o.read().decode().strip())
    if e.read().decode().strip():
        print('ERR:', e.read().decode().strip())
c.close()
