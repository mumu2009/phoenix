import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.0.100', username='kali', password='kali', timeout=10)
for cmd in ['lspci | grep -i nvidia', 'lspci | grep -i vga', 'cat /proc/cpuinfo | grep processor | wc -l', 'free -h']:
    _, o, e = c.exec_command(cmd)
    print(f'--- {cmd} ---')
    print(o.read().decode().strip())
c.close()
