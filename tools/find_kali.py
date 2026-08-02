import paramiko

def test(host, user, pw):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        c.connect(host, username=user, password=pw, timeout=3, allow_agent=False, look_for_keys=False)
        _, out, _ = c.exec_command('hostname')
        return out.read().decode().strip()
    except Exception as e:
        return None
    finally:
        c.close()

for i in range(100, 111):
    host = f'192.168.0.{i}'
    hn = test(host, 'kali', 'kali')
    if hn:
        print('KALI:', host, 'hostname:', hn)
