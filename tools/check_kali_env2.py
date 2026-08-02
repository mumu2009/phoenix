import paramiko

def run(cmd):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('192.168.0.100', username='kali', password='kali', timeout=10)
    _, o, e = c.exec_command(cmd)
    out = o.read().decode().strip()
    err = e.read().decode().strip()
    print(f'--- {cmd} ---')
    print(out)
    if err:
        print('ERR:', err)
    c.close()

if __name__ == '__main__':
    for mod in ['numpy', 'soundfile', 'librosa', 'torch', 'onnx', 'scipy', 'tqdm']:
        run(f"python3 -c 'import {mod}; print(\"{mod} ok\", {mod}.__version__ if hasattr({mod}, \"__version__\") else \"\")'")
