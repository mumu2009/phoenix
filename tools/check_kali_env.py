import paramiko

def check():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('192.168.0.100', username='kali', password='kali', timeout=10)
    commands = [
        'which python3; python3 --version',
        'python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"',
        'python3 -c "import soundfile; print(\"soundfile ok\")"',
        'python3 -c "import librosa; print(\"librosa ok\")"',
        'python3 -c "import numpy; print(\"numpy ok\")"',
        'which pip3 || which pip',
        'nvidia-smi -L 2>/dev/null || echo no nvidia-smi',
        'df -h .',
    ]
    for cmd in commands:
        _, o, e = c.exec_command(cmd)
        out = o.read().decode().strip()
        err = e.read().decode().strip()
        print(f'--- {cmd} ---')
        print(out)
        if err:
            print('ERR:', err)
    c.close()

if __name__ == '__main__':
    check()
