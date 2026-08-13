import paramiko, os

HOST = '192.168.0.100'
USER = 'kali'
PASS = 'kali'
REMOTE_ROOT = '/home/kali/phoenix'

FILES = [
    'tools/train_audio.py',
    'tools/train_video.py',
    'tools/export_multimodal.py',
    'tools/export_speech.py',
    'tools/compile_bpu.sh',
    'tools/run_hb_mapper.py',
    'tools/hb_mapper_patch.py',
    'tools/train_pilot.py',
    'doc/remote_training_package.md',
]

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=20)
sftp = client.open_sftp()

for f in FILES:
    local_path = f
    if not os.path.exists(local_path):
        print(f'[skip] not found: {local_path}')
        continue
    remote = (REMOTE_ROOT + '/' + f).replace('\\', '/')
    data = open(local_path, 'rb').read()
    with sftp.file(remote, 'wb') as out:
        out.write(data)
    # verify
    stdin, stdout, stderr = client.exec_command(f'wc -c {remote}')
    size = int(stdout.read().decode().strip().split()[0])
    print(f'[put] {local_path} -> {remote} ({size} bytes)')

sftp.close()
client.close()
