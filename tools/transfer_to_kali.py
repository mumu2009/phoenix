import argparse
import os
from pathlib import Path
import paramiko
from scp import SCPClient


def transfer(local_dir, remote_dir, host='192.168.0.100', user='kali', pw='kali'):
    local_dir = Path(local_dir)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=pw, timeout=10, allow_agent=False, look_for_keys=False)
    # ensure remote dir
    sftp = client.open_sftp()
    try:
        sftp.mkdir(remote_dir)
    except IOError:
        pass
    scp = SCPClient(client.get_transport(), progress=_progress)
    print(f'Transferring {local_dir} to {host}:{remote_dir} ...')
    scp.put(str(local_dir), remote_path=remote_dir, recursive=True)
    scp.close()
    sftp.close()
    client.close()
    print('Done.')


def _progress(filename, size, sent):
    if sent % (10 * 1024 * 1024) == 0 or sent == size:
        print(f'{filename}: {sent}/{size} bytes ({100 * sent // size if size else 0}%)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', required=True)
    parser.add_argument('--remote', required=True)
    parser.add_argument('--host', default='192.168.0.100')
    parser.add_argument('--user', default='kali')
    parser.add_argument('--pw', default='kali')
    args = parser.parse_args()
    transfer(args.local, args.remote, args.host, args.user, args.pw)
