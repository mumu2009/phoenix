#!/usr/bin/env python3
"""Reverse SSH tunnel: Windows -> Kali, forwarding Kali:2222 -> X5:22.

Run this on the Windows host when the Kali VM cannot route to the RDK X5
(192.168.0.107) directly.  After it starts, on Kali use:

    ssh -p 2222 root@127.0.0.1

or set `bpu_evolve_additive.py` / `run_all_additive_training.py` to:

    --x5-host 127.0.0.1 --x5-port 2222 --x5-user root --x5-pass root
"""
import select
import socket
import threading
import paramiko

KALI_HOST = "192.168.0.100"
KALI_USER = "kali"
KALI_PASS = "kali"
X5_HOST = "192.168.0.107"
X5_PORT = 22
TUNNEL_BIND = ("127.0.0.1", 2222)


def handler(chan, host, port):
    sock = socket.socket()
    try:
        sock.connect((host, port))
    except Exception as e:
        print(f"! tunnel connect to {host}:{port} failed: {e}")
        chan.close()
        return

    while True:
        r, w, x = select.select([sock, chan], [], [])
        if sock in r:
            data = sock.recv(1024)
            if len(data) == 0:
                break
            chan.send(data)
        if chan in r:
            data = chan.recv(1024)
            if len(data) == 0:
                break
            sock.send(data)

    chan.close()
    sock.close()


def reverse_port_forward_tunnel(transport):
    transport.request_port_forward(*TUNNEL_BIND)
    print(f"[tunnel] Kali {TUNNEL_BIND[0]}:{TUNNEL_BIND[1]} -> {X5_HOST}:{X5_PORT}")
    while True:
        chan = transport.accept(1000)
        if chan is None:
            continue
        thr = threading.Thread(target=handler, args=(chan, X5_HOST, X5_PORT))
        thr.setDaemon(True)
        thr.start()


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(KALI_HOST, username=KALI_USER, password=KALI_PASS, timeout=30)
    print(f"[tunnel] connected to Kali {KALI_HOST}")
    reverse_port_forward_tunnel(client.get_transport())


if __name__ == "__main__":
    main()
