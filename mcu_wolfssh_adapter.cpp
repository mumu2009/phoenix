/* mcu_wolfssh_adapter.cpp - MCU wolfSSH adapter implementation */

#include "mcu_wolfssh_adapter.hpp"

#include <cstdio>
#include <cstring>

namespace mcu_wolfssh {

bool sshInit() {
    // TODO: wolfSSL_Init() + wolfSSH_Init()
    // 在集成 wolfSSL 后取消下面注释并调用真实 API
    // wolfSSL_Init();
    // wolfSSH_Init();
    return true;
}

void sshCleanup() {
    // wolfSSH_Cleanup();
    // wolfSSL_Cleanup();
}

SshSession sshListen(const SshServerConfig &cfg) {
    SshSession sess;
    sess.socket = mcu_posix::socketCreate(mcu_posix::SocketType::Tcp);
    if (!sess.socket.valid()) {
        return sess;
    }
    if (!mcu_posix::socketBind(sess.socket, cfg.listenPort)) {
        mcu_posix::socketClose(sess.socket);
        return sess;
    }
    // wolfSSH 层负责 listen；这里仅保持 socket 句柄
    sess.active = true;
    return sess;
}

SshSession sshAccept(SshSession &listener) {
    SshSession client;
    if (!listener.active || !listener.socket.valid()) {
        return client;
    }
    client.socket = mcu_posix::socketAccept(listener.socket);
    if (client.socket.valid()) {
        client.active = true;
        // TODO: 创建 WOLFSSH_CTX / WOLFSSH 对象
    }
    return client;
}

bool sshService(SshSession &session) {
    if (!session.active || !session.socket.valid()) {
        return false;
    }
    // TODO: 调用 wolfSSH_accept() 循环处理握手、认证、SFTP 命令
    // 直到连接关闭或出错。
    (void)session;
    return true;
}

void sshClose(SshSession &session) {
    if (session.socket.valid()) {
        mcu_posix::socketClose(session.socket);
    }
    session.active = false;
    // TODO: 释放 WOLFSSH / WOLFSSL 对象
}

} // namespace mcu_wolfssh
