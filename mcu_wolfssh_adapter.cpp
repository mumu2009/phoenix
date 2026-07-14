/* mcu_wolfssh_adapter.cpp - MCU wolfSSH adapter implementation
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   079 Project is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with 079 Project.  If not, see <http://www.gnu.org/licenses/>. */

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
