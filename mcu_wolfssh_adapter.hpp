/* mcu_wolfssh_adapter.hpp - wolfSSH/wolfSSL adapter for MCU
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

#pragma once

/**
 * wolfSSH / wolfSSL adapter (v1.0)
 *
 * Goal: Enable wolfSSH to run on FreeRTOS/lwIP or bare metal Wi-Fi modules,
 * using mcu_posix_compat socket interface for data transmission.
 *
 * Compile with WOLFSSL_USER_SETTINGS defined and provide user_settings.h
 * to trim wolfSSL.
 */

#include <cstdint>
#include <string>

#include "mcu_posix_compat.hpp"

namespace mcu_wolfssh {

/* SSH server configuration */
struct SshServerConfig {
    uint16_t listenPort{2222};              /* Listen port */
    std::string rootPath{"/"};              /* SFTP root directory, mapped to SD card */
    std::string banner{"GD32H759 minimal SSH server"}; /* Server banner */
    bool enablePasswordAuth{false};         /* Prefer public key authentication */
    bool enableSftp{true};                  /* Enable SFTP */
};

/* SSH session */
struct SshSession {
    mcu_posix::SocketHandle socket;         /* Socket handle */
    void *wolfSshCtx{nullptr};              /* WOLFSSH* opaque */
    void *wolfSslCtx{nullptr};              /* WOLFSSL* opaque */
    bool active{false};                     /* Session active flag */
};

/* Initialize wolfSSL/wolfSSH library */
bool sshInit();

/* Cleanup wolfSSH/wolfSSL global resources */
void sshCleanup();

/* Create SSH listening server */
SshSession sshListen(const SshServerConfig &cfg);

/* Accept a client connection (blocking) */
SshSession sshAccept(SshSession &listener);

/* Service an SSH session (handle SFTP/commands) */
/* Usually runs as a separate task in FreeRTOS */
bool sshService(SshSession &session);

/* Close session */
void sshClose(SshSession &session);

} // namespace mcu_wolfssh
