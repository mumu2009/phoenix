/* mcu_wolfssh_adapter.hpp - wolfSSH/wolfSSL adapter for MCU */

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
