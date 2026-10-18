/* mcu_posix_compat.hpp - MCU POSIX compatibility layer abstraction
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
 * MCU POSIX compatibility layer abstraction (v2.0)
 *
 * Goal: Enable existing C++ code (edge_platform, llama.cpp, etc.) to run on
 * low-cost MCU/SoC platforms (Allwinner T113-S3 / GD32H759 / ESP32-P4) without
 * depending on full Linux.
 *
 * Design principles:
 * 1. Only wrap the POSIX subset actually used: pthread, std::filesystem, socket, munmap/mmap.
 * 2. On musl minimal Linux, most interfaces can pass through directly;
 *    on bare metal/FreeRTOS, reimplement with low-level APIs.
 * 3. Do not introduce full libc, only minimal shim.
 */

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace mcu_posix {

/* Target platform types */
enum class TargetPlatform {
    PiZero2W_FullLinux,     /* Current solution: full glibc Linux */
    GD32H759_FreeRTOS,      /* Final solution: 600MHz Cortex-M7, FreeRTOS + minimal POSIX + SSH, external SDRAM 768MB */
    GD32F427_FreeRTOS,      /* Low-cost alternative: 168MHz Cortex-M4, FreeRTOS + minimal POSIX + SSH, external SDRAM 256MB */
    GD32H759_BareMetal,     /* 600MHz Cortex-M7, pure bare metal, GPIO 150MHz, external SDRAM 768MB */
    GD32F427_BareMetal,     /* 168MHz Cortex-M4, pure bare metal, GPIO ~80MHz, external SDRAM 256MB */
    ESP32P4_FreeRTOS,       /* ESP-IDF FreeRTOS, built-in Wi-Fi 6 / USB / SD */
};

/* SSH stack selection (for FreeRTOS/bare metal targets) */
enum class SshStack {
    None,       /* No SSH */
    WolfSSH,    /* Recommended: pairs with wolfSSL, embedded-friendly */
    TinySSH,    /* Minimal implementation, depends on libsodium/NaCl */
};

/* Thread and synchronization primitives.
   Maps to pthread on musl/full Linux; uses tasks/queues on FreeRTOS/bare metal. */
struct ThreadHandle {
    void *native{nullptr}; /* Native thread handle */
    bool valid() const { return native != nullptr; } /* Check if handle is valid */
};

struct MutexHandle {
    void *native{nullptr}; /* Native mutex handle */
    bool valid() const { return native != nullptr; } /* Check if handle is valid */
};

using ThreadFn = void (*)(void *arg); /* Thread function type */

ThreadHandle threadCreate(ThreadFn fn, void *arg, const char *name = nullptr, std::size_t stackBytes = 8192); /* Create thread */
bool threadJoin(ThreadHandle &h, uint32_t timeoutMs = 0); /* Join thread with timeout */
void threadSleepMs(uint32_t ms); /* Sleep for milliseconds */
void threadYield(); /* Yield CPU */

MutexHandle mutexCreate(); /* Create mutex */
void mutexDestroy(MutexHandle &h); /* Destroy mutex */
void mutexLock(MutexHandle &h); /* Lock mutex */
bool mutexTryLock(MutexHandle &h); /* Try to lock mutex (non-blocking) */
void mutexUnlock(MutexHandle &h); /* Unlock mutex */

/* Filesystem (minimal subset).
   Passes through std::filesystem behavior, but can map to FAT/SD card drivers
   on bare metal/FreeRTOS. */
bool fileExists(const std::string &path); /* Check if file exists */
bool directoryExists(const std::string &path); /* Check if directory exists */
bool createDirectory(const std::string &path); /* Create directory */
bool removeFile(const std::string &path); /* Remove file */
std::vector<std::string> listDirectory(const std::string &path); /* List directory contents */
std::string joinPath(const std::string &a, const std::string &b); /* Join path components */
std::string absolutePath(const std::string &path); /* Get absolute path */

/* Virtual memory (mmap/munmap replacement).
   Provides large weight mapping for llamacpp's ggml; still mmap on Linux,
   changed to DDR/SD card paging manager on MCU. */
struct MappedRegion {
    void *addr{nullptr}; /* Mapped address */
    std::size_t length{0}; /* Mapped length */
    int fd{-1}; /* File descriptor (Linux) or block ID (MCU) */
    bool mappedFromSd{false}; /* Mapped from SD card */
};

MappedRegion mmapFileReadOnly(const std::string &path, std::size_t offset, std::size_t length); /* Map file read-only */
MappedRegion mmapFilePrivate(const std::string &path, std::size_t offset, std::size_t length); /* Map file private (copy-on-write) */
bool munmapRegion(MappedRegion &region); /* Unmap region */

/* Hot page cache: When MCU has no DDR or insufficient DDR, load SD card weights
   in chunks to PSRAM/DDR. */
struct PagingContext {
    std::size_t pageSize{4096}; /* Page size in bytes */
    std::size_t hotPages{64}; /* Number of hot pages to cache */
    std::size_t sdReadAheadPages{4}; /* SD card read-ahead pages */
    bool enableDdrCache{true}; /* Enable DDR cache */
};

bool pagingInit(const PagingContext &ctx); /* Initialize paging system */
void pagingShutdown(); /* Shutdown paging system */
void *pageFaultHandler(std::size_t fileOffset, std::size_t length); /* Handle page fault */

/* Network (socket / SSH).
   Passes through BSD socket on musl; maps to lwip on ESP32; requires external
   WiFi module on bare metal GD32H759. */
struct SocketHandle {
    int fd{-1}; /* Socket file descriptor */
    bool valid() const { return fd >= 0; } /* Check if handle is valid */
};

enum class SocketType { Tcp, Udp }; /* Socket type */

SocketHandle socketCreate(SocketType type); /* Create socket */
bool socketConnect(SocketHandle &h, const std::string &host, uint16_t port, uint32_t timeoutMs = 5000); /* Connect socket */
bool socketBind(SocketHandle &h, uint16_t port); /* Bind socket to port */
SocketHandle socketAccept(SocketHandle &listener); /* Accept incoming connection */
int socketSend(SocketHandle &h, const void *data, std::size_t len); /* Send data */
int socketRecv(SocketHandle &h, void *buf, std::size_t maxLen, uint32_t timeoutMs = 0); /* Receive data */
void socketClose(SocketHandle &h); /* Close socket */

/* GPIO backend selection (set at runtime) */
void setTargetPlatform(TargetPlatform p); /* Set target platform */
TargetPlatform getTargetPlatform(); /* Get current target platform */

} // namespace mcu_posix
