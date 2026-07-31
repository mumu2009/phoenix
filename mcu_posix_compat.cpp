/* mcu_posix_compat.cpp - MCU POSIX compatibility layer implementation
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

#include "mcu_posix_compat.hpp"

#include <cstring>

/* MCU POSIX compatibility layer implementation (FreeRTOS target)
   Dependencies: FreeRTOS kernel, optional FatFs / lwIP / wolfSSH
   Compile with FREERTOS_ENABLED to enable real implementation; otherwise uses host fallback. */

#ifdef FREERTOS_ENABLED

#include "FreeRTOS.h"
#include "task.h"
#include "semphr.h"
#include "queue.h"

namespace mcu_posix {

static TargetPlatform g_platform = TargetPlatform::GD32H759_FreeRTOS;

// ─── 线程 ────────────────────────────────────────────────────────────────────

struct ThreadArgWrapper {
    ThreadFn userFn;
    void *userArg;
};

static void freertosTaskWrapper(void *arg) {
    auto *w = static_cast<ThreadArgWrapper *>(arg);
    ThreadFn fn = w->userFn;
    void *userArg = w->userArg;
    delete w;
    fn(userArg);
    vTaskDelete(nullptr);
}

ThreadHandle threadCreate(ThreadFn fn, void *arg, const char *name, std::size_t stackBytes) {
    ThreadHandle h;
    auto *w = new ThreadArgWrapper{fn, arg};
    TaskHandle_t task = nullptr;
    if (xTaskCreate(freertosTaskWrapper, name ? name : "mcu_task",
                    static_cast<uint16_t>(stackBytes / sizeof(StackType_t)),
                    w, tskIDLE_PRIORITY + 1, &task) != pdPASS) {
        delete w;
        return h;
    }
    h.native = task;
    return h;
}

bool threadJoin(ThreadHandle &h, uint32_t timeoutMs) {
    if (!h.valid()) return false;
    TickType_t ticks = (timeoutMs == 0) ? portMAX_DELAY : pdMS_TO_TICKS(timeoutMs);
    // FreeRTOS 中任务通常自删除；此处仅阻塞指定时间
    vTaskDelay(ticks);
    h.native = nullptr;
    return true;
}

void threadSleepMs(uint32_t ms) {
    vTaskDelay(pdMS_TO_TICKS(ms));
}

void threadYield() {
    taskYIELD();
}

// ─── 互斥 ────────────────────────────────────────────────────────────────────

MutexHandle mutexCreate() {
    MutexHandle h;
    SemaphoreHandle_t sem = xSemaphoreCreateMutex();
    h.native = sem;
    return h;
}

void mutexDestroy(MutexHandle &h) {
    if (h.valid()) {
        vSemaphoreDelete(static_cast<SemaphoreHandle_t>(h.native));
        h.native = nullptr;
    }
}

void mutexLock(MutexHandle &h) {
    if (h.valid()) {
        xSemaphoreTake(static_cast<SemaphoreHandle_t>(h.native), portMAX_DELAY);
    }
}

bool mutexTryLock(MutexHandle &h) {
    if (!h.valid()) return false;
    return xSemaphoreTake(static_cast<SemaphoreHandle_t>(h.native), 0) == pdTRUE;
}

void mutexUnlock(MutexHandle &h) {
    if (h.valid()) {
        xSemaphoreGive(static_cast<SemaphoreHandle_t>(h.native));
    }
}

// ─── 文件系统（FatFs 最小封装）────────────────────────────────────────────────
// 若未启用 FatFs，使用内存 fallback。

#if defined(FATFS_ENABLED)
#include "ff.h"

bool fileExists(const std::string &path) {
    FILINFO fi;
    return f_stat(path.c_str(), &fi) == FR_OK;
}

bool directoryExists(const std::string &path) {
    FILINFO fi;
    return f_stat(path.c_str(), &fi) == FR_OK && (fi.fattrib & AM_DIR);
}

bool createDirectory(const std::string &path) {
    return f_mkdir(path.c_str()) == FR_OK;
}

bool removeFile(const std::string &path) {
    return f_unlink(path.c_str()) == FR_OK;
}

std::vector<std::string> listDirectory(const std::string &path) {
    std::vector<std::string> out;
    DIR dir;
    FILINFO fi;
    if (f_opendir(&dir, path.c_str()) == FR_OK) {
        while (f_readdir(&dir, &fi) == FR_OK && fi.fname[0]) {
            out.emplace_back(fi.fname);
        }
        f_closedir(&dir);
    }
    return out;
}
#else
bool fileExists(const std::string &) { return false; }
bool directoryExists(const std::string &) { return false; }
bool createDirectory(const std::string &) { return false; }
bool removeFile(const std::string &) { return false; }
std::vector<std::string> listDirectory(const std::string &) { return {}; }
#endif

std::string joinPath(const std::string &a, const std::string &b) {
    if (a.empty()) return b;
    if (b.empty()) return a;
    if (a.back() == '/' || a.back() == '\\') return a + b;
    return a + '/' + b;
}

std::string absolutePath(const std::string &path) {
    // 在 MCU 中没有进程当前目录概念；直接返回原路径
    return path;
}

// ─── 虚拟内存 / mmap 最小实现 ────────────────────────────────────────────────
// 在 768MB SDRAM 方案中，模型权重可直接加载到 SDRAM，无需换页。

static bool g_pagingInit = false;
static PagingContext g_pagingCtx;

MappedRegion mmapFileReadOnly(const std::string &path, std::size_t offset, std::size_t length) {
    MappedRegion r;
    r.length = length;
    r.mappedFromSd = true;
    // 需要 FatFs 集成以从 SD 卡读取指定偏移到 SDRAM；当前为 fallback。
    (void)path;
    (void)offset;
    return r;
}

MappedRegion mmapFilePrivate(const std::string &path, std::size_t offset, std::size_t length) {
    return mmapFileReadOnly(path, offset, length);
}

bool munmapRegion(MappedRegion &region) {
    if (region.addr && region.length > 0) {
        // SDRAM 为物理内存，无需释放；标记即可
        region.addr = nullptr;
        region.length = 0;
    }
    return true;
}

bool pagingInit(const PagingContext &ctx) {
    g_pagingCtx = ctx;
    g_pagingInit = true;
    return true;
}

void pagingShutdown() {
    g_pagingInit = false;
}

void *pageFaultHandler(std::size_t fileOffset, std::size_t length) {
    (void)fileOffset;
    (void)length;
    if (!g_pagingInit) return nullptr;
    // 需要 FatFs 集成以从 SD 卡加载一页到 SDRAM 热缓存；当前为 fallback。
    return nullptr;
}

// ─── 网络（lwIP 最小封装）────────────────────────────────────────────────────
#if defined(LWIP_ENABLED)
#include "lwip/sockets.h"
#include "lwip/netdb.h"

SocketHandle socketCreate(SocketType type) {
    int domain = AF_INET;
    int sockType = (type == SocketType::Udp) ? SOCK_DGRAM : SOCK_STREAM;
    int fd = lwip_socket(domain, sockType, 0);
    return SocketHandle{fd};
}

bool socketConnect(SocketHandle &h, const std::string &host, uint16_t port, uint32_t timeoutMs) {
    if (!h.valid()) return false;
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    ipaddr_aton(host.c_str(), reinterpret_cast<ip_addr_t *>(&addr.sin_addr));
    (void)timeoutMs; // lwIP 阻塞 connect 由超时控制
    return lwip_connect(h.fd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) == 0;
}

bool socketBind(SocketHandle &h, uint16_t port) {
    if (!h.valid()) return false;
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;
    return lwip_bind(h.fd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) == 0;
}

SocketHandle socketAccept(SocketHandle &listener) {
    if (!listener.valid()) return SocketHandle{};
    struct sockaddr_in addr;
    socklen_t len = sizeof(addr);
    int fd = lwip_accept(listener.fd, reinterpret_cast<struct sockaddr *>(&addr), &len);
    return SocketHandle{fd};
}

int socketSend(SocketHandle &h, const void *data, std::size_t len) {
    if (!h.valid()) return -1;
    return lwip_send(h.fd, data, len, 0);
}

int socketRecv(SocketHandle &h, void *buf, std::size_t maxLen, uint32_t timeoutMs) {
    if (!h.valid()) return -1;
    (void)timeoutMs;
    return lwip_recv(h.fd, buf, maxLen, 0);
}

void socketClose(SocketHandle &h) {
    if (h.valid()) {
        lwip_close(h.fd);
        h.fd = -1;
    }
}
#else
SocketHandle socketCreate(SocketType) { return SocketHandle{-1}; }
bool socketConnect(SocketHandle &, const std::string &, uint16_t, uint32_t) { return false; }
bool socketBind(SocketHandle &, uint16_t) { return false; }
SocketHandle socketAccept(SocketHandle &) { return SocketHandle{}; }
int socketSend(SocketHandle &, const void *, std::size_t) { return -1; }
int socketRecv(SocketHandle &, void *, std::size_t, uint32_t) { return -1; }
void socketClose(SocketHandle &) {}
#endif

// ─── 平台选择 ───────────────────────────────────────────────────────────────

void setTargetPlatform(TargetPlatform p) { g_platform = p; }
TargetPlatform getTargetPlatform() { return g_platform; }

} // namespace mcu_posix

#else // !FREERTOS_ENABLED

// Host 最小 fallback：不依赖 POSIX 头，用于在 Windows / 任意 host 编译测试。
// 真实功能在 FreeRTOS 目标下启用。

#include <chrono>
#include <thread>
#include <mutex>

namespace mcu_posix {

static TargetPlatform g_platform = TargetPlatform::GD32H759_FreeRTOS;

struct ThreadArgWrapper {
    ThreadFn userFn;
    void *userArg;
};

ThreadHandle threadCreate(ThreadFn fn, void *arg, const char *name, std::size_t stackBytes) {
    ThreadHandle h;
    auto *w = new ThreadArgWrapper{fn, arg};
    (void)name;
    (void)stackBytes;
    // 使用 std::thread 作为 host 测试替身；真实目标使用 FreeRTOS 任务
    auto *t = new std::thread([w]() {
        w->userFn(w->userArg);
        delete w;
    });
    h.native = t;
    return h;
}

bool threadJoin(ThreadHandle &h, uint32_t timeoutMs) {
    if (!h.valid()) return false;
    (void)timeoutMs;
    auto *t = static_cast<std::thread *>(h.native);
    if (t->joinable()) t->join();
    delete t;
    h.native = nullptr;
    return true;
}

void threadSleepMs(uint32_t ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

void threadYield() {
    std::this_thread::yield();
}

MutexHandle mutexCreate() {
    MutexHandle h;
    h.native = new std::mutex();
    return h;
}

void mutexDestroy(MutexHandle &h) {
    if (h.valid()) {
        delete static_cast<std::mutex *>(h.native);
        h.native = nullptr;
    }
}

void mutexLock(MutexHandle &h) {
    if (h.valid()) static_cast<std::mutex *>(h.native)->lock();
}

bool mutexTryLock(MutexHandle &h) {
    if (!h.valid()) return false;
    return static_cast<std::mutex *>(h.native)->try_lock();
}

void mutexUnlock(MutexHandle &h) {
    if (h.valid()) static_cast<std::mutex *>(h.native)->unlock();
}

bool fileExists(const std::string &) { return false; }
bool directoryExists(const std::string &) { return false; }
bool createDirectory(const std::string &) { return false; }
bool removeFile(const std::string &) { return false; }
std::vector<std::string> listDirectory(const std::string &) { return {}; }

std::string joinPath(const std::string &a, const std::string &b) {
    if (a.empty()) return b;
    if (b.empty()) return a;
    if (a.back() == '/' || a.back() == '\\') return a + b;
    return a + '/' + b;
}

std::string absolutePath(const std::string &path) { return path; }

MappedRegion mmapFileReadOnly(const std::string &, std::size_t, std::size_t) { return {}; }
MappedRegion mmapFilePrivate(const std::string &, std::size_t, std::size_t) { return {}; }
bool munmapRegion(MappedRegion &) { return true; }
bool pagingInit(const PagingContext &) { return true; }
void pagingShutdown() {}
void *pageFaultHandler(std::size_t, std::size_t) { return nullptr; }

SocketHandle socketCreate(SocketType) { return SocketHandle{-1}; }
bool socketConnect(SocketHandle &, const std::string &, uint16_t, uint32_t) { return false; }
bool socketBind(SocketHandle &, uint16_t) { return false; }
SocketHandle socketAccept(SocketHandle &) { return SocketHandle{}; }
int socketSend(SocketHandle &, const void *, std::size_t) { return -1; }
int socketRecv(SocketHandle &, void *, std::size_t, uint32_t) { return -1; }
void socketClose(SocketHandle &) {}

void setTargetPlatform(TargetPlatform p) { g_platform = p; }
TargetPlatform getTargetPlatform() { return g_platform; }

} // namespace mcu_posix

#endif
