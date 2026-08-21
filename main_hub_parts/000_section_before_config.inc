/* Auto-generated from main.cpp
   Segment: section_before_Config
   Original lines: 1-990
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
#include <json/json.h>
#include <jwt-cpp/jwt.h>
#include <jwt-cpp/traits/nlohmann-json/traits.h>
#include <nlohmann/json.hpp>
#include <trantor/utils/Utilities.h>

#ifndef HAVE_LMDB
#define HAVE_LMDB 1
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <optional>
#include <queue>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../phoenix_config.hpp"
#include "../runtime_tuned_config.hpp"
#include "../version.hpp"

#include "../world_model.hpp"
#include "../DATABASE_079.hpp"
#include "../addon.hpp"
#include "../emergency_stop.hpp"
#include "../instance_registry.hpp"
#include "../addons/cli_json_addon.hpp"
#include "../autonomy_stack.hpp"
#include "../frontend_server.hpp"
#include "../loggerCXXH.hpp"
#include "../model_lifecycle.hpp"
#include "../module_mount.hpp"
#include "../transformer.hpp"
// transformer_main.cpp is only used as last resort; normally we use llamacpp
// #include "../transformer_main.cpp"

// New system includes
#include "../emotion_system.hpp"
#include "../llamacpp_emotion_adjuster.hpp"
#include "../plugin_system.hpp"
#include "../modern_context_system.hpp"
#include "../graph_diffusion_summarizer.hpp"
#include "../hierarchical_memory.hpp"
#include "../model_deployment.hpp"
#include "../mission_workspace.hpp"
#include "../context_window_pack.hpp"

#ifndef AI_OUTSIDES_LINK_MODE
#define AI_OUTSIDES_LINK_MODE "headers-only"
#endif

extern "C" {
int transformer_ollama_ft_bridge_set_module(const char *module_name,
                                            char *error_buf, size_t error_cap);
int transformer_ollama_ft_bridge_init(char *error_buf, size_t error_cap);
int transformer_ollama_ft_bridge_run_main(int argc, const char *const *argv,
                                          char *error_buf, size_t error_cap);
int transformer_ollama_ft_bridge_train(
    const char *append_corpus, const char *output_dir, const char *ollama_model,
    const char *hf_model, int self_play_pairs, int epochs, double lr,
    char *error_buf, size_t error_cap);
void transformer_ollama_ft_bridge_finalize(void);
}

#ifdef _WIN32
#include <psapi.h>
#include <windows.h>
#include <tlhelp32.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "psapi.lib")
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#ifdef HAVE_REDIS
#include <sw/redis++/redis++.h>
#endif

#ifdef HAVE_LMDB
#include <lmdb.h>
#endif

#ifdef HAVE_GUMBO
#include <gumbo.h>
#endif

#ifdef HAVE_POPPLER
#include <poppler-document.h>
#include <poppler-page.h>
#endif

#ifdef HAVE_CURL
#include <curl/curl.h>
#endif

#ifdef HAVE_TORCH
#include <torch/torch.h>
#endif

using json = nlohmann::json;
namespace fs = std::filesystem;

// Memory management utilities to prevent OOM kills during heavy inference.
#ifdef _WIN32
static size_t getAvailablePhysicalMemoryMB() {
  MEMORYSTATUSEX ms{};
  ms.dwLength = sizeof(ms);
  if (GlobalMemoryStatusEx(&ms))
    return (size_t)(ms.ullAvailPhys / (1024ULL * 1024ULL));
  return 0;
}

static size_t getProcessWorkingSetMB() {
  PROCESS_MEMORY_COUNTERS pmc{};
  pmc.cb = sizeof(pmc);
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
    return (size_t)(pmc.WorkingSetSize / (1024ULL * 1024ULL));
  return 0;
}

static void trimProcessWorkingSet() {
  SetProcessWorkingSetSize(GetCurrentProcess(), (SIZE_T)-1, (SIZE_T)-1);
}

// Returns true if there's enough memory to proceed with inference.
// Threshold: need at least 800MB free system RAM.
static bool preInferenceMemoryCheck(std::string &errorOut) {
  size_t availMB = getAvailablePhysicalMemoryMB();
  size_t wsMB = getProcessWorkingSetMB();
  if (availMB < 800) {
    // Try trimming our working set first to free pages back to OS
    trimProcessWorkingSet();
    availMB = getAvailablePhysicalMemoryMB();
    if (availMB < 400) {
      errorOut = "Insufficient system memory for inference (available=" +
                 std::to_string(availMB) + "MB, process_ws=" +
                 std::to_string(wsMB) + "MB). Please free memory and retry.";
      return false;
    }
  }
  return true;
}

static void postInferenceMemoryTrim() {
  trimProcessWorkingSet();
}
#else
static bool preInferenceMemoryCheck(std::string &) { return true; }
static void postInferenceMemoryTrim() {}
#endif

// 将 JsonCpp 的 Json::Value 递归转换为 nlohmann::json。
// 调用方式：当外部模块返回 JsonCpp 对象时通过本函数桥接。
// 实现思路：按 ValueType 分支递归处理对象与数组。
// 注意事项：未知类型会回退为 null，调用方需做容错。
static json fromJsoncpp(const Json::Value &v) {
  switch (v.type()) {
  case Json::nullValue:
    return nullptr;
  case Json::intValue:
    return (int64_t)v.asInt64();
  case Json::uintValue:
    return (uint64_t)v.asUInt64();
  case Json::realValue:
    return v.asDouble();
  case Json::stringValue:
    return v.asString();
  case Json::booleanValue:
    return v.asBool();
  case Json::arrayValue: {
    json out = json::array();
    for (const auto &item : v)
      out.push_back(fromJsoncpp(item));
    return out;
  }
  case Json::objectValue: {
    json out = json::object();
    for (auto it = v.begin(); it != v.end(); ++it) {
      out[it.name()] = fromJsoncpp(*it);
    }
    return out;
  }
  default:
    return nullptr;
  }
}

// 将 nlohmann::json 递归转换为 JsonCpp 的 Json::Value。
// 调用方式：需要与 JsonCpp API 交互时调用。
// 实现思路：按 json 类型分派并递归构建数组/对象。
// 注意事项：数值类型会按整型/浮点分支转换。
static Json::Value toJsoncpp(const json &v) {
  if (v.is_null())
    return Json::Value();
  if (v.is_boolean())
    return Json::Value(v.get<bool>());
  if (v.is_number_integer())
    return Json::Value((Json::Int64)v.get<long long>());
  if (v.is_number_unsigned())
    return Json::Value((Json::UInt64)v.get<unsigned long long>());
  if (v.is_number_float())
    return Json::Value(v.get<double>());
  if (v.is_string())
    return Json::Value(v.get<std::string>());
  if (v.is_array()) {
    Json::Value arr(Json::arrayValue);
    for (const auto &item : v)
      arr.append(toJsoncpp(item));
    return arr;
  }
  if (v.is_object()) {
    Json::Value obj(Json::objectValue);
    for (auto it = v.begin(); it != v.end(); ++it)
      obj[it.key()] = toJsoncpp(it.value());
    return obj;
  }
  return Json::Value();
}

static std::string g_selfPath;

static std::mutex gFrontendProcMu;
static std::mutex gPidFileMu;
static fs::path gPidFilePath = fs::path("runtime_store") / "phoenix_main.pid";
static bool gPidFileArmed = false;
#ifdef _WIN32
static HANDLE gFrontendProcessHandle = nullptr;
static DWORD gFrontendProcessId = 0;
static HANDLE gFrontendJobHandle = nullptr;
static HANDLE gPidFileHandle = INVALID_HANDLE_VALUE;
static std::atomic<bool> gCtrlForceExitArmed{false};
static std::atomic<DWORD> gCtrlForceExitMs{12000};
static bool gDisableWatchdog{false};

static DWORD WINAPI forceExitWatchdogThread(LPVOID) noexcept {
  DWORD waitMs = gCtrlForceExitMs.load(std::memory_order_relaxed);
  if (waitMs < 1000)
    waitMs = 1000;
  if (waitMs > 600000)
    waitMs = 600000;
  Sleep(waitMs);
  TerminateProcess(GetCurrentProcess(), 130);
  return 0;
}

static void armCtrlForceExitWatchdog() noexcept {
  if (gDisableWatchdog)
    return;
  bool expected = false;
  if (!gCtrlForceExitArmed.compare_exchange_strong(expected, true))
    return;
  HANDLE h =
      CreateThread(nullptr, 0, forceExitWatchdogThread, nullptr, 0, nullptr);
  if (h)
    CloseHandle(h);
}

static void ensureFrontendJobObjectLocked() noexcept {
  if (gFrontendJobHandle)
    return;
  HANDLE job = CreateJobObjectA(nullptr, nullptr);
  if (!job)
    return;
  JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};
  info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
  if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation, &info,
                               sizeof(info))) {
    CloseHandle(job);
    return;
  }
  gFrontendJobHandle = job;
}
#else
static pid_t gFrontendProcessPid = -1;
#endif

static void appendFatalLog(const std::string &line);

static unsigned long long currentProcessId() noexcept {
#ifdef _WIN32
  return static_cast<unsigned long long>(GetCurrentProcessId());
#else
  return static_cast<unsigned long long>(getpid());
#endif
}

static bool processIsRunning(unsigned long long pid) noexcept {
  if (pid == 0)
    return false;
#ifdef _WIN32
  HANDLE h = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, FALSE,
                         static_cast<DWORD>(pid));
  if (!h)
    return false;
  DWORD waitCode = WaitForSingleObject(h, 0);
  CloseHandle(h);
  return waitCode == WAIT_TIMEOUT;
#else
  return kill(static_cast<pid_t>(pid), 0) == 0;
#endif
}

static unsigned long long readPidFileValue(const fs::path &pidPath) noexcept {
  try {
    std::ifstream in(pidPath);
    unsigned long long pid = 0;
    in >> pid;
    return pid;
  } catch (...) {
    return 0;
  }
}

static void cleanupPidFile() noexcept {
  std::lock_guard<std::mutex> lock(gPidFileMu);
  if (!gPidFileArmed)
    return;
#ifdef _WIN32
  if (gPidFileHandle != INVALID_HANDLE_VALUE) {
    CloseHandle(gPidFileHandle);
    gPidFileHandle = INVALID_HANDLE_VALUE;
  }
#endif
  std::error_code ec;
  fs::remove(gPidFilePath, ec);
  gPidFileArmed = false;
}

static bool acquirePidFile(const fs::path &pidPath, std::string &errorOut) {
  std::lock_guard<std::mutex> lock(gPidFileMu);
  gPidFilePath = pidPath;
  std::error_code ec;
  fs::create_directories(gPidFilePath.parent_path(), ec);

#ifdef _WIN32
  auto tryCreate = [&]() -> bool {
    gPidFileHandle = CreateFileW(
        gPidFilePath.wstring().c_str(), GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    return gPidFileHandle != INVALID_HANDLE_VALUE;
  };

  if (!tryCreate()) {
    DWORD code = GetLastError();
    if (code == ERROR_FILE_EXISTS || code == ERROR_ALREADY_EXISTS) {
      unsigned long long existingPid = readPidFileValue(gPidFilePath);
      if (processIsRunning(existingPid)) {
        errorOut = "pid file exists and process is running, pid=" +
                   std::to_string(existingPid) +
                   " file=" + gPidFilePath.string();
        return false;
      }
      fs::remove(gPidFilePath, ec);
      if (!tryCreate()) {
        errorOut = "pid file create failed after stale cleanup, file=" +
                   gPidFilePath.string();
        return false;
      }
    } else {
      errorOut = "pid file create failed, code=" +
                 std::to_string(static_cast<unsigned long long>(code)) +
                 " file=" + gPidFilePath.string();
      return false;
    }
  }

  std::string body = std::to_string(currentProcessId()) + "\n";
  DWORD written = 0;
  BOOL ok = WriteFile(gPidFileHandle, body.data(),
                      static_cast<DWORD>(body.size()), &written, nullptr);
  FlushFileBuffers(gPidFileHandle);
  if (!ok || written != body.size()) {
    CloseHandle(gPidFileHandle);
    gPidFileHandle = INVALID_HANDLE_VALUE;
    fs::remove(gPidFilePath, ec);
    errorOut = "pid file write failed, file=" + gPidFilePath.string();
    return false;
  }
  gPidFileArmed = true;
  return true;
#else
  int fd =
      ::open(gPidFilePath.string().c_str(), O_CREAT | O_EXCL | O_WRONLY, 0644);
  if (fd < 0) {
    unsigned long long existingPid = readPidFileValue(gPidFilePath);
    if (processIsRunning(existingPid)) {
      errorOut = "pid file exists and process is running, pid=" +
                 std::to_string(existingPid) + " file=" + gPidFilePath.string();
      return false;
    }
    fs::remove(gPidFilePath, ec);
    fd = ::open(gPidFilePath.string().c_str(), O_CREAT | O_EXCL | O_WRONLY,
                0644);
    if (fd < 0) {
      errorOut = "pid file create failed, file=" + gPidFilePath.string();
      return false;
    }
  }
  std::string body = std::to_string(currentProcessId()) + "\n";
  ssize_t n = ::write(fd, body.data(), body.size());
  ::close(fd);
  if (n != static_cast<ssize_t>(body.size())) {
    fs::remove(gPidFilePath, ec);
    errorOut = "pid file write failed, file=" + gPidFilePath.string();
    return false;
  }
  gPidFileArmed = true;
  return true;
#endif
}

static void cleanupFrontendProcess() noexcept {
  std::lock_guard<std::mutex> lock(gFrontendProcMu);
#ifdef _WIN32
  if (gFrontendProcessHandle) {
    DWORD waitCode = WaitForSingleObject(gFrontendProcessHandle, 0);
    if (waitCode == WAIT_TIMEOUT) {
      TerminateProcess(gFrontendProcessHandle, 0);
      WaitForSingleObject(gFrontendProcessHandle, 1500);
    }
    CloseHandle(gFrontendProcessHandle);
    gFrontendProcessHandle = nullptr;
    gFrontendProcessId = 0;
  }
  if (gFrontendJobHandle) {
    CloseHandle(gFrontendJobHandle);
    gFrontendJobHandle = nullptr;
  }
#else
  if (gFrontendProcessPid > 0) {
    kill(gFrontendProcessPid, SIGTERM);
    gFrontendProcessPid = -1;
  }
#endif
}

#ifdef _WIN32
static BOOL WINAPI onConsoleControl(DWORD ctrlType) {
  switch (ctrlType) {
  case CTRL_C_EVENT:
  case CTRL_BREAK_EVENT:
    // Suppress CTRL+C/BREAK to prevent accidental termination of the server.
    // The server should only be stopped via explicit shutdown commands or CTRL_CLOSE.
    appendFatalLog(std::string("console control event suppressed type=") + std::to_string(ctrlType));
    return TRUE;
  case CTRL_CLOSE_EVENT:
  case CTRL_LOGOFF_EVENT:
  case CTRL_SHUTDOWN_EVENT:
    appendFatalLog(std::string("console close/shutdown event type=") + std::to_string(ctrlType));
    armCtrlForceExitWatchdog();
    cleanupFrontendProcess();
    cleanupPidFile();
    return TRUE;
  default:
    return FALSE;
  }
}

// Forward declaration for the minidump writer defined below.
namespace crash_dump {
bool writeMinidump(EXCEPTION_POINTERS *ep, const fs::path &dumpPath);
}

static LONG WINAPI onUnhandledException(EXCEPTION_POINTERS *exceptionInfo) {
  try {
    DWORD code = exceptionInfo && exceptionInfo->ExceptionRecord
                     ? exceptionInfo->ExceptionRecord->ExceptionCode
                     : 0;
    std::ostringstream oss;
    oss << "windows unhandled exception code=0x" << std::hex
        << (unsigned long)code;
    if (exceptionInfo && exceptionInfo->ExceptionRecord) {
      auto *rec = exceptionInfo->ExceptionRecord;
      oss << " addr=0x" << std::hex << (uintptr_t)rec->ExceptionAddress;
      if (code == 0xC0000005 && rec->NumberParameters >= 2) {
        oss << " access=" << (rec->ExceptionInformation[0] == 0 ? "read" : "write")
            << " target=0x" << std::hex << (uintptr_t)rec->ExceptionInformation[1];
      }
    }
    if (exceptionInfo && exceptionInfo->ContextRecord) {
#if defined(__x86_64__) || defined(_M_X64)
      oss << " rip=0x" << std::hex << exceptionInfo->ContextRecord->Rip
          << " rsp=0x" << exceptionInfo->ContextRecord->Rsp;
#elif defined(__i386__) || defined(_M_IX86)
      oss << " eip=0x" << std::hex << exceptionInfo->ContextRecord->Eip
          << " esp=0x" << exceptionInfo->ContextRecord->Esp;
#endif
    }
    HMODULE hMod = GetModuleHandleA(nullptr);
    uintptr_t base = (uintptr_t)hMod;
    oss << " base=0x" << std::hex << base;
    if (exceptionInfo && exceptionInfo->ExceptionRecord) {
      uintptr_t offset = (uintptr_t)exceptionInfo->ExceptionRecord->ExceptionAddress - base;
      oss << " offset=0x" << std::hex << offset;
    }
    oss << std::dec << " mem_avail_mb=" << getAvailablePhysicalMemoryMB()
        << " process_ws_mb=" << getProcessWorkingSetMB();
    appendFatalLog(oss.str());

    // Write a Windows minidump for post-mortem analysis.  The dump is
    // intentionally written without linking against dbghelp.lib (runtime load).
    try {
      fs::path dumpPath = fs::path("runtime_store") / "logs" / "crash.dmp";
      if (crash_dump::writeMinidump(exceptionInfo, dumpPath)) {
        appendFatalLog(std::string("wrote crash minidump to ") +
                       dumpPath.string());
      } else {
        appendFatalLog("failed to write crash minidump");
      }
    } catch (...) {
      appendFatalLog("exception while writing crash minidump");
    }
  } catch (...) {
  }
  return EXCEPTION_EXECUTE_HANDLER;
}
#endif

static std::mutex gFatalLogMu;
static fs::path gFatalLogPath =
    fs::path("runtime_store") / "logs" / "fatal_runtime.log";

static void appendFatalLog(const std::string &line) {
  try {
    std::lock_guard<std::mutex> lock(gFatalLogMu);
    fs::create_directories(gFatalLogPath.parent_path());
    std::ofstream out(gFatalLogPath, std::ios::app);
    if (!out)
      return;
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    out << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S") << " | "
        << line << "\n";
  } catch (...) {
  }
}

static void onFatalTerminate() {
  std::string msg = "std::terminate";
  try {
    auto ep = std::current_exception();
    if (ep) {
      try {
        std::rethrow_exception(ep);
      } catch (const std::exception &e) {
        msg += std::string(": ") + e.what();
      } catch (...) {
        msg += ": non-std exception";
      }
    }
  } catch (...) {
  }
  appendFatalLog(msg);
  cleanupPidFile();
  std::_Exit(134);
}

static void onFatalSignal(int signum) {
  appendFatalLog(std::string("fatal signal=") + std::to_string(signum));
  cleanupPidFile();
  std::_Exit(128 + signum);
}

static void installFatalHooks(const fs::path &baseDir) {
  try {
    gFatalLogPath = baseDir / "logs" / "fatal_runtime.log";
  } catch (...) {
  }
  std::set_terminate(onFatalTerminate);
  std::signal(SIGABRT, onFatalSignal);
  std::signal(SIGSEGV, onFatalSignal);
  std::signal(SIGILL, onFatalSignal);
  std::signal(SIGFPE, onFatalSignal);
  std::signal(SIGTERM, onFatalSignal);
}

#ifdef _WIN32
namespace crash_dump {

typedef enum _PHX_MINIDUMP_TYPE {
  PhxMiniDumpNormal = 0x00000000,
  PhxMiniDumpWithDataSegs = 0x00000001,
} PHX_MINIDUMP_TYPE;

typedef struct _PHX_MINIDUMP_EXCEPTION_INFORMATION {
  DWORD ThreadId;
  PEXCEPTION_POINTERS ExceptionPointers;
  BOOL ClientPointers;
} PHX_MINIDUMP_EXCEPTION_INFORMATION, *PPHX_MINIDUMP_EXCEPTION_INFORMATION;

typedef BOOL(WINAPI *MiniDumpWriteDumpFn)(
    HANDLE hProcess, DWORD ProcessId, HANDLE hFile, PHX_MINIDUMP_TYPE DumpType,
    PPHX_MINIDUMP_EXCEPTION_INFORMATION ExceptionParam, PVOID UserStreamParam,
    PVOID CallbackParam);

bool writeMinidump(EXCEPTION_POINTERS *ep, const fs::path &dumpPath) {
  HMODULE hDbgHelp = LoadLibraryA("dbghelp.dll");
  if (!hDbgHelp) return false;
  auto fn = (MiniDumpWriteDumpFn)GetProcAddress(hDbgHelp, "MiniDumpWriteDump");
  if (!fn) {
    FreeLibrary(hDbgHelp);
    return false;
  }
  try {
    fs::create_directories(dumpPath.parent_path());
  } catch (...) {
  }
  HANDLE hFile =
      CreateFileA(dumpPath.string().c_str(), GENERIC_WRITE, 0, nullptr,
                  CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (hFile == INVALID_HANDLE_VALUE) {
    FreeLibrary(hDbgHelp);
    return false;
  }
  PHX_MINIDUMP_EXCEPTION_INFORMATION mei{};
  mei.ThreadId = GetCurrentThreadId();
  mei.ExceptionPointers = ep;
  mei.ClientPointers = FALSE;
  BOOL ok = fn(GetCurrentProcess(), GetCurrentProcessId(), hFile,
               PhxMiniDumpWithDataSegs, &mei, nullptr, nullptr);
  CloseHandle(hFile);
  FreeLibrary(hDbgHelp);
  return ok == TRUE;
}

} // namespace crash_dump

static bool readProcessMemoryCounters(DWORD pid, size_t &wsMb, size_t &privMb) {
  wsMb = 0;
  privMb = 0;
  HANDLE hProc = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, pid);
  if (!hProc) return false;
  PROCESS_MEMORY_COUNTERS_EX pmc{};
  pmc.cb = sizeof(pmc);
  bool ok = false;
  if (GetProcessMemoryInfo(hProc, (PROCESS_MEMORY_COUNTERS *)&pmc, sizeof(pmc))) {
    wsMb = (size_t)(pmc.WorkingSetSize / (1024ULL * 1024ULL));
    privMb = (size_t)(pmc.PrivateUsage / (1024ULL * 1024ULL));
    ok = true;
  }
  CloseHandle(hProc);
  return ok;
}

static bool findProcessByName(const std::string &name, DWORD &outPid,
                              size_t &wsMb, size_t &privMb) {
  outPid = 0;
  wsMb = 0;
  privMb = 0;
  HANDLE hSnap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
  if (hSnap == INVALID_HANDLE_VALUE) return false;
  PROCESSENTRY32 pe{};
  pe.dwSize = sizeof(pe);
  bool found = false;
  if (Process32First(hSnap, &pe)) {
    do {
      std::string exe(pe.szExeFile);
      if (exe == name) {
        size_t ws = 0, priv = 0;
        if (readProcessMemoryCounters(pe.th32ProcessID, ws, priv)) {
          if (!found || ws > wsMb) {
            outPid = pe.th32ProcessID;
            wsMb = ws;
            privMb = priv;
            found = true;
          }
        }
      }
    } while (Process32Next(hSnap, &pe));
  }
  CloseHandle(hSnap);
  return found;
}

static std::atomic<bool> gMemoryTelemetryRunning{false};

static void memoryTelemetryLoop(const fs::path &logDir) {
  fs::path logPath = logDir / "memory_telemetry.log";
  try {
    fs::create_directories(logPath.parent_path());
  } catch (...) {
  }
  while (gMemoryTelemetryRunning.load()) {
    try {
      size_t availMb = getAvailablePhysicalMemoryMB();
      size_t selfWs = getProcessWorkingSetMB();
      size_t selfPriv = 0;
      size_t llamaWs = 0, llamaPriv = 0;
      size_t frontWs = 0, frontPriv = 0;
      DWORD llamaPid = 0, frontPid = 0;
      PROCESS_MEMORY_COUNTERS_EX pmc{};
      pmc.cb = sizeof(pmc);
      if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&pmc,
                               sizeof(pmc))) {
        selfPriv = (size_t)(pmc.PrivateUsage / (1024ULL * 1024ULL));
      }
      findProcessByName("llama-server.exe", llamaPid, llamaWs, llamaPriv);
      // The frontend child is also phoenix_main.exe.  Pick the one with the
      // largest working set that is NOT the current process.
      DWORD selfPid = GetCurrentProcessId();
      HANDLE hSnap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
      if (hSnap != INVALID_HANDLE_VALUE) {
        PROCESSENTRY32 pe{};
        pe.dwSize = sizeof(pe);
        if (Process32First(hSnap, &pe)) {
          do {
            if (pe.th32ProcessID != selfPid &&
                std::string(pe.szExeFile) == "phoenix_main.exe") {
              size_t ws = 0, priv = 0;
              if (readProcessMemoryCounters(pe.th32ProcessID, ws, priv)) {
                if (ws > frontWs) {
                  frontPid = pe.th32ProcessID;
                  frontWs = ws;
                  frontPriv = priv;
                }
              }
            }
          } while (Process32Next(hSnap, &pe));
        }
        CloseHandle(hSnap);
      }
      std::ofstream out(logPath.string(), std::ios::app | std::ios::out);
      if (!out) {
        std::cerr << "[memoryTelemetry] cannot open " << logPath.string()
                  << std::endl;
      } else {
        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        std::tm *tmBuf = std::localtime(&t);
        if (tmBuf) {
          char buf[64];
          if (std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", tmBuf))
            out << buf;
        }
        out << " "
            << "avail=" << availMb
            << " self_ws=" << selfWs << " self_priv=" << selfPriv
            << " llama_pid=" << llamaPid << " llama_ws=" << llamaWs
            << " llama_priv=" << llamaPriv
            << " front_pid=" << frontPid << " front_ws=" << frontWs
            << " front_priv=" << frontPriv << "\n";
      }
    } catch (...) {
    }
    std::this_thread::sleep_for(std::chrono::seconds(30));
  }
}

static void startMemoryTelemetry(const fs::path &logDir) {
  if (gMemoryTelemetryRunning.exchange(true)) return;
  std::cout << "[memoryTelemetry] starting, log="
            << (logDir / "memory_telemetry.log").string() << std::endl;
  std::thread([logDir]() { memoryTelemetryLoop(logDir); }).detach();
}

#endif // _WIN32

// 将常见环境变量名映射到 config/phoenix.json 中的 dot path。
// 未映射的变量返回空串，调用方应使用原始 fallback。
static std::string dotPathForEnv(const std::string &env) {
  if (env == "AI_BASE_DIR") return "main.baseDir";
  if (env == "AI_GATEWAY_HOST") return "main.gateway.host";
  if (env == "CONTROLLER_PORT") return "main.gateway.port";
  // v8.0 audit C1: AI_STUDY_PORT / main.studyPort retired; frontend_server.port is the single source.
  if (env == "AI_HTTP_LOG") return "main.httpLog";
  if (env == "AI_COUNT" || env == "AI_NUM") return "main.aiCount";
  if (env == "AI_GROUP_COUNT" || env == "GROUP_COUNT") return "main.groupCount";
  if (env == "AI_GROUP_SIZE" || env == "GROUP_SIZE") return "main.groupSize";
  if (env == "AI_SPARK_NUM_AI") return "main.sparkNumAI";
  if (env == "AI_SPARK_BUDGET") return "main.spark.budget";
  if (env == "AI_SINGLE_PROC") return "main.singleProc";
  if (env == "AI_GROUP_PROC") return "main.groupProc";
  if (env == "AI_GROUP_PROC_TIMEOUT_MS") return "main.groupProcTimeoutMs";
  if (env == "AI_INFER_MP") return "main.inferMP";
  if (env == "AI_INFER_WORKERS") return "main.inferWorkers";
  if (env == "AI_REDIS_CONNECT_TIMEOUT_MS") return "main.redis.connectTimeoutMs";
  if (env == "REDIS_URL") return "main.redis.url";
  if (env == "AI_REDIS_CHANNEL") return "main.redis.channel";
  if (env == "AI_DB_PATH") return "main.dbPath";
  if (env == "AI_REDIS_CACHE_DB") return "main.redis.cacheDb";
  if (env == "AI_REDIS_CACHE_PREFIX") return "main.redis.cachePrefix";
  if (env == "AI_SNAPSHOT_DIR") return "main.snapshotDir";
  if (env == "LMDB_DIR") return "main.lmdb.root";
  if (env == "AI_LMDB_MAP_MB") return "main.lmdb.mapMb";
  if (env == "AI_SEARCH_ENDPOINT") return "main.searchEndpoint";
  if (env == "AI_ROBOTS_DIR") return "main.robots.dir";
  if (env == "AI_LEMMA_CSV") return "main.lemma.csv";
  if (env == "AI_LEMMA_AUTOLOAD") return "main.lemma.autoload";
  if (env == "AI_LEMMA_MAX_MB") return "main.lemma.maxMb";
  if (env == "AI_LEMMA_FORCE") return "main.lemma.force";
  if (env == "AI_ROBOTS_LIMIT") return "main.robots.warmupLimit";
  if (env == "AI_ROBOTS_AUTOLOAD") return "main.robots.autoload";
  if (env == "AI_ROBOTS_WARMUP_SHUFFLE") return "main.robots.warmupShuffle";
  if (env == "AI_ROBOTS_CHUNK_MIN") return "main.robots.chunkMinWords";
  if (env == "AI_ROBOTS_CHUNK_MAX") return "main.robots.chunkMaxWords";
  if (env == "AI_KVM_CACHE_MAX") return "main.kvmCacheMaxEntries";
  if (env == "AI_LEARNING_WARMUP") return "main.learningWarmup";
  if (env == "AI_SYNC_STANDBY_ON_BOOT") return "main.syncStandbyOnBoot";
  if (env == "AI_TESTS_AUTOLOAD") return "main.tests.autoload";
  if (env == "AI_DISABLE_MEMEBARRIER") return "main.disable.memebarrier";
  if (env == "AI_DISABLE_RL") return "main.disable.rl";
  if (env == "AI_DISABLE_ADV") return "main.disable.adv";
  if (env == "AI_DISABLE_LEARNING") return "main.disable.learning";
  if (env == "AI_DISABLE_GNN_MODULE") return "main.disable.gnnModule";
  if (env == "AI_DISABLE_CONTEXT_MODULE") return "main.disable.contextModule";
  if (env == "AI_USING_OLLAMA") return "main.inference.usingOllama";
  if (env == "AI_OLLAMA_FINE_TUNING") return "main.inference.ollamaFineTuning";
  if (env == "AI_INFERENCE_ENABLED") return "main.inference.enabled";
  if (env == "AI_TRANSFORMER_MODE") return "main.inference.transformerMode";
  if (env == "AI_OLLAMA_MODEL") return "main.inference.ollamaModel";
  if (env == "AI_OLLAMA_BASE_URL") return "main.inference.ollamaBaseUrl";
  if (env == "AI_OLLAMA_TIMEOUT_MS") return "main.inference.ollamaTimeoutMs";
  if (env == "AI_LLAMACPP_MODEL") return "main.inference.llamaCppModel";
  if (env == "AI_LLAMACPP_BASE_URL") return "main.inference.llamaCppBaseUrl";
  if (env == "AI_LLAMACPP_TIMEOUT_MS") return "main.inference.llamaCppTimeoutMs";
  if (env == "AI_LLAMACPP_LORA_FILES") return "main.inference.llamaCppLoraFiles";
  if (env == "AI_LLAMACPP_LORA_INIT_WITHOUT_APPLY") return "main.inference.llamaCppLoraInitWithoutApply";
  if (env == "AI_LLAMACPP_FINE_TUNING") return "main.inference.llamaCppFineTuning";
  if (env == "AI_BITNET_MODEL") return "main.inference.bitnetModel";
  if (env == "AI_BITNET_BASE_URL") return "main.inference.bitnetBaseUrl";
  if (env == "AI_BITNET_TIMEOUT_MS") return "main.inference.bitnetTimeoutMs";
  if (env == "AI_TINYLLAMA_MODEL") return "main.inference.tinyllamaModel";
  if (env == "AI_TINYLLAMA_BASE_URL") return "main.inference.tinyllamaBaseUrl";
  if (env == "AI_TINYLLAMA_TIMEOUT_MS") return "main.inference.tinyllamaTimeoutMs";
  if (env == "AI_TINYLLAMA_ENABLED") return "main.inference.tinyllamaEnabled";
  if (env == "AI_CONCEPT_MATRIX_ENABLED") return "main.inference.conceptMatrixEnabled";
  if (env == "AI_CONCEPT_MATRIX_CLEAR_PER_REQUEST") return "main.inference.conceptMatrixClearPerRequest";
  if (env == "AI_GGUF_MODELS_DIR") return "main.inference.ggufModelsDir";
  if (env == "AI_EXTERNAL_STYLE_SIM" || env == "AI_EXTERNAL_BACKEND_STYLE_SIM") return "main.inference.externalBackendStyleSim";
  if (env == "AI_TRANSFORMER_BOOTSTRAP") return "main.inference.transformerBootstrap";
  if (env == "AI_TRANSFORMER_BOOTSTRAP_DOCS") return "main.inference.transformerBootstrapDocs";
  if (env == "AI_LOG_MODE") return "main.logMode";
  if (env == "AI_EXPORT_DIR") return "main.exportDir";
  if (env == "PHOENIX_TUNED_CONFIG") return "main.tunedConfigPath";
  if (env == "AI_MODEL_DEPLOYMENT_CONFIG") return "model_deployment.configPath";
  return "";
}

static void setRuntimeEnvValue(const char *key, const std::string &value) {
#ifdef _WIN32
  SetEnvironmentVariableA(key, value.c_str());
#else
  setenv(key, value.c_str(), 1);
#endif
}

static void
setRuntimeEnvIfArgPresent(const std::map<std::string, std::string> &args,
                          const std::string &argKey, const char *envKey) {
  auto it = args.find(argKey);
  if (it != args.end() && !it->second.empty()) {
    setRuntimeEnvValue(envKey, it->second);
  }
}

static void
applyFrontendEnvFromArgs(const std::map<std::string, std::string> &args) {
  setRuntimeEnvIfArgPresent(args, "frontend-enabled", "AI_FRONTEND_ENABLED");
  setRuntimeEnvIfArgPresent(args, "frontend-http-log", "FRONTEND_HTTP_LOG");
  setRuntimeEnvIfArgPresent(args, "frontend-chat-queue-wait-ms",
                            "FRONTEND_CHAT_QUEUE_WAIT_MS");
  setRuntimeEnvIfArgPresent(args, "frontend-chat-upstream-timeout-ms",
                            "FRONTEND_CHAT_UPSTREAM_TIMEOUT_MS");
  setRuntimeEnvIfArgPresent(args, "frontend-api-upstream-timeout-ms",
                            "FRONTEND_API_UPSTREAM_TIMEOUT_MS");
  setRuntimeEnvIfArgPresent(args, "frontend-chat-max-inflight",
                            "FRONTEND_CHAT_MAX_INFLIGHT");
  setRuntimeEnvIfArgPresent(args, "frontend-port", "FRONTEND_PORT");

  auto syncJwtSecret = [&](const std::string &value) {
    if (value.empty())
      return;
    setRuntimeEnvValue("AI_AUTH_JWT_SECRET", value);
    setRuntimeEnvValue("AUTH_JWT_SECRET", value);
    setRuntimeEnvValue("JWT_SECRET", value);
  };
  auto itAuthJwt = args.find("auth-jwt-secret");
  if (itAuthJwt != args.end() && !itAuthJwt->second.empty())
    syncJwtSecret(itAuthJwt->second);
  auto itJwt = args.find("jwt-secret");
  if (itJwt != args.end() && !itJwt->second.empty())
    syncJwtSecret(itJwt->second);

  std::string host = phoenix::resolveConfigAsString("api.gatewayHost", "127.0.0.1", "AI_GATEWAY_HOST");
  std::string port = phoenix::resolveConfigAsString("api.gatewayPort", "5080", "CONTROLLER_PORT");
  auto itHost = args.find("gateway-host");
  if (itHost != args.end() && !itHost->second.empty())
    host = itHost->second;
  auto itPort = args.find("port");
  if (itPort != args.end() && !itPort->second.empty())
    port = itPort->second;
  // Wildcard bind addresses (0.0.0.0/::/*) are not connectable destinations.
  // The frontend subprocess proxies to the gateway on the SAME machine, so map
  // them back to loopback - otherwise every proxied call fails with
  // "disconnected" (drogon HttpClient cannot connect to 0.0.0.0).
  if (host.empty() || host == "0.0.0.0" || host == "::" || host == "*")
    host = "127.0.0.1";
  setRuntimeEnvValue("AI_API_BASE", std::string("http://") + host + ":" + port);
}

struct Config;
static void applyFrontendEnvFromConfig(const Config &config);

// 将字符串解析为布尔值（宽松规则）。
// 调用方式：读取配置项后转换成布尔开关。
// 实现思路：先转小写，再对常见 false 字面量做否定判断。
// 注意事项：非 false 关键字均视为 true。
static bool boolFrom(const std::string &value, bool fallback = true) {
  std::string v = value;
  auto begin = v.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos)
    return fallback;
  auto end = v.find_last_not_of(" \t\r\n");
  v = v.substr(begin, end - begin + 1);
  if (v.empty())
    return fallback;
  std::transform(v.begin(), v.end(), v.begin(), ::tolower);
  if (v == "1" || v == "true" || v == "on" || v == "yes")
    return true;
  if (v == "0" || v == "false" || v == "off" || v == "no")
    return false;
  return fallback;
}

static int roughTokenCount(const std::string &s) {
  int c = 0;
  bool in = false;
  for (unsigned char ch : s) {
    bool sp = std::isspace(ch) != 0;
    if (!sp && !in) {
      c++;
      in = true;
    } else if (sp) {
      in = false;
    }
  }
  return c;
}

// 复制并裁剪字符串首尾空白字符。
// 调用方式：用于数值解析或文本预处理前标准化输入。
// 实现思路：定位首尾非空白下标后执行 substr。
// 注意事项：全空白字符串会返回空串。
static std::string trimCopy(const std::string &s) {
  auto start = s.find_first_not_of(" \t\r\n");
  if (start == std::string::npos)
    return "";
  auto end = s.find_last_not_of(" \t\r\n");
  return s.substr(start, end - start + 1);
}

// Extract the host part from an HTTP(S) URL. IPv6 hosts are returned without brackets.
static std::string urlHost(const std::string &url) {
  size_t schemeEnd = url.find("://");
  if (schemeEnd == std::string::npos)
    return "";
  size_t hostStart = schemeEnd + 3;
  if (hostStart >= url.size())
    return "";
  if (url[hostStart] == '[') {
    size_t hostEnd = url.find(']', hostStart + 1);
    if (hostEnd == std::string::npos)
      return "";
    return url.substr(hostStart + 1, hostEnd - hostStart - 1);
  }
  size_t hostEnd = url.find_first_of(":/", hostStart);
  if (hostEnd == std::string::npos)
    return url.substr(hostStart);
  return url.substr(hostStart, hostEnd - hostStart);
}

static bool isLoopbackHost(const std::string &host) {
  std::string h = trimCopy(host);
  std::transform(h.begin(), h.end(), h.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return h.empty() || h == "127.0.0.1" || h == "localhost" || h == "::1";
}

// A llama.cpp backend that talks to a remote llama-server does not need a local .gguf file.
static bool isRemoteLlamacppUrl(const std::string &url) {
  return !isLoopbackHost(urlHost(url));
}

// 按 JavaScript 风格解析数字文本。
// 调用方式：传入字符串和输出引用，成功返回 true。
// 实现思路：支持 NaN/Infinity 及 0x/0b/0o 前缀，再回退 stod。
// 注意事项：解析失败返回 false，调用方需自行处理默认值。
static bool parseNumberLikeJs(const std::string &s, double &out) {
  std::string t = trimCopy(s);
  if (t.empty()) {
    out = 0.0;
    return true;
  }
  if (t == "NaN" || t == "+NaN" || t == "-NaN") {
    out = std::numeric_limits<double>::quiet_NaN();
    return true;
  }
  if (t == "Infinity" || t == "+Infinity") {
    out = std::numeric_limits<double>::infinity();
    return true;
  }
  if (t == "-Infinity") {
    out = -std::numeric_limits<double>::infinity();
    return true;
  }

  int sign = 1;
  std::string body = t;
  if (!body.empty() && (body[0] == '+' || body[0] == '-')) {
    if (body[0] == '-')
      sign = -1;
    body = body.substr(1);
  }

  try {
    if (body.rfind("0x", 0) == 0 || body.rfind("0X", 0) == 0) {
      size_t idx = 0;
      unsigned long long v = std::stoull(body.substr(2), &idx, 16);
      if (idx != body.substr(2).size())
        return false;
      out = (double)v * sign;
      return true;
    }
    if (body.rfind("0b", 0) == 0 || body.rfind("0B", 0) == 0) {
      size_t idx = 0;
      unsigned long long v = std::stoull(body.substr(2), &idx, 2);
      if (idx != body.substr(2).size())
        return false;
      out = (double)v * sign;
      return true;
    }
    if (body.rfind("0o", 0) == 0 || body.rfind("0O", 0) == 0) {
      size_t idx = 0;
      unsigned long long v = std::stoull(body.substr(2), &idx, 8);
      if (idx != body.substr(2).size())
        return false;
      out = (double)v * sign;
      return true;
    }

    size_t idx = 0;
    double v = std::stod(t, &idx);
    if (idx != t.size())
      return false;
    out = v;
    return true;
  } catch (...) {
    return false;
  }
}

static bool isLikelyGibberishReply(const std::string &raw) {
  std::string s = trimCopy(raw);
  if (s.empty())
    return true;

  int letters = 0;
  int digits = 0;
  int spaces = 0;
  int punct = 0;
  int controls = 0;
  int cjk = 0;
  int vowels = 0;
  for (unsigned char ch : s) {
    if (ch < 0x20 || ch == 0x7F) {
      controls++;
      continue;
    }
    if (std::isspace(ch)) {
      spaces++;
      continue;
    }
    if (std::isalpha(ch)) {
      letters++;
      char lo = (char)std::tolower(ch);
      if (lo == 'a' || lo == 'e' || lo == 'i' || lo == 'o' || lo == 'u')
        vowels++;
      continue;
    }
    if (std::isdigit(ch)) {
      digits++;
      continue;
    }
    if (std::ispunct(ch)) {
      punct++;
      continue;
    }
    if (ch >= 0x80)
      cjk++;
  }

  if (controls > 0)
    return true;

  int nonSpace = std::max(1, (int)s.size() - spaces);
  double punctRatio = (double)punct / (double)nonSpace;
  if (punctRatio > 0.60)
    return true;

  int core = letters + digits + cjk;
  if (core < std::max(4, (int)s.size() / 10))
    return true;

  std::istringstream iss(s);
  std::string tok;
  int tokenCount = 0;
  int shortTokenCount = 0;
  int oddTokenCount = 0;
  int tokIdLike = 0;
  while (iss >> tok) {
    tokenCount++;
    if (tok.size() <= 2)
      shortTokenCount++;
    bool hasNonAlpha = false;
    bool hasDigit = false;
    for (unsigned char c : tok) {
      if (!std::isalpha(c))
        hasNonAlpha = true;
      if (std::isdigit(c))
        hasDigit = true;
    }
    if (hasNonAlpha || hasDigit)
      oddTokenCount++;
    if (tok.size() > 3 && tok.rfind("tok", 0) == 0) {
      bool allNum = true;
      for (size_t i = 3; i < tok.size(); i++) {
        if (!std::isdigit((unsigned char)tok[i])) {
          allNum = false;
          break;
        }
      }
      if (allNum)
        tokIdLike++;
    }
  }

  if (tokIdLike >= 2)
    return true;

  if (cjk == 0 && tokenCount >= 4) {
    double shortRatio =
        (double)shortTokenCount / (double)std::max(1, tokenCount);
    double oddRatio = (double)oddTokenCount / (double)std::max(1, tokenCount);
    if (shortRatio >= 0.75 && oddRatio >= 0.30)
      return true;
  }

  if (cjk == 0 && letters >= 12) {
    double vowelRatio = (double)vowels / (double)letters;
    if (vowelRatio < 0.18)
      return true;
  }

  return false;
}

static bool containsInternalContextMarkers(const std::string &raw) {
  std::string s = raw;
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return (char)std::tolower(c); });
  static const std::array<const char *, 11> markers = {
      "ctx_v2", "ctx_stage1", "nodes:", "edges:",     "summary:n=",  "|act=",
      "|deg=",  "|seed=",     "|kw=",   "[tool_call", "[tool_result"};
  for (const char *m : markers) {
    if (s.find(m) != std::string::npos)
      return true;
  }
  return false;
}

static bool hasExcessUnknownTokens(const std::string &raw) {
  std::istringstream iss(raw);
  std::string tok;
  int total = 0;
  int unk = 0;
  while (iss >> tok) {
    std::string t;
    t.reserve(tok.size());
    for (unsigned char c : tok)
      t.push_back((char)std::tolower(c));
    if (t == "unk" || t == "<unk>")
      unk++;
    total++;
    if (total >= 80)
      break;
  }
  if (total == 0)
    return false;
  return unk >= 3 && (double)unk / (double)total >= 0.25;
}

static double numberOr(const std::string &s, double fallback) {
  double v = 0.0;
  if (!parseNumberLikeJs(s, v))
    return fallback;
  if (!std::isfinite(v))
    return fallback;
  if (v == 0.0)
    return fallback;
  return v;
}

static double numberOrNaN(const std::string &s, double fallback) {
  double v = 0.0;
  if (!parseNumberLikeJs(s, v))
    return fallback;
  if (!std::isfinite(v))
    return fallback;
  return v;
}

static int64_t getRobotsRetrainLimitCap() {
  int64_t cap = phoenix::resolveConfig<int64_t>("robots.retrain.limitMax", 10000, "AI_ROBOTS_RETRAIN_LIMIT_MAX");
  if (cap < 1)
    cap = 1;
  if (cap > 1000000)
    cap = 1000000;
  return cap;
}

static int64_t getRobotsRetrainBatchDefault() {
  int64_t batch = phoenix::resolveConfig<int64_t>("robots.retrain.batch", 64, "AI_ROBOTS_RETRAIN_BATCH");
  if (batch < 16)
    batch = 16;
  if (batch > 512)
    batch = 512;
  return batch;
}

static int64_t getRobotsRetrainBatchCharsBudget() {
  int64_t budget = phoenix::resolveConfig<int64_t>("robots.retrain.batchChars", 4000000, "AI_ROBOTS_RETRAIN_BATCH_CHARS");
  if (budget < 200000)
    budget = 200000;
  if (budget > 100000000)
    budget = 100000000;
  return budget;
}

static int64_t getRobotsRetrainDocMaxChars() {
  int64_t cap = phoenix::resolveConfig<int64_t>("robots.retrain.docMaxChars", 6000, "AI_ROBOTS_RETRAIN_DOC_MAX_CHARS");
  if (cap < 512)
    cap = 512;
  if (cap > 200000)
    cap = 200000;
  return cap;
}

static int64_t getRobotsRetrainDocMaxTokens() {
  int64_t cap = phoenix::resolveConfig<int64_t>("robots.retrain.docMaxTokens", 256, "AI_ROBOTS_RETRAIN_DOC_MAX_TOKENS");
  if (cap < 16)
    cap = 16;
  if (cap > 2000)
    cap = 2000;
  return cap;
}

static void ensureDir(const fs::path &dir) {
  if (!fs::exists(dir))
    fs::create_directories(dir);
}

static fs::path getProjectRoot() {
  static fs::path cached;
  static bool inited = false;
  if (inited)
    return cached;
  inited = true;

  std::vector<fs::path> starts;
  if (!g_selfPath.empty()) {
    try {
      starts.push_back(fs::absolute(fs::path(g_selfPath)).parent_path());
    } catch (...) {
    }
  }
  try {
    starts.push_back(fs::current_path());
  } catch (...) {
  }
  try {
    fs::path sourceDir = fs::path(__FILE__).parent_path();
    if (sourceDir.empty()) {
      sourceDir = fs::current_path();
    }
    starts.push_back(fs::absolute(sourceDir));
  } catch (...) {
  }

  for (const auto &start : starts) {
    fs::path cur = start;
    for (int i = 0; i < 6; i++) {
      if (fs::exists(cur / "compile.bat") && fs::exists(cur / "main.cpp")) {
        cached = fs::absolute(cur);
        return cached;
      }
      if (fs::exists(cur / "conanfile.txt") && fs::exists(cur / "main.cpp")) {
        cached = fs::absolute(cur);
        return cached;
      }
      if (fs::exists(cur / "main.cjs")) {
        cached = fs::absolute(cur);
        return cached;
      }
      if (fs::exists(cur / "package.json")) {
        cached = fs::absolute(cur);
        return cached;
      }
      if (cur.has_parent_path())
        cur = cur.parent_path();
      else
        break;
    }
  }
  for (const auto &start : starts) {
    if (!start.empty()) {
      cached = start;
      return cached;
    }
  }
  try {
    cached = fs::current_path();
  } catch (...) {
    cached = fs::path(".");
  }
  return cached;
}

static std::map<std::string, std::string> parseArgs(int argc, char **argv) {
  std::map<std::string, std::string> out;
  for (int i = 1; i < argc; i++) {
    std::string item = argv[i];
    if (item.rfind("--", 0) != 0)
      continue;
    auto pos = item.find('=');
    if (pos == std::string::npos) {
      out[item.substr(2)] = "true";
    } else {
      out[item.substr(2, pos - 2)] = item.substr(pos + 1);
    }
  }
  return out;
}

// Config 汇总进程启动所需的运行期配置项。
// 调用方式：由配置加载流程填充后传入各子模块。
// 实现思路：按功能域聚合端口、路径、缓存和并发参数。
// 注意事项：字段默认值用于保障最小可运行配置。
// 注意事项：环境变量/命令行覆盖顺序由外层控制。
// 注意事项：该结构仅承载数据，不包含业务逻辑。
