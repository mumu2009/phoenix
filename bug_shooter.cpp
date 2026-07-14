/* bug_shooter.cpp - Bug shooter implementation
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

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#endif

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

int64_t nowMs() {
    return static_cast<int64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

std::string argValue(int argc, char **argv, const std::string &key, const std::string &fallback = "") {
    for (int i = 1; i + 1 < argc; ++i) {
        if (key == argv[i]) {
            return argv[i + 1];
        }
    }
    return fallback;
}

bool writeJsonFile(const fs::path &path, const json &doc) {
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }
    out << doc.dump(2);
    return static_cast<bool>(out);
}

#ifdef _WIN32
bool readMemoryCounters(DWORD pid, PROCESS_MEMORY_COUNTERS_EX &pmc) {
    HANDLE process = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ | PROCESS_SET_QUOTA, FALSE, pid);
    if (!process) {
        return false;
    }
    const BOOL ok = GetProcessMemoryInfo(process, reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&pmc), sizeof(pmc));
    CloseHandle(process);
    return ok == TRUE;
}

bool trimWorkingSet(DWORD pid) {
    HANDLE process = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_SET_QUOTA, FALSE, pid);
    if (!process) {
        return false;
    }
    const BOOL ok = EmptyWorkingSet(process) && SetProcessWorkingSetSizeEx(process, static_cast<SIZE_T>(-1), static_cast<SIZE_T>(-1), 0);
    CloseHandle(process);
    return ok == TRUE;
}

bool processStillRunning(DWORD pid) {
    HANDLE process = OpenProcess(SYNCHRONIZE, FALSE, pid);
    if (!process) {
        return false;
    }
    const DWORD wait = WaitForSingleObject(process, 0);
    CloseHandle(process);
    return wait == WAIT_TIMEOUT;
}
#endif

} // namespace

int main(int argc, char **argv) {
    const DWORD targetPid = static_cast<DWORD>(std::strtoul(argValue(argc, argv, "--target-pid", "0").c_str(), nullptr, 10));
    const std::string targetName = argValue(argc, argv, "--target-name", "phoenix_main.exe");
    const fs::path statusFile = argValue(argc, argv, "--status-file", "runtime_store/bug_shooter_status.json");
    const std::size_t softLimitMb = static_cast<std::size_t>(std::strtoull(argValue(argc, argv, "--soft-limit-mb", "3072").c_str(), nullptr, 10));
    const std::size_t hardLimitMb = static_cast<std::size_t>(std::strtoull(argValue(argc, argv, "--hard-limit-mb", "4096").c_str(), nullptr, 10));
    const int pollMs = std::max(250, std::atoi(argValue(argc, argv, "--poll-ms", "1500").c_str()));

    if (!targetPid) {
        std::cerr << "missing --target-pid\n";
        return 2;
    }

    json status = {
        {"ok", true},
        {"running", true},
        {"pid", static_cast<uint32_t>(GetCurrentProcessId())},
        {"targetPid", static_cast<uint32_t>(targetPid)},
        {"targetName", targetName},
        {"softLimitMb", softLimitMb},
        {"hardLimitMb", hardLimitMb},
        {"pollMs", pollMs},
        {"status", "starting"},
        {"events", json::array()}
    };
    writeJsonFile(statusFile, status);

#ifdef _WIN32
    // Keep a handle open from the start so we can always read the exit code
    HANDLE hTargetPersistent = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ | SYNCHRONIZE, FALSE, targetPid);
    if (!hTargetPersistent) {
        std::cerr << "[bug_shooter] cannot open target process " << targetPid << std::endl;
        status["status"] = "cannot_open_target";
        writeJsonFile(statusFile, status);
        return 3;
    }
    uint64_t trims = 0;
    uint64_t warnings = 0;
    std::size_t peakWorkingSetMb = 0;
    std::size_t peakPrivateMb = 0;
    int64_t startMs = nowMs();
    int64_t lastMemorySnapshotMs = 0;
    json memoryTimeline = json::array();
    
    while (processStillRunning(targetPid)) {
        PROCESS_MEMORY_COUNTERS_EX pmc{};
        status["updatedAtMs"] = nowMs();
        status["status"] = "monitoring";
        status["uptimeMs"] = nowMs() - startMs;
        if (readMemoryCounters(targetPid, pmc)) {
            const std::size_t workingSetMb = static_cast<std::size_t>(pmc.WorkingSetSize / (1024ull * 1024ull));
            const std::size_t privateMb = static_cast<std::size_t>(pmc.PrivateUsage / (1024ull * 1024ull));
            const std::size_t pageFaults = static_cast<std::size_t>(pmc.PageFaultCount);
            peakWorkingSetMb = std::max(peakWorkingSetMb, workingSetMb);
            peakPrivateMb = std::max(peakPrivateMb, privateMb);
            status["memory"] = {
                {"workingSetMb", workingSetMb},
                {"privateMb", privateMb},
                {"peakWorkingSetMb", peakWorkingSetMb},
                {"peakPrivateMb", peakPrivateMb},
                {"pageFaults", pageFaults}
            };
            // Record memory timeline every 30 seconds
            if (nowMs() - lastMemorySnapshotMs >= 30000) {
                lastMemorySnapshotMs = nowMs();
                memoryTimeline.push_back({
                    {"atMs", nowMs()},
                    {"workingSetMb", workingSetMb},
                    {"privateMb", privateMb},
                    {"pageFaults", pageFaults}
                });
                if (memoryTimeline.size() > 120) {
                    memoryTimeline.erase(memoryTimeline.begin());
                }
                status["memoryTimeline"] = memoryTimeline;
            }
            if (workingSetMb >= softLimitMb || privateMb >= softLimitMb) {
                ++warnings;
                status["lastAction"] = "soft_limit_warning";
                status["warningCount"] = warnings;
            }
            if (workingSetMb >= hardLimitMb || privateMb >= hardLimitMb) {
                const bool trimmed = trimWorkingSet(targetPid);
                ++trims;
                status["lastAction"] = trimmed ? "working_set_trim" : "trim_failed";
                status["trimCount"] = trims;
                status["events"].push_back({
                    {"atMs", nowMs()},
                    {"action", status["lastAction"]},
                    {"workingSetMb", workingSetMb},
                    {"privateMb", privateMb}
                });
                if (status["events"].size() > 64) {
                    status["events"].erase(status["events"].begin());
                }
            }
        } else {
            status["lastAction"] = "memory_read_failed";
        }
        writeJsonFile(statusFile, status);
        std::this_thread::sleep_for(std::chrono::milliseconds(pollMs));
    }
    
    // Capture exit code of target process using persistent handle
    DWORD exitCode = 0;
    GetExitCodeProcess(hTargetPersistent, &exitCode);
    CloseHandle(hTargetPersistent);
    status["exitCode"] = static_cast<uint32_t>(exitCode);
    status["exitCodeHex"] = ([&]() {
        std::ostringstream oss;
        oss << "0x" << std::hex << exitCode;
        return oss.str();
    })();
    if (exitCode == 0xC0000005) {
        status["exitReason"] = "ACCESS_VIOLATION";
    } else if (exitCode == 0xC00000FD) {
        status["exitReason"] = "STACK_OVERFLOW";
    } else if (exitCode == 0xC0000374) {
        status["exitReason"] = "HEAP_CORRUPTION";
    } else if (exitCode == 0) {
        status["exitReason"] = "NORMAL_EXIT";
    } else {
        status["exitReason"] = "UNKNOWN";
    }
#endif

    status["running"] = false;
    status["status"] = "target_exited";
    status["updatedAtMs"] = nowMs();
    status["totalUptimeMs"] = nowMs() - startMs;
    writeJsonFile(statusFile, status);
    std::cout << "[bug_shooter] target exited. exitCode=" << status.value("exitCodeHex", "?")
              << " reason=" << status.value("exitReason", "?")
              << " uptime=" << (nowMs() - startMs) / 1000 << "s"
              << " peakMemMb=" << peakWorkingSetMb << std::endl;
    return 0;
}