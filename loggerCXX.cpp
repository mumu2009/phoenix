/* loggerCXX.cpp - Logger implementation
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

#include "loggerCXXH.hpp"
#include "phoenix_config.hpp"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <cctype>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#ifdef ERROR
#undef ERROR
#endif
#else
#include <unistd.h>
#include <fstream>
#endif

LoggerCXX &LoggerCXX::instance()
{
    static LoggerCXX inst;
    return inst;
}

LoggerCXX::~LoggerCXX()
{
    stopMemorySamplerLocked();
    std::lock_guard<std::mutex> lock(mu_);
    if (out_.is_open())
        out_.close();
}

bool LoggerCXX::shouldLog(Type type) const
{
    if (!initialized_ || mode_ == Mode::Off)
        return false;
    if (mode_ == Mode::Debug)
        return true;
    return type == Type::FATAL || type == Type::ERROR;
}

void LoggerCXX::initialize(const std::string &mode, const std::filesystem::path &dir)
{
    stopMemorySamplerLocked();
    bool needStart = false;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (out_.is_open() && dir_ != dir)
            out_.close();
        dir_ = dir;
        std::string m = mode;
        for (auto &c : m)
            c = (char)std::tolower((unsigned char)c);
        if (m == "off" || m == "none" || m == "0")
        {
            mode_ = Mode::Off;
        }
        else if (m == "debug")
        {
            mode_ = Mode::Debug;
        }
        else
        {
            mode_ = Mode::Release;
        }
        int interval = phoenix::resolveConfig<int>("logger.memoryIntervalSec", 0, "AI_LOG_MEMORY_INTERVAL_SEC");
        if (interval < 0)
            interval = 0;
        memoryIntervalSec_ = interval;
        initialized_ = true;
        needStart = (mode_ == Mode::Debug && memoryIntervalSec_ > 0);
    }
    if (needStart)
    {
        startMemorySamplerLocked();
    }
}

void LoggerCXX::ensureOpen()
{
    if (!initialized_ || mode_ == Mode::Off)
        return;
    if (out_.is_open())
        return;
    std::error_code ec;
    std::filesystem::create_directories(dir_, ec);
    std::string name = (mode_ == Mode::Debug) ? "runtime_debug.log" : "runtime_release.log";
    filePath_ = dir_ / name;
    out_.open(filePath_, std::ios::out | std::ios::app);
}

std::string LoggerCXX::typeToString(Type type) const
{
    switch (type)
    {
    case Type::FATAL:
        return "FATAL";
    case Type::ERROR:
        return "ERROR";
    case Type::WARNING:
        return "WARNING";
    case Type::LOG:
        return "LOG";
    case Type::MEMORY:
        return "MEMORY";
    case Type::COMPUTE:
        return "COMPUTE";
    case Type::DEBUG:
        return "DEBUG";
    case Type::TEST:
        return "TEST";
    case Type::BENCHMARK:
        return "BENCHMARK";
    case Type::SECURITY:
        return "SECURITY";
    case Type::MONITORING:
        return "MONITORING";
    default:
        return "LOG";
    }
}

std::string LoggerCXX::levelToString(Type type) const
{
    switch (type)
    {
    case Type::FATAL:
        return "FATAL";
    case Type::ERROR:
        return "ERROR";
    case Type::WARNING:
        return "WARNING";
    case Type::LOG:
        return "LOG";
    case Type::DEBUG:
        return "DEBUG";
    default:
        return "INFO";
    }
}

std::string LoggerCXX::channelToString(Type type) const
{
    switch (type)
    {
    case Type::MEMORY:
        return "memory";
    case Type::COMPUTE:
        return "compute";
    case Type::DEBUG:
        return "debug";
    case Type::TEST:
        return "test";
    case Type::BENCHMARK:
        return "benchmark";
    case Type::SECURITY:
        return "security";
    case Type::MONITORING:
        return "monitoring";
    case Type::FATAL:
    case Type::ERROR:
    case Type::WARNING:
    case Type::LOG:
    default:
        return "runtime";
    }
}

std::string LoggerCXX::nowIso() const
{
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
    std::ostringstream oss;
    oss << buf << '.' << std::setfill('0') << std::setw(3) << ms.count() << 'Z';
    return oss.str();
}

void LoggerCXX::log(Type type, const std::string &value)
{
    std::lock_guard<std::mutex> lock(mu_);
    if (!shouldLog(type))
        return;
    logUnlocked(type, value);
}

std::string LoggerCXX::buildMemorySample() const
{
    std::ostringstream oss;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&pmc, sizeof(pmc)))
    {
        oss << "rss=" << (int64_t)pmc.WorkingSetSize
            << " private=" << (int64_t)pmc.PrivateUsage
            << " peak=" << (int64_t)pmc.PeakWorkingSetSize;
    }
    else
    {
        oss << "rss=0 private=0 peak=0";
    }
#else
    long pageSize = sysconf(_SC_PAGESIZE);
    std::ifstream statm("/proc/self/statm");
    long size = 0;
    long resident = 0;
    if (statm >> size >> resident)
    {
        int64_t rss = (int64_t)resident * pageSize;
        int64_t virt = (int64_t)size * pageSize;
        oss << "rss=" << rss << " virtual=" << virt;
    }
    else
    {
        oss << "rss=0 virtual=0";
    }
#endif
    return oss.str();
}

void LoggerCXX::shutdown()
{
    stopMemorySamplerLocked();
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (out_.is_open())
            out_.close();
        initialized_ = false;
        mode_ = Mode::Off;
        dir_.clear();
        filePath_.clear();
    }
}

void LoggerCXX::startMemorySamplerLocked()
{
    if (memorySamplerRunning_)
        return;
    if (!initialized_ || mode_ != Mode::Debug || memoryIntervalSec_ <= 0)
        return;
    memorySamplerRunning_ = true;
    memorySampler_ = std::thread([this]()
                                 {
        while (memorySamplerRunning_) {
            std::this_thread::sleep_for(std::chrono::seconds(memoryIntervalSec_));
            if (!memorySamplerRunning_) break;
            std::lock_guard<std::mutex> lock(mu_);
            if (!shouldLog(Type::MEMORY)) continue;
            logUnlocked(Type::MEMORY, buildMemorySample());
        } });
}

void LoggerCXX::stopMemorySamplerLocked()
{
    memorySamplerRunning_ = false;
    if (memorySampler_.joinable())
    {
        memorySampler_.join();
    }
}

void LoggerCXX::logUnlocked(Type type, const std::string &value)
{
    ensureOpen();
    if (!out_.is_open())
        return;
    out_ << "[" << levelToString(type) << "]"
         << "[type=" << typeToString(type) << "]"
         << "[channel=" << channelToString(type) << "]"
         << " ts=" << nowIso() << " " << value << "\n";
    out_.flush();
}
