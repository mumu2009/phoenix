/* loggerCXXH.hpp - C++ logging system with memory sampling
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

#include <filesystem>
#include <fstream>
#include <atomic>
#include <mutex>
#include <string>
#include <thread>

#ifdef ERROR
#undef ERROR
#endif

/* C++ logging system with multiple log types and memory sampling */
class LoggerCXX {
public:
    enum class Mode {
        Off,        /* Logging disabled */
        Release,    /* Release mode logging */
        Debug       /* Debug mode logging */
    };

    enum class Type {
        FATAL,      /* Fatal errors */
        ERROR,      /* General errors */
        WARNING,    /* Warnings */
        LOG,        /* General log messages */
        MEMORY,     /* Memory-related logs */
        COMPUTE,    /* Compute-related logs */
        DEBUG,      /* Debug messages */
        TEST,       /* Test-related logs */
        BENCHMARK,  /* Benchmark logs */
        SECURITY,   /* Security-related logs */
        MONITORING  /* Monitoring logs */
    };

    static LoggerCXX &instance(); /* Get singleton instance */

    void initialize(const std::string &mode, const std::filesystem::path &dir); /* Initialize logger */
    Mode mode() const { return mode_; } /* Get current mode */
    bool enabled() const { return mode_ != Mode::Off; } /* Check if logging enabled */
    bool shouldLog(Type type) const; /* Check if type should be logged */

    void log(Type type, const std::string &value); /* Log a message */

    void shutdown(); /* Stop sampler, close file, and reset logger for test cleanup */

    ~LoggerCXX();

private:
    LoggerCXX() = default;
    std::string typeToString(Type type) const; /* Convert type to string */
    std::string levelToString(Type type) const; /* Convert type to log level */
    std::string channelToString(Type type) const; /* Convert type to channel name */
    std::string nowIso() const; /* Get current ISO timestamp */
    std::string buildMemorySample() const; /* Build memory sample string */
    void ensureOpen(); /* Ensure log file is open */
    void startMemorySamplerLocked(); /* Start memory sampler thread */
    void stopMemorySamplerLocked(); /* Stop memory sampler thread */
    void logUnlocked(Type type, const std::string &value); /* Log without lock */

    std::mutex mu_;                    /* Mutex for thread safety */
    std::filesystem::path dir_;         /* Log directory */
    std::filesystem::path filePath_;    /* Log file path */
    std::ofstream out_;                 /* Output file stream */
    Mode mode_{Mode::Release};          /* Current logging mode */
    bool initialized_{false};           /* Initialization flag */
    int memoryIntervalSec_{0};         /* Memory sampling interval in seconds (0 = disabled by default) */
    std::atomic<bool> memorySamplerRunning_{false}; /* Memory sampler running flag */
    std::thread memorySampler_;         /* Memory sampler thread */
};
