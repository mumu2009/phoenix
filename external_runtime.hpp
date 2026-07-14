/* external_runtime.hpp - External inference backend runtime management
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

#include <cstdint>
#include <filesystem>
#include <string>

#include <nlohmann/json.hpp>

namespace external_runtime {

namespace fs = std::filesystem;
using json = nlohmann::json;

/* Specification for external inference backend runtime.
   Configures paths, parameters, and behavior for backends like Ollama, llama.cpp, and BitNet. */
struct BackendRuntimeSpec {
    std::string provider;          /* Backend provider name (ollama, llamacpp, bitnet) */
    std::string baseUrl;           /* Base URL for the backend service */
    std::string modelPath;         /* Path to the model file */
    fs::path runtimeDir;           /* Runtime directory for the backend */
    fs::path providerRoot;         /* Root directory for provider code */
    fs::path calculatorRoot;       /* Root directory for calculator code */
    fs::path divingAgreementRoot;  /* Root directory for diving agreement code */
    fs::path pythonExecutable;     /* Path to Python executable */
    fs::path adapterScriptPath;    /* Path to adapter script */
    std::string launchArgsTemplate; /* Template for launch arguments */
    int ctxSize{0};                /* Context size for the model */
    int batchSize{0};              /* Batch size for inference */
    int ubatchSize{0};             /* Micro-batch size */
    std::string ropeScaling;       /* RoPE scaling method */
    double ropeFreqBase{0.0};      /* RoPE frequency base */
    double ropeFreqScale{0.0};     /* RoPE frequency scale */
    int yarnOrigCtx{0};            /* YaRN original context size */
    double yarnExtFactor{0.0};     /* YaRN extension factor */
    double yarnAttnFactor{0.0};    /* YaRN attention factor */
    double yarnBetaFast{0.0};      /* YaRN beta fast */
    double yarnBetaSlow{0.0};      /* YaRN beta slow */
    bool autoLaunch{true};         /* Auto-launch backend if not running */
    int readyTimeoutMs{30000};     /* Timeout for backend readiness check */
    int healthPollMs{500};         /* Health check poll interval */
    std::string loraFiles;         /* LoRA adapter files */
    bool loraInitWithoutApply{false}; /* Initialize LoRA without applying */
    bool fineTuningEnabled{false}; /* Enable fine-tuning mode */
};

/* Runtime state for external inference backend.
   Tracks the current status, process information, and health of the backend. */
struct BackendRuntimeState {
    bool ready{false};              /* Backend is ready to serve requests */
    bool launchAttempted{false};    /* Launch attempt has been made */
    bool binaryFound{false};        /* Backend binary was found */
    bool brainMapReady{false};      /* Brain map is ready */
    uint32_t pid{0};               /* Process ID of the backend */
    int64_t checkedAtMs{0};         /* Last health check timestamp */
    std::string status;             /* Status message */
    std::string error;              /* Error message if any */
    std::string commandLine;        /* Command line used to launch */
    fs::path binaryPath;            /* Path to the binary */
    fs::path statusFile;            /* Path to status file */
    fs::path brainMapPath;         /* Path to brain map file */

    json toJson() const;            /* Serialize state to JSON */
};

/* Specification for BugShooter monitoring process.
   Configures memory monitoring and crash detection for the main process. */
struct BugShooterSpec {
    bool enabled{true};             /* Enable BugShooter monitoring */
    fs::path executablePath;        /* Path to BugShooter executable */
    fs::path runtimeDir;            /* Runtime directory */
    uint32_t targetPid{0};          /* Target process PID to monitor */
    std::string targetName;         /* Target process name */
    std::size_t softLimitMb{3072};  /* Soft memory limit in MB */
    std::size_t hardLimitMb{4096};  /* Hard memory limit in MB */
    int pollIntervalMs{1500};       /* Polling interval in milliseconds */
};

/* State for BugShooter monitoring process.
   Tracks the monitoring process status and health. */
struct BugShooterState {
    bool enabled{false};            /* BugShooter is enabled */
    bool running{false};            /* BugShooter is running */
    bool executableFound{false};    /* BugShooter executable was found */
    bool launchAttempted{false};   /* Launch attempt has been made */
    uint32_t pid{0};               /* BugShooter process ID */
    int64_t checkedAtMs{0};         /* Last check timestamp */
    std::string status;             /* Status message */
    std::string error;              /* Error message if any */
    fs::path statusFile;            /* Path to status file */

    json toJson() const;            /* Serialize state to JSON */
};

bool ensureBackendReady(const BackendRuntimeSpec &spec, BackendRuntimeState &state); /* Ensure backend is ready */
bool ensureBugShooterAttached(const BugShooterSpec &spec, BugShooterState &state); /* Ensure BugShooter is attached */
json readStatusFile(const fs::path &path); /* Read status from file */

} // namespace external_runtime