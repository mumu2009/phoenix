/* edge_platform.hpp - Edge platform and NPU control interface
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
#include <deque>
#include <filesystem>
#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

namespace edge_platform {

namespace fs = std::filesystem;
using json = nlohmann::json;

/* Runtime configuration for edge platform.
   Configures NPU, GPIO, SPI, and MCU target settings for edge deployment. */
struct RuntimeConfig {
    fs::path baseDir;               /* Base directory for runtime data */
    fs::path netlistRoot;           /* Root directory for netlist files */
    std::vector<fs::path> netlistFiles; /* List of netlist files to load */
    fs::path gerberConnectorMap;    /* Path to Gerber connector map */
    bool enabled{true};             /* Enable edge platform */
    bool npuEnabled{true};          /* Enable NPU functionality */
    bool peripheralSchedulingEnabled{true}; /* Enable peripheral scheduling */
    std::string preferredComputeBackend{"auto"}; /* Preferred compute backend (auto, cpu, npu, hybrid) */
    int maxComputeInflight{2};       /* Maximum concurrent compute operations */
    int maxPeripheralInflight{2};   /* Maximum concurrent peripheral operations */
    std::string npuSpiDevice{"/dev/spidev0.0"}; /* NPU SPI device path */
    int npuSpiSpeedHz{120000000};    /* NPU SPI speed in Hz */
    int npuSpiMode{0};              /* NPU SPI mode */
    bool npuAsyncGpioExecuteLocally{true}; /* Execute GPIO locally for NPU */
    std::string npuGpioSysfsRoot{"/sys/class/gpio"}; /* GPIO sysfs root */
    std::string gpioChipDevice{"/dev/gpiochip0"}; /* GPIO chip device */
    bool preferGpioChip{true};      /* Prefer GPIO chip over sysfs */
    int npuGpioPulseUs{0};          /* NPU GPIO pulse width in microseconds */
    bool npuVirtualMemoryEnabled{true}; /* Enable NPU virtual memory */
    int npuAdvertisedMemoryMb{8192}; /* Advertised NPU memory in MB */
    fs::path npuSdcardWeightsRoot{"runtime_store/sdcard_weights"}; /* SD card weights root */
    int npuHotWeightsLimit{12};     /* Hot weights limit */
    int npuHotPromoteHits{2};       /* Hits needed to promote to hot */
    int npuUnitCount{19};           /* Number of NPU units */
    bool npuEfficiencyProbeEnabled{true}; /* Enable NPU efficiency probing */
    double npuEfficiencyAnomalyThreshold{0.35}; /* NPU efficiency anomaly threshold */

    /* MCU v2.2 target: GD32H759ZMT6 + 768MB external SDRAM + FreeRTOS + SSH */
    std::string mcuTargetSoC{"gd32h759zmt6"}; /* Target SoC: gd32h759zmt6, gd32f427vet6, esp32_p4, pi_zero_2w */
    std::string mcuGpioBackend{"mcu_baremetal"}; /* GPIO backend: pi_mmap, linux_gpiochip, mcu_baremetal */
    std::string mcuPosixLayer{"freertos"}; /* POSIX layer: full_linux, musl_linux, freertos, none */
    std::string mcuSshStack{"wolfssh"}; /* SSH stack: wolfssh, tinyssh, none */
    bool mcuSdcardAsSwap{true}; /* Use SD card as swap */
    bool mcuDdrPagingEnabled{false}; /* Enable DDR paging (768MB SDRAM fits llama.cpp minimum; no paging needed) */
    bool mcuSdramAsHeap{true}; /* Map malloc to external SDRAM */
    int mcuSdramTotalMb{768}; /* Total SDRAM in MB (768MB = 3x 256MB 16-bit SDRAM via EXMC) */
    int mcuSdramChipCount{3}; /* Number of SDRAM chips (3 identical 256MB chips for unification) */
    int mcuPsrampTotalMb{0}; /* Total PSRAM in MB */
    int mcuGpioWriteRateMHz{150}; /* Expected GPIO toggle rate in MHz (GD32H7 bare metal) */
    int mcuDacLines{16}; /* Number of DAC lines */
    int mcuComparatorLines{16}; /* Number of comparator read-back lines */
};

/* Platform manager for edge hardware control.
   Manages NPU, GPIO, SPI, and peripheral scheduling for edge deployment. */
class PlatformManager {
public:
    PlatformManager();

    void reconfigure(const RuntimeConfig &config); /* Reconfigure with new settings */
    json status(); /* Get current platform status */
    json refreshTopology(); /* Refresh hardware topology */
    json applyPatch(const json &patch); /* Apply hardware patch */
    json planCompute(const json &payload); /* Plan compute operation */
    json dispatchCompute(const json &payload); /* Dispatch compute operation */
    json planMobility(const json &payload); /* Plan mobility operation */
    json dispatchPeripheral(const json &payload); /* Dispatch peripheral operation */
    json dispatchMobility(const json &payload); /* Dispatch mobility operation */
    json runSelfTest(const json &payload); /* Run hardware self-test */

    /* Signal binding from component to netlist signal */
    struct SignalBinding {
        std::string componentId;      /* Component identifier */
        std::string designator;       /* Component designator */
        std::string componentValue;   /* Component value */
        std::string pinNumber;        /* Pin number */
        std::string signal;           /* Signal name */
    };

    /* Summary of a hardware interface */
    struct InterfaceSummary {
        std::string id;               /* Interface ID */
        std::string type;             /* Interface type */
        bool available{false};        /* Interface availability */
        std::vector<std::string> signals; /* Available signals */
        std::vector<std::string> bindings; /* Signal bindings */

        json toJson() const;          /* Serialize to JSON */
    };

    /* Summary of a physical pin */
    struct PhysicalPinSummary {
        std::string connectorId;      /* Connector ID */
        std::string pinNumber;        /* Pin number */
        std::string logicalNet;       /* Logical net name */
        std::string boardNet;         /* Board net name */
        bool exactMatch{false};       /* Exact match flag */

        json toJson() const;          /* Serialize to JSON */
    };

    /* Summary of a connector */
    struct ConnectorSummary {
        std::string id;               /* Connector ID */
        std::string role;             /* Connector role */
        std::map<std::string, PhysicalPinSummary> pins; /* Pin mappings */

        json toJson() const;          /* Serialize to JSON */
    };

    /* Cached hardware topology information */
    struct TopologyCache {
        bool loaded{false};            /* Topology loaded flag */
        fs::path root;                /* Root directory */
        std::vector<fs::path> files;  /* Netlist files */
        std::vector<std::string> connectors; /* Connector IDs */
        std::vector<std::string> warnings; /* Loading warnings */
        std::unordered_map<std::string, std::vector<SignalBinding>> signalBindings; /* Signal bindings */
        std::unordered_map<std::string, InterfaceSummary> interfaces; /* Interface summaries */
        std::vector<std::string> npuControlSignals; /* NPU control signals */
        std::vector<std::string> npuTransportSignals; /* NPU transport signals */
        bool npuAvailable{false};     /* NPU availability */
        bool gerberLoaded{false};     /* Gerber loaded flag */
        fs::path gerberConnectorMap;  /* Gerber connector map path */
        std::unordered_map<std::string, ConnectorSummary> physicalConnectors; /* Physical connectors */
        std::vector<std::string> gerberWarnings; /* Gerber warnings */
        std::size_t componentCount{0}; /* Total component count */
    };

    /* Platform metrics and statistics */
    struct Metrics {
        uint64_t refreshCount{0};     /* Topology refresh count */
        uint64_t computePlanned{0};   /* Compute operations planned */
        uint64_t computeDispatched{0}; /* Compute operations dispatched */
        uint64_t routedCpu{0};        /* Operations routed to CPU */
        uint64_t routedNpu{0};        /* Operations routed to NPU */
        uint64_t routedHybrid{0};     /* Operations routed to hybrid */
        uint64_t rejectedDispatches{0}; /* Rejected dispatches */
        uint64_t weightHotHits{0};    /* Hot weight cache hits */
        uint64_t weightColdLoads{0};  /* Cold weight cache loads */
        uint64_t weightPromotions{0}; /* Weight promotions to hot */
        uint64_t fleetAssignments{0}; /* Fleet assignments */
        uint64_t nextDispatchId{1};   /* Next dispatch ID */
        double computeBacklog{0.0};    /* Compute backlog */
        double accelEfficiencyScore{1.0}; /* Accelerator efficiency score */
        int64_t lastDecayAtMs{0};     /* Last decay timestamp */
        std::deque<json> recentDispatches; /* Recent dispatch history */
    };

private:

    /* Weight residency tracking for hot/cold caching */
    struct WeightResidency {
        int weightBlockId{0};        /* Weight block ID */
        uint64_t totalHits{0};       /* Total access count */
        uint64_t hotHits{0};         /* Hot cache hits */
        bool hot{false};             /* Hot residency flag */
        bool presentOnSd{false};     /* Present on SD card */
        std::size_t weightBytes{0};  /* Weight block size in bytes */
        int64_t lastAccessMs{0};     /* Last access timestamp */
        int64_t promotedAtMs{0};     /* Promotion timestamp */
    };

    /* NPU lane state for parallel execution */
    struct NpuLaneState {
        int laneId{0};               /* Lane ID */
        int inflight{0};             /* Inflight operations */
        uint64_t assigned{0};        /* Total assigned operations */
        uint64_t completed{0};       /* Total completed operations */
        double ewmaLatencyMs{0.0};   /* EWMA latency in milliseconds */
        int64_t lastAssignedAtMs{0}; /* Last assignment timestamp */
    };

    TopologyCache buildTopologyLocked() const; /* Build topology (locked) */
    json buildStatusLocked() const; /* Build status (locked) */
    json buildComputePlanLocked(const json &payload); /* Build compute plan (locked) */
    void ensureNpuLanesLocked(); /* Ensure NPU lanes initialized (locked) */
    json buildFleetScheduleLocked(int shardCount, int preferredBatch) const; /* Build fleet schedule (locked) */
    json buildWeightVirtualizationViewLocked(const json &payload) const; /* Build weight view (locked) */
    void updateNpuWeightResidencyLocked(const json &result, bool accepted); /* Update weight residency (locked) */
    void updateFleetProbeLocked(const json &result, bool accepted, const json &execution); /* Update fleet probe (locked) */
    void decayBacklogLocked(int64_t nowMs); /* Decay backlog (locked) */
    void pushHistoryLocked(const json &event); /* Push event to history (locked) */

    mutable std::mutex mu_;          /* Mutex for thread safety */
    RuntimeConfig config_;          /* Runtime configuration */
    TopologyCache topology_;        /* Cached topology */
    Metrics metrics_;              /* Platform metrics */
    std::unordered_map<int, WeightResidency> weightResidency_; /* Weight residency tracking */
    std::vector<NpuLaneState> npuLanes_; /* NPU lane states */
};

} // namespace edge_platform