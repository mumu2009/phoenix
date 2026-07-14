/* autonomy_stack.hpp - Autonomous system managers for cognition, optimization, and data
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

#include <nlohmann/json.hpp>
#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace autonomy {

using json = nlohmann::json;

/* Build seed payload for cognition autonomy initialization */
json buildCognitionAutonomySeedPayload(const std::string &sessionId,
                                       const std::string &mission,
                                       double uncertainty = 0.68);

/* Manager for transformer cluster routing and load balancing */
class TransformerClusterManager {
public:
    TransformerClusterManager();
    json status() const; /* Get cluster status */
    json updateNodes(const json &payload); /* Update node list */
    json pickNode(const json &request); /* Select best node for request */
    json feedback(const json &payload); /* Process request feedback */

private:
    /* Node in transformer cluster */
    struct Node {
        std::string id;              /* Node identifier */
        std::string endpoint;       /* Node endpoint URL */
        double weight{1.0};         /* Routing weight */
        int inflight{0};            /* Inflight requests */
        uint64_t routed{0};        /* Total routed requests */
        uint64_t success{0};       /* Successful requests */
        uint64_t failure{0};       /* Failed requests */
        double emaLatencyMs{120.0}; /* EWMA latency in milliseconds */
        bool healthy{true};         /* Health status */
        std::string lastError;      /* Last error message */
    };

    mutable std::mutex mu_;         /* Mutex for thread safety */
    std::vector<Node> nodes_;      /* Cluster nodes */
    std::atomic<uint64_t> routeSeq_{0}; /* Route sequence counter */
};

/* Manager for web crawling and spider autonomy */
class SpiderAutonomyManager {
public:
    SpiderAutonomyManager();
    json status() const; /* Get spider status */
    json adapt(const json &payload, const json &monitoring); /* Adapt crawling strategy */

private:
    mutable std::mutex mu_;         /* Mutex for thread safety */
    bool enabled_{true};           /* Spider enabled */
    int iteration_{0};            /* Adaptation iteration */
    int crawlDepth_{3};            /* Maximum crawl depth */
    int maxPages_{256};           /* Maximum pages to crawl */
    int intervalSec_{30};         /* Crawl interval in seconds */
    bool selfHeal_{true};         /* Self-healing enabled */
    std::vector<std::string> banPatterns_; /* URL ban patterns */
    int64_t lastAdaptAtMs_{0};    /* Last adaptation timestamp */
};

/* Manager for performance optimization and tuning */
class OptimizerAutonomyManager {
public:
    OptimizerAutonomyManager();
    json status() const; /* Get optimizer status */
    json iterate(const json &payload, const json &monitoring, const json &transformerParams); /* Run optimization iteration */
    json applyPerfProfile(const json &payload, const json &transformerParams); /* Apply performance profile */
    json proposeGnnUpgrade(const json &payload, const json &transformerParams); /* Propose GNN upgrade */
    json proposeTransformerUpgrade(const json &payload, const json &transformerParams); /* Propose transformer upgrade */
    json modernizeTransformer(const json &payload, const json &transformerParams); /* Modernize transformer architecture */

private:
    mutable std::mutex mu_;         /* Mutex for thread safety */
    bool enabled_{true};           /* Optimizer enabled */
    int iteration_{0};            /* Optimization iteration */
    std::string devicePolicy_{"auto"}; /* Device selection policy */
    int workerProcesses_{1};      /* Number of worker processes */
    int workerThreads_{2};        /* Threads per worker */
    bool useGpu_{false};          /* Use GPU acceleration */
    bool useNpu_{false};          /* Use NPU acceleration */
    int64_t lastIterAtMs_{0};     /* Last iteration timestamp */
};

/* Manager for cognitive autonomy and self-reflection */
class CognitionAutonomyManager {
public:
    CognitionAutonomyManager();
    json status() const; /* Get cognition status */
    json observe(const json &payload, const json &worldState); /* Observe world state */
    json iterate(const json &payload, const json &worldState); /* Run cognition iteration */
    json session(const std::string &sessionId) const; /* Get session state */
    json exportState() const; /* Export cognition state */
    json importState(const json &state); /* Import cognition state */

private:
    mutable std::mutex mu_;         /* Mutex for thread safety */
    bool enabled_{true};           /* Cognition enabled */
    bool backgroundEnabled_{true}; /* Background iteration enabled */
    int iteration_{0};            /* Cognition iteration */
    int observations_{0};         /* Total observations */
    int backgroundEvery_{4};      /* Background iteration frequency */
    double uncertaintyThreshold_{0.45}; /* Uncertainty threshold */
    double reflectionThreshold_{0.60}; /* Reflection threshold */
    int64_t lastIterAtMs_{0};     /* Last iteration timestamp */
    std::unordered_map<std::string, json> sessions_; /* Session states */
};

/* Manager for dataset catalog and governance */
class DatasetCatalogManager {
public:
    DatasetCatalogManager();
    json status() const; /* Get catalog status */
    json list() const; /* List all datasets */
    json registerDataset(const json &payload); /* Register new dataset */
    json activate(const json &payload); /* Activate dataset */
    json updateCleaningProfile(const json &payload); /* Update cleaning profile */
    json collectData(const json &payload); /* Collect data */
    json governance() const; /* Get governance info */

private:
    void load(); /* Load catalog from disk */
    void persist() const; /* Persist catalog to disk */
    static std::string nowIso(); /* Get current ISO timestamp */

    mutable std::mutex mu_;         /* Mutex for thread safety */
    std::string filePath_;         /* Catalog file path */
    json catalog_;                /* Dataset catalog */
};

} // namespace autonomy
