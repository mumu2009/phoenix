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

#include "external_mixed_modal_io.hpp"
#include "instinct.hpp"
#include "primal_sensation.hpp"
#include "prompt_split.hpp"
#include "active_inference.hpp"
#include "subconscious_profile.hpp"
#include "agi_action_registry.hpp"
#include "mission_lifecycle.hpp"
#include "mission_workspace.hpp"
#include "mcp_client.hpp"
#include "emergency_stop.hpp"
#include "instance_registry.hpp"
#include "addon.hpp"
#include <functional>
#include <nlohmann/json.hpp>
#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace autonomy {

using PrimalSensationEngine = phoenix::primal::PrimalSensationEngine;
using InstinctEngine = phoenix::instinct::InstinctEngine;
using PromptComposer = phoenix::prompt::PromptComposer;
using MixedModalInputBuffer = phoenix::io::MixedModalInputBuffer;
using MixedModalOutputQueue = phoenix::io::MixedModalOutputQueue;
using MixedModalChannelRegistry = phoenix::io::MixedModalChannelRegistry;

using json = nlohmann::json;

/* Executor hook for real capability dispatch (set by the gateway). */
using AgiActionExecutor =
    std::function<json(const phoenix::agi::AgiActionSpec &spec, const json &context)>;

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
    ~CognitionAutonomyManager();
    json status() const; /* Get cognition status */
    json observe(const json &payload, const json &worldState); /* Observe world state */
    json iterate(const json &payload, const json &worldState); /* Run cognition iteration */
    json session(const std::string &sessionId) const; /* Get session state */
    json exportState() const; /* Export cognition state */
    json importState(const json &state); /* Import cognition state */

    /* v7.0 primal sensation / instinct layer */
    json ingestSensation(const json &payload); /* Ingest a primal sensation */
    json evaluateInstincts(); /* Run benefit-harm evaluation */

    /* v7.0 active inference / MPC (optional, config agi.*) */
    json configureAgi(const json &payload);        /* Configure the AGI controller. */
    json agiPlan();                                /* MPC action selection. */
    json ingestAgiTransition(const json &payload); /* Feed a real (z,a,z') transition. */
    json registerAgiAction(const json &payload);     /* Register an executable capability. */
    json listAgiActions() const;                     /* List registered actions. */
    void setAgiActionExecutor(AgiActionExecutor executor); /* C++ hook for real dispatch. */
    json executeAgiActionByName(const std::string &name, const json &context); /* Execute a registered action by name. */

    /* v7.0 subconscious profile (optional, config subconscious.*) */
    json configureSubconscious(const json &payload); /* Configure temperament/tuning. */

    /* v7.0 mission layer (optional, config mission.*): Meeseeks-style goal
       pressure + instance reproduction.  In-process lifecycle state machine;
       no separate runtime layer. */
    json assignMission(const json &payload);     /* Spawn THIS instance on one goal. */
    json missionStatus() const;                  /* Current mission + pressure. */
    json reportMissionOutcome(const json &payload); /* {goalAchieved:bool} ends pain. */
    json spawnMissionChild(const json &payload); /* Mutated child genome (heredity). */
    /* v8.3 self-verdict: a helper box declared its sub-task done; the loop
       stops rotating it (its workspace stays readable for the parent). */
    json markMissionChildDone(const json &payload); /* {childId} */

    /* v8.0 mission worker: the gateway registers an LLM-backed deliberator.
       While a mission is Running the heartbeat calls it OUTSIDE the manager
       lock (slow LLM replies must not stall interject/status); its output
       accumulates in the mission deliverable, which is the work product the
       human supervisor reads and judges.  Optional: without it the loop is
       pure introspection and produces no text deliverables. */
    using MissionDeliberator = std::function<std::string(
        const std::string &goal, const std::string &deliverable, int maxTokens,
        const std::string &scope)>; /* scope = workspace sub-path (missionId,
                                       or missionId/children/<childId>) */
    void setMissionDeliberator(MissionDeliberator fn);
    json appendMissionDeliverable(const json &payload); /* {text} appends. */
    /** Snapshot of mission context-packing options (ctx / summary mode / GNN). */
    json missionContextOptions() const;
    void setMissionGnnSummary(const std::string &summary);

    /* v7.0 MCP compatibility (optional, config mcp.*): launch external MCP
       servers (JSON-RPC over stdio) and expose their tools to the planner as
       AGI actions with category "mcp".  Mainstream plugin-market bridge. */
    json configureMcp(const json &payload);  /* {enabled, servers:[...]} */
    json listMcpTools() const;               /* aggregated tool snapshot */
    json callMcpTool(const json &payload);   /* {server, tool, arguments} */

    /* v7.0 prompt split */
    json composePrompt(const json &payload); /* Compose system+memory+user prompt */

    /* v7.0 human interjection (插话): inject an instruction mid-lifecycle
       without ending the mission.  Optionally amend the mission goal. */
    json interject(const json &payload);       /* {text, sessionId?, amendGoal?} */

    /* v7.0 long-term autonomous loop (optional, config autonomyLoop.*):
       an internal heartbeat that runs the plan/act/observe/learn cycle
       WITHOUT external messages, and persists the evolved state to disk so
       evolution survives restarts. */
    json configureAutonomyLoop(const json &payload); /* {enabled, intervalSec, ...} */
    json startAutonomyLoop();                       /* spawn the heartbeat thread */
    json stopAutonomyLoop();                        /* stop and join */
    json autonomyLoopStatus() const;

    /* v7.0 external mixed-modal I/O */
    json ingestMixedModalPacket(const json &payload); /* Accept external mixed-modal input */
    json pretrainSpeechConcept(const json &payload); /* Persistently align an audio packet with its transcript */
    json emitMixedModalOutput(const json &payload); /* Adapt a semantic unit to a requested external modality */
    json drainMixedModalOutputs(const json &payload); /* Retrieve outbound mixed-modal packets */

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

    /* v7.0 primal sensation / instinct layer */
    phoenix::primal::PrimalSensationEngine sensationEngine_;
    phoenix::instinct::InstinctEngine instinctEngine_;
    std::string lastBenefitHarmBias_;

    /* v7.0 active inference / MPC (optional) */
    bool agiEnabled_{false};
    double agiPragW_{1.0}, agiIntrinW_{1.0}, agiEpistW_{0.25};
    double agiAlpha_{0.05}, agiGamma_{0.9};
    size_t agiConsolidateEvery_{16};
    bool agiAdaptiveExploration_{true};
    phoenix::agi::ActiveInferenceController agiController_;
    std::vector<float> agiLatentState_;  /*!< last observed latent state (z). */
    std::string lastAgiAction_;          /*!< last chosen action (for the next transition). */

    /* v7.0 subconscious profile (optional) */
    bool subconsciousEnabled_{false};
    phoenix::subconscious::SubconsciousProfile subProfile_;

    /* v7.0 mission layer (optional) */
    bool missionEnabled_{false};
    float missionMutationRate_{0.05f};
    size_t missionMaxReplicas_{4};            /* guardrail on free replication */

    /* v7.0 human interjection queue (插话) */
    std::vector<std::pair<uint64_t, std::string>> interjections_;

    /* v7.0 long-term autonomous loop (heartbeat + persistence) */
    bool loopEnabled_{false};
    std::thread loopThread_;
    std::atomic<bool> loopStop_{true};
    int loopIntervalSec_{10};
    int loopMaxStepsPerTick_{8};
    int loopPersistEveryTicks_{5};
    std::string loopPersistPath_{"runtime_store/autonomy_state.json"};
    std::atomic<uint64_t> loopTickCount_{0};
    std::atomic<int64_t> loopLastTickAtMs_{0};
    uint64_t safetyRegId_{0}; /* entry in the system instance registry */
    bool safetyRegistered_{false};
    void loopRun();
    void ensureHeartbeatSession();
    void registerWithSafetyRegistry();
    void unregisterFromSafetyRegistry();
    phoenix::mission::MissionLifecycle mission_;
  MissionDeliberator missionDeliberator_;
  int loopDeliberateMaxTokens_{128}; /* smaller chunks = higher success rate on RDK */
  int loopChildDeliberateMaxTokens_{128}; /* helper boxes: same budget each */
  size_t loopMaxChildrenPerTick_{0}; /* 0 = run ALL helper boxes each tick */
  size_t childRoundRobin_{0};        /* rotates start index when budget < N */
  /* Context packing (sliding window + pinned summary / optional GNN). */
  int missionCtxTokens_{4096};                 /* 4096 or 16384 typical */
  std::string missionContextPack_{"full_and_summary"}; /* summary | full_and_summary */
  bool missionIncludeGnnSummary_{false};
  std::string missionGnnSummary_;              /* last known GNN/graph summary text */
    phoenix::mission::MissionGenome missionGenome_;

    /* v7.0 MCP compatibility (optional) */
    bool mcpEnabled_{false};
    phoenix::mcp::McpManager mcpManager_;
    std::map<std::string, std::pair<std::string, std::string>> mcpActionMap_;

    /* v7.0 AGI action space: real capabilities beyond the instinct verbs. */
    phoenix::agi::AgiActionRegistry agiActionRegistry_;
    AgiActionExecutor agiActionExecutor_;
    std::shared_ptr<addon::AddonManager> addonManager_; /* default tool dispatch */
    std::vector<std::string> goals_;                   /* goal_advance backlog */

    void registerDefaultAgiActions();
    nlohmann::json executeAgiAction(const phoenix::agi::AgiActionSpec &spec,
                                    const nlohmann::json &context);

    /* v7.0 prompt split */
    phoenix::prompt::PromptComposer promptComposer_;

    /* v7.0 external mixed-modal I/O */
    phoenix::io::MixedModalInputBuffer inputBuffer_;
    phoenix::io::MixedModalOutputQueue outputQueue_;
    phoenix::io::MixedModalChannelRegistry channelRegistry_;
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
