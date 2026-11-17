/* autonomy_stack.hpp - Autonomous system managers for cognition, optimization, and data */

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
#include "mission_experience.hpp"
#include "cross_context_memory.hpp"
#include "memory_scope.hpp"
#include "scoped_trainable_memory.hpp"
#include "mcp_client.hpp"
#include "emergency_stop.hpp"
#include "self_mind.hpp"
#include "study_unit_learn.hpp"
#include "population_layer.hpp"
#include "tiny_expert.hpp"
#include "instance_registry.hpp"
#include "addon.hpp"
#include <functional>
#include <nlohmann/json.hpp>
#include <atomic>
#include <condition_variable>

#ifdef PHOENIX_RACE_SENTINEL
#include "tests/gtest/support/race_track.hpp"
#endif

#include <deque>
#include <mutex>
#include <string>
#include <map>
#include <thread>
#include <unordered_map>
#include <vector>

/* ---------------------------------------------------------------------------
   phoenix::state_guard - serialised entry into RuntimeState's research state.
   ---------------------------------------------------------------------------
   2026-11-17 heap-corruption fix (0xC0000374, ntdll+0xff489).

   RuntimeState owns ONE mutex (stateMu_) that serialises every read/write of
   its mutable research state: MemeGraph graph_, TensorEngine tensor_,
   PatternMatrix pattern_ and Params params_.  The raw helpers
   RuntimeState::mapWordsToMemesOrdered() / RuntimeState::runPropagation()
   assume the caller already holds that mutex - processInput() and
   learnFromDialog() do.

   OnlineResearcher::fallback() lives outside RuntimeState and re-enters that
   same state (it maps query words and runs a propagation to suggest memes).
   It used to do so with NO lock at all, so with autonomyLoop.parallelMissions
   several mission threads executed PatternMatrix::rebuild() /
   GraphTensorBridge::buildMultiOrderCSR() - both of which mutate the matrix /
   bridge nlohmann::json trees - at the same time and corrupted the heap.  The
   2026-11-17 08:12, 08:14 and 08:15 dumps each show 2-3 threads simultaneously
   inside PatternMatrix::rebuild() reached through this fallback.

   ResearchStateGuard/withResearchStateLock is the only sanctioned way for a
   collaborator outside RuntimeState to enter the region, and it shares the
   RuntimeState mutex (no second lock, so it also excludes processInput()'s
   propagation).  Kept header-only so the gtest can pin its mutual-exclusion
   and exception-safety contract without linking the main.cpp unity TU. */
namespace phoenix {
namespace state_guard {

class ResearchStateGuard {
public:
  explicit ResearchStateGuard(std::mutex &mu) : lock_(mu) {}
  ResearchStateGuard(const ResearchStateGuard &) = delete;
  ResearchStateGuard &operator=(const ResearchStateGuard &) = delete;
  ResearchStateGuard(ResearchStateGuard &&) = delete;
  ResearchStateGuard &operator=(ResearchStateGuard &&) = delete;

  bool owns_lock() const { return lock_.owns_lock(); }
  void unlock() { lock_.unlock(); }

private:
  std::unique_lock<std::mutex> lock_;
};

/* Runs `fn` while holding the RuntimeState research-state mutex.  Callback
   form keeps the region exception-safe: the guard releases the mutex on every
   exit path, so a throwing propagation cannot wedge every other research /
   chat path. */
template <class Fn>
auto withResearchStateLock(std::mutex &mu, Fn &&fn) -> decltype(fn()) {
  ResearchStateGuard guard(mu);
  return fn();
}

} // namespace state_guard
} // namespace phoenix

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
    json configureAgi(const json &payload);        /* Configure the AGI template. */
    json agiPlan(const json &payload = json::object()); /* MPC; scoped hot path. */
    json ingestAgiTransition(const json &payload); /* Feed a scoped (z,a,z') transition. */
    json registerAgiAction(const json &payload);     /* Register an executable capability. */
    json listAgiActions() const;                     /* List registered actions. */
    void setAgiActionExecutor(AgiActionExecutor executor); /* C++ hook for real dispatch. */
    json executeAgiActionByName(const std::string &name, const json &context); /* Execute a registered action by name. */

    /* v7.0 subconscious profile (optional, config subconscious.*) */
    json configureSubconscious(const json &payload); /* Configure temperament/tuning. */

    /* Self-mind: dialogue bandit, inner thought, mirror/Turing probes.
       Thoughts are not answers. The chat causal string is left untouched. */
    json configureSelfMind(const json &payload);
    json ingestSelfCorpus(const json &payload);
    json selfMindStatus() const;
    json selfProbe();
    json tickSelfThought();

    /* Deepened unit learning, capped population layer, tiny goal expert.
       All three stay off the chat causal string. Population and the expert
       default off; unit learning follows study.unitLearn.enabled. */
    json configureStudy(const json &payload);
    json learnUnits(const json &payload);
    json runPopulationGoal(const json &payload);
    json summonTinyExpert(const json &payload);
    /* Tool-side live training. The caller detected a source-edit situation.
       Unit learning, the live-turn self-model, and one tiny-expert fit run
       as a side channel. Nothing returned here is a prompt. */
    json noteFieldOpportunity(const std::string &text);
    /* Project-path specialist score. Not a prompt and not a diff. */
    json projectPathScore(const std::string &text) const;

    /* v7.0 mission layer (optional, config mission.*): Meeseeks-style goal
       pressure + instance reproduction.  In-process lifecycle state machine;
       no separate runtime layer. */
    json assignMission(const json &payload);     /* Spawn/refresh ONE concurrent mission (id = key). */
    json missionStatus() const;                  /* ALL missions + pressure (multi-mission view). */
    json reportMissionOutcome(const json &payload); /* {goalAchieved, missionId?} ends pain. */
    json spawnMissionChild(const json &payload); /* Mutated child genome ({missionId?}). */
    /* v8.3 self-verdict: a helper box declared its sub-task done; the loop
       stops rotating it (its workspace stays readable for the parent). */
    json markMissionChildDone(const json &payload); /* {childId} */
    /* v8.x C3 evolution lineage: auditable history + human reset. */
    json missionLineage() const;      /* all missions (audit) */
    json resetMissionLineage();          /* clears every mission's lineage */
    /* default mission id for legacy single-mission callers (files route etc.) */
    std::string defaultMissionId() const;
    /* v8.x unified pipeline: per-context emotion appraisal (the pre-processing
       stage the mission deliberator runs before the GNN outline + main
       inference), returning {emotionTensor, inferenceOptions, benefitHarm}. */
    json evaluateInstinctsFor(const std::string &contextTag);
    /* current pressure of one mission (drives the steady-output loop). */
    float missionPressureFor(const std::string &missionId);
    /* Heritable genome of the mission/box that owns `scope`
       (scope = "<missionId>" or "<missionId>/children/<childId>").
       Returns {} when the mission/box is not found.  The gateway worker
       reads tokensPerTick from here so the per-tick generation budget is
       the genome's, not a global constant. */
    json missionGenomeFor(const std::string &scope);

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
    json setMissionDeliverable(const json &payload);    /* {text} replaces. */
    /** Snapshot of mission context-packing options (ctx / summary mode / GNN). */
    json missionContextOptions() const;
    void setMissionGnnSummary(const std::string &summary);
    void setMissionGnnSummaryFor(const std::string &missionId,
                                 const std::string &summary);
    std::string missionGnnSummaryFor(const std::string &missionId) const;

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
    json startAutonomyLoop(const json &opts = json::object()); /* spawn; never join */
    json stopAutonomyLoop(); /* flag+abort only; never join a long iterate */
    json autonomyLoopStatus() const;
    /* Global loop tick: one generation is one loop round.  The trace uses
       this instead of a per-scope ++ so all carriers in a round share the
       generation even if one scope was re-assigned earlier. */
    uint64_t loopTickCount() const {
      return loopTickCount_.load(std::memory_order_relaxed);
    }

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

    /* Person model under the chat window. Default on; chat prompts stay user-only. */
    phoenix::selfmind::SelfMind selfMind_;

    /* Study line: unit graph, population above instances, local tiny expert. */
    phoenix::study::UnitLearner unitLearner_;
    phoenix::study::PopulationLayer population_;
    phoenix::study::TinyExpert tinyExpert_;
    std::string lastFieldGoal_;
    /* v10.1: the expert used to train only once per process (lastFieldGoal_
       was never re-armed).  Train on a CHANGED field goal, throttled to one
       run per 5 minutes so trainProjectTree never hogs the mission loop. */
    long long lastFieldTrainMs_{0};
    /* Last field state fed to the population layer (guarded by mu_). */
    json lastFieldState_ = json::object();

    /* v10.2 asynchronous field training.  Expert training is pure CPU work
       (corpus scan + SGD epochs) and used to run inline in the caller, so a
       training pass blocked the mission loop - and therefore the GPU - for
       its whole duration.  It now runs on a single background worker: the
       caller only enqueues, so CPU training and GPU decoding overlap. */
    void ensureTrainingWorker();
    void stopTrainingWorker();
    void trainingWorkerLoop();
    void enqueueExpertTraining(const std::string &slice);
    /* queued-but-not-started passes (status() is const, so it locks). */
    int trainingPending() const {
        std::lock_guard<std::mutex> lk(trainMu_);
        return static_cast<int>(trainQueue_.size());
    }
    std::thread trainThread_;
    mutable std::mutex trainMu_;
    std::condition_variable trainCv_;
    std::deque<std::string> trainQueue_;
    bool trainStop_{false};
    std::atomic<bool> trainBusy_{false};
    std::atomic<uint64_t> trainJobsDone_{0};
    /* guarded by mu_ */
    json lastExpertTrain_ = json::object();

    /* v7.0 mission layer (optional) */
    bool missionEnabled_{false};
    float missionMutationRate_{0.05f};
    size_t missionMaxReplicaDepth_{0};       /* v8.x config default for new missions */
bool missionEvolutionEnabled_{false};      /* v8.x C3 gate (applied on assign) */
    double missionAnnealPerGen_{0.0};       /* v10.1: mutation-factor anneal per gen */
    bool missionEarlyStop_{false};          /* v10.1: stall after 2 flat adoptions */
size_t missionMaxReplicas_{4};            /* guardrail on free replication */

    /* v7.0 human interjection queue (插话).  scopeKey isolates delivery: a
       tagged interjection is consumed only by an iterate() running under the
       same MemoryScope; an empty scopeKey is legacy-global (any tick). */
    struct Interjection {
        uint64_t atMs{0};
        std::string scopeKey;   /* "chat:<sid>" / "mission:<id>" / "" = global */
        std::string text;
    };
    std::vector<Interjection> interjections_;

    /* v7.0 long-term autonomous loop (heartbeat + persistence) */
    bool loopEnabled_{false};
    std::thread loopThread_;
    std::atomic<bool> loopStop_{true};
    /* True only while loopRun() is executing on loopThread_.  joinable()
       stays true after the thread exits until join(); never use joinable()
       alone as "running". */
    std::atomic<bool> loopRunning_{false};
    /* Bumped on every start. A detached leftover iterate must not keep
       ticking after a replacement thread is spawned. */
    std::atomic<uint64_t> loopGeneration_{0};
    int loopIntervalSec_{10};
    int loopMaxStepsPerTick_{8};
    int loopPersistEveryTicks_{5};
    std::string loopPersistPath_{"runtime_store/autonomy_state.json"};
    std::atomic<uint64_t> loopTickCount_{0};
    std::atomic<int64_t> loopLastTickAtMs_{0};
    /* Stall watchdog: the loop thread publishes its current stage + the time
       it entered it; a separate watchdog thread logs [mission-stall] when a
       stage exceeds autonomyLoop.stallWarnSec (default 1800s).  The 4h speed
       baseline froze silently right after the deliverable crossed
       minDeliverableChars - stage markers make the next hang self-naming. */
    std::atomic<int> loopStage_{0};
    std::atomic<int64_t> loopStageAtMs_{0};
    std::atomic<uint64_t> loopStageTick_{0};
    std::thread loopWatchdogThread_;
    std::atomic<bool> loopWatchdogStop_{false};
    int loopStallWarnSec_{1800};
    void setLoopStage(int stage);
    static const char *loopStageName(int stage);
    void ensureLoopWatchdog();
    uint64_t safetyRegId_{0}; /* entry in the system instance registry */
    bool safetyRegistered_{false};
    void loopRun(uint64_t gen);
    void ensureHeartbeatSession();
    void registerWithSafetyRegistry();
    void unregisterFromSafetyRegistry();
    /* Caller holds mu_. Point defaultMissionId_ at a Running mission,
       or clear it so observe/file do not keep reading a completed draft. */
    void retargetDefaultMissionLocked();
    /* v10.1: config owns the enable switches / paths, persisted state owns
       the learned parameters.  Called after importState() so a state file
       written while a layer was disabled cannot switch it off again. */
    void reapplyStudyConfig();
    /* Config owns the behaviour switches too (agi/mission/subconscious).
       Must run at the END of importState: the state restore of those three
       flags happens after the study re-apply and used to overwrite it. */
    void reapplyBehaviourSwitches();
    /* Apply one study.* payload.  afterRestore keeps the restored population
       size (a size change reseeds and would discard the individuals). */
    void applyStudyConfigPayload(const json &cfg, bool afterRestore);
    /* guarded by mu_ */
    json studyCfgCache_ = json::object();
    /* v8.x concurrent missions: independent lifecycles keyed by mission id.
       Each task owns its state machine / deliverable / children / lineage;
       the ONLY shared layers are cross-context ones (AGI learner, genome
       baseline, GNN graph, experience stores) - see context_isolation.md.
       defaultMissionId_ = most recently assigned (legacy single-mission
       callers keep working unmodified). */
    std::map<std::string, phoenix::mission::MissionLifecycle> missions_;
    std::string defaultMissionId_;
    /* resolve payload missionId: explicit id -> that mission, else the
       default (last assigned).  Caller must hold mu_. */
    phoenix::mission::MissionLifecycle *missionByIdLocked(const std::string &id);
    /* aggregate status of every mission (multi-mission view). */
    nlohmann::json missionsStatusLocked() const;
  MissionDeliberator missionDeliberator_;
  int loopDeliberateMaxTokens_{48}; /* CPU: ~300ms/tok -> 48 tok ~15s decode */
  int loopChildDeliberateMaxTokens_{48}; /* helper boxes: same budget each */
  size_t loopMaxChildrenPerTick_{0}; /* 0 = run ALL helper boxes each tick */
  size_t childRoundRobin_{0};        /* rotates start index when budget < N */
  /* v8.x C2 self-pause, per mission (concurrent missions pause independently). */
  std::map<std::string, int> missionPauseTicks_;
  int loopMaxMissionsPerTick_{0};      /* 0 = advance ALL running missions each tick */
  /* Context packing (sliding window + pinned summary / optional GNN). */
  int missionCtxTokens_{4096};                 /* 4096 or 16384 typical */
  std::string missionContextPack_{"full_and_summary"}; /* summary | full_and_summary */
  bool missionIncludeGnnSummary_{false};
  std::string missionGnnSummary_;              /* legacy: default mission only */
  std::map<std::string, std::string> missionGnnSummaries_; /* per mission id */
    phoenix::mission::MissionGenome missionGenome_;

    /* v7.0 MCP compatibility (optional) */
    bool mcpEnabled_{false};
    phoenix::mcp::McpManager mcpManager_;
#ifdef PHOENIX_RACE_SENTINEL
    race_sentinel::RaceTrack<
        std::map<std::string, std::pair<std::string, std::string>>>
        mcpActionMap_{"autonomy_stack::mcpActionMap_"};
#else
    std::map<std::string, std::pair<std::string, std::string>> mcpActionMap_;
#endif
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
