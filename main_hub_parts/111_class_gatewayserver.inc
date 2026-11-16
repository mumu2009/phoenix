// Auto-generated from main.cpp
// Segment: class_GatewayServer
// Original lines: 13037-13883

#include "peer_hot_update.hpp"

class GatewayServer : public IGatewayServer {
public:
  /* Forward decl only - definition is out-of-line after GatewayServer closes
     (see 116_section_tail.inc).  A second in-class definition would collide. */
  std::string buildMissionOutline(const std::string &goal);
  /** Pretrain meme graph + memory/emotion from robots/wikitext (unit-sequence
   *  ingest, not text into infer).  Returns documents ingested this call. */
  int seedGnnGraphFromRobotsCorpus(int maxDocs, const char *sourceTag);
  void attachEmotionSystem(
      std::shared_ptr<phoenix::emotion::EmotionSystem> emo) {
    emotionSystem_ = std::move(emo);
  }

  // v7.4 world model / tinyllama integration
  void setWorldModelDatabase(std::shared_ptr<Database079> db) {
    if (!db || worldModel_)
      return;
    auto kvm = db->createStore("kvm");
    auto meme = db->createStore("meme_graph");
    auto session = db->createStore("session");
    if (logger_.enabled()) {
      std::string msg = "[gateway-world] setWorldModelDatabase: kvm=" + std::to_string(kvm ? 1 : 0)
                        + " meme=" + std::to_string(meme ? 1 : 0)
                        + " session=" + std::to_string(session ? 1 : 0);
      logger_.log(LoggerCXX::Type::MONITORING, msg);
    }
    if (kvm && meme && session) {
      worldModel_ = std::make_shared<world_model::WorldModelStore>(kvm, meme, session);
      if (logger_.enabled())
        logger_.log(LoggerCXX::Type::MONITORING, "[gateway-world] world model store attached");
    } else if (logger_.enabled()) {
      logger_.log(LoggerCXX::Type::WARNING, "[gateway-world] failed to create one or more world model stores");
    }
  }

  void ingestWorldEvidence(const std::string &sessionId,
                           const std::string &text,
                           const std::string &graphSummary,
                           const std::string &source = "gateway-chat") {
    if (!worldModel_)
      return;
    try {
      /* Salad quarantine (write path): a collapsed generation must never
         become world-model evidence — the WM serves its graphContext back
         into the next chat prompt, so one bad deposit self-reinforces
         (observed: "peer views - g1_ai1 vaults buildings romanesque..."
         looping through world:session/*).  Drop the whole evidence when the
         user text itself is collapsed; strip a collapsed graphSummary but
         keep the clean user text. */
      if (phoenix::mission::looksLikeCollapsedProse(text))
        return;
      nlohmann::json evidence;
      evidence["sessionId"] = sessionId;
      evidence["modality"] = "text";
      evidence["text"] = text;
      evidence["graphSummary"] =
          phoenix::mission::looksLikeCollapsedProse(graphSummary)
              ? std::string()
              : graphSummary;
      evidence["metadata"] = nlohmann::json{{"source", source}};
      /* Chat must not inherit the default running mission.  Only an
         explicit mission MemoryScope tags evidence with missionId. */
      const auto scope = phoenix::memory::currentMemoryScope();
      if (scope.kind == phoenix::memory::MemoryKind::Mission && scope.valid()) {
        evidence["metadata"]["missionId"] = scope.id;
        /* Mission scene state lives under "mission:<id>", never under the
           chat session key — a chat turn must not read back mission
           evidence via sessionState(sessionId). */
        evidence["sessionId"] = scope.key();
      }
      worldModel_->ingestEvidence(evidence);
    } catch (...) {
    }
  }

  json chatWithTinyllama(const std::string &text,
                         const std::string &graphContext,
                         int maxTokens,
                         const json &options = json::object()) {
    int summaryTimeoutMs = std::max(1000, config_.tinyllamaRequestTimeoutMs);
    if (config_.tinyllamaBaseUrl.empty()) {
      return json{{"ok", false}, {"error", "tinyllamaBaseUrl not configured"},
                  {"provider", "tinyllama"}};
    }
    // Split llama-server exposes /phx/*, not Ollama /api/chat.
    json r = phoenix::v7::llamaSplitChat(
        config_.tinyllamaBaseUrl, summaryTimeoutMs, config_.tinyllamaModel,
        text, graphContext, maxTokens, options);
    if (config_.httpLog)
      std::cout << "[chat] tinyllama(split) ok=" << r.value("ok", false)
                << " replyLen=" << r.value("reply", std::string()).size()
                << std::endl;
    return r;
  }

  GatewayServer(std::shared_ptr<ControllerPoolBase> pool,
                std::shared_ptr<ISnapshotManager> snapshots,
                std::shared_ptr<IRedisSynchronizer> redisSync,
                std::shared_ptr<RotationManager> rotation,
                std::shared_ptr<IStudyEngine> study,
                std::shared_ptr<Database079> db079, const Config &config)
      : pool_(std::move(pool)), snapshots_(std::move(snapshots)),
        redis_(std::move(redisSync)), rotation_(std::move(rotation)),
        study_(std::move(study)), db_(std::move(db079)), config_(config) {
    startedAt_ = std::chrono::steady_clock::now();
    shards_ = std::make_shared<ShardManager>(pool_);
    json sparkOptions;
    sparkOptions["groupIds"] = pool_->listGroupIds();
    sparkOptions["numAI"] = config_.sparkNumAI;
    sparkOptions["rfInstances"] = std::max(3, config_.sparkNumAI);
    sparkOptions["budget"] = config_.sparkBudget;
    auto sparkFactory =
        module_mount::ModuleRegistry::instance().sparkArrayFactory();
    if (sparkFactory)
      spark_ = sparkFactory(pool_, shards_, sparkOptions);
    else
      spark_ = std::make_shared<SparkArray>(pool_, shards_, sparkOptions);
    authEnabled_ =
        phoenix::resolveConfig<bool>("auth.enabled", true, "AI_AUTH_ENABLED");
    acceptLocalToken_ =
        phoenix::resolveConfig<bool>("auth.acceptLocalToken", true, "AI_AUTH_ACCEPT_LOCAL_TOKEN");
    jwtSecret_ = phoenix::resolveConfig<std::string>(
        "auth.jwtSecret", "dev-secret-change-me", "AI_AUTH_JWT_SECRET",
        "AUTH_JWT_SECRET");
    cognitionAutonomyEnabled_ =
        phoenix::resolveConfig<bool>("main.cognitionAutonomyEnabled", true, "AI_ENABLE_COGNITION_AUTONOMY");
    /* v7.0 active inference / MPC: load agi.* config and configure the
       optional controller at startup (disabled by default). */
    {
      json agiCfg;
      agiCfg["enabled"] = phoenix::resolveConfig<bool>("agi.enabled", false, "AI_AGI_ENABLED");
      agiCfg["dim"] = phoenix::cfgOr<int>("agi.dim", 128);
      agiCfg["horizon"] = phoenix::cfgOr<int>("agi.horizon", 3);
      agiCfg["pragmaticWeight"] = phoenix::cfgOr<double>("agi.pragmaticWeight", 1.0);
      agiCfg["intrinsicWeight"] = phoenix::cfgOr<double>("agi.intrinsicWeight", 1.0);
      agiCfg["epistemicWeight"] = phoenix::cfgOr<double>("agi.epistemicWeight", 0.25);
      agiCfg["learningRate"] = phoenix::cfgOr<double>("agi.learningRate", 0.05);
      agiCfg["discount"] = phoenix::cfgOr<double>("agi.discount", 0.9);
      agiCfg["consolidateEvery"] = phoenix::cfgOr<int>("agi.consolidateEvery", 16);
      agiCfg["adaptiveExploration"] = phoenix::cfgOr<bool>("agi.adaptiveExploration", true);
      cognitionManager_.configureAgi(agiCfg);
    }
    /* v7.0 subconscious profile: load subconscious.* config (disabled by
       default; every parameter defaults to the current behaviour). */
    {
      json subCfg;
      subCfg["enabled"] = phoenix::cfgOr<bool>("subconscious.enabled", false);
      subCfg["baselineValence"] = phoenix::cfgOr<double>("subconscious.baselineValence", 0.0);
      subCfg["baselineArousal"] = phoenix::cfgOr<double>("subconscious.baselineArousal", 0.0);
      subCfg["baselineDominance"] = phoenix::cfgOr<double>("subconscious.baselineDominance", 0.0);
      subCfg["temperamentStrength"] = phoenix::cfgOr<double>("subconscious.temperamentStrength", 0.0);
      subCfg["riskAversion"] = phoenix::cfgOr<double>("subconscious.riskAversion", 1.0);
      subCfg["anticipatoryGain"] = phoenix::cfgOr<double>("subconscious.anticipatoryGain", 1.0);
      subCfg["sensationTuning"] = phoenix::cfgOr<nlohmann::json>("subconscious.sensationTuning", nlohmann::json::object());
      subCfg["instincts"] = phoenix::cfgOr<nlohmann::json>("subconscious.instincts", nlohmann::json::array());
      cognitionManager_.configureSubconscious(subCfg);
    }
    /* Self-mind: person bandit + inner thought. Chat causal text stays the
       user message; thoughts are a separate channel (selfMind.*). */
    {
      json selfCfg;
      selfCfg["enabled"] = phoenix::resolveConfig<bool>("selfMind.enabled", true, "AI_SELF_MIND_ENABLED");
      selfCfg["name"] = phoenix::cfgOr<std::string>("selfMind.name", std::string("self"));
      selfCfg["learningRate"] = phoenix::cfgOr<double>("selfMind.learningRate", 0.35);
      selfCfg["corpusPath"] = phoenix::cfgOr<std::string>("selfMind.corpusPath", std::string());
      selfCfg["learnFromLiveTurns"] = phoenix::cfgOr<bool>("selfMind.learnFromLiveTurns", false);
      selfCfg["holdoutEvery"] = phoenix::cfgOr<int>("selfMind.holdoutEvery", 4);
      cognitionManager_.configureSelfMind(selfCfg);
    }
    /* v10.1 study layer (unitLearn / population / expert).  This was never
       wired at startup, so tinyExpert_ kept its default disabled_=false and
       every field-train opportunity reported expert=skipped while the expert
       weights stayed frozen.  Configure all three from study.* config. */
    {
      json studyCfg;
      studyCfg["unitLearn"] = json{
          {"enabled", phoenix::cfgOr<bool>("study.unitLearn.enabled", true)},
          {"steps", phoenix::cfgOr<int>("study.unitLearn.steps", 3)},
          {"dim", phoenix::cfgOr<int>("study.unitLearn.dim", 16)},
          {"maxMerges", phoenix::cfgOr<int>("study.unitLearn.maxMerges", 32)}};
      studyCfg["population"] = json{
          {"enabled", phoenix::cfgOr<bool>("study.population.enabled", false)},
          {"n", phoenix::cfgOr<int>("study.population.n", 4)},
          /* v10.2: the layer must accept the 8-feature field state vector.
             Without these the gateway payload replaced the config-derived
             population block, so the restored 4x2 dims survived and every
             runGoal() was rejected with goal-dim (goalsRun stayed 0). */
          {"inputDim", phoenix::cfgOr<int>("study.population.inputDim", 8)},
          {"hidden", phoenix::cfgOr<int>("study.population.hidden", 6)},
          {"outputDim", phoenix::cfgOr<int>("study.population.outputDim", 1)},
          {"maxPopulation", phoenix::cfgOr<int>("study.population.maxPopulation", 8)},
          {"mutationScale", phoenix::cfgOr<double>("study.population.mutationScale", 0.05)},
          {"tolerance", phoenix::cfgOr<double>("study.population.tolerance", 0.2)}};
      studyCfg["expert"] = json{
          {"enabled", phoenix::cfgOr<bool>("study.expert.enabled", false)},
          {"corpusPath", phoenix::cfgOr<std::string>("study.expert.corpusPath", std::string("./runtime_store/datasets/study_expert/corpus.txt"))},
          {"weightsPath", phoenix::cfgOr<std::string>("study.expert.weightsPath", std::string("./runtime_store/experts/study_tiny/weights.json"))},
          {"hidden", phoenix::cfgOr<int>("study.expert.hidden", 16)},
          {"vocab", phoenix::cfgOr<int>("study.expert.vocab", 32)},
          {"epochs", phoenix::cfgOr<int>("study.expert.epochs", 8)}};
      cognitionManager_.configureStudy(studyCfg);
    }
    /* ---- v10.2 config hot reload -------------------------------------
       cfgOr() reads an atomic copy-on-write snapshot, so every cfgOr-based
       parameter (sampling temperatures, semantic floor/weight, recall
       diversity, summon anneal, slot counts, plugin thresholds ...) is already
       hot.  The parameters below are the ones CACHED at startup, so they need
       an explicit re-apply: AGI, subconscious, the study layers, and the
       autonomy-loop cadence.  A watcher thread stats the base config and the
       research profile every 2 s and only fires on an mtime change; a failed
       parse keeps the previous snapshot (a half-written file cannot break a
       running gateway). */
    if (phoenix::cfgOr<bool>("main.configHotReload", true)) {
      phoenix::PhoenixConfig::instance().setReloadHook([this] {
        try {
          json agiCfg;
          agiCfg["enabled"] = phoenix::cfgOr<bool>("agi.enabled", true);
          agiCfg["dim"] = phoenix::cfgOr<int>("agi.dim", 128);
          agiCfg["horizon"] = phoenix::cfgOr<int>("agi.horizon", 3);
          agiCfg["pragmaticWeight"] = phoenix::cfgOr<double>("agi.pragmaticWeight", 1.0);
          agiCfg["intrinsicWeight"] = phoenix::cfgOr<double>("agi.intrinsicWeight", 1.0);
          agiCfg["epistemicWeight"] = phoenix::cfgOr<double>("agi.epistemicWeight", 0.25);
          agiCfg["learningRate"] = phoenix::cfgOr<double>("agi.learningRate", 0.05);
          agiCfg["discount"] = phoenix::cfgOr<double>("agi.discount", 0.9);
          agiCfg["consolidateEvery"] = phoenix::cfgOr<int>("agi.consolidateEvery", 16);
          agiCfg["adaptiveExploration"] =
              phoenix::cfgOr<bool>("agi.adaptiveExploration", true);
          cognitionManager_.configureAgi(agiCfg);
          json subCfg;
          subCfg["enabled"] = phoenix::cfgOr<bool>("subconscious.enabled", true);
          subCfg["baselineValence"] =
              phoenix::cfgOr<double>("subconscious.baselineValence", 0.0);
          subCfg["baselineArousal"] =
              phoenix::cfgOr<double>("subconscious.baselineArousal", 0.0);
          subCfg["baselineDominance"] =
              phoenix::cfgOr<double>("subconscious.baselineDominance", 0.0);
          subCfg["temperamentStrength"] =
              phoenix::cfgOr<double>("subconscious.temperamentStrength", 0.0);
          subCfg["riskAversion"] =
              phoenix::cfgOr<double>("subconscious.riskAversion", 1.0);
          subCfg["anticipatoryGain"] =
              phoenix::cfgOr<double>("subconscious.anticipatoryGain", 1.0);
          cognitionManager_.configureSubconscious(subCfg);
          json studyCfg;
          studyCfg["unitLearn"] =
              json{{"enabled", phoenix::cfgOr<bool>("study.unitLearn.enabled", true)},
                   {"steps", phoenix::cfgOr<int>("study.unitLearn.steps", 3)},
                   {"dim", phoenix::cfgOr<int>("study.unitLearn.dim", 16)},
                   {"maxMerges", phoenix::cfgOr<int>("study.unitLearn.maxMerges", 32)}};
          studyCfg["population"] =
              json{{"enabled", phoenix::cfgOr<bool>("study.population.enabled", true)},
                   {"inputDim", phoenix::cfgOr<int>("study.population.inputDim", 8)},
                   {"hidden", phoenix::cfgOr<int>("study.population.hidden", 6)},
                   {"outputDim", phoenix::cfgOr<int>("study.population.outputDim", 1)},
                   {"maxPopulation", phoenix::cfgOr<int>("study.population.maxPopulation", 8)},
                   {"mutationScale", phoenix::cfgOr<double>("study.population.mutationScale", 0.12)},
                   {"tolerance", phoenix::cfgOr<double>("study.population.tolerance", 0.2)}};
          studyCfg["expert"] =
              json{{"enabled", phoenix::cfgOr<bool>("study.expert.enabled", true)},
                   {"corpusPath", phoenix::cfgOr<std::string>("study.expert.corpusPath", std::string("./runtime_store/datasets/study_expert/corpus.txt"))},
                   {"weightsPath", phoenix::cfgOr<std::string>("study.expert.weightsPath", std::string("./runtime_store/experts/study_tiny/weights.json"))},
                   {"hidden", phoenix::cfgOr<int>("study.expert.hidden", 16)},
                   {"vocab", phoenix::cfgOr<int>("study.expert.vocab", 32)},
                   {"epochs", phoenix::cfgOr<int>("study.expert.epochs", 8)}};
          cognitionManager_.configureStudy(studyCfg);
          cognitionManager_.configureAutonomyLoop(
              json{{"intervalSec", phoenix::cfgOr<int>("autonomyLoop.intervalSec", 3)},
                   {"maxMissionsPerTick",
                    phoenix::cfgOr<int>("autonomyLoop.maxMissionsPerTick", 8)},
                   {"deliberateMaxTokens",
                    phoenix::cfgOr<int>("mission.deliberateMaxTokens", 256)}});
          std::cout << "[config] hot reload applied (agi/subconscious/study/"
                       "autonomyLoop)" << std::endl;
        } catch (const std::exception &ex) {
          std::cout << "[config] hot reload apply failed: " << ex.what()
                    << std::endl;
        } catch (...) {
          std::cout << "[config] hot reload apply failed" << std::endl;
        }
      });
      std::thread([this] {
        for (;;) {
          std::this_thread::sleep_for(std::chrono::seconds(2));
          if (phoenix::inference::shutdownRequested()) return;
          try {
            if (phoenix::PhoenixConfig::instance().reloadIfChanged())
              phoenix::PhoenixConfig::instance().fireReloadHook();
          } catch (...) {
          }
        }
      }).detach();
    }
    /* v7.0 mission layer: load mission.* config (disabled by default).  When
       enabled AND a goal is configured, this instance is born as a Meeseeks:
       ONE mission whose time-increasing pain ends only when the mission is
       judged complete (reportMissionOutcome).  With no goal the layer stays
       armed but idle, and missions can be assigned at runtime. */
    {
      json missionCfg;
      missionCfg["enabled"] = phoenix::resolveConfig<bool>("mission.enabled", false, "AI_MISSION_ENABLED");
      missionCfg["id"] = phoenix::cfgOr<std::string>("mission.id", std::string());
      missionCfg["goal"] = phoenix::resolveConfig<std::string>("mission.goal", std::string(), "AI_MISSION_GOAL");
      missionCfg["deadlineSec"] = phoenix::cfgOr<double>("mission.defaultDeadlineSec", 300.0);
      missionCfg["painGainPerSec"] = phoenix::cfgOr<double>("mission.painGainPerSec", 0.01);
      missionCfg["maxPain"] = phoenix::cfgOr<double>("mission.maxPain", 1.0);
      missionCfg["pressureMode"] = phoenix::resolveConfig<std::string>("mission.pressureMode", std::string("asymptotic"), "AI_MISSION_PRESSURE_MODE");
      missionCfg["pressureHorizonSec"] = phoenix::resolveConfig<double>("mission.pressureHorizonSec", 3600.0, "AI_MISSION_PRESSURE_HORIZON_SEC");
      missionCfg["pressureTauSec"] = phoenix::resolveConfig<double>("mission.pressureTauSec", 86400.0, "AI_MISSION_PRESSURE_TAU_SEC");
      missionCfg["pressureExpr"] = phoenix::resolveConfig<std::string>("mission.pressureExpr", std::string("Pmax*tanh(t/tau)"), "AI_MISSION_PRESSURE_EXPR");
      missionCfg["pressureFloor"] = phoenix::resolveConfig<double>("mission.pressureFloor", 0.05, "AI_MISSION_PRESSURE_FLOOR");
      missionCfg["mutationRate"] = phoenix::cfgOr<double>("mission.mutationRate", 0.05);
      missionCfg["maxReplicas"] = phoenix::cfgOr<int>("mission.maxReplicas", 4);
      /* 0 = UNLIMITED task-tree depth (boxes may summon their own boxes);
         user can cap it via mission.maxReplicaDepth / AI_MISSION_MAX_REPLICA_DEPTH. */
      missionCfg["maxReplicaDepth"] = phoenix::resolveConfig<int>("mission.maxReplicaDepth", 0, "AI_MISSION_MAX_REPLICA_DEPTH");
      missionCfg["evolutionEnabled"] =
          phoenix::cfgOr<bool>("mission.evolution.enabled", false);
      cognitionManager_.assignMission(missionCfg);
      /* v8.1 mission worker: LLM-backed deliberator with a sandboxed file
         workspace AND tool calls.  Registered UNCONDITIONALLY - the mission
         layer can be armed at runtime via /api/mission/assign, and a later
         assign must not spin with an empty deliverable.
         Reply forms the model may use:
           1) plain markdown the model chose (including its own headings)
           2) {"action":write|replace|read|list|run,"path","content","find"}
           3) {"tool":"script"|"math"|...,"arguments":{...}}
           4) {"done":true,"summary":"..."} -> SELF-VERDICT */
        cognitionManager_.setMissionDeliberator(
            [this](const std::string &goal, const std::string &deliverable,
                 int maxTokens, const std::string &scope) -> std::string {
            /* v10.1 concurrency guarantee: parallel missions (autonomyLoop
               .parallelMissions) run this lambda on separate std::async
               threads.  The body touches shared gateway state (outline cache,
               experience/ccm stores, runtime graph, file workspaces) plus a
               non-thread-safe llama channel.  Two layers exist:
               - the FINE locks each shared resource already owns
                 (llamaCallMu_, outlineMu_, graphMu_, gWsCacheMu, ...);
               - this COARSE backstop, switchable with
                 autonomyLoop.missionDelibLock (default true).  Set false only
                 after the fine locks are proven sufficient; see
                 doc/v10.1/artifact_contract.md. */
            std::unique_lock<std::mutex> delibLock(missionDelibMu_, std::defer_lock);
            if (phoenix::cfgOr<bool>("autonomyLoop.missionDelibLock", true))
                delibLock.lock();
            try {
              const std::string msnId =
                  phoenix::memory::missionIdFromWorkspaceScope(scope);
              /* Deterministic force driver: the schedule is carried in the
                 goal (k=<number>); the carrier index is parsed from the
                 scope (q0..qN).  No model is called, so p is exactly k/N.
                 A per-round schedule file (runtime_store/force_schedule.txt:
                 first line = start loop tick, then one k per round) overrides
                 the goal k so three consecutive rounds can follow
                 0.375 -> 0.4375 -> 0.46875 in one run. */
              if (goal.find("FORCE-EXPERIMENT") != std::string::npos &&
                  phoenix::cfgOr<bool>("research.force.enabled", false)) {
                int scheduleK = 0;
                const int loopTick =
                    static_cast<int>(cognitionManager_.loopTickCount());
                {
                  /* One schedule line per loop round.  The first tick of a
                     round advances the round index; every carrier in that
                     tick reads the same index (no start-tick lookup).  A
                     missing trace account (the harness cleared it) resets
                     the round counter, so the measured run starts at line 0. */
                  static std::atomic<int> forceLastTick{-1};
                  static std::atomic<int> forceRound{-1};
                  {
                    std::ifstream acct(
                        "runtime_store/replica_trace/account.json");
                    if (!acct) {
                      forceRound.store(-1, std::memory_order_relaxed);
                      forceLastTick.store(-1, std::memory_order_relaxed);
                    }
                  }
                  int prev = forceLastTick.load(std::memory_order_relaxed);
                  if (loopTick != prev &&
                      forceLastTick.compare_exchange_strong(prev, loopTick)) {
                    /* First carrier of this round advances the round index;
                       every carrier in the same tick reads the same value. */
                    forceRound.fetch_add(1, std::memory_order_relaxed);
                  }
                  const int r = forceRound.load(std::memory_order_relaxed);
                  std::ifstream sf(phoenix::cfgOr<std::string>(
                      "research.force.schedulePath",
                      std::string("runtime_store/force_schedule.txt")));
                  std::string line;
                  int cur = 0;
                  while (std::getline(sf, line)) {
                    if (cur == r) {
                      scheduleK = std::atoi(line.c_str());
                      break;
                    }
                    ++cur;
                  }
                }
                if (scheduleK <= 0) {
                  const auto kp = goal.find("k=");
                  if (kp != std::string::npos)
                    scheduleK = std::atoi(goal.c_str() + kp + 2);
                }
                if (scheduleK <= 0) {
                  auto keyAt = [&](const char *key) -> size_t {
                    const std::string needle(key);
                    size_t pos = 0;
                    while ((pos = goal.find(needle, pos)) != std::string::npos) {
                      if (pos == 0 || goal[pos - 1] == ' ' ||
                          goal[pos - 1] == '\n')
                        return pos;
                      pos += needle.size();
                    }
                    return std::string::npos;
                  };
                  const auto rulePos = keyAt("rule=");
                  const auto nPos = keyAt("n=");
                  const auto pPos = keyAt("p=");
                  const auto paramPos = keyAt("param=");
                  if (rulePos != std::string::npos &&
                      nPos != std::string::npos &&
                      pPos != std::string::npos &&
                      paramPos != std::string::npos) {
                    const auto ruleEnd = goal.find_first_of(" \n", rulePos + 5);
                    const std::string rule = goal.substr(
                        rulePos + 5, ruleEnd == std::string::npos
                                         ? std::string::npos
                                         : ruleEnd - (rulePos + 5));
                    const int n = std::atoi(goal.c_str() + nPos + 2);
                    const double p = std::atof(goal.c_str() + pPos + 2);
                    const double param =
                        std::atof(goal.c_str() + paramPos + 6);
                    if (!rule.empty() && n > 0 && p >= 0.0)
                      scheduleK = phoenix::memetic::plannedEditCount(
                          rule, p, n, param);
                  }
                }
                int carrierIndex = 0;
                const auto qp = scope.rfind('q');
                if (qp != std::string::npos)
                  carrierIndex = std::atoi(scope.c_str() + qp + 1);
                const bool edit = carrierIndex < scheduleK;
                phoenix::memetic::ReplicaTick forceTick;
                forceTick.instance = scope;
                forceTick.mission = msnId;
                forceTick.replyClass = edit ? "source-edit" : "prose";
                forceTick.replyChars = edit ? 104 : 22;
                forceTick.sourceEdit = edit;
                /* Generation 0 is filtered by the series; the first measured
                   round is 1. */
                forceTick.generation = loopTick + 1;
                phoenix::memetic::appendReplicaTick(
                    "runtime_store/replica_trace", forceTick);
                return edit
                           ? "FILE active_inference.cpp\nOLD\n"
                             "  surpriseEma_ = 0.0;\nNEW\n"
                             "  surpriseEma_ = 0.0;  // deterministic force tick\n"
                           : "Reviewed: no edit.";
              }
              /* Edit boxes in a force experiment need no model call: a
                 deterministic patch makes the generation frequency exactly
                 k/N, so the force identification is not contaminated by the
                 coder's ~85% patch rate or by timeouts. */
              if (phoenix::cfgOr<bool>("research.deterministicCarriers",
                                       false) &&
                  goal.find("make one small clarifying code edit") !=
                      std::string::npos) {
                phoenix::memetic::ReplicaTick editTick;
                editTick.instance = scope;
                editTick.mission = msnId;
                editTick.replyClass = "source-edit";
                editTick.replyChars = 104;
                editTick.sourceEdit = true;
                editTick.generation = static_cast<int>(
                    cognitionManager_.loopTickCount());
                phoenix::memetic::appendReplicaTick(
                    "runtime_store/replica_trace", editTick);
                return "FILE active_inference.cpp\nOLD\n"
                       "  surpriseEma_ = 0.0;\nNEW\n"
                       "  surpriseEma_ = 0.0;  // deterministic force tick\n";
              }
              /* Review boxes need no model call: their verdict is "no
                 edit".  A deterministic prose tick keeps an N-carrier round
                 at CPU speed and keeps the force frequency honest (a review
                 tick never contributes a source edit). */
              if (phoenix::cfgOr<bool>("research.deterministicCarriers",
                                       false) &&
                  (goal.find("do not edit") != std::string::npos ||
                   goal.find("summarize") != std::string::npos)) {
                phoenix::memetic::ReplicaTick reviewTick;
                reviewTick.instance = scope;
                reviewTick.mission = msnId;
                reviewTick.replyClass = "prose";
                reviewTick.replyChars = 22;
                reviewTick.sourceEdit = false;
                reviewTick.generation = static_cast<int>(
                    cognitionManager_.loopTickCount());
                phoenix::memetic::appendReplicaTick(
                    "runtime_store/replica_trace", reviewTick);
                return "Reviewed: no edit.";
              }
              phoenix::memory::MemoryScopeGuard memGuard(
                  phoenix::memory::makeMissionScope(msnId));
              static std::mutex gDelibSkipMu;
              static std::unordered_map<std::string, int> gDelibSkipCount;
              static std::unordered_map<std::string, size_t> gDelibLastDelivAtLlm;
              static std::unordered_map<std::string, int> gDelibCacheSkipsNoGrowth;
              const std::string workspaceRoot = phoenix::resolveConfig<std::string>(
                  "mission.workspaceDir", std::string("./runtime_store/mission_workspace"),
                  "AI_MISSION_WORKSPACE");
              /* scope = workspace sub-path: missionId for the parent,
                 missionId/children/<childId> for a helper box. */
              const bool isChildBox = scope.find("/children/") != std::string::npos;
              const int configuredMinChars = isChildBox
                  ? phoenix::resolveConfig<int>("mission.childMinDeliverableChars", 200)
                  : phoenix::resolveConfig<int>("mission.minDeliverableChars", 1200);
              const int minDeliverableChars = isChildBox
                  ? configuredMinChars
                  : phoenix::mission::inferMinDeliverableChars(goal, configuredMinChars);
              const std::string deliverablePath = "deliverable.md";
              const auto listRes = phoenix::mission::workspaceExecute(
                  workspaceRoot, scope, nlohmann::json{{"action", "list"}});
              const auto readRes = phoenix::mission::workspaceExecute(
                  workspaceRoot, scope,
                  nlohmann::json{{"action", "read"}, {"path", deliverablePath}});
              std::string currentFile = readRes.value("ok", false)
                                            ? readRes.value("content", std::string())
                                            : std::string();
              const bool peerEvolveGoal =
                  goal.find("PEER-EVOLVE") != std::string::npos;
              const bool selfModGoal =
                  goal.find("SELF-MOD") != std::string::npos;
              const bool codeMission =
                  !selfModGoal &&
                  goal.find("ALGO-OPT") != std::string::npos;
              /* Plain-continuation worker: the reply is prose appended to
                 the document, with no JSON/tool/plan parsing.  This makes
                 the per-tick output length a function of num_predict, so
                 the genome's tokensPerTick is the only live lever. */
              const bool plainTextOnly =
                  phoenix::cfgOr<bool>("mission.plainTextOnly", false);
              /* Same-process router-order override: runtime_store/
                 router_order.cfg (true/false) beats the config, so a new
                 and an old order can be compared inside ONE gateway
                 process instead of two separate logs. */
              auto routerOrderFromFiles = [&]() -> bool {
                bool first = phoenix::cfgOr<bool>(
                    "mission.routerBeforeCond", true);
                std::ifstream rf("runtime_store/router_order.cfg");
                if (rf) {
                  std::string v;
                  std::getline(rf, v);
                  if (v.find("false") != std::string::npos)
                    first = false;
                  else if (v.find("true") != std::string::npos)
                    first = true;
                }
                return first;
              };
              bool pendingScore = false;
              float pendingScoreReward = 0.f;
              std::string pendingScoreReply;
              if (currentFile.empty() && !deliverable.empty())
                currentFile = deliverable; /* before the first write */
              /* A code submission is one function. The document rule that
                 a longer in-memory body wins would paste earlier failed
                 attempts back over the function they just wrote. */
              if (!peerEvolveGoal && deliverable.size() > currentFile.size())
                currentFile = deliverable;
              /* One function. A shorter rewrite used to be glued back onto
                 the previous failed attempt, so the causal tip never moved. */
              std::string peerEditBase = currentFile;
              /* Real source span for SELF-MOD. Filled when the file is read.
                 The causal tip and the summoned expert both use this carrier,
                 so the expert is not handed the score paragraph. */
              std::string ownCarrier;
              /* The self-mod target path, hoisted so the summon-tool can
                 name it in a retry prompt ("FILE <path>") when the coder's
                 first reply is prose instead of a patch. */
              std::string focusTarget;
              /* Repository explore list: immutable after the first build.
                 A function-local magic static is the C++11 thread-safe way to
                 publish it, so parallel missions only race the FIRST call and
                 never the contents.  The old lazy `if (explorePaths.empty())`
                 let two threads push_back into the same vector while a third
                 iterated it - a use-after-free that the ntdll heap check
                 reports as 0xC0000374. */
              static const std::vector<std::string> &explorePaths =
                  gatewayExplorePaths();
              /* Last repository path a reply named.  Per mission scope: a
                 process-global leaked mission A's target into mission B's
                 focus, and the concurrent write was a data race on a
                 std::string (realloc under a reader = heap corruption). */
              static std::mutex namedPathMu;
              static std::unordered_map<std::string, std::string>
                  namedPathByScope;
              auto namedPathGet = [&]() -> std::string {
                std::lock_guard<std::mutex> lk(namedPathMu);
                auto it = namedPathByScope.find(scope);
                return it == namedPathByScope.end() ? std::string()
                                                    : it->second;
              };
              auto namedPathSet = [&](const std::string &p) {
                std::lock_guard<std::mutex> lk(namedPathMu);
                namedPathByScope[scope] = p;
              };
              std::string exploreMark = "none";
              std::string summonMark = "none";
              if (peerEvolveGoal && !codeMission &&
                  phoenix::peer::evolveHasControl(peerEditBase))
                phoenix::peer::evolveTrimToDeclaration(peerEditBase);
              /* A draft that only grows pushes the earlier lines out of
                 the window. Keep the tail they are still writing. */
              if (codeMission && peerEditBase.size() > 1800)
                peerEditBase = peerEditBase.substr(peerEditBase.size() - 1800);
              if (selfModGoal && peerEditBase.size() > 16000) {
                const auto cut = peerEditBase.rfind("\ndiff ");
                if (cut != std::string::npos)
                  peerEditBase = peerEditBase.substr(cut + 1);
              }

              const bool draftEmpty = currentFile.empty();

              /* Causal has word order: first-layer working brief +
                 packContext recent draft. RAG (memory/GNN/search) only
                 hints; it does not write the file. */
              std::vector<phoenix::inference::UnitQueryIO> missionMemorySnippets;
              std::vector<phoenix::inference::UnitQueryIO> missionGnnPkts;
              const bool unitPipelineEnabled =
                  phoenix::cfgOr<bool>("main.inference.unitPipeline", true);
              phoenix::context::PackOptions packOpt;
              packOpt.mode = "full_and_summary";
              packOpt.includeGnnSummary =
                  phoenix::cfgOr<bool>("mission.includeGnnSummary", true);
              packOpt.ctxTokens =
                  phoenix::cfgOr<int>("mission.ctxSize", 4096);
              packOpt.ngramMerge =
                  phoenix::cfgOr<int>("main.inference.ngramMerge", 2);
              if (!phoenix::cfgOr<bool>("main.inference.packPrompt", false))
                packOpt.ngramMerge = 1;
              std::string gnnText =
                  cognitionManager_.missionGnnSummaryFor(msnId);
              nlohmann::json missionGraphResult = nlohmann::json::object();
              try {
                const auto copts = cognitionManager_.missionContextOptions();
                const auto &cr = copts.value("result", nlohmann::json::object());
                packOpt = phoenix::context::optionsFromJson(cr, packOpt);
                if (!phoenix::cfgOr<bool>("main.inference.packPrompt", false))
                  packOpt.ngramMerge = 1;
              } catch (...) {
              }
              const auto packed =
                  phoenix::context::packContext(currentFile, gnnText, packOpt);
              const std::string packedBody = packed.packed;
              const std::string recallDraft = packed.recentFull.empty()
                  ? currentFile : packed.recentFull;
              const bool pinOnly = draftEmpty;
              const bool stillDrafting =
                  static_cast<int>(currentFile.size()) < minDeliverableChars;
              const bool onesJunkDraft =
                  phoenix::mission::looksLikeOnesJunk(currentFile) ||
                  phoenix::mission::looksLikeRewriteInstructionEcho(
                      currentFile);
              if (config_.httpLog)
                std::cout << "[mission-deliberate] stage=enter deliv="
                          << currentFile.size()
                          << " minChars=" << minDeliverableChars
                          << " stillDrafting=" << (stillDrafting ? 1 : 0)
                          << std::endl;
              /* GNN pretrain gap: empty meme graph => empty diffusion prior.
                 The graph covers the whole deliverable. Only the unseen
                 suffix is linked; the outline query still sees every
                 character. Pin-only is not a draft to graph. A 160–250
                 char first write used to fall into this block and pin
                 the heartbeat (no second tokBudget for minutes). */
              if (!pinOnly && !stillDrafting &&
                  (packOpt.includeGnnSummary || unitPipelineEnabled)) {
                if (config_.httpLog)
                  std::cout << "[mission-deliberate] stage=gnn-pretrain"
                            << std::endl;
                try {
                  const int lim =
                      phoenix::cfgOr<int>("mission.gnnPretrainDocs", 48);
                  seedGnnGraphFromRobotsCorpus(lim, "gnn-pretrain");
                  if (unitPipelineEnabled) {
                    const auto gnnT0 = std::chrono::steady_clock::now();
                    /* The graph is the outline of the whole deliverable.
                       packed.recentFull is only the causal window for the
                       llama continuation; it must not replace the document. */
                    const std::string gnnSrc = currentFile;
                    struct GnnCursor {
                      size_t chars{0};
                      uint64_t hash{0};
                      nlohmann::json graph = nlohmann::json::object();
                      bool haveGraph{false};
                    };
                    static std::mutex gnnCursorMu;
                    static std::unordered_map<std::string, GnnCursor> gnnCursors;
                    auto fnvPrefix = [](const std::string &s, size_t n) {
                      uint64_t h = 1469598103934665603ull;
                      const size_t lim = std::min(n, s.size());
                      for (size_t i = 0; i < lim; ++i) {
                        h ^= static_cast<unsigned char>(s[i]);
                        h *= 1099511628211ull;
                      }
                      return h;
                    };
                    std::string delta = gnnSrc;
                    bool reuseGraph = false;
                    {
                      std::lock_guard<std::mutex> lk(gnnCursorMu);
                      auto &cur = gnnCursors[scope];
                      if (cur.chars > 0 && cur.chars <= gnnSrc.size() &&
                          fnvPrefix(gnnSrc, cur.chars) == cur.hash) {
                        if (cur.chars == gnnSrc.size()) {
                          delta.clear();
                          reuseGraph = cur.haveGraph;
                          if (reuseGraph)
                            missionGraphResult = cur.graph;
                        } else {
                          /* Keep one sentence of overlap so the new suffix
                             stays linked to the graph of the earlier text. */
                          const size_t overlap = std::min(cur.chars, size_t{256});
                          delta = gnnSrc.substr(cur.chars - overlap);
                        }
                      } else {
                        cur.chars = 0;
                        cur.hash = 0;
                        cur.haveGraph = false;
                      }
                    }
                    if (config_.httpLog)
                      std::cout << "[mission-deliberate] gnnSrc="
                                << gnnSrc.size() << " delta=" << delta.size()
                                << std::endl;
                    if (!pinOnly && currentFile.size() >= 2500 &&
                        !gnnSrc.empty() &&
                        !phoenix::mission::draftTailLooksCollapsed(gnnSrc)) {
                      if (!reuseGraph) {
                        auto rt = activeRuntime();
                        auto st = rt ? std::dynamic_pointer_cast<RuntimeState>(rt)
                                     : nullptr;
                        if (st && !delta.empty()) {
                          const auto ingestT0 = std::chrono::steady_clock::now();
                          st->ingestDocument(
                              nlohmann::json{{"text", delta},
                                             {"source", "mission-draft"},
                                             {"mode", "chain"}});
                          if (config_.httpLog) {
                            const auto ingestMs =
                                std::chrono::duration_cast<
                                    std::chrono::milliseconds>(
                                    std::chrono::steady_clock::now() - ingestT0)
                                    .count();
                            std::cout << "[mission-deliberate] gnnIngestMs="
                                      << ingestMs << std::endl;
                          }
                        }
                        const auto queryT0 = std::chrono::steady_clock::now();
                        missionGraphResult = buildMissionGraphResult(gnnSrc);
                        if (config_.httpLog) {
                          const auto queryMs =
                              std::chrono::duration_cast<
                                  std::chrono::milliseconds>(
                                  std::chrono::steady_clock::now() - queryT0)
                                  .count();
                          std::cout << "[mission-deliberate] gnnQueryMs="
                                    << queryMs << std::endl;
                        }
                        {
                          std::lock_guard<std::mutex> lk(gnnCursorMu);
                          auto &cur = gnnCursors[scope];
                          cur.chars = gnnSrc.size();
                          cur.hash = fnvPrefix(gnnSrc, gnnSrc.size());
                          cur.graph = missionGraphResult;
                          cur.haveGraph = true;
                        }
                      }
                      const int llamaDim = phoenix::cfgOr<int>(
                          "main.inference.llamaEmbDim", 4096);
                      auto gnnRows =
                          buildGraphEmbeddings(missionGraphResult, llamaDim);
                      gnnRows = phoenix::mission::projectRowsToLlamaDim(
                          gnnRows, llamaDim);
                      if (gnnRows.size() > 32)
                        gnnRows.resize(32);
                      if (!gnnRows.empty())
                        phoenix::inference::appendUnitQuery(
                            missionGnnPkts,
                            phoenix::inference::unitQueryFromRows(
                                std::move(gnnRows), "unit"));
                    }
                    if (config_.httpLog) {
                      const auto gnnMs =
                          std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - gnnT0)
                              .count();
                      std::cout << "[mission-deliberate] gnnMs=" << gnnMs
                                << std::endl;
                    }
                  }
                  /* Do not invent a document outline from GNN memes.
                     The model writes plan.md after it has a draft. */
                } catch (const std::exception &ex) {
                  std::cout << "[mission] GNN pretrain/diffuse failed: "
                            << ex.what() << std::endl;
              } catch (...) {
              }
              }

              /* RDK llama prompt eval is ~1.5-2s/token.  Dumping every AGI
                 action (name+category+description) pushed the deliberator
                 prompt to ~780 tokens (~24 min) and exceeded the HTTP
                 timeout → llama cancel_tasks → empty deliverable.  Keep the
                 prompt tiny: names-only tool hint, capped pack/outline. */
              /* v8.x tool-friendly: the model says its INTENT in natural
                 language and the gateway resolves it (the tool understands
                 the AI, not the other way round).  The static prefix lists
                 name + one-line purpose per tool - it is prefilled once and
                 then reused by the slot cache, so the one-time cost is
                 amortized across ticks. */
              std::string toolHint = "math, search, replicate";
              try {
                const auto la = cognitionManager_.listAgiActions();
                const auto &arr = la.value("result", nlohmann::json::array());
                std::string names;
                int n = 0;
                for (const auto &spec : arr) {
                  if (!spec.is_object()) continue;
                  const std::string nm = spec.value("name", std::string());
                  if (nm.empty()) continue;
                  if (n++) names += ", ";
                  names += nm;
                  if (n >= 16) break;
                }
                if (!names.empty()) toolHint = names;
              } catch (...) {
              }

              std::string goalClip = goal;
              {
                const size_t goalMax =
                    phoenix::cfgOr<size_t>("mission.goalClipChars", 16000);
                if (goalClip.size() > goalMax)
                  goalClip = phoenix::mission::clipGoalKeepRequiredStructure(
                      goalClip, goalMax);
              }
              std::string listClip = listRes.dump();
              if (listClip.size() > 300) listClip = listClip.substr(0, 300) + "...";
              /* packed.recentFull is the causal short-term window.
                 pinnedSummary / gnnPinned are RAG, not causal text. */
              if (!packed.pinnedSummary.empty())
                phoenix::inference::appendUnitQueriesFromParagraphs(
                    missionMemorySnippets, packed.pinnedSummary, "text");
              if (!packed.gnnPinned.empty())
                phoenix::inference::appendUnitQueriesFromParagraphs(
                    missionMemorySnippets, packed.gnnPinned, "text");
              (void)packedBody;

              /* Working brief is first-layer recent context (word order).
                 Catalog / appendix stay out of this prefix so the tip
                 is not a chapter table. Full assignment is RAG hint. */
              std::string staticPrompt =
                  phoenix::mission::extractGoalWorkingContext(goalClip);
              /* The model continues the END of the causal text. The working
                 brief is context. The tip is the one chapter whose body is
                 still missing or only a requirements list. A heading alone
                 is not a finished section. */
              const std::string sectionFocus =
                  phoenix::mission::firstUnderfilledGoalChapter(goal,
                                                                currentFile);
              /* Per-scope "the model restated instead of continuing this
                 section" counter.  Guarded: parallel missions inserting into a
                 bare std::map from several `std::async` threads rebalanced the
                 tree under a reader - the second heap-corruption source that
                 the coarse missionDelibMu_ had been masking. */
              struct ScopeCounter {
                std::mutex mu;
                std::map<std::string, int> values;
                int get(const std::string &k) {
                  std::lock_guard<std::mutex> lk(mu);
                  auto it = values.find(k);
                  return it == values.end() ? 0 : it->second;
                }
                void set(const std::string &k, int v) {
                  std::lock_guard<std::mutex> lk(mu);
                  values[k] = v;
                }
                void inc(const std::string &k) {
                  std::lock_guard<std::mutex> lk(mu);
                  ++values[k];
                }
              };
              static ScopeCounter sectionDropStreak;
              std::string sectionTip;
              const int sectionStreak = sectionDropStreak.get(scope);
              if (!sectionFocus.empty()) {
                sectionTip = phoenix::mission::chapterContinuationTip(
                    sectionFocus, currentFile, sectionStreak);
                if (config_.httpLog && sectionStreak > 0)
                  std::cout << "[mission-deliberate] section-variant="
                            << sectionStreak << std::endl;
                if (config_.httpLog) {
                  std::string shown = sectionFocus;
                  if (shown.size() > 120) shown.resize(120);
                  std::cout << "[mission-deliberate] next-section=" << shown
                            << std::endl;
                }
              } else if (!phoenix::mission::collectGoalChapterHeadings(goal)
                              .empty() &&
                         !stillDrafting && !isChildBox) {
                std::string msnId = scope;
                const auto spDone = msnId.find("/children/");
                if (spDone != std::string::npos) msnId = msnId.substr(0, spDone);
                cognitionManager_.reportMissionOutcome(
                    nlohmann::json{{"goalAchieved", true},
                                   {"missionId", msnId}});
                return "[auto-complete] every required section has a body";
              }
              {
                const std::string expPath = phoenix::resolveConfig<std::string>(
                    "mission.experiencePath",
                    std::string("./runtime_store/mission_experience.json"));
                /* Pin-only / empty draft: do not recall prior missions.
                   Assign writes the seed before tick 1; treating that
                   as a real window pulled tutorial/game-review RAG. */
                if (!pinOnly && !stillDrafting && !recallDraft.empty()) {
                  if (config_.httpLog)
                    std::cout << "[mission-deliberate] stage=recall"
                              << std::endl;
                  /* callerTag buckets the read path: other missions'
                     deposits must clear a goal-relevance bar before they
                     may enter THIS mission's prompt. */
                  phoenix::mission::appendRecalledMemory(
                      missionMemorySnippets, phoenix::instance::ccmStorePath(),
                      expPath, goal, recallDraft, 3, 3, "mission:" + scope);
                }
                /* Pretrain wrote HierarchicalMemory under corpus:* and
                   never read it back on mission ticks. Same recall query
                   as CCM — not a second causal stream. */
                if (!pinOnly && !stillDrafting) {
                if (config_.httpLog)
                  std::cout << "[mission-deliberate] stage=corpus-query"
                            << std::endl;
                try {
                  const std::string q = phoenix::mission::buildMemoryRecallQuery(
                      goalClip, recallDraft);
                  const auto hits = cognitionMemory_.query("corpus:", 12);
                  int kept = 0;
                  for (const auto &kv : hits) {
                    if (!kv.second.is_object()) continue;
                    std::string t = kv.second.value("text", std::string());
                    if (t.empty() ||
                        phoenix::mission::looksLikeCollapsedProse(t))
                      continue;
                    if (phoenix::memory::ccmOverlap(q, t) < 0.04)
                      continue;
                    if (t.size() > 800) t.resize(800);
                    phoenix::inference::appendUnitQueriesFromParagraphs(
                        missionMemorySnippets, t, "text");
                    if (++kept >= 3) break;
                  }
                } catch (...) {
                }
                }
                /* Full assignment / catalog / tools: RAG hint only.
                   They do not replace the causal recent window. */
                {
                const std::string brief = goalClip;
                const size_t memBefore = missionMemorySnippets.size();
                phoenix::inference::appendUnitQueriesFromParagraphs(
                    missionMemorySnippets, brief, "text");
                const size_t goalCap = 24;
                if (missionMemorySnippets.size() > memBefore + goalCap)
                  missionMemorySnippets.resize(memBefore + goalCap);
                /* Numbered outline on a short/ones-junk draft is what
                   taught the board to file-edit "1." forever. Keep it
                   off the RAG stream until there is real prose. */
                if (phoenix::cfgOr<bool>("mission.outlineEnabled", true) &&
                    !stillDrafting && !onesJunkDraft) {
                  if (config_.httpLog)
                    std::cout << "[mission-deliberate] stage=outline"
                              << std::endl;
                  try {
                    const std::string outline = buildMissionOutline(goalClip);
                    /* The outline is a GLOBAL meme skeleton, not
                       goal-scoped: a graph dominated by the previous
                       mission's theme would bleed those keywords into
                       this mission's prompt. Skip it when it shares no
                       vocabulary with the goal at all. */
                    if (outline.find("\n1. ") != std::string::npos &&
                        phoenix::memory::ccmOverlap(goalClip, outline) > 0.0)
                      phoenix::inference::appendUnitQuery(
                          missionMemorySnippets,
                          phoenix::inference::unitQueryFromText(outline,
                                                                "text"));
                  } catch (...) {
                  }
                }
                }
                /* Observe step: draft defects go to RAG, not the causal
                   prefix and not a refuse gate. The causal stream can
                   still append; the model is shown what to revise. */
                if (!pinOnly && !currentFile.empty()) {
                  const std::string health =
                      phoenix::mission::inspectDeliverableHealth(currentFile,
                                                                 goal);
                  if (!health.empty())
                    phoenix::inference::appendUnitQueriesFromParagraphs(
                        missionMemorySnippets, health, "text");
                }
                /* Anti-junk belongs on RAG. Putting "Do not omit 1.1.1"
                   on the causal tip made rdk-retest7-111500 file-edit
                   that sentence as the only body. */
                if (draftEmpty || onesJunkDraft) {
                  phoenix::inference::appendUnitQueriesFromParagraphs(
                      missionMemorySnippets,
                      phoenix::mission::onesJunkRewriteHint(), "text");
                }
                /* Plugins self-select. Host broadcasts the situation;
                   each installed addon understands it and may contribute
                   text or unit-query packets. Both become unit queries
                   on the RAG stream. They are not a third protocol and
                   they are not concatenated onto the causal prefix. */
                /* Self-evolve DLLs hot-load on this path. A short task is
                   still drafting, and that used to skip the reload entirely. */
                if (phoenix::cfgOr<bool>("mission.pluginOffers", true) &&
                    !goal.empty()) {
                  if (config_.httpLog)
                    std::cout << "[mission-deliberate] stage=plugin-offers"
                              << std::endl;
                  try {
                    auto rt = activeRuntime();
                    auto st =
                        rt ? std::dynamic_pointer_cast<RuntimeState>(rt)
                           : nullptr;
                    if (st) {
                      const int seeded = st->ensureSearchIndexSeeded(
                          phoenix::cfgOr<int>("mission.pluginSearchDocs",
                                             64));
                      nlohmann::json sit;
                      sit["phase"] = "mission-deliberate";
                      sit["goal"] = goal;
                      sit["draft"] = currentFile;
                      sit["allowWeb"] = phoenix::cfgOr<bool>(
                          "mission.pluginSearchWeb", true);
                      sit["allowIndex"] = true;
                      sit["pinOnly"] = pinOnly;
                      const auto offers = st->collectAddonOffers(sit);
                      std::string srcAll;
                      std::string names;
                      size_t pluginUnits = 0;
                      for (const auto &off : offers) {
                        if (!names.empty()) names += ",";
                        names += off.name;
                        const std::string src =
                            off.result.meta.value("source", std::string());
                        if (!src.empty() &&
                            srcAll.find(src) == std::string::npos) {
                          if (!srcAll.empty()) srcAll += "+";
                          srcAll += src;
                        }
                        if (off.result.meta.contains("results") &&
                            off.result.meta["results"].is_array()) {
                          for (const auto &hit : off.result.meta["results"]) {
                            if (!hit.is_object()) continue;
                            const std::string url =
                                hit.value("url", std::string());
                            const std::string title =
                                hit.value("title", std::string());
                            const std::string snip =
                                hit.value("snippet", std::string());
                            if (snip.size() < 40) continue;
                            if (url.rfind("http://", 0) == 0 ||
                                url.rfind("https://", 0) == 0)
                              st->indexSearchDocument(url, title.empty()
                                                               ? url
                                                               : title,
                                                      snip);
                          }
                        }
                        const size_t before = missionMemorySnippets.size();
                        if (off.result.units.is_array() &&
                            !off.result.units.empty()) {
                          phoenix::inference::appendUnitQueriesFromJsonArray(
                              missionMemorySnippets, off.result.units);
                        }
                        pluginUnits +=
                            missionMemorySnippets.size() - before;
                        if (config_.httpLog)
                          std::cout
                              << "[mission-deliberate] plugin-offer name="
                              << off.name << " score=" << off.score
                              << " src="
                              << (src.empty() ? std::string("-") : src)
                              << " units="
                              << (missionMemorySnippets.size() - before)
                              << " reply=" << off.result.reply.size()
                              << std::endl;
                        const bool selfEvolve =
                            off.name.find("self_evolve") != std::string::npos ||
                            off.type.find("self-evolve") != std::string::npos ||
                            off.type.find("self_evolve") != std::string::npos;
                        /* A quiz plugin's return value is not a
                           self-modification. SELF-MOD keeps the source. */
                        if (selfEvolve && !selfModGoal &&
                            !off.result.reply.empty()) {
                          std::string hot = off.result.reply;
                          if (hot.size() > 4000) hot.resize(4000);
                          phoenix::mission::workspaceExecute(
                              workspaceRoot, scope,
                              nlohmann::json{{"action", "write"},
                                             {"path", "hot_result.txt"},
                                             {"content", hot}});
                          if (config_.httpLog)
                            std::cout << "[mission-deliberate] hot-result bytes="
                                      << hot.size() << std::endl;
                        }
                      }
                      if (config_.httpLog && seeded > 0)
                        std::cout
                            << "[mission-deliberate] plugin-index seeded="
                            << seeded << std::endl;
                      if (config_.httpLog && !offers.empty()) {
                        std::cout
                            << "[mission-deliberate] plugin-offers who="
                            << names << " src="
                            << (srcAll.empty() ? std::string("-") : srcAll)
                            << " pluginUnits=" << pluginUnits
                            << " memUnits=" << missionMemorySnippets.size()
                            << std::endl;
                      } else if (config_.httpLog && offers.empty()) {
                        std::cout
                            << "[mission-deliberate] plugin-offers none"
                            << std::endl;
                      }
                    }
                  } catch (const std::exception &ex) {
                    if (config_.httpLog)
                      std::cout << "[mission-deliberate] plugin-offers error "
                                << ex.what() << std::endl;
                  } catch (...) {
                    if (config_.httpLog)
                      std::cout << "[mission-deliberate] plugin-offers error"
                                << std::endl;
                  }
                }
                /* C1: the model's own plan.md is RAG emphasis, not a
                   [NEXT] block on the causal stream. Resume stays the
                   file tail. Skip on pin-only — there is no draft yet. */
                if (selfModGoal) {
                  const auto measured = phoenix::mission::workspaceExecute(
                      workspaceRoot, scope,
                      nlohmann::json{{"action", "read"},
                                     {"path", "measure.md"}});
                  if (measured.value("ok", false)) {
                    std::string line = measured.value("content", std::string());
                    if (line.rfind("score ", 0) == 0) {
                      phoenix::inference::appendUnitQuery(
                          missionMemorySnippets,
                          phoenix::inference::unitQueryFromText(line, "text"));
                      const auto seen = phoenix::mission::workspaceExecute(
                          workspaceRoot, scope,
                          nlohmann::json{{"action", "read"},
                                         {"path", "score_seen.txt"}});
                      const std::string prev =
                          seen.value("content", std::string());
                      if (prev != line) {
                        float reward = -0.35f;
                        try {
                          if (std::stod(line.substr(6)) > 0) reward = 0.6f;
                        } catch (...) {
                        }
                        pendingScore = true;
                        pendingScoreReward = reward;
                        pendingScoreReply =
                            currentFile.empty() ? line : currentFile;
                        phoenix::mission::workspaceExecute(
                            workspaceRoot, scope,
                            nlohmann::json{{"action", "write"},
                                           {"path", "score_seen.txt"},
                                           {"content", line}});
                      }
                    }
                  }
                  const char *peerEnv = std::getenv("AI_PEER_NAME");
                  std::ifstream otherEdit(
                      std::string("runtime_store/peers/") +
                      (peerEnv ? peerEnv : "") + "/self_edit.diff");
                  std::string other((std::istreambuf_iterator<char>(otherEdit)),
                                    std::istreambuf_iterator<char>());
                  if (other.find("diff ") != std::string::npos ||
                      other.find("@@") != std::string::npos ||
                      other.find("FILE ") != std::string::npos) {
                    if (other.size() > 4000)
                      other = other.substr(other.size() - 4000);
                    phoenix::inference::appendUnitQuery(
                        missionMemorySnippets,
                        phoenix::inference::unitQueryFromText(
                            "other edit\n" + other, "text"));
                  }
                  std::string focus;
                  /* Alias-probe mode: keep the source-edit focus empty so
                     the description-only alias recall runs instead of the
                     source-edit tool (the tool otherwise wins whenever the
                     goal or a reply names a file). */
                  const bool aliasProbe =
                      phoenix::cfgOr<bool>("mission.aliasProbe", false);
                  if (!aliasProbe) {
                  auto takePath = [&](const std::string &mark) {
                    const auto at0 = currentFile.rfind(mark);
                    if (at0 == std::string::npos) return;
                    size_t at = at0 + mark.size();
                    while (at < currentFile.size() &&
                           std::isspace(static_cast<unsigned char>(
                               currentFile[at])))
                      ++at;
                    auto end = currentFile.find_first_of("\r\n", at);
                    if (end == std::string::npos) end = currentFile.size();
                    const std::string rel = currentFile.substr(at, end - at);
                    if (rel.find("..") != std::string::npos) return;
                    if (rel.find("runtime_store") != std::string::npos) return;
                    if (rel.find("peer_hot") != std::string::npos) return;
                    const bool hub =
                        rel.rfind("main_hub_parts/", 0) == 0 &&
                        rel.find(".inc") != std::string::npos;
                    const bool nested =
                        rel.rfind("phoenix/", 0) == 0 ||
                        rel.rfind("addons/", 0) == 0 ||
                        rel.rfind("module_overrides/", 0) == 0;
                    const bool top =
                        rel.find('/') == std::string::npos &&
                        (rel.find(".cpp") != std::string::npos ||
                         rel.find(".hpp") != std::string::npos ||
                         rel.find(".inc") != std::string::npos);
                    if (hub || nested || top) focus = rel;
                  };
                  /* The goal itself names the target.  It is authoritative
                     over the workspace: a stale deliverable (or a seed-write
                     race) can still carry "FILE ActiveInferenceController.cpp"
                     from a previous run, which made ownCarrier 0 on the very
                     tick that had to edit. */
                  if (focus.empty()) {
                    for (const std::string ext :
                         {std::string(".cpp"), std::string(".hpp"),
                          std::string(".inc")}) {
                      const auto at = goal.find(ext);
                      if (at == std::string::npos) continue;
                      size_t begin = at;
                      while (begin > 0 &&
                             (std::isalnum(static_cast<unsigned char>(
                                  goal[begin - 1])) ||
                              goal[begin - 1] == '_' || goal[begin - 1] == '/' ||
                              goal[begin - 1] == '.'))
                        --begin;
                      focus = goal.substr(begin, at + ext.size() - begin);
                      break;
                    }
                  }
                  if (focus.empty()) takePath("FILE ");
                  /* A previous reply named a real repository path. The body
                     stays off the causal tip; only the expert receives it. */
                  const std::string namedRealPath = namedPathGet();
                  if (focus.empty() && !namedRealPath.empty() &&
                      namedRealPath.find("..") == std::string::npos) {
                    std::error_code namedEc;
                    if (fs::is_regular_file(namedRealPath, namedEc))
                      focus = namedRealPath;
                  }
                  /* Persist the target per scope: after the first patch the
                     deliverable no longer carries "FILE <path>", so the
                     next tick used to lose ownCarrier and the expert idled
                     ("no-file-yet").  A stable target keeps the source-edit
                     rate constant, which is what the force fit needs. */
                  {
                    static std::mutex focusMu;
                    static std::unordered_map<std::string, std::string>
                        focusByScope;
                    std::lock_guard<std::mutex> flock(focusMu);
                    if (focus.empty()) {
                      auto it = focusByScope.find(scope);
                      if (it != focusByScope.end()) focus = it->second;
                    } else {
                      focusByScope[scope] = focus;
                    }
                  }
                  if (!focus.empty()) focusTarget = focus;
                  /* An explicit path still opens a file for the expert only.
                     A description with no path is ranked after the explore
                     list exists. The file body is not pasted here. */
                  if (!focus.empty()) {
                  std::ifstream ownSrc(focus, std::ios::binary);
                  std::string src((std::istreambuf_iterator<char>(ownSrc)),
                                  std::istreambuf_iterator<char>());
                  if (!src.empty()) {
                    const auto mark = src.find("processMessage");
                    if (mark != std::string::npos && mark + 1500 < src.size())
                      src = src.substr(mark, 1500);
                    else if (src.size() > 1500)
                      src = src.substr(src.size() - 1500);
                    ownCarrier =
                        phoenix::memetic::exclusiveEditCarrier(focus, src);
                  }
                  }
                  }
                  if (config_.httpLog)
                    std::cout << "[sig] focus scope=" << scope
                              << " path="
                              << (focus.empty() ? std::string("-") : focus)
                              << " own=" << ownCarrier.size() << std::endl;
                  /* explorePaths is built once by gatewayExplorePaths() (see
                     the magic static at the top of this lambda); no lazy
                     mutation here any more. */
                  const auto ranked = phoenix::memetic::rankExplorePaths(
                      goal, explorePaths, 1);
                  /* Width: rankExplorePaths only returns a path when EXACTLY
                     one candidate matches (ambiguity guard).  When several
                     match it returns nothing, so the mission never tried any
                     of them and SELF-MOD kept re-editing whatever file it had
                     already named.  Sample among the plausible files with a
                     softmax whose temperature anneals per loop tick, so early
                     ticks explore and late ticks converge. */
                  std::string exploredPath =
                      ranked.empty()
                          ? phoenix::memetic::pickExplorePath(
                                goal, explorePaths,
                                phoenix::cfgOr<double>("mission.exploreTemperature",
                                                       0.8),
                                static_cast<unsigned>(
                                    cognitionManager_.loopTickCount()))
                          : ranked.front();
                  std::string note;
                  if (!exploredPath.empty()) {
                    note = "repository file\n" + exploredPath;
                  } else {
                    note = "repository files\n";
                    int shown = 0;
                    for (const auto &rel : explorePaths) {
                      note += rel;
                      note += "\n";
                      if (++shown >= 8)
                        break;
                    }
                  }
                  phoenix::inference::appendUnitQuery(
                      missionMemorySnippets,
                      phoenix::inference::unitQueryFromText(note, "text"));
                  exploreMark = exploredPath.empty()
                                    ? "listed"
                                    : std::string("pick ") + exploredPath;
                  /* The draft is the recommender context. File profiles are
                     the items. A unique top score names the file the expert
                     receives. Echoes of the goal are not a new conception.
                     The body stays in ownCarrier, off the causal tip. */
                  if (config_.httpLog)
                    std::cout << "[sig] latent-guard focus="
                              << (focus.empty() ? 0 : 1) << " selfMod="
                              << (selfModGoal ? 1 : 0) << " curLen="
                              << currentFile.size() << " classify="
                              << phoenix::memetic::classifyReplicaReply(
                                     goal, currentFile, false)
                              << std::endl;
                  if (focus.empty() && selfModGoal && !currentFile.empty() &&
                      phoenix::memetic::classifyReplicaReply(
                          goal, currentFile, false) != "echo") {
                    struct AliasFile {
                      std::string path;
                      std::vector<phoenix::memetic::GlossTerm> terms;
                      phoenix::memetic::GlossLineage line;
                      std::vector<std::string> pending;
                      std::vector<std::string> links;
                      std::vector<std::string> pendingLinks;
                      /* Recent independent conditional hits for this
                         file. A repeat of any of the last four is not a
                         second witness (confirmCondWords checks it);
                         keeping four closes the A,B,A alternation. */
                      std::vector<std::string> condRecent;
                    };
                    std::vector<AliasFile> aliasRows;
                    {
                      std::ifstream aliasIn("runtime_store/meme_alias.json");
                      if (aliasIn) {
                        try {
                          nlohmann::json saved;
                          aliasIn >> saved;
                          if (saved.is_array()) {
                            for (const auto &row : saved) {
                              if (!row.is_object())
                                continue;
                              AliasFile file;
                              file.path = row.value("path", std::string());
                              if (file.path.empty() || !row.contains("words") ||
                                  !row["words"].is_array())
                                continue;
                              for (const auto &w : row["words"]) {
                                phoenix::memetic::GlossTerm term;
                                if (w.is_string())
                                  term.word = w.get<std::string>();
                                else if (w.is_object())
                                  term.word = w.value("word", std::string());
                                if (term.word.empty())
                                  continue;
                                term.tf = w.is_object() ? w.value("tf", 1) : 1;
                                if (term.tf < 1)
                                  term.tf = 1;
                                file.terms.push_back(std::move(term));
                              }
                              file.line.state.fecundity =
                                  row.value("fecundity", 0);
                              file.line.state.longevity =
                                  row.value("longevity", 1);
                              file.line.state.streak = row.value("streak", 0);
                              if (row.contains("carriers") &&
                                  row["carriers"].is_array()) {
                                for (const auto &c : row["carriers"]) {
                                  if (c.is_string())
                                    file.line.carriers.push_back(
                                        c.get<std::string>());
                                }
                              }
                              if (row.contains("pending") &&
                                  row["pending"].is_array()) {
                                for (const auto &w : row["pending"]) {
                                  if (w.is_string())
                                    file.pending.push_back(w.get<std::string>());
                                }
                              }
                              if (row.contains("blocked") &&
                                  row["blocked"].is_array()) {
                                for (const auto &w : row["blocked"]) {
                                  if (w.is_string())
                                    file.links.push_back(w.get<std::string>());
                                }
                              }
                              if (row.contains("blockedPending") &&
                                  row["blockedPending"].is_array()) {
                                for (const auto &w : row["blockedPending"]) {
                                  if (w.is_string())
                                    file.pendingLinks.push_back(
                                        w.get<std::string>());
                                }
                              }
                              if (row.contains("condRecent") &&
                                  row["condRecent"].is_array()) {
                                for (const auto &w : row["condRecent"]) {
                                  if (w.is_string())
                                    file.condRecent.push_back(
                                        w.get<std::string>());
                                }
                              } else if (row.contains("condLast") &&
                                         row["condLast"].is_string()) {
                                const std::string legacy =
                                    row["condLast"].get<std::string>();
                                if (!legacy.empty())
                                  file.condRecent.push_back(legacy);
                              }
                              aliasRows.push_back(std::move(file));
                            }
                          }
                        } catch (const std::exception &) {
                        }
                      }
                    }
                    auto saveAliases = [&]() {
                      nlohmann::json rows = nlohmann::json::array();
                      for (const auto &file : aliasRows) {
                        nlohmann::json words = nlohmann::json::array();
                        for (const auto &term : file.terms)
                          words.push_back(
                              {{"word", term.word}, {"tf", term.tf}});
                        rows.push_back(
                            {{"path", file.path},
                             {"words", words},
                             {"pending", file.pending},
                             {"blocked", file.links},
                             {"blockedPending", file.pendingLinks},
                             {"condRecent", file.condRecent},
                             {"regime",
                              phoenix::memetic::lineageRegime(
                                  file.line.carriers)
                                  .name},
                             {"fecundity", file.line.state.fecundity},
                             {"longevity", file.line.state.longevity},
                             {"streak", file.line.state.streak},
                             {"carriers", file.line.carriers}});
                      }
                      /* Continuous replicator flow at t = 1.  The
                         discrete map p'_i = p_i f_i / sum p_j f_j is the
                         unit-time solution of dp_i/dt = p_i (f_i - fbar),
                         so the flow share must order the rows exactly as
                         fecundity x longevity does.  This connects the
                         continuous equation to the same population the
                         recall ordering already uses; it does not change
                         the ordering. */
                      if (aliasRows.size() >= 2) {
                        const size_t n = aliasRows.size();
                        std::vector<double> freq(n, 1.0 / static_cast<double>(n));
                        std::vector<double> fit(n, 1.0);
                        for (size_t i = 0; i < n; ++i) {
                          const int f = std::max(1, aliasRows[i].line.state.fecundity);
                          const int l = std::max(1, aliasRows[i].line.state.longevity);
                          fit[i] = static_cast<double>(f) *
                                   static_cast<double>(l);
                        }
                        const auto share =
                            phoenix::memetic::replicatorFlow(freq, fit, 1.0);
                        std::vector<size_t> byFit(n), byFlow(n);
                        for (size_t i = 0; i < n; ++i) {
                          byFit[i] = i;
                          byFlow[i] = i;
                        }
                        std::sort(byFit.begin(), byFit.end(),
                                  [&](size_t a, size_t b) {
                                    return fit[a] > fit[b];
                                  });
                        std::sort(byFlow.begin(), byFlow.end(),
                                  [&](size_t a, size_t b) {
                                    return share[a] > share[b];
                                  });
                        bool sameOrder = true;
                        for (size_t i = 0; i < n; ++i)
                          if (byFit[i] != byFlow[i]) sameOrder = false;
                        if (config_.httpLog) {
                          std::cout << "[flow]";
                          for (size_t i = 0; i < n; ++i)
                            std::cout << " " << aliasRows[i].path << "="
                                      << share[i];
                          std::cout << " orderEqual=" << (sameOrder ? 1 : 0)
                                    << std::endl;
                        }
                      }
                      std::vector<double> fit;
                      std::vector<double> trait;
                      fit.reserve(aliasRows.size());
                      for (const auto &file : aliasRows) {
                        fit.push_back(
                            static_cast<double>(file.line.state.fecundity));
                        trait.push_back(
                            phoenix::memetic::lineageAdmitted(file.line.state)
                                ? 1.0
                                : 0.0);
                      }
                      const double selection =
                          phoenix::memetic::priceSelectionOf(fit, trait);
                      if (config_.httpLog)
                        std::cout << "[gloss-price] selection=" << selection
                                  << std::endl;
                      std::error_code dirEc;
                      fs::create_directories("runtime_store", dirEc);
                      std::ofstream aliasOut("runtime_store/meme_alias.json",
                                             std::ios::trunc);
                      if (aliasOut)
                        aliasOut << rows.dump();
                    };
                    /* Independent alias recall: the persistent alias table
                       (path + learned words) is a file set of its own.  It
                       does NOT depend on the self-mod explore list, so a
                       normal code-description mission can exercise the
                       MLP-before-cond order.  The router is trained on the
                       alias words; the teacher profiles are the same words. */
                    bool aliasFromMlp = false;
                    bool aliasFromCond = false;
                    bool aliasCondStable = false;
                    std::string aliasChosen;
                    {
                      /* Signal trace: the alias query is the workspace
                         deliverable.md (the user's description), NOT the
                         self-mod source body that may have replaced
                         currentFile/deliverable by this point. */
                      std::string aliasQuery;
                      try {
                        const auto qr = phoenix::mission::workspaceExecute(
                            workspaceRoot, scope,
                            nlohmann::json{{"action", "read"},
                                           {"path", "deliverable.md"}});
                        if (qr.value("ok", false))
                          aliasQuery = qr.value("content", std::string());
                      } catch (...) {
                      }
                      /* Trace found: the self-mod worker overwrites
                         deliverable.md/currentFile with the 1-2 KB source
                         body, so the user's short description signal is
                         lost before the alias recall.  A long deliverable
                         is a work product, not a description: fall back to
                         the goal (the stable description). */
                      if (aliasQuery.empty() || aliasQuery.size() > 400)
                        aliasQuery = goal;
                      if (config_.httpLog)
                        std::cout << "[sig] alias-query len="
                                  << aliasQuery.size() << " delivLen="
                                  << deliverable.size() << " curLen="
                                  << currentFile.size() << " focus="
                                  << (focus.empty() ? 0 : 1) << " selfMod="
                                  << (selfModGoal ? 1 : 0) << " rows="
                                  << aliasRows.size() << std::endl;
                      const auto q = phoenix::memetic::condTokens(aliasQuery);
                      if (q.size() >= 2 && q.size() <= 40 &&
                          aliasRows.size() >= 2) {
                        if (config_.httpLog)
                          std::cout << "[sig] alias-recall q=" << q.size()
                                    << " rows=" << aliasRows.size()
                                    << std::endl;
                        std::vector<std::pair<std::string,
                                              std::vector<std::string>>>
                            aliasBags;
                        std::vector<std::pair<std::string, std::string>>
                            aliasProfiles;
                        std::unordered_set<std::string> aliasVocab;
                        int bodyBags = 0;
                        for (const auto &row : aliasRows) {
                          std::vector<std::string> toks;
                          std::string words;
                          /* The router bag comes from the SOURCE TEXT of the
                             candidate file (distinctive words vs the other
                             candidates), not from the alias table's stored
                             terms.  The table only supplies the candidate path
                             list; this is what makes the MLP train the
                             file<->word mapping from the project itself. */
                          std::string body;
                          {
                            std::ifstream af(row.path, std::ios::binary);
                            body.assign(
                                (std::istreambuf_iterator<char>(af)),
                                std::istreambuf_iterator<char>());
                          }
                          if (!body.empty()) {
                            std::vector<std::pair<std::string, std::string>>
                                others;
                            for (const auto &r2 : aliasRows) {
                              if (r2.path == row.path) continue;
                              std::ifstream bf(r2.path, std::ios::binary);
                              others.emplace_back(
                                  r2.path,
                                  std::string(
                                      (std::istreambuf_iterator<char>(bf)),
                                      std::istreambuf_iterator<char>()));
                            }
                            const std::string prof =
                                phoenix::memetic::distinctiveProfile(
                                    row.path, body, others);
                            for (const auto &t :
                                 phoenix::memetic::glossTokens(prof)) {
                              toks.push_back(t);
                              aliasVocab.insert(t);
                              if (!words.empty()) words.push_back(' ');
                              words += t;
                            }
                          }
                          const bool fromBody = !toks.empty();
                          if (toks.empty()) {
                            for (const auto &term : row.terms) {
                              auto tt = phoenix::memetic::glossTokens(term.word);
                              for (const auto &t : tt) {
                                toks.push_back(t);
                                aliasVocab.insert(t);
                                if (!words.empty()) words.push_back(' ');
                                words += t;
                              }
                            }
                          }
                          if (toks.empty()) continue;
                          if (fromBody) ++bodyBags;
                          aliasBags.emplace_back(row.path, std::move(toks));
                          aliasProfiles.emplace_back(row.path, words);
                        }
                        auto runAliasRouter = [&]() -> bool {
                          if (aliasBags.size() < 2) return false;
                          phoenix::memetic::OpenColumnRouter ar;
                          ar.init(std::max(
                              64, static_cast<int>(aliasVocab.size()) + 8));
                          ar.enableBackground();
                          for (const auto &b : aliasBags)
                            ar.addFileCe(b.first, b.second);
                          const auto r = ar.route(
                              phoenix::memetic::glossTokens(aliasQuery));
                          if (config_.httpLog)
                            std::cout << "[sig] alias-router idx=" << r.index
                                      << " margin=" << r.margin
                                      << " known=" << r.knownTokens
                                      << " bags=" << aliasBags.size()
                                      << std::endl;
                          if (r.index < 0) return false;
                          aliasChosen = ar.fileId(r.index);
                          aliasFromMlp = !aliasChosen.empty();
                          if (config_.httpLog && aliasFromMlp) {
                            const char *src =
                                (bodyBags ==
                                 static_cast<int>(aliasBags.size()))
                                    ? "body"
                                    : "alias-terms";
                            std::cout << "[open-column] mode=ce alias=1 source="
                                      << src << " path=" << aliasChosen
                                      << " margin=" << r.margin
                                      << " known=" << r.knownTokens
                                      << std::endl;
                            /* Explicit zero: the teacher was NOT called on
                               this tick.  Absence of a line is not used.
                               source=body means every bag came from
                               distinctiveProfile of the file text. */
                            std::cout << "[cond-count] alias=1 source=" << src
                                      << " path=" << aliasChosen
                                      << " calls=0" << std::endl;
                          }
                          return aliasFromMlp;
                        };
                        auto runAliasCond = [&]() -> bool {
                          if (aliasProfiles.size() < 2) return false;
                          const int cb =
                              phoenix::memetic::condProfileExchangeCount()
                                  .load();
                          aliasChosen =
                              phoenix::memetic::condProfilePathAudited(
                                  aliasQuery, aliasProfiles,
                                  &aliasCondStable);
                          const int calls =
                              phoenix::memetic::condProfileExchangeCount()
                                  .load() -
                              cb;
                          aliasFromCond = !aliasChosen.empty();
                          if (config_.httpLog)
                            std::cout << "[cond-count] alias=1 path="
                                      << aliasChosen << " calls=" << calls
                                      << std::endl;
                          return aliasFromCond;
                        };
                        const bool routerFirst = routerOrderFromFiles();
                        if (routerFirst) {
                          if (!runAliasRouter()) runAliasCond();
                        } else {
                          if (!runAliasCond()) runAliasRouter();
                        }
                      }
                    }
                    std::vector<phoenix::memetic::OrderedMeme> items;
                    std::vector<std::pair<std::string, std::string>> rawBodies;
                    std::vector<int> fecundity;
                    items.reserve(explorePaths.size());
                    for (const auto &rel : explorePaths) {
                      std::error_code relEc;
                      if (!fs::is_regular_file(rel, relEc))
                        continue;
                      std::ifstream in(rel, std::ios::binary);
                      std::string body((std::istreambuf_iterator<char>(in)),
                                       std::istreambuf_iterator<char>());
                      if (body.size() > 48000)
                        body.resize(48000);
                      rawBodies.emplace_back(rel, body);
                      int fileFecundity = 0;
                      for (const auto &file : aliasRows)
                        if (file.path == rel)
                          fileFecundity = file.line.state.fecundity;
                      fecundity.push_back(fileFecundity);
                      std::string profile = rel + "\n" + body;
                      for (const auto &file : aliasRows) {
                        if (file.path != rel)
                          continue;
                        for (const auto &term : file.terms) {
                          profile.push_back('\n');
                          profile += term.word;
                        }
                      }
                      items.push_back(
                          phoenix::memetic::OrderedMeme{rel, profile});
                    }
                    const auto edit = phoenix::memetic::rankLatentEdit(
                        currentFile, items, fecundity);
                    std::string chosen =
                        !aliasChosen.empty() ? aliasChosen : edit.path;
                    bool fromMlp = aliasFromMlp;
                    bool fromCond = aliasFromCond;
                    bool fromLink = false;
                    bool condStable = aliasCondStable;
                    if (chosen.empty()) {
                      std::vector<std::pair<std::string, std::vector<std::string>>>
                          linkRows;
                      std::vector<int> linkFecundity;
                      for (const auto &file : aliasRows) {
                        if (file.links.empty())
                          continue;
                        linkRows.emplace_back(file.path, file.links);
                        linkFecundity.push_back(file.line.state.fecundity);
                      }
                      chosen = phoenix::memetic::rankBlockedLinks(
                          currentFile, linkRows, linkFecundity);
                      fromLink = !chosen.empty();
                    }
                    /* Router before the conditional teacher: the open-column
                       MLP is trained from the file bodies themselves (no
                       teacher call); it resolves the descriptions the
                       lexical index and the side table cannot, and only
                       the rest reaches the 8107 conditional model.
                       mission.routerBeforeCond=false restores the old
                       order for a same-mission call-count comparison. */
                    auto tryRouter = [&]() -> bool {
                      if (!chosen.empty() || rawBodies.size() < 2) return false;
                      std::vector<std::pair<std::string, std::vector<std::string>>>
                          routerBags;
                      std::unordered_set<std::string> routerVocab;
                      for (const auto &body : rawBodies) {
                        const std::string profile =
                            phoenix::memetic::distinctiveProfile(
                                body.first, body.second, rawBodies);
                        auto tokens = phoenix::memetic::glossTokens(profile);
                        for (const auto &t : tokens)
                          routerVocab.insert(t);
                        routerBags.emplace_back(body.first, std::move(tokens));
                      }
                      const int routerHidden = std::max(
                          64, static_cast<int>(routerVocab.size()) + 8);
                      std::string mode;
                      phoenix::memetic::OpenColumnRouter router;
                      router.init(routerHidden);
                      router.enableBackground();
                      for (const auto &row : routerBags)
                        router.addFileCe(row.first, row.second);
                      const auto route =
                          router.route(phoenix::memetic::glossTokens(currentFile));
                      if (route.index >= 0) {
                        chosen = router.fileId(route.index);
                        fromMlp = !chosen.empty();
                        mode = "ce";
                        if (config_.httpLog)
                          std::cout << "[open-column] mode=ce source=body path="
                                    << chosen
                                    << " margin=" << route.margin
                                    << " known=" << route.knownTokens
                                    << " hidden=" << routerHidden
                                    << std::endl;
                      }
                      if (!fromMlp) {
                        phoenix::memetic::OpenColumnRouter hebb;
                        hebb.init(routerHidden);
                        for (const auto &row : routerBags)
                          hebb.addFile(row.first, row.second);
                        const auto h = hebb.route(
                            phoenix::memetic::glossTokens(currentFile));
                        if (h.index >= 0) {
                          chosen = hebb.fileId(h.index);
                          fromMlp = !chosen.empty();
                          mode = "hebb";
                          if (config_.httpLog)
                            std::cout << "[open-column] mode=hebb path=" << chosen
                                      << " margin=" << h.margin
                                      << " known=" << h.knownTokens
                                      << std::endl;
                        }
                      }
                      if (fromMlp && config_.httpLog)
                        std::cout << "[cond-count] source=body path=" << chosen
                                  << " calls=0" << std::endl;
                      (void)mode;
                      return fromMlp;
                    };
                    const bool routerFirst = routerOrderFromFiles();
                    if (routerFirst) tryRouter();
                    const auto condWords = phoenix::memetic::condTokens(currentFile);
                    if (chosen.empty() && condWords.size() >= 2 &&
                        condWords.size() <= 40) {
                      std::vector<std::pair<std::string, std::string>> condProfiles;
                      for (const auto &body : rawBodies) {
                        const std::string words =
                            phoenix::memetic::distinctiveProfile(
                                body.first, body.second, rawBodies);
                        if (!words.empty())
                          condProfiles.emplace_back(body.first, words);
                      }
                      /* Stability audit: the same description with the
                         profile words in the opposite order must point
                         at the same file.  An unstable teacher hit does
                         not open a file and does not publish words. */
                      const int condBefore =
                          phoenix::memetic::condProfileExchangeCount().load();
                      chosen = phoenix::memetic::condProfilePathAudited(
                          currentFile, condProfiles, &condStable);
                      const int condCalls =
                          phoenix::memetic::condProfileExchangeCount().load() -
                          condBefore;
                      fromCond = !chosen.empty();
                      if (config_.httpLog)
                        std::cout << "[cond-count] path=" << chosen
                                  << " calls=" << condCalls << std::endl;
                    }
                    if (!routerFirst && chosen.empty()) tryRouter();
                    if (!chosen.empty()) {
                      AliasFile *slot = nullptr;
                      for (auto &file : aliasRows)
                        if (file.path == chosen)
                          slot = &file;
                      if (fromMlp) {
                        /* The router picked the file; it did not witness
                           a new conditional pair, so no promotion and no
                           alias growth. */
                      } else if (!fromCond && !fromLink) {
                        if (slot == nullptr) {
                          aliasRows.push_back(AliasFile{chosen, {}, {}});
                          slot = &aliasRows.back();
                        }
                        std::vector<std::string> prior;
                        prior.reserve(slot->terms.size());
                        for (const auto &term : slot->terms)
                          prior.push_back(term.word);
                        const auto grown = phoenix::memetic::witnessedAliases(
                            currentFile, items, chosen, prior);
                        const bool grew = grown.size() > prior.size();
                        if (grew) {
                          std::unordered_set<std::string> have;
                          for (const auto &term : slot->terms)
                            have.insert(term.word);
                          for (const auto &w : grown) {
                            if (!have.insert(w).second)
                              continue;
                            slot->terms.push_back(
                                phoenix::memetic::GlossTerm{w, 1});
                          }
                          slot->line = phoenix::memetic::GlossLineage{};
                        }
                        std::string raw;
                        for (const auto &body : rawBodies)
                          if (body.first == chosen)
                            raw = body.second;
                        const int fecundityBefore = slot->line.state.fecundity;
                        if (!grew && phoenix::memetic::glossCarried(
                                         currentFile, raw, slot->terms))
                          phoenix::memetic::reinforceGloss(
                              slot->terms, slot->line, currentFile);
                        if (grew ||
                            slot->line.state.fecundity != fecundityBefore)
                          saveAliases();
                        if (config_.httpLog &&
                            slot->line.state.fecundity != fecundityBefore)
                          std::cout << "[gloss-lineage] path=" << chosen
                                    << " fecundity="
                                    << slot->line.state.fecundity
                                    << " admitted="
                                    << (phoenix::memetic::lineageAdmitted(
                                            slot->line.state)
                                            ? 1
                                            : 0)
                                    << std::endl;
                      } else if (fromCond) {
                        if (slot == nullptr) {
                          aliasRows.push_back(AliasFile{chosen, {}, {}, {}});
                          slot = &aliasRows.back();
                        }
                        std::vector<std::string> active;
                        active.reserve(slot->terms.size());
                        for (const auto &term : slot->terms)
                          active.push_back(term.word);
                        const auto pendingBefore = slot->pending.size();
                        const auto activeBefore = active.size();
                        /* Two distinct observations are required.  A
                           repeat of any of the last four hits is not a
                           second teacher hit; it is dropped before it
                           can publish words.  The recent window closes
                           the A,B,A alternation. */
                        /* Student cross-check: the self-supervised
                           open-column router (trained on the file bodies,
                           independent of the 8107 teacher) must not point
                           at a different file.  If it does, the teacher
                           hits are not published (the pending list is
                           kept).  No student opinion = allow: local
                           ignorance is not evidence against the hit. */
                        bool studentAgrees = true;
                        {
                          std::vector<std::pair<std::string,
                                                std::vector<std::string>>>
                              sbags;
                          std::unordered_set<std::string> svocab;
                          for (const auto &body : rawBodies) {
                            const std::string profile =
                                phoenix::memetic::distinctiveProfile(
                                    body.first, body.second, rawBodies);
                            auto toks = phoenix::memetic::glossTokens(profile);
                            for (const auto &t : toks) svocab.insert(t);
                            sbags.emplace_back(body.first, std::move(toks));
                          }
                          if (sbags.size() >= 2) {
                            phoenix::memetic::OpenColumnRouter sr;
                            sr.init(std::max(
                                64, static_cast<int>(svocab.size()) + 8));
                            sr.enableBackground();
                            for (const auto &row : sbags)
                              sr.addFileCe(row.first, row.second);
                            const auto srR = sr.route(
                                phoenix::memetic::glossTokens(currentFile));
                            if (srR.index >= 0) {
                              const std::string sp = sr.fileId(srR.index);
                              if (!sp.empty() && sp != chosen)
                                studentAgrees = false;
                            }
                          }
                        }
                        if (!studentAgrees && config_.httpLog)
                          std::cout << "[cond-veto] path=" << chosen
                                    << " student=other" << std::endl;
                        bool condIndependent =
                            studentAgrees &&
                            phoenix::memetic::condRecentIndependent(
                                slot->condRecent, currentFile);
                        const int promoted =
                            condIndependent
                                ? phoenix::memetic::confirmCondWords(
                                      currentFile, items, active,
                                      slot->pending,
                                      slot->condRecent.empty()
                                          ? nullptr
                                          : &slot->condRecent.back())
                                : 0;
                        if (condIndependent) {
                          slot->condRecent.push_back(currentFile);
                          if (slot->condRecent.size() > 4)
                            slot->condRecent.erase(slot->condRecent.begin());
                        }
                        if (config_.httpLog)
                          std::cout << "[cond-hit] path=" << chosen
                                    << " independent="
                                    << (condIndependent ? 1 : 0)
                                    << " stable=" << (condStable ? 1 : 0)
                                    << " promoted=" << promoted
                                    << " pending=" << slot->pending.size()
                                    << std::endl;
                        if (active.size() != activeBefore) {
                          std::unordered_set<std::string> have;
                          for (const auto &term : slot->terms)
                            have.insert(term.word);
                          for (const auto &w : active) {
                            if (!have.insert(w).second)
                              continue;
                            slot->terms.push_back(
                                phoenix::memetic::GlossTerm{w, 1});
                          }
                          slot->line = phoenix::memetic::GlossLineage{};
                        }
                        const auto linksBefore = slot->links.size();
                        const auto pendingLinksBefore = slot->pendingLinks.size();
                        const int linked = phoenix::memetic::confirmBlockedLinks(
                            currentFile, items, slot->links, slot->pendingLinks);
                        if (promoted > 0 || condIndependent ||
                            slot->pending.size() != pendingBefore ||
                            linked > 0 ||
                            slot->pendingLinks.size() != pendingLinksBefore)
                          saveAliases();
                        if (config_.httpLog && promoted > 0)
                          std::cout << "[cond-promote] path=" << chosen
                                    << " words=" << promoted << std::endl;
                        if (config_.httpLog && linked > 0)
                          std::cout << "[link-step] path=" << chosen
                                    << " words=" << linked
                                    << " kept=" << slot->links.size()
                                    << " before=" << linksBefore << std::endl;
                      }
                      std::ifstream ownSrc(chosen, std::ios::binary);
                      std::string src((std::istreambuf_iterator<char>(ownSrc)),
                                      std::istreambuf_iterator<char>());
                      if (!src.empty()) {
                        src = phoenix::memetic::latentCarrierSpan(currentFile,
                                                                 src);
                        if (!src.empty())
                          ownCarrier = phoenix::memetic::exclusiveEditCarrier(
                              chosen, src);
                      }
                      if (!ownCarrier.empty()) {
                        focus = chosen;
                        exploreMark = std::string(fromMlp ? "mlp "
                                                    : fromLink ? "link "
                                                    : fromCond ? "cond "
                                                               : "latent ") +
                                      chosen;
                      }
                      if (config_.httpLog)
                        std::cout << "[latent-edit] path=" << chosen
                                  << " present=" << edit.presentCount
                                  << " cond=" << (fromCond ? 1 : 0)
                                  << " link=" << (fromLink ? 1 : 0)
                                  << " mlp=" << (fromMlp ? 1 : 0)
                                  << " carrier=" << ownCarrier.size()
                                  << std::endl;
                    }
                  }
                  if (config_.httpLog)
                    std::cout << "[explore-tool] " << exploreMark
                              << " candidates=" << explorePaths.size()
                              << std::endl;
                }
                if (codeMission) {
                  const auto measured = phoenix::mission::workspaceExecute(
                      workspaceRoot, scope,
                      nlohmann::json{{"action", "read"},
                                     {"path", "measure.md"}});
                  if (measured.value("ok", false)) {
                    std::string line = measured.value("content", std::string());
                    if (line.size() > 400) line.resize(400);
                    if (!line.empty())
                      phoenix::inference::appendUnitQuery(
                          missionMemorySnippets,
                          phoenix::inference::unitQueryFromText(line, "text"));
                  }
                  std::string peerSrc = phoenix::peer::latestIncomingSource();
                  const bool showPeer = peerSrc.find("384") != std::string::npos;
                  if (showPeer) {
                    if (peerSrc.size() > 1800)
                      peerSrc = peerSrc.substr(peerSrc.size() - 1800);
                    phoenix::inference::appendUnitQuery(
                        missionMemorySnippets,
                        phoenix::inference::unitQueryFromText(
                            "peer function\n" + peerSrc,
                            "text"));
                  }
                }
                if (!pinOnly) {
                  const auto planRes = phoenix::mission::workspaceExecute(
                      workspaceRoot, scope,
                      nlohmann::json{{"action", "read"},
                                     {"path", "plan.md"}});
                  if (planRes.value("ok", false)) {
                    std::string plan = planRes.value("content", std::string());
                    if (plan.size() > 2400)
                      plan = plan.substr(plan.size() - 2400);
                    if (!plan.empty())
                      phoenix::inference::appendUnitQuery(
                          missionMemorySnippets,
                          phoenix::inference::unitQueryFromText(
                              "[plan] " + plan, "text"));
                  }
                }
              }
              nlohmann::json opts = nlohmann::json{
                  {"temperature", phoenix::cfgOr<double>("mission.temperature", 0.35)},
                  {"top_p", phoenix::cfgOr<double>("mission.topP", 0.9)},
                  {"frequency_penalty",
                   phoenix::cfgOr<double>("mission.frequencyPenalty", 0.35)},
                  {"repeat_penalty",
                   phoenix::cfgOr<double>("mission.repeatPenalty", 1.18)},
                  {"repeat_last_n",
                   phoenix::cfgOr<int>("mission.repeatLastN", 512)},
                  {"presence_penalty",
                   phoenix::cfgOr<double>("mission.presencePenalty", 0.15)},
                  {"max_side_tokens",
                   phoenix::cfgOr<int>("mission.maxSideTokens", 768)},
                  {"timeoutMs", phoenix::cfgOr<int>("mission.llmTimeoutMs", 5400000)},
                  {"ctxTokenBudget", packOpt.ctxTokens},
                  {"stop", nlohmann::json::array({
                       "Is there anything else",
                       "Would you like me to",
                       "How can I help",
                       "Please let me know if",
                       "<|eom_id|>",
                       "<|eot_id|>",
                       "<|im_end|>",
                       "<|start_header_id|>",
                       "<|end_of_text|>"})}};
              /* GNN text reply stays off the causal stream. Rows (if
                 any) are already in missionGnnPkts from the graph step. */
              (void)gnnText;
              opts = buildUnitPipelineOptions(nlohmann::json::object(),
                                              missionMemorySnippets,
                                              missionGnnPkts, scope, true, opts);
              opts["rag_mix_cap"] =
                  phoenix::cfgOr<int>("mission.ragMixCap", 384);
              if (!phoenix::cfgOr<bool>("mission.disableParallelDecode", true) &&
                  phoenix::cfgOr<bool>("llama.parallelDecode.enabled", false)) {
                opts["n_parallel"] =
                    phoenix::cfgOr<int>("llama.parallelDecode.N", 4);
                opts["parallel_mode"] =
                    phoenix::cfgOr<std::string>("llama.parallelDecode.mode", "jacobi");
              }
              /* Emotion + pressure modulate sampling and unit weights.
                 Never dump the tensor into the prompt. Instincts only
                 see this contextTag; corpus pretrain used "corpus".
                 Feed the same recent window infer sees. */
              {
                std::string msnId = scope;
                const auto sp0 = msnId.find("/children/");
                if (sp0 != std::string::npos) msnId = msnId.substr(0, sp0);
                const std::string tag = "mission:" + msnId;
                const std::string emoSrc =
                    recallDraft.size() > 4000
                        ? recallDraft.substr(recallDraft.size() - 4000)
                        : recallDraft;
                nlohmann::json esOpts = nlohmann::json::object();
                nlohmann::json esTensor = nlohmann::json::object();
                if (!pinOnly && (!stillDrafting || peerEvolveGoal) &&
                    !emoSrc.empty()) {
                  if (config_.httpLog)
                    std::cout << "[mission-deliberate] stage=emotion"
                              << std::endl;
                  try {
                    cognitionManager_.ingestSensation(nlohmann::json{
                        {"type", "novelty"},
                        {"intensity", 0.28},
                        {"valence", 0.0},
                        {"source", tag + ":draft"}});
                  } catch (...) {
                  }
                  if (emotionSystem_) {
                    try {
                      const auto st = emotionSystem_->processMessage(
                          tag, emoSrc,
                          static_cast<int64_t>(currentFile.size()));
                      esOpts = st.current.inferenceOptions(
                          static_cast<float>(opts.value("temperature", 0.35)),
                          static_cast<float>(opts.value("top_p", 0.9)));
                      esTensor = st.current.toJson();
                      const auto toks =
                          phoenix::mission::splitEmotionVocabTokens(emoSrc);
                      emotionSystem_->observeVocab(
                          tag, toks,
                          static_cast<int>(currentFile.size()));
                      const auto bias = emotionSystem_->getVocabLogitBias(
                          tag, toks,
                          static_cast<int>(currentFile.size()));
                      if (bias.is_object() && !bias.empty())
                        opts["logit_bias"] = bias;
                    } catch (...) {
                    }
                  }
                  try {
                    ingestWorldEvidence(tag, emoSrc, packed.pinnedSummary,
                                        "mission-draft");
                  } catch (...) {
                  }
                }
                const auto emo = cognitionManager_.evaluateInstinctsFor(tag);
                const auto &emoRes =
                    emo.value("result", nlohmann::json::object());
                const float pressure =
                    cognitionManager_.missionPressureFor(msnId);
                phoenix::mission::applyEmotionSampling(
                    opts, emoRes.value("inferenceOptions", nlohmann::json::object()),
                    emoRes.value("emotionTensor", nlohmann::json::object()),
                    pressure);
                if (!esOpts.empty())
                  phoenix::mission::applyEmotionSampling(
                      opts, esOpts, esTensor, pressure * 0.5f);
              }
              /* Causal I/O is the deliverable file. Assignment /
                 plugin / memory / GNN are unit queries on RAG.
                 Emotion is the matrix-control interface. */
              (void)toolHint;
              (void)isChildBox;
              if (pinOnly)
                opts["repeat_penalty"] = 1.0;
              std::string dynamicPrompt = sectionTip;
              (void)listClip;
              std::string resumeSuffix;
              /* The causal tip is the unfinished section, not the requirements
                 tail. Continuing that tail writes another "shall" list. */
              if (!sectionTip.empty())
                resumeSuffix = sectionTip;
              else if (!draftEmpty)
                resumeSuffix = packed.recentFull.empty() ? currentFile
                                                         : packed.recentFull;
              if (onesJunkDraft) {
                resumeSuffix.clear();
                opts.erase("resume_suffix");
              }
              if (draftEmpty) {
                opts.erase("gnn_units");
                opts.erase("gnn_content");
              }
              /* SELF-MOD does not paste a source span or the previous
                 fictional draft onto the causal tip. A file body goes only
                 to the expert, and only after the draft names a real path. */
              if (selfModGoal && ownCarrier.empty()) {
                resumeSuffix.clear();
                opts.erase("resume_suffix");
              } else if (peerEvolveGoal && !peerEditBase.empty()) {
                resumeSuffix = peerEditBase;
              }
              if (selfModGoal && ownCarrier.empty() && resumeSuffix.size() > 1800)
                resumeSuffix = resumeSuffix.substr(resumeSuffix.size() - 1800);
              if (!resumeSuffix.empty())
                opts["resume_suffix"] = resumeSuffix;
              if (config_.httpLog) {
                std::string tip = !resumeSuffix.empty() ? resumeSuffix
                                                        : staticPrompt;
                if (tip.size() > 160) tip = tip.substr(tip.size() - 160);
                for (char &c : tip) {
                  if (c == '\n' || c == '\r') c = ' ';
                }
                std::cout << "[mission-deliberate] causalStaticLen="
                          << staticPrompt.size()
                          << " causalTokBudget=" << packed.causalTokenBudget
                          << " recentChars=" << packed.fullCharsUsed
                          << " tip=" << tip
                          << std::endl;
              }
              /* v10.1 hot-path: variant-aware prefix slots.  The hidden
                 condition = the session-varying static-prefix shape (its
                 length, the memory/GNN units, the pressure quantile).  The
                 fingerprint is computed at runtime and never compiled in,
                 so a changed session lands in its own slot and the server's
                 KV reuse follows the variant instead of colliding across
                 them.  Gated by research.hotPath.prefixSlots (default off =
                 the pre-v10.1 single-slot behaviour). */
              if (phoenix::cfgOr<bool>("research.hotPath.prefixSlots",
                                       false)) {
                std::string msnBase = scope;
                const auto spP2 = msnBase.find("/children/");
                if (spP2 != std::string::npos)
                  msnBase = msnBase.substr(0, spP2);
                const float slotPressure =
                    cognitionManager_.missionPressureFor(msnBase);
                const std::string fpInput =
                    std::to_string(staticPrompt.size()) + "|" +
                    std::to_string(packed.fullCharsUsed) + "|" +
                    std::to_string(missionMemorySnippets.size()) + "|" +
                    std::to_string(missionGnnPkts.size()) + "|" +
                    std::to_string(static_cast<int>(slotPressure * 1000.0f));
                uint64_t fp = 0xcbf29ce484222325ULL;
                for (unsigned char c : fpInput) {
                  fp ^= static_cast<uint64_t>(c);
                  fp *= 0x100000001b3ULL;
                }
                opts["id_slot"] = "phx-" + std::to_string(fp);
                if (config_.httpLog)
                  std::cout << "[prefix-cache] slot=" << opts["id_slot"]
                            << " staticLen=" << staticPrompt.size()
                            << " recent=" << packed.fullCharsUsed
                            << " mem=" << missionMemorySnippets.size()
                            << " gnn=" << missionGnnPkts.size()
                            << " press=" << slotPressure << std::endl;
              }
              if (config_.httpLog) {
                std::string msnId = scope;
                const auto spP = msnId.find("/children/");
                if (spP != std::string::npos) msnId = msnId.substr(0, spP);
                const float pressure =
                    cognitionManager_.missionPressureFor(msnId);
                std::cout << "[mission-deliberate] resumeSuffixLen="
                          << resumeSuffix.size()
                          << " recentChars=" << packed.fullCharsUsed
                          << " droppedChars=" << packed.droppedChars
                          << " causalTokBudget=" << packed.causalTokenBudget
                          << " ngram=" << packOpt.ngramMerge
                          << " memUnits=" << missionMemorySnippets.size()
                          << " gnnUnits=" << missionGnnPkts.size()
                          << " pressure=" << pressure
                          << " top_p=" << opts.value("top_p", 0.0)
                          << " freqP=" << opts.value("frequency_penalty", 0.0)
                          << std::endl;
              }
              /* v8.x D feedback: read the one-shot hint from the last
                 failed tool call, then consume it (delete). */
              {
                const auto fbRes = phoenix::mission::workspaceExecute(
                    workspaceRoot, scope,
                    nlohmann::json{{"action", "read"},
                                   {"path", "tool-feedback.md"}});
                if (fbRes.value("ok", false)) {
                  std::string fb = fbRes.value("content", std::string());
                  if (fb.size() > 300) fb = fb.substr(fb.size() - 300);
                  if (!fb.empty()) {
                    /* RAG, not causal. Compact mode used to swallow this. */
                    const auto pkt = phoenix::inference::unitQueryIOToJson(
                        phoenix::inference::unitQueryFromText(
                            "[tool-feedback] " + fb, "text"));
                    if (!opts.contains("context_units") ||
                        !opts["context_units"].is_array())
                      opts["context_units"] = nlohmann::json::array();
                    opts["context_units"].push_back(pkt);
                    phoenix::mission::workspaceExecute(
                        workspaceRoot, scope,
                        nlohmann::json{{"action", "delete"},
                                       {"path", "tool-feedback.md"}});
                  }
                }
              }
              /* Successful tool output from the previous tick (one-shot). */
              {
                const auto trRes = phoenix::mission::workspaceExecute(
                    workspaceRoot, scope,
                    nlohmann::json{{"action", "read"},
                                   {"path", "tool-last-result.md"}});
                if (trRes.value("ok", false)) {
                  std::string trb = trRes.value("content", std::string());
                  if (trb.size() > 400) trb = trb.substr(trb.size() - 400);
                  if (!trb.empty()) {
                    const auto pkt = phoenix::inference::unitQueryIOToJson(
                        phoenix::inference::unitQueryFromText(
                            "[tool-result] " + trb, "text"));
                    if (!opts.contains("context_units") ||
                        !opts["context_units"].is_array())
                      opts["context_units"] = nlohmann::json::array();
                    opts["context_units"].push_back(pkt);
                    phoenix::mission::workspaceExecute(
                        workspaceRoot, scope,
                        nlohmann::json{{"action", "delete"},
                                       {"path", "tool-last-result.md"}});
                  }
                }
              }
              /* Image concept notes and helper-box ids are units, not
                 a second text dump on the causal document. */
              {
                const auto attRes = phoenix::mission::workspaceExecute(
                    workspaceRoot, scope,
                    nlohmann::json{{"action", "read"},
                                   {"path", "attachments.json"}});
                if (attRes.value("ok", false)) {
                  const auto att = nlohmann::json::parse(
                      attRes.value("content", std::string("{}")), nullptr,
                      false);
                  if (!att.is_discarded() && att.is_object() &&
                      att.contains("images") && att["images"].is_array() &&
                      !att["images"].empty()) {
                    std::string blob;
                    for (const auto &img : att["images"]) {
                      if (!blob.empty()) blob += "\n";
                      blob += img.value("file", std::string()) + ": " +
                              img.value("concept", std::string());
                    }
                    if (!blob.empty()) {
                      const auto pkt = phoenix::inference::unitQueryIOToJson(
                          phoenix::inference::unitQueryFromText(blob, "text"));
                      if (!opts.contains("context_units") ||
                          !opts["context_units"].is_array())
                        opts["context_units"] = nlohmann::json::array();
                      opts["context_units"].push_back(pkt);
                    }
                  }
                }
              }
              if (!isChildBox) {
                try {
                  std::string msnId = scope;
                  const auto slashPos = msnId.find("/children/");
                  if (slashPos != std::string::npos)
                    msnId = msnId.substr(0, slashPos);
                  auto ms2 = cognitionManager_.missionStatus();
                  nlohmann::json children2 = nlohmann::json::array();
                  for (const auto &mm : ms2.value("result", nlohmann::json::object())
                                             .value("missions", nlohmann::json::array())) {
                    if (mm.value("id", std::string()) == msnId) {
                      children2 = mm.value("stats", nlohmann::json::object())
                                     .value("children", nlohmann::json::array());
                      break;
                    }
                  }
                  if (children2.is_array() && !children2.empty()) {
                    std::string blob;
                    for (const auto &box : children2) {
                      const std::string id = box.value("id", std::string());
                      if (id.empty()) continue;
                      if (!blob.empty()) blob += " ";
                      blob += id;
                    }
                    if (!blob.empty()) {
                      const auto pkt = phoenix::inference::unitQueryIOToJson(
                          phoenix::inference::unitQueryFromText(blob, "text"));
                      if (!opts.contains("context_units") ||
                          !opts["context_units"].is_array())
                        opts["context_units"] = nlohmann::json::array();
                      opts["context_units"].push_back(pkt);
                    }
                  }
              } catch (...) {
              }
              }

              /* Heritable per-tick generation budget.  The mission's
                 genome (or the helper box's own genome, for a child
                 scope) decides how many tokens this tick asks for.  The
                 genome is read before the prompt reserve is computed so
                 a large budget still leaves room for the prompt. */
              float genomeTokens = 0.0f;
              try {
                const auto genomeJson =
                    cognitionManager_.missionGenomeFor(scope);
                if (genomeJson.is_object())
                  genomeTokens = genomeJson.value("tokensPerTick", 0.0f);
              } catch (...) {
              }
              if (config_.httpLog)
                std::cout << "[mission-genome] scope=" << scope
                          << " tokensPerTick=" << genomeTokens << std::endl;
              const int genReserve = std::max(
                  16, genomeTokens > 0.0f
                          ? std::min(512, static_cast<int>(genomeTokens + 0.5f))
                          : phoenix::cfgOr<int>("mission.deliberateMaxTokens", 48));
              const int tmplReserve =
                  phoenix::cfgOr<int>("mission.templateOverheadTokens", 64);
              const int safePromptTokens = std::max(
                  512, packOpt.ctxTokens - genReserve - tmplReserve -
                           packOpt.overheadTokens);
              const bool packOn =
                  phoenix::cfgOr<bool>("main.inference.packPrompt", false);
              const int ngramFac = phoenix::inference::causalBudgetNgram(
                  packOn, packOpt.ngramMerge);
              const size_t maxPromptChars =
                  static_cast<size_t>(safePromptTokens) *
                  phoenix::context::charsPerPackedToken() *
                  static_cast<size_t>(ngramFac);
              /* Plain-continuation contract.  The goal text is not enough;
                 the worker's RAG stream advertises tools, so the system
                 side has to say no. */
              if (plainTextOnly) {
                staticPrompt +=
                    "\n\nWRITE PLAIN PROSE ONLY. Continue the document with "
                    "new sentences. Do not emit JSON, tool calls, plans, "
                    "markdown headings, or code fences.\n";
              }
              /* Split path: staticPrompt is the document head.
                 Tail is resume_suffix, not a second prompt protocol. */
              {
                const auto fitted = phoenix::mission::fitMissionPromptSplit(
                    staticPrompt, dynamicPrompt, maxPromptChars);
                staticPrompt = fitted.first;
                dynamicPrompt = fitted.second;
              }
              /* Mission ticks need a longer socket timeout than interactive
                 chat: even a slim prompt is slow on RDK CPU (~2s/token). */
              /* v8.x L1 lookup: if the dynamic turn is (near-)identical to
                 the previous tick the model would repeat itself - skip the
                 LLM entirely (saves minutes/tick on RDK).  After 3
                 consecutive skips, force an advance hint + hotter sampling
                 on the next real call to break the cycle.
                 Never skip while deliverable is still below minDeliverableChars.
                 Do not append rewrite hints onto dynamicPrompt: that tip
                 became the deliverable on the board (Do not omit 1.1.1). */
              {
                const bool cacheEnabled =
                    phoenix::cfgOr<bool>("mission.cacheEnabled", false);
                const double cacheSim =
                    phoenix::cfgOr<double>("mission.cacheSim", 0.95);
                const size_t delivSize = currentFile.size();
                /* L1 cache is OFF by default: skipping the LLM while the prompt
                   is similar caused llama idle + zero/stalled deliverable on
                   RDK.  When enabled, never skip unless deliverable grew since
                   the last real generation. */
                bool allowCacheSkip = false;
                if (cacheEnabled && !stillDrafting && !currentFile.empty() &&
                    phoenix::mission::workspaceCacheGet(scope, dynamicPrompt,
                                                        cacheSim)) {
                  size_t lastSz = 0;
                  int noGrowth = 0;
                  {
                    std::lock_guard<std::mutex> lk(gDelibSkipMu);
                    lastSz = gDelibLastDelivAtLlm[scope];
                    noGrowth = gDelibCacheSkipsNoGrowth[scope];
                  }
                  if (delivSize > lastSz) {
                    allowCacheSkip = true;
                  } else if (noGrowth < 2) {
                    {
                      std::lock_guard<std::mutex> lk(gDelibSkipMu);
                      gDelibCacheSkipsNoGrowth[scope] = noGrowth + 1;
                    }
                    allowCacheSkip = true;
                  }
                }
                if (allowCacheSkip) {
                  int cnt = 0;
                  {
                    std::lock_guard<std::mutex> lk(gDelibSkipMu);
                    cnt = ++gDelibSkipCount[scope];
                  }
                  if (cnt >= 3) {
                    {
                      std::lock_guard<std::mutex> lk(gDelibSkipMu);
                      gDelibSkipCount[scope] = 0;
                    }
                    opts["temperature"] = 0.7;
                    dynamicPrompt += "\n[loop-break] You repeated yourself. ADVANCE to the next "
                                       "unfinished section; do NOT rewrite existing text.\n";
                    {
                      const auto fitted = phoenix::mission::fitMissionPromptSplit(
                          staticPrompt, dynamicPrompt, maxPromptChars);
                      staticPrompt = fitted.first;
                      dynamicPrompt = fitted.second;
                    }
                    /* fall through: a real call with the loop-break hint */
                  } else {
                    return "[cache] prompt unchanged; skipped identical "
                           "re-generation (cycle guard)";
                  }
                }
              }
              int tokBudget = std::max(16, maxTokens > 0 ? maxTokens : 48);
              {
                const int draftCap = phoenix::cfgOr<int>(
                    "mission.draftTickMaxTokens", 64);
                const int genomeBudget =
                    genomeTokens > 0.0f
                        ? phoenix::mission::missionTickBudget(
                              genomeTokens, 512, false, draftCap)
                        : 0;
                /* 192-token ticks plus a fat prompt ate the 8–15 min
                   short-test window (one 160-char write, no second
                   body). Cap while the draft is still under min chars
                   so a second tick can finish and complete can wait
                   the last generate out in << 180s. */
                if (peerEvolveGoal && (codeMission || selfModGoal)) {
                  tokBudget = std::max(tokBudget, 512);
                } else if (genomeBudget > 0) {
                  tokBudget = genomeBudget;
                } else if (stillDrafting) {
                  tokBudget = std::min(std::max(tokBudget, 32),
                                       std::max(32, draftCap));
                } else {
                  const int deliberateTok = phoenix::cfgOr<int>(
                      "mission.deliberateMaxTokens", tokBudget);
                  tokBudget = std::max(tokBudget, std::max(16, deliberateTok));
                }
                opts["num_predict"] = tokBudget;
                opts["deliberateMaxTokens"] = tokBudget;
                /* Split /phx/generate does not preserve prefix-cache across
                   requests (KV is cleared per call); forcing cache_prompt only
                   adds unused fields. */
                opts["cache_prompt"] = false;
                if (config_.httpLog)
                  std::cout << "[mission-deliberate] tokBudget=" << tokBudget
                            << " genome=" << (genomeBudget > 0 ? 1 : 0)
                            << " stillDrafting=" << (stillDrafting ? 1 : 0)
                            << " onesJunk=" << (onesJunkDraft ? 1 : 0)
                            << " deliv=" << currentFile.size() << std::endl;
              }
              /* v8.6 summon routing: a helper box spawned with a model
                 field has its workspace scope registered in the summon
                 registry; its ticks go to that external endpoint (the
                 adapter translates the prompt - the model is never taught
                 a protocol).  Base-model boxes take the normal slot path. */
              json r;
              const std::string summonModel =
                  phoenix::summon::SummonRegistry::instance().modelForScope(
                      scope);
              const auto *summonEp =
                  summonModel.empty()
                      ? nullptr
                      : phoenix::summon::SummonRegistry::instance().find(
                            summonModel);
              phoenix::summon::ModelEndpoint toolSummoned;
              bool summonedByTool = false;
              if (summonEp == nullptr && selfModGoal) {
                /* The model is not told that summon or field training exist.
                   The tools look at the source-edit situation and run. */
                const std::string fieldText =
                    currentFile.empty() ? goal : currentFile;
                try {
                  const auto trained =
                      cognitionManager_.noteFieldOpportunity(fieldText);
                  if (config_.httpLog)
                    std::cout << "[field-train] unit="
                              << (trained.value("unitLearn", false) ? 1 : 0)
                              << " live="
                              << (trained.value("liveTurns", false) ? 1 : 0)
                              << " expert="
                              << trained.value("expert", std::string())
                              << " mode="
                              << trained.value("expertMode", std::string())
                              << " acc="
                              << trained.value("expertAccuracy", 0.0)
                              << " params="
                              << trained.value("expertParams", 0)
                              << " trainLmax="
                              << trained.value("expertLmax", 0.0)
                              << std::endl;
                } catch (const std::exception &ex) {
                  if (config_.httpLog)
                    std::cout << "[field-train] FAIL " << ex.what()
                              << std::endl;
                } catch (...) {
                  if (config_.httpLog)
                    std::cout << "[field-train] FAIL" << std::endl;
                }
                toolSummoned = phoenix::summon::SummonRegistry::instance()
                                   .selectForSourceEdit();
                if (!toolSummoned.name.empty()) {
                  summonEp = &toolSummoned;
                  summonedByTool = true;
                  if (config_.httpLog)
                    std::cout << "[summon-tool] source-edit model="
                              << toolSummoned.name
                              << " kind=" << toolSummoned.kind << std::endl;
                } else if (config_.httpLog) {
                  std::cout << "[summon-tool] source-edit idle "
                               "reason=no-programming-moe"
                            << std::endl;
                }
              }
              opts["system_content"] = staticPrompt;
              /* Code has to keep its spaces. The global pack/ngram path
                 glues identifiers together, so this mission sends plain tokens. */
              if (peerEvolveGoal && (codeMission || selfModGoal)) {
                opts["pack_prompt"] = false;
                opts["group_n"] = 1;
                opts["ngram_merge"] = 1;
                opts["n_parallel"] = 1;
                opts["speculative"] = false;
                opts["parallel_mode"] = "";
              }
              if (summonedByTool && ownCarrier.empty()) {
                summonMark = "idle:no-file-yet";
                if (config_.httpLog)
                  std::cout << "[summon-tool] idle reason=no-file-yet"
                            << std::endl;
                summonEp = nullptr;
                summonedByTool = false;
              } else if (summonedByTool) {
                summonMark = "called";
                /* v10.2 scored summoning: count this call as in-flight so a
                   busy expert is not picked again while it works. */
                phoenix::summon::SummonRegistry::instance().noteSummonStart(
                    summonEp->name);
              }
              if (summonEp) {
                if (config_.httpLog)
                  std::cout << "[mission-deliberate] summon scope=" << scope
                            << " model=" << summonEp->name
                            << " kind=" << summonEp->kind
                            << " url=" << summonEp->url << std::endl;
                const int summonTokens =
                    summonedByTool ? std::max(summonEp->maxTokens, 256)
                                   : tokBudget;
                /* Declared outside the carrier guard so the 8B fallback
                   can reuse the same patch instruction. */
                std::string editPrompt;
                json expertOpts = opts;
                expertOpts["temperature"] = 0.0;
                expertOpts["top_p"] = 0.9;
                expertOpts["repeat_penalty"] = 1.05;
                if (summonedByTool && !ownCarrier.empty()) {
                  if (config_.httpLog)
                    std::cout << "[summon-tool] carrier bytes="
                              << ownCarrier.size() << std::endl;
                  /* Expert side: ask the programming model for exactly one
                     patch.  The plugin side (code_edit_parser) normalises
                     a fenced/raw code reply into the same FILE/OLD/NEW
                     form, so both paths reach the source-edit detector.
                     A 1B expert degenerates on the full 1.5 KB carrier
                     (token salad observed live), so send a bounded tail and
                     greedy decoding. */
                  std::string expertCode = ownCarrier;
                  if (expertCode.size() > 600)
                    expertCode = expertCode.substr(expertCode.size() - 600);
                  editPrompt =
                      "You edit C++ source. Reply with only one patch for "
                      "the file `" +
                      (focusTarget.empty() ? std::string("the given file")
                                           : focusTarget) +
                      "`, no prose. Example reply for a file a.cpp:\n"
                      "FILE a.cpp\nOLD\nint x = 1;\nNEW\nint x = 2;\n\n"
                      "Now edit this code the same way:\n" +
                      expertCode;
                  /* p-injection (the controlled variable of the live force
                     experiment): the goal carries "[population-rate <p>]";
                     the carrier's decision context then sees the current
                     trait frequency, which is the conformity term's input. */
                  const auto ratePos = goal.find("[population-rate ");
                  if (ratePos != std::string::npos &&
                      phoenix::cfgOr<bool>("research.forceLive.enabled",
                                           false)) {
                    editPrompt += "\n[population] the current share of "
                                  "carriers that produced a source edit in "
                                  "the previous round is: " +
                                  goal.substr(ratePos) + "\n";
                  }
                  r = chatWithSummonEndpoint(*summonEp, editPrompt,
                                             std::string(), summonTokens,
                                             expertOpts);
                } else {
                  r = chatWithSummonEndpoint(*summonEp, dynamicPrompt,
                                             staticPrompt, summonTokens, opts);
                }
                if (summonedByTool) {
                  std::string summonedReply =
                      r.value("reply", std::string());
                  bool patch =
                      summonedReply.find("diff --git ") != std::string::npos ||
                      summonedReply.find("--- a/") != std::string::npos ||
                      (summonedReply.find("FILE ") != std::string::npos &&
                       summonedReply.find("\nOLD") != std::string::npos &&
                       summonedReply.find("\nNEW") != std::string::npos);
                  if (!patch && r.value("ok", false)) {
                    const std::string normalised =
                        phoenix::code_edit::extractCodeEdit(summonedReply,
                                                            std::string());
                    if (!normalised.empty()) {
                      r["reply"] = normalised;
                      summonedReply = normalised;
                      patch = true;
                      if (config_.httpLog)
                        std::cout << "[summon-tool] plugin-normalised code "
                                  << normalised.size() << " bytes"
                                  << std::endl;
                    }
                  }
                  /* One pointed retry before the base fallback: the coder's
                     first answer is sometimes prose.  Naming the target and
                     the exact three-part shape lifts the source-edit rate. */
                  if (!patch && r.value("ok", false) && !focusTarget.empty()) {
                    std::string retryCode = ownCarrier;
                    if (retryCode.size() > 600)
                      retryCode = retryCode.substr(retryCode.size() - 600);
                    const std::string retryPrompt =
                        "Reply with ONLY this exact format, no prose and no "
                        "fences:\nFILE " + focusTarget +
                        "\nOLD\n<one original line>\nNEW\n<one changed line>\n\n"
                        "Edit this code:\n" + retryCode;
                    json r2 = chatWithSummonEndpoint(
                        *summonEp, retryPrompt, std::string(), summonTokens,
                        expertOpts);
                    std::string rr = r2.value("reply", std::string());
                    bool p2 = rr.find("--- a/") != std::string::npos ||
                              (rr.find("FILE ") != std::string::npos &&
                               rr.find("\nOLD") != std::string::npos &&
                               rr.find("\nNEW") != std::string::npos);
                    if (!p2 && r2.value("ok", false)) {
                      const std::string nz =
                          phoenix::code_edit::extractCodeEdit(rr, focusTarget);
                      if (!nz.empty()) {
                        rr = nz;
                        p2 = true;
                      }
                    }
                    if (p2) {
                      r = r2;
                      r["reply"] = rr;
                      patch = true;
                      if (config_.httpLog)
                        std::cout << "[summon-tool] retry-ok bytes="
                                  << rr.size() << std::endl;
                    } else if (config_.httpLog) {
                      std::cout << "[summon-tool] retry-fail len="
                                << rr.size() << std::endl;
                    }
                  }
                  /* Live-force instrument (research.forceLive.enabled):
                     the source-edit tool has run and `patch` is the actual
                     outcome, so the tick is honest (prose replies count as
                     non-edits).  The generation is the global loop round. */
                  if (phoenix::cfgOr<bool>("research.forceLive.enabled",
                                           false)) {
                    phoenix::memetic::ReplicaTick liveTick;
                    liveTick.instance = scope;
                    liveTick.mission = scope;
                    liveTick.replyClass = patch ? "source-edit" : "prose";
                    liveTick.replyChars = static_cast<int>(
                        r.value("reply", std::string()).size());
                    liveTick.sourceEdit = patch;
                    liveTick.generation = static_cast<int>(
                        cognitionManager_.loopTickCount());
                    phoenix::memetic::appendReplicaTick(
                        "runtime_store/replica_trace", liveTick);
                  }
                  /* v10.2: feed the scored selector.  A reply that yielded a
                     patch is a success; prose/failure is a failure - which is
                     what makes EndpointRanker stop summoning a degenerate
                     expert after a couple of tries instead of forever. */
                  phoenix::summon::SummonRegistry::instance().noteSummonOutcome(
                      summonEp->name, patch);
                  if (!r.value("ok", false) || !patch) {
                    if (config_.httpLog) {
                      std::string head = summonedReply.substr(
                          0, std::min<size_t>(120, summonedReply.size()));
                      for (char &c : head)
                        if (c == '\n' || c == '\r') c = ' ';
                      std::cout << "[summon-tool] fallback base reason="
                                << (!r.value("ok", false)
                                        ? "call-failed"
                                        : "not-a-source-edit")
                                << " replyLen=" << summonedReply.size()
                                << " head=" << head << std::endl;
                    }
                    /* The 1B programming MoE degenerates on the carrier;
                       the 8B base is the stronger code writer.  Ask it
                       with the SAME patch instruction, not the generic
                       mission prompt, so a source edit can come back. */
                    r = chatWithLlamaCpp(editPrompt, std::string(),
                                         tokBudget, expertOpts);
                  }
                }
              } else {
                r = chatWithLlamaCpp(dynamicPrompt, staticPrompt, tokBudget,
                                     opts);
              }
              if (config_.httpLog && !r.value("ok", false)) {
                std::cout << "[mission-deliberate] llamacpp FAIL err="
                          << r.value("error", std::string()).substr(0, 320)
                          << " staticLen=" << staticPrompt.size()
                          << " dynLen=" << dynamicPrompt.size()
                          << std::endl;
              }
              if (!r.value("ok", false)) {
                const std::string err = r.value("error", std::string("llm failed"));
                return "[llm-error] " + err.substr(0, 240);
              }
              std::string reply = r.value("reply", std::string());
              if (reply.empty())
                return "[llm-error] empty reply (possible cancel/timeout)";
              if (config_.httpLog) {
                std::cout << "[mission-deliberate] replyLen=" << reply.size()
                          << " head=" << reply.substr(0, 160);
                if (reply.size() < 8) {
                  std::cout << " hex=";
                  for (unsigned char c : reply) {
                    char b[4];
                    std::snprintf(b, sizeof(b), "%02x", c);
                    std::cout << b;
                  }
                }
                std::cout << std::endl;
              }

              /* structured reply: tool call or file action */
              {
                /* v8.x tool-friendly intent resolver: natural language
                   ("calculate 3*9", "summon a box for the examples section")
                   or malformed JSON maps onto the registered tools so the
                   model does not need to emit perfect schemas. */
                auto resolveToolIntent = [&](const std::string &text,
                                             std::string &toolName,
                                             nlohmann::json &ctx) -> bool {
                  std::string lower = text;
                  std::transform(lower.begin(), lower.end(), lower.begin(),
                                 [](unsigned char c) {
                                   return static_cast<char>(::tolower(c));
                                 });
                  nlohmann::json toolList = nlohmann::json::array();
                  try {
                    const auto la = cognitionManager_.listAgiActions();
                    toolList = la.value("result", nlohmann::json::array());
                  } catch (...) {
                  }
                  /* 1) exact tool-name mention - skip for long document drafts
                     (a long chapter can casually mention a tool name). */
                  const bool looksLikeDocument =
                      text.size() > 180 ||
                      lower.rfind("**", 0) == 0 ||
                      lower.rfind("#", 0) == 0 ||
                      lower.rfind("chapter ", 0) == 0 ||
                      lower.find("\n**") != std::string::npos ||
                      lower.find("\n#") != std::string::npos;
                  auto wholeToken = [&](const std::string &word) {
                    if (word.empty()) return false;
                    std::size_t pos = 0;
                    while ((pos = lower.find(word, pos)) != std::string::npos) {
                      const bool left =
                          pos == 0 ||
                          !std::isalnum(static_cast<unsigned char>(lower[pos - 1]));
                      const bool right =
                          pos + word.size() >= lower.size() ||
                          !std::isalnum(static_cast<unsigned char>(
                              lower[pos + word.size()]));
                      if (left && right) return true;
                      pos += word.size();
                    }
                    return false;
                  };
                  if (!looksLikeDocument) {
                    for (const auto &t : toolList) {
                      const std::string nm = t.value("name", std::string());
                      if (nm.empty()) continue;
                      std::string nmLower = nm;
                      std::transform(nmLower.begin(), nmLower.end(),
                                     nmLower.begin(),
                                     [](unsigned char c) {
                                       return static_cast<char>(::tolower(c));
                                     });
                      /* The tool name has to occur as its own token.
                         "research" is not the token "search". A synonym
                         list is not a copy of the tool name. */
                      if (wholeToken(nmLower)) {
                        toolName = nm;
                        break;
                      }
                    }
                  }
                  /* Index over ordered niches. Not a replicator test:
                     lift and word order can name a tool, and one shared
                     word cannot. A mention of a tool name above is still
                     only the name token; this branch does not add synonyms. */
                  if (toolName.empty()) {
                    const auto guess = phoenix::memetic::inferLatentTool(text);
                    if (!guess.name.empty()) {
                      toolName = guess.name;
                      if (config_.httpLog)
                        std::cout << "[index-tool] niche=" << guess.name
                                  << " lift=" << guess.share
                                  << " order=" << guess.ipr
                                  << " hits=" << guess.hits << std::endl;
                    }
                  }
                  if (toolName.empty()) return false;
                  /* 3) argument: longest quoted span, else the text itself */
                  std::string arg;
                  const auto q1 = text.find('"');
                  if (q1 != std::string::npos) {
                    const auto q2 = text.find('"', q1 + 1);
                    arg = q2 == std::string::npos
                              ? text.substr(q1 + 1)
                              : text.substr(q1 + 1, q2 - q1 - 1);
                  }
                  if (arg.empty()) arg = text;
                  if (arg.size() > 400) arg = arg.substr(0, 400);
                  ctx = nlohmann::json{{"userPrompt", arg}};
                  return true;
                };
                /* v8.x D feedback loop: a failed tool call (or a broken
                   reply) writes ONE hint file; the next tick reads it and
                   deletes it - the model learns the expected form once.
                   Written as a whole file so it never accumulates. */
                auto writeToolFeedback = [&](const std::string &msg) {
                  std::string fb = msg;
                  if (fb.size() > 300) fb = fb.substr(0, 300);
                  phoenix::mission::workspaceExecute(
                      workspaceRoot, scope,
                      nlohmann::json{{"action", "write"},
                                     {"path", "tool-feedback.md"},
                                     {"content", fb}});
                };
                auto writeToolLastResult = [&](const std::string &msg) {
                  std::string body = msg;
                  if (body.size() > 400) body = body.substr(0, 400);
                  phoenix::mission::workspaceExecute(
                      workspaceRoot, scope,
                      nlohmann::json{{"action", "write"},
                                     {"path", "tool-last-result.md"},
                                     {"content", body}});
                };
                auto missionIdFromScope = [&]() -> std::string {
                  std::string msnId = scope;
                  const auto sp = msnId.find("/children/");
                  if (sp != std::string::npos) msnId = msnId.substr(0, sp);
                  return msnId;
                };
                auto syncDeliverableFromWorkspace = [&]() {
                  const auto rd = phoenix::mission::workspaceExecute(
                      workspaceRoot, scope,
                      nlohmann::json{{"action", "read"},
                                     {"path", deliverablePath}});
                  if (!rd.value("ok", false)) return;
                  cognitionManager_.setMissionDeliverable(
                      nlohmann::json{{"text", rd.value("content", std::string())},
                                     {"missionId", missionIdFromScope()}});
                };
                /* Outcome -> existing vocab weights, then the table is
                   saved. Not a reflection paragraph in the prompt. */
                auto learnReply = [&](const std::string &reply, float reward) {
                  if (!emotionSystem_ || reply.empty() || reward == 0.f) return;
                  try {
                    const std::string tag = "mission:" + missionIdFromScope();
                    emotionSystem_->updateVocabFromResponse(
                        tag, phoenix::mission::splitEmotionVocabTokens(goal),
                        phoenix::mission::splitEmotionVocabTokens(reply), reward,
                        static_cast<int>(reply.size()));
                    if (config_.httpLog)
                      std::cout << "[peer-evolve] vocab reward=" << reward
                                << " saved" << std::endl;
                  } catch (...) {
                  }
                };
                if (pendingScore)
                  learnReply(pendingScoreReply, pendingScoreReward);
                const auto trim = [](const std::string &s) -> std::string {
                  return phoenix::mission::trimCopy(s);
                };
                std::vector<nlohmann::json> embedded =
                    phoenix::mission::extractJsonObjects(reply);
                auto execWorkspaceAction =
                    [&](nlohmann::json act) -> nlohmann::json {
                  if (act.contains("action") && act["action"].is_string()) {
                    const std::string actName = act["action"].get<std::string>();
                    const std::string path = act.value("path", std::string());
                    if ((actName == "write" || actName == "append" ||
                         actName == "replace") &&
                        phoenix::mission::isDeliverablePath(path) &&
                        act.contains("content") && act["content"].is_string()) {
                      const std::string safe =
                          phoenix::mission::sanitizeDeliverableActionContent(
                              act["content"].get<std::string>());
                      if (safe.empty()) {
                        return nlohmann::json{
                            {"ok", false},
                            {"error",
                             "deliverable content looked like JSON/internal "
                             "state; write markdown prose only"}};
                      }
                      act["content"] = safe;
                    }
                  }
                  if (peerEvolveGoal) {
                    const std::string path = act.value("path", std::string());
                    if (path.empty() ||
                        phoenix::mission::isDeliverablePath(path))
                      act["allowShrink"] = true;
                  }
                  return phoenix::mission::workspaceExecute(workspaceRoot, scope,
                                                            act);
                };
                if (!embedded.empty() &&
                    !phoenix::mission::replyLeadsWithJson(reply)) {
                  if (config_.httpLog)
                    std::cout << "[mission-deliberate] ignore-buried-json n="
                              << embedded.size() << std::endl;
                  embedded.clear();
                }
                if (!plainTextOnly && !embedded.empty()) {
                  for (const auto &act : embedded) {
                    if (!act.is_object() || !act.contains("plan") ||
                        !act["plan"].is_string())
                      continue;
                    std::string planText = act["plan"].get<std::string>();
                    if (planText.size() > 800)
                      planText = planText.substr(0, 800);
                    const int planMinChars = isChildBox
                        ? configuredMinChars
                        : minDeliverableChars;
                    const size_t bodyChars =
                        std::max(currentFile.size(), deliverable.size());
                    const int planFloor =
                        std::max(600, std::min(planMinChars / 8, 1500));
                    if (static_cast<int>(bodyChars) < planFloor) {
                      return "[plan-refused] deliverable too small to plan; "
                             "keep drafting a minimal version first.";
                    }
                    phoenix::mission::workspaceExecute(
                        workspaceRoot, scope,
                        nlohmann::json{{"action", "write"},
                                       {"path", "plan.md"},
                                       {"content", planText}});
                    return "[plan] saved to plan.md; the next tick follows "
                           "it before acting.";
                  }
                  bool touchedDeliverable = false;
                  bool ranWorkspaceAction = false;
                  for (const auto &act : embedded) {
                    if (!act.is_object() || !act.contains("action") ||
                        !act["action"].is_string())
                      continue;
                    ranWorkspaceAction = true;
                    const std::string path = act.value("path", std::string());
                    const auto opRes = execWorkspaceAction(act);
                    if (!opRes.value("ok", false) &&
                        (phoenix::mission::isDeliverablePath(path) ||
                         path.empty())) {
                      writeToolFeedback(
                          "deliverable " + act.value("action", std::string()) +
                          " rejected: " + opRes.value("error", std::string()));
                    }
                    if (opRes.value("refusedShrink", false) && config_.httpLog)
                      std::cout << "[mission-deliberate] keep-file refused-shrink "
                                << opRes.value("bytes", 0) << std::endl;
                    if (path == deliverablePath || path == "deliverable.md" ||
                        path.empty())
                      touchedDeliverable = true;
                  }
                  if (ranWorkspaceAction) {
                    if (touchedDeliverable) syncDeliverableFromWorkspace();
                    if (embedded.size() > 1) {
                      writeToolFeedback(
                          "multiple JSON objects in one reply - send exactly ONE "
                          "form per tick (plain markdown OR one JSON object).");
                    }
                    return std::string();
                  }
                  /* Braces that are not a workspace action (plan/done/tool, or
                     prose with `{...}`) must fall through. Swallowing them
                     left tick-2 mission replies on the floor. */
                }
                std::string trimmed = trim(reply);
                /* Only treat as JSON when the reply itself is a JSON
                   object. Scanning for the first `{` ate LaTeX
                   $\boxed{8-40}$ and wrote tool-feedback.md, which the
                   next tick copied into deliverable.md. */
                {
                  /* Only a real control object. Fenced ```json / ```python
                     / ```bash drafts used to match and get executed as
                     workspace JSON, which swallowed the tick. */
                  if (phoenix::mission::replyLeadsWithJson(trimmed)) {
                    const size_t brace =
                        phoenix::mission::firstJsonBrace(reply);
                    if (brace < reply.size())
                      trimmed = trim(reply.substr(brace));
                  }
                }
                if (!plainTextOnly && !trimmed.empty() &&
                    trimmed.front() == '{') {
                  try {
                    nlohmann::json act = nlohmann::json::parse(trimmed);
                    if (!act.is_object()) throw 0;
                    if (act.contains("tool") && act["tool"].is_string()) {
                      nlohmann::json args = act.value("arguments", nlohmann::json::object());
                      if (!args.is_object()) args = nlohmann::json::object();
                      const std::string toolName =
                          act["tool"].get<std::string>();
                      if (toolName == "script" || toolName == "python" ||
                          toolName == "python3") {
                        std::string sp = args.value("path", std::string());
                        if (sp.empty())
                          sp = args.value("userPrompt", std::string());
                        const auto opRes = execWorkspaceAction(
                            nlohmann::json{{"action", "run"},
                                           {"path", sp},
                                           {"timeoutMs",
                                            args.value("timeoutMs", 30000)}});
                        if (opRes.value("ok", false))
                          writeToolLastResult(
                              "script " + sp + ": " +
                              opRes.value("stdout", std::string()));
                        else
                          writeToolFeedback(
                              "script " + sp + " failed: " +
                              opRes.value("error", std::string()));
                        return std::string();
                      }
                      std::string up = args.value("userPrompt", std::string());
                      if (up.empty()) up = args.value("input", std::string());
                      if (!up.empty() && up.front() == '{') up.clear();
                      if (up.empty()) up = args.value("query", std::string());
                      nlohmann::json ctx = args;
                      ctx["userPrompt"] = up;
                      /* task-tree lineage: a helper box that spawns records
                         the child under ITSELF (parentId = this box id) */
                      if (isChildBox) {
                        const auto slash = scope.rfind('/');
                        if (slash != std::string::npos && slash + 1 < scope.size())
                          ctx["parentId"] = scope.substr(slash + 1);
                          {
                            std::string mn = scope.substr(0, slash);
                            const auto sp = mn.find("/children/");
                            if (sp != std::string::npos) mn = mn.substr(0, sp);
                            ctx["missionId"] = mn;
                          }
                      }
                      const auto tr = cognitionManager_.executeAgiActionByName(
                          act["tool"].get<std::string>(), ctx);
                      if (tr.value("ok", false)) {
                        const std::string toolReply =
                            phoenix::mission::agiActionReply(tr);
                        if (!toolReply.empty())
                          writeToolLastResult(act["tool"].get<std::string>() +
                                              ": " + toolReply);
                      } else {
                        writeToolFeedback(
                            "tool \"" + act["tool"].get<std::string>() +
                            "\" failed: " + tr.value("error", std::string()) +
                            " - expected form {\"tool\":\"<name>\","
                            "\"arguments\":{\"userPrompt\":\"...\"}}, or just"
                            " state your intent in plain words.");
                      }
                      return std::string();
                    }
                    if (act.contains("action") && act["action"].is_string()) {
                      const auto opRes = execWorkspaceAction(act);
                      if (!opRes.value("ok", false)) {
                        writeToolFeedback(
                            "file " + act.value("action", std::string()) +
                            " on " + act.value("path", std::string()) +
                            " failed: " + opRes.value("error", std::string()));
                      } else if (phoenix::mission::isDeliverablePath(
                                     act.value("path", std::string())) ||
                                 act.value("path", std::string()).empty()) {
                        if (opRes.value("refusedShrink", false) &&
                            config_.httpLog)
                          std::cout
                              << "[mission-deliberate] keep-file refused-shrink "
                              << opRes.value("bytes", 0) << std::endl;
                        syncDeliverableFromWorkspace();
                      }
                      return std::string();
                    }
                    /* v8.x C1 self-planning: {"plan":"<next steps>"} writes
                       plan.md; the next tick reads it back and follows it.
                       Gate: a minimal draft must exist first (no planning
                       in the void).  Humans can read/rewrite plan.md via
                       /api/mission/file (human veto). */
                    if (act.contains("plan") && act["plan"].is_string()) {
                      std::string planText = act["plan"].get<std::string>();
                      if (planText.size() > 500)
                        planText = planText.substr(0, 500);
                      const int planMinChars = isChildBox
                          ? configuredMinChars
                          : minDeliverableChars;
                      const size_t bodyChars =
                          std::max(currentFile.size(), deliverable.size());
                      /* Allow self-outline once a real draft exists. Do not
                         wait for 1/3 of a 30k-word target. */
                      const int planFloor =
                          std::max(600, std::min(planMinChars / 8, 1500));
                      if (static_cast<int>(bodyChars) < planFloor) {
                        return "[plan-refused] deliverable too small to plan; "
                               "keep drafting a minimal version first.";
                      }
                      phoenix::mission::workspaceExecute(
                          workspaceRoot, scope,
                          nlohmann::json{{"action", "write"},
                                         {"path", "plan.md"},
                                         {"content", planText}});
                      return "[plan] saved to plan.md; the next tick follows "
                             "it before acting.";
                    }
                    /* v8.x C2 self-directed loop:
                       {"loop":{"action":"pause|replicate|continue",
                        "reason":"...", "ticks":N}}.  The AI decides what
                       the loop does next; humans always outrank it
                       (interject/amendGoal/E-stop unchanged). */
                    if (act.contains("loop") && act["loop"].is_object()) {
                      const std::string loopAction =
                          act["loop"].value("action", std::string());
                      if (loopAction == "pause") {
                        const int cap = phoenix::cfgOr<int>(
                            "mission.maxSelfPauseTicks", 6);
                        int n = act["loop"].value("ticks", 3);
                        n = std::max(1, std::min(cap, n));
                        return "[loop-pause:" + std::to_string(n) + "] " +
                               act["loop"].value("reason", std::string());
                      }
                      if (loopAction == "replicate") {
                        std::string sub = act["loop"].value("reason", std::string());
                        if (sub.empty())
                          sub = act["loop"].value("subgoal", std::string());
                        nlohmann::json spawnCtx = nlohmann::json{{"subgoal", sub}};
                        if (isChildBox) {
                          const auto slash = scope.rfind('/');
                          if (slash != std::string::npos &&
                              slash + 1 < scope.size())
                            spawnCtx["parentId"] = scope.substr(slash + 1);
                        {
                          std::string mn = scope.substr(0, slash);
                          const auto sp = mn.find("/children/");
                          if (sp != std::string::npos) mn = mn.substr(0, sp);
                          spawnCtx["missionId"] = mn;
                        }
                        }
                        const auto cr =
                            cognitionManager_.spawnMissionChild(spawnCtx);
                        if (cr.value("ok", false))
                          return "[box] summoned " +
                                 cr.value("result", nlohmann::json::object())
                                     .value("id", std::string());
                        return "[box] FAILED " +
                               cr.value("error", std::string());
                      }
                      return "[loop] continue acknowledged (" +
                             act["loop"].value("reason", std::string()) + ")";
                    }
                    /* v8.3 SELF-VERDICT: {"done":true,"summary":"..."}.
                       The agent judges its own task complete; the gate only
                       refuses declarations on empty/micro deliverables.  A
                       parent verdict completes the mission (pressure ends);
                       a box verdict marks the box done (loop skips it, the
                       parent merges its file). */
                    if (act.contains("done") && act["done"].is_boolean() &&
                        act["done"].get<bool>() &&
                        (codeMission || selfModGoal)) {
                      learnReply(reply, -0.2f);
                      return std::string();
                    }
                    if (act.contains("done") && act["done"].is_boolean() &&
                        act["done"].get<bool>()) {
                      std::string summary = act.value("summary", std::string());
                      if (summary.size() > 400)
                        summary.resize(400);
                      /* Prefer the longer of workspace file vs in-memory prior so
                         a stale/truncated file cannot gate-keep a real body. */
                      const size_t bodyChars =
                          std::max(currentFile.size(), deliverable.size());
                      if (static_cast<int>(bodyChars) < minDeliverableChars) {
                        return "[auto-verify] REFUSED: completion declared too "
                               "early - deliverable.md is only " +
                               std::to_string(bodyChars) + " chars (need >= " +
                               std::to_string(minDeliverableChars) +
                               "). Keep writing section by section; never "
                               "declare done before the work actually exists.";
                      }
                      phoenix::mission::workspaceExecute(
                          workspaceRoot, scope,
                          nlohmann::json{{"action", "append"},
                                         {"path", deliverablePath},
                                         {"content", "\n\n[DONE] " + summary}});
                      if (isChildBox) {
                        const auto slash = scope.rfind('/');
                        if (slash != std::string::npos && slash + 1 < scope.size()) {
                          const std::string childId = scope.substr(slash + 1);
                          cognitionManager_.markMissionChildDone(
                              nlohmann::json{{"childId", childId}});
                        }
                        return "[box-done] sub-task declared complete by the "
                               "box itself; parent will read and merge.";
                      }
                      /* v8.x B2: deposit this mission as experience (section
                         titles + excerpt) so similar future tasks start
                         better.  Bounded: 200 entries, summary <= 400 chars. */
                      try {
                        std::string expSum;
                        size_t sp = 0;
                        int secs = 0;
                        while (sp < currentFile.size() && secs < 4) {
                          const auto eol = currentFile.find('\n', sp);
                          const std::string line = currentFile.substr(
                              sp, eol == std::string::npos
                                      ? std::string::npos
                                      : eol - sp);
                          const auto b = line.find_first_not_of(" \t\r");
                          if (b != std::string::npos && line[b] == '#') {
                            expSum += line.substr(b) + "; ";
                            ++secs;
                          }
                          if (eol == std::string::npos) break;
                          sp = eol + 1;
                        }
                        expSum += currentFile.substr(0, 200);
                        if (expSum.size() > 400)
                          expSum = expSum.substr(0, 400);
                        phoenix::mission::experienceAdd(
                            phoenix::resolveConfig<std::string>(
                                "mission.experiencePath",
                                "./runtime_store/mission_experience.json"),
                            goal, expSum);
                        /* v8.x cross-context memory: the finished mission
                           deposits its summary so OTHER contexts (chat,
                           future missions) can recall it. Gated: an
                           off-goal summary is a bled-in foreign theme and
                           must not be stored under this mission's tag. */
                        if (phoenix::memory::ccmMissionTextRelevant(goal,
                                                                    expSum))
                          phoenix::memory::ccmRemember(
                              phoenix::resolveConfig<std::string>(
                                  "crossContextMemory.path",
                                  "./runtime_store/cross_context_memory.json"),
                              "mission:" + scope, expSum);
                        /* v8.x B3 GNN graph evolution: feed the finished
                           mission back through the EXISTING learning
                           pipeline (learnFromDialog extracts memes and
                           builds/strengthens edges) - the long-lived graph
                           genuinely evolves, and future outlines for similar
                           goals get better.  Bounded: summary <= 600 chars,
                           best-effort, never blocks completion. */
                        try {
                          auto lr = activeRuntime();
                          if (lr) {
                            std::string learnText = expSum;
                            if (learnText.size() > 600)
                              learnText = learnText.substr(0, 600);
                            lr->learnFromDialog(
                                nlohmann::json{{"text", goal},
                                               {"sessionId", "mission:" + msnId},
                                               {"missionId", msnId},
                                               {"memoryKind", "mission"}},
                                nlohmann::json{{"reply", learnText}});
                          }
                        } catch (...) {
                          /* graph learning is best-effort */
                        }
                      } catch (...) {
                        /* experience deposit is best-effort */
                      }
                      /* v8.x concurrent: target THIS mission (scope id) */
                      {
                        std::string msnId = scope;
                        const auto sp = msnId.find("/children/");
                        if (sp != std::string::npos) msnId = msnId.substr(0, sp);
                        cognitionManager_.reportMissionOutcome(
                            nlohmann::json{{"goalAchieved", true},
                                           {"missionId", msnId}});
                      }
                      return "[auto-complete] mission self-verdict: task "
                             "complete. " + summary;
                    }
                    writeToolFeedback(
                        "unrecognized JSON shape; reply with plain markdown "
                        "text OR exactly one of {\"action\":...}, "
                        "{\"tool\":...}, {\"plan\":...}, {\"loop\":...}, "
                        "{\"done\":...}.");
                    return std::string();
                  } catch (...) {
                    /* malformed JSON: try the natural-language intent
                       resolver before falling back to plain text */
                    std::string tName;
                    nlohmann::json tCtx;
                    if (resolveToolIntent(reply, tName, tCtx)) {
                      if (barrier_) {
                        const auto insp = barrier_->inspectText(reply);
                        if (insp.blocked) {
                          if (config_.httpLog)
                            std::cout << "[meme-tool] safety block "
                                      << insp.reason << std::endl;
                          writeToolFeedback(
                              "blocked by safety module before any tool ran; "
                              "continue without that material.");
                          return std::string();
                        }
                      }
                      if (isChildBox) {
                        const auto slash = scope.rfind('/');
                        if (slash != std::string::npos &&
                            slash + 1 < scope.size())
                          tCtx["parentId"] = scope.substr(slash + 1);
                        {
                          std::string mn = scope.substr(0, slash);
                          const auto sp = mn.find("/children/");
                          if (sp != std::string::npos) mn = mn.substr(0, sp);
                          tCtx["missionId"] = mn;
                        }
                      }
                      const auto tr =
                          cognitionManager_.executeAgiActionByName(tName, tCtx);
                      if (tr.value("ok", false)) {
                        const std::string toolReply =
                            phoenix::mission::agiActionReply(tr);
                        if (!toolReply.empty())
                          writeToolLastResult(tName + ": " + toolReply);
                      } else {
                        writeToolFeedback(
                            "tool \"" + tName + "\" failed: " +
                            tr.value("error", std::string()) +
                            " - expected form {\"tool\":\"<name>\","
                            "\"arguments\":{\"userPrompt\":\"...\"}}, or just"
                            " state your intent in plain words.");
                      }
                      return std::string();
                    }
                    /* a reply that LOOKS like JSON but parses to nothing and
                       matches no tool intent would pollute the deliverable -
                       ask for a retry instead of appending the broken text */
                    if (!trimmed.empty() && trimmed.front() == '{') {
                      writeToolFeedback(
                          "last reply was not valid JSON and no tool intent "
                          "matched; retry with {\"tool\":...}, "
                          "{\"action\":...}, {\"done\":...}, "
                          "{\"plan\":...}, {\"loop\":...} or plain text.");
                      return std::string();
                    }
                  }
                } else {
                /* non-JSON reply: natural-language tool intent wins, else
                   the text is appended as the next deliverable chunk */
                std::string tName;
                nlohmann::json tCtx;
                if (resolveToolIntent(reply, tName, tCtx)) {
                  if (barrier_) {
                    const auto insp = barrier_->inspectText(reply);
                    if (insp.blocked) {
                      if (config_.httpLog)
                        std::cout << "[meme-tool] safety block "
                                  << insp.reason << std::endl;
                      writeToolFeedback(
                          "blocked by safety module before any tool ran; "
                          "continue without that material.");
                      return std::string();
                    }
                  }
                  if (isChildBox) {
                    const auto slash = scope.rfind('/');
                    if (slash != std::string::npos &&
                        slash + 1 < scope.size())
                      tCtx["parentId"] = scope.substr(slash + 1);
                      {
                        std::string mn = scope.substr(0, slash);
                        const auto sp = mn.find("/children/");
                        if (sp != std::string::npos) mn = mn.substr(0, sp);
                        tCtx["missionId"] = mn;
                      }
                  }
                  const auto tr =
                      cognitionManager_.executeAgiActionByName(tName, tCtx);
                  if (tr.value("ok", false)) {
                    const std::string toolReply =
                        phoenix::mission::agiActionReply(tr);
                    if (!toolReply.empty())
                      writeToolLastResult(tName + ": " + toolReply);
                  } else {
                    writeToolFeedback(
                        "tool \"" + tName + "\" failed: " +
                        tr.value("error", std::string()) +
                        " - expected form {\"tool\":\"<name>\","
                        "\"arguments\":{\"userPrompt\":\"...\"}}, or just"
                        " state your intent in plain words.");
                  }
                  return std::string();
                }
              }

              /* plain text: edit the first open assigned section. */
              if (phoenix::mission::looksLikeStructuredMissionReply(reply)) {
                if (config_.httpLog)
                  std::cout << "[mission-deliberate] drop structured len="
                            << reply.size() << std::endl;
                writeToolFeedback(
                    "structured JSON replies must be a SINGLE valid object per "
                    "tick, or use plain markdown text without a JSON prefix.");
                return std::string();
              }
              const auto fenceTick =
                  phoenix::mission::applyMarkdownFenceProtocol(
                      phoenix::mission::cutAtChatMarkup(reply), currentFile);
              if (!fenceTick.write) {
                if (config_.httpLog)
                  std::cout << "[mission-deliberate] " << fenceTick.reason
                            << std::endl;
                return std::string();
              }
              std::string chunk = fenceTick.chunk;
              const bool closingOddFence = fenceTick.closingOddFence;
              if (closingOddFence && config_.httpLog)
                std::cout << "[mission-deliberate] close odd fence"
                          << std::endl;
              if (barrier_) {
                const auto insp = barrier_->inspectText(chunk);
                if (insp.blocked) {
                  if (config_.httpLog)
                    std::cout << "[mission-deliberate] memebarrier block "
                              << insp.reason << std::endl;
                  writeToolFeedback(
                      "blocked by safety module; continue the document "
                      "without that material.");
                  return std::string();
                }
              }
              if (phoenix::mission::isMissionMetaReply(chunk) || chunk.empty()) {
                if (config_.httpLog)
                  std::cout << "[mission-deliberate] drop meta/empty len="
                            << chunk.size() << " head="
                            << chunk.substr(0, 120) << std::endl;
                if (chunk.empty() || chunk.size() < 8)
                  writeToolFeedback(
                      "empty-reply; continue after the last sentence.");
                return std::string();
              }
              /* Quality/repeat gates are archived. Exact already-written
                 spans are stripped as I/O hygiene; a leftover under the
                 old 80-char reject is still written. Reliability was
                 supposed to come from memory/GNN units — those are 0
                 today, so do not invent another title/content judge. */
              {
                const std::string cut =
                    phoenix::mission::prefixBeforeForbiddenCloser(chunk);
                if (cut != chunk) {
                  if (config_.httpLog)
                    std::cout << "[mission-deliberate] closer-cut kept="
                              << cut.size() << " from=" << chunk.size()
                              << std::endl;
                  chunk = cut;
                }
              }
              if (chunk.empty() ||
                  phoenix::mission::replyLooksLikeShortLineLoop(chunk)) {
                if (config_.httpLog)
                  std::cout << "[mission-deliberate] drop "
                            << (chunk.empty() ? "empty" : "short-line-loop")
                            << " len=" << chunk.size() << " head="
                            << chunk.substr(0, 120) << std::endl;
                return std::string();
              }
              if (!peerEvolveGoal) {
                std::string kept =
                    phoenix::mission::keepUniqueContinuation(chunk,
                                                             currentFile);
                if (kept != chunk) {
                  if (config_.httpLog)
                    std::cout << "[mission-deliberate] unique-cut kept="
                              << kept.size() << " from=" << chunk.size()
                              << std::endl;
                  if (!kept.empty())
                    chunk = std::move(kept);
                }
              }
              if (!peerEvolveGoal && !closingOddFence) {
                const std::string raw = chunk;
                const std::string kept =
                    phoenix::mission::keepCompleteParagraphIo(chunk);
                if (kept != chunk) {
                  if (config_.httpLog)
                    std::cout << "[mission-deliberate] paragraph-io kept="
                              << kept.size() << " from=" << chunk.size()
                              << std::endl;
                  chunk = kept;
                }
                if (chunk.size() < 20 && raw.size() >= 20) {
                  if (config_.httpLog)
                    std::cout << "[mission-deliberate] keep raw-after-io "
                              << raw.size() << std::endl;
                  chunk = phoenix::mission::trimCopy(raw);
                }
              }
              if (chunk.size() < 20 && !closingOddFence) {
                if (config_.httpLog)
                  std::cout << "[mission-deliberate] drop short-after-io"
                            << std::endl;
                return std::string();
              }
              if (!peerEvolveGoal) {
                const std::string stripped =
                    phoenix::mission::stripRewriteInstructionParagraphs(chunk);
                if (stripped != chunk) {
                  if (config_.httpLog)
                    std::cout << "[mission-deliberate] instruction-cut kept="
                              << stripped.size() << " from=" << chunk.size()
                              << std::endl;
                  chunk = stripped;
                }
                if (chunk.empty() ||
                    phoenix::mission::looksLikeRewriteInstructionEcho(chunk) ||
                    (phoenix::mission::looksLikeOnesJunk(chunk) &&
                     phoenix::mission::deliverableInformationChars(chunk) <
                         20)) {
                  if (config_.httpLog)
                    std::cout << "[mission-deliberate] drop instruction-echo "
                              << "len=" << chunk.size() << " head="
                              << chunk.substr(0, 80) << std::endl;
                  writeToolFeedback(
                      phoenix::mission::onesJunkRewriteHint());
                  return std::string();
                }
              }
              if (chunk.size() < 20 && !closingOddFence) {
                if (config_.httpLog)
                  std::cout << "[mission-deliberate] drop short-after-instruction"
                            << std::endl;
                return std::string();
              }
              {
                const auto fresh = phoenix::mission::workspaceExecute(
                    workspaceRoot, scope,
                    nlohmann::json{{"action", "read"},
                                   {"path", deliverablePath}});
                if (fresh.value("ok", false)) {
                  const std::string onDisk =
                      fresh.value("content", std::string());
                  if (onDisk.size() > currentFile.size())
                    currentFile = onDisk;
                }
                const bool chunkHasProse =
                    phoenix::mission::deliverableInformationChars(chunk) >=
                        20 &&
                    !phoenix::mission::looksLikeOnesJunk(chunk) &&
                    !phoenix::mission::looksLikeRewriteInstructionEcho(chunk);
                std::string edited;
                if (onesJunkDraft && chunkHasProse) {
                  edited = phoenix::mission::trimCopy(chunk);
                  if (config_.httpLog)
                    std::cout << "[mission-deliberate] replace-ones-junk "
                              << currentFile.size() << "->" << edited.size()
                              << std::endl;
                } else if (phoenix::mission::isOnesJunkAppend(currentFile,
                                                              chunk) ||
                           phoenix::mission::looksLikeRewriteInstructionEcho(
                               chunk)) {
                  if (config_.httpLog)
                    std::cout << "[mission-deliberate] drop ones-junk-append"
                              << std::endl;
                  writeToolFeedback(
                      phoenix::mission::onesJunkRewriteHint());
                  return std::string();
                } else if (peerEvolveGoal && selfModGoal) {
                  const std::string body = phoenix::mission::trimCopy(chunk);
                  /* Population generation: one round in which each carrier
                     ticks once.  The force is fitted per generation, not
                     per carrier tick (a per-tick series is a first-
                     appearance ramp). */
                  /* Population generation = the global loop round.  A
                     per-scope ++ drifted (a scope re-assigned earlier got a
                     higher counter), splitting one 8-carrier round into
                     several generations. */
                  const int genNow = static_cast<int>(
                      cognitionManager_.loopTickCount());
                  /* A review box whose goal says "do not edit" counts as
                     prose even if its reply happens to contain patch
                     markers.  The word "summarize" alone does not: an
                     ordinary writing mission can use that word and still
                     be a real edit. */
                  const bool reviewRole =
                      goal.find("do not edit") != std::string::npos;
                  const bool sourceEdit = !reviewRole && [&]() {
                    if (body.rfind("diff --git ", 0) == 0 ||
                        body.find("\ndiff --git ") != std::string::npos)
                      return true;
                    if (body.rfind("--- a/", 0) == 0 ||
                        body.find("\n--- a/") != std::string::npos)
                      return true;
                    const auto fileAt = body.find("FILE ");
                    if (fileAt == std::string::npos) return false;
                    const auto oldAt = body.find("\nOLD", fileAt);
                    const auto newAt = body.find("\nNEW", fileAt);
                    return oldAt != std::string::npos &&
                           newAt != std::string::npos && oldAt < newAt;
                  }();
                  if (!sourceEdit) {
                    learnReply(chunk, -0.2f);
                    if (selfModGoal) {
                      for (const auto &rel : explorePaths) {
                        if (!rel.empty() &&
                            chunk.find(rel) != std::string::npos) {
                          namedPathSet(rel);
                          break;
                        }
                      }
                      double projectScore = 0.0;
                      try {
                        const auto scored =
                            cognitionManager_.projectPathScore(chunk);
                        projectScore = scored.value("project", 0.0);
                      } catch (...) {
                      }
                      const char *inst = std::getenv("AI_INSTANCE_ID");
                      const auto copied = phoenix::memetic::judgeCopy(goal, chunk);
                      phoenix::memetic::ReplicaTick mark;
                      /* No AI_INSTANCE_ID: use the mission scope as the
                         carrier id, so two concurrent missions still give
                         an interior frequency instead of an all-unknown
                         binary series.  Explicit alpha/beta still wins. */
                      mark.instance = inst ? inst : ("scope:" + scope);
                      mark.mission = scope;
                      mark.explore = exploreMark;
                      mark.summon = summonMark;
                      mark.replyClass = phoenix::memetic::classifyReplicaReply(
                          goal, chunk, false);
                      mark.replyChars = static_cast<int>(chunk.size());
                      mark.sourceEdit = false;
                      mark.generation = genNow;
                      mark.projectScore = projectScore;
                      mark.copyFidelity = copied.fidelity;
                      mark.indexOverlap = copied.indexOverlap;
                      mark.replicator = copied.replicator;
                      std::string peerDraft;
                      if (mark.instance == "alpha" || mark.instance == "beta") {
                        const char *peerName =
                            mark.instance == "alpha" ? "beta" : "alpha";
                        const char *peerMission =
                            mark.instance == "alpha" ? "self-mod-b"
                                                     : "self-mod-a";
                        std::ifstream peerIn(
                            std::string("runtime_store/peer_") + peerName +
                                "/mission_workspace/" + peerMission +
                                "/deliverable.md",
                            std::ios::binary);
                        if (peerIn)
                          peerDraft.assign(std::istreambuf_iterator<char>(peerIn),
                                           std::istreambuf_iterator<char>());
                      }
                      const auto population = phoenix::memetic::relatePopulation(
                          goal, {chunk, peerDraft});
                      mark.attractionBias = population.attractionBias;
                      mark.peerCopies = population.peerCopies;
                      mark.mechanism = population.mechanism;
                      const auto account = phoenix::memetic::appendReplicaTick(
                          "runtime_store/replica_trace", mark);
                      if (config_.httpLog)
                        std::cout << "[replica-trace] instance=" << mark.instance
                                  << " reply=" << mark.replyClass
                                  << " explore=" << mark.explore
                                  << " summon=" << mark.summon
                                  << " q=" << account.fidelity
                                  << " Lmax=" << account.eigenLmax
                                  << " selection=" << account.priceSelection
                                  << " transmission=" << account.priceTransmission
                                  << " binding=" << account.binding
                                  << " copyF=" << mark.copyFidelity
                                  << " index=" << mark.indexOverlap
                                  << " replicator=" << (mark.replicator ? 1 : 0)
                                  << " attract=" << mark.attractionBias
                                  << " peerCopies=" << mark.peerCopies
                                  << " mechanism=" << mark.mechanism
                                  << " force=" << account.force
                                  << " forceResidual=" << account.forceResidual
                                  << " forceSamples=" << account.forceSamples
                                  << " replicators=" << account.replicators
                                  << std::endl;
                    }
                    if (config_.httpLog)
                      std::cout << "[peer-evolve] not a source edit"
                                << std::endl;
                    return std::string();
                  }
                  {
                    const char *inst = std::getenv("AI_INSTANCE_ID");
                    phoenix::memetic::ReplicaTick mark;
                    mark.instance = inst ? inst : ("scope:" + scope);
                    mark.mission = scope;
                    mark.explore = exploreMark;
                    mark.summon = summonMark;
                    mark.replyClass = "source-edit";
                    mark.replyChars = static_cast<int>(body.size());
                    mark.sourceEdit = true;
                    mark.generation = genNow;
                    const auto account = phoenix::memetic::appendReplicaTick(
                        "runtime_store/replica_trace", mark);
                    if (config_.httpLog)
                      std::cout << "[replica-trace] instance=" << mark.instance
                                << " reply=source-edit q=" << account.fidelity
                                << " Lmax=" << account.eigenLmax
                                << " binding=" << account.binding << std::endl;
                  }
                  const bool fresh = body.find("diff ") != std::string::npos ||
                                     body.find("--- ") != std::string::npos ||
                                     body.find("FILE ") != std::string::npos;
                  if (!peerEditBase.empty() && !fresh) {
                    edited = peerEditBase;
                    if (!edited.empty() && !body.empty() &&
                        !std::isspace(
                            static_cast<unsigned char>(edited.back())) &&
                        !std::isspace(static_cast<unsigned char>(body.front())))
                      edited.push_back('\n');
                    edited += body;
                  } else {
                    edited = body;
                  }
                  if (edited.size() > 16000) {
                    const auto cut = edited.rfind("\ndiff ");
                    if (cut != std::string::npos) edited = edited.substr(cut + 1);
                  }
                } else if (peerEvolveGoal && codeMission) {
                  const std::string body = phoenix::mission::trimCopy(chunk);
                  const bool shortBrace =
                      body.size() < 80 &&
                      (body.find('{') != std::string::npos ||
                       body.find('}') != std::string::npos);
                  const bool codeish =
                      body.find("384") != std::string::npos ||
                      body.find("int64") != std::string::npos ||
                      body.find("for") != std::string::npos ||
                      body.find(';') != std::string::npos ||
                      body.find("return") != std::string::npos ||
                      body.find("evolveCompute") != std::string::npos ||
                      shortBrace;
                  if (!codeish) {
                    learnReply(chunk, -0.2f);
                    if (config_.httpLog)
                      std::cout << "[peer-evolve] kept source; non-code reply "
                                   "not appended"
                                << std::endl;
                    return std::string();
                  }
                  /* A later tick that still says evolveCompute is a new
                     draft. Anything else continues the function already
                     on disk, including a for-loop that was cut mid-line. */
                  const bool fresh =
                      body.find("evolveCompute") != std::string::npos;
                  if (!peerEditBase.empty() && !fresh) {
                    edited = peerEditBase;
                    if (!edited.empty() && !body.empty() &&
                        !std::isspace(static_cast<unsigned char>(edited.back())) &&
                        !std::isspace(static_cast<unsigned char>(body.front())))
                      edited.push_back('\n');
                    edited += body;
                  } else {
                    edited = body;
                  }
                } else if (peerEvolveGoal &&
                           phoenix::peer::evolveHasControl(chunk)) {
                  edited = phoenix::mission::trimCopy(chunk);
                } else if (peerEvolveGoal &&
                           phoenix::peer::evolveHasControl(peerEditBase)) {
                  const std::string cont = phoenix::mission::trimCopy(chunk);
                  const bool codeCont =
                      cont.find(';') != std::string::npos ||
                      cont.find('}') != std::string::npos ||
                      cont.find("return") != std::string::npos ||
                      cont.find(">>") != std::string::npos ||
                      cont.find("<<") != std::string::npos ||
                      cont.find("^=") != std::string::npos ||
                      cont.find("state") != std::string::npos;
                  if (!codeCont) {
                    learnReply(chunk, -0.2f);
                    if (config_.httpLog)
                      std::cout << "[peer-evolve] kept source; non-code reply "
                                   "not appended"
                                << std::endl;
                    return std::string();
                  }
                  edited = phoenix::mission::joinDeliverableText(peerEditBase,
                                                                 cont);
                } else if (peerEvolveGoal) {
                  learnReply(chunk, -0.2f);
                  if (config_.httpLog)
                    std::cout << "[peer-evolve] kept source; non-code reply "
                                 "not appended"
                              << std::endl;
                  return std::string();
                } else if (!currentFile.empty() &&
                           (phoenix::mission::continuationRestatesDraft(
                                chunk, currentFile) ||
                            phoenix::mission::deliverableInformationChars(
                                phoenix::mission::stripRequirementBoilerplate(
                                    chunk)) < 40)) {
                  if (config_.httpLog)
                    std::cout << "[mission-deliberate] drop restate len="
                              << chunk.size() << std::endl;
                  sectionDropStreak.inc(scope);
                  writeToolFeedback(
                      sectionFocus.empty()
                          ? "That paragraph restates text already in the draft. Continue from the last new sentence."
                          : "That paragraph is a requirements list. Continue this section in prose: " +
                                sectionFocus);
                  return std::string();
                } else {
                  std::string toWrite =
                      phoenix::mission::stripRequirementBoilerplate(chunk);
                  if (toWrite.empty()) toWrite = chunk;
                  if (!sectionFocus.empty())
                    edited = phoenix::mission::insertChapterProse(
                        currentFile, sectionFocus, toWrite);
                  else
                    edited = currentFile.empty()
                        ? toWrite
                        : phoenix::mission::joinDeliverableText(currentFile,
                                                                toWrite);
                  if (phoenix::mission::isZeroInformationGrowth(currentFile,
                                                                edited) ||
                      phoenix::mission::looksLikeRewriteInstructionEcho(
                          edited)) {
                    if (config_.httpLog)
                      std::cout
                          << "[mission-deliberate] drop zero-info-growth"
                          << std::endl;
                    writeToolFeedback(
                        phoenix::mission::onesJunkRewriteHint());
                    return std::string();
                  }
                }
                if (edited == currentFile ||
                    (peerEvolveGoal && edited == peerEditBase)) {
                  if (peerEvolveGoal) learnReply(chunk, -0.2f);
                  if (config_.httpLog)
                    std::cout << "[mission-deliberate] drop no-file-edit"
                              << std::endl;
                  return std::string();
                }
                if (config_.httpLog)
                  std::cout << "[mission-deliberate] file-edit "
                            << currentFile.size() << "->" << edited.size()
                            << std::endl;
                sectionDropStreak.set(scope, 0);
                nlohmann::json writeAct = {
                    {"action", "write"},
                    {"path", deliverablePath},
                    {"content", edited}};
                if (peerEvolveGoal) writeAct["allowShrink"] = true;
                phoenix::mission::workspaceExecute(workspaceRoot, scope,
                                                   writeAct);
                /* Sync memory before optional CCM/emotion so a hang
                   there cannot hide the write from the next tick. */
                syncDeliverableFromWorkspace();
                phoenix::peer::PeerPushResult peerPush;
                /* An unfinished draft has no return yet. Compiling it
                   only teaches the weight table that a loop is wrong. */
                bool algoReady = true;
                if (selfModGoal) {
                  algoReady = false;
                  std::ofstream own("runtime_store/peers/" +
                                    std::string(std::getenv("AI_INSTANCE_ID")
                                                    ? std::getenv("AI_INSTANCE_ID")
                                                    : "alpha") +
                                    "/self_edit.diff",
                                    std::ios::binary);
                  own << edited;
                } else if (codeMission) {
                  const std::string fn =
                      phoenix::peer::extractEvolve(goal, edited);
                  algoReady = fn.find("return") != std::string::npos;
                }
                if (algoReady) try {
                  peerPush = phoenix::peer::pushEvolveToPeer(goal, edited);
                } catch (const std::exception &ex) {
                  std::cout << "[peer-evolve] " << ex.what() << std::endl;
                } catch (...) {
                  std::cout << "[peer-evolve] failed" << std::endl;
                }
                if (codeMission &&
                    peerPush.kind != phoenix::peer::PeerPushResult::Kind::Idle) {
                  long long us = -1;
                  if (peerPush.detail.rfind("OK ", 0) == 0) {
                    try {
                      us = std::stoll(peerPush.detail.substr(3));
                    } catch (...) {
                      us = -1;
                    }
                  }
                  long long best = -1;
                  const auto bestRd = phoenix::mission::workspaceExecute(
                      workspaceRoot, scope,
                      nlohmann::json{{"action", "read"}, {"path", "best_us.txt"}});
                  if (bestRd.value("ok", false)) {
                    try {
                      best = std::stoll(bestRd.value("content", std::string()));
                    } catch (...) {
                      best = -1;
                    }
                  }
                  std::string note = peerPush.detail;
                  if (note.size() > 180) note.resize(180);
                  if (best >= 0) note += "\nbest " + std::to_string(best);
                  phoenix::mission::workspaceExecute(
                      workspaceRoot, scope,
                      nlohmann::json{{"action", "write"},
                                     {"path", "measure.md"},
                                     {"content", note}});
                  if (us >= 0 && (best < 0 || us < best)) {
                    phoenix::mission::workspaceExecute(
                        workspaceRoot, scope,
                        nlohmann::json{{"action", "write"},
                                       {"path", "best_us.txt"},
                                       {"content", std::to_string(us)}});
                    learnReply(edited, 0.6f);
                  } else if (us >= 0) {
                    learnReply(edited, 0.15f);
                  } else {
                    learnReply(edited, -0.35f);
                  }
                } else if (peerPush.kind ==
                           phoenix::peer::PeerPushResult::Kind::Rejected)
                  learnReply(edited, -0.35f);
                else if (peerPush.kind ==
                         phoenix::peer::PeerPushResult::Kind::Loaded)
                  learnReply(edited, 0.6f);
                if (!stillDrafting) try {
                  phoenix::mission::depositMissionProgress(
                      phoenix::instance::ccmStorePath(),
                      phoenix::resolveConfig<std::string>(
                          "mission.experiencePath",
                          "./runtime_store/mission_experience.json"),
                      "mission:" + scope, goal, edited, chunk);
                } catch (...) {
                }
                if (emotionSystem_ && !stillDrafting && !peerEvolveGoal) {
                  try {
                    std::string msnId = scope;
                    const auto spE = msnId.find("/children/");
                    if (spE != std::string::npos)
                      msnId = msnId.substr(0, spE);
                    const std::string tag = "mission:" + msnId;
                    const auto promptToks =
                        phoenix::mission::splitEmotionVocabTokens(
                            recallDraft);
                    const auto replyToks =
                        phoenix::mission::splitEmotionVocabTokens(chunk);
                    float reward = 0.15f;
                    if (!phoenix::mission::inspectDeliverableHealth(edited,
                                                                    goal)
                             .empty())
                      reward = 0.05f;
                    if (phoenix::mission::contradictedOperationalFact(edited,
                                                                      goal))
                      reward = 0.0f;
                    emotionSystem_->updateVocabFromResponse(
                        tag, promptToks, replyToks, reward,
                        static_cast<int>(edited.size()));
                  } catch (...) {
                  }
                }
              }
              syncDeliverableFromWorkspace();
              {
                std::lock_guard<std::mutex> lk(gDelibSkipMu);
                const auto rd = phoenix::mission::workspaceExecute(
                    workspaceRoot, scope,
                    nlohmann::json{{"action", "read"}, {"path", deliverablePath}});
                const size_t sz =
                    rd.value("ok", false)
                        ? rd.value("content", std::string()).size()
                        : currentFile.size();
                gDelibLastDelivAtLlm[scope] = sz;
                gDelibCacheSkipsNoGrowth[scope] = 0;
              }
              if (!stillDrafting)
                phoenix::mission::workspaceCachePut(scope, dynamicPrompt, chunk);
              if (config_.httpLog)
                std::cout << "[mission-deliberate] wrote-return deliv="
                          << currentFile.size() << std::endl;
              return std::string(); /* file + memory synced; loop must not re-append */
              }
            } catch (const std::exception &ex) {
              /* A throwing tick used to return "" here with zero trace -
                 indistinguishable from a dead loop in the gateway log. */
              std::cout << "[mission-deliberate] EXCEPTION: "
                        << ex.what() << std::endl;
            } catch (...) {
              std::cout << "[mission-deliberate] EXCEPTION (unknown)"
                        << std::endl;
            }
            return std::string();
          });
    }
    /* v7.0 MCP compatibility: load mcp.* config (disabled by default).  When
       enabled, each configured MCP server (JSON-RPC over stdio) is launched
       and its tools become planner actions named "mcp.<server>.<tool>". */
    {
      json mcpCfg;
      mcpCfg["enabled"] = phoenix::cfgOr<bool>("mcp.enabled", false);
      mcpCfg["servers"] = phoenix::cfgOr<nlohmann::json>("mcp.servers", nlohmann::json::array());
      cognitionManager_.configureMcp(mcpCfg);
    }
    /* v7.0 long-term autonomous loop: the heartbeat that runs the
       plan/act/observe/learn cycle without external messages and persists
       the evolved state (long-term evolution).  Disabled by default. */
    {
      json loopCfg;
      loopCfg["enabled"] = phoenix::cfgOr<bool>("autonomyLoop.enabled", false);
      loopCfg["intervalSec"] = phoenix::cfgOr<int>("autonomyLoop.intervalSec", 10);
      loopCfg["maxStepsPerTick"] = phoenix::cfgOr<int>("autonomyLoop.maxStepsPerTick", 8);
      loopCfg["maxChildrenPerTick"] = phoenix::cfgOr<int>("autonomyLoop.maxChildrenPerTick", 0);
      /* v8.x concurrent missions: 0 = advance ALL running missions each tick */
      loopCfg["maxMissionsPerTick"] = phoenix::cfgOr<int>("autonomyLoop.maxMissionsPerTick", 0);
      loopCfg["persistEveryTicks"] = phoenix::cfgOr<int>("autonomyLoop.persistEveryTicks", 5);
      loopCfg["persistPath"] = phoenix::resolveConfig<std::string>(
          "autonomyLoop.persistPath",
          std::string("runtime_store/autonomy_state.json"),
          "AI_AUTONOMY_PERSIST");
      loopCfg["deliberateMaxTokens"] =
          phoenix::cfgOr<int>("mission.deliberateMaxTokens", 192);
      loopCfg["childDeliberateMaxTokens"] =
          phoenix::cfgOr<int>("mission.childDeliberateMaxTokens", 96);
      loopCfg["stallWarnSec"] =
          phoenix::cfgOr<int>("autonomyLoop.stallWarnSec", 1800);
      cognitionManager_.configureAutonomyLoop(loopCfg);
      if (phoenix::cfgOr<bool>("autonomyLoop.enabled", false)) {
        cognitionManager_.startAutonomyLoop();
      }
    }
    {
      static std::string cancelUrl;
      cancelUrl = config_.llamaCppBaseUrl.empty()
                      ? std::string("http://127.0.0.1:8082")
                      : config_.llamaCppBaseUrl;
      phoenix::inference::setAbortNotify([]() {
        phoenix::v7::cancelPhxGenerate(cancelUrl);
      });
    }
    /* v7.0 system-level E-stop: when pressed, the latch kills every
       registered instance, persists the evolved state, and quits this
       process (the system shuts ITSELF down - it cannot keep running
       uncontrolled). */
    phoenix::safety::EmergencyStop::instance().setShutdownHandler([this] {
      try {
        auto state = cognitionManager_.exportState();
        std::filesystem::create_directories("runtime_store");
        std::ofstream f("runtime_store/estop_state.json");
        f << state.dump(2);
      } catch (...) {
        /* persistence is best-effort during an emergency */
      }
      drogon::app().quit();
    });
    /* v7.0 cli-json bridge: whitelist of external CLI tools ("any software
       becomes a plugin").  Empty by default (nothing is runnable). */
    {
      nlohmann::json cliTools = phoenix::cfgOr<nlohmann::json>("cliTools", nlohmann::json::array());
      if (!cliTools.is_array()) cliTools = nlohmann::json::array();
      std::string cliErr;
      addon::builtins::setCliJsonRegistry(cliTools, &cliErr);
    }
    /* AGI action space: register the real addon capabilities as planner
       actions and set the executor that dispatches the chosen tool.  The
       planner may now choose math/search/research/web in addition to the
       instinct verbs; the executor performs the actual tool call. */
    {
      json agiActions = phoenix::cfgOr<nlohmann::json>("agi.actions", nlohmann::json::array());
      if (!agiActions.is_array()) agiActions = nlohmann::json::array();
      // The manager already registers its default capabilities in its
      // constructor (math/search/computer/goal_advance); config entries ADD
      // or REPLACE them (registerAgiAction is idempotent by name).
      for (const auto &item : agiActions) {
        cognitionManager_.registerAgiAction(item);
      }
      cognitionManager_.setAgiActionExecutor(
          [this](const phoenix::agi::AgiActionSpec &spec, const json &context) -> json {
            AddonToolCall call;
            call.requested = true;
            call.type = spec.addonType.empty() ? spec.name : spec.addonType;
            call.input = context.value("userPrompt", std::string());
            addon::AddonResult result;
            const bool handled = invokeAddonTool(call, context, result);
            if (!handled) {
              /* v8.6: non-addon categories (replicate / summon / goal /
                 script sandbox / mcp) are dispatched by the manager itself.
                 Without this fallback a planner-picked "replicate" returned
                 "tool not handled" and never spawned the box.  iterate()
                 releases mu_ before invoking the executor, so this re-lock
                 inside executeAgiActionByName cannot deadlock. */
              return cognitionManager_.executeAgiActionByName(spec.name,
                                                              context);
            }
            return json{{"ok", handled},
                        {"handled", handled},
                        {"action", spec.name},
                        {"reply", result.reply},
                        {"meta", result.meta}};
          });
    }
    /* v8.6 model summoning: each registered summon.models entry becomes a
       planner action "summon.<name>" (category "summon").  The EFE planner
       itself decides when a knowledge gap calls for a bigger/expert model -
       the base model is NEVER taught a call syntax (no prompt text, no
       fine-tuning): the framework adapts to the model, not vice versa.
       Endpoint adapters translate the prompt into each backend's API.
       Disabled by default (summon.enabled=false). */
    {
      const bool summonEnabled = phoenix::resolveConfig<bool>(
          "summon.enabled", false, "AI_SUMMON_ENABLED");
      json summonModels = phoenix::cfgOr<nlohmann::json>(
          "summon.models", nlohmann::json::array());
      auto &summonReg = phoenix::summon::SummonRegistry::instance();
      const auto summonLoad = summonReg.loadFromConfig(summonEnabled,
                                                       summonModels);
      /* Optional single-MoE spec. Absent object is a no-op. upsertMoe adds
         or replaces that one row and does not start a process. */
      {
        json summonMoe = phoenix::cfgOr<nlohmann::json>(
            "summon.moe", nlohmann::json::object());
        if (summonMoe.is_object() && !summonMoe.empty()) {
          const auto moeUp = summonReg.upsertMoe(summonMoe);
          /* autoStart spawns Granite in-process-adjacent: one llama-server
             child, CPU only, expert pin via the child environment. The
             parent process environment is not modified, so 8082 is untouched.
             No /health probe. */
          const bool autoStart = summonMoe.value("autoStart", false);
          std::string modelPath = summonMoe.value("modelPath", std::string());
          if (modelPath.empty() && summonMoe.contains("gguf") &&
              summonMoe["gguf"].is_string())
            modelPath = summonMoe["gguf"].get<std::string>();
          std::string lowered = modelPath;
          for (char &c : lowered) {
            if (c >= 'A' && c <= 'Z')
              c = static_cast<char>(c - 'A' + 'a');
          }
          const bool blocked = lowered.find("qwen3-30b") != std::string::npos ||
                               lowered.find("deepseek") != std::string::npos;
          std::error_code moeEc;
          const bool ggufPresent =
              !modelPath.empty() &&
              std::filesystem::is_regular_file(modelPath, moeEc);
          const std::string envName =
              summonMoe.value("moeEnv", std::string("LLAMA_FORCE_MOE_EXPERT"));
          const std::string envValue =
              summonMoe.value("moeEnvValue", std::string("0"));
          const int moeK = summonMoe.value("k", summonMoe.value("activeExperts", 1));
          const int pinned = summonMoe.value(
              "pinnedExpert", summonMoe.value("expertId", -1));
          const int moePort = summonMoe.value("port", 0);
          int launchPid = 0;
          int launch = 0;
          std::string launchReason = "autoStart false";
          if (blocked)
            launchReason = "refused model";
          else if (autoStart && !ggufPresent)
            launchReason = "gguf missing";
          else if (autoStart && moePort <= 0)
            launchReason = "port missing";
          else if (autoStart) {
#ifdef _WIN32
            const auto serverExe = std::filesystem::absolute(
                "outsides/llamacpp/build-gcc/bin/llama-server.exe", moeEc);
            const auto modelAbs = std::filesystem::absolute(modelPath, moeEc);
            if (moeEc || !std::filesystem::is_regular_file(serverExe)) {
              launchReason = "llama-server.exe missing";
            } else {
              std::filesystem::create_directories("runtime_store", moeEc);
              const std::string errPath =
                  "runtime_store/moe_8100_gateway.err.log";
              SECURITY_ATTRIBUTES sa{};
              sa.nLength = sizeof(sa);
              sa.bInheritHandle = TRUE;
              HANDLE hNul = CreateFileA(
                  "NUL", GENERIC_READ | GENERIC_WRITE,
                  FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING,
                  FILE_ATTRIBUTE_NORMAL, nullptr);
              HANDLE hErr = CreateFileA(
                  errPath.c_str(), GENERIC_WRITE, FILE_SHARE_READ, &sa,
                  CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
              auto quoteArg = [](const std::string &s) {
                std::string o = "\"";
                for (char c : s) {
                  if (c == '"')
                    o += "\\\"";
                  else
                    o.push_back(c);
                }
                o.push_back('"');
                return o;
              };
              const std::string cmd =
                  quoteArg(serverExe.string()) + " -m " +
                  quoteArg(modelAbs.string()) +
                  " --host 127.0.0.1 --port " + std::to_string(moePort) +
                  " --ctx-size 4096 --parallel 1 --threads 6 --n-gpu-layers 0"
                  /* v10.1: the Granite MoE slot is a CPU helper; -dev none
                     stops llama.cpp from creating a second CUDA context that
                     otherwise shows up as an extra compute process on the GPU
                     and steals VRAM from the 8B (the only real GPU consumer). */
                  " -dev none"
                  " --batch-size 4096 --ubatch-size 512";
              std::string envBlock;
              if (LPCH env = GetEnvironmentStringsA()) {
                for (LPCH p = env; *p;) {
                  const std::string entry(p);
                  p += entry.size() + 1;
                  if (entry.size() >= 6 &&
                      entry.compare(0, 6, "LLAMA_") == 0)
                    continue;
                  envBlock.append(entry);
                  envBlock.push_back('\0');
                }
                FreeEnvironmentStringsA(env);
              }
              envBlock.append(envName);
              envBlock.push_back('=');
              envBlock.append(envValue);
              envBlock.push_back('\0');
              envBlock.push_back('\0');
              STARTUPINFOA si{};
              si.cb = sizeof(si);
              si.dwFlags = STARTF_USESTDHANDLES;
              si.hStdInput = hNul;
              si.hStdOutput = hErr;
              si.hStdError = hErr;
              PROCESS_INFORMATION pi{};
              std::vector<char> cmdBuf(cmd.begin(), cmd.end());
              cmdBuf.push_back('\0');
              const DWORD flags = CREATE_NEW_PROCESS_GROUP |
                                  DETACHED_PROCESS | CREATE_NO_WINDOW;
              if (hNul == INVALID_HANDLE_VALUE ||
                  hErr == INVALID_HANDLE_VALUE) {
                launchReason = "log handle failed";
              } else if (!CreateProcessA(
                             nullptr, cmdBuf.data(), nullptr, nullptr, TRUE,
                             flags, envBlock.empty() ? nullptr : &envBlock[0],
                             nullptr, &si, &pi)) {
                launchReason = "CreateProcess failed " +
                               std::to_string(GetLastError());
              } else {
                launchPid = static_cast<int>(pi.dwProcessId);
                launch = 1;
                launchReason = "gateway spawned granite";
                CloseHandle(pi.hThread);
                CloseHandle(pi.hProcess);
                {
                  std::ofstream pidf("runtime_store/moe_8100_gateway.pid",
                                     std::ios::trunc);
                  if (pidf)
                    pidf << launchPid;
                }
                bool sawPin = false;
                bool sawLoaded = false;
                for (int wait = 0; wait < 120 && !(sawPin && sawLoaded);
                     ++wait) {
                  Sleep(500);
                  HANDLE hRead = CreateFileA(
                      errPath.c_str(), GENERIC_READ,
                      FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
                  if (hRead == INVALID_HANDLE_VALUE)
                    continue;
                  std::string log;
                  char buf[4096];
                  DWORD n = 0;
                  while (ReadFile(hRead, buf, sizeof(buf), &n, nullptr) &&
                         n > 0)
                    log.append(buf, buf + n);
                  CloseHandle(hRead);
                  if (log.find("n_expert_used=1") != std::string::npos)
                    sawPin = true;
                  if (log.find("model loaded") != std::string::npos)
                    sawLoaded = true;
                  if (log.find("failed to bind") != std::string::npos ||
                      log.find("address already in use") != std::string::npos) {
                    launchReason = "port busy";
                    break;
                  }
                }
                if (sawPin && sawLoaded)
                  launchReason = "gateway spawned granite, model loaded";
                else if (launchReason == "gateway spawned granite")
                  launchReason = "spawned but model not loaded yet";
              }
              if (hNul != INVALID_HANDLE_VALUE)
                CloseHandle(hNul);
              if (hErr != INVALID_HANDLE_VALUE)
                CloseHandle(hErr);
            }
#else
            launchReason = "autoStart spawn is implemented on Windows";
#endif
          }
          std::cerr << "[summon] moe name="
                    << summonMoe.value("name", std::string("moe"))
                    << " port=" << moePort
                    << " k=" << moeK
                    << " pinnedExpert=" << pinned
                    << " env=" << envName << "=" << envValue
                    << " ggufPresent=" << (ggufPresent ? 1 : 0)
                    << " autoStart=" << (autoStart ? 1 : 0)
                    << " launch=" << launch
                    << " pid=" << launchPid
                    << " reason=" << launchReason
                    << " upsert=" << moeUp.dump()
                    << std::endl;
        }
      }
      if (logger_.enabled())
        logger_.log(LoggerCXX::Type::MONITORING,
                    std::string("summon registry: ") + summonLoad.dump());
      if (summonEnabled) {
        const auto endpoints = summonReg.list();
        /* One-hot action embeddings past the five default actions; one
           shared dim so only the first registration grows actionDim. */
        const size_t summonDim = 5 + endpoints.size();
        size_t summonIdx = 0;
        for (const auto &ep : endpoints) {
          json summonSpec;
          summonSpec["name"] = "summon." + ep.name;
          summonSpec["category"] = "summon";
          summonSpec["addonType"] = "";
          summonSpec["description"] =
              ep.description.empty()
                  ? "Summon a helper box running external model " + ep.name
                  : ep.description;
          std::vector<float> emb(summonDim, 0.0f);
          emb[5 + summonIdx] = 1.0f;
          summonSpec["embedding"] = emb;
          cognitionManager_.registerAgiAction(summonSpec);
          ++summonIdx;
        }
      }
    }
    rlDisabled_ = config_.disableLearning || config_.disableRL;
    advDisabled_ = config_.disableLearning || config_.disableADV;
    gnnGaDisabled_ = config_.disableLearning;
    dialogLearningEnabled_ = !(config_.disableLearning == true);
    dialogTransformerLearning_ =
        phoenix::resolveConfig<bool>("dialog.transformerLearning", false, "AI_DIALOG_TRANSFORMER_LEARNING");
    dialogLearnGraphContext_ =
        phoenix::resolveConfig<bool>("dialog.learnGraphContext", false, "AI_DIALOG_LEARN_GRAPH_CONTEXT");
    retrainTrace_ = phoenix::resolveConfig<bool>("learning.retrainTrace", false, "AI_RETRAIN_TRACE");
    loadDialogConfig();
    logger_.initialize(config_.logMode, config_.exportDir / "logs");
    if (logger_.enabled())
      logger_.log(LoggerCXX::Type::LOG,
                  std::string("gateway init host=") + config_.gatewayHost +
                      " port=" + std::to_string(config_.portGateway));
    matrixCache_ =
        std::make_shared<HotMatrixCache>(config_.redisUrl, config_.redisCacheDb,
                                         config_.redisCachePrefix + ":matrix");
    // transformer::setAttentionCacheConfig(config_.redisUrl, config_.redisCacheDb,
    //                                      config_.redisCachePrefix + ":attn",
    //                                      900, 3, 200000);
    if (!config_.disableBarrier) {
      try {
        auto rt = std::dynamic_pointer_cast<RuntimeState>(
            pool_->getActive()->runtime());
        if (rt) {
          barrier_ = std::make_shared<MemeBarrier>(rt);
          barrier_->start();
          if (logger_.enabled())
            logger_.log(LoggerCXX::Type::MONITORING, "memebarrier started");
        }
      } catch (...) {
        barrier_.reset();
        if (logger_.enabled())
          logger_.log(LoggerCXX::Type::WARNING, "memebarrier start failed");
      }
    } else if (logger_.enabled()) {
      logger_.log(LoggerCXX::Type::MONITORING,
                  "memebarrier disabled by config");
    }

    try {
      phoenix::v7::AsyncTaskSystem::global().start();
    } catch (...) {}

    try {
      auto aheadCfgJson = phoenix::cfgOr<nlohmann::json>("v7.ahead", nlohmann::json::object());
      auto aheadCfg = phoenix::v7::AheadConfig::fromJson(aheadCfgJson);
      aheadModule_ = std::make_unique<phoenix::v7::AheadModule>(aheadCfg);
      if (!aheadModule_->initialize())
        aheadModule_.reset();
      else if (logger_.enabled())
        logger_.log(LoggerCXX::Type::MONITORING, "ahead module initialized");
    } catch (...) {
      aheadModule_.reset();
    }

    setupRoutes();
  }

  void listen() override {
    unsigned hw = std::thread::hardware_concurrency();
    unsigned defaultThreads = hw > 0 ? std::max(4u, std::min(16u, hw)) : 4u;
    int threadNum = phoenix::resolveConfig<int>(
        "frontend_server.httpThreads", static_cast<int>(defaultThreads),
        "AI_HTTP_THREADS");
    if (threadNum < 2)
      threadNum = 2;
    drogon::app().setThreadNum(threadNum);
    /* Default idle timeout is 60s. A 64–80 token generate on host
       (~50–70s) or RDK (~150–210s) then looks like an empty client
       close while the server still logs 200. Keep the HTTP socket
       until llama finishes. */
    drogon::app().setIdleConnectionTimeout(3600);
    drogon::app().addListener(config_.gatewayHost, config_.portGateway);
    drogon::app().disableSigtermHandling();
    drogon::app().run();
  }

  void warmupLearning() override {
    if (!config_.disableLearning && !config_.disableRL) {
      try {
        ensureRL()->learn(1);
      } catch (...) {
        rlDisabled_ = true;
      }
    }
    if (!config_.disableLearning && !config_.disableADV) {
      try {
        auto docs = pool_->getActive()->runtime()->collectRobotsDocuments(
            json{{"limit", 3}, {"shuffle", true}});
        json samples = json::array();
        for (auto &d : docs) {
          if (!d.text.empty())
            samples.push_back(d.text);
          if (samples.size() >= 3)
            break;
        }
        if (!samples.empty())
          ensureADV()->attackAndDefend(samples);
      } catch (...) {
      }
    }

    /* Corpus pretrain is not online RL. disableLearning still blocks
       RL/ADV/GNN-GA above; wikitext -> GNN/memory/emotion always runs. */
    try {
      const int lim = phoenix::cfgOr<int>(
          "mission.gnnPretrainDocs",
          phoenix::cfgOr<int>("main.inference.gnnBootstrapDocs", 256));
      seedGnnGraphFromRobotsCorpus(lim, "gnn-warmup-pretrain");
        } catch (...) {
    }
  }

  void bootstrapTransformerFromCorpus(int maxDocs = 48) override {
    /* Self-built transformer bootstrap is deprecated (infer = llama-server).
       This is the startup GNN/memory/emotion corpus pass (workflow §8/§12).
       disableLearning must not skip it. */
    const int lim = maxDocs > 0
                        ? maxDocs
                        : phoenix::cfgOr<int>("mission.gnnPretrainDocs", 256);
    seedGnnGraphFromRobotsCorpus(lim, "gnn-bootstrap-pretrain");
  }

private:
  std::shared_ptr<world_model::WorldModelStore> worldModel_;

  struct RouteMetric {
    uint64_t calls{0};
    uint64_t errors{0};
    uint64_t totalMs{0};
    uint64_t maxMs{0};
    uint64_t lastMs{0};
  };

  struct TrainMonitorState {
    uint64_t id{0};
    bool running{false};
    bool ok{false};
    std::string route;
    std::string stage;
    std::string error;
    int64_t startedAtMs{0};
    int64_t updatedAtMs{0};
    int64_t finishedAtMs{0};
    uint64_t rssBefore{0};
    uint64_t rssPeak{0};
    uint64_t rssAfter{0};
    uint64_t docs{0};
    uint64_t processedDocs{0};
    uint64_t samples{0};
    uint64_t chars{0};
    json requestSummary = json::object();
    json resultSummary = json::object();
  };

  /* Repository file list for the SELF-MOD explore step.  Built once and then
     read-only from every deliberator thread: the previous version mutated a
     lambda-level static vector lazily, so two parallel missions could
     push_back while a third iterated it (use-after-free -> 0xC0000374). */
  static const std::vector<std::string> &gatewayExplorePaths() {
    static const std::vector<std::string> paths = [] {
      std::vector<std::string> out;
      const char *fixed[] = {"emotion_system.cpp",
                             "instinct.cpp",
                             "autonomy_stack.cpp",
                             "self_mind.cpp",
                             "active_inference.cpp",
                             "mission_lifecycle.cpp",
                             "memetic_existence.hpp"};
      for (const char *rel : fixed) {
        std::error_code ec;
        if (fs::is_regular_file(rel, ec))
          out.push_back(rel);
      }
      std::error_code ec;
      if (fs::is_directory("main_hub_parts", ec)) {
        int n = 0;
        for (const auto &ent : fs::directory_iterator("main_hub_parts", ec)) {
          if (ec || !ent.is_regular_file())
            continue;
          const std::string name = ent.path().filename().string();
          if (name.size() < 5 || name.find(".inc") == std::string::npos)
            continue;
          out.push_back("main_hub_parts/" + name);
          if (++n >= 12)
            break;
        }
      }
      std::sort(out.begin(), out.end());
      return out;
    }();
    return paths;
  }

  std::shared_ptr<ControllerPoolBase> pool_;
  std::shared_ptr<ISnapshotManager> snapshots_;
  std::shared_ptr<IRedisSynchronizer> redis_;
  std::shared_ptr<RotationManager> rotation_;
  std::shared_ptr<IStudyEngine> study_;
  std::shared_ptr<Database079> db_;
  std::shared_ptr<HotMatrixCache> matrixCache_;
  std::shared_ptr<ShardManager> shards_;
  std::shared_ptr<ISparkArray> spark_;
  /* v8.3 mission outline cache: the GNN graph is long-lived, so the outline
     for a goal is built once and reused every deliberator tick. */
  mutable std::mutex outlineMu_;
  mutable std::unordered_map<std::string, std::string> missionOutlineCache_;
  /* v10.1: serializes the whole mission-deliberator critical section.  Parallel
     missions are scheduled on separate threads by loopRun (std::async), but the
     deliberator touches non-thread-safe shared state (llama channel, workspace
     file IO, experience/ccm stores, runtime graph); this mutex is the single
     guarantee that no two missions corrupt it. */
  mutable std::mutex missionDelibMu_;
  std::shared_ptr<IReinforcementLearner> rl_;
  std::shared_ptr<IAdversarialLearner> adv_;
  std::shared_ptr<IGnnGaLearner> gnnGa_;
  transformer::TransformerService transformer_;
  model_lifecycle::ModelLifecycleManager modelLifecycle_;
  autonomy::TransformerClusterManager clusterManager_;
  autonomy::SpiderAutonomyManager spiderAutonomy_;
  autonomy::OptimizerAutonomyManager optimizerAutonomy_;
  autonomy::DatasetCatalogManager datasetCatalog_;
  autonomy::CognitionAutonomyManager cognitionManager_;
  phoenix::memory::HierarchicalMemory cognitionMemory_;
  std::shared_ptr<phoenix::emotion::EmotionSystem> emotionSystem_;
  /* Exclusive llama lease: liveLlamaUsers() is 1 so instances queue, never
     overlap one llama_context.  Chat waiters outrank mission ticks. */
  mutable std::mutex llamaSlotMu_;
  mutable std::condition_variable llamaSlotCv_;
  std::atomic<int> llamaSlotsInUse_{0};
  std::atomic<int> llamaChatWaiters_{0};
  std::mutex transformerMu_;
  std::mutex chatMu_;
  Config config_;
  std::chrono::steady_clock::time_point startedAt_;
  bool authEnabled_{true};
  bool acceptLocalToken_{true};
  std::string jwtSecret_;
  std::shared_ptr<MemeBarrier> barrier_;
  LoggerCXX &logger_{LoggerCXX::instance()};
  bool rlDisabled_{false};
  bool advDisabled_{false};
  bool gnnGaDisabled_{false};
  bool dialogLearningEnabled_{true};
  bool dialogTransformerLearning_{false};
  bool dialogLearnGraphContext_{false};
  bool dataCleaningEnabled_{true};
  bool retrainTrace_{false};
  bool cognitionAutonomyEnabled_{true};
  GatewayContextService contextService_;
  size_t dataCleanMaxChars_{2048};
  std::atomic<uint64_t> cleanedInputCount_{0};
  std::atomic<uint64_t> cleanedSampleCount_{0};
  std::mutex routeMetricMu_;
  std::unordered_map<std::string, RouteMetric> routeMetrics_;
  std::mutex trainMonitorMu_;
  uint64_t trainMonitorSeq_{0};
  TrainMonitorState trainMonitorActive_;
  std::deque<json> trainMonitorHistory_;
  std::mutex fineTuningMu_;
  bool fineTuningRunning_{false};
  std::atomic<int> dialogAsyncInFlight_{0};
  int dialogAsyncLimit_{2};
  struct DialogCounters {
    int total{0};
    int lastRL{0};
    int lastADV{0};
    int lastGNN{0};
  } dialogCounters_;
  struct DialogThresholds {
    double rlEvery{20};
    double advEvery{30};
    double gnnEvery{40};
  } dialogThresholds_;
  bool releaseSnapshotLogged_{false};

  std::unique_ptr<phoenix::v7::AheadModule> aheadModule_;

  struct CognitionPack {
    std::string context;
    std::string instanceMemoryText;
    nlohmann::json inferenceOptions;
    nlohmann::json emotionTensor;
    std::vector<nlohmann::json> mixedModalOutputs;
    nlohmann::json worldEvidence;
  };

  CognitionPack runCognitionAutonomy(const nlohmann::json &body,
                                     const std::string &userText,
                                     const std::string &baseContext) {
    CognitionPack pack;
    pack.context = baseContext;
    if (!cognitionAutonomyEnabled_)
      return pack;
    try {
      std::string sessionId = body.value("sessionId", "");
      const auto chatScope = phoenix::memory::memoryScopeForChatRoute(body);
      phoenix::memory::MemoryScopeGuard chatGuard(chatScope);
      auto &hotHier =
          phoenix::memory::ScopedTrainableMemory::instance().hot(chatScope).hier;
      nlohmann::json observePayload;
      observePayload["sessionId"] = chatScope.key();
      observePayload["contextTag"] = chatScope.key();
      observePayload["memoryKind"] = "chat";
      observePayload["reply"] = userText;
      observePayload["autonomySource"] = "gateway-chat";
      observePayload["worldUncertainty"] = body.value("worldUncertainty", 0.25);
      nlohmann::json worldState;
      worldState["graphContext"] = baseContext;
      worldState["graphResult"] = body.value("graphResult", nlohmann::json::object());
      if (!sessionId.empty()) {
        auto prev = hotHier.get("cognition");
        if (prev)
          worldState["previousCognition"] = *prev;
        if (worldModel_) {
          try {
            auto wmState = worldModel_->sessionState(sessionId, 8);
            if (wmState.value("ok", false)) {
              worldState["worldModelState"] = wmState;
              world_model::PromptContextOptions wmOpts;
              wmOpts.maxRecentEvidence = 0;
              wmOpts.maxSummaryChars = 220;
              /* Raw causal chat on the small unit-loop model: telemetry
                 lines (relations/goals/prediction/calibration) AND scene
                 summary/object/evidence lines dominate the short prompt and
                 the model pattern-continues them into salad at EVERY
                 temperature (ablation _temp_ab3.py on the exact live chat
                 prompt: 0.40..0.15 all salad; emotion head + question alone
                 answers cleanly).  The world-model pipeline still runs
                 (ingest, calibration, /api/world-model/*) — only its chat
                 prompt rendering is fully suppressed. */
              wmOpts.telemetryLines = false;
              wmOpts.sceneLines = false;
              const std::string wmOnly =
                  world_model::mergePromptContext("", wmState, wmOpts);
              if (!wmOnly.empty())
                pack.instanceMemoryText += wmOnly;
              pack.context = world_model::mergePromptContext(pack.context, wmState, wmOpts);
              worldState["graphContext"] = pack.context;
              /* v8.0 audit A: the epistemic signal now comes from the world
                 model\'s prediction-alignment calibration.  A misaligned
                 prediction (low alignmentScore) raises worldUncertainty and
                 therefore exploration, closing the predict->align->calibrate
                 loop into the AGI planner. */
              if (wmState.contains("prediction") && wmState["prediction"].is_object()) {
                const auto &pred = wmState["prediction"];
                if (pred.contains("predictionCalibration") &&
                    pred["predictionCalibration"].is_object() &&
                    pred["predictionCalibration"].contains("lastAlignmentScore") &&
                    pred["predictionCalibration"]["lastAlignmentScore"].is_number()) {
                  double alignment = pred["predictionCalibration"]["lastAlignmentScore"].get<double>();
                  alignment = std::max(0.0, std::min(1.0, alignment));
                  observePayload["worldUncertainty"] = 1.0 - alignment;
                  observePayload["worldAlignmentScore"] = alignment;
                }
              }
            }
          } catch (...) {
          }
        }
      }
      auto obs = cognitionManager_.observe(observePayload, worldState);
      if (!obs.value("ok", false))
        return pack;

      nlohmann::json iterPayload;
      iterPayload["sessionId"] = chatScope.key();
      iterPayload["contextTag"] = chatScope.key();
      iterPayload["memoryKind"] = "chat";
      iterPayload["userPrompt"] = userText;
      iterPayload["graphContext"] = baseContext;
      iterPayload["body"] = body;
      /* audit A: expose the world model\'s expected next state to the
         iterate() payload so the EFE planner can use it as a forward-model
         prior (field is optional; iterate() ignores it when absent). */
      if (worldState.contains("worldModelState") &&
          worldState["worldModelState"].is_object() &&
          worldState["worldModelState"].contains("prediction"))
        iterPayload["worldPrediction"] = worldState["worldModelState"]["prediction"];
      /* audit A: expose the world model\'s expected next state to the
         iterate() payload so the EFE planner can use it as a forward-model
         prior (field is optional; iterate() ignores it when absent). */
      if (worldState.contains("worldModelState") &&
          worldState["worldModelState"].is_object() &&
          worldState["worldModelState"].contains("prediction"))
        iterPayload["worldPrediction"] = worldState["worldModelState"]["prediction"];
      auto iter = cognitionManager_.iterate(iterPayload, worldState);
      if (!iter.value("ok", false))
        return pack;

      const nlohmann::json &r = iter.value("result", nlohmann::json::object());
      pack.inferenceOptions = r.value("inferenceOptions", nlohmann::json::object());
      pack.emotionTensor = r.value("emotionTensor", nlohmann::json::object());
      pack.mixedModalOutputs = r.value("mixedModalOutputs", nlohmann::json::array()).get<std::vector<nlohmann::json>>();
      pack.worldEvidence = r.value("worldEvidence", nlohmann::json::array());
      // AGI action execution: fold the chosen tool's output back into the
      // context so the generation step can use it (closes the plan->act->
      // observe loop).
      if (r.contains("agiPlan") && r["agiPlan"].is_object()) {
        const auto &agiPlan = r["agiPlan"];
        if (agiPlan.contains("execution") && agiPlan["execution"].is_object()) {
          const auto &exec = agiPlan["execution"];
          if (exec.value("handled", false)) {
            const std::string toolOut = exec.value("reply", std::string());
            const std::string toolAction = exec.value("action", std::string());
            if (!toolOut.empty()) {
              const std::string note =
                  "\n[tool_result action=" + toolAction +
                              "] output=" + toolOut + " [/tool_result]";
              pack.context += note;
              pack.instanceMemoryText += note;
            }
          }
        }
      }
      /* Architecture rule (user directive 2026-10-04): the prompt carries
         ONLY the task / ask message.  Emotion state modulates generation
         through the pre-processing matrix channels — inferenceOptions
         (temperature/top_p/penalties) and the vocabulary logit_bias map —
         never as prompt text.  The cognitionModulation string
         (EmotionTensor::modulationHint, "high-arousal; risk-tolerant; ...")
         stays in the iterate() result json for observability but is no
         longer prepended to the prompt context.  Ablation _temp_ab5.py:
         pure message prompts answer correctly at 0.25..0.40. */
      (void)r.value("cognitionModulation", "");

      // Run graph-diffusion summarization over the provided graph result.
      phoenix::graph::GraphDiffusionSummarizer diffusor;
      auto diffSummary = diffusor.summarizeFromJson(worldState["graphResult"]);
      if (!diffSummary.rankedNodes.empty()) {
        if (!pack.worldEvidence.is_array())
          pack.worldEvidence = nlohmann::json::array();
        pack.worldEvidence.push_back(
            {{"stage", "graphDiffusion"}, {"summary", diffSummary.toJson()}});
        if (!diffSummary.summaryText.empty()) {
          /* Chat graph diffusion stays in the chat bucket.  Never write
             the default mission GNN summary from a chat turn.
             v10.2: go through setGnnSummary so a concurrent writer of the
             same scope cannot reallocate the string under us. */
          phoenix::memory::ScopedTrainableMemory::instance().setGnnSummary(
              chatScope, diffSummary.summaryText);
          auto &chatHot =
              phoenix::memory::ScopedTrainableMemory::instance().hot(chatScope);
          chatHot.graphHint = diffSummary.summaryText;
        }
      }

      // Persist the cognition pack in hierarchical memory for the next turn.
      if (!sessionId.empty()) {
        nlohmann::json packJson;
        packJson["inferenceOptions"] = pack.inferenceOptions;
        packJson["emotionTensor"] = pack.emotionTensor;
        packJson["worldEvidence"] = pack.worldEvidence;
        packJson["mixedModalOutputs"] = pack.mixedModalOutputs;
        double activation = 0.5;
        if (pack.emotionTensor.is_object() &&
            pack.emotionTensor.contains("arousal") &&
            pack.emotionTensor["arousal"].is_number()) {
          activation = 0.5 + 0.5 * pack.emotionTensor["arousal"].get<double>();
        }
        hotHier.put("cognition", packJson, activation);
      }
    } catch (...) {
    }
    return pack;
  }

  void loadDialogConfig() {
    dialogThresholds_.rlEvery =
        phoenix::cfgOr<double>("dialog.rlEvery", dialogThresholds_.rlEvery);
    dialogThresholds_.advEvery =
        phoenix::cfgOr<double>("dialog.advEvery", dialogThresholds_.advEvery);
    dialogThresholds_.gnnEvery =
        phoenix::cfgOr<double>("dialog.gnnEvery", dialogThresholds_.gnnEvery);
    dialogAsyncLimit_ =
        phoenix::cfgOr<int>("dialog.dialogAsyncLimit", dialogAsyncLimit_);
  }
  struct AuthError {
    std::string error;
    std::string message;
    drogon::HttpStatusCode code{drogon::k401Unauthorized};
  };
  static thread_local AuthError lastAuthError_;
  static thread_local bool hasAuthError_;

  template <typename Fn> void launchDialogTask(Fn &&fn) {
    int now = dialogAsyncInFlight_.load(std::memory_order_relaxed);
    if (now >= std::max(1, dialogAsyncLimit_))
      return;
    dialogAsyncInFlight_.fetch_add(1, std::memory_order_relaxed);
    std::thread([this,
                 task = std::function<void()>(std::forward<Fn>(fn))]() mutable {
      try {
        task();
      } catch (...) {
      }
      dialogAsyncInFlight_.fetch_sub(1, std::memory_order_relaxed);
    }).detach();
  }

  json parseRequestBody(const drogon::HttpRequestPtr &req, bool &ok) const {
    ok = true;
    auto payload = req->getJsonObject();
    if (payload)
      return fromJsoncpp(*payload);
    auto body = req->getBody();
    if (body.empty())
      return json::object();
    auto contentType = req->getHeader("content-type");
    std::string lowered = contentType;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), ::tolower);
    auto tryParseJsonBody = [&]() -> std::optional<json> {
      try {
        return json::parse(std::string(body));
      } catch (...) {
        return std::nullopt;
      }
    };
    if (lowered.find("application/json") != std::string::npos) {
      auto parsed = tryParseJsonBody();
      if (parsed)
        return *parsed;
    }
    if (lowered.find("application/x-www-form-urlencoded") !=
        std::string::npos) {
      return parseFormUrlEncoded(std::string(body));
    }
    std::string raw = std::string(body);
    auto l = raw.find_first_not_of(" \t\r\n");
    if (l != std::string::npos) {
      char c = raw[l];
      if (c == '{' || c == '[') {
        auto parsed = tryParseJsonBody();
        if (parsed)
          return *parsed;
      }
    }
    ok = false;
    return json();
  }

  bool jsTruthy(const json &v) const {
    if (v.is_null())
      return false;
    if (v.is_boolean())
      return v.get<bool>();
    if (v.is_number())
      return v.get<double>() != 0.0;
    if (v.is_string())
      return !v.get<std::string>().empty();
    return true;
  }

  struct AddonToolCall {
    bool requested{false};
    std::string type;
    std::string input;
  };

  static std::string trimText(std::string s) {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos)
      return "";
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
  }

  std::string sanitizeInputText(const std::string &raw, bool &changed) const {
    changed = false;
    if (!dataCleaningEnabled_)
      return raw;
    std::string out;
    out.reserve(raw.size());
    bool previousSpace = false;
    for (unsigned char ch : raw) {
      if (ch < 0x20 || ch == 0x7F) {
        if (ch == '\t' || ch == '\n' || ch == '\r') {
          if (!previousSpace) {
            out.push_back(' ');
            previousSpace = true;
          }
        }
        changed = true;
        continue;
      }
      if (std::isspace(ch)) {
        if (!previousSpace) {
          out.push_back(' ');
          previousSpace = true;
        }
        continue;
      }
      out.push_back((char)ch);
      previousSpace = false;
    }
    auto trimmed = trimText(out);
    if (trimmed.size() > (size_t)dataCleanMaxChars_) {
      size_t cut = (size_t)dataCleanMaxChars_;
      while (cut > 0 &&
             (static_cast<unsigned char>(trimmed[cut]) & 0xC0) == 0x80)
        --cut;
      if (cut == 0)
        cut =
            (size_t)dataCleanMaxChars_; // fall back to raw cut if no boundary found
      trimmed.resize(cut);
      changed = true;
    }
    if (trimmed != raw)
      changed = true;
    return trimmed;
  }

  static std::string attachAddonToolContract(const std::string &graphContext) {
    std::string out = graphContext;
    if (!out.empty())
      out += "\n";
    out += "[tool_contract] For arithmetic/calculation tasks, emit: "
           "[[ADDON:math|<expression>]] as the first line; for web/online fact "
           "lookup tasks, emit: [[ADDON:search|<query>]] as the first line. "
           "Keep payload plain and executable. [/tool_contract]";
    return out;
  }

  static AddonToolCall parseAddonToolCall(const std::string &reply,
                                          const std::string &fallbackInput) {
    AddonToolCall out;
    try {
      std::regex re(R"(\[\[ADDON:([a-zA-Z0-9_\-]+)\|(.+?)\]\])",
                    std::regex::icase);
      std::smatch m;
      if (std::regex_search(reply, m, re) && m.size() >= 3) {
        out.requested = true;
        out.type = trimText(m[1].str());
        out.input = trimText(m[2].str());
        if (out.input.empty())
          out.input = trimText(fallbackInput);
      }
    } catch (...) {
      out = AddonToolCall{};
    }
    return out;
  }

  static std::string stripAddonToolDirective(const std::string &reply) {
    try {
      std::regex re(R"(\s*\[\[ADDON:[^\]]+\]\]\s*)", std::regex::icase);
      return trimText(std::regex_replace(reply, re, " "));
    } catch (...) {
      return trimText(reply);
    }
  }

  static std::string buildAddonResultContext(const std::string &graphContext,
                                             const AddonToolCall &call,
                                             const addon::AddonResult &result) {
    std::string out = graphContext;
    if (!out.empty())
      out += "\n";
    std::string addonType =
        call.type.empty() ? result.meta.value("addon", std::string("unknown"))
                          : call.type;
    out += "[tool_result addon=" + addonType + "]";
    if (!call.input.empty())
      out += " input=" + call.input;
    out += " output=" + result.reply;
    out += " [/tool_result]";
    return out;
  }

  bool invokeAddonTool(const AddonToolCall &call, const json &payload,
                       addon::AddonResult &result) const {
    result = addon::AddonResult{};
    if (!call.requested)
      return false;
    std::string typeLower = call.type;
    std::transform(typeLower.begin(), typeLower.end(), typeLower.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    if (!typeLower.empty() && typeLower != "math" && typeLower != "search" &&
        typeLower != "research" && typeLower != "web" &&
        typeLower != "computer" && typeLower != "shell" &&
        typeLower != "cli-json" && typeLower != "cli" &&
        typeLower != "script")
      return false;
    auto controller = pool_->getActive();
    if (!controller)
      return false;
    auto runtime =
        std::dynamic_pointer_cast<RuntimeState>(controller->runtime());
    if (!runtime)
      return false;
    json toolPayload = payload;
    if (!typeLower.empty())
      toolPayload["__addonType"] = typeLower;
    result = runtime->runAddonTool(call.input, toolPayload);
    addon::sealAddonResultUnits(result);
    if (result.handled && logger_.enabled()) {
      logger_.log(LoggerCXX::Type::COMPUTE,
                  std::string("addon_tool handled type=") +
                      (typeLower.empty() ? "auto" : typeLower) +
                      " input=" + call.input + " addon=" +
                      result.meta.value("addon", std::string("unknown")));
    }
    return result.handled;
  }

  bool isGroupProc() const {
    return dynamic_cast<GroupProcessPool *>(pool_.get()) != nullptr;
  }

  bool authOK(const drogon::HttpRequestPtr &req) {
    hasAuthError_ = false;
    if (!authEnabled_)
      return true;
    if (req->method() == drogon::Options)
      return true;
    auto path = req->path();
    if (path == "/api/system/status")
      return true;
    if (path == "/context/reset")
      return true;
    if (path == "/context/knowledge")
      return true;
    if (path.rfind("/api/", 0) != 0)
      return true;
    auto auth = req->getHeader("authorization");
    std::string token;
    try {
      std::regex re("^Bearer\\s+(.+)$", std::regex::icase);
      std::smatch m;
      if (std::regex_match(auth, m, re) && m.size() >= 2) {
        token = m[1].str();
      }
    } catch (...) {
      token.clear();
    }
    if (token.empty()) {
      lastAuthError_ = {"unauthorized", "", drogon::k401Unauthorized};
      hasAuthError_ = true;
      logSecurityEvent(std::string("auth failed path=") + path +
                       " reason=missing-token");
      return false;
    }
    if (acceptLocalToken_ && token.rfind("local-", 0) == 0) {
      return true;
    }
    try {
      auto dec = jwt::decode<jwt::traits::nlohmann_json>(token);
      jwt::verify<jwt::traits::nlohmann_json>()
          .allow_algorithm(jwt::algorithm::hs256{jwtSecret_})
          .verify(dec);
      return true;
    } catch (const std::exception &e) {
      lastAuthError_ = {"invalid-token", e.what(), drogon::k401Unauthorized};
      hasAuthError_ = true;
      logSecurityEvent(std::string("auth failed path=") + path +
                       " reason=" + e.what());
      return false;
    } catch (...) {
      lastAuthError_ = {"invalid-token", "invalid token",
                        drogon::k401Unauthorized};
      hasAuthError_ = true;
      logSecurityEvent(std::string("auth failed path=") + path +
                       " reason=invalid-token");
      return false;
    }
  }

  /* v8.6 summon endpoint adapter: translate Phoenix's (dynamic, static)
     prompt split into the target backend's API shape.  This is where the
     framework adapts to the model - the model itself never sees a tool
     protocol.  Bounded by ep.timeoutMs; never touches the main llama slot
     bookkeeping (a summoned box does not occupy the base model's slot). */
  json chatWithSummonEndpoint(const phoenix::summon::ModelEndpoint &ep,
                              const std::string &text,
                              const std::string &graphContext, int maxTokens,
                              const json &inferenceOptions = json::object()) {
    json out = json::object();
    out["provider"] = std::string("summon:") + ep.kind;
    out["model"] = ep.name;
    out["ok"] = false;
    std::string prompt = text;
    if (!graphContext.empty()) {
      prompt = graphContext;
      if (!text.empty()) {
        if (!prompt.empty() && prompt.back() != '\n') prompt.push_back('\n');
        prompt += text;
      }
    }
    const int numPredict = std::max(16, maxTokens > 0 ? maxTokens : ep.maxTokens);
    json body;
    std::string path;
    if (ep.kind == "ollama") {
      path = "/api/chat";
      json options = json{{"num_predict", numPredict}};
      if (inferenceOptions.is_object()) {
        for (auto it = inferenceOptions.begin(); it != inferenceOptions.end();
             ++it) {
          if (it.value().is_number()) options[it.key()] = it.value();
        }
      }
      body = {{"model", ep.model},
              {"stream", false},
              {"messages",
               json::array({json{{"role", "user"}, {"content", prompt}}})},
              {"options", options}};
    } else if (ep.kind == "openai") {
      path = "/v1/chat/completions";
      body = {{"model", ep.model},
              {"stream", false},
              {"messages",
               json::array({json{{"role", "user"}, {"content", prompt}}})},
              {"max_tokens", numPredict}};
    } else { /* llamacpp */
      path = "/completion";
      body = {{"prompt", prompt},
              {"n_predict", numPredict},
              {"stream", false},
              {"cache_prompt", false}};
      if (inferenceOptions.is_object()) {
        for (auto it = inferenceOptions.begin(); it != inferenceOptions.end();
             ++it) {
          if (it.value().is_number()) body[it.key()] = it.value();
        }
      }
    }
    try {
      auto client = drogon::HttpClient::newHttpClient(ep.url);
      if (!client) {
        out["error"] = "summon client init failed";
        return out;
      }
      auto req = drogon::HttpRequest::newHttpRequest();
      req->setMethod(drogon::Post);
      req->setPath(path);
      req->setContentTypeCode(drogon::CT_APPLICATION_JSON);
      req->setBody(body.dump());
      /* The promise must outlive a timed-out wait: the client callback runs
         on its own thread and used to write through a dangling reference
         after the local promise was destroyed on timeout (the queue/crash
         seen when several carriers hit a slow coder).  A shared_ptr keeps it
         alive for the callback; the future is intentionally dropped on
         timeout. */
      auto pr = std::make_shared<
          std::promise<std::pair<drogon::ReqResult, drogon::HttpResponsePtr>>>();
      auto fut = pr->get_future();
      client->sendRequest(
          req, [pr](drogon::ReqResult result,
                    const drogon::HttpResponsePtr &resp) {
            try {
              pr->set_value({result, resp});
            } catch (...) {
            }
          });
      const int waitMs = std::max(1000, ep.timeoutMs);
      if (fut.wait_for(std::chrono::milliseconds(waitMs)) !=
          std::future_status::ready) {
        out["error"] = "summon endpoint timeout";
        return out;
      }
      auto pair = fut.get();
      if (pair.first != drogon::ReqResult::Ok || !pair.second) {
        out["error"] = "summon endpoint unreachable";
        return out;
      }
      if (pair.second->statusCode() < 200 || pair.second->statusCode() >= 300) {
        out["error"] = std::string("summon endpoint http ") +
                       std::to_string(static_cast<int>(pair.second->statusCode()));
        return out;
      }
      json doc = json::parse(pair.second->getBody(), nullptr, false);
      if (doc.is_discarded()) {
        out["error"] = "summon endpoint bad json";
        return out;
      }
      std::string reply;
      if (ep.kind == "ollama") {
        reply = doc.value("message", json::object()).value("content",
                                                           std::string());
      } else if (ep.kind == "openai") {
        const auto choices = doc.value("choices", json::array());
        if (!choices.empty() && choices[0].is_object())
          reply = choices[0].value("message", json::object())
                      .value("content", std::string());
      } else {
        reply = doc.value("content", std::string());
      }
      if (reply.empty()) {
        out["error"] = "summon endpoint empty reply";
        return out;
      }
      out["ok"] = true;
      out["reply"] = reply;
      return out;
    } catch (const std::exception &ex) {
      out["error"] = std::string("summon endpoint exception: ") + ex.what();
      return out;
    } catch (...) {
      out["error"] = "summon endpoint exception";
      return out;
    }
  }

  json chatWithOllama(const std::string &text, const std::string &graphContext,
                      int maxTokens,
                      const json &inferenceOptions = json::object()) {
    json out = json::object();
    out["provider"] =
        config_.ollamaFineTuning ? "ollama+fine-tuning" : "ollama";
    try {
      auto pickAvailableOllamaModel = [&]() -> std::string {
        try {
          auto client = drogon::HttpClient::newHttpClient(
              config_.ollamaBaseUrl.empty()
                  ? std::string("http://127.0.0.1:11434")
                  : config_.ollamaBaseUrl);
          if (!client)
            return "";
          auto tagsReq = drogon::HttpRequest::newHttpRequest();
          tagsReq->setMethod(drogon::Get);
          tagsReq->setPath("/api/tags");
          std::promise<std::pair<drogon::ReqResult, drogon::HttpResponsePtr>>
              tagsPromise;
          auto tagsFuture = tagsPromise.get_future();
          client->sendRequest(
              tagsReq, [&tagsPromise](drogon::ReqResult result,
                                      const drogon::HttpResponsePtr &resp) {
                try {
                  tagsPromise.set_value({result, resp});
                } catch (...) {
                }
              });
          int waitMs =
              std::max(1000, std::min(config_.ollamaRequestTimeoutMs, 10000));
          if (tagsFuture.wait_for(std::chrono::milliseconds(waitMs)) !=
              std::future_status::ready)
            return "";
          auto tagsPair = tagsFuture.get();
          if (tagsPair.first != drogon::ReqResult::Ok || !tagsPair.second)
            return "";
          if (tagsPair.second->statusCode() < 200 ||
              tagsPair.second->statusCode() >= 300)
            return "";
          json tagsDoc =
              json::parse(tagsPair.second->getBody(), nullptr, false);
          if (tagsDoc.is_discarded() || !tagsDoc.contains("models") ||
              !tagsDoc["models"].is_array())
            return "";
          for (const auto &m : tagsDoc["models"]) {
            if (!m.is_object())
              continue;
            std::string name = trimCopy(m.value("name", ""));
            if (!name.empty())
              return name;
            std::string model = trimCopy(m.value("model", ""));
            if (!model.empty())
              return model;
          }
        } catch (...) {
        }
        return "";
      };

      std::string selectedModel = trimCopy(config_.ollamaModel);
      if (selectedModel.empty()) {
        selectedModel = pickAvailableOllamaModel();
      }
      if (selectedModel.empty()) {
        out["ok"] = false;
        out["error"] = "ollama model not available";
        out["reply"] = "";
        return out;
      }
      out["model"] = selectedModel;

      std::string prompt = text;
      if (!graphContext.empty()) {
        prompt = graphContext;
        if (!text.empty()) {
          if (!prompt.empty() && prompt.back() != '\n') prompt.push_back('\n');
          prompt += text;
        }
      }
      json options = json{{"num_predict", std::max(16, maxTokens)}};
      if (inferenceOptions.is_object() && !inferenceOptions.empty()) {
        for (auto it = inferenceOptions.begin(); it != inferenceOptions.end(); ++it) {
          if (it.value().is_number())
            options[it.key()] = it.value();
        }
      }
      json payload = {
          {"model", selectedModel},
          {"stream", false},
          {"messages",
           json::array({json{{"role", "user"}, {"content", prompt}}})},
          {"options", options}};

      auto client = drogon::HttpClient::newHttpClient(
          config_.ollamaBaseUrl.empty() ? std::string("http://127.0.0.1:11434")
                                        : config_.ollamaBaseUrl);
      if (!client) {
        out["ok"] = false;
        out["error"] = "ollama client init failed";
        out["reply"] = "";
        return out;
      }

      auto req = drogon::HttpRequest::newHttpRequest();
      req->setMethod(drogon::Post);
      req->setPath("/api/chat");
      req->setContentTypeCode(drogon::CT_APPLICATION_JSON);
      req->setBody(payload.dump());

      std::promise<std::pair<drogon::ReqResult, drogon::HttpResponsePtr>>
          promise;
      auto future = promise.get_future();
      client->sendRequest(req, [&promise](drogon::ReqResult result,
                                          const drogon::HttpResponsePtr &resp) {
        try {
          promise.set_value({result, resp});
        } catch (...) {
        }
      });

      if (future.wait_for(std::chrono::milliseconds(
              config_.ollamaRequestTimeoutMs)) != std::future_status::ready) {
        out["ok"] = false;
        out["error"] = "ollama request timeout";
        out["timeoutMs"] = config_.ollamaRequestTimeoutMs;
        out["reply"] = "";
        return out;
      }

      auto pair = future.get();
      auto reqResult = pair.first;
      auto resp = pair.second;
      if (reqResult != drogon::ReqResult::Ok || !resp) {
        out["ok"] = false;
        out["error"] = "ollama request failed";
        out["reply"] = "";
        return out;
      }
      if (resp->statusCode() < 200 || resp->statusCode() >= 300) {
        out["ok"] = false;
        out["status"] = (int)resp->statusCode();
        out["error"] = "ollama bad status";
        out["reply"] = "";
        return out;
      }

      json doc = json::parse(resp->getBody(), nullptr, false);
      if (doc.is_discarded()) {
        out["ok"] = false;
        out["error"] = "ollama invalid json";
        out["reply"] = "";
        return out;
      }

      std::string reply;
      if (doc.contains("message") && doc["message"].is_object() &&
          doc["message"].contains("content") &&
          doc["message"]["content"].is_string()) {
        reply = doc["message"]["content"].get<std::string>();
      } else if (doc.contains("response") && doc["response"].is_string()) {
        reply = doc["response"].get<std::string>();
      }

      out["ok"] = !reply.empty();
      out["reply"] = reply;
      if (doc.contains("done"))
        out["done"] = doc["done"];
      if (doc.contains("total_duration"))
        out["total_duration"] = doc["total_duration"];
      return out;
    } catch (const std::exception &e) {
      out["ok"] = false;
      out["error"] = e.what();
      out["reply"] = "";
      return out;
    } catch (...) {
      out["ok"] = false;
      out["error"] = "ollama unknown error";
      out["reply"] = "";
      return out;
    }
  }

  json chatWithExternalAdapter(const std::string &provider,
                               const std::string &baseUrl,
                               const std::string &model, int timeoutMs,
                               const std::string &text,
                               const std::string &graphContext, int maxTokens,
                               const json &inferenceOptions = json::object()) {
    json out = json::object();

