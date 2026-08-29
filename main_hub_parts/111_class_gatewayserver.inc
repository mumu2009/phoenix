// Auto-generated from main.cpp
// Segment: class_GatewayServer
// Original lines: 13037-13883

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
      nlohmann::json evidence;
      evidence["sessionId"] = sessionId;
      evidence["modality"] = "text";
      evidence["text"] = text;
      evidence["graphSummary"] = graphSummary;
      evidence["metadata"] = nlohmann::json{{"source", source}};
      /* v8.0 audit B: tag the running mission so prediction docs aggregate
         per goal (world evidence becomes goal-scoped observability). */
      try {
        /* v8.x concurrent missions: tag with the DEFAULT mission (the one a
           chat session most likely talks about); per-session tagging stays a
           caller decision via sessionId. */
        auto ms = cognitionManager_.missionStatus();
        const auto &res = ms.value("result", nlohmann::json::object());
        const std::string defId = res.value("defaultMissionId", std::string());
        const auto &arr = res.value("missions", nlohmann::json::array());
        for (const auto &mm : arr) {
          if (mm.value("id", std::string()) != defId) continue;
          const auto &st = mm.value("stats", nlohmann::json::object());
          if (st.value("state", 0) == 1)
            evidence["metadata"]["missionId"] = defId;
          break;
        }
      } catch (...)
      {
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
      missionCfg["mutationRate"] = phoenix::cfgOr<double>("mission.mutationRate", 0.05);
      missionCfg["maxReplicas"] = phoenix::cfgOr<int>("mission.maxReplicas", 4);
      /* 0 = UNLIMITED task-tree depth (boxes may summon their own boxes);
         user can cap it via mission.maxReplicaDepth / AI_MISSION_MAX_REPLICA_DEPTH. */
      missionCfg["maxReplicaDepth"] = phoenix::resolveConfig<int>("mission.maxReplicaDepth", 0, "AI_MISSION_MAX_REPLICA_DEPTH");
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
            try {
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
              if (currentFile.empty() && !deliverable.empty())
                currentFile = deliverable; /* before the first write */
              /* If the in-memory prior grew past a stale/shorter workspace
                 snapshot, prefer the longer body so the model is not stuck
                 rewriting only the first 256KB forever. */
              if (deliverable.size() > currentFile.size())
                currentFile = deliverable;

              /* Seed before pack. Packing an empty file made
                 recentChars=0 and dropped the short-term window from
                 the causal stream (2868973). No ChatML roles here —
                 this is the document opening, not a user/assistant
                 turn. */
              bool seededEmpty = false;
              if (currentFile.empty()) {
                const std::string seed =
                    phoenix::mission::formatEmptyFileSeed(goal);
                if (!seed.empty()) {
                  const auto wr = phoenix::mission::workspaceExecute(
                      workspaceRoot, scope,
                      nlohmann::json{{"action", "write"},
                                     {"path", deliverablePath},
                                     {"content", seed}});
                  if (wr.value("ok", false)) {
                    currentFile = seed;
                    seededEmpty = true;
                  }
                }
              }

              /* Two infer streams: causal = pin + working brief + draft
                 tail (resume); RAG = memory/GNN unit-query sequences as
                 residual emphasis. They are not concatenated onto the
                 causal tail. */
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
              std::string gnnText;
              nlohmann::json missionGraphResult = nlohmann::json::object();
              try {
                const auto copts = cognitionManager_.missionContextOptions();
                const auto &cr = copts.value("result", nlohmann::json::object());
                packOpt = phoenix::context::optionsFromJson(cr, packOpt);
              } catch (...) {
              }
              const auto packed =
                  phoenix::context::packContext(currentFile, gnnText, packOpt);
              const std::string packedBody = packed.packed;
              const std::string recallDraft = packed.recentFull.empty()
                  ? currentFile : packed.recentFull;
              const bool pinOnly =
                  phoenix::mission::deliverableIsOnlyPin(currentFile, goal) ||
                  seededEmpty;
              /* GNN pretrain gap: empty meme graph => empty diffusion prior.
                 Graph the same recent window infer sees, not a 2k tail and
                 not the human assignment. Pin-only is not a draft to graph. */
              if (!pinOnly &&
                  (packOpt.includeGnnSummary || unitPipelineEnabled)) {
                try {
                  const int lim =
                      phoenix::cfgOr<int>("mission.gnnPretrainDocs", 48);
                  seedGnnGraphFromRobotsCorpus(lim, "gnn-pretrain");
                  if (unitPipelineEnabled) {
                    std::string gnnSrc = packed.recentFull.empty()
                                             ? currentFile
                                             : packed.recentFull;
                    const size_t gnnCap = std::max(
                        static_cast<size_t>(phoenix::cfgOr<int>(
                            "mission.gnnSourceChars", 2000)),
                        packed.recentFull.size());
                    if (gnnSrc.size() > gnnCap)
                      gnnSrc = gnnSrc.substr(gnnSrc.size() - gnnCap);
                    if (!pinOnly && currentFile.size() >= 2500 &&
                        !gnnSrc.empty() &&
                        !phoenix::mission::draftTailLooksCollapsed(gnnSrc)) {
                      auto rt = activeRuntime();
                      auto st = rt ? std::dynamic_pointer_cast<RuntimeState>(rt)
                                   : nullptr;
                      if (st)
                        st->ingestDocument(
                            nlohmann::json{{"text", gnnSrc},
                                           {"source", "mission-draft"}});
                      missionGraphResult = buildMissionGraphResult(gnnSrc);
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
                    phoenix::cfgOr<size_t>("mission.goalClipChars", 8000);
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

              /* Do not dump the human assignment (Required Chapters / SRS
                 headings) into the chat system turn. That framed each tick
                 as "answer the brief" and the 8B recopied Background
                 Assumptions / Testing and Validation. Facts live in the
                 seed + pin; the file tail is the continuation. */
              std::string staticPrompt;
              {
                const std::string expPath = phoenix::resolveConfig<std::string>(
                    "mission.experiencePath",
                    std::string("./runtime_store/mission_experience.json"));
                /* Pin-only / empty draft: do not recall prior missions.
                   Assign writes the seed before tick 1; treating that
                   as a real window pulled tutorial/game-review RAG. */
                if (!pinOnly && !recallDraft.empty())
                  phoenix::mission::appendRecalledMemory(
                      missionMemorySnippets, phoenix::instance::ccmStorePath(),
                      expPath, goal, recallDraft, 3, 3);
                /* Pretrain wrote HierarchicalMemory under corpus:* and
                   never read it back on mission ticks. Same recall query
                   as CCM — not a second causal stream. */
                if (!pinOnly) {
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
                /* Assignment is RAG emphasis after a real draft exists.
                   On a pin-only first tick the working brief is already
                   in the causal prefix; dumping Required Chapters as RAG
                   framed the tick as "answer the exam". */
                if (!pinOnly) {
                const size_t memBefore = missionMemorySnippets.size();
                phoenix::inference::appendUnitQueriesFromParagraphs(
                    missionMemorySnippets, goalClip, "text");
                const size_t goalCap = 24;
                if (missionMemorySnippets.size() > memBefore + goalCap)
                  missionMemorySnippets.resize(memBefore + goalCap);
                if (phoenix::cfgOr<bool>("mission.outlineEnabled", true)) {
                  try {
                    const std::string outline = buildMissionOutline(goalClip);
                    if (outline.find("\n1. ") != std::string::npos)
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
                /* C1: the model's own plan.md is RAG emphasis, not a
                   [NEXT] block on the causal stream. Resume stays the
                   file tail. Skip on pin-only — there is no draft yet. */
                if (!pinOnly) {
                  const auto planRes = phoenix::mission::workspaceExecute(
                      workspaceRoot, scope,
                      nlohmann::json{{"action", "read"},
                                     {"path", "plan.md"}});
                  if (planRes.value("ok", false)) {
                    std::string plan = planRes.value("content", std::string());
                    if (plan.size() > 800)
                      plan = plan.substr(plan.size() - 800);
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
                   phoenix::cfgOr<int>("mission.maxSideTokens", 512)},
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
                if (!pinOnly && !emoSrc.empty()) {
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
              /* Names only. "Output is deliverable.md" + helper-box
                 copy made the 8B wrap ```markdown and invent
                 "Helper boxes spawned / please merge". The file tail
                 is the continuation; this prefix is operating facts. */
              if (phoenix::cfgOr<bool>("mission.compactDeliberatorPrompt", true)) {
                /* Facts only. A JSON tool schema in this prefix taught
                   every tick that the output *is* write/run/script, and
                   the 8B wrote a station-ops manual instead of the
                   design document. Bare JSON still hits the tool path;
                   do not advertise the schema on the causal stream. */
                (void)toolHint;
                staticPrompt +=
                    phoenix::mission::formatGoalConstraintPin(goalClip);
                staticPrompt +=
                    phoenix::mission::extractGoalWorkingContext(goalClip);
                if (pinOnly)
                  staticPrompt +=
                      phoenix::mission::extractGoalOpeningHeading(goalClip);
              } else {
                (void)toolHint;
                staticPrompt +=
                    phoenix::mission::formatGoalConstraintPin(goalClip);
                staticPrompt +=
                    phoenix::mission::extractGoalWorkingContext(goalClip);
                if (pinOnly)
                  staticPrompt +=
                      phoenix::mission::extractGoalOpeningHeading(goalClip);
                (void)isChildBox;
              }
              if (pinOnly)
                opts["repeat_penalty"] = 1.0;
              /* Pressure stays on the sampling / loop side. Putting
                 "p=0.xxx" in the prefix made the 8B write
                 "probability of failure (p) is 0.000". */
              /* Llama body is empty (no user turn). Draft tail is
                 resume_suffix so n_resume is real. Pin-only has no tail. */
              std::string dynamicPrompt;
              (void)listClip;
              std::string resumeSuffix;
              if (pinOnly || seededEmpty || currentFile.empty()) {
                opts.erase("gnn_units");
                opts.erase("gnn_content");
                resumeSuffix = phoenix::mission::formatMissionResumeSuffix(
                    currentFile, goalClip);
              } else {
                /* Do not put the model's own headings into llama stop.
                   ### / ## already written then blocked every later tick
                   (7159339: heading-cut kept=0 after tick 2). */
                const std::string resumeSrc = packed.recentFull.empty()
                    ? currentFile : packed.recentFull;
                const size_t resumeN = std::max(
                    resumeSrc.size(),
                    static_cast<size_t>(std::max(
                        200, phoenix::cfgOr<int>("mission.resumeChars", 720))));
                resumeSuffix = phoenix::mission::formatMissionResumeSuffix(
                    resumeSrc, goalClip, resumeN);
              }
              if (!resumeSuffix.empty())
                opts["resume_suffix"] = resumeSuffix;
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
              /* v8.x B1: attached images (encoded to concept notes) */
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
                    staticPrompt += "[image-context] attached images (concept):\n";
                    for (const auto &img : att["images"]) {
                      staticPrompt += "- " + img.value("file", std::string()) +
                                ": " + img.value("concept", std::string()) +
                                "\n";
                    }
                    staticPrompt += "\n";
                  }
                }
              }
              /* Resume is the file tail only. Headings / plan / outline
                 stay in context_units, not in this suffix. */
              if (!isChildBox) {
                /* DYNAMIC suffix: the parent's helper boxes (their files stay
                   readable; the parent remains fully responsible for the
                   final deliverable and keeps working itself). */
                try {
                  /* v8.x concurrent: the mission id lives in the scope */
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
                    staticPrompt += "boxes:";
                    for (const auto &box : children2) {
                      staticPrompt += " " + box.value("id", std::string());
                    }
                    staticPrompt += "\n";
                  }
              } catch (...) {
              }
              }

              const int genReserve = std::max(
                  16, phoenix::cfgOr<int>("mission.deliberateMaxTokens", 48));
              const int tmplReserve =
                  phoenix::cfgOr<int>("mission.templateOverheadTokens", 64);
              const int safePromptTokens = std::max(
                  512, packOpt.ctxTokens - genReserve - tmplReserve -
                           packOpt.overheadTokens);
              const int ngramFac =
                  phoenix::inference::ngramMergeFactor(packOpt.ngramMerge);
              const size_t maxPromptChars =
                  static_cast<size_t>(safePromptTokens) * 3 *
                  static_cast<size_t>(ngramFac);
              /* Split path: staticPrompt is pin + working brief.
                 Recent window is resume_suffix, not dynamicPrompt. */
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
                 Never skip while deliverable is still below minDeliverableChars. */
              {
                const bool stillDrafting =
                    static_cast<int>(currentFile.size()) < minDeliverableChars;
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
                const int deliberateTok = phoenix::cfgOr<int>(
                    "mission.deliberateMaxTokens", tokBudget);
                tokBudget = std::max(tokBudget, std::max(16, deliberateTok));
                opts["num_predict"] = tokBudget;
                /* Split /phx/generate does not preserve prefix-cache across
                   requests (KV is cleared per call); forcing cache_prompt only
                   adds unused fields. */
                opts["cache_prompt"] = false;
              }
              const auto r =
                  chatWithLlamaCpp(dynamicPrompt, staticPrompt, tokBudget, opts);
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
                     (a Helios chapter can casually mention a tool name). */
                  const bool looksLikeDocument =
                      text.size() > 180 ||
                      lower.rfind("**", 0) == 0 ||
                      lower.rfind("#", 0) == 0 ||
                      lower.rfind("chapter ", 0) == 0 ||
                      lower.find("\n**") != std::string::npos ||
                      lower.find("\n#") != std::string::npos;
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
                      if (lower.find(nmLower) != std::string::npos) {
                        toolName = nm;
                        break;
                      }
                    }
                  }
                  /* 2) intent keywords - ONLY for short imperative turns.
                     Long markdown prose (mission deliverable drafts) often
                     contains words like "research" / "box" / "helper"; treating
                     those as tool calls swallowed every tick (deliv stayed 0). */
                  if (toolName.empty() && !looksLikeDocument) {
                    if (lower.find("calculate") != std::string::npos ||
                        lower.find("compute") != std::string::npos ||
                        lower.find("math") != std::string::npos)
                      toolName = "math";
                    else if (lower.find("search for") != std::string::npos ||
                             lower.find("web search") != std::string::npos ||
                             lower.find("look up") != std::string::npos)
                      toolName = "search";
                    else if (lower.find("summon") != std::string::npos ||
                             lower.find("spawn a") != std::string::npos ||
                             lower.find("replicate") != std::string::npos)
                      toolName = "replicate";
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
                if (!embedded.empty()) {
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
                     left tick-2 Helios replies on the floor. */
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
                if (!trimmed.empty() && trimmed.front() == '{') {
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
                           future missions) can recall it. */
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
                                nlohmann::json{{"text", goal}},
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
              {
                const bool exam =
                    phoenix::mission::replyLooksLikeExamDump(chunk);
                const bool leftover =
                    phoenix::mission::replyLooksLikeLeftoverLoop(chunk);
                const bool homework =
                    phoenix::mission::replyLooksLikeHomeworkCodeDump(chunk);
                if (chunk.empty() || exam || leftover || homework) {
                  if (config_.httpLog)
                    std::cout << "[mission-deliberate] drop "
                              << (exam ? "exam-dump"
                                       : leftover ? "leftover-loop"
                                                  : homework ? "homework-code"
                                                             : "empty")
                              << " len=" << chunk.size() << " head="
                              << chunk.substr(0, 120) << std::endl;
                  return std::string();
                }
              }
              {
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
              if (!closingOddFence) {
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
                const std::string edited = currentFile.empty()
                    ? chunk
                    : phoenix::mission::joinDeliverableText(currentFile,
                                                            chunk);
                if (edited == currentFile) {
                  if (config_.httpLog)
                    std::cout << "[mission-deliberate] drop no-file-edit"
                              << std::endl;
                  return std::string();
                }
                if (config_.httpLog)
                  std::cout << "[mission-deliberate] file-edit "
                            << currentFile.size() << "->" << edited.size()
                            << std::endl;
                phoenix::mission::workspaceExecute(
                    workspaceRoot, scope,
                    nlohmann::json{{"action", "write"},
                                   {"path", deliverablePath},
                                   {"content", edited}});
                try {
                  phoenix::mission::depositMissionProgress(
                      phoenix::instance::ccmStorePath(),
                      phoenix::resolveConfig<std::string>(
                          "mission.experiencePath",
                          "./runtime_store/mission_experience.json"),
                      "mission:" + scope, goal, edited, chunk);
                } catch (...) {
                }
                if (emotionSystem_) {
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
              phoenix::mission::workspaceCachePut(scope, dynamicPrompt, chunk);
              return std::string(); /* file + memory synced; loop must not re-append */
              }
            } catch (...) {
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
      loopCfg["persistPath"] = phoenix::cfgOr<std::string>(
          "autonomyLoop.persistPath", "runtime_store/autonomy_state.json");
      loopCfg["deliberateMaxTokens"] =
          phoenix::cfgOr<int>("mission.deliberateMaxTokens", 192);
      loopCfg["childDeliberateMaxTokens"] =
          phoenix::cfgOr<int>("mission.childDeliberateMaxTokens", 96);
      cognitionManager_.configureAutonomyLoop(loopCfg);
      if (phoenix::cfgOr<bool>("autonomyLoop.enabled", false)) {
        cognitionManager_.startAutonomyLoop();
      }
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
            return json{{"ok", handled},
                        {"handled", handled},
                        {"action", spec.name},
                        {"reply", result.reply},
                        {"meta", result.meta}};
          });
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
  phoenix::conceptmatrix::ConceptMatrix conceptMatrix_;
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
      nlohmann::json observePayload;
      observePayload["sessionId"] = sessionId;
      observePayload["reply"] = userText;
      observePayload["autonomySource"] = "gateway-chat";
      observePayload["worldUncertainty"] = body.value("worldUncertainty", 0.25);
      nlohmann::json worldState;
      worldState["graphContext"] = baseContext;
      worldState["graphResult"] = body.value("graphResult", nlohmann::json::object());
      if (!sessionId.empty()) {
        auto prev = cognitionMemory_.get("cognition:" + sessionId);
        if (prev)
          worldState["previousCognition"] = *prev;
        if (worldModel_) {
          try {
            auto wmState = worldModel_->sessionState(sessionId, 8);
            if (wmState.value("ok", false)) {
              worldState["worldModelState"] = wmState;
              world_model::PromptContextOptions wmOpts;
              wmOpts.maxRecentEvidence = 3;
              wmOpts.maxSummaryChars = 220;
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
      iterPayload["sessionId"] = body.value("sessionId", "");
      iterPayload["contextTag"] =
          "chat:" + std::string(body.value("sessionId", "anonymous"));
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
      std::string mod = r.value("cognitionModulation", "");
      if (!mod.empty()) {
        if (pack.context.empty())
          pack.context = mod;
        else
          pack.context = mod + "\n" + pack.context;
        if (pack.instanceMemoryText.empty())
          pack.instanceMemoryText = mod;
        else
          pack.instanceMemoryText = mod + "\n" + pack.instanceMemoryText;
      }

      // Run graph-diffusion summarization over the provided graph result.
      phoenix::graph::GraphDiffusionSummarizer diffusor;
      auto diffSummary = diffusor.summarizeFromJson(worldState["graphResult"]);
      if (!diffSummary.rankedNodes.empty()) {
        if (!pack.worldEvidence.is_array())
          pack.worldEvidence = nlohmann::json::array();
        pack.worldEvidence.push_back(
            {{"stage", "graphDiffusion"}, {"summary", diffSummary.toJson()}});
        if (!diffSummary.summaryText.empty())
          cognitionManager_.setMissionGnnSummary(diffSummary.summaryText);
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
        cognitionMemory_.put("cognition:" + sessionId, packJson, activation);
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
