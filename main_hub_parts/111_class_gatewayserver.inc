// Auto-generated from main.cpp
// Segment: class_GatewayServer
// Original lines: 13037-13883

class GatewayServer : public IGatewayServer {
public:
  /* Forward decl only - definition is out-of-line after GatewayServer closes
     (see 116_section_tail.inc).  A second in-class definition would collide. */
  std::string buildMissionOutline(const std::string &goal);

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
                           const std::string &graphSummary) {
    if (!worldModel_)
      return;
    try {
      nlohmann::json evidence;
      evidence["sessionId"] = sessionId;
      evidence["modality"] = "text";
      evidence["text"] = text;
      evidence["graphSummary"] = graphSummary;
      evidence["metadata"] = nlohmann::json{{"source", "gateway-chat"}};
      /* v8.0 audit B: tag the running mission so prediction docs aggregate
         per goal (world evidence becomes goal-scoped observability). */
      try {
        auto ms = cognitionManager_.missionStatus();
        const auto &stats = ms.value("result", nlohmann::json::object()).value("stats", nlohmann::json::object());
        const auto &mission = stats.value("mission", nlohmann::json::object());
        if (mission.value("state", 0) == 1 && !mission.value("id", std::string()).empty())
          evidence["metadata"]["missionId"] = mission.value("id", std::string());
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
    int summaryTimeoutMs =
        phoenix::cfgOr<int>("tinyllama.summaryTimeoutMs", 5000);
    return chatWithExternalAdapter(
        "tinyllama", config_.tinyllamaBaseUrl, config_.tinyllamaModel,
        summaryTimeoutMs, text, graphContext, maxTokens, options);
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
      missionCfg["pressureTauSec"] = phoenix::resolveConfig<double>("mission.pressureTauSec", 1800.0, "AI_MISSION_PRESSURE_TAU_SEC");
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
           1) plain text            -> appended to deliverable.md (first minimal draft)
           2) {"action":write|append|read|list,"path","content"} -> workspace file op
           3) {"tool":"<action-name>","arguments":{...}}          -> registered AGI action
              (math/search/research/web/replicate/mcp.*) executed via the executor
           4) {"done":true,"summary":"..."} -> SELF-VERDICT: the task is complete
              (v8.3; the gate refuses it until deliverable.md is long enough)
         Tool/action results are appended to the deliverable so the NEXT tick
         sees them - the agent iteratively refines instead of looping. */
      cognitionManager_.setMissionDeliberator(
          [this](const std::string &goal, const std::string &deliverable,
                 int maxTokens, const std::string &scope) -> std::string {
            try {
              const std::string workspaceRoot = phoenix::resolveConfig<std::string>(
                  "mission.workspaceDir", std::string("./runtime_store/mission_workspace"),
                  "AI_MISSION_WORKSPACE");
              /* scope = workspace sub-path: missionId for the parent,
                 missionId/children/<childId> for a helper box. */
              const bool isChildBox = scope.find("/children/") != std::string::npos;
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

              /* Sliding-window pack: pinned summary (+ optional GNN) + recent
                 full text, sized to the user-selected ctx (4k/16k). */
              phoenix::context::PackOptions packOpt;
              packOpt.mode = "full_and_summary";
              packOpt.includeGnnSummary = false;
              packOpt.ctxTokens = phoenix::cfgOr<int>("llama_server.ctx_size", 4096);
              std::string gnnText;
              try {
                const auto copts = cognitionManager_.missionContextOptions();
                const auto &cr = copts.value("result", nlohmann::json::object());
                packOpt = phoenix::context::optionsFromJson(cr, packOpt);
                gnnText = cr.value("gnnSummary", std::string());
              } catch (...) {
              }
              if (packOpt.includeGnnSummary && gnnText.empty()) {
                /* lightweight stand-in: keyword-ish extractive of the goal
                   when no graph diffusion summary has been pushed yet */
                gnnText = "goal focus: " + goal;
              }
              const auto packed = phoenix::context::packContext(currentFile, gnnText, packOpt);
              const std::string packedBody = packed.packed;

              /* RDK llama prompt eval is ~1.5-2s/token.  Dumping every AGI
                 action (name+category+description) pushed the deliberator
                 prompt to ~780 tokens (~24 min) and exceeded the HTTP
                 timeout → llama cancel_tasks → empty deliverable.  Keep the
                 prompt tiny: names-only tool hint, capped pack/outline. */
              std::string toolHint = "math,search,research,web,replicate";
              try {
                const auto la = cognitionManager_.listAgiActions();
                const auto &arr = la.value("result", nlohmann::json::array());
                std::string names;
                int n = 0;
                for (const auto &spec : arr) {
                  if (!spec.is_object()) continue;
                  const std::string nm = spec.value("name", std::string());
                  if (nm.empty()) continue;
                  if (n++) names += ",";
                  names += nm;
                  if (n >= 16) break;
                }
                if (!names.empty()) toolHint = names;
              } catch (...) {
              }

              auto clip = [](std::string s, size_t maxChars) {
                if (s.size() <= maxChars) return s;
                return s.substr(s.size() - maxChars);
              };
              std::string goalClip = goal;
              if (goalClip.size() > 400) goalClip = goalClip.substr(0, 400);
              std::string packClip = clip(packedBody, 900);
              std::string listClip = listRes.dump();
              if (listClip.size() > 300) listClip = listClip.substr(0, 300) + "...";

              std::string prompt =
                  std::string("You are an autonomous agent working toward this goal:\n") +
                  goalClip + "\n\n";
              prompt += "Workspace files:\n" + listClip + "\n\n";
              if (!packClip.empty()) {
                prompt += "Context pack (tail, approxTokens=" +
                          std::to_string(packed.estimatedTokens) + "):\n" +
                          packClip + "\n\n";
              } else {
                prompt += "deliverable.md does not exist yet - first write a MINIMAL draft.\n\n";
              }
              prompt += "Tools (name only): " + toolHint + "\n\n";
              /* v8.3: stable OUTLINE from the long-lived GNN graph (parent
                 only - a box's sub-task is too small for a 2K outline) plus
                 the section headers already present, so chapters neither
                 duplicate nor get lost. */
              if (!isChildBox) {
                std::string outline = buildMissionOutline(goal);
                if (outline.size() > 700) outline.resize(700);
                if (!outline.empty())
                  prompt += outline + "\n\n";
              }
              {
                std::string sections;
                size_t pos = 0;
                while (pos < currentFile.size()) {
                  const auto eol = currentFile.find('\n', pos);
                  const std::string line = currentFile.substr(
                      pos, eol == std::string::npos ? std::string::npos
                                                    : eol - pos);
                  const auto b = line.find_first_not_of(" \t\r");
                  if (b != std::string::npos && line[b] == '#') {
                    if (sections.empty())
                      sections =
                          "Existing sections in deliverable.md (never rewrite "
                          "these - continue with the NEXT one):\n";
                    sections += line.substr(b) + "\n";
                  }
                  if (eol == std::string::npos)
                    break;
                  pos = eol + 1;
                }
                if (!sections.empty())
                  prompt += sections + "\n";
              }
              if (isChildBox) {
                prompt +=
                    "You are a HELPER BOX working on the sub-task above. You may summon your OWN helper boxes for bounded sub-parts (tool: replicate, arguments.userPrompt = the specific sub-task); keep every sub-task bounded and concrete. ";
              } else {
                /* the parent knows about its helper boxes and can READ + MERGE
                   their drafts - but it remains fully responsible for the
                   final deliverable and keeps working itself */
                try {
                  auto ms2 = cognitionManager_.missionStatus();
                  const auto &stats2 = ms2.value("result", nlohmann::json::object())
                                               .value("stats", nlohmann::json::object());
                  const auto &children2 = stats2.value("children", nlohmann::json::array());
                  if (children2.is_array() && !children2.empty()) {
                    prompt += "Helper boxes you spawned (read their files with the file tool and merge the best parts; you remain responsible for the final deliverable):\n";
                    for (const auto &box : children2) {
                      prompt += "- " + box.value("id", std::string()) + ": " +
                                box.value("goal", std::string()) +
                                " (file: children/" + box.value("id", std::string()) + "/deliverable.md)\n";
                    }
                    prompt += "\n";
                  }
                } catch (...) {
                }
              }
              prompt +=
                  "Reply with exactly ONE of these four forms:\n"
                  "1) Plain text - APPENDED to deliverable.md (use for the first minimal draft).\n"
                  "2) JSON {\"action\":\"write\"|\"append\"|\"read\"|\"list\",\"path\":\"...\",\"content\":\"...\"} - file operation in your sandboxed workspace (relative paths, no \"..\").\n"
                  "3) JSON {\"tool\":\"<tool-name>\",\"arguments\":{\"userPrompt\":\"...\"}} - invoke a tool; its result comes back to you next turn.\n"
                  "4) JSON {\"done\":true,\"summary\":\"...\"} - declare the task COMPLETE (ONLY after every outline item and section is actually written; a premature declaration is refused).\n"
                  "Iterate: minimal correct version first, then expand section by section following the OUTLINE strictly - advance item by item, never skip ahead, never repeat a finished section; use tools (search, math, replicate) when they help; never repeat text already in the file.";

              /* Mission ticks need a longer socket timeout than interactive
                 chat: even a slim prompt is slow on RDK CPU (~2s/token). */
              nlohmann::json opts = nlohmann::json{
                  {"temperature", 0.3},
                  {"timeoutMs", phoenix::cfgOr<int>("mission.llmTimeoutMs", 900000)}};
              if (prompt.size() > 3500)
                prompt = prompt.substr(0, 3500);
              const auto r = chatWithLlamaCpp(prompt, "", std::max(64, maxTokens), opts);
              if (!r.value("ok", false)) {
                const std::string err = r.value("error", std::string("llm failed"));
                return "[llm-error] " + err.substr(0, 240);
              }
              std::string reply = r.value("reply", std::string());
              if (reply.empty())
                return "[llm-error] empty reply (possible cancel/timeout)";

              /* structured reply: tool call or file action */
              {
                const auto trim = [](const std::string &s) -> std::string {
                  const auto b = s.find_first_not_of(" \t\r\n");
                  if (b == std::string::npos) return std::string();
                  const auto e = s.find_last_not_of(" \t\r\n");
                  return s.substr(b, e - b + 1);
                };
                const std::string trimmed = trim(reply);
                if (!trimmed.empty() && trimmed.front() == '{') {
                  try {
                    nlohmann::json act = nlohmann::json::parse(trimmed);
                    if (!act.is_object()) throw 0;
                    if (act.contains("tool") && act["tool"].is_string()) {
                      nlohmann::json args = act.value("arguments", nlohmann::json::object());
                      if (!args.is_object()) args = nlohmann::json::object();
                      std::string up = args.value("userPrompt", std::string());
                      if (up.empty()) up = args.value("query", std::string());
                      if (up.empty()) up = args.value("input", std::string());
                      if (up.empty()) up = args.dump();
                      nlohmann::json ctx = args;
                      ctx["userPrompt"] = up;
                      /* task-tree lineage: a helper box that spawns records
                         the child under ITSELF (parentId = this box id) */
                      if (isChildBox) {
                        const auto slash = scope.rfind('/');
                        if (slash != std::string::npos && slash + 1 < scope.size())
                          ctx["parentId"] = scope.substr(slash + 1);
                      }
                      const auto tr = cognitionManager_.executeAgiActionByName(
                          act["tool"].get<std::string>(), ctx);
                      std::string note = "[tool:" + act["tool"].get<std::string>();
                      if (tr.value("ok", false))
                        note += " ok " + tr.value("reply", std::string()).substr(0, 400);
                      else
                        note += " FAILED " + tr.value("error", std::string());
                      note += "]";
                      return note;
                    }
                    if (act.contains("action") && act["action"].is_string()) {
                      const auto opRes = phoenix::mission::workspaceExecute(
                          workspaceRoot, scope, act);
                      if (opRes.value("ok", false))
                        return "[file:" + act.value("action", std::string()) + " ok " +
                               std::to_string(opRes.value("bytes", 0)) + "B " +
                               act.value("path", std::string()) + "]";
                      return "[file:" + act.value("action", std::string()) + " FAILED " +
                             opRes.value("error", std::string()) + "]";
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
                      const int minChars = isChildBox
                          ? phoenix::resolveConfig<int>("mission.childMinDeliverableChars", 200)
                          : phoenix::resolveConfig<int>("mission.minDeliverableChars", 1200);
                      /* Prefer the longer of workspace file vs in-memory prior so
                         a stale/truncated file cannot gate-keep a real body. */
                      const size_t bodyChars =
                          std::max(currentFile.size(), deliverable.size());
                      if (static_cast<int>(bodyChars) < minChars) {
                        return "[auto-verify] REFUSED: completion declared too "
                               "early - deliverable.md is only " +
                               std::to_string(bodyChars) + " chars (need >= " +
                               std::to_string(minChars) +
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
                      cognitionManager_.reportMissionOutcome(
                          nlohmann::json{{"goalAchieved", true}});
                      return "[auto-complete] mission self-verdict: task "
                             "complete. " + summary;
                    }
                  } catch (...) {
                  }
                }
              }

              /* plain text: mirror into the workspace file AND return for
                 the in-memory deliverable (both stay in sync) */
              phoenix::mission::workspaceExecute(
                  workspaceRoot, scope,
                  nlohmann::json{{"action", "append"},
                                 {"path", deliverablePath},
                                 {"content", reply}});
              return reply;
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
      loopCfg["persistEveryTicks"] = phoenix::cfgOr<int>("autonomyLoop.persistEveryTicks", 5);
      loopCfg["persistPath"] = phoenix::cfgOr<std::string>(
          "autonomyLoop.persistPath", "runtime_store/autonomy_state.json");
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

    if (!config_.disableLearning) {
      try {
        std::vector<transformer::TrainSample> samples;
        samples.reserve(160);

        auto isGoodText = [](const std::string &text) {
          if (text.size() < 24)
            return false;
          int letters = 0;
          int digits = 0;
          int spaces = 0;
          int punct = 0;
          int controls = 0;
          int cjk = 0;
          for (unsigned char ch : text) {
            if (ch < 0x20 || ch == 0x7F) {
              controls++;
              continue;
            }
            if (std::isspace(ch)) {
              spaces++;
              continue;
            }
            if (std::isalpha(ch)) {
              letters++;
              continue;
            }
            if (std::isdigit(ch)) {
              digits++;
              continue;
            }
            if (std::ispunct(ch)) {
              punct++;
              continue;
            }
            if (ch >= 0x80)
              cjk++;
          }
          if (controls > 0)
            return false;
          int nonSpace = std::max(1, (int)text.size() - spaces);
          double punctRatio = (double)punct / (double)nonSpace;
          if (punctRatio > 0.45)
            return false;
          int semantic = letters + cjk;
          if (semantic < std::max(10, (int)text.size() / 5))
            return false;
          return true;
        };

        auto addSample = [&](const std::string &raw) {
          std::string text = trimCopy(raw);
          if (!isGoodText(text))
            return;
          if (text.size() > 320)
            text = text.substr(0, 320);
          samples.push_back(transformer::TrainSample{text, text, ""});
        };

        try {
          auto docs = pool_->getActive()->runtime()->collectRobotsDocuments(
              json{{"limit", 120}, {"shuffle", true}});
          for (auto &d : docs) {
            addSample(d.text);
            if (samples.size() >= 120)
              break;
          }
        } catch (...) {
        }

        try {
          if (fs::exists(config_.testsDir)) {
            for (auto &e : fs::directory_iterator(config_.testsDir)) {
              if (!e.is_regular_file())
                continue;
              auto ext = e.path().extension().string();
              std::string lowerExt;
              lowerExt.reserve(ext.size());
              for (char c : ext)
                lowerExt.push_back((char)std::tolower((unsigned char)c));
              if (lowerExt != ".txt")
                continue;
              std::string text = readTextFile(e.path());
              if (text.empty())
                continue;
              std::istringstream iss(text);
              std::string line;
              while (std::getline(iss, line)) {
                addSample(line);
                if (samples.size() >= 180)
                  break;
              }
              if (samples.size() >= 180)
                break;
            }
          }
        } catch (...) {
        }

        if (!samples.empty()) {
          int epochs =
              std::max(1, std::min(3, (int)(samples.size() >= 120 ? 2 : 1)));
          float lr = 0.0006f;
          std::lock_guard<std::mutex> lock(transformerMu_);
          transformer_.pretrain(samples, epochs, lr);
        }
      } catch (...) {
      }
    }
  }

  void bootstrapTransformerFromCorpus(int maxDocs = 48) override {
    if (config_.disableLearning)
      return;
    std::vector<transformer::TrainSample> samples;
    samples.reserve((size_t)std::max(8, maxDocs));
    auto isGoodText = [](const std::string &text) {
      if (text.size() < 24)
        return false;
      int letters = 0;
      int spaces = 0;
      int punct = 0;
      int controls = 0;
      int cjk = 0;
      for (unsigned char ch : text) {
        if (ch < 0x20 || ch == 0x7F) {
          controls++;
          continue;
        }
        if (std::isspace(ch)) {
          spaces++;
          continue;
        }
        if (std::isalpha(ch)) {
          letters++;
          continue;
        }
        if (std::ispunct(ch)) {
          punct++;
          continue;
        }
        if (ch >= 0x80)
          cjk++;
      }
      if (controls > 0)
        return false;
      int nonSpace = std::max(1, (int)text.size() - spaces);
      double punctRatio = (double)punct / (double)nonSpace;
      if (punctRatio > 0.45)
        return false;
      int semantic = letters + cjk;
      return semantic >= std::max(10, (int)text.size() / 6);
    };

    auto addSample = [&](const std::string &raw) {
      std::string text = trimCopy(raw);
      if (!isGoodText(text))
        return;
      if (text.size() > 280)
        text = text.substr(0, 280);
      samples.push_back(transformer::TrainSample{text, text, ""});
    };

    try {
      auto docs = pool_->getActive()->runtime()->collectRobotsDocuments(
          json{{"limit", std::max(8, maxDocs)}, {"shuffle", true}});
      for (auto &d : docs) {
        addSample(d.text);
        if ((int)samples.size() >= maxDocs)
          break;
      }
    } catch (...) {
    }

    if (samples.size() < 8)
      return;

    try {
      std::lock_guard<std::mutex> lock(transformerMu_);
      transformer_.pretrain(samples, 1, 0.0008f);
    } catch (...) {
    }
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
  /* Serialize llama-server calls (chat + mission deliberator).  With
     --parallel 1 on the RDK, concurrent clients cancel each other and the
     deliberator returns empty forever. */
  mutable std::mutex llamaCallMu_;
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
              pack.context += "\n[tool_result action=" + toolAction +
                              "] output=" + toolOut + " [/tool_result]";
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
        typeLower != "research" && typeLower != "web")
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
        prompt =
            std::string("Context:\n") + graphContext + "\n\nUser:\n" + text;
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
