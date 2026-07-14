// Auto-generated from main.cpp
// Segment: class_GatewayServer
// Original lines: 13037-13883

class GatewayServer : public IGatewayServer {
public:
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
    {
      std::string authFlag = getEnv("AI_AUTH_ENABLED", "true");
      std::transform(authFlag.begin(), authFlag.end(), authFlag.begin(),
                     ::tolower);
      authEnabled_ = (authFlag != "false");
    }
    acceptLocalToken_ =
        boolFrom(getEnv("AI_AUTH_ACCEPT_LOCAL_TOKEN", "true"), true);
    jwtSecret_ = getEnv("AI_AUTH_JWT_SECRET",
                        getEnv("AUTH_JWT_SECRET", "dev-secret-change-me"));
    rlDisabled_ = config_.disableLearning || config_.disableRL;
    advDisabled_ = config_.disableLearning || config_.disableADV;
    gnnGaDisabled_ = config_.disableLearning;
    dialogLearningEnabled_ = !(config_.disableLearning == true);
    dialogTransformerLearning_ =
        boolFrom(getEnv("AI_DIALOG_TRANSFORMER_LEARNING", "false"), false);
    dialogLearnGraphContext_ =
        boolFrom(getEnv("AI_DIALOG_LEARN_GRAPH_CONTEXT", "false"), false);
    retrainTrace_ = boolFrom(getEnv("AI_RETRAIN_TRACE", "false"), false);
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
    setupRoutes();
  }

  void listen() override {
    unsigned hw = std::thread::hardware_concurrency();
    unsigned defaultThreads = hw > 0 ? std::max(4u, std::min(16u, hw)) : 4u;
    int threadNum = static_cast<int>(
        numberOr(getEnv("AI_HTTP_THREADS", std::to_string(defaultThreads)),
                 static_cast<long long>(defaultThreads)));
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
  std::shared_ptr<IReinforcementLearner> rl_;
  std::shared_ptr<IAdversarialLearner> adv_;
  std::shared_ptr<IGnnGaLearner> gnnGa_;
  transformer::TransformerService transformer_;
  model_lifecycle::ModelLifecycleManager modelLifecycle_;
  autonomy::TransformerClusterManager clusterManager_;
  autonomy::SpiderAutonomyManager spiderAutonomy_;
  autonomy::OptimizerAutonomyManager optimizerAutonomy_;
  autonomy::DatasetCatalogManager datasetCatalog_;
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
    if (trimmed.size() > dataCleanMaxChars_) {
      trimmed.resize(dataCleanMaxChars_);
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
                      int maxTokens) {
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
      json payload = {
          {"model", selectedModel},
          {"stream", false},
          {"messages",
           json::array({json{{"role", "user"}, {"content", prompt}}})},
          {"options", json{{"num_predict", std::max(16, maxTokens)}}}};

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
                               const std::string &graphContext, int maxTokens) {
    json out = json::object();
