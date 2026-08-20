/* Auto-generated from main.cpp
   Segment: section_before_SharedMemorySlice
   Original lines: 1061-1627
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

static Config loadConfig(int argc, char **argv) {
  const auto args = parseArgs(argc, argv);
  const auto rootDir = getProjectRoot();
  // Load the single authoritative runtime parameter file.
  phoenix::PhoenixConfig::instance().load(rootDir / "config" / "phoenix.json");

  // Load the optional auto-tuned override file.
  phoenix::tuned::loadTunedConfig(rootDir / "config" / "phoenix_tuned.json");
  auto arg = [&](const std::string &k,
                 const std::string &def = "") -> std::string {
    auto it = args.find(k);
    if (it != args.end())
      return it->second;
    return def;
  };
  auto argOrEnv = [&](const std::string &argKey, const std::string &env,
                      const std::string &def = "") {
    auto it = args.find(argKey);
    if (it != args.end() && !it->second.empty())
      return it->second;
    return phoenix::resolveConfigAsString(dotPathForEnv(env), def, env.c_str());
  };
  auto pathArgOrEnv = [&](const std::string &argKey, const std::string &env,
                          const fs::path &defPath) {
    std::string raw = argOrEnv(argKey, env, defPath.string());
    fs::path p = raw.empty() ? defPath : fs::path(raw);
    if (p.is_relative())
      p = rootDir / p;
    return fs::absolute(p);
  };

  // Load the v7.0 model deployment topology (LLM / vision / speech placement).
  phoenix::deployment::ModelDeploymentConfig::instance().load(
      args, rootDir.string(),
      argOrEnv("model-deployment-config", "AI_MODEL_DEPLOYMENT_CONFIG", ""));

  Config c;
  c.baseDir =
      pathArgOrEnv("base-dir", "AI_BASE_DIR", rootDir / "runtime_store");
  c.gatewayHost = argOrEnv("gateway-host", "AI_GATEWAY_HOST", "127.0.0.1");
  c.portGateway =
      (int)numberOrNaN(argOrEnv("port", "CONTROLLER_PORT", "5080"), 5080);
  c.portStudy =
      (int)numberOrNaN(argOrEnv("study-port", "AI_STUDY_PORT", "5081"), 5081);
  c.httpLog = boolFrom(argOrEnv("http-log", "AI_HTTP_LOG", "false"), false);

  auto aiCountRaw = argOrEnv("ai-count", "AI_COUNT",
                             phoenix::resolveConfigAsString("main.aiCount", "7", "AI_NUM"));
  auto groupCountRaw =
      argOrEnv("group-count", "AI_GROUP_COUNT",
               phoenix::resolveConfigAsString("main.groupCount", "3", "GROUP_COUNT"));
  auto groupSizeRaw =
      argOrEnv("group-size", "AI_GROUP_SIZE",
               phoenix::resolveConfigAsString("main.groupSize", aiCountRaw, "GROUP_SIZE"));
  auto sparkNumAiRaw =
      argOrEnv("spark-num-ai", "AI_SPARK_NUM_AI", groupSizeRaw);

  double aiCountNum = numberOr(aiCountRaw, 7);
  double groupCountNum = numberOr(groupCountRaw, 3);
  double groupSizeNum = numberOr(groupSizeRaw, numberOr(aiCountRaw, 7));
  double sparkNumAiNum =
      numberOr(sparkNumAiRaw, numberOr(groupSizeRaw, numberOr(aiCountRaw, 7)));

  c.aiCount = std::max(3, (int)aiCountNum);
  c.groupCount = std::max(1, (int)groupCountNum);
  c.groupSize = std::max(1, (int)groupSizeNum);
  c.sparkNumAI = std::max(1, (int)sparkNumAiNum);
  c.sparkBudget = argOrEnv("spark-budget", "AI_SPARK_BUDGET", "default");

  c.singleProc =
      boolFrom(argOrEnv("single-proc", "AI_SINGLE_PROC", "false"), false);
  c.groupProc =
      c.singleProc
          ? false
          : boolFrom(argOrEnv("group-proc", "AI_GROUP_PROC", "false"), false);
  c.groupProcTimeoutMs =
      std::max(500, (int)numberOr(argOrEnv("group-proc-timeout-ms",
                                           "AI_GROUP_PROC_TIMEOUT_MS", "12000"),
                                  12000));

  c.inferMP =
      c.singleProc
          ? false
          : boolFrom(argOrEnv("infer-mp", "AI_INFER_MP", "false"), false);
  int cpuCount = std::max(2u, std::thread::hardware_concurrency());
  int inferWorkersRaw = 0;
  inferWorkersRaw =
      (int)numberOr(argOrEnv("infer-workers", "AI_INFER_WORKERS", "0"), 0);
  c.inferWorkers =
      std::max(1, inferWorkersRaw ? inferWorkersRaw : (cpuCount - 1));

  c.redisConnectTimeoutMs = std::max(
      200, (int)numberOr(argOrEnv("redis-timeout-ms",
                                  "AI_REDIS_CONNECT_TIMEOUT_MS", "1500"),
                         1500));
  c.redisUrl = argOrEnv("redis-url", "REDIS_URL", "redis://127.0.0.1:6379");
  c.redisChannel =
      argOrEnv("channel", "AI_REDIS_CHANNEL", "AI-model-workspace");
  c.dbPath =
      pathArgOrEnv("db-path", "AI_DB_PATH", c.baseDir / "ai_store.sqlite");
  c.redisCacheDb = std::max(
      0,
      (int)numberOr(argOrEnv("redis-cache-db", "AI_REDIS_CACHE_DB", "1"), 1));
  c.redisCachePrefix =
      argOrEnv("redis-cache-prefix", "AI_REDIS_CACHE_PREFIX", "AI079");

  c.snapshotDir =
      pathArgOrEnv("snapshot-dir", "AI_SNAPSHOT_DIR", rootDir / "snapshots");
  c.workerFile = fs::absolute(rootDir / "memeMergeWorker.cjs");
  c.maxWorkers = std::max(1, (int)cpuCount - 1);
  c.shardCache = fs::absolute(rootDir / "shards_cache.json");

  c.lmdbRoot = pathArgOrEnv("lmdb-dir", "LMDB_DIR", rootDir / "lmdb");
  c.lmdbMapSizeBytes =
      std::max<std::size_t>(
          64, (std::size_t)numberOr(
                  argOrEnv("lmdb-map-mb", "AI_LMDB_MAP_MB", "4096"), 4096)) *
      1024ull * 1024ull;

  c.searchEndpoint = argOrEnv("search-endpoint", "AI_SEARCH_ENDPOINT", "");
  c.robotsDir = pathArgOrEnv("robots-dir", "AI_ROBOTS_DIR", rootDir / "robots");
  c.lemmaCsv = pathArgOrEnv("lemma-csv", "AI_LEMMA_CSV", rootDir / "lemma.csv");

  c.lemmaAutoload =
      boolFrom(argOrEnv("lemma-autoload", "AI_LEMMA_AUTOLOAD", "false"), false);
  c.lemmaMaxBytes =
      std::max<std::size_t>(
          1, (std::size_t)numberOr(
                 argOrEnv("lemma-max-mb", "AI_LEMMA_MAX_MB", "64"), 64)) *
      1024ull * 1024ull;
  c.lemmaForce =
      boolFrom(argOrEnv("lemma-force", "AI_LEMMA_FORCE", "false"), false);

  c.robotsWarmupLimit = std::max(
      0, (int)numberOr(argOrEnv("robots-limit", "AI_ROBOTS_LIMIT", "0"), 0));
  c.robotsAutoload =
      boolFrom(argOrEnv("robots-autoload", "AI_ROBOTS_AUTOLOAD", "false"), false);
  c.robotsWarmupShuffle = boolFrom(
      argOrEnv("robots-warmup-shuffle", "AI_ROBOTS_WARMUP_SHUFFLE", "false"),
      false);
  c.robotsChunkMinWords = std::max(
      1, (int)numberOr(argOrEnv("robots-chunk-min", "AI_ROBOTS_CHUNK_MIN", "3"),
                       2));
  c.robotsChunkMaxWords = std::max(
      c.robotsChunkMinWords,
      (int)numberOr(argOrEnv("robots-chunk-max", "AI_ROBOTS_CHUNK_MAX", "20"),
                    5));

  c.kvmCacheMaxEntries = std::max(
      0,
      (int)numberOr(argOrEnv("kvm-cache-max", "AI_KVM_CACHE_MAX", "50000"), 0));
  c.learningWarmup = boolFrom(
      argOrEnv("learning-warmup", "AI_LEARNING_WARMUP", "false"), false);
  c.syncStandbyOnBoot = boolFrom(
      argOrEnv("sync-standby", "AI_SYNC_STANDBY_ON_BOOT", "false"), false);
  c.testsAutoload =
      boolFrom(argOrEnv("tests-autoload", "AI_TESTS_AUTOLOAD", "false"), false);
  c.testsDir = fs::absolute(rootDir / "tests");

  c.disableBarrier = boolFrom(
      argOrEnv("disable-memebarrier", "AI_DISABLE_MEMEBARRIER", "false"),
      false);
  c.disableRL =
      boolFrom(argOrEnv("disable-rl", "AI_DISABLE_RL", "false"), false);
  c.disableADV =
      boolFrom(argOrEnv("disable-adv", "AI_DISABLE_ADV", "false"), false);
  c.disableLearning = boolFrom(
      argOrEnv("disable-learning", "AI_DISABLE_LEARNING", "false"), false);
  c.disableGnnModule = boolFrom(
      argOrEnv("disable-gnn-module", "AI_DISABLE_GNN_MODULE", "false"), false);
  c.disableContextModule = boolFrom(
      argOrEnv("disable-context-module", "AI_DISABLE_CONTEXT_MODULE", "false"),
      false);
  c.usingOllama =
      boolFrom(argOrEnv("using-ollama", "AI_USING_OLLAMA", "false"), false);
  c.ollamaFineTuning = boolFrom(
      argOrEnv("ollama-fine-tuning", "AI_OLLAMA_FINE_TUNING", "false"), false);
  c.inferenceEnabled = boolFrom(
      argOrEnv("inference-enabled", "AI_INFERENCE_ENABLED", "true"), true);

  {
    std::string explicitMode =
        trimCopy(argOrEnv("transformer-mode", "AI_TRANSFORMER_MODE", ""));
    std::transform(
        explicitMode.begin(), explicitMode.end(), explicitMode.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (explicitMode == "native") {
      c.inferenceEnabled = true;
      c.usingOllama = false;
      c.ollamaFineTuning = false;
      c.transformerBackend = "native";
    } else if (explicitMode == "ollama") {
      c.inferenceEnabled = true;
      c.usingOllama = true;
      c.ollamaFineTuning = false;
      c.transformerBackend = "ollama";
    } else if (explicitMode == "ollama-fine-tuning" ||
               explicitMode == "ollama_fine_tuning") {
      c.inferenceEnabled = true;
      c.usingOllama = true;
      c.ollamaFineTuning = true;
      c.transformerBackend = "ollama-fine-tuning";
    } else if (explicitMode == "llamacpp" || explicitMode == "llama.cpp" ||
               explicitMode == "llama_cpp") {
      c.inferenceEnabled = true;
      c.usingOllama = false;
      c.ollamaFineTuning = false;
      c.transformerBackend = "llamacpp";
    } else if (explicitMode == "bitnet") {
      c.inferenceEnabled = true;
      c.usingOllama = false;
      c.ollamaFineTuning = false;
      c.transformerBackend = "bitnet";
    } else if (explicitMode == "off" || explicitMode == "false" ||
               explicitMode == "disabled") {
      c.inferenceEnabled = false;
      c.transformerBackend = "off";
    } else {
      // No explicit transformer mode.  Default to a real backend; do NOT
      // silently fall back to the slow native Python transformer.
      if (c.inferenceEnabled) {
        c.usingOllama = false;
        c.ollamaFineTuning = false;
        c.transformerBackend = "llamacpp";
      } else {
        c.transformerBackend = "off";
      }
    }
  }

  c.ollamaModel = argOrEnv("ollama-model", "AI_OLLAMA_MODEL", "");
  c.ollamaBaseUrl = argOrEnv("ollama-base-url", "AI_OLLAMA_BASE_URL",
                             "http://127.0.0.1:11434");
  c.ollamaRequestTimeoutMs =
      std::max(1000, (int)numberOr(argOrEnv("ollama-timeout-ms",
                                            "AI_OLLAMA_TIMEOUT_MS", "120000"),
                                   120000));
  c.llamaCppModel = argOrEnv("llamacpp-model", "AI_LLAMACPP_MODEL", "");
  c.llamaCppBaseUrl = argOrEnv("llamacpp-base-url", "AI_LLAMACPP_BASE_URL",
                               "http://127.0.0.1:8080");
  c.llamaCppRequestTimeoutMs =
      std::max(1000, (int)numberOr(argOrEnv("llamacpp-timeout-ms",
                                            "AI_LLAMACPP_TIMEOUT_MS", "120000"),
                                   120000));
  c.llamaCppLoraFiles = argOrEnv("llamacpp-lora-files", "AI_LLAMACPP_LORA_FILES", "");
  c.llamaCppLoraInitWithoutApply = boolFrom(
      argOrEnv("llamacpp-lora-init-without-apply", "AI_LLAMACPP_LORA_INIT_WITHOUT_APPLY", "false"), false);
  c.llamaCppFineTuning = boolFrom(
      argOrEnv("llamacpp-fine-tuning", "AI_LLAMACPP_FINE_TUNING", "false"), false);
  c.bitnetModel = argOrEnv("bitnet-model", "AI_BITNET_MODEL", "");
  c.bitnetBaseUrl = argOrEnv("bitnet-base-url", "AI_BITNET_BASE_URL",
                             "http://127.0.0.1:8090");
  c.bitnetRequestTimeoutMs =
      std::max(1000, (int)numberOr(argOrEnv("bitnet-timeout-ms",
                                            "AI_BITNET_TIMEOUT_MS", "120000"),
                                   120000));
  c.tinyllamaModel = argOrEnv("tinyllama-model", "AI_TINYLLAMA_MODEL", "");
  c.tinyllamaBaseUrl = argOrEnv("tinyllama-base-url", "AI_TINYLLAMA_BASE_URL",
                                "http://127.0.0.1:8085");
  c.tinyllamaRequestTimeoutMs =
      std::max(1000, (int)numberOr(argOrEnv("tinyllama-timeout-ms",
                                            "AI_TINYLLAMA_TIMEOUT_MS", "120000"),
                                   120000));
  c.tinyllamaEnabled = boolFrom(
      argOrEnv("tinyllama-enabled", "AI_TINYLLAMA_ENABLED", "false"), false);
  c.conceptMatrixEnabled = boolFrom(
      argOrEnv("concept-matrix-enabled", "AI_CONCEPT_MATRIX_ENABLED", "false"), false);
  c.conceptMatrixClearPerRequest = boolFrom(
      argOrEnv("concept-matrix-clear-per-request", "AI_CONCEPT_MATRIX_CLEAR_PER_REQUEST", "true"), true);
  c.ggufModelsDir = pathArgOrEnv("gguf-models-dir", "AI_GGUF_MODELS_DIR",
                                 rootDir / "GGUF_models");
  {
    std::string styleSimRaw =
        argOrEnv("external-style-sim", "AI_EXTERNAL_STYLE_SIM", "");
    if (styleSimRaw.empty()) {
      styleSimRaw = argOrEnv("external-backend-style-sim",
                             "AI_EXTERNAL_BACKEND_STYLE_SIM", "false");
    }
    c.externalBackendStyleSim = boolFrom(styleSimRaw, false);
  }

  // Apply v7.0 remote LLM deployment overrides.  If the user asked for an
  // edge LLM, the backend, base URL and model name are taken from the
  // deployment record after the legacy flags, so the deployment config wins.
  const auto &llmDeployment =
      phoenix::deployment::ModelDeploymentConfig::instance().llm();
  const bool llmRemote =
      (llmDeployment.placement == phoenix::deployment::ModelPlacement::Remote ||
       llmDeployment.placement == phoenix::deployment::ModelPlacement::Auto) &&
      !llmDeployment.remote.url.empty();
  if (llmRemote) {
    std::string method = llmDeployment.remote.method;
    if (method != "ollama" && method != "llamacpp" && method != "bitnet")
      method = "llamacpp";
    c.inferenceEnabled = true;
    c.transformerBackend = method;
    if (method == "ollama") {
      c.usingOllama = true;
      c.ollamaFineTuning = false;
      c.ollamaBaseUrl = llmDeployment.remote.url;
      if (!llmDeployment.remote.modelName.empty())
        c.ollamaModel = llmDeployment.remote.modelName;
      if (llmDeployment.remote.timeoutMs > 0)
        c.ollamaRequestTimeoutMs = llmDeployment.remote.timeoutMs;
    } else if (method == "llamacpp") {
      c.usingOllama = false;
      c.ollamaFineTuning = false;
      c.llamaCppBaseUrl = llmDeployment.remote.url;
      if (!llmDeployment.remote.modelName.empty())
        c.llamaCppModel = llmDeployment.remote.modelName;
      if (llmDeployment.remote.timeoutMs > 0)
        c.llamaCppRequestTimeoutMs = llmDeployment.remote.timeoutMs;
    } else if (method == "bitnet") {
      c.usingOllama = false;
      c.ollamaFineTuning = false;
      c.bitnetBaseUrl = llmDeployment.remote.url;
      if (!llmDeployment.remote.modelName.empty())
        c.bitnetModel = llmDeployment.remote.modelName;
      if (llmDeployment.remote.timeoutMs > 0)
        c.bitnetRequestTimeoutMs = llmDeployment.remote.timeoutMs;
    }
  }

  {
    std::error_code ec;
    fs::create_directories(c.ggufModelsDir, ec);

    auto collectGgufFiles = [&](const fs::path &dir) {
      std::vector<fs::path> files;
      std::error_code iterEc;
      if (!fs::exists(dir, iterEc) || !fs::is_directory(dir, iterEc))
        return files;
      fs::recursive_directory_iterator it(
          dir, fs::directory_options::skip_permission_denied, iterEc);
      fs::recursive_directory_iterator end;
      while (!iterEc && it != end) {
        const auto &entry = *it;
        std::error_code fileEc;
        if (entry.is_regular_file(fileEc)) {
          std::string ext = entry.path().extension().string();
          std::transform(ext.begin(), ext.end(), ext.begin(),
                         [](unsigned char ch) {
                           return static_cast<char>(std::tolower(ch));
                         });
          if (ext == ".gguf")
            files.push_back(fs::absolute(entry.path()));
        }
        it.increment(iterEc);
      }
      std::sort(files.begin(), files.end());
      return files;
    };

    auto ensureGgufModelForBackend = [&](const std::string &backend,
                                         const std::string &flagName,
                                         std::string &modelField) {
      auto candidates = collectGgufFiles(c.ggufModelsDir);
      std::string configured = trimCopy(modelField);
      if (configured.empty()) {
        if (candidates.size() == 1) {
          modelField = candidates.front().string();
          return;
        }
        if (candidates.empty()) {
          throw std::runtime_error("transformer-mode=" + backend +
                                   " requires a .gguf model. Put model into " +
                                   c.ggufModelsDir.string() + " or pass --" +
                                   flagName);
        }
        std::ostringstream oss;
        oss << "transformer-mode=" << backend
            << " found multiple .gguf models in " << c.ggufModelsDir.string()
            << ". please specify --" << flagName << ". candidates=";
        for (size_t i = 0; i < candidates.size(); i++) {
          if (i)
            oss << ",";
          oss << candidates[i].filename().string();
        }
        throw std::runtime_error(oss.str());
      }

      fs::path modelPath = fs::path(configured);
      if (modelPath.is_relative()) {
        fs::path underGguf = c.ggufModelsDir / modelPath;
        std::error_code existsEc;
        if (fs::exists(underGguf, existsEc))
          modelPath = underGguf;
      }
      modelPath = fs::absolute(modelPath);

      std::error_code fileEc;
      if (!fs::exists(modelPath, fileEc) ||
          !fs::is_regular_file(modelPath, fileEc))
        throw std::runtime_error("invalid " + backend +
                                 " model path: " + modelPath.string());

      std::string ext = modelPath.extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
      });
      if (ext != ".gguf" && !ext.empty())
        throw std::runtime_error(
            backend + " model must be a .gguf file or an extensionless raw blob: " + modelPath.string());

      modelField = modelPath.string();
    };

    if (c.transformerBackend == "llamacpp") {
      // When Phoenix on X5 talks to a host-side llama-server, there is no .gguf on the device.
      if (isRemoteLlamacppUrl(c.llamaCppBaseUrl)) {
        std::string configured = trimCopy(c.llamaCppModel);
        if (configured.empty())
          configured = "llamacpp";
        c.llamaCppModel = configured;
      } else {
        ensureGgufModelForBackend("llamacpp", "llamacpp-model", c.llamaCppModel);
      }
    } else if (c.transformerBackend == "bitnet") {
      ensureGgufModelForBackend("bitnet", "bitnet-model", c.bitnetModel);
    }
  }

  c.transformerBootstrap = boolFrom(
      argOrEnv("transformer-bootstrap", "AI_TRANSFORMER_BOOTSTRAP", "false"),
      false);
  c.transformerBootstrapDocs = std::max(
      16, (int)numberOr(argOrEnv("transformer-bootstrap-docs",
                                 "AI_TRANSFORMER_BOOTSTRAP_DOCS", "256"),
                        256));

  c.logMode = argOrEnv("log-mode", "AI_LOG_MODE", "release");
  c.exportDir =
      pathArgOrEnv("export-dir", "AI_EXPORT_DIR", rootDir / "runtime_store");
  c.tunedConfigPath = pathArgOrEnv(
      "tuned-config", "PHOENIX_TUNED_CONFIG",
      rootDir / "config" / "phoenix_tuned.json");
  phoenix::tuned::loadTunedConfig(c.tunedConfigPath);

  return c;
}

static void applyFrontendEnvFromConfig(const Config &config) {
  setRuntimeEnvValue("FRONTEND_PORT", std::to_string(config.portStudy));
  std::string gatewayHostForApi = config.gatewayHost;
  // Same rule as applyFrontendEnvFromArgs: a wildcard BIND address is not a
  // connectable destination for the same-machine frontend proxy.
  if (gatewayHostForApi.empty() || gatewayHostForApi == "0.0.0.0" ||
      gatewayHostForApi == "::" || gatewayHostForApi == "*")
    gatewayHostForApi = "127.0.0.1";
  setRuntimeEnvValue("AI_API_BASE", std::string("http://") +
                                        gatewayHostForApi + ":" +
                                        std::to_string(config.portGateway));
  std::string gatewayJwtSecret = phoenix::resolveConfig<std::string>(
      "auth.jwtSecret", "dev-secret-change-me", "AI_AUTH_JWT_SECRET",
      "AUTH_JWT_SECRET");
  if (!gatewayJwtSecret.empty())
    setRuntimeEnvValue("JWT_SECRET", gatewayJwtSecret);
}

// ------------------ Utilities ------------------

static std::string nowIso() {
  auto now = std::chrono::system_clock::now();
  auto t = std::chrono::system_clock::to_time_t(now);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) %
            1000;
  std::tm tm{};
#ifdef _WIN32
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  char buf[64];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
  std::ostringstream oss;
  oss << buf << '.' << std::setfill('0') << std::setw(3) << ms.count() << 'Z';
  return oss.str();
}

static int64_t nowEpochMs() {
  return (int64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

static json getLoadAvg() {
  json out = json::array();
#ifdef _WIN32
  out.push_back(0.0);
  out.push_back(0.0);
  out.push_back(0.0);
#else
  double loads[3] = {0.0, 0.0, 0.0};
  if (getloadavg(loads, 3) != -1) {
    out.push_back(loads[0]);
    out.push_back(loads[1]);
    out.push_back(loads[2]);
  } else {
    out.push_back(0.0);
    out.push_back(0.0);
    out.push_back(0.0);
  }
#endif
  return out;
}

static json getMemoryUsage() {
  json mem{{"rss", 0},
           {"heapTotal", 0},
           {"heapUsed", 0},
           {"external", 0},
           {"arrayBuffers", 0}};
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS_EX pmc{};
  if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&pmc,
                           sizeof(pmc))) {
    mem["rss"] = (int64_t)pmc.WorkingSetSize;
    mem["heapTotal"] = (int64_t)pmc.PrivateUsage;
    mem["heapUsed"] = (int64_t)pmc.PrivateUsage;
    mem["external"] = 0;
    mem["arrayBuffers"] = 0;
  }
#else
  long pageSize = sysconf(_SC_PAGESIZE);
  std::ifstream statm("/proc/self/statm");
  long size = 0, resident = 0;
  if (statm >> size >> resident) {
    int64_t rss = (int64_t)resident * pageSize;
    mem["rss"] = rss;
    mem["heapTotal"] = rss;
    mem["heapUsed"] = rss;
    mem["external"] = 0;
    mem["arrayBuffers"] = 0;
  }
#endif
  return mem;
}

static double getProcessCpuPercent() {
#ifdef _WIN32
  auto toU64 = [](const FILETIME &ft) -> uint64_t {
    ULARGE_INTEGER u{};
    u.LowPart = ft.dwLowDateTime;
    u.HighPart = ft.dwHighDateTime;
    return u.QuadPart;
  };

  FILETIME creation{}, exit{}, kernel{}, user{}, now{};
  if (!GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user))
    return 0.0;
  GetSystemTimeAsFileTime(&now);

  static std::mutex cpuMu;
  static uint64_t lastWall = 0;
  static uint64_t lastProc = 0;
  std::lock_guard<std::mutex> lock(cpuMu);

  uint64_t wall = toU64(now);
  uint64_t proc = toU64(kernel) + toU64(user);
  if (lastWall == 0 || wall <= lastWall || proc < lastProc) {
    lastWall = wall;
    lastProc = proc;
    return 0.0;
  }
  uint64_t dWall = wall - lastWall;
  uint64_t dProc = proc - lastProc;
  lastWall = wall;
  lastProc = proc;

  const double cpuCount =
      (double)std::max(1u, std::thread::hardware_concurrency());
  double pct = (double)dProc / ((double)dWall * cpuCount) * 100.0;
  if (!std::isfinite(pct))
    return 0.0;
  return std::max(0.0, std::min(100.0, pct));
#else
  static std::mutex cpuMu;
  static auto lastWall = std::chrono::steady_clock::now();
  static clock_t lastCpu = std::clock();
  std::lock_guard<std::mutex> lock(cpuMu);
  auto now = std::chrono::steady_clock::now();
  clock_t cpuNow = std::clock();
  auto wallMs =
      std::chrono::duration_cast<std::chrono::milliseconds>(now - lastWall)
          .count();
  if (wallMs <= 0)
    return 0.0;
  double cpuMs = (double)(cpuNow - lastCpu) * 1000.0 / (double)CLOCKS_PER_SEC;
  lastWall = now;
  lastCpu = cpuNow;
  const double cpuCount =
      (double)std::max(1u, std::thread::hardware_concurrency());
  double pct = cpuMs / ((double)wallMs * cpuCount) * 100.0;
  if (!std::isfinite(pct))
    return 0.0;
  return std::max(0.0, std::min(100.0, pct));
#endif
}

static std::string sha1Hex(const std::string &s) {
  auto digest = trantor::utils::sha1(s);
  return trantor::utils::toHexString(digest);
}

static std::string randomHex(size_t bytes) {
  std::random_device rd;
  std::uniform_int_distribution<int> dist(0, 255);
  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (size_t i = 0; i < bytes; i++) {
    int v = dist(rd);
    oss << std::setw(2) << (v & 0xff);
  }
  return oss.str();
}

static std::string randomSessionId() { return randomHex(8); }

static uint32_t hashStrSimple(const std::string &s,
                              uint32_t seed = 1315423911u) {
  uint32_t hash = seed;
  for (unsigned char c : s) {
    hash ^= ((hash << 5) + c + (hash >> 2));
  }
  return hash;
}

static std::string readTextFile(const fs::path &file) {
  std::ifstream in(file, std::ios::binary);
  if (!in)
    return {};
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

#ifdef _WIN32
using SocketHandle = SOCKET;
static constexpr SocketHandle kInvalidSocket = INVALID_SOCKET;
static void closeSocket(SocketHandle s) {
  if (s != INVALID_SOCKET)
    closesocket(s);
}
#else
using SocketHandle = int;
static constexpr SocketHandle kInvalidSocket = -1;
static void closeSocket(SocketHandle s) {
  if (s >= 0)
    close(s);
}
#endif

static bool sendAll(SocketHandle sock, const std::string &data) {
  const char *buf = data.c_str();
  size_t total = 0;
  while (total < data.size()) {
#ifdef _WIN32
    int sent = ::send(sock, buf + total, (int)(data.size() - total), 0);
#else
    ssize_t sent = ::send(sock, buf + total, data.size() - total, 0);
#endif
    if (sent <= 0)
      return false;
    total += (size_t)sent;
  }
  return true;
}

// SharedMemorySlice 表示共享内存中的一段偏移与大小信息。
// 调用方式：在跨进程数据交换时作为切片描述符传递。
// 实现思路：记录起始偏移和长度并附带标识字段。
// 注意事项：偏移与长度需由调用方确保边界合法。
// 注意事项：该结构不负责共享内存生命周期管理。
// 注意事项：跨平台使用时需关注地址和对齐约束。
