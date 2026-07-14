// Auto-generated from main.cpp
// Segment: class_GroupProcessPool
// Original lines: 10470-10875

class GroupProcessPool : public ControllerPoolBase {
public:
  GroupProcessPool(std::shared_ptr<KeyValueStore> kvmStore,
                   std::shared_ptr<KeyValueStore> memeStore,
                   std::shared_ptr<KeyValueStore> sessionStore,
                   const Config &config)
      : config_(config), kvmStore_(std::move(kvmStore)),
        memeStore_(std::move(memeStore)),
        sessionStore_(std::move(sessionStore)) {
    for (int i = 0; i < config_.groupCount; i++) {
      std::string gid = "G" + std::to_string(i + 1);
      groupIds_.push_back(gid);
      std::vector<std::string> names;
      for (int j = 0; j < config_.groupSize; j++) {
        std::string name = gid + "_AI" + std::to_string(j + 1);
        names.push_back(name);
        controllerToGroup_[name] = gid;
        controllers_[name] = std::make_shared<ProxyController>(this, name);
      }
      groups_[gid] = names;
    }
    auto g1 = groupIds_.empty() ? "G1" : groupIds_[0];
    auto list = groups_[g1];
    servingName_ = list.empty() ? "" : list[0];
    standbyName_ = list.size() > 1 ? list[1] : servingName_;
    validationName_ = list.size() > 2 ? list[2] : servingName_;
    spawnWorkers();
  }

  ~GroupProcessPool() override { shutdownWorkers(); }

  std::shared_ptr<Controller> getActive() override {
    return getByName(servingName_);
  }
  std::shared_ptr<Controller> getStandby() override {
    return getByName(standbyName_);
  }
  std::shared_ptr<Controller> getValidation() override {
    return getByName(validationName_);
  }

  std::shared_ptr<Controller> getByName(const std::string &name) override {
    auto it = controllers_.find(name);
    return it == controllers_.end() ? nullptr : it->second;
  }

  std::vector<std::string> listControllerNames() const override {
    std::vector<std::string> out;
    for (auto &gid : groupIds_) {
      auto it = groups_.find(gid);
      if (it == groups_.end())
        continue;
      for (auto &name : it->second)
        out.push_back(name);
    }
    return out;
  }

  std::vector<std::string> listGroupIds() const override { return groupIds_; }

  std::vector<std::string>
  listControllersInGroup(const std::string &gid) override {
    return groups_.count(gid) ? groups_.at(gid) : std::vector<std::string>{};
  }

  std::vector<json> listMetrics() const override {
    std::vector<json> out;
    for (auto &gid : groupIds_) {
      try {
        auto result = rpc(gid, json{{"cmd", "listMetrics"}});
        if (result.is_array()) {
          for (auto &item : result)
            out.push_back(item);
        }
      } catch (...) {
        // ignore failed groups
      }
    }
    return out;
  }

  json ingestDocument(const json &doc) override {
    json results = json::array();
    for (auto &gid : groupIds_) {
      auto part = rpc(gid, json{{"cmd", "ingestDocument"}, {"doc", doc}});
      if (part.is_array()) {
        for (auto &item : part)
          results.push_back(item);
      } else {
        results.push_back(part);
      }
    }
    return results;
  }

  json ingestDocumentToGroup(const std::string &gid, const json &doc) override {
    if (!groups_.count(gid))
      return json{{"ok", false}, {"reason", "group-not-found"}};
    return rpc(gid, json{{"cmd", "ingestDocumentToGroup"}, {"doc", doc}});
  }

  json forgetMemes(const json &payload) override {
    json results = json::array();
    for (auto &gid : groupIds_) {
      auto part = rpc(gid, json{{"cmd", "forgetMemes"}, {"criteria", payload}});
      if (part.is_array()) {
        for (auto &item : part)
          results.push_back(item);
      } else {
        results.push_back(part);
      }
    }
    return results;
  }

  json onlineResearch(const json &queryOrTokens, const json &options) override {
    auto ctrl = getActive();
    if (!ctrl)
      throw std::runtime_error("no-active-controller");
    return rpc(controllerToGroup_[ctrl->name()],
               json{{"cmd", "runtimeCall"},
                    {"controllerName", ctrl->name()},
                    {"method", "onlineLookup"},
                    {"args", json::array({queryOrTokens, options})}});
  }

  void hotSwap(const json &snapshot) override {
    for (auto &gid : groupIds_) {
      rpc(gid, json{{"cmd", "applySnapshotAll"}, {"snapshot", snapshot}});
    }
    std::swap(servingName_, standbyName_);
  }

  void applySnapshotAll(const json &snapshot) {
    for (auto &gid : groupIds_) {
      rpc(gid, json{{"cmd", "applySnapshotAll"}, {"snapshot", snapshot}});
    }
  }

  const Config &config() const override { return config_; }

  json respondByName(const std::string &controllerName,
                     const json &payload) override {
    auto gid = controllerToGroup_.count(controllerName)
                   ? controllerToGroup_[controllerName]
                   : std::string();
    if (gid.empty())
      throw std::runtime_error("group-not-found");
    return rpc(gid, json{{"cmd", "respond"},
                         {"controllerName", controllerName},
                         {"payload", payload}});
  }

  json rpc(const std::string &groupId, const json &payload) const {
    auto it = workers_.find(groupId);
    if (it == workers_.end())
      throw std::runtime_error("group-worker-unavailable:" + groupId);
    auto &worker = *it->second;
    std::lock_guard<std::mutex> lock(worker.mu);
    json msg = payload;
    msg["id"] = std::to_string(worker.seq++);
    msg["shmProtocolVersion"] = kSharedMemoryProtocolVersion;
    msg["transport"] = "socket-msgpack-shm2";
    msg["backendHint"] = config_.transformerBackend;
    std::vector<uint8_t> requestBytes = json::to_msgpack(msg);
    auto attempt = [&](bool reconnect) -> std::optional<json> {
      if (reconnect) {
        closeSocket(worker.sock);
        worker.sock = connectWorker(worker.port);
      }
      if (worker.sock == kInvalidSocket)
        worker.sock = connectWorker(worker.port);
      if (worker.sock == kInvalidSocket ||
          !sendPacket(worker.sock, requestBytes))
        return std::nullopt;

      std::vector<uint8_t> replyBytes;
      if (!recvPacket(worker.sock, replyBytes, config_.groupProcTimeoutMs))
        return std::nullopt;

      json reply;
      try {
        reply = json::from_msgpack(replyBytes);
      } catch (...) {
        return std::nullopt;
      }
      if (reply.contains("id") && msg.contains("id") &&
          reply["id"].is_string() && msg["id"].is_string()) {
        if (reply["id"].get<std::string>() != msg["id"].get<std::string>())
          return std::nullopt;
      }
      if (reply.contains("shmProtocolVersion") &&
          reply["shmProtocolVersion"].is_number_integer()) {
        int proto = reply["shmProtocolVersion"].get<int>();
        if (proto <= 0 || proto > kSharedMemoryProtocolVersion)
          return std::nullopt;
      }
      return reply;
    };

    auto first = attempt(false);
    auto reply = first ? first : attempt(true);
    if (!reply) {
      std::string cmd = payload.value("cmd", "");
      throw std::runtime_error("group-worker-timeout:" + groupId + ":" + cmd);
    }
    if ((*reply).value("ok", false))
      return (*reply).value("result", json::object());
    std::string err = (*reply).value("error", "");
    if (err.empty())
      err = "group-worker-error:" + groupId;
    throw std::runtime_error(err);
  }

  json runtimeCall(const std::string &controllerName, const std::string &method,
                   const json &args) const {
    auto gid = controllerToGroup_.count(controllerName)
                   ? controllerToGroup_.at(controllerName)
                   : std::string();
    if (gid.empty())
      throw std::runtime_error("group-not-found");
    return rpc(gid, json{{"cmd", "runtimeCall"},
                         {"controllerName", controllerName},
                         {"method", method},
                         {"args", args}});
  }

  json memebarrierStart(double threshold) const {
    auto gid = controllerToGroup_.count(servingName_)
                   ? controllerToGroup_.at(servingName_)
                   : std::string();
    if (gid.empty())
      throw std::runtime_error("group-not-found");
    return rpc(gid, json{{"cmd", "memebarrierStart"},
                         {"maliciousThreshold", threshold}});
  }

  json memebarrierStop() const {
    auto gid = controllerToGroup_.count(servingName_)
                   ? controllerToGroup_.at(servingName_)
                   : std::string();
    if (gid.empty())
      throw std::runtime_error("group-not-found");
    return rpc(gid, json{{"cmd", "memebarrierStop"}});
  }

  json memebarrierStats() const {
    auto gid = controllerToGroup_.count(servingName_)
                   ? controllerToGroup_.at(servingName_)
                   : std::string();
    if (gid.empty())
      throw std::runtime_error("group-not-found");
    return rpc(gid, json{{"cmd", "memebarrierStats"}});
  }

  json memebarrierSetThreshold(double threshold) const {
    auto gid = controllerToGroup_.count(servingName_)
                   ? controllerToGroup_.at(servingName_)
                   : std::string();
    if (gid.empty())
      throw std::runtime_error("group-not-found");
    return rpc(gid, json{{"cmd", "memebarrierSetThreshold"},
                         {"maliciousThreshold", threshold}});
  }

  std::string groupForController(const std::string &name) const {
    auto it = controllerToGroup_.find(name);
    return it == controllerToGroup_.end() ? std::string() : it->second;
  }

private:
  struct WorkerInfo {
    int port{0};
    mutable SocketHandle sock{kInvalidSocket};
    mutable std::mutex mu;
    mutable uint64_t seq{1};
#ifdef _WIN32
    PROCESS_INFORMATION proc{0};
#else
    pid_t pid{-1};
#endif
  };

  Config config_;
  std::shared_ptr<KeyValueStore> kvmStore_;
  std::shared_ptr<KeyValueStore> memeStore_;
  std::shared_ptr<KeyValueStore> sessionStore_;
  std::vector<std::string> groupIds_;
  std::unordered_map<std::string, std::vector<std::string>> groups_;
  std::unordered_map<std::string, std::string> controllerToGroup_;
  std::unordered_map<std::string, std::shared_ptr<Controller>> controllers_;
  mutable std::unordered_map<std::string, std::shared_ptr<WorkerInfo>> workers_;
  std::string servingName_;
  std::string standbyName_;
  std::string validationName_;

  void spawnWorkers() {
    int basePort = std::max(1, config_.portStudy + 100);
    for (size_t i = 0; i < groupIds_.size(); i++) {
      std::string gid = groupIds_[i];
      int port = basePort + (int)i;
      auto info = std::make_shared<WorkerInfo>();
      info->port = port;
      spawnWorkerProcess(gid, port, *info);
      info->sock = connectWorker(port);
      workers_[gid] = info;
    }
  }

  void shutdownWorkers() {
    for (auto &kv : workers_) {
      if (!kv.second)
        continue;
      auto &worker = *kv.second;
      closeSocket(worker.sock);
#ifdef _WIN32
      if (worker.proc.hProcess) {
        TerminateProcess(worker.proc.hProcess, 0);
        CloseHandle(worker.proc.hProcess);
        CloseHandle(worker.proc.hThread);
      }
#else
      if (worker.pid > 0) {
        kill(worker.pid, SIGTERM);
      }
#endif
    }
  }

  void spawnWorkerProcess(const std::string &groupId, int port,
                          WorkerInfo &info) const {
    std::string portStr = std::to_string(port);
    std::string groupSize = std::to_string(config_.groupSize);
#ifdef _WIN32
    STARTUPINFOA si{};
    PROCESS_INFORMATION pi{};
    si.cb = sizeof(si);
    std::ostringstream cmd;
    cmd << '"' << g_selfPath << '"' << " --group-worker=1";
    std::string cmdLine = cmd.str();

    std::string envBlock;
    auto addEnv = [&](const std::string &k, const std::string &v) {
      envBlock += k + "=" + v + "\0";
    };
    addEnv("AI_GROUP_ID", groupId);
    addEnv("AI_GROUP_SIZE", groupSize);
    addEnv("AI_GROUP_COUNT", std::to_string(config_.groupCount));
    addEnv("AI_LMDB_ROOT", config_.lmdbRoot.string());
    addEnv("AI_LMDB_MAP_BYTES", std::to_string(config_.lmdbMapSizeBytes));
    addEnv("AI_BASE_DIR", config_.baseDir.string());
    addEnv("AI_GROUP_WORKER_PORT", portStr);
    envBlock.push_back('\0');

    BOOL ok = CreateProcessA(nullptr, cmdLine.data(), nullptr, nullptr, FALSE,
                             0, (LPVOID)envBlock.c_str(), nullptr, &si, &pi);
    if (!ok) {
      throw std::runtime_error("failed-to-spawn-group-worker");
    }
    info.proc = pi;
    CloseHandle(pi.hThread);
#else
    pid_t pid = fork();
    if (pid == 0) {
      setenv("AI_GROUP_ID", groupId.c_str(), 1);
      setenv("AI_GROUP_SIZE", groupSize.c_str(), 1);
      setenv("AI_GROUP_COUNT", std::to_string(config_.groupCount).c_str(), 1);
      setenv("AI_LMDB_ROOT", config_.lmdbRoot.string().c_str(), 1);
      setenv("AI_LMDB_MAP_BYTES",
             std::to_string(config_.lmdbMapSizeBytes).c_str(), 1);
      setenv("AI_BASE_DIR", config_.baseDir.string().c_str(), 1);
      setenv("AI_GROUP_WORKER_PORT", portStr.c_str(), 1);
      execl(g_selfPath.c_str(), g_selfPath.c_str(), "--group-worker=1",
            nullptr);
      _exit(1);
    }
    info.pid = pid;
#endif
  }

  SocketHandle connectWorker(int port) const {
#ifdef _WIN32
    static std::atomic<bool> wsaReady{false};
    if (!wsaReady) {
      WSADATA wsaData{};
      WSAStartup(MAKEWORD(2, 2), &wsaData);
      wsaReady = true;
    }
#endif
    SocketHandle sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == kInvalidSocket)
      return kInvalidSocket;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    for (int i = 0; i < 50; i++) {
      if (connect(sock, (sockaddr *)&addr, sizeof(addr)) == 0) {
        return sock;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    closeSocket(sock);
    return kInvalidSocket;
  }
};
