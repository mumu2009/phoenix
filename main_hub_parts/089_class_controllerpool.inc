// Auto-generated from main.cpp
// Segment: class_ControllerPool
// Original lines: 10185-10364

class ControllerPool : public ControllerPoolBase {
public:
  ControllerPool(std::shared_ptr<KeyValueStore> kvmStore,
                 std::shared_ptr<KeyValueStore> memeStore,
                 std::shared_ptr<KeyValueStore> sessionStore,
                 const Config &config)
      : config_(config) {
#ifdef HAVE_LMDB
    bool preferSqlite = Database079::isSqliteStore(kvmStore) ||
                        Database079::isSqliteStore(memeStore) ||
                        Database079::isSqliteStore(sessionStore);
    if (!preferSqlite) {
      auto lmdbK = std::make_shared<LmdbStore>("kvm", config_.lmdbRoot,
                                               config_.lmdbMapSizeBytes);
      auto lmdbM = std::make_shared<LmdbStore>("meme_graph", config_.lmdbRoot,
                                               config_.lmdbMapSizeBytes);
      auto lmdbS = std::make_shared<LmdbStore>("session", config_.lmdbRoot,
                                               config_.lmdbMapSizeBytes);
      if (lmdbK->ok())
        kvmStore = lmdbK;
      if (lmdbM->ok())
        memeStore = lmdbM;
      if (lmdbS->ok())
        sessionStore = lmdbS;
    }
#endif
    for (int i = 0; i < config_.groupCount; i++) {
      std::string gid = "G" + std::to_string(i + 1);
      groupIds_.push_back(gid);
      std::vector<std::string> names;
      auto kvmNs = std::make_shared<NamespacedStore>(kvmStore, gid + ":kvm");
      auto memeNs =
          std::make_shared<NamespacedStore>(memeStore, gid + ":graph");
      auto sessNs =
          std::make_shared<NamespacedStore>(sessionStore, gid + ":session");
      for (int j = 0; j < config_.groupSize; j++) {
        std::string name = gid + "_AI" + std::to_string(j + 1);
        controllers_[name] = std::make_shared<LocalController>(
            name,
            std::make_shared<RuntimeState>(kvmNs, memeNs, sessNs, config_));
        names.push_back(name);
      }
      groups_[gid] = names;
    }
    auto g1 = groupIds_.empty() ? "G1" : groupIds_[0];
    auto list = groups_[g1];
    serving_ = controllers_[list[0]];
    standby_ = controllers_[list.size() > 1 ? list[1] : list[0]];
    validation_ = controllers_[list.size() > 2 ? list[2] : list[0]];
    if (config_.inferMP && !config_.groupProc) {
      inferencePool_ = std::make_unique<InferenceProcessPool>(
          this, config_.inferWorkers, config_.groupProcTimeoutMs);
    }
  }

  std::shared_ptr<Controller> getActive() override { return serving_; }
  std::shared_ptr<Controller> getStandby() override { return standby_; }
  std::shared_ptr<Controller> getValidation() override { return validation_; }

  std::shared_ptr<Controller> getByName(const std::string &name) override {
    if (name == "A" || name == "B" || name == "C") {
      auto g1 = groupIds_.empty() ? "G1" : groupIds_[0];
      auto list = groups_[g1];
      if (name == "A" && !list.empty())
        return controllers_[list[0]];
      if (name == "B" && list.size() > 1)
        return controllers_[list[1]];
      if (name == "C" && list.size() > 2)
        return controllers_[list[2]];
    }
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
    for (auto &name : listControllerNames()) {
      auto it = controllers_.find(name);
      if (it != controllers_.end())
        out.push_back(it->second->metrics());
    }
    return out;
  }

  json ingestDocument(const json &doc) override {
    json results = json::array();
    for (auto &name : listControllerNames()) {
      auto ctrl = getByName(name);
      if (!ctrl) {
        results.push_back(
            json{{"ok", false}, {"reason", "controller-not-found"}});
        continue;
      }
      results.push_back(ctrl->runtime()->ingestDocument(doc));
    }
    return results;
  }

  json ingestDocumentToGroup(const std::string &gid, const json &doc) override {
    auto list = listControllersInGroup(gid);
    if (list.empty())
      return json{{"ok", false}, {"reason", "group-not-found"}};
    json results = json::array();
    for (auto &name : list) {
      auto ctrl = getByName(name);
      if (!ctrl) {
        results.push_back(
            json{{"ok", false}, {"reason", "controller-not-found"}});
        continue;
      }
      results.push_back(ctrl->runtime()->ingestDocument(doc));
    }
    return results;
  }

  json forgetMemes(const json &payload) override {
    json results = json::array();
    for (auto &name : listControllerNames()) {
      auto ctrl = getByName(name);
      if (!ctrl) {
        results.push_back(
            json{{"ok", false}, {"reason", "controller-not-found"}});
        continue;
      }
      results.push_back(ctrl->runtime()->forgetMemes(payload));
    }
    return results;
  }

  json onlineResearch(const json &queryOrTokens, const json &options) override {
    return serving_->runtime()->onlineLookup(queryOrTokens, options);
  }

  void hotSwap(const json &snapshot) override {
    standby_->applySnapshot(snapshot);
    std::swap(serving_, standby_);
  }

  const Config &config() const override { return config_; }

  json respondByName(const std::string &controllerName,
                     const json &payload) override {
    if (inferencePool_) {
      return inferencePool_->respond(controllerName, payload);
    }
    auto ctrl = getByName(controllerName);
    if (!ctrl)
      throw std::runtime_error("controller-offline");
    return ctrl->respond(payload);
  }

private:
  Config config_;
  std::unordered_map<std::string, std::shared_ptr<Controller>> controllers_;
  std::unordered_map<std::string, std::vector<std::string>> groups_;
  std::vector<std::string> groupIds_;
  std::shared_ptr<Controller> serving_;
  std::shared_ptr<Controller> standby_;
  std::shared_ptr<Controller> validation_;
  std::unique_ptr<InferenceProcessPool> inferencePool_;
};
