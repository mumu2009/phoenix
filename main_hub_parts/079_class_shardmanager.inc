// Auto-generated from main.cpp
// Segment: class_ShardManager
// Original lines: 9993-10068

class ShardManager {
public:
  explicit ShardManager(std::shared_ptr<ControllerPoolBase> pool)
      : pool_(std::move(pool)) {
    auto names = pool_->listControllerNames();
    for (auto &name : names) {
      shards_.emplace(name, ShardDescriptor(name));
    }
    lastEmbedding_ = std::vector<float>(64, 0.0f);
  }

  std::shared_ptr<Controller> chooseShard(const std::string &text) {
    auto requestEmbedding = textToMiniEmbedding(text, 64);
    lastEmbedding_ = requestEmbedding;
    auto best = pool_->getActive();
    double bestScore = -1e9;
    for (auto &kv : shards_) {
      if (kv.second.isOffline())
        continue;
      auto score = cosineSim(requestEmbedding, kv.second.embedding());
      if (score > bestScore) {
        auto ctrl = pool_->getByName(kv.first);
        if (ctrl) {
          bestScore = score;
          best = ctrl;
        }
      }
    }
    if (!best)
      best = pool_->getActive();
    return best;
  }

  json metrics() const {
    json out = json::array();
    for (auto &kv : shards_) {
      out.push_back({{"name", kv.first}, {"shard", kv.second.toJson()}});
    }
    return out;
  }

  void record(const std::string &controllerName,
              const std::vector<float> &requestEmbedding,
              const std::string &replyText, int latency) {
    auto it = shards_.find(controllerName);
    if (it == shards_.end())
      return;
    auto replyEmbedding =
        textToMiniEmbedding(replyText, (int)requestEmbedding.size());
    it->second.updateEmbedding(replyEmbedding);
    double affinity = cosineSim(requestEmbedding, replyEmbedding);
    it->second.record(json{{"ok", true},
                           {"latency", latency},
                           {"affinity", affinity},
                           {"ts", (int64_t)std::chrono::system_clock::now()
                                      .time_since_epoch()
                                      .count()}});
  }

  void recordFailure(const std::string &controllerName, int latency = 0) {
    auto it = shards_.find(controllerName);
    if (it == shards_.end())
      return;
    it->second.record(json{{"ok", false},
                           {"latency", latency},
                           {"affinity", 0.0},
                           {"ts", (int64_t)std::chrono::system_clock::now()
                                      .time_since_epoch()
                                      .count()}});
  }

private:
  std::shared_ptr<ControllerPoolBase> pool_;
  std::unordered_map<std::string, ShardDescriptor> shards_;
  std::vector<float> lastEmbedding_;
};
