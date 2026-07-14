// Auto-generated from main.cpp
// Segment: section_before_SnapshotManager
// Original lines: 12791-12908


StudyEngine::StudyEngine(std::shared_ptr<ControllerPoolBase> pool,
                         std::shared_ptr<IRedisSynchronizer> redis)
    : pool_(std::move(pool)), redis_(std::move(redis)) {}

void StudyEngine::start() {
  if (running_)
    return;
  running_ = true;
  worker_ = std::thread([this]() {
    while (running_) {
      std::this_thread::sleep_for(std::chrono::seconds(60));
      if (!running_)
        break;
      try {
        tick();
      } catch (...) {
      }
    }
  });
}

void StudyEngine::enqueueDocument(const json &doc) {
  std::lock_guard<std::mutex> lock(mu_);
  queue_.push_back(doc);
  metrics_["enqueued"] = metrics_.value("enqueued", 0) + 1;
}

json StudyEngine::status() const {
  std::lock_guard<std::mutex> lock(mu_);
  return json{{"running", running_.load()},
              {"queue", (int)queue_.size()},
              {"metrics", metrics_}};
}

std::shared_ptr<LemmaWorkerPool> StudyEngine::ensureWorkerPool() {
  std::lock_guard<std::mutex> lock(workerMu_);
  if (workerPool_)
    return workerPool_;
  int workers = 1;
  try {
    auto cfg = pool_ ? pool_->config() : Config{};
    workers = std::max(1, cfg.maxWorkers);
  } catch (...) {
    workers = 1;
  }
  workerPool_ = std::make_shared<LemmaWorkerPool>(workers);
  return workerPool_;
}

void StudyEngine::tick() {
  json doc;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (!running_ || queue_.empty())
      return;
    doc = queue_.front();
    queue_.pop_front();
  }
  try {
    {
      std::lock_guard<std::mutex> lock(mu_);
      metrics_["lastTickAt"] =
          (int64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
    }
    if (!doc.is_null()) {
      pool_->ingestDocument(doc);
    }
    try {
      if (!doc.is_null()) {
        std::string text = doc.value("text", "");
        auto tokens = tokenize(text);
        std::string lemmaCsv;
        if (pool_) {
          auto active = pool_->getActive();
          if (active) {
            try {
              lemmaCsv = active->runtime()->config().lemmaCsv.string();
            } catch (...) {
            }
          }
        }
        auto wp = ensureWorkerPool();
        if (wp) {
          wp->execBatchLemmatize(tokens, lemmaCsv);
        }
      }
    } catch (...) {
      // ignore worker errors
    }
    {
      std::lock_guard<std::mutex> lock(mu_);
      metrics_["processed"] = metrics_.value("processed", 0) + 1;
      metrics_["lastError"] = nullptr;
    }
  } catch (const std::exception &e) {
    std::lock_guard<std::mutex> lock(mu_);
    metrics_["lastError"] = std::string(e.what());
  }
  if (redis_) {
    try {
      auto snap = pool_->getStandby()->snapshot();
      redis_->publish(snap);
    } catch (...) {
    }
  }
}

// ------------------ Snapshot ------------------

// SnapshotManager 管理运行时快照的创建与恢复。
// 调用方式：网关路由调用 create/list/restore/remove。
// 实现思路：快照 JSON 原子写入并按数量进行清理。
// 注意事项：恢复时需校验快照结构兼容当前版本。
// 注意事项：文件系统异常会抛错并由上层处理。
// 注意事项：快照只覆盖运行态，不覆盖外部依赖。
