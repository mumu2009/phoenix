// Auto-generated from main.cpp
// Segment: class_LemmaWorkerPool
// Original lines: 6145-6236

class LemmaWorkerPool {
public:
  explicit LemmaWorkerPool(int workers) {
    int count = std::max(1, workers);
    running_ = true;
    threads_.reserve(count);
    for (int i = 0; i < count; i++) {
      threads_.emplace_back([this]() { workerLoop(); });
    }
  }

  ~LemmaWorkerPool() { stop(); }

  std::vector<std::string>
  execBatchLemmatize(const std::vector<std::string> &tokens,
                     const std::string &lemmaCsvPath) {
    if (tokens.empty())
      return {};
    auto task = std::make_shared<Task>();
    task->tokens = tokens;
    task->lemmaCsvPath = lemmaCsvPath;
    auto fut = task->promise.get_future();
    {
      std::lock_guard<std::mutex> lock(mu_);
      queue_.push(task);
    }
    cv_.notify_one();
    return fut.get();
  }

  void stop() {
    if (!running_)
      return;
    running_ = false;
    cv_.notify_all();
    for (auto &t : threads_) {
      if (t.joinable())
        t.join();
    }
    threads_.clear();
  }

private:
  struct Task {
    std::vector<std::string> tokens;
    std::string lemmaCsvPath;
    std::promise<std::vector<std::string>> promise;
  };

  std::atomic<bool> running_{false};
  std::vector<std::thread> threads_;
  std::queue<std::shared_ptr<Task>> queue_;
  std::mutex mu_;
  std::condition_variable cv_;

  void workerLoop() {
    while (running_) {
      std::shared_ptr<Task> task;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&]() { return !running_ || !queue_.empty(); });
        if (!running_ && queue_.empty())
          return;
        task = queue_.front();
        queue_.pop();
      }
      try {
        auto lemmaMap = loadLemmaMapCached(task->lemmaCsvPath);
        std::vector<std::string> out;
        out.reserve(task->tokens.size());
        for (auto &w : task->tokens) {
          std::string normalized = trimString(lowerAscii(w));
          if (normalized.empty()) {
            out.push_back(std::string());
            continue;
          }
          auto it = lemmaMap.find(normalized);
          if (it != lemmaMap.end())
            out.push_back(it->second);
          else
            out.push_back(normalized);
        }
        task->promise.set_value(out);
      } catch (...) {
        try {
          task->promise.set_value({});
        } catch (...) {
        }
      }
    }
  }
};
