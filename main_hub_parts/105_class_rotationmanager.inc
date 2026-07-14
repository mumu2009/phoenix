// Auto-generated from main.cpp
// Segment: class_RotationManager
// Original lines: 12684-12790

class RotationManager {
public:
  RotationManager(std::shared_ptr<ControllerPoolBase> pool,
                  int cycleMs = 5 * 60 * 1000, int learnIters = 3,
                  double minImprove = 0.05)
      : pool_(std::move(pool)), cycleMs_(cycleMs), learnIters_(learnIters),
        minImprove_(minImprove) {}

  void start() {
    if (running_)
      return;
    running_ = true;
    worker_ = std::thread([this]() {
      while (running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(cycleMs_));
        if (!running_)
          break;
        try {
          runCycle();
        } catch (...) {
        }
      }
    });
  }

  void stop() {
    running_ = false;
    if (worker_.joinable())
      worker_.join();
  }

  bool running() const { return running_; }
  int cycleMs() const { return cycleMs_; }

private:
  double evaluateController(const std::shared_ptr<Controller> &ctrl,
                            const std::vector<std::string> &prompts) {
    if (!ctrl)
      return -1e9;
    double quality = 0.0;
    int ok = 0;
    for (const auto &prompt : prompts) {
      try {
        auto out = ctrl->respond(json{{"text", prompt}});
        std::string reply = out.value("reply", "");
        if (reply.empty())
          continue;
        auto q = tokenize(prompt);
        auto r = tokenize(reply);
        if (r.empty())
          continue;
        std::unordered_set<std::string> rq(r.begin(), r.end());
        int hit = 0;
        for (auto &w : q)
          if (rq.count(w))
            hit++;
        double overlap =
            q.empty() ? 0.0 : (double)hit / std::max(1.0, (double)q.size());
        double diversity = (double)rq.size() / std::max(1.0, (double)r.size());
        quality += 0.65 * overlap + 0.35 * diversity;
        ok++;
      } catch (...) {
      }
    }
    if (ok == 0)
      return -1e9;
    return quality / ok;
  }

  void runCycle() {
    static const std::vector<std::string> prompts = {
        "请总结输入并给出关键结论", "解释这个问题的因果关系",
        "给出一个可执行的解决步骤", "将这段描述转换成结构化要点",
        "从风险和收益两个角度分析"};

    auto ctrlA = pool_->getActive();
    auto ctrlB = pool_->getStandby();
    auto ctrlC = pool_->getValidation();
    if (!ctrlA || !ctrlB || !ctrlC)
      return;

    double baseScore = evaluateController(ctrlA, prompts);
    double evalScore = evaluateController(ctrlB, prompts);

    if (!std::isfinite(baseScore) || !std::isfinite(evalScore))
      return;

    auto params = ctrlB->runtime()->cloneParams();
    int iteration = params.value("iteration", getModelDefaults().iteration);
    iteration += std::max(1, learnIters_);
    params["iteration"] = iteration;
    ctrlB->applyParams(params);

    if (evalScore - baseScore >= minImprove_) {
      pool_->hotSwap(ctrlB->snapshot());
    }
    ctrlC->applySnapshot(ctrlB->snapshot());
  }

  std::shared_ptr<ControllerPoolBase> pool_;
  int cycleMs_{5 * 60 * 1000};
  int learnIters_{3};
  double minImprove_{0.05};
  std::atomic<bool> running_{false};
  std::thread worker_;
  std::mt19937 rng_{std::random_device{}()};
};
