// Auto-generated from main.cpp
// Segment: class_StudyEngine
// Original lines: 6238-6264

class StudyEngine : public IStudyEngine {
public:
  StudyEngine(std::shared_ptr<ControllerPoolBase> pool,
              std::shared_ptr<IRedisSynchronizer> redis);

  void start() override;
  void enqueueDocument(const json &doc) override;
  json status() const override;
  bool running() const override { return running_; }

private:
  void tick();
  std::shared_ptr<LemmaWorkerPool> ensureWorkerPool();

  std::shared_ptr<ControllerPoolBase> pool_;
  std::shared_ptr<IRedisSynchronizer> redis_;
  std::atomic<bool> running_{false};
  std::thread worker_;
  mutable std::mutex mu_;
  std::deque<json> queue_;
  json metrics_ = json{{"enqueued", 0},
                       {"processed", 0},
                       {"lastTickAt", 0},
                       {"lastError", nullptr}};
  mutable std::mutex workerMu_;
  std::shared_ptr<LemmaWorkerPool> workerPool_;
};
