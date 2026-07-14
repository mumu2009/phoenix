// Auto-generated from main.cpp
// Segment: class_InferenceProcessPool
// Original lines: 10072-10097

class InferenceProcessPool {
public:
  InferenceProcessPool(ControllerPool *pool, int workers, int timeoutMs);
  ~InferenceProcessPool();

  json respond(const std::string &controllerName, const json &payload);
  void stop();

private:
  struct Task {
    std::string controllerName;
    SharedMemorySlice payload;
    std::shared_ptr<std::promise<SharedMemorySlice>> promise;
  };

  ControllerPool *pool_{nullptr};
  int timeoutMs_{12000};
  size_t maxQueue_{1024};
  std::atomic<bool> running_{false};
  std::vector<std::thread> workers_;
  std::queue<std::shared_ptr<Task>> queue_;
  std::mutex mu_;
  std::condition_variable cv_;

  void workerLoop();
};
