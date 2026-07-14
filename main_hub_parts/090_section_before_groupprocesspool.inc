// Auto-generated from main.cpp
// Segment: section_before_GroupProcessPool
// Original lines: 10365-10469


InferenceProcessPool::InferenceProcessPool(ControllerPool *pool, int workers,
                                           int timeoutMs)
    : pool_(pool), timeoutMs_(std::max(200, timeoutMs)) {
  int count = std::max(1, workers);
  running_ = true;
  workers_.reserve(count);
  for (int i = 0; i < count; i++) {
    workers_.emplace_back([this]() { workerLoop(); });
  }
}

InferenceProcessPool::~InferenceProcessPool() { stop(); }

void InferenceProcessPool::stop() {
  if (!running_)
    return;
  running_ = false;
  cv_.notify_all();
  for (auto &t : workers_) {
    if (t.joinable())
      t.join();
  }
  workers_.clear();
}

json InferenceProcessPool::respond(const std::string &controllerName,
                                   const json &payload) {
  if (!running_)
    throw std::runtime_error("inference-pool-stopped");
  auto task = std::make_shared<Task>();
  task->controllerName = controllerName;
  auto packed =
      std::make_shared<std::vector<uint8_t>>(json::to_msgpack(buildShmEnvelope(
          payload, controllerName,
          pool_ ? pool_->config().transformerBackend : std::string(),
          "request")));
  task->payload = SharedMemorySlice::fromVector(packed);
  task->promise = std::make_shared<std::promise<SharedMemorySlice>>();
  auto fut = task->promise->get_future();
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (queue_.size() >= maxQueue_)
      throw std::runtime_error("inference-queue-busy");
    queue_.push(task);
  }
  cv_.notify_one();
  if (fut.wait_for(std::chrono::milliseconds(timeoutMs_)) ==
      std::future_status::timeout) {
    throw std::runtime_error("inference-timeout");
  }
  auto out = fut.get();
  if (!out.owner)
    return json::object();
  try {
    json decoded = json::from_msgpack(*out.owner);
    json payloadDecoded;
    if (unpackShmEnvelope(decoded, payloadDecoded))
      return payloadDecoded;
    return decoded;
  } catch (...) {
    throw std::runtime_error("inference-invalid-response");
  }
}

void InferenceProcessPool::workerLoop() {
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
      if (!pool_)
        throw std::runtime_error("pool-null");
      auto ctrl = pool_->getByName(task->controllerName);
      if (!ctrl)
        throw std::runtime_error("controller-offline");
      json payload = json::object();
      if (!task->payload.empty() && task->payload.owner) {
        auto raw = json::from_msgpack(*task->payload.owner);
        if (!unpackShmEnvelope(raw, payload))
          payload = raw;
      }
      auto out = ctrl->respond(payload);
      auto packedOut = std::make_shared<std::vector<uint8_t>>(
          json::to_msgpack(buildShmEnvelope(
              out, task->controllerName,
              pool_ ? pool_->config().transformerBackend : std::string(),
              "response")));
      task->promise->set_value(SharedMemorySlice::fromVector(packedOut));
    } catch (const std::exception &e) {
      try {
        task->promise->set_exception(
            std::make_exception_ptr(std::runtime_error(e.what())));
      } catch (...) {
      }
    }
  }
}

