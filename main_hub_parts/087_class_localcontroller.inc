// Auto-generated from main.cpp
// Segment: class_LocalController
// Original lines: 10154-10183

class LocalController : public Controller {
public:
  LocalController(std::string name, std::shared_ptr<RuntimeState> runtime)
      : name_(std::move(name)), runtime_(std::move(runtime)),
        since_(nowEpochMs()) {}

  json respond(const json &payload) override {
    return runtime_->processInput(payload);
  }
  json metrics() const override {
    return json{
        {"name", name_},
        {"online", true},
        {"health", json{{"status", "ok"}, {"since", since_}, {"failures", 0}}},
        {"metrics", runtime_->metrics()},
        {"params", runtime_->cloneParams()}};
  }
  void applyParams(const json &p) override { runtime_->setParams(p); }
  json snapshot() const override { return runtime_->toSnapshot(); }
  void applySnapshot(const json &snap) override {
    runtime_->fromSnapshot(snap);
  }
  const std::string &name() const override { return name_; }
  std::shared_ptr<RuntimeBase> runtime() const override { return runtime_; }

private:
  std::string name_;
  std::shared_ptr<RuntimeState> runtime_;
  int64_t since_{0};
};
