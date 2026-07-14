// Auto-generated from main.cpp
// Segment: class_ProxyController
// Original lines: 10134-10152

class ProxyController : public Controller {
public:
  ProxyController(GroupProcessPool *pool, std::string name)
      : pool_(pool), name_(std::move(name)),
        runtime_(std::make_shared<ProxyRuntime>(pool, name_)) {}

  json respond(const json &payload) override;
  json metrics() const override;
  void applyParams(const json &p) override;
  json snapshot() const override;
  void applySnapshot(const json &snap) override;
  const std::string &name() const override { return name_; }
  std::shared_ptr<RuntimeBase> runtime() const override { return runtime_; }

private:
  GroupProcessPool *pool_{nullptr};
  std::string name_;
  std::shared_ptr<ProxyRuntime> runtime_;
};
