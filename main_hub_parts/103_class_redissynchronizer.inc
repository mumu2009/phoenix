// Auto-generated from main.cpp
// Segment: class_RedisSynchronizer
// Original lines: 12636-12682

class RedisSynchronizer : public IRedisSynchronizer {
public:
  RedisSynchronizer(std::shared_ptr<ControllerPoolBase> pool,
                    const std::string &url, const std::string &channel)
      : pool_(std::move(pool)), url_(url), channel_(channel) {}

  void start() override {
#ifdef HAVE_REDIS
    sw::redis::ConnectionOptions opts(url_);
    pub_ = std::make_unique<sw::redis::Redis>(opts);
    sub_ = std::make_unique<sw::redis::Redis>(opts);
    auto sub = sub_->subscriber();
    sub.subscribe(channel_);
    subThread_ = std::thread([this, sub = std::move(sub)]() mutable {
      while (running_) {
        sub.consume([&](std::string, std::string msg) {
          try {
            auto snapshot = json::parse(msg);
            pool_->hotSwap(snapshot);
          } catch (...) {
          }
        });
      }
    });
#endif
  }

  void publish(const json &snapshot) override {
#ifdef HAVE_REDIS
    if (pub_)
      pub_->publish(channel_, snapshot.dump());
#endif
  }

  const std::string &channel() const override { return channel_; }

private:
  std::shared_ptr<ControllerPoolBase> pool_;
  std::string url_;
  std::string channel_;
#ifdef HAVE_REDIS
  std::unique_ptr<sw::redis::Redis> pub_;
  std::unique_ptr<sw::redis::Redis> sub_;
#endif
  std::thread subThread_;
  std::atomic<bool> running_{true};
};
