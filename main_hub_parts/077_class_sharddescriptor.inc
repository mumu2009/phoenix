// Auto-generated from main.cpp
// Segment: class_ShardDescriptor
// Original lines: 9930-9991

class ShardDescriptor {
public:
  explicit ShardDescriptor(std::string name)
      : name_(std::move(name)), label_("AI-" + name_), embedding_(64, 0.0f) {}

  const std::string &name() const { return name_; }
  const std::string &label() const { return label_; }
  const std::vector<float> &embedding() const { return embedding_; }
  bool isOffline() const { return offline_; }

  void updateEmbedding(const std::vector<float> &vec) {
    double alpha = 0.2;
    if (vec.size() != embedding_.size())
      return;
    for (size_t i = 0; i < embedding_.size(); i++) {
      embedding_[i] = (float)((1 - alpha) * embedding_[i] + alpha * vec[i]);
    }
  }

  void record(const json &entry) {
    requests_++;
    bool ok = entry.value("ok", true);
    if (!ok)
      failures_++;
    int latency = entry.value("latency", 0);
    if (latency > 0) {
      if (emaLatencyMs_ <= 0.0)
        emaLatencyMs_ = latency;
      else
        emaLatencyMs_ = 0.9 * emaLatencyMs_ + 0.1 * latency;
    }
    lastSeenMs_ = nowEpochMs();
    double failRate = requests_ > 0 ? (double)failures_ / requests_ : 0.0;
    offline_ = failRate > 0.45 || (emaLatencyMs_ > 3500.0 && requests_ > 6);

    history_.push_back(entry);
    if (history_.size() > 50)
      history_.erase(history_.begin());
  }

  json toJson() const {
    return json{{"name", name_},
                {"label", label_},
                {"offline", offline_},
                {"emaLatencyMs", emaLatencyMs_},
                {"requests", requests_},
                {"failures", failures_},
                {"lastSeenMs", lastSeenMs_},
                {"history", history_}};
  }

private:
  std::string name_;
  std::string label_;
  std::vector<float> embedding_;
  std::vector<json> history_;
  bool offline_{false};
  int requests_{0};
  int failures_{0};
  double emaLatencyMs_{0.0};
  int64_t lastSeenMs_{0};
};
