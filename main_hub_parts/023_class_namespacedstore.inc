/* Auto-generated from main.cpp
   Segment: class_NamespacedStore
   Original lines: 3361-3402 */

class NamespacedStore : public KeyValueStore {
public:
  NamespacedStore(std::shared_ptr<KeyValueStore> base, const std::string &ns)
      : base_(std::move(base)) {
    std::string trimmed = ns;
    trimmed.erase(trimmed.begin(), std::find_if(trimmed.begin(), trimmed.end(),
                                                [](unsigned char ch) {
                                                  return !std::isspace(ch);
                                                }));
    trimmed.erase(
        std::find_if(trimmed.rbegin(), trimmed.rend(),
                     [](unsigned char ch) { return !std::isspace(ch); })
            .base(),
        trimmed.end());
    if (trimmed.empty())
      throw std::runtime_error("NamespacedStore requires a namespace");
    prefix_ = "ns:" + trimmed + ":";
  }

  std::optional<json> get(const std::string &key) override {
    return base_->get(prefix_ + key);
  }
  void put(const std::string &key, const json &value) override {
    base_->put(prefix_ + key, value);
  }
  void del(const std::string &key) override { base_->del(prefix_ + key); }
  std::vector<std::pair<std::string, json>>
  entries(const std::string &prefix) override {
    auto all = base_->entries(prefix_ + prefix);
    std::vector<std::pair<std::string, json>> out;
    for (auto &kv : all) {
      if (kv.first.rfind(prefix_, 0) == 0) {
        out.push_back({kv.first.substr(prefix_.size()), kv.second});
      }
    }
    return out;
  }

private:
  std::shared_ptr<KeyValueStore> base_;
  std::string prefix_;
};
