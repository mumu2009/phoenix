// Auto-generated from main.cpp
// Segment: class_MemeSurfaceLexicon
// Original lines: 5267-5381

class MemeSurfaceLexicon {
public:
  MemeSurfaceLexicon(std::shared_ptr<KeyValueStore> store,
                     int maxEntriesPerMeme = 64, double decay = 0.985)
      : store_(std::move(store)), maxEntries_(std::max(8, maxEntriesPerMeme)),
        decay_(decay) {}

  void learn(const std::string &memeId, const std::string &replyText,
             double weight = 1.0) {
    if (memeId.empty())
      return;
    auto rec = load(memeId);
    json phrases = rec.value("phrases", json::object());
    for (auto it = phrases.begin(); it != phrases.end();) {
      double v = 0.0;
      if (it.value().is_number()) {
        v = it.value().get<double>();
      } else if (it.value().is_string()) {
        try {
          v = std::stod(it.value().get<std::string>());
        } catch (...) {
          v = 0.0;
        }
      }
      double nv = v * decay_;
      if (nv <= 1e-6) {
        it = phrases.erase(it);
      } else {
        it.value() = nv;
        ++it;
      }
    }
    for (auto &p : extractSurfacePhrases(replyText, 24)) {
      double prev = phrases.value(p.phrase, 0.0);
      phrases[p.phrase] = prev + p.weight * weight;
    }
    std::vector<std::pair<std::string, double>> ordered;
    for (auto it = phrases.begin(); it != phrases.end(); ++it) {
      if (it.value().is_number())
        ordered.push_back({it.key(), it.value().get<double>()});
    }
    std::sort(ordered.begin(), ordered.end(),
              [](auto &a, auto &b) { return a.second > b.second; });
    if ((int)ordered.size() > maxEntries_)
      ordered.resize(maxEntries_);
    json next;
    next["updatedAt"] =
        (int64_t)std::chrono::system_clock::now().time_since_epoch().count();
    json obj = json::object();
    for (auto &kv : ordered)
      obj[kv.first] = kv.second;
    next["phrases"] = obj;
    store_->put("m:" + memeId, next);
  }

  std::vector<SurfacePhrase> getTop(const std::string &memeId, int limit = 6) {
    auto rec = load(memeId);
    std::vector<SurfacePhrase> out;
    json phrases = rec.value("phrases", json::object());
    for (auto it = phrases.begin(); it != phrases.end(); ++it) {
      double v = 0.0;
      if (it.value().is_number()) {
        v = it.value().get<double>();
      } else if (it.value().is_string()) {
        try {
          v = std::stod(it.value().get<std::string>());
        } catch (...) {
          v = 0.0;
        }
      }
      out.push_back({it.key(), v});
    }
    std::sort(out.begin(), out.end(),
              [](auto &a, auto &b) { return a.weight > b.weight; });
    if ((int)out.size() > limit)
      out.resize(limit);
    return out;
  }

  json exportSnapshot(int limitMemes = 512) {
    json out = json::array();
    auto items = store_->entries("m:");
    for (auto &kv : items) {
      out.push_back({kv.first, kv.second});
      if ((int)out.size() >= limitMemes)
        break;
    }
    return out;
  }

  void importSnapshot(const json &entries) {
    if (!entries.is_array())
      return;
    for (auto &item : entries) {
      if (!item.is_array() || item.size() != 2)
        continue;
      std::string key = item[0].get<std::string>();
      if (key.rfind("m:", 0) != 0)
        continue;
      store_->put(key, item[1]);
    }
  }

private:
  std::shared_ptr<KeyValueStore> store_;
  int maxEntries_;
  double decay_;

  json load(const std::string &memeId) {
    auto v = store_->get("m:" + memeId);
    if (v && v->is_object())
      return *v;
    return json{{"phrases", json::object()}, {"updatedAt", 0}};
  }
};
