// Auto-generated from main.cpp
// Segment: class_DialogMemory
// Original lines: 5389-5544

class DialogMemory {
public:
  DialogMemory(std::shared_ptr<KeyValueStore> store, int maxItems = 2048,
               int maxPerIndex = 64)
      : store_(std::move(store)), maxItems_(std::max(128, maxItems)),
        maxPerIndex_(std::max(8, maxPerIndex)) {}

  json remember(const std::string &sessionId, const std::string &signature,
                const std::vector<std::string> &memes,
                const std::string &question, const std::string &reply,
                double scoreHint = 0.0) {
    std::string sid = scopeKey(sessionId);
    if (sid.empty() || signature.empty() || reply.empty())
      return json();
    std::string id = sid + ":" + sha1Hex(signature);
    auto key = "d:" + id;
    auto prev = store_->get(key);
    int count = prev ? prev->value("count", 0) : 0;
    double prevHint = prev ? prev->value("scoreHint", 0.0) : 0.0;
    double hint = std::isfinite(scoreHint) ? scoreHint : prevHint;
    json rec = {
        {"id", id},
        {"sessionId", sid},
        {"signature", signature},
        {"memes", memes},
        {"question", question.substr(0, 800)},
        {"reply", reply.substr(0, 1200)},
        {"updatedAt",
         (int64_t)std::chrono::system_clock::now().time_since_epoch().count()},
        {"count", count + 1},
        {"scoreHint", hint}};
    store_->put(key, rec);
    for (const auto &memeId :
         std::unordered_set<std::string>(memes.begin(), memes.end())) {
      std::string ik = "i:" + sid + ":" + memeId;
      auto ids = store_->get(ik);
      std::vector<std::string> list;
      if (ids && ids->is_array()) {
        for (auto &x : *ids) {
          if (x.is_string())
            list.push_back(x.get<std::string>());
        }
      }
      list.erase(std::remove(list.begin(), list.end(), id), list.end());
      list.insert(list.begin(), id);
      if ((int)list.size() > maxPerIndex_)
        list.resize(maxPerIndex_);
      store_->put(ik, list);
    }
    return rec;
  }

  json retrieve(const std::string &sessionId,
                const std::vector<std::string> &memes,
                const std::string &signature, double minSim = 0.45) {
    std::string sid = scopeKey(sessionId);
    if (sid.empty())
      return json();
    std::unordered_set<std::string> candidateIds;
    for (size_t i = 0; i < std::min<size_t>(8, memes.size()); i++) {
      auto ids = store_->get("i:" + sid + ":" + memes[i]);
      if (ids && ids->is_array()) {
        for (auto &x : *ids) {
          if (x.is_string() && !x.get<std::string>().empty())
            candidateIds.insert(x.get<std::string>());
        }
      }
    }
    if (candidateIds.empty() && !signature.empty())
      candidateIds.insert(sid + ":" + sha1Hex(signature));

    auto toSet = [](const std::string &sig) {
      std::unordered_set<std::string> out;
      std::stringstream ss(sig);
      std::string item;
      while (std::getline(ss, item, '|'))
        if (!item.empty())
          out.insert(item);
      return out;
    };

    auto sigSet = toSet(signature);
    json best;
    double bestScore = 0;
    for (const auto &id : candidateIds) {
      auto rec = store_->get("d:" + id);
      if (!rec || !rec->contains("reply"))
        continue;
      if ((*rec)["reply"].is_string()) {
        auto r = (*rec)["reply"].get<std::string>();
        auto start = r.find_first_not_of(" \n\r\t");
        if (start == std::string::npos)
          continue;
      }
      auto recSet = toSet(rec->value("signature", ""));
      double inter = 0;
      for (auto &x : sigSet)
        if (recSet.count(x))
          inter++;
      double uni = sigSet.size() + recSet.size() - inter;
      double sim = (uni <= 0) ? 0 : inter / uni;
      if (sim < minSim)
        continue;
      double freq = std::log(1 + rec->value("count", 0));
      double score = sim * (1 + 0.15 * freq);
      if (score > bestScore) {
        bestScore = score;
        best = *rec;
        best["similarity"] = sim;
        best["score"] = score;
      }
    }
    return best;
  }

  json exportSnapshot(int limit = 512) {
    json out{{"dialogs", json::array()}, {"indexes", json::array()}};
    auto dialogs = store_->entries("d:");
    for (auto &kv : dialogs) {
      out["dialogs"].push_back({kv.first, kv.second});
      if ((int)out["dialogs"].size() >= limit)
        break;
    }
    auto indexes = store_->entries("i:");
    for (auto &kv : indexes) {
      out["indexes"].push_back({kv.first, kv.second});
      if ((int)out["indexes"].size() >= limit)
        break;
    }
    return out;
  }

  void importSnapshot(const json &snapshot) {
    for (auto &item : snapshot.value("dialogs", json::array())) {
      if (!item.is_array() || item.size() != 2)
        continue;
      store_->put(item[0].get<std::string>(), item[1]);
    }
    for (auto &item : snapshot.value("indexes", json::array())) {
      if (!item.is_array() || item.size() != 2)
        continue;
      store_->put(item[0].get<std::string>(), item[1]);
    }
  }

private:
  static std::string scopeKey(const std::string &sessionId) {
    if (sessionId.empty())
      return std::string();
    return "s" + sha1Hex(sessionId);
  }

  std::shared_ptr<KeyValueStore> store_;
  int maxItems_;
  int maxPerIndex_;
};
