// Auto-generated from main.cpp
// Segment: class_SessionManager
// Original lines: 5851-6043

class SessionManager {
public:
  SessionManager(std::shared_ptr<KeyValueStore> store,
                 int idleMs = 10 * 60 * 1000, int maxSessions = 200)
      : store_(std::move(store)), idleMs_(idleMs), maxSessions_(maxSessions) {}

  std::string ensure(const std::string &sessionId) {
    std::lock_guard<std::mutex> lock(mu_);
    loadFromStoreLocked();
    cleanupExpiredLocked(nowMs());

    if (!sessionId.empty()) {
      std::string sid = sessionId;
      auto it = active_.find(sid);
      if (it != active_.end()) {
        auto &data = it->second;
        data["lastActivity"] = nowMs();
        data["count"] = data.value("count", 0) + 1;
        saveLocked(sid, data);
        if ((int)active_.size() > maxSessions_)
          truncateLocked();
        return sid;
      }
      auto stored = store_->get("session:" + sid);
      if (stored && stored->is_object()) {
        json data = *stored;
        data["id"] = data.value("id", sid);
        data["lastActivity"] = nowMs();
        data["count"] = data.value("count", 0) + 1;
        active_[sid] = data;
        if (std::find(order_.begin(), order_.end(), sid) == order_.end())
          order_.push_back(sid);
        saveLocked(sid, data);
        if ((int)active_.size() > maxSessions_)
          truncateLocked();
        return sid;
      }
    }

    std::string id = newIdLocked();
    json data{{"id", id},
              {"createdAt", nowMs()},
              {"lastActivity", nowMs()},
              {"count", 1},
              {"meta", json::object()}};
    active_[id] = data;
    order_.push_back(id);
    saveLocked(id, data);
    if ((int)active_.size() > maxSessions_)
      truncateLocked();
    return id;
  }

  json exportSessions() const {
    std::lock_guard<std::mutex> lock(mu_);
    json out = json::array();
    for (auto &id : order_) {
      auto it = active_.find(id);
      if (it != active_.end())
        out.push_back(it->second);
    }
    return out;
  }

  void importSessions(const json &list) {
    std::lock_guard<std::mutex> lock(mu_);
    active_.clear();
    order_.clear();
    loaded_ = true;
    if (!list.is_array())
      return;
    for (auto &item : list) {
      if (!item.is_object())
        continue;
      std::string id = item.value("id", "");
      if (id.empty())
        continue;
      active_[id] = item;
      order_.push_back(id);
      saveLocked(id, item);
    }
    cleanupExpiredLocked(nowMs());
    if ((int)active_.size() > maxSessions_)
      truncateLocked();
  }

private:
  std::shared_ptr<KeyValueStore> store_;
  int idleMs_{10 * 60 * 1000};
  int maxSessions_{200};
  std::unordered_map<std::string, json> active_;
  std::vector<std::string> order_;
  bool loaded_{false};
  mutable std::mutex mu_;

  static int64_t nowMs() {
    return (int64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
  }

  void saveLocked(const std::string &sid, const json &data) {
    store_->put("session:" + sid, data);
  }

  std::string newIdLocked() {
    for (int i = 0; i < 8; i++) {
      std::string candidate = randomSessionId();
      if (!active_.count(candidate))
        return candidate;
    }
    return randomHex(16);
  }

  void loadFromStoreLocked() {
    if (loaded_)
      return;
    loaded_ = true;
    auto items = store_->entries("session:");
    for (auto &kv : items) {
      if (!kv.second.is_object())
        continue;
      std::string sid = kv.second.value("id", "");
      if (sid.empty()) {
        if (kv.first.rfind("session:", 0) == 0)
          sid = kv.first.substr(8);
      }
      if (sid.empty())
        continue;
      active_[sid] = kv.second;
      active_[sid]["id"] = sid;
      order_.push_back(sid);
    }
  }

  void cleanupExpiredLocked(int64_t now) {
    if (idleMs_ <= 0)
      return;
    std::unordered_set<std::string> keep;
    for (auto it = active_.begin(); it != active_.end();) {
      const auto &obj = it->second;
      int64_t last = obj.value("lastActivity", obj.value("createdAt", 0ll));
      if (now - last > idleMs_) {
        store_->del("session:" + it->first);
        it = active_.erase(it);
      } else {
        keep.insert(it->first);
        ++it;
      }
    }
    if (!order_.empty()) {
      std::vector<std::string> next;
      next.reserve(order_.size());
      for (const auto &id : order_) {
        if (keep.count(id))
          next.push_back(id);
      }
      order_.swap(next);
    }
  }

  void truncateLocked() {
    std::vector<json> items;
    items.reserve(active_.size());
    for (auto &kv : active_)
      items.push_back(kv.second);
    std::sort(items.begin(), items.end(), [](const json &a, const json &b) {
      return a.value("lastActivity", 0) > b.value("lastActivity", 0);
    });
    if ((int)items.size() <= maxSessions_)
      return;
    std::unordered_set<std::string> keep;
    for (int i = 0; i < maxSessions_ && i < (int)items.size(); i++)
      keep.insert(items[i].value("id", ""));
    for (auto it = active_.begin(); it != active_.end();) {
      if (!keep.count(it->first)) {
        store_->del("session:" + it->first);
        it = active_.erase(it);
      } else {
        ++it;
      }
    }
    if (!order_.empty()) {
      std::vector<std::string> next;
      next.reserve(order_.size());
      for (auto &id : order_) {
        if (keep.count(id))
          next.push_back(id);
      }
      order_.swap(next);
    }
  }
};
