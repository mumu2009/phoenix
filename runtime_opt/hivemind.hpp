#pragma once

#include <cstdint>
#include <list>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace phoenix {
namespace runtime_opt {

struct HiveConfig {
  bool enabled = true;
  bool neverDualSlot = true;
  int cacheTtlSec = 120;
  size_t cacheMax = 32;
};

struct OrganStep {
  std::string name;
  bool usesLlamaSlot = false;
};

struct OrchestraPlan {
  std::vector<OrganStep> serial;
  bool dualSlot = false;
  bool cancellable = true;
  std::string reason = "serial_no_dual_slot";
};

inline bool organUsesLlama(std::string_view name) {
  return name == "mission" || name == "llama" || name == "deliberate" ||
         name == "completion";
}

inline OrchestraPlan planOrchestra(const std::vector<std::string> &jobs,
                                   const HiveConfig &cfg) {
  OrchestraPlan p;
  if (!cfg.enabled) {
    p.reason = "hivemind_off";
    return p;
  }
  p.cancellable = true;
  p.dualSlot = false;
  bool sawLlama = false;
  for (const auto &j : jobs) {
    OrganStep s;
    s.name = j;
    s.usesLlamaSlot = organUsesLlama(j);
    if (s.usesLlamaSlot) {
      if (sawLlama && cfg.neverDualSlot)
        continue;
      sawLlama = true;
    }
    p.serial.push_back(s);
  }
  p.reason = sawLlama ? "serial_one_llama_slot" : "serial_helpers_only";
  return p;
}

inline uint64_t fnv1a(std::string_view s) {
  uint64_t h = 14695981039346656037ull;
  for (unsigned char c : s) {
    h ^= c;
    h *= 1099511628211ull;
  }
  return h;
}

inline std::string organCacheKey(std::string_view scope, std::string_view name,
                                 std::string_view input) {
  return std::string(scope) + "|" + std::string(name) + "|" +
         std::to_string(fnv1a(input));
}

class OrganResultCache {
public:
  explicit OrganResultCache(size_t maxEntries = 32) : max_(maxEntries) {}

  bool get(const std::string &key, std::string &out) const {
    auto it = map_.find(key);
    if (it == map_.end())
      return false;
    out = it->second;
    return true;
  }

  void put(const std::string &key, std::string value) {
    auto it = map_.find(key);
    if (it != map_.end()) {
      it->second = std::move(value);
      return;
    }
    if (order_.size() >= max_ && !order_.empty()) {
      map_.erase(order_.front());
      order_.pop_front();
    }
    order_.push_back(key);
    map_[key] = std::move(value);
  }

  size_t size() const { return map_.size(); }

private:
  size_t max_;
  std::unordered_map<std::string, std::string> map_;
  std::list<std::string> order_;
};

} // namespace runtime_opt
} // namespace phoenix
