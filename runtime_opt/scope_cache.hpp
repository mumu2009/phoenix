#pragma once

#include "runtime_opt/hivemind.hpp"

#include <list>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace phoenix {
namespace runtime_opt {

struct ScopeCacheConfig {
  bool enabled = true;
  size_t maxEntries = 64;
};

/* Read-only retrieval cache keyed by isolation scope.
   Does not train RNN/LSTM/GNN or write memegraph. */
inline std::string scopeCacheKey(std::string_view scope, std::string_view query) {
  return std::string(scope) + "#" + std::to_string(fnv1a(query));
}

class ScopeReadCache {
public:
  explicit ScopeReadCache(size_t maxEntries = 64) : max_(maxEntries) {}

  bool get(std::string_view scope, std::string_view query, std::string &out) const {
    const std::string k = scopeCacheKey(scope, query);
    auto it = map_.find(k);
    if (it == map_.end())
      return false;
    out = it->second;
    return true;
  }

  void put(std::string_view scope, std::string_view query, std::string value) {
    const std::string k = scopeCacheKey(scope, query);
    auto it = map_.find(k);
    if (it != map_.end()) {
      it->second = std::move(value);
      return;
    }
    if (order_.size() >= max_ && !order_.empty()) {
      map_.erase(order_.front());
      order_.pop_front();
    }
    order_.push_back(k);
    map_[k] = std::move(value);
  }

  size_t size() const { return map_.size(); }

private:
  size_t max_;
  std::unordered_map<std::string, std::string> map_;
  std::list<std::string> order_;
};

} // namespace runtime_opt
} // namespace phoenix
