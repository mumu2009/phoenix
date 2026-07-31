/* hierarchical_memory.cpp - Hierarchical memory tier wrapper
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version. */

#include "hierarchical_memory.hpp"

#include <algorithm>
#include <chrono>

namespace phoenix {
namespace memory {

int64_t HierarchicalMemory::nowMs() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

void HierarchicalMemory::prune(std::map<std::string, MemorySlot> &tier) {
  for (auto it = tier.begin(); it != tier.end();) {
    if (it->second.accessCount == 0 &&
        (nowMs() - it->second.createdMs) > cfg_.shortTermTtlMs) {
      it = tier.erase(it);
    } else {
      ++it;
    }
  }
}

MemorySlot &HierarchicalMemory::touchLocked(const std::string &key) {
  auto it = working_.find(key);
  if (it != working_.end()) {
    it->second.lastAccessMs = nowMs();
    ++it->second.accessCount;
    return it->second;
  }
  it = shortTerm_.find(key);
  if (it != shortTerm_.end()) {
    it->second.lastAccessMs = nowMs();
    ++it->second.accessCount;
    if (it->second.activation >= cfg_.promotionThreshold ||
        it->second.accessCount > 1) {
      working_[key] = std::move(it->second);
      shortTerm_.erase(it);
      return working_[key];
    }
    return it->second;
  }
  it = longTerm_.find(key);
  if (it != longTerm_.end()) {
    it->second.lastAccessMs = nowMs();
    ++it->second.accessCount;
    if (it->second.activation >= cfg_.promotionThreshold ||
        it->second.accessCount > 2) {
      shortTerm_[key] = std::move(it->second);
      longTerm_.erase(it);
      return shortTerm_[key];
    }
    return it->second;
  }
  // Should not reach if get/put called with valid key.
  static MemorySlot empty;
  return empty;
}

void HierarchicalMemory::put(const std::string &key,
                             const nlohmann::json &value,
                             double activation) {
  std::lock_guard<std::mutex> lock(mu_);
  int64_t t = nowMs();
  MemorySlot slot;
  slot.key = key;
  slot.value = value;
  slot.createdMs = t;
  slot.lastAccessMs = t;
  slot.activation = activation;

  auto it = working_.find(key);
  if (it != working_.end()) {
    slot.accessCount = it->second.accessCount + 1;
    working_[key] = std::move(slot);
    return;
  }
  it = shortTerm_.find(key);
  if (it != shortTerm_.end()) {
    slot.accessCount = it->second.accessCount + 1;
    shortTerm_[key] = std::move(slot);
    return;
  }
  it = longTerm_.find(key);
  if (it != longTerm_.end()) {
    slot.accessCount = it->second.accessCount + 1;
    longTerm_[key] = std::move(slot);
    return;
  }

  if (activation >= cfg_.promotionThreshold && working_.size() < cfg_.workingCapacity) {
    working_[key] = std::move(slot);
  } else if (shortTerm_.size() < cfg_.shortTermCapacity) {
    shortTerm_[key] = std::move(slot);
  } else {
    longTerm_[key] = std::move(slot);
  }
}

std::optional<nlohmann::json> HierarchicalMemory::get(const std::string &key) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = working_.find(key);
  if (it != working_.end()) {
    it->second.lastAccessMs = nowMs();
    ++it->second.accessCount;
    return it->second.value;
  }
  it = shortTerm_.find(key);
  if (it != shortTerm_.end()) {
    it->second.lastAccessMs = nowMs();
    ++it->second.accessCount;
    if (it->second.activation >= cfg_.promotionThreshold ||
        it->second.accessCount > 1) {
      working_[key] = std::move(it->second);
      shortTerm_.erase(it);
      return working_[key].value;
    }
    return it->second.value;
  }
  it = longTerm_.find(key);
  if (it != longTerm_.end()) {
    it->second.lastAccessMs = nowMs();
    ++it->second.accessCount;
    if (it->second.activation >= cfg_.promotionThreshold ||
        it->second.accessCount > 2) {
      shortTerm_[key] = std::move(it->second);
      longTerm_.erase(it);
      return shortTerm_[key].value;
    }
    return it->second.value;
  }
  return std::nullopt;
}

void HierarchicalMemory::del(const std::string &key) {
  std::lock_guard<std::mutex> lock(mu_);
  working_.erase(key);
  shortTerm_.erase(key);
  longTerm_.erase(key);
}

std::vector<std::pair<std::string, nlohmann::json>>
HierarchicalMemory::query(const std::string &prefix, size_t topK) const {
  std::lock_guard<std::mutex> lock(mu_);
  std::vector<std::pair<std::string, nlohmann::json>> out;
  auto collect = [&](const std::map<std::string, MemorySlot> &tier) {
    for (const auto &kv : tier) {
      if (!prefix.empty() && kv.first.rfind(prefix, 0) != 0)
        continue;
      out.push_back({kv.first, kv.second.value});
    }
  };
  collect(working_);
  collect(shortTerm_);
  collect(longTerm_);
  std::sort(out.begin(), out.end(),
            [&](const auto &a, const auto &b) {
              // Higher activation / more recent first.
              auto itA = working_.find(a.first);
              if (itA == working_.end())
                itA = shortTerm_.find(a.first);
              if (itA == shortTerm_.end())
                itA = longTerm_.find(a.first);
              auto itB = working_.find(b.first);
              if (itB == working_.end())
                itB = shortTerm_.find(b.first);
              if (itB == shortTerm_.end())
                itB = longTerm_.find(b.first);
              double scoreA = (itA == longTerm_.end() ? 0.0 : itA->second.activation) +
                              1e-6 * itA->second.lastAccessMs;
              double scoreB = (itB == longTerm_.end() ? 0.0 : itB->second.activation) +
                              1e-6 * itB->second.lastAccessMs;
              return scoreA > scoreB;
            });
  if (out.size() > topK)
    out.resize(topK);
  return out;
}

void HierarchicalMemory::rebalance() {
  std::lock_guard<std::mutex> lock(mu_);
  int64_t now = nowMs();
  auto moveIfCold = [&](std::map<std::string, MemorySlot> &src,
                        std::map<std::string, MemorySlot> &dst,
                        size_t capacity, int64_t ttlMs, double activationFloor) {
    for (auto it = src.begin(); it != src.end();) {
      bool cold = it->second.activation < activationFloor;
      bool expired = (now - it->second.lastAccessMs) > ttlMs;
      bool overCapacity = src.size() > capacity;
      if (cold || expired || overCapacity) {
        if (dst.size() < capacity * 2)
          dst[it->first] = std::move(it->second);
        it = src.erase(it);
      } else {
        ++it;
      }
    }
  };

  // Demote from working to short-term.
  moveIfCold(working_, shortTerm_, cfg_.workingCapacity, cfg_.shortTermTtlMs,
             cfg_.promotionThreshold);

  // Demote from short-term to long-term.
  moveIfCold(shortTerm_, longTerm_, cfg_.shortTermCapacity,
             cfg_.longTermTtlMs, 0.0);

  // Long-term cleanup.
  prune(longTerm_);
}

nlohmann::json HierarchicalMemory::toJson() const {
  std::lock_guard<std::mutex> lock(mu_);
  nlohmann::json j;
  j["working"] = nlohmann::json::array();
  j["shortTerm"] = nlohmann::json::array();
  j["longTerm"] = nlohmann::json::array();
  auto dump = [](const std::map<std::string, MemorySlot> &tier,
                 nlohmann::json &arr) {
    for (const auto &kv : tier) {
      arr.push_back(nlohmann::json{{"key", kv.first}, {"value", kv.second.value}});
    }
  };
  dump(working_, j["working"]);
  dump(shortTerm_, j["shortTerm"]);
  dump(longTerm_, j["longTerm"]);
  return j;
}

void HierarchicalMemory::fromJson(const nlohmann::json &j) {
  std::lock_guard<std::mutex> lock(mu_);
  working_.clear();
  shortTerm_.clear();
  longTerm_.clear();
  // Restore key/value pairs into short-term as a neutral starting point.
  auto restore = [&](const nlohmann::json &arr) {
    if (!arr.is_array())
      return;
    for (const auto &v : arr) {
      std::string key = v.value("key", v.dump());
      nlohmann::json value = v.value("value", v);
      MemorySlot slot;
      slot.key = key;
      slot.value = std::move(value);
      slot.createdMs = nowMs();
      slot.lastAccessMs = slot.createdMs;
      shortTerm_[key] = std::move(slot);
    }
  };
  if (j.contains("working"))
    restore(j["working"]);
  if (j.contains("shortTerm"))
    restore(j["shortTerm"]);
  if (j.contains("longTerm"))
    restore(j["longTerm"]);
}

size_t HierarchicalMemory::size() const {
  std::lock_guard<std::mutex> lock(mu_);
  return working_.size() + shortTerm_.size() + longTerm_.size();
}

}  // namespace memory
}  // namespace phoenix
