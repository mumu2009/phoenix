/* hierarchical_memory.hpp - Hierarchical memory tier wrapper
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version. */

#pragma once

#include <chrono>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace phoenix {
namespace memory {

/**
 * @brief Tiered memory slot.
 *
 * A slot is a typed memory unit with access-time metadata.  Tier placement
 * is determined by recency and access frequency rather than by a single
 * flag, allowing gradual promotion/demotion across working, short-term and
 * long-term tiers.
 */
struct MemorySlot {
  std::string key;
  nlohmann::json value;
  int64_t createdMs{0};
  int64_t lastAccessMs{0};
  uint64_t accessCount{0};
  double activation{0.0};  // affective / attentional weight
};

/**
 * @brief Hierarchical memory with three tiers.
 *
 *  - Working: recently accessed, high activation, small capacity.
 *  - ShortTerm: moderately recent, bounded by time or capacity.
 *  - LongTerm: durable storage, promoted from short-term on rehearsal.
 *
 * The wrapper keeps an in-memory hot cache and optionally mirrors writes to
 * a KeyValueStore backend when flush() is called.  It is used by the
 * CognitionAutonomyManager to carry multi-turn context and affective
 * modulators.
 */
class HierarchicalMemory {
 public:
  struct Config {
    size_t workingCapacity{64};
    size_t shortTermCapacity{512};
    int64_t shortTermTtlMs{10 * 60 * 1000};  // 10 minutes
    int64_t longTermTtlMs{24 * 60 * 60 * 1000};  // 24 hours
    double promotionThreshold{0.5};  // activation to move to next tier
  };

  HierarchicalMemory() = default;
  explicit HierarchicalMemory(const Config &cfg) : cfg_(cfg) {}

  /** Store or overwrite a slot in the appropriate tier. */
  void put(const std::string &key, const nlohmann::json &value,
           double activation = 0.0);

  /** Retrieve a slot; updates access statistics and may promote it. */
  std::optional<nlohmann::json> get(const std::string &key);

  /** Remove a slot from all tiers. */
  void del(const std::string &key);

  /** Query all slots whose key starts with prefix, ordered by activation. */
  std::vector<std::pair<std::string, nlohmann::json>> query(const std::string &prefix,
                                                            size_t topK = 16) const;

  /** Recalculate tier placement and prune expired/cold slots. */
  void rebalance();

  /** Convert current state to JSON (useful for snapshots or prompts). */
  nlohmann::json toJson() const;

  /** Load state from a snapshot. */
  void fromJson(const nlohmann::json &j);

  size_t size() const;

 private:
  mutable std::mutex mu_;
  Config cfg_;
  std::map<std::string, MemorySlot> working_;
  std::map<std::string, MemorySlot> shortTerm_;
  std::map<std::string, MemorySlot> longTerm_;

  int64_t nowMs() const;
  void prune(std::map<std::string, MemorySlot> &tier);
  MemorySlot &touchLocked(const std::string &key);
};

}  // namespace memory
}  // namespace phoenix
