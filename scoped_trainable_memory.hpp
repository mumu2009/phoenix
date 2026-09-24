/* scoped_trainable_memory.hpp - one live trainable bucket per MemoryScope

   Anything that writes, trains, or updates hidden state is stored here.
   Cold / read-only recall (official GNN ingest, experience JSON, CCM)
   stays on the shared stores and is not owned by this registry. */
#pragma once

#include "active_inference.hpp"
#include "concept_matrix.hpp"
#include "hierarchical_memory.hpp"
#include "memory_scope.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace phoenix {
namespace memory {

struct RecurrentState {
  std::vector<float> rnnHidden;
  std::vector<float> lstmHidden;
  std::vector<float> lstmCell;
  int messageCount{0};
  std::string lastMode;
};

struct GnnEdgeDelta {
  std::string from;
  std::string to;
  double weight{0.0};
  int direction{0};
};

class GnnOnlineOverlay {
 public:
  void add(const std::string &from, const std::string &to, double weight,
           int direction = 0) {
    if (from.empty() || to.empty() || from == to)
      return;
    const std::string k = from + '\0' + to;
    auto it = edges_.find(k);
    if (it == edges_.end()) {
      GnnEdgeDelta e;
      e.from = from;
      e.to = to;
      e.weight = weight;
      e.direction = direction;
      edges_.emplace(k, std::move(e));
      return;
    }
    it->second.weight += weight;
    it->second.direction = direction;
  }

  double weight(const std::string &from, const std::string &to) const {
    const std::string k = from + '\0' + to;
    auto it = edges_.find(k);
    return it == edges_.end() ? 0.0 : it->second.weight;
  }

  std::vector<GnnEdgeDelta> all() const {
    std::vector<GnnEdgeDelta> out;
    out.reserve(edges_.size());
    for (const auto &kv : edges_)
      out.push_back(kv.second);
    return out;
  }

  bool empty() const { return edges_.empty(); }
  size_t size() const { return edges_.size(); }
  void clear() { edges_.clear(); }

 private:
  std::unordered_map<std::string, GnnEdgeDelta> edges_;
};

struct DialogOverlayEntry {
  std::string signature;
  std::string question;
  std::string reply;
  std::vector<std::string> memes;
  double scoreHint{0.0};
};

class DialogOverlay {
 public:
  void remember(const std::string &signature, const std::string &question,
                const std::string &reply, const std::vector<std::string> &memes,
                double scoreHint) {
    DialogOverlayEntry e;
    e.signature = signature;
    e.question = question.substr(0, 800);
    e.reply = reply.substr(0, 1200);
    e.memes = memes;
    e.scoreHint = scoreHint;
    entries_.push_back(std::move(e));
    if (entries_.size() > 256)
      entries_.erase(entries_.begin());
  }

  const std::vector<DialogOverlayEntry> &entries() const { return entries_; }
  bool empty() const { return entries_.empty(); }
  void clear() { entries_.clear(); }

 private:
  std::vector<DialogOverlayEntry> entries_;
};

struct ResidualStats {
  double ema{0.0};
  double last{0.0};
  int64_t count{0};
};

/**
 * Per-scope live trainable state.  ConceptMatrix / HierarchicalMemory are
 * not copyable (mutex); the registry holds unique_ptr buckets.
 */
struct TrainableMemoryBucket {
  RecurrentState recurrent;
  GnnOnlineOverlay gnn;
  DialogOverlay dialog;
  conceptmatrix::ConceptMatrix conceptMatrix;
  HierarchicalMemory hier;
  std::string graphHint;
  std::string gnnSummary;
  std::vector<float> agiLatent;
  std::string lastAgiAction;
  std::unique_ptr<phoenix::agi::ActiveInferenceController> agi;
  ResidualStats residual;
  std::string lastPressureSource;
  /* Last benefit/harm action bias produced by an iterate() under this scope.
     The manager-level lastBenefitHarmBias_ is only a process-wide mirror of
     the most recent tick; per-scope history lives here. */
  std::string lastBenefitHarmBias;
  uint64_t createdAtMs{0};
  uint64_t lastAccessMs{0};
};

class ScopedTrainableMemory {
 public:
  static ScopedTrainableMemory &instance() {
    static ScopedTrainableMemory g;
    return g;
  }

  /** Clone the process AGI template into this scope on first use. */
  phoenix::agi::ActiveInferenceController &agiFor(
      const MemoryScope &scope,
      const phoenix::agi::ActiveInferenceController &tmpl) {
    const std::string k = scope.key();
    std::lock_guard<std::mutex> lock(mu_);
    auto it = live_.find(k);
    if (it == live_.end()) {
      auto bucket = std::make_unique<TrainableMemoryBucket>();
      it = live_.emplace(k, std::move(bucket)).first;
    }
    if (!it->second->agi)
      it->second->agi =
          std::make_unique<phoenix::agi::ActiveInferenceController>(tmpl);
    return *it->second->agi;
  }

  TrainableMemoryBucket &hot(const MemoryScope &scope) {
    const std::string k = scope.key();
    std::lock_guard<std::mutex> lock(mu_);
    auto it = live_.find(k);
    if (it == live_.end()) {
      auto bucket = std::make_unique<TrainableMemoryBucket>();
      it = live_.emplace(k, std::move(bucket)).first;
    }
    return *it->second;
  }

  const TrainableMemoryBucket *peek(const MemoryScope &scope) const {
    const std::string k = scope.key();
    std::lock_guard<std::mutex> lock(mu_);
    auto it = live_.find(k);
    return it == live_.end() ? nullptr : it->second.get();
  }

  bool hasLive(const MemoryScope &scope) const {
    std::lock_guard<std::mutex> lock(mu_);
    return live_.count(scope.key()) != 0;
  }

  /** Drop live state.  Optionally keep a snapshot in the archive (capped). */
  void release(const MemoryScope &scope, bool keepArchive = true) {
    const std::string k = scope.key();
    std::lock_guard<std::mutex> lock(mu_);
    auto it = live_.find(k);
    if (it == live_.end())
      return;
    if (keepArchive) {
      archived_[k] = std::move(it->second);
      if (archived_.size() > 64) {
        archived_.erase(archived_.begin());
      }
    }
    live_.erase(it);
  }

  void releaseChat(const std::string &sessionId) {
    release(parseMemoryScope(sessionId), false);
    release(makeChatScope(sessionId), false);
  }

  void releaseMission(const std::string &missionId) {
    release(makeMissionScope(missionId), true);
  }

  size_t liveCount() const {
    std::lock_guard<std::mutex> lock(mu_);
    return live_.size();
  }

  /** Test helper: wipe every bucket (does not touch cold stores). */
  void resetForTest() {
    std::lock_guard<std::mutex> lock(mu_);
    live_.clear();
    archived_.clear();
  }

  nlohmann::json status() const {
    std::lock_guard<std::mutex> lock(mu_);
    nlohmann::json keys = nlohmann::json::array();
    for (const auto &kv : live_)
      keys.push_back(kv.first);
    return nlohmann::json{{"live", live_.size()},
                          {"archived", archived_.size()},
                          {"keys", std::move(keys)}};
  }

 private:
  ScopedTrainableMemory() = default;
  mutable std::mutex mu_;
  std::unordered_map<std::string, std::unique_ptr<TrainableMemoryBucket>> live_;
  std::unordered_map<std::string, std::unique_ptr<TrainableMemoryBucket>>
      archived_;
};

}  // namespace memory
}  // namespace phoenix
