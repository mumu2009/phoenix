/* mission_lifecycle.cpp - Mission layer implementation
   Copyright (C) 2026 079 Project */

#include "mission_lifecycle.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

namespace phoenix {
namespace mission {

namespace {
uint64_t sysNowMs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

float clampF(float v, float lo, float hi) {
  return std::max(lo, std::min(hi, v));
}

// Gaussian perturbation of one heritable scalar.
float mutateScalar(float x, float rate, float lo, float hi, std::mt19937 &rng) {
  if (rate <= 0.0f) return clampF(x, lo, hi);
  std::normal_distribution<float> d(0.0f, rate);
  return clampF(x + d(rng), lo, hi);
}

/* v8.x C3 lineage: FNV-1a fingerprint of the genome JSON (16 hex chars). */
std::string genomeFingerprint(const MissionGenome &g) {
  const std::string raw = g.toJson().dump();
  uint64_t h = 1469598103934665603ULL;
  for (const unsigned char c : raw) {
    h ^= c;
    h *= 1099511628211ULL;
  }
  char buf[24];
  std::snprintf(buf, sizeof(buf), "%016llx",
                static_cast<unsigned long long>(h));
  return std::string(buf);
}
}  // namespace

nlohmann::json Mission::toJson() const {
  return {{"id", id},
          {"goal", goal},
          {"deadlineSec", deadlineSec},
          {"painGainPerSec", painGainPerSec},
          {"maxPain", maxPain},
          {"pressureMode", pressureMode},
          {"pressureHorizonSec", pressureHorizonSec},
          {"pressureTauSec", pressureTauSec},
          {"pressureExpr", pressureExpr},
          {"deliverable", deliverable},
          {"state", static_cast<int>(state)},
          {"startMs", startMs},
          {"endMs", endMs}};
}

Mission Mission::fromJson(const nlohmann::json &j) {
  Mission m;
  if (!j.is_object()) return m;
  if (j.contains("id") && j["id"].is_string()) m.id = j["id"].get<std::string>();
  if (j.contains("goal") && j["goal"].is_string()) m.goal = j["goal"].get<std::string>();
  if (j.contains("deadlineSec") && j["deadlineSec"].is_number()) m.deadlineSec = j["deadlineSec"].get<double>();
  if (j.contains("painGainPerSec") && j["painGainPerSec"].is_number()) m.painGainPerSec = j["painGainPerSec"].get<float>();
  if (j.contains("maxPain") && j["maxPain"].is_number()) m.maxPain = j["maxPain"].get<float>();
  if (j.contains("pressureMode") && j["pressureMode"].is_string()) {
    const std::string mode = j["pressureMode"].get<std::string>();
    if (mode == "linear" || mode == "logarithmic" || mode == "expression" ||
        mode == "asymptotic") {
      m.pressureMode = mode;
    } else {
      m.pressureMode = "asymptotic";
    }
  }
  if (j.contains("pressureHorizonSec") && j["pressureHorizonSec"].is_number())
    m.pressureHorizonSec = j["pressureHorizonSec"].get<double>();
  if (j.contains("pressureTauSec") && j["pressureTauSec"].is_number())
    m.pressureTauSec = j["pressureTauSec"].get<double>();
  if (j.contains("pressureExpr") && j["pressureExpr"].is_string())
    m.pressureExpr = j["pressureExpr"].get<std::string>();
  if (j.contains("deliverable") && j["deliverable"].is_string())
    m.deliverable = j["deliverable"].get<std::string>();
  if (j.contains("state") && j["state"].is_number_integer()) {
    const int s = j["state"].get<int>();
    if (s >= 0 && s <= 3) m.state = static_cast<MissionState>(s);
  }
  if (j.contains("startMs") && j["startMs"].is_number()) m.startMs = j["startMs"].get<uint64_t>();
  if (j.contains("endMs") && j["endMs"].is_number()) m.endMs = j["endMs"].get<uint64_t>();
  return m;
}

MissionGenome MissionGenome::mutate(float rate, std::mt19937 &rng) const {
  MissionGenome g = *this;
  // Temperament baseline: PAD disposition.
  g.profile.baselineValence = mutateScalar(g.profile.baselineValence, rate, -1.0f, 1.0f, rng);
  g.profile.baselineArousal = mutateScalar(g.profile.baselineArousal, rate, -1.0f, 1.0f, rng);
  g.profile.baselineDominance = mutateScalar(g.profile.baselineDominance, rate, -1.0f, 1.0f, rng);
  g.profile.temperamentStrength = mutateScalar(g.profile.temperamentStrength, rate, 0.0f, 2.0f, rng);
  // Risk attitude and anticipatory gain.
  g.profile.riskAversion = mutateScalar(g.profile.riskAversion, rate * 0.25f, 0.2f, 3.0f, rng);
  g.profile.anticipatoryGain = mutateScalar(g.profile.anticipatoryGain, rate * 0.25f, 0.0f, 3.0f, rng);
  // Per-sensation tuning: gains and setpoints.
  for (auto &kv : g.profile.sensationTuning) {
    kv.second.gain = mutateScalar(kv.second.gain, rate * 0.5f, 0.0f, 5.0f, rng);
    kv.second.setpoint = mutateScalar(kv.second.setpoint, rate, -1.0f, 1.0f, rng);
    kv.second.halfLifeSec = mutateScalar(kv.second.halfLifeSec, rate * 10.0f, 1.0f, 3600.0f, rng);
  }
  // Custom instinct table: drives and sensitivities.
  for (auto &inst : g.profile.instincts) {
    inst.activation = mutateScalar(inst.activation, rate, 0.0f, 1.0f, rng);
    inst.benefitWeight = mutateScalar(inst.benefitWeight, rate, 0.0f, 1.0f, rng);
    inst.harmWeight = mutateScalar(inst.harmWeight, rate, 0.0f, 1.0f, rng);
  }
  // Learning rate of the value learner.
  g.learningRate = mutateScalar(g.learningRate, rate * 0.05f, 0.001f, 0.5f, rng);
  return g;
}

nlohmann::json MissionGenome::toJson() const {
  return {{"profile", profile.toJson()}, {"learningRate", learningRate}};
}

MissionGenome MissionGenome::fromJson(const nlohmann::json &j) {
  MissionGenome g;
  if (!j.is_object()) return g;
  if (j.contains("profile")) g.profile = subconscious::SubconsciousProfile::fromJson(j["profile"]);
  if (j.contains("learningRate") && j["learningRate"].is_number()) {
    g.learningRate = j["learningRate"].get<float>();
  }
  return g;
}

nlohmann::json MissionChild::toJson() const {
  return {{"id", id},
          {"genome", genome.toJson()},
          {"goal", goal},
          {"subgoal", subgoal},
          {"parentId", parentId},
          {"depth", depth},
          {"bornMs", bornMs},
          {"generation", generation},
          {"done", done}};
}

MissionLifecycle::MissionLifecycle()
    : rng_(static_cast<uint32_t>(sysNowMs() ^ 0x4D495353ULL)) {}

uint64_t MissionLifecycle::nowMs() const { return sysNowMs(); }

void MissionLifecycle::assign(const Mission &m) {
  assign(m, MissionGenome{});
}

void MissionLifecycle::assign(const Mission &m, const MissionGenome &genome) {
  std::lock_guard<std::mutex> lock(mu_);
  mission_ = m;
  mission_.state = MissionState::Running;
  mission_.startMs = sysNowMs();
  mission_.endMs = 0;
  genome_ = genome;
  children_.clear(); /* fresh lineage: a new mission does not inherit stale
                        successors from a previous one */
  ++generation_;
}

Mission MissionLifecycle::mission() const {
  std::lock_guard<std::mutex> lock(mu_);
  return mission_;
}

bool MissionLifecycle::active() const {
  std::lock_guard<std::mutex> lock(mu_);
  return mission_.state == MissionState::Running;
}

float MissionLifecycle::pressureNow() const {
  std::lock_guard<std::mutex> lock(mu_);
  return mission_.pressure(sysNowMs());
}

void MissionLifecycle::appendDeliverable(const std::string &text) {
  if (text.empty()) return;
  std::lock_guard<std::mutex> lock(mu_);
  if (mission_.state != MissionState::Running) return;
  constexpr size_t kMaxDeliverableBytes = 4u * 1024u * 1024u; /* 4 MiB: long tutorials */
  mission_.deliverable += text;
  if (mission_.deliverable.size() > kMaxDeliverableBytes)
    mission_.deliverable = mission_.deliverable.substr(
        mission_.deliverable.size() - kMaxDeliverableBytes);
}

void MissionLifecycle::markComplete() {
  std::lock_guard<std::mutex> lock(mu_);
  if (mission_.state != MissionState::Running) return;
  mission_.state = MissionState::Completed;
  mission_.endMs = sysNowMs();
  ++completeCount_;
  /* v8.x C3: the outcome feeds the lineage (auditable history).  Locked
     variant - recordLineage() is the public entry for unlocked callers. */
  const uint64_t elapsed =
      mission_.endMs > mission_.startMs ? mission_.endMs - mission_.startMs : 0;
  const double coverage = std::min(
      1.0, static_cast<double>(mission_.deliverable.size()) / 5000.0);
  lineage_.push_back({genomeFingerprint(genome_), elapsed, coverage});
  if (lineage_.size() > 1000) lineage_.erase(lineage_.begin());
}

void MissionLifecycle::markFailed() {
  std::lock_guard<std::mutex> lock(mu_);
  if (mission_.state != MissionState::Running) return;
  mission_.state = MissionState::Failed;
  mission_.endMs = sysNowMs();
}

bool MissionLifecycle::amendGoal(const std::string &newGoal) {
  std::lock_guard<std::mutex> lock(mu_);
  if (mission_.state != MissionState::Running || newGoal.empty()) return false;
  mission_.goal = newGoal;
  return true;
}

MissionGenome MissionLifecycle::spawnChild(const MissionGenome &parent,
                                           float mutationRate) {
  std::lock_guard<std::mutex> lock(mu_);
  ++spawnCount_;
  return parent.mutate(mutationRate, rng_);
}

MissionChild MissionLifecycle::recordChild(const MissionGenome &parent,
                                             float mutationRate,
                                             const std::string &subgoal,
                                             const std::string &parentId) {
  std::lock_guard<std::mutex> lock(mu_);
  if (children_.size() >= maxReplicas_) {
    throw std::runtime_error("maxReplicas reached (" +
                             std::to_string(maxReplicas_) +
                             "); replication is bounded");
  }
  /* task-tree depth: parentId empty = spawned by the root mission;
     otherwise depth = parent depth + 1.  0 = unlimited (user-decided). */
  int depth = 0;
  if (!parentId.empty()) {
    for (const auto &box : children_) {
      if (box.id == parentId) {
        depth = box.depth + 1;
        break;
      }
    }
    if (maxReplicaDepth_ > 0 && depth > static_cast<int>(maxReplicaDepth_)) {
      throw std::runtime_error("maxReplicaDepth reached (" +
                               std::to_string(maxReplicaDepth_) +
                               "); recursion is capped by config");
    }
  }
  /* v8.x C3: the lineage shapes the mutation step (when enabled).
     mu_ is ALREADY held here - use the Locked variant (the locking one
     would self-deadlock on this same mutex). */
  const float effRate =
      evolutionEnabled_ ? effectiveMutationRateLocked(mutationRate) : mutationRate;
  const MissionGenome child = parent.mutate(effRate, rng_);
  ++spawnCount_;
  MissionChild rec;
  rec.id = "child-" + std::to_string(generation_) + "-" + std::to_string(children_.size());
  rec.genome = child;
  rec.subgoal = subgoal;
  rec.parentId = parentId;
  rec.depth = depth;
  /* bounded helper semantics: an empty request - or one that just echoes the
     whole parent goal - becomes "assist with: <parent goal>".  The parent
     can never completely delegate its own work to a box. */
  const auto trim = [](const std::string &s) -> std::string {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return std::string();
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
  };
  const std::string req = trim(subgoal);
  if (req.empty() || req == trim(mission_.goal))
    rec.goal = "assist with: " + mission_.goal;
  else
    rec.goal = req;
  rec.bornMs = sysNowMs();
  rec.generation = static_cast<int>(generation_);
  children_.push_back(rec);
  return rec;
}

MissionGenome MissionLifecycle::replicate(float mutationRate) {
  return replicate(mutationRate, std::string());
}

MissionGenome MissionLifecycle::replicate(float mutationRate,
                                           const std::string &subgoal) {
  return recordChild(genome_, mutationRate, subgoal).genome;
}

size_t MissionLifecycle::maxReplicas() const {
  std::lock_guard<std::mutex> lock(mu_);
  return maxReplicas_;
}

void MissionLifecycle::setMaxReplicas(size_t n) {
  std::lock_guard<std::mutex> lock(mu_);
  maxReplicas_ = n;
}

size_t MissionLifecycle::maxReplicaDepth() const {
  std::lock_guard<std::mutex> lock(mu_);
  return maxReplicaDepth_;
}

void MissionLifecycle::setMaxReplicaDepth(size_t n) {
  std::lock_guard<std::mutex> lock(mu_);
  maxReplicaDepth_ = n;
}

std::vector<MissionChild> MissionLifecycle::children() const {
  std::lock_guard<std::mutex> lock(mu_);
  return children_;
}

bool MissionLifecycle::markChildDone(const std::string &childId) {
  std::lock_guard<std::mutex> lock(mu_);
  for (auto &box : children_) {
    if (box.id == childId) {
      box.done = true;
      return true;
    }
  }
  return false;
}

void MissionLifecycle::recordLineage(uint64_t completionMs, double coverage) {
  std::lock_guard<std::mutex> lock(mu_);
  lineage_.push_back({genomeFingerprint(genome_), completionMs, coverage});
  if (lineage_.size() > 1000) lineage_.erase(lineage_.begin());
}

nlohmann::json MissionLifecycle::lineage() const {
  std::lock_guard<std::mutex> lock(mu_);
  nlohmann::json arr = nlohmann::json::array();
  for (const auto &e : lineage_) {
    arr.push_back(nlohmann::json{{"fingerprint", e.fingerprint},
                                 {"completionMs", e.completionMs},
                                 {"coverage", e.coverage}});
  }
  return nlohmann::json{{"ok", true},
                        {"result", nlohmann::json{{"enabled", evolutionEnabled_},
                                                  {"entries", arr},
                                                  {"count", lineage_.size()}}}};
}

void MissionLifecycle::resetLineage() {
  std::lock_guard<std::mutex> lock(mu_);
  lineage_.clear();
}

void MissionLifecycle::setEvolutionEnabled(bool enabled) {
  std::lock_guard<std::mutex> lock(mu_);
  evolutionEnabled_ = enabled;
}

bool MissionLifecycle::evolutionEnabled() const {
  std::lock_guard<std::mutex> lock(mu_);
  return evolutionEnabled_;
}

/* Softmax-weighted step factor: faster completions with higher coverage
   shrink the mutation step (1 -> 0.5), slow/poor ones widen it (-> 2.0).
   No history -> 1.0 (unchanged).  Used only when evolution is enabled. */
float MissionLifecycle::effectiveMutationRateLocked(float baseRate) const {
  if (lineage_.empty()) return baseRate;
  double sumW = 0.0, sumS = 0.0;
  for (const auto &e : lineage_) {
    const double speed =
        e.completionMs > 0 ? 1.0 / (1.0 + static_cast<double>(e.completionMs) / 1000.0) : 0.0;
    const double s = std::max(0.0, 0.6 * speed + 0.4 * e.coverage);
    sumW += s;
    sumS += s * s;
  }
  if (sumW <= 0.0) return baseRate;
  const double avg = sumS / sumW;
  /* avg in [0,1]; 1 = great history -> shrink step; 0 = poor -> widen */
  const double factor = 2.0 - 1.5 * avg;
  const float clamped =
      static_cast<float>(std::max(0.5, std::min(2.0, factor)));
  return baseRate * clamped;
}

float MissionLifecycle::effectiveMutationRate(float baseRate) const {
  std::lock_guard<std::mutex> lock(mu_);
  return effectiveMutationRateLocked(baseRate);
}

nlohmann::json MissionLifecycle::statsLocked() const {
  /* completionTimeMs is the SELECTION signal for the human supervisor: the
     model exists to serve human needs, so the decision to spawn a mutated
     retry (or to keep a genome) is made OUTSIDE this process, from this
     number.  There is deliberately no in-process fitness field, no elite
     retention and no autonomous cross-generation loop. */
  const bool terminal = (mission_.state == MissionState::Completed ||
                         mission_.state == MissionState::Failed);
  const int64_t completionTimeMs =
      (terminal && mission_.endMs >= mission_.startMs)
          ? static_cast<int64_t>(mission_.endMs - mission_.startMs)
          : -1;
  nlohmann::json childArr = nlohmann::json::array();
  for (const auto &c : children_) childArr.push_back(c.toJson());
  return {{"generations", generation_},
          {"spawns", spawnCount_},
          {"completions", completeCount_},
          {"mission", mission_.toJson()},
          {"pressure", mission_.pressure(sysNowMs())},
          {"completionTimeMs", completionTimeMs},
          {"maxReplicas", maxReplicas_},
          {"children", childArr}};
}

nlohmann::json MissionLifecycle::stats() const {
  std::lock_guard<std::mutex> lock(mu_);
  return statsLocked();
}

nlohmann::json MissionLifecycle::toJson() const {
  std::lock_guard<std::mutex> lock(mu_);
  nlohmann::json out = statsLocked();
  out["genome"] = genome_.toJson();
  return out;
}

void MissionLifecycle::fromJson(const nlohmann::json &j) {
  std::lock_guard<std::mutex> lock(mu_);
  if (j.contains("mission")) mission_ = Mission::fromJson(j["mission"]);
  if (j.contains("spawns") && j["spawns"].is_number()) spawnCount_ = j["spawns"].get<size_t>();
  if (j.contains("completions") && j["completions"].is_number()) completeCount_ = j["completions"].get<size_t>();
  if (j.contains("generations") && j["generations"].is_number()) generation_ = j["generations"].get<size_t>();
  if (j.contains("maxReplicas") && j["maxReplicas"].is_number()) maxReplicas_ = j["maxReplicas"].get<size_t>();
  if (j.contains("genome") && j["genome"].is_object()) {
    genome_ = MissionGenome::fromJson(j["genome"]);
  }
  if (j.contains("children") && j["children"].is_array()) {
    children_.clear();
    for (const auto &ch : j["children"]) {
      if (!ch.is_object()) continue;
      MissionChild rec;
      rec.id = ch.value("id", std::string());
      if (ch.contains("genome")) rec.genome = MissionGenome::fromJson(ch["genome"]);
      rec.goal = ch.value("goal", std::string());
      rec.subgoal = ch.value("subgoal", std::string());
      rec.parentId = ch.value("parentId", std::string());
      rec.depth = ch.value("depth", 0);
      rec.done = ch.value("done", false);
      rec.bornMs = ch.value("bornMs", 0ull);
      rec.generation = ch.value("generation", 0);
      if (!rec.id.empty()) children_.push_back(std::move(rec));
    }
  }
}

}  // namespace mission
}  // namespace phoenix
