/* mission_lifecycle.cpp - Mission layer implementation
   Copyright (C) 2026 079 Project */

#include "mission_lifecycle.hpp"

#include <algorithm>
#include <chrono>

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
}  // namespace

nlohmann::json Mission::toJson() const {
  return {{"id", id},
          {"goal", goal},
          {"deadlineSec", deadlineSec},
          {"painGainPerSec", painGainPerSec},
          {"maxPain", maxPain},
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
          {"bornMs", bornMs},
          {"generation", generation}};
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

void MissionLifecycle::markComplete() {
  std::lock_guard<std::mutex> lock(mu_);
  if (mission_.state != MissionState::Running) return;
  mission_.state = MissionState::Completed;
  mission_.endMs = sysNowMs();
  ++completeCount_;
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

MissionGenome MissionLifecycle::replicate(float mutationRate) {
  std::lock_guard<std::mutex> lock(mu_);
  if (children_.size() >= maxReplicas_) {
    throw std::runtime_error("maxReplicas reached (" +
                             std::to_string(maxReplicas_) +
                             "); replication is bounded");
  }
  const MissionGenome child = genome_.mutate(mutationRate, rng_);
  ++spawnCount_;
  MissionChild rec;
  rec.id = "child-" + std::to_string(generation_) + "-" + std::to_string(children_.size());
  rec.genome = child;
  rec.goal = mission_.goal;
  rec.bornMs = sysNowMs();
  rec.generation = static_cast<int>(generation_);
  children_.push_back(rec);
  return child;
}

size_t MissionLifecycle::maxReplicas() const {
  std::lock_guard<std::mutex> lock(mu_);
  return maxReplicas_;
}

void MissionLifecycle::setMaxReplicas(size_t n) {
  std::lock_guard<std::mutex> lock(mu_);
  maxReplicas_ = n;
}

std::vector<MissionChild> MissionLifecycle::children() const {
  std::lock_guard<std::mutex> lock(mu_);
  return children_;
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
      rec.bornMs = ch.value("bornMs", 0ull);
      rec.generation = ch.value("generation", 0);
      if (!rec.id.empty()) children_.push_back(std::move(rec));
    }
  }
}

}  // namespace mission
}  // namespace phoenix
