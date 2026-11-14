/* mission_lifecycle.hpp - Meeseeks-style mission layer (goal pressure + reproduction)

   Formalises the "Meeseeks box" idea: an agent instance is born with ONE
   mission; pain grows with time since mission start (growth mode is user-
   configurable: asymptotic tanh by default, or linear / logarithmic /
   a free-form expression) and ends only when the mission is judged
   complete.  Because the agent's existing loop minimises accumulated pain
   (see doc/v7.0/mission_layer.md), the pressure forces shortest-time goal
   completion.  Asymptotic mode approaches maxPain but never saturates for
   finite t, avoiding pressure-driven hallucination / goal forgetting.

   The "outer layer" of the Meeseeks box is deliberately NOT a separate
   runtime layer (no cross-layer translation cost): it is a lifecycle state
   machine + a heritable genome object living inside the main system.
   spawnChild() mutates the genome (Gaussian perturbation of the heritable
   parameters) - heredity + variation - and the completion record is the
   selection signal (fitness = -completionTime): a (1+lambda) evolution
   strategy over agent instances.

   Theory references: allostatic urgency (Sterling 1988); accumulated free
   energy / prior preference for terminal states (Friston); time-minimisation
   MDP objective; evolution strategies (Beyer & Schwefel; Rechenberg).
*/
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <nlohmann/json.hpp>

#include "pressure_expr.hpp"
#include "subconscious_profile.hpp"

namespace phoenix {
namespace mission {

enum class MissionState { Idle = 0, Running = 1, Completed = 2, Failed = 3 };

/** @brief A single "Meeseeks mission". */
struct Mission {
  std::string id;
  std::string goal;              /*!< natural-language mission. */
  double deadlineSec{300.0};     /*!< time budget. */
  float painGainPerSec{0.01f};   /*!< urgency: pain growth rate (linear mode). */
  float maxPain{1.0f};
  /* Starvation floor: while the mission is Running and t>0, pressure never
     drops below this value (clamped to [0, Pmax)).  A near-zero pressure
     (e.g. asymptotic with tau=86400) made the first mission hours
     indistinguishable from "no mission" downstream (sampling weights and
     the Pain sensation both scale with pressure). */
  float pressureFloor{0.05f};
  /* pressureMode:
       "asymptotic" (default) = Pmax * tanh(t/tau) — approaches Pmax, never hits
       "linear"               = min(g*t, Pmax)
       "logarithmic"          = Pmax * ln(1+t)/ln(1+H)  (saturates at H)
       "expression"           = user formula in t (see pressureExpr); result
                                is clamped to [0, Pmax). */
  std::string pressureMode{"asymptotic"};
  double pressureHorizonSec{3600.0};       /*!< log mode horizon H (seconds). */
  double pressureTauSec{1800.0};           /*!< asymptotic tanh time-constant. */
  /* Default expression mirrors asymptotic: Pmax*tanh(t/tau).  Users may set
     any elementary formula (trig/hyperbolic/poly/exp/log/...); variable t is
     elapsed seconds; H, g, Pmax, tau are also bound. */
  std::string pressureExpr{"Pmax*tanh(t/tau)"};
  std::string deliverable;                 /*!< accumulated work product (LLM worker). */
  MissionState state{MissionState::Idle};
  uint64_t startMs{0};
  uint64_t endMs{0};
  /* Sum of this mission's own deliberation times (excluding queue/wait).
     Used by the (1+lambda) selection when >0; 0 = legacy wall clock. */
  uint64_t workMs{0};

  /** Elapsed seconds since start (0 if not started or clock skew). */
  double elapsedSec(uint64_t nowMs) const {
    if (state != MissionState::Running || nowMs <= startMs) {
      return 0.0;
    }
    return static_cast<double>(nowMs - startMs) / 1000.0;
  }

  /** Mission pressure (user-configurable growth mode).  Asymptotic default
      never reaches maxPain for finite t (open interval [0, Pmax)), so the
      agent keeps a usable gradient instead of saturating into panic. */
  float pressure(uint64_t nowMs) const {
    if (state != MissionState::Running) return 0.0f;
    const double t = elapsedSec(nowMs);
    if (t <= 0.0) return 0.0f;
    const double Pmax = static_cast<double>(maxPain);
    const double g = static_cast<double>(painGainPerSec);
    const double H = pressureHorizonSec >= 1.0 ? pressureHorizonSec : 1.0;
    const double tau = pressureTauSec >= 1.0 ? pressureTauSec : 1.0;
    double p = 0.0;
    if (pressureMode == "linear") {
      p = g * t;
      if (p > Pmax) p = Pmax;
    } else if (pressureMode == "logarithmic") {
      p = Pmax * std::log1p(t) / std::log1p(H);
      if (p > Pmax) p = Pmax;
    } else if (pressureMode == "expression") {
      std::unordered_map<std::string, double> vars{
          {"H", H}, {"g", g}, {"Pmax", Pmax}, {"tau", tau}};
      const auto ev = evalPressureExpr(pressureExpr, t, vars);
      if (!ev.ok || !std::isfinite(ev.value)) {
        /* Fail closed to asymptotic so a bad formula never stalls the loop. */
        p = Pmax * std::tanh(t / tau);
      } else {
        p = ev.value;
      }
      /* Open upper bound: approach Pmax but never claim saturation. */
      if (p < 0.0) p = 0.0;
      if (p >= Pmax) p = std::nextafter(Pmax, 0.0);
    } else {
      /* asymptotic (default): Pmax * tanh(t/tau) ∈ [0, Pmax) */
      p = Pmax * std::tanh(t / tau);
      if (p >= Pmax) p = std::nextafter(Pmax, 0.0);
    }
    if (p < 0.0) p = 0.0;
    /* Starvation floor: an unfinished mission must keep a non-zero urgency
       floor so downstream modulation never decays to "no mission". */
    double floor = static_cast<double>(pressureFloor);
    if (floor < 0.0) floor = 0.0;
    if (floor >= Pmax) floor = std::nextafter(Pmax, 0.0);
    if (p < floor) p = floor;
    return static_cast<float>(p);
  }

  nlohmann::json toJson() const;
  static Mission fromJson(const nlohmann::json &j);
};

/**
 * @brief The heritable genome of an agent instance.
 *
 * Heredity carries the subconscious profile (temperament, sensation tuning,
 * instincts, risk attitude) plus the learning rate.  Mutation is Gaussian
 * perturbation with clamping; completion time is the fitness.
 */
struct MissionGenome {
  subconscious::SubconsciousProfile profile;
  float learningRate{0.05f};
  /* Heritable per-tick generation budget.  The mission worker asks the
     language model for at most this many tokens per tick; the wall clock
     of a fixed deliverable therefore depends on it (more tokens per tick
     means fewer ticks and less repeated prompt prefill).  It is mutated
     in log space so the step is multiplicative, and clamped to
     [32, 1024].  Before 2026-10-26 the genome changed only the value
     learner's step size, which no generation ever read. */
  float tokensPerTick{256.0f};

  /** Gaussian mutation of all heritable scalars, rate = sigma (clamped). */
  MissionGenome mutate(float rate, std::mt19937 &rng) const;

  nlohmann::json toJson() const;
  static MissionGenome fromJson(const nlohmann::json &j);
};

/**
 * @brief A replicated successor instance ("Meeseeks summons another Meeseeks").
 *
 * Replication is a FREE CAPABILITY of the instance: the instance itself
 * decides (through its planner) WHEN to replicate.  There is no fixed trigger
 * and no hand-off - before the goal completes the pressure only grows and the
 * instance does not end; the child is simply another instance working the
 * same goal with a mutated genome.  When evolution is enabled and the parent
 * has completed, the finished genome with the strictly shorter completion
 * time becomes the next parent.  A tie keeps the earlier genome.  Children
 * are not removed.  Emergency stop is outside this object.
 */
struct MissionChild {
  std::string id;        /*!< "child-<generation>-<n>" */
  MissionGenome genome;  /*!< mutated heredity */
  std::string goal;      /*!< effective goal: the bounded sub-task (or "assist with: <parent goal>") */
  std::string subgoal;   /*!< the raw request the parent made when spawning this box */
  std::string parentId;  /*!< empty = spawned by the root mission; otherwise the spawning box id */
  /* v8.6 summon: which model this box runs on.  Empty = the base model
     (main llama-server).  A non-empty value must be a registered
     summon.models entry (see summon_registry.hpp) - the gateway deliberator
     routes this box's ticks to that endpoint.  The model itself is never
     told how to call anything; the framework adapts. */
  std::string model;
  int depth{0};          /*!< task-tree depth: 0 = direct child of the root mission */
  uint64_t bornMs{0};
  int generation{0};
  /* v8.3 self-verdict: the box itself declared its sub-task done (the parent
     still merges/verifies; a done box is skipped by the loop). */
  bool done{false};
  /* Elapsed ms from birth to done. 0 until the box finishes. Fitness is
     the negative of this time: lower completion is kept. */
  uint64_t completionMs{0};
  /* Own deliberation time (sum of this box's ticks), excluding the parent
     tick/queue wait.  >0 selects on work time; 0 = legacy birth clock. */
  uint64_t workMs{0};
  nlohmann::json toJson() const;
};

/* (1+lambda) parent retention. Index 0 is the parent. A later index is
   kept only when its completion time is strictly smaller. A tie keeps
   the earlier index, so the parent stays unless a child is faster.
   Empty input returns -1. */
inline int selectOnePlusLambda(const std::vector<double> &completionMs) {
  if (completionMs.empty()) return -1;
  int best = 0;
  for (int i = 1; i < static_cast<int>(completionMs.size()); ++i) {
    if (completionMs[static_cast<size_t>(i)] < completionMs[static_cast<size_t>(best)])
      best = i;
  }
  return best;
}

/* Phenotype of learningRate on a deterministic chain. State `length-1`
   is a fixed target of 1. Each episode sweeps backward and applies
   v <- v + alpha * (target - v). Returns how many episodes pass before
   the start state reaches 0.5. alpha <= 0 never moves and returns the
   cap 100000. A larger step inside (0, 1] reaches the criterion sooner.
   This is not a fitted curve. */
inline int episodesToCriterion(float learningRate, int length = 8) {
  if (length < 2) return 0;
  if (!(learningRate > 0.0f)) return 100000;
  std::vector<float> value(static_cast<size_t>(length - 1), 0.0f);
  for (int episode = 1; episode <= 100000; ++episode) {
    for (int i = length - 2; i >= 0; --i) {
      const float target = (i + 1 == length - 1)
                               ? 1.0f
                               : value[static_cast<size_t>(i + 1)];
      float &cell = value[static_cast<size_t>(i)];
      cell += learningRate * (target - cell);
    }
    if (value[0] >= 0.5f) return episode;
  }
  return 100000;
}

inline constexpr float kGenomeTokensFloor = 32.0f;
inline constexpr float kGenomeTokensCeil = 1024.0f;

inline float clampGenomeTokens(float v) {
  if (!(v > 0.0f)) return 256.0f;
  if (v < kGenomeTokensFloor) return kGenomeTokensFloor;
  if (v > kGenomeTokensCeil) return kGenomeTokensCeil;
  return v;
}

/* One shared policy for the genome's per-tick generation budget.

   The same function is called by the live gateway worker (111) and by
   the real-model wall-clock probe, so the probe measures the policy the
   running program executes rather than a copy of it.

   safetyCap is the caller's context-derived ceiling (the worker still
   has to leave room for the prompt).  stillDrafting keeps the existing
   short-draft cap.  There is deliberately no target-length term: the
   worker does not know how many characters remain and a budget that
   overshoots is charged to the wall clock, which is exactly the
   trade-off the evolution experiment measures. */
inline int missionTickBudget(float tokensPerTick, int safetyCap,
                             bool stillDrafting, int draftCap) {
  int want = static_cast<int>(std::lround(clampGenomeTokens(tokensPerTick)));
  if (want < 16) want = 16;
  int cap = safetyCap > 0 ? safetyCap : want;
  if (cap < 16) cap = 16;
  int budget = std::min(want, cap);
  if (stillDrafting) {
    const int draft = draftCap > 0 ? draftCap : 32;
    budget = std::min(budget, draft);
  }
  if (budget < 16) budget = 16;
  return budget;
}

/** @brief Lifecycle state machine + reproduction bookkeeping. */
class MissionLifecycle {
 public:
  MissionLifecycle();

  void assign(const Mission &m);
  void assign(const Mission &m, const MissionGenome &genome); /*!< bind heredity */
  Mission mission() const;   /*!< snapshot copy (mutex-guarded). */
  MissionGenome genome() const;
  bool active() const;       /*!< true while Running (mutex-guarded). */
  float pressureNow() const;
  /** v8.0 mission worker: append the LLM-produced work product (capped at
      4 MiB) so the human supervisor can read and judge the deliverable. */
  void appendDeliverable(const std::string &text);
  void setDeliverable(const std::string &text);

  /** @brief Replicate: mutate THIS instance's genome and record a successor
      bound to the current goal.  Returns the child genome.  Throws
      std::runtime_error when maxReplicas is exhausted (the only guardrail on
      the instance's free replication power - no runaway spawning). */
  MissionGenome replicate(float mutationRate);
  /* v8.2: children are Meeseeks-BOX TOOLS - the parent summons a bounded
     sub-task helper.  subgoal is the parent request; empty (or identical to
     the parent goal) becomes "assist with: <parent goal>" so the parent can
     never hand its whole job to a child. */
  MissionGenome replicate(float mutationRate, const std::string &subgoal);
  /* Record + mutate in one call (used by the manager spawnMissionChild).
     parentId: empty = spawned by the root mission; otherwise the spawning
     box id - the new box gets depth = parentDepth + 1.  Deep task trees are
     allowed but bounded by maxReplicaDepth (default 3): boxes CAN summon
     their own boxes so the mission can decompose into nested sub-tasks. */
  MissionChild recordChild(const MissionGenome &parent, float mutationRate,
                           const std::string &subgoal,
                           const std::string &parentId = std::string(),
                           const std::string &model = std::string());
  size_t maxReplicaDepth() const;
  void setMaxReplicaDepth(size_t n);
  size_t maxReplicas() const;
  void setMaxReplicas(size_t n);
  std::vector<MissionChild> children() const; /*!< successors (observability) */
  /* v8.3: the box declared its sub-task complete (self-verdict).  Returns
     false when the id is unknown.  workMs>0 records the box's own
     deliberation time; 0 uses the accumulated noteChildWorkMs or, when
     that is also 0, the birth-based elapsed time (legacy). */
  bool markChildDone(const std::string &childId, uint64_t workMs = 0);
  /* Fair-selection clocks: the work actually spent by the parent's own
     ticks and by each box's ticks, excluding queue/wait time. */
  void noteWorkMs(uint64_t ms);
  void noteChildWorkMs(const std::string &childId, uint64_t ms);

  /* ===================== v8.x C3 autonomous evolution =====================
   * Historical selection loop (NOT online culling): every completed mission
   * records (genome fingerprint, completion time, structural coverage) into
   * an auditable lineage; the NEXT replication uses the lineage to adjust
   * its mutation step (softmax-weighted - good history shrinks the step,
   * bad history widens it).  Humans can inspect/reset the lineage; the
   * whitelist of mutable genome fields is fixed in MissionGenome::mutate and
   * cannot be extended via config.  Disabled by default (mission.evolution
   * .enabled); when disabled the lineage is still recorded for audit. */
  void recordLineage(uint64_t completionMs, double coverage);
  nlohmann::json lineage() const;
  void resetLineage();
  void setEvolutionEnabled(bool enabled);
  bool evolutionEnabled() const;
  /* v10.1 efficiency knobs (config-gated, default off). */
  void setEvolutionAnnealPerGen(double perGen);
  void setEvolutionEarlyStop(bool enabled);
  /* True when earlyStop_ is on and two consecutive adoptions did not
     shorten the measured workMs; the replicate action must no-op then. */
  bool evolutionStalled() const;
  /* softmax-weighted mutation-step factor (1.0 = unchanged).  The Locked
     variant assumes mu_ is already held (recordChild calls it under the
     lock - calling the locking version there would self-deadlock). */
  float effectiveMutationRate(float baseRate) const;
  float effectiveMutationRateLocked(float baseRate) const;

  void markComplete();
  void markFailed();
  /* Run the same retention markComplete uses, after a restored record.
     Does nothing while evolution is off or the parent is not completed. */
  void retainFinishedGenome();

  /** Amend the goal mid-flight (human interjection): keeps Running and the
      original start time, so pressure keeps growing - the mission is
      redirected, not restarted.  Returns false when not Running. */
  bool amendGoal(const std::string &newGoal);

  /** Reproduction step: mutate the parent genome (heredity + variation). */
  MissionGenome spawnChild(const MissionGenome &parent, float mutationRate);

  nlohmann::json stats() const;      /*!< generations / spawns / completions. */
  nlohmann::json toJson() const;
  void fromJson(const nlohmann::json &j);

 private:
  nlohmann::json statsLocked() const; /*!< stats body without locking */
  /* When evolution is enabled and the parent mission has completed,
     replace this genome with the finished parent or child that has the
     strictly lowest completion time. */
  void adoptEliteLocked();
  mutable std::mutex mu_;
  Mission mission_;
  MissionGenome genome_;      /*!< heredity of THIS instance */
  std::vector<MissionChild> children_; /*!< replicated successors (observability) */
  size_t maxReplicas_{4};    /*!< guardrail cap on free replication */
  /* 0 = UNLIMITED task-tree depth (user-decided via mission.maxReplicaDepth);
     >0 = hard cap on how deep boxes may summon their own boxes. */
  size_t maxReplicaDepth_{0};
  /* v8.x C3 lineage (auditable history) */
  struct LineageEntry {
    std::string fingerprint;
    uint64_t completionMs{0};
    double coverage{0.0};
  };
  std::vector<LineageEntry> lineage_;
  bool evolutionEnabled_{false};
  /* v10.1 evolution-efficiency knobs (config-gated, default off):
     annealPerGen_ shrinks the mutation-rate factor's deviation per
     generation; earlyStop_ + stall tracking skip new children after two
     consecutive generations without a strictly shorter measured workMs. */
  double annealPerGen_{0.0};
  bool earlyStop_{false};
  uint64_t lastBestWorkMs_{0};
  int stallCount_{0};
  std::mt19937 rng_;
  size_t spawnCount_{0};
  size_t completeCount_{0};
  size_t generation_{0};
  uint64_t nowMs() const;
};

}  // namespace mission
}  // namespace phoenix
