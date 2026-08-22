/* mission_lifecycle.hpp - Meeseeks-style mission layer (goal pressure + reproduction)
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

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

#include <cmath>
#include <cstdint>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
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
 * same goal with a mutated genome.  Mutation provides diversity only; there
 * is no fitness comparison and no culling (not natural selection).
 */
struct MissionChild {
  std::string id;        /*!< "child-<generation>-<n>" */
  MissionGenome genome;  /*!< mutated heredity */
  std::string goal;      /*!< effective goal: the bounded sub-task (or "assist with: <parent goal>") */
  std::string subgoal;   /*!< the raw request the parent made when spawning this box */
  std::string parentId;  /*!< empty = spawned by the root mission; otherwise the spawning box id */
  int depth{0};          /*!< task-tree depth: 0 = direct child of the root mission */
  uint64_t bornMs{0};
  int generation{0};
  /* v8.3 self-verdict: the box itself declared its sub-task done (the parent
     still merges/verifies; a done box is skipped by the loop). */
  bool done{false};
  nlohmann::json toJson() const;
};

/** @brief Lifecycle state machine + reproduction bookkeeping. */
class MissionLifecycle {
 public:
  MissionLifecycle();

  void assign(const Mission &m);
  void assign(const Mission &m, const MissionGenome &genome); /*!< bind heredity */
  Mission mission() const;   /*!< snapshot copy (mutex-guarded). */
  bool active() const;       /*!< true while Running (mutex-guarded). */
  float pressureNow() const;
  /** v8.0 mission worker: append the LLM-produced work product (capped at
      4 MiB) so the human supervisor can read and judge the deliverable. */
  void appendDeliverable(const std::string &text);

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
                           const std::string &parentId = std::string());
  size_t maxReplicaDepth() const;
  void setMaxReplicaDepth(size_t n);
  size_t maxReplicas() const;
  void setMaxReplicas(size_t n);
  std::vector<MissionChild> children() const; /*!< successors (observability) */
  /* v8.3: the box declared its sub-task complete (self-verdict).  Returns
     false when the id is unknown. */
  bool markChildDone(const std::string &childId);

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
  /* softmax-weighted mutation-step factor (1.0 = unchanged).  The Locked
     variant assumes mu_ is already held (recordChild calls it under the
     lock - calling the locking version there would self-deadlock). */
  float effectiveMutationRate(float baseRate) const;
  float effectiveMutationRateLocked(float baseRate) const;

  void markComplete();
  void markFailed();

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
  std::mt19937 rng_;
  size_t spawnCount_{0};
  size_t completeCount_{0};
  size_t generation_{0};
  uint64_t nowMs() const;
};

}  // namespace mission
}  // namespace phoenix
