/* mission_lifecycle.hpp - Meeseeks-style mission layer (goal pressure + reproduction)
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   Formalises the "Meeseeks box" idea: an agent instance is born with ONE
   mission; pain grows with time since mission start (growth mode is user-
   configurable: linear or logarithmic, logarithmic by default) and ends
   only when the mission is judged complete.  Because the agent's existing
   loop minimises accumulated pain (see doc/v7.0/mission_layer.md for the
   proofs: linear mode total pain = g*T^2/2; logarithmic mode total pain =
   k*((T+1)*ln(T+1) - T); both strictly monotone in completion time T), the
   pressure forces shortest-time goal completion.

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
#include <nlohmann/json.hpp>

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
  std::string pressureMode{"logarithmic"}; /*!< "linear" | "logarithmic" (default). */
  double pressureHorizonSec{3600.0};       /*!< log mode: seconds until maxPain. */
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

  /** Mission pressure (user-configurable growth mode):
      linear:      p(t) = min(g*t, Pmax)                g = painGainPerSec
      logarithmic: p(t) = Pmax * ln(1+t) / ln(1+H)      H = pressureHorizonSec
      (default logarithmic: gentle early urgency that reaches maxPain exactly
      at the horizon H, giving long tasks like text deliverables time to work).
      Both curves are strictly increasing in t, so total accumulated pain
      stays monotone in completion time T (mission_layer.md section 2). */
  float pressure(uint64_t nowMs) const {
    if (state != MissionState::Running) return 0.0f;
    const double t = elapsedSec(nowMs);
    if (t <= 0.0) return 0.0f;
    double p = 0.0;
    if (pressureMode == "linear") {
      p = static_cast<double>(painGainPerSec) * t;
    } else {
      const double H = pressureHorizonSec >= 1.0 ? pressureHorizonSec : 1.0;
      p = static_cast<double>(maxPain) * std::log1p(t) / std::log1p(H);
    }
    return static_cast<float>(p >= maxPain ? maxPain : p);
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
  std::string goal;      /*!< inherited goal */
  uint64_t bornMs{0};
  int generation{0};
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
      64 KiB) so the human supervisor can read and judge the deliverable. */
  void appendDeliverable(const std::string &text);

  /** @brief Replicate: mutate THIS instance's genome and record a successor
      bound to the current goal.  Returns the child genome.  Throws
      std::runtime_error when maxReplicas is exhausted (the only guardrail on
      the instance's free replication power - no runaway spawning). */
  MissionGenome replicate(float mutationRate);
  size_t maxReplicas() const;
  void setMaxReplicas(size_t n);
  std::vector<MissionChild> children() const; /*!< successors (observability) */

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
  std::mt19937 rng_;
  size_t spawnCount_{0};
  size_t completeCount_{0};
  size_t generation_{0};
  uint64_t nowMs() const;
};

}  // namespace mission
}  // namespace phoenix
