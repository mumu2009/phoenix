/* mission_lifecycle.hpp - Meeseeks-style mission layer (goal pressure + reproduction)
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   Formalises the "Meeseeks box" idea: an agent instance is born with ONE
   mission; pain grows linearly with time since mission start and ends only
   when the mission is judged complete.  Because the agent's existing loop
   minimises accumulated pain (see doc/v7.0/mission_layer.md for the proof
   that total pain = g * T^2 / 2, monotone in the completion time T), the
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
  float painGainPerSec{0.01f};   /*!< urgency: pain growth rate. */
  float maxPain{1.0f};
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

  /** Mission pressure: pain grows linearly with elapsed time until maxPain. */
  float pressure(uint64_t nowMs) const {
    if (state != MissionState::Running) return 0.0f;
    const double p = static_cast<double>(painGainPerSec) * elapsedSec(nowMs);
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

/** @brief Lifecycle state machine + reproduction bookkeeping. */
class MissionLifecycle {
 public:
  MissionLifecycle();

  void assign(const Mission &m);
  Mission mission() const;   /*!< snapshot copy (mutex-guarded). */
  bool active() const;       /*!< true while Running (mutex-guarded). */
  float pressureNow() const;

  void markComplete();
  void markFailed();

  /** Reproduction step: mutate the parent genome (heredity + variation). */
  MissionGenome spawnChild(const MissionGenome &parent, float mutationRate);

  nlohmann::json stats() const;      /*!< generations / spawns / completions. */
  nlohmann::json toJson() const;
  void fromJson(const nlohmann::json &j);

 private:
  mutable std::mutex mu_;
  Mission mission_;
  std::mt19937 rng_;
  size_t spawnCount_{0};
  size_t completeCount_{0};
  size_t generation_{0};
  uint64_t nowMs() const;
};

}  // namespace mission
}  // namespace phoenix
