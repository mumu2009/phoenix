/* subconscious_profile.hpp - Configurable subconscious (allostatic) layer
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   The "subconscious" here is the innate, configurable part of the brainstem
   layer (primal sensations + instincts + affective baselines).  It extends
   趋利避害 with a set of theory-grounded initial parameters so each Phoenix
   instance can be born with a different temperament, sensitivity profile,
   homeostatic setpoints and risk attitude.

   Theory (see doc/v7.0/subconscious.md for the derivations):
    - temperament  = PAD disposition (Mehrabian & Russell 1974);
    - allostasis   = homeostatic setpoints + gains + per-drive time constants
                     (Sterling 1988; Solomon & Corbit 1974);
    - risk aversion = prospect-theory utility curvature (Kahneman & Tversky 1979);
    - precision weighting (Friston): anticipatoryGain scales how strongly the
      forward-model prediction error (surprise) feeds the Curiosity drive;
    - ethology: a custom instinct table may replace the built-in five drives.

   Every parameter defaults to the current behaviour, so the profile is inert
   until enabled (config subconscious.enabled = false).
*/
#pragma once

#include <map>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include "primal_sensation.hpp"
#include "instinct.hpp"
#include "active_inference.hpp"

namespace phoenix {
namespace subconscious {

struct SubconsciousProfile {
  /* Temperament: baseline PAD disposition ([-1,1]) and mixing strength. */
  float baselineValence{0.0f};
  float baselineArousal{0.0f};
  float baselineDominance{0.0f};
  float temperamentStrength{0.0f};   /* 0 = no baseline shift. */

  /* Prospect-theory utility curvature gamma; 1 = linear. */
  float riskAversion{1.0f};

  /* Surprise -> Novelty feedback gain; 0 disables the metacognitive loop. */
  float anticipatoryGain{1.0f};

  /* Per-sensation tuning keyed by sensation-type string. */
  std::map<std::string, primal::SensationTuning> sensationTuning;

  /* Custom instinct table; empty = keep the built-in five drives. */
  std::vector<instinct::Instinct> instincts;

  nlohmann::json toJson() const;
  static SubconsciousProfile fromJson(const nlohmann::json &j);
  static SubconsciousProfile defaults();

  /** Install per-sensation tuning into a sensation engine. */
  void applyTo(primal::PrimalSensationEngine &engine) const;
  /** Replace the instinct table when the profile provides one. */
  void applyTo(instinct::InstinctEngine &engine) const;
  /** Apply risk aversion to the MPC controller. */
  void applyTo(phoenix::agi::ActiveInferenceController &ctl) const;
  /** Shift a PAD emotion tensor toward the temperament baseline. */
  emotion::EmotionTensor applyTemperament(const emotion::EmotionTensor &t) const;
};

}  // namespace subconscious
}  // namespace phoenix
