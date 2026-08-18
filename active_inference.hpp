/* active_inference.hpp - Active inference / model-predictive control core
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This module closes the otherwise-open "sense -> drive -> act" loop into a
   full agent by adding a predictive forward model and a receding-horizon
   planner that minimises Expected Free Energy (EFE).  It is OPTIONAL and
   disabled by default (config `agi.enabled`), consistent with Phoenix's
   "every module is opt-in" philosophy.

   Theory: the agent picks a policy a_{1:H} minimising
       G = sum_t [ pragmatic_t + intrinsic_t + epistemic_t ],
   with
     pragmatic_t = - expected_utility   (preferred outcomes; the existing
                                         趋利避害 netUtility),
     intrinsic_t = homeostatic drive cost (primal sensations / instincts),
     epistemic_t = - expected information gain (forward-model surprise /
                                         curiosity).
   See doc/v7.0/active_inference.md for the full decomposition and the
   MPC <-> active-inference equivalence proof.
*/
#pragma once

#include <cstddef>
#include <limits>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace phoenix {
namespace agi {

/**
 * @brief The three additive terms of Expected Free Energy.
 *
 * Minimising total() trades off exploitation (pragmatic), homeostasis
 * (intrinsic) and exploration (epistemic).  All three are non-negative
 * costs by convention so `total()` is directly minimisable.
 */
struct ExpectedFreeEnergy {
  double pragmatic{0.0};  /*!< -expected utility (exploit preferred outcomes). */
  double intrinsic{0.0};  /*!< homeostatic drive cost (primal sensations). */
  double epistemic{0.0};  /*!< -expected information gain (explore). */

  double total() const { return pragmatic + intrinsic + epistemic; }
  nlohmann::json toJson() const;
};

/**
 * @brief Deterministic linear forward model z_{t+1} = A z_t + B a_t + b.
 *
 * Predictive (not merely reconstructive): given a latent unit query z_t and
 * an action embedding a_t it predicts the next latent z_{t+1}.  The L2
 * prediction error defines the Gaussian surprise used as the epistemic term
 * and as the learning signal for online least-squares-style updates.
 */
class LatentTransitionModel {
 public:
  explicit LatentTransitionModel(size_t dim = 128, size_t actionDim = 0);

  size_t dim() const { return dim_; }
  size_t actionDim() const { return actionDim_; }

  /** Rebuild the model for a new (dim, actionDim) pair (identity dynamics). */
  void reset(size_t dim, size_t actionDim);

  /** Predict z_{t+1} from z_t and an action embedding (may be empty). */
  std::vector<float> predict(const std::vector<float> &z,
                             const std::vector<float> &a) const;

  /** Gaussian surprise of an observed next-state against a prediction. */
  double surprise(const std::vector<float> &observed,
                  const std::vector<float> &predicted) const;

  /** One SGD step toward the observed transition. */
  void update(const std::vector<float> &z, const std::vector<float> &a,
              const std::vector<float> &zNext, float lr = 0.01f);

  nlohmann::json status() const;

  /** Full-state serialization (long-term evolution persistence). */
  nlohmann::json toJson() const;
  static LatentTransitionModel fromJson(const nlohmann::json &j, size_t dim,
                                        size_t actionDim);

 private:
  size_t dim_;
  size_t actionDim_;
  std::vector<float> A_;  /*!< dim x dim row-major transition. */
  std::vector<float> B_;  /*!< dim x actionDim control matrix. */
  std::vector<float> b_;  /*!< dim bias. */
};

/**
 * @brief A named, embedded action in the agent's action space.
 */
struct Action {
  std::string name;
  std::vector<float> embedding;  /*!< optional action vector for the forward model. */
};

/**
 * @brief Receding-horizon (MPC) planner that minimises Expected Free Energy.
 *
 * The controller holds a forward model, a linear preference/value head
 * (utility = w . z), and a rolling episodic memory of (z, a, z', utility,
 * surprise) transitions.  plan() enumerates the action space, rolls each
 * action forward H steps, accumulates EFE, and returns the best first action.
 * This is MPC, and it is exactly active inference when the pragmatic term is
 * the expected log-likelihood of preferred outcomes and the epistemic term is
 * the expected information gain.
 *
 * Self-evolution: observeRewarded() closes the learning loop - the realised
 * benefit-harm netUtility is the reward, TD(0) evolves the value head w, the
 * forward model is refined online, and consolidate() replays episodic memory.
 * With bootstrapPreferences() the drive vector only SEEDS w once; afterwards
 * experience owns it.
 */
class ActiveInferenceController {
 public:
  ActiveInferenceController(size_t dim = 128, size_t actionDim = 1, size_t horizon = 3);

  /** Rebuild the controller for a new configuration. */
  void configure(size_t dim, size_t actionDim, size_t horizon);

  /** Set the linear preference vector w (utility = w . z). */
  void setPreferences(const std::vector<float> &w);
  const std::vector<float> &preferences() const { return w_; }

  /** Bootstrap the preference vector once (subsequent calls are no-ops until
      resetPreferencesLearning()).  This is the initialisation hook for the TD
      value learner: the drive vector seeds w once, then experience evolves it
      (self-evolution); resetting w every turn would erase the learning. */
  void bootstrapPreferences(const std::vector<float> &w);
  void resetPreferencesLearning() { prefsBootstrapped_ = false; }
  bool preferencesBootstrapped() const { return prefsBootstrapped_; }

  /** Risk aversion gamma: utility curvature u' = sign(u)·|u|^gamma (prospect
      theory, Kahneman & Tversky 1979). 1.0 = linear, <1 risk-seeking, >1
      risk-averse. */
  void setRiskAversion(float gamma) { riskAversion_ = gamma > 0.0f ? gamma : 1.0f; }
  float riskAversion() const { return riskAversion_; }

  /** Clamp bound for the learned value weights (prevents divergence). */
  void setValueClamp(double v) { valueClamp_ = v > 0.0 ? v : 10.0; }
  double valueClamp() const { return valueClamp_; }

  /** Set the discrete action space. */
  void setActions(std::vector<Action> actions);
  const std::vector<Action> &actions() const { return actions_; }

  void setHorizon(size_t h);
  size_t horizon() const { return horizon_; }

  LatentTransitionModel &model() { return model_; }
  const LatentTransitionModel &model() const { return model_; }

  /** Scalar preference value of a latent state: utility = w . z. */
  double utility(const std::vector<float> &z) const;

  /** Roll out one action over the horizon and return its EFE. */
  ExpectedFreeEnergy evaluate(const std::vector<float> &z,
                              const Action &action, double driveCost,
                              double pragW, double intrinW, double epistW) const;

  struct Plan {
    int bestAction = -1;
    ExpectedFreeEnergy efe;
    double bestCost = std::numeric_limits<double>::infinity();
  };

  /** Pick the action minimising EFE over the horizon. */
  Plan plan(const std::vector<float> &z, double driveCost,
            double pragW = 1.0, double intrinW = 1.0, double epistW = 0.25) const;

  /** Record a real transition, refine the model, return its surprise. */
  double observe(const std::vector<float> &z, const std::vector<float> &a,
                 const std::vector<float> &zNext, float lr = 0.01f);

  /** Observe with a realised reward: forward-model update + TD(0) value
      learning (the self-evolution step).  Returns the transition surprise. */
  double observeRewarded(const std::vector<float> &z, const std::vector<float> &a,
                         const std::vector<float> &zNext, double reward,
                         float lr = 0.01f, double alpha = 0.05,
                         double gamma = 0.9);

  /** TD(0) value learning (Sutton & Barto 1988; Tsitsiklis & Van Roy 1997):
      w <- w + alpha * delta * z,  delta = r + gamma * V(z') - V(z).
      Weights are clamped to [-valueClamp_, valueClamp_].  Returns delta. */
  double learnValue(const std::vector<float> &z, const std::vector<float> &zNext,
                    double reward, double alpha = 0.05, double gamma = 0.9);

  /** Replay the most recent episodes to consolidate the forward model and
      the value function (sleep-like consolidation).  Returns mean surprise. */
  double consolidate(size_t maxReplays = 64, double alpha = 0.02,
                     double gamma = 0.9);

  /** Adaptive exploration multiplier (VDBE-style, Tokic 2010): > 1 when the
      last surprise is below its EMA (getting predictable -> explore), < 1
      when above (chaotic -> exploit).  Bounded to [0.5, 2.0]. */
  double explorationMultiplier() const;
  double surpriseEma() const { return surpriseEma_; }

  size_t episodeCount() const { return Episodes_.size(); }
  nlohmann::json status() const;

  /** Full-state serialization: learned value head, forward model, episodic
      memory and all evolution bookkeeping (long-term evolution survives
      restarts through exportState/importState). */
  nlohmann::json toJson() const;
  static ActiveInferenceController fromJson(const nlohmann::json &j);

 private:
  static constexpr size_t kMaxEpisodes_ = 4096;
  struct Episode {
    std::vector<float> z, a, zNext;
    float utility, surprise, reward;
  };
  LatentTransitionModel model_;
  size_t horizon_;
  std::vector<float> w_;
  float riskAversion_{1.0f};
  double surpriseEma_{0.0};
  double lastSurprise_{0.0};
  double valueClamp_{10.0};
  bool prefsBootstrapped_{false};
  std::vector<Action> actions_;
  std::vector<Episode> Episodes_;
};

}  // namespace agi
}  // namespace phoenix
