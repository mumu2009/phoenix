/* instinct.hpp - Instinct/benefit-harm layer for Phoenix v7.0 "Arthur"
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version. */

#pragma once

#include "emotion_system.hpp"
#include "primal_sensation.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace phoenix {
namespace instinct {

/**
 * @brief Innate behavioural drives that bias decision making.
 */
enum class InstinctType {
    Unknown = 0,
    Survival,      /*!< Preserve self / system integrity. */
    Exploration,   /*!< Seek novelty / information. */
    Avoidance,     /*!< Flee harm / threat. */
    Affiliation,   /*!< Seek social contact. */
    Curiosity      /*!< Resolve uncertainty. */
};

/**
 * @brief An instinct definition with its tuning parameters.
 */
struct Instinct {
    InstinctType type = InstinctType::Unknown;
    float activation = 0.0f;               /*!< Current activation [0, 1]. */
    float benefitWeight = 0.5f;            /*!< How strongly it pursues benefit. */
    float harmWeight = 0.5f;                 /*!< How strongly it avoids harm. */
    primal::SensationType targetSensation = primal::SensationType::Unknown;
    std::string actionBias;                /*!< Default action label (verb). */

    nlohmann::json toJson() const;
    static Instinct fromJson(const nlohmann::json &j);
    static std::string typeToString(InstinctType t);
    static InstinctType stringToType(const std::string &s);
};

/**
 * @brief Result of the benefit-harm (趋利避害) evaluation.
 */
struct BenefitHarmResult {
    float benefitScore = 0.0f;
    float harmScore = 0.0f;
    float netUtility = 0.0f;                 /*!< benefit - harm, normalized [-1, 1]. */
    std::string recommendedAction;
    std::vector<float> driveVector;          /*!< 8-dim emotion-direction vector. */

    nlohmann::json toJson() const;
};

/**
 * @brief Converts between primal sensations, instincts and emotional tensors.
 *
 * Implements the v7.0 primal sensation/instinct layer and prompt split
 * motivation signal.  It evaluates benefit vs harm and produces an emotion
 * tensor bias that downstream modules can apply.
 *
 * Dynamic activation:
 *   Each instinct has a base activation (stored in Instinct::activation) and a
 *   runtime current activation maintained by this engine.  Calling update()
 *   decays the current activation over time and then amplifies it based on
 *   matching primal sensations.  This implements the "base strength x
 *   sensation weight x time-decay" regulation described in the v7.0 design.
 */
class InstinctEngine {
public:
    InstinctEngine();
    explicit InstinctEngine(std::vector<Instinct> defaults);

    void registerInstinct(const Instinct &i);
    const std::vector<Instinct> &instincts() const { return instincts_; }

    /**
     * @brief Return the current runtime activation of each instinct.
     *
     * The returned vector is parallel to instincts().  Values are clamped to
     * [0, 1] and reflect the last update() call (or the base activation if
     * update() has never been called).
     */
    std::vector<float> currentActivations() const;

    /**
     * @brief Update runtime activations by decaying old state and amplifying
     *        matching sensations.
     *
     * Implements the v7.0 instinct strength regulation:
     *   current = baseActivation * (1 + sensationScore) * decayFactor
     * where decayFactor halves every activationDecayHalfLife_ seconds.
     *
     * @param sensations Current primal sensations.
     * @param dtSec      Elapsed seconds since last update.
     */
    void update(const std::vector<primal::PrimalSensation> &sensations, float dtSec);

    /**
     * @brief Evaluate sensations and return benefit/harm recommendation.
     *
     * Uses a goal-conduciveness appraisal (Smith & Lazarus, 1990): positive
     * valence contributes benefit, negative valence contributes harm, each
     * weighted by intensity, sensation/instinct affinity, and the instinct's
     * pursue/avoid sensitivity.  The 8-d drive vector is produced by the
     * canonical emotion::fromAppraisal mapping (no ad-hoc matrix).
     *
     * @param temperature Boltzmann inverse-temperature, reserved for stochastic
     *                    action selection; the deterministic recommendation is
     *                    the argmax and is temperature-invariant.
     */
    BenefitHarmResult evaluate(const std::vector<primal::PrimalSensation> &sensations,
                               float temperature = 1.0f) const;

    /**
     * @brief Convert a benefit-harm result into an emotion tensor offset.
     */
    static emotion::EmotionTensor driveToEmotion(const BenefitHarmResult &result);

    /**
     * @brief Compute a benefit-harm score for an action string.
     */
    BenefitHarmResult evaluateAction(const std::string &action,
                                     const std::vector<primal::PrimalSensation> &sensations) const;

    nlohmann::json toJson() const;
    static InstinctEngine fromJson(const nlohmann::json &j);

    /** Factory: returns a default set of human-like drives. */
    static InstinctEngine defaultEngine();

private:
    /** Goal-conduciveness appraisal of a single sensation for an instinct. */
    struct Appraisal {
        float benefit = 0.0f;   /*!< Signed benefit contribution. */
        float harm = 0.0f;      /*!< Signed harm contribution. */
    };

    std::vector<Instinct> instincts_;
    std::vector<float> currentActivations_;       /*!< Runtime activation per instinct. */
    float activationDecayHalfLife_ = 60.0f;       /*!< Seconds for current activation to halve. */

    Appraisal appraise(const primal::PrimalSensation &s,
                       const Instinct &instinct) const;
};

}  // namespace instinct
}  // namespace phoenix
