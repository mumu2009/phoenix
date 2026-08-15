/* instinct.cpp - Implementation for instinct/benefit-harm layer
   Copyright (C) 2026 079 Project */

#include "instinct.hpp"
#include <algorithm>
#include <cmath>

namespace phoenix {
namespace instinct {

namespace {

/**
 * @brief Soft affinity between a sensation type and an instinct type.
 *
 * Primary binding (per the v7.0 design's primal-sensation/instinct table)
 * maps to 1.0; biologically related secondary bindings map to intermediate
 * values; unrelated pairs keep a small floor so no instinct is ever perfectly
 * blind to any signal.  This replaces the previous hard 0/1 type match, which
 * made e.g. the Survival instinct insensitive to Pain and Fatigue.
 */
float typeAffinity(primal::SensationType s, InstinctType i) {
    switch (i) {
        case InstinctType::Survival:
            switch (s) {
                case primal::SensationType::Threat:      return 1.0f;
                case primal::SensationType::Fatigue:     return 0.6f;
                case primal::SensationType::Pain:        return 0.5f;
                case primal::SensationType::Temperature: return 0.4f;
                case primal::SensationType::Hunger:      return 0.3f;
                default: return 0.1f;
            }
        case InstinctType::Exploration:
            switch (s) {
                case primal::SensationType::Novelty:   return 1.0f;
                case primal::SensationType::Pleasure:  return 0.4f;
                default: return 0.15f;
            }
        case InstinctType::Avoidance:
            switch (s) {
                case primal::SensationType::Pain:        return 1.0f;
                case primal::SensationType::Threat:      return 0.8f;
                case primal::SensationType::Temperature: return 0.3f;
                default: return 0.1f;
            }
        case InstinctType::Affiliation:
            switch (s) {
                case primal::SensationType::SocialIsolation: return 1.0f;
                case primal::SensationType::Pleasure:        return 0.5f;
                default: return 0.1f;
            }
        case InstinctType::Curiosity:
            switch (s) {
                case primal::SensationType::Novelty:   return 1.0f;
                case primal::SensationType::Pleasure:  return 0.3f;
                default: return 0.15f;
            }
        default:
            return 0.0f;
    }
}

}  // namespace

std::string Instinct::typeToString(InstinctType t) {
    switch (t) {
        case InstinctType::Survival: return "survival";
        case InstinctType::Exploration: return "exploration";
        case InstinctType::Avoidance: return "avoidance";
        case InstinctType::Affiliation: return "affiliation";
        case InstinctType::Curiosity: return "curiosity";
        default: return "unknown";
    }
}

InstinctType Instinct::stringToType(const std::string &s) {
    static const std::unordered_map<std::string, InstinctType> map = {
        {"survival", InstinctType::Survival},
        {"exploration", InstinctType::Exploration},
        {"avoidance", InstinctType::Avoidance},
        {"affiliation", InstinctType::Affiliation},
        {"curiosity", InstinctType::Curiosity},
    };
    auto it = map.find(s);
    return it != map.end() ? it->second : InstinctType::Unknown;
}

nlohmann::json Instinct::toJson() const {
    return {
        {"type", typeToString(type)},
        {"activation", activation},
        {"benefitWeight", benefitWeight},
        {"harmWeight", harmWeight},
        {"targetSensation", primal::PrimalSensation::typeToString(targetSensation)},
        {"actionBias", actionBias}
    };
}

Instinct Instinct::fromJson(const nlohmann::json &j) {
    Instinct i;
    if (!j.is_object()) return i;
    if (j.contains("type") && j["type"].is_string()) {
        i.type = stringToType(j["type"].get<std::string>());
    }
    if (j.contains("activation") && j["activation"].is_number()) {
        i.activation = std::clamp(j["activation"].get<float>(), 0.0f, 1.0f);
    }
    if (j.contains("benefitWeight") && j["benefitWeight"].is_number()) {
        i.benefitWeight = std::clamp(j["benefitWeight"].get<float>(), 0.0f, 1.0f);
    }
    if (j.contains("harmWeight") && j["harmWeight"].is_number()) {
        i.harmWeight = std::clamp(j["harmWeight"].get<float>(), 0.0f, 1.0f);
    }
    if (j.contains("targetSensation") && j["targetSensation"].is_string()) {
        i.targetSensation = primal::PrimalSensation::stringToType(j["targetSensation"].get<std::string>());
    }
    if (j.contains("actionBias") && j["actionBias"].is_string()) {
        i.actionBias = j["actionBias"].get<std::string>();
    }
    return i;
}

nlohmann::json BenefitHarmResult::toJson() const {
    return {
        {"benefitScore", benefitScore},
        {"harmScore", harmScore},
        {"netUtility", netUtility},
        {"recommendedAction", recommendedAction},
        {"driveVector", driveVector}
    };
}

InstinctEngine::InstinctEngine() {}

InstinctEngine::InstinctEngine(std::vector<Instinct> defaults)
    : instincts_(std::move(defaults)) {
    for (const auto &i : instincts_) {
        currentActivations_.push_back(i.activation);
    }
}

void InstinctEngine::registerInstinct(const Instinct &i) {
    instincts_.push_back(i);
    currentActivations_.push_back(i.activation);
}

InstinctEngine::Appraisal InstinctEngine::appraise(
    const primal::PrimalSensation &s, const Instinct &instinct) const {
    // Goal-conduciveness appraisal (Smith & Lazarus, 1990): a sensation is
    // beneficial in proportion to its positive valence and harmful in
    // proportion to its negative valence, weighted by intensity, by how
    // attuned the instinct is to that sensation type, and by the instinct's
    // pursue/avoid sensitivity.
    const float drive = s.intensity * typeAffinity(s.type, instinct.type);
    Appraisal a;
    a.benefit = drive * std::max(0.0f, s.valence) * instinct.benefitWeight;
    a.harm = drive * std::max(0.0f, -s.valence) * instinct.harmWeight;
    return a;
}

std::vector<float> InstinctEngine::currentActivations() const {
    return currentActivations_;
}

void InstinctEngine::update(const std::vector<primal::PrimalSensation> &sensations,
                            float dtSec) {
    float decay = 1.0f;
    if (activationDecayHalfLife_ > 0.0f && dtSec > 0.0f) {
        decay = std::pow(0.5f, dtSec / activationDecayHalfLife_);
    }

    for (size_t i = 0; i < instincts_.size(); ++i) {
        const auto &instinct = instincts_[i];
        float sensationDrive = 0.0f;
        for (const auto &s : sensations) {
            const Appraisal a = appraise(s, instinct);
            sensationDrive += a.benefit + a.harm;
        }
        /* v7.0 dynamic regulation: current = base * (1 + drive) * decay */
        float current = instinct.activation * (1.0f + sensationDrive) * decay;
        currentActivations_[i] = std::clamp(current, 0.0f, 1.0f);
    }
}

BenefitHarmResult InstinctEngine::evaluate(
    const std::vector<primal::PrimalSensation> &sensations,
    float temperature) const {
    (void)temperature;  // Reserved: Boltzmann inverse-temperature (see below).
    BenefitHarmResult result;
    float benefit = 0.0f;
    float harm = 0.0f;
    std::unordered_map<std::string, float> actionUtility;

    for (size_t i = 0; i < instincts_.size(); ++i) {
        const auto &instinct = instincts_[i];
        float bi = 0.0f;
        float hi = 0.0f;
        for (const auto &s : sensations) {
            const Appraisal a = appraise(s, instinct);
            bi += a.benefit;
            hi += a.harm;
        }
        const float act = currentActivations_[i];
        benefit += bi * act;
        harm += hi * act;

        // Each instinct votes for its action bias with its own net utility.
        if (!instinct.actionBias.empty()) {
            actionUtility[instinct.actionBias] += (bi - hi) * act;
        }
    }

    // Normalized, bounded benefit/harm/utility.  Both benefit and harm live in
    // [0, 1]; netUtility is the signed normalised difference in [-1, 1].
    const float denom = benefit + harm + 1e-6f;
    result.benefitScore = std::clamp(benefit / denom, 0.0f, 1.0f);
    result.harmScore = std::clamp(harm / denom, 0.0f, 1.0f);
    result.netUtility = std::clamp((benefit - harm) / denom, -1.0f, 1.0f);

    // Action selection.  The deterministic recommendation is the argmax of the
    // per-action net utility.  temperature is a Boltzmann inverse-temperature
    // reserved for stochastic (sampling) action selection: as T -> 0 the policy
    // concentrates on argmax, as T -> inf it approaches a uniform distribution.
    // The argmax itself is temperature-invariant because softmax is monotonic.
    if (!actionUtility.empty()) {
        result.recommendedAction = std::max_element(
            actionUtility.begin(), actionUtility.end(),
            [](const auto &a, const auto &b) { return a.second < b.second; })->first;
    } else {
        result.recommendedAction.clear();
    }

    // Canonical appraisal -> 8-d drive vector (shared with the emotion module).
    // Replaces the previous ad-hoc 8x5 fixed matrix.  See emotion::fromAppraisal
    // and doc/v7.0/algorithm.md sections 7 and 15.
    const emotion::EmotionTensor t =
        emotion::fromAppraisal(result.benefitScore, result.harmScore);
    result.driveVector = {t.valence, t.arousal, t.dominance, t.trust,
                          t.joy, t.fear, t.anger, t.surprise};

    return result;
}

emotion::EmotionTensor InstinctEngine::driveToEmotion(const BenefitHarmResult &result) {
    // Reconstruct the 8-D emotion tensor from the drive vector when it is
    // available.  This preserves any explicit drive-vector values (e.g. in
    // unit tests) while still being lossless for vectors produced by
    // emotion::fromAppraisal() in evaluate().
    if (result.driveVector.size() >= 8) {
        return emotion::EmotionTensor(
            result.driveVector[0], result.driveVector[1], result.driveVector[2],
            result.driveVector[3], result.driveVector[4], result.driveVector[5],
            result.driveVector[6], result.driveVector[7]);
    }
    return emotion::fromAppraisal(result.benefitScore, result.harmScore);
}

BenefitHarmResult InstinctEngine::evaluateAction(
    const std::string &action,
    const std::vector<primal::PrimalSensation> &sensations) const {
    BenefitHarmResult base = evaluate(sensations);
    float actionMod = 0.0f;
    for (size_t i = 0; i < instincts_.size(); ++i) {
        const auto &instinct = instincts_[i];
        if (instinct.actionBias.find(action) != std::string::npos) {
            actionMod += currentActivations_[i] * (instinct.benefitWeight - instinct.harmWeight);
        }
    }
    base.netUtility = std::clamp(base.netUtility + actionMod * 0.25f, -1.0f, 1.0f);
    return base;
}

void InstinctEngine::replaceAll(const std::vector<Instinct> &instincts) {
    instincts_ = instincts;
    currentActivations_.clear();
    for (const auto &i : instincts_) {
        currentActivations_.push_back(i.activation);
    }
}

nlohmann::json InstinctEngine::toJson() const {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto &i : instincts_) arr.push_back(i.toJson());
    return arr;
}

InstinctEngine InstinctEngine::fromJson(const nlohmann::json &j) {
    InstinctEngine e;
    if (!j.is_array()) return e;
    for (const auto &item : j) e.registerInstinct(Instinct::fromJson(item));
    return e;
}

InstinctEngine InstinctEngine::defaultEngine() {
    InstinctEngine e;
    e.registerInstinct({InstinctType::Survival, 0.5f, 0.3f, 0.9f, primal::SensationType::Threat, "protect"});
    e.registerInstinct({InstinctType::Exploration, 0.5f, 0.9f, 0.2f, primal::SensationType::Novelty, "explore"});
    e.registerInstinct({InstinctType::Avoidance, 0.4f, 0.1f, 0.95f, primal::SensationType::Pain, "avoid"});
    e.registerInstinct({InstinctType::Affiliation, 0.5f, 0.8f, 0.3f, primal::SensationType::SocialIsolation, "connect"});
    e.registerInstinct({InstinctType::Curiosity, 0.5f, 0.9f, 0.4f, primal::SensationType::Novelty, "investigate"});
    return e;
}

}  // namespace instinct
}  // namespace phoenix
