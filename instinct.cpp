/* instinct.cpp - Implementation for instinct/benefit-harm layer
   Copyright (C) 2026 079 Project */

#include "instinct.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>

namespace phoenix {
namespace instinct {

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

float InstinctEngine::sensationInstinctScore(const primal::PrimalSensation &s,
                                              const Instinct &instinct) const {
    float typeMatch = (s.type == instinct.targetSensation) ? 1.0f : 0.0f;
    float valenceMatch = 1.0f - std::abs(s.valence - (instinct.benefitWeight - instinct.harmWeight));
    return typeMatch * s.intensity * (valenceMatch * 0.5f + 0.5f);
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
        float sensationScore = 0.0f;
        for (const auto &s : sensations) {
            sensationScore += sensationInstinctScore(s, instinct);
        }
        /* v7.0 dynamic regulation: current = base * (1 + sensationScore) * decay */
        float current = instinct.activation * (1.0f + sensationScore) * decay;
        currentActivations_[i] = std::clamp(current, 0.0f, 1.0f);
    }
}

BenefitHarmResult InstinctEngine::evaluate(
    const std::vector<primal::PrimalSensation> &sensations,
    float temperature) const {
    BenefitHarmResult result;
    float benefit = 0.0f;
    float harm = 0.0f;
    float totalActivation = 0.0f;
    std::unordered_map<std::string, float> actionScores;

    for (size_t i = 0; i < instincts_.size(); ++i) {
        const auto &instinct = instincts_[i];
        float score = 0.0f;
        for (const auto &s : sensations) {
            score += sensationInstinctScore(s, instinct);
        }
        score *= currentActivations_[i];
        if (temperature > 0.0f) {
            score = std::pow(score + 1e-6f, 1.0f / temperature);
        }
        benefit += score * instinct.benefitWeight;
        harm += score * instinct.harmWeight;
        totalActivation += score;

        if (!instinct.actionBias.empty()) {
            actionScores[instinct.actionBias] += score;
        }
    }

    float denom = totalActivation + 1e-6f;
    result.benefitScore = std::clamp(benefit / denom, 0.0f, 1.0f);
    result.harmScore = std::clamp(harm / denom, 0.0f, 1.0f);
    result.netUtility = std::clamp((benefit - harm) / denom, -1.0f, 1.0f);

    if (!actionScores.empty()) {
        auto best = std::max_element(
            actionScores.begin(), actionScores.end(),
            [](const auto &a, const auto &b) { return a.second < b.second; });
        result.recommendedAction = best->first;
    } else {
        result.recommendedAction.clear();  // Do not synthesize explicit action words.
    }

    /* Compute the 8-dim emotion operation weight vector via a fixed linear
       matrix applied to [benefit, harm, netUtility, abs(netUtility), activation].
       This vector is not an explicit emotion state; it is a set of weights that
       downstream modules (e.g. EmotionVocabWeightTable / logit-bias matrices)
       apply as a latent signal, avoiding hard-coded emotional labels. */
    static constexpr std::array<std::array<float, 5>, 8> kEmotionOperationMatrix = {{
        {{ 0.00f,  0.00f,  1.00f,  0.00f,  0.00f }},
        {{ 0.30f,  0.30f,  0.00f,  0.00f,  0.40f }},
        {{ 0.00f,  0.00f,  0.00f,  1.00f,  0.00f }},
        {{ 0.70f, -0.20f,  0.30f,  0.00f,  0.00f }},
        {{ 0.60f, -0.30f,  0.40f,  0.00f,  0.10f }},
        {{-0.20f,  0.80f,  0.00f,  0.00f,  0.20f }},
        {{ 0.00f,  0.60f, -0.60f,  0.00f,  0.10f }},
        {{ 0.10f,  0.10f,  0.00f,  0.30f,  0.50f }}
    }};
    const float absNet = std::abs(result.netUtility);
    const float activationNorm = std::clamp(totalActivation, 0.0f, 1.0f);
    const std::array<float, 5> input = {{
        result.benefitScore, result.harmScore, result.netUtility, absNet, activationNorm
    }};
    result.driveVector.assign(8, 0.0f);
    for (size_t i = 0; i < 8; ++i) {
        float v = 0.0f;
        for (size_t j = 0; j < 5; ++j) {
            v += kEmotionOperationMatrix[i][j] * input[j];
        }
        result.driveVector[i] = std::clamp(v, -1.0f, 1.0f);
    }

    return result;
}

emotion::EmotionTensor InstinctEngine::driveToEmotion(const BenefitHarmResult &result) {
    /* The driveVector holds the 8 operation weights produced by the fixed
       linear matrix in evaluate().  Convert them into an EmotionTensor so
       downstream modules can apply them as a latent signal to matrices such
       as token/logit weight tables, without turning them into explicit words. */
    emotion::EmotionTensor t;
    if (result.driveVector.size() >= 8) {
        t.valence = result.driveVector[0];
        t.arousal = result.driveVector[1];
        t.dominance = result.driveVector[2];
        t.trust = result.driveVector[3];
        t.joy = result.driveVector[4];
        t.fear = result.driveVector[5];
        t.anger = result.driveVector[6];
        t.surprise = result.driveVector[7];
    } else {
        // Fallback: derive a neutral-ish tensor directly from scores.
        t.valence = result.netUtility;
        t.arousal = (result.benefitScore + result.harmScore) * 0.5f;
        t.dominance = std::abs(result.netUtility);
        t.trust = result.benefitScore;
        t.joy = result.benefitScore;
        t.fear = result.harmScore;
        t.anger = result.netUtility < 0.0f ? -result.netUtility : 0.0f;
        t.surprise = t.arousal;
    }
    return t;
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
