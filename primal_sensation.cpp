/* primal_sensation.cpp - Implementation for primal sensation layer
   Copyright (C) 2026 079 Project */

#include "primal_sensation.hpp"
#include <algorithm>
#include <chrono>
#include <limits>
#include <unordered_map>

namespace phoenix {
namespace primal {

std::string PrimalSensation::typeToString(SensationType t) {
    switch (t) {
        case SensationType::Pain: return "pain";
        case SensationType::Pleasure: return "pleasure";
        case SensationType::Hunger: return "hunger";
        case SensationType::Temperature: return "temperature";
        case SensationType::Fatigue: return "fatigue";
        case SensationType::Threat: return "threat";
        case SensationType::SocialIsolation: return "social_isolation";
        case SensationType::Novelty: return "novelty";
        default: return "unknown";
    }
}

SensationType PrimalSensation::stringToType(const std::string &s) {
    static const std::unordered_map<std::string, SensationType> map = {
        {"pain", SensationType::Pain},
        {"pleasure", SensationType::Pleasure},
        {"hunger", SensationType::Hunger},
        {"temperature", SensationType::Temperature},
        {"fatigue", SensationType::Fatigue},
        {"threat", SensationType::Threat},
        {"social_isolation", SensationType::SocialIsolation},
        {"novelty", SensationType::Novelty},
    };
    auto it = map.find(s);
    return it != map.end() ? it->second : SensationType::Unknown;
}

nlohmann::json PrimalSensation::toJson() const {
    return {
        {"type", typeToString(type)},
        {"intensity", intensity},
        {"valence", valence},
        {"durationSec", durationSec},
        {"source", source},
        {"timestampMs", timestampMs}
    };
}

PrimalSensation PrimalSensation::fromJson(const nlohmann::json &j) {
    PrimalSensation s;
    if (!j.is_object()) return s;
    if (j.contains("type") && j["type"].is_string()) {
        s.type = stringToType(j["type"].get<std::string>());
    }
    if (j.contains("intensity") && j["intensity"].is_number()) {
        s.intensity = std::clamp(j["intensity"].get<float>(), 0.0f, 1.0f);
    }
    if (j.contains("valence") && j["valence"].is_number()) {
        s.valence = std::clamp(j["valence"].get<float>(), -1.0f, 1.0f);
    }
    if (j.contains("durationSec") && j["durationSec"].is_number()) {
        s.durationSec = j["durationSec"].get<float>();
    }
    if (j.contains("source") && j["source"].is_string()) {
        s.source = j["source"].get<std::string>();
    }
    if (j.contains("timestampMs") && j["timestampMs"].is_number_unsigned()) {
        s.timestampMs = j["timestampMs"].get<uint64_t>();
    } else if (s.timestampMs == 0) {
        s.timestampMs = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
    }
    return s;
}

void PrimalSensationEngine::add(const PrimalSensation &s) {
    sensations_.push_back(s);
}

void PrimalSensationEngine::decay(float halfLifeSec, float dtSec) {
    for (auto &s : sensations_) {
        s.decay(halfLifeSec, dtSec);
    }
    constexpr float kIntensityEpsilon = 1e-3f;
    /* Prune sensations that are non-positive or have decayed to a negligible
       intensity. kIntensityEpsilon > 0, so this single pass also covers the
       "intensity <= 0" case that used to be a separate erase-remove pass. */
    sensations_.erase(std::remove_if(sensations_.begin(), sensations_.end(),
                                       [kIntensityEpsilon](const PrimalSensation &s) {
                                           return s.intensity < kIntensityEpsilon;
                                       }),
                        sensations_.end());
}

float PrimalSensationEngine::netValence() const {
    if (sensations_.empty()) return 0.0f;
    float total = 0.0f;
    float weight = 0.0f;
    for (const auto &s : sensations_) {
        total += s.valence * s.intensity;
        weight += s.intensity;
    }
    if (weight <= 0.0f) return 0.0f;
    return std::clamp(total / weight, -1.0f, 1.0f);
}

float PrimalSensationEngine::netArousal() const {
    float arousal = 0.0f;
    for (const auto &s : sensations_) {
        arousal = std::max(arousal, s.intensity);
    }
    return std::clamp(arousal, 0.0f, 1.0f);
}

std::optional<PrimalSensation> PrimalSensationEngine::dominant() const {
    if (sensations_.empty()) return std::nullopt;
    auto it = std::max_element(sensations_.begin(), sensations_.end(),
                               [](const PrimalSensation &a, const PrimalSensation &b) {
                                   return a.intensity < b.intensity;
                               });
    return *it;
}

nlohmann::json PrimalSensationEngine::toJson() const {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto &s : sensations_) arr.push_back(s.toJson());
    return arr;
}

PrimalSensationEngine PrimalSensationEngine::fromJson(const nlohmann::json &j) {
    PrimalSensationEngine e;
    if (!j.is_array()) return e;
    for (const auto &item : j) e.add(PrimalSensation::fromJson(item));
    return e;
}

}  // namespace primal
}  // namespace phoenix
