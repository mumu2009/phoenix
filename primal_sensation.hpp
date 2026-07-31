/* primal_sensation.hpp - Primal sensation layer for Phoenix v7.0 "Arthur"
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version. */

#pragma once

#include <cmath>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace phoenix {
namespace primal {

/**
 * @brief Biological primary sensations distinct from learned emotions.
 *
 * Based on interoceptive signals such as pain, hunger, fatigue, thermal state,
 * threat, social isolation and novelty.  These sensations provide the
 * valence/arousal foundation that drives higher-level affect.
 */
enum class SensationType {
    Unknown = 0,
    Pain,
    Pleasure,
    Hunger,
    Temperature,
    Fatigue,
    Threat,
    SocialIsolation,
    Novelty,
    MAX
};

/**
 * @brief A primary interoceptive/exteroceptive sensation event.
 */
struct PrimalSensation {
    SensationType type = SensationType::Unknown;
    float intensity = 0.0f;      /*!< Signal magnitude, normalised [0, 1]. */
    float valence = 0.0f;        /*!< Negative (harm) to positive (benefit), [-1, 1]. */
    float durationSec = 0.0f;      /*!< Estimated duration of the sensation. */
    std::string source;          /*!< Free-form source tag (sensor, text, etc.). */
    uint64_t timestampMs = 0;     /*!< UTC milliseconds. */

    nlohmann::json toJson() const;
    static PrimalSensation fromJson(const nlohmann::json &j);
    static std::string typeToString(SensationType t);
    static SensationType stringToType(const std::string &s);

    /**
     * @brief Decay the intensity of this sensation.
     */
    void decay(float halfLifeSec, float dtSec) {
        if (halfLifeSec <= 0.0f || dtSec <= 0.0f || intensity <= 0.0f) return;
        intensity *= std::pow(0.5f, dtSec / halfLifeSec);
        if (intensity < 1e-5f) intensity = 0.0f;
    }
};

/**
 * @brief Aggregation layer for primal sensations.
 *
 * Tracks active sensations, decays them over time and exposes net valence
 * and arousal signals to the instinct/effect systems.
 */
class PrimalSensationEngine {
public:
    void add(const PrimalSensation &s);

    /** Decay all sensations. */
    void decay(float halfLifeSec, float dtSec);

    /** Remove all sensations. */
    void clear() { sensations_.clear(); }

    std::vector<PrimalSensation> active() const { return sensations_; }

    /** Net valence across all sensations, range [-1, 1]. */
    float netValence() const;

    /** Net arousal: how strongly the body is activated, range [0, 1]. */
    float netArousal() const;

    /** Dominant sensation by intensity. */
    std::optional<PrimalSensation> dominant() const;

    nlohmann::json toJson() const;
    static PrimalSensationEngine fromJson(const nlohmann::json &j);

private:
    std::vector<PrimalSensation> sensations_;
};

}  // namespace primal
}  // namespace phoenix
