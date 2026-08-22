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
#include <unordered_map>
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
 * @brief Per-sensation homeostatic tuning (subconscious profile parameters).
 *
 * Grounded in allostasis (Sterling 1988) and precision weighting (Friston):
 *  - gain       precision/sensitivity multiplier for the sensation intensity;
 *  - halfLifeSec per-type decay half-life (opponent-process time constant,
 *               Solomon & Corbit 1974); 0 = use the engine's default;
 *  - setpoint   homeostatic setpoint: the desired intensity level; deviation
 *               in either direction is the homeostatic cost.
 */
struct SensationTuning {
    float gain = 1.0f;
    float halfLifeSec = 0.0f;
    float setpoint = 0.0f;
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

    /** Decay all sensations (explicit default half-life). */
    void decay(float halfLifeSec, float dtSec);

    /** Decay all sensations using each type's tuned half-life (falls back to
        defaultHalfLifeSec_, 300 s).  Opponent-process decay: without this,
        sensations only accumulate and a completed mission would leave
        permanent Pain pinning valence at -1. */
    void decayAuto(float dtSec);

    /** Engine-wide default half-life for untuned sensation types. */
    void setDefaultHalfLife(float sec) { defaultHalfLifeSec_ = sec; }
    float defaultHalfLife() const { return defaultHalfLifeSec_; }

    /** Remove all sensations. */
    void clear() { sensations_.clear(); }

    std::vector<PrimalSensation> active() const { return sensations_; }

    /* v8.x context isolation: return only the sensations visible to one
       context.  A source WITHOUT a ':' is GLOBAL (visible everywhere, e.g.
       externally ingested signals); a source WITH a prefix is visible only
       when the prefix matches contextTag.  This keeps mission pressure out
       of chat mood and chat noise out of mission appraisal while the shared
       engines (AGI learner, graph, experience store) stay cross-context.
       Empty contextTag -> all (legacy callers unchanged). */
    std::vector<PrimalSensation> activeFor(const std::string &contextTag) const {
        if (contextTag.empty()) return sensations_;
        const std::string prefix = contextTag + ":";
        std::vector<PrimalSensation> out;
        out.reserve(sensations_.size());
        for (const auto &s : sensations_) {
            const auto pos = s.source.find(':');
            if (pos == std::string::npos)
                out.push_back(s);
            else if (s.source.size() >= prefix.size() &&
                     s.source.compare(0, prefix.size(), prefix) == 0)
                out.push_back(s);
        }
        return out;
    }

    /** Net valence across all sensations, range [-1, 1]. */
    float netValence() const;

    /** Net arousal: how strongly the body is activated, range [0, 1]. */
    float netArousal() const;

    /** Dominant sensation by intensity. */
    std::optional<PrimalSensation> dominant() const;

    /** Install per-sensation tuning (subconscious profile). Empty = untuned. */
    void setTuning(const std::unordered_map<SensationType, SensationTuning> &t) { tuning_ = t; }
    bool hasTuning() const { return !tuning_.empty(); }

    /** Homeostatic cost: Σ gain · |intensity − setpoint| (allostatic drive). */
    float homeostaticCost() const;

    nlohmann::json toJson() const;
    static PrimalSensationEngine fromJson(const nlohmann::json &j);

private:
    std::vector<PrimalSensation> sensations_;
    std::unordered_map<SensationType, SensationTuning> tuning_;
    float defaultHalfLifeSec_{300.0f};
};

}  // namespace primal
}  // namespace phoenix
