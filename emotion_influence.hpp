/* emotion_influence.hpp - Evidence-based emotion influence on LLM generation
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version. */

#pragma once

#include "emotion_system.hpp"

#include <nlohmann/json.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace phoenix {

/**
 * @brief Emotion-conditioned generation steering.
 *
 * This namespace deliberately does NOT perform a raw "small matrix multiply"
 * with backend weight matrices.  Instead it uses evidence-based, interpretable
 * levers that are already supported by production inference servers:
 *   - prompt-level modulation (system-prompt / prefix injection),
 *   - token-level logit bias (OpenAI/Ollama/llama-server compatible),
 *   - sampling-parameter adjustment (temperature, top_p, presence/frequency).
 *
 * The mapping from the 8-d emotion tensor to these levers is a small
 * learnable or hand-crafted projection.  For the default implementation we use
 * a deterministic valence-arousal-dominance (PAD) decomposition and a
 * small lexicon of affective tokens.  This is in line with affective computing
 * literature (Mehrabian PAD, Warriner et al. affective norms, EmotionPrompt
 * style prompt conditioning) and keeps the influence inspectable and testable.
 */
namespace emotion_influence {

struct EmotionInfluenceResult {
    /** Logit bias map: token string -> bias value. */
    nlohmann::json logitBias;

    /** Natural-language directive to prepend / append to the system prompt. */
    std::string promptModulation;

    /** Sampling adjustments. */
    nlohmann::json inferenceOptions;

    /** Human-readable summary of the influence. */
    std::string explanation;
};

struct EmotionInfluenceConfig {
    /** Overall strength of the influence. */
    float strength{1.0f};

    /** Bias clamp. */
    float maxBias{2.0f};
    float minBias{-2.0f};

    /** Default temperature when emotion is neutral. */
    float baseTemperature{0.7f};
    float baseTopP{0.9f};

    /** Vocabulary used for token-level steering. */
    std::unordered_map<std::string, std::vector<std::string>> affectiveLexicon;

    /** Per-token score decay between turns. */
    float decay{0.95f};

    static EmotionInfluenceConfig fromJson(const nlohmann::json &j);
    nlohmann::json toJson() const;
};

/**
 * @brief Small learnable / deterministic emotion-to-control projection.
 *
 * The design is intentionally modular so it can be replaced with a trained
 * adapter (LoRA, prompt-tuning, etc.) without touching the backend model.
 */
class EmotionInfluence {
public:
    explicit EmotionInfluence(const EmotionInfluenceConfig &cfg = EmotionInfluenceConfig{});
    ~EmotionInfluence();

    /** Set or replace the affective lexicon at runtime. */
    void setAffectiveLexicon(
        const std::unordered_map<std::string, std::vector<std::string>> &lexicon);

    /**
     * @brief Compute logit bias, prompt modulation and sampling options.
     *
     * @param emotion   Current 8-d emotion tensor.
     * @param context   Optional recent text context (for token scoring).
     * @param turn      Conversation turn number.
     * @return EmotionInfluenceResult with all three levers.
     */
    EmotionInfluenceResult compute(
        const phoenix::emotion::EmotionTensor &emotion,
        const std::string &context = "",
        int64_t turn = 0) const;

    /** Same as above but stateful: applies internal decay. */
    EmotionInfluenceResult computeStateful(
        const phoenix::emotion::EmotionTensor &emotion,
        const std::string &sessionId,
        const std::string &context = "",
        int64_t turn = 0);

    /** Reset per-session state. */
    void resetSession(const std::string &sessionId);

    /** Status. */
    nlohmann::json status() const;

private:
    EmotionInfluenceConfig cfg_;
    struct State;
    std::unordered_map<std::string, std::unique_ptr<State>> sessionStates_;

    float tokenScore_(const std::string &emotionKey,
                      float emotionValue) const;
    std::string buildPromptModulation_(const phoenix::emotion::EmotionTensor &e) const;
    nlohmann::json buildInferenceOptions_(const phoenix::emotion::EmotionTensor &e) const;
};

} // namespace emotion_influence
} // namespace phoenix
