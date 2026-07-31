/* prompt_split.hpp - Prompt split (system vs memory) for Phoenix v7.0 "Arthur"
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version. */

#pragma once

#include "emotion_system.hpp"
#include "modern_context_system.hpp"
#include "semantic_unit.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace phoenix {
namespace prompt {

/**
 * @brief Immutable core system prompt.
 *
 * Contains the identity, hard constraints and non-negotiable directives.
 * This portion is kept stable across turns and is not derived from memory.
 */
struct SystemPrompt {
    std::string identity;       /*!< Who/what the assistant is. */
    std::string version;        /*!< Version tag, e.g. "Arthur v7.0". */
    std::string constraints;      /*!< Hard rules and safety guardrails. */
    std::string coreDirective;  /*!< Primary mission statement. */

    nlohmann::json toJson() const;
    static SystemPrompt fromJson(const nlohmann::json &j);

    /** Default Phoenix Arthur v7.0 system prompt. */
    static SystemPrompt arthurDefault();
};

/**
 * @brief Dynamic, memory-derived prompt content.
 *
 * The memory prompt is regenerated each turn from relevant context entries,
 * semantic memory and instinctive drives.  It is a "soft" overlay that can
 * be updated, trimmed or replaced without changing the immutable system core.
 */
struct MemoryPrompt {
    std::string summary;                     /*!< Compact situation summary. */
    std::vector<std::string> relevantFacts;  /*!< Key facts from memory. */
    std::vector<std::string> activeGoals;    /*!< Currently tracked goals. */
    std::string emotionalTone;               /*!< Emotional/primal tone hint. */
    std::string benefitHarmBias;             /*!< Human-readable "approach/avoid/wait" directive. */
    std::vector<float> driveVector;          /*!< 8-dim numeric emotion-direction vector. */
    phoenix::emotion::EmotionTensor emotionTensor; /*!< 8-dim affect tensor. */
    nlohmann::json inferenceOptions;         /*!< LLM sampling options derived from affect. */

    nlohmann::json toJson() const;
    static MemoryPrompt fromJson(const nlohmann::json &j);

    static MemoryPrompt empty();
};

/**
 * @brief Composes the final prompt from immutable system + dynamic memory + user.
 *
 * The split allows the runtime to keep the system prompt fixed while
 * continuously rewriting the memory portion from context and affect signals.
 */
class PromptComposer {
public:
    PromptComposer(const SystemPrompt &system, const MemoryPrompt &memory = MemoryPrompt::empty());

    /** Compose the final text for a downstream LLM call. */
    std::string compose(const std::string &userPrompt,
                        bool includeMemory = true,
                        const std::string &separator = "\n---\n") const;

    /** Compose structured messages suitable for JSON chat APIs. */
    nlohmann::json composeMessages(const std::string &userPrompt, bool includeMemory = true) const;

    /** Build a memory prompt from a list of context entries. */
    static MemoryPrompt fromContext(const std::vector<phoenix::context::ContextEntry> &context,
                                    size_t maxFacts = 5);

    /** Build a memory prompt from semantic memory search. */
    static MemoryPrompt fromSemanticMemory(const phoenix::multimodal::SemanticMemory &memory,
                                           const phoenix::multimodal::SemanticUnit &query,
                                           size_t topK = 5);

    void setMemory(const MemoryPrompt &memory) { memory_ = memory; }
    const SystemPrompt &system() const { return system_; }
    const MemoryPrompt &memory() const { return memory_; }

    /** Numeric drive vector currently encoded in memory. */
    std::vector<float> driveVector() const { return memory_.driveVector; }

    /** Emotion tensor currently encoded in memory. */
    phoenix::emotion::EmotionTensor emotionTensor() const { return memory_.emotionTensor; }

    /** LLM sampling options derived from the affective state. */
    nlohmann::json inferenceOptions() const { return memory_.inferenceOptions; }

    /** Human-readable directive summarising the affective state. */
    std::string modulationHint() const { return memory_.emotionTensor.modulationHint(); }

private:
    SystemPrompt system_;
    MemoryPrompt memory_;
};

}  // namespace prompt
}  // namespace phoenix
