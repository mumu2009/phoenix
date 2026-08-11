/* memory_module.hpp - Explicit memory branch with dual-summary design
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version. */

#pragma once

#include "semantic_unit.hpp"

#include <nlohmann/json.hpp>

#include <memory>
#include <string>
#include <vector>

namespace phoenix {
namespace v7 {

/**
 * @brief Configuration for the memory module.
 *
 * The memory module can summarize context with RNN, LSTM or a tiny
 * transformer.  It produces parallel summaries for the backend and the
 * GNN/MemeBarrier.
 */
struct MemoryConfig {
    enum class CellType { RNN, LSTM, Transformer };

    CellType cellType{CellType::LSTM};
    int inputDim{128};
    int hiddenDim{128};
    int outputDim{128};
    int maxSummaryTokens{256};
    bool useTinyLlamaFallback{true};
    std::string tinyLlamaUrl{"http://127.0.0.1:8086"};
    int tinyLlamaTimeoutMs{5000};

    static MemoryConfig fromJson(const nlohmann::json &j);
    nlohmann::json toJson() const;
};

/**
 * @brief Result of the memory branch.
 *
 * Three conceptual handlings of a context summary:
 *   1. forBackend  - a textual / vector summary passed to the main LLM backend.
 *   2. forGnn      - a semantic unit passed to the GNN / MemeBarrier.
 *   3. forUser     - a human-readable summary that can be shown directly.
 */
struct MemorySummary {
    phoenix::multimodal::SemanticUnit forBackend;
    phoenix::multimodal::SemanticUnit forGnn;
    phoenix::multimodal::SemanticUnit forUser;
    bool empty{false};
};

/**
 * @brief Post-generation routing.
 *
 * After sentence generation (TinyLlama / backend), the sentence is converted
 * back into a semantic unit and two parallel copies are produced:
 *   - one for the backend (so it can be used as the next turn's context),
 *   - one for the MemeBarrier / GNN (so it can be stored / aligned).
 */
struct SentenceMemoryFork {
    phoenix::multimodal::SemanticUnit forBackend;
    phoenix::multimodal::SemanticUnit forMemeBarrier;
    std::string text;
};

/**
 * @brief Memory module implementing RNN/LSTM/Transformer context summaries.
 *
 * The module is deliberately independent of the http frontend code.  It
 * operates on SemanticUnits and can be called from any processing stage.
 */
class MemoryModule {
public:
    explicit MemoryModule(const MemoryConfig &cfg = MemoryConfig{});
    ~MemoryModule();

    /** Initialize the recurrent / transformer sub-networks. */
    bool initialize();

    /**
     * @brief Summarize a sequence of context units for the current turn.
     * @param sessionId   Session id.
     * @param context     Previous and current SemanticUnits.
     * @param userPrompt  The incoming user prompt text (if any).
     * @return MemorySummary with forBackend, forGnn and forUser populated.
     */
    MemorySummary summarize(const std::string &sessionId,
                            const std::vector<phoenix::multimodal::SemanticUnit> &context,
                            const std::string &userPrompt);

    /**
     * @brief Fork a generated sentence into backend + MemeBarrier copies.
     */
    SentenceMemoryFork forkSentence(const std::string &sentence) const;

    /** Save / load session memory to a JSON store. */
    bool saveSession(const std::string &sessionId, const std::string &path) const;
    bool loadSession(const std::string &sessionId, const std::string &path);

    /** Health / status. */
    nlohmann::json status() const;

    /** Reset session. */
    void resetSession(const std::string &sessionId);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace v7
} // namespace phoenix
