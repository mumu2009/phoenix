/* ahead_module.hpp - Pre-GNN "ahead" processing module for Phoenix v7.0 "Arthur"
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version. */

#pragma once

#include "external_mixed_modal_io.hpp"
#include "semantic_unit.hpp"
#include "emotion_system.hpp"
#include "memory_module.hpp"
#include "emotion_influence.hpp"

#include <nlohmann/json.hpp>

#include <memory>
#include <string>
#include <vector>

namespace phoenix {
namespace v7 {

/**
 * @brief Output produced by the ahead-of-GNN processing stage.
 *
 * The aheadModule sits before the GNN/MemeBarrier and the backend.  It
 * converts raw incoming packets to SemanticUnits, produces memory summaries,
 * emotion-conditioned inference options, and pre-routes generated sentences
 * along the two required branches (backend + MemeBarrier).
 */
struct AheadResult {
    /** Encoded semantic units from the current user turn. */
    std::vector<phoenix::multimodal::SemanticUnit> units;

    /** Memory summary intended for the main LLM backend. */
    phoenix::multimodal::SemanticUnit memoryForBackend;

    /** Memory summary intended for the GNN / MemeBarrier. */
    phoenix::multimodal::SemanticUnit memoryForGnn;

    /** Emotion state inferred from input and previous context. */
    phoenix::emotion::EmotionTensor emotion;

    /** Emotion-derived logit bias / prompt modulation for the backend. */
    phoenix::emotion_influence::EmotionInfluenceResult emotionInfluence;

    /** Whether the memory branch produced usable context. */
    bool memoryReady{false};
};

/**
 * @brief Configuration for the ahead module.
 */
struct AheadConfig {
    /** Concept vector dimension. */
    size_t conceptDim{128};

    /** Whether to enable the memory branch. */
    bool memoryEnabled{true};

    /** Whether to enable emotion conditioning. */
    bool emotionEnabled{true};

    /** Whether audio/video encoder/decoder are active. */
    bool audioVideoEncDecEnabled{true};

    /** Async scheduler name for downstream work. */
    std::string schedulerName{"ahead_default"};

    static AheadConfig fromJson(const nlohmann::json &j);
    nlohmann::json toJson() const;
};

/**
 * @brief Pre-GNN broad preprocessing module.
 *
 * This is NOT the http reverse-proxy in frontend_server.cpp.  The aheadModule
 * is a conceptual first stage: it receives a list of MixedModalPackets,
 * encodes them into SemanticUnits, runs the memory branch and the emotion
 * branch in parallel, and forwards results to the GNN/MemeBarrier and the
 * backend.
 */
class AheadModule {
public:
    explicit AheadModule(const AheadConfig &cfg = AheadConfig{});
    ~AheadModule();

    /** Initialize sub-modules (memory, emotion). */
    bool initialize();

    /**
     * @brief Process a batch of incoming packets.
     * @param packets  Raw multimodal input packets.
     * @param sessionId  Session id for emotion / memory.
     * @return AheadResult with all branches populated.
     */
    AheadResult processBatch(
        const std::vector<phoenix::io::MixedModalPacket> &packets,
        const std::string &sessionId);

    /**
     * @brief Post-process a generated sentence.
     *
     * After the backend sentence generator (TinyLlama / llama3.1) produces a
     * sentence, it is passed back to the aheadModule.  The memory branch emits
     * one copy for the backend and one copy for the MemeBarrier; these two
     * branches are NOT serial.
     */
    struct SentenceResult {
        std::string textForBackend;
        std::string textForMemeBarrier;
        phoenix::multimodal::SemanticUnit unitForBackend;
        phoenix::multimodal::SemanticUnit unitForMemeBarrier;
        bool blockedByMemeBarrier{false};
    };

    SentenceResult routeSentence(const std::string &sentence,
                                 const std::string &sessionId);

    /** Status / health of all encoders and sub-modules. */
    nlohmann::json status() const;

    /** Reset per-session state. */
    void resetSession(const std::string &sessionId);

private:
    AheadConfig cfg_;
    std::unique_ptr<MemoryModule> memory_;
    std::unique_ptr<phoenix::emotion::EmotionSystem> emotion_;
    std::unique_ptr<emotion_influence::EmotionInfluence> emotionInfluence_;

    std::vector<phoenix::multimodal::SemanticUnit> encodePackets_(
        const std::vector<phoenix::io::MixedModalPacket> &packets);

    phoenix::emotion::EmotionTensor observeEmotion_(
        const std::string &sessionId,
        const std::vector<phoenix::multimodal::SemanticUnit> &units);
};

} // namespace v7
} // namespace phoenix
