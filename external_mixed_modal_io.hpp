/* external_mixed_modal_io.hpp - Mixed-modal external I/O for Phoenix v7.0 "Arthur"
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version. */

#pragma once

#include "semantic_unit.hpp"
#include <cstdint>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace phoenix {
namespace io {

/**
 * @brief Modalities supported by the external mixed-modal I/O layer.
 *
 * Mirrors phoenix::multimodal::Modality but adds an explicit Unknown value.
 */
enum class MixedModalModality {
    Unknown = 0,
    Text,
    Image,
    Audio,
    Video,
    Sensor,
    Structured
};

/**
 * @brief A single mixed-modal packet from or to an external channel.
 */
struct MixedModalPacket {
    std::string id;                         /*!< Correlation/trace id. */
    MixedModalModality modality = MixedModalModality::Unknown;
    std::vector<uint8_t> payload;           /*!< Raw payload bytes. */
    std::string mimeType;                   /*!< MIME hint (e.g. image/png). */
    std::string source;                     /*!< Channel/source name. */
    uint64_t timestampMs = 0;                /*!< UTC milliseconds. */
    nlohmann::json metadata;                /*!< Extra modality metadata. */

    nlohmann::json toJson() const;
    static MixedModalPacket fromJson(const nlohmann::json &j);

    static std::string modalityToString(MixedModalModality m);
    static MixedModalModality stringToModality(const std::string &s);

    /**
     * @brief Convert to a concept-space semantic unit for downstream fusion.
     *
     * Text uses the token-encoder adapter, visual media accepts world-model
     * concept vectors or visual features, and audio uses the persistent speech
     * concept model trained from audio-transcript pairs.
     */
    phoenix::multimodal::SemanticUnit toSemanticUnit(size_t targetDim = 0,
                                                     const std::string &contentHint = "") const;
};

class MixedModalConceptBridge {
public:
    static phoenix::multimodal::SemanticUnit encode(const MixedModalPacket &packet,
                                                     size_t targetDim = 0,
                                                     const std::string &contentHint = "");
    static bool pretrainSpeech(const MixedModalPacket &audio,
                               const std::string &transcript,
                               size_t targetDim = 0);
    static MixedModalPacket decode(const phoenix::multimodal::SemanticUnit &unit,
                                   MixedModalModality target,
                                   const std::string &source = "");
    static nlohmann::json status();
};

/**
 * @brief Thread-safe inbound packet buffer.
 */
class MixedModalInputBuffer {
public:
    void push(MixedModalPacket packet);
    std::vector<MixedModalPacket> flush();
    bool empty() const;
    size_t size() const;
    nlohmann::json toJson() const;

private:
    mutable std::mutex mutex_;
    std::vector<MixedModalPacket> buffer_;
};

/**
 * @brief Thread-safe outbound packet queue.
 */
class MixedModalOutputQueue {
public:
    void push(MixedModalPacket packet);
    std::vector<MixedModalPacket> drain(size_t max = 0);
    bool empty() const;
    size_t size() const;
    nlohmann::json toJson() const;

private:
    mutable std::mutex mutex_;
    std::vector<MixedModalPacket> queue_;
};

/**
 * @brief Registry of named external mixed-modal channels.
 */
class MixedModalChannelRegistry {
public:
    void registerSource(const std::string &name, const std::string &mimeType);
    std::vector<std::string> sources() const;
    nlohmann::json toJson() const;

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::string> sources_;
};

}  // namespace io
}  // namespace phoenix
