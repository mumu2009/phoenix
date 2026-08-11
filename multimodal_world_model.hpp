/* multimodal_world_model.hpp - LLaVA / Qwen2-Audio based encoders
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version. */

#pragma once

#include "semantic_unit.hpp"

#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace phoenix {
namespace io {

/**
 * @brief Result of encoding an image or video frame.
 *
 * Unlike the old JEPA world model that produced a single 64/128-d concept
 * vector, the LLaVA vision projector outputs one 4096-d unit query per image
 * patch (576 patches for a 336x336 image).  These queries already live in the
 * llama3.1 8b hidden space, so the main matrix can consume them directly.
 *
 * The mean unit query is kept as the semantic vector; the per-patch sequence
 * is stored in the SemanticUnit so the backend can attend over the full image.
 */
struct MultimodalImageEncResult {
  std::vector<std::vector<float>> unitQueries;
  std::vector<float> meanUnitQuery;
  std::string model;
  std::string error;
};

/**
 * @brief Result of encoding an audio clip.
 *
 * Qwen2-Audio's audio tower + connector emits one 4096-d unit query per
 * 40ms temporal frame.  The mean is the summary; the sequence can be fed to
 * the main matrix as a continuous-time audio token stream.
 */
struct MultimodalAudioEncResult {
  std::vector<std::vector<float>> unitQueries;
  std::vector<float> meanUnitQuery;
  std::string model;
  std::string error;
};

/**
 * @brief Decoder output for image or audio.
 */
struct MultimodalDecResult {
  std::vector<uint8_t> payload;
  std::string mimeType;
  std::string model;
  std::string error;
};

/**
 * @brief Configuration for the external multimodal enc/dec Python service.
 */
struct MultimodalEncDecConfig {
  std::string baseUrl = "http://127.0.0.1:8085";
  std::string imageEncoderModel = "llava-1.5-7b";
  std::string audioEncoderModel = "qwen2-audio-7b";
  int timeoutMs = 120000;
};

/**
 * @brief Image / video encoder using an external LLaVA-based service.
 *
 * The service is the single source of truth for converting raw pixels into the
 * llama3.1 8b unit-query space.  C++ code is intentionally thin: it ships the
 * payload over HTTP and packs the returned queries into a SemanticUnit.
 */
class MultimodalImageWorldModel {
 public:
  explicit MultimodalImageWorldModel(const MultimodalEncDecConfig &cfg = {});
  ~MultimodalImageWorldModel();

  MultimodalImageEncResult encode(const std::vector<uint8_t> &imageBytes,
                                  int width,
                                  int height,
                                  const std::string &mimeType);

  MultimodalDecResult decode(const std::vector<float> &meanUnitQuery,
                             const std::vector<std::vector<float>> &unitQueries,
                             const std::string &mimeType,
                             int width,
                             int height);

  nlohmann::json status() const;
  const std::string &model() const;

 private:
  MultimodalEncDecConfig cfg_;
  mutable nlohmann::json lastStatus_;
  mutable bool statusLoaded_ = false;
};

/**
 * @brief Audio encoder using an external Qwen2-Audio-based service.
 */
class MultimodalAudioWorldModel {
 public:
  explicit MultimodalAudioWorldModel(const MultimodalEncDecConfig &cfg = {});
  ~MultimodalAudioWorldModel();

  MultimodalAudioEncResult encode(const std::vector<uint8_t> &audioBytes,
                                  int sampleRate,
                                  const std::string &mimeType);

  MultimodalDecResult decode(const std::vector<float> &meanUnitQuery,
                             const std::vector<std::vector<float>> &unitQueries,
                             const std::string &mimeType,
                             size_t lengthHint);

  nlohmann::json status() const;
  const std::string &model() const;

 private:
  MultimodalEncDecConfig cfg_;
  mutable nlohmann::json lastStatus_;
  mutable bool statusLoaded_ = false;
};

}  // namespace io
}  // namespace phoenix
