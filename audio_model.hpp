/* audio_model.hpp - Semantic interface for 1D speech world models
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version. */

#pragma once

#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace phoenix {
namespace io {

/**
 * @brief Description of a 1D audio world-model variant.
 *
 * Variants differ by target sample rate and patch stride.  They correspond to
 * the standard wav2vec 2.0 / HuBERT style 25ms windows with 10ms stride.
 */
struct AudioModelConfig {
  std::string id;          /**< internal variant id, e.g. "audio-16k" */
  std::string arch;        /**< architecture string, e.g. "jepa_1d" */
  std::string repo;        /**< HuggingFace repo or checkpoint URL */
  int sampleRate = 16000;  /**< intended sample rate in Hz */
  int windowSamples = 400; /**< analysis window length in samples */
  int strideSamples = 160; /**< window hop in samples */
  long long params = 0;    /**< approximate parameter count */
  std::string dataset;     /**< pretraining dataset, e.g. "LibriSpeech" */
  std::string weightsFile = "model.safetensors"; /**< primary checkpoint filename inside localWeightsDir */
  std::string localWeightsDir; /**< directory under runtime_store/models/ijepa/; empty means "<id>/" */
};

/**
 * @brief Semantic interface for a 1D audio world model.
 *
 * A 1D JEPA splits an audio waveform into temporal patches (context/target
 * windows), encodes them into a joint embedding space, and predicts masked
 * targets from visible context.  The interface also exposes an audio-text
 * contrastive pretraining step so the audio encoder can align with a paired
 * text concept vector.
 */
class AudioModel {
 public:
  virtual ~AudioModel() = default;

  /**
   * @brief Encode an audio waveform into a semantic concept vector.
   */
  virtual std::vector<float> encode(const std::vector<uint8_t> &audioBytes,
                                    int sampleRate,
                                    const std::string &mimeType) = 0;

  /**
   * @brief Encode the visible context windows of an audio waveform.
   *
   * @param mask  flattened bool mask, true = visible window, false = masked.
   *              If empty the full waveform is used as context.
   */
  virtual std::vector<float> encodeContext(const std::vector<uint8_t> &audioBytes,
                                           int sampleRate,
                                           const std::string &mimeType,
                                           const std::vector<bool> &mask) = 0;

  /**
   * @brief Encode a set of target windows.
   *
   * @param windowIndices list of window indices forming the target block.
   */
  virtual std::vector<float> encodeTarget(const std::vector<uint8_t> &audioBytes,
                                          int sampleRate,
                                          const std::string &mimeType,
                                          const std::vector<int> &windowIndices) = 0;

  /**
   * @brief Predict a target representation from a context representation.
   *
   * @param contextRepr  context embedding from encodeContext().
   * @param targetPos    positional tokens of the target window to predict.
   */
  virtual std::vector<float> predictTarget(const std::vector<float> &contextRepr,
                                           const std::vector<int> &targetPos) = 0;

  /**
   * @brief Self-supervised domain adaptation step.
   *
   * Runs one JEPA prediction step on the provided audio: sample target windows,
   * predict them from the context, and update the predictor/encoder weights.
   *
   * @return approximate loss value (>=0), or -1 on failure.
   */
  virtual float adapt(const std::vector<uint8_t> &audioBytes,
                      int sampleRate,
                      const std::string &mimeType,
                      int steps = 1,
                      float lr = 1e-4f) = 0;

  /**
   * @brief Speech-text contrastive pretraining step.
   *
   * Encodes the audio, compares it with the supplied text concept vector, and
   * updates an internal speech-to-text alignment buffer.
   *
   * @param textConcept   concept vector produced by a text encoder for the
   *                      transcript paired with the audio.
   * @param temperature   softmax temperature used for the contrastive loss.
   * @return contrastive loss value (>=0), or -1 on failure.
   */
  virtual float contrastiveAdapt(const std::vector<uint8_t> &audioBytes,
                                 int sampleRate,
                                 const std::string &mimeType,
                                 const std::vector<float> &textConcept,
                                 float temperature = 0.1f) = 0;

  /**
   * @brief Render a concept vector back to an audio payload.
   *
   * A world model is not a generative decoder, so implementations may emit a
   * concept-laden payload with a downstream-decoder flag or a low-level
   * deterministic waveform.
   */
  virtual std::vector<uint8_t> decode(const std::vector<float> &conceptVector,
                                      const std::string &mimeType = "audio/wav",
                                      size_t lengthHint = 0) = 0;

  /** @brief Runtime status (variant, loaded backend, sample count). */
  virtual nlohmann::json status() const = 0;

  /** @brief Configuration of this model instance. */
  virtual const AudioModelConfig &config() const = 0;

  /** @brief Optional per-instance role used for the "backend" status field. */
  mutable std::string kind_;
};

/**
 * @brief List the 1D audio variants by sample rate.
 */
inline std::vector<AudioModelConfig> audioOfficialVariants() {
  return {
      {"audio-encoder",      "jepa_1d", "runtime_store/models/additive_jepa/speech_encoder", 16000, 400, 160, 0, "LibriSpeech", "best.onnx", "speech_encoder"},
      {"audio-16k",          "jepa_1d", "runtime_store/models/ijepa/speech_16k", 16000, 400, 160, 0, "LibriSpeech", "model.safetensors", "speech_16k"},
      {"audio-22k",          "jepa_1d", "facebook/audio-22k", 22050, 512, 256, 0, "LibriLight", "model.safetensors", "speech_22k"},
      {"audio-44k",          "jepa_1d", "facebook/audio-44k", 44100, 1024, 512, 0, "VoxPopuli", "model.safetensors", "speech_44k"},
      {"audio-48k",          "jepa_1d", "facebook/audio-48k", 48000, 1024, 512, 0, "VoxPopuli", "model.safetensors", "speech_48k"}};
}

/**
 * @brief Factory: create a 1D audio world model instance.
 *
 * @param variantId   one of the ids returned by audioOfficialVariants().
 * @param targetDim   desired output concept dimension.
 * @param backend     retained for API compatibility.
 * @return a concrete implementation. Local execution uses the compiled RDK X5
 *         speech JEPA BPU model; fallback is used when BPU runtime is unavailable.
 */
std::unique_ptr<AudioModel> createAudioModel(
    const std::string &variantId = "audio-16k",
    int targetDim = 0,
    const std::string &backend = "auto");

using AudioEncoder = AudioModel;
using AudioDecoder = AudioModel;

std::unique_ptr<AudioEncoder> createAudioEncoder(
    const std::string &variantId = "audio-16k",
    int targetDim = 0,
    const std::string &backend = "auto");

std::unique_ptr<AudioDecoder> createAudioDecoder(
    const std::string &variantId = "audio-16k",
    int targetDim = 0,
    const std::string &backend = "auto");

/**
 * @brief Find an official variant config by id.
 */
inline const AudioModelConfig *findAudioModelVariant(
    const std::string &id) {
  static const std::vector<AudioModelConfig> kVariants =
      audioOfficialVariants();
  for (const auto &v : kVariants) {
    if (v.id == id) return &v;
  }
  return nullptr;
}

}  // namespace io
}  // namespace phoenix
