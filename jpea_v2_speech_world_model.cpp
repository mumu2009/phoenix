/* jpea_v2_speech_world_model.cpp - Factory and HBDNN/fallback for 1D speech world model interface
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version. */

#include "jpea_v2_speech_world_model.hpp"
#include "model_deployment.hpp"
#include "phoenix_config.hpp"
#include "rdk_x5_bpu.hpp"
#include "semantic_unit.hpp"

#ifndef PHOENIX_EDGE_SPEECH_ENABLED
#define PHOENIX_EDGE_SPEECH_ENABLED 1
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>

namespace phoenix {
namespace io {

namespace {

constexpr int kFixedInputSamples = 16000;
constexpr int kEncoderOutputDim = 128;
constexpr int kDecoderOutputSamples = 15872;

std::string resolveBpuModelPath(const std::string &envOverride,
                                const std::string &modelKind,
                                const std::string &variantId) {
  if (!envOverride.empty()) return envOverride;

  const std::vector<std::string> names = [modelKind]() {
    if (modelKind == "encoder") return std::vector<std::string>{"model_encoder.bin", "encoder.bin", "model.bin", "ijepa.bin"};
    if (modelKind == "decoder") return std::vector<std::string>{"model_decoder.bin", "decoder.bin", "decode.bin"};
    return std::vector<std::string>{"model.bin"};
  }();

  // Map the 16 kHz speech variant to the real on-device folder layout.
  std::string folder = variantId;
  if (variantId == "jpea_v2_speech_16k" || variantId == "speech_16k") {
    folder = "speech_16k";
  }

  std::vector<std::string> roots;
  roots.push_back(std::string("runtime_store/models/ijepa/") + folder);
  if (folder != variantId) {
    roots.push_back(std::string("runtime_store/models/ijepa/") + variantId);
  }
  roots.push_back(std::string("runtime_store/models/ijepa/"));

  std::error_code ec;
  for (const auto &root : roots) {
    for (const auto &name : names) {
      std::filesystem::path p = std::filesystem::path(root) / name;
      if (std::filesystem::is_regular_file(p, ec)) return p.string();
    }
  }
  return {};
}

static std::filesystem::path temporaryInputPath() {
  static std::atomic<uint64_t> sequence{0};
  return std::filesystem::temp_directory_path() /
         ("phoenix-jpea-speech-" + std::to_string(sequence.fetch_add(1)) + ".tensor");
}

/**
 * @brief Local HBDNN-backed 1D speech world model.
 *
 * Uses the RDK X5 hbDNN runtime for both the whole-clip speech encoder
 * (1x1x1x16000 -> 1x128) and the waveform decoder (1x128x1x1 -> 1x1x1x15872).
 */
class JpeaV2SpeechHbdnnModel : public JpeaV2SpeechWorldModel {
 public:
  JpeaV2SpeechHbdnnModel(JpeaV2SpeechWorldModelConfig cfg, int targetDim)
      : cfg_(std::move(cfg)),
        targetDim_(targetDim > 0 ? targetDim : 128),
        modelPath_(resolveBpuModelPath(phoenix::resolveConfig<std::string>("jpea.speech.horizonModel", "", "JPEA_SPEECH_HORIZON_MODEL"), "encoder", cfg_.id)),
        decoderPath_(resolveBpuModelPath(phoenix::resolveConfig<std::string>("jpea.speech.horizonDecoderModel", "", "JPEA_SPEECH_HORIZON_DECODER_MODEL"), "decoder", cfg_.id)),
        predictorWeights_(static_cast<size_t>(targetDim_ * targetDim_), 0.0f),
        predictorBias_(static_cast<size_t>(targetDim_), 0.0f),
        textAlignment_(static_cast<size_t>(targetDim_), 0.0f) {
    for (int i = 0; i < targetDim_; ++i) {
      predictorWeights_[static_cast<size_t>(i * targetDim_ + i)] = 1.0f;
    }
  }

  std::vector<float> encode(const std::vector<uint8_t> &audioBytes,
                            int sampleRate,
                            const std::string &mimeType) override {
    auto samples = preprocessAudio(audioBytes, sampleRate, mimeType);
    auto input = prepareInput(samples, {});
    return runEncoderBpu(input);
  }

  std::vector<float> encodeContext(const std::vector<uint8_t> &audioBytes,
                                   int sampleRate,
                                   const std::string &mimeType,
                                   const std::vector<bool> &mask) override {
    auto samples = preprocessAudio(audioBytes, sampleRate, mimeType);
    auto input = prepareInput(samples, mask);
    return runEncoderBpu(input);
  }

  std::vector<float> encodeTarget(const std::vector<uint8_t> &audioBytes,
                                  int sampleRate,
                                  const std::string &mimeType,
                                  const std::vector<int> &windowIndices) override {
    auto samples = preprocessAudio(audioBytes, sampleRate, mimeType);
    const int windows = countWindows(static_cast<int>(samples.size()));
    std::vector<bool> mask;
    if (!windowIndices.empty()) {
      mask.assign(static_cast<size_t>(windows), false);
      for (int idx : windowIndices) {
        if (idx >= 0 && idx < windows) mask[static_cast<size_t>(idx)] = true;
      }
    }
    auto input = prepareInput(samples, mask);
    return runEncoderBpu(input);
  }

  std::vector<float> predictTarget(const std::vector<float> &contextRepr,
                                   const std::vector<int> & /*targetPos*/) override {
    if (contextRepr.size() != static_cast<size_t>(targetDim_)) {
      lastError_ = "predictor input dimension mismatch";
      return {};
    }
    std::vector<float> out(static_cast<size_t>(targetDim_), 0.0f);
    for (int i = 0; i < targetDim_; ++i) {
      float v = predictorBias_[static_cast<size_t>(i)];
      for (int j = 0; j < targetDim_; ++j) {
        v += predictorWeights_[static_cast<size_t>(i * targetDim_ + j)] * contextRepr[static_cast<size_t>(j)];
      }
      out[static_cast<size_t>(i)] = v;
    }
    return out;
  }

  float adapt(const std::vector<uint8_t> &audioBytes,
              int sampleRate,
              const std::string &mimeType,
              int steps,
              float lr) override {
    if (steps < 1 || lr < 0.0f) {
      lastError_ = "invalid adaptation parameters";
      return -1.0f;
    }
    if (targetDim_ <= 0) {
      lastError_ = "target dimension not set";
      return -1.0f;
    }

    auto samples = preprocessAudio(audioBytes, sampleRate, mimeType);
    const int totalWindows = countWindows(static_cast<int>(samples.size()));
    if (totalWindows < 2) {
      lastError_ = "not enough windows for predictor adaptation";
      return -1.0f;
    }

    std::mt19937 rng(static_cast<unsigned>(adaptSteps_ + 0x1DEA));
    float totalLoss = 0.0f;
    for (int step = 0; step < steps; ++step) {
      std::vector<int> order(totalWindows);
      std::iota(order.begin(), order.end(), 0);
      std::shuffle(order.begin(), order.end(), rng);

      const int targetCount = std::max(1, totalWindows / 3);
      std::vector<int> targetIndices(order.begin(), order.begin() + targetCount);

      std::vector<bool> contextMask(static_cast<size_t>(totalWindows), true);
      for (int t : targetIndices) contextMask[static_cast<size_t>(t)] = false;

      auto context = encodeContext(audioBytes, sampleRate, mimeType, contextMask);
      auto target = encodeTarget(audioBytes, sampleRate, mimeType, targetIndices);
      if (context.size() != static_cast<size_t>(targetDim_) ||
          target.size() != static_cast<size_t>(targetDim_)) {
        lastError_ = "context/target dimension mismatch";
        return -1.0f;
      }

      auto pred = predictTarget(context, targetIndices);
      if (pred.size() != target.size()) {
        lastError_ = "predictor output dimension mismatch";
        return -1.0f;
      }

      double loss = 0.0;
      std::vector<float> error(pred.size());
      for (size_t i = 0; i < pred.size(); ++i) {
        error[i] = pred[i] - target[i];
        loss += static_cast<double>(error[i]) * static_cast<double>(error[i]);
      }
      loss /= static_cast<double>(pred.size());
      totalLoss += static_cast<float>(loss);

      for (int i = 0; i < targetDim_; ++i) {
        for (int j = 0; j < targetDim_; ++j) {
          predictorWeights_[static_cast<size_t>(i * targetDim_ + j)] -=
              lr * error[static_cast<size_t>(i)] * context[static_cast<size_t>(j)];
        }
        predictorBias_[static_cast<size_t>(i)] -= lr * error[static_cast<size_t>(i)];
      }
      ++adaptSteps_;
      ++samples_;
    }

    lastError_.clear();
    return totalLoss / static_cast<float>(steps);
  }

  float contrastiveAdapt(const std::vector<uint8_t> &audioBytes,
                         int sampleRate,
                         const std::string &mimeType,
                         const std::vector<float> &textConcept,
                         float /*temperature*/) override {
    if (audioBytes.empty() || textConcept.empty()) {
      lastError_ = "contrastive adaptation requires audio and text concept";
      return -1.0f;
    }
    auto speechConcept = encode(audioBytes, sampleRate, mimeType);
    if (speechConcept.empty()) {
      lastError_ = "speech encoding failed";
      return -1.0f;
    }

    auto textInSpeechSpace = phoenix::multimodal::projectToDimension(textConcept, static_cast<size_t>(targetDim_), 0x53505458U);
    if (textInSpeechSpace.size() != speechConcept.size()) {
      lastError_ = "contrastive adaptation dimension mismatch";
      return -1.0f;
    }

    const float n = static_cast<float>(contrastiveSamples_);
    for (size_t i = 0; i < textAlignment_.size(); ++i) {
      const float diff = textInSpeechSpace[i] - speechConcept[i];
      textAlignment_[i] = (textAlignment_[i] * n + diff) / (n + 1.0f);
    }
    ++contrastiveSamples_;

    double dot = 0.0;
    double a2 = 0.0;
    double b2 = 0.0;
    for (size_t i = 0; i < speechConcept.size(); ++i) {
      dot += static_cast<double>(speechConcept[i]) * static_cast<double>(textInSpeechSpace[i]);
      a2 += static_cast<double>(speechConcept[i]) * static_cast<double>(speechConcept[i]);
      b2 += static_cast<double>(textInSpeechSpace[i]) * static_cast<double>(textInSpeechSpace[i]);
    }
    const double denom = std::sqrt(a2 * b2);
    return denom > 1e-8 ? static_cast<float>(1.0 - dot / denom) : 1.0f;
  }

  std::vector<uint8_t> decode(const std::vector<float> &conceptVector,
                              const std::string &mimeType,
                              size_t lengthHint) override {
    if (decoderPath_.empty()) {
      lastError_ = "JPEA_SPEECH_HORIZON_DECODER_MODEL is required for decode";
      return {};
    }
    if (!rdk_x5_bpu::available()) {
      lastError_ = "RDK X5 hbDNN runtime is unavailable";
      return {};
    }
    if (static_cast<int>(conceptVector.size()) != targetDim_) {
      lastError_ = "decode concept dimension does not match targetDim";
      return {};
    }

    std::vector<float> decoderConcept = conceptVector;
    if (static_cast<int>(decoderConcept.size()) != kEncoderOutputDim) {
      decoderConcept = phoenix::multimodal::projectToDimension(decoderConcept, static_cast<size_t>(kEncoderOutputDim), 0x1DEA);
    }

    const std::filesystem::path inputPath = temporaryInputPath();
    {
      std::ofstream output(inputPath, std::ios::binary | std::ios::trunc);
      output.write(reinterpret_cast<const char *>(decoderConcept.data()),
                   static_cast<std::streamsize>(decoderConcept.size() * sizeof(float)));
      if (!output) {
        lastError_ = "unable to write decoder concept tensor";
        return {};
      }
    }

    const auto result = rdk_x5_bpu::execute(nlohmann::json{{"bpuModelPath", decoderPath_},
                                                              {"bpuInputFloatsPath", inputPath.string()},
                                                              {"maxBpuOutputValues", kDecoderOutputSamples}});
    std::error_code ec;
    std::filesystem::remove(inputPath, ec);
    if (!result.value("executed", false)) {
      lastError_ = result.value("error", std::string("hbDNN decoder inference failed"));
      return {};
    }
    const auto outputs = result.value("outputs", nlohmann::json::array());
    if (!outputs.is_array() || outputs.empty() || !outputs[0].contains("values") || !outputs[0]["values"].is_array()) {
      lastError_ = "JPEA Horizon speech decoder must expose a float waveform output";
      return {};
    }
    const auto values = outputs[0]["values"].get<std::vector<float>>();

    const size_t targetLen = lengthHint > 0 ? lengthHint : static_cast<size_t>(cfg_.sampleRate);
    std::vector<float> waveform(targetLen, 0.0f);
    const size_t toCopy = std::min(values.size(), targetLen);
    std::copy(values.begin(), values.begin() + toCopy, waveform.begin());

    std::vector<uint8_t> raw(targetLen);
    for (size_t i = 0; i < targetLen; ++i) {
      float norm = (waveform[i] + 1.0f) * 127.5f;
      raw[i] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, norm)));
    }

    if (mimeType == "audio/wav") {
      auto header = buildWavHeader(static_cast<uint32_t>(targetLen), static_cast<uint32_t>(cfg_.sampleRate));
      std::vector<uint8_t> out;
      out.reserve(header.size() + raw.size());
      out.insert(out.end(), header.begin(), header.end());
      out.insert(out.end(), raw.begin(), raw.end());
      ++samples_;
      lastError_.clear();
      return out;
    }

    ++samples_;
    lastError_.clear();
    return raw;
  }

  nlohmann::json status() const override {
    std::error_code ec;
    bool modelReady = !modelPath_.empty() && std::filesystem::is_regular_file(modelPath_, ec);
    bool decoderReady = !decoderPath_.empty() && std::filesystem::is_regular_file(decoderPath_, ec);
    bool bpuReady = rdk_x5_bpu::available();
    return nlohmann::json{{"id", cfg_.id}, {"arch", cfg_.arch}, {"backend", "horizon-hbdnn"},
                          {"sampleRate", cfg_.sampleRate}, {"windowSamples", cfg_.windowSamples},
                          {"strideSamples", cfg_.strideSamples}, {"targetDim", targetDim_},
                          {"samples", samples_}, {"modelPath", modelPath_},
                          {"decoderPath", decoderPath_},
                          {"ready", modelReady && bpuReady},
                          {"decoderReady", decoderReady && bpuReady},
                          {"error", lastError_}};
  }

  const JpeaV2SpeechWorldModelConfig &config() const override { return cfg_; }

 private:
  std::vector<float> preprocessAudio(const std::vector<uint8_t> &payload,
                                     int sampleRate,
                                     const std::string &mimeType) const {
    size_t offset = 0;
    if (mimeType == "audio/wav" && payload.size() >= 4) {
      if (payload[0] == 'R' && payload[1] == 'I' && payload[2] == 'F' && payload[3] == 'F') {
        offset = 44;
      }
    }
    if (offset > payload.size()) offset = payload.size();

    std::vector<float> samples(payload.size() - offset);
    for (size_t i = 0; i < samples.size(); ++i) {
      samples[i] = (static_cast<float>(payload[offset + i]) / 127.5f) - 1.0f;
    }

    if (sampleRate > 0 && sampleRate != cfg_.sampleRate && !samples.empty()) {
      const float ratio = static_cast<float>(cfg_.sampleRate) / static_cast<float>(sampleRate);
      const size_t outLen = static_cast<size_t>(static_cast<float>(samples.size()) * ratio + 0.5f);
      if (outLen > 0) {
        std::vector<float> resampled(outLen);
        for (size_t i = 0; i < outLen; ++i) {
          const float src = static_cast<float>(i) / ratio;
          const size_t lo = static_cast<size_t>(std::floor(src));
          const size_t hi = std::min(lo + 1, samples.size() - 1);
          const float frac = src - static_cast<float>(lo);
          resampled[i] = samples[lo] * (1.0f - frac) + samples[hi] * frac;
        }
        samples = std::move(resampled);
      }
    }
    return samples;
  }

  int countWindows(int sampleCount) const {
    if (sampleCount < cfg_.windowSamples) return 0;
    return 1 + (sampleCount - cfg_.windowSamples) / cfg_.strideSamples;
  }

  std::vector<float> prepareInput(const std::vector<float> &samples,
                                  const std::vector<bool> &mask) const {
    std::vector<float> masked = samples;
    const int windows = countWindows(static_cast<int>(masked.size()));
    if (!mask.empty()) {
      for (int w = 0; w < windows; ++w) {
        if (w < static_cast<int>(mask.size()) && mask[static_cast<size_t>(w)]) continue;
        const int start = w * cfg_.strideSamples;
        for (int k = 0; k < cfg_.windowSamples; ++k) {
          const size_t idx = static_cast<size_t>(start + k);
          if (idx < masked.size()) masked[idx] = 0.0f;
        }
      }
    }

    std::vector<float> input(static_cast<size_t>(kFixedInputSamples), 0.0f);
    const size_t toCopy = std::min(masked.size(), static_cast<size_t>(kFixedInputSamples));
    std::copy(masked.begin(), masked.begin() + toCopy, input.begin());
    return input;
  }

  std::vector<float> runEncoderBpu(const std::vector<float> &input) {
    if (modelPath_.empty()) {
      lastError_ = "JPEA_SPEECH_HORIZON_MODEL is required";
      return {};
    }
    if (!rdk_x5_bpu::available()) {
      lastError_ = "RDK X5 hbDNN runtime is unavailable";
      return {};
    }
    if (static_cast<int>(input.size()) != kFixedInputSamples) {
      lastError_ = "BPU speech input must contain 16000 samples";
      return {};
    }

    const std::filesystem::path inputPath = temporaryInputPath();
    {
      std::ofstream output(inputPath, std::ios::binary | std::ios::trunc);
      output.write(reinterpret_cast<const char *>(input.data()),
                   static_cast<std::streamsize>(input.size() * sizeof(float)));
      if (!output) {
        lastError_ = "unable to write BPU speech input tensor";
        return {};
      }
    }

    const auto result = rdk_x5_bpu::execute(nlohmann::json{{"bpuModelPath", modelPath_},
                                                              {"bpuInputFloatsPath", inputPath.string()},
                                                              {"maxBpuOutputValues", kEncoderOutputDim}});
    std::error_code ec;
    std::filesystem::remove(inputPath, ec);
    if (!result.value("executed", false)) {
      lastError_ = result.value("error", std::string("hbDNN inference failed"));
      return {};
    }

    const auto outputs = result.value("outputs", nlohmann::json::array());
    if (!outputs.is_array() || outputs.empty() || !outputs[0].contains("values") || !outputs[0]["values"].is_array()) {
      lastError_ = "JPEA Horizon speech model must expose a float embedding output";
      return {};
    }

    auto values = outputs[0]["values"].get<std::vector<float>>();
    if (static_cast<int>(values.size()) != kEncoderOutputDim) {
      lastError_ = "JPEA speech embedding output dimension is not 128";
      return {};
    }

    if (targetDim_ != kEncoderOutputDim) {
      values = phoenix::multimodal::projectToDimension(values, static_cast<size_t>(targetDim_), 0x1DEA);
    }
    if (textAlignment_.size() == values.size()) {
      for (size_t i = 0; i < values.size(); ++i) values[i] += textAlignment_[i];
    }
    return phoenix::multimodal::normalizeVector(values);
  }

  static std::vector<uint8_t> buildWavHeader(uint32_t dataSize, uint32_t sampleRate,
                                             uint16_t numChannels = 1, uint16_t bitsPerSample = 8) {
    std::vector<uint8_t> header(44, 0);
    const uint16_t audioFormat = 1;  // PCM
    const uint32_t byteRate = sampleRate * numChannels * bitsPerSample / 8;
    const uint16_t blockAlign = numChannels * bitsPerSample / 8;

    auto write4 = [&](size_t off, uint32_t v) {
      header[off] = static_cast<uint8_t>(v & 0xff);
      header[off + 1] = static_cast<uint8_t>((v >> 8) & 0xff);
      header[off + 2] = static_cast<uint8_t>((v >> 16) & 0xff);
      header[off + 3] = static_cast<uint8_t>((v >> 24) & 0xff);
    };
    auto write2 = [&](size_t off, uint16_t v) {
      header[off] = static_cast<uint8_t>(v & 0xff);
      header[off + 1] = static_cast<uint8_t>((v >> 8) & 0xff);
    };

    header[0] = 'R'; header[1] = 'I'; header[2] = 'F'; header[3] = 'F';
    write4(4, 36 + dataSize);
    header[8] = 'W'; header[9] = 'A'; header[10] = 'V'; header[11] = 'E';
    header[12] = 'f'; header[13] = 'm'; header[14] = 't'; header[15] = ' ';
    write4(16, 16);
    write2(20, audioFormat);
    write2(22, numChannels);
    write4(24, sampleRate);
    write4(28, byteRate);
    write2(32, blockAlign);
    write2(34, bitsPerSample);
    header[36] = 'd'; header[37] = 'a'; header[38] = 't'; header[39] = 'a';
    write4(40, dataSize);
    return header;
  }

  JpeaV2SpeechWorldModelConfig cfg_;
  int targetDim_;
  std::string modelPath_;
  std::string decoderPath_;
  std::vector<float> predictorWeights_;
  std::vector<float> predictorBias_;
  std::vector<float> textAlignment_;
  size_t samples_ = 0;
  size_t contrastiveSamples_ = 0;
  size_t adaptSteps_ = 0;
  std::string lastError_;
};

/* Deterministic fallback that implements the 1D JPEA-v2 speech interface.
   It is not the real model; it is here so the semantic contract can be
   exercised before a compiled backend is wired in. */
class JpeaV2SpeechFallbackModel : public JpeaV2SpeechWorldModel {
 public:
  JpeaV2SpeechFallbackModel(JpeaV2SpeechWorldModelConfig cfg, int targetDim)
      : cfg_(std::move(cfg)),
        targetDim_(targetDim > 0 ? targetDim : 128),
        textAlignment_(static_cast<size_t>(targetDim_), 0.0f),
        predictorWeights_(static_cast<size_t>(targetDim_ * targetDim_), 0.0f),
        predictorBias_(static_cast<size_t>(targetDim_), 0.0f) {
    for (int i = 0; i < targetDim_; ++i) {
      predictorWeights_[static_cast<size_t>(i * targetDim_ + i)] = 1.0f;
    }
  }

  std::vector<float> encode(const std::vector<uint8_t> &audioBytes,
                            int sampleRate,
                            const std::string &mimeType) override {
    auto samples = preprocess(audioBytes, sampleRate, mimeType);
    return windowStatsToConcept(samples, {});
  }

  std::vector<float> encodeContext(const std::vector<uint8_t> &audioBytes,
                                   int sampleRate,
                                   const std::string &mimeType,
                                   const std::vector<bool> &mask) override {
    auto samples = preprocess(audioBytes, sampleRate, mimeType);
    return windowStatsToConcept(samples, mask);
  }

  std::vector<float> encodeTarget(const std::vector<uint8_t> &audioBytes,
                                  int sampleRate,
                                  const std::string &mimeType,
                                  const std::vector<int> &windowIndices) override {
    auto samples = preprocess(audioBytes, sampleRate, mimeType);
    const int windows = static_cast<int>(countWindows(static_cast<int>(samples.size())));
    std::vector<bool> mask;
    if (!windowIndices.empty()) {
      mask.assign(windows, false);
      for (int idx : windowIndices) {
        if (idx >= 0 && idx < windows) mask[idx] = true;
      }
    }
    return windowStatsToConcept(samples, mask);
  }

  std::vector<float> predictTarget(const std::vector<float> &contextRepr,
                                   const std::vector<int> & /*targetPos*/) override {
    if (contextRepr.size() != static_cast<size_t>(targetDim_)) {
      lastError_ = "predictor input dimension mismatch";
      return {};
    }
    std::vector<float> out(static_cast<size_t>(targetDim_), 0.0f);
    for (int i = 0; i < targetDim_; ++i) {
      float v = predictorBias_[static_cast<size_t>(i)];
      for (int j = 0; j < targetDim_; ++j) {
        v += predictorWeights_[static_cast<size_t>(i * targetDim_ + j)] * contextRepr[static_cast<size_t>(j)];
      }
      out[static_cast<size_t>(i)] = v;
    }
    return out;
  }

  float adapt(const std::vector<uint8_t> &audioBytes,
              int sampleRate,
              const std::string &mimeType,
              int steps,
              float lr) override {
    if (audioBytes.empty()) {
      lastError_ = "audio bytes are empty";
      return -1.0f;
    }
    if (steps < 1) steps = 1;
    if (lr < 0.0f) lr = 0.0f;

    auto samples = preprocess(audioBytes, sampleRate, mimeType);
    const int totalWindows = countWindows(static_cast<int>(samples.size()));
    if (totalWindows < 2) {
      lastError_ = "not enough windows for self-supervised adaptation";
      return -1.0f;
    }

    float totalLoss = 0.0f;
    std::mt19937 rng(0x1DEA);
    for (int step = 0; step < steps; ++step) {
      std::vector<int> windowOrder(totalWindows);
      std::iota(windowOrder.begin(), windowOrder.end(), 0);
      std::shuffle(windowOrder.begin(), windowOrder.end(), rng);

      const int targetCount = std::max(1, totalWindows / 3);
      std::vector<int> targetIndices(windowOrder.begin(), windowOrder.begin() + targetCount);
      std::vector<bool> contextMask(static_cast<size_t>(totalWindows), true);
      for (int t : targetIndices) contextMask[static_cast<size_t>(t)] = false;

      auto context = encodeContext(audioBytes, sampleRate, mimeType, contextMask);
      auto target = encodeTarget(audioBytes, sampleRate, mimeType, targetIndices);
      if (context.size() != static_cast<size_t>(targetDim_) ||
          target.size() != static_cast<size_t>(targetDim_)) {
        lastError_ = "context/target dimension mismatch";
        return -1.0f;
      }

      auto pred = predictTarget(context, targetIndices);
      if (pred.size() != target.size()) {
        lastError_ = "predictor output dimension mismatch";
        return -1.0f;
      }

      double loss = 0.0;
      std::vector<float> error(pred.size());
      for (size_t i = 0; i < pred.size(); ++i) {
        error[i] = pred[i] - target[i];
        loss += static_cast<double>(error[i]) * static_cast<double>(error[i]);
      }
      loss /= static_cast<double>(pred.size());
      totalLoss += static_cast<float>(loss);

      const float stepSize = lr;
      for (int i = 0; i < targetDim_; ++i) {
        for (int j = 0; j < targetDim_; ++j) {
          predictorWeights_[static_cast<size_t>(i * targetDim_ + j)] -=
              stepSize * error[static_cast<size_t>(i)] * context[static_cast<size_t>(j)];
        }
        predictorBias_[static_cast<size_t>(i)] -= stepSize * error[static_cast<size_t>(i)];
      }
    }

    ++samples_;
    lastError_.clear();
    return totalLoss / static_cast<float>(steps);
  }

  float contrastiveAdapt(const std::vector<uint8_t> &audioBytes,
                         int sampleRate,
                         const std::string &mimeType,
                         const std::vector<float> &textConcept,
                         float /*temperature*/) override {
    if (audioBytes.empty() || textConcept.empty()) return -1.0f;
    auto speechConcept = encode(audioBytes, sampleRate, mimeType);
    if (speechConcept.empty()) return -1.0f;

    auto textInSpeechSpace = phoenix::multimodal::projectToDimension(textConcept, static_cast<size_t>(targetDim_), 0x53505458U);
    if (textInSpeechSpace.size() != speechConcept.size()) return -1.0f;

    const float n = static_cast<float>(contrastiveSamples_);
    for (size_t i = 0; i < textAlignment_.size(); ++i) {
      const float diff = textInSpeechSpace[i] - speechConcept[i];
      textAlignment_[i] = (textAlignment_[i] * n + diff) / (n + 1.0f);
    }
    ++contrastiveSamples_;

    double dot = 0.0;
    double a2 = 0.0;
    double b2 = 0.0;
    for (size_t i = 0; i < speechConcept.size(); ++i) {
      dot += static_cast<double>(speechConcept[i]) * static_cast<double>(textInSpeechSpace[i]);
      a2 += static_cast<double>(speechConcept[i]) * static_cast<double>(speechConcept[i]);
      b2 += static_cast<double>(textInSpeechSpace[i]) * static_cast<double>(textInSpeechSpace[i]);
    }
    const double denom = std::sqrt(a2 * b2);
    return denom > 1e-8 ? static_cast<float>(1.0 - dot / denom) : 1.0f;
  }

  std::vector<uint8_t> decode(const std::vector<float> &conceptVector,
                              const std::string & /*mimeType*/,
                              size_t lengthHint) override {
    const size_t len = lengthHint > 0 ? lengthHint : static_cast<size_t>(cfg_.sampleRate);
    std::vector<float> waveform(len, 0.0f);
    for (size_t t = 0; t < len; ++t) {
      float v = 0.0f;
      for (size_t i = 0; i < conceptVector.size(); ++i) {
        constexpr float kPi = 3.14159265358979323846f;
        const float freq = static_cast<float>(i + 1) * 2.0f * kPi / static_cast<float>(len);
        v += conceptVector[i] * std::sin(freq * static_cast<float>(t));
      }
      waveform[t] = v;
    }
    float lo = *std::min_element(waveform.begin(), waveform.end());
    float hi = *std::max_element(waveform.begin(), waveform.end());
    float range = hi - lo;
    std::vector<uint8_t> raw(len);
    for (size_t i = 0; i < len; ++i) {
      float norm = (range > 1e-8f) ? (waveform[i] - lo) / range : 0.5f;
      raw[i] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, norm * 255.0f)));
    }
    return raw;
  }

  nlohmann::json status() const override {
    nlohmann::json j;
    j["id"] = cfg_.id;
    j["arch"] = cfg_.arch;
    j["backend"] = "fallback";
    j["sampleRate"] = cfg_.sampleRate;
    j["windowSamples"] = cfg_.windowSamples;
    j["strideSamples"] = cfg_.strideSamples;
    j["targetDim"] = targetDim_;
    j["samples"] = samples_;
    j["contrastiveSamples"] = contrastiveSamples_;
    j["ready"] = true;
    j["error"] = lastError_;
    j["note"] = "deterministic CPU fallback for environments without BPU runtime";
    return j;
  }

  const JpeaV2SpeechWorldModelConfig &config() const override { return cfg_; }

 private:
  JpeaV2SpeechWorldModelConfig cfg_;
  int targetDim_;
  size_t samples_ = 0;
  size_t contrastiveSamples_ = 0;
  std::vector<float> textAlignment_;
  std::vector<float> predictorWeights_;
  std::vector<float> predictorBias_;
  std::string lastError_;

  int countWindows(int sampleCount) const {
    if (sampleCount < cfg_.windowSamples) return 0;
    return 1 + (sampleCount - cfg_.windowSamples) / cfg_.strideSamples;
  }

  std::vector<float> preprocess(const std::vector<uint8_t> &payload,
                                int sampleRate,
                                const std::string & /*mimeType*/) const {
    std::vector<float> samples(payload.size());
    for (size_t i = 0; i < payload.size(); ++i) {
      samples[i] = (static_cast<float>(payload[i]) / 127.5f) - 1.0f;
    }
    if (sampleRate > 0 && sampleRate != cfg_.sampleRate && !samples.empty()) {
      const float ratio = static_cast<float>(cfg_.sampleRate) / static_cast<float>(sampleRate);
      const size_t outLen = static_cast<size_t>(static_cast<float>(samples.size()) * ratio + 0.5f);
      if (outLen > 0) {
        std::vector<float> resampled(outLen);
        for (size_t i = 0; i < outLen; ++i) {
          const float src = static_cast<float>(i) / ratio;
          const size_t lo = static_cast<size_t>(std::floor(src));
          const size_t hi = std::min(lo + 1, samples.size() - 1);
          const float frac = src - static_cast<float>(lo);
          resampled[i] = samples[lo] * (1.0f - frac) + samples[hi] * frac;
        }
        samples = std::move(resampled);
      }
    }
    return samples;
  }

  std::vector<float> windowStatsToConcept(const std::vector<float> &samples,
                                          const std::vector<bool> &mask) const {
    using namespace phoenix::multimodal;
    const int windows = countWindows(static_cast<int>(samples.size()));
    std::vector<float> stats;
    stats.reserve(2 * std::max(0, windows));
    for (int w = 0; w < windows; ++w) {
      if (!mask.empty() && !mask[w]) continue;
      float mean = 0.0f;
      float sq = 0.0f;
      int count = 0;
      const int start = w * cfg_.strideSamples;
      for (int k = 0; k < cfg_.windowSamples; ++k) {
        const size_t idx = static_cast<size_t>(start + k);
        if (idx >= samples.size()) continue;
        float v = samples[idx];
        mean += v;
        sq += v * v;
        ++count;
      }
      if (count == 0) continue;
      mean /= static_cast<float>(count);
      float var = (sq / static_cast<float>(count)) - (mean * mean);
      stats.push_back(mean);
      stats.push_back(std::sqrt(std::max(0.0f, var)));
    }
    if (stats.empty()) {
      stats.assign(2, 0.0f);
    }
    auto projected = projectToDimension(stats, static_cast<size_t>(targetDim_), 0x1DEA);
    if (textAlignment_.size() == projected.size()) {
      for (size_t i = 0; i < projected.size(); ++i) projected[i] += textAlignment_[i];
    }
    return normalizeVector(projected);
  }
};

class JpeaV2SpeechUnavailableModel : public JpeaV2SpeechWorldModel {
 public:
  JpeaV2SpeechUnavailableModel(JpeaV2SpeechWorldModelConfig cfg, int targetDim)
      : cfg_(std::move(cfg)), targetDim_(targetDim > 0 ? targetDim : 128) {}

  std::vector<float> encode(const std::vector<uint8_t> &, int, const std::string &) override { return {}; }
  std::vector<float> encodeContext(const std::vector<uint8_t> &, int, const std::string &, const std::vector<bool> &) override { return {}; }
  std::vector<float> encodeTarget(const std::vector<uint8_t> &, int, const std::string &, const std::vector<int> &) override { return {}; }
  std::vector<float> predictTarget(const std::vector<float> &, const std::vector<int> &) override { return {}; }
  float adapt(const std::vector<uint8_t> &, int, const std::string &, int, float) override { return -1.0f; }
  float contrastiveAdapt(const std::vector<uint8_t> &, int, const std::string &, const std::vector<float> &, float) override { return -1.0f; }
  std::vector<uint8_t> decode(const std::vector<float> &, const std::string &, size_t) override { return {}; }
  nlohmann::json status() const override {
    return nlohmann::json{{"id", cfg_.id}, {"arch", cfg_.arch}, {"backend", "unavailable"},
                          {"targetDim", targetDim_}, {"ready", false},
                          {"error", "no compiled RDK X5 speech JPEA model is configured"}};
  }
  const JpeaV2SpeechWorldModelConfig &config() const override { return cfg_; }

 private:
  JpeaV2SpeechWorldModelConfig cfg_;
  int targetDim_;
};

/**
 * @brief Remote implementation of the 1-D speech world model.
 *
 * Forwards encode/decode requests to a remote HTTP/JSON endpoint.  The encode
 * call expects a float embedding of size @c targetDim_; the decode call expects
 * a base64-encoded payload in return.  Local adaptation and contrastive
 * learning are not supported by a remote inference-only model.
 */
class JpeaV2SpeechRemoteModel : public JpeaV2SpeechWorldModel {
 public:
  JpeaV2SpeechRemoteModel(JpeaV2SpeechWorldModelConfig cfg, int targetDim,
                          const phoenix::deployment::RemoteEndpoint &endpoint)
      : cfg_(std::move(cfg)), targetDim_(targetDim > 0 ? targetDim : 128),
        endpoint_(endpoint) {}

  std::vector<float> encode(const std::vector<uint8_t> &audioBytes,
                            int sampleRate,
                            const std::string &mimeType) override {
    nlohmann::json payload;
    payload["modality"] = "audio";
    payload["mimeType"] = mimeType;
    payload["sampleRate"] = sampleRate;
    payload["payloadBase64"] = phoenix::deployment::RemoteModelClient::base64Encode(audioBytes);
    payload["conceptDim"] = targetDim_;

    auto result = phoenix::deployment::RemoteModelClient::call(endpoint_, payload);
    if (!result.value("ok", false)) {
      lastError_ = result.value("error", std::string("remote speech encode failed"));
      return {};
    }
    auto embedding = parseFloatVectorFromJson(result.value("embedding", nlohmann::json::array()));
    if (static_cast<int>(embedding.size()) != targetDim_) {
      lastError_ = "remote speech embedding dimension mismatch";
      return {};
    }
    ++samples_;
    lastError_.clear();
    return embedding;
  }

  std::vector<float> encodeContext(const std::vector<uint8_t> &audioBytes,
                                   int sampleRate,
                                   const std::string &mimeType,
                                   const std::vector<bool> &) override {
    return encode(audioBytes, sampleRate, mimeType);
  }

  std::vector<float> encodeTarget(const std::vector<uint8_t> &audioBytes,
                                  int sampleRate,
                                  const std::string &mimeType,
                                  const std::vector<int> &) override {
    // Remote endpoints are inference-only; return the full utterance embedding.
    return encode(audioBytes, sampleRate, mimeType);
  }

  std::vector<float> predictTarget(const std::vector<float> &contextRepr,
                                   const std::vector<int> &) override {
    // No remote predictor exposed; identity is the safest deterministic default.
    return std::vector<float>(contextRepr.begin(), contextRepr.end());
  }

  float adapt(const std::vector<uint8_t> &, int, const std::string &, int, float) override {
    lastError_ = "remote runtime adaptation is not supported";
    return -1.0f;
  }

  float contrastiveAdapt(const std::vector<uint8_t> &audioBytes,
                         int sampleRate,
                         const std::string &mimeType,
                         const std::vector<float> &textConcept,
                         float /*temperature*/) override {
    if (audioBytes.empty() || textConcept.empty()) {
      lastError_ = "remote contrastive adaptation requires audio and text concept";
      return -1.0f;
    }
    auto speechConcept = encode(audioBytes, sampleRate, mimeType);
    if (speechConcept.empty()) return -1.0f;

    auto textInSpeechSpace = phoenix::multimodal::projectToDimension(textConcept, static_cast<size_t>(targetDim_), 0x53505458U);
    if (textInSpeechSpace.size() != speechConcept.size()) {
      lastError_ = "remote contrastive adaptation dimension mismatch";
      return -1.0f;
    }

    double dot = 0.0;
    double a2 = 0.0;
    double b2 = 0.0;
    for (size_t i = 0; i < speechConcept.size(); ++i) {
      dot += static_cast<double>(speechConcept[i]) * static_cast<double>(textInSpeechSpace[i]);
      a2 += static_cast<double>(speechConcept[i]) * static_cast<double>(speechConcept[i]);
      b2 += static_cast<double>(textInSpeechSpace[i]) * static_cast<double>(textInSpeechSpace[i]);
    }
    const double denom = std::sqrt(a2 * b2);
    if (denom < 1e-8) {
      lastError_ = "remote contrastive adaptation received a zero vector";
      return -1.0f;
    }
    lastError_.clear();
    return static_cast<float>(1.0 - dot / denom);
  }

  std::vector<uint8_t> decode(const std::vector<float> &conceptVector,
                              const std::string &mimeType,
                              size_t lengthHint) override {
    nlohmann::json payload;
    payload["modality"] = "audio";
    payload["decode"] = true;
    payload["mimeType"] = mimeType;
    payload["lengthHint"] = lengthHint;
    payload["conceptVector"] = conceptVector;

    auto result = phoenix::deployment::RemoteModelClient::call(endpoint_, payload);
    if (!result.value("ok", false)) {
      lastError_ = result.value("error", std::string("remote speech decode failed"));
      return {};
    }
    auto encoded = result.value("payloadBase64", std::string());
    if (encoded.empty()) {
      lastError_ = "remote speech decode returned empty payload";
      return {};
    }
    auto decoded = phoenix::deployment::RemoteModelClient::base64Decode(encoded);
    if (decoded.empty() && !encoded.empty()) {
      lastError_ = "remote speech decode returned invalid base64";
      return {};
    }
    return decoded;
  }

  nlohmann::json status() const override {
    return nlohmann::json{{"id", cfg_.id}, {"arch", cfg_.arch}, {"backend", "remote"},
                          {"url", endpoint_.url}, {"method", endpoint_.method},
                          {"targetDim", targetDim_}, {"samples", samples_},
                          {"ready", !endpoint_.url.empty()},
                          {"error", lastError_}};
  }

  const JpeaV2SpeechWorldModelConfig &config() const override { return cfg_; }

 private:
  static std::vector<float> parseFloatVectorFromJson(const nlohmann::json &j) {
    std::vector<float> out;
    if (!j.is_array()) return out;
    for (const auto &v : j) {
      if (v.is_number()) out.push_back(v.get<double>());
    }
    return out;
  }

  JpeaV2SpeechWorldModelConfig cfg_;
  int targetDim_;
  phoenix::deployment::RemoteEndpoint endpoint_;
  size_t samples_ = 0;
  std::string lastError_;
};

}  // namespace

/**
 * @brief Factory that selects a local HBDNN, remote, or fallback speech world model.
 *
 * Preference order: remote edge endpoint (when configured), local HBDNN BPU
 * model (when the runtime and compiled model are both available), and finally
 * the deterministic fallback.  The fallback is also used for ServerClient
 * placement or disabled edge builds.
 */
std::unique_ptr<JpeaV2SpeechWorldModel> createJpeaV2SpeechWorldModel(
    const std::string &variantId, int targetDim, const std::string & /*backend*/) {
  const auto *v = findJpeaV2SpeechVariant(variantId);
  if (!v) {
    v = findJpeaV2SpeechVariant("jpea_v2_speech_16k");
  }

  const auto &deployment = phoenix::deployment::ModelDeploymentConfig::instance().speech();

  if (deployment.placement == phoenix::deployment::ModelPlacement::ServerClient) {
    return std::make_unique<JpeaV2SpeechFallbackModel>(*v, targetDim);
  }

  const bool useRemote =
      (deployment.placement == phoenix::deployment::ModelPlacement::Remote ||
       deployment.placement == phoenix::deployment::ModelPlacement::Auto) &&
      !deployment.remote.url.empty();
  if (useRemote) {
    return std::make_unique<JpeaV2SpeechRemoteModel>(*v, targetDim, deployment.remote);
  }

#if PHOENIX_EDGE_SPEECH_ENABLED
  // Prefer local HBDNN only when the compiled model and BPU runtime are both
  // available.  On non-X5 builds this check fails gracefully and we fall back
  // to the deterministic CPU implementation, preserving a working audio<->concept
  // bridge everywhere.
  auto hbdnn = std::make_unique<JpeaV2SpeechHbdnnModel>(*v, targetDim);
  if (hbdnn->status().value("ready", false)) {
    return hbdnn;
  }
#endif

  (void)deployment;
  return std::make_unique<JpeaV2SpeechFallbackModel>(*v, targetDim);
}

}  // namespace io
}  // namespace phoenix
