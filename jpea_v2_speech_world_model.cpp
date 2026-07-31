/* jpea_v2_speech_world_model.cpp - Factory and fallback for 1D speech world model interface
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version. */

#include "jpea_v2_speech_world_model.hpp"
#include "model_deployment.hpp"
#include "semantic_unit.hpp"

#ifndef PHOENIX_EDGE_SPEECH_ENABLED
#define PHOENIX_EDGE_SPEECH_ENABLED 1
#endif

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace phoenix {
namespace io {

namespace {

/* Deterministic fallback that implements the 1D JPEA-v2 speech interface.
   It is not the real model; it is here so the semantic contract can be
   exercised before a PyTorch/HuggingFace backend is wired in. */
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
    j["note"] = "fallback implementation; replace with PyTorch/HuggingFace backend for real 1D JPEA";
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

}  // namespace

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

/**
 * @brief Factory that selects a local fallback or remote speech world model.
 *
 * Uses ModelDeploymentConfig to decide whether to send audio to a remote edge
 * endpoint.  For local execution it returns the deterministic fallback, which
 * always produces a concept vector and keeps the multimodal bridge / speech
 * pretraining functional when no compiled 1D JPEA model is available.
 */
std::unique_ptr<JpeaV2SpeechWorldModel> createJpeaV2SpeechWorldModel(
    const std::string &variantId, int targetDim, const std::string & /*backend*/) {
  const auto *v = findJpeaV2SpeechVariant(variantId);
  if (!v) {
    v = findJpeaV2SpeechVariant("jpea_v2_speech_16k");
  }

  const auto &deployment = phoenix::deployment::ModelDeploymentConfig::instance().speech();

#if PHOENIX_EDGE_SPEECH_ENABLED
  const bool useRemote =
      (deployment.placement == phoenix::deployment::ModelPlacement::Remote ||
       deployment.placement == phoenix::deployment::ModelPlacement::Auto) &&
      !deployment.remote.url.empty();
  if (useRemote) {
    return std::make_unique<JpeaV2SpeechRemoteModel>(*v, targetDim, deployment.remote);
  }
#else
  (void)deployment;
#endif
  return std::make_unique<JpeaV2SpeechFallbackModel>(*v, targetDim);
}

}  // namespace io
}  // namespace phoenix
