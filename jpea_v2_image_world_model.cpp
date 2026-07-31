/* jpea_v2_image_world_model.cpp - Factory and fallback for image world model interface
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version. */

#include "jpea_v2_image_world_model.hpp"
#include "model_deployment.hpp"
#include "rdk_x5_bpu.hpp"
#include "semantic_unit.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>

#if __has_include(<opencv2/opencv.hpp>)
#define JPEA_HAVE_OPENCV 1
#include <opencv2/opencv.hpp>
#else
#define JPEA_HAVE_OPENCV 0
#endif

#ifndef PHOENIX_EDGE_IMAGE_ENABLED
#define PHOENIX_EDGE_IMAGE_ENABLED 1
#endif

namespace phoenix {
namespace io {

namespace {

std::string resolveBpuModelPath(const std::string &envOverride,
                                const std::string &modelKind,
                                const std::string &variantId) {
  if (!envOverride.empty()) return envOverride;

  const std::vector<std::string> names = [modelKind]() {
    if (modelKind == "encoder") return std::vector<std::string>{"model.bin", "encoder.bin", "ijepa.bin"};
    if (modelKind == "decoder") return std::vector<std::string>{"decoder.bin", "decode.bin"};
    return std::vector<std::string>{"model.bin"};
  }();

  const std::vector<std::string> roots = {
      std::string("runtime_store/models/ijepa/") + variantId,
      std::string("runtime_store/models/ijepa/"),
  };

  std::error_code ec;
  for (const auto &root : roots) {
    for (const auto &name : names) {
      std::filesystem::path p = std::filesystem::path(root) / name;
      if (std::filesystem::is_regular_file(p, ec)) return p.string();
    }
  }
  return {};
}

/**
 * @brief Local HBDNN-backed image world model.
 *
 * Uses OpenCV to decode and preprocess images and dispatches them to the RDK
 * X5 hbDNN runtime.  If the model or runtime is unavailable, encode() returns
 * an empty vector so the multimodal concept bridge can fall back to a
 * deterministic media concept.
 */
class JpeaV2ImageHbdnnModel : public JpeaV2ImageWorldModel {
 public:
  JpeaV2ImageHbdnnModel(JpeaV2ImageWorldModelConfig cfg, int targetDim)
      : cfg_(std::move(cfg)), targetDim_(targetDim > 0 ? targetDim : 128),
        modelPath_(resolveBpuModelPath(environment("JPEA_IMAGE_HORIZON_MODEL"), "encoder", cfg_.id)),
        decoderPath_(resolveBpuModelPath(environment("JPEA_IMAGE_HORIZON_DECODER_MODEL"), "decoder", cfg_.id)),
        color_(environment("JPEA_IMAGE_INPUT_COLOR", "bgr")),
        mean_(parseFloatVector(environment("JPEA_IMAGE_PIXEL_MEAN", "0.485 0.456 0.406"), 3)),
        std_(parseFloatVector(environment("JPEA_IMAGE_PIXEL_STD", "0.229 0.224 0.225"), 3)) {}

  std::vector<float> encode(const std::vector<uint8_t> &imageBytes, int width, int height,
                            const std::string &mimeType) override {
#if !JPEA_HAVE_OPENCV
    lastError_ = "OpenCV image decoding is unavailable in this build";
    return {};
#else
    if (modelPath_.empty()) {
      lastError_ = "JPEA_IMAGE_HORIZON_MODEL is required";
      return {};
    }
    if (!rdk_x5_bpu::available()) {
      lastError_ = "RDK X5 hbDNN runtime is unavailable";
      return {};
    }
    cv::Mat image;
    if (mimeType == "application/x-bgr") {
      if (width <= 0 || height <= 0 || imageBytes.size() != static_cast<size_t>(width) * static_cast<size_t>(height) * 3U) {
        lastError_ = "direct BGR frame does not match its declared dimensions";
        return {};
      }
      image = cv::Mat(height, width, CV_8UC3, const_cast<uint8_t *>(imageBytes.data())).clone();
    } else {
      std::vector<uchar> compressed(imageBytes.begin(), imageBytes.end());
      image = cv::imdecode(compressed, cv::IMREAD_COLOR);
      if (image.empty()) {
        lastError_ = "unable to decode image payload";
        return {};
      }
    }
    cv::resize(image, image, cv::Size(cfg_.resolution, cfg_.resolution), 0, 0, cv::INTER_AREA);
    if (color_ == "rgb") cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    if (color_ != "bgr" && color_ != "rgb") {
      lastError_ = "JPEA_IMAGE_INPUT_COLOR must be bgr or rgb";
      return {};
    }
    // Convert to float NCHW and normalize (matches the ONNX featuremap input).
    image.convertTo(image, CV_32FC3, 1.0 / 255.0);
    cv::subtract(image, cv::Scalar(mean_[2], mean_[1], mean_[0]), image);  // BGR order
    cv::divide(image, cv::Scalar(std_[2], std_[1], std_[0]), image);
    std::vector<cv::Mat> channels;
    cv::split(image, channels);
    std::vector<float> nchw(channels[0].total() * 3);
    for (int c = 0; c < 3; ++c) {
      const cv::Mat &ch = channels[c];
      std::memcpy(nchw.data() + c * ch.total(), ch.ptr<float>(), ch.total() * sizeof(float));
    }
    const std::filesystem::path inputPath = temporaryInputPath();
    {
      std::ofstream output(inputPath, std::ios::binary | std::ios::trunc);
      output.write(reinterpret_cast<const char *>(nchw.data()),
                   static_cast<std::streamsize>(nchw.size() * sizeof(float)));
      if (!output) {
        lastError_ = "unable to write BPU input tensor";
        return {};
      }
    }
    const auto result = rdk_x5_bpu::execute(nlohmann::json{{"bpuModelPath", modelPath_},
                                                              {"bpuInputFloatsPath", inputPath.string()},
                                                              {"maxBpuOutputValues", targetDim_}});
    std::error_code ec;
    std::filesystem::remove(inputPath, ec);
    if (!result.value("executed", false)) {
      lastError_ = result.value("error", std::string("hbDNN inference failed"));
      return {};
    }
    const auto outputs = result.value("outputs", nlohmann::json::array());
    if (!outputs.is_array() || outputs.empty() || !outputs[0].contains("values") || !outputs[0]["values"].is_array()) {
      lastError_ = "JPEA Horizon model must expose a float embedding output";
      return {};
    }
    const auto embedding = outputs[0]["values"].get<std::vector<float>>();
    if (static_cast<int>(embedding.size()) != targetDim_) {
      lastError_ = "JPEA embedding output dimension does not match JPEA_IMAGE_CONCEPT_DIM";
      return {};
    }
    ++samples_;
    lastError_.clear();
    return embedding;
#endif
  }

  std::vector<float> encodeContext(const std::vector<uint8_t> &imageBytes, int width, int height,
                                   const std::string &mimeType, const std::vector<bool> &) override {
    return encode(imageBytes, width, height, mimeType);
  }

  std::vector<float> encodeTarget(const std::vector<uint8_t> &imageBytes, int width, int height,
                                  const std::string &mimeType,
                                  const std::vector<int> &) override {
    // This model was compiled as a plain encoder, not a masked JPEA target graph.
    // Returning the full-image embedding is a deterministic fallback for callers
    // that expect a target concept.
    return encode(imageBytes, width, height, mimeType);
  }

  std::vector<float> predictTarget(const std::vector<float> &contextRepr,
                                   const std::vector<int> &) override {
    // No learned predictor available; the best deterministic estimate is the
    // context representation itself, especially before any adaptation has run.
    return std::vector<float>(contextRepr.begin(), contextRepr.end());
  }

  float adapt(const std::vector<uint8_t> &, int, int, const std::string &, int, float) override {
    lastError_ = "runtime adaptation is not supported by an immutable Horizon model";
    return -1.0f;
  }

  std::vector<uint8_t> decode(const std::vector<float> &conceptVector,
                              const std::string &mimeType) override {
#if !JPEA_HAVE_OPENCV
    lastError_ = "OpenCV image encoding is unavailable in this build";
    return {};
#else
    if (decoderPath_.empty()) {
      lastError_ = "JPEA_IMAGE_HORIZON_DECODER_MODEL is required for decode";
      return {};
    }
    if (!rdk_x5_bpu::available()) {
      lastError_ = "RDK X5 hbDNN runtime is unavailable";
      return {};
    }
    if (static_cast<int>(conceptVector.size()) != targetDim_) {
      lastError_ = "concept vector dimension does not match targetDim";
      return {};
    }
    const std::filesystem::path inputPath = temporaryInputPath();
    {
      std::ofstream output(inputPath, std::ios::binary | std::ios::trunc);
      output.write(reinterpret_cast<const char *>(conceptVector.data()),
                   static_cast<std::streamsize>(conceptVector.size() * sizeof(float)));
      if (!output) {
        lastError_ = "unable to write decoder concept tensor";
        return {};
      }
    }
    const int outPixels = cfg_.resolution * cfg_.resolution * 3;
    const auto result = rdk_x5_bpu::execute(nlohmann::json{{"bpuModelPath", decoderPath_},
                                                              {"bpuInputFloatsPath", inputPath.string()},
                                                              {"maxBpuOutputValues", outPixels}});
    std::error_code ec;
    std::filesystem::remove(inputPath, ec);
    if (!result.value("executed", false)) {
      lastError_ = result.value("error", std::string("hbDNN decoder inference failed"));
      return {};
    }
    const auto outputs = result.value("outputs", nlohmann::json::array());
    if (!outputs.is_array() || outputs.empty() || !outputs[0].contains("values") || !outputs[0]["values"].is_array()) {
      lastError_ = "JPEA Horizon decoder must expose a float image output";
      return {};
    }
    const auto values = outputs[0]["values"].get<std::vector<float>>();
    if (static_cast<int>(values.size()) != outPixels) {
      lastError_ = "decoder output size does not match resolution";
      return {};
    }
    // values are in NCHW [1,3,H,W]; convert to HWC float image.
    std::vector<float> hwc(outPixels);
    const int H = cfg_.resolution;
    const int W = cfg_.resolution;
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          hwc[(h * W + w) * 3 + c] = values[c * H * W + h * W + w];
        }
      }
    }
    cv::Mat floatImg(H, W, CV_32FC3, hwc.data());
    // Denormalize: (img * std + mean) * 255.
    std::vector<cv::Mat> channels;
    cv::split(floatImg, channels);
    for (int c = 0; c < 3; ++c) {
      channels[c] = (channels[c] * std_[c] + mean_[c]) * 255.0;
    }
    cv::merge(channels, floatImg);
    cv::Mat uint8Img;
    cv::max(cv::min(floatImg, 255.0), 0.0, floatImg);  // clip to [0,255]
    floatImg.convertTo(uint8Img, CV_8UC3);
    if (color_ == "rgb") cv::cvtColor(uint8Img, uint8Img, cv::COLOR_RGB2BGR);
    std::vector<uchar> encoded;
    if (!cv::imencode(mimeType == "image/png" ? ".png" : ".jpg", uint8Img, encoded)) {
      lastError_ = "OpenCV failed to encode decoded image";
      return {};
    }
    ++samples_;
    lastError_.clear();
    return {encoded.begin(), encoded.end()};
#endif
  }

  nlohmann::json status() const override {
    return nlohmann::json{{"id", cfg_.id}, {"arch", cfg_.arch}, {"backend", "horizon-hbdnn"},
                          {"resolution", cfg_.resolution}, {"patchSize", cfg_.patchSize},
                          {"targetDim", targetDim_}, {"samples", samples_}, {"modelPath", modelPath_},
                          {"decoderPath", decoderPath_},
                          {"inputColor", color_}, {"pixelMean", mean_}, {"pixelStd", std_},
                          {"ready", !modelPath_.empty() && std::filesystem::is_regular_file(modelPath_) && rdk_x5_bpu::available()},
                          {"decoderReady", !decoderPath_.empty() && std::filesystem::is_regular_file(decoderPath_) && rdk_x5_bpu::available()},
                          {"error", lastError_}};
  }

  const JpeaV2ImageWorldModelConfig &config() const override { return cfg_; }

 private:
  static std::vector<float> parseFloatVectorFromJson(const nlohmann::json &j) {
    std::vector<float> out;
    if (!j.is_array()) return out;
    for (const auto &v : j) {
      if (v.is_number()) out.push_back(v.get<double>());
    }
    return out;
  }

  static std::string environment(const char *name, const std::string &defaultValue = {}) {
    const char *value = std::getenv(name);
    return value != nullptr ? std::string(value) : defaultValue;
  }

  static std::vector<float> parseFloatVector(const std::string &s, std::size_t expected) {
    std::vector<float> out;
    std::istringstream iss(s);
    float v;
    while (iss >> v) out.push_back(v);
    if (out.size() != expected) out.resize(expected, expected > 0 ? 0.0f : 0.0f);
    return out;
  }

  static std::filesystem::path temporaryInputPath() {
    static std::atomic<uint64_t> sequence{0};
    return std::filesystem::temp_directory_path() /
           ("phoenix-jpea-" + std::to_string(sequence.fetch_add(1)) + ".tensor");
  }

  JpeaV2ImageWorldModelConfig cfg_;
  int targetDim_;
  std::string modelPath_;
  std::string decoderPath_;
  std::string color_;
  std::vector<float> mean_;
  std::vector<float> std_;
  size_t samples_ = 0;
  std::string lastError_;
};

/**
 * @brief Remote implementation of the image world model.
 *
 * Dispatches encode/decode requests to a remote HTTP/JSON edge endpoint.  The
 * encode call expects a float embedding of size @c targetDim_; the decode call
 * expects a base64-encoded payload in return.  Local adaptation and target
 * prediction are not supported by a remote inference-only model.
 */
/**
 * @brief JEPA-style deterministic fallback image world model.
 *
 * When no BPU/remote backend is available, this fallback still implements the
 * full JEPA v2 contract: it decodes the image, splits it into a patch grid,
 * computes patch statistics, and can run an SGD-based linear predictor for
 * self-supervised adaptation.  This makes the image <-> concept unit
 * conversion functional and testable even without a real I-JEPA checkpoint.
 */
class JpeaV2ImageFallbackModel : public JpeaV2ImageWorldModel {
 public:
  JpeaV2ImageFallbackModel(JpeaV2ImageWorldModelConfig cfg, int targetDim)
      : cfg_(std::move(cfg)),
        targetDim_(targetDim > 0 ? targetDim : 128),
        predictorWeights_(static_cast<size_t>(targetDim_ * targetDim_), 0.0f),
        predictorBias_(static_cast<size_t>(targetDim_), 0.0f) {
    for (int i = 0; i < targetDim_; ++i) {
      predictorWeights_[static_cast<size_t>(i * targetDim_ + i)] = 1.0f;
    }
  }

  std::vector<float> encode(const std::vector<uint8_t> &imageBytes, int width, int height,
                            const std::string &mimeType) override {
    auto grid = preprocessImage(imageBytes, width, height, mimeType);
    if (grid.empty()) {
      if (lastError_.empty()) lastError_ = "image preprocessing failed";
      return {};
    }
    return patchStatsToConcept(grid, {});
  }

  std::vector<float> encodeContext(const std::vector<uint8_t> &imageBytes, int width, int height,
                                   const std::string &mimeType,
                                   const std::vector<bool> &mask) override {
    auto grid = preprocessImage(imageBytes, width, height, mimeType);
    if (grid.empty()) {
      if (lastError_.empty()) lastError_ = "image preprocessing failed";
      return {};
    }
    return patchStatsToConcept(grid, mask);
  }

  std::vector<float> encodeTarget(const std::vector<uint8_t> &imageBytes, int width, int height,
                                  const std::string &mimeType,
                                  const std::vector<int> &blockIndices) override {
    auto grid = preprocessImage(imageBytes, width, height, mimeType);
    if (grid.empty()) {
      if (lastError_.empty()) lastError_ = "image preprocessing failed";
      return {};
    }
    const int totalPatches = countPatches();
    std::vector<bool> mask;
    if (!blockIndices.empty()) {
      mask.assign(static_cast<size_t>(totalPatches), false);
      for (int idx : blockIndices) {
        if (idx >= 0 && idx < totalPatches) mask[static_cast<size_t>(idx)] = true;
      }
    }
    return patchStatsToConcept(grid, mask);
  }

  std::vector<float> predictTarget(const std::vector<float> &contextRepr,
                                   const std::vector<int> &) override {
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

  float adapt(const std::vector<uint8_t> &imageBytes, int width, int height,
              const std::string &mimeType, int steps, float lr) override {
    if (imageBytes.empty() || steps < 1) {
      lastError_ = "empty image or non-positive steps";
      return -1.0f;
    }
    if (lr < 0.0f) lr = 0.0f;

    auto grid = preprocessImage(imageBytes, width, height, mimeType);
    if (grid.empty()) {
      if (lastError_.empty()) lastError_ = "image preprocessing failed";
      return -1.0f;
    }
    const int totalPatches = countPatches();
    if (totalPatches < 2) {
      lastError_ = "not enough patches for JEPA adaptation";
      return -1.0f;
    }

    float totalLoss = 0.0f;
    std::mt19937 rng(0x1DEA);
    for (int step = 0; step < steps; ++step) {
      std::vector<int> patchOrder(totalPatches);
      std::iota(patchOrder.begin(), patchOrder.end(), 0);
      std::shuffle(patchOrder.begin(), patchOrder.end(), rng);

      const int targetCount = std::max(1, totalPatches / 3);
      std::vector<int> targetIndices(patchOrder.begin(), patchOrder.begin() + targetCount);

      std::vector<bool> contextMask(static_cast<size_t>(totalPatches), true);
      for (int t : targetIndices) contextMask[static_cast<size_t>(t)] = false;

      auto context = encodeContext(imageBytes, width, height, mimeType, contextMask);
      auto target = encodeTarget(imageBytes, width, height, mimeType, targetIndices);
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

  std::vector<uint8_t> decode(const std::vector<float> &conceptVector,
                              const std::string &mimeType) override {
#if !JPEA_HAVE_OPENCV
    (void)conceptVector;
    (void)mimeType;
    lastError_ = "OpenCV image decoding is unavailable in this build";
    return {};
#else
    if (static_cast<int>(conceptVector.size()) != targetDim_) {
      lastError_ = "decode concept dimension mismatch";
      return {};
    }
    const int patchGrid = std::max(1, cfg_.resolution / std::max(1, cfg_.patchSize));
    cv::Mat patchImg(patchGrid, patchGrid, CV_32FC3, cv::Scalar(0, 0, 0));
    for (int p = 0; p < patchGrid * patchGrid && p < targetDim_; ++p) {
      int y = p / patchGrid;
      int x = p % patchGrid;
      float v = std::max(-1.0f, std::min(1.0f, conceptVector[static_cast<size_t>(p)]));
      cv::Vec3f &pix = patchImg.at<cv::Vec3f>(y, x);
      pix[0] = v;
      pix[1] = -v;
      pix[2] = std::abs(v);
    }
    cv::Mat fullImg;
    cv::resize(patchImg, fullImg, cv::Size(cfg_.resolution, cfg_.resolution), 0, 0, cv::INTER_NEAREST);
    cv::max(cv::min(fullImg, 1.0f), -1.0f, fullImg);
    fullImg = (fullImg + 1.0f) * 127.5f;
    cv::Mat uint8Img;
    fullImg.convertTo(uint8Img, CV_8UC3);
    std::vector<uchar> encoded;
    if (!cv::imencode(mimeType == "image/png" ? ".png" : ".jpg", uint8Img, encoded)) {
      lastError_ = "OpenCV failed to encode fallback image";
      return {};
    }
    lastError_.clear();
    return {encoded.begin(), encoded.end()};
#endif
  }

  nlohmann::json status() const override {
    return nlohmann::json{{"id", cfg_.id}, {"arch", cfg_.arch}, {"backend", "fallback-jepa"},
                          {"resolution", cfg_.resolution}, {"patchSize", cfg_.patchSize},
                          {"targetDim", targetDim_}, {"samples", samples_}, {"ready", true},
                          {"error", lastError_}};
  }

  const JpeaV2ImageWorldModelConfig &config() const override { return cfg_; }

 private:
  int countPatches() const {
    const int grid = cfg_.resolution / std::max(1, cfg_.patchSize);
    return grid * grid;
  }

  std::vector<float> preprocessImage(const std::vector<uint8_t> &imageBytes, int width, int height,
                                     const std::string &mimeType) {
#if !JPEA_HAVE_OPENCV
    (void)imageBytes;
    (void)width;
    (void)height;
    (void)mimeType;
    lastError_ = "OpenCV unavailable in this build";
    return {};
#else
    cv::Mat image;
    if (mimeType == "application/x-bgr") {
      if (width <= 0 || height <= 0 ||
          imageBytes.size() != static_cast<size_t>(width) * static_cast<size_t>(height) * 3U) {
        lastError_ = "direct BGR frame does not match its declared dimensions";
        return {};
      }
      image = cv::Mat(height, width, CV_8UC3, const_cast<uint8_t *>(imageBytes.data())).clone();
    } else {
      if (imageBytes.empty()) {
        lastError_ = "empty image payload";
        return {};
      }
      std::vector<uchar> compressed(imageBytes.begin(), imageBytes.end());
      image = cv::imdecode(compressed, cv::IMREAD_COLOR);
      if (image.empty()) {
        lastError_ = "unable to decode image payload";
        return {};
      }
    }
    cv::resize(image, image, cv::Size(cfg_.resolution, cfg_.resolution), 0, 0, cv::INTER_AREA);
    cv::Mat gray;
    if (image.channels() == 3) {
      cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 1) {
      gray = image;
    } else {
      lastError_ = "unsupported image channel count";
      return {};
    }
    cv::Mat floatGray;
    gray.convertTo(floatGray, CV_32F, 1.0 / 255.0);
    std::vector<float> grid(static_cast<size_t>(cfg_.resolution * cfg_.resolution));
    std::memcpy(grid.data(), floatGray.ptr<float>(), grid.size() * sizeof(float));
    lastError_.clear();
    return grid;
#endif
  }

  std::vector<float> patchStatsToConcept(const std::vector<float> &grid,
                                         const std::vector<bool> &mask) const {
    using namespace phoenix::multimodal;
    const int patchGrid = cfg_.resolution / std::max(1, cfg_.patchSize);
    if (patchGrid <= 0) return std::vector<float>(static_cast<size_t>(targetDim_), 0.0f);
    std::vector<float> stats;
    stats.reserve(2 * patchGrid * patchGrid);
    for (int py = 0; py < patchGrid; ++py) {
      for (int px = 0; px < patchGrid; ++px) {
        const int patchIdx = py * patchGrid + px;
        if (!mask.empty() && !mask[static_cast<size_t>(patchIdx)]) continue;
        float mean = 0.0f;
        float sq = 0.0f;
        int count = 0;
        const int y0 = py * cfg_.patchSize;
        const int x0 = px * cfg_.patchSize;
        for (int y = 0; y < cfg_.patchSize; ++y) {
          for (int x = 0; x < cfg_.patchSize; ++x) {
            const int gy = y0 + y;
            const int gx = x0 + x;
            const size_t idx = static_cast<size_t>(gy * cfg_.resolution + gx);
            if (idx >= grid.size()) continue;
            float v = grid[idx];
            mean += v;
            sq += v * v;
            ++count;
          }
        }
        if (count == 0) continue;
        mean /= static_cast<float>(count);
        float var = (sq / static_cast<float>(count)) - (mean * mean);
        stats.push_back(mean);
        stats.push_back(std::sqrt(std::max(0.0f, var)));
      }
    }
    if (stats.empty()) stats.assign(2, 0.0f);
    auto projected = projectToDimension(stats, static_cast<size_t>(targetDim_), 0x1DEA);
    return normalizeVector(projected);
  }

  JpeaV2ImageWorldModelConfig cfg_;
  int targetDim_;
  size_t samples_ = 0;
  std::vector<float> predictorWeights_;
  std::vector<float> predictorBias_;
  std::string lastError_;
};

class JpeaV2ImageRemoteModel : public JpeaV2ImageWorldModel {
 public:
  JpeaV2ImageRemoteModel(JpeaV2ImageWorldModelConfig cfg, int targetDim,
                         const phoenix::deployment::RemoteEndpoint &endpoint)
      : cfg_(std::move(cfg)), targetDim_(targetDim > 0 ? targetDim : 128),
        endpoint_(endpoint) {}

  std::vector<float> encode(const std::vector<uint8_t> &imageBytes, int width, int height,
                            const std::string &mimeType) override {
    nlohmann::json payload;
    payload["modality"] = "image";
    payload["mimeType"] = mimeType;
    payload["width"] = width;
    payload["height"] = height;
    payload["payloadBase64"] = phoenix::deployment::RemoteModelClient::base64Encode(imageBytes);
    payload["conceptDim"] = targetDim_;

    auto result = phoenix::deployment::RemoteModelClient::call(endpoint_, payload);
    if (!result.value("ok", false)) {
      lastError_ = result.value("error", std::string("remote image encode failed"));
      return {};
    }
    auto embedding = parseFloatVectorFromJson(result.value("embedding", nlohmann::json::array()));
    if (static_cast<int>(embedding.size()) != targetDim_) {
      lastError_ = "remote image embedding dimension mismatch";
      return {};
    }
    ++samples_;
    lastError_.clear();
    return embedding;
  }

  std::vector<float> encodeContext(const std::vector<uint8_t> &imageBytes, int width, int height,
                                   const std::string &mimeType, const std::vector<bool> &) override {
    return encode(imageBytes, width, height, mimeType);
  }

  std::vector<float> encodeTarget(const std::vector<uint8_t> &imageBytes, int width, int height,
                                  const std::string &mimeType,
                                  const std::vector<int> &) override {
    // Remote endpoints are currently inference-only; return the full-image
    // embedding as a deterministic target fallback.
    return encode(imageBytes, width, height, mimeType);
  }

  std::vector<float> predictTarget(const std::vector<float> &contextRepr,
                                   const std::vector<int> &) override {
    // No remote predictor exposed; identity is the safest deterministic default.
    return std::vector<float>(contextRepr.begin(), contextRepr.end());
  }

  float adapt(const std::vector<uint8_t> &, int, int, const std::string &, int, float) override {
    lastError_ = "remote runtime adaptation is not supported";
    return -1.0f;
  }

  std::vector<uint8_t> decode(const std::vector<float> &conceptVector,
                              const std::string &mimeType) override {
    nlohmann::json payload;
    payload["modality"] = "image";
    payload["decode"] = true;
    payload["mimeType"] = mimeType;
    payload["conceptVector"] = conceptVector;

    auto result = phoenix::deployment::RemoteModelClient::call(endpoint_, payload);
    if (!result.value("ok", false)) {
      lastError_ = result.value("error", std::string("remote image decode failed"));
      return {};
    }
    auto encoded = result.value("payloadBase64", std::string());
    if (encoded.empty()) {
      lastError_ = "remote image decode returned empty payload";
      return {};
    }
    auto decoded = phoenix::deployment::RemoteModelClient::base64Decode(encoded);
    if (decoded.empty() && !encoded.empty()) {
      lastError_ = "remote image decode returned invalid base64";
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

  const JpeaV2ImageWorldModelConfig &config() const override { return cfg_; }

 private:
  static std::vector<float> parseFloatVectorFromJson(const nlohmann::json &j) {
    std::vector<float> out;
    if (!j.is_array()) return out;
    for (const auto &v : j) {
      if (v.is_number()) out.push_back(v.get<double>());
    }
    return out;
  }

  JpeaV2ImageWorldModelConfig cfg_;
  int targetDim_;
  phoenix::deployment::RemoteEndpoint endpoint_;
  size_t samples_ = 0;
  std::string lastError_;
};

}  // namespace

/**
 * @brief Factory that selects the best available image world model.
 *
 * Preference order: remote edge endpoint (when configured), local HBDNN BPU
 * model (when the runtime and compiled model are both available), and finally
 * the JEPA-style deterministic fallback.  The fallback is also used for
 * ServerClient placement, disabled edge builds, or any situation where the
 * HBDNN backend is not ready.
 */
std::unique_ptr<JpeaV2ImageWorldModel> createJpeaV2ImageWorldModel(
    const std::string &variantId, int targetDim, const std::string & /*backend*/) {
  const auto *v = findJpeaV2ImageVariant(variantId);
  if (!v) {
    v = findJpeaV2ImageVariant("ijepa_vith14_1k");
  }

  const auto &deployment = phoenix::deployment::ModelDeploymentConfig::instance().vision();

  if (deployment.placement == phoenix::deployment::ModelPlacement::ServerClient) {
    return std::make_unique<JpeaV2ImageFallbackModel>(*v, targetDim);
  }

#if PHOENIX_EDGE_IMAGE_ENABLED
  const bool useRemote =
      (deployment.placement == phoenix::deployment::ModelPlacement::Remote ||
       deployment.placement == phoenix::deployment::ModelPlacement::Auto) &&
      !deployment.remote.url.empty();
  if (useRemote) {
    return std::make_unique<JpeaV2ImageRemoteModel>(*v, targetDim, deployment.remote);
  }

  // Local path: prefer HBDNN when BPU runtime + compiled model are present.
  auto hbdnn = std::make_unique<JpeaV2ImageHbdnnModel>(*v, targetDim);
  if (hbdnn->status().value("ready", false)) {
    return hbdnn;
  }
  // Otherwise the JEPA fallback keeps image -> concept unit conversion alive.
  return std::make_unique<JpeaV2ImageFallbackModel>(*v, targetDim);
#else
  (void)deployment;
  return std::make_unique<JpeaV2ImageFallbackModel>(*v, targetDim);
#endif
}

}  // namespace io
}  // namespace phoenix
