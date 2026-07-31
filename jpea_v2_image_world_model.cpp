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

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
 * @brief Deterministic fallback image world model.
 *
 * Used when edge image inference (BPU or remote) is disabled at compile time
 * or when the selected backend is unavailable.  encode() returns an empty
 * vector so the multimodal concept bridge can use its media-concept fallback.
 */
class JpeaV2ImageFallbackModel : public JpeaV2ImageWorldModel {
 public:
  JpeaV2ImageFallbackModel(JpeaV2ImageWorldModelConfig cfg, int targetDim)
      : cfg_(std::move(cfg)), targetDim_(targetDim > 0 ? targetDim : 128) {}

  std::vector<float> encode(const std::vector<uint8_t> &, int, int,
                            const std::string &) override {
    lastError_ = "edge image inference disabled";
    return {};
  }
  std::vector<float> encodeContext(const std::vector<uint8_t> &imageBytes, int width, int height,
                                   const std::string &mimeType,
                                   const std::vector<bool> &) override {
    return encode(imageBytes, width, height, mimeType);
  }
  std::vector<float> encodeTarget(const std::vector<uint8_t> &imageBytes, int width, int height,
                                  const std::string &mimeType,
                                  const std::vector<int> &) override {
    return encode(imageBytes, width, height, mimeType);
  }
  std::vector<float> predictTarget(const std::vector<float> &contextRepr,
                                   const std::vector<int> &) override {
    return std::vector<float>(contextRepr.begin(), contextRepr.end());
  }
  float adapt(const std::vector<uint8_t> &, int, int, const std::string &, int, float) override {
    lastError_ = "runtime adaptation is not supported";
    return -1.0f;
  }
  std::vector<uint8_t> decode(const std::vector<float> &, const std::string &) override {
    lastError_ = "edge image decoding disabled";
    return {};
  }
  nlohmann::json status() const override {
    return nlohmann::json{{"id", cfg_.id}, {"arch", cfg_.arch}, {"backend", "disabled"},
                          {"resolution", cfg_.resolution}, {"patchSize", cfg_.patchSize},
                          {"targetDim", targetDim_}, {"samples", 0u}, {"ready", false},
                          {"error", lastError_}};
  }
  const JpeaV2ImageWorldModelConfig &config() const override { return cfg_; }

 private:
  JpeaV2ImageWorldModelConfig cfg_;
  int targetDim_;
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
 * @brief Factory that selects a local HBDNN or remote image world model.
 *
 * The decision is driven by the global ModelDeploymentConfig singleton.  When
 * vision is configured as remote and a URL is present, a JpeaV2ImageRemoteModel
 * is returned; otherwise the local HBDNN-backed implementation is used.
 */
std::unique_ptr<JpeaV2ImageWorldModel> createJpeaV2ImageWorldModel(
    const std::string &variantId, int targetDim, const std::string & /*backend*/) {
  const auto *v = findJpeaV2ImageVariant(variantId);
  if (!v) {
    v = findJpeaV2ImageVariant("ijepa_vith14_1k");
  }

  const auto &deployment = phoenix::deployment::ModelDeploymentConfig::instance().vision();

#if PHOENIX_EDGE_IMAGE_ENABLED
  const bool useRemote =
      (deployment.placement == phoenix::deployment::ModelPlacement::Remote ||
       deployment.placement == phoenix::deployment::ModelPlacement::Auto) &&
      !deployment.remote.url.empty();
  if (useRemote) {
    return std::make_unique<JpeaV2ImageRemoteModel>(*v, targetDim, deployment.remote);
  }
  // Local path: use the HBDNN model.  If no BPU / no model is configured it
  // will return empty vectors and the concept bridge will fall back.
  return std::make_unique<JpeaV2ImageHbdnnModel>(*v, targetDim);
#else
  (void)deployment;
  return std::make_unique<JpeaV2ImageFallbackModel>(*v, targetDim);
#endif
}

}  // namespace io
}  // namespace phoenix
