/* video_model.cpp - video world model factory and fallbacks
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version. */

#include "video_model.hpp"
#include "local_onnx.hpp"
#include "model_deployment.hpp"
#include "phoenix_config.hpp"
#include "rdk_x5_bpu.hpp"
#include "semantic_unit.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#ifndef PHOENIX_EDGE_IMAGE_ENABLED
#define PHOENIX_EDGE_IMAGE_ENABLED 1
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>

namespace phoenix {
namespace io {

namespace {

constexpr int kDefaultResolution = 224;
constexpr int kDefaultConceptDim = 128;

std::string imageWeightsDir(const std::string &variantId) {
  if (variantId == "video-encoder" || variantId == "video-decoder" ||
      variantId == "vision_encoder") return "vision_encoder";
  if (variantId == "resnet18_224") return "resnet18_224";
  return variantId;
}

std::string additiveDirFor(const std::string &ijepaDir, const std::string &modelKind) {
  std::string dir = ijepaDir;
  const bool isEncoder = (modelKind == "encoder");
  auto pos = dir.find("_decoder");
  if (isEncoder && pos != std::string::npos) {
    dir.replace(pos, 8, "_encoder");
    return dir;
  }
  pos = dir.find("_encoder");
  if (!isEncoder && pos != std::string::npos) {
    dir.replace(pos, 8, "_decoder");
  }
  return dir;
}

std::string resolveBpuModelPath(const std::string &envOverride,
                                const std::string &modelKind,
                                const std::string &ijepaDir) {
  if (!envOverride.empty()) return envOverride;

  const std::vector<std::string> names = [modelKind]() {
    if (modelKind == "encoder") return std::vector<std::string>{"best.bin", "model_encoder.bin", "encoder.bin", "model.bin", "ijepa.bin"};
    if (modelKind == "decoder") return std::vector<std::string>{"best.bin", "model_decoder.bin", "decoder.bin", "decode.bin"};
    return std::vector<std::string>{"best.bin", "model.bin"};
  }();

  const std::vector<std::string> roots = {
      std::string("runtime_store/models/additive_jepa/") + additiveDirFor(ijepaDir, modelKind),
      std::string("runtime_store/models/ijepa/") + ijepaDir,
      std::string("runtime_store/models/bpu_jepa/") + ijepaDir,
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

std::string resolveOnnxModelPath(const std::string &modelKind,
                                 const std::string &ijepaDir) {
  const std::vector<std::string> names = {"best.onnx", "model.onnx"};
  const std::vector<std::string> roots = {
      std::string("runtime_store/models/additive_jepa/") + additiveDirFor(ijepaDir, modelKind),
      std::string("runtime_store/models/ijepa/") + ijepaDir,
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

nlohmann::json readModelManifest(const std::filesystem::path &modelDir) {
  std::filesystem::path manifestPath = modelDir / "model.manifest.json";
  std::ifstream in(manifestPath, std::ios::binary);
  if (!in) return {};
  try {
    nlohmann::json j;
    in >> j;
    return j;
  } catch (const std::exception &) {
    return {};
  }
}

static std::filesystem::path temporaryOnnxPath() {
  static std::atomic<uint64_t> sequence{0};
  std::error_code ec;
  std::filesystem::path base;
  const char *env = std::getenv("PHOENIX_ONNX_TMP");
  if (env && *env) {
    base = std::filesystem::path(env);
  } else {
    base = std::filesystem::path("build") / "tmp";
  }
  std::filesystem::create_directories(base, ec);
  if (!std::filesystem::is_directory(base, ec)) {
    base = std::filesystem::temp_directory_path(ec);
  }
  return base / ("phoenix-onnx-image-" + std::to_string(sequence.fetch_add(1)));
}

static std::string pythonExecutable() {
  std::error_code ec;
  const char *env = std::getenv("PHOENIX_PYTHON");
  if (env && *env) return std::string(env);
  const std::vector<std::string> candidates = {
      "Python314/python", "Python314/python.exe", "Python314/pythonw.exe",
      "python3", "python", "py"};
  for (const auto &c : candidates) {
    if (std::filesystem::is_regular_file(c, ec))
      return std::filesystem::absolute(c, ec).string();
  }
  return "python3";
}

static std::string toShapeString(const std::vector<int> &shape) {
  std::ostringstream oss;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) oss << "x";
    oss << shape[i];
  }
  return oss.str();
}

static nlohmann::json runLocalOnnx(
    const std::string &modelPath,
    const std::string &inputName,
    const std::vector<int> &inputShape,
    const std::vector<float> &inputFloats,
    const std::string &outputName,
    const std::vector<int> &outputShape,
    bool gpu) {
  std::error_code ec;
  const auto inPath = temporaryOnnxPath().replace_extension(".in");
  const auto outPath = temporaryOnnxPath().replace_extension(".out");

  {
    std::ofstream out(inPath, std::ios::binary | std::ios::trunc);
    if (!out) {
      return {{"ok", false}, {"error", "failed to write ONNX input binary"}};
    }
    out.write(reinterpret_cast<const char *>(inputFloats.data()),
              static_cast<std::streamsize>(inputFloats.size() * sizeof(float)));
  }

  const std::string py = pythonExecutable();
  if (py.find_first_of(" \t") != std::string::npos) {
    std::filesystem::remove(inPath, ec);
    return {{"ok", false}, {"error", "python executable path contains spaces; set PHOENIX_PYTHON"}};
  }
  auto quoteIfNeeded = [](const std::string &s) -> std::string {
    if (s.find_first_of(" \t\"&|<>^%;") == std::string::npos) return s;
    std::string out;
    out.push_back('\"');
    for (char c : s) {
      if (c == '\"') out.push_back('\"');
      out.push_back(c);
    }
    out.push_back('\"');
    return out;
  };

  std::ostringstream cmd;
  cmd << py << " "
      << "tools/local_onnx_runner.py "
      << "--model " << quoteIfNeeded(modelPath) << " "
      << "--input " << quoteIfNeeded(inPath.string()) << " "
      << "--input-name " << quoteIfNeeded(inputName) << " "
      << "--input-shape " << toShapeString(inputShape) << " "
      << "--output " << quoteIfNeeded(outPath.string()) << " "
      << "--output-name " << quoteIfNeeded(outputName) << " "
      << "--output-shape " << toShapeString(outputShape);
  if (gpu) cmd << " --gpu";

  std::string outputJson;
  FILE *pipe = _popen(cmd.str().c_str(), "r");
  if (!pipe) {
    std::filesystem::remove(inPath, ec);
    return {{"ok", false}, {"error", "failed to start local ONNX runner"}};
  }
  char buffer[4096];
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    outputJson += buffer;
  }
  const int rc = _pclose(pipe);
  std::filesystem::remove(inPath, ec);

  if (outputJson.empty()) {
    return {{"ok", false}, {"error", "local ONNX runner produced no output"}};
  }
  auto result = nlohmann::json::parse(outputJson, nullptr, false);
  if (result.is_discarded()) {
    return {{"ok", false}, {"error", "local ONNX runner returned invalid JSON"}, {"raw", outputJson}};
  }
  if (!result.value("ok", false) || rc != 0) {
    return result;
  }

  std::ifstream in(outPath, std::ios::binary);
  if (!in) {
    return {{"ok", false}, {"error", "local ONNX runner did not write output file"}};
  }
  const size_t expected = static_cast<size_t>(std::accumulate(outputShape.begin(), outputShape.end(), 1, std::multiplies<int>()));
  std::vector<float> outputFloats(expected);
  in.read(reinterpret_cast<char *>(outputFloats.data()),
          static_cast<std::streamsize>(expected * sizeof(float)));
  if (!in) {
    return {{"ok", false}, {"error", "failed to read ONNX output binary"}};
  }
  std::filesystem::remove(outPath, ec);
  result["floats"] = std::move(outputFloats);
  return result;
}

class VideoFallbackModel : public VideoModel {
 public:
  VideoFallbackModel(VideoModelConfig cfg, int targetDim)
      : cfg_(std::move(cfg)), targetDim_(targetDim > 0 ? targetDim : kDefaultConceptDim) {}

  std::vector<float> encode(const std::vector<uint8_t> &, int, int, const std::string &) override {
    lastError_ = "no compiled image model available; deterministic image fallback returns empty concept";
    return {};
  }

  std::vector<float> encodeContext(const std::vector<uint8_t> &, int, int, const std::string &, const std::vector<bool> &) override { return {}; }
  std::vector<float> encodeTarget(const std::vector<uint8_t> &, int, int, const std::string &, const std::vector<int> &) override { return {}; }
  std::vector<float> predictTarget(const std::vector<float> &, const std::vector<int> &) override { return {}; }

  float adapt(const std::vector<uint8_t> &, int, int, const std::string &, int, float) override {
    lastError_ = "fallback model does not support adaptation";
    return -1.0f;
  }

  std::vector<uint8_t> decode(const std::vector<float> &, const std::string &) override {
    lastError_ = "no compiled image decoder available";
    return {};
  }

  nlohmann::json status() const override {
    return nlohmann::json{{"id", cfg_.id}, {"arch", cfg_.arch}, {"backend", kind_.empty() ? std::string("fallback") : kind_},
                          {"resolution", cfg_.resolution}, {"targetDim", targetDim_},
                          {"samples", samples_}, {"ready", false}, {"error", lastError_}};
  }

  const VideoModelConfig &config() const override { return cfg_; }

 private:
  VideoModelConfig cfg_;
  int targetDim_;
  size_t samples_ = 0;
  mutable std::string lastError_;
};

class VideoUnavailableModel : public VideoModel {
 public:
  VideoUnavailableModel(VideoModelConfig cfg, int targetDim, std::string reason)
      : cfg_(std::move(cfg)), targetDim_(targetDim > 0 ? targetDim : kDefaultConceptDim), reason_(std::move(reason)) {}

  std::vector<float> encode(const std::vector<uint8_t> &, int, int, const std::string &) override { return {}; }
  std::vector<float> encodeContext(const std::vector<uint8_t> &, int, int, const std::string &, const std::vector<bool> &) override { return {}; }
  std::vector<float> encodeTarget(const std::vector<uint8_t> &, int, int, const std::string &, const std::vector<int> &) override { return {}; }
  std::vector<float> predictTarget(const std::vector<float> &, const std::vector<int> &) override { return {}; }
  float adapt(const std::vector<uint8_t> &, int, int, const std::string &, int, float) override { return -1.0f; }
  std::vector<uint8_t> decode(const std::vector<float> &, const std::string &) override { return {}; }

  nlohmann::json status() const override {
    return nlohmann::json{{"id", cfg_.id}, {"arch", cfg_.arch}, {"backend", kind_.empty() ? std::string("unavailable") : kind_},
                          {"resolution", cfg_.resolution}, {"targetDim", targetDim_},
                          {"samples", samples_}, {"ready", false}, {"error", reason_}};
  }

  const VideoModelConfig &config() const override { return cfg_; }

 private:
  VideoModelConfig cfg_;
  int targetDim_;
  size_t samples_ = 0;
  std::string reason_;
};

class VideoServerClientModel : public VideoModel {
 public:
  VideoServerClientModel(VideoModelConfig cfg, int targetDim)
      : cfg_(std::move(cfg)), targetDim_(targetDim > 0 ? targetDim : kDefaultConceptDim) {}

  std::vector<float> encode(const std::vector<uint8_t> &, int, int, const std::string &) override {
    lastError_ = "server-client mode expects a client-supplied concept vector";
    return {};
  }
  std::vector<float> encodeContext(const std::vector<uint8_t> &, int, int, const std::string &, const std::vector<bool> &) override { return {}; }
  std::vector<float> encodeTarget(const std::vector<uint8_t> &, int, int, const std::string &, const std::vector<int> &) override { return {}; }
  std::vector<float> predictTarget(const std::vector<float> &contextRepr, const std::vector<int> &) override { return contextRepr; }
  float adapt(const std::vector<uint8_t> &, int, int, const std::string &, int, float) override {
    lastError_ = "server-client mode does not support adaptation";
    return -1.0f;
  }
  std::vector<uint8_t> decode(const std::vector<float> &, const std::string &) override {
    lastError_ = "server-client mode expects a client-supplied decoded payload";
    return {};
  }
  nlohmann::json status() const override {
    return nlohmann::json{{"id", cfg_.id}, {"arch", cfg_.arch}, {"backend", kind_.empty() ? std::string("server-client") : kind_},
                          {"resolution", cfg_.resolution}, {"targetDim", targetDim_},
                          {"samples", samples_}, {"ready", false},
                          {"error", "server-client: expects client concept vectors"}};
  }
  const VideoModelConfig &config() const override { return cfg_; }

 private:
  VideoModelConfig cfg_;
  int targetDim_;
  size_t samples_ = 0;
  mutable std::string lastError_;
};

class VideoRemoteModel : public VideoModel {
 public:
  VideoRemoteModel(VideoModelConfig cfg, int targetDim,
                         const phoenix::deployment::RemoteEndpoint &endpoint)
      : cfg_(std::move(cfg)), targetDim_(targetDim > 0 ? targetDim : kDefaultConceptDim), endpoint_(endpoint) {}

  std::vector<float> encode(const std::vector<uint8_t> &imageBytes,
                            int width,
                            int height,
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

  std::vector<float> encodeContext(const std::vector<uint8_t> &imageBytes,
                                   int width,
                                   int height,
                                   const std::string &mimeType,
                                   const std::vector<bool> &) override {
    return encode(imageBytes, width, height, mimeType);
  }

  std::vector<float> encodeTarget(const std::vector<uint8_t> &imageBytes,
                                  int width,
                                  int height,
                                  const std::string &mimeType,
                                  const std::vector<int> &) override {
    return encode(imageBytes, width, height, mimeType);
  }

  std::vector<float> predictTarget(const std::vector<float> &contextRepr, const std::vector<int> &) override { return contextRepr; }
  float adapt(const std::vector<uint8_t> &, int, int, const std::string &, int, float) override {
    lastError_ = "remote runtime adaptation is not supported";
    return -1.0f;
  }

  std::vector<uint8_t> decode(const std::vector<float> &conceptVector, const std::string &mimeType) override {
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
    return nlohmann::json{{"id", cfg_.id}, {"arch", cfg_.arch}, {"backend", kind_.empty() ? std::string("remote") : kind_},
                          {"url", endpoint_.url}, {"method", endpoint_.method},
                          {"resolution", cfg_.resolution}, {"targetDim", targetDim_},
                          {"samples", samples_}, {"ready", !endpoint_.url.empty()}, {"error", lastError_}};
  }

  const VideoModelConfig &config() const override { return cfg_; }

 private:
  static std::vector<float> parseFloatVectorFromJson(const nlohmann::json &j) {
    std::vector<float> out;
    if (!j.is_array()) return out;
    for (const auto &v : j) {
      if (v.is_number()) out.push_back(v.get<double>());
    }
    return out;
  }

  VideoModelConfig cfg_;
  int targetDim_;
  phoenix::deployment::RemoteEndpoint endpoint_;
  size_t samples_ = 0;
  mutable std::string lastError_;
};

class VideoLocalOnnxModel : public VideoModel {
 public:
  VideoLocalOnnxModel(VideoModelConfig cfg, int targetDim, bool gpu)
      : cfg_(std::move(cfg)), targetDim_(targetDim > 0 ? targetDim : kDefaultConceptDim), gpu_(gpu),
        modelPath_(resolveOnnxModelPath("encoder", imageWeightsDir(cfg_.id))),
        decoderPath_(resolveOnnxModelPath("decoder", imageWeightsDir(cfg_.id))) {
    loadManifest("encoder", encoderInputName_, encoderOutputName_, encoderInputShape_, encoderOutputShape_, conceptDim_);
    loadManifest("decoder", decoderInputName_, decoderOutputName_, decoderInputShape_, decoderOutputShape_, conceptDim_);
    if (conceptDim_ <= 0) conceptDim_ = kDefaultConceptDim;
    if (modelPath_.empty()) {
      lastError_ = "local ONNX image encoder model (best.onnx) is not configured";
    }
  }

  std::vector<float> encode(const std::vector<uint8_t> &imageBytes,
                            int width,
                            int height,
                            const std::string &mimeType) override {
    if (modelPath_.empty()) {
      lastError_ = "local ONNX image encoder model (best.onnx) is missing";
      return {};
    }
    auto input = prepareImageInput(imageBytes, width, height, mimeType);
    if (input.empty()) {
      if (lastError_.empty()) lastError_ = "local ONNX image encoder preprocessing failed";
      return {};
    }

    auto result = phoenix::io::runLocalOnnx(
        modelPath_, encoderInputName_, encoderInputShape_, input,
        encoderOutputName_, encoderOutputShape_, gpu_);
    if (!result.value("ok", false)) {
      lastError_ = result.value("error", std::string("local ONNX image encode failed"));
      return {};
    }
    auto values = result.value("floats", std::vector<float>{});
    if (static_cast<int>(values.size()) != conceptDim_) {
      lastError_ = "ONNX image encoder output dimension mismatch";
      return {};
    }
    if (targetDim_ != conceptDim_) {
      values = phoenix::multimodal::projectToDimension(values, static_cast<size_t>(targetDim_), 0x57494D47U);
    }
    ++samples_;
    lastError_.clear();
    return phoenix::multimodal::normalizeVector(values);
  }

  std::vector<float> encodeContext(const std::vector<uint8_t> &imageBytes,
                                   int width,
                                   int height,
                                   const std::string &mimeType,
                                   const std::vector<bool> &) override {
    return encode(imageBytes, width, height, mimeType);
  }

  std::vector<float> encodeTarget(const std::vector<uint8_t> &imageBytes,
                                  int width,
                                  int height,
                                  const std::string &mimeType,
                                  const std::vector<int> &) override {
    return encode(imageBytes, width, height, mimeType);
  }

  std::vector<float> predictTarget(const std::vector<float> &contextRepr, const std::vector<int> &) override {
    return std::vector<float>(contextRepr.begin(), contextRepr.end());
  }

  float adapt(const std::vector<uint8_t> &, int, int, const std::string &, int, float) override {
    lastError_ = "local ONNX image model does not support adaptation";
    return -1.0f;
  }

  std::vector<uint8_t> decode(const std::vector<float> &conceptVector, const std::string &mimeType) override {
    if (decoderPath_.empty()) {
      lastError_ = "local ONNX image decoder model (best.onnx) is not configured";
      return {};
    }
    std::vector<float> decoderConcept = conceptVector;
    if (static_cast<int>(decoderConcept.size()) != conceptDim_) {
      decoderConcept = phoenix::multimodal::projectToDimension(decoderConcept, static_cast<size_t>(conceptDim_), 0x57494D47U);
    }

    auto result = phoenix::io::runLocalOnnx(
        decoderPath_, decoderInputName_, decoderInputShape_, decoderConcept,
        decoderOutputName_, decoderOutputShape_, gpu_);
    if (!result.value("ok", false)) {
      lastError_ = result.value("error", std::string("local ONNX image decode failed"));
      return {};
    }
    auto values = result.value("floats", std::vector<float>{});
    return renderImage(values, mimeType);
  }

  nlohmann::json status() const override {
    std::error_code ec;
    bool modelReady = !modelPath_.empty() && std::filesystem::is_regular_file(modelPath_, ec);
    bool decoderReady = !decoderPath_.empty() && std::filesystem::is_regular_file(decoderPath_, ec);
    std::string backend = kind_.empty() ? (gpu_ ? std::string("local-gpu") : std::string("local-onnx")) : kind_;
    return nlohmann::json{{"id", cfg_.id}, {"arch", cfg_.arch}, {"backend", backend},
                          {"resolution", cfg_.resolution}, {"targetDim", targetDim_},
                          {"conceptDim", conceptDim_}, {"samples", samples_},
                          {"modelPath", modelPath_}, {"decoderPath", decoderPath_},
                          {"ready", modelReady}, {"decoderReady", decoderReady}, {"error", lastError_}};
  }

  const VideoModelConfig &config() const override { return cfg_; }

 private:
  void loadManifest(const std::string &kind, std::string &inputName, std::string &outputName,
                    std::vector<int> &inputShape, std::vector<int> &outputShape, int &conceptDim) {
    std::error_code ec;
    const std::string &path = (kind == "encoder") ? modelPath_ : decoderPath_;
    if (path.empty()) return;
    auto dir = std::filesystem::path(path).parent_path();
    auto manifest = readModelManifest(dir);
    if (!manifest.is_object()) return;
    inputName = manifest.value("input_name", kind == "encoder" ? "pixel_values" : "concept");
    outputName = manifest.value("output_name", kind == "encoder" ? "concept" : "reconstruction");
    if (manifest.contains("input_shape") && manifest["input_shape"].is_array())
      inputShape = manifest["input_shape"].get<std::vector<int>>();
    if (manifest.contains("output_shape") && manifest["output_shape"].is_array())
      outputShape = manifest["output_shape"].get<std::vector<int>>();
    if (manifest.contains("concept_dim") && manifest["concept_dim"].is_number())
      conceptDim = manifest["concept_dim"].get<int>();
    if (inputShape.empty())
      inputShape = (kind == "encoder") ? std::vector<int>{1, 3, kDefaultResolution, kDefaultResolution}
                                         : std::vector<int>{1, conceptDim, 1, 1};
    if (outputShape.empty())
      outputShape = (kind == "encoder") ? std::vector<int>{1, conceptDim, 1, 1}
                                         : std::vector<int>{1, 3, kDefaultResolution, kDefaultResolution};
  }

  std::vector<float> prepareImageInput(const std::vector<uint8_t> &imageBytes,
                                       int width,
                                       int height,
                                       const std::string &mimeType) {
    cv::Mat img;
    if (!imageBytes.empty() && width > 0 && height > 0 &&
        static_cast<int>(imageBytes.size()) == width * height * 3) {
      img = cv::Mat(height, width, CV_8UC3, const_cast<uint8_t *>(imageBytes.data()));
    } else if (!imageBytes.empty()) {
      std::vector<int> params;
      const std::vector<uint8_t> *source = &imageBytes;
      std::vector<uint8_t> tmp;
      if (!mimeType.empty() && mimeType.find("/raw") != std::string::npos &&
          width > 0 && height > 0 && static_cast<int>(imageBytes.size()) >= width * height * 3) {
        img = cv::Mat(height, width, CV_8UC3, const_cast<uint8_t *>(imageBytes.data()));
      } else {
        cv::Mat decoded = cv::imdecode(imageBytes, cv::IMREAD_COLOR);
        if (!decoded.empty()) img = decoded;
      }
    }

    if (img.empty()) {
      int targetRes = cfg_.resolution > 0 ? cfg_.resolution : kDefaultResolution;
      if (width > 0 && height > 0) {
        img = cv::Mat::zeros(height, width, CV_8UC3);
      } else {
        img = cv::Mat::zeros(targetRes, targetRes, CV_8UC3);
      }
    }

    const int res = cfg_.resolution > 0 ? cfg_.resolution : kDefaultResolution;
    if (img.rows != res || img.cols != res) {
      cv::resize(img, img, cv::Size(res, res), 0, 0, cv::INTER_LINEAR);
    }

    // BGR -> RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    cv::Mat floatImg;
    img.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);

    // ImageNet normalization by default
    constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
    constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};
    for (int y = 0; y < floatImg.rows; ++y) {
      auto *row = floatImg.ptr<cv::Vec3f>(y);
      for (int x = 0; x < floatImg.cols; ++x) {
        for (int c = 0; c < 3; ++c) {
          row[x][c] = (row[x][c] - kMean[c]) / kStd[c];
        }
      }
    }

    std::vector<float> input(static_cast<size_t>(res * res * 3));
    for (int c = 0; c < 3; ++c) {
      for (int y = 0; y < res; ++y) {
        const auto *row = floatImg.ptr<cv::Vec3f>(y);
        for (int x = 0; x < res; ++x) {
          input[static_cast<size_t>(c * res * res + y * res + x)] = row[x][c];
        }
      }
    }
    return input;
  }

  std::vector<uint8_t> renderImage(const std::vector<float> &output,
                                   const std::string &mimeType) {
    const int res = cfg_.resolution > 0 ? cfg_.resolution : kDefaultResolution;
    const size_t expected = static_cast<size_t>(res * res * 3);
    if (output.size() < expected) {
      lastError_ = "ONNX image decoder output size mismatch";
      return {};
    }

    cv::Mat img(res, res, CV_32FC3);
    for (int c = 0; c < 3; ++c) {
      for (int y = 0; y < res; ++y) {
        auto *row = img.ptr<cv::Vec3f>(y);
        for (int x = 0; x < res; ++x) {
          row[x][c] = output[static_cast<size_t>(c * res * res + y * res + x)];
        }
      }
    }

    // Assume decoder outputs roughly [-1, 1] or [0, 1] normalized logits.
    // Robust linear map: find min/max, stretch to [0, 255] while preserving shape.
    double minVal = 0, maxVal = 0;
    cv::minMaxLoc(img.reshape(1), &minVal, &maxVal);
    double range = maxVal - minVal;
    if (range < 1e-6) range = 1.0;
    cv::Mat scaled;
    cv::Mat((img - minVal) * (255.0 / range)).convertTo(scaled, CV_8UC3);

    cv::Mat bgr;
    cv::cvtColor(scaled, bgr, cv::COLOR_RGB2BGR);

    std::vector<int> compressParams;
    std::string ext = ".png";
    if (mimeType == "image/jpeg" || mimeType == "image/jpg") {
      ext = ".jpg";
      compressParams = {cv::IMWRITE_JPEG_QUALITY, 90};
    }
    std::vector<uint8_t> out;
    if (!cv::imencode(ext, bgr, out, compressParams)) {
      lastError_ = "local ONNX image decoder: cv::imencode failed";
      return {};
    }
    lastError_.clear();
    return out;
  }

  VideoModelConfig cfg_;
  int targetDim_;
  bool gpu_;
  std::string modelPath_;
  std::string decoderPath_;
  size_t samples_ = 0;
  mutable std::string lastError_;
  std::string encoderInputName_;
  std::string encoderOutputName_;
  std::vector<int> encoderInputShape_;
  std::vector<int> encoderOutputShape_;
  std::string decoderInputName_;
  std::string decoderOutputName_;
  std::vector<int> decoderInputShape_;
  std::vector<int> decoderOutputShape_;
  int conceptDim_ = 0;
};

class VideoHbdnnModel : public VideoModel {
 public:
  VideoHbdnnModel(VideoModelConfig cfg, int targetDim)
      : cfg_(std::move(cfg)), targetDim_(targetDim > 0 ? targetDim : kDefaultConceptDim),
        modelPath_(resolveBpuModelPath(phoenix::resolveConfig<std::string>("jepa.image.horizonModel", "", "JEPA_IMAGE_HORIZON_MODEL"), "encoder", imageWeightsDir(cfg_.id))),
        decoderPath_(resolveBpuModelPath(phoenix::resolveConfig<std::string>("jepa.image.horizonDecoderModel", "", "JEPA_IMAGE_HORIZON_DECODER_MODEL"), "decoder", imageWeightsDir(cfg_.id))) {
    if (modelPath_.empty()) {
      lastError_ = "local BPU image encoder model (model_encoder.bin) is not configured";
    } else if (!rdk_x5_bpu::available()) {
      lastError_ = "RDK X5 hbDNN runtime is unavailable";
    }
  }

  std::vector<float> encode(const std::vector<uint8_t> &, int, int, const std::string &) override {
    if (modelPath_.empty()) {
      lastError_ = "local BPU image encoder model (model_encoder.bin) is not configured";
      return {};
    }
    if (!rdk_x5_bpu::available()) {
      lastError_ = "RDK X5 hbDNN runtime is unavailable";
      return {};
    }
    lastError_ = "BPU image encoder preprocessing is not implemented in this build";
    return {};
  }

  std::vector<float> encodeContext(const std::vector<uint8_t> &imageBytes,
                                   int width,
                                   int height,
                                   const std::string &mimeType,
                                   const std::vector<bool> &) override {
    return encode(imageBytes, width, height, mimeType);
  }

  std::vector<float> encodeTarget(const std::vector<uint8_t> &imageBytes,
                                  int width,
                                  int height,
                                  const std::string &mimeType,
                                  const std::vector<int> &) override {
    return encode(imageBytes, width, height, mimeType);
  }

  std::vector<float> predictTarget(const std::vector<float> &contextRepr, const std::vector<int> &) override { return contextRepr; }

  float adapt(const std::vector<uint8_t> &, int, int, const std::string &, int, float) override {
    lastError_ = "BPU image model does not support in-process adaptation";
    return -1.0f;
  }

  std::vector<uint8_t> decode(const std::vector<float> &, const std::string &) override {
    if (decoderPath_.empty()) {
      lastError_ = "local BPU image decoder model (model_decoder.bin) is not configured";
      return {};
    }
    if (!rdk_x5_bpu::available()) {
      lastError_ = "RDK X5 hbDNN runtime is unavailable";
      return {};
    }
    lastError_ = "BPU image decoder rendering is not implemented in this build";
    return {};
  }

  nlohmann::json status() const override {
    std::error_code ec;
    bool modelReady = !modelPath_.empty() && std::filesystem::is_regular_file(modelPath_, ec);
    bool decoderReady = !decoderPath_.empty() && std::filesystem::is_regular_file(decoderPath_, ec);
    bool bpuReady = rdk_x5_bpu::available();
    std::string backend = kind_.empty() ? std::string("horizon-hbdnn") : kind_;
    if (!bpuReady || !modelReady) backend += "-unavailable";
    return nlohmann::json{{"id", cfg_.id}, {"arch", cfg_.arch}, {"backend", backend},
                          {"resolution", cfg_.resolution}, {"targetDim", targetDim_},
                          {"samples", samples_}, {"modelPath", modelPath_},
                          {"decoderPath", decoderPath_},
                          {"ready", modelReady && bpuReady}, {"decoderReady", decoderReady && bpuReady},
                          {"error", lastError_}};
  }

  const VideoModelConfig &config() const override { return cfg_; }

 private:
  VideoModelConfig cfg_;
  int targetDim_;
  std::string modelPath_;
  std::string decoderPath_;
  size_t samples_ = 0;
  mutable std::string lastError_;
};

static bool bpuModelAvailable(const std::string &variantId) {
  using phoenix::deployment::LocalBackendType;
  std::error_code ec;
  if (!rdk_x5_bpu::available()) return false;
  // Compiled BPU models are stored in the legacy ijepa tree; newer end-to-end
  // .bin exports may land in additive_jepa.  Accept either.
  auto ijepaDir = std::filesystem::path("runtime_store") / "models" / "ijepa" / variantId;
  auto additiveBpuDir = std::filesystem::path("runtime_store") / "models" / "additive_jepa" / variantId;
  return std::filesystem::is_regular_file(ijepaDir / "best.bin", ec) ||
         std::filesystem::is_regular_file(ijepaDir / "model_encoder.bin", ec) ||
         std::filesystem::is_regular_file(ijepaDir / "model_decoder.bin", ec) ||
         std::filesystem::is_regular_file(additiveBpuDir / "best.bin", ec) ||
         std::filesystem::is_regular_file(additiveBpuDir / "model_encoder.bin", ec) ||
         std::filesystem::is_regular_file(additiveBpuDir / "model_decoder.bin", ec);
}

static phoenix::deployment::LocalBackendType detectLocalBackend(const std::string &variantId) {
  using phoenix::deployment::LocalBackendType;
  std::error_code ec;
  if (bpuModelAvailable(variantId)) return LocalBackendType::Bpu;
  auto additiveDir = std::filesystem::path("runtime_store") / "models" / "additive_jepa" / variantId;
  if (std::filesystem::is_regular_file(additiveDir / "best.onnx", ec)) {
    return LocalBackendType::Cpu;
  }
  return LocalBackendType::Auto;
}

static phoenix::deployment::LocalBackendType chooseLocalBackend(
    const std::string &variantId, const phoenix::deployment::ModelDeploymentRecord &record) {
  auto backend = record.localBackend;
  if (backend == phoenix::deployment::LocalBackendType::Auto) {
    backend = detectLocalBackend(variantId);
  }
  return backend;
}

}  // namespace

std::unique_ptr<VideoModel> createVideoModel(
    const std::string &variantId, int targetDim, const std::string & /*backend*/) {
  const auto *v = findVideoModelVariant(variantId);
  if (!v) {
    VideoModelConfig unknownCfg;
    unknownCfg.id = variantId;
    unknownCfg.arch = "unknown";
    return std::make_unique<VideoUnavailableModel>(
        unknownCfg, targetDim, "unknown image variant: " + variantId);
  }

  const auto &deployment = phoenix::deployment::ModelDeploymentConfig::instance().vision();

  if (deployment.placement == phoenix::deployment::ModelPlacement::ServerClient) {
    return std::make_unique<VideoServerClientModel>(*v, targetDim);
  }

  if (deployment.placement == phoenix::deployment::ModelPlacement::Remote) {
    if (deployment.remote.url.empty()) {
      return std::make_unique<VideoUnavailableModel>(
          *v, targetDim, "remote image placement configured but remote.url is empty");
    }
    return std::make_unique<VideoRemoteModel>(*v, targetDim, deployment.remote);
  }

  auto backend = chooseLocalBackend(imageWeightsDir(v->id), deployment);
  if (backend == phoenix::deployment::LocalBackendType::Bpu) {
#if PHOENIX_EDGE_IMAGE_ENABLED
    auto hbdnn = std::make_unique<VideoHbdnnModel>(*v, targetDim);
    if (hbdnn->status().value("ready", false)) return hbdnn;
#endif
    // Graceful fallback: if the BPU runtime or compiled .bin is unavailable,
    // use the local ONNX model so the deployment is still feasible.
    std::error_code ec;
    auto additiveDir = std::filesystem::path("runtime_store") / "models" / "additive_jepa" / imageWeightsDir(v->id);
    if (std::filesystem::is_regular_file(additiveDir / "best.onnx", ec)) {
      return std::make_unique<VideoLocalOnnxModel>(*v, targetDim, false);
    }
    return std::make_unique<VideoUnavailableModel>(
        *v, targetDim, "local BPU image backend is unavailable and no ONNX fallback found");
  }

  if (backend == phoenix::deployment::LocalBackendType::Cpu ||
      backend == phoenix::deployment::LocalBackendType::Gpu) {
    bool gpu = backend == phoenix::deployment::LocalBackendType::Gpu;
    return std::make_unique<VideoLocalOnnxModel>(*v, targetDim, gpu);
  }

  if (backend == phoenix::deployment::LocalBackendType::Js) {
    return std::make_unique<VideoFallbackModel>(*v, targetDim);
  }

  // Auto-detected nothing: use fallback so the bridge still returns a clear error.
  return std::make_unique<VideoFallbackModel>(*v, targetDim);
}

std::unique_ptr<VideoEncoder> createVideoEncoder(
    const std::string &variantId, int targetDim, const std::string &backend) {
  auto model = createVideoModel(variantId, targetDim, backend);
  if (model) model->kind_ = "video-encoder";
  return model;
}

std::unique_ptr<VideoDecoder> createVideoDecoder(
    const std::string &variantId, int targetDim, const std::string &backend) {
  auto model = createVideoModel(variantId, targetDim, backend);
  if (model) model->kind_ = "video-decoder";
  return model;
}

}  // namespace io
}  // namespace phoenix
