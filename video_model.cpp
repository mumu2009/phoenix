/* jepa_v2_image_world_model.cpp - JEPA-v2 image world model factory and fallbacks
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version. */

#include "jepa_v2_image_world_model.hpp"
#include "model_deployment.hpp"
#include "phoenix_config.hpp"
#include "rdk_x5_bpu.hpp"
#include "semantic_unit.hpp"

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
  if (variantId == "vision_encoder") return "vision_encoder";
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

class JepaV2ImageFallbackModel : public JepaV2ImageWorldModel {
 public:
  JepaV2ImageFallbackModel(JepaV2ImageWorldModelConfig cfg, int targetDim)
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
    return nlohmann::json{{"id", cfg_.id}, {"arch", cfg_.arch}, {"backend", "fallback"},
                          {"resolution", cfg_.resolution}, {"targetDim", targetDim_},
                          {"samples", samples_}, {"ready", false}, {"error", lastError_}};
  }

  const JepaV2ImageWorldModelConfig &config() const override { return cfg_; }

 private:
  JepaV2ImageWorldModelConfig cfg_;
  int targetDim_;
  size_t samples_ = 0;
  mutable std::string lastError_;
};

class JepaV2ImageUnavailableModel : public JepaV2ImageWorldModel {
 public:
  JepaV2ImageUnavailableModel(JepaV2ImageWorldModelConfig cfg, int targetDim, std::string reason)
      : cfg_(std::move(cfg)), targetDim_(targetDim > 0 ? targetDim : kDefaultConceptDim), reason_(std::move(reason)) {}

  std::vector<float> encode(const std::vector<uint8_t> &, int, int, const std::string &) override { return {}; }
  std::vector<float> encodeContext(const std::vector<uint8_t> &, int, int, const std::string &, const std::vector<bool> &) override { return {}; }
  std::vector<float> encodeTarget(const std::vector<uint8_t> &, int, int, const std::string &, const std::vector<int> &) override { return {}; }
  std::vector<float> predictTarget(const std::vector<float> &, const std::vector<int> &) override { return {}; }
  float adapt(const std::vector<uint8_t> &, int, int, const std::string &, int, float) override { return -1.0f; }
  std::vector<uint8_t> decode(const std::vector<float> &, const std::string &) override { return {}; }

  nlohmann::json status() const override {
    return nlohmann::json{{"id", cfg_.id}, {"arch", cfg_.arch}, {"backend", "unavailable"},
                          {"resolution", cfg_.resolution}, {"targetDim", targetDim_},
                          {"samples", samples_}, {"ready", false}, {"error", reason_}};
  }

  const JepaV2ImageWorldModelConfig &config() const override { return cfg_; }

 private:
  JepaV2ImageWorldModelConfig cfg_;
  int targetDim_;
  size_t samples_ = 0;
  std::string reason_;
};

class JepaV2ImageServerClientModel : public JepaV2ImageWorldModel {
 public:
  JepaV2ImageServerClientModel(JepaV2ImageWorldModelConfig cfg, int targetDim)
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
    return nlohmann::json{{"id", cfg_.id}, {"arch", cfg_.arch}, {"backend", "server-client"},
                          {"resolution", cfg_.resolution}, {"targetDim", targetDim_},
                          {"samples", samples_}, {"ready", false},
                          {"error", "server-client: expects client concept vectors"}};
  }
  const JepaV2ImageWorldModelConfig &config() const override { return cfg_; }

 private:
  JepaV2ImageWorldModelConfig cfg_;
  int targetDim_;
  size_t samples_ = 0;
  mutable std::string lastError_;
};

class JepaV2ImageRemoteModel : public JepaV2ImageWorldModel {
 public:
  JepaV2ImageRemoteModel(JepaV2ImageWorldModelConfig cfg, int targetDim,
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
    return nlohmann::json{{"id", cfg_.id}, {"arch", cfg_.arch}, {"backend", "remote"},
                          {"url", endpoint_.url}, {"method", endpoint_.method},
                          {"resolution", cfg_.resolution}, {"targetDim", targetDim_},
                          {"samples", samples_}, {"ready", !endpoint_.url.empty()}, {"error", lastError_}};
  }

  const JepaV2ImageWorldModelConfig &config() const override { return cfg_; }

 private:
  static std::vector<float> parseFloatVectorFromJson(const nlohmann::json &j) {
    std::vector<float> out;
    if (!j.is_array()) return out;
    for (const auto &v : j) {
      if (v.is_number()) out.push_back(v.get<double>());
    }
    return out;
  }

  JepaV2ImageWorldModelConfig cfg_;
  int targetDim_;
  phoenix::deployment::RemoteEndpoint endpoint_;
  size_t samples_ = 0;
  mutable std::string lastError_;
};

class JepaV2ImageLocalOnnxModel : public JepaV2ImageWorldModel {
 public:
  JepaV2ImageLocalOnnxModel(JepaV2ImageWorldModelConfig cfg, int targetDim, bool gpu)
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
                            const std::string & /*mimeType*/) override {
    if (modelPath_.empty()) {
      lastError_ = "local ONNX image encoder model (best.onnx) is missing";
      return {};
    }
    (void)width;
    (void)height;
    (void)imageBytes;
    lastError_ = "local ONNX image encoder preprocessing is not implemented in this build";
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

  std::vector<float> predictTarget(const std::vector<float> &contextRepr, const std::vector<int> &) override {
    return std::vector<float>(contextRepr.begin(), contextRepr.end());
  }

  float adapt(const std::vector<uint8_t> &, int, int, const std::string &, int, float) override {
    lastError_ = "local ONNX image model does not support adaptation";
    return -1.0f;
  }

  std::vector<uint8_t> decode(const std::vector<float> &conceptVector, const std::string & /*mimeType*/) override {
    if (decoderPath_.empty()) {
      lastError_ = "local ONNX image decoder model (best.onnx) is not configured";
      return {};
    }
    (void)conceptVector;
    lastError_ = "local ONNX image decoder rendering is not implemented in this build";
    return {};
  }

  nlohmann::json status() const override {
    std::error_code ec;
    bool modelReady = !modelPath_.empty() && std::filesystem::is_regular_file(modelPath_, ec);
    bool decoderReady = !decoderPath_.empty() && std::filesystem::is_regular_file(decoderPath_, ec);
    return nlohmann::json{{"id", cfg_.id}, {"arch", cfg_.arch}, {"backend", gpu_ ? "local-gpu" : "local-onnx"},
                          {"resolution", cfg_.resolution}, {"targetDim", targetDim_},
                          {"conceptDim", conceptDim_}, {"samples", samples_},
                          {"modelPath", modelPath_}, {"decoderPath", decoderPath_},
                          {"ready", modelReady}, {"decoderReady", decoderReady}, {"error", lastError_}};
  }

  const JepaV2ImageWorldModelConfig &config() const override { return cfg_; }

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

  JepaV2ImageWorldModelConfig cfg_;
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

class JepaV2ImageHbdnnModel : public JepaV2ImageWorldModel {
 public:
  JepaV2ImageHbdnnModel(JepaV2ImageWorldModelConfig cfg, int targetDim)
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
    std::string backend = "horizon-hbdnn";
    if (!bpuReady || !modelReady) backend += "-unavailable";
    return nlohmann::json{{"id", cfg_.id}, {"arch", cfg_.arch}, {"backend", backend},
                          {"resolution", cfg_.resolution}, {"targetDim", targetDim_},
                          {"samples", samples_}, {"modelPath", modelPath_},
                          {"decoderPath", decoderPath_},
                          {"ready", modelReady && bpuReady}, {"decoderReady", decoderReady && bpuReady},
                          {"error", lastError_}};
  }

  const JepaV2ImageWorldModelConfig &config() const override { return cfg_; }

 private:
  JepaV2ImageWorldModelConfig cfg_;
  int targetDim_;
  std::string modelPath_;
  std::string decoderPath_;
  size_t samples_ = 0;
  mutable std::string lastError_;
};

static phoenix::deployment::LocalBackendType detectLocalBackend(const std::string &variantId) {
  using phoenix::deployment::LocalBackendType;
  std::error_code ec;
  auto bpuDir = std::filesystem::path("runtime_store") / "models" / "ijepa" / variantId;
  if (std::filesystem::is_regular_file(bpuDir / "model_encoder.bin", ec) ||
      std::filesystem::is_regular_file(bpuDir / "model_decoder.bin", ec)) {
    return LocalBackendType::Bpu;
  }
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

std::unique_ptr<JepaV2ImageWorldModel> createJepaV2ImageWorldModel(
    const std::string &variantId, int targetDim, const std::string & /*backend*/) {
  const auto *v = findJepaV2ImageVariant(variantId);
  if (!v) {
    JepaV2ImageWorldModelConfig unknownCfg;
    unknownCfg.id = variantId;
    unknownCfg.arch = "unknown";
    return std::make_unique<JepaV2ImageUnavailableModel>(
        unknownCfg, targetDim, "unknown image variant: " + variantId);
  }

  const auto &deployment = phoenix::deployment::ModelDeploymentConfig::instance().vision();

  if (deployment.placement == phoenix::deployment::ModelPlacement::ServerClient) {
    return std::make_unique<JepaV2ImageServerClientModel>(*v, targetDim);
  }

  if (deployment.placement == phoenix::deployment::ModelPlacement::Remote) {
    if (deployment.remote.url.empty()) {
      return std::make_unique<JepaV2ImageUnavailableModel>(
          *v, targetDim, "remote image placement configured but remote.url is empty");
    }
    return std::make_unique<JepaV2ImageRemoteModel>(*v, targetDim, deployment.remote);
  }

  auto backend = chooseLocalBackend(imageWeightsDir(v->id), deployment);
  if (backend == phoenix::deployment::LocalBackendType::Bpu) {
#if PHOENIX_EDGE_IMAGE_ENABLED
    auto hbdnn = std::make_unique<JepaV2ImageHbdnnModel>(*v, targetDim);
    return hbdnn;
#else
    return std::make_unique<JepaV2ImageUnavailableModel>(
        *v, targetDim, "local BPU image backend is disabled at compile time");
#endif
  }

  if (backend == phoenix::deployment::LocalBackendType::Cpu ||
      backend == phoenix::deployment::LocalBackendType::Gpu) {
    bool gpu = backend == phoenix::deployment::LocalBackendType::Gpu;
    return std::make_unique<JepaV2ImageLocalOnnxModel>(*v, targetDim, gpu);
  }

  if (backend == phoenix::deployment::LocalBackendType::Js) {
    return std::make_unique<JepaV2ImageFallbackModel>(*v, targetDim);
  }

  // Auto-detected nothing: use fallback so the bridge still returns a clear error.
  return std::make_unique<JepaV2ImageFallbackModel>(*v, targetDim);
}

}  // namespace io
}  // namespace phoenix
