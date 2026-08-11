/* jepa_v2_image_world_model.cpp - Legacy JEPA image interface backed by LLaVA
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   The implementation is now a thin wrapper over the new external multimodal
   Python service (MultimodalImageWorldModel).  It is kept only so that
   existing call sites such as frontend_server.cpp continue to compile while
   the v7.x+1 enc/dec rework is completed.  New code should use
   MultimodalImageWorldModel directly. */

#include "jepa_v2_image_world_model.hpp"
#include "multimodal_world_model.hpp"
#include "phoenix_config.hpp"

namespace phoenix {
namespace io {

class MultimodalImageJepaWrapper : public JepaV2ImageWorldModel {
 public:
  explicit MultimodalImageJepaWrapper(const std::string &variantId,
                                      int targetDim,
                                      const std::string &backend) {
    (void)backend;
    cfg_ = findJepaV2ImageVariant(variantId) ? *findJepaV2ImageVariant(variantId)
                                             : jepaV2ImageOfficialVariants().front();
    if (!variantId.empty()) cfg_.id = variantId;
    (void)targetDim;
    MultimodalEncDecConfig mcfg;
    mcfg.baseUrl = phoenix::resolveConfigAsString("multimodal.encDecBaseUrl", "http://127.0.0.1:8085");
    mcfg.timeoutMs = phoenix::resolveConfig<int>("multimodal.encDecTimeoutMs", 120000);
    mcfg.imageEncoderModel = cfg_.id;
    model_ = std::make_unique<MultimodalImageWorldModel>(mcfg);
  }

  std::vector<float> encode(const std::vector<uint8_t> &imageBytes,
                            int width,
                            int height,
                            const std::string &mimeType) override {
    auto result = model_->encode(imageBytes, width, height, mimeType);
    if (!result.error.empty()) lastError_ = result.error;
    return std::move(result.meanUnitQuery);
  }

  std::vector<float> encodeContext(const std::vector<uint8_t> &,
                                   int,
                                   int,
                                   const std::string &,
                                   const std::vector<bool> &) override {
    return {};
  }

  std::vector<float> encodeTarget(const std::vector<uint8_t> &,
                                  int,
                                  int,
                                  const std::string &,
                                  const std::vector<int> &) override {
    return {};
  }

  std::vector<float> predictTarget(const std::vector<float> &,
                                   const std::vector<int> &) override {
    return {};
  }

  float adapt(const std::vector<uint8_t> &,
              int,
              int,
              const std::string &,
              int,
              float) override {
    return -1.0f;
  }

  std::vector<uint8_t> decode(const std::vector<float> &conceptVector,
                              const std::string &mimeType) override {
    std::vector<std::vector<float>> emptySeq;
    auto result = model_->decode(conceptVector, emptySeq, mimeType, 224, 224);
    if (!result.error.empty()) lastError_ = result.error;
    return std::move(result.payload);
  }

  nlohmann::json status() const override {
    nlohmann::json j = model_->status();
    j["jepaV2Wrapper"] = true;
    if (!lastError_.empty()) j["lastError"] = lastError_;
    return j;
  }

  const JepaV2ImageWorldModelConfig &config() const override { return cfg_; }

 private:
  JepaV2ImageWorldModelConfig cfg_;
  std::unique_ptr<MultimodalImageWorldModel> model_;
  mutable std::string lastError_;
};

std::unique_ptr<JepaV2ImageWorldModel> createJepaV2ImageWorldModel(
    const std::string &variantId,
    int targetDim,
    const std::string &backend) {
  return std::make_unique<MultimodalImageJepaWrapper>(variantId, targetDim, backend);
}

}  // namespace io
}  // namespace phoenix
