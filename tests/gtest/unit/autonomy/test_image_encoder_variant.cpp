/* Regression tests for the 2026-11 image-encoder adapter wiring.
 *
 * The image->concept encoder variant is operator-selected
 * (jepa.image.variant / JEPA_IMAGE_VARIANT) and must FAIL OPEN to the shipped
 * 128-d "video-encoder": an unknown registry id, a registered variant whose
 * ONNX is not installed, or a not-ready model must still produce the shipped
 * encoder instead of empty concept vectors.
 *
 * The new variant (frozen timm ResNet50 + linear 2048->128 adapter, exported as
 * one ONNX) is registered as "vision_resnet50_a1_in1k_adapter" and lives under
 * runtime_store/models/additive_jepa/vision_resnet50_a1_in1k_adapter/.
 *
 * The real-inference test is SKIPPED when the adapter ONNX is not installed or
 * when the local ONNX runner environment (Python314 + onnxruntime) is absent,
 * so CI stays green without large runtime artifacts.
 */
#include "model_deployment.hpp"
#include "video_model.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace {

constexpr const char *kShippedVariant = "video-encoder";
constexpr const char *kAdapterVariant = "vision_resnet50_a1_in1k_adapter";

/* All-local topology so createVideoModel() cannot pick a remote/server-client
 * model left behind by another suite. */
void forceLocalImageDeployment() {
  auto &cfg = phoenix::deployment::ModelDeploymentConfig::instance();
  cfg.reset();
  nlohmann::json j = cfg.toJson();
  j["vision"]["placement"] = "local";
  j["vision"]["localBackend"] = "cpu";
  cfg.fromJson(j);
}

std::filesystem::path adapterOnnxPath() {
  return std::filesystem::path("runtime_store") / "models" / "additive_jepa" /
         kAdapterVariant / "best.onnx";
}

/* Deterministic 224x224x3 BGR payload (raw, no image codec needed). */
std::vector<uint8_t> probeImageBytes(int width = 224, int height = 224) {
  std::vector<uint8_t> bytes(static_cast<size_t>(width) * height * 3);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t i = (static_cast<size_t>(y) * width + x) * 3;
      bytes[i + 0] = static_cast<uint8_t>((x * 255) / (width - 1));
      bytes[i + 1] = static_cast<uint8_t>((y * 255) / (height - 1));
      bytes[i + 2] = static_cast<uint8_t>(((x + y) * 255) / (width + height - 2));
    }
  }
  return bytes;
}

}  // namespace

TEST(ImageEncoderVariantTest, AdapterVariantIsRegistered) {
  const auto *cfg = phoenix::io::findVideoModelVariant(kAdapterVariant);
  ASSERT_NE(cfg, nullptr) << "createVideoModel() would reject the adapter variant";
  EXPECT_EQ(cfg->id, kAdapterVariant);
  EXPECT_EQ(cfg->weightsFile, "best.onnx");
  EXPECT_EQ(cfg->localWeightsDir, kAdapterVariant);
  EXPECT_EQ(cfg->arch, "resnet50_adapter");
  EXPECT_EQ(cfg->resolution, 224);
}

TEST(ImageEncoderVariantTest, UnknownVariantFailsOpenToShippedEncoder) {
  forceLocalImageDeployment();
  auto choice = phoenix::io::createImageEncoderWithFallback(
      "no_such_image_variant", 128, "cpu");
  EXPECT_TRUE(choice.fellBack);
  EXPECT_EQ(choice.requestedVariant, "no_such_image_variant");
  EXPECT_EQ(choice.variant, kShippedVariant);
  EXPECT_NE(choice.reason.find("unknown-variant-registry"), std::string::npos)
      << "reason was: " << choice.reason;
  EXPECT_NE(choice.model, nullptr);
}

TEST(ImageEncoderVariantTest, RegisteredButMissingOnnxFailsOpen) {
  forceLocalImageDeployment();
  /* Registered official variant with no local ONNX installed -> not usable. */
  auto choice =
      phoenix::io::createImageEncoderWithFallback("ijepa_vith14_1k", 128, "cpu");
  EXPECT_TRUE(choice.fellBack);
  EXPECT_EQ(choice.variant, kShippedVariant);
  EXPECT_NE(choice.reason.find("onnx-missing"), std::string::npos)
      << "reason was: " << choice.reason;
  EXPECT_NE(choice.model, nullptr);
}

TEST(ImageEncoderVariantTest, ShippedVariantDoesNotFallBack) {
  forceLocalImageDeployment();
  auto choice =
      phoenix::io::createImageEncoderWithFallback(kShippedVariant, 128, "cpu");
  EXPECT_FALSE(choice.fellBack);
  EXPECT_TRUE(choice.reason.empty());
  EXPECT_EQ(choice.variant, kShippedVariant);
  EXPECT_NE(choice.model, nullptr);
}

TEST(ImageEncoderVariantTest, EmptyRequestUsesShippedVariant) {
  forceLocalImageDeployment();
  auto choice = phoenix::io::createImageEncoderWithFallback("", 128, "cpu");
  EXPECT_FALSE(choice.fellBack);
  EXPECT_EQ(choice.variant, kShippedVariant);
}

TEST(ImageEncoderVariantTest, AdapterEncoderEmits128dConcept) {
  if (!std::filesystem::is_regular_file(adapterOnnxPath())) {
    GTEST_SKIP() << "adapter ONNX not installed at " << adapterOnnxPath();
  }
  forceLocalImageDeployment();
  auto choice = phoenix::io::createImageEncoderWithFallback(
      kAdapterVariant, 128, "cpu");
  ASSERT_NE(choice.model, nullptr);
  if (!choice.model->status().value("ready", false)) {
    GTEST_SKIP() << "adapter ONNX present but model not ready (artifact issue)";
  }
  EXPECT_FALSE(choice.fellBack) << "reason: " << choice.reason;
  EXPECT_EQ(choice.variant, kAdapterVariant);

  auto bytes = probeImageBytes();
  const auto conceptVec = choice.model->encode(bytes, 224, 224, "application/x-bgr");
  if (conceptVec.empty()) {
    const std::string err =
        choice.model->status().value("error", std::string());
    if (err.find("python") != std::string::npos ||
        err.find("ONNX runner") != std::string::npos ||
        err.find("onnxruntime") != std::string::npos) {
      GTEST_SKIP() << "local ONNX runner unavailable: " << err;
    }
    FAIL() << "adapter encode returned empty: " << err;
  }
  EXPECT_EQ(conceptVec.size(), 128u)
      << "adapter must emit the native 128-d concept (no post-head)";
  // The native conceptDim is 128, so targetDim==128 means no projection was
  // applied (the legacy 128->128 head must not be used after the adapter).
  EXPECT_EQ(choice.model->status().value("conceptDim", 0), 128);
  EXPECT_NE(choice.model->status().value("decoderReady", true), true)
      << "decoder must not aliase the encoder ONNX for encode-only variants";
}
