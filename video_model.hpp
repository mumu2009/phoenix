/* video_model.hpp - Semantic interface for image world models
 */

#pragma once

#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace phoenix {
namespace io {

/**
 * @brief Description of an official I-JEPA / video world-model variant.
 *
 * The four shipped variants below correspond to the official Meta I-JEPA
 * checkpoints (see https://github.com/facebookresearch/ijepa).  The default
 * is the lowest-compute variant: ViT-H/14 trained on ImageNet-1K at 224x224.
 */
struct VideoModelConfig {
  std::string id;            /**< internal variant id, e.g. "ijepa_vith14_1k" */
  std::string arch;          /**< architecture string, e.g. "vit_h14" */
  std::string repo;          /**< HuggingFace repo or checkpoint URL */
  int patchSize = 14;        /**< spatial patch size */
  int resolution = 224;      /**< input resolution */
  long long params = 0;      /**< approximate parameter count */
  std::string dataset;       /**< pretraining dataset, e.g. "IN1K" */
  std::string weightsFile = "model.safetensors"; /**< primary checkpoint filename inside localWeightsDir */
  std::string localWeightsDir; /**< directory under runtime_store/models/ijepa/; empty means "<id>/" */
};

/**
 * @brief Semantic interface for a video world model.
 *
 * The interface is organised around the JEPA joint-embedding predictive
 * architecture.  Implementations must be able to:
 *   1. split an image into a context and one or more target patch blocks,
 *   2. encode context patches and target patches into latent representations,
 *   3. predict target representations from the context representation,
 *   4. map the pooled representation into a semantic concept vector.
 *
 * This is not just a format wrapper: it exposes the model's internal
 * representation contract so downstream modules can reason about concepts.
 */
class VideoModel {
 public:
  virtual ~VideoModel() = default;

  /**
   * @brief Encode an image into a semantic concept vector.
   *
   * This is the high-level entry point used by MixedModalConceptBridge.
   * It should internally run the JEPA context encoder, pool patch
   * representations, and project to the requested concept dimension.
   */
  virtual std::vector<float> encode(const std::vector<uint8_t> &imageBytes,
                                    int width,
                                    int height,
                                    const std::string &mimeType) = 0;

  /**
   * @brief Encode the visible context patches of an image.
   *
   * @param mask  flattened bool mask, true = visible patch, false = masked.
   *              If empty the full image is used as context.
   * @return context representation vector in embedding space.
   */
  virtual std::vector<float> encodeContext(const std::vector<uint8_t> &imageBytes,
                                           int width,
                                           int height,
                                           const std::string &mimeType,
                                           const std::vector<bool> &mask) = 0;

  /**
   * @brief Encode a set of target patch blocks.
   *
   * In JEPA this is the teacher branch used to provide the prediction target.
   *
   * @param blockIndices list of patch indices forming each target block.
   * @return target representation vector in embedding space.
   */
  virtual std::vector<float> encodeTarget(const std::vector<uint8_t> &imageBytes,
                                          int width,
                                          int height,
                                          const std::string &mimeType,
                                          const std::vector<int> &blockIndices) = 0;

  /**
   * @brief Predict a target representation from a context representation.
   *
   * The predictor is the small JEPA world-model head that models spatial
   * relationships between visible and masked regions.
   *
   * @param contextRepr  context embedding from encodeContext().
   * @param targetPos    positional tokens of the target block to predict.
   * @return predicted target embedding.
   */
  virtual std::vector<float> predictTarget(const std::vector<float> &contextRepr,
                                           const std::vector<int> &targetPos) = 0;

  /**
   * @brief Unsupervised / self-supervised domain adaptation step.
   *
   * Runs one JEPA prediction step on the provided image: sample target blocks,
   * predict them from the context, and update the predictor/encoder weights.
   *
   * @return approximate loss value (>=0), or -1 on failure.
   */
  virtual float adapt(const std::vector<uint8_t> &imageBytes,
                      int width,
                      int height,
                      const std::string &mimeType,
                      int steps = 1,
                      float lr = 1e-4f) = 0;

  /**
   * @brief Render a concept vector back to an image payload.
   *
   * A world model is not a generative decoder, so implementations may emit a
   * concept-laden payload with a downstream-decoder flag or a low-level
   * deterministic visualization.
   */
  virtual std::vector<uint8_t> decode(const std::vector<float> &conceptVector,
                                      const std::string &mimeType = "image/png") = 0;

  /** @brief Runtime status (variant, loaded backend, sample count). */
  virtual nlohmann::json status() const = 0;

  /** @brief Configuration of this model instance. */
  virtual const VideoModelConfig &config() const = 0;

  /** @brief Optional per-instance role used for the "backend" status field. */
  mutable std::string kind_;
};

/**
 * @brief List the four official I-JEPA variants.
 *
 * Order is from lowest to highest compute / parameter count.
 */
inline std::vector<VideoModelConfig> videoOfficialVariants() {
  return {
      {"resnet18_224",     "resnet18",
       "timm/resnet18.a1_in1k",  16, 224, 11689512LL, "IN1K"},
      {"video-encoder",    "additive_resnet18",
       "runtime_store/models/additive_jepa/vision_encoder",  16, 224, 0, "additive", "best.onnx", "vision_encoder"},
      /* 2026-11-18 io-head: frozen timm ResNet50 + linear(2048->128) adapter.
         Self-contained ONNX (backbone+adapter inside), outputs 128-d directly;
         installed under runtime_store/models/additive_jepa/<dir>/best.onnx so
         resolveOnnxModelPath()/detectLocalBackend() find it without changes.
         Selected only when jepa.image.variant/JEPA_IMAGE_VARIANT names it; the
         default stays "video-encoder". */
      {"vision_resnet50_a1_in1k_adapter", "resnet50_adapter",
       "runtime_store/models/additive_jepa/vision_resnet50_a1_in1k_adapter", 16, 224, 0, "adapter", "best.onnx", "vision_resnet50_a1_in1k_adapter"},
      {"ijepa_vith14_1k",  "vit_h14",
       "facebook/ijepa_vith14_1k",  14, 224, 632000000LL, "IN1K"},
      {"ijepa_vith16_448", "vit_h16",
       "facebook/ijepa_vith16_1k", 16, 448, 632000000LL, "IN1K"},
      {"ijepa_vith14_22k", "vit_h14",
       "facebook/ijepa_vith14_22k", 14, 224, 632000000LL, "IN22K"},
      {"ijepa_vitg16_22k", "vit_g16",
       "facebook/ijepa_vitg16_22k", 16, 224, 1000000000LL, "IN22K"}};
}

/**
 * @brief Factory: create an image world model instance.
 *
 * @param variantId   one of the ids returned by videoOfficialVariants().
 * @param targetDim   desired output concept dimension.
 * @param backend     retained for API compatibility; production always uses Horizon hbDNN.
 * @return a concrete implementation. Missing BPU runtime or a compiled model is reported
 *         through status() and encode() returns an empty vector.
 */
std::unique_ptr<VideoModel> createVideoModel(
    const std::string &variantId = "ijepa_vith14_1k",
    int targetDim = 0,
    const std::string &backend = "auto");

using VideoEncoder = VideoModel;
using VideoDecoder = VideoModel;

std::unique_ptr<VideoEncoder> createVideoEncoder(
    const std::string &variantId = "ijepa_vith14_1k",
    int targetDim = 0,
    const std::string &backend = "auto");

std::unique_ptr<VideoDecoder> createVideoDecoder(
    const std::string &variantId = "ijepa_vith14_1k",
    int targetDim = 0,
    const std::string &backend = "auto");

/**
 * @brief Outcome of resolving the configured image encoder variant.
 *
 * The image encoder variant is operator-selected (jepa.image.variant /
 * JEPA_IMAGE_VARIANT) but the image->concept path must fail OPEN to the
 * shipped 128-d encoder: an unknown registry id, a missing ONNX or a model
 * that is not ready must never turn image packets into empty concept vectors.
 * Every decision is logged with the "[io-head]" prefix and carries the
 * requested variant, the reason and the selected variant so the behaviour is
 * observable in the gateway log (2026-11 image adapter wiring).
 */
struct ImageEncoderChoice {
  std::string requestedVariant; /**< configured value; empty = shipped default */
  std::string variant;          /**< variant actually selected */
  bool fellBack = false;        /**< true if requestedVariant was not usable */
  std::string reason;           /**< why the fallback happened (empty if none) */
  std::unique_ptr<VideoModel> model;
};

/**
 * @brief Create the image encoder for `requestedVariant`, failing open to the
 *        shipped "video-encoder" (additive 128-d) encoder.
 *
 * Logs (stdout):
 *   [io-head] image encoder variant=<v> conceptDim=<n> targetDim=<n> backend=<b>
 *   [io-head] image encoder fallback requested=<v> reason=<r> selected=<v>
 * `reason` is one of: unknown-variant-registry, onnx-missing:<detail>,
 * status-not-ready:<detail>.
 */
ImageEncoderChoice createImageEncoderWithFallback(
    const std::string &requestedVariant, int targetDim = 0,
    const std::string &backend = "auto");

/**
 * @brief Find an official variant config by id.
 */
inline const VideoModelConfig *findVideoModelVariant(
    const std::string &id) {
  static const std::vector<VideoModelConfig> kVariants =
      videoOfficialVariants();
  for (const auto &v : kVariants) {
    if (v.id == id) return &v;
  }
  return nullptr;
}

/**
 * @brief Expected local directory for downloaded weights of a variant.
 *
 * This matches the layout created by tools/download_ijepa_models.py:
 * runtime_store/models/ijepa/<id>/.
 */
inline std::string videoModelLocalWeightsDir(const VideoModelConfig &cfg) {
  return cfg.localWeightsDir.empty() ? cfg.id : cfg.localWeightsDir;
}

/**
 * @brief Expected full local checkpoint path for a variant.
 *
 * Example: "runtime_store/models/ijepa/ijepa_vith14_1k/model.safetensors".
 */
inline std::string videoModelExpectedWeightsPath(const VideoModelConfig &cfg) {
  return std::string("runtime_store/models/ijepa/") +
         videoModelLocalWeightsDir(cfg) + "/" + cfg.weightsFile;
}

}  // namespace io
}  // namespace phoenix
