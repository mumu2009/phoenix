/* inference_backend_router.hpp - route legacy transformer_* calls to llama/tinyllama

   Self-built transformer is deprecated.  NativeTransformer backend resolves to
   llama-server split (/phx/*) or TinyLlama via config — same call sites, new
   implementation ("reflection" style dispatch). */
#pragma once

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace phoenix {
namespace inference {

/** Project a unit query vector to llama n_embd (repeat/pad). */
inline std::vector<float> projectUnitQuery(const std::vector<float> &src,
                                           int targetDim) {
  if (targetDim <= 0) return {};
  std::vector<float> out(static_cast<size_t>(targetDim), 0.f);
  if (src.empty()) return out;
  for (int i = 0; i < targetDim; ++i)
    out[static_cast<size_t>(i)] = src[static_cast<size_t>(i % src.size())];
  return out;
}

inline nlohmann::json graphEmbeddingsToPrefixHidden(
    const std::vector<std::vector<float>> &embs, int targetDim) {
  nlohmann::json arr = nlohmann::json::array();
  for (const auto &e : embs) {
    if (targetDim > 0 && static_cast<int>(e.size()) == targetDim)
      arr.push_back(e);
    else
      arr.push_back(projectUnitQuery(e, targetDim));
  }
  return arr;
}

enum class LegacyBackend {
  DeprecatedNativeTransformer,
  LlamaSplit,
};

inline LegacyBackend resolveLegacyBackend(const std::string &transformerBackend,
                                          bool transformerEnabled,
                                          bool /*tinyllamaEnabled*/) {
  (void)transformerEnabled;
  if (transformerBackend == "native" || transformerBackend == "off")
    return LegacyBackend::DeprecatedNativeTransformer;
  return LegacyBackend::LlamaSplit;
}

}  // namespace inference
}  // namespace phoenix
