#pragma once

#include "runtime_opt/hooks.hpp"

#include <algorithm>
#include <string>

namespace phoenix {
namespace runtime_opt {

struct ValenceBiasConfig {
  bool enabled = false;
  double maxTempDelta = 0.15;
  double maxTopPDelta = 0.05;
};

struct SampleBias {
  double temperatureDelta = 0.0;
  double topPDelta = 0.0;
  double suggestedTemperature = 0.35;
  double suggestedTopP = 0.9;
  std::string reason = "off";
};

/* Read-only PAD hint. Does not train vocab weights. */
inline SampleBias sampleBiasFromPad(float valence, float arousal, float dominance,
                                    double baseTemp, double baseTopP,
                                    const ValenceBiasConfig &cfg) {
  SampleBias b;
  b.suggestedTemperature = baseTemp;
  b.suggestedTopP = baseTopP;
  if (!cfg.enabled) {
    b.reason = "valence_bias_off";
    return b;
  }
  const auto va = wireValenceArousal(valence, arousal, "emotion_existing");
  (void)dominance;
  double td = static_cast<double>(va.arousal) * cfg.maxTempDelta -
              static_cast<double>(va.valence) * (cfg.maxTempDelta * 0.35);
  if (td > cfg.maxTempDelta)
    td = cfg.maxTempDelta;
  if (td < -cfg.maxTempDelta)
    td = -cfg.maxTempDelta;
  double pd = static_cast<double>(va.arousal) * cfg.maxTopPDelta;
  if (pd > cfg.maxTopPDelta)
    pd = cfg.maxTopPDelta;
  if (pd < -cfg.maxTopPDelta)
    pd = -cfg.maxTopPDelta;
  b.temperatureDelta = td;
  b.topPDelta = pd;
  b.suggestedTemperature = std::max(0.05, std::min(1.2, baseTemp + td));
  b.suggestedTopP = std::max(0.5, std::min(1.0, baseTopP + pd));
  b.reason = "pad_sample_hint";
  return b;
}

} // namespace runtime_opt
} // namespace phoenix
