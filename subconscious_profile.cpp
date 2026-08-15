/* subconscious_profile.cpp - Subconscious profile implementation
   Copyright (C) 2026 079 Project */

#include "subconscious_profile.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace phoenix {
namespace subconscious {

namespace {

std::unordered_map<primal::SensationType, primal::SensationTuning>
parseTuning(const std::map<std::string, primal::SensationTuning> &m) {
  std::unordered_map<primal::SensationType, primal::SensationTuning> out;
  for (const auto &kv : m) {
    const primal::SensationType t = primal::PrimalSensation::stringToType(kv.first);
    if (t != primal::SensationType::Unknown) {
      out[t] = kv.second;
    }
  }
  return out;
}

}  // namespace

nlohmann::json SubconsciousProfile::toJson() const {
  nlohmann::json j;
  j["baselineValence"] = baselineValence;
  j["baselineArousal"] = baselineArousal;
  j["baselineDominance"] = baselineDominance;
  j["temperamentStrength"] = temperamentStrength;
  j["riskAversion"] = riskAversion;
  j["anticipatoryGain"] = anticipatoryGain;
  nlohmann::json tuning = nlohmann::json::object();
  for (const auto &kv : sensationTuning) {
    tuning[kv.first] = {{"gain", kv.second.gain},
                        {"halfLifeSec", kv.second.halfLifeSec},
                        {"setpoint", kv.second.setpoint}};
  }
  j["sensationTuning"] = tuning;
  nlohmann::json insts = nlohmann::json::array();
  for (const auto &i : instincts) insts.push_back(i.toJson());
  j["instincts"] = insts;
  return j;
}

SubconsciousProfile SubconsciousProfile::fromJson(const nlohmann::json &j) {
  SubconsciousProfile p;
  if (!j.is_object()) return p;
  auto g = [&](const char *k, float &dst) {
    if (j.contains(k) && j[k].is_number()) dst = j[k].get<float>();
  };
  g("baselineValence", p.baselineValence);
  g("baselineArousal", p.baselineArousal);
  g("baselineDominance", p.baselineDominance);
  g("temperamentStrength", p.temperamentStrength);
  g("riskAversion", p.riskAversion);
  g("anticipatoryGain", p.anticipatoryGain);

  if (j.contains("sensationTuning") && j["sensationTuning"].is_object()) {
    for (auto it = j["sensationTuning"].begin(); it != j["sensationTuning"].end(); ++it) {
      const auto &v = it.value();
      if (!v.is_object()) continue;
      primal::SensationTuning st;
      if (v.contains("gain") && v["gain"].is_number()) st.gain = v["gain"].get<float>();
      if (v.contains("halfLifeSec") && v["halfLifeSec"].is_number())
        st.halfLifeSec = v["halfLifeSec"].get<float>();
      if (v.contains("setpoint") && v["setpoint"].is_number())
        st.setpoint = v["setpoint"].get<float>();
      p.sensationTuning[it.key()] = st;
    }
  }
  if (j.contains("instincts") && j["instincts"].is_array()) {
    for (const auto &item : j["instincts"]) {
      p.instincts.push_back(instinct::Instinct::fromJson(item));
    }
  }
  return p;
}

SubconsciousProfile SubconsciousProfile::defaults() { return SubconsciousProfile{}; }

void SubconsciousProfile::applyTo(primal::PrimalSensationEngine &engine) const {
  engine.setTuning(parseTuning(sensationTuning));
}

void SubconsciousProfile::applyTo(instinct::InstinctEngine &engine) const {
  if (!instincts.empty()) engine.replaceAll(instincts);
}

void SubconsciousProfile::applyTo(phoenix::agi::ActiveInferenceController &ctl) const {
  ctl.setRiskAversion(riskAversion);
}

emotion::EmotionTensor SubconsciousProfile::applyTemperament(
    const emotion::EmotionTensor &t) const {
  if (temperamentStrength <= 0.0f) return t;
  emotion::EmotionTensor out = t;
  out.valence = std::clamp(out.valence + temperamentStrength * baselineValence, -1.0f, 1.0f);
  out.arousal = std::clamp(out.arousal + temperamentStrength * baselineArousal, -1.0f, 1.0f);
  out.dominance = std::clamp(out.dominance + temperamentStrength * baselineDominance, -1.0f, 1.0f);
  return out;
}

}  // namespace subconscious
}  // namespace phoenix
