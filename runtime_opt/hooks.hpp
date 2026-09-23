#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <string_view>

namespace phoenix {
namespace runtime_opt {

/* Isolation agent owns RNN/LSTM/GNN + chat/mission memory isolation.
   This overlay only reserves an HMT/HELT-shaped hook and does not store state. */
struct HierarchicalMemoryReserve {
  static constexpr const char *kKind = "hmt_helt_reserve";
  bool isolationOwnedElsewhere = true;
  bool implementHere = false;
};

struct ValenceArousal {
  float valence = 0.0f;
  float arousal = 0.0f;
  std::string source = "emotion_or_subconscious";
};

inline float clampUnit(float v) {
  return std::max(-1.0f, std::min(1.0f, v));
}

inline ValenceArousal wireValenceArousal(float valence, float arousal,
                                         std::string_view source) {
  ValenceArousal out;
  out.valence = clampUnit(valence);
  out.arousal = clampUnit(arousal);
  if (!source.empty())
    out.source = std::string(source);
  return out;
}

enum class OrchestrationHook {
  Will,
  Kairos,
  HiveMind,
  ExoMem,
  AuraOS,
  Survivability,
};

inline const char *hookName(OrchestrationHook h) {
  switch (h) {
  case OrchestrationHook::Will:
    return "will";
  case OrchestrationHook::Kairos:
    return "kairos";
  case OrchestrationHook::HiveMind:
    return "hivemind";
  case OrchestrationHook::ExoMem:
    return "exomem";
  case OrchestrationHook::AuraOS:
    return "auraos";
  case OrchestrationHook::Survivability:
    return "survivability";
  }
  return "unknown";
}

struct HookDecision {
  std::string name;
  bool allowUncontrolledEvolution = false;
  bool mutateGenome = false;
  std::string action = "observe_only";
};

inline HookDecision planOrchestration(OrchestrationHook hook, bool noUncontrolled) {
  HookDecision d;
  d.name = hookName(hook);
  d.allowUncontrolledEvolution = false;
  d.mutateGenome = false;
  (void)noUncontrolled;
  switch (hook) {
  case OrchestrationHook::Will:
    d.action = "will_gate";
    break;
  case OrchestrationHook::Kairos:
    d.action = "kairos_tick";
    break;
  case OrchestrationHook::HiveMind:
    d.action = "hivemind_serial";
    break;
  case OrchestrationHook::ExoMem:
    d.action = "exomem_shm_frames";
    break;
  case OrchestrationHook::AuraOS:
    d.action = "helper_sandbox";
    break;
  case OrchestrationHook::Survivability:
    d.action = "survive_degrade";
    break;
  }
  return d;
}

inline bool reuseSecurityPluginInsteadOfNewWeapon() { return true; }

inline bool dannModulePresent() { return false; }

inline bool dannShouldLink(bool modulePresent) { return modulePresent; }

} // namespace runtime_opt
} // namespace phoenix
