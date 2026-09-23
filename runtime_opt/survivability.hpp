#pragma once

#include <algorithm>
#include <string>
#include <string_view>

namespace phoenix {
namespace runtime_opt {

struct SurviveConfig {
  bool enabled = true;
  int diskFreeLowPct = 10;
  int swapHighPct = 70;
  int swapCriticalPct = 85;
  int maxInFlightFloor = 1;
  bool neverTouchLlama = true;
};

struct HostPressure {
  int diskFreePct = 100;
  int swapUsedPct = 0;
  bool llamaAlive = true;
  bool gatewayAlive = true;
};

struct SurviveDecision {
  int maxInFlight = 1;
  bool skipHeavyHelpers = false;
  bool restartGatewayOnly = true;
  bool leaveLlama = true;
  bool cleanupChildTree = true;
  bool diskTight = false;
  bool swapHigh = false;
  std::string reason = "nominal";
};

inline SurviveDecision planSurvive(const HostPressure &h, const SurviveConfig &cfg,
                                   int defaultInFlight) {
  SurviveDecision d;
  d.maxInFlight = std::max(1, defaultInFlight);
  d.leaveLlama = cfg.neverTouchLlama && h.llamaAlive;
  d.restartGatewayOnly = true;
  d.cleanupChildTree = true;
  d.diskTight = h.diskFreePct <= cfg.diskFreeLowPct;
  d.swapHigh = h.swapUsedPct >= cfg.swapHighPct;
  if (!cfg.enabled) {
    d.reason = "survivability_off";
    return d;
  }
  if (d.diskTight || h.swapUsedPct >= cfg.swapCriticalPct) {
    d.maxInFlight = cfg.maxInFlightFloor;
    d.skipHeavyHelpers = true;
    d.reason = d.diskTight ? "disk_low_serialize" : "swap_critical_serialize";
    return d;
  }
  if (d.swapHigh) {
    d.maxInFlight = std::min(d.maxInFlight, std::max(1, cfg.maxInFlightFloor));
    d.skipHeavyHelpers = true;
    d.reason = "swap_high_drop_helpers";
    return d;
  }
  d.reason = "nominal";
  return d;
}

inline bool surviveMayRestart(std::string_view role, const SurviveDecision &d) {
  if (role == "llama" || role == "inference")
    return false;
  if (role == "gateway")
    return d.restartGatewayOnly;
  return role == "frontend";
}

} // namespace runtime_opt
} // namespace phoenix
