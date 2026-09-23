#pragma once

#include <algorithm>
#include <string>

namespace phoenix {
namespace runtime_opt {

struct KairosConfig {
  bool enabled = true;
  double heavySkipPressure = 0.85;
  int minTickMs = 200;
  int maxTickMs = 5000;
};

struct TickPlan {
  int delayMs = 1000;
  bool skipHeavy = false;
  bool skipOrganFanout = false;
  std::string reason = "nominal";
};

/* Consumes deadline/pressure as inputs. Does not rewrite mission pressure. */
inline TickPlan planTick(int deadlineRemainSec, double pressure01,
                         bool diskTight, const KairosConfig &cfg) {
  TickPlan t;
  if (!cfg.enabled) {
    t.reason = "kairos_off";
    return t;
  }
  double p = pressure01;
  if (p < 0.0)
    p = 0.0;
  if (p > 1.0)
    p = 1.0;
  const int span = std::max(0, cfg.maxTickMs - cfg.minTickMs);
  t.delayMs = cfg.minTickMs + static_cast<int>(span * p);
  if (deadlineRemainSec >= 0 && deadlineRemainSec < 30)
    t.delayMs = cfg.minTickMs;
  t.skipHeavy = diskTight || p >= cfg.heavySkipPressure ||
                (deadlineRemainSec >= 0 && deadlineRemainSec < 20);
  t.skipOrganFanout = t.skipHeavy;
  if (t.skipHeavy && deadlineRemainSec >= 0 && deadlineRemainSec < 20)
    t.reason = "deadline_near_skip_heavy";
  else if (diskTight)
    t.reason = "disk_tight_skip_heavy";
  else if (p >= cfg.heavySkipPressure)
    t.reason = "pressure_skip_heavy";
  else
    t.reason = "nominal";
  return t;
}

} // namespace runtime_opt
} // namespace phoenix
