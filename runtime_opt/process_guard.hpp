#pragma once

#include "runtime_opt/platform.hpp"

#include <string>
#include <string_view>

namespace phoenix {
namespace runtime_opt {

inline bool roleIsLlama(std::string_view role) {
  return role == "llama" || role == "inference";
}

struct ProcessGuardConfig {
  TriSwitch enabled = TriSwitch::Auto;
  bool windowsJobObject = true;
  bool linuxCgroupV2 = true;
  bool cgroupRequireRoot = true;
  bool cgroupDegradeToOom = true;
  bool killTreeOnClose = false;
  bool neverScoreLlamaAsExpendable = true;
  int oomLlama = -200;
  int oomGateway = 300;
  int oomFrontend = 400;
  int oomHelper = 500;
};

struct GuardPlan {
  std::string role;
  int oomScoreAdj = 0;
  bool useJobObject = false;
  bool useCgroupV2 = false;
  bool killTreeOnClose = false;
  bool mayCleanupTree = false;
  bool skip = false;
  std::string reason;
};

inline int oomScoreForRole(std::string_view role, const ProcessGuardConfig &cfg) {
  if (roleIsLlama(role))
    return cfg.neverScoreLlamaAsExpendable ? cfg.oomLlama : cfg.oomHelper;
  if (role == "gateway")
    return cfg.oomGateway;
  if (role == "frontend")
    return cfg.oomFrontend;
  return cfg.oomHelper;
}

inline bool processGuardAutoOn(HostPlatform p) {
  /* Grouping is safe on both hosts; Linux/RDK also writes oom_score_adj. */
  return p == HostPlatform::Windows || isLinuxLike(p);
}

inline GuardPlan planProcessGuard(std::string_view role, const ProcessGuardConfig &cfg,
                                  HostPlatform platform) {
  GuardPlan plan;
  plan.role = std::string(role);
  const bool on = resolveTri(cfg.enabled, processGuardAutoOn(platform));
  if (!on) {
    plan.skip = true;
    plan.reason = "process_guard_off";
    return plan;
  }
  plan.oomScoreAdj = oomScoreForRole(role, cfg);
  plan.mayCleanupTree = !roleIsLlama(role);
  if (roleIsLlama(role)) {
    plan.useJobObject = false;
    plan.useCgroupV2 = false;
    plan.killTreeOnClose = false;
    plan.reason = "llama_protect_score_only";
    return plan;
  }
  plan.useJobObject = platform == HostPlatform::Windows && cfg.windowsJobObject;
  plan.useCgroupV2 = isLinuxLike(platform) && cfg.linuxCgroupV2;
  plan.killTreeOnClose = cfg.killTreeOnClose && plan.mayCleanupTree;
  plan.reason = plan.useJobObject ? "windows_job_object"
                                  : (cfg.cgroupDegradeToOom ? "linux_cgroup_or_oom_degrade"
                                                            : "linux_cgroup_or_oom");
  return plan;
}

inline bool mayCleanupProcessTree(std::string_view role) { return !roleIsLlama(role); }

} // namespace runtime_opt
} // namespace phoenix
