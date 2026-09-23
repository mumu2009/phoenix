#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace phoenix {
namespace runtime_opt {

struct SpeculativeConfig {
  bool enabled = false;
  std::string mode = "llama_server_draft";
  std::string draftModel;
  int draftMax = 4;
  int draftMin = 2;
  double draftPMin = 0.6;
  bool forbidSecondServerWhenParallel1 = true;
  bool neverCurlHealth = true;
  bool neverRestartLiveLlama = true;
};

struct SpeculativePlan {
  bool enabled = false;
  bool spawnSecondServer = false;
  bool attachToExistingProcess = false;
  bool refuse = false;
  std::string reason;
  std::vector<std::string> sameProcessArgs;
  std::vector<std::string> requestFields;
  int reqNMax = 0;
  int reqNMin = 0;
  double reqPMin = 0.0;
};

inline int clampDraftMax(int n) { return n < 1 ? 1 : (n > 16 ? 16 : n); }
inline int clampDraftMin(int n, int mx) {
  if (n < 1)
    n = 1;
  return n > mx ? mx : n;
}
inline double clampDraftPMin(double p) {
  if (p < 0.1)
    return 0.1;
  if (p > 0.95)
    return 0.95;
  return p;
}

inline bool refuseSecondSlotServer(int parallel, bool wantStandalone) {
  return wantStandalone && parallel <= 1;
}

inline bool llamaHealthHttpForbidden() { return true; }

inline bool urlLooksLikeLlamaHealth(std::string_view url) {
  return url.find("/health") != std::string_view::npos;
}

inline SpeculativePlan planSpeculative(const SpeculativeConfig &cfg, int parallel,
                                       bool llamaAlreadyRunning,
                                       bool wantStandaloneDraftServer) {
  SpeculativePlan plan;
  if (cfg.neverCurlHealth)
    plan.requestFields.push_back("never_curl_/health");

  const bool parallelOne = parallel <= 1;
  if (wantStandaloneDraftServer &&
      (cfg.forbidSecondServerWhenParallel1 && parallelOne)) {
    plan.refuse = true;
    plan.spawnSecondServer = false;
    plan.reason = "parallel_1_forbids_second_slot_server";
    return plan;
  }
  if (wantStandaloneDraftServer) {
    plan.refuse = true;
    plan.spawnSecondServer = false;
    plan.reason = "draft_must_share_llama_server_process";
    return plan;
  }
  if (!cfg.enabled) {
    plan.reason = "speculative_off";
    return plan;
  }
  if (cfg.draftModel.empty()) {
    plan.refuse = true;
    plan.reason = "draft_model_empty_degrade";
    return plan;
  }
  if (llamaAlreadyRunning && cfg.neverRestartLiveLlama) {
    plan.attachToExistingProcess = false;
    plan.reason = "llama_alive_leave_no_reflag";
    return plan;
  }
  plan.enabled = true;
  plan.attachToExistingProcess = true;
  plan.reqNMax = clampDraftMax(cfg.draftMax);
  plan.reqNMin = clampDraftMin(cfg.draftMin, plan.reqNMax);
  plan.reqPMin = clampDraftPMin(cfg.draftPMin);
  plan.sameProcessArgs = {"--model-draft",
                          cfg.draftModel,
                          "--draft-max",
                          std::to_string(plan.reqNMax),
                          "--draft-min",
                          std::to_string(plan.reqNMin),
                          "--draft-p-min",
                          std::to_string(plan.reqPMin)};
  plan.requestFields.push_back("speculative.n_max");
  plan.requestFields.push_back("speculative.n_min");
  plan.requestFields.push_back("speculative.p_min");
  plan.reason = "llama_server_same_process_draft";
  return plan;
}

} // namespace runtime_opt
} // namespace phoenix
