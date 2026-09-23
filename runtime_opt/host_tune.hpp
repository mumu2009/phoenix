#pragma once

#include "runtime_opt/hivemind.hpp"

#include <string>
#include <string_view>

namespace phoenix {
namespace runtime_opt {

struct HostTuneConfig {
  bool enabled = true;
  bool cpuAffinity = true;
  bool nice = true;
  bool hugepageProbe = true;
  int threadCap = 4;
  bool neverTuneLlama = true;
};

struct TunePlan {
  int nice = 0;
  bool setAffinity = false;
  int affinityKeepLastCpus = 1;
  int threadCap = 4;
  bool hugepageOk = false;
  bool hugepageDegrade = true;
  std::string promptCacheKey;
  std::string reason = "off";
};

inline TunePlan planHostTune(std::string_view role, int nproc, bool hugepageAvail,
                             std::string_view scope, std::string_view prefix,
                             const HostTuneConfig &cfg) {
  TunePlan t;
  t.threadCap = cfg.threadCap < 1 ? 1 : cfg.threadCap;
  t.promptCacheKey = std::string(scope) + ":" + std::to_string(fnv1a(prefix));
  if (!cfg.enabled) {
    t.reason = "host_tune_off";
    return t;
  }
  if (cfg.neverTuneLlama && (role == "llama" || role == "inference")) {
    t.reason = "never_tune_llama";
    return t;
  }
  if (cfg.nice)
    t.nice = role == "helper" ? 10 : 5;
  if (cfg.cpuAffinity && nproc >= 4) {
    t.setAffinity = true;
    t.affinityKeepLastCpus = 1;
  }
  t.hugepageOk = hugepageAvail;
  t.hugepageDegrade = !hugepageAvail;
  t.reason = "gateway_or_helper_tune";
  return t;
}

} // namespace runtime_opt
} // namespace phoenix
