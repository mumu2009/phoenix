#pragma once

#include "runtime_opt/hivemind.hpp"
#include "runtime_opt/hooks.hpp"
#include "runtime_opt/host_tune.hpp"
#include "runtime_opt/kairos.hpp"
#include "runtime_opt/mlock_guard.hpp"
#include "runtime_opt/model_router.hpp"
#include "runtime_opt/platform.hpp"
#include "runtime_opt/process_guard.hpp"
#include "runtime_opt/scope_cache.hpp"
#include "runtime_opt/shm_channel.hpp"
#include "runtime_opt/speculative.hpp"
#include "runtime_opt/survivability.hpp"
#include "runtime_opt/valence_bias.hpp"

#include <nlohmann/json.hpp>
#include <string>

namespace phoenix {
namespace runtime_opt {

struct OverlayConfig {
  bool enabled = true;
  std::string profile = "auto";
  ProcessGuardConfig processGuard;
  MlockConfig mlock;
  SpeculativeConfig speculative;
  RouterConfig router;
  ShmConfig shm;
  SurviveConfig survive;
  KairosConfig kairos;
  HiveConfig hive;
  ValenceBiasConfig valence;
  ScopeCacheConfig scopeCache;
  HostTuneConfig hostTune;
  bool willGate = true;
  bool securityAgg = true;
  bool watchdogEnabled = true;
  bool hierarchicalMemoryReserve = true;
  bool valenceArousalWireOnly = true;
  bool orchestrationHooks = true;
  bool reuseSecurityPlugin = true;
  bool dannLinkIfPresent = true;
  bool noUncontrolledEvolution = true;
};

inline TriSwitch triFromJson(const nlohmann::json &v, TriSwitch fallback) {
  if (v.is_boolean())
    return v.get<bool>() ? TriSwitch::On : TriSwitch::Off;
  if (v.is_string())
    return parseTri(v.get<std::string>(), fallback);
  return fallback;
}

inline OverlayConfig loadOverlay(const nlohmann::json &root) {
  OverlayConfig out;
  if (!root.is_object() || !root.contains("runtime_opt") ||
      !root["runtime_opt"].is_object())
    return out;
  const auto &j = root["runtime_opt"];
  if (j.contains("enabled") && j["enabled"].is_boolean())
    out.enabled = j["enabled"].get<bool>();
  if (j.contains("profile") && j["profile"].is_string())
    out.profile = j["profile"].get<std::string>();

  auto obj = [&](const char *key) -> const nlohmann::json * {
    if (j.contains(key) && j[key].is_object())
      return &j[key];
    return nullptr;
  };

  if (const auto *g = obj("process_guard")) {
    if (g->contains("enabled"))
      out.processGuard.enabled = triFromJson((*g)["enabled"], TriSwitch::Auto);
    if (g->contains("windowsJobObject"))
      out.processGuard.windowsJobObject = (*g)["windowsJobObject"].get<bool>();
    if (g->contains("linuxCgroupV2"))
      out.processGuard.linuxCgroupV2 = (*g)["linuxCgroupV2"].get<bool>();
    if (g->contains("cgroupRequireRoot"))
      out.processGuard.cgroupRequireRoot = (*g)["cgroupRequireRoot"].get<bool>();
    if (g->contains("cgroupDegradeToOom"))
      out.processGuard.cgroupDegradeToOom = (*g)["cgroupDegradeToOom"].get<bool>();
    if (g->contains("killTreeOnClose"))
      out.processGuard.killTreeOnClose = (*g)["killTreeOnClose"].get<bool>();
    if (g->contains("neverScoreLlamaAsExpendable"))
      out.processGuard.neverScoreLlamaAsExpendable =
          (*g)["neverScoreLlamaAsExpendable"].get<bool>();
    if (g->contains("oomScoreAdj") && (*g)["oomScoreAdj"].is_object()) {
      const auto &o = (*g)["oomScoreAdj"];
      if (o.contains("llama"))
        out.processGuard.oomLlama = o["llama"].get<int>();
      if (o.contains("gateway"))
        out.processGuard.oomGateway = o["gateway"].get<int>();
      if (o.contains("frontend"))
        out.processGuard.oomFrontend = o["frontend"].get<int>();
      if (o.contains("helper"))
        out.processGuard.oomHelper = o["helper"].get<int>();
    }
  }

  if (const auto *m = obj("mlock")) {
    if (m->contains("enabled"))
      out.mlock.enabled = triFromJson((*m)["enabled"], TriSwitch::Auto);
    if (m->contains("degradeOnFail"))
      out.mlock.degradeOnFail = (*m)["degradeOnFail"].get<bool>();
    if (m->contains("maxBytes") && (*m)["maxBytes"].is_number_unsigned())
      out.mlock.maxBytes = (*m)["maxBytes"].get<size_t>();
    if (m->contains("hardCapBytes") && (*m)["hardCapBytes"].is_number_unsigned())
      out.mlock.hardCapBytes = (*m)["hardCapBytes"].get<size_t>();
  }
  out.mlock.llamaServerFlag = false;
  out.mlock.maxBytes = effectiveMlockCap(out.mlock);
  out.mlock.hardCapBytes = kMlockHardCapBytes;

  if (const auto *s = obj("speculative")) {
    if (s->contains("enabled") && (*s)["enabled"].is_boolean())
      out.speculative.enabled = (*s)["enabled"].get<bool>();
    if (s->contains("draftModel") && (*s)["draftModel"].is_string())
      out.speculative.draftModel = (*s)["draftModel"].get<std::string>();
    if (s->contains("draftMax"))
      out.speculative.draftMax = (*s)["draftMax"].get<int>();
    if (s->contains("draftMin"))
      out.speculative.draftMin = (*s)["draftMin"].get<int>();
    if (s->contains("draftPMin"))
      out.speculative.draftPMin = (*s)["draftPMin"].get<double>();
    if (s->contains("forbidSecondServerWhenParallel1"))
      out.speculative.forbidSecondServerWhenParallel1 =
          (*s)["forbidSecondServerWhenParallel1"].get<bool>();
    if (s->contains("neverRestartLiveLlama"))
      out.speculative.neverRestartLiveLlama =
          (*s)["neverRestartLiveLlama"].get<bool>();
  }

  if (const auto *r = obj("model_router")) {
    if (r->contains("enabled") && (*r)["enabled"].is_boolean())
      out.router.enabled = (*r)["enabled"].get<bool>();
    if (r->contains("infoOnly"))
      out.router.infoOnly = (*r)["infoOnly"].get<bool>();
    if (r->contains("neverDualSlot"))
      out.router.neverDualSlot = (*r)["neverDualSlot"].get<bool>();
    if (r->contains("longTaskChars"))
      out.router.longTaskChars = (*r)["longTaskChars"].get<int>();
    if (r->contains("organBackends") && (*r)["organBackends"].is_object()) {
      const auto &o = (*r)["organBackends"];
      if (o.contains("caption"))
        out.router.organCaption = o["caption"].get<std::string>();
      if (o.contains("summary"))
        out.router.organSummary = o["summary"].get<std::string>();
      if (o.contains("mission"))
        out.router.organMission = o["mission"].get<std::string>();
      if (o.contains("chat"))
        out.router.organChat = o["chat"].get<std::string>();
    }
  }

  if (const auto *h = obj("shm_ipc")) {
    if (h->contains("enabled"))
      out.shm.enabled = triFromJson((*h)["enabled"], TriSwitch::Auto);
    if (h->contains("name") && (*h)["name"].is_string())
      out.shm.name = (*h)["name"].get<std::string>();
    if (h->contains("bytes") && (*h)["bytes"].is_number_unsigned())
      out.shm.bytes = (*h)["bytes"].get<size_t>();
    if (h->contains("fallback") && (*h)["fallback"].is_string())
      out.shm.fallback = (*h)["fallback"].get<std::string>();
  }

  if (const auto *sv = obj("survivability")) {
    if (sv->contains("enabled"))
      out.survive.enabled = resolveTri(triFromJson((*sv)["enabled"], TriSwitch::On), true);
    if (sv->contains("diskFreeLowPct"))
      out.survive.diskFreeLowPct = (*sv)["diskFreeLowPct"].get<int>();
    if (sv->contains("swapHighPct"))
      out.survive.swapHighPct = (*sv)["swapHighPct"].get<int>();
    if (sv->contains("swapCriticalPct"))
      out.survive.swapCriticalPct = (*sv)["swapCriticalPct"].get<int>();
  }
  if (const auto *k = obj("kairos")) {
    if (k->contains("enabled"))
      out.kairos.enabled = resolveTri(triFromJson((*k)["enabled"], TriSwitch::Auto),
                                      isLinuxLike(currentPlatform()));
    if (k->contains("heavySkipPressure"))
      out.kairos.heavySkipPressure = (*k)["heavySkipPressure"].get<double>();
  }
  if (const auto *hv = obj("hivemind")) {
    if (hv->contains("enabled"))
      out.hive.enabled = resolveTri(triFromJson((*hv)["enabled"], TriSwitch::Auto), true);
    if (hv->contains("neverDualSlot"))
      out.hive.neverDualSlot = (*hv)["neverDualSlot"].get<bool>();
  }
  if (const auto *vb = obj("valence_bias")) {
    if (vb->contains("enabled") && (*vb)["enabled"].is_boolean())
      out.valence.enabled = (*vb)["enabled"].get<bool>();
    if (vb->contains("maxTempDelta"))
      out.valence.maxTempDelta = (*vb)["maxTempDelta"].get<double>();
  }
  if (const auto *sc = obj("scope_cache")) {
    if (sc->contains("enabled"))
      out.scopeCache.enabled =
          resolveTri(triFromJson((*sc)["enabled"], TriSwitch::Auto), true);
    if (sc->contains("maxEntries") && (*sc)["maxEntries"].is_number_unsigned())
      out.scopeCache.maxEntries = (*sc)["maxEntries"].get<size_t>();
  }
  if (const auto *ht = obj("host_tune")) {
    if (ht->contains("enabled"))
      out.hostTune.enabled = resolveTri(triFromJson((*ht)["enabled"], TriSwitch::Auto),
                                        isLinuxLike(currentPlatform()));
    if (ht->contains("threadCap"))
      out.hostTune.threadCap = (*ht)["threadCap"].get<int>();
    if (ht->contains("neverTuneLlama"))
      out.hostTune.neverTuneLlama = (*ht)["neverTuneLlama"].get<bool>();
  }
  if (j.contains("will_gate") && j["will_gate"].is_object() &&
      j["will_gate"].contains("enabled") && j["will_gate"]["enabled"].is_boolean())
    out.willGate = j["will_gate"]["enabled"].get<bool>();
  if (j.contains("security_agg") && j["security_agg"].is_object() &&
      j["security_agg"].contains("enabled") && j["security_agg"]["enabled"].is_boolean())
    out.securityAgg = j["security_agg"]["enabled"].get<bool>();
  if (const auto *w = obj("watchdog")) {
    if (w->contains("enabled") && (*w)["enabled"].is_boolean())
      out.watchdogEnabled = (*w)["enabled"].get<bool>();
  }
  if (const auto *hk = obj("hooks")) {
    if (hk->contains("noUncontrolledEvolution"))
      out.noUncontrolledEvolution = (*hk)["noUncontrolledEvolution"].get<bool>();
    if (hk->contains("reuseSecurityPlugin"))
      out.reuseSecurityPlugin = (*hk)["reuseSecurityPlugin"].get<bool>();
    if (hk->contains("dannLinkIfPresent"))
      out.dannLinkIfPresent = (*hk)["dannLinkIfPresent"].get<bool>();
  }
  if (out.profile == "safe") {
    if (out.processGuard.enabled == TriSwitch::Auto)
      out.processGuard.enabled = TriSwitch::Off;
    if (out.mlock.enabled == TriSwitch::Auto)
      out.mlock.enabled = TriSwitch::Off;
    if (out.shm.enabled == TriSwitch::Auto)
      out.shm.enabled = TriSwitch::Off;
    out.kairos.enabled = false;
    out.hostTune.enabled = false;
    out.valence.enabled = false;
  }
  if (out.profile == "linux_rdk") {
    if (out.processGuard.enabled == TriSwitch::Auto)
      out.processGuard.enabled = TriSwitch::On;
    if (out.mlock.enabled == TriSwitch::Auto)
      out.mlock.enabled = TriSwitch::On;
    if (out.shm.enabled == TriSwitch::Auto)
      out.shm.enabled = TriSwitch::On;
    out.survive.enabled = true;
    out.kairos.enabled = true;
    out.hive.enabled = true;
    out.scopeCache.enabled = true;
    out.hostTune.enabled = true;
  }
  return out;
}

struct EffectiveFlags {
  bool processGuard = false;
  bool mlock = false;
  bool speculative = false;
  bool watchdog = false;
  bool modelRouter = false;
  bool shmIpc = false;
  bool survivability = false;
  bool kairos = false;
  bool willGate = false;
  bool hivemind = false;
  bool valence = false;
  bool scopeCache = false;
  bool hostTune = false;
};

inline EffectiveFlags resolveFlags(const OverlayConfig &cfg, HostPlatform p) {
  EffectiveFlags f;
  if (!cfg.enabled)
    return f;
  f.processGuard = resolveTri(cfg.processGuard.enabled, processGuardAutoOn(p));
  f.mlock = resolveTri(cfg.mlock.enabled, mlockAutoOn(p));
  f.speculative = cfg.speculative.enabled && !cfg.speculative.draftModel.empty();
  f.watchdog = cfg.watchdogEnabled;
  f.modelRouter = cfg.router.enabled;
  f.shmIpc = resolveTri(cfg.shm.enabled, shmAutoOn(p));
  f.survivability = cfg.survive.enabled;
  f.kairos = cfg.kairos.enabled;
  f.willGate = cfg.willGate;
  f.hivemind = cfg.hive.enabled;
  f.valence = cfg.valence.enabled;
  f.scopeCache = cfg.scopeCache.enabled;
  f.hostTune = cfg.hostTune.enabled;
  return f;
}

} // namespace runtime_opt
} // namespace phoenix
