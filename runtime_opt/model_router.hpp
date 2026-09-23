#pragma once

#include <algorithm>
#include <string>
#include <string_view>

namespace phoenix {
namespace runtime_opt {

enum class BackendId { None, Llama, TinyLlama, Bitnet };

inline const char *backendName(BackendId id) {
  switch (id) {
  case BackendId::Llama:
    return "llama";
  case BackendId::TinyLlama:
    return "tinyllama";
  case BackendId::Bitnet:
    return "bitnet";
  default:
    return "none";
  }
}

inline BackendId parseBackend(std::string_view raw) {
  if (raw == "llama" || raw == "llamacpp" || raw == "inference")
    return BackendId::Llama;
  if (raw == "tinyllama" || raw == "summary")
    return BackendId::TinyLlama;
  if (raw == "bitnet")
    return BackendId::Bitnet;
  return BackendId::None;
}

struct RouterConfig {
  bool enabled = false;
  bool infoOnly = true;
  bool neverDualSlot = true;
  int longTaskChars = 4000;
  std::string organCaption = "tinyllama";
  std::string organSummary = "tinyllama";
  std::string organMission = "llama";
  std::string organChat = "auto";
};

struct RouteDecision {
  BackendId backend = BackendId::None;
  std::string reason;
  bool dualSlot = false;
  bool infoOnly = true;
};

inline RouteDecision routeTask(int taskChars, std::string_view organ,
                               const RouterConfig &cfg, bool llamaUp, bool tinyUp,
                               bool bitnetUp) {
  RouteDecision d;
  d.infoOnly = cfg.infoOnly;
  d.dualSlot = false;
  if (!cfg.enabled) {
    d.backend = llamaUp ? BackendId::Llama : BackendId::None;
    d.reason = "router_off_keep_primary";
    return d;
  }

  BackendId organHint = BackendId::None;
  if (organ == "caption")
    organHint = parseBackend(cfg.organCaption);
  else if (organ == "summary")
    organHint = parseBackend(cfg.organSummary);
  else if (organ == "mission")
    organHint = parseBackend(cfg.organMission);
  else if (organ == "chat" && cfg.organChat != "auto")
    organHint = parseBackend(cfg.organChat);

  auto available = [&](BackendId id) {
    if (id == BackendId::Llama)
      return llamaUp;
    if (id == BackendId::TinyLlama)
      return tinyUp;
    if (id == BackendId::Bitnet)
      return bitnetUp;
    return false;
  };

  if (organHint != BackendId::None && available(organHint)) {
    d.backend = organHint;
    d.reason = "organ_backend";
    return d;
  }
  if (taskChars >= std::max(1, cfg.longTaskChars) && llamaUp) {
    d.backend = BackendId::Llama;
    d.reason = "long_task_primary";
    return d;
  }
  if (tinyUp && taskChars < std::max(1, cfg.longTaskChars)) {
    d.backend = BackendId::TinyLlama;
    d.reason = "short_task_tiny";
    return d;
  }
  if (llamaUp) {
    d.backend = BackendId::Llama;
    d.reason = "fallback_primary";
    return d;
  }
  if (bitnetUp) {
    d.backend = BackendId::Bitnet;
    d.reason = "fallback_bitnet";
    return d;
  }
  d.reason = "no_backend";
  return d;
}

} // namespace runtime_opt
} // namespace phoenix
