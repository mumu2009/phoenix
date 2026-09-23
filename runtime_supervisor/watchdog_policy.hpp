#pragma once

#include "runtime_supervisor/supervisor.hpp"

#include <string>
#include <string_view>

namespace phoenix {
namespace runtime_supervisor {

inline bool llamaHealthHttpForbidden() { return true; }

inline bool mayUseHttpHealthProbe(std::string_view role) {
  return !roleIsLlama(role);
}

inline bool urlLooksLikeLlamaHealth(std::string_view url) {
  return url.find("/health") != std::string_view::npos;
}

inline bool refuseLlamaHealthUrl(std::string_view role, std::string_view url) {
  return roleIsLlama(role) && urlLooksLikeLlamaHealth(url);
}

struct RssFuseDecision {
  std::string role;
  bool restart = false;
  std::string reason;
};

inline RssFuseDecision planRssFuse(std::string_view role, int rssMb, int maxRssMb) {
  RssFuseDecision d;
  d.role = std::string(role);
  if (roleIsLlama(role)) {
    d.reason = "llama_rss_ignored";
    return d;
  }
  if (role != "gateway") {
    d.reason = "rss_fuse_gateway_only";
    return d;
  }
  if (maxRssMb > 0 && rssMb >= maxRssMb) {
    d.restart = true;
    d.reason = "rss_fuse";
    return d;
  }
  d.reason = "rss_ok";
  return d;
}

inline bool mayCleanupProcessTree(std::string_view role) { return !roleIsLlama(role); }

struct WatchdogPolicy {
  bool restartDeadGateway = true;
  bool restartDeadFrontend = true;
  bool leaveLlamaIfAlive = true;
  bool neverKillLlama = true;
  bool neverCurlLlamaHealth = true;
  bool neverLanProbeLlama = true;
  bool rssFuseGatewayOnly = true;
};

inline bool isLoopbackHost(std::string_view host) {
  return host == "127.0.0.1" || host == "localhost" || host == "::1" ||
         host == "0.0.0.0";
}

inline std::string llamaProbeHost(std::string_view host) {
  if (host == "::1")
    return "::1";
  if (isLoopbackHost(host))
    return "127.0.0.1";
  return "127.0.0.1";
}

inline RestartDecision planWatchdogOne(const ServiceRecord &svc,
                                       const WatchdogPolicy &wp) {
  RestartDecision d;
  d.role = svc.role;
  if (svc.alive) {
    d.action = RestartAction::Leave;
    d.reason = roleIsLlama(svc.role) && wp.leaveLlamaIfAlive ? "llama_alive_leave"
                                                             : "service_alive_leave";
    return d;
  }
  if (roleIsLlama(svc.role)) {
    d.action = RestartAction::Leave;
    d.reason = "llama_dead_no_autostart";
    return d;
  }
  if (svc.role == "gateway" && wp.restartDeadGateway) {
    d.action = RestartAction::Start;
    d.reason = "watchdog_gateway_dead";
    return d;
  }
  if (svc.role == "frontend" && wp.restartDeadFrontend) {
    d.action = RestartAction::Start;
    d.reason = "watchdog_frontend_dead";
    return d;
  }
  d.action = RestartAction::Leave;
  d.reason = "watchdog_leave";
  return d;
}

} // namespace runtime_supervisor
} // namespace phoenix
