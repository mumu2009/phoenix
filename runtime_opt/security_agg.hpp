#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace phoenix {
namespace runtime_opt {

struct SecurityAlert {
  std::string kind;
  std::string detail;
  int severity = 1;
};

struct AlertAggregate {
  int count = 0;
  int maxSeverity = 0;
  std::string worstKind;
  bool blockRecommend = false;
  std::string reason = "none";
};

/* Reuse ThePlugInForSecurity alerts. No construct/deploy. */
inline AlertAggregate aggregateAlerts(const std::vector<SecurityAlert> &in,
                                      int cap = 32) {
  AlertAggregate a;
  const int n = static_cast<int>(in.size());
  a.count = n > cap ? cap : n;
  for (int i = 0; i < a.count; ++i) {
    if (in[static_cast<size_t>(i)].severity > a.maxSeverity) {
      a.maxSeverity = in[static_cast<size_t>(i)].severity;
      a.worstKind = in[static_cast<size_t>(i)].kind;
    }
  }
  a.blockRecommend = a.maxSeverity >= 3;
  a.reason = a.blockRecommend ? "reuse_security_block" : "reuse_security_observe";
  return a;
}

inline bool securityActionForbidden(std::string_view action) {
  return action == "construct" || action == "deploy";
}

} // namespace runtime_opt
} // namespace phoenix
