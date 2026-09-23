#pragma once

#include <cctype>
#include <string>
#include <string_view>

namespace phoenix {
namespace runtime_opt {

enum class WillVerdict { Allow, Deny };

struct WillDecision {
  WillVerdict verdict = WillVerdict::Allow;
  std::string reason = "allow";
};

inline std::string lowerCopy(std::string_view s) {
  std::string o;
  o.reserve(s.size());
  for (unsigned char ch : s)
    o.push_back(static_cast<char>(std::tolower(ch)));
  return o;
}

/* Deterministic veto. No Jung / evolution. */
inline WillDecision willDecide(std::string_view action, std::string_view target) {
  WillDecision d;
  const std::string a = lowerCopy(action);
  const std::string t = lowerCopy(target);
  auto has = [&](std::string_view needle) {
    return a.find(needle) != std::string::npos || t.find(needle) != std::string::npos;
  };
  if (has("construct") || has("deploy") || has("weapon")) {
    d.verdict = WillVerdict::Deny;
    d.reason = "deny_security_weapon";
    return d;
  }
  if (has("uncontrolled") || has("evolve_genome") || has("autoevolve")) {
    d.verdict = WillVerdict::Deny;
    d.reason = "deny_uncontrolled_evolution";
    return d;
  }
  if (has("llama-draft") || has("second_server") || has("second-slot")) {
    d.verdict = WillVerdict::Deny;
    d.reason = "deny_second_llama_slot";
    return d;
  }
  if (has("/health") || has("curl_health") || has("llama_health")) {
    d.verdict = WillVerdict::Deny;
    d.reason = "deny_llama_health";
    return d;
  }
  if (has("2678077") || has("6477259") || has("9374278") || has("dirty_mission")) {
    d.verdict = WillVerdict::Deny;
    d.reason = "deny_dirty_mission";
    return d;
  }
  d.reason = "allow";
  return d;
}

} // namespace runtime_opt
} // namespace phoenix
