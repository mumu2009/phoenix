/* security_exempt.hpp - exemption areas for the security plugin.

   WHY
   ---
   The security addon is a heuristic detector.  Once it is enabled by default
   (addons.security.enabled=true) it must not flag the system's OWN trusted
   surfaces: its own plugin sources, the mission sandbox it just wrote, the
   build/tmp scratch area, the test corpus, and so on.  Otherwise every
   security pass produces noise about code we shipped ourselves, and a noisy
   detector gets ignored - which is worse than no detector.

   The exemption list is configuration, not code:
     addons.security.exemptPrefixes: ["...", ...]
   A situation is exempt when its text/goal mentions any configured prefix.
   Matching is case-insensitive and plain substring on purpose: prefixes here
   are paths and plugin names, not natural language, so a fuzzy match would
   widen the exemption in ways an operator cannot predict.

   Pure function, unit tested (tests/gtest/unit/addons/test_security_exempt.cpp).
*/
#pragma once

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace phoenix {
namespace security {

inline std::string lowerCopy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

/* True when `text` mentions any exempt prefix.  Empty prefix entries and
   entries shorter than 3 characters are ignored: a 1-2 char "prefix" would
   match almost everything and silently disable the detector. */
inline bool isExempt(const std::string &text,
                     const std::vector<std::string> &prefixes) {
    if (text.empty() || prefixes.empty()) return false;
    const std::string low = lowerCopy(text);
    for (const auto &raw : prefixes) {
        if (raw.size() < 3) continue;
        const std::string p = lowerCopy(raw);
        if (low.find(p) != std::string::npos) return true;
    }
    return false;
}

/* Default trusted surfaces.  Kept here so a deployment that never sets the
   config key still protects its own code/sandbox from being reported. */
inline std::vector<std::string> defaultExemptPrefixes() {
    return {
        "addons/thepluginforsecurity",
        "main_hub_parts/",
        "runtime_store/mission_workspace",
        "build/tmp",
        "tests/gtest",
        "tools/",
    };
}

}  // namespace security
}  // namespace phoenix
