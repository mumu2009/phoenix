#pragma once

#include <string>
#include <string_view>

namespace phoenix {
namespace runtime_opt {

enum class HostPlatform { Windows, Linux, Unknown };

inline HostPlatform currentPlatform() {
#ifdef _WIN32
  return HostPlatform::Windows;
#elif defined(__linux__)
  return HostPlatform::Linux;
#else
  return HostPlatform::Unknown;
#endif
}

inline bool isLinuxLike(HostPlatform p) {
  return p == HostPlatform::Linux;
}

inline const char *platformName(HostPlatform p) {
  switch (p) {
  case HostPlatform::Windows:
    return "windows";
  case HostPlatform::Linux:
    return "linux";
  default:
    return "unknown";
  }
}

enum class TriSwitch { Off, On, Auto };

inline TriSwitch parseTri(std::string_view raw, TriSwitch fallback = TriSwitch::Auto) {
  if (raw.empty())
    return fallback;
  if (raw == "on" || raw == "true" || raw == "1" || raw == "yes")
    return TriSwitch::On;
  if (raw == "off" || raw == "false" || raw == "0" || raw == "no")
    return TriSwitch::Off;
  if (raw == "auto")
    return TriSwitch::Auto;
  return fallback;
}

inline bool resolveTri(TriSwitch sw, bool autoValue) {
  if (sw == TriSwitch::On)
    return true;
  if (sw == TriSwitch::Off)
    return false;
  return autoValue;
}

} // namespace runtime_opt
} // namespace phoenix
