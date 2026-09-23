#pragma once

#include <cstddef>
#include <string_view>

namespace phoenix {
namespace runtime_opt {

/* RDK X5 192.168.1.107 (wlan0 DHCP, sunrise sudo -n):
   RAM 6.9Gi, llama ~4.4Gi RSS + ~1.15Gi swap, ulimit -l ~885MiB, VmLck=0,
   8082 bind 127.0.0.1 --parallel 1. cgroup v2 writes need root/sudo. */
inline constexpr std::string_view kRdkX5BoardCapNote =
    "RDK X5 192.168.1.107: 6.9Gi RAM, llama ~4.4Gi RSS + ~1.15Gi swap, "
    "ulimit -l ~885MiB, VmLck=0, 8082=127.0.0.1 --parallel 1. "
    "mlock hot cap <=64MiB (never --mlock live llama). "
    "cgroup v2 needs sudo; write fail -> oom_score_adj only; "
    "llama stays out of kill cgroup.";

inline constexpr size_t kMlockHardCapBytes = 64ull * 1024ull * 1024ull;
inline constexpr size_t kMlockDefaultHotBytes = 4ull * 1024ull * 1024ull;

inline size_t clampMlockBytes(size_t want) {
  const size_t n = want == 0 ? kMlockDefaultHotBytes : want;
  return n > kMlockHardCapBytes ? kMlockHardCapBytes : n;
}

} // namespace runtime_opt
} // namespace phoenix
