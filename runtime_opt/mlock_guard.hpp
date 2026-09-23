#pragma once

#include "runtime_opt/board_cap.hpp"
#include "runtime_opt/platform.hpp"

#include <cstddef>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace phoenix {
namespace runtime_opt {

struct MlockConfig {
  TriSwitch enabled = TriSwitch::Auto;
  bool hotWeights = true;
  bool sharedBuffers = true;
  bool degradeOnFail = true;
  size_t maxBytes = kMlockDefaultHotBytes;
  size_t hardCapBytes = kMlockHardCapBytes;
  bool llamaServerFlag = false;
};

struct MlockResult {
  bool locked = false;
  bool degraded = false;
  std::string reason;
  size_t bytes = 0;
};

inline bool mlockAutoOn(HostPlatform p) {
  /* Linux/RDK: try pin small overlay buffers. Windows: available but not forced. */
  return isLinuxLike(p);
}

inline bool mlockWanted(const MlockConfig &cfg, HostPlatform platform) {
  return resolveTri(cfg.enabled, mlockAutoOn(platform));
}

inline size_t effectiveMlockCap(const MlockConfig &cfg) {
  const size_t cfgCap =
      cfg.hardCapBytes == 0 ? kMlockHardCapBytes : cfg.hardCapBytes;
  const size_t hard = cfgCap > kMlockHardCapBytes ? kMlockHardCapBytes : cfgCap;
  return clampMlockBytes(cfg.maxBytes == 0 ? kMlockDefaultHotBytes
                                           : (cfg.maxBytes > hard ? hard : cfg.maxBytes));
}

inline MlockResult tryLockPages(void *ptr, size_t bytes, const MlockConfig &cfg) {
  MlockResult out;
  out.bytes = bytes;
  const size_t allow = effectiveMlockCap(cfg);
  if (bytes > allow) {
    out.degraded = true;
    out.reason = "over_max_bytes";
    return out;
  }
  if (!ptr || bytes == 0) {
    out.degraded = true;
    out.reason = "empty_region";
    return out;
  }
#ifdef _WIN32
  if (VirtualLock(ptr, bytes)) {
    out.locked = true;
    out.reason = "VirtualLock";
    return out;
  }
  out.degraded = true;
  out.reason = "VirtualLock_failed";
#else
  if (mlock(ptr, bytes) == 0) {
    out.locked = true;
    out.reason = "mlock";
    return out;
  }
  out.degraded = true;
  out.reason = "mlock_failed";
#endif
  return out;
}

inline MlockResult tryUnlockPages(void *ptr, size_t bytes) {
  MlockResult out;
  out.bytes = bytes;
  if (!ptr || bytes == 0) {
    out.degraded = true;
    out.reason = "empty_region";
    return out;
  }
#ifdef _WIN32
  if (VirtualUnlock(ptr, bytes)) {
    out.locked = false;
    out.reason = "VirtualUnlock";
    return out;
  }
  out.degraded = true;
  out.reason = "VirtualUnlock_failed";
#else
  if (munlock(ptr, bytes) == 0) {
    out.locked = false;
    out.reason = "munlock";
    return out;
  }
  out.degraded = true;
  out.reason = "munlock_failed";
#endif
  return out;
}

} // namespace runtime_opt
} // namespace phoenix
