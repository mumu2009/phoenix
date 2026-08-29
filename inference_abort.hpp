/* inference_abort.hpp - Cross-thread cancel for in-flight llama HTTP. */
#pragma once

#include <atomic>
#include <cstdint>

namespace phoenix {
namespace inference {

inline std::atomic<uint64_t> &abortEpoch() {
  static std::atomic<uint64_t> epoch{0};
  return epoch;
}

inline std::atomic<bool> &shutdownFlag() {
  static std::atomic<bool> flag{false};
  return flag;
}

inline uint64_t currentAbortEpoch() {
  return abortEpoch().load(std::memory_order_acquire);
}

inline void requestAbort() {
  abortEpoch().fetch_add(1, std::memory_order_acq_rel);
}

inline void requestShutdownAbort() {
  shutdownFlag().store(true, std::memory_order_release);
  requestAbort();
}

inline bool shutdownRequested() {
  return shutdownFlag().load(std::memory_order_acquire);
}

inline bool shouldAbort(uint64_t startEpoch) {
  return shutdownRequested() || currentAbortEpoch() != startEpoch;
}

inline void clearShutdownAbortForTesting() {
  shutdownFlag().store(false, std::memory_order_release);
}

} /* namespace inference */
} /* namespace phoenix */
