/* inference_abort.hpp - Cross-thread cancel for in-flight llama HTTP. */
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

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

inline std::atomic<int> &inFlightGenerates() {
  static std::atomic<int> n{0};
  return n;
}

inline std::atomic<int> &pendingCancelNotifies() {
  static std::atomic<int> n{0};
  return n;
}

/* chatWithLlamaCpp takes llamaSlotsInUse_ before httpRequest bumps
   inFlight. Report that only watched inFlight returned in ~150ms
   (retest8 board reportMs=182) while --parallel 1 was already leased
   or about to be leased by the next tick. */
inline std::atomic<int> &llamaSlotsHeld() {
  static std::atomic<int> n{0};
  return n;
}

/* Last mission complete: refuse new lowPriority / mission generates so
   the loop cannot grab the slot after waitForGenerateIdle returns.
   Interactive chat is not lowPriority and still proceeds. */
inline std::atomic<bool> &blockLowPriorityGenerates() {
  static std::atomic<bool> flag{false};
  return flag;
}

inline uint64_t currentAbortEpoch() {
  return abortEpoch().load(std::memory_order_acquire);
}

/* Optional side effect after the epoch bump (e.g. POST /phx/cancel).
   Must not RST llama or call /health. The notify itself must ignore
   shouldAbort() or it will cancel its own HTTP. */
inline std::atomic<void (*)()> &abortNotify() {
  static std::atomic<void (*)()> fn{nullptr};
  return fn;
}

inline void setAbortNotify(void (*fn)()) {
  abortNotify().store(fn, std::memory_order_release);
}

inline void noteInFlightGenerate() {
  inFlightGenerates().fetch_add(1, std::memory_order_acq_rel);
}

inline void noteInFlightGenerateDone() {
  inFlightGenerates().fetch_sub(1, std::memory_order_acq_rel);
}

struct InFlightGenerateGuard {
  InFlightGenerateGuard() { noteInFlightGenerate(); }
  ~InFlightGenerateGuard() { noteInFlightGenerateDone(); }
  InFlightGenerateGuard(const InFlightGenerateGuard &) = delete;
  InFlightGenerateGuard &operator=(const InFlightGenerateGuard &) = delete;
};

inline void notifyAbortFinished() {
  int cur = pendingCancelNotifies().load(std::memory_order_acquire);
  while (cur > 0 && !pendingCancelNotifies().compare_exchange_weak(
                        cur, cur - 1, std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
  }
}

/* Client-side cancel only. HTTP callers must drain-or-return; they must
   not RST the unique llama-server or send /health.
   If nothing is in flight, bump the epoch so the finished caller sees
   abort, but do not POST /phx/cancel or release a llama slot — that
   leftover cancel is what emptied the next short ask after complete. */
inline void requestAbort() {
  abortEpoch().fetch_add(1, std::memory_order_acq_rel);
  if (inFlightGenerates().load(std::memory_order_acquire) <= 0)
    return;
  auto fn = abortNotify().load(std::memory_order_acquire);
  if (!fn)
    return;
  pendingCancelNotifies().fetch_add(1, std::memory_order_acq_rel);
  fn();
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

inline bool waitForAbortNotifyIdle(int timeoutMs = 4000) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  while (pendingCancelNotifies().load(std::memory_order_acquire) > 0) {
    if (std::chrono::steady_clock::now() >= deadline)
      return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return true;
}

/* Board last-tick at n_predict=256 is ~2s/token. A fat prompt plus
   reporting at the tokBudget line (generate just starting) can exceed
   256×2s+60s: r8slot-rdk-131748 waited 572s and still had slots=1.
   Floor 256×2s + 3min prefill; cap 15 minutes. After this budget the
   caller must /phx/cancel rather than return still-busy. */
inline int lastTickIdleWaitMs(int nPredict = 256) {
  int n = nPredict;
  if (n < 16)
    n = 16;
  int need = n * 2000 + 180000;
  const int floorMs = 256 * 2000 + 180000;
  if (need < floorMs)
    need = floorMs;
  const int capMs = 15 * 60 * 1000;
  if (need > capMs)
    need = capMs;
  return need;
}

/* After /phx/cancel: wait until inFlight=0 and slotsHeld=0. Live llama
   may not honor /phx/cancel (not rebuilt). The aborting HTTP must drain
   leftover tokens synchronously so these counters stay held until llama
   actually finishes or this cap elapses. 8 min covers the rest of a
   256-token board tick after the 692s budget. */
inline int lastTickCancelDrainWaitMs() { return 8 * 60 * 1000; }

/* Small extra settle after the generate socket is truly idle. */
inline int lastTickCancelSettleMs() { return 20 * 1000; }

/* gtest / standalone: 0 means use production default. */
inline std::atomic<int> &testLastTickIdleWaitMsOverride() {
  static std::atomic<int> n{0};
  return n;
}
inline std::atomic<int> &testLastTickCancelDrainWaitMsOverride() {
  static std::atomic<int> n{0};
  return n;
}
inline std::atomic<int> &testLastTickCancelSettleMsOverride() {
  static std::atomic<int> n{0};
  return n;
}

inline int effectiveLastTickIdleWaitMs(int nPredict = 256) {
  const int over = testLastTickIdleWaitMsOverride().load(std::memory_order_acquire);
  if (over > 0)
    return over;
  return lastTickIdleWaitMs(nPredict);
}

inline int effectiveLastTickCancelDrainWaitMs() {
  const int over =
      testLastTickCancelDrainWaitMsOverride().load(std::memory_order_acquire);
  if (over > 0)
    return over;
  return lastTickCancelDrainWaitMs();
}

inline int effectiveLastTickCancelSettleMs() {
  const int over =
      testLastTickCancelSettleMsOverride().load(std::memory_order_acquire);
  if (over > 0)
    return over;
  return lastTickCancelSettleMs();
}

struct LastTickReleaseInfo {
  bool idle = false;
  bool cancelled = false;
  int waitMs = 0;
  int drainMs = 0;
  int waitInFlight = 0;
  int waitSlots = 0;
  int inFlight = 0;
  int slots = 0;
};

inline bool generateChannelBusy() {
  return inFlightGenerates().load(std::memory_order_acquire) > 0 ||
         llamaSlotsHeld().load(std::memory_order_acquire) > 0 ||
         pendingCancelNotifies().load(std::memory_order_acquire) > 0;
}

/* Last mission tick may still own --parallel 1 after report/stop.
   inFlight hits 0 a few instructions before the slot is released, and
   the next loop snap can take the slot after file-edit (inFlight=0)
   before HTTP starts. Wait until inFlight=0 AND the slot is free;
   settle and re-check so a raced next tick cannot sneak through. */
inline bool waitForGenerateIdle(int timeoutMs = 30000) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  while (true) {
    if (!generateChannelBusy()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(150));
      if (!generateChannelBusy())
        return true;
    }
    if (std::chrono::steady_clock::now() >= deadline)
      return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}

/* Wait the n_predict budget. If last-tick still holds the slot, POST
   /phx/cancel (drain+FIN, no RST) and wait until inFlight=0 AND
   slotsHeld=0. Do not bump the epoch again after idle: interactive
   chat snapshots after this returns, so it must not inherit cancel. */
inline LastTickReleaseInfo waitThenCancelLastTickIfBusy(int idleWaitMs,
                                                      int drainWaitMs,
                                                      int settleMs) {
  LastTickReleaseInfo info;
  const auto t0 = std::chrono::steady_clock::now();
  info.idle = waitForGenerateIdle(idleWaitMs);
  info.waitMs = static_cast<int>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - t0)
          .count());
  info.waitInFlight = inFlightGenerates().load(std::memory_order_acquire);
  info.waitSlots = llamaSlotsHeld().load(std::memory_order_acquire);
  info.inFlight = info.waitInFlight;
  info.slots = info.waitSlots;
  if (info.idle || !generateChannelBusy()) {
    info.idle = !generateChannelBusy();
    return info;
  }
  requestAbort();
  info.cancelled = true;
  const auto t1 = std::chrono::steady_clock::now();
  info.idle = waitForGenerateIdle(drainWaitMs);
  int drained = static_cast<int>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - t1)
          .count());
  if (settleMs > drained)
    std::this_thread::sleep_for(std::chrono::milliseconds(settleMs - drained));
  const int remain = drainWaitMs > drained ? drainWaitMs - drained : 200;
  if (!waitForGenerateIdle(std::max(200, remain)))
    info.idle = !generateChannelBusy();
  else
    info.idle = true;
  info.drainMs = static_cast<int>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - t1)
          .count());
  info.inFlight = inFlightGenerates().load(std::memory_order_acquire);
  info.slots = llamaSlotsHeld().load(std::memory_order_acquire);
  return info;
}

/* Interactive chat must not inherit the mission report/stop abort.
   Wait until in-flight generate and /phx/cancel are gone, then snapshot. */
inline uint64_t beginInteractiveEpoch(int timeoutMs = 30000) {
  waitForGenerateIdle(timeoutMs);
  return currentAbortEpoch();
}

inline void clearShutdownAbortForTesting() {
  shutdownFlag().store(false, std::memory_order_release);
}

inline void resetAbortStateForTesting() {
  abortEpoch().store(0, std::memory_order_release);
  shutdownFlag().store(false, std::memory_order_release);
  inFlightGenerates().store(0, std::memory_order_release);
  pendingCancelNotifies().store(0, std::memory_order_release);
  llamaSlotsHeld().store(0, std::memory_order_release);
  blockLowPriorityGenerates().store(false, std::memory_order_release);
  abortNotify().store(nullptr, std::memory_order_release);
  testLastTickIdleWaitMsOverride().store(0, std::memory_order_release);
  testLastTickCancelDrainWaitMsOverride().store(0, std::memory_order_release);
  testLastTickCancelSettleMsOverride().store(0, std::memory_order_release);
}

} /* namespace inference */
} /* namespace phoenix */
