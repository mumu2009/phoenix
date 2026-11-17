/* Regression pin for the 2026-11-17 heap-corruption fix (0xC0000374).
 *
 * Root cause: OnlineResearcher::fallback() re-entered RuntimeState's mutable
 * research state (MemeGraph graph_, TensorEngine tensor_, PatternMatrix
 * pattern_) through mapWordsToMemesOrdered()/runPropagation() without holding
 * RuntimeState::stateMu_, the mutex every other caller of those helpers holds.
 * With autonomyLoop.parallelMissions=true, mission threads reached it in
 * parallel through collectAddonOffers -> invokeAddonOnlineLookup ->
 * onlineLookup -> lookup -> fallback; the 08:12/08:14/08:15 dumps each show
 * 2-3 threads inside PatternMatrix::rebuild() at once, corrupting the
 * nlohmann::json heap of PatternMatrix/GraphTensorBridge.
 *
 * The fix routes the fallback through
 * phoenix::state_guard::withResearchStateLock(RuntimeState::researchStateMutex()).
 * RuntimeState itself lives in the main.cpp unity translation unit, which the
 * gtest runner does not link, so the reachable, testable seam is the guard:
 * these tests pin the contract the production call site depends on:
 *   1. it acquires the SAME mutex object RuntimeState's own paths use
 *      (a plain lock_guard on that mutex excludes it and vice versa),
 *   2. concurrent guarded callbacks never overlap,
 *   3. the mutex is released on every exit path (including exceptions).
 */
#include "autonomy_stack.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using phoenix::state_guard::ResearchStateGuard;
using phoenix::state_guard::withResearchStateLock;

/* Spin-waits (bounded) until pred() is true so the tests are deterministic
 * instead of sleep-based. Returns false on timeout. */
template <class Pred>
bool waitFor(Pred pred, int timeoutMs = 5000) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  while (!pred()) {
    if (std::chrono::steady_clock::now() > deadline)
      return false;
    std::this_thread::yield();
  }
  return true;
}

} // namespace

/* 1. The guarded region and a plain lock_guard on the same mutex are mutually
 *    exclusive - this is what makes the fallback exclude processInput()'s
 *    runPropagation() (and vice versa). */
TEST(ResearchStateGuardTest, SerialisesWithPlainLockGuardOnSameMutex) {
  std::mutex mu;
  std::atomic<bool> researchInside{false};
  std::atomic<bool> plainEntered{false};
  std::atomic<bool> releaseResearch{false};

  std::thread research([&] {
    withResearchStateLock(mu, [&] {
      researchInside.store(true, std::memory_order_release);
      while (!releaseResearch.load(std::memory_order_acquire))
        std::this_thread::yield();
      EXPECT_FALSE(plainEntered.load());
    });
  });
  ASSERT_TRUE(waitFor([&] { return researchInside.load(); }));

  std::thread plain([&] {
    std::lock_guard<std::mutex> lk(mu);
    plainEntered.store(true, std::memory_order_release);
  });

  /* The plain locker must stay blocked while the research callback runs. */
  std::this_thread::sleep_for(std::chrono::milliseconds(80));
  EXPECT_FALSE(plainEntered.load());

  releaseResearch.store(true);
  research.join();
  plain.join();
  EXPECT_TRUE(plainEntered.load());
}

/* 2. Parallel guarded callbacks are serialised (the parallel-mission case:
 *    previously N fallbacks executed PatternMatrix::rebuild concurrently). */
TEST(ResearchStateGuardTest, ConcurrentResearchCallbacksNeverOverlap) {
  std::mutex mu;
  constexpr int kThreads = 4;
  constexpr int kIterations = 400;
  std::atomic<int> inside{0};
  std::atomic<int> maxInside{0};
  std::atomic<int> completed{0};

  auto worker = [&] {
    for (int i = 0; i < kIterations; ++i) {
      withResearchStateLock(mu, [&] {
        const int now = inside.fetch_add(1) + 1;
        int prev = maxInside.load();
        while (now > prev &&
               !maxInside.compare_exchange_weak(prev, now)) {
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        inside.fetch_sub(1);
        completed.fetch_add(1);
      });
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int i = 0; i < kThreads; ++i)
    threads.emplace_back(worker);
  for (auto &t : threads)
    t.join();

  EXPECT_EQ(completed.load(), kThreads * kIterations);
  EXPECT_EQ(maxInside.load(), 1)
      << "two research callbacks entered the protected region at once";
  EXPECT_EQ(inside.load(), 0);
}

/* 3a. A throwing callback must not leak the mutex (otherwise one bad
 *     propagation would hang every chat/mission/study path forever). */
TEST(ResearchStateGuardTest, ReleasesLockWhenCallbackThrows) {
  std::mutex mu;
  EXPECT_THROW(
      withResearchStateLock(mu, []() -> int {
        throw std::runtime_error("propagation failure");
      }),
      std::runtime_error);

  /* Immediately lockable again from another thread. */
  std::atomic<bool> acquired{false};
  std::thread t([&] {
    std::lock_guard<std::mutex> lk(mu);
    acquired.store(true);
  });
  t.join();
  EXPECT_TRUE(acquired.load());
}

/* 3b. RAII ownership reporting used by the call site comment/contract. */
TEST(ResearchStateGuardTest, OwnsLockTracksOwnership) {
  std::mutex mu;
  ResearchStateGuard guard(mu);
  EXPECT_TRUE(guard.owns_lock());
  guard.unlock();
  EXPECT_FALSE(guard.owns_lock());
}
