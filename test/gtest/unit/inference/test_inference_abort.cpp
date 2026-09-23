/* report/stop abort must not stick onto the next interactive chat. */

#include "inference_abort.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

using phoenix::inference::InFlightGenerateGuard;
using phoenix::inference::beginInteractiveEpoch;
using phoenix::inference::currentAbortEpoch;
using phoenix::inference::inFlightGenerates;
using phoenix::inference::pendingCancelNotifies;
using phoenix::inference::requestAbort;
using phoenix::inference::resetAbortStateForTesting;
using phoenix::inference::setAbortNotify;
using phoenix::inference::shouldAbort;
using phoenix::inference::waitForAbortNotifyIdle;
using phoenix::inference::lastTickIdleWaitMs;
using phoenix::inference::lastTickCancelDrainWaitMs;
using phoenix::inference::llamaSlotsHeld;
using phoenix::inference::waitForGenerateIdle;
using phoenix::inference::waitThenCancelLastTickIfBusy;

class InferenceAbortTest : public ::testing::Test {
 protected:
  void SetUp() override { resetAbortStateForTesting(); }
  void TearDown() override { resetAbortStateForTesting(); }
};

TEST_F(InferenceAbortTest, CompleteWithNoInFlightDoesNotNotify) {
  static std::atomic<int> fired{0};
  fired.store(0);
  setAbortNotify([]() { fired.fetch_add(1); });
  requestAbort();
  EXPECT_EQ(fired.load(), 0);
  EXPECT_EQ(pendingCancelNotifies().load(), 0);
  const uint64_t epoch = beginInteractiveEpoch(200);
  EXPECT_FALSE(shouldAbort(epoch));
}

TEST_F(InferenceAbortTest, InFlightAbortNotifiesThenInteractiveIsFresh) {
  static std::atomic<int> fired{0};
  fired.store(0);
  setAbortNotify([]() {
    fired.fetch_add(1);
    phoenix::inference::notifyAbortFinished();
  });
  {
    InFlightGenerateGuard guard;
    EXPECT_EQ(inFlightGenerates().load(), 1);
    const uint64_t old = currentAbortEpoch();
    requestAbort();
    EXPECT_EQ(fired.load(), 1);
    EXPECT_TRUE(shouldAbort(old));
  }
  EXPECT_EQ(inFlightGenerates().load(), 0);
  EXPECT_TRUE(waitForAbortNotifyIdle(200));
  const uint64_t chatEpoch = beginInteractiveEpoch(200);
  EXPECT_FALSE(shouldAbort(chatEpoch));
  EXPECT_EQ(pendingCancelNotifies().load(), 0);
}

TEST_F(InferenceAbortTest, InteractiveDoesNotInheritPriorEpoch) {
  {
    InFlightGenerateGuard guard;
    requestAbort();
  }
  const uint64_t chatEpoch = beginInteractiveEpoch(200);
  EXPECT_FALSE(shouldAbort(chatEpoch));
  EXPECT_EQ(chatEpoch, currentAbortEpoch());
}

TEST_F(InferenceAbortTest, LastTickWaitCoversBoard256Tokens) {
  EXPECT_EQ(lastTickIdleWaitMs(256), 256 * 2000 + 180000);
  EXPECT_EQ(lastTickIdleWaitMs(64), 256 * 2000 + 180000);
  EXPECT_EQ(lastTickIdleWaitMs(2048), 15 * 60 * 1000);
  EXPECT_LE(lastTickIdleWaitMs(256), 15 * 60 * 1000);
  EXPECT_EQ(lastTickCancelDrainWaitMs(), 8 * 60 * 1000);
  EXPECT_EQ(phoenix::inference::lastTickCancelSettleMs(), 20 * 1000);
}

TEST_F(InferenceAbortTest, WaitIdleWaitsForSlotsHeld) {
  llamaSlotsHeld().store(1);
  std::thread releaser([]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    llamaSlotsHeld().store(0);
  });
  const auto t0 = std::chrono::steady_clock::now();
  EXPECT_TRUE(waitForGenerateIdle(800));
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
  releaser.join();
  EXPECT_GE(ms, 50);
  EXPECT_EQ(llamaSlotsHeld().load(), 0);
}

TEST_F(InferenceAbortTest, WaitIdleSettleRechecksSlots) {
  std::thread grabber([]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    llamaSlotsHeld().store(1);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    llamaSlotsHeld().store(0);
  });
  const auto t0 = std::chrono::steady_clock::now();
  EXPECT_TRUE(waitForGenerateIdle(1200));
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
  grabber.join();
  EXPECT_GE(ms, 220) << "settle must re-check; a raced slot grab is not idle";
  EXPECT_EQ(llamaSlotsHeld().load(), 0);
}

TEST_F(InferenceAbortTest, WaitIdleDoesNotNotifyAbort) {
  static std::atomic<int> fired{0};
  fired.store(0);
  setAbortNotify([]() { fired.fetch_add(1); });
  auto *guard = new InFlightGenerateGuard();
  std::thread releaser([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    delete guard;
  });
  EXPECT_TRUE(waitForGenerateIdle(500));
  releaser.join();
  EXPECT_EQ(fired.load(), 0);
  EXPECT_EQ(inFlightGenerates().load(), 0);
}

TEST_F(InferenceAbortTest, WaitThenCancelDoesNotNotifyWhenIdleSoon) {
  static std::atomic<int> fired{0};
  fired.store(0);
  setAbortNotify([]() { fired.fetch_add(1); });
  auto *guard = new InFlightGenerateGuard();
  std::thread releaser([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    delete guard;
  });
  const auto info = waitThenCancelLastTickIfBusy(500, 400, 1);
  releaser.join();
  EXPECT_TRUE(info.idle);
  EXPECT_FALSE(info.cancelled);
  EXPECT_EQ(fired.load(), 0);
  EXPECT_EQ(inFlightGenerates().load(), 0);
}

TEST_F(InferenceAbortTest, WaitThenCancelCancelsWhenStillBusyThenInteractiveFresh) {
  static std::atomic<int> fired{0};
  fired.store(0);
  setAbortNotify([]() {
    fired.fetch_add(1);
    phoenix::inference::notifyAbortFinished();
  });
  const uint64_t start = currentAbortEpoch();
  auto *guard = new InFlightGenerateGuard();
  llamaSlotsHeld().store(1);
  std::thread worker([&]() {
    while (!shouldAbort(start))
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    delete guard;
    llamaSlotsHeld().store(0);
  });
  const auto info = waitThenCancelLastTickIfBusy(40, 400, 1);
  worker.join();
  EXPECT_TRUE(info.cancelled);
  EXPECT_TRUE(info.idle);
  EXPECT_EQ(fired.load(), 1);
  EXPECT_EQ(inFlightGenerates().load(), 0);
  EXPECT_EQ(llamaSlotsHeld().load(), 0);
  const uint64_t chatEpoch = beginInteractiveEpoch(200);
  EXPECT_FALSE(shouldAbort(chatEpoch));
  EXPECT_EQ(pendingCancelNotifies().load(), 0);
}

TEST_F(InferenceAbortTest, BeginInteractiveWaitsForInFlightThenIsFresh) {
  static std::atomic<int> fired{0};
  fired.store(0);
  setAbortNotify([]() {
    fired.fetch_add(1);
    phoenix::inference::notifyAbortFinished();
  });
  auto *guard = new InFlightGenerateGuard();
  requestAbort();
  EXPECT_EQ(fired.load(), 1);
  EXPECT_EQ(pendingCancelNotifies().load(), 0);
  EXPECT_EQ(inFlightGenerates().load(), 1);
  std::atomic<bool> released{false};
  std::thread releaser([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    delete guard;
    released.store(true);
  });
  const auto t0 = std::chrono::steady_clock::now();
  const uint64_t chatEpoch = beginInteractiveEpoch(500);
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
  releaser.join();
  EXPECT_TRUE(released.load());
  EXPECT_GE(ms, 50);
  EXPECT_EQ(inFlightGenerates().load(), 0);
  EXPECT_FALSE(shouldAbort(chatEpoch));
}
