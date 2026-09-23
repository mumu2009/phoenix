/* test_autonomy_loop.cpp - Long-term autonomous loop, persistence and
   human interjection (插话) for the autonomy stack. */

#include "autonomy_stack.hpp"

#include "emergency_stop.hpp"
#include "inference_abort.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

using autonomy::CognitionAutonomyManager;
using autonomy::buildCognitionAutonomySeedPayload;
using autonomy::json;

namespace {

class AutonomyLoopTest : public ::testing::Test {
protected:
    std::unique_ptr<CognitionAutonomyManager> mgr_;
    void SetUp() override {
        phoenix::inference::resetAbortStateForTesting();
        mgr_ = std::make_unique<CognitionAutonomyManager>();
    }
    void TearDown() override {
        mgr_.reset();
        phoenix::inference::resetAbortStateForTesting();
    }
};

}  // namespace

TEST_F(AutonomyLoopTest, InterjectQueuesAndIterateConsumes) {
    auto r = mgr_->interject(json{{"text", "remember to double-check every answer"}});
    ASSERT_TRUE(r.value("ok", false));
    EXPECT_EQ(r.value("queued", 0), 1);

    mgr_->observe(buildCognitionAutonomySeedPayload("inj-sess", "mission", 0.5), json{});
    auto iter = mgr_->iterate(json{{"sessionId", "inj-sess"}}, json{});
    EXPECT_TRUE(iter.value("ok", false));
    EXPECT_EQ(iter["result"].value("interjectionsConsumed", 0), 1);
    const std::string mod = iter["result"].value("cognitionModulation", "");
    EXPECT_NE(mod.find("double-check"), std::string::npos);

    /* consumed exactly once */
    auto iter2 = mgr_->iterate(json{{"sessionId", "inj-sess"}}, json{});
    EXPECT_EQ(iter2["result"].value("interjectionsConsumed", 0), 0);
}

TEST_F(AutonomyLoopTest, InterjectAmendGoalRedirectsRunningMission) {
    mgr_->assignMission(json{{"enabled", true},
                             {"goal", "original"},
                             {"painGainPerSec", 1.0},
                             {"maxPain", 1.0}});
    auto st = mgr_->missionStatus();
    const uint64_t startMs = st["result"]["stats"]["mission"].value("startMs", 0ull);
    EXPECT_GT(startMs, 0ull);

    auto r = mgr_->interject(json{{"text", "change of plans"},
                                  {"amendGoal", "redirected"}});
    EXPECT_TRUE(r.value("ok", false));
    EXPECT_TRUE(r.value("goalAmended", false));

    st = mgr_->missionStatus();
    EXPECT_EQ(st["result"]["stats"]["mission"].value("goal", std::string()), "redirected");
    EXPECT_EQ(st["result"]["stats"]["mission"].value("state", 0), 1); /* still Running */
    EXPECT_EQ(st["result"]["stats"]["mission"].value("startMs", 0ull), startMs);
}

TEST_F(AutonomyLoopTest, InterjectAmendGoalWithoutMissionWarns) {
    auto r = mgr_->interject(json{{"text", "note"}, {"amendGoal", "x"}});
    EXPECT_TRUE(r.value("ok", false));
    EXPECT_FALSE(r.value("goalAmended", true));
    EXPECT_TRUE(r.contains("warning"));
}

TEST_F(AutonomyLoopTest, AutonomyLoopTicksWithoutExternalIterate) {
    auto cfg = mgr_->configureAutonomyLoop(json{{"enabled", true},
                                                {"intervalSec", 1},
                                                {"maxStepsPerTick", 2},
                                                {"persistEveryTicks", 10000}});
    EXPECT_TRUE(cfg.value("ok", false));

    auto start = mgr_->startAutonomyLoop();
    ASSERT_TRUE(start.value("ok", false)) << start.dump();

    std::this_thread::sleep_for(std::chrono::milliseconds(2600));

    auto st = mgr_->autonomyLoopStatus();
    EXPECT_TRUE(st["result"].value("running", false));
    EXPECT_GE(st["result"].value("tickCount", 0), 1);

    auto status = mgr_->status();
    EXPECT_GE(status["result"].value("iteration", 0), 1)
        << "the heartbeat must run the cycle with no external iterate calls";

    auto stop = mgr_->stopAutonomyLoop();
    EXPECT_TRUE(stop.value("ok", false));
    EXPECT_TRUE(stop["result"].value("stopping", false));
    for (int i = 0; i < 40 && mgr_->autonomyLoopStatus()["result"].value("running", false); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(mgr_->autonomyLoopStatus()["result"].value("running", true));
}

TEST_F(AutonomyLoopTest, RestartsAfterEstopBreaksLoopThread) {
    auto cfg = mgr_->configureAutonomyLoop(json{{"enabled", true},
                                                {"intervalSec", 1},
                                                {"maxStepsPerTick", 1},
                                                {"persistEveryTicks", 10000}});
    ASSERT_TRUE(cfg.value("ok", false));

    ASSERT_TRUE(mgr_->startAutonomyLoop().value("ok", false));
    std::this_thread::sleep_for(std::chrono::milliseconds(2200));
    EXPECT_TRUE(mgr_->autonomyLoopStatus()["result"].value("running", false));
    const uint64_t ticksBefore =
        mgr_->autonomyLoopStatus()["result"].value("tickCount", 0ull);
    EXPECT_GE(ticksBefore, 1u);

    /* Simulate the production bug: loopRun breaks on latched E-stop while
       loopStop_ is still false, leaving a joinable-but-dead thread that
       blocked startAutonomyLoop() from spawning a replacement. */
    phoenix::safety::EmergencyStop::instance().press("gtest-estop");
  std::this_thread::sleep_for(std::chrono::milliseconds(400));
  phoenix::safety::EmergencyStop::instance().resetForTesting();

  ASSERT_TRUE(mgr_->startAutonomyLoop().value("ok", false))
      << "must restart after the loop thread exits";
  std::this_thread::sleep_for(std::chrono::milliseconds(2200));
  EXPECT_TRUE(mgr_->autonomyLoopStatus()["result"].value("running", false));
  EXPECT_GT(mgr_->autonomyLoopStatus()["result"].value("tickCount", 0ull),
            ticksBefore);

  mgr_->stopAutonomyLoop();
}

TEST_F(AutonomyLoopTest, StartAutonomyLoopReturnsWithoutJoiningStuckIterate) {
    mgr_->setMissionDeliberator([](const std::string &, const std::string &, int,
                                   const std::string &) -> std::string {
        std::this_thread::sleep_for(std::chrono::milliseconds(2500));
        return "slow-deliberate";
    });
    mgr_->configureAutonomyLoop(json{{"enabled", true},
                                     {"intervalSec", 1},
                                     {"maxStepsPerTick", 1},
                                     {"persistEveryTicks", 10000}});
    mgr_->assignMission(json{{"enabled", true},
                             {"id", "start-join-1"},
                             {"goal", "occupy the loop"}});
    ASSERT_TRUE(mgr_->startAutonomyLoop(json{{"restoreState", false}}).value("ok", false));
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    mgr_->stopAutonomyLoop();
    const auto t0 = std::chrono::steady_clock::now();
    auto start = mgr_->startAutonomyLoop(json{{"restoreState", false}});
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    EXPECT_TRUE(start.value("ok", false));
    EXPECT_LT(ms, 800) << "start must not join a multi-second iterate";
    mgr_->stopAutonomyLoop();
}

TEST_F(AutonomyLoopTest, StopAutonomyLoopReturnsWithoutJoiningStuckIterate) {
    mgr_->setMissionDeliberator([](const std::string &, const std::string &, int,
                                   const std::string &) -> std::string {
        std::this_thread::sleep_for(std::chrono::milliseconds(2500));
        return "slow-deliberate";
    });
    mgr_->configureAutonomyLoop(json{{"enabled", true},
                                     {"intervalSec", 1},
                                     {"maxStepsPerTick", 1},
                                     {"persistEveryTicks", 10000}});
    mgr_->assignMission(json{{"enabled", true},
                             {"id", "stop-join-1"},
                             {"goal", "occupy the loop"}});
    ASSERT_TRUE(mgr_->startAutonomyLoop(json{{"restoreState", false}}).value("ok", false));
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const auto t0 = std::chrono::steady_clock::now();
    auto stop = mgr_->stopAutonomyLoop();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    EXPECT_TRUE(stop.value("ok", false));
    EXPECT_LT(ms, 800) << "stop must not join a multi-second iterate";
    for (int i = 0; i < 80 && mgr_->autonomyLoopStatus()["result"].value("running", false); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

TEST_F(AutonomyLoopTest, FileEditEmptyReturnStillTicksAgain) {
    std::atomic<int> calls{0};
    mgr_->setMissionDeliberator([&](const std::string &, const std::string &prior,
                                    int, const std::string &) -> std::string {
        const int n = ++calls;
        if (n == 1)
            return std::string(); /* file-edit already wrote; same as soak */
        return "second-tick-revision-" + std::to_string(prior.size());
    });
    mgr_->configureAutonomyLoop(json{{"enabled", true},
                                     {"intervalSec", 1},
                                     {"maxStepsPerTick", 1},
                                     {"persistEveryTicks", 10000}});
    mgr_->assignMission(json{{"enabled", true},
                             {"id", "loop-revise-1"},
                             {"goal", "keep writing the draft"}});
    ASSERT_TRUE(mgr_->startAutonomyLoop(json{{"restoreState", false}}).value("ok", false));
    for (int i = 0; i < 80 && calls.load() < 2; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    mgr_->stopAutonomyLoop();
    EXPECT_GE(calls.load(), 2) << "150-char file-edit must not stop the loop";
    const std::string body =
        mgr_->missionStatus()["result"]["stats"]["mission"].value(
            "deliverable", std::string());
    EXPECT_NE(body.find("second-tick-revision"), std::string::npos);
}

TEST_F(AutonomyLoopTest, ReportOutcomeDoesNotAbortLastTick) {
    phoenix::inference::resetAbortStateForTesting();
    static std::atomic<int> fired{0};
    fired.store(0);
    phoenix::inference::setAbortNotify([]() { fired.fetch_add(1); });
    auto *guard = new phoenix::inference::InFlightGenerateGuard();
    std::thread releaser([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
        delete guard;
    });
    mgr_->assignMission(json{{"enabled", true},
                             {"id", "report-no-abort-1"},
                             {"goal", "keep last tick"}});
    auto r = mgr_->reportMissionOutcome(json{{"goalAchieved", true},
                                             {"missionId", "report-no-abort-1"}});
    releaser.join();
    EXPECT_TRUE(r.value("ok", false));
    EXPECT_EQ(fired.load(), 0) << "report must wait the last tick out, not /phx/cancel";
    phoenix::inference::resetAbortStateForTesting();
}

TEST_F(AutonomyLoopTest, ReportOutcomeCancelsAfterBudgetThenInteractiveFresh) {
    phoenix::inference::resetAbortStateForTesting();
    phoenix::inference::testLastTickIdleWaitMsOverride().store(40);
    phoenix::inference::testLastTickCancelDrainWaitMsOverride().store(400);
    phoenix::inference::testLastTickCancelSettleMsOverride().store(1);
    static std::atomic<int> fired{0};
    fired.store(0);
    phoenix::inference::setAbortNotify([]() {
        fired.fetch_add(1);
        phoenix::inference::notifyAbortFinished();
    });
    const uint64_t start = phoenix::inference::currentAbortEpoch();
    auto *guard = new phoenix::inference::InFlightGenerateGuard();
    phoenix::inference::llamaSlotsHeld().store(1);
    std::thread worker([&]() {
        while (!phoenix::inference::shouldAbort(start))
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        delete guard;
        phoenix::inference::llamaSlotsHeld().store(0);
    });
    mgr_->assignMission(json{{"enabled", true},
                             {"id", "report-cancel-1"},
                             {"goal", "cancel last tick after budget"}});
    auto r = mgr_->reportMissionOutcome(json{{"goalAchieved", true},
                                             {"missionId", "report-cancel-1"}});
    worker.join();
    EXPECT_TRUE(r.value("ok", false));
    EXPECT_EQ(fired.load(), 1) << "timeout must /phx/cancel the last tick";
    EXPECT_EQ(phoenix::inference::inFlightGenerates().load(), 0);
    EXPECT_EQ(phoenix::inference::llamaSlotsHeld().load(), 0);
    const uint64_t chatEpoch = phoenix::inference::beginInteractiveEpoch(200);
    EXPECT_FALSE(phoenix::inference::shouldAbort(chatEpoch))
        << "next chat must not inherit the report cancel epoch";
    phoenix::inference::resetAbortStateForTesting();
}

TEST_F(AutonomyLoopTest, ReportOutcomeRetargetsDefaultAndStopsWhenLastDone) {
    mgr_->configureAutonomyLoop(json{{"enabled", true},
                                     {"intervalSec", 1},
                                     {"maxStepsPerTick", 1},
                                     {"persistEveryTicks", 10000}});
    mgr_->assignMission(json{{"enabled", true},
                             {"id", "keep-live"},
                             {"goal", "first"}});
    mgr_->assignMission(json{{"enabled", true},
                             {"id", "done-soon"},
                             {"goal", "second"}});
    EXPECT_EQ(mgr_->missionStatus()["result"].value("defaultMissionId", std::string()),
              "done-soon");
    auto r = mgr_->reportMissionOutcome(json{{"goalAchieved", true},
                                             {"missionId", "done-soon"}});
    EXPECT_TRUE(r.value("ok", false));
    EXPECT_EQ(mgr_->missionStatus()["result"].value("defaultMissionId", std::string()),
              "keep-live");
    ASSERT_TRUE(mgr_->startAutonomyLoop(json{{"restoreState", false}}).value("ok", false));
    auto last = mgr_->reportMissionOutcome(json{{"goalAchieved", true},
                                                {"missionId", "keep-live"}});
    EXPECT_TRUE(last.value("ok", false));
    EXPECT_TRUE(mgr_->missionStatus()["result"].value("defaultMissionId", std::string()).empty());
    for (int i = 0; i < 40 && mgr_->autonomyLoopStatus()["result"].value("running", false); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(mgr_->autonomyLoopStatus()["result"].value("running", true));
}

TEST_F(AutonomyLoopTest, IterateWithoutSessionBindsCurrentMission) {
    phoenix::memory::ScopedTrainableMemory::instance().resetForTest();
    mgr_->configureAgi(json{{"enabled", true}, {"dim", 8}});
    mgr_->assignMission(json{{"enabled", true},
                             {"id", "scope-bind-1"},
                             {"goal", "keep isolated"}});
    auto r = mgr_->iterate(json::object(), json{});
    ASSERT_TRUE(r.value("ok", false)) << r.dump();
    EXPECT_EQ(r["result"].value("memoryScope", std::string()), "mission:scope-bind-1");
    auto plan = mgr_->agiPlan();
    EXPECT_EQ(plan["result"].value("memoryScope", std::string()),
              "mission:scope-bind-1");
}

TEST_F(AutonomyLoopTest, IterateWithoutSessionDoesNotSweepForeignChat) {
    phoenix::memory::ScopedTrainableMemory::instance().resetForTest();
    mgr_->observe(buildCognitionAutonomySeedPayload("foreign-chat", "other goal", 0.8),
                  json{});
    auto r = mgr_->iterate(json::object(), json{});
    ASSERT_TRUE(r.value("ok", false)) << r.dump();
    EXPECT_EQ(r["result"].value("memoryScope", std::string()),
              "chat:__autonomy_heartbeat__");
    const auto &sess = r["result"].value("sessions", json::array());
    for (const auto &s : sess) {
        EXPECT_NE(s.value("sessionId", std::string()), "foreign-chat");
    }
}

TEST_F(AutonomyLoopTest, AssignMissionResetsIterationCounter) {
    mgr_->importState(json{{"sessions", json::object()},
                           {"iteration", 1007},
                           {"missionEnabled", false}});
    EXPECT_EQ(mgr_->status()["result"].value("iteration", 0), 1007);

    mgr_->assignMission(json{{"enabled", true}, {"goal", "fresh goal"}});
    EXPECT_EQ(mgr_->status()["result"].value("iteration", 0), 0);
}

TEST_F(AutonomyLoopTest, ExportImportRoundTripsEvolution) {
    mgr_->configureAgi(json{{"enabled", true}, {"dim", 8}});
    mgr_->configureSubconscious(json{{"enabled", true}});
    mgr_->assignMission(json{{"enabled", true}, {"goal", "persist me"}});
    mgr_->observe(buildCognitionAutonomySeedPayload("persist-sess", "persist me", 0.5), json{});

    /* run a few transitions so the value head / model actually change */
    for (int i = 0; i < 3; ++i) {
        mgr_->iterate(json{{"sessionId", "persist-sess"}}, json{});
    }
    nlohmann::json saved = mgr_->exportState();
    ASSERT_TRUE(saved.contains("agi"));
    ASSERT_TRUE(saved.contains("sensations"));
    ASSERT_TRUE(saved.contains("mission"));

    auto other = std::make_unique<CognitionAutonomyManager>();
    auto imp = other->importState(saved);
    ASSERT_TRUE(imp.value("ok", false)) << imp.dump();
    nlohmann::json re = other->exportState();

    /* learned state must survive the round-trip */
    EXPECT_EQ(re["agi"]["preferences"].dump(), saved["agi"]["preferences"].dump());
    EXPECT_EQ(re["mission"]["mission"].value("goal", ""),
              saved["mission"]["mission"].value("goal", ""));
    EXPECT_GT(re["iteration"].get<int>(), 0);
}
