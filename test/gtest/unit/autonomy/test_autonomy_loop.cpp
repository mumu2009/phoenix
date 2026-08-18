/* test_autonomy_loop.cpp - Long-term autonomous loop, persistence and
   human interjection (插话) for the autonomy stack. */

#include "autonomy_stack.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

using autonomy::CognitionAutonomyManager;
using autonomy::buildCognitionAutonomySeedPayload;
using autonomy::json;

namespace {

class AutonomyLoopTest : public ::testing::Test {
protected:
    std::unique_ptr<CognitionAutonomyManager> mgr_;
    void SetUp() override { mgr_ = std::make_unique<CognitionAutonomyManager>(); }
    void TearDown() override { mgr_.reset(); }
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
    EXPECT_FALSE(mgr_->autonomyLoopStatus()["result"].value("running", true));
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
