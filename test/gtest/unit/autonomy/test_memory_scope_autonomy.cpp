/* Dual-mission / chat isolation through CognitionAutonomyManager. */

#include "autonomy_stack.hpp"
#include "memory_scope.hpp"
#include "scoped_trainable_memory.hpp"

#include <gtest/gtest.h>

using autonomy::CognitionAutonomyManager;
using autonomy::json;
using phoenix::memory::ScopedTrainableMemory;
using phoenix::memory::makeChatScope;
using phoenix::memory::makeMissionScope;

class MemoryScopeAutonomyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ScopedTrainableMemory::instance().resetForTest();
    mgr_ = std::make_unique<CognitionAutonomyManager>();
  }
  void TearDown() override {
    mgr_.reset();
    ScopedTrainableMemory::instance().resetForTest();
  }
  std::unique_ptr<CognitionAutonomyManager> mgr_;
};

TEST_F(MemoryScopeAutonomyTest, TwoMissionsKeepIndependentGnnSummaries) {
  auto ha = mgr_->assignMission(json{{"enabled", true},
                                     {"id", "helios"},
                                     {"goal", "Design Helios station"}});
  ASSERT_TRUE(ha.value("ok", false)) << ha.dump();
  auto hb = mgr_->assignMission(json{{"enabled", true},
                                     {"id", "ops"},
                                     {"goal", "Night bus isolation"}});
  ASSERT_TRUE(hb.value("ok", false)) << hb.dump();

  mgr_->setMissionGnnSummaryFor("helios", "helios-gnn-pin");
  mgr_->setMissionGnnSummaryFor("ops", "ops-gnn-pin");

  EXPECT_EQ(mgr_->missionGnnSummaryFor("helios"), "helios-gnn-pin");
  EXPECT_EQ(mgr_->missionGnnSummaryFor("ops"), "ops-gnn-pin");
  EXPECT_NE(mgr_->missionGnnSummaryFor("helios"),
            mgr_->missionGnnSummaryFor("ops"));

  EXPECT_EQ(ScopedTrainableMemory::instance()
                .hot(makeMissionScope("helios"))
                .gnnSummary,
            "helios-gnn-pin");
  EXPECT_EQ(ScopedTrainableMemory::instance()
                .hot(makeMissionScope("ops"))
                .gnnSummary,
            "ops-gnn-pin");
}

TEST_F(MemoryScopeAutonomyTest, ChatBucketStaysEmptyWhileMissionsTrain) {
  mgr_->assignMission(json{{"enabled", true},
                           {"id", "helios"},
                           {"goal", "Helios"}});
  mgr_->setMissionGnnSummaryFor("helios", "mission-only-hint");
  auto &hm = ScopedTrainableMemory::instance().hot(makeMissionScope("helios"));
  hm.recurrent.rnnHidden = {1.f, 2.f};
  hm.gnn.add("mh", "ms", 1.0);

  auto *chat = ScopedTrainableMemory::instance().peek(makeChatScope("fresh"));
  EXPECT_EQ(chat, nullptr);
  auto &hc = ScopedTrainableMemory::instance().hot(makeChatScope("fresh"));
  EXPECT_TRUE(hc.recurrent.rnnHidden.empty());
  EXPECT_TRUE(hc.gnn.empty());
  EXPECT_TRUE(hc.gnnSummary.empty());
}

TEST_F(MemoryScopeAutonomyTest, CompleteMissionReleasesTrainableBucket) {
  mgr_->assignMission(json{{"enabled", true},
                           {"id", "helios"},
                           {"goal", "Helios"}});
  mgr_->setMissionGnnSummaryFor("helios", "live-hint");
  ScopedTrainableMemory::instance()
      .hot(makeMissionScope("helios"))
      .recurrent.rnnHidden = {3.f};

  auto done = mgr_->reportMissionOutcome(
      json{{"goalAchieved", true}, {"missionId", "helios"}});
  ASSERT_TRUE(done.value("ok", false)) << done.dump();

  EXPECT_FALSE(ScopedTrainableMemory::instance().hasLive(
      makeMissionScope("helios")));
  EXPECT_TRUE(mgr_->missionGnnSummaryFor("helios").empty());

  auto &chat = ScopedTrainableMemory::instance().hot(
      makeChatScope("after-complete"));
  EXPECT_TRUE(chat.recurrent.rnnHidden.empty());
  EXPECT_TRUE(chat.gnn.empty());
  EXPECT_TRUE(chat.gnnSummary.empty());
}

TEST_F(MemoryScopeAutonomyTest, SecondAssignDoesNotWipeOtherMissionSummary) {
  mgr_->assignMission(json{{"enabled", true},
                           {"id", "helios"},
                           {"goal", "Helios"}});
  mgr_->setMissionGnnSummaryFor("helios", "keep-me");
  mgr_->assignMission(json{{"enabled", true},
                           {"id", "ops"},
                           {"goal", "Ops"}});
  EXPECT_EQ(mgr_->missionGnnSummaryFor("helios"), "keep-me");
  EXPECT_TRUE(mgr_->missionGnnSummaryFor("ops").empty());
}
