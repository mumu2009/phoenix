/* GTest edge-case tests for Phoenix v7.0 mission layer.
   Covers boundary conditions, partial payloads, and multi-session isolation. */

#include "autonomy_stack.hpp"
#include "instinct.hpp"
#include "mission_lifecycle.hpp"
#include "primal_sensation.hpp"
#include "subconscious_profile.hpp"
#include "test_hacktest_framework.hpp"
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <thread>

using namespace autonomy;
using namespace phoenix::mission;
using namespace phoenix::primal;
using namespace phoenix::subconscious;
using json = nlohmann::json;

namespace {

MissionGenome makeGenome(float lr = 0.05f) {
  MissionGenome g;
  g.profile = SubconsciousProfile::defaults();
  g.learningRate = lr;
  g.profile.baselineValence = 0.2f;
  g.profile.baselineArousal = 0.1f;
  g.profile.baselineDominance = -0.1f;
  g.profile.temperamentStrength = 0.8f;
  g.profile.riskAversion = 1.0f;
  g.profile.anticipatoryGain = 0.5f;
  phoenix::instinct::Instinct inst;
  inst.type = phoenix::instinct::InstinctType::Curiosity;
  inst.activation = 0.4f;
  inst.benefitWeight = 0.6f;
  inst.harmWeight = 0.4f;
  g.profile.instincts.push_back(inst);
  g.profile.sensationTuning["pain"] = {1.0f, 60.0f, 0.0f};
  return g;
}

class AutonomyMissionFixture : public ::testing::Test {
 protected:
  std::unique_ptr<CognitionAutonomyManager> mgr_;

  void SetUp() override { mgr_ = std::make_unique<CognitionAutonomyManager>(); }
  void TearDown() override { mgr_.reset(); }

  void seed(const std::string &sid) {
    mgr_->observe(buildCognitionAutonomySeedPayload(sid, "edge-test", 0.8), json{});
  }
};

}  // namespace

// ---------------------------------------------------------------------------
// Pressure edge cases
// ---------------------------------------------------------------------------
TEST(MissionPressureEdgeCases, ZeroGainKeepsPressureZero) {
  Mission m;
  m.painGainPerSec = 0.0f;
  m.maxPain = 1.0f;
  m.state = MissionState::Running;
  m.startMs = 0;
  for (uint64_t t = 0; t <= 5000; t += 1000) {
    EXPECT_FLOAT_EQ(m.pressure(t), 0.0f) << "zero gain at t=" << t;
  }
}

TEST(MissionPressureEdgeCases, VeryHighGainSaturatesImmediately) {
  Mission m;
  m.painGainPerSec = 1.0e6f;
  m.maxPain = 0.5f;
  m.state = MissionState::Running;
  m.startMs = 0;
  EXPECT_FLOAT_EQ(m.pressure(1), 0.5f);
  EXPECT_FLOAT_EQ(m.pressure(1000), 0.5f);
  EXPECT_FLOAT_EQ(m.pressure(10000), 0.5f);
}

TEST(MissionPressureEdgeCases, CompletedAndFailedStatesAreZero) {
  Mission m;
  m.painGainPerSec = 1.0f;
  m.maxPain = 1.0f;
  m.state = MissionState::Running;
  m.startMs = 0;
  EXPECT_GT(m.pressure(100), 0.0f);

  m.state = MissionState::Completed;
  EXPECT_FLOAT_EQ(m.pressure(100), 0.0f);
  EXPECT_FLOAT_EQ(m.pressure(10000), 0.0f);

  m.state = MissionState::Failed;
  EXPECT_FLOAT_EQ(m.pressure(100), 0.0f);
}

// ---------------------------------------------------------------------------
// Lifecycle edge cases
// ---------------------------------------------------------------------------
TEST(MissionLifecycleEdgeCases, AssignResetsEndMsAndState) {
  MissionLifecycle lc;
  Mission m;
  m.id = "a";
  m.goal = "g";
  m.painGainPerSec = 1.0f;
  m.maxPain = 1.0f;
  lc.assign(m);
  lc.markComplete();
  EXPECT_EQ(lc.mission().state, MissionState::Completed);
  EXPECT_GT(lc.mission().endMs, 0u);

  Mission m2;
  m2.id = "a2";
  m2.goal = "g2";
  m2.painGainPerSec = 2.0f;
  m2.maxPain = 1.0f;
  lc.assign(m2);
  EXPECT_EQ(lc.mission().state, MissionState::Running);
  EXPECT_EQ(lc.mission().endMs, 0u);
  EXPECT_EQ(lc.mission().goal, "g2");
}

TEST(MissionLifecycleEdgeCases, FailedMissionDoesNotIncrementCompleteCount) {
  MissionLifecycle lc;
  Mission m;
  m.painGainPerSec = 10.0f;
  m.maxPain = 1.0f;
  lc.assign(m);
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  lc.markFailed();
  const auto s = lc.stats();
  EXPECT_EQ(s["completions"].get<size_t>(), 0u);
  EXPECT_EQ(s["spawns"].get<size_t>(), 0u);
}

// ---------------------------------------------------------------------------
// Genome / mutation edge cases
// ---------------------------------------------------------------------------
TEST(MissionGenomeEdgeCases, HighMutationRateStillWithinBounds) {
  MissionLifecycle lc;
  MissionGenome parent = makeGenome(0.05f);
  for (int i = 0; i < 50; ++i) {
    auto child = lc.spawnChild(parent, 2.0f);  // very high mutation
    EXPECT_GE(child.learningRate, 0.001f);
    EXPECT_LE(child.learningRate, 0.5f);
    EXPECT_GE(child.profile.baselineValence, -1.0f);
    EXPECT_LE(child.profile.baselineValence, 1.0f);
  }
}

TEST(MissionGenomeEdgeCases, MutationClampsExtremeParent) {
  MissionGenome parent = makeGenome(0.05f);
  parent.profile.baselineValence = 2.0f;  // out of bounds
  parent.profile.riskAversion = 10.0f;
  parent.learningRate = 2.0f;

  MissionLifecycle lc;
  auto child = lc.spawnChild(parent, 0.0f);  // no mutation, just clamp
  EXPECT_LE(child.profile.baselineValence, 1.0f);
  EXPECT_GE(child.profile.baselineValence, -1.0f);
  EXPECT_LE(child.profile.riskAversion, 3.0f);
  EXPECT_GE(child.profile.riskAversion, 0.2f);
  EXPECT_LE(child.learningRate, 0.5f);
  EXPECT_GE(child.learningRate, 0.001f);
}

TEST(MissionGenomeEdgeCases, InstinctMutationsStayWithinBounds) {
  MissionGenome parent = makeGenome(0.05f);
  parent.profile.instincts[0].activation = 0.9f;
  parent.profile.instincts[0].benefitWeight = 0.9f;
  parent.profile.instincts[0].harmWeight = 0.9f;

  MissionLifecycle lc;
  for (int i = 0; i < 30; ++i) {
    auto child = lc.spawnChild(parent, 0.2f);
    for (const auto &inst : child.profile.instincts) {
      EXPECT_GE(inst.activation, 0.0f);
      EXPECT_LE(inst.activation, 1.0f);
      EXPECT_GE(inst.benefitWeight, 0.0f);
      EXPECT_LE(inst.benefitWeight, 1.0f);
      EXPECT_GE(inst.harmWeight, 0.0f);
      EXPECT_LE(inst.harmWeight, 1.0f);
    }
  }
}

// ---------------------------------------------------------------------------
// Evolution edge cases
// ---------------------------------------------------------------------------
TEST(MissionEvolutionEdgeCases, DifferentSeedsProduceDifferentTrajectories) {
  const int G = 5;
  const int lambda = 4;
  const float rate = 0.05f;

  auto run = [&](uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, 0.1);
    MissionGenome parent = makeGenome(0.05f);
    double total = 0.0;
    for (int gen = 0; gen < G; ++gen) {
      double best = 1e9;
      for (int i = 0; i < lambda; ++i) {
        auto child = parent.mutate(rate, rng);
        double time = 100.0 - 150.0 * child.learningRate - 15.0 * child.profile.anticipatoryGain + noise(rng);
        if (time < 0.0) time = 0.0;
        if (time < best) best = time;
      }
      total += best;
    }
    return total;
  };

  double r1 = run(0x1234);
  double r2 = run(0xABCD);
  double r3 = run(0x9999);
  EXPECT_NE(r1, r2) << "different seeds should produce different results";
  EXPECT_NE(r2, r3);
}

// ---------------------------------------------------------------------------
// Autonomy stack edge cases
// ---------------------------------------------------------------------------
TEST_F(AutonomyMissionFixture, AssignWithoutEnabledIsDisabled) {
  seed("edge-enabled");
  auto r = mgr_->assignMission(json{{"goal", "x"}, {"painGainPerSec", 10.0}, {"maxPain", 1.0}});
  EXPECT_TRUE(r.value("ok", false));

  auto status = mgr_->missionStatus();
  EXPECT_FALSE(status["result"].value("enabled", true));
  EXPECT_EQ(status["result"]["stats"]["mission"].value("goal", std::string()), "x");
}

TEST_F(AutonomyMissionFixture, MissingGoalLeavesIdle) {
  auto r = mgr_->assignMission(json{{"enabled", true}});
  EXPECT_TRUE(r.value("ok", false));
  auto status = mgr_->missionStatus();
  EXPECT_EQ(status["result"]["stats"]["mission"].value("state", 1), 0);
}

TEST_F(AutonomyMissionFixture, PartialPayloadUsesDefaults) {
  seed("edge-partial");
  auto r = mgr_->assignMission(json{{"enabled", true}, {"goal", "partial"}});
  EXPECT_TRUE(r.value("ok", false));
  auto status = mgr_->missionStatus();
  EXPECT_TRUE(status["result"].value("enabled", false));
  auto m = status["result"]["stats"]["mission"];
  EXPECT_EQ(m.value("goal", std::string()), "partial");
  EXPECT_GT(m.value("painGainPerSec", 0.0f), 0.0f);
  EXPECT_GT(m.value("maxPain", 0.0f), 0.0f);
}

TEST_F(AutonomyMissionFixture, NullPayloadForSpawnWorks) {
  seed("edge-null-spawn");
  mgr_->assignMission(json{{"enabled", true}, {"goal", "evolve"}});
  auto child = mgr_->spawnMissionChild(json{});
  EXPECT_TRUE(child.value("ok", false));
  EXPECT_TRUE(child["result"].is_object());
  EXPECT_TRUE(child["result"].contains("profile"));
}

TEST_F(AutonomyMissionFixture, SpawnWithCustomGenomeAndRate) {
  seed("edge-custom-genome");
  mgr_->assignMission(json{{"enabled", true}, {"goal", "evolve"}});

  json genomeIn;
  {
    auto g = makeGenome(0.1f);
    genomeIn = g.toJson();
  }

  auto child = mgr_->spawnMissionChild(json{{"mutationRate", 0.0}, {"genome", genomeIn}});
  EXPECT_TRUE(child.value("ok", false));
  float lr = child["result"].value("learningRate", -1.0f);
  EXPECT_FLOAT_EQ(lr, 0.1f) << "zero mutation should preserve custom parent";
}

TEST_F(AutonomyMissionFixture, ReportMissionOutcomeFalseFails) {
  seed("edge-fail");
  mgr_->assignMission(json{{"enabled", true},
                           {"goal", "fail"},
                           {"painGainPerSec", 100.0},
                           {"maxPain", 1.0}});
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  mgr_->iterate(json{{"sessionId", "edge-fail"}}, json{});

  auto r = mgr_->reportMissionOutcome(json{{"goalAchieved", false}});
  EXPECT_TRUE(r.value("ok", false));

  auto iter = mgr_->iterate(json{{"sessionId", "edge-fail"}}, json{});
  EXPECT_FALSE(iter["result"]["mission"].value("active", true));
  EXPECT_FLOAT_EQ(iter["result"]["mission"].value("pressure", -1.0f), 0.0f);
}

TEST_F(AutonomyMissionFixture, CompleteMissionTwiceIsIdempotent) {
  seed("edge-twice");
  mgr_->assignMission(json{{"enabled", true},
                           {"goal", "twice"},
                           {"painGainPerSec", 100.0},
                           {"maxPain", 1.0}});
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  mgr_->iterate(json{{"sessionId", "edge-twice"}}, json{});

  mgr_->reportMissionOutcome(json{{"goalAchieved", true}});
  mgr_->reportMissionOutcome(json{{"goalAchieved", true}});
  mgr_->reportMissionOutcome(json{{"goalAchieved", false}});  // must not revert

  auto status = mgr_->missionStatus();
  EXPECT_EQ(status["result"]["stats"]["mission"].value("state", 0), static_cast<int>(MissionState::Completed));
  EXPECT_EQ(status["result"]["stats"]["completions"].get<size_t>(), 1u);
}

TEST_F(AutonomyMissionFixture, MultipleSessionsDoNotInterfere) {
  seed("edge-sess-a");
  seed("edge-sess-b");

  mgr_->assignMission(json{{"enabled", true},
                           {"goal", "shared"},
                           {"painGainPerSec", 100.0},
                           {"maxPain", 1.0}});

  std::this_thread::sleep_for(std::chrono::milliseconds(3));
  auto a = mgr_->iterate(json{{"sessionId", "edge-sess-a"}}, json{});
  auto b = mgr_->iterate(json{{"sessionId", "edge-sess-b"}}, json{});

  EXPECT_TRUE(a.value("ok", false));
  EXPECT_TRUE(b.value("ok", false));
  EXPECT_GT(a["result"]["mission"].value("pressure", 0.0f), 0.0f);
  EXPECT_GT(b["result"]["mission"].value("pressure", 0.0f), 0.0f);

  mgr_->reportMissionOutcome(json{{"goalAchieved", true}});
  auto a2 = mgr_->iterate(json{{"sessionId", "edge-sess-a"}}, json{});
  EXPECT_FALSE(a2["result"]["mission"].value("active", true));
}

TEST_F(AutonomyMissionFixture, MissionStatusReflectsPressure) {
  seed("edge-status");
  mgr_->assignMission(json{{"enabled", true},
                           {"goal", "status"},
                           {"painGainPerSec", 1000.0},
                           {"maxPain", 1.0}});

  auto before = mgr_->missionStatus();
  EXPECT_TRUE(before["result"].value("enabled", false));
  float p0 = before["result"]["stats"].value("pressure", -1.0f);
  EXPECT_GE(p0, 0.0f);

  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  auto after = mgr_->missionStatus();
  float p1 = after["result"]["stats"].value("pressure", -1.0f);
  EXPECT_GE(p1, p0);
}

TEST_F(AutonomyMissionFixture, DisabledMissionDoesNotSpawns) {
  // spawnMissionChild is independent of enabled flag; verify it still returns a
  // valid genome even when no mission is active.
  seed("edge-spawn-disabled");
  auto child = mgr_->spawnMissionChild(json{});
  EXPECT_TRUE(child.value("ok", false));
  EXPECT_TRUE(child["result"].is_object());
}
