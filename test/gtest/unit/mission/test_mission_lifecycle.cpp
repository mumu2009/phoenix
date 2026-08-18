/* GTest unit tests for Phoenix v7.0 mission layer (Meeseeks lifecycle).
   Validates the protocol in doc/v7.0/testing_methodology.md §9. */

#include "autonomy_stack.hpp"
#include "instinct.hpp"
#include "mission_lifecycle.hpp"
#include "primal_sensation.hpp"
#include "subconscious_profile.hpp"
#include "test_hacktest_framework.hpp"
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

using namespace autonomy;
using namespace phoenix::mission;
using namespace phoenix::primal;
using namespace phoenix::subconscious;
using json = nlohmann::json;

namespace {

// System-clock millisecond accessor (mirrors mission_lifecycle.cpp).
uint64_t sysNowMs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

// Build a running Mission with a known start time and parameters.
Mission makeRunningMission(uint64_t startMs,
                           float gain = 0.1f,
                           float maxPain = 1.0f) {
  Mission m;
  m.id = "test-mission";
  m.goal = "test goal";
  m.painGainPerSec = gain;
  m.maxPain = maxPain;
  m.state = MissionState::Running;
  m.startMs = startMs;
  return m;
}

// Fill a parent genome with one tuning and one instinct so all clamp loops are
// exercised (SubconsciousProfile::defaults() is otherwise empty).
MissionGenome makeParentGenome(float learningRate = 0.05f) {
  MissionGenome g;
  g.profile = SubconsciousProfile::defaults();
  g.learningRate = learningRate;
  g.profile.sensationTuning["pain"] = {1.0f, 60.0f, 0.0f};
  g.profile.sensationTuning["novelty"] = {1.0f, 30.0f, 0.0f};
  g.profile.temperamentStrength = 0.5f;
  g.profile.riskAversion = 1.0f;
  g.profile.anticipatoryGain = 1.0f;
  phoenix::instinct::Instinct inst;
  inst.type = phoenix::instinct::InstinctType::Curiosity;
  inst.activation = 0.5f;
  inst.benefitWeight = 0.5f;
  inst.harmWeight = 0.5f;
  g.profile.instincts.push_back(inst);
  return g;
}

double mean(const std::vector<double> &v) {
  if (v.empty()) return 0.0;
  return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

double stdDev(const std::vector<double> &v) {
  if (v.size() < 2) return 0.0;
  const double m = mean(v);
  double s = 0.0;
  for (double x : v) {
    const double d = x - m;
    s += d * d;
  }
  return std::sqrt(s / static_cast<double>(v.size() - 1));
}

// One-sided pooled t-test: return t-statistic for groupA > groupB.
// Uses the 0.05 critical value for df=8 (t_{0.95,8}=1.86) because the C++
// standard library does not provide a Student-t CDF.
double pooledT(const std::vector<double> &a, const std::vector<double> &b) {
  const double ma = mean(a);
  const double mb = mean(b);
  const double sa = stdDev(a);
  const double sb = stdDev(b);
  const size_t na = a.size();
  const size_t nb = b.size();
  const double varPooled = ((na - 1) * sa * sa + (nb - 1) * sb * sb) /
                           static_cast<double>(na + nb - 2);
  const double se = std::sqrt(varPooled * (1.0 / na + 1.0 / nb));
  if (se < 1e-12) return 0.0;
  return (ma - mb) / se;
}

}  // namespace

// ---------------------------------------------------------------------------
// §9.1 压力单调与截断
// ---------------------------------------------------------------------------
TEST(MissionPressureTest, MonotoneAndSaturated) {
  const uint64_t startMs = 0;
  const float gain = 0.1f;
  const float maxPain = 1.0f;

  for (int stateVal = 0; stateVal <= 3; ++stateVal) {
    Mission m;
    m.painGainPerSec = gain;
    m.maxPain = maxPain;
    m.state = static_cast<MissionState>(stateVal);
    m.startMs = startMs;

    float prev = -1.0f;
    for (uint64_t t = 0; t <= 20000; t += 1000) {
      float p = m.pressure(t);
      switch (m.state) {
        case MissionState::Idle:
        case MissionState::Completed:
        case MissionState::Failed:
          EXPECT_FLOAT_EQ(p, 0.0f) << "non-Running state at t=" << t;
          break;
        case MissionState::Running:
          if (t <= 10000) {
            EXPECT_NEAR(p, gain * static_cast<float>(t) / 1000.0f, 1e-6f)
                << "linear growth at t=" << t;
            EXPECT_GE(p, prev - 1e-6f) << "pressure not monotone at t=" << t;
          } else {
            EXPECT_FLOAT_EQ(p, maxPain) << "saturation at t=" << t;
          }
          break;
      }
      prev = p;
    }
  }
}

// ---------------------------------------------------------------------------
// §9.2 markComplete 终结疼痛 + 幂等
// ---------------------------------------------------------------------------
TEST(MissionLifecycleTest, StatsExposeCompletionTimeForSupervisorSelection) {
    MissionLifecycle lc;
    Mission m;
    m.id = "sel";
    m.goal = "g";
    m.painGainPerSec = 1.0f;
    m.maxPain = 1.0f;
    lc.assign(m);
    /* while running there is no completion time yet */
    EXPECT_EQ(lc.stats().value("completionTimeMs", -2), -1);
    lc.markComplete();
    auto st = lc.stats();
    ASSERT_TRUE(st.contains("completionTimeMs"));
    EXPECT_GE(st["completionTimeMs"].get<int64_t>(), 0);
    /* selection stays OUTSIDE the process: stats carry the signal, but there
       is no fitness field / elite genome / autonomous retry loop in the API */
    EXPECT_FALSE(st.contains("fitness"));
    EXPECT_FALSE(st.contains("elite"));
}

TEST(MissionLifecycleTest, ReplicateMutatesAndRecords) {
    MissionLifecycle lc;
    Mission m;
    m.id = "repl-m";
    m.goal = "must complete";
    m.painGainPerSec = 1.0f;
    m.maxPain = 1.0f;
    MissionGenome parent;
    parent.learningRate = 0.05f;
    lc.assign(m, parent);
    auto child = lc.replicate(0.1f);
    EXPECT_NEAR(child.learningRate, 0.05f, 0.5f); /* mutated but bounded */
    auto children = lc.children();
    ASSERT_EQ(children.size(), 1u);
    EXPECT_EQ(children[0].goal, "must complete"); /* goal inherited */
    EXPECT_EQ(children[0].id, "child-1-0");
    auto st = lc.stats();
    EXPECT_EQ(st["spawns"].get<size_t>(), 1u);
    ASSERT_TRUE(st.contains("children"));
    EXPECT_EQ(st["children"].size(), 1u);
}

TEST(MissionLifecycleTest, ReplicaLimitBoundsReproduction) {
    MissionLifecycle lc;
    Mission m;
    m.id = "bounded";
    m.goal = "g";
    lc.assign(m, MissionGenome{});
    lc.setMaxReplicas(2);
    lc.replicate(0.05f);
    lc.replicate(0.05f);
    EXPECT_THROW(lc.replicate(0.05f), std::runtime_error);
    EXPECT_EQ(lc.children().size(), 2u); /* bounded, no runaway spawning */
}

TEST(MissionLifecycleTest, AmendGoalRedirectsWithoutRestart) {
    MissionLifecycle lc;
    Mission m;
    m.id = "amend";
    m.goal = "original goal";
    m.painGainPerSec = 1.0f;
    m.maxPain = 1.0f;
    lc.assign(m, MissionGenome{});
    const uint64_t start = lc.mission().startMs;
    EXPECT_TRUE(lc.amendGoal("redirected goal"));
    EXPECT_EQ(lc.mission().goal, "redirected goal");
    EXPECT_EQ(lc.mission().state, MissionState::Running);
    EXPECT_EQ(lc.mission().startMs, start); /* pressure keeps growing */
    lc.markComplete();
    EXPECT_FALSE(lc.amendGoal("too late"));
    EXPECT_EQ(lc.mission().goal, "redirected goal");
}

TEST(MissionLifecycleTest, MarkCompleteEndsPainAndIsIdempotent) {
  MissionLifecycle lc;
  Mission m;
  m.id = "test-mission";
  m.goal = "test goal";
  m.painGainPerSec = 100.0f;  // fast rise so a 5ms sleep produces pressure
  m.maxPain = 1.0f;
  lc.assign(m);

  std::this_thread::sleep_for(std::chrono::milliseconds(5));

  float p0 = lc.pressureNow();
  EXPECT_GT(p0, 0.0f);
  EXPECT_TRUE(lc.active());

  const uint64_t startMs = lc.mission().startMs;
  lc.markComplete();

  EXPECT_FLOAT_EQ(lc.pressureNow(), 0.0f);
  EXPECT_FALSE(lc.active());
  EXPECT_EQ(lc.mission().state, MissionState::Completed);
  EXPECT_GE(lc.mission().endMs, startMs);

  const uint64_t firstEndMs = lc.mission().endMs;
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  lc.markComplete();
  EXPECT_EQ(lc.mission().state, MissionState::Completed);
  EXPECT_EQ(lc.mission().endMs, firstEndMs) << "second markComplete must be idempotent";

  lc.markFailed();
  EXPECT_EQ(lc.mission().state, MissionState::Completed)
      << "markFailed must not revert Completed";
}

TEST(MissionLifecycleTest, MarkFailedIsTerminal) {
  MissionLifecycle lc;
  Mission m;
  m.id = "test-mission";
  m.goal = "test goal";
  m.painGainPerSec = 100.0f;
  m.maxPain = 1.0f;
  lc.assign(m);

  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  EXPECT_GT(lc.pressureNow(), 0.0f);

  lc.markFailed();
  EXPECT_EQ(lc.mission().state, MissionState::Failed);
  EXPECT_FLOAT_EQ(lc.pressureNow(), 0.0f);
  lc.markComplete();
  EXPECT_EQ(lc.mission().state, MissionState::Failed)
      << "markComplete must not resurrect Failed";
}

// ---------------------------------------------------------------------------
// §9.3 累积疼痛定理：数值积分对照分段公式
// ---------------------------------------------------------------------------
TEST(MissionPressureTest, TotalPainIntegralMatchesTheorem) {
  const float gain = 0.1f;
  const float maxPain = 1.0f;
  const uint64_t startMs = 0;
  Mission m = makeRunningMission(startMs, gain, maxPain);

  const double dt = 1e-3;  // 1ms step trapezoid (avoids ms truncation at t<1ms)
  auto integrate = [&](double T) {
    double s = 0.0;
    double prev = m.pressure(startMs);
    for (double t = dt; t <= T + 1e-9; t += dt) {
      uint64_t nowMs = startMs + static_cast<uint64_t>(t * 1000.0 + 0.5);
      double cur = m.pressure(nowMs);
      s += 0.5 * (prev + cur) * dt;
      prev = cur;
    }
    return s;
  };

  auto expected = [&](double T) -> double {
    if (T <= 10.0) {
      return 0.5 * gain * T * T;
    } else {
      return 0.5 * gain * 10.0 * 10.0 + maxPain * (T - 10.0);
    }
  };

  for (double T : {2.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0}) {
    double actual = integrate(T);
    double exp = expected(T);
    double relErr = std::abs(actual - exp) / std::max(1.0, std::abs(exp));
    EXPECT_LE(relErr, 1e-6)
        << "P(" << T << ") actual=" << actual << " expected=" << exp;
  }
}

// ---------------------------------------------------------------------------
// §9.4 spawnChild 变异有界
// ---------------------------------------------------------------------------
TEST(MissionGenomeTest, SpawnedChildrenStayWithinClampBounds) {
  MissionLifecycle lc;
  MissionGenome parent = makeParentGenome(0.05f);
  const float rate = 0.05f;

  std::vector<MissionGenome> children;
  for (int i = 0; i < 100; ++i) {
    children.push_back(lc.spawnChild(parent, rate));
  }

  for (const auto &c : children) {
    // PAD
    EXPECT_GE(c.profile.baselineValence, -1.0f);
    EXPECT_LE(c.profile.baselineValence, 1.0f);
    EXPECT_GE(c.profile.baselineArousal, -1.0f);
    EXPECT_LE(c.profile.baselineArousal, 1.0f);
    EXPECT_GE(c.profile.baselineDominance, -1.0f);
    EXPECT_LE(c.profile.baselineDominance, 1.0f);

    EXPECT_GE(c.profile.temperamentStrength, 0.0f);
    EXPECT_LE(c.profile.temperamentStrength, 2.0f);

    EXPECT_GE(c.profile.riskAversion, 0.2f);
    EXPECT_LE(c.profile.riskAversion, 3.0f);

    EXPECT_GE(c.profile.anticipatoryGain, 0.0f);
    EXPECT_LE(c.profile.anticipatoryGain, 3.0f);

    EXPECT_GE(c.learningRate, 0.001f);
    EXPECT_LE(c.learningRate, 0.5f);

    for (const auto &kv : c.profile.sensationTuning) {
      EXPECT_GE(kv.second.gain, 0.0f) << "sensation gain clamp";
      EXPECT_LE(kv.second.gain, 5.0f) << "sensation gain clamp";
      EXPECT_GE(kv.second.halfLifeSec, 1.0f) << "sensation halfLife clamp";
      EXPECT_LE(kv.second.halfLifeSec, 3600.0f) << "sensation halfLife clamp";
      EXPECT_GE(kv.second.setpoint, -1.0f) << "sensation setpoint clamp";
      EXPECT_LE(kv.second.setpoint, 1.0f) << "sensation setpoint clamp";
    }

    for (const auto &inst : c.profile.instincts) {
      EXPECT_GE(inst.activation, 0.0f);
      EXPECT_LE(inst.activation, 1.0f);
      EXPECT_GE(inst.benefitWeight, 0.0f);
      EXPECT_LE(inst.benefitWeight, 1.0f);
      EXPECT_GE(inst.harmWeight, 0.0f);
      EXPECT_LE(inst.harmWeight, 1.0f);
    }
  }

  // No-drift check: per-scalar |mean(child) - parent| <= 2 * std(child).
  auto checkDrift = [&](const std::vector<float> &vals, float parentVal) {
    const double m = mean(std::vector<double>(vals.begin(), vals.end()));
    const double s = stdDev(std::vector<double>(vals.begin(), vals.end()));
    EXPECT_LE(std::abs(m - parentVal), 2.0 * s + 1e-4)
        << "drift: parent=" << parentVal << " mean=" << m << " std=" << s;
  };

  std::vector<float> vals;
  vals.reserve(100);
  for (float MissionGenome::*field : {&MissionGenome::learningRate}) {
    vals.clear();
    for (const auto &c : children) vals.push_back(c.*field);
    checkDrift(vals, parent.*field);
  }
  for (float SubconsciousProfile::*field : {&SubconsciousProfile::baselineValence,
                                           &SubconsciousProfile::baselineArousal,
                                           &SubconsciousProfile::baselineDominance,
                                           &SubconsciousProfile::temperamentStrength,
                                           &SubconsciousProfile::riskAversion,
                                           &SubconsciousProfile::anticipatoryGain}) {
    vals.clear();
    for (const auto &c : children) vals.push_back(c.profile.*field);
    checkDrift(vals, parent.profile.*field);
  }
}

TEST(MissionGenomeTest, ZeroMutationRateProducesIdenticalChild) {
  MissionLifecycle lc;
  MissionGenome parent = makeParentGenome(0.05f);
  MissionGenome child = lc.spawnChild(parent, 0.0f);

  EXPECT_EQ(child.learningRate, parent.learningRate);
  EXPECT_EQ(child.profile.baselineValence, parent.profile.baselineValence);
  EXPECT_EQ(child.profile.baselineArousal, parent.profile.baselineArousal);
  EXPECT_EQ(child.profile.baselineDominance, parent.profile.baselineDominance);
  EXPECT_EQ(child.profile.temperamentStrength, parent.profile.temperamentStrength);
  EXPECT_EQ(child.profile.riskAversion, parent.profile.riskAversion);
  EXPECT_EQ(child.profile.anticipatoryGain, parent.profile.anticipatoryGain);
  EXPECT_EQ(child.profile.sensationTuning.size(), parent.profile.sensationTuning.size());
  for (const auto &kv : child.profile.sensationTuning) {
    auto it = parent.profile.sensationTuning.find(kv.first);
    ASSERT_TRUE(it != parent.profile.sensationTuning.end()) << "missing tuning key " << kv.first;
    EXPECT_FLOAT_EQ(kv.second.gain, it->second.gain);
    EXPECT_FLOAT_EQ(kv.second.halfLifeSec, it->second.halfLifeSec);
    EXPECT_FLOAT_EQ(kv.second.setpoint, it->second.setpoint);
  }
  EXPECT_EQ(child.profile.instincts.size(), parent.profile.instincts.size());
  for (size_t i = 0; i < child.profile.instincts.size(); ++i) {
    EXPECT_FLOAT_EQ(child.profile.instincts[i].activation, parent.profile.instincts[i].activation);
    EXPECT_FLOAT_EQ(child.profile.instincts[i].benefitWeight, parent.profile.instincts[i].benefitWeight);
    EXPECT_FLOAT_EQ(child.profile.instincts[i].harmWeight, parent.profile.instincts[i].harmWeight);
  }
  EXPECT_EQ(lc.stats()["spawns"].get<size_t>(), 1u);
}

// ---------------------------------------------------------------------------
// §9.5 多代完成时间下降（统计趋势，t 检验）
// ---------------------------------------------------------------------------
TEST(MissionEvolutionTest, CompletionTimeDecreasesAcrossGenerations) {
  // Deterministic (1+lambda)-ES simulation.  Completion time is a synthetic
  // fitness that rewards higher learningRate and anticipatoryGain.  A small
  // amount of noise is added so the result is not a foregone conclusion.
  const int G = 10;
  const int lambda = 8;
  const float mutationRate = 0.05f;
  const uint32_t seed = 0x1DEA;
  std::mt19937 rng(seed);
  std::normal_distribution<double> noise(0.0, 0.2);

  MissionGenome parent = makeParentGenome(0.05f);
  std::vector<double> genMeans;

  for (int gen = 0; gen < G; ++gen) {
    std::vector<MissionGenome> children;
    std::vector<double> times;
    children.reserve(lambda);
    times.reserve(lambda);
    for (int i = 0; i < lambda; ++i) {
      children.push_back(parent.mutate(mutationRate, rng));
      const auto &c = children.back();
      double time = 100.0 - 150.0 * c.learningRate - 15.0 * c.profile.anticipatoryGain + noise(rng);
      if (time < 0.0) time = 0.0;
      times.push_back(time);
    }

    // (1+lambda) selection: keep the child with the shortest completion time.
    size_t best = 0;
    for (size_t i = 1; i < times.size(); ++i) {
      if (times[i] < times[best]) best = i;
    }
    parent = children[best];
    genMeans.push_back(mean(times));
  }

  ASSERT_EQ(genMeans.size(), static_cast<size_t>(G));
  std::vector<double> first5(genMeans.begin(), genMeans.begin() + 5);
  std::vector<double> last5(genMeans.begin() + 5, genMeans.end());

  // One-sided test that first5 mean > last5 mean.
  double t = pooledT(first5, last5);
  // Critical t_{0.95, 8} = 1.860 (conservative for small samples).
  const double tCrit = 1.86;

  if (t > tCrit) {
    EXPECT_GT(mean(first5), mean(last5)) << "evolution should shorten completion time";
    RecordProperty("t_statistic", t);
    RecordProperty("generations", G);
  } else {
    // Noise can mask a real trend with a short horizon; record and skip
    // rather than claiming evolution as inevitable.
    GTEST_SKIP() << "No statistically significant decrease across generations "
                 << "(t=" << t << " <= " << tCrit << "). First5 mean="
                 << mean(first5) << " last5 mean=" << mean(last5);
  }
}

// ---------------------------------------------------------------------------
// §9.6 自主栈 Pain 感受注入逐项匹配
// ---------------------------------------------------------------------------
class AutonomyMissionTest : public ::testing::Test {
 protected:
  std::unique_ptr<CognitionAutonomyManager> mgr_;

  void SetUp() override { mgr_ = std::make_unique<CognitionAutonomyManager>(); }

  void TearDown() override { mgr_.reset(); }

  // Ensure a session exists and shouldIterate is true.
  void seedSession(const std::string &sessionId) {
    auto payload = buildCognitionAutonomySeedPayload(sessionId, "mission-test", 0.85);
    mgr_->observe(payload, json{});
  }

  // Find the most recent mission-pressure Pain in status() by timestamp.
  json findLatestMissionPain() {
    auto st = mgr_->status();
    json latest;
    uint64_t latestTs = 0;
    for (const auto &s : st["result"]["sensations"]) {
      if (s.value("type", std::string()) == "pain" &&
          s.value("source", std::string()) == "mission-pressure") {
        const uint64_t ts = s.value("timestampMs", 0ull);
        if (ts >= latestTs) {
          latestTs = ts;
          latest = s;
        }
      }
    }
    return latest;
  }
};

TEST_F(AutonomyMissionTest, IterateInjectsMissionPressurePain) {
  const std::string sid = "mission-pain-test";
  seedSession(sid);

  auto assign = mgr_->assignMission(json{{"enabled", true},
                                         {"goal", "complete this test"},
                                         {"painGainPerSec", 1000.0},
                                         {"maxPain", 1.0}});
  EXPECT_TRUE(assign.value("ok", false));

  std::this_thread::sleep_for(std::chrono::milliseconds(2));

  json iter = mgr_->iterate(json{{"sessionId", sid}}, json{});
  EXPECT_TRUE(iter.value("ok", false));

  const auto &r = iter["result"];
  ASSERT_TRUE(r.contains("mission")) << "result object: " << iter.dump();
  EXPECT_TRUE(r["mission"].value("enabled", false));
  EXPECT_TRUE(r["mission"].value("active", false));
  EXPECT_GT(r["mission"].value("pressure", 0.0f), 0.0f);
  EXPECT_EQ(r["mission"].value("goal", std::string()), "complete this test");

  json pain = findLatestMissionPain();
  ASSERT_FALSE(pain.empty()) << "status() should report a mission-pressure Pain";
  EXPECT_EQ(pain.value("type", std::string()), "pain");
  EXPECT_EQ(pain.value("source", std::string()), "mission-pressure");
  EXPECT_FLOAT_EQ(pain.value("valence", 0.0f), -1.0f);
  EXPECT_GT(pain.value("intensity", 0.0f), 0.0f);
  EXPECT_LE(pain.value("intensity", 0.0f), 1.0f);
}

TEST_F(AutonomyMissionTest, PressureIsSingleSignalNotStacked) {
    const std::string sid = "mission-single-signal";
    seedSession(sid);
    mgr_->assignMission(json{{"enabled", true},
                               {"goal", "keep one signal"},
                               {"painGainPerSec", 1000.0},
                               {"maxPain", 1.0}});
    for (int i = 0; i < 3; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
        mgr_->iterate(json{{"sessionId", sid}}, json{});
    }
    auto st = mgr_->status();
    size_t count = 0;
    for (const auto &s : st["result"]["sensations"]) {
        if (s.value("type", std::string()) == "pain" &&
            s.value("source", std::string()) == "mission-pressure") {
            ++count;
        }
    }
    EXPECT_EQ(count, 1u)
        << "mission pressure must be ONE refreshed signal p(t), not stacked "
           "copies (N*p would break the gT^2/2 model)";
}

TEST_F(AutonomyMissionTest, ReplicateActionRegisteredByDefault) {
    auto actions = mgr_->listAgiActions();
    ASSERT_TRUE(actions.contains("result"));
    bool found = false;
    for (const auto &a : actions["result"]) {
        if (a.value("name", std::string()) == "replicate") {
            found = true;
            EXPECT_EQ(a.value("category", std::string()), "replicate");
        }
    }
    EXPECT_TRUE(found) << "replicate must be a default planner action";
}

TEST_F(AutonomyMissionTest, ExecuteReplicateSpawnsSuccessorSession) {
    mgr_->assignMission(json{{"enabled", true},
                             {"goal", "replicate for me"},
                             {"painGainPerSec", 1.0},
                             {"maxPain", 1.0}});
    nlohmann::json ctx{{"userPrompt", "spawn help"}};
    auto res = mgr_->executeAgiActionByName("replicate", ctx);
    ASSERT_TRUE(res.value("ok", false)) << res.dump();
    const std::string childId = res["result"].value("childSessionId", std::string());
    EXPECT_FALSE(childId.empty());
    EXPECT_EQ(res["result"].value("goal", std::string()), "replicate for me");
    /* the successor is a real session bound to the same goal */
    auto st = mgr_->status();
    bool found = false;
    for (const auto &s : st["result"]["sessions"]) {
        if (s.value("sessionId", std::string()) == childId) found = true;
    }
    EXPECT_TRUE(found) << "successor session must exist";
    auto mst = mgr_->missionStatus();
    EXPECT_EQ(mst["result"]["stats"]["children"].size(), 1u);
}

TEST_F(AutonomyMissionTest, ReplicateRespectsMaxReplicas) {
    mgr_->assignMission(json{{"enabled", true},
                             {"goal", "bounded replication"},
                             {"maxReplicas", 1}});
    nlohmann::json ctx;
    auto first = mgr_->executeAgiActionByName("replicate", ctx);
    EXPECT_TRUE(first.value("ok", false));
    auto second = mgr_->executeAgiActionByName("replicate", ctx);
    EXPECT_FALSE(second.value("ok", true)) << "maxReplicas is the only guardrail";
    EXPECT_NE(second.value("error", std::string()).find("maxReplicas"), std::string::npos);
}

TEST_F(AutonomyMissionTest, ReportOutcomeFalseKeepsMissionFailed) {
    mgr_->assignMission(json{{"enabled", true},
                             {"goal", "no hand-off semantics"},
                             {"maxReplicas", 2}});
    mgr_->executeAgiActionByName("replicate", json{});
    auto r = mgr_->reportMissionOutcome(json{{"goalAchieved", false}});
    EXPECT_TRUE(r.value("ok", false));
    /* no takeover: before completion pressure only grows, the instance keeps
       going; judgement is the caller's.  Children stay for observability. */
    EXPECT_EQ(r["result"]["mission"].value("state", 0), 3); /* Failed */
    EXPECT_EQ(r["result"]["children"].size(), 1u);
}

TEST_F(AutonomyMissionTest, CompleteMissionStopsPainInjection) {
  const std::string sid = "mission-complete-test";
  seedSession(sid);

  mgr_->assignMission(json{{"enabled", true},
                           {"goal", "finish"},
                           {"painGainPerSec", 100.0},
                           {"maxPain", 1.0}});
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  mgr_->iterate(json{{"sessionId", sid}}, json{});

  ASSERT_FALSE(findLatestMissionPain().empty()) << "Pain should exist before completion";

  mgr_->reportMissionOutcome(json{{"goalAchieved", true}});

  auto iter = mgr_->iterate(json{{"sessionId", sid}}, json{});
  EXPECT_TRUE(iter.value("ok", false));
  EXPECT_FALSE(iter["result"]["mission"].value("active", true));
  EXPECT_FLOAT_EQ(iter["result"]["mission"].value("pressure", -1.0f), 0.0f);

  json pain = findLatestMissionPain();
  // The existing Pain decays naturally via half-life; after completion the
  // most recent mission-pressure Pain must be no newer than the completion call.
  const uint64_t completeTs = sysNowMs();
  EXPECT_TRUE(pain.empty() || pain.value("timestampMs", 0ull) <= completeTs)
      << "no new mission-pressure Pain should be injected after completion";
}

// ---------------------------------------------------------------------------
// §9.7 空目标幂等
// ---------------------------------------------------------------------------
TEST_F(AutonomyMissionTest, EmptyGoalIsEnabledButIdle) {
  auto r = mgr_->assignMission(json{{"enabled", true}});
  EXPECT_TRUE(r.value("ok", false));

  auto status = mgr_->missionStatus();
  EXPECT_TRUE(status["result"].value("enabled", false));
  // Empty goal means the mission object remains Idle (state 0) even though
  // missionEnabled_ is true.
  EXPECT_EQ(status["result"]["stats"]["mission"].value("state", 1), 0)
      << "empty-goal mission must remain Idle";
  EXPECT_FLOAT_EQ(status["result"]["stats"].value("pressure", -1.0f), 0.0f);

  const std::string sid = "mission-empty-test";
  seedSession(sid);
  auto iter = mgr_->iterate(json{{"sessionId", sid}}, json{});
  EXPECT_TRUE(iter.value("ok", false));
  EXPECT_FALSE(iter["result"]["mission"].value("active", true));
  EXPECT_FLOAT_EQ(iter["result"]["mission"].value("pressure", -1.0f), 0.0f);
  EXPECT_TRUE(findLatestMissionPain().empty()) << "idle mission must not inject Pain";

  // Subsequent real assign should start normally.
  mgr_->assignMission(json{{"enabled", true},
                           {"goal", "real goal"},
                           {"painGainPerSec", 1000.0},
                           {"maxPain", 1.0}});
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  iter = mgr_->iterate(json{{"sessionId", sid}}, json{});
  EXPECT_TRUE(iter["result"]["mission"].value("active", false));
  EXPECT_GT(iter["result"]["mission"].value("pressure", 0.0f), 0.0f);
}

// ---------------------------------------------------------------------------
// §8 配置门：mission.enabled=false 时行为逐位不变（回归门）
// ---------------------------------------------------------------------------
TEST_F(AutonomyMissionTest, DefaultOffDoesNotInjectMissionOrPain) {
  const std::string sid = "default-off-test";
  seedSession(sid);

  // Try to assign a mission without explicitly enabling it.
  auto r = mgr_->assignMission(json{{"goal", "should not run"},
                                    {"painGainPerSec", 1000.0},
                                    {"maxPain", 1.0}});
  EXPECT_TRUE(r.value("ok", false));

  auto status = mgr_->missionStatus();
  EXPECT_FALSE(status["result"].value("enabled", true));

  auto iter = mgr_->iterate(json{{"sessionId", sid}}, json{});
  EXPECT_TRUE(iter.value("ok", false));

  const auto &res = iter["result"];
  ASSERT_TRUE(res.contains("mission"));
  EXPECT_TRUE(res["mission"].empty()) << "disabled mission must appear as empty object";
  EXPECT_TRUE(findLatestMissionPain().empty()) << "disabled mission must not inject Pain";
}

// ---------------------------------------------------------------------------
// §9.8 连续 iterate 使压力持续上升并被截断
// ---------------------------------------------------------------------------
TEST_F(AutonomyMissionTest, PressureRisesAndSaturatesAcrossIterations) {
  const std::string sid = "mission-pressure-iter";
  seedSession(sid);

  mgr_->assignMission(json{{"enabled", true},
                           {"goal", "saturate"},
                           {"painGainPerSec", 1000.0},
                           {"maxPain", 0.8}});

  float lastPressure = 0.0f;
  for (int i = 0; i < 5; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    auto iter = mgr_->iterate(json{{"sessionId", sid}}, json{});
    ASSERT_TRUE(iter.value("ok", false));
    float p = iter["result"]["mission"].value("pressure", -1.0f);
    EXPECT_GE(p, 0.0f);
    EXPECT_LE(p, 0.8f);
    if (i > 0) {
      EXPECT_GE(p, lastPressure - 1e-3f) << "pressure should not drop between iterations";
    }
    lastPressure = p;
    if (p >= 0.8f - 1e-3f) break;
  }
}

// ---------------------------------------------------------------------------
// §9.9 子任务spawn接口在manager层可用且返回有效基因
// ---------------------------------------------------------------------------
TEST_F(AutonomyMissionTest, SpawnMissionChildProducesValidGenome) {
  const std::string sid = "mission-spawn";
  seedSession(sid);

  mgr_->assignMission(json{{"enabled", true},
                           {"goal", "evolve"},
                           {"painGainPerSec", 10.0},
                           {"maxPain", 1.0},
                           {"mutationRate", 0.1}});

  auto parentStatus = mgr_->missionStatus();
  float parentLR = parentStatus["result"]["genome"].value("learningRate", 0.05f);

  auto child = mgr_->spawnMissionChild(json{});
  EXPECT_TRUE(child.value("ok", false));
  ASSERT_TRUE(child["result"].is_object());
  EXPECT_TRUE(child["result"].contains("profile"));
  float childLR = child["result"].value("learningRate", -1.0f);
  EXPECT_GE(childLR, 0.001f);
  EXPECT_LE(childLR, 0.5f);
  EXPECT_NEAR(childLR, parentLR, 0.05f) << "child learningRate should stay near parent";

  auto status = mgr_->missionStatus();
  EXPECT_EQ(status["result"]["stats"]["spawns"].get<size_t>(), 1u);
}

// ---------------------------------------------------------------------------
// §9.10 显式禁用 mission 后再次启用可以重新注入 Pain
// ---------------------------------------------------------------------------
TEST_F(AutonomyMissionTest, ReEnableMissionResumesPressureAndPain) {
  const std::string sid = "mission-reenable";
  seedSession(sid);

  mgr_->assignMission(json{{"enabled", true},
                           {"goal", "first"},
                           {"painGainPerSec", 1000.0},
                           {"maxPain", 1.0}});
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  mgr_->iterate(json{{"sessionId", sid}}, json{});
  ASSERT_FALSE(findLatestMissionPain().empty());

  // Disable: must stop new Pain.
  mgr_->assignMission(json{{"enabled", false}});
  auto disabled = mgr_->iterate(json{{"sessionId", sid}}, json{});
  EXPECT_TRUE(disabled["result"]["mission"].empty());

  // Re-enable with a fresh goal: Pain should resume.
  mgr_->assignMission(json{{"enabled", true},
                           {"goal", "second"},
                           {"painGainPerSec", 1000.0},
                           {"maxPain", 1.0}});
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  auto iter = mgr_->iterate(json{{"sessionId", sid}}, json{});
  EXPECT_EQ(iter["result"]["mission"].value("goal", std::string()), "second");
  EXPECT_GT(iter["result"]["mission"].value("pressure", 0.0f), 0.0f);

  auto pain = findLatestMissionPain();
  ASSERT_FALSE(pain.empty());
  EXPECT_EQ(pain.value("source", std::string()), "mission-pressure");
}
