/* Runtime (non-trainable) state isolation across scopes.
   Covers: primal sensation context filtering, per-owner FIFO eviction,
   scope-tagged interjections, and per-scope benefit/harm bias writes. */

#include "autonomy_stack.hpp"
#include "memory_scope.hpp"
#include "primal_sensation.hpp"
#include "scoped_trainable_memory.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using autonomy::CognitionAutonomyManager;
using autonomy::json;
using phoenix::memory::ScopedTrainableMemory;
using phoenix::memory::evictScopedFifoForInsert;
using phoenix::memory::makeChatScope;
using phoenix::primal::PrimalSensation;
using phoenix::primal::PrimalSensationEngine;
using phoenix::primal::SensationType;

namespace {

PrimalSensation makeSensation(SensationType type, float intensity, float valence,
                              const std::string &source) {
  PrimalSensation s;
  s.type = type;
  s.intensity = intensity;
  s.valence = valence;
  s.source = source;
  return s;
}

} // namespace

/* Mission A's pain must not leak into chat B's arousal/cost aggregates. */
TEST(RuntimeScopeIsolationTest, SensationAggregatesAreContextScoped) {
  PrimalSensationEngine eng;
  eng.add(makeSensation(SensationType::Pain, 0.9f, -0.9f,
                        "mission:helios:pressure"));
  eng.add(makeSensation(SensationType::Pleasure, 0.4f, 0.4f,
                        "chat:s1:dialog"));

  /* Chat scope sees only its own (and global) signals.  Homeostatic cost is
     gain*|intensity-setpoint| over the visible set, so s1's pleasure (0.4,
     untuned setpoint 0) contributes 0.4 while s2 sees nothing. */
  EXPECT_FLOAT_EQ(eng.netArousalFor("chat:s2"), 0.0f);
  EXPECT_FLOAT_EQ(eng.homeostaticCostFor("chat:s2"), 0.0f);
  EXPECT_FLOAT_EQ(eng.netArousalFor("chat:s1"), 0.4f);
  EXPECT_FLOAT_EQ(eng.homeostaticCostFor("chat:s1"), 0.4f);

  /* Mission scope sees its own pain; prefix match must not cross missions. */
  EXPECT_FLOAT_EQ(eng.netArousalFor("mission:helios"), 0.9f);
  EXPECT_FLOAT_EQ(eng.homeostaticCostFor("mission:helios"), 0.9f);
  EXPECT_FLOAT_EQ(eng.netArousalFor("mission:ops"), 0.0f);
  EXPECT_FLOAT_EQ(eng.homeostaticCostFor("mission:ops"), 0.0f);

  /* Empty tag keeps legacy global behavior. */
  EXPECT_FLOAT_EQ(eng.netArousalFor(""), eng.netArousal());
  EXPECT_FLOAT_EQ(eng.homeostaticCostFor(""), eng.homeostaticCost());

  /* Global (source-less) signals are visible to every scope. */
  eng.add(makeSensation(SensationType::Fatigue, 0.5f, -0.5f, ""));
  EXPECT_FLOAT_EQ(eng.netArousalFor("chat:s2"), 0.5f);
  EXPECT_FLOAT_EQ(eng.netArousalFor("mission:ops"), 0.5f);
  EXPECT_FLOAT_EQ(eng.homeostaticCostFor("chat:s2"), 0.5f);
  EXPECT_FLOAT_EQ(eng.homeostaticCostFor("mission:helios"), 1.4f);
}

/* Per-owner FIFO: A exceeding its own cap evicts A's oldest only.
   Call order matches the frontend write path: evict BEFORE the insert. */
TEST(RuntimeScopeIsolationTest, ScopedFifoEvictionIsolatesOwners) {
  struct Entry {
    std::string owner;
    std::string text;
  };
  std::vector<Entry> entries;
  auto ownerOf = [](const Entry &e) { return e.owner; };

  for (int i = 0; i < 3; ++i) {
    evictScopedFifoForInsert(entries, std::string("chat:a"), 2, 100, ownerOf);
    entries.push_back({"chat:a", "a" + std::to_string(i)});
  }
  /* chat:a capped at 2, oldest dropped. */
  ASSERT_EQ(entries.size(), 2u);
  EXPECT_EQ(entries.front().text, "a1");
  EXPECT_EQ(entries.back().text, "a2");

  /* B's writes must not evict A's entries. */
  for (int i = 0; i < 5; ++i) {
    evictScopedFifoForInsert(entries, std::string("chat:b"), 2, 100, ownerOf);
    entries.push_back({"chat:b", "b" + std::to_string(i)});
  }
  ASSERT_EQ(entries.size(), 4u);
  int aCount = 0, bCount = 0;
  for (const auto &e : entries) {
    if (e.owner == "chat:a") ++aCount;
    if (e.owner == "chat:b") ++bCount;
  }
  EXPECT_EQ(aCount, 2);
  EXPECT_EQ(bCount, 2);
  EXPECT_EQ(entries.front().text, "a1");

  /* Global cap is only a memory guard: it kicks in past the total limit. */
  entries.clear();
  for (int i = 0; i < 4; ++i) {
    evictScopedFifoForInsert(entries, std::string("chat:a"), 10, 5, ownerOf);
    entries.push_back({"chat:a", "ga" + std::to_string(i)});
  }
  for (int i = 0; i < 4; ++i) {
    evictScopedFifoForInsert(entries, std::string("chat:b"), 10, 5, ownerOf);
    entries.push_back({"chat:b", "gb" + std::to_string(i)});
  }
  EXPECT_EQ(entries.size(), 5u);
}

/* A scope-tagged interjection is consumed only by its own scope's tick. */
TEST(RuntimeScopeIsolationTest, TaggedInterjectionWaitsForItsOwnScope) {
  ScopedTrainableMemory::instance().resetForTest();
  CognitionAutonomyManager mgr;

  mgr.interject(json{{"text", "for-helios"}, {"missionId", "helios"}});
  mgr.interject(json{{"text", "for-chat1"}, {"sessionId", "chat-1"}});

  /* Unrelated chat tick must not consume either tagged interjection. */
  json r0 = mgr.iterate(json{{"sessionId", "chat-2"}}, json{});
  EXPECT_EQ(r0["result"].value("interjectionsConsumed", -1), 0);

  /* The owning chat consumes only its own. */
  json r1 = mgr.iterate(json{{"sessionId", "chat-1"}}, json{});
  EXPECT_EQ(r1["result"].value("interjectionsConsumed", -1), 1);

  /* The owning mission consumes the remaining one. */
  json r2 = mgr.iterate(json{{"missionId", "helios"}}, json{});
  EXPECT_EQ(r2["result"].value("interjectionsConsumed", -1), 1);

  /* Nothing left for anyone. */
  json r3 = mgr.iterate(json{{"sessionId", "chat-9"}}, json{});
  EXPECT_EQ(r3["result"].value("interjectionsConsumed", -1), 0);
  ScopedTrainableMemory::instance().resetForTest();
}

/* Legacy untagged interjections stay global: any tick drains them. */
TEST(RuntimeScopeIsolationTest, UntaggedInterjectionStillGlobal) {
  ScopedTrainableMemory::instance().resetForTest();
  CognitionAutonomyManager mgr;

  mgr.interject(json{{"text", "global-note"}});
  json r = mgr.iterate(json{{"sessionId", "chat-x"}}, json{});
  EXPECT_EQ(r["result"].value("interjectionsConsumed", -1), 1);
  ScopedTrainableMemory::instance().resetForTest();
}

/* iterate() writes the benefit/harm bias into the ticking scope's bucket. */
TEST(RuntimeScopeIsolationTest, IterateWritesBenefitHarmBiasPerScope) {
  ScopedTrainableMemory::instance().resetForTest();
  CognitionAutonomyManager mgr;

  mgr.iterate(json{{"sessionId", "bias-a"}}, json{});

  auto &reg = ScopedTrainableMemory::instance();
  const auto *bucketA = reg.peek(makeChatScope("bias-a"));
  ASSERT_NE(bucketA, nullptr);
  EXPECT_FALSE(bucketA->lastBenefitHarmBias.empty());
  EXPECT_EQ(reg.peek(makeChatScope("bias-b")), nullptr);
  ScopedTrainableMemory::instance().resetForTest();
}

/* agiPlan drive cost must only aggregate the requesting scope's pain. */
TEST(RuntimeScopeIsolationTest, AgiPlanDriveCostIsScopeFiltered) {
  ScopedTrainableMemory::instance().resetForTest();
  CognitionAutonomyManager mgr;
  json cfg = mgr.configureAgi(json{{"enabled", true}, {"dim", 8}});
  ASSERT_TRUE(cfg.value("ok", false)) << cfg.dump();

  mgr.ingestSensation(json{{"type", "pain"},
                           {"intensity", 0.9},
                           {"source", "mission:helios:pressure"}});

  json planOps = mgr.agiPlan(json{{"missionId", "ops"}});
  EXPECT_FLOAT_EQ(planOps["result"].value("driveCost", -1.0), 0.0);

  json planHelios = mgr.agiPlan(json{{"missionId", "helios"}});
  EXPECT_FLOAT_EQ(planHelios["result"].value("driveCost", -1.0), 0.9);
  ScopedTrainableMemory::instance().resetForTest();
}
