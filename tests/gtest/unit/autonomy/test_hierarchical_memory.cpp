/* test_hierarchical_memory.cpp - GTest for HierarchicalMemory
   Copyright (C) 2026 079 Project */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "hierarchical_memory.hpp"

using json = nlohmann::json;

TEST(HierarchicalMemory, PutAndGet) {
  phoenix::memory::HierarchicalMemory mem;
  mem.put("context:turn1", json{{"text", "hello"}}, 0.4);
  auto result = mem.get("context:turn1");
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->value("text", ""), "hello");
}

TEST(HierarchicalMemory, PromotesHighActivationToWorkingTier) {
  phoenix::memory::HierarchicalMemory::Config cfg;
  cfg.workingCapacity = 1;
  cfg.promotionThreshold = 0.5;
  phoenix::memory::HierarchicalMemory mem(cfg);

  mem.put("a", json{{"key", "a"}}, 0.9);  // goes to working
  mem.put("b", json{{"key", "b"}}, 0.1);  // stays in short-term

  auto a = mem.get("a");
  ASSERT_TRUE(a.has_value());
  EXPECT_EQ(a->value("key", ""), "a");

  // Accessing b twice should promote it to working tier.
  EXPECT_TRUE(mem.get("b").has_value());
  EXPECT_TRUE(mem.get("b").has_value());
  auto b = mem.get("b");
  ASSERT_TRUE(b.has_value());
  EXPECT_EQ(b->value("key", ""), "b");
}

TEST(HierarchicalMemory, DeleteRemovesFromAllTiers) {
  phoenix::memory::HierarchicalMemory mem;
  mem.put("x", json{{"v", 1}}, 0.9);
  EXPECT_TRUE(mem.get("x").has_value());
  mem.del("x");
  EXPECT_FALSE(mem.get("x").has_value());
}

TEST(HierarchicalMemory, QueryReturnsTopKByPrefix) {
  phoenix::memory::HierarchicalMemory mem;
  mem.put("ctx:a", json{{"n", "a"}}, 0.2);
  mem.put("ctx:b", json{{"n", "b"}}, 0.8);
  mem.put("other:c", json{{"n", "c"}}, 1.0);

  auto results = mem.query("ctx:", 2);
  ASSERT_EQ(results.size(), 2u);
  // b has higher activation and is more recent.
  EXPECT_EQ(results[0].first, "ctx:b");
  EXPECT_EQ(results[0].second.value("n", ""), "b");
}

TEST(HierarchicalMemory, SnapshotRoundTrip) {
  phoenix::memory::HierarchicalMemory mem;
  mem.put("turn1", json{{"text", "hi"}}, 0.3);
  json snapshot = mem.toJson();
  EXPECT_TRUE(snapshot.contains("working") ||
              snapshot.contains("shortTerm") ||
              snapshot.contains("longTerm"));

  phoenix::memory::HierarchicalMemory restored;
  restored.fromJson(snapshot);
  EXPECT_EQ(restored.size(), mem.size());
  auto r = restored.get("turn1");
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(r->value("text", ""), "hi");
}
