/* Isolation of live trainable memory by MemoryScope{Chat|Mission, id}. */

#include "memory_scope.hpp"
#include "scoped_trainable_memory.hpp"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

using phoenix::memory::GnnOnlineOverlay;
using phoenix::memory::MemoryKind;
using phoenix::memory::MemoryScope;
using phoenix::memory::MemoryScopeGuard;
using phoenix::memory::ScopedTrainableMemory;
using phoenix::memory::makeChatScope;
using phoenix::memory::makeMissionScope;
using phoenix::memory::memoryScopeForChatRoute;
using phoenix::memory::memoryScopeFromIds;
using phoenix::memory::memoryScopeFromPayload;
using phoenix::memory::parseMemoryScope;
using json = nlohmann::json;

class MemoryScopeIsolationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ScopedTrainableMemory::instance().resetForTest();
  }
  void TearDown() override {
    ScopedTrainableMemory::instance().resetForTest();
  }
};

TEST_F(MemoryScopeIsolationTest, KeySeparatesSameIdAcrossKinds) {
  auto chat = makeChatScope("abc");
  auto mission = makeMissionScope("abc");
  EXPECT_EQ(chat.key(), "chat:abc");
  EXPECT_EQ(mission.key(), "mission:abc");
  EXPECT_NE(chat, mission);
}

TEST_F(MemoryScopeIsolationTest, ChatPayloadNeverInheritsMissionId) {
  json body{{"sessionId", "chat-sess-1"}, {"text", "hello"}};
  auto s = memoryScopeFromPayload(body);
  EXPECT_EQ(s.kind, MemoryKind::Chat);
  EXPECT_EQ(s.id, "chat-sess-1");
}

TEST_F(MemoryScopeIsolationTest, MissionIdWinsOverSessionId) {
  json body{{"sessionId", "chat-sess-1"},
            {"missionId", "helios-1"},
            {"text", "hello"}};
  auto s = memoryScopeFromPayload(body);
  EXPECT_EQ(s.kind, MemoryKind::Mission);
  EXPECT_EQ(s.id, "helios-1");
}

TEST_F(MemoryScopeIsolationTest, ParsePrefixedSessionId) {
  EXPECT_EQ(parseMemoryScope("mission:ops-9").kind, MemoryKind::Mission);
  EXPECT_EQ(parseMemoryScope("mission:ops-9").id, "ops-9");
  EXPECT_EQ(parseMemoryScope("chat:s1").kind, MemoryKind::Chat);
}

TEST_F(MemoryScopeIsolationTest, TwoMissionsDoNotShareTrainableWrites) {
  auto a = makeMissionScope("helios");
  auto b = makeMissionScope("ops");
  auto &ha = ScopedTrainableMemory::instance().hot(a);
  auto &hb = ScopedTrainableMemory::instance().hot(b);

  ha.recurrent.rnnHidden = {1.0f, 2.0f, 3.0f};
  ha.recurrent.lstmHidden = {9.0f};
  ha.recurrent.lstmCell = {8.0f};
  ha.gnn.add("meme-helios", "meme-station", 1.5);
  ha.conceptMatrix.encodeText("Helios mars station 1000 sols");
  ha.hier.put("cognition", json{{"goal", "helios"}}, 0.8);
  ha.dialog.remember("sig-a", "q-helios", "reply-helios", {"m1"}, 0.9);
  ha.gnnSummary = "helios-outline";

  hb.recurrent.rnnHidden = {0.1f};
  hb.gnn.add("meme-ops", "meme-bus", 0.4);
  hb.conceptMatrix.encodeText("night bus contactor isolation");
  hb.hier.put("cognition", json{{"goal", "ops"}}, 0.2);
  hb.gnnSummary = "ops-outline";

  EXPECT_EQ(ha.recurrent.rnnHidden.size(), 3u);
  EXPECT_EQ(hb.recurrent.rnnHidden.size(), 1u);
  EXPECT_FLOAT_EQ(ha.gnn.weight("meme-helios", "meme-station"), 1.5);
  EXPECT_FLOAT_EQ(hb.gnn.weight("meme-helios", "meme-station"), 0.0);
  EXPECT_FLOAT_EQ(hb.gnn.weight("meme-ops", "meme-bus"), 0.4);
  EXPECT_FLOAT_EQ(ha.gnn.weight("meme-ops", "meme-bus"), 0.0);
  EXPECT_EQ(ha.gnnSummary, "helios-outline");
  EXPECT_EQ(hb.gnnSummary, "ops-outline");
  EXPECT_GT(ha.conceptMatrix.activeCount(), 0u);
  EXPECT_GT(hb.conceptMatrix.activeCount(), 0u);
  EXPECT_NE(ha.conceptMatrix.toContextString(8, 0.01f),
            hb.conceptMatrix.toContextString(8, 0.01f));
  EXPECT_TRUE(ha.hier.get("cognition").has_value());
  EXPECT_EQ(ha.hier.get("cognition")->value("goal", ""), "helios");
  EXPECT_EQ(hb.hier.get("cognition")->value("goal", ""), "ops");
  EXPECT_EQ(ha.dialog.entries().size(), 1u);
  EXPECT_TRUE(hb.dialog.empty());
}

TEST_F(MemoryScopeIsolationTest, ChatDoesNotConsumeMissionIncrements) {
  auto mission = makeMissionScope("helios");
  auto chat = makeChatScope("user-session");
  auto &hm = ScopedTrainableMemory::instance().hot(mission);
  hm.recurrent.rnnHidden.assign(8, 0.75f);
  hm.recurrent.lstmHidden.assign(8, 0.5f);
  hm.recurrent.lstmCell.assign(8, 0.25f);
  hm.gnn.add("m1", "m2", 2.0);
  hm.conceptMatrix.encodeText("Helios uncrewed science station");
  hm.graphHint = "mission-graph-hint";

  auto *hc = ScopedTrainableMemory::instance().peek(chat);
  EXPECT_EQ(hc, nullptr);

  auto &fresh = ScopedTrainableMemory::instance().hot(chat);
  EXPECT_TRUE(fresh.recurrent.rnnHidden.empty());
  EXPECT_TRUE(fresh.recurrent.lstmHidden.empty());
  EXPECT_TRUE(fresh.gnn.empty());
  EXPECT_TRUE(fresh.graphHint.empty());
  EXPECT_EQ(fresh.conceptMatrix.activeCount(), 0u);
  EXPECT_FLOAT_EQ(fresh.gnn.weight("m1", "m2"), 0.0);
}

TEST_F(MemoryScopeIsolationTest, CompletedMissionLeavesNewChatClean) {
  auto mission = makeMissionScope("helios");
  auto &hm = ScopedTrainableMemory::instance().hot(mission);
  hm.recurrent.rnnHidden = {1.0f, 1.0f};
  hm.gnn.add("a", "b", 3.0);
  hm.conceptMatrix.encodeText("Helios");
  hm.dialog.remember("s", "q", "r", {"m"}, 1.0);
  EXPECT_TRUE(ScopedTrainableMemory::instance().hasLive(mission));

  ScopedTrainableMemory::instance().releaseMission("helios");
  EXPECT_FALSE(ScopedTrainableMemory::instance().hasLive(mission));

  auto chat = makeChatScope("post-mission-chat");
  auto &hc = ScopedTrainableMemory::instance().hot(chat);
  EXPECT_TRUE(hc.recurrent.rnnHidden.empty());
  EXPECT_TRUE(hc.gnn.empty());
  EXPECT_TRUE(hc.dialog.empty());
  EXPECT_EQ(hc.conceptMatrix.activeCount(), 0u);
}

TEST_F(MemoryScopeIsolationTest, GraphLinkUsesCurrentScopeOverlay) {
  MemoryScopeGuard ga(makeMissionScope("helios"));
  ScopedTrainableMemory::instance()
      .hot(phoenix::memory::currentMemoryScope())
      .gnn.add("x", "y", 1.0);
  {
    MemoryScopeGuard gb(makeMissionScope("ops"));
    ScopedTrainableMemory::instance()
        .hot(phoenix::memory::currentMemoryScope())
        .gnn.add("p", "q", 4.0);
  }
  EXPECT_FLOAT_EQ(ScopedTrainableMemory::instance()
                      .hot(makeMissionScope("helios"))
                      .gnn.weight("x", "y"),
                  1.0);
  EXPECT_FLOAT_EQ(ScopedTrainableMemory::instance()
                      .hot(makeMissionScope("helios"))
                      .gnn.weight("p", "q"),
                  0.0);
  EXPECT_FLOAT_EQ(ScopedTrainableMemory::instance()
                      .hot(makeMissionScope("ops"))
                      .gnn.weight("p", "q"),
                  4.0);
}

TEST_F(MemoryScopeIsolationTest, FromIdsNeverPromotesBareChatToMission) {
  auto s = memoryScopeFromIds("sess-9", "");
  EXPECT_EQ(s.kind, MemoryKind::Chat);
  EXPECT_EQ(s.id, "sess-9");
}

TEST_F(MemoryScopeIsolationTest, ChatRouteIgnoresStrayMissionId) {
  json body{{"sessionId", "chat-sess-1"},
            {"missionId", "helios"},
            {"text", "hello"}};
  auto s = memoryScopeForChatRoute(body);
  EXPECT_EQ(s.kind, MemoryKind::Chat);
  EXPECT_EQ(s.id, "chat-sess-1");
}

TEST_F(MemoryScopeIsolationTest, ChatRouteHonorsExplicitMemoryKind) {
  json body{{"sessionId", "chat-sess-1"},
            {"missionId", "helios"},
            {"memoryKind", "mission"},
            {"text", "hello"}};
  auto s = memoryScopeForChatRoute(body);
  EXPECT_EQ(s.kind, MemoryKind::Mission);
  EXPECT_EQ(s.id, "helios");
}

TEST_F(MemoryScopeIsolationTest, TwoMissionsDoNotShareAgiUpdates) {
  phoenix::agi::ActiveInferenceController tmpl(8, 1, 2);
  auto a = makeMissionScope("helios");
  auto b = makeMissionScope("ops");
  auto &agiA = ScopedTrainableMemory::instance().agiFor(a, tmpl);
  auto &agiB = ScopedTrainableMemory::instance().agiFor(b, tmpl);
  EXPECT_NE(&agiA, &agiB);
  std::vector<float> z(8, 0.1f);
  std::vector<float> act = {1.0f};
  std::vector<float> z2(8, 0.5f);
  agiA.observeRewarded(z, act, z2, 1.0, 0.1f, 0.2, 0.9);
  EXPECT_EQ(agiA.episodeCount(), 1u);
  EXPECT_EQ(agiB.episodeCount(), 0u);
}
