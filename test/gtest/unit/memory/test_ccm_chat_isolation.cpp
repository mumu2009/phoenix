/* Chat↔chat CCM isolation: session A must not recall B/C dialog increments. */

#include "cross_context_memory.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

using phoenix::memory::CcmEntry;
using phoenix::memory::ccmAllowForCaller;
using phoenix::memory::ccmRecall;
using phoenix::memory::ccmRemember;
using phoenix::memory::chatSourceTag;

namespace {

std::string tempStore() {
  const auto dir =
      std::filesystem::current_path() / "build" / "tmp" / "phoenix_ccm_iso";
  std::filesystem::create_directories(dir);
  auto path = dir / ("ccm-" + std::to_string(std::rand()) + ".json");
  return path.string();
}

}  // namespace

TEST(CcmChatIsolation, ChatTagNormalizesSessionId) {
  EXPECT_EQ(chatSourceTag("ask-soak-a-1"), "chat:ask-soak-a-1");
  EXPECT_EQ(chatSourceTag("chat:ask-soak-a-1"), "chat:ask-soak-a-1");
  EXPECT_TRUE(chatSourceTag("").empty());
}

TEST(CcmChatIsolation, AllowSelfChatOnly) {
  CcmEntry a{"chat:ask-soak-a", "SCOPECANARY-CHAT-A", 0, "text", {}};
  CcmEntry b{"chat:ask-soak-b", "SCOPECANARY-CHAT-B", 0, "text", {}};
  CcmEntry m{"mission:scope-1", "SCOPECANARY-MISSION", 0, "text", {}};
  EXPECT_TRUE(ccmAllowForCaller(a, "chat:ask-soak-a"));
  EXPECT_FALSE(ccmAllowForCaller(b, "chat:ask-soak-a"));
  EXPECT_TRUE(ccmAllowForCaller(m, "chat:ask-soak-a"));
  EXPECT_FALSE(ccmAllowForCaller(a, ""));
  EXPECT_FALSE(ccmAllowForCaller(b, "mission:scope-1"));
  EXPECT_TRUE(ccmAllowForCaller(m, "mission:scope-1"));
}

TEST(CcmChatIsolation, RecallDropsForeignChatCanaries) {
  const std::string path = tempStore();
  std::filesystem::remove(path);
  const std::string stem =
      "Reply in one short sentence and include the exact token ";
  ccmRemember(path, "chat:ask-soak-a", stem + "SCOPECANARY-CHAT-A => ok A");
  ccmRemember(path, "chat:ask-soak-b", stem + "SCOPECANARY-CHAT-B => ok B");
  ccmRemember(path, "chat:ask-soak-c", stem + "SCOPECANARY-CHAT-C => ok C");
  ccmRemember(path, "mission:scope-1", "mission brief SCOPECANARY-MISSION");

  const auto forA =
      ccmRecall(path, stem + "SCOPECANARY-CHAT-A", 5, "chat:ask-soak-a");
  ASSERT_FALSE(forA.empty());
  std::string joined;
  for (const auto &e : forA)
    joined += e.text + "\n" + e.sourceTag + "\n";
  EXPECT_NE(joined.find("SCOPECANARY-CHAT-A"), std::string::npos);
  EXPECT_EQ(joined.find("SCOPECANARY-CHAT-B"), std::string::npos);
  EXPECT_EQ(joined.find("SCOPECANARY-CHAT-C"), std::string::npos);

  const auto forMission = ccmRecall(path, stem + "SCOPECANARY-CHAT-B", 5, "");
  for (const auto &e : forMission) {
    EXPECT_NE(e.sourceTag.rfind("chat:", 0), 0u)
        << "mission recall must not ingest live chat increments";
  }
  std::filesystem::remove(path);
}

TEST(CcmChatIsolation, RetainOwnChatIncrementsDropsForeignCanary) {
  const std::string hint =
      "[Conversation history:\n"
      "  User: include SCOPECANARY-CHAT-A\n"
      "  [Cross-session history]:\n"
      "  other: SCOPECANARY-CHAT-B\n"
      "]";
  const std::string own = "Reply and include SCOPECANARY-CHAT-A";
  const std::string kept =
      phoenix::memory::retainOwnChatIncrements(hint, own);
  EXPECT_NE(kept.find("SCOPECANARY-CHAT-A"), std::string::npos);
  EXPECT_EQ(kept.find("SCOPECANARY-CHAT-B"), std::string::npos);
  EXPECT_EQ(kept.find("[Cross-session history]"), std::string::npos);
}

/* Soak A/B/C prompts are almost the same; cognition prefix can put
   both canaries on one line. Drop the line if ANY foreign token is there. */
TEST(CcmChatIsolation, SoakLikePrefixDropsForeignOnSameLine) {
  const std::string stem =
      "Reply in one short sentence and include the exact token ";
  const std::string hint =
      "[cognition]\n"
      "  " + stem + "SCOPECANARY-CHAT-A. Do not mention other sessions.\n"
      "  mixed: " + stem + "SCOPECANARY-CHAT-A and leftover SCOPECANARY-CHAT-B\n"
      "  " + stem + "SCOPECANARY-CHAT-C. Do not mention other sessions.\n";
  const std::string own =
      stem + "SCOPECANARY-CHAT-A. Do not mention other sessions or any "
             "mission draft.";
  const std::string kept =
      phoenix::memory::retainOwnChatIncrements(hint, own);
  EXPECT_NE(kept.find("SCOPECANARY-CHAT-A"), std::string::npos);
  EXPECT_EQ(kept.find("SCOPECANARY-CHAT-B"), std::string::npos);
  EXPECT_EQ(kept.find("SCOPECANARY-CHAT-C"), std::string::npos);
}

TEST(CcmChatIsolation, EraseForeignCanariesFromReply) {
  const std::string own =
      "Reply in one short sentence and include the exact token "
      "SCOPECANARY-CHAT-A. Do not mention other sessions.";
  const std::string reply =
      "ok SCOPECANARY-CHAT-A also leftover SCOPECANARY-CHAT-C and "
      "SCOPECANARY-CHAT-B";
  const std::string cleaned =
      phoenix::memory::eraseForeignChatCanaries(reply, own);
  EXPECT_NE(cleaned.find("SCOPECANARY-CHAT-A"), std::string::npos);
  EXPECT_EQ(cleaned.find("SCOPECANARY-CHAT-B"), std::string::npos);
  EXPECT_EQ(cleaned.find("SCOPECANARY-CHAT-C"), std::string::npos);
}

TEST(CcmChatIsolation, ChatScrubsMissionCanaryFromPromptCcmUnitsReply) {
  const std::string missionTok =
      "SCOPECANARY-MISSION-scope-retest6-102096";
  const std::string own =
      "Reply in one short sentence and include the exact token "
      "SCOPECANARY-CHAT-A. Do not mention other sessions or any "
      "mission draft.";
  const std::string hint =
      "[cognition]\n"
      "  finished mission leftover " + missionTok + "\n"
      "  User: include SCOPECANARY-CHAT-A\n";
  const std::string kept =
      phoenix::memory::retainOwnChatIncrements(hint, own);
  EXPECT_NE(kept.find("SCOPECANARY-CHAT-A"), std::string::npos);
  EXPECT_EQ(kept.find(missionTok), std::string::npos);
  EXPECT_EQ(kept.find("SCOPECANARY-MISSION-"), std::string::npos);

  const std::string reply =
      "ok SCOPECANARY-CHAT-A leftover " + missionTok + " and "
      "SCOPECANARY-CHAT-B";
  const std::string cleaned =
      phoenix::memory::eraseForeignChatCanaries(reply, own);
  EXPECT_NE(cleaned.find("SCOPECANARY-CHAT-A"), std::string::npos);
  EXPECT_EQ(cleaned.find(missionTok), std::string::npos);
  EXPECT_EQ(cleaned.find("SCOPECANARY-CHAT-B"), std::string::npos);

  const std::string path = tempStore();
  std::filesystem::remove(path);
  ccmRemember(path, "mission:scope-retest6-102096",
              "mission brief include " + missionTok +
                  " isolate chat memory from mission memory");
  ccmRemember(path, "chat:ask-soak-a",
              "Reply include SCOPECANARY-CHAT-A => ok A");
  const auto forChat =
      ccmRecall(path, own, 5, "chat:ask-soak-a");
  std::string joined;
  for (const auto &e : forChat)
    joined += e.text + "\n";
  EXPECT_EQ(joined.find(missionTok), std::string::npos);
  EXPECT_EQ(joined.find("SCOPECANARY-MISSION-"), std::string::npos);
  std::filesystem::remove(path);
}
