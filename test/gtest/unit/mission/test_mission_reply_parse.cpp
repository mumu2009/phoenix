#include <gtest/gtest.h>

#include "mission_reply_parse.hpp"

using namespace phoenix::mission;

TEST(MissionReplyParse, ExtractMultipleJsonObjects) {
  const std::string raw =
      R"(JSON {"action":"write","path":"a.md","content":"hi"} JSON {"action":"append","path":"a.md","content":"there"})";
  const auto objs = extractJsonObjects(raw);
  ASSERT_EQ(objs.size(), 2u);
  EXPECT_EQ(objs[0]["action"], "write");
  EXPECT_EQ(objs[1]["action"], "append");
}

TEST(MissionReplyParse, MetaReplyDetection) {
  EXPECT_TRUE(isMissionMetaReply("[tool:search FAILED x]"));
  EXPECT_TRUE(isMissionMetaReply("[parse-fail] bad json"));
  EXPECT_FALSE(isMissionMetaReply("## Chapter 1\nReal content"));
}

TEST(MissionReplyParse, OutlineFromGoalChapters) {
  const std::string goal =
      "Task\nChapter 1: Alpha\n\nChapter 2: Beta\n\nChapter 3: Gamma";
  const std::string outline = outlineFromGoalChapters(goal);
  EXPECT_NE(outline.find("Chapter 1"), std::string::npos);
  EXPECT_NE(outline.find("Chapter 3"), std::string::npos);
}
