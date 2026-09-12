#include <gtest/gtest.h>

#include "mission_unit_workflow.hpp"
#include "context_window_pack.hpp"

#include <filesystem>
#include <fstream>

using namespace phoenix::mission;

TEST(MissionUnitWorkflow, EmotionVocabTokensAlnumLower) {
  const auto toks = splitEmotionVocabTokens("Helios 2035, uncrewed!");
  ASSERT_GE(toks.size(), 3u);
  EXPECT_EQ(toks[0], "helios");
  EXPECT_EQ(toks[1], "2035");
  EXPECT_EQ(toks[2], "uncrewed");
}

TEST(MissionUnitWorkflow, RecallQueryPutsGoalFirst) {
  const std::string draft(800, 'x');
  const std::string q = buildMemoryRecallQuery("GOALWORD helios", draft + " TAILMARK",
                                               80, 120);
  EXPECT_NE(q.find("TAILMARK"), std::string::npos);
  EXPECT_NE(q.find("GOALWORD"), std::string::npos);
  EXPECT_LT(q.find("GOALWORD"), q.find("TAILMARK"));
}

TEST(MissionUnitWorkflow, RecallQueryDefaultKeepsGoalAndDraftTail) {
  /* Goal leads. Draft tail is short so a drifted body cannot dominate. */
  std::string draft(9000, 'a');
  draft.replace(8200, 8, "MIDMARKX");
  draft.replace(8988, 8, "TAILMARK");
  const std::string q = buildMemoryRecallQuery("GOALWORD", draft);
  EXPECT_NE(q.find("TAILMARK"), std::string::npos);
  EXPECT_EQ(q.find("MIDMARKX"), std::string::npos);
  EXPECT_NE(q.find("GOALWORD"), std::string::npos);
  EXPECT_LT(q.find("GOALWORD"), q.find("TAILMARK"));
}

TEST(MissionUnitWorkflow, PluginSearchQueryUsesThisGoal) {
  const std::string goal =
      "It is 2035. A science station has successfully landed.\n"
      "Due to communications delays of 8-40 minutes, operate 1000 sols.\n\n"
      "Chapter 1: Mission and Requirements Analysis\n";
  const std::string q = buildPluginSearchQuery(goal);
  EXPECT_NE(q.find("science station"), std::string::npos);
  EXPECT_EQ(q.find("Chapter 1"), std::string::npos);
  EXPECT_EQ(q.find("##"), std::string::npos);
}

TEST(MissionUnitWorkflow, PluginSearchSkipsTitleLineWithoutSentence) {
  const std::string goal =
      "Task Name\n"
      "Design the Helios Mars Surface Autonomous Science Station "
      "Long-Duration Operations Software System\n\n"
      "It is 2035. A Mars science station has successfully landed.\n";
  const auto qs = buildPluginSearchQueries(goal, "", 3);
  ASSERT_FALSE(qs.empty());
  EXPECT_EQ(qs.front().find("Design the"), std::string::npos);
  EXPECT_NE(qs.front().find("Mars science station"), std::string::npos);
  EXPECT_EQ(qs.front().rfind("It is 2035.", 0), std::string::npos);
}

TEST(MissionUnitWorkflow, PluginSearchQueriesBackgroundBeforeTitle) {
  const std::string goal =
      "Task Name\n"
      "Design the Surface Station Operations Software\n\n"
      "It is 2035. A science station has successfully landed.\n"
      "Due to communications delays of 8-40 minutes, operate 1000 sols.\n\n"
      "## Chapter 1: Mission and Requirements Analysis\n";
  const auto qs = buildPluginSearchQueries(goal, "", 3);
  ASSERT_FALSE(qs.empty());
  EXPECT_NE(qs.front().find("science station"), std::string::npos);
  bool sawTitle = false;
  bool sawDelay = false;
  for (const auto &q : qs) {
    if (q.find("Surface Station") != std::string::npos) sawTitle = true;
    if (q.find("8-40") != std::string::npos) sawDelay = true;
    EXPECT_EQ(q.find("## Chapter"), std::string::npos);
  }
  EXPECT_TRUE(sawTitle);
  EXPECT_TRUE(sawDelay);
}

TEST(MissionUnitWorkflow, ProjectRowsRepeatsShortVectors) {
  const auto rows = projectRowsToLlamaDim({{1.f, 2.f}}, 4);
  ASSERT_EQ(rows.size(), 1u);
  ASSERT_EQ(rows[0].size(), 4u);
  EXPECT_FLOAT_EQ(rows[0][0], 1.f);
  EXPECT_FLOAT_EQ(rows[0][1], 2.f);
  EXPECT_FLOAT_EQ(rows[0][2], 1.f);
  EXPECT_FLOAT_EQ(rows[0][3], 2.f);
}

TEST(MissionUnitWorkflow, DepositThenRecallSameMission) {
  const auto dir = std::filesystem::current_path() / "build" /
                   "phoenix_unit_workflow_test";
  std::filesystem::create_directories(dir);
  const std::string ccm = (dir / "ccm.json").string();
  const std::string exp = (dir / "exp.json").string();
  std::filesystem::remove(ccm);
  std::filesystem::remove(exp);
  const std::string draft =
      "### Night bus\n\nThe isolator sheds noncritical loads before dawn.";
  depositMissionProgress(ccm, exp, "mission:t1",
                         "Helios uncrewed science station 1000 sols", draft,
                         "The isolator sheds noncritical loads before dawn.");
  std::vector<phoenix::inference::UnitQueryIO> pkts;
  appendRecalledMemory(pkts, ccm, exp,
                       "Helios uncrewed science station 1000 sols", draft, 3,
                       3);
  ASSERT_FALSE(pkts.empty());
  bool saw = false;
  for (const auto &p : pkts) {
    if (p.content.find("isolator") != std::string::npos) saw = true;
  }
  EXPECT_TRUE(saw);
}

TEST(MissionUnitWorkflow, IoEncSplitsParagraphsToUnits) {
  const std::string blob = "First paragraph here.\n\nSecond paragraph here.";
  const auto paras = phoenix::inference::splitIoParagraphs(blob);
  ASSERT_EQ(paras.size(), 2u);
  EXPECT_EQ(paras[0], "First paragraph here.");
  EXPECT_EQ(paras[1], "Second paragraph here.");
  std::vector<phoenix::inference::UnitQueryIO> pkts;
  phoenix::inference::appendUnitQueriesFromParagraphs(pkts, blob, "text");
  ASSERT_EQ(pkts.size(), 2u);
  EXPECT_EQ(pkts[0].content, "First paragraph here.");
  EXPECT_EQ(pkts[1].content, "Second paragraph here.");
  EXPECT_TRUE(pkts[0].rows.empty());
}

TEST(MissionUnitWorkflow, CollapsedProseIsRejected) {
  EXPECT_TRUE(looksLikeCollapsedProse(
      "://acks toscalpel/tool forthe of switch: to. Huto, get<|ing/tear_The "
      "asparagus the. On agrad 2 D. endof 1) when 0. data-uml;](worst to,"));
  EXPECT_FALSE(looksLikeCollapsedProse(
      "The isolator sheds noncritical loads before dawn on sol 1000."));
}

TEST(MissionUnitWorkflow, IoEncKeepsSingleBlobAsOneUnit) {
  const std::string one = "Only one paragraph, even with a single\nline break.";
  const auto paras = phoenix::inference::splitIoParagraphs(one);
  ASSERT_EQ(paras.size(), 1u);
}

TEST(MissionUnitWorkflow, RagMixIsResidualNotConcat) {
  using phoenix::inference::modulateEncHidden;
  using phoenix::inference::ragAttend;
  using phoenix::inference::ragMixIntoCausal;
  const std::vector<std::vector<float>> causal = {
      {1.f, 0.f, 0.f, 0.f},
      {0.f, 1.f, 0.f, 0.f},
      {0.f, 0.f, 1.f, 0.f},
      {0.f, 0.f, 0.f, 1.f},
  };
  const std::vector<std::vector<float>> mem = {{10.f, 0.f, 0.f, 0.f}};
  const auto att = ragAttend(causal[0], mem);
  ASSERT_EQ(att.size(), 4u);
  EXPECT_FLOAT_EQ(att[0], 10.f);
  auto mixed = causal;
  ragMixIntoCausal(mixed, mem, 0.2f, 2);
  ASSERT_EQ(mixed.size(), causal.size());
  EXPECT_GT(mixed[0][0], causal[0][0]);
  EXPECT_FLOAT_EQ(mixed[2][2], 1.f);
  EXPECT_FLOAT_EQ(mixed[3][3], 1.f);
  const auto out = modulateEncHidden(causal, mem, {}, 1.f, 1.f);
  ASSERT_EQ(out.size(), causal.size());
}

TEST(MissionUnitWorkflow, NgramMergeCompactsAdjacentUnits) {
  using phoenix::inference::mergeNgramUnits;
  using phoenix::inference::ngramMergeFactor;
  std::vector<std::vector<float>> rows = {
      {2.f, 0.f}, {4.f, 2.f}, {6.f, 4.f}, {8.f, 6.f}, {10.f, 8.f},
  };
  auto a = rows;
  mergeNgramUnits(a, 2, 1);
  ASSERT_EQ(a.size(), 3u);
  EXPECT_FLOAT_EQ(a[0][0], 3.f);
  EXPECT_FLOAT_EQ(a[1][0], 7.f);
  EXPECT_FLOAT_EQ(a[2][0], 10.f);
  auto b = rows;
  mergeNgramUnits(b, 3, 0);
  ASSERT_EQ(b.size(), 2u);
  EXPECT_FLOAT_EQ(b[0][0], 4.f);
  EXPECT_FLOAT_EQ(b[1][0], 9.f);
  auto c = rows;
  mergeNgramUnits(c, 1, 0);
  ASSERT_EQ(c.size(), 5u);
  EXPECT_EQ(ngramMergeFactor(2), 2);
  EXPECT_EQ(ngramMergeFactor(0), 1);
}

TEST(MissionUnitWorkflow, RagProtectLeavesShortPrefixOpen) {
  using phoenix::inference::ragProtectUnits;
  EXPECT_EQ(ragProtectUnits(40, 0, 16), 0);
  EXPECT_EQ(ragProtectUnits(40, 8, 16), 0);
  EXPECT_EQ(ragProtectUnits(200, 80, 16), 16);
  EXPECT_EQ(ragProtectUnits(20, 80, 16), 5);
}

TEST(MissionUnitWorkflow, EstimateTokensTwoWordsPerPackedToken) {
  EXPECT_EQ(phoenix::context::estimateTokens("alpha beta gamma delta"), 2u);
  EXPECT_EQ(phoenix::context::charsPerPackedToken(), 6u);
}

TEST(MissionUnitWorkflow, PackSplitsRecentCausalFromSummaryRag) {
  phoenix::context::PackOptions opt;
  opt.ctxTokens = 4096;
  opt.replyReserveTokens = 512;
  opt.overheadTokens = 256;
  opt.ngramMerge = 2;
  opt.includeGnnSummary = true;
  const std::string full(50000, 'a');
  const auto packed = phoenix::context::packContext(full, "gnn-pin-text", opt);
  EXPECT_FALSE(packed.recentFull.empty());
  EXPECT_EQ(packed.recentFull, full.substr(full.size() - packed.recentFull.size()));
  EXPECT_EQ(packed.causalTokenBudget, 8192u);
  EXPECT_GT(packed.fullCharsUsed, 8000u);
  EXPECT_TRUE(packed.usedGnn);
  EXPECT_EQ(packed.gnnPinned.find("gnn-pin"), 0u);
  auto opt1 = opt;
  opt1.ngramMerge = 1;
  const auto thin = phoenix::context::packContext(full, "", opt1);
  EXPECT_GT(packed.fullCharsUsed, thin.fullCharsUsed);
}

TEST(MissionUnitWorkflow, EmotionScalesUnitWeightsAndTemperature) {
  nlohmann::json opts{{"temperature", 0.35},
                      {"top_p", 0.9},
                      {"memoryWeight", 1.0},
                      {"gnnWeight", 1.0}};
  const nlohmann::json emo{{"temperature", 1.2}, {"top_p", 0.5}};
  const nlohmann::json tensor{{"arousal", 0.8}};
  applyEmotionSampling(opts, emo, tensor, 0.6f);
  EXPECT_GT(opts["temperature"].get<double>(), 0.35);
  EXPECT_GT(opts["memoryWeight"].get<double>(), 1.0);
  EXPECT_GT(opts["gnnWeight"].get<double>(), 1.0);
}
