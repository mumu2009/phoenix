/* test_graph_diffusion_summarizer.cpp - GTest for GraphDiffusionSummarizer
   Copyright (C) 2026 079 Project */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "graph_diffusion_summarizer.hpp"

using json = nlohmann::json;

TEST(GraphDiffusionSummarizer, RanksSeedNodesWhenNoEdges) {
  phoenix::graph::GraphDiffusionSummarizer ds;
  std::vector<std::string> ids = {"a", "b", "c"};
  std::vector<std::vector<std::tuple<size_t, double, int>>> adjacency(ids.size());
  std::vector<double> seedScores = {0.1, 0.6, 0.3};
  auto result = ds.summarize(ids, adjacency, seedScores, 4, 0.85, 8);
  ASSERT_EQ(result.rankedNodes.size(), 3u);
  EXPECT_EQ(result.rankedNodes[0].first, "b");
  EXPECT_GT(result.rankedNodes[0].second, 0.0);
  EXPECT_FALSE(result.summaryText.empty());
}

TEST(GraphDiffusionSummarizer, DiffusionBoostsHighlyConnectedNode) {
  phoenix::graph::GraphDiffusionSummarizer ds;
  std::vector<std::string> ids = {"hub", "leaf1", "leaf2"};
  std::vector<std::vector<std::tuple<size_t, double, int>>> adjacency(ids.size());
  // hub <- leaves (inbound, lower weight)
  adjacency[1].push_back({0, 1.0, 1});  // leaf1 -> hub
  adjacency[2].push_back({0, 1.0, 1});  // leaf2 -> hub
  std::vector<double> seedScores = {0.1, 0.45, 0.45};  // hub starts low
  auto result = ds.summarize(ids, adjacency, seedScores, 8, 0.85, 3);
  ASSERT_EQ(result.rankedNodes.size(), 3u);
  // hub should rise to top after diffusion despite low seed score.
  EXPECT_EQ(result.rankedNodes[0].first, "hub");
}

TEST(GraphDiffusionSummarizer, SummarizeFromJson) {
  json graphResult;
  graphResult["memes"] = json::array({"x", "y", "z"});
  graphResult["activation"] = json::array({0.2, 0.7, 0.1});
  graphResult["edges"] = json::array({
      json{{"from", "y"}, {"to", "x"}, {"weight", 0.9}, {"direction", 0}},
      json{{"from", "y"}, {"to", "z"}, {"weight", 0.5}, {"direction", 0}}
  });

  phoenix::graph::GraphDiffusionSummarizer ds;
  auto result = ds.summarizeFromJson(graphResult, 4, 0.85, 2);
  ASSERT_EQ(result.rankedNodes.size(), 2u);
  // y is the strongest seed and also a diffusion source.
  EXPECT_EQ(result.rankedNodes[0].first, "y");
  json j = result.toJson();
  EXPECT_TRUE(j.contains("summaryText"));
  EXPECT_TRUE(j.contains("rankedNodes"));
}

TEST(GraphDiffusionSummarizer, ExplicitZeroSeedsDoNotActivateTheWholeGraph) {
  phoenix::graph::GraphDiffusionSummarizer ds;
  std::vector<std::string> ids = {"a", "b", "c"};
  std::vector<std::vector<std::tuple<size_t, double, int>>> adjacency(ids.size());
  adjacency[0].push_back({1, 1.0, 0});
  adjacency[1].push_back({0, 1.0, 0});
  auto result = ds.summarize(ids, adjacency, {0.0, 0.0, 0.0}, 4, 0.85, 3);
  ASSERT_EQ(result.rankedNodes.size(), 3u);
  for (const auto &row : result.rankedNodes)
    EXPECT_DOUBLE_EQ(row.second, 0.0);
}

TEST(GraphDiffusionSummarizer, EmptyInputReturnsEmptySummary) {
  phoenix::graph::GraphDiffusionSummarizer ds;
  auto result = ds.summarizeFromJson(json::object(), 4, 0.85, 8);
  EXPECT_TRUE(result.rankedNodes.empty());
  EXPECT_TRUE(result.summaryText.empty());
}
