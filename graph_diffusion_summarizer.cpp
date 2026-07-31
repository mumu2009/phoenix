/* graph_diffusion_summarizer.cpp - Graph-diffusion summarization for MemeGraph
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version. */

#include "graph_diffusion_summarizer.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

namespace phoenix {
namespace graph {

nlohmann::json DiffusionSummary::toJson() const {
  nlohmann::json j;
  j["summaryText"] = summaryText;
  j["rankedNodes"] = nlohmann::json::array();
  for (const auto &p : rankedNodes) {
    j["rankedNodes"].push_back({{"id", p.first}, {"score", p.second}});
  }
  return j;
}

DiffusionSummary GraphDiffusionSummarizer::summarize(
    const std::vector<std::string> &ids,
    const std::vector<std::vector<std::tuple<size_t, double, int>>> &adjacency,
    const std::vector<double> &seedScores,
    int rounds,
    double damping,
    size_t topK) const {
  DiffusionSummary out;
  if (ids.empty())
    return out;

  const size_t n = ids.size();
  std::vector<double> scores(n, 0.0);
  if (!seedScores.empty() && seedScores.size() == n) {
    scores = seedScores;
  } else {
    std::fill(scores.begin(), scores.end(), 1.0 / static_cast<double>(n));
  }

  // Normalize seed scores to a probability distribution (the teleport/personalization).
  double total = std::accumulate(scores.begin(), scores.end(), 0.0);
  std::vector<double> seedDist = scores;
  if (total > 0.0) {
    for (auto &s : seedDist)
      s /= total;
  } else {
    std::fill(seedDist.begin(), seedDist.end(), 1.0 / static_cast<double>(n));
  }

  // Pre-normalize adjacency weights per source node.
  std::vector<std::vector<std::pair<size_t, double>>> normalized(n);
  for (size_t i = 0; i < n; ++i) {
    if (i >= adjacency.size())
      continue;
    double wsum = 0.0;
    for (const auto &edge : adjacency[i]) {
      double w = std::get<1>(edge);
      int direction = std::get<2>(edge);
      if (direction == 1)
        w *= 0.85;  // inbound
      else if (direction == 2)
        w *= 1.15;  // outbound
      w = std::abs(w);
      if (w > 0.0)
        wsum += w;
    }
    if (wsum <= 0.0)
      continue;
    for (const auto &edge : adjacency[i]) {
      size_t to = std::get<0>(edge);
      double w = std::get<1>(edge);
      int direction = std::get<2>(edge);
      if (direction == 1)
        w *= 0.85;
      else if (direction == 2)
        w *= 1.15;
      w = std::abs(w) / wsum;
      if (w > 0.0 && to < n)
        normalized[i].push_back({to, w});
    }
  }

  scores = seedDist;
  for (int r = 0; r < rounds; ++r) {
    std::vector<double> next(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
      next[i] = (1.0 - damping) * seedDist[i];
      for (const auto &edge : normalized[i]) {
        next[edge.first] += damping * scores[i] * edge.second;
      }
    }
    scores = std::move(next);
  }

  std::vector<std::pair<size_t, double>> ranked;
  ranked.reserve(n);
  for (size_t i = 0; i < n; ++i)
    ranked.push_back({i, scores[i]});
  std::sort(ranked.begin(), ranked.end(),
            [](const auto &a, const auto &b) { return a.second > b.second; });

  out.rankedNodes.reserve(std::min(topK, n));
  for (size_t i = 0; i < std::min(topK, n); ++i)
    out.rankedNodes.push_back({ids[ranked[i].first], ranked[i].second});

  // Build a short natural-language summary from the top nodes.
  std::ostringstream oss;
  oss << "MemeGraph diffusion top context:";
  for (size_t i = 0; i < out.rankedNodes.size(); ++i) {
    oss << " " << (i + 1) << "." << out.rankedNodes[i].first
        << "(" << std::fixed << std::setprecision(3)
        << out.rankedNodes[i].second << ")";
  }
  out.summaryText = oss.str();
  return out;
}

DiffusionSummary GraphDiffusionSummarizer::summarizeFromJson(
    const nlohmann::json &graphResult,
    int rounds,
    double damping,
    size_t topK) const {
  DiffusionSummary out;
  if (!graphResult.is_object())
    return out;

  auto memes = graphResult.value("memes", nlohmann::json::array());
  auto act = graphResult.value("activation", nlohmann::json::array());
  auto edges = graphResult.value("edges", nlohmann::json::array());

  std::vector<std::string> ids;
  std::vector<double> seedScores;
  for (size_t i = 0; i < memes.size(); ++i) {
    if (!memes[i].is_string())
      continue;
    ids.push_back(memes[i].get<std::string>());
    double s = 0.0;
    if (act.is_array() && i < act.size() && act[i].is_number())
      s = act[i].get<double>();
    seedScores.push_back(s);
  }

  if (ids.empty())
    return out;

  std::unordered_map<std::string, size_t> index;
  for (size_t i = 0; i < ids.size(); ++i)
    index[ids[i]] = i;

  std::vector<std::vector<std::tuple<size_t, double, int>>> adjacency(ids.size());
  for (const auto &e : edges) {
    if (!e.is_object())
      continue;
    std::string from = e.value("from", "");
    std::string to = e.value("to", "");
    if (from.empty() || to.empty())
      continue;
    auto itFrom = index.find(from);
    auto itTo = index.find(to);
    if (itFrom == index.end() || itTo == index.end())
      continue;
    double w = e.value("weight", 1.0);
    int direction = e.value("direction", 0);
    adjacency[itFrom->second].push_back({itTo->second, w, direction});
  }

  return summarize(ids, adjacency, seedScores, rounds, damping, topK);
}

}  // namespace graph
}  // namespace phoenix
