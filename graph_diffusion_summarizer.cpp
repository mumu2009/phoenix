/* graph_diffusion_summarizer.cpp - Graph-diffusion summarization for MemeGraph */

#include "graph_diffusion_summarizer.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <unordered_map>

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
  const bool explicitSeeds = !seedScores.empty() && seedScores.size() == n;
  if (explicitSeeds)
    scores = seedScores;
  else
    std::fill(scores.begin(), scores.end(), 1.0 / static_cast<double>(n));

  // Normalize seed scores to a probability distribution (the teleport).
  // Explicit all-zero seeds stay zero: do not fall back to uniform activation,
  // which would light up every memory and look like a global variable.
  double total = std::accumulate(scores.begin(), scores.end(), 0.0);
  std::vector<double> seedDist = scores;
  if (total > 0.0) {
    for (auto &s : seedDist)
      s /= total;
  } else if (!explicitSeeds) {
    std::fill(seedDist.begin(), seedDist.end(), 1.0 / static_cast<double>(n));
  } else {
    std::fill(seedDist.begin(), seedDist.end(), 0.0);
  }

  // Raw unsigned weights (direction-scaled) plus two transitions:
  // random-walk (row-normalized) and GCN (symmetric D^{-1/2} A D^{-1/2}
  // with a self-loop). Dense / high-degree graphs mix toward GCN so
  // hubs absorb less query mass. Teleport grows with seed entropy:
  // a flat query-conditioned seed walks less.
  std::vector<std::vector<std::pair<size_t, double>>> raw(n);
  std::vector<double> deg(n, 1.0);
  double edgeSum = 0.0;
  int edgeN = 0;
  for (size_t i = 0; i < n; ++i) {
    if (i >= adjacency.size())
      continue;
    raw[i].reserve(adjacency[i].size());
    for (const auto &edge : adjacency[i]) {
      size_t to = std::get<0>(edge);
      if (to >= n)
        continue;
      double w = std::get<1>(edge);
      int direction = std::get<2>(edge);
      if (direction == 1)
        w *= 0.85;
      else if (direction == 2)
        w *= 1.15;
      w = std::abs(w);
      if (w <= 0.0)
        continue;
      raw[i].push_back({to, w});
      deg[i] += w;
      edgeSum += w;
      ++edgeN;
    }
  }
  std::vector<std::vector<std::pair<size_t, double>>> walk(n);
  std::vector<std::vector<std::pair<size_t, double>>> gcn(n);
  for (size_t i = 0; i < n; ++i) {
    double wsum = deg[i] - 1.0;
    walk[i].reserve(raw[i].size());
    gcn[i].reserve(raw[i].size());
    for (const auto &wp : raw[i]) {
      if (wsum > 0.0)
        walk[i].push_back({wp.first, wp.second / wsum});
      const double den = std::sqrt(deg[i] * deg[wp.first]);
      if (den > 0.0)
        gcn[i].push_back({wp.first, wp.second / den});
    }
  }
  const double meanDeg =
      edgeN > 0 ? edgeSum / static_cast<double>(n) : 0.0;
  const double hubMix = meanDeg / (meanDeg + 4.0);
  double entropy = 0.0;
  for (double s : seedDist) {
    if (s > 1e-15)
      entropy -= s * std::log(s);
  }
  const double hNorm =
      n > 1 ? entropy / std::log(static_cast<double>(n)) : 0.0;
  double teleport = 1.0 - damping;
  if (teleport < 0.05)
    teleport = 0.05;
  if (teleport > 0.50)
    teleport = 0.50;
  teleport *= 1.0 + std::max(0.0, std::min(1.0, hNorm));
  if (teleport > 0.40)
    teleport = 0.40;
  const double walkKeep = 1.0 - teleport;

  scores = seedDist;
  const double kConvergenceTol = 1e-9;
  for (int r = 0; r < rounds; ++r) {
    std::vector<double> next(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
      next[i] = teleport * seedDist[i];
      next[i] += walkKeep * hubMix * (scores[i] / deg[i]);
    }
    for (size_t i = 0; i < n; ++i) {
      const double rwScale = walkKeep * (1.0 - hubMix) * scores[i];
      for (const auto &edge : walk[i])
        next[edge.first] += rwScale * edge.second;
      const double gcnScale = walkKeep * hubMix * scores[i];
      for (const auto &edge : gcn[i])
        next[edge.first] += gcnScale * edge.second;
    }
    // Power iteration converges geometrically; stop early when the L1 step is
    // negligible so lightly-connected graphs skip the remaining rounds.
    double delta = 0.0;
    for (size_t i = 0; i < n; ++i) {
      delta += std::fabs(next[i] - scores[i]);
    }
    scores = std::move(next);
    if (delta <= kConvergenceTol) {
      break;
    }
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
