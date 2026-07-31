/* graph_diffusion_summarizer.hpp - Graph-diffusion summarization for MemeGraph
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version. */

#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace phoenix {
namespace graph {

/**
 * @brief Result of a graph-diffusion summarization pass.
 *
 * The summarizer spreads seed activation through the MemeGraph edges using
 * a PageRank-like iterative diffusion.  The resulting 
 scores identify the
 * most contextually relevant memory nodes and can be turned into a short
 * natural-language summary or a ranked keyword list.
 */
struct DiffusionSummary {
  /** Ranked node ids with diffusion score. */
  std::vector<std::pair<std::string, double>> rankedNodes;
  /** Optional short text summary built from top nodes. */
  std::string summaryText;
  /** JSON payload for downstream consumers. */
  nlohmann::json toJson() const;
};

/**
 * @brief GNN-style graph-diffusion summarizer for MemeGraph.
 *
 * Accepts a sparse graph represented by ids and an adjacency/weight list
 * (the same format used by MemeGraph::WindowInfo) and runs a few rounds of
 * power-iteration diffusion.  The output can be consumed by the GNN hint
 * builder to weight prompt context.
 */
class GraphDiffusionSummarizer {
 public:
  GraphDiffusionSummarizer() = default;

  /**
   * @brief Run diffusion on an explicit adjacency list.
   *
   * @param ids          ordered list of node ids.
   * @param adjacency    for each node i, a list of {target-index, weight, direction}.
   * @param seedScores   optional initial activation per node (size must match ids).
   * @param rounds       number of diffusion iterations.
   * @param damping      restart probability (1 - damping keeps random-walk mass).
   * @param topK         number of top nodes to return.
   * @return DiffusionSummary containing ranked nodes and text.
   */
  DiffusionSummary summarize(
      const std::vector<std::string> &ids,
      const std::vector<std::vector<std::tuple<size_t, double, int>>> &adjacency,
      const std::vector<double> &seedScores = {},
      int rounds = 4,
      double damping = 0.85,
      size_t topK = 8) const;

  /**
   * @brief Convenience overload that works from a JSON graph result.
   *
   * Expected input:
   *   {
   *     "memes": ["id1", "id2", ...],
   *     "activation": [0.5, 0.3, ...],
   *     "edges": [{"from": 0, "to": 1, "weight": 0.9}, ...]
   *   }
   */
  DiffusionSummary summarizeFromJson(const nlohmann::json &graphResult,
                                     int rounds = 4,
                                     double damping = 0.85,
                                     size_t topK = 8) const;
};

}  // namespace graph
}  // namespace phoenix
