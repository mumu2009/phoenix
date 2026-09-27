/* meme_semantic_slicer.cpp - see header for the design note. */

#include "meme_semantic_slicer.hpp"

#include <algorithm>
#include <cmath>

namespace phoenix {
namespace secamp {

namespace {

/* Count meme occurrences over the non-empty units in [pos, pos+limit)
   walking forward, or [pos-1, pos-1-limit) walking backward. Empty units
   are skipped so a run of short sentences cannot fake a seam. */
std::unordered_map<std::string, int>
windowCounts(const std::vector<SliceUnitFootprint> &units, int start, int step,
             int limit, int *refs) {
  std::unordered_map<std::string, int> counts;
  int taken = 0;
  int total = 0;
  for (int i = start; i >= 0 && i < static_cast<int>(units.size()) &&
                    taken < limit;
       i += step) {
    const auto &seq = units[static_cast<size_t>(i)].memeSeq;
    if (seq.empty())
      continue;
    ++taken;
    total += static_cast<int>(seq.size());
    for (const auto &id : seq)
      counts[id]++;
  }
  if (refs)
    *refs = total;
  return counts;
}

} // namespace

std::vector<double>
SemanticSlicer::gapFlows(const std::vector<SliceUnitFootprint> &units,
                         int lookahead) {
  std::vector<double> flows;
  const int n = static_cast<int>(units.size());
  if (n < 2)
    return flows;
  const int look = std::max(1, lookahead);
  flows.reserve(static_cast<size_t>(n - 1));
  for (int gap = 0; gap + 1 < n; ++gap) {
    int refsL = 0;
    int refsR = 0;
    const auto left = windowCounts(units, gap, -1, look, &refsL);
    const auto right = windowCounts(units, gap + 1, +1, look, &refsR);
    double shared = 0.0;
    for (const auto &kv : left) {
      auto it = right.find(kv.first);
      if (it != right.end())
        shared += std::min(kv.second, it->second);
    }
    const double den = std::sqrt(static_cast<double>(refsL) *
                                 static_cast<double>(refsR));
    flows.push_back(den > 1e-12 ? shared / den : 0.0);
  }
  return flows;
}

std::vector<SemanticSlice>
SemanticSlicer::slice(const std::vector<SliceUnitFootprint> &units,
                      const SemanticSliceConfig &cfg) {
  std::vector<SemanticSlice> out;
  const int n = static_cast<int>(units.size());
  if (n <= 0)
    return out;
  SemanticSliceConfig c = cfg;
  c.minUnits = std::max(1, c.minUnits);
  c.maxUnits = std::max(c.minUnits, c.maxUnits);
  c.targetUnits = std::max(c.minUnits, std::min(c.targetUnits, c.maxUnits));

  const auto flows = gapFlows(units, c.lookahead);

  auto emit = [&](int begin, int end, double seam) {
    SemanticSlice s;
    s.beginUnit = begin;
    s.endUnit = end;
    s.seamAfter = seam;
    std::unordered_set<std::string> uniq;
    for (int i = begin; i < end; ++i) {
      const auto &seq = units[static_cast<size_t>(i)].memeSeq;
      s.memeRefs += static_cast<int>(seq.size());
      for (const auto &id : seq)
        uniq.insert(id);
    }
    s.uniqueMemes = static_cast<int>(uniq.size());
    out.push_back(std::move(s));
  };

  int begin = 0;
  while (begin < n) {
    const int latest = std::min(n, begin + c.maxUnits);
    if (latest >= n) {
      emit(begin, n, 0.0);
      break;
    }
    const int earliest = begin + c.minUnits;
    /* Gap g closes a slice ending at g+1. Search g in
       [earliest-1, latest-1]; end = g+1 lands in [earliest, latest].
       (latest < n here, so g <= latest-1 <= n-2 always indexes flows.)
       Skip ends that would leave a tail smaller than minUnits; if every
       candidate does, merge the tail into this slice (final slice may
       exceed maxUnits by less than minUnits - documented contract). */
    int bestGap = -1;
    double bestFlow = 0.0;
    int bestDist = 0;
    for (int g = earliest - 1; g < latest; ++g) {
      const int remainder = n - (g + 1);
      if (remainder > 0 && remainder < c.minUnits)
        continue;
      const double f = flows[static_cast<size_t>(g)];
      const int end = g + 1;
      const int dist = std::abs(end - begin - c.targetUnits);
      if (bestGap < 0 || f < bestFlow - 1e-12 ||
          (std::abs(f - bestFlow) <= 1e-12 && dist < bestDist)) {
        bestGap = g;
        bestFlow = f;
        bestDist = dist;
      }
    }
    if (bestGap < 0) {
      /* Every candidate cut leaves a too-short tail: absorb it. */
      emit(begin, n, 0.0);
      break;
    }
    const int end = bestGap + 1;
    emit(begin, end, bestFlow);
    begin = end;
  }
  return out;
}

} // namespace secamp
} // namespace phoenix
