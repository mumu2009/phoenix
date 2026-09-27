/* meme_semantic_slicer.hpp - Semantic dynamic slicing on GNN meme footprints

   The fixed 512-unit corpus slice cuts through semantic boundaries. This
   slicer instead reads the GNN's own output: each corpus unit resolves to
   an ordered meme sequence via the official clustering
   (phoenix::memetic::buildMemeSequence / ingestDocumentLike). A meme shared
   by units on both sides of a gap is semantic glue grown by the graph
   itself; a gap with no shared memes is a semantic seam. Slices are cut at
   seams, so slice size floats between minUnits and maxUnits instead of
   being fixed.

   Pure component: no GNN mutation, no locks, no I/O. The caller owns
   ingest and locking. Used by the screening pipeline tool and by the
   SecurityObservatory inspect path (footprints from a DiscreteGraph
   snapshot, never from the live graph mid-mutation). */

#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace phoenix {
namespace secamp {

/* One corpus unit's meme footprint: the ordered meme ids the unit resolved
   to on the official path. Units with too few tokens to ingest keep an
   empty memeSeq so indices stay aligned with splitCorpusUnits output. */
struct SliceUnitFootprint {
  int unitIndex{0};
  std::vector<std::string> memeSeq;
};

struct SemanticSliceConfig {
  int minUnits{128};    /* never cut before this many units accumulated */
  int maxUnits{1024};   /* hard cut at this many units */
  int targetUnits{512}; /* seam search prefers the cut closest to this */
  int lookahead{8};     /* non-empty units each side of a gap for flow */
};

struct SemanticSlice {
  int beginUnit{0};    /* inclusive corpus-unit index */
  int endUnit{0};      /* exclusive corpus-unit index */
  double seamAfter{0}; /* normalized flow across the closing gap
                          (0 when closed by corpus end / hard cap) */
  int memeRefs{0};     /* total meme references inside the slice */
  int uniqueMemes{0};  /* distinct memes inside the slice */
};

class SemanticSlicer {
public:
  /* Normalized semantic flow across the gap AFTER unit i, for
     i in [0, n-2). flow = sum over shared memes of min(cL, cR) divided by
     sqrt(refsL * refsR), windows spanning up to `lookahead` non-empty
     units each side. 0 = semantic seam, 1 = same meme mass both sides. */
  static std::vector<double>
  gapFlows(const std::vector<SliceUnitFootprint> &units, int lookahead);

  /* Greedy seam slicing: extend the current slice until size >= minUnits,
     then cut at the lowest-flow gap before maxUnits (ties prefer the cut
     closest to targetUnits). Cuts that would leave a tail smaller than
     minUnits are forbidden; if every candidate does, the tail is absorbed,
     so the final slice may exceed maxUnits by less than minUnits. */
  static std::vector<SemanticSlice>
  slice(const std::vector<SliceUnitFootprint> &units,
        const SemanticSliceConfig &cfg);
};

} // namespace secamp
} // namespace phoenix
