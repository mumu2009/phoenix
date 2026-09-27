/* meme_annealer.hpp - Simulated-annealing meme integration (multi-meme
   combination, method 2) plus the GNN organic-merge gate listing (method 1).

   Method 1 (organic): the GNN already integrates memes during ingest
   (resolveOrCreate + mappingMergeAllowed). organicMergePairs lists the
   existing meme pairs whose word sets would pass that same gate if seen
   as one incoming window - i.e. the merges the GNN itself sanctions.

   Method 2 (annealing): partition a meme pool into groups maximizing
   replication fitness + graph cohesion, under a hard word cap so no
   giant meme can appear (the screen already drops bags larger than one
   ngramMax window; we default the cap to 2*ngramMax and penalize from
   ngramMax up). Not "dual-meme integration": moves merge/split/move any
   number of groups, so the annealer itself decides how many memes
   integrate.

   Energy per group g:
     E(g) = -[ fitness(g) + cohesionWeight * cohesion(g) ]
            + sizePenalty * max(0, words(g) - softWords)^2
   fitness  = contentSpec(merged (word,alpha)) * typical-set sweet spot
              (mappingContentSpec / mappingTypicalAlpha from the official
              mapping header; df measured on the pool)
   cohesion = internal consecutive-edge density + shared-content-word
              fraction (edges are the GNN manifold's own linkConsecutive
              weights)

   Pure component: deterministic under a fixed seed (mt19937_64), no I/O,
   no GNN mutation. The caller owns the store and its locks; build the
   pool from a snapshot (DiscreteGraph / frozen snap), never while the
   live graph mutates. */

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace phoenix {
namespace secamp {

struct AnnealMeme {
  std::string id;
  /* Ordered (word, alpha) set, alpha = tf/df from the source graph. */
  std::vector<std::pair<std::string, double>> weights;
  int degree{0};
};

struct AnnealEdge {
  int a{-1}; /* pool indices */
  int b{-1};
  double w{1.0};
};

struct AnnealConfig {
  int maxMemeWords{28}; /* hard cap; merges above this are rejected */
  int softWords{14};    /* ngramMax; quadratic penalty beyond */
  double sizePenalty{0.05};
  double cohesionWeight{0.5};
  double t0{1.0};
  double cooling{0.999};
  int steps{20000};
  std::uint64_t seed{42};
};

struct AnnealGroupResult {
  std::vector<int> members; /* pool indices */
  /* Merged ordered (word, alpha): alpha = max over members, sorted desc. */
  std::vector<std::pair<std::string, double>> merged;
  std::vector<std::string> typical; /* mappingTypicalAlpha keys */
  double fitness{0.0};
  double cohesion{0.0};
  int wordCount{0};
};

struct AnnealResult {
  std::vector<AnnealGroupResult> groups; /* ranked: best first */
  double energy{0.0};
  double initialEnergy{0.0};
  int stepsDone{0};
};

class MemeAnnealer {
public:
  static AnnealResult run(const std::vector<AnnealMeme> &pool,
                          const std::vector<AnnealEdge> &edges,
                          const AnnealConfig &cfg);

  /* Method 1: pairs (i, j) whose union passes the official ingest gate
     mappingMergeAllowed in at least one direction and stays within
     maxMemeWords. Built from a word->memes inverted index, so only pairs
     sharing >= minOverlap words are gate-checked. */
  static std::vector<std::pair<int, int>>
  organicMergePairs(const std::vector<AnnealMeme> &pool, int minOverlap,
                    int maxMemeWords);
};

} // namespace secamp
} // namespace phoenix
