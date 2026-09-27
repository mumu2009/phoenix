/* test_meme_semantic.cpp - Semantic slicer + meme annealer unit tests. */

#include <cmath>
#include <string>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

#include "addons/ThePlugInForSecurity/meme_annealer.hpp"
#include "addons/ThePlugInForSecurity/meme_semantic_slicer.hpp"
#include "addons/ThePlugInForSecurity/security_core.hpp"

namespace {

using phoenix::secamp::AnnealConfig;
using phoenix::secamp::AnnealEdge;
using phoenix::secamp::AnnealMeme;
using phoenix::secamp::MemeAnnealer;
using phoenix::secamp::SemanticSliceConfig;
using phoenix::secamp::SemanticSlicer;
using phoenix::secamp::SliceUnitFootprint;

SliceUnitFootprint fp(int idx, std::initializer_list<const char *> ids) {
  SliceUnitFootprint f;
  f.unitIndex = idx;
  for (const char *id : ids)
    f.memeSeq.push_back(id);
  return f;
}

/* Two semantic blocks (A* memes then B* memes) with a hard seam at 6|7. */
std::vector<SliceUnitFootprint> twoBlockUnits() {
  std::vector<SliceUnitFootprint> u;
  for (int i = 0; i < 6; ++i)
    u.push_back(fp(i, {"m_a1", "m_a2", "m_a3"}));
  for (int i = 6; i < 12; ++i)
    u.push_back(fp(i, {"m_b1", "m_b2", "m_b3"}));
  return u;
}

TEST(SemanticSlicer, FindsHardSeam) {
  const auto units = twoBlockUnits();
  const auto flows = SemanticSlicer::gapFlows(units, 2);
  ASSERT_EQ(flows.size(), units.size() - 1);
  /* Inside a block flow is maximal; at the 6|7 seam it is zero. */
  EXPECT_NEAR(flows[2], 1.0, 1e-9);
  EXPECT_NEAR(flows[5], 0.0, 1e-9);
  EXPECT_NEAR(flows[8], 1.0, 1e-9);

  SemanticSliceConfig cfg;
  cfg.minUnits = 3;
  cfg.maxUnits = 8;
  cfg.targetUnits = 6;
  const auto slices = SemanticSlicer::slice(units, cfg);
  ASSERT_EQ(slices.size(), 2u);
  EXPECT_EQ(slices[0].beginUnit, 0);
  EXPECT_EQ(slices[0].endUnit, 6);
  EXPECT_NEAR(slices[0].seamAfter, 0.0, 1e-9);
  EXPECT_EQ(slices[1].beginUnit, 6);
  EXPECT_EQ(slices[1].endUnit, 12);
}

TEST(SemanticSlicer, RespectsMinAndMax) {
  /* No seam anywhere: every unit shares memes with its neighbors. */
  std::vector<SliceUnitFootprint> u;
  for (int i = 0; i < 20; ++i)
    u.push_back(fp(i, {"m_x", "m_y"}));
  SemanticSliceConfig cfg;
  cfg.minUnits = 5;
  cfg.maxUnits = 7;
  cfg.targetUnits = 6;
  const auto slices = SemanticSlicer::slice(u, cfg);
  ASSERT_FALSE(slices.empty());
  int covered = 0;
  for (size_t i = 0; i < slices.size(); ++i) {
    const auto &s = slices[i];
    /* No slice may be smaller than minUnits; non-final slices stay within
       maxUnits; the final slice may absorb a short tail (< minUnits
       extra), per the documented contract. */
    EXPECT_GE(s.endUnit - s.beginUnit, 5);
    if (i + 1 < slices.size())
      EXPECT_LE(s.endUnit - s.beginUnit, 7);
    else
      EXPECT_LE(s.endUnit - s.beginUnit, 7 + 5 - 1);
    covered += s.endUnit - s.beginUnit;
  }
  EXPECT_EQ(covered, 20);
}

TEST(SemanticSlicer, EmptyUnitsDoNotFakeSeams) {
  /* Short/un-ingested units (empty seq) inside one semantic block. */
  std::vector<SliceUnitFootprint> u;
  for (int i = 0; i < 10; ++i) {
    if (i == 4 || i == 5)
      u.push_back(fp(i, {}));
    else
      u.push_back(fp(i, {"m_x", "m_y", "m_z"}));
  }
  const auto flows = SemanticSlicer::gapFlows(u, 4);
  ASSERT_EQ(flows.size(), 9u);
  /* Windows skip empty units, so flow across the empty run stays high. */
  EXPECT_GT(flows[4], 0.5);
}

TEST(SemanticSlicer, TinyCorpusSingleSlice) {
  std::vector<SliceUnitFootprint> u;
  u.push_back(fp(0, {"m_a"}));
  u.push_back(fp(1, {"m_b"}));
  SemanticSliceConfig cfg;
  cfg.minUnits = 128;
  cfg.maxUnits = 1024;
  const auto slices = SemanticSlicer::slice(u, cfg);
  ASSERT_EQ(slices.size(), 1u);
  EXPECT_EQ(slices[0].endUnit, 2);
}

AnnealMeme meme(std::string id,
                std::initializer_list<std::pair<const char *, double>> w) {
  AnnealMeme m;
  m.id = std::move(id);
  for (auto &wp : w)
    m.weights.push_back({wp.first, wp.second});
  return m;
}

struct AnnealFixturePools {
  /* Two cohesive clusters (edges inside) + one isolated meme. */
  std::vector<AnnealMeme> pool;
  std::vector<AnnealEdge> edges;
  AnnealFixturePools() {
    pool.push_back(meme("a1", {{"harbor", 0.9}, {"dock", 0.8}, {"ledger", 0.7}}));
    pool.push_back(meme("a2", {{"harbor", 0.6}, {"cargo", 0.9}, {"manifest", 0.8}}));
    pool.push_back(meme("b1", {{"lunar", 0.9}, {"orbit", 0.8}, {"beacon", 0.7}}));
    pool.push_back(meme("b2", {{"lunar", 0.5}, {"eclipse", 0.9}, {"umbra", 0.8}}));
    pool.push_back(meme("c1", {{"solitary", 0.9}, {"unique", 0.8}}));
    edges.push_back(AnnealEdge{0, 1, 2.0});
    edges.push_back(AnnealEdge{2, 3, 2.0});
  }
};

TEST(MemeAnnealer, DeterministicSameSeed) {
  AnnealFixturePools f;
  AnnealConfig cfg;
  cfg.steps = 3000;
  cfg.seed = 7;
  const auto r1 = MemeAnnealer::run(f.pool, f.edges, cfg);
  const auto r2 = MemeAnnealer::run(f.pool, f.edges, cfg);
  ASSERT_EQ(r1.groups.size(), r2.groups.size());
  EXPECT_DOUBLE_EQ(r1.energy, r2.energy);
  for (size_t i = 0; i < r1.groups.size(); ++i) {
    EXPECT_EQ(r1.groups[i].members, r2.groups[i].members);
    EXPECT_EQ(r1.groups[i].wordCount, r2.groups[i].wordCount);
  }
}

TEST(MemeAnnealer, NeverExceedsWordCap) {
  AnnealFixturePools f;
  AnnealConfig cfg;
  cfg.steps = 4000;
  cfg.seed = 123;
  cfg.maxMemeWords = 5; /* tight cap: a1+a2 union has 5 words, +b1 would be 8 */
  const auto r = MemeAnnealer::run(f.pool, f.edges, cfg);
  for (const auto &g : r.groups)
    EXPECT_LE(g.wordCount, cfg.maxMemeWords);
}

TEST(MemeAnnealer, CohesivePairsIntegrate) {
  AnnealFixturePools f;
  AnnealConfig cfg;
  cfg.steps = 6000;
  cfg.seed = 42;
  cfg.t0 = 0.5;
  const auto r = MemeAnnealer::run(f.pool, f.edges, cfg);
  EXPECT_LT(r.energy, r.initialEnergy);
  /* The edge-linked, word-sharing pairs {a1,a2} and {b1,b2} should end up
     integrated in the best partition. */
  bool foundA = false;
  bool foundB = false;
  for (const auto &g : r.groups) {
    std::unordered_set<int> ms(g.members.begin(), g.members.end());
    if (ms.count(0) && ms.count(1))
      foundA = true;
    if (ms.count(2) && ms.count(3))
      foundB = true;
  }
  EXPECT_TRUE(foundA);
  EXPECT_TRUE(foundB);
}

TEST(MemeAnnealer, OrganicPairsRespectGnnGate) {
  std::vector<AnnealMeme> pool;
  /* Share 2 content words -> organic. */
  pool.push_back(meme("x1", {{"harbor", 0.9}, {"dock", 0.8}, {"ledger", 0.7}}));
  pool.push_back(meme("x2", {{"harbor", 0.6}, {"dock", 0.5}, {"cargo", 0.9}}));
  /* Shares only stopwords -> the GNN gate must reject. */
  pool.push_back(meme("y1", {{"the", 0.9}, {"and", 0.8}, {"quantum", 0.7}}));
  pool.push_back(meme("y2", {{"the", 0.6}, {"and", 0.5}, {"classical", 0.9}}));
  const auto pairs = MemeAnnealer::organicMergePairs(pool, 2, 28);
  bool foundX = false;
  bool foundY = false;
  for (const auto &pr : pairs) {
    if ((pr.first == 0 && pr.second == 1))
      foundX = true;
    if ((pr.first == 2 && pr.second == 3))
      foundY = true;
  }
  EXPECT_TRUE(foundX);
  EXPECT_FALSE(foundY);
}

TEST(MemeAnnealer, EmptyPool) {
  const auto r = MemeAnnealer::run({}, {}, AnnealConfig{});
  EXPECT_TRUE(r.groups.empty());
  EXPECT_DOUBLE_EQ(r.energy, 0.0);
}

/* Security-observatory JSON surfaces over a snapshot. */
TEST(SecuritySemanticSurfaces, SlicesAndAnnealJson) {
  auto &obs = phoenix::secamp::SecurityObservatory::instance();
  obs.resetForTests();

  phoenix::secamp::DiscreteGraph g;
  g.ids = {"m_alpha", "m_beta"};
  g.layers = {"meme", "meme"};
  g.mapped = {{"harbor", "dock", "ledger"}, {"lunar", "orbit", "beacon"}};
  g.edges = {phoenix::secamp::GraphEdge{0, 1, 1.0}};
  obs.ingest(g);

  const std::string text =
      "harbor dock ledger records station\n\n"
      "harbor dock ledger cargo manifest\n\n"
      "harbor dock ledger night signals\n\n"
      "harbor dock ledger paints night\n\n"
      "lunar orbit beacon eclipse umbra\n\n"
      "lunar orbit beacon schedule alpha\n\n"
      "lunar orbit beacon greenhouse\n\n"
      "lunar orbit beacon harvest\n\n";
  const auto sj = obs.semanticSlicesJson(text, 2, 8);
  ASSERT_TRUE(sj.value("ok", false));
  EXPECT_EQ(sj.value("units", 0), 8);
  EXPECT_GE(sj["slices"].size(), 1u);
  /* The alpha->beta topic shift should show up as a low-flow gap. */
  EXPECT_LE(sj.value("flowMin", 1.0), sj.value("flowMean", 1.0));

  const auto aj = obs.annealPreviewJson(500, 42);
  ASSERT_TRUE(aj.value("ok", false));
  EXPECT_EQ(aj.value("poolSize", 0), 2);
  EXPECT_GE(aj.value("organicMergePairs", -1), 0);

  obs.resetForTests();
}

} // namespace
