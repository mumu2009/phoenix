/* v10.2 retrieval width (MMR).

   Pinned contract of phoenix::retrieval::mmrSelect:
     - lambda = 1 is exactly the legacy greedy order (no silent change);
     - with a lower lambda, near-duplicate candidates stop filling every slot,
       so the mission sees different past recipes instead of one repeated;
     - k = 0 / empty input / k > n are all safe;
     - the result is deterministic (ties broken by lower index);
     - bad similarity values (NaN / out of range) are ignored, not propagated. */
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "retrieval_diversity.hpp"

using phoenix::retrieval::mmrSelect;

TEST(RetrievalDiversity, Sample1_LambdaOneIsLegacyGreedyOrder) {
  const std::vector<double> rel{0.9, 0.8, 0.7, 0.1};
  auto sim = [](size_t, size_t) { return 1.0; }; /* maximal similarity */
  const auto picked = mmrSelect(
      rel.size(), [&rel](size_t i) { return rel[i]; }, sim, 3, 1.0);
  ASSERT_EQ(picked.size(), 3u);
  EXPECT_EQ(picked[0], 0);
  EXPECT_EQ(picked[1], 1);
  EXPECT_EQ(picked[2], 2);
}

TEST(RetrievalDiversity, Sample2_DuplicatesDoNotFillEverySlot) {
  /* 0 and 1 are near duplicates (rel 0.9, 0.88); 2 is distinct and slightly
     less relevant (0.7).  Greedy picks 0,1; MMR must take 2 as well. */
  const std::vector<double> rel{0.9, 0.88, 0.7};
  auto sim = [](size_t a, size_t b) {
    if ((a == 0 && b == 1) || (a == 1 && b == 0)) return 0.98;
    return 0.05;
  };
  const auto legacy = mmrSelect(
      rel.size(), [&rel](size_t i) { return rel[i]; }, sim, 2, 1.0);
  EXPECT_EQ(legacy[0], 0);
  EXPECT_EQ(legacy[1], 1);

  const auto diverse = mmrSelect(
      rel.size(), [&rel](size_t i) { return rel[i]; }, sim, 2, 0.7);
  ASSERT_EQ(diverse.size(), 2u);
  EXPECT_EQ(diverse[0], 0);
  EXPECT_EQ(diverse[1], 2) << "the diverse candidate must win the second slot";
}

TEST(RetrievalDiversity, Sample3_EdgeCasesAreSafe) {
  const std::vector<double> rel{1.0, 0.5};
  auto sim = [](size_t, size_t) { return 0.0; };
  EXPECT_TRUE(mmrSelect(0, [](size_t) { return 0.0; }, sim, 3, 0.7).empty());
  EXPECT_TRUE(mmrSelect(rel.size(), [&rel](size_t i) { return rel[i]; }, sim,
                        0, 0.7)
                  .empty());
  const auto all = mmrSelect(rel.size(), [&rel](size_t i) { return rel[i]; },
                             sim, 99, 0.7);
  EXPECT_EQ(all.size(), rel.size());
}

TEST(RetrievalDiversity, Sample4_DeterministicForEqualScores) {
  const std::vector<double> rel{0.5, 0.5, 0.5};
  auto sim = [](size_t, size_t) { return 0.0; };
  const auto a = mmrSelect(rel.size(), [&rel](size_t i) { return rel[i]; },
                           sim, 3, 0.7);
  const auto b = mmrSelect(rel.size(), [&rel](size_t i) { return rel[i]; },
                           sim, 3, 0.7);
  EXPECT_EQ(a, b);
  EXPECT_EQ(a[0], 0);
}

TEST(RetrievalDiversity, Sample5_NaNSimilarityIsIgnored) {
  const std::vector<double> rel{0.9, 0.8};
  auto sim = [](size_t, size_t) {
    return std::numeric_limits<double>::quiet_NaN();
  };
  const auto picked = mmrSelect(
      rel.size(), [&rel](size_t i) { return rel[i]; }, sim, 2, 0.5);
  ASSERT_EQ(picked.size(), 2u);
  EXPECT_EQ(picked[0], 0);
  EXPECT_EQ(picked[1], 1);
}
