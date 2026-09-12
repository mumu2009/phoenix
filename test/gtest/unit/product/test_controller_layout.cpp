#include <gtest/gtest.h>

#include "controller_layout.hpp"

TEST(ControllerLayout, AiCountOneIsNotInflatedToThreeOrSeven) {
  auto layout = phoenix::planControllerLayout(1, 3, 7);
  EXPECT_EQ(layout.totalControllers, 1);
  ASSERT_EQ(layout.groupCount, 1);
  EXPECT_EQ(layout.groupSizes[0], 1);
  EXPECT_EQ(phoenix::clampPositiveCount(1), 1);
}

TEST(ControllerLayout, ProductJsonSevenByThreeIsSevenNotTwentyOne) {
  auto layout = phoenix::planControllerLayout(7, 3, 7);
  EXPECT_EQ(layout.totalControllers, 7);
  EXPECT_EQ(layout.groupCount, 1);
  EXPECT_EQ(layout.groupSizes[0], 7);
}

TEST(ControllerLayout, ExplicitTwentyOneKeepsThreeGroups) {
  auto layout = phoenix::planControllerLayout(21, 3, 7);
  EXPECT_EQ(layout.totalControllers, 21);
  EXPECT_EQ(layout.groupCount, 3);
}

TEST(BoundedHitMap, CapsAndDoesNotGrowUnbounded) {
  phoenix::BoundedHitMap hits(8);
  for (int i = 0; i < 64; ++i)
    hits.increment("k" + std::to_string(i));
  EXPECT_LE(hits.size(), 8u);
}

TEST(ProjectionBound, LlamaEmbDim4096DoesNotAllocateTargetOverThreeTriplets) {
  const size_t nz = phoenix::boundedProjectionNonZeros(4096, 4096, 0);
  EXPECT_LE(nz * 4096u, phoenix::kMaxProjectionTriplets);
  EXPECT_LT(nz, 4096u / 3u);
}
