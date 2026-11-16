/* v10.2 exploration width for the SELF-MOD explore step.

   Pinned behaviour of phoenix::memetic::pickExplorePath:
     - one matching candidate      -> that candidate (unchanged legacy path);
     - several candidates:
         T <= 0                    -> deterministic argmax (an explicit choice);
         T >  0                    -> sampled from softmax(score/T), so the
                                      plausible files all get mass and the
                                      mission stops re-editing the one file it
                                      happened to name first;
     - no candidate                -> empty string (the caller then lists files
                                      instead of naming one).
   `rankExplorePaths(keep=1)` keeps its ambiguity guard (it returns nothing
   when several match); this function is the width-preserving fallback.

   Determinism: the sampler is seeded from `seedShift`, so a fixed seed gives a
   reproducible choice.  With the same seed and a different seedShift the
   choice may differ, which is what the temperature test relies on. */
#include <gtest/gtest.h>

#include <set>
#include <string>
#include <vector>

#include "memetic_existence.hpp"

using phoenix::memetic::pickExplorePath;
using phoenix::memetic::rankExplorePaths;

TEST(ExploreWidth, Sample1_SingleCandidateIsReturnedUnchanged) {
  const std::vector<std::string> paths{"emotion_system.cpp", "instinct.cpp"};
  /* "instinct" only matches instinct.cpp */
  const auto picked = pickExplorePath("instinct cpp", paths, 0.9, 11u);
  EXPECT_EQ(picked, "instinct.cpp");
}

TEST(ExploreWidth, Sample2_NoCandidateReturnsEmpty) {
  const std::vector<std::string> paths{"emotion_system.cpp", "instinct.cpp"};
  EXPECT_TRUE(pickExplorePath("", paths, 0.9, 1u).empty());
  EXPECT_TRUE(pickExplorePath("zzz-nothing-matches", paths, 0.9, 1u).empty());
}

TEST(ExploreWidth, Sample3_ZeroTemperatureIsDeterministicArgmax) {
  const std::vector<std::string> paths{"autonomy_stack.cpp",
                                      "autonomy_memory.cpp"};
  const auto a = pickExplorePath("autonomy stack memory cpp", paths, 0.0, 5u);
  const auto b = pickExplorePath("autonomy stack memory cpp", paths, 0.0, 5u);
  EXPECT_EQ(a, b);
  EXPECT_FALSE(a.empty());
}

TEST(ExploreWidth, Sample4_AmbiguityGuardStillReturnsNothingForKeep1) {
  const std::vector<std::string> paths{"autonomy_stack.cpp",
                                      "autonomy_memory.cpp"};
  /* Both match "autonomy": rankExplorePaths(keep=1) refuses to name one.
     That refusal is the reason the width-preserving fallback exists. */
  EXPECT_TRUE(rankExplorePaths("autonomy stack memory cpp", paths, 1).empty());
  /* ... and the fallback does name one. */
  EXPECT_FALSE(pickExplorePath("autonomy stack memory cpp", paths, 0.8, 3u).empty());
}

TEST(ExploreWidth, Sample5_HighTemperatureReachesBothCandidates) {
  const std::vector<std::string> paths{"alpha_one.cpp", "alpha_two.cpp"};
  std::set<std::string> seen;
  /* A wide temperature must not collapse onto one row: over many seeds both
     candidates have to appear.  This is the property that keeps the search
     from locking into a local optimum. */
  for (unsigned seed = 1; seed <= 64; ++seed)
    seen.insert(pickExplorePath("alpha one two cpp", paths, 50.0, seed));
  EXPECT_EQ(seen.size(), 2u) << "only " << *seen.begin()
                             << " was ever selected at high temperature";
}

TEST(ExploreWidth, Sample6_EmptyPathListIsSafe) {
  const std::vector<std::string> empty;
  EXPECT_TRUE(pickExplorePath("anything", empty, 0.8, 1u).empty());
}
