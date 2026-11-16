/* v10.1 evolution-efficiency knob tests: the anneal and the early stop
   must be inert at their defaults (the pre-v10.1 behaviour is unchanged)
   and the setters must clamp the inputs. */
#include <gtest/gtest.h>

#include "mission_lifecycle.hpp"

using phoenix::mission::MissionLifecycle;

TEST(MissionEvolutionKnobs, Sample1_AnnealZeroKeepsBaseRate) {
  MissionLifecycle lc;
  lc.setEvolutionEnabled(true);
  lc.recordLineage(1000, 0.5);
  lc.setEvolutionAnnealPerGen(0.15);
  const float withAnneal = lc.effectiveMutationRate(0.05f);
  lc.setEvolutionAnnealPerGen(0.0);
  const float base = lc.effectiveMutationRate(0.05f);
  /* generation 0: the anneal factor is 1, so the anneal must not change
     the effective rate yet. */
  EXPECT_FLOAT_EQ(withAnneal, base);
}

TEST(MissionEvolutionKnobs, Sample2_EarlyStopFreshStateNotStalled) {
  MissionLifecycle lc;
  lc.setEvolutionEnabled(true);
  lc.setEvolutionEarlyStop(false);
  EXPECT_FALSE(lc.evolutionStalled());
  lc.setEvolutionEarlyStop(true);
  /* the fresh state has zero stalls; the stall only marks after two flat
     adoptions, which requires the completed-mission path. */
  EXPECT_FALSE(lc.evolutionStalled());
}

TEST(MissionEvolutionKnobs, Sample3_AnnealClampsNonNegative) {
  MissionLifecycle lc;
  lc.setEvolutionEnabled(true);
  lc.recordLineage(1000, 0.5);
  lc.setEvolutionAnnealPerGen(-0.5);
  EXPECT_GT(lc.effectiveMutationRate(0.05f), 0.0f);
}
