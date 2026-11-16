/* v10.2 population-layer domain coupling.

   `study.population` used to compete on nothing: runGoal() needs {input,
   target} and no caller produced one, so goalsRun stayed 0 forever. These
   tests pin the feature/target contract that feeds it real mission data:

     - features are normalised into [0,1] and have a fixed length (8);
     - richer/more repetitive text is reflected in the right slots;
     - the target rewards NEW information, not mere length (a draft that grew
       only by repeating itself scores 0);
     - the payload is rejected-by-shape only when it cannot be built. */
#include <gtest/gtest.h>

#include "study_population_features.hpp"

using namespace phoenix::study;

TEST(PopulationFeatures, Sample1_FeatureVectorIsFixedLengthAndNormalised) {
  const auto f = fieldStateFeatures(nlohmann::json{{"deliverable", "hello world"}});
  ASSERT_EQ(static_cast<int>(f.size()), kFieldFeatureCount);
  for (float v : f) {
    EXPECT_GE(v, 0.0f);
    EXPECT_LE(v, 1.0f);
  }
  /* empty/non-object input is safe */
  const auto z = fieldStateFeatures(nlohmann::json::object());
  EXPECT_EQ(static_cast<int>(z.size()), kFieldFeatureCount);
}

TEST(PopulationFeatures, Sample2_LongerDeliverableRaisesTheSizeFeature) {
  nlohmann::json a{{"deliverable", "short text here"}};
  nlohmann::json b{{"deliverable", std::string(2000, 'x')}};
  EXPECT_GT(fieldStateFeatures(b)[0], fieldStateFeatures(a)[0]);
}

TEST(PopulationFeatures, Sample3_RepetitionIsDetected) {
  /* three distinct sentences -> low repetition */
  const std::string varied =
      "Alpha measures one thing. Beta measures another. Gamma is different.";
  /* the same sentence three times -> high repetition */
  const std::string repeated =
      "Same sentence again. Same sentence again. Same sentence again.";
  EXPECT_LT(repetitionRatio(varied), repetitionRatio(repeated));
  EXPECT_GT(repetitionRatio(repeated), 0.0);
}

TEST(PopulationFeatures, Sample4_GainRewardsNewInformation) {
  nlohmann::json prev{{"deliverable", "Intro. Alpha one. Beta two."}};
  nlohmann::json grew{{"deliverable",
                       "Intro. Alpha one. Beta two. Gamma three. Delta four."}};
  const auto target = fieldGainTarget(prev, grew);
  ASSERT_EQ(target.size(), 1u);
  EXPECT_GT(target[0], 0.0f);
  /* no growth -> no gain */
  const auto flat = fieldGainTarget(prev, prev);
  ASSERT_EQ(flat.size(), 1u);
  EXPECT_FLOAT_EQ(flat[0], 0.0f);
}

TEST(PopulationFeatures, Sample5_GrowthByPureRepetitionScoresZero) {
  nlohmann::json prev{{"deliverable", "One fact. Two facts."}};
  /* longer, but every added sentence repeats an existing one */
  nlohmann::json bloated{{"deliverable",
                          "One fact. Two facts. One fact. Two facts. One fact."}};
  const auto target = fieldGainTarget(prev, bloated);
  ASSERT_EQ(target.size(), 1u);
  EXPECT_FLOAT_EQ(target[0], 0.0f)
      << "repetition must not be rewarded as information gain";
}

TEST(PopulationFeatures, Sample6_GoalPayloadHasBothShapes) {
  const auto goal = populationGoal(nlohmann::json{{"deliverable", "a"}},
                                   nlohmann::json{{"deliverable", "a b"}});
  ASSERT_TRUE(goal.is_object());
  ASSERT_TRUE(goal.contains("input"));
  ASSERT_TRUE(goal.contains("target"));
  EXPECT_EQ(static_cast<int>(goal["input"].size()), kFieldFeatureCount);
  EXPECT_EQ(static_cast<int>(goal["target"].size()), 1);
}
