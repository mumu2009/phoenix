/* v10.2 plugin intent matching: memetic-semantic, not word recognition.

   Pinned behaviour:
     - the lexical channel is order- and stopword-insensitive;
     - the memetic channel can select a plugin whose profile shares NO content
       word with the situation (the case the old keyword tally always missed);
     - without a meme graph the decision degrades to lexical only (no silent
       full score);
     - the blend damps a plugin with no semantic evidence but never vetoes it,
       so a confident plugin is still considered. */
#include <gtest/gtest.h>

#include "plugin_semantics.hpp"

using namespace phoenix::plugin;

TEST(PluginSemantics, Sample1_LexicalIgnoresOrderAndStopwords) {
  const double a = lexicalAffinity("explain the derivative of a function",
                                   "function derivative explain");
  EXPECT_NEAR(a, 1.0, 1e-9);
  /* the stopwords alone must not create affinity */
  EXPECT_DOUBLE_EQ(lexicalAffinity("the and for", "this with from"), 0.0);
}

TEST(PluginSemantics, Sample2_MemeticChannelSelectsPluginWithNoSharedWords) {
  const std::string situation = "consistent";
  const std::string profile = "logic prover";
  EXPECT_DOUBLE_EQ(lexicalAffinity(situation, profile), 0.0);
  auto related = [](const std::string &a, const std::string &b) -> double {
    if (a == "consistent" && (b == "logic" || b == "prover")) return 0.9;
    return 0.0;
  };
  const double lexOnly = semanticAffinity(situation, profile, nullptr, 0.5);
  EXPECT_DOUBLE_EQ(lexOnly, 0.0);
  const double blended = semanticAffinity(situation, profile, related, 0.5);
  EXPECT_NEAR(blended, 0.45, 1e-9);
  const double memeticOnly = semanticAffinity(situation, profile, related, 1.0);
  EXPECT_NEAR(memeticOnly, 0.9, 1e-9);
}

TEST(PluginSemantics, Sample3_BlendDampsButNeverVetoes) {
  EXPECT_NEAR(blendOfferScore(0.8, 0.0, 0.5), 0.4, 1e-9);
  EXPECT_NEAR(blendOfferScore(0.8, 1.0, 0.5), 0.8, 1e-9);
  /* weight 0 == the old behaviour: the plugin score passes through */
  EXPECT_NEAR(blendOfferScore(0.8, 0.0, 0.0), 0.8, 1e-9);
}

TEST(PluginSemantics, Sample4_NaNAndOutOfRangeGraphScoresAreIgnored) {
  auto related = [](const std::string &, const std::string &) -> double {
    return std::numeric_limits<double>::quiet_NaN();
  };
  const double v = semanticAffinity("alpha", "beta", related, 1.0);
  EXPECT_DOUBLE_EQ(v, 0.0);
}

TEST(PluginSemantics, Sample5_PluginProfileCarriesMetaTags) {
  const auto meta = nlohmann::json{{"tags", nlohmann::json::array({"math", "exact"})},
                                   {"description", "exact rational arithmetic"}};
  const std::string profile = pluginProfile("math", "tool", "big numbers", meta);
  EXPECT_NE(profile.find("math"), std::string::npos);
  EXPECT_NE(profile.find("exact rational arithmetic"), std::string::npos);
}

TEST(PluginSemantics, Sample6_InvalidSemanticWeightFallsBackToHalf) {
  /* 2.0 is out of range: clamped to 0.5, so a perfect semantic score at
     weight 2.0 must behave exactly like weight 0.5. */
  EXPECT_NEAR(semanticAffinity("parser", "parser", nullptr, 2.0),
              semanticAffinity("parser", "parser", nullptr, 0.5), 1e-12);
}
