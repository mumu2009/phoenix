/* GTest unit tests for Phoenix v7.0 primal sensation / instinct layer. */

#include "instinct.hpp"
#include "primal_sensation.hpp"
#include "test_hacktest_framework.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace phoenix::primal;
using namespace phoenix::instinct;
using namespace phoenix::testing;

class PrimalSensationTest : public HacktestBase {};

TEST_F(PrimalSensationTest, TypeStringConversions) {
    EXPECT_EQ(PrimalSensation::typeToString(SensationType::Pain), "pain");
    EXPECT_EQ(PrimalSensation::typeToString(SensationType::Pleasure), "pleasure");
    EXPECT_EQ(PrimalSensation::typeToString(SensationType::Hunger), "hunger");
    EXPECT_EQ(PrimalSensation::typeToString(SensationType::Temperature), "temperature");
    EXPECT_EQ(PrimalSensation::typeToString(SensationType::Fatigue), "fatigue");
    EXPECT_EQ(PrimalSensation::typeToString(SensationType::Threat), "threat");
    EXPECT_EQ(PrimalSensation::typeToString(SensationType::SocialIsolation), "social_isolation");
    EXPECT_EQ(PrimalSensation::typeToString(SensationType::Novelty), "novelty");
    EXPECT_EQ(PrimalSensation::typeToString(SensationType::Unknown), "unknown");

    EXPECT_EQ(PrimalSensation::stringToType("pain"), SensationType::Pain);
    EXPECT_EQ(PrimalSensation::stringToType("threat"), SensationType::Threat);
    EXPECT_EQ(PrimalSensation::stringToType("not_a_type"), SensationType::Unknown);
}

TEST_F(PrimalSensationTest, JsonRoundTrip) {
    PrimalSensation s;
    s.type = SensationType::Pain;
    s.intensity = 0.8f;
    s.valence = -0.9f;
    s.source = "thermal sensor";

    auto j = s.toJson();
    auto loaded = PrimalSensation::fromJson(j);

    EXPECT_EQ(loaded.type, s.type);
    EXPECT_FLOAT_EQ(loaded.intensity, s.intensity);
    EXPECT_FLOAT_EQ(loaded.valence, s.valence);
    EXPECT_EQ(loaded.source, s.source);
}

TEST_F(PrimalSensationTest, InvalidJsonClamps) {
    nlohmann::json j = {
        {"type", "pain"},
        {"intensity", 1.5f},
        {"valence", -2.0f}
    };
    auto s = PrimalSensation::fromJson(j);
    EXPECT_FLOAT_EQ(s.intensity, 1.0f);
    EXPECT_FLOAT_EQ(s.valence, -1.0f);
}

TEST_F(PrimalSensationTest, SameSourceRefreshesInsteadOfStacking) {
    PrimalSensationEngine engine;
    PrimalSensation p1;
    p1.type = SensationType::Pain;
    p1.intensity = 0.3f;
    p1.valence = -1.0f;
    p1.source = "mission-pressure";
    PrimalSensation p2 = p1;
    p2.intensity = 0.8f;
    engine.add(p1);
    engine.add(p2);
    /* mission pressure must stay ONE signal p(t): stacking copies would make
       homeostaticCost grow like N*p and break the gT^2/2 model */
    ASSERT_EQ(engine.active().size(), 1u);
    EXPECT_FLOAT_EQ(engine.active()[0].intensity, 0.8f);
    EXPECT_FLOAT_EQ(engine.homeostaticCost(), 0.8f);
}

TEST_F(PrimalSensationTest, EmptySourceStillStacks) {
    PrimalSensationEngine engine;
    PrimalSensation a, b;
    a.type = SensationType::Pain;
    a.intensity = 0.5f;
    b = a;
    engine.add(a);
    engine.add(b);
    /* untagged transient events keep accumulate semantics */
    EXPECT_EQ(engine.active().size(), 2u);
}

TEST_F(PrimalSensationTest, DecayAutoReleasesCompletedPressure) {
    PrimalSensationEngine engine;
    PrimalSensation p;
    p.type = SensationType::Pain;
    p.intensity = 1.0f;
    p.valence = -1.0f;
    p.source = "mission-pressure";
    engine.add(p);
    /* default half-life is 300 s; 150 s = 0.5 half-life -> 2^-0.5 */
    engine.decayAuto(150.0f);
    ASSERT_EQ(engine.active().size(), 1u);
    EXPECT_NEAR(engine.active()[0].intensity, 0.70710678f, 1e-4f);
    /* 10 more half-lives -> below the 1e-3 prune threshold */
    engine.decayAuto(3000.0f);
    EXPECT_TRUE(engine.active().empty());
    EXPECT_FLOAT_EQ(engine.homeostaticCost(), 0.0f);
}

TEST_F(PrimalSensationTest, DecayAutoUsesPerTypeTuning) {
    PrimalSensationEngine engine;
    engine.setTuning({{SensationType::Pain, {1.0f, 60.0f, 0.0f}}});
    PrimalSensation p;
    p.type = SensationType::Pain;
    p.intensity = 1.0f;
    p.valence = -1.0f;
    p.source = "mission-pressure";
    engine.add(p);
    engine.decayAuto(60.0f);  /* one tuned half-life */
    ASSERT_EQ(engine.active().size(), 1u);
    EXPECT_NEAR(engine.active()[0].intensity, 0.5f, 1e-4f);
}

TEST_F(PrimalSensationTest, EngineDecay) {
    PrimalSensationEngine engine;
    PrimalSensation s;
    s.type = SensationType::Pain;
    s.intensity = 1.0f;
    engine.add(s);
    EXPECT_EQ(engine.active().size(), 1u);

    engine.decay(1.0f, 10.0f);  // 10 half-lives -> ~0
    EXPECT_TRUE(engine.active().empty());
    EXPECT_FLOAT_EQ(engine.netValence(), 0.0f);
    EXPECT_FLOAT_EQ(engine.netArousal(), 0.0f);
}

TEST_F(PrimalSensationTest, NetValenceAndArousal) {
    PrimalSensationEngine engine;
    PrimalSensation pleasure, pain;
    pleasure.type = SensationType::Pleasure;
    pleasure.intensity = 0.8f;
    pleasure.valence = 0.9f;
    pain.type = SensationType::Pain;
    pain.intensity = 0.4f;
    pain.valence = -0.9f;

    engine.add(pleasure);
    engine.add(pain);

    float valence = engine.netValence();
    EXPECT_GT(valence, 0.0f) << "Pleasure should dominate";
    EXPECT_LE(valence, 1.0f);

    float arousal = engine.netArousal();
    EXPECT_FLOAT_EQ(arousal, 0.8f);
}

class InstinctTest : public HacktestBase {};

TEST_F(InstinctTest, DefaultEngineHasInstincts) {
    auto engine = InstinctEngine::defaultEngine();
    EXPECT_FALSE(engine.instincts().empty());
}

TEST_F(InstinctTest, BenefitHarmEvaluation) {
    auto engine = InstinctEngine::defaultEngine();
    std::vector<PrimalSensation> sensations;
    PrimalSensation threat;
    threat.type = SensationType::Threat;
    threat.intensity = 0.9f;
    threat.valence = -0.9f;
    sensations.push_back(threat);

    auto result = engine.evaluate(sensations);
    EXPECT_GT(result.harmScore, result.benefitScore) << "Threat should raise harm";
    EXPECT_LT(result.netUtility, 0.0f);
    EXPECT_FALSE(result.recommendedAction.empty());
    EXPECT_FALSE(result.driveVector.empty());
}

TEST_F(InstinctTest, DriveToEmotion) {
    BenefitHarmResult result;
    result.benefitScore = 0.3f;
    result.harmScore = 0.1f;
    result.netUtility = 0.2f;
    result.driveVector.assign(8, 0.0f);
    result.driveVector[0] = 0.2f;
    result.driveVector[1] = 0.3f;
    result.driveVector[3] = 0.1f;

    auto emotion = InstinctEngine::driveToEmotion(result);
    EXPECT_FLOAT_EQ(emotion.valence, 0.2f);
    EXPECT_FLOAT_EQ(emotion.arousal, 0.3f);
    EXPECT_FLOAT_EQ(emotion.trust, 0.1f);
}

TEST_F(InstinctTest, ActionEvaluationBoostsAlignedInstinct) {
    auto engine = InstinctEngine::defaultEngine();
    std::vector<PrimalSensation> sensations;
    PrimalSensation novel;
    novel.type = SensationType::Novelty;
    novel.intensity = 0.7f;
    novel.valence = 0.5f;
    sensations.push_back(novel);

    auto base = engine.evaluate(sensations);
    auto action = engine.evaluateAction("explore", sensations);
    EXPECT_GE(action.netUtility, base.netUtility) << "Explore should align with novelty";
}

TEST_F(InstinctTest, JsonRoundTrip) {
    Instinct i;
    i.type = InstinctType::Survival;
    i.activation = 0.9f;
    i.benefitWeight = 0.2f;
    i.harmWeight = 0.95f;
    i.targetSensation = SensationType::Threat;
    i.actionBias = "protect";

    auto j = i.toJson();
    auto loaded = Instinct::fromJson(j);

    EXPECT_EQ(loaded.type, i.type);
    EXPECT_FLOAT_EQ(loaded.activation, i.activation);
    EXPECT_EQ(loaded.targetSensation, i.targetSensation);
    EXPECT_EQ(loaded.actionBias, i.actionBias);
}
