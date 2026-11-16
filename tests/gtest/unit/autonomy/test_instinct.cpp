/* test_instinct.cpp - GTest unit tests for instinct.cpp
   Covers all public functions and classes in instinct.hpp. */

#include "instinct.hpp"
#include "primal_sensation.hpp"
#include "test_hacktest_framework.hpp"
#include <gtest/gtest.h>
#include <vector>

using namespace phoenix::instinct;
using namespace phoenix::primal;
using namespace phoenix::testing;
using json = nlohmann::json;

namespace {
InstinctEngine makeDefaultEngine() { return InstinctEngine::defaultEngine(); }
}

// Instinct type conversions
TEST(InstinctTypeConversion, TypeToStringKnown) {
    EXPECT_EQ(Instinct::typeToString(InstinctType::Survival), "survival");
    EXPECT_EQ(Instinct::typeToString(InstinctType::Exploration), "exploration");
    EXPECT_EQ(Instinct::typeToString(InstinctType::Avoidance), "avoidance");
    EXPECT_EQ(Instinct::typeToString(InstinctType::Affiliation), "affiliation");
    EXPECT_EQ(Instinct::typeToString(InstinctType::Curiosity), "curiosity");
}

TEST(InstinctTypeConversion, TypeToStringUnknown) {
    EXPECT_EQ(Instinct::typeToString(InstinctType::Unknown), "unknown");
}

TEST(InstinctTypeConversion, StringToTypeKnown) {
    EXPECT_EQ(Instinct::stringToType("survival"), InstinctType::Survival);
    EXPECT_EQ(Instinct::stringToType("exploration"), InstinctType::Exploration);
    EXPECT_EQ(Instinct::stringToType("avoidance"), InstinctType::Avoidance);
    EXPECT_EQ(Instinct::stringToType("affiliation"), InstinctType::Affiliation);
    EXPECT_EQ(Instinct::stringToType("curiosity"), InstinctType::Curiosity);
}

TEST(InstinctTypeConversion, StringToTypeUnknown) {
    EXPECT_EQ(Instinct::stringToType("invalid"), InstinctType::Unknown);
    EXPECT_EQ(Instinct::stringToType(""), InstinctType::Unknown);
}

TEST(InstinctTypeConversion, RoundTrip) {
    for (const auto t : {InstinctType::Survival, InstinctType::Exploration,
                         InstinctType::Avoidance, InstinctType::Affiliation,
                         InstinctType::Curiosity, InstinctType::Unknown}) {
        EXPECT_EQ(Instinct::stringToType(Instinct::typeToString(t)), t);
    }
}

// Instinct JSON
TEST(InstinctJson, ToJsonContainsFields) {
    Instinct i{InstinctType::Curiosity, 0.7f, 0.8f, 0.2f,
               SensationType::Novelty, "investigate"};
    json j = i.toJson();
    EXPECT_EQ(j["type"], "curiosity");
    EXPECT_FLOAT_EQ(j["activation"].get<float>(), 0.7f);
    EXPECT_FLOAT_EQ(j["benefitWeight"].get<float>(), 0.8f);
    EXPECT_FLOAT_EQ(j["harmWeight"].get<float>(), 0.2f);
    EXPECT_EQ(j["actionBias"], "investigate");
    EXPECT_TRUE(j.contains("targetSensation"));
}

TEST(InstinctJson, FromJsonRoundTrip) {
    Instinct i{InstinctType::Exploration, 0.6f, 0.9f, 0.1f,
               SensationType::Pleasure, "explore"};
    Instinct r = Instinct::fromJson(i.toJson());
    EXPECT_EQ(r.type, i.type);
    EXPECT_FLOAT_EQ(r.activation, i.activation);
    EXPECT_FLOAT_EQ(r.benefitWeight, i.benefitWeight);
    EXPECT_FLOAT_EQ(r.harmWeight, i.harmWeight);
    EXPECT_EQ(r.actionBias, i.actionBias);
    EXPECT_EQ(r.targetSensation, i.targetSensation);
}

TEST(InstinctJson, FromJsonInvalid) {
    Instinct i = Instinct::fromJson(json(nullptr));
    EXPECT_EQ(i.type, InstinctType::Unknown);
    EXPECT_FLOAT_EQ(i.activation, 0.0f);
}

// BenefitHarmResult
TEST(BenefitHarmResultTest, ToJsonRoundTrip) {
    BenefitHarmResult r;
    r.benefitScore = 0.8f;
    r.harmScore = 0.2f;
    r.netUtility = 0.6f;
    r.recommendedAction = "explore";
    r.driveVector = {0.5f, 0.4f, 0.3f, 0.2f, 0.1f, 0.0f, -0.1f, 0.6f};
    json j = r.toJson();
    EXPECT_FLOAT_EQ(j["benefitScore"].get<float>(), 0.8f);
    EXPECT_FLOAT_EQ(j["harmScore"].get<float>(), 0.2f);
    EXPECT_FLOAT_EQ(j["netUtility"].get<float>(), 0.6f);
    EXPECT_EQ(j["recommendedAction"], "explore");
    EXPECT_EQ(j["driveVector"].size(), 8u);
}

// InstinctEngine construction
TEST(InstinctEngineTest, DefaultConstructor) {
    InstinctEngine e;
    EXPECT_TRUE(e.instincts().empty());
}

TEST(InstinctEngineTest, DefaultEngineHasInstincts) {
    InstinctEngine e = InstinctEngine::defaultEngine();
    EXPECT_FALSE(e.instincts().empty());
}

TEST(InstinctEngineTest, ConstructorWithDefaults) {
    std::vector<Instinct> defs = {
        {InstinctType::Survival, 0.5f, 0.5f, 0.5f, SensationType::Threat, "run"}
    };
    InstinctEngine e(defs);
    EXPECT_EQ(e.instincts().size(), 1u);
}

TEST(InstinctEngineTest, RegisterInstinct) {
    InstinctEngine e;
    e.registerInstinct({InstinctType::Curiosity, 0.5f, 0.9f, 0.1f,
                        SensationType::Novelty, "look"});
    EXPECT_EQ(e.instincts().size(), 1u);
}

// Evaluate
TEST(InstinctLogicTest, EvaluateEmptySensations) {
    auto engine = makeDefaultEngine();
    BenefitHarmResult r = engine.evaluate({}, 1.0f);
    EXPECT_FALSE(std::isnan(r.benefitScore));
    EXPECT_FALSE(std::isnan(r.harmScore));
    EXPECT_FALSE(std::isnan(r.netUtility));
    EXPECT_GE(r.benefitScore, 0.0f);
    EXPECT_LE(r.benefitScore, 1.0f);
    EXPECT_GE(r.harmScore, 0.0f);
    EXPECT_LE(r.harmScore, 1.0f);
    EXPECT_GE(r.netUtility, -1.0f);
    EXPECT_LE(r.netUtility, 1.0f);
    EXPECT_FALSE(r.recommendedAction.empty());
    EXPECT_EQ(r.driveVector.size(), 8u);
}

TEST(InstinctLogicTest, EvaluateWithThreat) {
    auto engine = makeDefaultEngine();
    PrimalSensation s{SensationType::Threat, 0.9f, -0.8f, 0.0f, "test", 0};
    BenefitHarmResult r = engine.evaluate({s}, 1.0f);
    EXPECT_GE(r.harmScore, 0.0f);
    EXPECT_LE(r.netUtility, 1.0f);
    EXPECT_FALSE(r.recommendedAction.empty());
}

TEST(InstinctLogicTest, EvaluateWithPleasure) {
    auto engine = makeDefaultEngine();
    PrimalSensation s{SensationType::Pleasure, 0.9f, 0.9f, 0.0f, "test", 0};
    BenefitHarmResult r = engine.evaluate({s}, 1.0f);
    EXPECT_GE(r.benefitScore, 0.0f);
    EXPECT_LE(r.benefitScore, 1.0f);
    EXPECT_FALSE(r.recommendedAction.empty());
}

TEST(InstinctLogicTest, EvaluateTemperatureVariations) {
    auto engine = makeDefaultEngine();
    PrimalSensation s{SensationType::Novelty, 0.7f, 0.5f, 0.0f, "test", 0};
    for (float t : {0.5f, 1.0f, 2.0f, 5.0f}) {
        BenefitHarmResult r = engine.evaluate({s}, t);
        EXPECT_FALSE(std::isnan(r.netUtility));
        EXPECT_GE(r.netUtility, -1.0f);
        EXPECT_LE(r.netUtility, 1.0f);
    }
}

TEST(InstinctLogicTest, EvaluateActionProtect) {
    auto engine = makeDefaultEngine();
    PrimalSensation s{SensationType::Threat, 0.8f, -0.7f, 0.0f, "test", 0};
    BenefitHarmResult r = engine.evaluateAction("protect", {s});
    EXPECT_GE(r.netUtility, -1.0f);
    EXPECT_LE(r.netUtility, 1.0f);
}

TEST(InstinctLogicTest, EvaluateActionExplore) {
    auto engine = makeDefaultEngine();
    PrimalSensation s{SensationType::Novelty, 0.8f, 0.7f, 0.0f, "test", 0};
    BenefitHarmResult r = engine.evaluateAction("explore", {s});
    EXPECT_GE(r.netUtility, -1.0f);
    EXPECT_LE(r.netUtility, 1.0f);
}

// Drive to emotion
TEST(InstinctLogicTest, DriveToEmotionFullVector) {
    auto engine = makeDefaultEngine();
    PrimalSensation s{SensationType::Pleasure, 1.0f, 0.9f, 0.0f, "test", 0};
    BenefitHarmResult r = engine.evaluate({s}, 1.0f);
    auto tensor = InstinctEngine::driveToEmotion(r);
    EXPECT_FALSE(std::isnan(tensor.valence));
    EXPECT_FALSE(std::isnan(tensor.arousal));
    EXPECT_FALSE(std::isnan(tensor.dominance));
    EXPECT_GE(tensor.valence, -1.0f);
    EXPECT_LE(tensor.valence, 1.0f);
    EXPECT_GE(tensor.arousal, -1.0f);
    EXPECT_LE(tensor.arousal, 1.0f);
}

// Engine JSON
TEST(InstinctLogicTest, EngineToJson) {
    auto engine = makeDefaultEngine();
    json j = engine.toJson();
    EXPECT_TRUE(j.is_array());
    EXPECT_FALSE(j.empty());
}

TEST(InstinctLogicTest, EngineFromJsonRoundTrip) {
    auto engine = makeDefaultEngine();
    json j = engine.toJson();
    InstinctEngine restored = InstinctEngine::fromJson(j);
    EXPECT_EQ(restored.instincts().size(), engine.instincts().size());
}

TEST(InstinctEngineTest, EngineFromJsonInvalid) {
    InstinctEngine e = InstinctEngine::fromJson(json(nullptr));
    EXPECT_TRUE(e.instincts().empty());
}

// Edge cases
TEST(InstinctEngineTest, EvaluateNoInstincts) {
    InstinctEngine e;
    PrimalSensation s{SensationType::Pain, 1.0f, -1.0f, 0.0f, "test", 0};
    BenefitHarmResult r = e.evaluate({s}, 1.0f);
    EXPECT_FLOAT_EQ(r.benefitScore, 0.0f);
    EXPECT_FLOAT_EQ(r.harmScore, 0.0f);
    EXPECT_FLOAT_EQ(r.netUtility, 0.0f);
    EXPECT_EQ(r.driveVector.size(), 8u);
    EXPECT_TRUE(r.recommendedAction.empty())
        << "No action word should be synthesized when no instinct provides one";
}

TEST(InstinctEngineTest, EvaluateActionNoInstincts) {
    InstinctEngine e;
    BenefitHarmResult r = e.evaluateAction("run", {});
    EXPECT_FLOAT_EQ(r.netUtility, 0.0f);
}

TEST(InstinctLogicTest, MultipleSensations) {
    auto engine = makeDefaultEngine();
    std::vector<PrimalSensation> sensations = {
        {SensationType::Threat, 0.5f, -0.6f, 0.0f, "a", 0},
        {SensationType::Novelty, 0.4f, 0.5f, 0.0f, "b", 0},
        {SensationType::SocialIsolation, 0.3f, -0.3f, 0.0f, "c", 0}
    };
    BenefitHarmResult r = engine.evaluate(sensations, 1.0f);
    EXPECT_FALSE(std::isnan(r.netUtility));
    EXPECT_GE(r.netUtility, -1.0f);
    EXPECT_LE(r.netUtility, 1.0f);
    EXPECT_FALSE(r.recommendedAction.empty());
}

TEST(InstinctLogicTest, DriveVectorValuesClamped) {
    auto engine = makeDefaultEngine();
    PrimalSensation s{SensationType::Pain, 1.0f, -1.0f, 0.0f, "test", 0};
    BenefitHarmResult r = engine.evaluate({s}, 1.0f);
    for (float v : r.driveVector) {
        EXPECT_GE(v, -1.0f);
        EXPECT_LE(v, 1.0f);
    }
}

// Dynamic instinct activation (v7.0 regulation)
TEST(InstinctEngineTest, CurrentActivationsParallelToInstincts) {
    InstinctEngine e;
    e.registerInstinct({InstinctType::Survival, 0.5f, 0.3f, 0.9f,
                        SensationType::Threat, "protect"});
    EXPECT_EQ(e.currentActivations().size(), e.instincts().size());
    EXPECT_FLOAT_EQ(e.currentActivations()[0], 0.5f);
}

TEST(InstinctEngineTest, UpdateAmplifiesMatchingSensation) {
    InstinctEngine e;
    e.registerInstinct({InstinctType::Avoidance, 0.4f, 0.1f, 0.95f,
                        SensationType::Pain, "avoid"});
    PrimalSensation s{SensationType::Pain, 1.0f, -1.0f, 0.0f, "test", 0};
    e.update({s}, 1.0f);
    auto activations = e.currentActivations();
    EXPECT_GT(activations[0], 0.4f);
    EXPECT_LE(activations[0], 1.0f);
}

TEST(InstinctEngineTest, UpdateDecaysWhenNoSensation) {
    InstinctEngine e;
    e.registerInstinct({InstinctType::Exploration, 0.8f, 0.9f, 0.2f,
                        SensationType::Novelty, "explore"});
    e.update({}, 60.0f);
    auto activations = e.currentActivations();
    EXPECT_LT(activations[0], 0.8f);
    EXPECT_GE(activations[0], 0.0f);
}

TEST(InstinctEngineTest, EvaluateUsesUpdatedActivation) {
    InstinctEngine e;
    e.registerInstinct({InstinctType::Avoidance, 0.2f, 0.1f, 0.95f,
                        SensationType::Pain, "avoid"});
    PrimalSensation s{SensationType::Pain, 1.0f, -1.0f, 0.0f, "test", 0};
    BenefitHarmResult before = e.evaluate({s}, 1.0f);
    e.update({s}, 1.0f);
    BenefitHarmResult after = e.evaluate({s}, 1.0f);
    EXPECT_GE(after.harmScore, before.harmScore);
}
