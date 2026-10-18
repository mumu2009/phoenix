/* Chat affect map: finer appraisal, sampling band, thought coupling.
   The standalone driver in test/emotion_sampling_driver.cpp is the
   one-file run; this file is the same invariants for the gtest suite. */

#include "emotion_system.hpp"
#include "emotion_influence.hpp"
#include "instinct.hpp"
#include "self_mind.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <string>

using phoenix::emotion::EmotionTensor;
using phoenix::emotion::FineAppraisal;

TEST(EmotionSampling, NeutralFineAppraisalMatchesBenefitHarm) {
    const float pairs[][2] = {{0.f, 0.f}, {1.f, 0.f}, {0.f, 1.f}, {0.3f, 0.7f}};
    for (const auto &p : pairs) {
        FineAppraisal fa;
        fa.benefit = p[0];
        fa.harm = p[1];
        const auto rich = phoenix::emotion::fromFineAppraisal(fa);
        const auto old = phoenix::emotion::fromAppraisal(p[0], p[1]);
        EXPECT_NEAR(rich.valence, old.valence, 1e-5f);
        EXPECT_NEAR(rich.arousal, old.arousal, 1e-5f);
        EXPECT_NEAR(rich.dominance, old.dominance, 1e-5f);
        EXPECT_NEAR(rich.surprise, old.surprise, 1e-5f);
        EXPECT_NEAR(rich.anger, old.anger, 1e-5f);
        EXPECT_NEAR(rich.trust, old.trust, 1e-5f);
    }
}

TEST(EmotionSampling, ChatTemperatureStaysInBand) {
    const float grid[] = {-1.f, 0.f, 1.f};
    for (float arousal : grid) {
        for (float fear : grid) {
            for (float surprise : grid) {
                EmotionTensor e;
                e.arousal = arousal;
                e.fear = fear;
                e.surprise = surprise;
                e.anger = 1.f;
                e.valence = -1.f;
                e.joy = 1.f;
                e.trust = -1.f;
                const auto opts = phoenix::emotion::mapChatSampling(e);
                const double t = opts["temperature"].get<double>();
                /* 0.40f promoted to double sits one ulp above the double literal 0.4. */
                EXPECT_GE(t, 0.15 - 1e-6);
                EXPECT_LE(t, 0.40 + 1e-6);
                EXPECT_GE(opts["frequency_penalty"].get<double>(), 0.35 - 1e-5);
                EXPECT_GE(opts["presence_penalty"].get<double>(), 0.15 - 1e-5);
                EXPECT_GE(opts["top_p"].get<double>(), 0.50 - 1e-6);
                EXPECT_LE(opts["top_p"].get<double>(), 0.95 + 1e-6);
            }
        }
    }
    EmotionTensor neutral;
    EXPECT_NEAR(phoenix::emotion::mapChatSampling(neutral)["temperature"].get<double>(), 0.30, 1e-6);
}

TEST(EmotionSampling, ThoughtChangesAffectButNotTheCausalStringOrPolicy) {
    phoenix::selfmind::SelfMind mind;
    mind.configure(nlohmann::json{{"enabled", true}, {"learningRate", 0.35}});
    const auto tick = mind.tickThought(nlohmann::json{{"userPrompt", "ZZZ-QUESTION-TOKEN?"}});
    const auto snap = mind.toJson();
    const std::string thought = tick.value("text", std::string());
    const auto prefer = phoenix::emotion::blendThoughtAffect(
        EmotionTensor{}, "prefer", thought, 0.22f);
    const auto service = phoenix::emotion::blendThoughtAffect(
        EmotionTensor{}, "service", "", 0.22f);
    EXPECT_GT(prefer.valence, service.valence);
    EXPECT_EQ(mind.toJson(), snap);
    const std::string user = "What is two plus two?";
    const std::string causal = phoenix::emotion::chatCausalText(user);
    EXPECT_EQ(causal, user + "\n");
    EXPECT_EQ(causal.find(thought), std::string::npos);
    EXPECT_EQ(causal.find("Affect:"), std::string::npos);
    EXPECT_EQ(causal.find("<emotion"), std::string::npos);
    EXPECT_EQ(causal.find("[self-kernel]"), std::string::npos);

    phoenix::emotion_influence::EmotionInfluence influence;
    const auto r = influence.compute(prefer, user, 1);
    EXPECT_TRUE(r.promptModulation.empty());
    const double t = r.inferenceOptions["temperature"].get<double>();
    EXPECT_GE(t, 0.15 - 1e-6);
    EXPECT_LE(t, 0.40 + 1e-6);

    const auto felt = phoenix::instinct::sensationFromThought("ask", "chat:s1");
    EXPECT_EQ(felt.source, "chat:s1:thought");
    EXPECT_EQ(felt.source.find(thought), std::string::npos);
}

TEST(EmotionSampling, InstinctNoveltyLiftsArousalWithoutBreakingTheOldCore) {
    auto eng = phoenix::instinct::InstinctEngine::defaultEngine();
    const auto none = eng.evaluate({});
    const auto plain = phoenix::emotion::fromAppraisal(none.benefitScore, none.harmScore);
    ASSERT_EQ(none.driveVector.size(), 8u);
    EXPECT_NEAR(none.driveVector[1], plain.arousal, 1e-4f);

    phoenix::primal::PrimalSensation nov;
    nov.type = phoenix::primal::SensationType::Novelty;
    nov.intensity = 0.9f;
    nov.valence = 0.0f;
    const auto with = eng.evaluate({nov});
    const auto withPlain = phoenix::emotion::fromAppraisal(with.benefitScore, with.harmScore);
    EXPECT_GT(with.novelty, 0.5f);
    EXPECT_GT(with.driveVector[1], withPlain.arousal + 0.2f);
}
