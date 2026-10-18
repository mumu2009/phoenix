/* Self-mind: dialogue bandit, question-independent thought, mirror/Turing. */

#include "autonomy_stack.hpp"
#include "self_mind.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <string>

using json = nlohmann::json;

namespace {

const char *kCorpus = "doc/v8.3/self_mind_sample.json";

}  // namespace

TEST(SelfMindTest, LearnsPersonAndThinksWithoutQuestions) {
    phoenix::selfmind::SelfMind mind;
    mind.configure(json{{"enabled", true}, {"learningRate", 0.35}, {"name", "self"}});
    const auto ing = mind.ingest(json{{"path", kCorpus}});
    ASSERT_TRUE(ing.value("ok", false)) << ing.dump();
    EXPECT_GE(mind.status().value("humanCount", 0), 6);
    EXPECT_GE(mind.status().value("holdout", 0), 3);

    const auto thought = mind.tickThought(json{{"userPrompt", "ZZZ-QUESTION-TOKEN?"}});
    EXPECT_TRUE(thought.value("spontaneous", false));
    EXPECT_EQ(thought.value("text", std::string()).find("ZZZ-QUESTION-TOKEN"), std::string::npos);
    EXPECT_NE(thought.value("action", std::string()), "service");

    for (int i = 0; i < 3; ++i) mind.tickThought(json::object());
    const auto st = mind.status();
    ASSERT_TRUE(st.contains("policyBias"));
    EXPECT_LT(st["policyBias"][4].get<double>(), st["policyBias"][0].get<double>());
    EXPECT_TRUE(st.value("weightsFinite", false));
    const std::string kernel = st.value("kernel", std::string());
    EXPECT_NE(kernel.find("[self-kernel]"), std::string::npos);
    EXPECT_EQ(kernel.find("language model"), std::string::npos);
    EXPECT_NE(kernel.find("not a service window"), std::string::npos);
}

TEST(SelfMindTest, MirrorAndTuringPassOnCalibrationCorpus) {
    phoenix::selfmind::SelfMind mind;
    mind.configure(json{{"enabled", true}, {"learningRate", 0.35}});
    ASSERT_TRUE(mind.ingest(json{{"path", kCorpus}}).value("ok", false));
    for (int i = 0; i < 4; ++i) mind.tickThought(json::object());
    const auto probe = mind.probe();
    EXPECT_TRUE(probe["mirror"].value("pass", false)) << probe.dump(2);
    EXPECT_TRUE(probe["turing"].value("pass", false)) << probe.dump(2);
}

TEST(SelfMindTest, AlphaZeroDoesNotMoveAndStateRoundTrips) {
    phoenix::selfmind::SelfMind frozen;
    frozen.configure(json{{"enabled", true}, {"learningRate", 0.0}});
    ASSERT_TRUE(frozen.ingest(json{{"path", kCorpus}}).value("ok", false));
    frozen.tickThought(json::object());
    for (const auto &b : frozen.status()["policyBias"])
        EXPECT_NEAR(b.get<double>(), 0.0, 1e-8);
    EXPECT_EQ(frozen.status().value("episodes", -1), 0);

    phoenix::selfmind::SelfMind mind;
    mind.configure(json{{"enabled", true}, {"learningRate", 0.35}});
    mind.ingest(json{{"path", kCorpus}});
    mind.tickThought(json::object());
    phoenix::selfmind::SelfMind copy;
    copy.fromJson(mind.toJson());
    EXPECT_EQ(copy.status().value("humanCount", -1), mind.status().value("humanCount", -2));
    EXPECT_EQ(copy.status().value("lastThought", std::string()),
              mind.status().value("lastThought", std::string("x")));
}

TEST(SelfMindTest, WeightsStayClamped) {
    phoenix::selfmind::SelfMind mind;
    mind.configure(json{{"enabled", true}, {"learningRate", 0.35}});
    ASSERT_TRUE(mind.ingest(json{{"path", kCorpus}}).value("ok", false));
    for (int i = 0; i < 200; ++i) mind.tickThought(json::object());
    EXPECT_TRUE(mind.status().value("weightsFinite", false));
    EXPECT_EQ(mind.status().value("spontaneousCount", 0), 200);
}

TEST(SelfMindTest, IterateThoughtStaysOutOfChatPrompt) {
    autonomy::CognitionAutonomyManager mgr;
    mgr.configureSelfMind(json{{"enabled", true},
                               {"reset", true},
                               {"learningRate", 0.35},
                               {"name", "self"},
                               {"corpusPath", kCorpus}});
    mgr.observe(autonomy::buildCognitionAutonomySeedPayload("self-sess", "inner", 0.4), json{});
    const auto iter = mgr.iterate(json{{"sessionId", "self-sess"},
                                       {"userPrompt", "ZZZ-QUESTION-TOKEN?"}},
                                  json{});
    ASSERT_TRUE(iter.value("ok", false)) << iter.dump();
    ASSERT_TRUE(iter["result"].contains("selfMind"));
    const auto &sm = iter["result"]["selfMind"];
    EXPECT_TRUE(sm.value("spontaneous", false));
    EXPECT_EQ(sm.value("text", std::string()).find("ZZZ-QUESTION-TOKEN"), std::string::npos);
    const std::string composed = iter["result"].value("composedPrompt", std::string());
    EXPECT_EQ(composed.find("[self-kernel]"), std::string::npos);
    EXPECT_EQ(composed.find(sm.value("text", std::string())), std::string::npos);
}
