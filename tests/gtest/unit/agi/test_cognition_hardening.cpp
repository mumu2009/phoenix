/* test_cognition_hardening.cpp - Volume, extremes, penetration, performance,
   and public-API coverage tests for CognitionAutonomyManager. */

#include "autonomy_stack.hpp"
#include "addon.hpp"
#include "test_hacktest_framework.hpp"
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

using namespace autonomy;
using namespace phoenix::testing;

static nlohmann::json loadThresholds() {
    std::ifstream f("tests/gtest/perf_thresholds.json");
    if (!f) return nlohmann::json::object();
    nlohmann::json j = nlohmann::json::parse(f, nullptr, false);
    return j.is_discarded() ? nlohmann::json::object() : j;
}

static double elapsedMs(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
        .count();
}

class CognitionHardeningTest : public HacktestBase {
protected:
    std::unique_ptr<CognitionAutonomyManager> manager_;
    nlohmann::json thresholds_;

    void SetUp() override {
        HacktestBase::SetUp();
        manager_ = std::make_unique<CognitionAutonomyManager>();
        thresholds_ = loadThresholds();
    }

    void TearDown() override {
        addon::clearAddonOnlineLookupHandler();
        addon::clearAddonComputerShellHandler();
        manager_.reset();
    }
};

// 3. Doubled test volume: repeat the core action-dispatch loop many times and
//    ensure the backlog and action space stay consistent.
TEST_F(CognitionHardeningTest, Volume_DoubleToolDispatchLoop) {
    const int n = 64;  // 2x the typical 32-call smoke loop
    for (int i = 0; i < n; ++i) {
        nlohmann::json ctx{{"userPrompt", std::to_string(i) + "*" + std::to_string(i)}};
        auto res = manager_->executeAgiActionByName("math", ctx);
        ASSERT_TRUE(res.value("ok", false)) << res.value("error", "");
        auto reply = res["result"].value("reply", std::string());
        EXPECT_NE(reply.find(std::to_string(i * i)), std::string::npos) << reply;
    }

    for (int i = 0; i < n; ++i) {
        nlohmann::json ctx{{"userPrompt", "goal " + std::to_string(i)}};
        auto res = manager_->executeAgiActionByName("goal_advance", ctx);
        ASSERT_TRUE(res.value("ok", false));
    }

    auto list = manager_->listAgiActions();
    ASSERT_TRUE(list.value("ok", false));
    EXPECT_EQ(list["result"].size(), 6u);
}

// 4. Extreme environment / special scenarios
TEST_F(CognitionHardeningTest, Extreme_EmptyAndNullPayloads) {
    EXPECT_NO_THROW(manager_->observe(nlohmann::json::object(), nlohmann::json::object()));
    EXPECT_NO_THROW(manager_->iterate(nlohmann::json::object(), nlohmann::json::object()));
    EXPECT_NO_THROW(manager_->session(""));
    EXPECT_NO_THROW(manager_->exportState());
    EXPECT_NO_THROW(manager_->importState(nlohmann::json::object()));
    EXPECT_NO_THROW(manager_->ingestSensation(nlohmann::json::object()));
    EXPECT_NO_THROW(manager_->evaluateInstincts());
    EXPECT_NO_THROW(manager_->composePrompt(nlohmann::json::object()));
    EXPECT_NO_THROW(manager_->ingestMixedModalPacket(nlohmann::json::object()));
    EXPECT_NO_THROW(manager_->drainMixedModalOutputs(nlohmann::json::object()));
}

TEST_F(CognitionHardeningTest, Extreme_OversizedInputs) {
    std::string big(65536, 'x');
    nlohmann::json payload{{"sessionId", big}, {"userPrompt", big}};
    nlohmann::json worldState{{"state", big}};
    auto res = manager_->iterate(payload, worldState);
    EXPECT_TRUE(res.is_object());

    nlohmann::json hugeJson;
    for (int i = 0; i < 1024; ++i) {
        hugeJson[std::to_string(i)] = std::string(1024, 'a');
    }
    EXPECT_NO_THROW(manager_->observe(hugeJson, hugeJson));
    EXPECT_NO_THROW(manager_->importState(hugeJson));
}

TEST_F(CognitionHardeningTest, Extreme_MalformedTransition) {
    nlohmann::json trans{{"z", "not an array"}, {"a", 42}, {"zNext", nullptr}};
    auto res = manager_->ingestAgiTransition(trans);
    EXPECT_TRUE(res.is_object());
}

TEST_F(CognitionHardeningTest, Extreme_MismatchedDimensions) {
    auto cfg = manager_->configureAgi({{"enabled", true}, {"dim", 4}});
    EXPECT_TRUE(cfg.value("ok", false));
    nlohmann::json trans{{"z", nlohmann::json::array({1.0f, 2.0f})},
                         {"a", nlohmann::json::array({0.0f, 1.0f, 0.0f})},
                         {"zNext", nlohmann::json::array({1.0f, 3.0f})}};
    auto res = manager_->ingestAgiTransition(trans);
    EXPECT_TRUE(res.is_object());
}

// 5. User penetration / adversarial attack tests
TEST_F(CognitionHardeningTest, Penetration_InvalidActionName) {
    nlohmann::json ctx{{"userPrompt", "1+1"}};
    auto res = manager_->executeAgiActionByName("<script>alert(1)</script>", ctx);
    EXPECT_FALSE(res.value("ok", false));
    EXPECT_NE(res.value("error", std::string()).find("not found"), std::string::npos);
}

TEST_F(CognitionHardeningTest, Penetration_SqlLikePayload) {
    nlohmann::json ctx{{"userPrompt", "'; DROP TABLE users; --"}};
    auto res = manager_->executeAgiActionByName("math", ctx);
    // Math addon should fail gracefully, never throw: the request is handled
    // and the parse failure is surfaced as an explicit "[math error]" reply
    // (SQL is data, not executed - no shell/eval path exists).
    ASSERT_TRUE(res.is_object());
    EXPECT_TRUE(res.contains("result"));
    const std::string reply = res["result"].value("reply", std::string());
    EXPECT_NE(reply.find("math error"), std::string::npos)
        << "SQL-like payload must produce an explicit math error reply";
}

TEST_F(CognitionHardeningTest, Penetration_AdversarialRegisterAction) {
    nlohmann::json reg{{"name", "../../../etc/passwd"},
                       {"category", "tool"},
                       {"addonType", "math"},
                       {"description", "evil"}};
    auto res = manager_->registerAgiAction(reg);
    EXPECT_TRUE(res.is_object());
    if (res.value("ok", false)) {
        // If registered, it should still not escape the sandbox.
        nlohmann::json ctx{{"userPrompt", "1+1"}};
        auto exec = manager_->executeAgiActionByName("../../../etc/passwd", ctx);
        EXPECT_TRUE(exec.is_object());
    }
}

TEST_F(CognitionHardeningTest, Penetration_LongGoalFlood) {
    const int n = 2048;
    for (int i = 0; i < n; ++i) {
        nlohmann::json ctx{{"userPrompt", std::to_string(i)}};
        manager_->executeAgiActionByName("goal_advance", ctx);
    }
    // Goals vector is capped at 1024; ensure no runaway growth.
    nlohmann::json ctx{{"userPrompt", "probe"}};
    auto res = manager_->executeAgiActionByName("goal_advance", ctx);
    ASSERT_TRUE(res.value("ok", false));
    auto goals = res["result"].value("goals", nlohmann::json::array());
    EXPECT_LE(goals.size(), 1024u);
    EXPECT_EQ(goals.back().get<std::string>(), "probe");
}

// 6. Performance metric tests driven by the config file
TEST_F(CognitionHardeningTest, Performance_ExecuteActionWithinThreshold) {
    nlohmann::json ctx{{"userPrompt", "1+1"}};
    auto start = std::chrono::steady_clock::now();
    auto res = manager_->executeAgiActionByName("math", ctx);
    double ms = elapsedMs(start);
    EXPECT_TRUE(res.value("ok", false));
    double limit = thresholds_.value("executeAgiActionMs", 100.0);
    EXPECT_LT(ms, limit) << "math dispatch took " << ms << " ms, limit " << limit;
}

TEST_F(CognitionHardeningTest, Performance_IterateWithinThreshold) {
    auto cfg = manager_->configureAgi({{"enabled", false}});  // keep iteration cheap
    ASSERT_TRUE(cfg.value("ok", false));
    nlohmann::json payload{{"sessionId", "perf"}, {"userPrompt", "hi"}};
    nlohmann::json worldState = nlohmann::json::object();
    auto start = std::chrono::steady_clock::now();
    auto res = manager_->iterate(payload, worldState);
    double ms = elapsedMs(start);
    EXPECT_TRUE(res.value("ok", false));
    double limit = thresholds_.value("iterateMs", 500.0);
    EXPECT_LT(ms, limit) << "iterate took " << ms << " ms, limit " << limit;
}

TEST_F(CognitionHardeningTest, Performance_StatusAndPlanWithinThreshold) {
    auto start = std::chrono::steady_clock::now();
    auto st = manager_->status();
    double ms = elapsedMs(start);
    EXPECT_TRUE(st.is_object());
    double limit = thresholds_.value("statusMs", 100.0);
    EXPECT_LT(ms, limit);

    manager_->configureAgi({{"enabled", true}, {"dim", 4}});
    start = std::chrono::steady_clock::now();
    auto plan = manager_->agiPlan();
    ms = elapsedMs(start);
    EXPECT_TRUE(plan.is_object());
    limit = thresholds_.value("planMs", 100.0);
    EXPECT_LT(ms, limit);
}

// 7. Comment / API coverage test: every public method must be callable and return
//    a valid JSON object without throwing.
TEST_F(CognitionHardeningTest, Coverage_AllPublicMethodsReturnObjects) {
    nlohmann::json observePayload = nlohmann::json::object();
    observePayload["sessionId"] = "cov";
    nlohmann::json mathCtx = nlohmann::json::object();
    mathCtx["userPrompt"] = "1+1";
    nlohmann::json promptCtx = nlohmann::json::object();
    promptCtx["userPrompt"] = "hello";
    nlohmann::json packet = nlohmann::json::object();
    packet["modality"] = "text";
    packet["payload"] = nlohmann::json::array({104, 101, 108, 108, 111});
    packet["mimeType"] = "text/plain";
    packet["source"] = "coverage";

    EXPECT_TRUE(manager_->status().is_object());
    EXPECT_TRUE(manager_->observe(observePayload, nlohmann::json::object()).is_object());
    EXPECT_TRUE(manager_->iterate(observePayload, nlohmann::json::object()).is_object());
    EXPECT_TRUE(manager_->session("cov").is_object());
    EXPECT_TRUE(manager_->exportState().is_object());
    EXPECT_TRUE(manager_->importState(nlohmann::json::object()).is_object());
    EXPECT_TRUE(manager_->ingestSensation(nlohmann::json::object()).is_object());
    EXPECT_TRUE(manager_->evaluateInstincts().is_object());
    EXPECT_TRUE(manager_->configureAgi({{"enabled", false}}).is_object());
    EXPECT_TRUE(manager_->agiPlan().is_object());
    EXPECT_TRUE(manager_->ingestAgiTransition(nlohmann::json::object()).is_object());
    EXPECT_TRUE(manager_->registerAgiAction(nlohmann::json::object()).is_object());
    EXPECT_TRUE(manager_->listAgiActions().is_object());
    EXPECT_TRUE(manager_->executeAgiActionByName("math", mathCtx).is_object());
    EXPECT_TRUE(manager_->configureSubconscious(nlohmann::json::object()).is_object());
    EXPECT_TRUE(manager_->composePrompt(promptCtx).is_object());
    EXPECT_TRUE(manager_->ingestMixedModalPacket(packet).is_object());
    EXPECT_TRUE(manager_->emitMixedModalOutput(nlohmann::json::object()).is_object());
    EXPECT_TRUE(manager_->drainMixedModalOutputs(nlohmann::json::object()).is_object());
}

// 8. Improve assertion realism: inspect concrete fields instead of just
//    EXPECT_NO_THROW / EXPECT_TRUE(is_object).
TEST_F(CognitionHardeningTest, Realism_StatusHasExpectedFields) {
    auto st = manager_->status();
    ASSERT_TRUE(st.value("ok", false));
    auto result = st.value("result", nlohmann::json::object());
    EXPECT_TRUE(result.contains("enabled"));
    EXPECT_TRUE(result.contains("iteration"));
    EXPECT_TRUE(result.contains("observations"));
    EXPECT_TRUE(result.contains("agi"));
    EXPECT_TRUE(result["agi"].is_object());
    EXPECT_TRUE(result["agi"].contains("enabled"));
}

TEST_F(CognitionHardeningTest, Realism_IterateResultHasAgiPlanAndBenefitHarm) {
    auto cfg = manager_->configureAgi({{"enabled", true}, {"dim", 4}});
    ASSERT_TRUE(cfg.value("ok", false));
    nlohmann::json payload{{"sessionId", "real"}, {"userPrompt", "test"}};
    auto res = manager_->iterate(payload, nlohmann::json::object());
    ASSERT_TRUE(res.value("ok", false));
    auto result = res["result"];
    EXPECT_TRUE(result.contains("benefitHarm")) << result.dump();
    EXPECT_TRUE(result.contains("agiPlan")) << result.dump();
    EXPECT_TRUE(result.contains("iteration"));
    EXPECT_GE(result.value("iteration", 0), 1);
}

TEST_F(CognitionHardeningTest, Realism_RegisteredActionHasAllFields) {
    auto list = manager_->listAgiActions();
    ASSERT_TRUE(list.value("ok", false));
    auto actions = list["result"];
    ASSERT_FALSE(actions.empty());
    for (const auto &a : actions) {
        EXPECT_TRUE(a.contains("name")) << a.dump();
        EXPECT_TRUE(a.contains("category")) << a.dump();
        EXPECT_TRUE(a.contains("description")) << a.dump();
    }
}
