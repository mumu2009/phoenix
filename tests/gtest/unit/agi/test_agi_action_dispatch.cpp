/* test_agi_action_dispatch.cpp - Verify the AGI planner dispatches to real tools. */

#include "autonomy_stack.hpp"
#include "addon.hpp"
#include "test_hacktest_framework.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <unordered_set>

using namespace autonomy;
using namespace phoenix::testing;

class AgiActionDispatchTest : public HacktestBase {
protected:
    std::unique_ptr<CognitionAutonomyManager> manager_;

    void SetUp() override {
        HacktestBase::SetUp();
        manager_ = std::make_unique<CognitionAutonomyManager>();
    }

    void TearDown() override {
        addon::clearAddonOnlineLookupHandler();
        addon::clearAddonComputerShellHandler();
        manager_.reset();
    }
};

TEST_F(AgiActionDispatchTest, DefaultActionsAreRegistered) {
    auto list = manager_->listAgiActions();
    ASSERT_TRUE(list.value("ok", false));
    auto actions = list.value("result", nlohmann::json::array());
    EXPECT_EQ(actions.size(), 6u);

    std::unordered_set<std::string> names;
    for (const auto &a : actions) names.insert(a.value("name", std::string()));
    EXPECT_TRUE(names.count("math"));
    EXPECT_TRUE(names.count("search"));
    EXPECT_TRUE(names.count("computer"));
    EXPECT_TRUE(names.count("goal_advance"));
    EXPECT_TRUE(names.count("replicate"));
    EXPECT_TRUE(names.count("script"));
}

TEST_F(AgiActionDispatchTest, ExecuteMathTool) {
    nlohmann::json ctx{{"userPrompt", "1+1"}};
    auto res = manager_->executeAgiActionByName("math", ctx);
    ASSERT_TRUE(res.value("ok", false)) << res.value("error", "");
    auto reply = res["result"].value("reply", std::string());
    EXPECT_NE(reply.find("2"), std::string::npos) << "math tool should evaluate 1+1: " << reply;
}

TEST_F(AgiActionDispatchTest, ExecuteSearchToolWithMockLookup) {
    addon::setAddonOnlineLookupHandler([](const nlohmann::json &input, const nlohmann::json &) -> nlohmann::json {
        return nlohmann::json{{"ok", true}, {"snippet", "mock search result for " + input.get<std::string>()}};
    });

    nlohmann::json ctx{{"userPrompt", "phoenix"}};
    auto res = manager_->executeAgiActionByName("search", ctx);
    ASSERT_TRUE(res.value("ok", false)) << res.value("error", "");
    auto reply = res["result"].value("reply", std::string());
    EXPECT_NE(reply.find("mock search result"), std::string::npos) << reply;
}

TEST_F(AgiActionDispatchTest, ExecuteComputerToolWithMockShell) {
    addon::setAddonComputerShellHandler([](const nlohmann::json &request, const nlohmann::json &) -> nlohmann::json {
        if (request.value("op", std::string()) == "pwd") {
            return nlohmann::json{{"ok", true}, {"content", "/mock/pwd"}};
        }
        return nlohmann::json{{"ok", false}, {"error", "unsupported op"}};
    });

    nlohmann::json ctx{{"userPrompt", "pwd"}};
    auto res = manager_->executeAgiActionByName("computer", ctx);
    ASSERT_TRUE(res.value("ok", false)) << res.value("error", "");
    auto reply = res["result"].value("reply", std::string());
    EXPECT_NE(reply.find("/mock/pwd"), std::string::npos) << reply;
}

TEST_F(AgiActionDispatchTest, ExecuteGoalToolAdvancesBacklog) {
    nlohmann::json ctx{{"userPrompt", "build rocket"}};
    auto res = manager_->executeAgiActionByName("goal_advance", ctx);
    ASSERT_TRUE(res.value("ok", false)) << res.value("error", "");
    auto goals = res["result"].value("goals", nlohmann::json::array());
    ASSERT_FALSE(goals.empty());
    EXPECT_EQ(goals.back().get<std::string>(), "build rocket");
}

TEST_F(AgiActionDispatchTest, IterateIncludesExecutionWhenPlannerPicksTool) {
    // Configure AGI so the planner strongly prefers the 'goal_advance' tool:
    // w aligns with one-hot index 3 and the forward model learns that action.
    auto cfg = manager_->configureAgi({
        {"enabled", true},
        {"dim", 4},
        {"horizon", 1},
        {"pragmaticWeight", 1.0},
        {"intrinsicWeight", 0.0},
        {"epistemicWeight", 0.0},
        {"preferences", nlohmann::json::array({0.0f, 0.0f, 0.0f, 1.0f})}});
    ASSERT_TRUE(cfg.value("ok", false));

    // Train the latent transition model: the goal one-hot (index 3) shifts z along dim 3.
    nlohmann::json trans{{"z", nlohmann::json::array({0.0f, 0.0f, 0.0f, 1.0f})},
                         {"a", nlohmann::json::array({0.0f, 0.0f, 0.0f, 1.0f})},
                         {"zNext", nlohmann::json::array({0.0f, 0.0f, 0.0f, 2.0f})}};
    auto ingest = manager_->ingestAgiTransition(trans);
    ASSERT_TRUE(ingest.value("ok", false));

    nlohmann::json payload{{"sessionId", "agi_tool_test"}, {"userPrompt", "deploy satellite"}};
    nlohmann::json worldState = nlohmann::json::object();
    auto it = manager_->iterate(payload, worldState);
    ASSERT_TRUE(it.value("ok", false));

    auto agiPlan = it["result"].value("agiPlan", nlohmann::json::object());
    EXPECT_EQ(agiPlan.value("bestAction", std::string()), "goal_advance");

    auto execution = agiPlan.value("execution", nlohmann::json::object());
    ASSERT_TRUE(execution.value("ok", false)) << execution.value("error", "");
    auto goals = execution["result"].value("goals", nlohmann::json::array());
    ASSERT_FALSE(goals.empty());
    EXPECT_EQ(goals.back().get<std::string>(), "deploy satellite");
}
