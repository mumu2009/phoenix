/* GTest unit tests for Phoenix v8.6 model summoning (summon_registry +
   model-carrying helper boxes).

   Hard constraint under test: the framework adapts to the model - no prompt
   teaching, no fine-tuning.  Summon is a planner action (category
   "summon"); the endpoint registry is config-driven; an empty model field
   always means the base model. */

#include "autonomy_stack.hpp"
#include "mission_lifecycle.hpp"
#include "summon_registry.hpp"
#include "test_hacktest_framework.hpp"
#include <gtest/gtest.h>

using namespace autonomy;
using namespace phoenix::mission;
using phoenix::summon::SummonRegistry;
using json = nlohmann::json;

namespace {

json twoModelConfig() {
    return json::array({
        json{{"name", "qwen3-30b"},
             {"kind", "ollama"},
             {"url", "http://127.0.0.1:11434"},
             {"model", "qwen3:30b"},
             {"timeoutMs", 120000},
             {"maxTokens", 256},
             {"description", "big MoE whole model"}},
        json{{"name", "qwen3-30b_e3"},
             {"kind", "llamacpp"},
             {"url", "http://127.0.0.1:8093"},
             {"timeoutMs", 60000},
             {"maxTokens", 128},
             {"description", "forced expert 3"}},
    });
}

class SummonRegistryTest : public ::testing::Test {
 protected:
    void SetUp() override {
        /* Singleton: always reset to a known state. */
        SummonRegistry::instance().loadFromConfig(false, json::array());
    }
    void TearDown() override {
        SummonRegistry::instance().loadFromConfig(false, json::array());
    }
};

TEST_F(SummonRegistryTest, DisabledByDefaultRejectsEverything) {
    EXPECT_FALSE(SummonRegistry::instance().enabled());
    EXPECT_EQ(SummonRegistry::instance().find("qwen3-30b"), nullptr);
}

TEST_F(SummonRegistryTest, LoadsValidEntriesAndSkipsInvalid) {
    json models = twoModelConfig();
    /* invalid: empty url, unknown kind, missing name */
    models.push_back(json{{"name", "bad1"}, {"kind", "ollama"}});
    models.push_back(json{{"name", "bad2"}, {"kind", "tensorrt"},
                          {"url", "http://x"}});
    models.push_back(json{{"kind", "ollama"}, {"url", "http://x"}});
    const auto rep = SummonRegistry::instance().loadFromConfig(true, models);
    EXPECT_TRUE(rep.value("enabled", false));
    EXPECT_EQ(rep["loaded"].size(), 2u);
    EXPECT_EQ(rep["skipped"].size(), 3u);

    const auto *whole = SummonRegistry::instance().find("qwen3-30b");
    ASSERT_NE(whole, nullptr);
    EXPECT_EQ(whole->kind, "ollama");
    EXPECT_EQ(whole->model, "qwen3:30b");
    const auto *expert = SummonRegistry::instance().find("qwen3-30b_e3");
    ASSERT_NE(expert, nullptr);
    EXPECT_EQ(expert->kind, "llamacpp");
    EXPECT_EQ(expert->url, "http://127.0.0.1:8093");
    EXPECT_EQ(SummonRegistry::instance().find("nonexistent"), nullptr);
}

TEST_F(SummonRegistryTest, ScopeOverrideRoundtrip) {
    SummonRegistry::instance().loadFromConfig(true, twoModelConfig());
    auto &reg = SummonRegistry::instance();
    reg.setScopeModel("m1/children/child-1-0", "qwen3-30b_e3");
    EXPECT_EQ(reg.modelForScope("m1/children/child-1-0"), "qwen3-30b_e3");
    /* unknown scope -> empty (base model path) */
    EXPECT_TRUE(reg.modelForScope("m1").empty());
    EXPECT_TRUE(reg.modelForScope("m1/children/child-1-1").empty());
    /* empty model is never stored */
    reg.setScopeModel("m2/children/child-1-0", "");
    EXPECT_TRUE(reg.modelForScope("m2/children/child-1-0").empty());
    reg.clearScopeModel("m1/children/child-1-0");
    EXPECT_TRUE(reg.modelForScope("m1/children/child-1-0").empty());
}

TEST_F(SummonRegistryTest, DisabledRegistryHidesOverrides) {
    auto &reg = SummonRegistry::instance();
    reg.loadFromConfig(true, twoModelConfig());
    reg.setScopeModel("m1/children/child-1-0", "qwen3-30b_e3");
    /* Flipping enabled off must dark the whole feature. */
    reg.loadFromConfig(false, json::array());
    EXPECT_TRUE(reg.modelForScope("m1/children/child-1-0").empty());
    EXPECT_EQ(reg.find("qwen3-30b_e3"), nullptr);
}

/* ---- MissionChild model field ---- */

TEST(SummonMissionChildTest, ModelFieldPersistsThroughJson) {
    MissionLifecycle lc;
    Mission m;
    m.id = "m-summon";
    m.goal = "write the geology appendix";
    lc.assign(m);
    MissionGenome parent;
    const auto child = lc.recordChild(parent, 0.0f, "draft the basalt section",
                                      std::string(), "qwen3-30b_e3");
    EXPECT_EQ(child.model, "qwen3-30b_e3");

    /* roundtrip through export/import */
    MissionLifecycle restored;
    restored.fromJson(lc.toJson());
    const auto kids = restored.children();
    ASSERT_EQ(kids.size(), 1u);
    EXPECT_EQ(kids[0].model, "qwen3-30b_e3");
    EXPECT_EQ(kids[0].goal, "draft the basalt section");
}

TEST(SummonMissionChildTest, EmptyModelMeansBaseAndStaysEmpty) {
    MissionLifecycle lc;
    Mission m;
    m.id = "m-base";
    m.goal = "plain mission";
    lc.assign(m);
    MissionGenome parent;
    const auto child = lc.recordChild(parent, 0.0f, "help out");
    EXPECT_TRUE(child.model.empty());
    MissionLifecycle restored;
    restored.fromJson(lc.toJson());
    ASSERT_EQ(restored.children().size(), 1u);
    EXPECT_TRUE(restored.children()[0].model.empty());
}

/* ---- manager-level summon action ---- */

class SummonActionTest : public ::testing::Test {
 protected:
    void SetUp() override {
        mgr_ = std::make_unique<CognitionAutonomyManager>();
        SummonRegistry::instance().loadFromConfig(true, twoModelConfig());
        mgr_->assignMission(json{{"enabled", true},
                                 {"goal", "summon test mission"},
                                 {"maxReplicas", 8},
                                 {"painGainPerSec", 0.01},
                                 {"maxPain", 1.0}});
    }
    void TearDown() override {
        mgr_.reset();
        SummonRegistry::instance().loadFromConfig(false, json::array());
    }
    void registerSummon(const std::string &model) {
        mgr_->registerAgiAction(json{{"name", "summon." + model},
                                     {"category", "summon"},
                                     {"addonType", ""},
                                     {"description", "test summon"},
                                     {"embedding", json::array({0, 0, 0, 0, 0, 1})}});
    }
    std::unique_ptr<CognitionAutonomyManager> mgr_;
};

TEST_F(SummonActionTest, SummonActionSpawnsBoxOnExternalModel) {
    registerSummon("qwen3-30b_e3");
    auto res = mgr_->executeAgiActionByName(
        "summon.qwen3-30b_e3", json{{"userPrompt", "draft the basalt section"}});
    ASSERT_TRUE(res.value("ok", false)) << res.dump();
    EXPECT_EQ(res["result"].value("model", std::string()), "qwen3-30b_e3");
    const std::string boxId = res["result"].value("boxId", std::string());
    ASSERT_FALSE(boxId.empty());
    /* the scope override must be registered for the deliberator */
    const auto mst = mgr_->missionStatus();
    const std::string mid =
        mst["result"].value("defaultMissionId", std::string());
    ASSERT_FALSE(mid.empty());
    EXPECT_EQ(SummonRegistry::instance().modelForScope(
                  mid + "/children/" + boxId),
              "qwen3-30b_e3");
    /* child record carries the model */
    bool found = false;
    for (const auto &c : mst["result"]["stats"]["children"]) {
        if (c.value("id", std::string()) == boxId) {
            found = true;
            EXPECT_EQ(c.value("model", std::string()), "qwen3-30b_e3");
        }
    }
    EXPECT_TRUE(found) << "child record must carry the model field";
}

TEST_F(SummonActionTest, SummonRejectsUnregisteredModel) {
    /* action registered but the endpoint is NOT in summon.models */
    registerSummon("ghost-model");
    auto res = mgr_->executeAgiActionByName(
        "summon.ghost-model", json{{"userPrompt", "hello"}});
    EXPECT_FALSE(res.value("ok", true));
    EXPECT_NE(res.value("error", std::string()).find("not registered"),
              std::string::npos);
    /* no child may have been spawned */
    const auto mst = mgr_->missionStatus();
    EXPECT_EQ(mst["result"]["stats"]["children"].size(), 0u);
}

TEST_F(SummonActionTest, SpawnMissionChildAcceptsModelField) {
    auto res = mgr_->spawnMissionChild(
        json{{"subgoal", "expert sub-task"}, {"model", "qwen3-30b"}});
    ASSERT_TRUE(res.value("ok", false)) << res.dump();
    EXPECT_EQ(res["result"].value("model", std::string()), "qwen3-30b");
}

TEST_F(SummonActionTest, SpawnMissionChildRejectsUnknownModel) {
    auto res = mgr_->spawnMissionChild(
        json{{"subgoal", "x"}, {"model", "does-not-exist"}});
    EXPECT_FALSE(res.value("ok", true));
    EXPECT_NE(res.value("error", std::string()).find("unknown summon model"),
              std::string::npos);
}

TEST_F(SummonActionTest, SpawnMissionChildEmptyModelIsBase) {
    auto res = mgr_->spawnMissionChild(json{{"subgoal", "plain helper"}});
    ASSERT_TRUE(res.value("ok", false)) << res.dump();
    EXPECT_TRUE(res["result"].value("model", std::string()).empty());
    /* no scope override for a base-model box */
    const std::string boxId = res["result"].value("id", std::string());
    const auto mst = mgr_->missionStatus();
    const std::string mid =
        mst["result"].value("defaultMissionId", std::string());
    EXPECT_TRUE(SummonRegistry::instance()
                    .modelForScope(mid + "/children/" + boxId)
                    .empty());
}

TEST_F(SummonActionTest, ReplicateStillWorksWithoutModel) {
    /* regression: the shared replicate/summon branch must keep the old
       behaviour for plain replicate */
    auto res = mgr_->executeAgiActionByName(
        "replicate", json{{"userPrompt", "classic box"}});
    ASSERT_TRUE(res.value("ok", false)) << res.dump();
    EXPECT_TRUE(res["result"].value("model", std::string()).empty());
}

}  // namespace
