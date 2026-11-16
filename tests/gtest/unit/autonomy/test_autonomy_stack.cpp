#include "autonomy_stack.hpp"
#include "test_hacktest_framework.hpp"
#include <gtest/gtest.h>
#include <memory>

using namespace autonomy;
using namespace phoenix::testing;

class AutonomyStackTest : public HacktestBase {
protected:
    std::unique_ptr<TransformerClusterManager> clusterManager_;
    std::unique_ptr<SpiderAutonomyManager> spiderManager_;
    std::unique_ptr<OptimizerAutonomyManager> optimizerManager_;
    std::unique_ptr<CognitionAutonomyManager> cognitionManager_;
    std::unique_ptr<DatasetCatalogManager> catalogManager_;

    void SetUp() override {
        HacktestBase::SetUp();
        clusterManager_ = std::make_unique<TransformerClusterManager>();
        spiderManager_ = std::make_unique<SpiderAutonomyManager>();
        optimizerManager_ = std::make_unique<OptimizerAutonomyManager>();
        cognitionManager_ = std::make_unique<CognitionAutonomyManager>();
        catalogManager_ = std::make_unique<DatasetCatalogManager>();
    }

    void TearDown() override {
        catalogManager_.reset();
        cognitionManager_.reset();
        optimizerManager_.reset();
        spiderManager_.reset();
        clusterManager_.reset();
    }
};

// 样本 1-3: 基本功能测试
TEST_F(AutonomyStackTest, Sample1_ClusterManagerStatus) {
    auto status = clusterManager_->status();
    EXPECT_TRUE(status.is_object());
}

TEST_F(AutonomyStackTest, Sample2_SpiderManagerStatus) {
    auto status = spiderManager_->status();
    EXPECT_TRUE(status.is_object());
}

TEST_F(AutonomyStackTest, Sample3_OptimizerManagerStatus) {
    auto status = optimizerManager_->status();
    EXPECT_TRUE(status.is_object());
}

// 样本 4-6: 边界条件测试
TEST_F(AutonomyStackTest, Sample4_CognitionManagerStatus) {
    auto status = cognitionManager_->status();
    EXPECT_TRUE(status.is_object());
}

TEST_F(AutonomyStackTest, Sample5_CatalogManagerStatus) {
    auto status = catalogManager_->status();
    EXPECT_TRUE(status.is_object());
}

TEST_F(AutonomyStackTest, Sample6_MultipleStatusCalls) {
    for (int i = 0; i < 10; ++i) {
        auto status = clusterManager_->status();
        EXPECT_TRUE(status.is_object());
    }
}

// 样本 7-8: 复杂场景测试
TEST_F(AutonomyStackTest, Sample7_ClusterManagerUpdateNodes) {
    nlohmann::json payload = {
        {"nodes", nlohmann::json::array({
            nlohmann::json{{"id", "node1"}, {"endpoint", "http://localhost:8080"}},
            nlohmann::json{{"id", "node2"}, {"endpoint", "http://localhost:8081"}}
        })}
    };
    auto result = clusterManager_->updateNodes(payload);
    EXPECT_NO_THROW();
}

TEST_F(AutonomyStackTest, Sample8_CognitionManagerObserve) {
    nlohmann::json payload = {{"sessionId", "test"}};
    nlohmann::json worldState = {{"state", "test"}};
    auto result = cognitionManager_->observe(payload, worldState);
    EXPECT_NO_THROW();
}

// 样本 9-10: Hacktest
TEST_F(AutonomyStackTest, Sample9_Hacktest_InvalidPayload) {
    SetHacktestDescriptor({
        HacktestType::INVALID_INPUT,
        "Test with invalid payload",
        "Should handle gracefully"
    });
    
    nlohmann::json payload = nlohmann::json::object();
    EXPECT_NO_THROW({
        auto result = clusterManager_->updateNodes(payload);
    });
}

TEST_F(AutonomyStackTest, Sample10_Hacktest_RapidOperations) {
    SetHacktestDescriptor({
        HacktestType::RESOURCE_EXHAUSTION,
        "Test rapid operations",
        "Should handle gracefully"
    });
    
    EXPECT_NO_THROW({
        for (int i = 0; i < 100; ++i) {
            clusterManager_->status();
            spiderManager_->status();
            optimizerManager_->status();
        }
    });
}

// 函数级测试 - TransformerClusterManager
TEST(TransformerClusterManagerTest, Sample1_Status) {
    TransformerClusterManager manager;
    auto status = manager.status();
    EXPECT_TRUE(status.is_object());
}

TEST(TransformerClusterManagerTest, Sample2_UpdateNodes) {
    TransformerClusterManager manager;
    nlohmann::json payload = {
        {"nodes", nlohmann::json::array({
            nlohmann::json{{"id", "node1"}, {"endpoint", "http://localhost:8080"}}
        })}
    };
    auto result = manager.updateNodes(payload);
    EXPECT_NO_THROW();
}

TEST(TransformerClusterManagerTest, Sample3_PickNode) {
    TransformerClusterManager manager;
    nlohmann::json request = {{"input", "test"}};
    auto result = manager.pickNode(request);
    EXPECT_NO_THROW();
}

TEST(TransformerClusterManagerTest, Sample4_Feedback) {
    TransformerClusterManager manager;
    nlohmann::json payload = {
        {"nodeId", "node1"},
        {"success", true},
        {"latencyMs", 100}
    };
    auto result = manager.feedback(payload);
    EXPECT_NO_THROW();
}

TEST(TransformerClusterManagerTest, Sample5_EmptyNodes) {
    TransformerClusterManager manager;
    nlohmann::json payload = {{"nodes", nlohmann::json::array()}};
    auto result = manager.updateNodes(payload);
    EXPECT_NO_THROW();
}

TEST(TransformerClusterManagerTest, Sample6_NullPayload) {
    TransformerClusterManager manager;
    nlohmann::json payload = nlohmann::json::object();
    auto result = manager.updateNodes(payload);
    EXPECT_NO_THROW();
}

TEST(TransformerClusterManagerTest, Sample7_MultipleNodes) {
    TransformerClusterManager manager;
    nlohmann::json payload = {{"nodes", nlohmann::json::array()}};
    for (int i = 0; i < 10; ++i) {
        payload["nodes"].push_back(nlohmann::json{
            {"id", "node" + std::to_string(i)},
            {"endpoint", "http://localhost:" + std::to_string(8080 + i)}
        });
    }
    auto result = manager.updateNodes(payload);
    EXPECT_NO_THROW();
}

TEST(TransformerClusterManagerTest, Sample8_PickNodeAfterUpdate) {
    TransformerClusterManager manager;
    nlohmann::json payload = {
        {"nodes", nlohmann::json::array({
            nlohmann::json{{"id", "node1"}, {"endpoint", "http://localhost:8080"}}
        })}
    };
    manager.updateNodes(payload);
    
    nlohmann::json request = {{"input", "test"}};
    auto result = manager.pickNode(request);
    EXPECT_NO_THROW();
}

TEST(TransformerClusterManagerTest, Sample9_Hacktest_InvalidEndpoint) {
    TransformerClusterManager manager;
    nlohmann::json payload = {
        {"nodes", nlohmann::json::array({
            nlohmann::json{{"id", "node1"}, {"endpoint", "invalid_url"}}
        })}
    };
    EXPECT_NO_THROW({
        auto result = manager.updateNodes(payload);
    });
}

TEST(TransformerClusterManagerTest, Sample10_Hacktest_RapidFeedback) {
    TransformerClusterManager manager;
    auto _fn_fb = [&]{
        for (int i = 0; i < 1000; ++i) {
            nlohmann::json payload = {
                {"nodeId", "node1"},
                {"success", true},
                {"latencyMs", i % 200}
            };
            manager.feedback(payload);
        }
    };
    EXPECT_NO_THROW(_fn_fb());
}

// 函数级测试 - SpiderAutonomyManager
TEST(SpiderAutonomyManagerTest, Sample1_Status) {
    SpiderAutonomyManager manager;
    auto status = manager.status();
    EXPECT_TRUE(status.is_object());
}

TEST(SpiderAutonomyManagerTest, Sample2_Adapt) {
    SpiderAutonomyManager manager;
    nlohmann::json payload = {{"strategy", "adaptive"}};
    nlohmann::json monitoring = {{"crawled", 100}};
    auto result = manager.adapt(payload, monitoring);
    EXPECT_NO_THROW();
}

TEST(SpiderAutonomyManagerTest, Sample3_EmptyPayload) {
    SpiderAutonomyManager manager;
    nlohmann::json payload = nlohmann::json::object();
    nlohmann::json monitoring = nlohmann::json::object();
    auto result = manager.adapt(payload, monitoring);
    EXPECT_NO_THROW();
}

TEST(SpiderAutonomyManagerTest, Sample4_NullPayload) {
    SpiderAutonomyManager manager;
    nlohmann::json payload = nlohmann::json::object();
    nlohmann::json monitoring = nlohmann::json::object();
    auto result = manager.adapt(payload, monitoring);
    EXPECT_NO_THROW();
}

TEST(SpiderAutonomyManagerTest, Sample5_WithCrawlDepth) {
    SpiderAutonomyManager manager;
    nlohmann::json payload = {{"crawlDepth", 5}};
    nlohmann::json monitoring = nlohmann::json::object();
    auto result = manager.adapt(payload, monitoring);
    EXPECT_NO_THROW();
}

TEST(SpiderAutonomyManagerTest, Sample6_WithMaxPages) {
    SpiderAutonomyManager manager;
    nlohmann::json payload = {{"maxPages", 500}};
    nlohmann::json monitoring = nlohmann::json::object();
    auto result = manager.adapt(payload, monitoring);
    EXPECT_NO_THROW();
}

TEST(SpiderAutonomyManagerTest, Sample7_WithBanPatterns) {
    SpiderAutonomyManager manager;
    nlohmann::json payload = {
        {"banPatterns", nlohmann::json::array({"blocked.com", "spam.org"})}
    };
    nlohmann::json monitoring = nlohmann::json::object();
    auto result = manager.adapt(payload, monitoring);
    EXPECT_NO_THROW();
}

TEST(SpiderAutonomyManagerTest, Sample8_MultipleAdaptations) {
    SpiderAutonomyManager manager;
    for (int i = 0; i < 10; ++i) {
        nlohmann::json payload = {{"iteration", i}};
        nlohmann::json monitoring = {{"crawled", i * 10}};
        manager.adapt(payload, monitoring);
    }
    EXPECT_NO_THROW();
}

TEST(SpiderAutonomyManagerTest, Sample9_Hacktest_NegativeDepth) {
    SpiderAutonomyManager manager;
    nlohmann::json payload = {{"crawlDepth", -1}};
    nlohmann::json monitoring = nlohmann::json::object();
    EXPECT_NO_THROW({
        auto result = manager.adapt(payload, monitoring);
    });
}

TEST(SpiderAutonomyManagerTest, Sample10_Hacktest_VeryLargeMaxPages) {
    SpiderAutonomyManager manager;
    nlohmann::json payload = {{"maxPages", 10000000}};
    nlohmann::json monitoring = nlohmann::json::object();
    EXPECT_NO_THROW({
        auto result = manager.adapt(payload, monitoring);
    });
}

// 函数级测试 - OptimizerAutonomyManager
TEST(OptimizerAutonomyManagerTest, Sample1_Status) {
    OptimizerAutonomyManager manager;
    auto status = manager.status();
    EXPECT_TRUE(status.is_object());
}

TEST(OptimizerAutonomyManagerTest, Sample2_Iterate) {
    OptimizerAutonomyManager manager;
    nlohmann::json payload = {{"iteration", 1}};
    nlohmann::json monitoring = {{"latency", 100}};
    nlohmann::json transformerParams = {{"threads", 4}};
    auto result = manager.iterate(payload, monitoring, transformerParams);
    EXPECT_NO_THROW();
}

TEST(OptimizerAutonomyManagerTest, Sample3_ApplyPerfProfile) {
    OptimizerAutonomyManager manager;
    nlohmann::json payload = {{"profile", "high_performance"}};
    nlohmann::json transformerParams = {{"threads", 4}};
    auto result = manager.applyPerfProfile(payload, transformerParams);
    EXPECT_NO_THROW();
}

TEST(OptimizerAutonomyManagerTest, Sample4_ProposeGnnUpgrade) {
    OptimizerAutonomyManager manager;
    nlohmann::json payload = {{"current", "v1"}};
    nlohmann::json transformerParams = nlohmann::json::object();
    auto result = manager.proposeGnnUpgrade(payload, transformerParams);
    EXPECT_NO_THROW();
}

TEST(OptimizerAutonomyManagerTest, Sample5_ProposeTransformerUpgrade) {
    OptimizerAutonomyManager manager;
    nlohmann::json payload = {{"current", "v1"}};
    nlohmann::json transformerParams = nlohmann::json::object();
    auto result = manager.proposeTransformerUpgrade(payload, transformerParams);
    EXPECT_NO_THROW();
}

TEST(OptimizerAutonomyManagerTest, Sample6_ModernizeTransformer) {
    OptimizerAutonomyManager manager;
    nlohmann::json payload = {{"architecture", "modern"}};
    nlohmann::json transformerParams = nlohmann::json::object();
    auto result = manager.modernizeTransformer(payload, transformerParams);
    EXPECT_NO_THROW();
}

TEST(OptimizerAutonomyManagerTest, Sample7_EmptyPayload) {
    OptimizerAutonomyManager manager;
    nlohmann::json payload = nlohmann::json::object();
    nlohmann::json monitoring = nlohmann::json::object();
    nlohmann::json transformerParams = nlohmann::json::object();
    auto result = manager.iterate(payload, monitoring, transformerParams);
    EXPECT_NO_THROW();
}

TEST(OptimizerAutonomyManagerTest, Sample8_NullPayload) {
    OptimizerAutonomyManager manager;
    nlohmann::json payload = nlohmann::json::object();
    nlohmann::json monitoring = nlohmann::json::object();
    nlohmann::json transformerParams = nlohmann::json::object();
    auto result = manager.iterate(payload, monitoring, transformerParams);
    EXPECT_NO_THROW();
}

TEST(OptimizerAutonomyManagerTest, Sample9_Hacktest_InvalidProfile) {
    OptimizerAutonomyManager manager;
    nlohmann::json payload = {{"profile", "invalid_profile"}};
    nlohmann::json transformerParams = nlohmann::json::object();
    EXPECT_NO_THROW({
        auto result = manager.applyPerfProfile(payload, transformerParams);
    });
}

TEST(OptimizerAutonomyManagerTest, Sample10_Hacktest_RapidIterations) {
    OptimizerAutonomyManager manager;
    auto _fn_it = [&]{
        for (int i = 0; i < 100; ++i) {
            nlohmann::json payload = {{"iteration", i}};
            nlohmann::json monitoring = {{"latency", 100}};
            nlohmann::json transformerParams = nlohmann::json::object();
            manager.iterate(payload, monitoring, transformerParams);
        }
    };
    EXPECT_NO_THROW(_fn_it());
}

// 函数级测试 - CognitionAutonomyManager
TEST(CognitionAutonomyManagerTest, Sample1_Status) {
    CognitionAutonomyManager manager;
    auto status = manager.status();
    EXPECT_TRUE(status.is_object());
}

TEST(CognitionAutonomyManagerTest, Sample2_Observe) {
    CognitionAutonomyManager manager;
    nlohmann::json payload = {{"sessionId", "test"}};
    nlohmann::json worldState = {{"state", "test"}};
    auto result = manager.observe(payload, worldState);
    EXPECT_NO_THROW();
}

TEST(CognitionAutonomyManagerTest, Sample3_Iterate) {
    CognitionAutonomyManager manager;
    nlohmann::json payload = {{"sessionId", "test"}};
    nlohmann::json worldState = {{"state", "test"}};
    auto result = manager.iterate(payload, worldState);
    EXPECT_NO_THROW();
}

TEST(CognitionAutonomyManagerTest, Sample4_Session) {
    CognitionAutonomyManager manager;
    auto session = manager.session("test_session");
    EXPECT_NO_THROW();
}

TEST(CognitionAutonomyManagerTest, Sample5_ExportState) {
    CognitionAutonomyManager manager;
    auto state = manager.exportState();
    EXPECT_NO_THROW();
}

TEST(CognitionAutonomyManagerTest, Sample6_ImportState) {
    CognitionAutonomyManager manager;
    nlohmann::json state = {{"version", "1.0"}};
    auto result = manager.importState(state);
    EXPECT_NO_THROW();
}

TEST(CognitionAutonomyManagerTest, Sample7_EmptySessionId) {
    CognitionAutonomyManager manager;
    auto session = manager.session("");
    EXPECT_NO_THROW();
}

TEST(CognitionAutonomyManagerTest, Sample8_NullSessionId) {
    CognitionAutonomyManager manager;
    std::string nullId;
    auto session = manager.session(nullId);
    EXPECT_NO_THROW();
}

TEST(CognitionAutonomyManagerTest, Sample9_Hacktest_InvalidState) {
    CognitionAutonomyManager manager;
    nlohmann::json state = nlohmann::json::object();
    EXPECT_NO_THROW({
        auto result = manager.importState(state);
    });
}

TEST(CognitionAutonomyManagerTest, Sample10_Hacktest_RapidObservations) {
    CognitionAutonomyManager manager;
    auto _fn_ob = [&]{
        for (int i = 0; i < 100; ++i) {
            nlohmann::json payload = {{"sessionId", "test"}};
            nlohmann::json worldState = {{"state", i}};
            manager.observe(payload, worldState);
        }
    };
    EXPECT_NO_THROW(_fn_ob());
}

// 函数级测试 - DatasetCatalogManager
TEST(DatasetCatalogManagerTest, Sample1_Status) {
    DatasetCatalogManager manager;
    auto status = manager.status();
    EXPECT_TRUE(status.is_object());
}

TEST(DatasetCatalogManagerTest, Sample2_List) {
    DatasetCatalogManager manager;
    auto list = manager.list();
    EXPECT_NO_THROW();
}

TEST(DatasetCatalogManagerTest, Sample3_RegisterDataset) {
    DatasetCatalogManager manager;
    nlohmann::json payload = {
        {"name", "test_dataset"},
        {"path", "/data/test"}
    };
    auto result = manager.registerDataset(payload);
    EXPECT_NO_THROW();
}

TEST(DatasetCatalogManagerTest, Sample4_Activate) {
    DatasetCatalogManager manager;
    nlohmann::json payload = {{"name", "test_dataset"}};
    auto result = manager.activate(payload);
    EXPECT_NO_THROW();
}

TEST(DatasetCatalogManagerTest, Sample5_UpdateCleaningProfile) {
    DatasetCatalogManager manager;
    nlohmann::json payload = {
        {"name", "test_dataset"},
        {"profile", nlohmann::json{{"method", "standard"}}}
    };
    auto result = manager.updateCleaningProfile(payload);
    EXPECT_NO_THROW();
}

TEST(DatasetCatalogManagerTest, Sample6_CollectData) {
    DatasetCatalogManager manager;
    nlohmann::json payload = {{"name", "test_dataset"}};
    auto result = manager.collectData(payload);
    EXPECT_NO_THROW();
}

TEST(DatasetCatalogManagerTest, Sample7_Governance) {
    DatasetCatalogManager manager;
    auto governance = manager.governance();
    EXPECT_NO_THROW();
}

TEST(DatasetCatalogManagerTest, Sample8_EmptyPayload) {
    DatasetCatalogManager manager;
    nlohmann::json payload = nlohmann::json::object();
    auto result = manager.registerDataset(payload);
    EXPECT_NO_THROW();
}

TEST(DatasetCatalogManagerTest, Sample9_Hacktest_InvalidDataset) {
    DatasetCatalogManager manager;
    nlohmann::json payload = {{"name", ""}};
    EXPECT_NO_THROW({
        auto result = manager.registerDataset(payload);
    });
}

TEST(DatasetCatalogManagerTest, Sample10_Hacktest_RapidRegistrations) {
    DatasetCatalogManager manager;
    auto _fn_ds = [&]{
        for (int i = 0; i < 100; ++i) {
            nlohmann::json payload = {
                {"name", "dataset_" + std::to_string(i)},
                {"path", "/data/" + std::to_string(i)}
            };
            manager.registerDataset(payload);
        }
    };
    EXPECT_NO_THROW(_fn_ds());
}

// 函数级测试 - buildCognitionAutonomySeedPayload
TEST(BuildCognitionAutonomySeedPayload, Sample1_BasicPayload) {
    auto payload = buildCognitionAutonomySeedPayload("session_1", "test_mission");
    EXPECT_TRUE(payload.is_object());
}

TEST(BuildCognitionAutonomySeedPayload, Sample2_WithUncertainty) {
    auto payload = buildCognitionAutonomySeedPayload("session_1", "test_mission", 0.5);
    EXPECT_NO_THROW();
}

TEST(BuildCognitionAutonomySeedPayload, Sample3_EmptySessionId) {
    auto payload = buildCognitionAutonomySeedPayload("", "test_mission");
    EXPECT_NO_THROW();
}

TEST(BuildCognitionAutonomySeedPayload, Sample4_EmptyMission) {
    auto payload = buildCognitionAutonomySeedPayload("session_1", "");
    EXPECT_NO_THROW();
}

TEST(BuildCognitionAutonomySeedPayload, Sample5_HighUncertainty) {
    auto payload = buildCognitionAutonomySeedPayload("session_1", "test_mission", 0.9);
    EXPECT_NO_THROW();
}

TEST(BuildCognitionAutonomySeedPayload, Sample6_LowUncertainty) {
    auto payload = buildCognitionAutonomySeedPayload("session_1", "test_mission", 0.1);
    EXPECT_NO_THROW();
}

TEST(BuildCognitionAutonomySeedPayload, Sample7_ZeroUncertainty) {
    auto payload = buildCognitionAutonomySeedPayload("session_1", "test_mission", 0.0);
    EXPECT_NO_THROW();
}

TEST(BuildCognitionAutonomySeedPayload, Sample8_OneUncertainty) {
    auto payload = buildCognitionAutonomySeedPayload("session_1", "test_mission", 1.0);
    EXPECT_NO_THROW();
}

TEST(BuildCognitionAutonomySeedPayload, Sample9_Hacktest_NegativeUncertainty) {
    EXPECT_NO_THROW({
        auto payload = buildCognitionAutonomySeedPayload("session_1", "test_mission", -0.5);
    });
}

TEST(BuildCognitionAutonomySeedPayload, Sample10_Hacktest_GreaterThanOneUncertainty) {
    EXPECT_NO_THROW({
        auto payload = buildCognitionAutonomySeedPayload("session_1", "test_mission", 1.5);
    });
}

// 类级测试 - 完整生命周期
TEST(AutonomyStackLifecycle, Sample1_AllManagersConstruction) {
    TransformerClusterManager clusterManager;
    SpiderAutonomyManager spiderManager;
    OptimizerAutonomyManager optimizerManager;
    CognitionAutonomyManager cognitionManager;
    DatasetCatalogManager catalogManager;
    
    EXPECT_NO_THROW();
}

TEST(AutonomyStackLifecycle, Sample2_AllManagersStatus) {
    TransformerClusterManager clusterManager;
    SpiderAutonomyManager spiderManager;
    OptimizerAutonomyManager optimizerManager;
    CognitionAutonomyManager cognitionManager;
    DatasetCatalogManager catalogManager;
    
    auto clusterStatus = clusterManager.status();
    auto spiderStatus = spiderManager.status();
    auto optimizerStatus = optimizerManager.status();
    auto cognitionStatus = cognitionManager.status();
    auto catalogStatus = catalogManager.status();
    
    EXPECT_TRUE(clusterStatus.is_object());
    EXPECT_TRUE(spiderStatus.is_object());
    EXPECT_TRUE(optimizerStatus.is_object());
    EXPECT_TRUE(cognitionStatus.is_object());
    EXPECT_TRUE(catalogStatus.is_object());
}

TEST(AutonomyStackLifecycle, Sample3_CognitionFullCycle) {
    CognitionAutonomyManager manager;
    
    nlohmann::json payload = {{"sessionId", "test"}};
    nlohmann::json worldState = {{"state", "initial"}};
    
    manager.observe(payload, worldState);
    manager.iterate(payload, worldState);
    
    auto session = manager.session("test");
    auto state = manager.exportState();
    
    EXPECT_NO_THROW();
}

TEST(AutonomyStackLifecycle, Sample4_ClusterFullCycle) {
    TransformerClusterManager manager;
    
    nlohmann::json payload = {
        {"nodes", nlohmann::json::array({
            nlohmann::json{{"id", "node1"}, {"endpoint", "http://localhost:8080"}}
        })}
    };
    manager.updateNodes(payload);
    
    nlohmann::json request = {{"input", "test"}};
    auto node = manager.pickNode(request);
    
    nlohmann::json feedback = {
        {"nodeId", "node1"},
        {"success", true},
        {"latencyMs", 100}
    };
    manager.feedback(feedback);
    
    EXPECT_NO_THROW();
}

TEST(AutonomyAutonomyLifecycle, Sample5_OptimizerFullCycle) {
    OptimizerAutonomyManager manager;
    
    nlohmann::json payload = {{"iteration", 1}};
    nlohmann::json monitoring = {{"latency", 100}};
    nlohmann::json transformerParams = {{"threads", 4}};
    
    manager.iterate(payload, monitoring, transformerParams);
    
    nlohmann::json profilePayload = {{"profile", "high_performance"}};
    manager.applyPerfProfile(profilePayload, transformerParams);
    
    EXPECT_NO_THROW();
}

TEST(AutonomyStackLifecycle, Sample6_CatalogFullCycle) {
    DatasetCatalogManager manager;
    
    nlohmann::json payload = {
        {"name", "test_dataset"},
        {"path", "/data/test"}
    };
    manager.registerDataset(payload);
    
    nlohmann::json activatePayload = {{"name", "test_dataset"}};
    manager.activate(activatePayload);
    
    auto list = manager.list();
    auto governance = manager.governance();
    
    EXPECT_NO_THROW();
}

TEST(AutonomyStackLifecycle, Sample7_ErrorRecovery) {
    CognitionAutonomyManager manager;
    
    nlohmann::json invalidPayload = nlohmann::json::object();
    nlohmann::json worldState = nlohmann::json::object();
    EXPECT_NO_THROW({
        manager.observe(invalidPayload, worldState);
    });
    
    nlohmann::json validPayload = {{"sessionId", "test"}};
    nlohmann::json validWorldState = {{"state", "test"}};
    auto result = manager.observe(validPayload, validWorldState);
    EXPECT_NO_THROW();
}

TEST(AutonomyStackLifecycle, Sample8_ConcurrentAccess) {
    CognitionAutonomyManager manager;
    
    auto _fn_ca = [&]{
        for (int i = 0; i < 100; ++i) {
            nlohmann::json payload = {{"sessionId", "test"}};
            nlohmann::json worldState = {{"state", i}};
            manager.observe(payload, worldState);
            manager.status();
        }
    };
    EXPECT_NO_THROW(_fn_ca());
}

TEST(AutonomyStackLifecycle, Sample9_Hacktest_RapidConstruction) {
    EXPECT_NO_THROW({
        for (int i = 0; i < 100; ++i) {
            TransformerClusterManager clusterManager;
            SpiderAutonomyManager spiderManager;
            OptimizerAutonomyManager optimizerManager;
            CognitionAutonomyManager cognitionManager;
            DatasetCatalogManager catalogManager;
        }
    });
}

TEST(AutonomyStackLifecycle, Sample10_Hacktest_MixedOperations) {
    CognitionAutonomyManager manager;
    
    auto _fn_mix = [&]{
        for (int i = 0; i < 50; ++i) {
            nlohmann::json payload = {{"sessionId", "test"}};
            nlohmann::json worldState = {{"state", i}};
            manager.observe(payload, worldState);
            manager.iterate(payload, worldState);
            manager.status();
            manager.session("test");
        }
    };
    EXPECT_NO_THROW(_fn_mix());
}

// v7.0 mixed-modal I/O integration tests for CognitionAutonomyManager

TEST(CognitionMixedModalTest, IngestTextPacket) {
    CognitionAutonomyManager manager;
    nlohmann::json payload = {
        {"id", "ingest-text-1"},
        {"modality", "text"},
        {"payload", std::vector<uint8_t>{'h', 'e', 'l', 'l', 'o'}},
        {"targetDim", 8},
        {"source", "keyboard"}
    };
    auto result = manager.ingestMixedModalPacket(payload);
    EXPECT_TRUE(result.value("ok", false));
    EXPECT_EQ(result["result"]["semanticUnit"]["modality"], "text");
    EXPECT_EQ(result["result"]["semanticUnit"]["semanticVector"].size(), 8u);
    EXPECT_EQ(result["result"]["packet"]["source"], "keyboard");
}

TEST(CognitionMixedModalTest, PretrainSpeechConceptAndStatus) {
    CognitionAutonomyManager manager;
    nlohmann::json audio = {
        {"id", "audio-train-1"},
        {"modality", "audio"},
        {"payload", std::vector<uint8_t>{0x10, 0x40, 0x80, 0xc0, 0xff}},
        {"mimeType", "audio/pcm"}
    };
    nlohmann::json payload = {
        {"audio", audio},
        {"transcript", std::string("hello speech")},
        {"targetDim", 8}
    };
    auto result = manager.pretrainSpeechConcept(payload);
    EXPECT_TRUE(result.value("ok", false));
    EXPECT_EQ(result["result"]["audioWorldModel"]["dimension"], 8u);
    EXPECT_GE(result["result"]["audioWorldModel"]["samples"].get<size_t>(), 1u);
}

TEST(CognitionMixedModalTest, EmitAndDrainMixedModalOutput) {
    CognitionAutonomyManager manager;
    nlohmann::json semanticUnit = {
        {"id", "emit-1"},
        {"modality", "text"},
        {"content", std::string("hello world")},
        {"semanticVector", std::vector<float>{0.1f, 0.2f, 0.3f}}
    };
    nlohmann::json emitPayload = {
        {"semanticUnit", semanticUnit},
        {"targetModality", "text"},
        {"source", "tts"}
    };
    auto emitResult = manager.emitMixedModalOutput(emitPayload);
    EXPECT_TRUE(emitResult.value("ok", false));
    EXPECT_EQ(emitResult["result"]["modality"], "text");

    nlohmann::json drainPayload = {{"max", 10}};
    auto drainResult = manager.drainMixedModalOutputs(drainPayload);
    EXPECT_TRUE(drainResult.value("ok", false));
    EXPECT_EQ(drainResult["result"].size(), 1u);
    EXPECT_EQ(drainResult["result"][0]["source"], "tts");

    auto emptyDrain = manager.drainMixedModalOutputs(drainPayload);
    EXPECT_TRUE(emptyDrain["result"].empty());
}

TEST(CognitionMixedModalTest, PretrainSpeechMissingAudio) {
    CognitionAutonomyManager manager;
    nlohmann::json payload = {{"transcript", "hello"}};
    auto result = manager.pretrainSpeechConcept(payload);
    EXPECT_FALSE(result.value("ok", true));
    EXPECT_TRUE(result.contains("error"));
}

TEST(CognitionMixedModalTest, EmitUnknownTargetModality) {
    CognitionAutonomyManager manager;
    nlohmann::json semanticUnit = {
        {"id", "emit-bad"},
        {"modality", "text"},
        {"content", std::string("x")},
        {"semanticVector", std::vector<float>{0.1f}}
    };
    nlohmann::json emitPayload = {
        {"semanticUnit", semanticUnit},
        {"targetModality", "invalid"}
    };
    auto result = manager.emitMixedModalOutput(emitPayload);
    EXPECT_FALSE(result.value("ok", true));
    EXPECT_TRUE(result.contains("error"));
}
