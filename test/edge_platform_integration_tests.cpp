#include "../edge_platform.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

void requireTrue(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool jsonArrayContains(const json &values, const std::string &expected) {
    if (!values.is_array()) {
        return false;
    }
    for (const auto &item : values) {
        if (item.is_string() && item.get<std::string>() == expected) {
            return true;
        }
    }
    return false;
}

void configureRealManager(edge_platform::PlatformManager &manager) {
    const fs::path root = fs::current_path();
    const fs::path unifiedNetlist = root / "catastrophe" / "eext_netlist.json";
    const fs::path partition1 = root / "catastrophe" / "partitionByFunction_netlist1.json";
    const fs::path partition2 = root / "catastrophe" / "partitionByFunction_netlist2.json";
    const fs::path partition3 = root / "catastrophe" / "partitionByFunction_netlist3.json";
    const fs::path partition4 = root / "catastrophe" / "partitionByFunction_netlist4.json";
    const fs::path gerberConnectorMap = root / "catastrophe" / "gerber_catastrophe1_20260417_connector_map.json";
    requireTrue(fs::exists(unifiedNetlist), "missing catastrophe/eext_netlist.json");
    requireTrue(fs::exists(partition1), "missing catastrophe/partitionByFunction_netlist1.json");
    requireTrue(fs::exists(partition2), "missing catastrophe/partitionByFunction_netlist2.json");
    requireTrue(fs::exists(partition3), "missing catastrophe/partitionByFunction_netlist3.json");
    requireTrue(fs::exists(partition4), "missing catastrophe/partitionByFunction_netlist4.json");
    requireTrue(fs::exists(gerberConnectorMap), "missing catastrophe/gerber_catastrophe1_20260417_connector_map.json");

    edge_platform::RuntimeConfig config;
    config.baseDir = root;
    config.netlistFiles = {unifiedNetlist, partition1, partition2, partition3, partition4};
    config.gerberConnectorMap = gerberConnectorMap;

    manager.reconfigure(config);
    manager.refreshTopology();
}

void testRealNetlistDiscoversDirectNpuBoundary() {
    edge_platform::PlatformManager manager;
    configureRealManager(manager);
    const auto status = manager.status();
    const auto &result = status["result"];
    const auto &topology = result["topology"];

    requireTrue(topology.value("loaded", false), "real topology should load");
    requireTrue(topology["interfaces"].contains("npu-boundary"), "real topology should expose the gpio-attached npu boundary");
    requireTrue(topology["interfaces"].contains("weight-config"), "real topology should expose the GD32 weight-config sideband");
    requireTrue(topology["interfaces"].contains("gpio-bank"), "real topology should expose the GPIO bank");
    requireTrue(topology["npu"].value("available", false), "real topology should expose the gpio-attached npu route");
    requireTrue(topology["npu"].value("transport", std::string()) == "gpio-async", "real topology should advertise the direct gpio async transport");
    requireTrue(jsonArrayContains(topology["npu"]["controlSignals"], "GPIO2_WEIGHT_CFG_I2C_SDA"), "real topology should expose GPIO2 as the weight-config SDA line");
    requireTrue(jsonArrayContains(topology["npu"]["controlSignals"], "GPIO3_WEIGHT_CFG_I2C_SCL"), "real topology should expose GPIO3 as the weight-config SCL line");
    requireTrue(jsonArrayContains(topology["npu"]["controlSignals"], "GPIO19_WEIGHT_CFG_INT"), "real topology should expose GPIO19 as the weight-config interrupt line");
    requireTrue(topology["gerber"].value("loaded", false), "real topology should load the 20260417 gerber connector map");
    requireTrue(topology["gerber"]["connectors"].contains("J1"), "real topology should expose the Pi header connector map");
    requireTrue(!result.contains("brainstem"), "status should not expose any brainstem state");
    requireTrue(!result["metrics"].contains("peripheralDispatched"), "status metrics should not retain peripheral counters");
    requireTrue(status["result"]["config"]["hardware"].contains("preferredAsyncGpioDriver"), "real hardware status should expose the preferred async gpio driver");
}

void testLongAttentionRequestUsesHybridRoute() {
    edge_platform::PlatformManager manager;
    configureRealManager(manager);
    const auto plan = manager.planCompute(json{{"operation", "attention"},
                                               {"tokens", 2048},
                                               {"tensorBytes", 524288},
                                               {"postProcessOnCpu", true}});
    requireTrue(plan["result"]["route"].value("mode", std::string()) == "hybrid",
                "long attention requests should use hybrid routing on the real platform");

    const auto dispatched = manager.dispatchCompute(json{{"operation", "attention"},
                                                         {"tokens", 2048},
                                                         {"tensorBytes", 524288},
                                                         {"postProcessOnCpu", true}});
    requireTrue(dispatched.value("accepted", false), "hybrid compute request should be accepted");
}

void testNpuToggleFallsBackToCpu() {
    edge_platform::PlatformManager manager;
    configureRealManager(manager);
    manager.applyPatch(json{{"npuEnabled", false}});
    const auto cpuPlan = manager.planCompute(json{{"operation", "matmul"}, {"tokens", 512}, {"tensorBytes", 262144}});
    requireTrue(cpuPlan["result"]["route"].value("mode", std::string()) == "cpu",
                "disabling the npu at runtime should force cpu routing");
}

void testRealNetlistBuildsAsyncGpioProtocolForAnalogRequests() {
    edge_platform::PlatformManager manager;
    configureRealManager(manager);
    const auto plan = manager.planCompute(json{{"operation", "analog-matmul"},
                                               {"tokens", 256},
                                               {"tensorBytes", 131072},
                                               {"preferredBackend", "npu"}});
    requireTrue(plan["result"]["route"].value("transport", std::string()) == "gpio-async",
                "real page9 boundary should advertise the direct gpio async transport");
    requireTrue(plan["result"].contains("protocol"), "real page9 boundary should expose the async protocol envelope");
    requireTrue(plan["result"]["protocol"].value("executionModel", std::string()) == "async-overlapped",
                "real page9 boundary should prefer async-overlapped execution");
    requireTrue(plan["result"]["protocol"]["signals"]["txData"].size() == 12,
                "real page9 boundary should expose the twelve DAC write GPIOs");
    requireTrue(plan["result"]["protocol"]["signals"]["rxSelect"].empty(),
                "real page9 boundary should not invent an rx selector plane");
    requireTrue(plan["result"]["protocol"]["signals"]["rxData"].size() >= 10,
                "real page16/page17 boundary should expose comparator readback GPIOs");
    requireTrue(plan["result"]["protocol"]["signals"]["weightCfgI2c"].size() == 2,
                "real protocol should expose the GPIO2/GPIO3 weight-config sideband");
    requireTrue(plan["result"]["protocol"].contains("weightControl") && plan["result"]["protocol"]["weightControl"].value("available", false),
                "real protocol should expose the GD32 weight-control metadata");
    requireTrue(plan["result"]["protocol"]["weightControl"]["matrix"].value("totalTargets", 0) == 128,
                "real protocol should advertise the full 128-point weight matrix");
}

void testRealNetlistBuildsExecutableAsyncCyclePlan() {
    edge_platform::PlatformManager manager;
    configureRealManager(manager);
    const auto dispatched = manager.dispatchCompute(json{{"operation", "analog-matmul"},
                                                         {"tokens", 256},
                                                         {"tensorBytes", 131072},
                                                         {"preferredBackend", "npu"},
                                                         {"executeGpioAsync", true},
                                                         {"inputCycles",
                                                          json::array({json{{"txDataValue", 6}, {"mockReadbackValue", 3}}})}});
    requireTrue(dispatched.value("accepted", false), "real gpio async request should be accepted");
    requireTrue(dispatched["result"].contains("protocol"), "real gpio async request should keep protocol details");
    requireTrue(dispatched["result"]["protocol"].contains("cyclePlan") && dispatched["result"]["protocol"]["cyclePlan"].size() == 1,
                "real gpio async request should keep explicit GPIO input cycle metadata");
    requireTrue(dispatched["result"]["protocol"].contains("gpioPlan") && !dispatched["result"]["protocol"]["gpioPlan"].empty(),
                "real gpio async request should emit a gpioPlan alongside board-controller frames");
    requireTrue(dispatched["result"]["protocol"].value("ioCycleGranularity", std::string()) == "per-gpio-timestep",
                "real gpio async request should declare per-gpio-timestep I/O cycles");
    requireTrue(dispatched["result"]["protocol"]["cyclePlan"][0].value("txDataValue", -1) == 6,
                "real gpio async request should preserve explicit per-cycle txDataValue metadata");
    requireTrue(dispatched["result"].contains("execution") && dispatched["result"]["execution"].value("simulated", false),
                "Windows execution should run through the simulated async gpio driver");
}

void testRealNetlistWeightControlFaultKeepsComputePathAlive() {
    edge_platform::PlatformManager manager;
    configureRealManager(manager);
    const auto dispatched = manager.dispatchCompute(json{{"operation", "analog-matmul"},
                                                         {"tokens", 256},
                                                         {"tensorBytes", 131072},
                                                         {"preferredBackend", "npu"},
                                                         {"executeGpioAsync", true},
                                                         {"weightControllerFaultMode", "damaged"},
                                                         {"weightConfig",
                                                          json{{"targets", json::array({json{{"row", 0}, {"col", 15}, {"value", 1.0}}})}}},
                                                         {"inputCycles", json::array({json{{"txDataValue", 6}, {"mockReadbackValue", 3}}})}});
    requireTrue(dispatched.value("accepted", false), "faulted real-netlist weight-controller requests should still be accepted");
    requireTrue(dispatched["result"]["route"].value("weightControlAvailable", false),
                "real dispatch should still report an available weight-control plane");
    requireTrue(dispatched["result"]["route"].value("weightControlDegraded", false),
                "real dispatch should flag the injected weight-controller degradation");
    requireTrue(dispatched["result"]["protocol"]["weightControl"].value("effectiveMode", std::string()) == "bypass-retain-current-weights",
                "real dispatch should bypass sideband updates when the controller is faulted");
    requireTrue(dispatched["result"]["protocol"]["weightControl"].value("hotPathIndependent", false),
                "real dispatch should preserve the Pi hot-path independence guarantee");
}

} // namespace

int main() {
    try {
        testRealNetlistDiscoversDirectNpuBoundary();
        testLongAttentionRequestUsesHybridRoute();
        testNpuToggleFallsBackToCpu();
        testRealNetlistBuildsAsyncGpioProtocolForAnalogRequests();
        testRealNetlistBuildsExecutableAsyncCyclePlan();
        testRealNetlistWeightControlFaultKeepsComputePathAlive();
        std::cout << "edge_platform_integration_tests: ok" << std::endl;
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "edge_platform_integration_tests: failed: " << ex.what() << std::endl;
        return 1;
    }
}