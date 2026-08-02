#include "../edge_platform.hpp"

#include <filesystem>
#include <fstream>
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

template <typename AddPinFn>
void forEachSyntheticPiHeaderPin(AddPinFn addPin) {
    addPin("1", "3V3");
    addPin("3", "GPIO2_WEIGHT_CFG_I2C_SDA");
    addPin("5", "GPIO3_WEIGHT_CFG_I2C_SCL");
    addPin("6", "GND");
    addPin("7", "GPIO4_SLICE_SUM0_CMP");
    addPin("8", "GPIO14_CARRG_CMP");
    addPin("10", "GPIO15_CARRH_CMP");
    addPin("11", "GPIO17_DAC_IN4");
    addPin("12", "GPIO18_DAC_IN5");
    addPin("13", "GPIO27_DAC_IN6");
    addPin("15", "GPIO22_DAC_IN7");
    addPin("16", "GPIO23_DAC_IN8");
    addPin("17", "3V3");
    addPin("18", "GPIO24_DAC_IN9");
    addPin("19", "GPIO10_DAC_IN0");
    addPin("20", "GND");
    addPin("21", "GPIO9_DAC_IN10");
    addPin("22", "GPIO25_DAC_IN11");
    addPin("23", "GPIO11_DAC_IN1");
    addPin("24", "GPIO8_DAC_IN2");
    addPin("25", "GND");
    addPin("26", "GPIO7_DAC_IN3");
    addPin("29", "GPIO5_CARRA_CMP");
    addPin("30", "GND");
    addPin("31", "GPIO6_NPU_SAMPLE_SYNC");
    addPin("32", "GPIO12_CARRB_CMP");
    addPin("33", "GPIO13_CARRC_CMP");
    addPin("34", "GND");
    addPin("35", "GPIO19_WEIGHT_CFG_INT");
    addPin("36", "GPIO16_CARRD_CMP");
    addPin("37", "GPIO26_SLICE_SUM1_CMP");
    addPin("38", "GPIO20_CARRE_CMP");
    addPin("39", "GND");
    addPin("40", "GPIO21_CARRF_CMP");
}

fs::path writeSyntheticNetlist(const std::string &name, const json &doc) {
    const fs::path root = fs::current_path() / "build" / "edge_platform_testdata";
    fs::create_directories(root);
    const fs::path path = root / name;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to create synthetic netlist file");
    }
    out << doc.dump(2);
    return path;
}

fs::path writeSyntheticConnectorMap(const std::string &name) {
    json pins = json::object();
    auto addPin = [&](const std::string &pin, const std::string &net) {
        pins[pin] = json{{"logicalNet", net}, {"boardNet", net}, {"exactMatch", true}};
    };
    forEachSyntheticPiHeaderPin(addPin);

    const json doc{{"gerberVersion", "synthetic"},
                   {"sourceGerber", "synthetic"},
                   {"sourceNetlist", "synthetic"},
                   {"connectors", json{{"J1", json{{"role", "rpi-header"}, {"pins", pins}}}}}};
    return writeSyntheticNetlist(name, doc);
}

void configureSyntheticManager(edge_platform::PlatformManager &manager) {
    json pins = json::object();
    auto addPin = [&](const std::string &pin, const std::string &signal) {
        pins[pin] = signal;
    };
    forEachSyntheticPiHeaderPin(addPin);

    const json doc{{"j1", json{{"props", json{{"Designator", "J1"}, {"value", "Synthetic Pi Header"}}}, {"pins", pins}}}};

    edge_platform::RuntimeConfig config;
    config.baseDir = fs::current_path();
    config.netlistFiles = {writeSyntheticNetlist("eext_netlist_unit_synthetic.json", doc)};
    config.gerberConnectorMap = writeSyntheticConnectorMap("gerber_connector_map_unit_synthetic.json");
    config.maxComputeInflight = 4;
    manager.reconfigure(config);
    manager.refreshTopology();
}

void testSyntheticNetlistBuildsExpectedInterfaces() {
    edge_platform::PlatformManager manager;
    configureSyntheticManager(manager);
    const auto status = manager.status();
    const auto &result = status["result"];
    const auto &topology = result["topology"];
    const auto &hardware = result["config"]["hardware"];

    requireTrue(topology.value("loaded", false), "synthetic topology should load");
    requireTrue(topology["interfaces"].contains("npu-boundary"), "synthetic topology should expose the NPU GPIO boundary");
    requireTrue(topology["interfaces"].contains("weight-config"), "synthetic topology should expose the weight-config sideband");
    requireTrue(topology["interfaces"].contains("gpio-bank"), "synthetic topology should expose the GPIO bank");
    requireTrue(!topology["interfaces"].contains("spi0"), "synthetic topology should not fabricate a spi fallback plane");
    requireTrue(topology["npu"].value("available", false), "synthetic topology should expose the NPU control plane");
    requireTrue(topology["npu"].value("transport", std::string()) == "gpio-async", "synthetic topology should advertise the direct gpio async transport");
    requireTrue(topology["npu"]["controlSignals"].size() >= 24,
                "synthetic topology should expose both the transport boundary and the weight-config sideband");
    requireTrue(topology["gerber"].value("loaded", false), "synthetic topology should load the synthetic connector map");
    requireTrue(topology["gerber"]["connectors"].contains("J1"), "synthetic topology should expose the Pi header connector map");
    requireTrue(!result.contains("brainstem"), "status should not expose any brainstem state");
    requireTrue(!result["metrics"].contains("peripheralDispatched"), "status metrics should not retain peripheral counters");
    requireTrue(hardware.contains("preferredAsyncGpioDriver"), "hardware status should expose the preferred async gpio driver");
}

void testDenseComputePrefersNpuRoute() {
    edge_platform::PlatformManager manager;
    configureSyntheticManager(manager);
    const auto plan = manager.planCompute(json{{"operation", "matmul"}, {"tokens", 512}, {"tensorBytes", 262144}});
    const std::string mode = plan["result"]["route"].value("mode", std::string());
    requireTrue(mode == "npu" || mode == "hybrid", "dense compute should prefer npu or hybrid routing");
}

void testControlHeavyRequestStaysOnCpu() {
    edge_platform::PlatformManager manager;
    configureSyntheticManager(manager);
    const auto plan = manager.planCompute(json{{"operation", "tool-routing"},
                                               {"controlHeavy", true},
                                               {"tokens", 32},
                                               {"tensorBytes", 4096},
                                               {"latencyBudgetMs", 20}});
    requireTrue(plan["result"]["route"].value("mode", std::string()) == "cpu",
                "latency sensitive control-heavy requests should stay on cpu");
}

void testAsyncProtocolPrefersDirectGpioBoundaryWhenAvailable() {
    edge_platform::PlatformManager manager;
    configureSyntheticManager(manager);
    const auto plan = manager.planCompute(json{{"operation", "analog-matmul"},
                                               {"tokens", 256},
                                               {"tensorBytes", 131072},
                                               {"preferredBackend", "npu"}});
    const auto &result = plan["result"];
    requireTrue(result["route"].value("transport", std::string()) == "gpio-async",
                "direct gpio async boundary should be preferred when the required signals exist");
    requireTrue(result.contains("protocol"), "npu route should expose a protocol envelope");
    requireTrue(result["protocol"].value("executionModel", std::string()) == "async-overlapped",
                "protocol should switch to async-overlapped execution when tx and rx paths are independent");
    requireTrue(result["protocol"]["signals"]["txData"].size() == 12,
                "protocol should expose the twelve DAC write lines");
    requireTrue(result["protocol"]["signals"]["rxSelect"].empty(),
                "protocol should not invent a receive selector plane that is absent from the netlist");
    requireTrue(result["protocol"]["signals"]["rxData"].size() >= 10,
                "protocol should expose the comparator readback group on the rx side");
    requireTrue(result["protocol"]["signals"]["weightCfgI2c"].size() == 2,
                "protocol should expose the GPIO2/GPIO3 weight-config I2C sideband");
    requireTrue(result["protocol"]["signals"].value("weightCfgInt", std::string()) == "GPIO19_WEIGHT_CFG_INT",
                "protocol should expose the GPIO19 weight-config interrupt");
    requireTrue(result["protocol"].contains("weightControl") && result["protocol"]["weightControl"].value("available", false),
                "protocol should expose the onboard weight-control sideband metadata");
    requireTrue(result["protocol"]["weightControl"]["matrix"].value("totalTargets", 0) == 128,
                "protocol should advertise the 16x8 weight matrix coverage");
    requireTrue(result["protocol"]["transportFrames"].size() > 3,
                "protocol should auto-build a frame plan for the board-side controller");
    requireTrue(result["protocol"].contains("gpioPlan") && result["protocol"]["gpioPlan"].is_array() && !result["protocol"]["gpioPlan"].empty(),
                "protocol should also emit a gpioPlan for host-side execution");
}

void testWeightControlFaultFallsBackToIndependentHotPath() {
    edge_platform::PlatformManager manager;
    configureSyntheticManager(manager);
    const auto dispatched = manager.dispatchCompute(json{{"operation", "analog-matmul"},
                                                         {"tokens", 256},
                                                         {"tensorBytes", 131072},
                                                         {"preferredBackend", "npu"},
                                                         {"executeGpioAsync", true},
                                                         {"weightControllerFaultMode", "damaged"},
                                                         {"weightConfig",
                                                          json{{"targets", json::array({json{{"row", 1}, {"col", 2}, {"value", 0.75}}})}}},
                                                         {"inputCycles", json::array({json{{"txDataValue", 3}, {"mockReadbackValue", 1}}})}});
    requireTrue(dispatched.value("accepted", false), "faulted weight-controller requests should still keep the compute hot path alive");
    requireTrue(dispatched["result"]["route"].value("weightControlDegraded", false),
                "dispatch route should flag the injected weight-controller fault");
    requireTrue(dispatched["result"]["protocol"]["weightControl"].value("degraded", false),
                "protocol should surface the injected weight-controller fault");
    requireTrue(dispatched["result"]["protocol"]["weightControl"].value("effectiveMode", std::string()) == "bypass-retain-current-weights",
                "fault injection should bypass sideband updates and retain current analog weights");
    requireTrue(dispatched["result"]["protocol"]["weightControl"].value("hotPathIndependent", false),
                "fault injection should preserve the Pi compute hot path independence guarantee");
}

void testDispatchComputeAutoBuildsFramesForAsyncProtocol() {
    edge_platform::PlatformManager manager;
    configureSyntheticManager(manager);
    const auto dispatched = manager.dispatchCompute(json{{"operation", "analog-matmul"},
                                                         {"tokens", 256},
                                                         {"tensorBytes", 131072},
                                                         {"preferredBackend", "npu"},
                                                         {"requestId", 7}});
    requireTrue(dispatched.value("accepted", false), "async protocol dispatch should be accepted");
    requireTrue(dispatched["result"]["dispatch"].value("autoBuiltFrames", false),
                "dispatch should auto-build transport frames when the caller supplies only a high-level request");
    const std::string status = dispatched["result"]["dispatch"].value("status", std::string());
    requireTrue(status == "scheduled" || status == "executed", "async dispatch should either execute or schedule frames");
}

void testDispatchComputeExecutesAsyncProtocolViaMockDriver() {
    edge_platform::PlatformManager manager;
    configureSyntheticManager(manager);
    const auto dispatched = manager.dispatchCompute(json{{"operation", "analog-matmul"},
                                                         {"tokens", 256},
                                                         {"tensorBytes", 131072},
                                                         {"preferredBackend", "npu"},
                                                         {"executeGpioAsync", true},
                                                         {"inputCycles",
                                                          json::array({json{{"txDataValue", 5}, {"mockReadbackValue", 2}},
                                                                       json{{"txDataValue", 2}, {"mockReadbackValue", 1}}})}});
    requireTrue(dispatched.value("accepted", false), "mock async gpio dispatch should be accepted");
    requireTrue(dispatched["result"]["dispatch"].value("status", std::string()) == "executed",
                "mock async gpio dispatch should execute immediately");
    requireTrue(dispatched["result"].contains("execution"), "mock async gpio dispatch should include execution details");
    requireTrue(dispatched["result"]["execution"].value("simulated", false),
                "non-linux async gpio execution should fall back to mock mode");
    requireTrue(dispatched["result"]["execution"]["readbacks"].size() >= 2,
                "mock execution should emit per-cycle readbacks");
}

void testVirtualMemoryHotWeightPromotionAndSdFallback() {
    edge_platform::PlatformManager manager;
    configureSyntheticManager(manager);
    manager.applyPatch(json{{"npuHotPromoteHits", 2},
                            {"npuHotWeightsLimit", 4},
                            {"npuVirtualMemoryEnabled", true},
                            {"npuAdvertisedMemoryMb", 8192}});

    const json request{{"operation", "analog-matmul"},
                       {"preferredBackend", "npu"},
                       {"tokens", 512},
                       {"tensorBytes", 524288},
                       {"weightBlockId", 9},
                       {"weightBytes", 4194304}};

    const auto first = manager.dispatchCompute(request);
    requireTrue(first.value("accepted", false), "first virtual-memory dispatch should be accepted");
    requireTrue(first["result"]["virtualMemory"].value("hot", true) == false,
                "first touch should come from sdcard fallback path");

    const auto second = manager.dispatchCompute(request);
    requireTrue(second.value("accepted", false), "second virtual-memory dispatch should be accepted");

    const auto third = manager.dispatchCompute(request);
    requireTrue(third.value("accepted", false), "third virtual-memory dispatch should be accepted");
    requireTrue(third["result"]["virtualMemory"].value("hot", false),
                "reused weight block should be promoted to hot residency");
    requireTrue(third["result"]["virtualMemory"].value("streamWeights", true) == false,
                "hot-resident weights should stop streaming from sdcard");
}

void testFleetPlannerAndMiniEfficiencyProbeAreExposed() {
    edge_platform::PlatformManager manager;
    configureSyntheticManager(manager);
    manager.applyPatch(json{{"npuUnitCount", 19}, {"npuEfficiencyProbeEnabled", true}});

    const auto plan = manager.planCompute(json{{"operation", "analog-matmul"},
                                               {"preferredBackend", "npu"},
                                               {"tokens", 1024},
                                               {"tensorBytes", 1048576}});
    const auto &result = plan["result"];
    requireTrue(result.contains("fleet") && result["fleet"].is_object(), "compute plan should expose fleet schedule");
    requireTrue(result["fleet"].value("laneCount", 0) == 19, "fleet scheduler should default to 19 external npu lanes");
    requireTrue(static_cast<int>(result["fleet"]["assignments"].size()) == result["chunking"].value("shardCount", 0),
                "fleet assignments should cover every shard");

    const auto dispatched = manager.dispatchCompute(json{{"operation", "analog-matmul"},
                                                         {"preferredBackend", "npu"},
                                                         {"tokens", 1024},
                                                         {"tensorBytes", 1048576}});
    requireTrue(dispatched.value("accepted", false), "fleet dispatch should be accepted");
    requireTrue(dispatched["result"].contains("efficiencyProbe") && dispatched["result"]["efficiencyProbe"].is_object(),
                "dispatch should include mini efficiency probe output");
    requireTrue(dispatched["result"]["efficiencyProbe"].contains("score"),
                "mini efficiency probe should emit an efficiency score");
}

} // namespace

int main() {
    try {
        testSyntheticNetlistBuildsExpectedInterfaces();
        testDenseComputePrefersNpuRoute();
        testControlHeavyRequestStaysOnCpu();
        testAsyncProtocolPrefersDirectGpioBoundaryWhenAvailable();
        testWeightControlFaultFallsBackToIndependentHotPath();
        testDispatchComputeAutoBuildsFramesForAsyncProtocol();
        testDispatchComputeExecutesAsyncProtocolViaMockDriver();
        testVirtualMemoryHotWeightPromotionAndSdFallback();
        testFleetPlannerAndMiniEfficiencyProbeAreExposed();
        std::cout << "edge_platform_unit_tests: ok" << std::endl;
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "edge_platform_unit_tests: failed: " << ex.what() << std::endl;
        return 1;
    }
}