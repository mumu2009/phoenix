#include "physics_world.hpp"
#include "physics_world_runtime.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using json = nlohmann::json;
namespace fs = std::filesystem;

void requireTrue(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void testNormalizeEarthMapRequestDefaultsToBundledHeightfield() {
    const json request = physics_world::normalizeEarthMapImportRequest(json{{"enabled", true}});
    requireTrue(request.value("enabled", false), "earth map request should stay enabled");
    requireTrue(request.value("format", std::string()) == "heightfield", "earth map format should default to heightfield");
    requireTrue(request.value("coordinateFrame", std::string()) == "wgs84-local-enu", "earth map coordinate frame should default to local ENU");
    requireTrue(request.value("sourceUri", std::string()) == physics_world::bundledEarthHeightfieldUri(), "earth map request should default to the bundled heightfield asset");
}

void testExecuteNativePhysicsSceneRunsWithBundledHeightfield() {
    const fs::path workspaceRoot = fs::current_path();
    const json runtime = physics_world::inspectBullet3Runtime(workspaceRoot / "outsides" / "bullet3");
    requireTrue(runtime.value("repoPresent", false), "bullet3 repository should be present for native runtime test");
    requireTrue(runtime.value("nativeCompiled", false), "native Bullet sources should be compiled into the current test build");

    const json physicsScene{{"enabled", true},
                            {"backend", "bullet3"},
                            {"substeps", 4},
                            {"summary", "Bundled terrain rollout for embodied coordination."},
                            {"earthMap", physics_world::normalizeEarthMapImportRequest(json{{"enabled", true},
                                                                                           {"sourceUri", physics_world::bundledEarthHeightfieldUri()},
                                                                                           {"format", "heightfield"},
                                                                                           {"regionLabel", "china-relief-demo"}})},
                            {"rigidBodies", json::array({json{{"id", "agent-planner-1"},
                                                               {"shape", "capsule"},
                                                               {"massKg", 78.0},
                                                               {"bodyClass", "dynamic-agent"},
                                                               {"role", "planner"},
                                                               {"position", json{{"x", 1}, {"y", 1}, {"z", 1}}}},
                                                         json{{"id", "agent-explorer-2"},
                                                               {"shape", "sphere"},
                                                               {"massKg", 65.0},
                                                               {"bodyClass", "dynamic-agent"},
                                                               {"role", "explorer"},
                                                               {"position", json{{"x", 4}, {"y", 3}, {"z", 1}}}}})},
                            {"constraints", json::array({json{{"type", "formation-link"},
                                                               {"bodyA", "agent-planner-1"},
                                                               {"bodyB", "agent-explorer-2"}}})}};

    const json execution = physics_world::executeNativePhysicsScene(physicsScene, workspaceRoot, 8, 1.0 / 16.0);

    requireTrue(execution.value("ok", false), "native physics execution should succeed");
    requireTrue(execution.value("status", std::string()) == "executed", "native physics execution should report executed status");
    requireTrue(execution.value("backend", std::string()) == "bullet3-native", "native physics execution should report the Bullet backend");
    requireTrue(execution.contains("terrain") && execution["terrain"].is_object(), "native physics execution should return terrain metadata");
    requireTrue(execution["terrain"].value("format", std::string()) == "heightfield", "native physics terrain should use heightfield format");
    requireTrue(execution["terrain"].value("sourceStatus", std::string()) == "loaded-local", "native physics terrain should load the bundled local heightfield");
    requireTrue(execution.contains("bodySummaries") && execution["bodySummaries"].is_array() && execution["bodySummaries"].size() == 2,
                "native physics execution should emit both dynamic body traces");
    requireTrue(execution.contains("trainSamples") && execution["trainSamples"].is_array() && !execution["trainSamples"].empty(),
                "native physics execution should emit runtime training samples");
}

} // namespace

int main() {
    try {
        testNormalizeEarthMapRequestDefaultsToBundledHeightfield();
        testExecuteNativePhysicsSceneRunsWithBundledHeightfield();
        std::cout << "physics_world_runtime_tests: ok" << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "physics_world_runtime_tests: " << e.what() << std::endl;
        return 1;
    }
}