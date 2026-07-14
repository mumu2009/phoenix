/* physics_world_runtime.cpp - Physics world runtime implementation
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   079 Project is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with 079 Project.  If not, see <http://www.gnu.org/licenses/>. */

#include "physics_world_runtime.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <unordered_map>

#if defined(AI_HAVE_BULLET3_NATIVE) && __has_include("btBulletDynamicsCommon.h")
#include "btBulletDynamicsCommon.h"
#include "BulletCollision/CollisionShapes/btHeightfieldTerrainShape.h"
#define PHYSICS_WORLD_HAS_NATIVE_BULLET 1
#else
#define PHYSICS_WORLD_HAS_NATIVE_BULLET 0
#endif

namespace physics_world {
namespace {

struct LoadedHeightfield {
    int width{0};
    int height{0};
    std::vector<float> samples;
    double metersPerCell{750.0};
    float minHeight{0.0F};
    float maxHeight{0.0F};
    json metadata{json::object()};
};

struct SceneExtents {
    double maxXCells{1.0};
    double maxYCells{1.0};
};

#if PHYSICS_WORLD_HAS_NATIVE_BULLET
struct BodyTrace {
    std::string id;
    std::string role;
    std::string bodyClass;
    bool dynamic{false};
    btRigidBody *body{nullptr};
    std::vector<json> frames;
};
#endif

inline std::uint64_t fnv1a64(const std::string &text) {
    std::uint64_t hash = 1469598103934665603ull;
    for (unsigned char ch : text) {
        hash ^= static_cast<std::uint64_t>(ch);
        hash *= 1099511628211ull;
    }
    return hash;
}

inline double clamp01(double value) {
    if (value < 0.0) {
        return 0.0;
    }
    if (value > 1.0) {
        return 1.0;
    }
    return value;
}

template <typename T>
inline T clampValue(T value, T low, T high) {
    return std::max(low, std::min(high, value));
}

inline fs::path resolveLocalSourcePath(const std::string &sourceUri, const fs::path &workspaceRoot) {
    if (sourceUri.empty() || detectSourceType(sourceUri) != "local-file") {
        return {};
    }
    fs::path candidate(sourceUri);
    if (candidate.is_absolute() && fs::exists(candidate)) {
        return candidate;
    }
    if (fs::exists(workspaceRoot / candidate)) {
        return workspaceRoot / candidate;
    }
    if (fs::exists(fs::current_path() / candidate)) {
        return fs::current_path() / candidate;
    }
    return {};
}

inline bool extractHeightSamples(const json &rawHeights, int expectedWidth, int expectedHeight, std::vector<float> &out) {
    out.clear();
    if (!rawHeights.is_array()) {
        return false;
    }

    if (!rawHeights.empty() && rawHeights.front().is_array()) {
        int height = static_cast<int>(rawHeights.size());
        int width = 0;
        for (const auto &row : rawHeights) {
            if (!row.is_array()) {
                return false;
            }
            width = std::max(width, static_cast<int>(row.size()));
        }
        if ((expectedWidth > 0 && width != expectedWidth) || (expectedHeight > 0 && height != expectedHeight)) {
            return false;
        }
        out.reserve(static_cast<std::size_t>(width * height));
        for (const auto &row : rawHeights) {
            for (const auto &entry : row) {
                out.push_back(entry.is_number() ? static_cast<float>(entry.get<double>()) : 0.0F);
            }
        }
        return true;
    }

    const int total = static_cast<int>(rawHeights.size());
    if (expectedWidth > 0 && expectedHeight > 0 && total != expectedWidth * expectedHeight) {
        return false;
    }
    out.reserve(rawHeights.size());
    for (const auto &entry : rawHeights) {
        out.push_back(entry.is_number() ? static_cast<float>(entry.get<double>()) : 0.0F);
    }
    return !out.empty();
}

inline LoadedHeightfield buildProceduralHeightfield(const json &manifest) {
    LoadedHeightfield out;
    out.width = clampValue(manifest.value("gridWidth", 12), 4, 64);
    out.height = clampValue(manifest.value("gridHeight", 10), 4, 64);
    out.metersPerCell = std::max(1.0, manifest.value("metersPerCell", 750.0));
    const json geoBounds = manifest.value("geoBounds", json::object());
    const double latMin = geoBounds.value("latMin", 18.0);
    const double latMax = geoBounds.value("latMax", 54.0);
    const double lonMin = geoBounds.value("lonMin", 73.0);
    const double lonMax = geoBounds.value("lonMax", 136.0);
    const std::string regionLabel = manifest.value("regionLabel", std::string("china-relief-demo"));
    const std::uint64_t seed = fnv1a64(regionLabel + manifest.value("sourceUri", std::string()));
    const double ridge = 350.0 + static_cast<double>(seed % 800);
    const double basin = 120.0 + static_cast<double>((seed / 17ull) % 260ull);
    const double ripple = 40.0 + static_cast<double>((seed / 97ull) % 120ull);

    out.samples.reserve(static_cast<std::size_t>(out.width * out.height));
    out.minHeight = std::numeric_limits<float>::max();
    out.maxHeight = std::numeric_limits<float>::lowest();
    for (int row = 0; row < out.height; ++row) {
        const double v = out.height > 1 ? static_cast<double>(row) / static_cast<double>(out.height - 1) : 0.5;
        const double lat = latMin + (latMax - latMin) * v;
        for (int col = 0; col < out.width; ++col) {
            const double u = out.width > 1 ? static_cast<double>(col) / static_cast<double>(out.width - 1) : 0.5;
            const double lon = lonMin + (lonMax - lonMin) * u;
            const double westernRise = ridge * std::pow(1.0 - u, 1.35);
            const double coastalFalloff = basin * std::pow(u, 2.1);
            const double latWave = std::sin((lat + 8.0) * 0.13) * ripple;
            const double lonWave = std::cos((lon - 18.0) * 0.09) * (ripple * 0.8);
            const double plateau = 180.0 + westernRise - coastalFalloff + latWave + lonWave;
            const float sample = static_cast<float>(plateau);
            out.samples.push_back(sample);
            out.minHeight = std::min(out.minHeight, sample);
            out.maxHeight = std::max(out.maxHeight, sample);
        }
    }

    out.metadata = json{{"format", "heightfield"},
                        {"sourceStatus", "generated-procedural"},
                        {"sourceUri", manifest.value("sourceUri", bundledEarthHeightfieldUri())},
                        {"regionLabel", regionLabel},
                        {"metersPerCell", out.metersPerCell},
                        {"gridWidth", out.width},
                        {"gridHeight", out.height},
                        {"minHeight", out.minHeight},
                        {"maxHeight", out.maxHeight},
                        {"summary", truncateText("procedural heightfield proxy prepared for " + regionLabel, 220)}};
    return out;
}

inline LoadedHeightfield loadHeightfield(const json &earthMap, const fs::path &workspaceRoot) {
    const json manifest = normalizeEarthMapImportRequest(earthMap);
    const fs::path resolved = resolveLocalSourcePath(manifest.value("sourceUri", std::string()), workspaceRoot);
    if (resolved.empty()) {
        return buildProceduralHeightfield(manifest);
    }

    std::ifstream input(resolved, std::ios::binary);
    if (!input) {
        return buildProceduralHeightfield(manifest);
    }

    json fileJson = json::object();
    try {
        input >> fileJson;
    } catch (...) {
        return buildProceduralHeightfield(manifest);
    }

    LoadedHeightfield out;
    out.width = clampValue(fileJson.value("gridWidth", fileJson.value("width", 0)), 4, 256);
    out.height = clampValue(fileJson.value("gridHeight", fileJson.value("height", 0)), 4, 256);
    if (out.width <= 0 || out.height <= 0) {
        if (fileJson.contains("heights") && fileJson["heights"].is_array() && !fileJson["heights"].empty() && fileJson["heights"].front().is_array()) {
            out.height = static_cast<int>(fileJson["heights"].size());
            out.width = static_cast<int>(fileJson["heights"].front().size());
        }
    }
    out.metersPerCell = std::max(1.0, fileJson.value("metersPerCell", manifest.value("metersPerCell", 750.0)));
    if (!extractHeightSamples(fileJson.value("heights", json::array()), out.width, out.height, out.samples)) {
        return buildProceduralHeightfield(manifest);
    }

    out.minHeight = std::numeric_limits<float>::max();
    out.maxHeight = std::numeric_limits<float>::lowest();
    for (float sample : out.samples) {
        out.minHeight = std::min(out.minHeight, sample);
        out.maxHeight = std::max(out.maxHeight, sample);
    }

    out.metadata = json{{"format", "heightfield"},
                        {"sourceStatus", "loaded-local"},
                        {"sourceUri", manifest.value("sourceUri", std::string())},
                        {"resolvedPath", resolved.string()},
                        {"regionLabel", fileJson.value("regionLabel", manifest.value("regionLabel", std::string("global-earth")))},
                        {"coordinateFrame", fileJson.value("coordinateFrame", manifest.value("coordinateFrame", preferredEarthCoordinateFrame()))},
                        {"metersPerCell", out.metersPerCell},
                        {"gridWidth", out.width},
                        {"gridHeight", out.height},
                        {"minHeight", out.minHeight},
                        {"maxHeight", out.maxHeight},
                        {"summary", truncateText(fileJson.value("summary", std::string("local heightfield loaded for native Bullet terrain execution")), 220)}};
    return out;
}

inline double sampleHeightMeters(const LoadedHeightfield &heightfield, double u, double v) {
    if (heightfield.samples.empty() || heightfield.width <= 0 || heightfield.height <= 0) {
        return 0.0;
    }
    const int x = clampValue(static_cast<int>(std::lround(clamp01(u) * static_cast<double>(heightfield.width - 1))), 0, heightfield.width - 1);
    const int y = clampValue(static_cast<int>(std::lround(clamp01(v) * static_cast<double>(heightfield.height - 1))), 0, heightfield.height - 1);
    const std::size_t index = static_cast<std::size_t>(y * heightfield.width + x);
    if (index >= heightfield.samples.size()) {
        return 0.0;
    }
    return static_cast<double>(heightfield.samples[index]);
}

inline SceneExtents deriveSceneExtents(const json &physicsScene) {
    SceneExtents out;
    if (!physicsScene.contains("rigidBodies") || !physicsScene["rigidBodies"].is_array()) {
        return out;
    }
    for (const auto &body : physicsScene["rigidBodies"]) {
        if (!body.is_object() || !body.contains("position") || !body["position"].is_object()) {
            continue;
        }
        const auto &position = body["position"];
        out.maxXCells = std::max(out.maxXCells, static_cast<double>(position.value("x", 0)) + 1.0);
        out.maxYCells = std::max(out.maxYCells, static_cast<double>(position.value("y", 0)) + 1.0);
    }
    return out;
}

#if PHYSICS_WORLD_HAS_NATIVE_BULLET
inline btVector3 toBulletPosition(const json &position,
                                  const SceneExtents &extents,
                                  const LoadedHeightfield *heightfield,
                                  double planeCellSize) {
    const double cellX = static_cast<double>(position.value("x", 0));
    const double cellY = static_cast<double>(position.value("y", 0));
    const double layer = static_cast<double>(position.value("z", 0));
    if (heightfield != nullptr && heightfield->width > 1 && heightfield->height > 1) {
        const double u = extents.maxXCells > 1.0 ? clamp01(cellX / (extents.maxXCells - 1.0)) : 0.5;
        const double v = extents.maxYCells > 1.0 ? clamp01(cellY / (extents.maxYCells - 1.0)) : 0.5;
        const double worldX = (u * static_cast<double>(heightfield->width - 1) - static_cast<double>(heightfield->width - 1) * 0.5) * heightfield->metersPerCell;
        const double worldZ = (v * static_cast<double>(heightfield->height - 1) - static_cast<double>(heightfield->height - 1) * 0.5) * heightfield->metersPerCell;
        const double terrainY = sampleHeightMeters(*heightfield, u, v);
        return btVector3(static_cast<btScalar>(worldX),
                         static_cast<btScalar>(terrainY + 5.0 + layer * 4.0),
                         static_cast<btScalar>(worldZ));
    }
    return btVector3(static_cast<btScalar>((cellX - extents.maxXCells * 0.5) * planeCellSize),
                     static_cast<btScalar>(2.0 + layer * 1.2),
                     static_cast<btScalar>((cellY - extents.maxYCells * 0.5) * planeCellSize));
}

inline std::unique_ptr<btCollisionShape> makeCollisionShape(const std::string &shape, double scaleHint) {
    const double bodyScale = clampValue(scaleHint * 0.015, 0.6, 6.0);
    if (shape == "sphere") {
        return std::make_unique<btSphereShape>(static_cast<btScalar>(bodyScale * 0.6));
    }
    if (shape == "box" || shape == "sensor-volume") {
        return std::make_unique<btBoxShape>(btVector3(static_cast<btScalar>(bodyScale * 0.65),
                                                      static_cast<btScalar>(bodyScale * 0.35),
                                                      static_cast<btScalar>(bodyScale * 0.65)));
    }
    return std::make_unique<btCapsuleShape>(static_cast<btScalar>(bodyScale * 0.42),
                                            static_cast<btScalar>(bodyScale * 1.25));
}

inline btVector3 deriveInitialVelocity(const std::string &role, double motionScale) {
    const double speed = clampValue(motionScale * 0.012, 0.8, 8.0);
    if (role == "explorer" || role == "scout") {
        return btVector3(static_cast<btScalar>(speed * 1.4), 0.0, static_cast<btScalar>(speed * 0.6));
    }
    if (role == "planner" || role == "builder") {
        return btVector3(static_cast<btScalar>(speed * 0.8), 0.0, static_cast<btScalar>(speed * 0.9));
    }
    if (role == "critic") {
        return btVector3(static_cast<btScalar>(-speed * 0.45), 0.0, static_cast<btScalar>(speed * 0.35));
    }
    if (role == "memory") {
        return btVector3(static_cast<btScalar>(speed * 0.2), 0.0, static_cast<btScalar>(-speed * 0.2));
    }
    return btVector3(static_cast<btScalar>(speed * 0.5), 0.0, static_cast<btScalar>(speed * 0.3));
}

inline json captureFrameState(const btRigidBody &body, int frameIndex) {
    const btVector3 origin = body.getWorldTransform().getOrigin();
    const btVector3 velocity = body.getLinearVelocity();
    return json{{"frame", frameIndex},
                {"position", {origin.getX(), origin.getY(), origin.getZ()}},
                {"velocity", {velocity.getX(), velocity.getY(), velocity.getZ()}},
                {"speed", velocity.length()}};
}
#endif

inline std::string summarizeBodyMotion(const json &bodySummary) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(2);
    oss << bodySummary.value("id", std::string("body"))
        << " displaced " << bodySummary.value("displacementMeters", 0.0)
        << "m with peak speed " << bodySummary.value("peakSpeedMps", 0.0) << "m/s";
    return oss.str();
}

} // namespace

json executeNativePhysicsScene(const json &physicsScene,
                               const fs::path &workspaceRoot,
                               std::size_t frameCount,
                               double stepSeconds) {
    if (!physicsScene.is_object() || !physicsScene.value("enabled", false)) {
        return json{{"ok", true},
                    {"status", "disabled"},
                    {"backend", physicsScene.value("backend", std::string("bullet3"))},
                    {"summary", "native physics execution skipped because the physics scene is disabled"},
                    {"trainSamples", json::array()}};
    }

    const std::string backend = lowerCopy(trimCopy(physicsScene.value("backend", std::string("bullet3"))));
    if (backend != "bullet3") {
        return json{{"ok", false},
                    {"status", "unsupported-backend"},
                    {"backend", backend},
                    {"summary", "native physics execution currently supports bullet3 only"},
                    {"trainSamples", json::array()}};
    }

#if !PHYSICS_WORLD_HAS_NATIVE_BULLET
    return json{{"ok", false},
                {"status", "native-bullet-unavailable"},
                {"backend", backend},
                {"summary", "native Bullet execution was requested but the current build does not include embedded Bullet sources"},
                {"trainSamples", json::array()}};
#else
    const std::size_t frames = std::max<std::size_t>(4, std::min<std::size_t>(frameCount, 24));
    const double dt = stepSeconds > 0.0 ? stepSeconds : (1.0 / 12.0);
    const int substeps = std::max(1, physicsScene.value("substeps", 4));
    const SceneExtents extents = deriveSceneExtents(physicsScene);
    const bool hasEarthMap = physicsScene.contains("earthMap") && physicsScene["earthMap"].is_object() && physicsScene["earthMap"].value("enabled", false);
    const bool useHeightfield = hasEarthMap && lowerCopy(trimCopy(physicsScene["earthMap"].value("format", std::string()))) == "heightfield";
    std::unique_ptr<LoadedHeightfield> heightfield;
    if (useHeightfield) {
        heightfield = std::make_unique<LoadedHeightfield>(loadHeightfield(physicsScene["earthMap"], workspaceRoot));
    }

    btDefaultCollisionConfiguration collisionConfig;
    btCollisionDispatcher dispatcher(&collisionConfig);
    btDbvtBroadphase broadphase;
    btSequentialImpulseConstraintSolver solver;
    btDiscreteDynamicsWorld world(&dispatcher, &broadphase, &solver, &collisionConfig);

    btVector3 gravity(0.0, -9.81, 0.0);
    if (physicsScene.contains("gravity") && physicsScene["gravity"].is_array() && physicsScene["gravity"].size() >= 3) {
        gravity = btVector3(static_cast<btScalar>(physicsScene["gravity"][0].get<double>()),
                            static_cast<btScalar>(physicsScene["gravity"][1].get<double>()),
                            static_cast<btScalar>(physicsScene["gravity"][2].get<double>()));
    }
    world.setGravity(gravity);

    std::vector<std::unique_ptr<btCollisionShape>> shapes;
    std::vector<std::unique_ptr<btMotionState>> motionStates;
    std::vector<std::unique_ptr<btRigidBody>> rigidBodies;
    std::vector<std::unique_ptr<btTypedConstraint>> constraints;
    std::unordered_map<std::string, btRigidBody *> bodyLookup;
    std::unordered_map<const btCollisionObject *, std::string> collisionNames;
    std::vector<BodyTrace> traces;

    const double planeCellSize = heightfield ? heightfield->metersPerCell : 6.0;

    if (heightfield && !heightfield->samples.empty()) {
        auto terrainShape = std::make_unique<btHeightfieldTerrainShape>(heightfield->width,
                                                                        heightfield->height,
                                                                        heightfield->samples.data(),
                                                                        btScalar(1.0),
                                                                        btScalar(heightfield->minHeight),
                                                                        btScalar(heightfield->maxHeight),
                                                                        1,
                                                                        PHY_FLOAT,
                                                                        false);
        terrainShape->setLocalScaling(btVector3(static_cast<btScalar>(heightfield->metersPerCell), 1.0, static_cast<btScalar>(heightfield->metersPerCell)));
        btTransform terrainTransform;
        terrainTransform.setIdentity();
        terrainTransform.setOrigin(btVector3(0.0,
                                             static_cast<btScalar>((heightfield->maxHeight + heightfield->minHeight) * 0.5),
                                             0.0));
        auto motionState = std::make_unique<btDefaultMotionState>(terrainTransform);
        btRigidBody::btRigidBodyConstructionInfo terrainInfo(0.0, motionState.get(), terrainShape.get(), btVector3(0.0, 0.0, 0.0));
        auto terrainBody = std::make_unique<btRigidBody>(terrainInfo);
        terrainBody->setFriction(btScalar(1.3));
        world.addRigidBody(terrainBody.get());
        collisionNames[terrainBody.get()] = "earth-terrain";
        bodyLookup["earth-terrain"] = terrainBody.get();
        motionStates.push_back(std::move(motionState));
        shapes.push_back(std::move(terrainShape));
        rigidBodies.push_back(std::move(terrainBody));
    } else {
        auto groundShape = std::make_unique<btStaticPlaneShape>(btVector3(0.0, 1.0, 0.0), 0.0);
        btTransform groundTransform;
        groundTransform.setIdentity();
        auto motionState = std::make_unique<btDefaultMotionState>(groundTransform);
        btRigidBody::btRigidBodyConstructionInfo groundInfo(0.0, motionState.get(), groundShape.get(), btVector3(0.0, 0.0, 0.0));
        auto groundBody = std::make_unique<btRigidBody>(groundInfo);
        groundBody->setFriction(btScalar(1.1));
        world.addRigidBody(groundBody.get());
        collisionNames[groundBody.get()] = "world-ground-plane";
        bodyLookup["world-ground-plane"] = groundBody.get();
        motionStates.push_back(std::move(motionState));
        shapes.push_back(std::move(groundShape));
        rigidBodies.push_back(std::move(groundBody));
    }

    if (physicsScene.contains("rigidBodies") && physicsScene["rigidBodies"].is_array()) {
        for (const auto &bodySpec : physicsScene["rigidBodies"]) {
            if (!bodySpec.is_object()) {
                continue;
            }
            const std::string id = trimCopy(bodySpec.value("id", std::string()));
            if (id.empty() || id == "earth-terrain" || id == "world-ground-plane") {
                continue;
            }

            const std::string bodyClass = trimCopy(bodySpec.value("bodyClass", std::string()));
            const std::string role = trimCopy(bodySpec.value("role", std::string()));
            const std::string shapeName = lowerCopy(trimCopy(bodySpec.value("shape", std::string("capsule"))));
            const double massKg = std::max(0.0, bodySpec.value("massKg", 0.0));
            auto shape = makeCollisionShape(shapeName, planeCellSize);
            btVector3 inertia(0.0, 0.0, 0.0);
            if (massKg > 0.0) {
                shape->calculateLocalInertia(static_cast<btScalar>(massKg), inertia);
            }

            const json position = bodySpec.value("position", json::object());
            btTransform transform;
            transform.setIdentity();
            transform.setOrigin(toBulletPosition(position, extents, heightfield.get(), planeCellSize));
            auto motionState = std::make_unique<btDefaultMotionState>(transform);
            btRigidBody::btRigidBodyConstructionInfo bodyInfo(static_cast<btScalar>(massKg), motionState.get(), shape.get(), inertia);
            auto body = std::make_unique<btRigidBody>(bodyInfo);
            body->setFriction(bodyClass == "dynamic-agent" ? btScalar(0.9) : btScalar(1.2));
            body->setDamping(btScalar(0.08), btScalar(0.2));
            if (massKg > 0.0) {
                body->setLinearVelocity(deriveInitialVelocity(lowerCopy(role), planeCellSize));
                body->setActivationState(DISABLE_DEACTIVATION);
            }

            world.addRigidBody(body.get());
            bodyLookup[id] = body.get();
            collisionNames[body.get()] = id;
            traces.push_back(BodyTrace{id, role, bodyClass, massKg > 0.0, body.get(), {captureFrameState(*body, 0)}});
            motionStates.push_back(std::move(motionState));
            shapes.push_back(std::move(shape));
            rigidBodies.push_back(std::move(body));
        }
    }

    if (physicsScene.contains("constraints") && physicsScene["constraints"].is_array()) {
        for (const auto &constraintSpec : physicsScene["constraints"]) {
            if (!constraintSpec.is_object()) {
                continue;
            }
            if (lowerCopy(trimCopy(constraintSpec.value("type", std::string()))) != "formation-link") {
                continue;
            }
            const std::string bodyAId = trimCopy(constraintSpec.value("bodyA", std::string()));
            const std::string bodyBId = trimCopy(constraintSpec.value("bodyB", std::string()));
            auto itA = bodyLookup.find(bodyAId);
            auto itB = bodyLookup.find(bodyBId);
            if (itA == bodyLookup.end() || itB == bodyLookup.end() || itA->second == nullptr || itB->second == nullptr) {
                continue;
            }
            btRigidBody *bodyA = itA->second;
            btRigidBody *bodyB = itB->second;
            if (bodyA->getInvMass() == 0.0 && bodyB->getInvMass() == 0.0) {
                continue;
            }
            const btVector3 worldA = bodyA->getWorldTransform().getOrigin();
            const btVector3 worldB = bodyB->getWorldTransform().getOrigin();
            const btVector3 worldMid = (worldA + worldB) * btScalar(0.5);
            const btVector3 pivotA = bodyA->getCenterOfMassTransform().inverse() * worldMid;
            const btVector3 pivotB = bodyB->getCenterOfMassTransform().inverse() * worldMid;
            auto link = std::make_unique<btPoint2PointConstraint>(*bodyA, *bodyB, pivotA, pivotB);
            link->m_setting.m_tau = btScalar(0.18);
            link->m_setting.m_impulseClamp = btScalar(0.4);
            world.addConstraint(link.get(), true);
            constraints.push_back(std::move(link));
        }
    }

    std::set<std::string> contactPairs;
    for (std::size_t frame = 1; frame <= frames; ++frame) {
        world.stepSimulation(static_cast<btScalar>(dt), substeps, static_cast<btScalar>(dt / static_cast<double>(substeps)));
        for (auto &trace : traces) {
            if (trace.body != nullptr) {
                trace.frames.push_back(captureFrameState(*trace.body, static_cast<int>(frame)));
            }
        }
        for (int manifoldIndex = 0; manifoldIndex < dispatcher.getNumManifolds(); ++manifoldIndex) {
            const btPersistentManifold *manifold = dispatcher.getManifoldByIndexInternal(manifoldIndex);
            if (manifold == nullptr || manifold->getNumContacts() <= 0) {
                continue;
            }
            const btCollisionObject *objectA = static_cast<const btCollisionObject *>(manifold->getBody0());
            const btCollisionObject *objectB = static_cast<const btCollisionObject *>(manifold->getBody1());
            const std::string nameA = collisionNames.count(objectA) ? collisionNames[objectA] : std::string("body-a");
            const std::string nameB = collisionNames.count(objectB) ? collisionNames[objectB] : std::string("body-b");
            if (nameA < nameB) {
                contactPairs.insert(nameA + "<->" + nameB);
            } else {
                contactPairs.insert(nameB + "<->" + nameA);
            }
        }
    }

    json bodySummaries = json::array();
    for (const auto &trace : traces) {
        if (trace.frames.empty()) {
            continue;
        }
        const auto &start = trace.frames.front();
        const auto &end = trace.frames.back();
        const auto &startPos = start["position"];
        const auto &endPos = end["position"];
        const double dx = endPos[0].get<double>() - startPos[0].get<double>();
        const double dy = endPos[1].get<double>() - startPos[1].get<double>();
        const double dz = endPos[2].get<double>() - startPos[2].get<double>();
        double peakSpeed = 0.0;
        for (const auto &frame : trace.frames) {
            peakSpeed = std::max(peakSpeed, frame.value("speed", 0.0));
        }
        bodySummaries.push_back(json{{"id", trace.id},
                                     {"role", trace.role},
                                     {"bodyClass", trace.bodyClass},
                                     {"dynamic", trace.dynamic},
                                     {"displacementMeters", std::sqrt(dx * dx + dy * dy + dz * dz)},
                                     {"peakSpeedMps", peakSpeed},
                                     {"trace", trace.frames}});
    }

    json trainSamples = json::array();
    std::string graphHint;
    if (physicsScene.contains("summary") && physicsScene["summary"].is_string()) {
        graphHint = physicsScene["summary"].get<std::string>();
    }
    const std::string terrainSummary = heightfield ? heightfield->metadata.value("summary", std::string()) : std::string("static plane ground used for native Bullet execution");
    if (!graphHint.empty()) {
        trainSamples.push_back(json{{"input", "How did the native Bullet physics world evolve across the latest embodied rollout?"},
                                    {"target", truncateText(graphHint, 180)},
                                    {"graph", truncateText(graphHint, 220)},
                                    {"source", "sim_physics_runtime"}});
    }
    if (!terrainSummary.empty()) {
        trainSamples.push_back(json{{"input", "Which terrain asset anchored the native embodied simulation?"},
                                    {"target", truncateText(terrainSummary, 180)},
                                    {"graph", truncateText(graphHint.empty() ? terrainSummary : graphHint, 220)},
                                    {"source", "sim_earth_heightfield_runtime"}});
    }
    if (!bodySummaries.empty()) {
        const auto *best = &bodySummaries.front();
        for (const auto &summary : bodySummaries) {
            if (summary.value("displacementMeters", 0.0) > best->value("displacementMeters", 0.0)) {
                best = &summary;
            }
        }
        trainSamples.push_back(json{{"input", "Which embodied agent showed the strongest physical adaptation during the Bullet rollout?"},
                                    {"target", truncateText(summarizeBodyMotion(*best), 180)},
                                    {"graph", truncateText(graphHint.empty() ? summarizeBodyMotion(*best) : graphHint, 220)},
                                    {"source", "sim_physics_motion"}});
    }

    const std::string overallSummary = truncateText(
        "native Bullet execution produced " + std::to_string(bodySummaries.size()) +
            " body traces across " + std::to_string(frames) + " frames with " + std::to_string(contactPairs.size()) +
            " unique contact pairs on " + (heightfield ? std::string("a heightfield terrain") : std::string("a plane terrain")),
        220);

    json terrain = heightfield ? heightfield->metadata : json{{"format", "plane"}, {"summary", "static plane ground used for native Bullet execution"}};
    terrain["nativeRuntime"] = "bullet3";
    terrain["executed"] = true;

    return json{{"ok", true},
                {"status", "executed"},
                {"backend", "bullet3-native"},
                {"frames", frames},
                {"stepSeconds", dt},
                {"substeps", substeps},
                {"gravity", {gravity.getX(), gravity.getY(), gravity.getZ()}},
                {"terrain", terrain},
                {"contactPairs", json(contactPairs)},
                {"bodySummaries", bodySummaries},
                {"summary", overallSummary},
                {"trainSamples", trainSamples}};
#endif
}

} // namespace physics_world