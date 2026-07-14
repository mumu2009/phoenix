/* physics_world_runtime.hpp - Physics world runtime execution */

#pragma once

#include <cstddef>
#include <filesystem>

#include "physics_world.hpp"

namespace physics_world {

/* Execute native physics scene simulation */
json executeNativePhysicsScene(const json &physicsScene,
                               const fs::path &workspaceRoot,
                               std::size_t frameCount = 12,
                               double stepSeconds = 1.0 / 12.0);

} // namespace physics_world