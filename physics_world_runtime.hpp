/* physics_world_runtime.hpp - Physics world runtime execution
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