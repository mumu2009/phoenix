/* builtin_registry.hpp - Builtin addon registry
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

#include <memory>
#include <string>
#include <vector>

#include "../addon.hpp"

namespace addon::builtins {

/* Create builtin addon by type */
std::shared_ptr<Addon> createBuiltinAddon(const std::string &type, const std::string &name, std::string *error);
/* Create default builtin addons */
std::vector<std::shared_ptr<Addon>> createDefaultBuiltinAddons(const std::string &selection = std::string());

} // namespace addon::builtins
