/* builtin_registry.hpp - Builtin addon registry */

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
