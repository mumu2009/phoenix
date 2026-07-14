/* math_addon.hpp - Math addon */

#pragma once

#include <memory>
#include <string>

#include "../addon.hpp"

namespace addon::builtins {

/* Create math addon */
std::shared_ptr<Addon> createMathAddon(const std::string &name);

} // namespace addon::builtins
