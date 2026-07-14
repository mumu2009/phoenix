/* computer_shell_addon.hpp - Computer shell addon */

#pragma once

#include <memory>
#include <string>

#include "../addon.hpp"

namespace addon::builtins {

/* Create computer shell addon */
std::shared_ptr<Addon> createComputerShellAddon(const std::string &name = std::string());

} // namespace addon::builtins