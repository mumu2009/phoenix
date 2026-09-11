/* security_addon.hpp - Optional addon wrapper (not default-mounted) */

#pragma once

#include <memory>
#include <string>

#include "../../addon.hpp"

namespace addon {
namespace builtins {
std::shared_ptr<Addon> createSecurityAddon(const std::string &name);
}
} // namespace addon
