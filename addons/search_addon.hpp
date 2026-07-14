/* search_addon.hpp - Search addon */

#pragma once

#include <memory>
#include <string>

#include "../addon.hpp"

namespace addon::builtins {

/* Create search addon */
std::shared_ptr<Addon> createSearchAddon(const std::string &name);

} // namespace addon::builtins
