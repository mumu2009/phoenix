/* math_addon.hpp - Math addon */

#pragma once

#include <memory>
#include <string>

#include "../addon.hpp"

namespace addon::builtins {

/* Create math addon */
std::shared_ptr<Addon> createMathAddon(const std::string &name);

/* Evaluate a math expression directly (exact + float modes).
   Returns {"ok":true,"value":"...","exact":bool,"mode":"exact"|"float",
            "numeric":number|null,"expression":...}
   or      {"ok":false,"error":"...","position":n,"expression":...}. */
json evaluateMathExpression(const std::string &expr);

} // namespace addon::builtins
