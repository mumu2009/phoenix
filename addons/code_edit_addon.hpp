/* code_edit_addon.hpp - recognise + normalise model code output. */

#pragma once

#include <memory>
#include <string>

#include "../addon.hpp"

namespace addon::builtins {

std::shared_ptr<Addon> createCodeEditAddon(const std::string &name);

/* Direct helper (also used by the gateway): turn a model reply with a
   fenced/raw code block into the FILE/OLD/NEW patch form.  Empty when the
   reply carries no code. */
std::string normaliseCodeEdit(const std::string &reply,
                              const std::string &path);

}  // namespace addon::builtins
