/* cli_json_addon.hpp - Generic "any CLI software becomes a plugin" addon. */
#pragma once

#include <memory>
#include <string>

#include "../addon.hpp"

namespace addon::builtins {

/* Create the cli-json addon.  Commands are dispatched through a process-wide
   whitelist registry (set by the gateway from config cliTools.*); without a
   registered template the addon refuses to run (fail-closed). */
std::shared_ptr<Addon> createCliJsonAddon(const std::string &name = std::string());

/* Registry: template name -> {command, fixedArgs[], timeoutMs, jsonOutput}.
   Process-global; the gateway installs it from config. */
bool setCliJsonRegistry(const json &registry, std::string *error = nullptr);
void clearCliJsonRegistry();
json getCliJsonRegistry();

/* Run one whitelisted command; exposed for tests. */
json runCliJsonCommand(const std::string &tool, const std::vector<std::string> &args,
                       const json &options);

} // namespace addon::builtins
