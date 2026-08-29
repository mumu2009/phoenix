/* instance_isolation.hpp - one live llama context per instance

   High concurrency must not let two instances share one llama_context.
   On the RDK the 8B weights fit once, so isolation is exclusive use of
   that single llama-server process (queue, never overlap).  Spawning a
   second 8B process is a host-only option.

   Shared across instances: cross_context_memory + mission experience.
   Not shared: KV / unit-query buffers / workspace / sensations. */
#pragma once

#include <algorithm>
#include <string>

#include "phoenix_config.hpp"

namespace phoenix {
namespace instance {

inline bool exclusiveLlamaProcess() {
  const std::string mode =
      phoenix::cfgOr<std::string>("llama.instanceIsolation",
                                  std::string("exclusive"));
  return mode != "shared";
}

/** Concurrent llama users. Exclusive mode is always 1. */
inline int liveLlamaUsers() {
  if (exclusiveLlamaProcess())
    return 1;
  return std::max(1, phoenix::cfgOr<int>("llama_server.parallel", 1));
}

inline std::string ccmStorePath() {
  return phoenix::resolveConfig<std::string>(
      "crossContextMemory.path",
      std::string("./runtime_store/cross_context_memory.json"));
}

}  // namespace instance
}  // namespace phoenix
