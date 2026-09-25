/* instance_registry.hpp - Process-wide instance registry (multi-instance
   safety).

   Every autonomous instance (autonomy loop, mission lifecycle, spawned
   successors, MCP servers) registers here when its lifecycle begins, with a
   stop handler that can kill it.  The emergency stop (emergency_stop.hpp)
   walks this registry and stops EVERYTHING - this is the multi-instance,
   system-level layer above the single-instance memebarrier.
*/
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace phoenix {
namespace safety {

using StopHandler = std::function<void()>;

class InstanceRegistry {
public:
  static InstanceRegistry &instance();

  /* Register an instance; returns its id.  stop is invoked by stopAll(). */
  uint64_t registerInstance(const std::string &name, const std::string &kind,
                            StopHandler stop);
  bool unregister(uint64_t id);
  size_t count() const;
  nlohmann::json snapshot() const; /* [{id,name,kind,registeredAtMs}] */

  struct StopReport {
    size_t total{0};
    size_t stopped{0};
    std::vector<std::string> errors;
    nlohmann::json toJson() const;
  };
  /* Invoke every registered stop handler (once).  Handlers must be
     idempotent and non-throwing; exceptions are captured per handler. */
  StopReport stopAll();

  /* TEST ONLY: drop all entries (never call in production). */
  void clearForTesting();

private:
  InstanceRegistry() = default;
  struct Entry {
    uint64_t id{0};
    std::string name;
    std::string kind;
    int64_t registeredAtMs{0};
    StopHandler stop;
  };
  mutable std::mutex mu_;
  uint64_t nextId_{1};
  std::map<uint64_t, Entry> entries_;
};

} /* namespace safety */
} /* namespace phoenix */
