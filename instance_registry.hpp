/* instance_registry.hpp - Process-wide instance registry (multi-instance
   safety).  Copyright (C) 2026 079 Project.

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   079 Project is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with 079 Project.  If not, see <http://www.gnu.org/licenses/>.

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
