/* emergency_stop.hpp - System-level E-stop (multi-instance kill switch).
   Copyright (C) 2026 079 Project

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

   Industrial E-stop semantics, applied to the multi-instance system:

   1. LATCHING: press() latches permanently - it cannot be un-pressed from
      code (only a process restart clears it; resetForTesting() is the
      test-only exception).  This is fail-safe: a kill switch that can be
      "un-killed" by the very system it guards protects nothing.
   2. HIGHEST PRIORITY: the latch is checked by iterate()/interject()/the
      autonomy heartbeat before any work; once latched, every instance
      refuses new work and every registered instance is stopped.
   3. KILL ALL INSTANCES: press() walks the InstanceRegistry (instances
      register when their lifecycle begins) and invokes every stop handler.
   4. SELF-SHUTDOWN: after stopping all instances, the registered shutdown
      handler runs (the gateway persists state and quits the process), so
      the system cannot keep running uncontrolled.

   Layering: single-instance safety is the memebarrier (graph safety scan);
   this module is the system-level layer above it (defense in depth).
*/
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <nlohmann/json.hpp>

namespace phoenix {
namespace safety {

class EmergencyStop {
public:
  static EmergencyStop &instance();

  /* Press the E-stop.  Idempotent: the first press latches, stops all
     registered instances and invokes the shutdown handler; later presses
     return the original report.  Returns the stop report + status. */
  nlohmann::json press(const std::string &reason);

  bool latched() const { return latched_.load(std::memory_order_acquire); }
  std::string reason() const;
  nlohmann::json status() const;

  /* The gateway installs this: persist state and quit the process. */
  void setShutdownHandler(std::function<void()> fn);
  bool hasShutdownHandler() const;

  /* TEST ONLY: clear the latch so the test process can continue. */
  void resetForTesting();

private:
  EmergencyStop() = default;

  std::atomic<bool> latched_{false};
  mutable std::mutex mu_;
  std::string reason_;
  int64_t pressedAtMs_{0};
  nlohmann::json lastReport_ = nlohmann::json::object();
  std::function<void()> shutdownHandler_;
};

} /* namespace safety */
} /* namespace phoenix */
