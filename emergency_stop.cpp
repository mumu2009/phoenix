/* emergency_stop.cpp - Implementation, see header. */
#include "emergency_stop.hpp"

#include "instance_registry.hpp"

#include <chrono>

namespace phoenix {
namespace safety {

namespace {
int64_t nowMs() {
  return static_cast<int64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}
} /* namespace */

EmergencyStop &EmergencyStop::instance() {
  static EmergencyStop estop;
  return estop;
}

nlohmann::json EmergencyStop::press(const std::string &reason) {
  /* Fail-safe ordering: latch FIRST (so concurrent workers see it before
     anything else happens), then stop everything, then self-shutdown. */
  const bool already = latched_.exchange(true, std::memory_order_acq_rel);
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (already) {
      return {{"ok", true},
              {"alreadyLatched", true},
              {"latched", true},
              {"reason", reason_},
              {"pressedAtMs", pressedAtMs_},
              {"report", lastReport_}};
    }
    reason_ = reason.empty() ? "unspecified" : reason;
    pressedAtMs_ = nowMs();
  }
  const InstanceRegistry::StopReport report = InstanceRegistry::instance().stopAll();
  {
    std::lock_guard<std::mutex> lock(mu_);
    lastReport_ = report.toJson();
  }
  /* 4) self-shutdown: the gateway's handler persists state and quits. */
  std::function<void()> handler;
  {
    std::lock_guard<std::mutex> lock(mu_);
    handler = shutdownHandler_;
  }
  if (handler) {
    try {
      handler();
    } catch (...) {
      /* the process quit is best-effort from here */
    }
  }
  return {{"ok", true},
          {"alreadyLatched", false},
          {"latched", true},
          {"reason", reason_},
          {"pressedAtMs", pressedAtMs_},
          {"report", report.toJson()}};
}

std::string EmergencyStop::reason() const {
  std::lock_guard<std::mutex> lock(mu_);
  return reason_;
}

nlohmann::json EmergencyStop::status() const {
  std::lock_guard<std::mutex> lock(mu_);
  return {{"latched", latched_.load(std::memory_order_acquire)},
          {"reason", reason_},
          {"pressedAtMs", pressedAtMs_},
          {"registeredInstances", InstanceRegistry::instance().count()},
          {"instances", InstanceRegistry::instance().snapshot()},
          {"report", lastReport_}};
}

void EmergencyStop::setShutdownHandler(std::function<void()> fn) {
  std::lock_guard<std::mutex> lock(mu_);
  shutdownHandler_ = std::move(fn);
}

bool EmergencyStop::hasShutdownHandler() const {
  std::lock_guard<std::mutex> lock(mu_);
  return static_cast<bool>(shutdownHandler_);
}

void EmergencyStop::resetForTesting() {
  latched_.store(false, std::memory_order_release);
  std::lock_guard<std::mutex> lock(mu_);
  reason_.clear();
  pressedAtMs_ = 0;
  lastReport_ = nlohmann::json::object();
}

} /* namespace safety */
} /* namespace phoenix */
