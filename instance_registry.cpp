/* instance_registry.cpp - Implementation, see header. */
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

InstanceRegistry &InstanceRegistry::instance() {
  static InstanceRegistry reg;
  return reg;
}

uint64_t InstanceRegistry::registerInstance(const std::string &name,
                                            const std::string &kind,
                                            StopHandler stop) {
  std::lock_guard<std::mutex> lock(mu_);
  const uint64_t id = nextId_++;
  entries_[id] = Entry{id, name, kind, nowMs(), std::move(stop)};
  return id;
}

bool InstanceRegistry::unregister(uint64_t id) {
  std::lock_guard<std::mutex> lock(mu_);
  return entries_.erase(id) > 0;
}

size_t InstanceRegistry::count() const {
  std::lock_guard<std::mutex> lock(mu_);
  return entries_.size();
}

nlohmann::json InstanceRegistry::snapshot() const {
  std::lock_guard<std::mutex> lock(mu_);
  nlohmann::json out = nlohmann::json::array();
  for (const auto &kv : entries_) {
    out.push_back({{"id", kv.second.id},
                   {"name", kv.second.name},
                   {"kind", kv.second.kind},
                   {"registeredAtMs", kv.second.registeredAtMs}});
  }
  return out;
}

nlohmann::json InstanceRegistry::StopReport::toJson() const {
  return {{"total", total}, {"stopped", stopped}, {"errors", errors}};
}

InstanceRegistry::StopReport InstanceRegistry::stopAll() {
  /* snapshot handlers under the lock, then invoke OUTSIDE it: handlers may
     unregister themselves or block (killing loops, joining threads). */
  std::vector<std::pair<uint64_t, StopHandler>> handlers;
  {
    std::lock_guard<std::mutex> lock(mu_);
    handlers.reserve(entries_.size());
    for (const auto &kv : entries_) handlers.push_back({kv.first, kv.second.stop});
  }
  StopReport report;
  report.total = handlers.size();
  for (const auto &h : handlers) {
    if (!h.second) continue;
    try {
      h.second();
      ++report.stopped;
    } catch (const std::exception &e) {
      report.errors.push_back(std::string("handler #") + std::to_string(h.first) +
                              ": " + e.what());
    } catch (...) {
      report.errors.push_back(std::string("handler #") + std::to_string(h.first) +
                              ": unknown exception");
    }
  }
  return report;
}

void InstanceRegistry::clearForTesting() {
  std::lock_guard<std::mutex> lock(mu_);
  entries_.clear();
}

} /* namespace safety */
} /* namespace phoenix */
