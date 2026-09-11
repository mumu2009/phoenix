#pragma once

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>
#include <vector>

namespace phoenix {
namespace runtime_supervisor {

inline std::vector<std::string> builtinDenyResumeMissions() {
  return {"2678077", "6477259", "9374278"};
}

inline std::string normalizeMissionId(std::string_view raw) {
  std::string out;
  out.reserve(raw.size());
  for (unsigned char ch : raw) {
    if (std::isdigit(ch))
      out.push_back(static_cast<char>(ch));
  }
  return out;
}

struct ServiceRecord {
  std::string role;
  unsigned long long pid = 0;
  int port = 0;
  bool alive = false;
  std::string lastCleanCheckpoint;
};

struct Checkpoint {
  int version = 1;
  std::string writtenAt;
  std::vector<ServiceRecord> services;
  std::vector<std::string> denyResumeMissions = builtinDenyResumeMissions();
  std::string lastMissionId;
  bool resumeMissions = false;
};

struct SupervisorPolicy {
  bool leaveLlamaIfAlive = true;
  bool neverKillLlama = true;
  bool neverAssignMissionOnRestart = true;
  bool restartDeadNonLlama = true;
  bool restartDeadLlama = true;
};

enum class RestartAction {
  Leave,
  Start,
};

struct RestartDecision {
  std::string role;
  RestartAction action = RestartAction::Leave;
  std::string reason;
};

inline bool roleIsLlama(std::string_view role) {
  return role == "llama" || role == "inference";
}

inline nlohmann::json serviceToJson(const ServiceRecord &s) {
  return nlohmann::json{{"role", s.role},
                        {"pid", s.pid},
                        {"port", s.port},
                        {"alive", s.alive},
                        {"lastCleanCheckpoint", s.lastCleanCheckpoint}};
}

inline bool serviceFromJson(const nlohmann::json &j, ServiceRecord &out) {
  if (!j.is_object())
    return false;
  ServiceRecord tmp;
  if (j.contains("role") && j["role"].is_string())
    tmp.role = j["role"].get<std::string>();
  if (j.contains("pid") && j["pid"].is_number_unsigned())
    tmp.pid = j["pid"].get<unsigned long long>();
  else if (j.contains("pid") && j["pid"].is_number_integer() &&
           j["pid"].get<long long>() >= 0)
    tmp.pid = static_cast<unsigned long long>(j["pid"].get<long long>());
  if (j.contains("port") && j["port"].is_number_integer())
    tmp.port = j["port"].get<int>();
  if (j.contains("alive") && j["alive"].is_boolean())
    tmp.alive = j["alive"].get<bool>();
  if (j.contains("lastCleanCheckpoint") && j["lastCleanCheckpoint"].is_string())
    tmp.lastCleanCheckpoint = j["lastCleanCheckpoint"].get<std::string>();
  if (tmp.role.empty())
    return false;
  out = tmp;
  return true;
}

inline nlohmann::json toJson(const Checkpoint &cp) {
  nlohmann::json services = nlohmann::json::array();
  for (const auto &s : cp.services)
    services.push_back(serviceToJson(s));
  nlohmann::json deny = nlohmann::json::array();
  for (const auto &id : cp.denyResumeMissions)
    deny.push_back(id);
  return nlohmann::json{{"version", cp.version},
                        {"writtenAt", cp.writtenAt},
                        {"services", services},
                        {"denyResumeMissions", deny},
                        {"lastMissionId", cp.lastMissionId},
                        {"resumeMissions", cp.resumeMissions},
                        {"product", "phoenix"}};
}

inline bool fromJson(const nlohmann::json &j, Checkpoint &out) {
  if (!j.is_object())
    return false;
  Checkpoint tmp;
  if (j.contains("version") && j["version"].is_number_integer())
    tmp.version = j["version"].get<int>();
  if (j.contains("writtenAt") && j["writtenAt"].is_string())
    tmp.writtenAt = j["writtenAt"].get<std::string>();
  if (j.contains("lastMissionId") && j["lastMissionId"].is_string())
    tmp.lastMissionId = j["lastMissionId"].get<std::string>();
  if (j.contains("resumeMissions") && j["resumeMissions"].is_boolean())
    tmp.resumeMissions = j["resumeMissions"].get<bool>();
  if (j.contains("denyResumeMissions") && j["denyResumeMissions"].is_array()) {
    tmp.denyResumeMissions.clear();
    for (const auto &item : j["denyResumeMissions"]) {
      if (item.is_string())
        tmp.denyResumeMissions.push_back(item.get<std::string>());
      else if (item.is_number_integer())
        tmp.denyResumeMissions.push_back(std::to_string(item.get<long long>()));
    }
  }
  if (j.contains("services") && j["services"].is_array()) {
    for (const auto &item : j["services"]) {
      ServiceRecord rec;
      if (serviceFromJson(item, rec))
        tmp.services.push_back(rec);
    }
  }
  out = tmp;
  return true;
}

inline std::string serialize(const Checkpoint &cp) { return toJson(cp).dump(2); }

inline bool deserialize(const std::string &text, Checkpoint &out) {
  if (!nlohmann::json::accept(text))
    return false;
  nlohmann::json j = nlohmann::json::parse(text, nullptr, false);
  if (j.is_discarded())
    return false;
  return fromJson(j, out);
}

inline bool isDeniedMission(const Checkpoint &cp, std::string_view missionId) {
  const std::string want = normalizeMissionId(missionId);
  if (want.empty())
    return false;
  auto match = [&](const std::string &raw) {
    return normalizeMissionId(raw) == want;
  };
  if (std::any_of(cp.denyResumeMissions.begin(), cp.denyResumeMissions.end(),
                  match))
    return true;
  const auto builtin = builtinDenyResumeMissions();
  return std::any_of(builtin.begin(), builtin.end(), match);
}

inline bool canRestoreMission(const Checkpoint &cp, std::string_view missionId) {
  if (missionId.empty())
    return false;
  if (isDeniedMission(cp, missionId))
    return false;
  if (!cp.resumeMissions)
    return false;
  return true;
}

inline bool mayTerminateRole(const SupervisorPolicy &policy,
                             std::string_view role) {
  if (roleIsLlama(role) && policy.neverKillLlama)
    return false;
  return !roleIsLlama(role);
}

inline std::vector<RestartDecision>
planRestart(const Checkpoint &observed, const SupervisorPolicy &policy) {
  std::vector<RestartDecision> out;
  for (const auto &svc : observed.services) {
    RestartDecision d;
    d.role = svc.role;
    if (svc.alive) {
      d.action = RestartAction::Leave;
      if (roleIsLlama(svc.role) && policy.leaveLlamaIfAlive)
        d.reason = "llama_alive_leave";
      else
        d.reason = "service_alive_leave";
    } else if (roleIsLlama(svc.role)) {
      if (policy.restartDeadLlama) {
        d.action = RestartAction::Start;
        d.reason = "llama_dead_start_only";
      } else {
        d.action = RestartAction::Leave;
        d.reason = "llama_dead_no_autostart";
      }
    } else if (policy.restartDeadNonLlama) {
      d.action = RestartAction::Start;
      d.reason = "service_dead_start";
    } else {
      d.action = RestartAction::Leave;
      d.reason = "autostart_disabled";
    }
    out.push_back(d);
  }
  return out;
}

inline bool planWouldTerminateLlama(const std::vector<RestartDecision> &plan,
                                    const SupervisorPolicy &policy) {
  if (!policy.neverKillLlama)
    return false;
  for (const auto &d : plan) {
    if (roleIsLlama(d.role) && d.action != RestartAction::Leave &&
        d.reason.find("dead") == std::string::npos)
      return true;
  }
  return false;
}

inline bool shouldAssignMissionOnRestart(const Checkpoint &cp,
                                         const SupervisorPolicy &policy) {
  (void)cp;
  if (policy.neverAssignMissionOnRestart)
    return false;
  return false;
}

} // namespace runtime_supervisor
} // namespace phoenix
