/* agi_action_registry.cpp - AGI action registry implementation
   Copyright (C) 2026 079 Project */

#include "agi_action_registry.hpp"

#include <algorithm>

namespace phoenix {
namespace agi {

nlohmann::json AgiActionSpec::toJson() const {
  return {{"name", name},
          {"category", category},
          {"embedding", embedding},
          {"addonType", addonType},
          {"description", description}};
}

AgiActionSpec AgiActionSpec::fromJson(const nlohmann::json &j) {
  AgiActionSpec s;
  if (!j.is_object()) return s;
  if (j.contains("name") && j["name"].is_string()) s.name = j["name"].get<std::string>();
  if (j.contains("category") && j["category"].is_string()) s.category = j["category"].get<std::string>();
  if (j.contains("embedding") && j["embedding"].is_array()) {
    for (const auto &e : j["embedding"]) {
      if (e.is_number()) s.embedding.push_back(e.get<float>());
    }
  }
  if (j.contains("addonType") && j["addonType"].is_string()) s.addonType = j["addonType"].get<std::string>();
  if (j.contains("description") && j["description"].is_string()) s.description = j["description"].get<std::string>();
  return s;
}

void AgiActionRegistry::registerAction(AgiActionSpec spec) {
  for (auto &existing : specs_) {
    if (existing.name == spec.name) {
      existing = std::move(spec);
      return;
    }
  }
  specs_.push_back(std::move(spec));
}

bool AgiActionRegistry::unregister(const std::string &name) {
  const auto it = std::find_if(specs_.begin(), specs_.end(),
                               [&name](const AgiActionSpec &s) { return s.name == name; });
  if (it == specs_.end()) return false;
  specs_.erase(it);
  return true;
}

const AgiActionSpec *AgiActionRegistry::find(const std::string &name) const {
  for (const auto &s : specs_) {
    if (s.name == name) return &s;
  }
  return nullptr;
}

std::vector<Action> AgiActionRegistry::toPlannerActions() const {
  std::vector<Action> out;
  out.reserve(specs_.size());
  for (const auto &s : specs_) {
    Action a;
    a.name = s.name;
    a.embedding = s.embedding;
    out.push_back(std::move(a));
  }
  return out;
}

nlohmann::json AgiActionRegistry::toJson() const {
  nlohmann::json arr = nlohmann::json::array();
  for (const auto &s : specs_) arr.push_back(s.toJson());
  return arr;
}

void AgiActionRegistry::fromJson(const nlohmann::json &j) {
  specs_.clear();
  if (!j.is_array()) return;
  for (const auto &item : j) specs_.push_back(AgiActionSpec::fromJson(item));
}

}  // namespace agi
}  // namespace phoenix
