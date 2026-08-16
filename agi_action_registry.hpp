/* agi_action_registry.hpp - Registry of executable AGI actions
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   Completes the action-space expansion: the MPC planner is no longer limited
   to the five instinct verbs.  Any capability (addon tool, memory retrieval,
   goal advancement) can be registered as a named action with an embedding;
   the planner picks among ALL of them and the chosen action is dispatched to
   the real executor (see CognitionAutonomyManager::setAgiActionExecutor).

   Theory: in active inference the action space is the agent's effector set;
   enlarging it (adding tools) enlarges the policy space over which EFE is
   minimised, so plan quality can only improve (the old actions remain
   available - monotonicity of min).
*/
#pragma once

#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include "active_inference.hpp"

namespace phoenix {
namespace agi {

/** @brief A capability the MPC planner may choose. */
struct AgiActionSpec {
  std::string name;                 /*!< planner action name. */
  std::string category{"tool"};     /*!< "instinct" | "tool" | "goal". */
  std::vector<float> embedding;     /*!< control input for the forward model. */
  std::string addonType;            /*!< for "tool": math/search/research/web. */
  std::string description;          /*!< human-readable description. */

  nlohmann::json toJson() const;
  static AgiActionSpec fromJson(const nlohmann::json &j);
};

/** @brief Ordered registry of AGI actions. */
class AgiActionRegistry {
 public:
  /** Register or replace an action by name. */
  void registerAction(AgiActionSpec spec);

  /** Remove an action; returns false if it was not present. */
  bool unregister(const std::string &name);

  /** Find an action by name (nullptr if absent). */
  const AgiActionSpec *find(const std::string &name) const;

  /** Convert the registry to the planner's Action list (name + embedding). */
  std::vector<Action> toPlannerActions() const;

  size_t size() const { return specs_.size(); }
  bool empty() const { return specs_.empty(); }

  nlohmann::json toJson() const;
  void fromJson(const nlohmann::json &j);

 private:
  std::vector<AgiActionSpec> specs_;
};

}  // namespace agi
}  // namespace phoenix
