/* motion_adapter.hpp - Mobility command adapter for edge platform
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
   along with 079 Project.  If not, see <http://www.gnu.org/licenses/>. */

#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace edge_platform::motion_adapter {

using json = nlohmann::json;

/* Trim whitespace from string */
inline std::string trimCopy(const std::string &value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

/* Convert string to lowercase */
inline std::string lowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

/* Check if haystack contains any needle */
inline bool containsAny(const std::string &haystack, std::initializer_list<const char *> needles) {
    for (const char *needle : needles) {
        if (needle != nullptr && *needle != '\0' && haystack.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

/* Clamp value to [-1, 1] */
inline double clampUnit(double value) {
    return std::max(-1.0, std::min(1.0, value));
}

/* Clamp value to [0, 1] */
inline double clampPositiveUnit(double value) {
    return std::max(0.0, std::min(1.0, value));
}

/* Clamp duration to [50, 10000] ms */
inline int clampDurationMs(int value, int fallback) {
    return std::max(50, std::min(10000, value > 0 ? value : fallback));
}

/* Check if value is nearly zero */
inline bool nearlyZero(double value) {
    return std::fabs(value) < 0.01;
}

/* Adapt mobility command with safety gating */
inline json adaptMobilityCommand(const json &request) {
    const json adapter = request.contains("actionAdapter") && request["actionAdapter"].is_object()
                             ? request["actionAdapter"]
                             : json::object();
    const json constraints = request.contains("actionConstraints") && request["actionConstraints"].is_object()
                                 ? request["actionConstraints"]
                                 : json::object();
    const json currentMotion = request.contains("currentMotion") && request["currentMotion"].is_object()
                                   ? request["currentMotion"]
                                   : json::object();

    const std::string strategicGoal = trimCopy(request.value("strategicGoal", std::string("guard locomotion stability")));
    const std::string routePolicy = trimCopy(request.value("routePolicy", std::string("follow verified waypoints")));
    const std::string speedPolicy = trimCopy(request.value("speedPolicy", std::string("stable")));
    const std::string action = lowerCopy(trimCopy(request.value("action", request.value("selectedAction", std::string("stop")))));
    const bool platformAvailable = request.value("platformAvailable", false);
    const double verifyScore = std::max(0.0, std::min(1.0, request.value("verifyScore", 1.0)));
    const double minVerifyScore = std::max(0.0, std::min(1.0, adapter.value("minVerifyScore", constraints.value("minVerifyScore", 0.0))));
    const double worldUncertainty = std::max(0.0, std::min(1.0, request.value("worldUncertainty", request.value("uncertainty", 0.0))));
    const double maxUncertainty = std::max(0.0, std::min(1.0, adapter.value("maxUncertainty", constraints.value("maxUncertainty", 1.0))));
    const int cooldownMs = std::max(0, std::min(10000, adapter.value("cooldownMs", constraints.value("cooldownMs", 0))));
    const int64_t recentDispatchAgeMs = request.value("recentDispatchAgeMs", static_cast<int64_t>(-1));
    const bool cooldownActive = cooldownMs > 0 && recentDispatchAgeMs >= 0 && recentDispatchAgeMs < cooldownMs;
    const bool explicitStop = action.empty() || containsAny(action, {"stop", "hold", "idle"});
    const bool doNotMove = request.value("doNotMove", false) || constraints.value("doNotMove", false);
    const bool verifyBlocked = minVerifyScore > 0.0 && verifyScore < minVerifyScore;
    const bool uncertaintyBlocked = worldUncertainty > maxUncertainty;
    const bool continueStabilization = request.value("continueStabilization",
                                                     adapter.value("continueStabilizationWhenIdle",
                                                                   constraints.value("continueStabilizationWhenIdle", false)));
    const double holdScale = clampPositiveUnit(adapter.value("holdScale", constraints.value("holdScale", 0.65)));
    const bool currentMotionValid = currentMotion.is_object() &&
                                    currentMotion.contains("leftDuty") && currentMotion["leftDuty"].is_number() &&
                                    currentMotion.contains("rightDuty") && currentMotion["rightDuty"].is_number();

    const double requestedLeftDuty = clampUnit(request.value("leftDuty", 0.0));
    const double requestedRightDuty = clampUnit(request.value("rightDuty", 0.0));
    const int requestedDurationMs = clampDurationMs(request.value("durationMs", 450), 450);
    const bool requestedStopAfter = request.value("stopAfter", true);
    const bool requestedCameraVerify = request.value("cameraVerify", false);
    const double requestedMotionThreshold = std::max(0.001, std::min(1.0, request.value("motionThreshold", 0.015)));

    bool allowActuation = platformAvailable && !explicitStop && !doNotMove && !verifyBlocked && !uncertaintyBlocked && !cooldownActive;
    std::string primaryReason = allowActuation ? "intent-accepted" : "platform-unavailable";
    if (!platformAvailable) {
        primaryReason = "platform-unavailable";
    } else if (explicitStop) {
        primaryReason = "explicit-stop";
    } else if (doNotMove) {
        primaryReason = "do-not-move";
    } else if (verifyBlocked) {
        primaryReason = "verify-below-threshold";
    } else if (uncertaintyBlocked) {
        primaryReason = "uncertainty-too-high";
    } else if (cooldownActive) {
        primaryReason = "cooldown-active";
    }

    std::string commandMode = "hold";
    double leftDuty = 0.0;
    double rightDuty = 0.0;
    int durationMs = requestedDurationMs;
    bool stopAfter = requestedStopAfter;
    bool cameraVerify = requestedCameraVerify;
    double motionThreshold = requestedMotionThreshold;
    bool stabilizationActive = false;

    if (allowActuation) {
        commandMode = "move";
        leftDuty = requestedLeftDuty;
        rightDuty = requestedRightDuty;
    } else if (continueStabilization && currentMotionValid) {
        commandMode = "stabilize";
        stabilizationActive = true;
        leftDuty = clampUnit(currentMotion.value("leftDuty", 0.0) * holdScale);
        rightDuty = clampUnit(currentMotion.value("rightDuty", 0.0) * holdScale);
        durationMs = clampDurationMs(currentMotion.value("durationMs", requestedDurationMs), requestedDurationMs);
        stopAfter = false;
        cameraVerify = currentMotion.value("cameraVerify", requestedCameraVerify);
        motionThreshold = std::max(0.001, std::min(1.0, currentMotion.value("motionThreshold", requestedMotionThreshold)));
    }

    if (nearlyZero(leftDuty) && nearlyZero(rightDuty)) {
        commandMode = "hold";
        stabilizationActive = false;
        leftDuty = 0.0;
        rightDuty = 0.0;
    }

    const bool willActuate = commandMode != "hold";
    std::vector<std::string> guardNotes;
    if (doNotMove) {
        guardNotes.push_back("do-not-move asserted");
    }
    if (verifyBlocked) {
        guardNotes.push_back("verify score below threshold");
    }
    if (uncertaintyBlocked) {
        guardNotes.push_back("uncertainty budget exceeded");
    }
    if (cooldownActive) {
        guardNotes.push_back("command cooldown still active");
    }
    if (stabilizationActive) {
        guardNotes.push_back("stabilization keeps the body active without a fresh move command");
    }
    if (guardNotes.empty()) {
        guardNotes.push_back("intent cleared for actuation");
    }

    return json{{"schemaVersion", 1},
                {"intent",
                 json{{"strategicGoal", strategicGoal},
                      {"selectedAction", action},
                      {"routePolicy", routePolicy},
                      {"speedPolicy", speedPolicy},
                      {"cameraVerify", requestedCameraVerify}}},
                {"gating",
                 json{{"allowActuation", allowActuation},
                      {"willActuate", willActuate},
                      {"continueStabilization", stabilizationActive},
                      {"primaryReason", primaryReason},
                      {"verifyScore", verifyScore},
                      {"minVerifyScore", minVerifyScore},
                      {"worldUncertainty", worldUncertainty},
                      {"maxUncertainty", maxUncertainty},
                      {"doNotMove", doNotMove},
                      {"cooldownMs", cooldownMs},
                      {"recentDispatchAgeMs", recentDispatchAgeMs},
                      {"cooldownActive", cooldownActive},
                      {"notes", guardNotes}}},
                {"command",
                 json{{"mode", commandMode},
                      {"leftDuty", leftDuty},
                      {"rightDuty", rightDuty},
                      {"durationMs", durationMs},
                      {"stopAfter", stopAfter},
                      {"cameraVerify", cameraVerify},
                      {"motionThreshold", motionThreshold}}}};
}

} // namespace edge_platform::motion_adapter