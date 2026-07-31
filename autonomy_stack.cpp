/* autonomy_stack.cpp - Autonomy stack implementation
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

#include "autonomy_stack.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <unordered_set>

namespace autonomy {

namespace {

int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

double clampDouble(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

int clampInt(int v, int lo, int hi) {
    return std::max(lo, std::min(hi, v));
}

std::string trimLocal(const std::string &value) {
    std::size_t start = value.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return std::string();
    }
    std::size_t end = value.find_last_not_of(" \t\r\n");
    return value.substr(start, end - start + 1);
}

std::string lowerLocal(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::string truncateLocal(const std::string &value, std::size_t maxChars) {
    if (maxChars == 0 || value.size() <= maxChars) {
        return value;
    }
    if (maxChars <= 3) {
        return value.substr(0, maxChars);
    }
    return value.substr(0, maxChars - 3) + "...";
}

void appendUniqueString(std::vector<std::string> &items,
                        std::unordered_set<std::string> &seen,
                        const std::string &value,
                        std::size_t maxCount,
                        std::size_t maxChars = 0) {
    if (items.size() >= maxCount) {
        return;
    }
    std::string normalized = trimLocal(value);
    if (maxChars > 0) {
        normalized = truncateLocal(normalized, maxChars);
    }
    if (normalized.empty()) {
        return;
    }
    const std::string lowered = lowerLocal(normalized);
    if (!seen.insert(lowered).second) {
        return;
    }
    items.push_back(normalized);
}

std::vector<std::string> collectStrings(const json &value, std::size_t maxCount) {
    std::vector<std::string> out;
    std::unordered_set<std::string> seen;
    if (value.is_string()) {
        appendUniqueString(out, seen, value.get<std::string>(), maxCount);
        return out;
    }
    if (!value.is_array()) {
        return out;
    }
    for (const auto &entry : value) {
        if (!entry.is_string()) {
            continue;
        }
        appendUniqueString(out, seen, entry.get<std::string>(), maxCount);
        if (out.size() >= maxCount) {
            break;
        }
    }
    return out;
}

json toJsonArray(const std::vector<std::string> &items) {
    json out = json::array();
    for (const auto &item : items) {
        out.push_back(item);
    }
    return out;
}

std::string joinStringsLocal(const std::vector<std::string> &items, const std::string &separator) {
    std::ostringstream oss;
    for (std::size_t index = 0; index < items.size(); ++index) {
        if (index > 0) {
            oss << separator;
        }
        oss << items[index];
    }
    return oss.str();
}

json buildSeedChecklist(const std::string &mission) {
    json checklist = json::array();
    if (!mission.empty()) {
        checklist.push_back("stay aligned with the startup mission");
        checklist.push_back("prefer actions that produce observable outcomes");
    }
    return checklist;
}

json buildSeedQuestions(const std::string &mission) {
    json questions = json::array();
    if (!mission.empty()) {
        questions.push_back("what next step best advances the startup mission with observable evidence?");
    }
    return questions;
}

json buildSeedHypotheses(const std::string &mission) {
    json hypotheses = json::array();
    if (!mission.empty()) {
        hypotheses.push_back("the startup mission can be advanced by staged planning and evidence-backed iteration");
    }
    return hypotheses;
}

json buildSeedSimulationTargets(const std::string &mission) {
    json targets = json::array();
    if (!mission.empty()) {
        targets.push_back("simulate the expected next state after the first autonomous step");
    }
    return targets;
}

json buildCognitionHead(const std::string &name,
                        int priority,
                        double weight,
                        const std::string &trigger,
                        const std::vector<std::string> &focus,
                        const std::vector<std::string> &queuedActions,
                        const std::vector<std::string> &triggerEvents) {
    return json{{"name", name},
                {"priority", clampInt(priority, 0, 100)},
                {"weight", clampDouble(weight, 0.0, 1.0)},
                {"trigger", trimLocal(trigger)},
                {"focus", toJsonArray(focus)},
                {"queuedActions", toJsonArray(queuedActions)},
                {"triggerEvents", toJsonArray(triggerEvents)},
                {"active", !focus.empty() || !queuedActions.empty() || !triggerEvents.empty()}};
}

double cognitionHeadScore(const json &head) {
    const double weight = clampDouble(head.value("weight", 0.0), 0.0, 1.0);
    const double priority = clampDouble(head.value("priority", 0) / 100.0, 0.0, 1.0);
    return clampDouble((weight * 0.65) + (priority * 0.35), 0.0, 1.0);
}

void sortCognitionHeads(std::vector<json> &heads) {
    std::sort(heads.begin(), heads.end(), [](const json &a, const json &b) {
        const double scoreA = cognitionHeadScore(a);
        const double scoreB = cognitionHeadScore(b);
        if (std::abs(scoreA - scoreB) > 1e-6) {
            return scoreA > scoreB;
        }
        const int priorityA = a.value("priority", 0);
        const int priorityB = b.value("priority", 0);
        if (priorityA != priorityB) {
            return priorityA > priorityB;
        }
        return a.value("name", std::string()) < b.value("name", std::string());
    });
}

} // namespace

json buildCognitionAutonomySeedPayload(const std::string &sessionId,
                                const std::string &mission,
                                double uncertainty) {
    const std::string normalizedSessionId = trimLocal(sessionId);
    const std::string normalizedMission = truncateLocal(trimLocal(mission), 240);
    const double boundedUncertainty = clampDouble(uncertainty, 0.0, 1.0);
    return json{{"sessionId", normalizedSessionId},
             {"reply", normalizedMission},
             {"seedMission", normalizedMission},
             {"autonomySource", "startup-seed"},
             {"worldUncertainty", boundedUncertainty},
             {"worldReflectionSuggested", boundedUncertainty >= 0.45},
             {"verify", json{{"score", 0.0}}},
             {"reasoningAgenda",
              json{{"hypotheses", buildSeedHypotheses(normalizedMission)},
                  {"openQuestions", buildSeedQuestions(normalizedMission)},
                  {"nextStep", normalizedMission.empty() ? std::string() : "decompose the startup mission into concrete subgoals and verification steps"},
                  {"shouldReflect", boundedUncertainty >= 0.45},
                  {"reflectionReason", "startup autonomy seed mission requires an initial plan"},
                  {"uncertainty", boundedUncertainty}}},
             {"responsePlan",
              json{{"goalFrame", normalizedMission},
                  {"simulationTargets", buildSeedSimulationTargets(normalizedMission)},
                  {"critiqueChecklist", buildSeedChecklist(normalizedMission)},
                  {"revisionBudget", boundedUncertainty}}}};
}

TransformerClusterManager::TransformerClusterManager() {
    nodes_.push_back(Node{"local-a", "local://transformer/a", 1.0, 0, 0, 0, 0, 80.0, true, ""});
    nodes_.push_back(Node{"local-b", "local://transformer/b", 1.0, 0, 0, 0, 0, 90.0, true, ""});
    nodes_.push_back(Node{"local-c", "local://transformer/c", 1.0, 0, 0, 0, 0, 95.0, true, ""});
}

json TransformerClusterManager::status() const {
    std::lock_guard<std::mutex> lock(mu_);
    json nodes = json::array();
    for (const auto &n : nodes_) {
        nodes.push_back(json{{"id", n.id},
                             {"endpoint", n.endpoint},
                             {"weight", n.weight},
                             {"inflight", n.inflight},
                             {"routed", n.routed},
                             {"success", n.success},
                             {"failure", n.failure},
                             {"emaLatencyMs", n.emaLatencyMs},
                             {"healthy", n.healthy},
                             {"lastError", n.lastError}});
    }
    return json{{"ok", true}, {"result", json{{"nodes", nodes}, {"nextRouteSeq", routeSeq_.load() + 1}}}};
}

json TransformerClusterManager::updateNodes(const json &payload) {
    if (!payload.contains("nodes") || !payload["nodes"].is_array()) {
        return json{{"ok", false}, {"error", "nodes array required"}};
    }
    std::unique_lock<std::mutex> lock(mu_);
    std::vector<Node> next;
    for (const auto &item : payload["nodes"]) {
        if (!item.is_object()) continue;
        Node n;
        n.id = item.value("id", std::string());
        n.endpoint = item.value("endpoint", std::string());
        n.weight = clampDouble(item.value("weight", 1.0), 0.1, 20.0);
        n.healthy = item.value("healthy", true);
        if (!n.id.empty() && !n.endpoint.empty()) {
            next.push_back(n);
        }
    }
    if (next.empty()) {
        return json{{"ok", false}, {"error", "no valid nodes"}};
    }
    nodes_ = std::move(next);
    lock.unlock();
    return status();
}

json TransformerClusterManager::pickNode(const json &request) {
    std::lock_guard<std::mutex> lock(mu_);
    if (nodes_.empty()) {
        return json{{"ok", false}, {"error", "cluster empty"}};
    }

    const int maxTokens = request.value("maxTokens", 128);
    const bool preferLowLatency = request.value("preferLowLatency", true);
    const double tokenPenalty = clampDouble(maxTokens / 512.0, 0.0, 2.0);

    int chosen = -1;
    double bestScore = std::numeric_limits<double>::max();
    for (int i = 0; i < (int)nodes_.size(); ++i) {
        auto &n = nodes_[i];
        double healthPenalty = n.healthy ? 0.0 : 1000.0;
        double inflightPenalty = (double)n.inflight / std::max(0.1, n.weight);
        double latencyPenalty = preferLowLatency ? (n.emaLatencyMs / 100.0) : 0.0;
        double score = inflightPenalty + latencyPenalty + tokenPenalty + healthPenalty;
        if (score < bestScore) {
            bestScore = score;
            chosen = i;
        }
    }

    if (chosen < 0) {
        return json{{"ok", false}, {"error", "no routable node"}};
    }

    auto &node = nodes_[chosen];
    node.inflight += 1;
    node.routed += 1;
    uint64_t rid = routeSeq_.fetch_add(1) + 1;

    return json{{"ok", true},
                {"result", json{{"routeId", rid},
                                 {"nodeId", node.id},
                                 {"endpoint", node.endpoint},
                                 {"score", bestScore},
                                 {"inflight", node.inflight}}}};
}

json TransformerClusterManager::feedback(const json &payload) {
    std::string nodeId = payload.value("nodeId", std::string());
    double latencyMs = clampDouble(payload.value("latencyMs", 120.0), 1.0, 60000.0);
    bool success = payload.value("success", true);
    std::string error = payload.value("error", std::string());

    std::lock_guard<std::mutex> lock(mu_);
    for (auto &n : nodes_) {
        if (n.id != nodeId) continue;
        n.inflight = std::max(0, n.inflight - 1);
        if (success) n.success += 1;
        else n.failure += 1;
        n.emaLatencyMs = 0.85 * n.emaLatencyMs + 0.15 * latencyMs;
        n.healthy = success || n.failure < n.success + 5;
        if (!success) n.lastError = error;
        return json{{"ok", true}, {"result", json{{"nodeId", n.id}, {"healthy", n.healthy}, {"emaLatencyMs", n.emaLatencyMs}}}};
    }

    return json{{"ok", false}, {"error", "node not found"}};
}

SpiderAutonomyManager::SpiderAutonomyManager() {
    banPatterns_ = {"javascript:", "data:text/html", "about:blank"};
}

json SpiderAutonomyManager::status() const {
    std::lock_guard<std::mutex> lock(mu_);
    return json{{"ok", true},
                {"result", json{{"enabled", enabled_},
                                 {"iteration", iteration_},
                                 {"crawlDepth", crawlDepth_},
                                 {"maxPages", maxPages_},
                                 {"intervalSec", intervalSec_},
                                 {"selfHeal", selfHeal_},
                                 {"banPatterns", banPatterns_},
                                 {"lastAdaptAtMs", lastAdaptAtMs_}}}};
}

json SpiderAutonomyManager::adapt(const json &payload, const json &monitoring) {
    std::unique_lock<std::mutex> lock(mu_);
    enabled_ = payload.value("enabled", enabled_);
    if (!enabled_) {
        lock.unlock();
        return status();
    }

    double avgMs = 0.0;
    if (monitoring.is_object() && monitoring.contains("routes") && monitoring["routes"].is_object()) {
        const auto &routes = monitoring["routes"];
        if (routes.contains("/api/chat") && routes["/api/chat"].is_object()) {
            avgMs = routes["/api/chat"].value("avgMs", 0.0);
        }
    }

    if (avgMs > 1200.0) {
        intervalSec_ = clampInt(intervalSec_ + 10, 20, 300);
        maxPages_ = clampInt((int)std::round(maxPages_ * 0.8), 64, 2048);
    } else if (avgMs > 1.0 && avgMs < 450.0) {
        intervalSec_ = clampInt(intervalSec_ - 2, 10, 300);
        maxPages_ = clampInt((int)std::round(maxPages_ * 1.1), 64, 4096);
    }

    if (payload.contains("crawlDepth") && payload["crawlDepth"].is_number_integer()) {
        crawlDepth_ = clampInt(payload["crawlDepth"].get<int>(), 1, 8);
    }
    if (payload.contains("banPatterns") && payload["banPatterns"].is_array()) {
        banPatterns_.clear();
        for (const auto &x : payload["banPatterns"]) {
            if (x.is_string()) banPatterns_.push_back(x.get<std::string>());
        }
    }

    iteration_ += 1;
    lastAdaptAtMs_ = nowMs();
    lock.unlock();
    return status();
}

OptimizerAutonomyManager::OptimizerAutonomyManager() = default;

json OptimizerAutonomyManager::status() const {
    std::lock_guard<std::mutex> lock(mu_);
    return json{{"ok", true},
                {"result", json{{"enabled", enabled_},
                                 {"iteration", iteration_},
                                 {"devicePolicy", devicePolicy_},
                                 {"workerProcesses", workerProcesses_},
                                 {"workerThreads", workerThreads_},
                                 {"useGpu", useGpu_},
                                 {"useNpu", useNpu_},
                                 {"lastIterAtMs", lastIterAtMs_}}}};
}

json OptimizerAutonomyManager::iterate(const json &payload, const json &monitoring, const json &transformerParams) {
    std::unique_lock<std::mutex> lock(mu_);
    enabled_ = payload.value("enabled", enabled_);
    if (!enabled_) {
        lock.unlock();
        return status();
    }

    double avgMs = 0.0;
    double errRate = 0.0;
    if (monitoring.is_object() && monitoring.contains("routes") && monitoring["routes"].is_object()) {
        auto routes = monitoring["routes"];
        if (routes.contains("/api/transformer/chat") && routes["/api/transformer/chat"].is_object()) {
            avgMs = routes["/api/transformer/chat"].value("avgMs", 0.0);
            errRate = routes["/api/transformer/chat"].value("errorRate", 0.0);
        }
    }

    json patch = json::object();
    int nLayers = transformerParams.value("nLayers", 4);
    int dModel = transformerParams.value("dModel", 128);

    if (avgMs > 900.0 || errRate > 0.2) {
        patch["nLayers"] = clampInt(nLayers - 1, 2, 12);
        patch["dModel"] = clampInt(dModel - 16, 64, 512);
        workerThreads_ = clampInt(workerThreads_ + 1, 2, 16);
    } else if (avgMs > 1.0 && avgMs < 350.0 && errRate < 0.05) {
        patch["nLayers"] = clampInt(nLayers + 1, 2, 12);
        patch["dModel"] = clampInt(dModel + 16, 64, 512);
    }

    if (payload.contains("devicePolicy") && payload["devicePolicy"].is_string()) {
        devicePolicy_ = payload["devicePolicy"].get<std::string>();
    }
    if (devicePolicy_ == "gpu") {
        useGpu_ = true;
        useNpu_ = false;
    } else if (devicePolicy_ == "npu") {
        useNpu_ = true;
        useGpu_ = false;
    } else if (devicePolicy_ == "hybrid") {
        useGpu_ = true;
        useNpu_ = true;
    } else {
        useGpu_ = false;
        useNpu_ = false;
    }

    iteration_ += 1;
    lastIterAtMs_ = nowMs();

    json optimizerStatus = json{{"enabled", enabled_},
                                {"iteration", iteration_},
                                {"devicePolicy", devicePolicy_},
                                {"workerProcesses", workerProcesses_},
                                {"workerThreads", workerThreads_},
                                {"useGpu", useGpu_},
                                {"useNpu", useNpu_},
                                {"lastIterAtMs", lastIterAtMs_}};

    return json{{"ok", true},
                {"result", json{{"optimizer", optimizerStatus},
                                 {"transformerPatch", patch},
                                 {"perfPlan", json{{"workerProcesses", workerProcesses_},
                                                    {"workerThreads", workerThreads_},
                                                    {"devicePolicy", devicePolicy_},
                                                    {"useGpu", useGpu_},
                                                    {"useNpu", useNpu_}}}}}};
}

json OptimizerAutonomyManager::applyPerfProfile(const json &payload, const json &transformerParams) {
    std::lock_guard<std::mutex> lock(mu_);
    std::string profile = payload.value("profile", std::string("balanced"));
    json patch = json::object();

    int nLayers = transformerParams.value("nLayers", 4);
    int dModel = transformerParams.value("dModel", 128);

    if (profile == "throughput") {
        workerThreads_ = clampInt(payload.value("workerThreads", 8), 2, 32);
        workerProcesses_ = clampInt(payload.value("workerProcesses", 2), 1, 16);
        devicePolicy_ = payload.value("devicePolicy", std::string("hybrid"));
        patch["nLayers"] = clampInt(nLayers - 1, 2, 12);
    } else if (profile == "quality") {
        workerThreads_ = clampInt(payload.value("workerThreads", 4), 2, 32);
        workerProcesses_ = clampInt(payload.value("workerProcesses", 1), 1, 16);
        devicePolicy_ = payload.value("devicePolicy", std::string("gpu"));
        patch["nLayers"] = clampInt(nLayers + 1, 2, 12);
        patch["dModel"] = clampInt(dModel + 32, 64, 512);
    } else {
        workerThreads_ = clampInt(payload.value("workerThreads", 6), 2, 32);
        workerProcesses_ = clampInt(payload.value("workerProcesses", 1), 1, 16);
        devicePolicy_ = payload.value("devicePolicy", std::string("auto"));
    }

    useGpu_ = devicePolicy_ == "gpu" || devicePolicy_ == "hybrid";
    useNpu_ = devicePolicy_ == "npu" || devicePolicy_ == "hybrid";

    iteration_ += 1;
    lastIterAtMs_ = nowMs();
    return json{{"ok", true},
                {"result", json{{"profile", profile},
                                 {"perfPlan", json{{"workerProcesses", workerProcesses_}, {"workerThreads", workerThreads_}, {"devicePolicy", devicePolicy_}, {"useGpu", useGpu_}, {"useNpu", useNpu_}}},
                                 {"transformerPatch", patch}}}};
}

json OptimizerAutonomyManager::proposeGnnUpgrade(const json &payload, const json &transformerParams) {
    std::lock_guard<std::mutex> lock(mu_);
    std::string level = payload.value("level", std::string("v2"));
    json patch = json::object();
    patch["enableDynamicReasoning"] = true;
    if (level == "v3") {
        patch["simpleTokenThreshold"] = clampInt(transformerParams.value("simpleTokenThreshold", 10) + 2, 6, 48);
        patch["nLayers"] = clampInt(transformerParams.value("nLayers", 4) + 1, 2, 12);
    } else {
        patch["simpleTokenThreshold"] = clampInt(transformerParams.value("simpleTokenThreshold", 10) + 1, 6, 48);
    }
    iteration_ += 1;
    lastIterAtMs_ = nowMs();
    return json{{"ok", true}, {"result", json{{"upgrade", "gnn"}, {"level", level}, {"transformerPatch", patch}}}};
}

json OptimizerAutonomyManager::proposeTransformerUpgrade(const json &payload, const json &transformerParams) {
    std::lock_guard<std::mutex> lock(mu_);
    std::string arch = payload.value("arch", std::string("modern"));
    json patch = json::object();
    int nLayers = transformerParams.value("nLayers", 4);
    int dModel = transformerParams.value("dModel", 128);

    if (arch == "efficient") {
        patch["nLayers"] = clampInt(nLayers - 1, 2, 12);
        patch["dModel"] = clampInt(dModel - 16, 64, 512);
    } else if (arch == "reasoning") {
        patch["nLayers"] = clampInt(nLayers + 2, 2, 12);
        patch["dModel"] = clampInt(dModel + 32, 64, 512);
        patch["enableDynamicReasoning"] = true;
    } else {
        patch["nLayers"] = clampInt(nLayers + 1, 2, 12);
        patch["dModel"] = clampInt(dModel + 16, 64, 512);
    }

    iteration_ += 1;
    lastIterAtMs_ = nowMs();
    return json{{"ok", true}, {"result", json{{"upgrade", "transformer"}, {"arch", arch}, {"transformerPatch", patch}}}};
}

json OptimizerAutonomyManager::modernizeTransformer(const json &payload, const json &transformerParams) {
    std::lock_guard<std::mutex> lock(mu_);
    std::string profile = payload.value("profile", std::string("sota-balanced"));
    json patch = json::object();

    int nLayers = transformerParams.value("nLayers", 4);
    int dModel = transformerParams.value("dModel", 128);
    int nHeads = transformerParams.value("nHeads", 4);

    patch["enableMoE"] = true;
    patch["enableMLA"] = true;
    patch["enableCoefAttention"] = true;
    patch["enableHierEmbedding"] = true;
    patch["enableMultiTokenObjective"] = true;
    patch["enableCoTScaffold"] = true;
    patch["enableProgramSynthesis"] = true;
    patch["enableDynamicReasoning"] = true;
    patch["dynamicSampling"] = true;
    patch["useAlibi"] = true;
    patch["repetitionSegmented"] = true;

    if (profile == "sota-reasoning") {
        patch["nLayers"] = clampInt(nLayers + 2, 4, 12);
        patch["dModel"] = clampInt(dModel + 32, 128, 512);
        patch["nHeads"] = clampInt(nHeads + 2, 4, 16);
        patch["moeExperts"] = 8;
        patch["moeTopK"] = 2;
        patch["mlaRank"] = 16;
        patch["multiTokenHeads"] = 4;
        patch["cotSteps"] = 5;
        patch["draftTokens"] = 3;
        patch["windowClipTokens"] = 384;
        patch["attnChunkSize"] = 128;
    } else if (profile == "sota-efficient") {
        patch["nLayers"] = clampInt(nLayers, 3, 10);
        patch["dModel"] = clampInt(dModel, 96, 320);
        patch["nHeads"] = clampInt(nHeads, 4, 12);
        patch["moeExperts"] = 4;
        patch["moeTopK"] = 1;
        patch["mlaRank"] = 8;
        patch["multiTokenHeads"] = 2;
        patch["cotSteps"] = 3;
        patch["draftTokens"] = 2;
        patch["windowClipTokens"] = 256;
        patch["attnChunkSize"] = 64;
    } else {
        patch["nLayers"] = clampInt(nLayers + 1, 4, 12);
        patch["dModel"] = clampInt(dModel + 16, 128, 512);
        patch["nHeads"] = clampInt(nHeads + 1, 4, 16);
        patch["moeExperts"] = 6;
        patch["moeTopK"] = 2;
        patch["mlaRank"] = 12;
        patch["multiTokenHeads"] = 3;
        patch["cotSteps"] = 4;
        patch["draftTokens"] = 2;
        patch["windowClipTokens"] = 320;
        patch["attnChunkSize"] = 96;
    }

    iteration_ += 1;
    lastIterAtMs_ = nowMs();
    return json{{"ok", true},
                {"result", json{{"upgrade", "transformer-modern"},
                                 {"profile", profile},
                                 {"strategy", "moe+mla+coef-attention+hier-embedding+multi-token"},
                                 {"transformerPatch", patch}}}};
}

CognitionAutonomyManager::CognitionAutonomyManager()
    : promptComposer_(phoenix::prompt::SystemPrompt::arthurDefault(),
                      phoenix::prompt::MemoryPrompt::empty()),
      instinctEngine_(phoenix::instinct::InstinctEngine::defaultEngine()) {}

json CognitionAutonomyManager::status() const {
    std::lock_guard<std::mutex> lock(mu_);
    json sessions = json::array();
    for (const auto &entry : sessions_) {
        const auto &record = entry.second;
        sessions.push_back(json{{"sessionId", entry.first},
                                {"shouldIterate", record.value("shouldIterate", false)},
                                {"priority", record.value("priority", 0.0)},
                                {"dominantHead", record.value("dominantHead", std::string())},
                                {"headCount", record.value("heads", json::array()).size()},
                                {"lastUncertainty", record.value("lastUncertainty", 0.0)},
                                {"lastVerifyScore", record.value("lastVerifyScore", 0.0)},
                                {"lastObservedAtMs", record.value("lastObservedAtMs", 0ll)}});
    }
    std::sort(sessions.begin(), sessions.end(), [](const json &a, const json &b) {
        return a.value("priority", 0.0) > b.value("priority", 0.0);
    });
    if (sessions.size() > 12) {
        sessions.erase(sessions.begin() + 12, sessions.end());
    }
    return json{{"ok", true},
                {"result", json{{"enabled", enabled_},
                                 {"backgroundEnabled", backgroundEnabled_},
                                 {"iteration", iteration_},
                                 {"observations", observations_},
                                 {"backgroundEvery", backgroundEvery_},
                                 {"uncertaintyThreshold", uncertaintyThreshold_},
                                 {"reflectionThreshold", reflectionThreshold_},
                                 {"lastIterAtMs", lastIterAtMs_},
                                 {"sessionsTracked", sessions_.size()},
                                 {"sensations", sensationEngine_.toJson()},
                                 {"instincts", instinctEngine_.toJson()},
                                 {"lastBenefitHarmBias", lastBenefitHarmBias_},
                                 {"mixedModalInputSize", inputBuffer_.size()},
                                 {"mixedModalOutputSize", outputQueue_.size()},
                                 {"channels", channelRegistry_.toJson()},
                                 {"sessions", sessions}}}};
}

json CognitionAutonomyManager::observe(const json &payload, const json &worldState) {
    std::unique_lock<std::mutex> lock(mu_);
    enabled_ = payload.value("enabled", enabled_);
    backgroundEnabled_ = payload.value("backgroundEnabled", backgroundEnabled_);
    backgroundEvery_ = clampInt(payload.value("backgroundEvery", backgroundEvery_), 1, 128);
    uncertaintyThreshold_ = clampDouble(payload.value("uncertaintyThreshold", uncertaintyThreshold_), 0.0, 1.0);
    reflectionThreshold_ = clampDouble(payload.value("reflectionThreshold", reflectionThreshold_), 0.0, 1.0);

    /* v7.0 primal sensation / mixed-modal I/O ingestion */
    if (payload.contains("sensation") && payload["sensation"].is_object()) {
        sensationEngine_.add(phoenix::primal::PrimalSensation::fromJson(payload["sensation"]));
    }
    if (payload.contains("mixedModalPacket") && payload["mixedModalPacket"].is_object()) {
        auto packet = phoenix::io::MixedModalPacket::fromJson(payload["mixedModalPacket"]);
        inputBuffer_.push(packet);
        if (!packet.source.empty()) {
            channelRegistry_.registerSource(packet.source, packet.mimeType);
        }
    }

    std::string sessionId = trimLocal(payload.value("sessionId", std::string()));
    if (sessionId.empty() && worldState.is_object()) {
        sessionId = trimLocal(worldState.value("sessionId", std::string()));
    }
    if (sessionId.empty()) {
        lock.unlock();
        return status();
    }

    json reasoningAgenda = payload.value("reasoningAgenda", json::object());
    json responsePlan = payload.value("responsePlan", json::object());
    double uncertainty = payload.value("worldUncertainty", reasoningAgenda.value("uncertainty", 0.0));
    bool reflectionSuggested = payload.value("worldReflectionSuggested", reasoningAgenda.value("shouldReflect", false));
    double verifyScore = payload.value("verifyScore", 0.0);
    if (verifyScore <= 0.0 && payload.contains("verify") && payload["verify"].is_object()) {
        verifyScore = payload["verify"].value("score", 0.0);
    }
    auto &record = sessions_[sessionId];
    if (!record.is_object()) {
        record = json::object();
    }
    const int previousObservations = record.value("observations", 0);
    const double previousAverageUncertainty = record.value("avgUncertainty", uncertainty);

    record["sessionId"] = sessionId;
    record["observations"] = previousObservations + 1;
    observations_ += 1;
    record["lastObservedAtMs"] = nowMs();
    record["lastUncertainty"] = uncertainty;
    record["avgUncertainty"] = ((previousAverageUncertainty * previousObservations) + uncertainty) /
                                std::max(1, previousObservations + 1);
    record["lastVerifyScore"] = verifyScore;
    record["reflectionSuggested"] = reflectionSuggested;
    record["worldEvidenceCount"] = worldState.value("evidenceCount", worldState.value("recentEvidence", json::array()).size());
    if (reasoningAgenda.is_object()) {
        record["lastAgenda"] = reasoningAgenda;
    }
    if (responsePlan.is_object()) {
        record["lastResponsePlan"] = responsePlan;
    }

    std::vector<std::string> subgoals;
    std::unordered_set<std::string> subgoalSeen;
    const auto questions = collectStrings(reasoningAgenda.value("openQuestions", json::array()), 4);
    const auto hypotheses = collectStrings(reasoningAgenda.value("hypotheses", json::array()), 4);
    const auto contradictions = collectStrings(reasoningAgenda.value("contradictions", json::array()), 4);
    const auto simulationTargets = collectStrings(responsePlan.value("simulationTargets", json::array()), 4);
    const auto critiqueChecklist = collectStrings(responsePlan.value("critiqueChecklist", json::array()), 4);
    const std::string nextStep = trimLocal(reasoningAgenda.value("nextStep", std::string()));
    const std::string goalFrame = trimLocal(responsePlan.value("goalFrame", std::string()));
    const double revisionBudget = clampDouble(responsePlan.value("revisionBudget", uncertainty), 0.0, 1.0);
    json mobilityPlan = responsePlan.value("mobilityPlan", worldState.value("mobilityPlan", json::object()));

    std::vector<std::string> mobilitySimulationTargets;
    std::vector<std::string> mobilityWaypoints;
    std::string mobilityGoal;
    std::string mobilitySpeedPolicy;
    if (mobilityPlan.is_object()) {
        mobilityGoal = trimLocal(mobilityPlan.value("strategicGoal", std::string()));
        if (mobilityPlan.contains("bigBrain") && mobilityPlan["bigBrain"].is_object()) {
            mobilitySimulationTargets = collectStrings(mobilityPlan["bigBrain"].value("simulationTargets", json::array()), 4);
            mobilityWaypoints = collectStrings(mobilityPlan["bigBrain"].value("routeWaypoints", json::array()), 4);
        }
        if (mobilityPlan.contains("littleBrain") && mobilityPlan["littleBrain"].is_object()) {
            mobilitySpeedPolicy = trimLocal(mobilityPlan["littleBrain"].value("speedPolicy", std::string()));
        }
    }

    if (!goalFrame.empty()) {
        appendUniqueString(subgoals, subgoalSeen, goalFrame, 6, 120);
    }
    if (!nextStep.empty()) {
        appendUniqueString(subgoals, subgoalSeen, nextStep, 6, 120);
    }
    if (!mobilityGoal.empty()) {
        appendUniqueString(subgoals, subgoalSeen, "ground-route goal: " + mobilityGoal, 6, 120);
    }
    for (const auto &waypoint : mobilityWaypoints) {
        appendUniqueString(subgoals, subgoalSeen, "route waypoint: " + waypoint, 6, 120);
    }
    for (const auto &question : questions) {
        appendUniqueString(subgoals, subgoalSeen, "resolve question: " + question, 6, 120);
    }

    std::vector<std::string> hypothesisBacklog;
    std::unordered_set<std::string> backlogSeen;
    for (const auto &item : hypotheses) {
        appendUniqueString(hypothesisBacklog, backlogSeen, item, 6, 120);
    }
    for (const auto &item : contradictions) {
        appendUniqueString(hypothesisBacklog, backlogSeen, "check contradiction: " + item, 6, 120);
    }

    std::vector<std::string> reflectionTasks;
    std::unordered_set<std::string> reflectionSeen;
    if (reflectionSuggested) {
        appendUniqueString(reflectionTasks,
                           reflectionSeen,
                           "reflect on: " + reasoningAgenda.value("reflectionReason", std::string("uncertainty is high")),
                           6,
                           120);
    }
    if (verifyScore > 0.0 && verifyScore < reflectionThreshold_) {
        appendUniqueString(reflectionTasks,
                           reflectionSeen,
                           "repair low-verify answer sections before the next response",
                           6,
                           120);
    }
    for (const auto &item : critiqueChecklist) {
        appendUniqueString(reflectionTasks, reflectionSeen, item, 6, 120);
    }
    if (!mobilitySpeedPolicy.empty()) {
        appendUniqueString(reflectionTasks,
                           reflectionSeen,
                           "check rover stability policy: " + mobilitySpeedPolicy,
                           6,
                           120);
    }
    std::vector<std::string> combinedSimulationTargets = simulationTargets;
    std::unordered_set<std::string> combinedSimulationSeen;
    for (const auto &item : combinedSimulationTargets) {
        combinedSimulationSeen.insert(lowerLocal(item));
    }
    for (const auto &item : mobilitySimulationTargets) {
        appendUniqueString(combinedSimulationTargets, combinedSimulationSeen, item, 8, 120);
    }

    const double verifySignal = verifyScore > 0.0 ? clampDouble(1.0 - verifyScore, 0.0, 1.0) : 0.0;
    const double basePriority = clampDouble(std::max({uncertainty,
                                                      revisionBudget,
                                                      reflectionSuggested ? 0.7 : 0.0,
                                                      verifySignal}),
                                            0.0,
                                            1.0);

    std::vector<json> heads;
    auto addHead = [&](const std::string &name,
                       int priority,
                       double weight,
                       const std::string &trigger,
                       const std::vector<std::string> &focus,
                       const std::vector<std::string> &queuedActions,
                       const std::vector<std::string> &triggerEvents) {
        if (focus.empty() && queuedActions.empty() && triggerEvents.empty()) {
            return;
        }
        heads.push_back(buildCognitionHead(name, priority, weight, trigger, focus, queuedActions, triggerEvents));
    };

    {
        std::vector<std::string> evidenceFocus;
        std::unordered_set<std::string> focusSeen;
        for (const auto &item : questions) {
            appendUniqueString(evidenceFocus, focusSeen, item, 6, 120);
        }
        if (uncertainty >= uncertaintyThreshold_) {
            appendUniqueString(evidenceFocus, focusSeen, "collect missing world evidence under high uncertainty", 6, 120);
        }
        std::vector<std::string> evidenceEvents;
        std::unordered_set<std::string> eventSeen;
        if (uncertainty >= uncertaintyThreshold_) {
            appendUniqueString(evidenceEvents, eventSeen, "high-uncertainty", 6, 48);
        }
        if (!questions.empty()) {
            appendUniqueString(evidenceEvents, eventSeen, "open-questions", 6, 48);
        }
        addHead("evidence",
                uncertainty >= uncertaintyThreshold_ ? 88 : 68,
                std::max(uncertainty, 0.25 + (questions.empty() ? 0.0 : 0.12)),
                uncertainty >= uncertaintyThreshold_ ? "high-uncertainty" : "open-questions",
                evidenceFocus,
                (uncertainty >= uncertaintyThreshold_ || !questions.empty()) ? std::vector<std::string>{"collect-evidence"} : std::vector<std::string>{},
                evidenceEvents);
    }

    {
        std::vector<std::string> goalEvents;
        std::unordered_set<std::string> eventSeen;
        if (!goalFrame.empty()) {
            appendUniqueString(goalEvents, eventSeen, "goal-frame", 6, 48);
        }
        if (!nextStep.empty()) {
            appendUniqueString(goalEvents, eventSeen, "next-step", 6, 48);
        }
        if (!mobilityGoal.empty()) {
            appendUniqueString(goalEvents, eventSeen, "mobility-goal", 6, 48);
        }
        addHead("goal",
                74 + static_cast<int>(std::min<std::size_t>(4, subgoals.size())),
                clampDouble(0.32 + (revisionBudget * 0.38) + (subgoals.empty() ? 0.0 : 0.16), 0.0, 1.0),
                !nextStep.empty() ? "next-step" : "goal-frame",
                subgoals,
                subgoals.empty() ? std::vector<std::string>{} : std::vector<std::string>{"stabilize-subgoals"},
                goalEvents);
    }

    {
        std::vector<std::string> hypothesisEvents;
        std::unordered_set<std::string> eventSeen;
        if (!contradictions.empty()) {
            appendUniqueString(hypothesisEvents, eventSeen, "contradiction-present", 6, 48);
        }
        if (!hypotheses.empty()) {
            appendUniqueString(hypothesisEvents, eventSeen, "hypothesis-refresh", 6, 48);
        }
        if (verifySignal > 0.0) {
            appendUniqueString(hypothesisEvents, eventSeen, "low-verify", 6, 48);
        }
        addHead("hypothesis",
                contradictions.empty() ? 78 : 86,
                clampDouble(std::max({0.28, uncertainty * 0.72, verifySignal}), 0.0, 1.0),
                !contradictions.empty() ? "contradiction-present" : "hypothesis-refresh",
                hypothesisBacklog,
                hypothesisBacklog.empty() ? std::vector<std::string>{} : std::vector<std::string>{"refresh-hypotheses"},
                hypothesisEvents);
    }

    {
        std::vector<std::string> reflectionEvents;
        std::unordered_set<std::string> eventSeen;
        if (reflectionSuggested) {
            appendUniqueString(reflectionEvents, eventSeen, "reflection-suggested", 6, 48);
        }
        if (verifySignal > 0.0 && verifyScore < reflectionThreshold_) {
            appendUniqueString(reflectionEvents, eventSeen, "low-verify", 6, 48);
        }
        addHead("reflection",
                reflectionSuggested ? 84 : 76,
                clampDouble(std::max({0.25, reflectionSuggested ? 0.72 : 0.0, verifySignal}), 0.0, 1.0),
                reflectionSuggested ? "reflection-suggested" : "critique-checklist",
                reflectionTasks,
                reflectionTasks.empty() ? std::vector<std::string>{} : std::vector<std::string>{"run-reflection"},
                reflectionEvents);
    }

    {
        std::vector<std::string> simulationEvents;
        std::unordered_set<std::string> eventSeen;
        if (!simulationTargets.empty()) {
            appendUniqueString(simulationEvents, eventSeen, "simulation-requested", 6, 48);
        }
        if (!mobilitySimulationTargets.empty()) {
            appendUniqueString(simulationEvents, eventSeen, "mobility-simulation", 6, 48);
        }
        addHead("simulation",
                72 + static_cast<int>(std::min<std::size_t>(6, combinedSimulationTargets.size())),
                clampDouble(0.30 + (revisionBudget * 0.35) + (combinedSimulationTargets.empty() ? 0.0 : 0.18), 0.0, 1.0),
                "simulation-requested",
                combinedSimulationTargets,
                combinedSimulationTargets.empty() ? std::vector<std::string>{} : std::vector<std::string>{"run-simulation"},
                simulationEvents);
    }

    {
        std::vector<std::string> mobilityFocus;
        std::unordered_set<std::string> focusSeen;
        if (!mobilityGoal.empty()) {
            appendUniqueString(mobilityFocus, focusSeen, mobilityGoal, 6, 120);
        }
        for (const auto &waypoint : mobilityWaypoints) {
            appendUniqueString(mobilityFocus, focusSeen, waypoint, 6, 120);
        }
        if (!mobilitySpeedPolicy.empty()) {
            appendUniqueString(mobilityFocus, focusSeen, "speed policy: " + mobilitySpeedPolicy, 6, 120);
        }
        std::vector<std::string> mobilityEvents;
        std::unordered_set<std::string> eventSeen;
        if (!mobilityGoal.empty() || !mobilityWaypoints.empty()) {
            appendUniqueString(mobilityEvents, eventSeen, "mobility-plan-present", 6, 48);
        }
        std::vector<std::string> mobilityActions;
        if (!mobilityWaypoints.empty() || !mobilityGoal.empty()) {
            mobilityActions.push_back("plan-ground-route");
        }
        if (!mobilitySpeedPolicy.empty()) {
            mobilityActions.push_back("stabilize-locomotion");
        }
        addHead("mobility",
                80,
                clampDouble(0.42 + (revisionBudget * 0.22) + ((mobilityGoal.empty() && mobilityWaypoints.empty()) ? 0.0 : 0.16),
                            0.0,
                            1.0),
                "mobility-plan-present",
                mobilityFocus,
                mobilityActions,
                mobilityEvents);
    }

    sortCognitionHeads(heads);

    json headArray = json::array();
    std::vector<std::string> actionQueue;
    std::unordered_set<std::string> actionSeen;
    std::vector<std::string> eventQueue;
    std::unordered_set<std::string> eventSeen;
    for (const auto &head : heads) {
        headArray.push_back(head);
        for (const auto &queuedAction : collectStrings(head.value("queuedActions", json::array()), 4)) {
            appendUniqueString(actionQueue, actionSeen, queuedAction, 8, 48);
        }
        for (const auto &triggerEvent : collectStrings(head.value("triggerEvents", json::array()), 8)) {
            appendUniqueString(eventQueue, eventSeen, triggerEvent, 12, 48);
        }
    }

    const double headDrivenPriority = heads.empty() ? 0.0 : cognitionHeadScore(heads.front());
    const double priority = clampDouble(std::max(basePriority, headDrivenPriority), 0.0, 1.0);
    bool shouldIterate = enabled_ && backgroundEnabled_ && (priority >= 0.45 || !actionQueue.empty());

    record["subgoals"] = toJsonArray(subgoals);
    record["hypothesisBacklog"] = toJsonArray(hypothesisBacklog);
    record["simulationRequests"] = toJsonArray(combinedSimulationTargets);
    record["reflectionTasks"] = toJsonArray(reflectionTasks);
    record["actionQueue"] = toJsonArray(actionQueue);
    record["eventQueue"] = toJsonArray(eventQueue);
    record["heads"] = headArray;
    record["dominantHead"] = heads.empty() ? std::string() : heads.front().value("name", std::string());
    if (mobilityPlan.is_object() && !mobilityPlan.empty()) {
        record["mobilityPlan"] = mobilityPlan;
    }
    record["priority"] = priority;
    record["shouldIterate"] = shouldIterate;

    return json{{"ok", true},
                {"result", json{{"sessionId", sessionId},
                                 {"shouldIterate", shouldIterate},
                                 {"priority", priority},
                                 {"dominantHead", record.value("dominantHead", std::string())},
                                 {"heads", record["heads"]},
                                 {"eventQueue", record["eventQueue"]},
                                 {"subgoals", record["subgoals"]},
                                 {"hypothesisBacklog", record["hypothesisBacklog"]},
                                 {"simulationRequests", record["simulationRequests"]},
                                 {"reflectionTasks", record["reflectionTasks"]},
                                 {"actionQueue", record["actionQueue"]},
                                 {"mobilityPlan", record.value("mobilityPlan", json::object())},
                                 {"lastUncertainty", uncertainty},
                                 {"lastVerifyScore", verifyScore}}}};
}

json CognitionAutonomyManager::iterate(const json &payload, const json &worldState) {
    std::unique_lock<std::mutex> lock(mu_);
    enabled_ = payload.value("enabled", enabled_);
    backgroundEnabled_ = payload.value("backgroundEnabled", backgroundEnabled_);
    backgroundEvery_ = clampInt(payload.value("backgroundEvery", backgroundEvery_), 1, 128);
    uncertaintyThreshold_ = clampDouble(payload.value("uncertaintyThreshold", uncertaintyThreshold_), 0.0, 1.0);
    reflectionThreshold_ = clampDouble(payload.value("reflectionThreshold", reflectionThreshold_), 0.0, 1.0);
    if (!enabled_) {
        lock.unlock();
        return status();
    }

    std::string requestedSessionId = trimLocal(payload.value("sessionId", std::string()));
    if (requestedSessionId.empty() && worldState.is_object()) {
        requestedSessionId = trimLocal(worldState.value("sessionId", std::string()));
    }

    std::vector<std::string> targetSessions;
    if (!requestedSessionId.empty()) {
        targetSessions.push_back(requestedSessionId);
    } else {
        for (const auto &entry : sessions_) {
            if (entry.second.value("shouldIterate", false)) {
                targetSessions.push_back(entry.first);
            }
        }
    }

    if (targetSessions.empty()) {
        return json{{"ok", true},
                    {"result", json{{"iteration", iteration_},
                                     {"sessions", json::array()},
                                     {"worldEvidence", json::array()},
                                     {"runtimeFeaturePatch", json::object()}}}};
    }

    json sessions = json::array();
    json worldEvidence = json::array();
    json runtimeFeaturePatch = json::object();
    if (payload.value("suggestRuntimePatch", true)) {
        runtimeFeaturePatch["reasoningPlannerEnabled"] = true;
        runtimeFeaturePatch["reasoningCriticEnabled"] = true;
        runtimeFeaturePatch["cognitionAutonomyEnabled"] = true;
    }

    for (const auto &sessionId : targetSessions) {
        auto it = sessions_.find(sessionId);
        if (it == sessions_.end() || !it->second.is_object()) {
            continue;
        }
        auto &record = it->second;
        const auto actions = collectStrings(record.value("actionQueue", json::array()), 8);
        const auto subgoals = collectStrings(record.value("subgoals", json::array()), 6);
        const auto simulationRequests = collectStrings(record.value("simulationRequests", json::array()), 6);
        const auto reflectionTasks = collectStrings(record.value("reflectionTasks", json::array()), 6);
        const auto hypothesisBacklog = collectStrings(record.value("hypothesisBacklog", json::array()), 6);
        std::vector<json> orderedHeads;
        if (record.contains("heads") && record["heads"].is_array()) {
            for (const auto &head : record["heads"]) {
                if (head.is_object()) {
                    orderedHeads.push_back(head);
                }
            }
        }
        sortCognitionHeads(orderedHeads);
        json scheduledHeads = json::array();
        for (const auto &head : orderedHeads) {
            scheduledHeads.push_back(json{{"name", head.value("name", std::string())},
                                          {"priority", head.value("priority", 0)},
                                          {"weight", head.value("weight", 0.0)},
                                          {"score", cognitionHeadScore(head)},
                                          {"trigger", head.value("trigger", std::string())},
                                          {"triggerEvents", head.value("triggerEvents", json::array())},
                                          {"queuedActions", head.value("queuedActions", json::array())}});
        }
        json mobilityPlan = record.value("mobilityPlan", json::object());
        record["mobilityPlan"] = mobilityPlan;

        sessions.push_back(json{{"sessionId", sessionId},
                                {"priority", record.value("priority", 0.0)},
                                {"dominantHead", record.value("dominantHead", std::string())},
                                {"scheduledHeads", scheduledHeads},
                                {"actionQueue", record.value("actionQueue", json::array())},
                                {"subgoals", record.value("subgoals", json::array())},
                                {"hypothesisBacklog", record.value("hypothesisBacklog", json::array())}});

        auto findHead = [&](const std::string &name) -> json {
            for (const auto &head : orderedHeads) {
                if (head.value("name", std::string()) == name) {
                    return head;
                }
            }
            return json::object();
        };

        auto pushEvidence = [&](const std::string &kind, const std::string &summary, const json &head) {
            json metadata{{"source", "gateway/cognition/autonomy"},
                          {"kind", kind},
                          {"priority", record.value("priority", 0.0)}};
            if (head.is_object() && !head.empty()) {
                metadata["head"] = head.value("name", std::string());
                metadata["headPriority"] = head.value("priority", 0);
                metadata["headWeight"] = head.value("weight", 0.0);
                metadata["headScore"] = cognitionHeadScore(head);
                metadata["trigger"] = head.value("trigger", std::string());
                metadata["triggerEvents"] = head.value("triggerEvents", json::array());
            }
            worldEvidence.push_back(json{{"sessionId", sessionId},
                                         {"modality", "cognition"},
                                         {"graphSummary", summary},
                                         {"metadata", metadata}});
        };

        if (!subgoals.empty()) {
            pushEvidence("subgoals", "cognition_subgoals: " + joinStringsLocal(subgoals, " || "), findHead("goal"));
        }
        if (!hypothesisBacklog.empty()) {
            pushEvidence("hypotheses", "cognition_hypotheses: " + joinStringsLocal(hypothesisBacklog, " || "), findHead("hypothesis"));
        }
        if (!simulationRequests.empty()) {
            pushEvidence("simulation", "cognition_simulation: " + joinStringsLocal(simulationRequests, " || "), findHead("simulation"));
        }
        if (!reflectionTasks.empty()) {
            pushEvidence("reflection", "cognition_reflection: " + joinStringsLocal(reflectionTasks, " || "), findHead("reflection"));
        }
        if (mobilityPlan.is_object() && !mobilityPlan.empty()) {
            std::string mobilitySummary = mobilityPlan.value("strategicGoal", std::string("ground-route planning active"));
            if (mobilityPlan.contains("bigBrain") && mobilityPlan["bigBrain"].is_object()) {
                const std::string routePolicy = mobilityPlan["bigBrain"].value("routePolicy", std::string());
                if (!routePolicy.empty()) {
                    mobilitySummary += mobilitySummary.empty() ? routePolicy : std::string(" | ") + routePolicy;
                }
            }
            if (mobilityPlan.contains("littleBrain") && mobilityPlan["littleBrain"].is_object()) {
                const std::string speedPolicy = mobilityPlan["littleBrain"].value("speedPolicy", std::string());
                if (!speedPolicy.empty()) {
                    mobilitySummary += mobilitySummary.empty() ? speedPolicy : std::string(" | ") + speedPolicy;
                }
            }
            json mobilityHead = findHead("mobility");
            json mobilityMetadata{{"source", "gateway/cognition/autonomy"},
                                  {"kind", "mobility"},
                                  {"priority", record.value("priority", 0.0)},
                                  {"mobilityPlan", mobilityPlan}};
            if (mobilityHead.is_object() && !mobilityHead.empty()) {
                mobilityMetadata["head"] = mobilityHead.value("name", std::string());
                mobilityMetadata["headPriority"] = mobilityHead.value("priority", 0);
                mobilityMetadata["headWeight"] = mobilityHead.value("weight", 0.0);
                mobilityMetadata["headScore"] = cognitionHeadScore(mobilityHead);
                mobilityMetadata["trigger"] = mobilityHead.value("trigger", std::string());
                mobilityMetadata["triggerEvents"] = mobilityHead.value("triggerEvents", json::array());
            }
            worldEvidence.push_back(json{{"sessionId", sessionId},
                                         {"modality", "cognition"},
                                         {"graphSummary", "cognition_mobility: " + mobilitySummary},
                                         {"metadata", mobilityMetadata}});
        }

        record["iteration"] = record.value("iteration", 0) + 1;
        record["lastIterAtMs"] = nowMs();
        record["shouldIterate"] = false;
        record["lastActionQueue"] = toJsonArray(actions);
        record["lastScheduledHeads"] = scheduledHeads;
    }

    /* v7.0 instinct / benefit-harm evaluation and prompt split update */
    float dtSec = 1.0f;
    if (lastIterAtMs_ > 0) {
        dtSec = static_cast<float>(std::max<int64_t>(0, nowMs() - lastIterAtMs_)) / 1000.0f;
        if (dtSec <= 0.0f) dtSec = 1.0f;
    }
    instinctEngine_.update(sensationEngine_.active(), dtSec);
    auto bh = instinctEngine_.evaluate(sensationEngine_.active());

    /* v7.0 affect signal: export the emotion operation weight vector as a
       numeric matrix rather than an explicit action word or emotional label. */
    std::string driveWeights = json(bh.driveVector).dump();
    lastBenefitHarmBias_ = driveWeights;

    phoenix::prompt::MemoryPrompt memory;
    memory.driveVector = bh.driveVector;
    memory.emotionTensor = phoenix::instinct::InstinctEngine::driveToEmotion(bh);
    memory.inferenceOptions = memory.emotionTensor.inferenceOptions();
    memory.benefitHarmBias = bh.recommendedAction;  // human-readable action label
    if (memory.benefitHarmBias.empty())
        memory.benefitHarmBias = memory.emotionTensor.modulationHint();
    memory.summary = "Benefit=" + std::to_string(bh.benefitScore) +
                     " Harm=" + std::to_string(bh.harmScore) +
                     " Net=" + std::to_string(bh.netUtility);
    promptComposer_.setMemory(memory);

    std::string composedPrompt;
    std::string cognitionModulation;
    if (payload.contains("userPrompt") && payload["userPrompt"].is_string()) {
        composedPrompt = promptComposer_.compose(payload["userPrompt"].get<std::string>(), true);
        cognitionModulation = promptComposer_.modulationHint();
    }

    auto outbound = outputQueue_.drain(0);
    nlohmann::json mixedModalOutputs = nlohmann::json::array();
    for (const auto &p : outbound) mixedModalOutputs.push_back(p.toJson());

    iteration_ += 1;
    lastIterAtMs_ = nowMs();
    return json{{"ok", true},
                {"result", json{{"iteration", iteration_},
                                 {"sessions", sessions},
                                 {"scheduledHeads", sessions.empty() ? json::array() : sessions.front().value("scheduledHeads", json::array())},
                                 {"worldEvidence", worldEvidence},
                                 {"runtimeFeaturePatch", runtimeFeaturePatch},
                                 {"benefitHarm", bh.toJson()},
                                 {"emotionTensor", memory.emotionTensor.toJson()},
                                 {"driveVector", memory.driveVector},
                                 {"inferenceOptions", memory.inferenceOptions},
                                 {"cognitionModulation", cognitionModulation},
                                 {"mixedModalOutputs", mixedModalOutputs},
                                 {"composedPrompt", composedPrompt}}}};
}

json CognitionAutonomyManager::session(const std::string &sessionId) const {
    std::lock_guard<std::mutex> lock(mu_);
    const std::string trimmedSessionId = trimLocal(sessionId);
    auto it = sessions_.find(trimmedSessionId);
    if (it == sessions_.end()) {
        return json{{"ok", false}, {"error", "session not found"}};
    }
    return json{{"ok", true}, {"result", it->second}};
}

json CognitionAutonomyManager::exportState() const {
    std::lock_guard<std::mutex> lock(mu_);
    json sessions = json::object();
    for (const auto &entry : sessions_) {
        sessions[entry.first] = entry.second;
    }
    return json{{"enabled", enabled_},
                {"backgroundEnabled", backgroundEnabled_},
                {"iteration", iteration_},
                {"observations", observations_},
                {"sessions", sessions}};
}

json CognitionAutonomyManager::importState(const json &state) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!state.is_object()) {
        return json{{"ok", false}, {"error", "state must be an object"}};
    }

    enabled_ = state.value("enabled", enabled_);
    backgroundEnabled_ = state.value("backgroundEnabled", backgroundEnabled_);
    iteration_ = std::max(0, state.value("iteration", iteration_));
    observations_ = std::max(0, state.value("observations", observations_));
    // Additional state variables can be restored here

    sessions_.clear();
    std::size_t sessionsLoaded = 0;
    if (state.contains("sessions") && state["sessions"].is_object()) {
        for (const auto &entry : state["sessions"].items()) {
            const std::string sessionId = trimLocal(entry.key());
            if (sessionId.empty() || !entry.value().is_object()) {
                continue;
            }
            auto record = entry.value();
            record["sessionId"] = sessionId;
            sessions_[sessionId] = std::move(record);
            sessionsLoaded += 1;
        }
    }

    return json{{"ok", true},
                {"result", json{{"sessionsLoaded", sessionsLoaded},
                                 {"iteration", iteration_},
                                 {"observations", observations_}}}};
}

json CognitionAutonomyManager::ingestSensation(const json &payload) {
    std::lock_guard<std::mutex> lock(mu_);
    phoenix::primal::PrimalSensation s = phoenix::primal::PrimalSensation::fromJson(payload);
    sensationEngine_.add(s);
    return json{{"ok", true}, {"result", s.toJson()}};
}

json CognitionAutonomyManager::evaluateInstincts() {
    std::lock_guard<std::mutex> lock(mu_);
    instinctEngine_.update(sensationEngine_.active(), 1.0f);
    auto bh = instinctEngine_.evaluate(sensationEngine_.active());
    return json{{"ok", true}, {"result", bh.toJson()}};
}

json CognitionAutonomyManager::composePrompt(const json &payload) {
    std::lock_guard<std::mutex> lock(mu_);
    std::string userPrompt = payload.value("userPrompt", std::string());
    bool includeMemory = payload.value("includeMemory", true);
    std::string composed = promptComposer_.compose(userPrompt, includeMemory);
    return json{{"ok", true},
                {"result", json{{"prompt", composed},
                                 {"messages", promptComposer_.composeMessages(userPrompt, includeMemory)}}}};
}

/**
 * Ingest a MixedModalPacket, convert it to a SemanticUnit, and buffer the raw packet.
 * The packet is also registered in the channel registry when it carries a source.
 */
json CognitionAutonomyManager::ingestMixedModalPacket(const json &payload) {
    std::lock_guard<std::mutex> lock(mu_);
    auto packet = phoenix::io::MixedModalPacket::fromJson(payload);
    const size_t targetDim = payload.value("targetDim", static_cast<size_t>(64));
    const std::string contentHint = payload.value("contentHint", std::string());
    auto unit = phoenix::io::MixedModalConceptBridge::encode(packet, targetDim, contentHint);
    inputBuffer_.push(packet);
    if (!packet.source.empty()) {
        channelRegistry_.registerSource(packet.source, packet.mimeType);
    }
    return json{{"ok", true},
                {"result", json{{"packet", packet.toJson()},
                                 {"semanticUnit", unit.toJson()},
                                 {"conceptBridge", phoenix::io::MixedModalConceptBridge::status()}}}};
}

/**
 * Pretrain the persistent speech concept model from an audio packet and transcript.
 * Returns whether alignment was updated and persisted.
 */
json CognitionAutonomyManager::pretrainSpeechConcept(const json &payload) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!payload.contains("audio") || !payload["audio"].is_object()) {
        return json{{"ok", false}, {"error", "missing audio packet"}};
    }
    const auto audio = phoenix::io::MixedModalPacket::fromJson(payload["audio"]);
    const std::string transcript = payload.value("transcript", std::string());
    const size_t targetDim = payload.value("targetDim", static_cast<size_t>(64));
    const bool trained = phoenix::io::MixedModalConceptBridge::pretrainSpeech(audio, transcript, targetDim);
    return json{{"ok", trained}, {"result", phoenix::io::MixedModalConceptBridge::status()}};
}

/**
 * Convert a SemanticUnit to a target-modality MixedModalPacket and enqueue it.
 */
json CognitionAutonomyManager::emitMixedModalOutput(const json &payload) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!payload.contains("semanticUnit") || !payload["semanticUnit"].is_object()) {
        return json{{"ok", false}, {"error", "missing semanticUnit"}};
    }
    const auto unit = phoenix::multimodal::SemanticUnit::fromJson(payload["semanticUnit"]);
    const auto target = phoenix::io::MixedModalPacket::stringToModality(payload.value("targetModality", std::string("text")));
    if (target == phoenix::io::MixedModalModality::Unknown) {
        return json{{"ok", false}, {"error", "unknown target modality"}};
    }
    auto packet = phoenix::io::MixedModalConceptBridge::decode(unit, target, payload.value("source", std::string()));
    outputQueue_.push(packet);
    return json{{"ok", true}, {"result", packet.toJson()}};
}

/**
 * Drain up to `max` outbound mixed-modal packets (0 = all) and return them as JSON.
 */
json CognitionAutonomyManager::drainMixedModalOutputs(const json &payload) {
    std::lock_guard<std::mutex> lock(mu_);
    size_t max = payload.value("max", 0);
    auto packets = outputQueue_.drain(max);
    nlohmann::json arr = nlohmann::json::array();
    for (const auto &p : packets) arr.push_back(p.toJson());
    return json{{"ok", true}, {"result", arr}};
}

DatasetCatalogManager::DatasetCatalogManager() {
    filePath_ = "doc/external_dataset_index.json";
    load();
}

std::string DatasetCatalogManager::nowIso() {
    auto tp = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(tp);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

void DatasetCatalogManager::load() {
    std::lock_guard<std::mutex> lock(mu_);
    namespace fs = std::filesystem;
    if (!fs::exists(filePath_)) {
        catalog_ = json{{"activeDataset", ""},
                        {"datasets", json::array()},
                        {"internalCorpus", json{{"testsDir", "tests"}, {"robotsDir", "robots"}, {"sampleCount", 0}}},
                        {"cleaningProfile", json{{"enabled", true}, {"maxChars", 2048}, {"removeControlChars", true}, {"normalizeSpace", true}, {"dropIllegalUtf8", true}}},
                        {"collection", json{{"lastCollectAt", ""}, {"lastSources", json::array()}, {"lastAdded", 0}}},
                        {"updatedAt", nowIso()}};
        return;
    }
    try {
        std::ifstream in(filePath_);
        in >> catalog_;
        if (!catalog_.is_object()) {
            catalog_ = json{{"activeDataset", ""}, {"datasets", json::array()}, {"updatedAt", nowIso()}};
        }
        if (!catalog_.contains("datasets") || !catalog_["datasets"].is_array()) {
            catalog_["datasets"] = json::array();
        }
        if (!catalog_.contains("internalCorpus") || !catalog_["internalCorpus"].is_object()) {
            catalog_["internalCorpus"] = json{{"testsDir", "tests"}, {"robotsDir", "robots"}, {"sampleCount", 0}};
        }
        if (!catalog_.contains("cleaningProfile") || !catalog_["cleaningProfile"].is_object()) {
            catalog_["cleaningProfile"] = json{{"enabled", true}, {"maxChars", 2048}, {"removeControlChars", true}, {"normalizeSpace", true}, {"dropIllegalUtf8", true}};
        }
        if (!catalog_.contains("collection") || !catalog_["collection"].is_object()) {
            catalog_["collection"] = json{{"lastCollectAt", ""}, {"lastSources", json::array()}, {"lastAdded", 0}};
        }
    } catch (...) {
        catalog_ = json{{"activeDataset", ""},
                        {"datasets", json::array()},
                        {"internalCorpus", json{{"testsDir", "tests"}, {"robotsDir", "robots"}, {"sampleCount", 0}}},
                        {"cleaningProfile", json{{"enabled", true}, {"maxChars", 2048}, {"removeControlChars", true}, {"normalizeSpace", true}, {"dropIllegalUtf8", true}}},
                        {"collection", json{{"lastCollectAt", ""}, {"lastSources", json::array()}, {"lastAdded", 0}}},
                        {"updatedAt", nowIso()}};
    }
}

void DatasetCatalogManager::persist() const {
    namespace fs = std::filesystem;
    fs::create_directories(fs::path(filePath_).parent_path());
    std::ofstream out(filePath_);
    out << catalog_.dump(2);
}

json DatasetCatalogManager::status() const {
    std::lock_guard<std::mutex> lock(mu_);
    return json{{"ok", true},
                {"result", json{{"file", filePath_},
                                 {"activeDataset", catalog_.value("activeDataset", std::string())},
                                 {"datasetCount", catalog_.contains("datasets") && catalog_["datasets"].is_array() ? (int)catalog_["datasets"].size() : 0},
                                 {"updatedAt", catalog_.value("updatedAt", std::string())}}}};
}

json DatasetCatalogManager::list() const {
    std::lock_guard<std::mutex> lock(mu_);
    return json{{"ok", true},
                {"result", json{{"file", filePath_},
                                 {"activeDataset", catalog_.value("activeDataset", std::string())},
                                 {"datasets", catalog_.value("datasets", json::array())},
                                 {"internalCorpus", catalog_.value("internalCorpus", json::object())},
                                 {"cleaningProfile", catalog_.value("cleaningProfile", json::object())},
                                 {"collection", catalog_.value("collection", json::object())},
                                 {"omittedByDefault", true},
                                 {"note", "large datasets are external-indexed and not vendored in repository"}}}};
}

json DatasetCatalogManager::registerDataset(const json &payload) {
    std::string id = payload.value("id", std::string());
    std::string uri = payload.value("uri", std::string());
    if (id.empty() || uri.empty()) {
        return json{{"ok", false}, {"error", "id and uri required"}};
    }

    std::lock_guard<std::mutex> lock(mu_);
    json item{{"id", id},
              {"uri", uri},
              {"checksum", payload.value("checksum", std::string())},
              {"size", payload.value("size", std::string())},
              {"domain", payload.value("domain", std::string("general"))},
              {"registeredAt", nowIso()}};

    bool updated = false;
    for (auto &x : catalog_["datasets"]) {
        if (x.is_object() && x.value("id", std::string()) == id) {
            x = item;
            updated = true;
            break;
        }
    }
    if (!updated) catalog_["datasets"].push_back(item);

    if (catalog_.value("activeDataset", std::string()).empty()) {
        catalog_["activeDataset"] = id;
    }
    catalog_["updatedAt"] = nowIso();
    persist();
    return json{{"ok", true},
                {"result", json{{"file", filePath_},
                                 {"activeDataset", catalog_.value("activeDataset", std::string())},
                                 {"datasets", catalog_.value("datasets", json::array())},
                                 {"internalCorpus", catalog_.value("internalCorpus", json::object())},
                                 {"cleaningProfile", catalog_.value("cleaningProfile", json::object())},
                                 {"collection", catalog_.value("collection", json::object())},
                                 {"omittedByDefault", true},
                                 {"note", "large datasets are external-indexed and not vendored in repository"}}}};
}

json DatasetCatalogManager::activate(const json &payload) {
    std::string id = payload.value("id", std::string());
    if (id.empty()) {
        return json{{"ok", false}, {"error", "id required"}};
    }

    std::lock_guard<std::mutex> lock(mu_);
    bool exists = false;
    for (const auto &x : catalog_["datasets"]) {
        if (x.is_object() && x.value("id", std::string()) == id) {
            exists = true;
            break;
        }
    }
    if (!exists) {
        return json{{"ok", false}, {"error", "dataset not found"}};
    }
    catalog_["activeDataset"] = id;
    catalog_["updatedAt"] = nowIso();
    persist();
    return json{{"ok", true},
                {"result", json{{"file", filePath_},
                                 {"activeDataset", catalog_.value("activeDataset", std::string())},
                                 {"datasets", catalog_.value("datasets", json::array())},
                                 {"internalCorpus", catalog_.value("internalCorpus", json::object())},
                                 {"cleaningProfile", catalog_.value("cleaningProfile", json::object())},
                                 {"collection", catalog_.value("collection", json::object())},
                                 {"omittedByDefault", true},
                                 {"note", "large datasets are external-indexed and not vendored in repository"}}}};
}

json DatasetCatalogManager::updateCleaningProfile(const json &payload) {
    std::lock_guard<std::mutex> lock(mu_);
    auto &cp = catalog_["cleaningProfile"];
    cp["enabled"] = payload.value("enabled", cp.value("enabled", true));
    cp["maxChars"] = clampInt(payload.value("maxChars", cp.value("maxChars", 2048)), 128, 65536);
    cp["removeControlChars"] = payload.value("removeControlChars", cp.value("removeControlChars", true));
    cp["normalizeSpace"] = payload.value("normalizeSpace", cp.value("normalizeSpace", true));
    cp["dropIllegalUtf8"] = payload.value("dropIllegalUtf8", cp.value("dropIllegalUtf8", true));
    catalog_["updatedAt"] = nowIso();
    persist();
    return json{{"ok", true},
                {"result", json{{"file", filePath_},
                                 {"activeDataset", catalog_.value("activeDataset", std::string())},
                                 {"datasets", catalog_.value("datasets", json::array())},
                                 {"internalCorpus", catalog_.value("internalCorpus", json::object())},
                                 {"cleaningProfile", catalog_.value("cleaningProfile", json::object())},
                                 {"collection", catalog_.value("collection", json::object())},
                                 {"omittedByDefault", true},
                                 {"note", "large datasets are external-indexed and not vendored in repository"}}}};
}

json DatasetCatalogManager::collectData(const json &payload) {
    std::lock_guard<std::mutex> lock(mu_);
    json sources = payload.value("sources", json::array({"tests", "robots", "external-index"}));
    if (!sources.is_array()) {
        sources = json::array({"tests", "robots", "external-index"});
    }
    int estimatedAdded = 0;
    for (const auto &s : sources) {
        if (!s.is_string()) continue;
        const std::string name = s.get<std::string>();
        if (name == "tests") estimatedAdded += 40;
        else if (name == "robots") estimatedAdded += 120;
        else if (name == "external-index") estimatedAdded += 500;
    }
    auto &ic = catalog_["internalCorpus"];
    ic["sampleCount"] = ic.value("sampleCount", 0) + estimatedAdded;
    catalog_["collection"] = json{{"lastCollectAt", nowIso()}, {"lastSources", sources}, {"lastAdded", estimatedAdded}};
    catalog_["updatedAt"] = nowIso();
    persist();
    return json{{"ok", true}, {"result", json{{"added", estimatedAdded}, {"catalog", catalog_}}}};
}

json DatasetCatalogManager::governance() const {
    std::lock_guard<std::mutex> lock(mu_);
    const auto &ic = catalog_["internalCorpus"];
    const auto &datasets = catalog_["datasets"];
    return json{{"ok", true},
                {"result", json{{"activeDataset", catalog_.value("activeDataset", std::string())},
                                 {"internalSmallDataset", ic},
                                 {"externalLargeDatasets", datasets},
                                 {"cleaningProfile", catalog_.value("cleaningProfile", json::object())},
                                 {"collection", catalog_.value("collection", json::object())}}}};
}

} // namespace autonomy
