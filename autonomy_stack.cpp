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

#include "addons/builtin_registry.hpp"
#include "phoenix_config.hpp"

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

void padActionEmbedding(std::vector<float> &emb, size_t targetDim) {
    while (emb.size() < targetDim) emb.push_back(0.0f);
    if (emb.size() > targetDim) emb.resize(targetDim);
}

std::vector<float> oneHotEmbedding(size_t index, size_t dim) {
    std::vector<float> out(dim, 0.0f);
    if (index < dim) out[index] = 1.0f;
    return out;
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
      instinctEngine_(phoenix::instinct::InstinctEngine::defaultEngine()),
      addonManager_(addon::createDefaultAddons()) {
    // Default AGI executor: real tool dispatch (math/search/shell) + goal tracking.
    agiActionExecutor_ = [this](const phoenix::agi::AgiActionSpec &spec, const json &ctx) {
        return this->executeAgiAction(spec, ctx);
    };
    registerDefaultAgiActions();
    try {
        missionCtxTokens_ = phoenix::cfgOr<int>("mission.ctxSize",
            phoenix::cfgOr<int>("llama_server.ctx_size", 4096));
        missionContextPack_ = phoenix::cfgOr<std::string>("mission.contextPack",
                                                          std::string("full_and_summary"));
        if (missionContextPack_ != "summary" && missionContextPack_ != "full_and_summary")
            missionContextPack_ = "full_and_summary";
        missionIncludeGnnSummary_ = phoenix::cfgOr<bool>("mission.includeGnnSummary", false);
    } catch (...) {
    }
}

CognitionAutonomyManager::~CognitionAutonomyManager() {
    /* never leave the heartbeat thread running */
    loopStop_.store(true, std::memory_order_release);
    if (loopThread_.joinable()) loopThread_.join();
    unregisterFromSafetyRegistry();
}

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
                                 {"agi", agiEnabled_ ? agiController_.status() : json{{"enabled", false}}},
                                 {"subconscious", subconsciousEnabled_ ? subProfile_.toJson() : json{{"enabled", false}}},
                                 {"agiActions", agiActionRegistry_.toJson()},
                                 {"agiGoals", goals_},
                                 {"mixedModalInputSize", inputBuffer_.size()},
                                 {"mixedModalOutputSize", outputQueue_.size()},
                                 {"channels", channelRegistry_.toJson()},
                                 {"sessions", sessions}}}};
}


void CognitionAutonomyManager::registerDefaultAgiActions() {
    // Four real capabilities: math, search, computer shell, goal advancement.
    // They use a fixed 4-D one-hot action embedding so the latent transition
    // model can learn distinct effects per tool.
    struct DefaultAction {
        std::string name;
        std::string category;
        std::string addonType;
        std::string description;
        size_t oneHotIndex;
    };
    const DefaultAction defaults[] = {
        {"math", "tool", "math", "Evaluate arithmetic expressions", 0},
        {"search", "tool", "search", "Retrieve indexed or online knowledge", 1},
        {"computer", "tool", "computer", "List or read files on the local computer", 2},
        {"goal_advance", "goal", "", "Push the current mission goal forward", 3},
        {"replicate", "replicate", "replicate",
         "Replicate: summon a successor instance (mutated genome, same goal)", 4},
    };
    for (const auto &d : defaults) {
        if (agiActionRegistry_.find(d.name) != nullptr) continue;  // idempotent
        phoenix::agi::AgiActionSpec spec;
        spec.name = d.name;
        spec.category = d.category;
        spec.addonType = d.addonType;
        spec.description = d.description;
        spec.embedding = oneHotEmbedding(d.oneHotIndex, 5);
        agiActionRegistry_.registerAction(std::move(spec));
    }
}

nlohmann::json CognitionAutonomyManager::executeAgiAction(
    const phoenix::agi::AgiActionSpec &spec, const json &context) {
    if (spec.category == "instinct") {
        return json{{"ok", true}, {"result", json{{"noop", true}, {"reason", "instinct verb"}}}};
    }

    if (spec.category == "goal") {
        const std::string goal = context.value("userPrompt", spec.description);
        goals_.push_back(goal);
        if (goals_.size() > 1024) goals_.erase(goals_.begin());
        return json{{"ok", true},
                    {"result",
                     json{{"goal", spec.name}, {"description", goal}, {"goals", goals_}}}};
    }

    if (spec.category == "replicate") {
        /* v8.2: replicate = summon a Meeseeks-BOX TOOL.  The sub-task comes
           from context.userPrompt; empty/identical requests degrade to
           "assist with: <parent goal>" inside recordChild, so the parent
           can never hand its whole job to a child. */
        if (!missionEnabled_ || !mission_.active()) {
            return json{{"ok", false}, {"error", "no active mission to replicate for"}};
        }
        try {
            const std::string subgoal = context.value("userPrompt", std::string());
            /* parentId lets a HELPER BOX summon its own boxes (task tree);
               empty = spawned by the root mission.  Depth is capped only if
               the user configured mission.maxReplicaDepth > 0. */
            const std::string parentId = context.value("parentId", std::string());
            const auto childRec = mission_.recordChild(missionGenome_,
                                                       missionMutationRate_, subgoal, parentId);
            const phoenix::mission::Mission snap = mission_.mission();
            const std::string childId =
                "child-" + snap.id + "-" + std::to_string(mission_.children().size());
            nlohmann::json childSession;
            childSession["sessionId"] = childId;
            childSession["observations"] = 1;
            childSession["seedMission"] = childRec.goal;
            childSession["lastObservedAtMs"] = nowMs();
            childSession["shouldIterate"] = true;
            childSession["missionGenome"] = childRec.genome.toJson();
            childSession["spawnedBy"] = "instance-replicate";
            childSession["parentBoxId"] = parentId;
            childSession["boxId"] = childRec.id;
            sessions_[childId] = childSession;
            return json{{"ok", true},
                        {"result", json{{"replicated", true},
                                        {"childSessionId", childId},
                                        {"boxId", childRec.id},
                                        {"goal", childRec.goal},
                                        {"parentGoal", snap.goal},
                                        {"genome", childRec.genome.toJson()},
                                        {"childrenCount", mission_.children().size()},
                                        {"reply", "summoned helper box " + childRec.id +
                                                  " for: " + childRec.goal}}}};
        } catch (const std::runtime_error &e) {
            return json{{"ok", false}, {"error", e.what()}};
        }
    }

    if (spec.category == "mcp") {
        /* MCP tool: dispatch through the running server session. */
        auto it = mcpActionMap_.find(spec.name);
        if (it == mcpActionMap_.end()) {
            return json{{"ok", false}, {"error", "mcp action not registered: " + spec.name}};
        }
        nlohmann::json args = nlohmann::json::object();
        if (context.contains("mcpArguments") && context["mcpArguments"].is_object()) {
            args = context["mcpArguments"];
        }
        nlohmann::json out = mcpManager_.callTool(it->second.first, it->second.second, args);
        /* flatten MCP text content into a reply for the prompt pipeline */
        std::string reply;
        if (out.contains("content") && out["content"].is_array()) {
            for (const auto &c : out["content"]) {
                if (c.is_object() && c.value("type", "") == "text") {
                    if (!reply.empty()) reply += "\n";
                    reply += c.value("text", std::string());
                }
            }
        }
        return json{{"ok", out.value("ok", false)},
                    {"result", json{{"server", it->second.first},
                                    {"tool", it->second.second},
                                    {"reply", reply},
                                    {"mcp", out}}}};
    }

    if (spec.category != "tool" || !addonManager_) {
        return json{{"ok", false}, {"error", "no executor for category " + spec.category}};
    }

    std::string userPrompt = context.value("userPrompt", std::string());
    std::string text = userPrompt;
    if (spec.addonType == "math" && !text.empty() && text.find("math:") != 0 && text.find("calc:") != 0) {
        text = "math: " + text;
    } else if (spec.addonType == "search" && !text.empty() && text.find("search:") != 0) {
        text = "search: " + text;
    } else if (spec.addonType == "computer" && !text.empty() && text.find("computer:") != 0 &&
               text.find("shell:") != 0) {
        text = "computer: " + text;
    }

    json payload = context;
    if (!spec.addonType.empty()) payload["__addonType"] = spec.addonType;

    addon::AddonResult res = addonManager_->run(text, payload);
    if (!res.handled) {
        return json{{"ok", false}, {"error", "tool not handled: " + spec.addonType}};
    }

    json out = json{{"ok", true},
                    {"result",
                     json{{"addon", res.meta.value("addon", spec.addonType)},
                          {"name", spec.name},
                          {"reply", res.reply},
                          {"meta", res.meta},
                          {"extraTokens", res.extraTokens}}}};
    return out;
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
    /* v7.0 autonomous evolution: turn the world model's uncertainty into a
       real epistemic signal.  With agi.enabled, every observation ingests a
       Novelty sensation scaled by the reported uncertainty, so the TD(0)
       reward (the benefit-harm netUtility) reflects genuine environment
       dynamics instead of staying neutral in text-only loops. */
    if (agiEnabled_ && payload.contains("worldUncertainty") &&
        payload["worldUncertainty"].is_number()) {
        phoenix::primal::PrimalSensation nov;
        nov.type = phoenix::primal::SensationType::Novelty;
        nov.intensity = std::clamp(payload["worldUncertainty"].get<float>(), 0.0f, 1.0f);
        nov.valence = 0.2f;  // novelty is mildly attractive (exploration drive)
        nov.source = "world-uncertainty";
        nov.timestampMs = static_cast<uint64_t>(nowMs());
        sensationEngine_.add(nov);
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
    record["worldEvidenceCount"] = worldState.is_object()
                                      ? worldState.value("evidenceCount",
                                                         worldState.value("recentEvidence", json::array()).size())
                                      : 0;
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
    json mobilityPlan = responsePlan.value(
        "mobilityPlan", worldState.is_object() ? worldState.value("mobilityPlan", json::object()) : json::object());

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
    if (phoenix::safety::EmergencyStop::instance().latched()) {
        lock.unlock();
        return json{{"ok", false}, {"error", "emergency stop engaged: iterate rejected"}};
    }
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

    /* v7.0 sensation hygiene: decay every sensation FIRST (opponent-process
       decay, half-life per subconscious tuning or the engine default 300 s).
       Without this call sensations only accumulate: a completed mission would
       leave its Pain forever, pinning valence at -1. */
    float dtSec = 1.0f;
    if (lastIterAtMs_ > 0) {
        dtSec = static_cast<float>(std::max<int64_t>(0, nowMs() - lastIterAtMs_)) / 1000.0f;
        if (dtSec <= 0.0f) dtSec = 1.0f;
    }
    sensationEngine_.decayAuto(dtSec);

    /* v7.0 mission layer: an active mission exerts time-increasing pain
       (allostatic urgency; total pain = g * T^2 / 2).  add() REFRESHES the
       single (pain, "mission-pressure") signal instead of stacking, so the
       allostatic cost IS p(t).  Pain ends when the mission is judged
       complete; decayAuto above then releases it along the half-life. */
    if (missionEnabled_) {
        const float pressure = mission_.pressureNow();
        if (pressure > 0.0f) {
            phoenix::primal::PrimalSensation pain;
            pain.type = phoenix::primal::SensationType::Pain;
            pain.intensity = std::clamp(pressure, 0.0f, 1.0f);
            pain.valence = -1.0f;  // maximal harm valence
            pain.source = "mission-pressure";
            pain.timestampMs = static_cast<uint64_t>(nowMs());
            sensationEngine_.add(pain);
        }
        /* v7.0 replication: the instance REPLICATES FREELY - there is no
           fixed trigger here.  "replicate" is a planner action (see
           registerDefaultAgiActions / executeAgiAction); the instance itself
           chooses when to summon a successor, and the successor is another
           instance working the same goal.  No hand-off: before the goal
           completes, pressure only grows and this instance keeps going. */
    }

    /* v7.0 instinct / benefit-harm evaluation and prompt split update */
    instinctEngine_.update(sensationEngine_.active(), dtSec);
    auto bh = instinctEngine_.evaluate(sensationEngine_.active());

    /* v7.0 affect signal: export the emotion operation weight vector as a
       numeric matrix rather than an explicit action word or emotional label. */
    std::string driveWeights = json(bh.driveVector).dump();
    lastBenefitHarmBias_ = driveWeights;

    phoenix::prompt::MemoryPrompt memory;
    memory.driveVector = bh.driveVector;
    memory.emotionTensor = phoenix::instinct::InstinctEngine::driveToEmotion(bh);
    if (subconsciousEnabled_) {
        /* Temperament: shift the appraisal toward the baseline PAD disposition. */
        memory.emotionTensor = subProfile_.applyTemperament(memory.emotionTensor);
    }
    memory.inferenceOptions = memory.emotionTensor.inferenceOptions();
    memory.benefitHarmBias = bh.recommendedAction;  // human-readable action label
    /* v7.0 active inference / MPC.  Three parts per iteration:
       1. observe the previous real transition (zPrev, aPrev, zNow) so the
          forward model learns online (metacognition / episodic memory);
       2. re-plan over the receding horizon; once the model has at least one
          real transition it overrides the instinct argmax (cold-start guard);
       3. record the chosen action for the next transition. */
    nlohmann::json agiPlanJson = nlohmann::json::object();
    if (agiEnabled_) {
        const size_t dim = agiController_.model().dim();
        const std::vector<float> zNow =
            phoenix::multimodal::projectToDimension(bh.driveVector, dim, 0x41474955U);
        const bool havePrev = !agiLatentState_.empty() && !lastAgiAction_.empty();
        if (havePrev) {
            std::vector<float> aPrev;
            for (const auto &act : agiController_.actions()) {
                if (act.name == lastAgiAction_) { aPrev = act.embedding; break; }
            }
            /* Self-evolution: the realised benefit-harm netUtility is the
               reward; observeRewarded refines the forward model AND runs the
               TD(0) value-learning step on the preference head. */
            const double agiSurprise = agiController_.observeRewarded(
                agiLatentState_, aPrev, zNow, static_cast<double>(bh.netUtility),
                0.01f, agiAlpha_, agiGamma_);
            /* Metacognitive loop: the forward-model prediction error becomes a
               Novelty primal sensation, so Curiosity/Exploration receive a real
               epistemic signal instead of a hard-coded match. */
            const float feedGain = subconsciousEnabled_ ? subProfile_.anticipatoryGain : 1.0f;
            if (agiSurprise > 1e-4 && feedGain > 0.0f) {
                phoenix::primal::PrimalSensation nov;
                nov.type = phoenix::primal::SensationType::Novelty;
                nov.intensity = std::clamp(static_cast<float>(std::min(1.0, agiSurprise * feedGain)), 0.0f, 1.0f);
                nov.valence = 0.2f;  // novelty is mildly attractive (exploration drive)
                nov.source = "agi-forward-model-surprise";
                nov.timestampMs = static_cast<uint64_t>(nowMs());
                sensationEngine_.add(nov);
            }
        }
        agiLatentState_ = zNow;
        /* Only SEED the preference head once; TD learning owns it afterwards. */
        agiController_.bootstrapPreferences(zNow);
        /* Allostasis: with a subconscious profile the intrinsic cost is the
           homeostatic deviation (Σ gain·|intensity−setpoint|); otherwise the
           legacy arousal proxy. */
        const double driveCost = subconsciousEnabled_
            ? static_cast<double>(sensationEngine_.homeostaticCost())
            : static_cast<double>(sensationEngine_.netArousal());
        /* Adaptive exploration (VDBE-style): amplify the epistemic term when
           the environment is getting predictable, damp it when chaotic. */
        const double epistW = agiEpistW_ * (agiAdaptiveExploration_
            ? agiController_.explorationMultiplier() : 1.0);
        const auto plan = agiController_.plan(agiLatentState_, driveCost, agiPragW_, agiIntrinW_, epistW);
        /* Periodic consolidation: replay episodic memory to refine the forward
           model and the value function (sleep-like). */
        if (agiConsolidateEvery_ > 0 && agiController_.episodeCount() > 0 &&
            agiController_.episodeCount() % agiConsolidateEvery_ == 0) {
            agiController_.consolidate(64, agiAlpha_ * 0.4, agiGamma_);
        }
        json executionResult = nullptr;
        if (plan.bestAction >= 0 && plan.bestAction < static_cast<int>(agiController_.actions().size())) {
            lastAgiAction_ = agiController_.actions()[static_cast<size_t>(plan.bestAction)].name;
            if (agiController_.episodeCount() >= 1) {
                memory.benefitHarmBias = lastAgiAction_;
            }
            /* Execute real capabilities chosen by the planner (tools / goals),
               not just inject the verb into the prompt. */
            const auto *spec = agiActionRegistry_.find(lastAgiAction_);
            if (spec && spec->category != "instinct" && agiActionExecutor_) {
                json execCtx;
                execCtx["userPrompt"] = payload.value("userPrompt", json(std::string()));
                if (payload.contains("graphContext")) execCtx["graphContext"] = payload["graphContext"];
                executionResult = agiActionExecutor_(*spec, execCtx);
            }
        }
        agiPlanJson = json{{"bestAction", lastAgiAction_},
                           {"bestActionIndex", plan.bestAction},
                           {"efe", plan.efe.toJson()},
                           {"driveCost", driveCost},
                           {"episodes", agiController_.episodeCount()},
                           {"surpriseEma", agiController_.surpriseEma()},
                           {"explorationMultiplier", agiController_.explorationMultiplier()},
                           {"execution", executionResult}};
    }
    if (memory.benefitHarmBias.empty())
        memory.benefitHarmBias = memory.emotionTensor.modulationHint();
    memory.summary = "Benefit=" + std::to_string(bh.benefitScore) +
                     " Harm=" + std::to_string(bh.harmScore) +
                     " Net=" + std::to_string(bh.netUtility);
    promptComposer_.setMemory(memory);

    std::string composedPrompt;
    std::string cognitionModulation;
    /* drain human interjections: delivered once, on the next tick/turn */
    std::vector<std::string> consumedInterjections;
    for (const auto &inj : interjections_) consumedInterjections.push_back(inj.second);
    interjections_.clear();
    std::string userText;
    if (payload.contains("userPrompt") && payload["userPrompt"].is_string()) {
        userText = payload["userPrompt"].get<std::string>();
    }
    if (userText.empty() && !consumedInterjections.empty()) {
        userText = consumedInterjections.back();
    }
    if (!userText.empty()) {
        composedPrompt = promptComposer_.compose(userText, true);
        cognitionModulation = promptComposer_.modulationHint();
    }
    if (!consumedInterjections.empty()) {
        std::string injBlock = "[human interjections]";
        for (const auto &s : consumedInterjections) injBlock += "\n- " + s;
        if (!cognitionModulation.empty()) cognitionModulation += "\n";
        cognitionModulation += injBlock;
    }

    auto outbound = outputQueue_.drain(0);
    nlohmann::json mixedModalOutputs = nlohmann::json::array();
    for (const auto &p : outbound) mixedModalOutputs.push_back(p.toJson());

    iteration_ += 1;
    lastIterAtMs_ = nowMs();
    nlohmann::json missionInfo = nlohmann::json::object();
    if (missionEnabled_) {
        missionInfo = json{{"enabled", true},
                           {"active", mission_.active()},
                           {"pressure", mission_.pressureNow()},
                           {"state", static_cast<int>(mission_.mission().state)},
                           {"goal", mission_.mission().goal},
                           {"children", mission_.stats().value("children", json::array())}};

    }
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
                                 {"composedPrompt", composedPrompt},
                                 {"agiPlan", agiPlanJson},
                                 {"mission", missionInfo},
                                 {"interjectionsConsumed", consumedInterjections.size()}}}};
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
                {"sessions", sessions},
                /* long-term evolution: everything the agent LEARNS */
                {"agiEnabled", agiEnabled_},
                {"agi", agiController_.toJson()},
                {"agiLatentState", agiLatentState_},
                {"lastAgiAction", lastAgiAction_},
                {"sensations", sensationEngine_.toJson()},
                {"mission", mission_.toJson()},
                {"missionEnabled", missionEnabled_},
                {"missionGenome", missionGenome_.toJson()},
                {"goals", goals_},
                {"subconsciousEnabled", subconsciousEnabled_},
                {"subconscious", subProfile_.toJson()}};
}

json CognitionAutonomyManager::importState(const json &state) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!state.is_object()) {
        return json{{"ok", false}, {"error", "state must be an object"}};
    }
    if (!state.contains("sessions") || !state["sessions"].is_object()) {
        return json{{"ok", false}, {"error", "state.sessions object required"}};
    }

    enabled_ = state.value("enabled", enabled_);
    backgroundEnabled_ = state.value("backgroundEnabled", backgroundEnabled_);
    iteration_ = std::max(0, state.value("iteration", iteration_));
    observations_ = std::max(0, state.value("observations", observations_));

    /* long-term evolution restoration */
    if (state.contains("agi") && state["agi"].is_object()) {
        agiController_ = phoenix::agi::ActiveInferenceController::fromJson(state["agi"]);
    }
    if (state.contains("agiLatentState") && state["agiLatentState"].is_array()) {
        agiLatentState_.clear();
        for (const auto &v : state["agiLatentState"]) agiLatentState_.push_back(v.get<float>());
    }
    lastAgiAction_ = state.value("lastAgiAction", std::string());
    agiEnabled_ = state.value("agiEnabled", agiEnabled_);
    if (state.contains("sensations") && state["sensations"].is_object()) {
        sensationEngine_ = phoenix::primal::PrimalSensationEngine::fromJson(state["sensations"]);
    }
    if (state.contains("mission") && state["mission"].is_object()) {
        mission_.fromJson(state["mission"]);
    }
    missionEnabled_ = state.value("missionEnabled", missionEnabled_);
    if (state.contains("missionGenome") && state["missionGenome"].is_object()) {
        missionGenome_ = phoenix::mission::MissionGenome::fromJson(state["missionGenome"]);
    }
    if (state.contains("goals") && state["goals"].is_array()) {
        goals_.clear();
        for (const auto &g : state["goals"])
            if (g.is_string()) goals_.push_back(g.get<std::string>());
    }
    subconsciousEnabled_ = state.value("subconsciousEnabled", subconsciousEnabled_);
    if (state.contains("subconscious") && state["subconscious"].is_object()) {
        subProfile_ = phoenix::subconscious::SubconsciousProfile::fromJson(state["subconscious"]);
        if (subconsciousEnabled_) {
            subProfile_.applyTo(sensationEngine_);
            subProfile_.applyTo(instinctEngine_);
            subProfile_.applyTo(agiController_);
        }
    }

    sessions_.clear();
    std::size_t sessionsLoaded = 0;
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

json CognitionAutonomyManager::configureAgi(const json &payload) {
    std::lock_guard<std::mutex> lock(mu_);
    auto asInt = [](const json &j, int dflt) -> int {
        return j.is_number() ? static_cast<int>(j.get<double>()) : dflt;
    };
    auto asDbl = [](const json &j, double dflt) -> double {
        return j.is_number() ? j.get<double>() : dflt;
    };
    agiEnabled_ = payload.value("enabled", agiEnabled_);
    const int dim = clampInt(asInt(payload.value("dim", json(128)), 128), 8, 4096);
    const int horizon = clampInt(asInt(payload.value("horizon", json(3)), 3), 1, 16);
    agiPragW_ = clampDouble(asDbl(payload.value("pragmaticWeight", json(1.0)), 1.0), 0.0, 10.0);
    agiIntrinW_ = clampDouble(asDbl(payload.value("intrinsicWeight", json(1.0)), 1.0), 0.0, 10.0);
    agiEpistW_ = clampDouble(asDbl(payload.value("epistemicWeight", json(0.25)), 0.25), 0.0, 10.0);
    agiAlpha_ = clampDouble(asDbl(payload.value("learningRate", json(0.05)), 0.05), 0.0, 1.0);
    agiGamma_ = clampDouble(asDbl(payload.value("discount", json(0.9)), 0.9), 0.0, 0.999);
    agiConsolidateEvery_ = static_cast<size_t>(clampInt(asInt(payload.value("consolidateEvery", json(16)), 16), 0, 4096));
    agiAdaptiveExploration_ = payload.value("adaptiveExploration", agiAdaptiveExploration_);

    // Seed the action space from the instinct action biases.
    std::vector<phoenix::agi::Action> actions;
    for (const auto &i : instinctEngine_.instincts()) {
        if (i.actionBias.empty()) continue;
        phoenix::agi::Action a;
        a.name = i.actionBias;
        a.embedding = {i.benefitWeight - i.harmWeight};  // scalar control direction
        actions.push_back(std::move(a));
    }
    /* Append registered real capabilities (tools / goals) so the MPC loop can
       choose among actual executable effects, not only instinct verbs. */
    for (auto &a : agiActionRegistry_.toPlannerActions()) {
        actions.push_back(std::move(a));
    }

    // Normalise all action embeddings to the same dimension so the forward
    // model has a fixed actionDim.  This is what lets the planner distinguish
    // tools with distinct one-hot embeddings.
    size_t actionDim = 1;
    for (const auto &a : actions) {
        actionDim = std::max(actionDim, a.embedding.size());
    }
    for (auto &a : actions) {
        padActionEmbedding(a.embedding, actionDim);
    }

    // Rebuild the controller: latent dim + actionDim + horizon.
    agiController_.configure(static_cast<size_t>(dim), actionDim, static_cast<size_t>(horizon));
    agiController_.setActions(std::move(actions));

    // Preferences track the current benefit-harm drive direction.
    auto bh = instinctEngine_.evaluate(sensationEngine_.active());
    agiController_.bootstrapPreferences(
        phoenix::multimodal::projectToDimension(bh.driveVector, static_cast<size_t>(dim), 0x41474955U));

    // Optional: caller-supplied preference vector (useful for tests / manual steering).
    if (payload.contains("preferences") && payload["preferences"].is_array()) {
        std::vector<float> w;
        for (const auto &e : payload["preferences"]) {
            if (e.is_number()) w.push_back(e.get<float>());
        }
        if (!w.empty()) agiController_.setPreferences(w);
    }

    return json{{"ok", true}, {"result", agiController_.status()}};
}

json CognitionAutonomyManager::agiPlan() {
    std::lock_guard<std::mutex> lock(mu_);
    if (!agiEnabled_) return json{{"ok", false}, {"error", "agi disabled"}};
    auto bh = instinctEngine_.evaluate(sensationEngine_.active());
    const size_t dim = agiController_.model().dim();
    std::vector<float> z = agiLatentState_;
    if (z.empty()) {
        z = phoenix::multimodal::projectToDimension(bh.driveVector, dim, 0x41474955U);
        agiLatentState_ = z;
    }
    agiController_.bootstrapPreferences(
        phoenix::multimodal::projectToDimension(bh.driveVector, dim, 0x41474955U));
    const double driveCost = static_cast<double>(sensationEngine_.netArousal());
    const auto plan = agiController_.plan(z, driveCost, agiPragW_, agiIntrinW_, agiEpistW_);
    json out;
    out["ok"] = true;
    out["result"]["bestActionIndex"] = plan.bestAction;
    out["result"]["bestAction"] =
        (plan.bestAction >= 0 && plan.bestAction < static_cast<int>(agiController_.actions().size()))
            ? agiController_.actions()[static_cast<size_t>(plan.bestAction)].name
            : std::string();
    out["result"]["efe"] = plan.efe.toJson();
    out["result"]["driveCost"] = driveCost;
    out["result"]["model"] = agiController_.model().status();
    return out;
}

json CognitionAutonomyManager::ingestAgiTransition(const json &payload) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!agiEnabled_) return json{{"ok", false}, {"error", "agi disabled"}};
    auto toVec = [](const json &j) {
        std::vector<float> v;
        if (j.is_array()) for (const auto &e : j) if (e.is_number()) v.push_back(e.get<float>());
        return v;
    };
    std::vector<float> z = toVec(payload.value("z", json::array()));
    std::vector<float> a = toVec(payload.value("a", json::array()));
    std::vector<float> zNext = toVec(payload.value("zNext", json::array()));
    if (z.empty() || zNext.empty()) return json{{"ok", false}, {"error", "z and zNext required"}};
    const double surprise = agiController_.observe(z, a, zNext);
    agiLatentState_ = zNext;
    return json{{"ok", true},
                {"result", json{{"surprise", surprise},
                                {"episodes", agiController_.episodeCount()}}}};
}

json CognitionAutonomyManager::registerAgiAction(const json &payload) {
    std::lock_guard<std::mutex> lock(mu_);
    phoenix::agi::AgiActionSpec spec = phoenix::agi::AgiActionSpec::fromJson(payload);
    if (spec.name.empty()) {
        return json{{"ok", false}, {"error", "action name required"}};
    }
    agiActionRegistry_.registerAction(std::move(spec));
    /* Rebuild the planner action space: instinct verbs + registered capabilities. */
    if (agiEnabled_) {
        std::vector<phoenix::agi::Action> actions;
        for (const auto &i : instinctEngine_.instincts()) {
            if (i.actionBias.empty()) continue;
            phoenix::agi::Action a;
            a.name = i.actionBias;
            a.embedding = {i.benefitWeight - i.harmWeight};
            actions.push_back(std::move(a));
        }
        for (auto &a : agiActionRegistry_.toPlannerActions()) actions.push_back(std::move(a));

        // Keep a fixed actionDim; pad shorter embeddings, reconfigure only if a
        // new action truly needs a larger embedding space.
        size_t actionDim = agiController_.model().actionDim();
        for (auto &a : actions) {
            padActionEmbedding(a.embedding, actionDim);
        }
        size_t neededDim = actionDim;
        for (const auto &a : actions) {
            neededDim = std::max(neededDim, a.embedding.size());
        }
        if (neededDim > actionDim) {
            agiController_.configure(
                agiController_.model().dim(), neededDim, agiController_.horizon());
            for (auto &a : actions) {
                padActionEmbedding(a.embedding, neededDim);
            }
        }
        agiController_.setActions(std::move(actions));
    }
    return json{{"ok", true}, {"result", agiActionRegistry_.toJson()}};
}

json CognitionAutonomyManager::listAgiActions() const {
    std::lock_guard<std::mutex> lock(mu_);
    return json{{"ok", true}, {"result", agiActionRegistry_.toJson()}};
}

void CognitionAutonomyManager::setAgiActionExecutor(AgiActionExecutor executor) {
    std::lock_guard<std::mutex> lock(mu_);
    agiActionExecutor_ = std::move(executor);
}

json CognitionAutonomyManager::executeAgiActionByName(const std::string &name,
                                                      const json &context) {
    std::lock_guard<std::mutex> lock(mu_);
    const auto *spec = agiActionRegistry_.find(name);
    if (!spec) return json{{"ok", false}, {"error", "action not found: " + name}};
    return executeAgiAction(*spec, context);
}

json CognitionAutonomyManager::configureSubconscious(const json &payload) {
    std::lock_guard<std::mutex> lock(mu_);
    subconsciousEnabled_ = payload.value("enabled", subconsciousEnabled_);
    subProfile_ = phoenix::subconscious::SubconsciousProfile::fromJson(payload);
    /* Install the innate parameters into the three subconscious layers. */
    subProfile_.applyTo(sensationEngine_);
    subProfile_.applyTo(instinctEngine_);
    subProfile_.applyTo(agiController_);
    return json{{"ok", true}, {"result", subProfile_.toJson()}};
}

json CognitionAutonomyManager::assignMission(const json &payload) {
    std::lock_guard<std::mutex> lock(mu_);
    const json &p = payload.is_null() ? json::object() : payload;
    missionEnabled_ = p.value("enabled", missionEnabled_);
    missionMutationRate_ = static_cast<float>(clampDouble(p.value("mutationRate", missionMutationRate_), 0.0, 10.0));
    missionMaxReplicas_ = static_cast<size_t>(
        clampInt(p.value("maxReplicas", static_cast<int>(missionMaxReplicas_)), 0, 64));
    mission_.setMaxReplicas(missionMaxReplicas_);
    mission_.setMaxReplicaDepth(static_cast<size_t>(
        clampInt(p.value("maxReplicaDepth", static_cast<int>(mission_.maxReplicaDepth())), 0, 16)));
    /* Context packing: user chooses summary vs full+summary, optional GNN,
       and ctx size (4096 / 16384). */
    if (p.contains("contextPack") && p["contextPack"].is_string()) {
        const std::string m = p["contextPack"].get<std::string>();
        if (m == "summary" || m == "full_and_summary") missionContextPack_ = m;
    }
    if (p.contains("includeGnnSummary") && p["includeGnnSummary"].is_boolean())
        missionIncludeGnnSummary_ = p["includeGnnSummary"].get<bool>();
    {
        int ctx = p.value("ctxSize", p.value("ctxTokens", missionCtxTokens_));
        if (ctx < 2048) ctx = 2048;
        if (ctx > 32768) ctx = 32768;
        missionCtxTokens_ = ctx;
    }
    if (p.contains("gnnSummary") && p["gnnSummary"].is_string())
        missionGnnSummary_ = p["gnnSummary"].get<std::string>();
    if (p.value("enabled", false)) {
        registerWithSafetyRegistry(); /* mission lifecycle begins: register */
    }
    phoenix::mission::Mission m;
    m.id = p.value("id", std::string());
    m.goal = p.value("goal", std::string());
    m.deadlineSec = clampDouble(p.value("deadlineSec", 300.0), 1.0, 3.2e7);
    m.painGainPerSec = static_cast<float>(clampDouble(p.value("painGainPerSec", 0.01), 0.0, 100.0));
    m.maxPain = static_cast<float>(clampDouble(p.value("maxPain", 1.0), 0.0, 1.0));
    m.pressureMode = p.value("pressureMode", std::string("asymptotic"));
    if (m.pressureMode != "linear" && m.pressureMode != "logarithmic" &&
        m.pressureMode != "expression" && m.pressureMode != "asymptotic") {
      m.pressureMode = "asymptotic";
    }
    m.pressureHorizonSec = clampDouble(p.value("pressureHorizonSec", 3600.0), 60.0, 3.2e7);
    m.pressureTauSec = clampDouble(p.value("pressureTauSec", 1800.0), 1.0, 3.2e7);
    m.pressureExpr = p.value("pressureExpr", std::string("Pmax*tanh(t/tau)"));
    if (m.pressureExpr.empty()) m.pressureExpr = "Pmax*tanh(t/tau)";
    if (m.id.empty()) m.id = "mission-" + std::to_string(iteration_ + 1);
    if (p.contains("genome") && p["genome"].is_object()) {
        missionGenome_ = phoenix::mission::MissionGenome::fromJson(p["genome"]);
    }
    /* A mission without a goal is a no-op (config-only enable). */
    if (m.goal.empty()) {
        return json{{"ok", true}, {"result", mission_.stats()}};
    }
    mission_.assign(m, missionGenome_);
    return json{{"ok", true}, {"result", mission_.stats()}};
}

json CognitionAutonomyManager::missionStatus() const {
    std::lock_guard<std::mutex> lock(mu_);
    return json{{"ok", true},
                {"result", json{{"enabled", missionEnabled_},
                                {"stats", mission_.stats()},
                                {"genome", missionGenome_.toJson()},
                                {"contextPack", missionContextPack_},
                                {"includeGnnSummary", missionIncludeGnnSummary_},
                                {"ctxSize", missionCtxTokens_}}}};
}

json CognitionAutonomyManager::missionContextOptions() const {
    std::lock_guard<std::mutex> lock(mu_);
    return json{{"ok", true},
                {"result", json{{"contextPack", missionContextPack_},
                                {"includeGnnSummary", missionIncludeGnnSummary_},
                                {"ctxSize", missionCtxTokens_},
                                {"gnnSummary", missionGnnSummary_}}}};
}

void CognitionAutonomyManager::setMissionGnnSummary(const std::string &summary) {
    std::lock_guard<std::mutex> lock(mu_);
    missionGnnSummary_ = summary;
    if (missionGnnSummary_.size() > 4096)
        missionGnnSummary_.resize(4096);
}

json CognitionAutonomyManager::reportMissionOutcome(const json &payload) {
    std::lock_guard<std::mutex> lock(mu_);
    const json &p = payload.is_null() ? json::object() : payload;
    const bool achieved = p.value("goalAchieved", false);
    /* No hand-off: before the goal completes, pressure only grows and this
       instance keeps going.  Successors replicated by this instance stay in
       the children list (observability); judgement stays with the caller. */
    if (achieved) {
        mission_.markComplete();
    } else {
        mission_.markFailed();
    }
    return json{{"ok", true}, {"result", mission_.stats()}};
}

void CognitionAutonomyManager::setMissionDeliberator(MissionDeliberator fn) {
    std::lock_guard<std::mutex> lock(mu_);
    missionDeliberator_ = std::move(fn);
}

json CognitionAutonomyManager::appendMissionDeliverable(const json &payload) {
    std::lock_guard<std::mutex> lock(mu_);
    const std::string text = trimLocal(payload.value("text", std::string()));
    if (text.empty())
        return json{{"ok", false}, {"error", "deliverable text required"}};
    mission_.appendDeliverable(text);
    return json{{"ok", true}, {"result", mission_.stats()}};
}

json CognitionAutonomyManager::spawnMissionChild(const json &payload) {
    std::lock_guard<std::mutex> lock(mu_);
    const json &p = payload.is_null() ? json::object() : payload;
    const float rate = static_cast<float>(clampDouble(p.value("mutationRate", missionMutationRate_), 0.0, 10.0));
    const auto parent = (p.contains("genome") && p["genome"].is_object())
                            ? phoenix::mission::MissionGenome::fromJson(p["genome"])
                            : missionGenome_;
    /* v8.2: subgoal = the bounded request this box must fulfil. */
    const std::string subgoal = p.value("subgoal", p.value("userPrompt", std::string()));
    const std::string parentId = p.value("parentId", std::string());
    const auto child = mission_.recordChild(parent, rate, subgoal, parentId);
    return json{{"ok", true}, {"result", child.toJson()}};
}

json CognitionAutonomyManager::configureMcp(const json &payload) {
    std::lock_guard<std::mutex> lock(mu_);
    const json &p = payload.is_null() ? json::object() : payload;
    std::string err;
    mcpManager_.configure(p, err);
    mcpEnabled_ = p.value("enabled", false);
    mcpActionMap_.clear();
    nlohmann::json report = mcpManager_.startAll();
    nlohmann::json toolSnapshot = nlohmann::json::array();
    if (mcpEnabled_) {
        /* expose every MCP tool as a planner action (category "mcp") */
        toolSnapshot = mcpManager_.listTools();
        for (const auto &entry : toolSnapshot) {
            const std::string server = entry.value("server", std::string());
            const nlohmann::json &tool = entry["tool"];
            const std::string toolName = tool.value("name", std::string());
            if (server.empty() || toolName.empty()) continue;
            const std::string actionName = "mcp." + server + "." + toolName;
            phoenix::agi::AgiActionSpec spec;
            spec.name = actionName;
            spec.category = "mcp";
            spec.addonType = "mcp";
            spec.description = tool.value("description", std::string("MCP tool " + toolName));
            agiActionRegistry_.registerAction(spec);
            mcpActionMap_[actionName] = {server, toolName};
        }
    }
    return json{{"ok", true},
                {"result", json{{"enabled", mcpEnabled_},
                                {"report", report},
                                {"tools", toolSnapshot}}}};
}

json CognitionAutonomyManager::listMcpTools() const {
    std::lock_guard<std::mutex> lock(mu_);
    return json{{"ok", true}, {"result", mcpManager_.listTools()}};
}

json CognitionAutonomyManager::callMcpTool(const json &payload) {
    std::lock_guard<std::mutex> lock(mu_);
    const json &p = payload.is_null() ? json::object() : payload;
    const std::string server = p.value("server", std::string());
    const std::string tool = p.value("tool", std::string());
    const nlohmann::json args = p.contains("arguments") ? p["arguments"]
                                                       : nlohmann::json::object();
    nlohmann::json out = mcpManager_.callTool(server, tool, args);
    return json{{"ok", out.value("ok", false)}, {"result", out}};
}


void CognitionAutonomyManager::registerWithSafetyRegistry() {
    if (safetyRegistered_) return;
    safetyRegistered_ = true;
    safetyRegId_ = phoenix::safety::InstanceRegistry::instance().registerInstance(
        "cognition-autonomy", "manager", [this] {
            /* E-stop stop handler: kill this instance's autonomous activity.
               Idempotent and safe to run on any thread except the loop thread
               itself (which is why we only flag, not join, there). */
            loopStop_.store(true, std::memory_order_release);
            mcpManager_.stopAll();
            if (loopThread_.joinable() &&
                loopThread_.get_id() != std::this_thread::get_id()) {
                loopThread_.join();
            }
        });
}

void CognitionAutonomyManager::unregisterFromSafetyRegistry() {
    if (safetyRegistered_) {
        phoenix::safety::InstanceRegistry::instance().unregister(safetyRegId_);
        safetyRegistered_ = false;
        safetyRegId_ = 0;
    }
}

json CognitionAutonomyManager::interject(const json &payload) {
    std::lock_guard<std::mutex> lock(mu_);
    if (phoenix::safety::EmergencyStop::instance().latched()) {
        return json{{"ok", false}, {"error", "emergency stop engaged: interjection rejected"}};
    }
    const json &p = payload.is_null() ? json::object() : payload;
    const std::string text = trimLocal(p.value("text", std::string()));
    if (text.empty()) {
        return json{{"ok", false}, {"error", "interjection text required"}};
    }
    interjections_.push_back({nowMs(), text});
    if (interjections_.size() > 64) interjections_.erase(interjections_.begin());
    nlohmann::json out = json{{"ok", true}, {"queued", interjections_.size()}};
    /* optional mid-flight goal amendment: the mission is REDIRECTED, not
       restarted (start time and pressure are preserved). */
    const std::string amend = trimLocal(p.value("amendGoal", std::string()));
    if (!amend.empty()) {
        out["goalAmended"] = mission_.amendGoal(amend);
        out["goal"] = mission_.mission().goal;
        if (!out["goalAmended"].get<bool>()) {
            out["warning"] = "no running mission; goal not amended";
        }
    }
    return out;
}

json CognitionAutonomyManager::configureAutonomyLoop(const json &payload) {
    std::lock_guard<std::mutex> lock(mu_);
    const json &p = payload.is_null() ? json::object() : payload;
    loopEnabled_ = p.value("enabled", loopEnabled_);
    loopIntervalSec_ = clampInt(p.value("intervalSec", loopIntervalSec_), 1, 3600);
    loopMaxStepsPerTick_ = clampInt(p.value("maxStepsPerTick", loopMaxStepsPerTick_), 1, 64);
    loopPersistEveryTicks_ = clampInt(p.value("persistEveryTicks", loopPersistEveryTicks_), 1, 10000);
    if (p.contains("persistPath") && p["persistPath"].is_string()) {
        loopPersistPath_ = p["persistPath"].get<std::string>();
    }
    return json{{"ok", true},
                {"result", json{{"enabled", loopEnabled_},
                                {"intervalSec", loopIntervalSec_},
                                {"maxStepsPerTick", loopMaxStepsPerTick_},
                                {"persistEveryTicks", loopPersistEveryTicks_},
                                {"persistPath", loopPersistPath_}}}};
}

json CognitionAutonomyManager::startAutonomyLoop() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (phoenix::safety::EmergencyStop::instance().latched()) {
            return json{{"ok", false}, {"error", "emergency stop engaged: loop will not start"}};
        }
        if (!loopEnabled_) {
            return json{{"ok", false}, {"error", "autonomy loop not enabled (configure first)"}};
        }
        registerWithSafetyRegistry(); /* lifecycle begins: register with the system */
        if (loopThread_.joinable() && !loopStop_.load(std::memory_order_acquire)) {
            return json{{"ok", true}, {"result", json{{"running", true}}}};
        }
        loopStop_.store(false, std::memory_order_release);
    }
    /* v8.0 fix: restore OUTSIDE the manager lock.  importState() takes mu_
       itself, and calling it while holding mu_ self-deadlocks (the assign
       route wedged every drogon worker on this). */
    nlohmann::json restored = nullptr;
    try {
        std::ifstream f(loopPersistPath_);
        if (f.good()) {
            nlohmann::json saved;
            f >> saved;
            restored = importState(saved);
        }
    } catch (...)
    {
        restored = json{{"ok", false}, {"error", "persist file unreadable; starting fresh"}};
    }
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (loopThread_.joinable()) loopThread_.join();
        loopThread_ = std::thread([this] { loopRun(); });
    }
    return json{{"ok", true},
                {"result", json{{"running", true}, {"restored", restored}}}};
}

json CognitionAutonomyManager::stopAutonomyLoop() {
    loopStop_.store(true, std::memory_order_release);
    if (loopThread_.joinable()) loopThread_.join();
    std::lock_guard<std::mutex> lock(mu_);
    return json{{"ok", true},
                {"result", json{{"running", false}, {"ticks", loopTickCount_.load()}}}};
}

json CognitionAutonomyManager::autonomyLoopStatus() const {
    std::lock_guard<std::mutex> lock(mu_);
    return json{{"ok", true},
                {"result", json{{"enabled", loopEnabled_},
                                {"running", loopThread_.joinable() &&
                                               !loopStop_.load(std::memory_order_acquire)},
                                {"intervalSec", loopIntervalSec_},
                                {"maxStepsPerTick", loopMaxStepsPerTick_},
                                {"persistEveryTicks", loopPersistEveryTicks_},
                                {"persistPath", loopPersistPath_},
                                {"tickCount", loopTickCount_.load()},
                                {"lastTickAtMs", loopLastTickAtMs_.load()}}}};
}

void CognitionAutonomyManager::ensureHeartbeatSession() {
    std::lock_guard<std::mutex> lock(mu_);
    const std::string id = "__autonomy_heartbeat__";
    auto it = sessions_.find(id);
    if (it != sessions_.end()) {
        it->second["shouldIterate"] = true;
        return;
    }
    nlohmann::json rec;
    rec["sessionId"] = id;
    rec["observations"] = 1;
    const std::string goal = mission_.mission().goal;
    rec["seedMission"] = goal.empty() ? "autonomous loop" : goal;
    rec["lastObservedAtMs"] = nowMs();
    rec["shouldIterate"] = true;
    sessions_[id] = std::move(rec);
}

void CognitionAutonomyManager::loopRun() {
    /* Heartbeat: the REAL autonomous loop.  Each tick runs the full
       plan/act/observe/learn cycle through iterate() - no external message
       required - and periodically persists the evolved state to disk so
       evolution is long-term. */
    while (!loopStop_.load(std::memory_order_acquire)) {
        if (phoenix::safety::EmergencyStop::instance().latched()) break;
        const int64_t tickStart = nowMs();
        try {
            /* v8.0 mission worker: actually WORK on a Running mission via the
               gateway-registered LLM deliberator.  Runs BEFORE the iterate
               steps and OUTSIDE the manager lock (a slow LLM reply must not
               stall interject/status/E-stop).  Output accumulates in the
               mission deliverable; the human supervisor judges completion.

               Snapshot goal/deliverable/running under the lock, then call the
               LLM without holding mu_ - do NOT parse missionStatus() JSON
               through dangling temporary references. */
            if (missionDeliberator_) {
                std::string goal;
                std::string prior;
                std::string missionId;
                bool running = false;
                std::vector<phoenix::mission::MissionChild> helpers;
                {
                    std::lock_guard<std::mutex> lock(mu_);
                    running = missionEnabled_ && mission_.active();
                    if (running) {
                        const auto snap = mission_.mission();
                        goal = snap.goal;
                        prior = snap.deliverable;
                        missionId = snap.id;
                        helpers = mission_.children();
                    }
                }
                /* Prior packing happens inside the deliberator (sliding window
                   + pinned summary).  Pass a modest tail here only as a
                   fallback when the workspace file is still empty. */
                if (prior.size() > 6000)
                    prior = prior.substr(prior.size() - 6000);

                /* the PARENT always works first - it can use helper boxes
                   but it never hands its whole job to them */
                if (running && !goal.empty()) {
                    std::string work;
                    try {
                        work = missionDeliberator_(goal, prior, loopDeliberateMaxTokens_, missionId);
                    } catch (...) {
                        work.clear();
                    }
                    if (!work.empty()) {
                        appendMissionDeliverable(nlohmann::json{{"text", work}});
                    }
                }
                /* helper boxes (children): ALL active boxes work each tick
                   (sequential LLM calls under llamaCallMu_; --parallel may be
                   >1 so a cancelled peer is less likely). Cap with
                   loopMaxChildrenPerTick_ (0 = no cap = all children). */
                if (running && !helpers.empty()) {
                    size_t budget = helpers.size();
                    if (loopMaxChildrenPerTick_ > 0)
                        budget = std::min(budget, loopMaxChildrenPerTick_);
                    for (size_t n = 0; n < budget; ++n) {
                        const size_t idx = (childRoundRobin_ + n) % helpers.size();
                        const auto &box = helpers[idx];
                        std::string childGoal = box.goal;
                        if (childGoal.empty()) childGoal = goal;
                        const std::string childScope =
                            missionId + "/children/" + box.id;
                        try {
                            const std::string boxWork = missionDeliberator_(
                                childGoal, "", loopChildDeliberateMaxTokens_, childScope);
                            (void)boxWork; /* output lives in the box workspace file */
                        } catch (...) {
                        }
                    }
                    childRoundRobin_ += budget;
                }
            }
            ensureHeartbeatSession();
            /* While a mission is producing text, keep iterate() light so the
               RDK CPU stays available for llama-server.  Pure autonomy (no
               mission) still uses the configured maxStepsPerTick. */
            int steps = loopMaxStepsPerTick_;
            {
                std::lock_guard<std::mutex> lock(mu_);
                if (missionEnabled_ && mission_.active()) steps = std::min(steps, 1);
            }
            for (int step = 0; step < steps; ++step) {
                if (loopStop_.load(std::memory_order_acquire)) break;
                iterate(nlohmann::json::object(), nlohmann::json::object());
            }
        } catch (...) {
            /* the loop must never die from one bad tick */
        }
        loopTickCount_.fetch_add(1, std::memory_order_relaxed);
        loopLastTickAtMs_.store(nowMs(), std::memory_order_relaxed);
        if (loopTickCount_.load(std::memory_order_relaxed) % loopPersistEveryTicks_ == 0) {
            try {
                nlohmann::json state = exportState();
                std::filesystem::path path(loopPersistPath_);
                if (!path.parent_path().empty()) {
                    std::filesystem::create_directories(path.parent_path());
                }
                std::ofstream f(path);
                f << state.dump(2);
            } catch (...) {
                /* persistence is best-effort */
            }
        }
        const int64_t elapsed = nowMs() - tickStart;
        const int64_t sleepMs =
            std::max<int64_t>(50, static_cast<int64_t>(loopIntervalSec_) * 1000 - elapsed);
        for (int64_t slept = 0;
             slept < sleepMs && !loopStop_.load(std::memory_order_acquire);
             slept += 50) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
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
