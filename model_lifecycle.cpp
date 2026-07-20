/* model_lifecycle.cpp - Model lifecycle implementation
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

#include "model_lifecycle.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_set>

namespace fs = std::filesystem;

namespace {

int64_t nowMs()
{
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

std::string trimCopy(std::string s)
{
    auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos)
        return "";
    auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

int clampInt(int x, int lo, int hi)
{
    return std::max(lo, std::min(hi, x));
}

double clampDouble(double x, double lo, double hi)
{
    return std::max(lo, std::min(hi, x));
}

bool inSet(const std::string &v, const std::unordered_set<std::string> &s)
{
    return s.find(v) != s.end();
}

double quantRatio(const std::string &q)
{
    if (q == "int4")
        return 0.125;
    if (q == "int8")
        return 0.25;
    if (q == "fp16")
        return 0.5;
    return 1.0;
}

} // namespace

namespace model_lifecycle {

ModelLifecycleManager::ModelLifecycleManager()
{
    activeTarget_ = "none";
    activeVersion_ = "v3.0-baseline";
    compression_ = json{{"enabled", false},
                        {"method", "none"},
                        {"pruneRatio", 0.0},
                        {"quant", "fp32"},
                        {"note", "not-applied"},
                        {"updatedAtMs", nowMs()}};
    explainability_ = json{{"enabled", true},
                           {"method", "verify+token-overlap"},
                           {"updatedAtMs", nowMs()}};
    deployment_ = json{{"target", activeTarget_},
                       {"version", activeVersion_},
                       {"updatedAtMs", nowMs()}};
    onlineUpdate_ = json{{"enabled", true},
                         {"lastPackage", ""},
                         {"lastChecksum", ""},
                         {"updatedAtMs", nowMs()}};
    servingCluster_ = json{{"replicas", 1},
                           {"routingPolicy", "latency-aware"},
                           {"fallback", "sticky-primary"},
                           {"updatedAtMs", nowMs()}};
    updateSeq_ = 0;
    loadManifestIfExists();
    persistManifest();
}

json ModelLifecycleManager::status() const
{
    std::lock_guard<std::mutex> lock(mu_);
    json out;
    out["ok"] = true;
    out["activeTarget"] = activeTarget_;
    out["activeVersion"] = activeVersion_;
    out["compression"] = compression_;
    out["explainability"] = explainability_;
    out["deployment"] = deployment_;
    out["onlineUpdate"] = onlineUpdate_;
    out["servingCluster"] = servingCluster_;
    out["updateSeq"] = updateSeq_;
    out["manifestPath"] = (fs::path("runtime_store") / "model_lifecycle" / "manifest.json").string();
    out["events"] = events_;
    return out;
}

json ModelLifecycleManager::compressPlan(const json &payload)
{
    std::lock_guard<std::mutex> lock(mu_);
    if (!payload.is_object()) {
        return json{{"ok", false}, {"error", "payload must be an object"}};
    }
    static const std::unordered_set<std::string> kMethod = {"prune", "quant", "prune+quant", "distill+quant"};
    static const std::unordered_set<std::string> kQuant = {"fp32", "fp16", "int8", "int4"};

    bool enabled = payload.value("enabled", true);
    std::string method = trimCopy(payload.value("method", std::string("prune+quant")));
    if (method.empty())
        method = "prune+quant";
    if (!inSet(method, kMethod))
        method = "prune+quant";
    double pruneRatio = payload.value("pruneRatio", 0.15);
    pruneRatio = clampDouble(pruneRatio, 0.0, 0.95);
    std::string quant = trimCopy(payload.value("quant", std::string("int8")));
    if (quant.empty())
        quant = "int8";
    if (!inSet(quant, kQuant))
        quant = "int8";

    const double remainRatio = std::max(0.05, 1.0 - pruneRatio);
    const double qRatio = quantRatio(quant);
    const double estimatedSizeRatio = clampDouble(remainRatio * qRatio, 0.03, 1.0);
    const double estimatedSpeedup = clampDouble(1.0 / std::max(0.2, estimatedSizeRatio), 1.0, 5.0);
    const std::string applyWindow = trimCopy(payload.value("applyWindow", std::string("offpeak")));

    compression_ = json{{"enabled", enabled},
                        {"method", method},
                        {"pruneRatio", pruneRatio},
                        {"quant", quant},
                        {"estimatedSizeRatio", estimatedSizeRatio},
                        {"estimatedSpeedup", estimatedSpeedup},
                        {"applyWindow", applyWindow.empty() ? "offpeak" : applyWindow},
                        {"note", "plan-only, algorithm remains GNN+Transformer"},
                        {"updatedAtMs", nowMs()}};
    events_.push_back(json{{"type", "compress"},
                           {"atMs", nowMs()},
                           {"payload", compression_},
                           {"summary", json{{"estimatedSizeRatio", estimatedSizeRatio}, {"estimatedSpeedup", estimatedSpeedup}}}});
    if (events_.size() > 64)
        events_.erase(events_.begin());
    persistManifest();
    return json{{"ok", true}, {"result", compression_}};
}

json ModelLifecycleManager::explainOutput(const json &payload)
{
    std::lock_guard<std::mutex> lock(mu_);
    if (!payload.is_object()) {
        return json{{"ok", false}, {"error", "payload must be an object"}};
    }
    std::string text = payload.value("text", "");
    std::string reply = payload.value("reply", "");
    std::string graph = payload.value("graphContext", "");
    std::string method = trimCopy(payload.value("method", std::string("verify+token-overlap")));
    if (method.empty())
        method = "verify+token-overlap";

    auto tokenize = [](const std::string &s)
    {
        std::vector<std::string> out;
        std::string cur;
        for (unsigned char ch : s)
        {
            if (std::isalnum(ch) || ch >= 0x80)
            {
                cur.push_back((char)std::tolower(ch));
            }
            else if (!cur.empty())
            {
                out.push_back(cur);
                cur.clear();
            }
        }
        if (!cur.empty())
            out.push_back(cur);
        return out;
    };

    auto inTokens = tokenize(text);
    auto outTokens = tokenize(reply);
    auto graphTokens = tokenize(graph);
    std::unordered_set<std::string> inSet(inTokens.begin(), inTokens.end());
    std::unordered_set<std::string> graphSet(graphTokens.begin(), graphTokens.end());
    int supportedByInput = 0;
    int supportedByGraph = 0;
    std::vector<std::string> salient;
    for (const auto &tk : outTokens)
    {
        bool hitInput = inSet.count(tk) > 0;
        bool hitGraph = graphSet.count(tk) > 0;
        if (hitInput)
            supportedByInput++;
        if (hitGraph)
            supportedByGraph++;
        if ((hitInput || hitGraph) && salient.size() < 16)
            salient.push_back(tk);
    }
    double denom = (double)std::max<size_t>(1, outTokens.size());
    double supportScore = std::min(1.0, (supportedByInput * 0.7 + supportedByGraph * 0.3) / denom);

    explainability_ = json{{"enabled", true}, {"method", method}, {"updatedAtMs", nowMs()}};
    events_.push_back(json{{"type", "explain"},
                           {"atMs", nowMs()},
                           {"supportScore", supportScore},
                           {"salientTokens", salient}});
    if (events_.size() > 64)
        events_.erase(events_.begin());
    persistManifest();

    return json{{"ok", true},
                {"result",
                 json{{"method", method},
                      {"supportScore", supportScore},
                      {"supportedByInput", supportedByInput},
                      {"supportedByGraph", supportedByGraph},
                      {"outputTokenCount", (int)outTokens.size()},
                      {"salientTokens", salient}}}};
}

json ModelLifecycleManager::deployTarget(const json &payload)
{
    std::lock_guard<std::mutex> lock(mu_);
    static const std::unordered_set<std::string> kTarget = {"local", "windows-local", "edge", "cloud", "hybrid"};
    static const std::unordered_set<std::string> kRouting = {"latency-aware", "weighted-round-robin", "least-loaded", "sticky-primary"};

    std::string target = trimCopy(payload.value("target", std::string("local")));
    std::string version = trimCopy(payload.value("version", std::string("v3.0")));
    if (target.empty())
        target = "local";
    if (!inSet(target, kTarget))
        target = "local";
    if (version.empty())
        version = "v3.0";

    int replicas = clampInt(payload.value("replicas", 1), 1, 64);
    std::string routing = trimCopy(payload.value("routingPolicy", std::string("latency-aware")));
    if (routing.empty() || !inSet(routing, kRouting))
        routing = "latency-aware";
    int canary = clampInt(payload.value("canaryPercent", 10), 0, 100);

    activeTarget_ = target;
    activeVersion_ = version;
    servingCluster_ = json{{"replicas", replicas},
                           {"routingPolicy", routing},
                           {"fallback", payload.value("fallback", "sticky-primary")},
                           {"updatedAtMs", nowMs()}};
    deployment_ = json{{"target", activeTarget_},
                       {"version", activeVersion_},
                       {"rolling", payload.value("rolling", true)},
                       {"canaryPercent", canary},
                       {"cluster", servingCluster_},
                       {"updatedAtMs", nowMs()}};
    events_.push_back(json{{"type", "deploy"},
                           {"atMs", nowMs()},
                           {"target", target},
                           {"version", version},
                           {"cluster", json{{"replicas", replicas}, {"routingPolicy", routing}, {"canaryPercent", canary}}}});
    if (events_.size() > 64)
        events_.erase(events_.begin());
    persistManifest();
    return json{{"ok", true}, {"result", deployment_}};
}

json ModelLifecycleManager::applyOnlineUpdate(const json &payload)
{
    std::lock_guard<std::mutex> lock(mu_);
    static const std::unordered_set<std::string> kStrategy = {"incremental", "online", "hot-swap", "rollback"};

    std::string package = trimCopy(payload.value("package", std::string("")));
    std::string checksum = trimCopy(payload.value("checksum", std::string("")));
    std::string strategy = trimCopy(payload.value("strategy", std::string("incremental")));
    if (strategy.empty())
        strategy = "incremental";
    if (!inSet(strategy, kStrategy))
        strategy = "incremental";
    if (package.empty())
        package = "unspecified";

    std::string beforeVersion = activeVersion_;
    std::string activateVersion = trimCopy(payload.value("activateVersion", std::string("")));
    if (!activateVersion.empty())
        activeVersion_ = activateVersion;

    updateSeq_ += 1;

    onlineUpdate_ = json{{"enabled", true},
                         {"lastPackage", package},
                         {"lastChecksum", checksum},
                         {"strategy", strategy},
                         {"seq", updateSeq_},
                         {"beforeVersion", beforeVersion},
                         {"activeVersion", activeVersion_},
                         {"warmupBatches", clampInt(payload.value("warmupBatches", 3), 0, 1000)},
                         {"updatedAtMs", nowMs()}};
    events_.push_back(json{{"type", "update"},
                           {"atMs", nowMs()},
                           {"package", package},
                           {"checksum", checksum},
                           {"strategy", strategy},
                           {"seq", updateSeq_},
                           {"beforeVersion", beforeVersion},
                           {"afterVersion", activeVersion_}});
    if (events_.size() > 64)
        events_.erase(events_.begin());
    persistManifest();
    return json{{"ok", true},
                {"result", onlineUpdate_},
                {"deployment", json{{"target", activeTarget_}, {"activeVersion", activeVersion_}}}};
}

void ModelLifecycleManager::loadManifestIfExists()
{
    fs::path manifest = fs::path("runtime_store") / "model_lifecycle" / "manifest.json";
    if (!fs::exists(manifest))
        return;

    std::ifstream in(manifest, std::ios::binary);
    if (!in.is_open())
        return;

    std::stringstream ss;
    ss << in.rdbuf();
    json doc = json::parse(ss.str(), nullptr, false);
    if (doc.is_discarded() || !doc.is_object())
        return;

    activeTarget_ = doc.value("activeTarget", activeTarget_);
    activeVersion_ = doc.value("activeVersion", activeVersion_);
    if (doc.contains("compression") && doc["compression"].is_object())
        compression_ = doc["compression"];
    if (doc.contains("explainability") && doc["explainability"].is_object())
        explainability_ = doc["explainability"];
    if (doc.contains("deployment") && doc["deployment"].is_object())
        deployment_ = doc["deployment"];
    if (doc.contains("onlineUpdate") && doc["onlineUpdate"].is_object())
        onlineUpdate_ = doc["onlineUpdate"];
    if (doc.contains("servingCluster") && doc["servingCluster"].is_object())
        servingCluster_ = doc["servingCluster"];
    updateSeq_ = doc.value("updateSeq", (uint64_t)0);
    if (doc.contains("events") && doc["events"].is_array())
    {
        events_.clear();
        for (const auto &e : doc["events"])
        {
            if (e.is_object())
                events_.push_back(e);
        }
        if (events_.size() > 64)
            events_.erase(events_.begin(), events_.end() - 64);
    }
}

void ModelLifecycleManager::persistManifest() const
{
    fs::path outDir = fs::path("runtime_store") / "model_lifecycle";
    fs::create_directories(outDir);
    json doc;
    doc["schema"] = "v3.0-model-lifecycle";
    doc["savedAtMs"] = nowMs();
    doc["activeTarget"] = activeTarget_;
    doc["activeVersion"] = activeVersion_;
    doc["compression"] = compression_;
    doc["explainability"] = explainability_;
    doc["deployment"] = deployment_;
    doc["onlineUpdate"] = onlineUpdate_;
    doc["servingCluster"] = servingCluster_;
    doc["updateSeq"] = updateSeq_;
    doc["events"] = events_;
    std::ofstream out(outDir / "manifest.json", std::ios::binary | std::ios::trunc);
    if (out.is_open())
        out << doc.dump(2);
}

} // namespace model_lifecycle
