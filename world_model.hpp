#pragma once

#include "DATABASE_079.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace world_model {

using json = nlohmann::json;

inline std::int64_t nowMs() {
    return static_cast<std::int64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                         std::chrono::system_clock::now().time_since_epoch())
                                         .count());
}

inline std::uint64_t stableHash64(const std::string &text) {
    std::uint64_t hash = 1469598103934665603ull;
    for (unsigned char ch : text) {
        hash ^= ch;
        hash *= 1099511628211ull;
    }
    return hash;
}

inline std::string trimCopy(const std::string &value) {
    std::size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start])) != 0) {
        ++start;
    }
    std::size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
        --end;
    }
    return value.substr(start, end - start);
}

inline std::string lowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

inline std::string truncateText(const std::string &value, std::size_t limit) {
    if (value.size() <= limit) {
        return value;
    }
    if (limit < 4) {
        return value.substr(0, limit);
    }
    return value.substr(0, limit - 3) + "...";
}

inline std::vector<std::string> extractKeywords(const std::string &text, std::size_t maxCount) {
    std::vector<std::string> out;
    std::unordered_set<std::string> seen;
    std::string current;
    auto flushToken = [&]() {
        if (current.size() >= 3) {
            auto lowered = lowerCopy(current);
            if (seen.insert(lowered).second) {
                out.push_back(lowered);
            }
        }
        current.clear();
    };
    for (unsigned char ch : text) {
        if (std::isalnum(ch) != 0 || ch == '_') {
            current.push_back(static_cast<char>(ch));
        } else {
            flushToken();
            if (out.size() >= maxCount) {
                break;
            }
        }
    }
    if (out.size() < maxCount) {
        flushToken();
    }
    if (out.size() > maxCount) {
        out.resize(maxCount);
    }
    return out;
}

struct PromptContextOptions {
    std::size_t maxRecentEvidence{3};
    std::size_t maxEvidenceChars{96};
    std::size_t maxSummaryChars{220};
    std::size_t maxObjectSlots{6};
};

struct HotspotAnalysis {
    std::string scope;
    std::string canonicalKey;
    std::vector<std::string> labels;
    double hotScore{0.0};
    bool hotCandidate{false};
};

class SelectiveKvCache {
public:
    struct Options {
        std::size_t hotLimit{128};
        int promoteHits{2};
        std::int64_t hotTtlMs{15 * 60 * 1000};
    };

    explicit SelectiveKvCache(std::shared_ptr<KeyValueStore> coldStore = nullptr)
        : SelectiveKvCache(std::move(coldStore), Options{}) {}

    SelectiveKvCache(std::shared_ptr<KeyValueStore> coldStore,
                     Options options)
        : coldStore_(std::move(coldStore)), options_(options) {}

    std::optional<json> get(const std::string &key) {
        const auto now = nowMs();
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto it = hot_.find(key);
            if (it != hot_.end()) {
                if (options_.hotTtlMs > 0 && now - it->second.lastAccessMs > options_.hotTtlMs) {
                    hot_.erase(it);
                } else {
                    it->second.hits += 1;
                    it->second.lastAccessMs = now;
                    return std::make_optional<json>(it->second.value);
                }
            }
        }

        if (!coldStore_) {
            return std::nullopt;
        }

        auto stored = coldStore_->get(key);
        if (!stored) {
            return std::nullopt;
        }

        HotspotAnalysis analysis;
        const json payload = unwrapStoredValue(*stored, &analysis);
        bool promote = analysis.hotCandidate;
        {
            std::lock_guard<std::mutex> lock(mu_);
            int &touches = coldTouches_[key];
            touches += 1;
            promote = promote || touches >= std::max(1, options_.promoteHits);
            if (promote) {
                hot_[key] = CacheEntry{payload, analysis, touches, now, now};
                trimHotLocked();
            }
        }
        return std::make_optional<json>(payload);
    }

    void put(const std::string &key, const json &value, const HotspotAnalysis &analysis = {}) {
        const auto now = nowMs();
        if (coldStore_) {
            coldStore_->put(key, wrapStoredValue(value, analysis, now));
        }

        std::lock_guard<std::mutex> lock(mu_);
        int hits = std::max(1, coldTouches_[key] + 1);
        coldTouches_[key] = hits;
        if (analysis.hotCandidate || hits >= std::max(1, options_.promoteHits)) {
            hot_[key] = CacheEntry{value, analysis, hits, now, now};
            trimHotLocked();
        }
    }

    bool hasHot(const std::string &key) const {
        std::lock_guard<std::mutex> lock(mu_);
        return hot_.find(key) != hot_.end();
    }

    json stats() const {
        std::lock_guard<std::mutex> lock(mu_);
        json hotKeys = json::array();
        for (const auto &entry : hot_) {
            hotKeys.push_back(entry.first);
        }
        return json{{"hotSize", hot_.size()},
                    {"hotLimit", options_.hotLimit},
                    {"promoteHits", options_.promoteHits},
                    {"hasColdStore", static_cast<bool>(coldStore_)},
                    {"hotKeys", hotKeys}};
    }

private:
    struct CacheEntry {
        json value;
        HotspotAnalysis hotspot;
        int hits{0};
        std::int64_t createdAtMs{0};
        std::int64_t lastAccessMs{0};
    };

    std::shared_ptr<KeyValueStore> coldStore_;
    Options options_;
    mutable std::mutex mu_;
    std::unordered_map<std::string, CacheEntry> hot_;
    std::unordered_map<std::string, int> coldTouches_;

    static json wrapStoredValue(const json &value,
                                const HotspotAnalysis &analysis,
                                std::int64_t updatedAtMs) {
        json meta = json::object();
        meta["updatedAtMs"] = updatedAtMs;
        meta["scope"] = analysis.scope;
        meta["canonicalKey"] = analysis.canonicalKey;
        meta["hotScore"] = analysis.hotScore;
        meta["hotCandidate"] = analysis.hotCandidate;
        meta["labels"] = analysis.labels;
        return json{{"payload", value}, {"meta", meta}};
    }

    static json unwrapStoredValue(const json &stored, HotspotAnalysis *analysis) {
        if (stored.is_object() && stored.contains("payload")) {
            if (analysis && stored.contains("meta") && stored["meta"].is_object()) {
                const auto &meta = stored["meta"];
                analysis->scope = meta.value("scope", std::string());
                analysis->canonicalKey = meta.value("canonicalKey", std::string());
                analysis->hotScore = meta.value("hotScore", 0.0);
                analysis->hotCandidate = meta.value("hotCandidate", false);
                if (meta.contains("labels") && meta["labels"].is_array()) {
                    for (const auto &entry : meta["labels"]) {
                        if (entry.is_string()) {
                            analysis->labels.push_back(entry.get<std::string>());
                        }
                    }
                }
            }
            return stored["payload"];
        }
        return stored;
    }

    void trimHotLocked() {
        while (hot_.size() > std::max<std::size_t>(1, options_.hotLimit)) {
            auto evict = hot_.begin();
            for (auto it = hot_.begin(); it != hot_.end(); ++it) {
                if (it->second.lastAccessMs < evict->second.lastAccessMs) {
                    evict = it;
                }
            }
            hot_.erase(evict);
        }
    }
};

enum class CompiledInvocationSymbol : std::uint8_t {
    ProviderOllama,
    ProviderLlamaCpp,
    ProviderBitNet,
    ProviderNative,
    HasGraphContext,
    NoGraphContext,
    HasAddonContract,
    NoAddonContract,
};

enum class CompiledInvocationState : std::uint8_t {
    Start,
    Ollama,
    LlamaCpp,
    BitNet,
    Native,
    ExternalWithContext,
    ExternalLean,
    NativeDirect,
};

enum class CompiledInvocationKind : std::uint8_t {
    OllamaAdapter,
    LlamaCppEmbedded,
    BitNetEmbedded,
    NativeDirect,
};

struct CompiledInvocationTransition {
    CompiledInvocationState from;
    CompiledInvocationSymbol symbol;
    CompiledInvocationState to;
};

struct CompiledInvocationPlan {
    CompiledInvocationKind kind{CompiledInvocationKind::NativeDirect};
    CompiledInvocationState finalState{CompiledInvocationState::NativeDirect};
    std::string_view planId{"compiled-native-direct"};
    std::string_view requestPath{""};
    bool useExternalAdapter{false};
    bool useLlamacppWorldShell{false};
    bool usePhoenixGuidanceShell{false};
    bool injectRuntimeIdentity{false};
    bool dispatchStyleTrainStep{false};
    bool compiledEmbedding{true};
};

constexpr std::array<CompiledInvocationTransition, 12> kCompiledInvocationTransitions{{
    {CompiledInvocationState::Start, CompiledInvocationSymbol::ProviderOllama, CompiledInvocationState::Ollama},
    {CompiledInvocationState::Start, CompiledInvocationSymbol::ProviderLlamaCpp, CompiledInvocationState::LlamaCpp},
    {CompiledInvocationState::Start, CompiledInvocationSymbol::ProviderBitNet, CompiledInvocationState::BitNet},
    {CompiledInvocationState::Start, CompiledInvocationSymbol::ProviderNative, CompiledInvocationState::Native},
    {CompiledInvocationState::Ollama, CompiledInvocationSymbol::HasGraphContext, CompiledInvocationState::ExternalWithContext},
    {CompiledInvocationState::Ollama, CompiledInvocationSymbol::NoGraphContext, CompiledInvocationState::ExternalLean},
    {CompiledInvocationState::LlamaCpp, CompiledInvocationSymbol::HasGraphContext, CompiledInvocationState::ExternalWithContext},
    {CompiledInvocationState::LlamaCpp, CompiledInvocationSymbol::NoGraphContext, CompiledInvocationState::ExternalLean},
    {CompiledInvocationState::BitNet, CompiledInvocationSymbol::HasGraphContext, CompiledInvocationState::ExternalWithContext},
    {CompiledInvocationState::BitNet, CompiledInvocationSymbol::NoGraphContext, CompiledInvocationState::ExternalLean},
    {CompiledInvocationState::Native, CompiledInvocationSymbol::HasGraphContext, CompiledInvocationState::NativeDirect},
    {CompiledInvocationState::Native, CompiledInvocationSymbol::NoGraphContext, CompiledInvocationState::NativeDirect},
}};

constexpr CompiledInvocationState stepCompiledInvocationState(CompiledInvocationState current,
                                                              CompiledInvocationSymbol symbol) {
    for (const auto &transition : kCompiledInvocationTransitions) {
        if (transition.from == current && transition.symbol == symbol) {
            return transition.to;
        }
    }
    return current;
}

constexpr CompiledInvocationPlan finalizeCompiledInvocationPlan(CompiledInvocationState providerState,
                                                                CompiledInvocationState finalState,
                                                                bool hasAddonContract) {
    switch (providerState) {
    case CompiledInvocationState::LlamaCpp:
        return CompiledInvocationPlan{CompiledInvocationKind::LlamaCppEmbedded,
                                      finalState,
                                      finalState == CompiledInvocationState::ExternalWithContext ? "compiled-llamacpp-context" : "compiled-llamacpp-lean",
                                      "/v1/chat/completions",
                                      true,
                                      finalState == CompiledInvocationState::ExternalWithContext,
                                      false,
                                      true,
                                      !hasAddonContract,
                                      true};
    case CompiledInvocationState::BitNet:
        return CompiledInvocationPlan{CompiledInvocationKind::BitNetEmbedded,
                                      finalState,
                                      finalState == CompiledInvocationState::ExternalWithContext ? "compiled-bitnet-context" : "compiled-bitnet-lean",
                                      "/api/chat",
                                      true,
                                      false,
                                      finalState == CompiledInvocationState::ExternalWithContext,
                                      false,
                                      true,
                                      true};
    case CompiledInvocationState::Ollama:
        return CompiledInvocationPlan{CompiledInvocationKind::OllamaAdapter,
                                      finalState,
                                      finalState == CompiledInvocationState::ExternalWithContext ? "compiled-ollama-context" : "compiled-ollama-lean",
                                      "/api/chat",
                                      true,
                                      false,
                                      finalState == CompiledInvocationState::ExternalWithContext,
                                      false,
                                      false,
                                      true};
    case CompiledInvocationState::Native:
    default:
        return CompiledInvocationPlan{CompiledInvocationKind::NativeDirect,
                                      CompiledInvocationState::NativeDirect,
                                      "compiled-native-direct",
                                      "",
                                      false,
                                      false,
                                      false,
                                      false,
                                      false,
                                      true};
    }
}

inline CompiledInvocationPlan buildCompiledInvocationPlan(const std::string &provider,
                                                          bool hasGraphContext,
                                                          bool hasAddonContract) {
    const auto lowered = lowerCopy(trimCopy(provider));
    CompiledInvocationSymbol providerSymbol = CompiledInvocationSymbol::ProviderNative;
    if (lowered == "ollama" || lowered == "ollama-fine-tuning") {
        providerSymbol = CompiledInvocationSymbol::ProviderOllama;
    } else if (lowered == "llamacpp" || lowered == "llama.cpp" || lowered == "llama_cpp") {
        providerSymbol = CompiledInvocationSymbol::ProviderLlamaCpp;
    } else if (lowered == "bitnet") {
        providerSymbol = CompiledInvocationSymbol::ProviderBitNet;
    }

    CompiledInvocationState providerState = stepCompiledInvocationState(CompiledInvocationState::Start, providerSymbol);
    CompiledInvocationState finalState = stepCompiledInvocationState(providerState,
                                                                     hasGraphContext ? CompiledInvocationSymbol::HasGraphContext
                                                                                     : CompiledInvocationSymbol::NoGraphContext);
    finalState = stepCompiledInvocationState(finalState,
                                             hasAddonContract ? CompiledInvocationSymbol::HasAddonContract
                                                              : CompiledInvocationSymbol::NoAddonContract);
    return finalizeCompiledInvocationPlan(providerState, finalState, hasAddonContract);
}

struct GroundedLearningSample {
    std::string input;
    std::string target;
    std::string graph;
    std::string source;
};

struct GroundedLearningOptions {
    std::size_t maxSamples{4};
    std::size_t maxRecentEvidence{4};
    std::size_t maxInputChars{160};
    std::size_t maxTargetChars{160};
    std::size_t maxGraphChars{512};
    std::size_t maxVideoCompressionLevels{3};
    std::size_t maxVideoTemporalWindows{3};
    bool includeFusionSample{true};
    bool includeSceneRecallSample{true};
    bool includeVideoCompressionSamples{true};
    bool includeVideoTemporalSamples{true};
};

struct CognitiveStateOptions {
    std::size_t maxWorkingMemory{4};
    std::size_t maxAttentionItems{4};
    std::size_t maxGoals{4};
    std::size_t maxRegionItems{4};
    std::size_t maxChars{120};
};

enum class BrainProfileKind {
    Functional,
    Structural,
};

struct BrainProfileOptions {
    BrainProfileKind kind{BrainProfileKind::Functional};
    std::size_t maxWorkingMemory{4};
    std::size_t maxAttentionItems{4};
    std::size_t maxGoals{4};
    std::size_t maxRegionItems{4};
    std::size_t maxChars{120};
};

struct ReasoningAssemblyOptions {
    PromptContextOptions promptOptions{};
    BrainProfileOptions brainOptions{};
    bool includeBrainContext{false};
    bool includeReasoningAgenda{false};
    bool includeReasoningPlan{false};
    std::size_t maxContextChars{160};
    std::size_t maxPlanItems{4};
};

inline std::string brainProfileKindToString(BrainProfileKind kind) {
    switch (kind) {
    case BrainProfileKind::Structural:
        return "structural";
    case BrainProfileKind::Functional:
    default:
        return "functional";
    }
}

inline BrainProfileKind parseBrainProfileKind(const std::string &raw) {
    const auto lowered = lowerCopy(trimCopy(raw));
    if (lowered == "structural" || lowered == "research") {
        return BrainProfileKind::Structural;
    }
    return BrainProfileKind::Functional;
}

inline bool isFunctionalBrainState(const json &state) {
    return state.is_object() && state.value("profile", std::string()) == "functional" && state.contains("corticalSystems");
}

inline bool isStructuralBrainState(const json &state) {
    return state.is_object() && state.value("profile", std::string()) == "structural" && state.contains("corticalMap");
}

struct VirtualSceneOptions {
    std::size_t maxAgents{3};
    std::size_t maxSteps{3};
    std::size_t maxTrainSamples{6};
    std::size_t maxEventChars{140};
    std::size_t mapWidth{6};
    std::size_t mapHeight{6};
    std::size_t mapDepth{3};
    std::size_t maxDialogueTurns{2};
    std::size_t maxEcologyClusters{2};
    bool includeCounterfactual{true};
    bool includeSceneRecallSample{true};
    bool include3DMap{true};
    bool includeEmbodiedAgents{true};
    bool includeEcologyFromVideo{true};
    bool physicsEnabled{true};
    std::size_t physicsSubsteps{4};
    std::string physicsBackend{"bullet3"};
    bool earthMapEnabled{false};
    json earthMapRequest{json::object()};
    BrainProfileKind brainProfile{BrainProfileKind::Functional};
};

inline json buildBrainProfile(const json &sessionState, BrainProfileOptions options);
inline json buildReasoningAgenda(const json &sessionState, std::size_t maxItems, std::size_t maxChars);
inline std::string buildReasoningAgendaPromptContext(const json &sessionState, std::size_t maxCharsPerLine);
inline json buildResponsePlan(const json &sessionState, std::size_t maxItems, std::size_t maxChars);
inline std::string buildResponsePlanPromptContext(const json &sessionState, std::size_t maxCharsPerLine);

inline std::string buildPromptEvidenceText(const json &evidence, const PromptContextOptions &options);

inline json buildReasoningAgendaFromCognitiveState(const json &cognitiveState, std::size_t maxItems, std::size_t maxChars);

inline std::vector<std::string> collectObjectLabels(const json &sceneState, std::size_t maxCount) {
    std::vector<std::string> labels;
    std::unordered_set<std::string> seen;
    if (!sceneState.contains("objectSlots") || !sceneState["objectSlots"].is_array()) {
        return labels;
    }
    for (const auto &slot : sceneState["objectSlots"]) {
        if (!slot.is_object() || !slot.contains("label") || !slot["label"].is_string()) {
            continue;
        }
        std::string label = trimCopy(slot["label"].get<std::string>());
        if (label.empty()) {
            continue;
        }
        auto lowered = lowerCopy(label);
        if (!seen.insert(lowered).second) {
            continue;
        }
        labels.push_back(label);
        if (labels.size() >= maxCount) {
            break;
        }
    }
    return labels;
}

inline std::vector<std::string> collectSceneTags(const json &sceneState, std::size_t maxCount) {
    std::vector<std::string> tags;
    std::unordered_set<std::string> seen;
    if (!sceneState.contains("tags") || !sceneState["tags"].is_array()) {
        return tags;
    }
    for (const auto &entry : sceneState["tags"]) {
        if (!entry.is_string()) {
            continue;
        }
        std::string tag = trimCopy(entry.get<std::string>());
        if (tag.empty()) {
            continue;
        }
        auto lowered = lowerCopy(tag);
        if (!seen.insert(lowered).second) {
            continue;
        }
        tags.push_back(tag);
        if (tags.size() >= maxCount) {
            break;
        }
    }
    return tags;
}

inline std::string buildStableCacheKey(const std::string &scope,
                                       const std::string &seed,
                                       const std::string &suffix = std::string()) {
    std::ostringstream oss;
    oss << scope << ":" << std::hex << stableHash64(seed);
    if (!suffix.empty()) {
        oss << ":" << suffix;
    }
    return oss.str();
}

inline HotspotAnalysis analyzeReasoningHotspot(const std::string &scope,
                                               const json &sessionState,
                                               const std::string &initialGraphContext = std::string()) {
    const json sceneState = (sessionState.contains("sceneState") && sessionState["sceneState"].is_object()) ? sessionState["sceneState"] : json::object();
    const json episode = (sessionState.contains("episode") && sessionState["episode"].is_object()) ? sessionState["episode"] : json::object();
    const json recentEvidence = (sessionState.contains("recentEvidence") && sessionState["recentEvidence"].is_array()) ? sessionState["recentEvidence"] : json::array();

    HotspotAnalysis out;
    out.scope = scope;
    out.labels = collectObjectLabels(sceneState, 4);
    const auto tags = collectSceneTags(sceneState, 4);
    out.labels.insert(out.labels.end(), tags.begin(), tags.end());
    if (out.labels.size() > 6) {
        out.labels.resize(6);
    }

    double hotScore = 0.0;
    if (!trimCopy(sceneState.value("summary", std::string())).empty()) {
        hotScore += 0.30;
    }
    if (!trimCopy(episode.value("summary", std::string())).empty()) {
        hotScore += 0.20;
    }
    hotScore += std::min<std::size_t>(out.labels.size(), 4) * 0.10;
    hotScore += std::min<std::size_t>(recentEvidence.size(), 3) * 0.10;
    if (!trimCopy(initialGraphContext).empty()) {
        hotScore += 0.10;
    }
    out.hotScore = std::min(1.0, hotScore);
    out.hotCandidate = out.hotScore >= 0.55;

    std::ostringstream labels;
    for (std::size_t index = 0; index < out.labels.size(); ++index) {
        if (index > 0) {
            labels << ',';
        }
        labels << out.labels[index];
    }

    std::ostringstream signature;
    signature << sessionState.value("sessionId", std::string()) << '|'
              << sceneState.value("summary", std::string()) << '|'
              << episode.value("summary", std::string()) << '|'
              << labels.str() << '|'
              << truncateText(initialGraphContext, 160) << '|'
              << recentEvidence.size();
    out.canonicalKey = buildStableCacheKey(scope, signature.str());
    return out;
}

inline HotspotAnalysis analyzeReplyHotspot(const std::string &text,
                                           const std::string &modelGraphContext,
                                           int maxTokens) {
    HotspotAnalysis out;
    out.scope = "reply-hotspot";
    out.labels = extractKeywords(text + " " + modelGraphContext, 6);
    double hotScore = 0.0;
    if (text.size() <= 256) {
        hotScore += 0.30;
    }
    if (!trimCopy(modelGraphContext).empty()) {
        hotScore += 0.25;
    }
    if (maxTokens <= 192) {
        hotScore += 0.15;
    }
    hotScore += std::min<std::size_t>(out.labels.size(), 4) * 0.10;
    out.hotScore = std::min(1.0, hotScore);
    out.hotCandidate = out.hotScore >= 0.60;
    out.canonicalKey = buildStableCacheKey(out.scope,
                                           text + "|" + truncateText(modelGraphContext, 192) + "|" + std::to_string(maxTokens));
    return out;
}

inline std::string joinStrings(const std::vector<std::string> &items, const std::string &separator, std::size_t maxItems = 0) {
    std::ostringstream oss;
    const std::size_t limit = maxItems > 0 ? std::min(maxItems, items.size()) : items.size();
    for (std::size_t index = 0; index < limit; ++index) {
        if (index > 0) {
            oss << separator;
        }
        oss << items[index];
    }
    return oss.str();
}

inline void appendUniqueLimited(std::vector<std::string> &items,
                                std::unordered_set<std::string> &seen,
                                const std::string &value,
                                std::size_t maxCount,
                                std::size_t maxChars = 0) {
    if (items.size() >= maxCount) {
        return;
    }
    std::string normalized = trimCopy(value);
    if (maxChars > 0) {
        normalized = truncateText(normalized, maxChars);
    }
    if (normalized.empty()) {
        return;
    }
    const auto lowered = lowerCopy(normalized);
    if (!seen.insert(lowered).second) {
        return;
    }
    items.push_back(normalized);
}

inline json toJsonArray(const std::vector<std::string> &items) {
    json out = json::array();
    for (const auto &item : items) {
        out.push_back(item);
    }
    return out;
}

inline std::vector<std::string> collectJsonStrings(const json &value, std::size_t maxCount) {
    std::vector<std::string> out;
    std::unordered_set<std::string> seen;
    if (value.is_string()) {
        appendUniqueLimited(out, seen, value.get<std::string>(), maxCount);
        return out;
    }
    if (!value.is_array()) {
        return out;
    }
    for (const auto &entry : value) {
        if (!entry.is_string()) {
            continue;
        }
        appendUniqueLimited(out, seen, entry.get<std::string>(), maxCount);
        if (out.size() >= maxCount) {
            break;
        }
    }
    return out;
}

inline std::vector<std::string> collectRelationSummaries(const json &relations, std::size_t maxCount) {
    std::vector<std::string> out;
    std::unordered_set<std::string> seen;

    auto appendRelation = [&](const std::string &summary) {
        appendUniqueLimited(out, seen, summary, maxCount, 120);
    };

    if (relations.is_object()) {
        const std::string subject = trimCopy(relations.value("subject", std::string()));
        const std::string predicate = trimCopy(relations.value("predicate", std::string()));
        const std::string object = trimCopy(relations.value("object", std::string()));
        const std::string summary = !subject.empty() || !predicate.empty() || !object.empty()
                                        ? trimCopy(subject + " " + predicate + " " + object)
                                        : trimCopy(relations.value("summary", std::string()));
        if (!summary.empty()) {
            appendRelation(summary);
        }
        return out;
    }
    if (!relations.is_array()) {
        return out;
    }

    for (const auto &entry : relations) {
        std::string summary;
        if (entry.is_string()) {
            summary = trimCopy(entry.get<std::string>());
        } else if (entry.is_object()) {
            const std::string subject = trimCopy(entry.value("subject", std::string()));
            const std::string predicate = trimCopy(entry.value("predicate", std::string()));
            const std::string object = trimCopy(entry.value("object", std::string()));
            summary = !subject.empty() || !predicate.empty() || !object.empty()
                          ? trimCopy(subject + " " + predicate + " " + object)
                          : trimCopy(entry.value("summary", std::string()));
        }
        if (summary.empty()) {
            continue;
        }
        appendRelation(summary);
        if (out.size() >= maxCount) {
            break;
        }
    }

    return out;
}

inline double computeStringSetOverlap(const std::vector<std::string> &left, const std::vector<std::string> &right) {
    if (left.empty() && right.empty()) {
        return 1.0;
    }
    if (left.empty() || right.empty()) {
        return 0.0;
    }

    std::unordered_set<std::string> leftSet;
    std::unordered_set<std::string> rightSet;
    for (const auto &item : left) {
        const auto normalized = lowerCopy(trimCopy(item));
        if (!normalized.empty()) {
            leftSet.insert(normalized);
        }
    }
    for (const auto &item : right) {
        const auto normalized = lowerCopy(trimCopy(item));
        if (!normalized.empty()) {
            rightSet.insert(normalized);
        }
    }
    if (leftSet.empty() && rightSet.empty()) {
        return 1.0;
    }
    if (leftSet.empty() || rightSet.empty()) {
        return 0.0;
    }

    std::size_t overlap = 0;
    for (const auto &item : leftSet) {
        if (rightSet.count(item) > 0) {
            ++overlap;
        }
    }

    const double unionSize = static_cast<double>(leftSet.size() + rightSet.size() - overlap);
    return unionSize > 0.0 ? static_cast<double>(overlap) / unionSize : 0.0;
}

inline double clampUnitValue(double value) {
    return std::max(0.0, std::min(1.0, value));
}

inline bool containsKeywordToken(const std::string &text, std::initializer_list<const char *> keywords) {
    const std::string lowered = lowerCopy(text);
    for (const auto *keyword : keywords) {
        if (keyword != nullptr && *keyword != '\0' && lowered.find(keyword) != std::string::npos) {
            return true;
        }
    }
    return false;
}

inline json buildGroundMobilityPlan(const json &sessionState,
                                    const json &agenda,
                                    const std::vector<std::string> &focus,
                                    const std::vector<std::string> &goals,
                                    std::size_t maxItems,
                                    std::size_t maxChars,
                                    bool forceEnabled = false,
                                    const json &earthMap = json::object()) {
    json sceneState = (sessionState.contains("sceneState") && sessionState["sceneState"].is_object()) ? sessionState["sceneState"] : json::object();
    json episode = (sessionState.contains("episode") && sessionState["episode"].is_object()) ? sessionState["episode"] : json::object();

    const auto objectLabels = collectObjectLabels(sceneState, std::max<std::size_t>(std::size_t(3), maxItems));
    const auto sceneTags = collectSceneTags(sceneState, std::max<std::size_t>(std::size_t(4), maxItems));
    const auto openQuestions = collectJsonStrings(agenda.value("openQuestions", json::array()), maxItems);
    const auto contradictions = collectJsonStrings(agenda.value("contradictions", json::array()), maxItems);

    const std::string sceneSummary = trimCopy(sessionState.value("sceneSummary", sceneState.value("summary", std::string())));
    const std::string episodeSummary = trimCopy(sessionState.value("episodeSummary", episode.value("summary", std::string())));
    const std::string baseSummary = !episodeSummary.empty() ? episodeSummary : sceneSummary;
    const std::string routeGoal = !goals.empty() ? goals.front()
                                                 : (agenda.contains("nextStep") && agenda["nextStep"].is_string() ? agenda["nextStep"].get<std::string>()
                                                                                                                      : (!focus.empty() ? std::string("inspect ") + focus.front()
                                                                                                                                        : std::string("advance the current ground mission")));

    std::string relevanceSeed = baseSummary;
    if (!objectLabels.empty()) {
        if (!relevanceSeed.empty()) {
            relevanceSeed += " ";
        }
        relevanceSeed += joinStrings(objectLabels, " ");
    }
    if (!sceneTags.empty()) {
        if (!relevanceSeed.empty()) {
            relevanceSeed += " ";
        }
        relevanceSeed += joinStrings(sceneTags, " ");
    }
    if (!goals.empty()) {
        if (!relevanceSeed.empty()) {
            relevanceSeed += " ";
        }
        relevanceSeed += joinStrings(goals, " ");
    }
    if (!focus.empty()) {
        if (!relevanceSeed.empty()) {
            relevanceSeed += " ";
        }
        relevanceSeed += joinStrings(focus, " ");
    }

    const bool relevant = forceEnabled ||
                          earthMap.value("enabled", false) ||
                          containsKeywordToken(relevanceSeed,
                                               {"rover", "terrain", "ground", "route", "path", "wheel", "vehicle", "trail", "repair", "inspect", "walkway", "slope", "obstacle", "marsh", "solar panel", "bot"});
    if (!relevant) {
        return json::object();
    }

    const double uncertainty = agenda.value("uncertainty", sessionState.value("uncertainty", 0.0));

    std::string terrainClass = "mixed firm ground";
    if (earthMap.value("enabled", false)) {
        terrainClass = "earth-heightfield terrain";
    } else if (containsKeywordToken(relevanceSeed, {"marsh", "water", "mud", "wet", "waterline"})) {
        terrainClass = "soft wet ground";
    } else if (containsKeywordToken(relevanceSeed, {"slope", "ridge", "hill", "relief", "rock", "rough"})) {
        terrainClass = "rough sloped ground";
    } else if (containsKeywordToken(relevanceSeed, {"walkway", "pad", "floor", "concrete", "dock", "corridor"})) {
        terrainClass = "structured hard ground";
    }

    std::vector<std::string> hazards;
    std::unordered_set<std::string> hazardSeen;
    if (containsKeywordToken(relevanceSeed, {"marsh", "water", "mud", "wet", "waterline"})) {
        appendUniqueLimited(hazards, hazardSeen, "soft ground may reduce wheel traction", maxItems, maxChars);
    }
    if (earthMap.value("enabled", false) || containsKeywordToken(relevanceSeed, {"slope", "ridge", "hill", "relief", "rough"})) {
        appendUniqueLimited(hazards, hazardSeen, "grade changes may destabilize the chassis during turns", maxItems, maxChars);
    }
    if (containsKeywordToken(relevanceSeed, {"damaged", "crack", "fault", "debris", "jam", "unstable"})) {
        appendUniqueLimited(hazards, hazardSeen, "damaged obstacles may block the preferred route", maxItems, maxChars);
    }
    if (hazards.empty()) {
        appendUniqueLimited(hazards, hazardSeen, "unknown obstacles beyond current evidence coverage", maxItems, maxChars);
    }

    std::vector<std::string> routeWaypoints;
    std::unordered_set<std::string> waypointSeen;
    appendUniqueLimited(routeWaypoints, waypointSeen, "hold a staging position on stable ground before committing to the route", maxItems, maxChars);
    if (!focus.empty()) {
        appendUniqueLimited(routeWaypoints, waypointSeen, "approach the mission focus from a line with escape clearance: " + focus.front(), maxItems, maxChars);
    }
    if (!objectLabels.empty()) {
        appendUniqueLimited(routeWaypoints, waypointSeen, "use landmark alignment around " + objectLabels.front() + " to anchor route corrections", maxItems, maxChars);
    }
    appendUniqueLimited(routeWaypoints, waypointSeen, "finish at the task waypoint for " + routeGoal, maxItems, maxChars);

    std::vector<std::string> terrainHypotheses;
    std::unordered_set<std::string> terrainSeen;
    appendUniqueLimited(terrainHypotheses, terrainSeen, "primary terrain class: " + terrainClass, maxItems, maxChars);
    if (!sceneTags.empty()) {
        appendUniqueLimited(terrainHypotheses, terrainSeen, "scene cues imply surface tags: " + joinStrings(sceneTags, ", ", 3), maxItems, maxChars);
    }
    if (earthMap.value("enabled", false) && earthMap.contains("regionLabel") && earthMap["regionLabel"].is_string()) {
        appendUniqueLimited(terrainHypotheses, terrainSeen, "earth map anchor: " + earthMap["regionLabel"].get<std::string>(), maxItems, maxChars);
    }

    std::vector<std::string> evidenceNeeds;
    std::unordered_set<std::string> evidenceSeen;
    appendUniqueLimited(evidenceNeeds, evidenceSeen, "confirm traversable ground width before committing to the narrowest segment", maxItems, maxChars);
    appendUniqueLimited(evidenceNeeds, evidenceSeen, "verify obstacle clearance around the final approach", maxItems, maxChars);
    for (const auto &question : openQuestions) {
        appendUniqueLimited(evidenceNeeds, evidenceSeen, question, maxItems, maxChars);
    }

    std::vector<std::string> simulationTargets;
    std::unordered_set<std::string> simulationSeen;
    appendUniqueLimited(simulationTargets, simulationSeen, "simulate rover traction along the first route segment", maxItems, maxChars);
    appendUniqueLimited(simulationTargets, simulationSeen, "simulate stop distance before the final waypoint", maxItems, maxChars);
    if (!hazards.empty()) {
        appendUniqueLimited(simulationTargets, simulationSeen, "simulate avoidance margin for: " + hazards.front(), maxItems, maxChars);
    }
    if (!focus.empty()) {
        appendUniqueLimited(simulationTargets, simulationSeen, "simulate a safe ground approach around " + focus.front(), maxItems, maxChars);
    }

    std::vector<std::string> stabilityChecks;
    std::unordered_set<std::string> stabilitySeen;
    appendUniqueLimited(stabilityChecks, stabilitySeen, "watch pitch and roll before increasing speed on unknown ground", maxItems, maxChars);
    appendUniqueLimited(stabilityChecks, stabilitySeen, "keep a braking reserve before sharp turns or crest transitions", maxItems, maxChars);
    if (!contradictions.empty()) {
        appendUniqueLimited(stabilityChecks, stabilitySeen, "pause route commitment when perception contradictions remain unresolved", maxItems, maxChars);
    }

    std::vector<std::string> reflexPolicies;
    std::unordered_set<std::string> reflexSeen;
    appendUniqueLimited(reflexPolicies, reflexSeen, "prefer wide turns over pivoting on unstable ground", maxItems, maxChars);
    appendUniqueLimited(reflexPolicies, reflexSeen, "drop to crawl speed when traction confidence falls", maxItems, maxChars);
    appendUniqueLimited(reflexPolicies, reflexSeen, "re-center the route after any obstacle bypass maneuver", maxItems, maxChars);

    const std::string routePolicy = uncertainty >= 0.55 || !hazards.empty()
                                        ? std::string("prefer a contour-following route with traction reserve and escape space")
                                        : std::string("prefer the shortest stable ground corridor with periodic evidence refresh");
    const std::string speedPolicy = uncertainty >= 0.55
                                        ? std::string("cautious crawl with stop-to-check windows")
                                        : std::string("measured cruise with conservative braking reserve");
    const std::string selectedAction = uncertainty >= 0.78
                                           ? std::string("hold and verify terrain")
                                           : (uncertainty >= 0.55 ? std::string("probe-forward") : std::string("forward"));
    const std::string tractionPolicy = terrainClass == "soft wet ground"
                                           ? std::string("maximize traction margin and avoid abrupt steering inputs")
                                           : (terrainClass == "rough sloped ground"
                                                  ? std::string("bias for uphill stability and limit cross-slope exposure")
                                                  : std::string("maintain steady wheel loading and avoid wheelspin spikes"));
    const json actionConstraints = json{{"doNotMove", uncertainty >= 0.82 && !forceEnabled},
                                        {"continueStabilizationWhenIdle", true},
                                        {"minVerifyScore", contradictions.empty() ? 0.0 : 0.4},
                                        {"maxUncertainty", !hazards.empty() ? 0.72 : 0.82},
                                        {"cooldownMs", uncertainty >= 0.55 ? 450 : 180},
                                        {"holdScale", 0.65}};

    return json{{"enabled", true},
                {"vehicleClass", "all-terrain-four-wheel-rover"},
                {"environmentConstraint", "ground-only"},
                {"controlLayer", "high-level mobility planning"},
                {"strategicGoal", truncateText(routeGoal, maxChars)},
                {"terrainClass", terrainClass},
                {"hazards", toJsonArray(hazards)},
                {"actionConstraints", actionConstraints},
                {"bigBrain", json{{"missionIntent", truncateText(routeGoal, maxChars)},
                                   {"routePolicy", truncateText(routePolicy, maxChars)},
                                   {"routeWaypoints", toJsonArray(routeWaypoints)},
                                   {"terrainHypotheses", toJsonArray(terrainHypotheses)},
                                   {"evidenceNeeds", toJsonArray(evidenceNeeds)},
                                   {"simulationTargets", toJsonArray(simulationTargets)}}},
                {"littleBrain", json{{"locomotionMode", "four-wheel ground drive with high-clearance chassis"},
                                      {"selectedAction", truncateText(selectedAction, maxChars)},
                                      {"cameraVerify", uncertainty >= 0.55 || !contradictions.empty()},
                                      {"speedPolicy", truncateText(speedPolicy, maxChars)},
                                      {"tractionPolicy", truncateText(tractionPolicy, maxChars)},
                                      {"stabilityChecks", toJsonArray(stabilityChecks)},
                                      {"reflexPolicies", toJsonArray(reflexPolicies)}}}};
}

inline CognitiveStateOptions toCognitiveStateOptions(const BrainProfileOptions &options) {
    CognitiveStateOptions out;
    out.maxWorkingMemory = options.maxWorkingMemory;
    out.maxAttentionItems = options.maxAttentionItems;
    out.maxGoals = options.maxGoals;
    out.maxRegionItems = options.maxRegionItems;
    out.maxChars = options.maxChars;
    return out;
}

inline std::vector<std::string> collectRecentModalitySummaries(const json &recentEvidence,
                                                               std::initializer_list<const char *> acceptedModalities,
                                                               std::size_t maxCount,
                                                               std::size_t maxChars) {
    std::unordered_set<std::string> accepted;
    for (const auto *item : acceptedModalities) {
        accepted.insert(lowerCopy(item));
    }
    std::vector<std::string> out;
    std::unordered_set<std::string> seen;
    PromptContextOptions promptOptions;
    promptOptions.maxEvidenceChars = maxChars;
    if (!recentEvidence.is_array()) {
        return out;
    }
    for (std::size_t index = recentEvidence.size(); index > 0 && out.size() < maxCount; --index) {
        const auto &evidence = recentEvidence[index - 1];
        if (!evidence.is_object()) {
            continue;
        }
        const std::string modality = lowerCopy(trimCopy(evidence.value("modality", std::string())));
        if (accepted.count(modality) == 0) {
            continue;
        }
        appendUniqueLimited(out, seen, buildPromptEvidenceText(evidence, promptOptions), maxCount, maxChars);
    }
    std::reverse(out.begin(), out.end());
    return out;
}

inline std::vector<std::string> deriveActionPrograms(const std::string &sceneSummary,
                                                     const std::string &episodeSummary,
                                                     const std::vector<std::string> &objectLabels,
                                                     std::size_t maxCount,
                                                     std::size_t maxChars) {
    std::vector<std::string> out;
    std::unordered_set<std::string> seen;
    if (!episodeSummary.empty()) {
        appendUniqueLimited(out, seen, "execute next-step policy: " + episodeSummary, maxCount, maxChars);
    }
    if (!objectLabels.empty()) {
        appendUniqueLimited(out, seen, "prepare interaction around " + joinStrings(objectLabels, ", ", 2), maxCount, maxChars);
    }
    if (!sceneSummary.empty()) {
        appendUniqueLimited(out, seen, "stabilize scene trajectory: " + sceneSummary, maxCount, maxChars);
    }
    return out;
}

inline std::vector<std::string> deriveBodySchema(const std::vector<std::string> &objectLabels,
                                                 const std::vector<std::string> &sceneTags,
                                                 const std::string &sceneSummary,
                                                 std::size_t maxCount,
                                                 std::size_t maxChars) {
    std::vector<std::string> out;
    std::unordered_set<std::string> seen;
    if (!objectLabels.empty()) {
        appendUniqueLimited(out, seen, "reachable objects: " + joinStrings(objectLabels, ", ", 3), maxCount, maxChars);
    }
    if (!sceneTags.empty()) {
        appendUniqueLimited(out, seen, "context tags: " + joinStrings(sceneTags, ", ", 3), maxCount, maxChars);
    }
    if (!sceneSummary.empty()) {
        appendUniqueLimited(out, seen, "body-scene estimate: " + sceneSummary, maxCount, maxChars);
    }
    return out;
}

inline std::vector<std::string> collectWorkingMemoryLines(const json &recentEvidence, std::size_t maxCount, std::size_t maxChars) {
    std::vector<std::string> lines;
    PromptContextOptions promptOptions;
    promptOptions.maxEvidenceChars = maxChars;
    if (!recentEvidence.is_array() || recentEvidence.empty()) {
        return lines;
    }
    std::size_t start = recentEvidence.size() > maxCount ? recentEvidence.size() - maxCount : 0;
    for (std::size_t index = start; index < recentEvidence.size(); ++index) {
        if (!recentEvidence[index].is_object()) {
            continue;
        }
        const auto &evidence = recentEvidence[index];
        const std::string summary = buildPromptEvidenceText(evidence, promptOptions);
        if (summary.empty()) {
            continue;
        }
        std::string modality = trimCopy(evidence.value("modality", std::string("evidence")));
        modality = modality.empty() ? std::string("evidence") : lowerCopy(modality);
        lines.push_back(modality + ": " + summary);
    }
    return lines;
}

inline double crossModalConflictScore(const json &recentEvidence) {
    std::string vision;
    std::string speech;
    if (!recentEvidence.is_array()) {
        return 0.0;
    }
    for (std::size_t index = recentEvidence.size(); index > 0; --index) {
        const auto &evidence = recentEvidence[index - 1];
        if (!evidence.is_object()) {
            continue;
        }
        const std::string modality = lowerCopy(trimCopy(evidence.value("modality", std::string())));
        if (vision.empty() && (modality == "vision" || modality == "image")) {
            PromptContextOptions options;
            vision = buildPromptEvidenceText(evidence, options);
        }
        if (speech.empty() && (modality == "speech" || modality == "audio")) {
            PromptContextOptions options;
            speech = buildPromptEvidenceText(evidence, options);
        }
        if (!vision.empty() && !speech.empty()) {
            break;
        }
    }
    if (vision.empty() || speech.empty()) {
        return 0.0;
    }
    const auto visionKeywords = extractKeywords(vision, 12);
    const auto speechKeywords = extractKeywords(speech, 12);
    if (visionKeywords.empty() || speechKeywords.empty()) {
        return 0.0;
    }
    std::unordered_set<std::string> visionSet(visionKeywords.begin(), visionKeywords.end());
    std::unordered_set<std::string> speechSet(speechKeywords.begin(), speechKeywords.end());
    std::size_t overlap = 0;
    for (const auto &token : visionSet) {
        if (speechSet.count(token) > 0) {
            ++overlap;
        }
    }
    const double unionSize = static_cast<double>(visionSet.size() + speechSet.size() - overlap);
    const double jaccard = unionSize > 0.0 ? static_cast<double>(overlap) / unionSize : 0.0;
    return std::max(0.0, 1.0 - jaccard);
}

inline std::string buildPromptEvidenceText(const json &evidence, const PromptContextOptions &options) {
    std::string summary;
    if (evidence.contains("graphSummary") && evidence["graphSummary"].is_string()) {
        summary = trimCopy(evidence["graphSummary"].get<std::string>());
    }
    if (summary.empty() && evidence.contains("text") && evidence["text"].is_string()) {
        summary = trimCopy(evidence["text"].get<std::string>());
    }
    if (summary.empty() && evidence.contains("metadata") && evidence["metadata"].is_object()) {
        const auto &metadata = evidence["metadata"];
        if (metadata.contains("graphContext") && metadata["graphContext"].is_string()) {
            summary = trimCopy(metadata["graphContext"].get<std::string>());
        }
    }
    if (summary.empty()) {
        return std::string();
    }
    return truncateText(summary, options.maxEvidenceChars);
}

inline std::string buildPromptContext(const json &sessionState, PromptContextOptions options = {}) {
    json sceneState = (sessionState.contains("sceneState") && sessionState["sceneState"].is_object()) ? sessionState["sceneState"] : json::object();
    json episode = (sessionState.contains("episode") && sessionState["episode"].is_object()) ? sessionState["episode"] : json::object();
    json recentEvidence = (sessionState.contains("recentEvidence") && sessionState["recentEvidence"].is_array()) ? sessionState["recentEvidence"] : json::array();
    json prediction = (sessionState.contains("prediction") && sessionState["prediction"].is_object()) ? sessionState["prediction"] : json::object();

    std::vector<std::string> lines;

    const std::string sceneSummary = truncateText(trimCopy(sceneState.value("summary", std::string())), options.maxSummaryChars);
    if (!sceneSummary.empty()) {
        lines.push_back("world_scene|summary: " + sceneSummary);
    }

    const std::string episodeSummary = truncateText(trimCopy(episode.value("summary", std::string())), options.maxSummaryChars);
    if (!episodeSummary.empty() && episodeSummary != sceneSummary) {
        lines.push_back("world_episode|summary: " + episodeSummary);
    }

    if (sceneState.contains("objectSlots") && sceneState["objectSlots"].is_array()) {
        std::vector<std::string> labels;
        std::unordered_set<std::string> seen;
        for (const auto &slot : sceneState["objectSlots"]) {
            if (!slot.is_object() || !slot.contains("label") || !slot["label"].is_string()) {
                continue;
            }
            auto label = trimCopy(slot["label"].get<std::string>());
            if (label.empty()) {
                continue;
            }
            auto lowered = lowerCopy(label);
            if (!seen.insert(lowered).second) {
                continue;
            }
            labels.push_back(label);
            if (labels.size() >= options.maxObjectSlots) {
                break;
            }
        }
        if (!labels.empty()) {
            std::ostringstream oss;
            oss << "world_objects|labels: ";
            for (std::size_t index = 0; index < labels.size(); ++index) {
                if (index > 0) {
                    oss << ", ";
                }
                oss << labels[index];
            }
            lines.push_back(oss.str());
        }
    }

    if (sceneState.contains("relations") && sceneState["relations"].is_array()) {
        const auto relations = collectRelationSummaries(sceneState["relations"], std::max<std::size_t>(std::size_t(2), options.maxObjectSlots));
        if (!relations.empty()) {
            lines.push_back("world_relations|active: " + truncateText(joinStrings(relations, " || "), options.maxSummaryChars));
        }
    }

    const auto predictionGoals = collectJsonStrings(prediction.value("goals", json::array()), 2);
    if (!predictionGoals.empty()) {
        lines.push_back("world_goals|active: " + truncateText(joinStrings(predictionGoals, " || "), options.maxSummaryChars));
    }

    const json expectedState = (prediction.contains("expectedNextState") && prediction["expectedNextState"].is_object())
                                   ? prediction["expectedNextState"]
                                   : json::object();
    const json observedState = (prediction.contains("observedNextState") && prediction["observedNextState"].is_object())
                                   ? prediction["observedNextState"]
                                   : json::object();
    const json calibration = (prediction.contains("calibration") && prediction["calibration"].is_object())
                                 ? prediction["calibration"]
                                 : json::object();

    const std::string expectedSummary = truncateText(trimCopy(expectedState.value("summary", std::string())), options.maxSummaryChars);
    if (!expectedSummary.empty()) {
        lines.push_back("world_prediction|expected: " + expectedSummary);
    }

    const std::string observedSummary = truncateText(trimCopy(observedState.value("summary", std::string())), options.maxSummaryChars);
    if (!observedSummary.empty()) {
        lines.push_back("world_prediction|observed: " + observedSummary);
    }

    if (calibration.is_object() && calibration.value("samples", 0) > 0) {
        std::ostringstream oss;
        oss << "world_prediction|calibration: alignment="
            << std::fixed
            << std::setprecision(2)
            << clampUnitValue(calibration.value("lastAlignmentScore", calibration.value("avgAlignmentScore", 0.0)))
            << " | matched=" << calibration.value("matched", 0) << "/" << calibration.value("samples", 0);
        const auto reason = truncateText(trimCopy(calibration.value("lastReason", std::string())), options.maxSummaryChars / 2);
        if (!reason.empty()) {
            oss << " | reason=" << reason;
        }
        lines.push_back(truncateText(oss.str(), options.maxSummaryChars));
    }

    if (recentEvidence.is_array() && !recentEvidence.empty()) {
        std::size_t start = recentEvidence.size() > options.maxRecentEvidence ? recentEvidence.size() - options.maxRecentEvidence : 0;
        for (std::size_t index = start; index < recentEvidence.size(); ++index) {
            if (!recentEvidence[index].is_object()) {
                continue;
            }
            const auto &evidence = recentEvidence[index];
            std::string evidenceText = buildPromptEvidenceText(evidence, options);
            if (evidenceText.empty()) {
                continue;
            }
            std::string modality = trimCopy(evidence.value("modality", std::string("evidence")));
            modality = modality.empty() ? std::string("evidence") : lowerCopy(modality);
            lines.push_back("world_recent|" + modality + ": " + evidenceText);
        }
    }

    if (lines.empty()) {
        return std::string();
    }

    std::ostringstream oss;
    for (std::size_t index = 0; index < lines.size(); ++index) {
        if (index > 0) {
            oss << '\n';
        }
        oss << lines[index];
    }
    return oss.str();
}

inline std::string mergePromptContext(const std::string &graphContext, const json &sessionState, PromptContextOptions options = {}) {
    const std::string promptContext = buildPromptContext(sessionState, options);
    if (promptContext.empty()) {
        return graphContext;
    }
    if (graphContext.empty()) {
        return promptContext;
    }
    return graphContext + "\n" + promptContext;
}

inline std::string normalizeLlamacppWorldPromptLine(const std::string &line) {
    std::string normalized = trimCopy(line);
    while (normalized.size() >= 2 && ((normalized[0] == '-' || normalized[0] == '*') && normalized[1] == ' ')) {
        normalized = trimCopy(normalized.substr(2));
    }
    if (lowerCopy(normalized) == "supporting evidence:") {
        return std::string();
    }
    return normalized;
}

inline bool isLlamacppWorldPromptLine(const std::string &line) {
    const std::string lowered = lowerCopy(normalizeLlamacppWorldPromptLine(line));
    if (lowered.empty()) {
        return false;
    }
    return lowered.rfind("world_", 0) == 0 ||
           lowered.rfind("gnn_stage2|", 0) == 0 ||
           lowered.find("capture path:") != std::string::npos ||
           lowered.find("micro-mipi") != std::string::npos ||
           lowered.find("v-jepa2") != std::string::npos ||
           lowered.find("vjepa2") != std::string::npos;
}

inline int rankLlamacppWorldPromptLine(const std::string &line) {
    const std::string lowered = lowerCopy(normalizeLlamacppWorldPromptLine(line));
    if (lowered.rfind("world_plan|goal:", 0) == 0) {
        return 0;
    }
    if (lowered.rfind("world_agenda|next_step:", 0) == 0) {
        return 1;
    }
    if (lowered.rfind("world_scene|summary:", 0) == 0) {
        return 2;
    }
    if (lowered.rfind("world_goals|active:", 0) == 0) {
        return 3;
    }
    if (lowered.rfind("world_prediction|expected:", 0) == 0) {
        return 4;
    }
    if (lowered.find("capture path:") != std::string::npos) {
        return 5;
    }
    if (lowered.rfind("world_recent|vision:", 0) == 0 || lowered.rfind("world_recent|video:", 0) == 0) {
        return 6;
    }
    if (lowered.rfind("world_plan|action:", 0) == 0) {
        return 7;
    }
    if (lowered.rfind("world_plan|action_guard:", 0) == 0) {
        return 8;
    }
    if (lowered.rfind("world_plan|simulate:", 0) == 0) {
        return 9;
    }
    if (lowered.rfind("world_prediction|observed:", 0) == 0) {
        return 10;
    }
    if (lowered.rfind("world_recent|", 0) == 0) {
        return 11;
    }
    if (lowered.rfind("world_agenda|question:", 0) == 0) {
        return 12;
    }
    if (lowered.rfind("world_agenda|hypothesis:", 0) == 0) {
        return 13;
    }
    if (lowered.rfind("world_prediction|calibration:", 0) == 0) {
        return 14;
    }
    if (lowered.rfind("world_reflection|needed:", 0) == 0) {
        return 15;
    }
    if (lowered.rfind("world_plan|evidence:", 0) == 0) {
        return 16;
    }
    if (lowered.rfind("world_plan|outline:", 0) == 0) {
        return 17;
    }
    if (lowered.rfind("world_plan|critic:", 0) == 0) {
        return 18;
    }
    if (lowered.rfind("world_relations|active:", 0) == 0) {
        return 19;
    }
    if (lowered.rfind("world_objects|labels:", 0) == 0) {
        return 20;
    }
    if (lowered.rfind("world_episode|summary:", 0) == 0) {
        return 21;
    }
    if (lowered.rfind("world_agenda|conflict:", 0) == 0) {
        return 22;
    }
    if (lowered.rfind("world_plan|rover_goal:", 0) == 0 ||
        lowered.rfind("world_plan|route:", 0) == 0 ||
        lowered.rfind("world_plan|locomotion:", 0) == 0 ||
        lowered.rfind("world_plan|revision_budget:", 0) == 0) {
        return 23;
    }
    if (lowered.rfind("gnn_stage2|", 0) == 0) {
        return 24;
    }
    return 25;
}

inline std::string buildLlamacppWorldPromptShell(const std::string &graphContext,
                                                 std::size_t maxLines = 10,
                                                 std::size_t maxCharsPerLine = 160) {
    if (graphContext.empty()) {
        return std::string();
    }

    struct RankedLine {
        int rank;
        std::size_t order;
        std::string text;
    };

    std::vector<RankedLine> ranked;
    std::unordered_set<std::string> seen;
    std::istringstream input(graphContext);
    std::string rawLine;
    std::size_t order = 0;
    while (std::getline(input, rawLine)) {
        const std::string normalized = normalizeLlamacppWorldPromptLine(rawLine);
        if (!isLlamacppWorldPromptLine(normalized)) {
            continue;
        }
        const std::string lowered = lowerCopy(normalized);
        if (!seen.insert(lowered).second) {
            continue;
        }
        ranked.push_back(RankedLine{rankLlamacppWorldPromptLine(normalized), order++, truncateText(normalized, maxCharsPerLine)});
    }

    if (ranked.empty()) {
        return std::string();
    }

    std::stable_sort(ranked.begin(), ranked.end(), [](const RankedLine &left, const RankedLine &right) {
        if (left.rank != right.rank) {
            return left.rank < right.rank;
        }
        return left.order < right.order;
    });
    if (ranked.size() > maxLines) {
        ranked.resize(maxLines);
    }

    std::ostringstream shell;
    shell << "World model context for llama.cpp. Treat each bullet as grounded evidence or plan guidance. Favor scene facts, direct capture-path hints, recent multimodal evidence, the current next-step goal, and any action guards before generating the answer.\n";
    for (const auto &item : ranked) {
        shell << "- " << item.text << '\n';
    }
    return trimCopy(shell.str());
}

inline json buildCognitiveState(const json &sessionState, CognitiveStateOptions options = {}) {
    json sceneState = (sessionState.contains("sceneState") && sessionState["sceneState"].is_object()) ? sessionState["sceneState"] : json::object();
    json episode = (sessionState.contains("episode") && sessionState["episode"].is_object()) ? sessionState["episode"] : json::object();
    json recentEvidence = (sessionState.contains("recentEvidence") && sessionState["recentEvidence"].is_array()) ? sessionState["recentEvidence"] : json::array();
    json prediction = (sessionState.contains("prediction") && sessionState["prediction"].is_object()) ? sessionState["prediction"] : json::object();

    auto objectLabels = collectObjectLabels(sceneState, options.maxAttentionItems);
    const auto sceneTags = collectSceneTags(sceneState, options.maxAttentionItems);
    auto workingMemory = collectWorkingMemoryLines(recentEvidence, options.maxWorkingMemory, options.maxChars);
    const auto predictedEntities = collectJsonStrings(prediction.value("entities", json::array()), options.maxAttentionItems);
    const json expectedState = (prediction.contains("expectedNextState") && prediction["expectedNextState"].is_object())
                                   ? prediction["expectedNextState"]
                                   : json::object();
    const json observedState = (prediction.contains("observedNextState") && prediction["observedNextState"].is_object())
                                   ? prediction["observedNextState"]
                                   : json::object();
    const json calibration = (prediction.contains("calibration") && prediction["calibration"].is_object())
                                 ? prediction["calibration"]
                                 : json::object();

    const std::string sceneSummary = truncateText(trimCopy(sceneState.value("summary", std::string())), options.maxChars);
    const std::string episodeSummary = truncateText(trimCopy(episode.value("summary", std::string())), options.maxChars);
    const std::string expectedSummary = truncateText(trimCopy(expectedState.value("summary", std::string())), options.maxChars);
    const std::string observedSummary = truncateText(trimCopy(observedState.value("summary", std::string())), options.maxChars);

    auto appendMemory = [&](const std::string &item) {
        if (item.empty()) {
            return;
        }
        const auto clipped = truncateText(item, options.maxChars);
        if (std::find(workingMemory.begin(), workingMemory.end(), clipped) == workingMemory.end()) {
            workingMemory.push_back(clipped);
        }
        while (workingMemory.size() > options.maxWorkingMemory) {
            workingMemory.erase(workingMemory.begin());
        }
    };
    if (!expectedSummary.empty()) {
        appendMemory("expected next state: " + expectedSummary);
    }
    if (!observedSummary.empty()) {
        appendMemory("observed next state: " + observedSummary);
    }

    if (workingMemory.empty()) {
        if (!episodeSummary.empty()) {
            workingMemory.push_back("episode: " + episodeSummary);
        } else if (!sceneSummary.empty()) {
            workingMemory.push_back("scene: " + sceneSummary);
        }
    }

    bool sawVision = false;
    bool sawSpeech = false;
    json modalities = json::array();
    std::unordered_set<std::string> modalitySeen;
    if (recentEvidence.is_array()) {
        for (const auto &evidence : recentEvidence) {
            if (!evidence.is_object()) {
                continue;
            }
            std::string modality = lowerCopy(trimCopy(evidence.value("modality", std::string())));
            if (modality.empty() || !modalitySeen.insert(modality).second) {
                continue;
            }
            modalities.push_back(modality);
            if (modality == "vision" || modality == "image") {
                sawVision = true;
            }
            if (modality == "speech" || modality == "audio") {
                sawSpeech = true;
            }
        }
    }

    std::vector<std::string> goals;
    auto appendGoal = [&](const std::string &goal) {
        if (goal.empty()) {
            return;
        }
        if (std::find(goals.begin(), goals.end(), goal) == goals.end()) {
            goals.push_back(goal);
        }
    };
    for (const auto &goal : collectJsonStrings(prediction.value("goals", json::array()), options.maxGoals)) {
        appendGoal(goal);
    }
    if (sawVision && sawSpeech) {
        appendGoal("align multimodal evidence");
    }
    if (!objectLabels.empty()) {
        appendGoal("track object state transitions");
    }
    if (recentEvidence.is_array() && recentEvidence.size() >= 2) {
        appendGoal("predict next scene change");
    }
    if (!episodeSummary.empty()) {
        appendGoal("consolidate episodic memory");
    }
    if (goals.empty() && !sceneSummary.empty()) {
        appendGoal("stabilize scene understanding");
    }
    if (calibration.is_object() && calibration.value("samples", 0) > 0 &&
        clampUnitValue(calibration.value("lastAlignmentScore", calibration.value("avgAlignmentScore", 1.0))) < 0.45) {
        appendGoal("calibrate the next-state model");
    }
    if (goals.size() > options.maxGoals) {
        goals.resize(options.maxGoals);
    }

    if (objectLabels.empty() && !predictedEntities.empty()) {
        objectLabels = predictedEntities;
    }
    std::vector<std::string> attentionFocus = !objectLabels.empty() ? objectLabels : sceneTags;
    if (attentionFocus.empty() && !sceneSummary.empty()) {
        attentionFocus = extractKeywords(sceneSummary, options.maxAttentionItems);
    }
    if (attentionFocus.size() > options.maxAttentionItems) {
        attentionFocus.resize(options.maxAttentionItems);
    }

    double uncertainty = 0.12;
    if (!(sawVision || sawSpeech)) {
        uncertainty += 0.35;
    }
    if (sceneSummary.empty() && episodeSummary.empty()) {
        uncertainty += 0.18;
    }
    uncertainty += 0.35 * crossModalConflictScore(recentEvidence);
    if (calibration.is_object() && calibration.value("samples", 0) > 0) {
        const double lastAlignment = clampUnitValue(calibration.value("lastAlignmentScore", calibration.value("avgAlignmentScore", 0.5)));
        uncertainty += 0.25 * (1.0 - lastAlignment);
        if (calibration.value("mismatched", 0) > calibration.value("matched", 0)) {
            uncertainty += 0.08;
        }
    }
    uncertainty = std::max(0.05, std::min(0.95, uncertainty));

    json activeGoals = json::array();
    for (const auto &goal : goals) {
        activeGoals.push_back(goal);
    }
    json focusArray = json::array();
    for (const auto &item : attentionFocus) {
        focusArray.push_back(item);
    }
    json wmArray = json::array();
    for (const auto &item : workingMemory) {
        wmArray.push_back(item);
    }

    json prefrontal = json::array();
    for (const auto &goal : goals) {
        prefrontal.push_back(goal);
        if (prefrontal.size() >= options.maxRegionItems) {
            break;
        }
    }
    if (prefrontal.size() < options.maxRegionItems && !workingMemory.empty()) {
        prefrontal.push_back(workingMemory.front());
    }

    json hippocampus = json::array();
    if (!episodeSummary.empty()) {
        hippocampus.push_back(episodeSummary);
    }
    for (const auto &item : workingMemory) {
        hippocampus.push_back(item);
        if (hippocampus.size() >= options.maxRegionItems) {
            break;
        }
    }

    json thalamus = json::array();
    for (const auto &item : attentionFocus) {
        thalamus.push_back(item);
        if (thalamus.size() >= options.maxRegionItems) {
            break;
        }
    }
    if (thalamus.empty()) {
        for (const auto &modality : modalities) {
            thalamus.push_back(modality);
            if (thalamus.size() >= options.maxRegionItems) {
                break;
            }
        }
    }

    json basalGanglia = json::array();
    if (sawVision || sawSpeech) {
        basalGanglia.push_back("observe");
    }
    if (std::find(goals.begin(), goals.end(), "predict next scene change") != goals.end()) {
        basalGanglia.push_back("predict");
    }
    if (std::find(goals.begin(), goals.end(), "align multimodal evidence") != goals.end()) {
        basalGanglia.push_back("verify");
    }
    basalGanglia.push_back("consolidate");

    json cortex = json::array();
    if (!sceneSummary.empty()) {
        cortex.push_back(sceneSummary);
    }
    for (const auto &label : objectLabels) {
        cortex.push_back(label);
        if (cortex.size() >= options.maxRegionItems) {
            break;
        }
    }

    const auto visualObservations = collectRecentModalitySummaries(recentEvidence, {"vision", "image"}, options.maxRegionItems, options.maxChars);
    const auto auditoryObservations = collectRecentModalitySummaries(recentEvidence, {"speech", "audio"}, options.maxRegionItems, options.maxChars);

    std::string languageSeed = joinStrings(auditoryObservations, " ");
    if (languageSeed.empty()) {
        languageSeed = !episodeSummary.empty() ? episodeSummary : sceneSummary;
    }
    std::vector<std::string> semanticTokens = extractKeywords(languageSeed, options.maxRegionItems);

    std::vector<std::string> spatialMap;
    {
        std::unordered_set<std::string> seen;
        for (const auto &label : objectLabels) {
            appendUniqueLimited(spatialMap, seen, label, options.maxRegionItems, options.maxChars);
        }
        for (const auto &tag : sceneTags) {
            appendUniqueLimited(spatialMap, seen, tag, options.maxRegionItems, options.maxChars);
            if (spatialMap.size() >= options.maxRegionItems) {
                break;
            }
        }
    }

    std::vector<std::string> multimodalBindings;
    {
        std::unordered_set<std::string> seen;
        if (!visualObservations.empty() && !auditoryObservations.empty()) {
            appendUniqueLimited(multimodalBindings,
                                seen,
                                "bind visual and auditory evidence around " + (!objectLabels.empty() ? joinStrings(objectLabels, ", ", 2) : std::string("the active scene")),
                                options.maxRegionItems,
                                options.maxChars);
        }
        if (!sceneSummary.empty()) {
            appendUniqueLimited(multimodalBindings, seen, sceneSummary, options.maxRegionItems, options.maxChars);
        }
        if (!episodeSummary.empty()) {
            appendUniqueLimited(multimodalBindings, seen, episodeSummary, options.maxRegionItems, options.maxChars);
        }
    }

    const auto motorIntentions = deriveActionPrograms(sceneSummary, episodeSummary, objectLabels, options.maxRegionItems, options.maxChars);
    const auto bodySchema = deriveBodySchema(objectLabels, sceneTags, sceneSummary, options.maxRegionItems, options.maxChars);

    std::vector<std::string> episodeTrace;
    {
        std::unordered_set<std::string> seen;
        if (!episodeSummary.empty()) {
            appendUniqueLimited(episodeTrace, seen, episodeSummary, options.maxRegionItems, options.maxChars);
        }
        for (const auto &item : workingMemory) {
            appendUniqueLimited(episodeTrace, seen, item, options.maxRegionItems, options.maxChars);
            if (episodeTrace.size() >= options.maxRegionItems) {
                break;
            }
        }
    }

    const double conflictScore = crossModalConflictScore(recentEvidence);

    json corticalSystems{{"visualCortex", json{{"observations", toJsonArray(visualObservations)},
                                                {"objects", toJsonArray(objectLabels)},
                                                {"spatialMap", toJsonArray(spatialMap)}}},
                         {"auditoryCortex", json{{"observations", toJsonArray(auditoryObservations)},
                                                  {"commands", toJsonArray(auditoryObservations)}}},
                         {"languageNetwork", json{{"utterances", toJsonArray(auditoryObservations)},
                                                   {"semanticTokens", toJsonArray(semanticTokens)}}},
                         {"multimodalAssociation", json{{"bindings", toJsonArray(multimodalBindings)},
                                                         {"fusionSummary", !episodeSummary.empty() ? episodeSummary : sceneSummary},
                                                         {"conflictScore", conflictScore}}},
                         {"somatosensoryCortex", json{{"bodySchema", toJsonArray(bodySchema)},
                                                       {"interactionTargets", toJsonArray(objectLabels)}}},
                         {"motorPlanner", json{{"intentions", toJsonArray(motorIntentions)},
                                                {"predictedActions", toJsonArray(motorIntentions)}}},
                         {"episodicMemory", json{{"episodeTrace", toJsonArray(episodeTrace)},
                                                  {"recallTarget", !episodeSummary.empty() ? episodeSummary : sceneSummary}}}};

    json cognitive{{"ok", true},
                   {"profile", "functional"},
                   {"purpose", "application"},
                   {"workingMemory", wmArray},
                   {"attentionFocus", focusArray},
                   {"activeGoals", activeGoals},
                   {"uncertainty", uncertainty},
                   {"modalities", modalities},
                   {"sceneSummary", sceneSummary},
                   {"episodeSummary", episodeSummary},
                   {"corticalSystems", corticalSystems},
                   {"brainRegions", json{{"prefrontal", prefrontal},
                                           {"hippocampus", hippocampus},
                                           {"thalamus", thalamus},
                                           {"basalGanglia", basalGanglia},
                                           {"cortex", cortex}}}};

    if (prediction.is_object() && !prediction.empty()) {
        cognitive["prediction"] = prediction;
    }

    const std::size_t agendaItems = std::max<std::size_t>(2, std::min(options.maxGoals, options.maxAttentionItems));
    cognitive["reasoningAgenda"] = buildReasoningAgendaFromCognitiveState(cognitive, agendaItems, options.maxChars);
    return cognitive;
}

inline json buildReasoningAgendaFromCognitiveState(const json &cognitiveState, std::size_t maxItems, std::size_t maxChars) {
    maxItems = std::max<std::size_t>(1, maxItems);
    maxChars = std::max<std::size_t>(32, maxChars);

    const auto focus = collectJsonStrings(cognitiveState.value("attentionFocus", json::array()), maxItems);
    const auto goals = collectJsonStrings(cognitiveState.value("activeGoals", json::array()), maxItems);
    const auto workingMemory = collectJsonStrings(cognitiveState.value("workingMemory", json::array()), maxItems);
    const auto modalities = collectJsonStrings(cognitiveState.value("modalities", json::array()), maxItems);
    const json prediction = (cognitiveState.contains("prediction") && cognitiveState["prediction"].is_object()) ? cognitiveState["prediction"] : json::object();
    const json expectedState = (prediction.contains("expectedNextState") && prediction["expectedNextState"].is_object())
                                   ? prediction["expectedNextState"]
                                   : json::object();
    const json observedState = (prediction.contains("observedNextState") && prediction["observedNextState"].is_object())
                                   ? prediction["observedNextState"]
                                   : json::object();
    const json calibration = (prediction.contains("calibration") && prediction["calibration"].is_object())
                                 ? prediction["calibration"]
                                 : json::object();
    const std::string sceneSummary = truncateText(trimCopy(cognitiveState.value("sceneSummary", std::string())), maxChars);
    const std::string episodeSummary = truncateText(trimCopy(cognitiveState.value("episodeSummary", std::string())), maxChars);
    const std::string selectedAction = truncateText(trimCopy(cognitiveState.value("selectedAction", std::string())), maxChars);
    const double uncertainty = std::max(0.0, std::min(1.0, cognitiveState.value("uncertainty", 0.5)));
    const std::string expectedSummary = truncateText(trimCopy(expectedState.value("summary", std::string())), maxChars);
    const std::string observedSummary = truncateText(trimCopy(observedState.value("summary", std::string())), maxChars);
    const double lastAlignmentScore = clampUnitValue(calibration.value("lastAlignmentScore", calibration.value("avgAlignmentScore", 0.5)));
    const bool predictionMismatch = !expectedSummary.empty() && !observedSummary.empty() && lastAlignmentScore < 0.45;

    std::vector<std::string> hypotheses;
    std::vector<std::string> contradictions;
    std::vector<std::string> openQuestions;
    std::vector<std::string> evidenceChecklist;
    std::unordered_set<std::string> hypothesisSeen;
    std::unordered_set<std::string> contradictionSeen;
    std::unordered_set<std::string> questionSeen;
    std::unordered_set<std::string> evidenceSeen;

    if (!focus.empty() && !goals.empty()) {
        appendUniqueLimited(hypotheses,
                            hypothesisSeen,
                            "the next useful step is to " + goals.front() + " around " + focus.front(),
                            maxItems,
                            maxChars);
    }
    if (!episodeSummary.empty()) {
        appendUniqueLimited(hypotheses,
                            hypothesisSeen,
                            "the current episode is still governed by " + episodeSummary,
                            maxItems,
                            maxChars);
    }
    if (modalities.size() >= 2 && !sceneSummary.empty()) {
        appendUniqueLimited(hypotheses,
                            hypothesisSeen,
                            "multimodal evidence should converge on " + sceneSummary,
                            maxItems,
                            maxChars);
    }
    if (!workingMemory.empty()) {
        appendUniqueLimited(hypotheses,
                            hypothesisSeen,
                            "recent evidence suggests " + workingMemory.front(),
                            maxItems,
                            maxChars);
    }
    for (const auto &item : collectJsonStrings(prediction.value("hypotheses", json::array()), maxItems)) {
        appendUniqueLimited(hypotheses, hypothesisSeen, item, maxItems, maxChars);
    }
    if (!expectedSummary.empty()) {
        appendUniqueLimited(hypotheses,
                            hypothesisSeen,
                            "expected next state: " + expectedSummary,
                            maxItems,
                            maxChars);
    }

    if (uncertainty >= 0.55) {
        appendUniqueLimited(contradictions,
                            contradictionSeen,
                            "scene understanding remains unstable and should be verified before action",
                            maxItems,
                            maxChars);
    }
    if (modalities.size() == 1) {
        appendUniqueLimited(contradictions,
                            contradictionSeen,
                            "only " + modalities.front() + " evidence is available; cross-modal grounding is incomplete",
                            maxItems,
                            maxChars);
    }
    if (focus.empty()) {
        appendUniqueLimited(contradictions,
                            contradictionSeen,
                            "no stable attention target is set for the next reasoning step",
                            maxItems,
                            maxChars);
    }
    if (goals.empty()) {
        appendUniqueLimited(contradictions,
                            contradictionSeen,
                            "no active goal is prioritized, so the next action may drift",
                            maxItems,
                            maxChars);
    }
    for (const auto &item : collectJsonStrings(prediction.value("contradictions", json::array()), maxItems)) {
        appendUniqueLimited(contradictions, contradictionSeen, item, maxItems, maxChars);
    }
    if (predictionMismatch) {
        appendUniqueLimited(contradictions,
                            contradictionSeen,
                            "observed next state diverged from expectation: " + observedSummary,
                            maxItems,
                            maxChars);
    }

    if (uncertainty >= 0.4) {
        appendUniqueLimited(openQuestions,
                            questionSeen,
                            "what evidence would reduce uncertainty fastest?",
                            maxItems,
                            maxChars);
    }
    if (!focus.empty()) {
        appendUniqueLimited(openQuestions,
                            questionSeen,
                            "what is the next state change around " + focus.front() + "?",
                            maxItems,
                            maxChars);
    }
    if (modalities.size() < 2) {
        appendUniqueLimited(openQuestions,
                            questionSeen,
                            "which missing modality should be observed next?",
                            maxItems,
                            maxChars);
    }
    if (!goals.empty()) {
        appendUniqueLimited(openQuestions,
                            questionSeen,
                            "what constraint must be satisfied before executing " + goals.front() + "?",
                            maxItems,
                            maxChars);
    }
    if (!expectedSummary.empty() && observedSummary.empty()) {
        appendUniqueLimited(openQuestions,
                            questionSeen,
                            "which observation will confirm or falsify the expected next state?",
                            maxItems,
                            maxChars);
    }
    if (predictionMismatch) {
        appendUniqueLimited(openQuestions,
                            questionSeen,
                            "what caused the mismatch between expected and observed next state?",
                            maxItems,
                            maxChars);
    }

    for (const auto &item : workingMemory) {
        appendUniqueLimited(evidenceChecklist, evidenceSeen, item, maxItems, maxChars);
    }
    for (const auto &item : focus) {
        appendUniqueLimited(evidenceChecklist,
                            evidenceSeen,
                            "track focus: " + item,
                            maxItems,
                            maxChars);
    }
    if (!expectedSummary.empty()) {
        appendUniqueLimited(evidenceChecklist,
                            evidenceSeen,
                            "verify expected transition: " + expectedSummary,
                            maxItems,
                            maxChars);
    }
    if (calibration.is_object() && calibration.value("samples", 0) > 0) {
        std::ostringstream oss;
        oss << "prediction alignment score: " << std::fixed << std::setprecision(2) << lastAlignmentScore;
        appendUniqueLimited(evidenceChecklist, evidenceSeen, oss.str(), maxItems, maxChars);
    }

    std::string nextStep;
    if (!selectedAction.empty()) {
        nextStep = selectedAction;
    } else if (predictionMismatch) {
        nextStep = "update the prediction model using the latest observed divergence";
    } else if (!contradictions.empty() && uncertainty >= 0.55) {
        nextStep = "pause and verify the scene before committing to an action";
    } else if (!openQuestions.empty()) {
        nextStep = "collect evidence for: " + openQuestions.front();
    } else if (!goals.empty()) {
        nextStep = goals.front();
    } else if (!sceneSummary.empty()) {
        nextStep = "stabilize the current scene model";
    } else {
        nextStep = "gather more scene evidence";
    }
    nextStep = truncateText(nextStep, maxChars);

    const bool shouldReflect = uncertainty >= 0.45 || !contradictions.empty() || predictionMismatch;
    std::string reflectionReason;
    if (predictionMismatch) {
        reflectionReason = "expected and observed next states diverged";
    } else if (uncertainty >= 0.45) {
        reflectionReason = "uncertainty is high";
    } else if (!contradictions.empty()) {
        reflectionReason = contradictions.front();
    }
    reflectionReason = truncateText(reflectionReason, maxChars);
    const double reflectionPriority = std::max(0.0,
                                               std::min(1.0,
                                                        uncertainty +
                                                            0.12 * static_cast<double>(std::min<std::size_t>(contradictions.size(), 3)) +
                                                            (predictionMismatch ? 0.18 * (1.0 - lastAlignmentScore) : 0.0)));

    return json{{"hypotheses", toJsonArray(hypotheses)},
                {"contradictions", toJsonArray(contradictions)},
                {"openQuestions", toJsonArray(openQuestions)},
                {"evidenceChecklist", toJsonArray(evidenceChecklist)},
                {"nextStep", nextStep},
                {"shouldReflect", shouldReflect},
                {"reflectionReason", reflectionReason},
                {"reflectionPriority", reflectionPriority},
                {"uncertainty", uncertainty}};
}

inline json buildReasoningAgenda(const json &sessionState, std::size_t maxItems = 4, std::size_t maxChars = 120) {
    if (sessionState.is_object() && sessionState.contains("reasoningAgenda") && sessionState["reasoningAgenda"].is_object()) {
        return sessionState["reasoningAgenda"];
    }
    if (sessionState.is_object() && sessionState.contains("workingMemory") && sessionState.contains("attentionFocus")) {
        return buildReasoningAgendaFromCognitiveState(sessionState, maxItems, maxChars);
    }
    CognitiveStateOptions options;
    options.maxWorkingMemory = std::max<std::size_t>(2, maxItems);
    options.maxAttentionItems = std::max<std::size_t>(2, maxItems);
    options.maxGoals = std::max<std::size_t>(2, maxItems);
    options.maxRegionItems = std::max<std::size_t>(2, maxItems);
    options.maxChars = maxChars;
    return buildReasoningAgendaFromCognitiveState(buildCognitiveState(sessionState, options), maxItems, maxChars);
}

inline std::string buildReasoningAgendaPromptContext(const json &sessionState, std::size_t maxCharsPerLine = 160) {
    const auto agenda = buildReasoningAgenda(sessionState, 4, maxCharsPerLine);
    if (!agenda.is_object()) {
        return std::string();
    }

    std::vector<std::string> lines;
    auto collectTop = [&](const char *key, std::size_t limit) {
        std::vector<std::string> out;
        if (!agenda.contains(key) || !agenda[key].is_array()) {
            return out;
        }
        for (const auto &entry : agenda[key]) {
            if (!entry.is_string()) {
                continue;
            }
            out.push_back(entry.get<std::string>());
            if (out.size() >= limit) {
                break;
            }
        }
        return out;
    };

    const auto hypotheses = collectTop("hypotheses", 2);
    if (!hypotheses.empty()) {
        lines.push_back("world_agenda|hypothesis: " + truncateText(joinStrings(hypotheses, " || "), maxCharsPerLine));
    }
    const auto questions = collectTop("openQuestions", 2);
    if (!questions.empty()) {
        lines.push_back("world_agenda|question: " + truncateText(joinStrings(questions, " || "), maxCharsPerLine));
    }
    const auto contradictions = collectTop("contradictions", 2);
    if (!contradictions.empty()) {
        lines.push_back("world_agenda|conflict: " + truncateText(joinStrings(contradictions, " || "), maxCharsPerLine));
    }
    if (agenda.contains("nextStep") && agenda["nextStep"].is_string()) {
        lines.push_back("world_agenda|next_step: " + truncateText(agenda["nextStep"].get<std::string>(), maxCharsPerLine));
    }
    if (agenda.value("shouldReflect", false)) {
        std::ostringstream oss;
        oss << "world_reflection|needed: yes";
        const std::string reason = agenda.value("reflectionReason", std::string());
        if (!reason.empty()) {
            oss << " | reason=" << truncateText(reason, maxCharsPerLine / 2);
        }
        oss << " | priority=" << std::fixed << std::setprecision(2) << agenda.value("reflectionPriority", 0.0);
        lines.push_back(truncateText(oss.str(), maxCharsPerLine));
    }

    return joinStrings(lines, "\n");
}

inline json buildResponsePlan(const json &sessionState, std::size_t maxItems = 4, std::size_t maxChars = 120) {
    if (sessionState.is_object() && sessionState.contains("responsePlan") && sessionState["responsePlan"].is_object()) {
        return sessionState["responsePlan"];
    }
    if (sessionState.is_object() && sessionState.contains("goalFrame") && sessionState.contains("answerOutline")) {
        return sessionState;
    }

    json cognitiveState;
    if (sessionState.is_object() && sessionState.contains("workingMemory") && sessionState.contains("attentionFocus")) {
        cognitiveState = sessionState;
    } else {
        CognitiveStateOptions options;
        options.maxWorkingMemory = std::max<std::size_t>(2, maxItems);
        options.maxAttentionItems = std::max<std::size_t>(2, maxItems);
        options.maxGoals = std::max<std::size_t>(2, maxItems);
        options.maxRegionItems = std::max<std::size_t>(2, maxItems);
        options.maxChars = maxChars;
        cognitiveState = buildCognitiveState(sessionState, options);
    }

    const auto agenda = buildReasoningAgenda(cognitiveState, maxItems, maxChars);
    const auto focus = collectJsonStrings(cognitiveState.value("attentionFocus", json::array()), maxItems);
    const auto goals = collectJsonStrings(cognitiveState.value("activeGoals", json::array()), maxItems);
    const auto workingMemory = collectJsonStrings(cognitiveState.value("workingMemory", json::array()), maxItems);
    const auto hypotheses = collectJsonStrings(agenda.value("hypotheses", json::array()), maxItems);
    const auto contradictions = collectJsonStrings(agenda.value("contradictions", json::array()), maxItems);
    const auto openQuestions = collectJsonStrings(agenda.value("openQuestions", json::array()), maxItems);
    const auto evidenceChecklist = collectJsonStrings(agenda.value("evidenceChecklist", json::array()), maxItems);

    std::vector<std::string> evidenceFocus;
    std::unordered_set<std::string> evidenceSeen;
    for (const auto &item : evidenceChecklist) {
        appendUniqueLimited(evidenceFocus, evidenceSeen, item, maxItems, maxChars);
    }
    for (const auto &item : focus) {
        appendUniqueLimited(evidenceFocus, evidenceSeen, "inspect focus: " + item, maxItems, maxChars);
    }
    for (const auto &item : workingMemory) {
        appendUniqueLimited(evidenceFocus, evidenceSeen, item, maxItems, maxChars);
    }

    std::vector<std::string> answerOutline;
    std::unordered_set<std::string> outlineSeen;
    if (!focus.empty()) {
        appendUniqueLimited(answerOutline, outlineSeen, "start from the observed focus: " + focus.front(), maxItems, maxChars);
    }
    if (!hypotheses.empty()) {
        appendUniqueLimited(answerOutline, outlineSeen, "state the strongest grounded hypothesis: " + hypotheses.front(), maxItems, maxChars);
    }
    const double uncertainty = agenda.value("uncertainty", cognitiveState.value("uncertainty", 0.0));
    if (uncertainty >= 0.35) {
        appendUniqueLimited(answerOutline, outlineSeen, "mark uncertainty briefly instead of overstating certainty", maxItems, maxChars);
    }
    if (!goals.empty()) {
        appendUniqueLimited(answerOutline, outlineSeen, "connect the answer to goal: " + goals.front(), maxItems, maxChars);
    }
    if (agenda.contains("nextStep") && agenda["nextStep"].is_string()) {
        appendUniqueLimited(answerOutline, outlineSeen, "end with next step: " + agenda["nextStep"].get<std::string>(), maxItems, maxChars);
    }

    std::vector<std::string> critiqueChecklist;
    std::unordered_set<std::string> critiqueSeen;
    if (!contradictions.empty()) {
        appendUniqueLimited(critiqueChecklist, critiqueSeen, "resolve contradiction: " + contradictions.front(), maxItems, maxChars);
    }
    if (!openQuestions.empty()) {
        appendUniqueLimited(critiqueChecklist, critiqueSeen, "do not skip open question: " + openQuestions.front(), maxItems, maxChars);
    }
    if (!evidenceFocus.empty()) {
        appendUniqueLimited(critiqueChecklist, critiqueSeen, "anchor claims to evidence focus", maxItems, maxChars);
    }
    if (uncertainty >= 0.35) {
        appendUniqueLimited(critiqueChecklist, critiqueSeen, "avoid unsupported certainty", maxItems, maxChars);
    }

    std::vector<std::string> simulationTargets;
    std::unordered_set<std::string> simulationSeen;
    if (!openQuestions.empty()) {
        appendUniqueLimited(simulationTargets, simulationSeen, "simulate evidence path for: " + openQuestions.front(), maxItems, maxChars);
    }
    if (!focus.empty()) {
        appendUniqueLimited(simulationTargets, simulationSeen, "simulate next state around " + focus.front(), maxItems, maxChars);
    }
    if (!goals.empty()) {
        appendUniqueLimited(simulationTargets, simulationSeen, "simulate constraint before " + goals.front(), maxItems, maxChars);
    }

    const json mobilityPlan = buildGroundMobilityPlan(sessionState, agenda, focus, goals, maxItems, maxChars);
    if (mobilityPlan.is_object() && !mobilityPlan.empty()) {
        appendUniqueLimited(answerOutline,
                            outlineSeen,
                            "separate strategic route planning from local rover stabilization",
                            maxItems,
                            maxChars);
        appendUniqueLimited(critiqueChecklist,
                            critiqueSeen,
                            "keep the rover ground-only and avoid assuming aerial motion",
                            maxItems,
                            maxChars);
        if (mobilityPlan.contains("bigBrain") && mobilityPlan["bigBrain"].is_object()) {
            for (const auto &item : collectJsonStrings(mobilityPlan["bigBrain"].value("evidenceNeeds", json::array()), maxItems)) {
                appendUniqueLimited(evidenceFocus, evidenceSeen, item, maxItems, maxChars);
            }
            for (const auto &item : collectJsonStrings(mobilityPlan["bigBrain"].value("simulationTargets", json::array()), maxItems)) {
                appendUniqueLimited(simulationTargets, simulationSeen, item, maxItems, maxChars);
            }
        }
    }

    std::string goalFrame;
    if (!goals.empty()) {
        goalFrame = goals.front();
    } else if (agenda.contains("nextStep") && agenda["nextStep"].is_string()) {
        goalFrame = agenda["nextStep"].get<std::string>();
    } else if (!focus.empty()) {
        goalFrame = "stabilize reasoning around " + focus.front();
    } else {
        goalFrame = "deliver the most grounded next answer";
    }
    goalFrame = truncateText(goalFrame, maxChars);

    const double revisionBudget = std::max(agenda.value("reflectionPriority", 0.0), uncertainty);

    return json{{"goalFrame", goalFrame},
                {"evidenceFocus", toJsonArray(evidenceFocus)},
                {"answerOutline", toJsonArray(answerOutline)},
                {"critiqueChecklist", toJsonArray(critiqueChecklist)},
                {"simulationTargets", toJsonArray(simulationTargets)},
                {"mobilityPlan", mobilityPlan},
                {"revisionBudget", std::max(0.0, std::min(1.0, revisionBudget))},
                {"uncertainty", uncertainty}};
}

inline std::string buildResponsePlanPromptContext(const json &sessionState, std::size_t maxCharsPerLine = 160) {
    const auto plan = buildResponsePlan(sessionState, 4, maxCharsPerLine);
    if (!plan.is_object()) {
        return std::string();
    }

    std::vector<std::string> lines;
    auto collectTop = [&](const char *key, std::size_t limit) {
        std::vector<std::string> out;
        if (!plan.contains(key) || !plan[key].is_array()) {
            return out;
        }
        for (const auto &entry : plan[key]) {
            if (!entry.is_string()) {
                continue;
            }
            out.push_back(entry.get<std::string>());
            if (out.size() >= limit) {
                break;
            }
        }
        return out;
    };

    if (plan.contains("goalFrame") && plan["goalFrame"].is_string()) {
        lines.push_back("world_plan|goal: " + truncateText(plan["goalFrame"].get<std::string>(), maxCharsPerLine));
    }
    const auto evidenceFocus = collectTop("evidenceFocus", 2);
    if (!evidenceFocus.empty()) {
        lines.push_back("world_plan|evidence: " + truncateText(joinStrings(evidenceFocus, " || "), maxCharsPerLine));
    }
    const auto answerOutline = collectTop("answerOutline", 3);
    if (!answerOutline.empty()) {
        lines.push_back("world_plan|outline: " + truncateText(joinStrings(answerOutline, " || "), maxCharsPerLine));
    }
    const auto critiqueChecklist = collectTop("critiqueChecklist", 2);
    if (!critiqueChecklist.empty()) {
        lines.push_back("world_plan|critic: " + truncateText(joinStrings(critiqueChecklist, " || "), maxCharsPerLine));
    }
    const auto simulationTargets = collectTop("simulationTargets", 2);
    if (!simulationTargets.empty()) {
        lines.push_back("world_plan|simulate: " + truncateText(joinStrings(simulationTargets, " || "), maxCharsPerLine));
    }
    if (plan.contains("mobilityPlan") && plan["mobilityPlan"].is_object()) {
        const auto &mobilityPlan = plan["mobilityPlan"];
        if (mobilityPlan.contains("strategicGoal") && mobilityPlan["strategicGoal"].is_string()) {
            lines.push_back("world_plan|rover_goal: " + truncateText(mobilityPlan["strategicGoal"].get<std::string>(), maxCharsPerLine));
        }
        if (mobilityPlan.contains("bigBrain") && mobilityPlan["bigBrain"].is_object()) {
            const auto &bigBrain = mobilityPlan["bigBrain"];
            if (bigBrain.contains("routePolicy") && bigBrain["routePolicy"].is_string()) {
                lines.push_back("world_plan|route: " + truncateText(bigBrain["routePolicy"].get<std::string>(), maxCharsPerLine));
            }
        }
        if (mobilityPlan.contains("littleBrain") && mobilityPlan["littleBrain"].is_object()) {
            const auto &littleBrain = mobilityPlan["littleBrain"];
            if (littleBrain.contains("speedPolicy") && littleBrain["speedPolicy"].is_string()) {
                lines.push_back("world_plan|locomotion: " + truncateText(littleBrain["speedPolicy"].get<std::string>(), maxCharsPerLine));
            }
            if (littleBrain.contains("selectedAction") && littleBrain["selectedAction"].is_string()) {
                lines.push_back("world_plan|action: " + truncateText(littleBrain["selectedAction"].get<std::string>(), maxCharsPerLine));
            }
        }
        if (mobilityPlan.contains("actionConstraints") && mobilityPlan["actionConstraints"].is_object()) {
            const auto &constraints = mobilityPlan["actionConstraints"];
            std::vector<std::string> guards;
            if (constraints.value("doNotMove", false)) {
                guards.push_back("hold position until evidence improves");
            }
            if (constraints.contains("minVerifyScore") && constraints["minVerifyScore"].is_number()) {
                std::ostringstream oss;
                oss << "verify>=" << std::fixed << std::setprecision(2) << constraints["minVerifyScore"].get<double>();
                guards.push_back(oss.str());
            }
            if (constraints.contains("maxUncertainty") && constraints["maxUncertainty"].is_number()) {
                std::ostringstream oss;
                oss << "uncertainty<=" << std::fixed << std::setprecision(2) << constraints["maxUncertainty"].get<double>();
                guards.push_back(oss.str());
            }
            if (constraints.contains("cooldownMs") && constraints["cooldownMs"].is_number_integer()) {
                guards.push_back("cooldown=" + std::to_string(constraints["cooldownMs"].get<int>()) + "ms");
            }
            if (!guards.empty()) {
                lines.push_back("world_plan|action_guard: " + truncateText(joinStrings(guards, " || "), maxCharsPerLine));
            }
        }
    }
    if (plan.contains("revisionBudget") && plan["revisionBudget"].is_number()) {
        std::ostringstream oss;
        oss << "world_plan|revision_budget: " << std::fixed << std::setprecision(2) << plan["revisionBudget"].get<double>();
        lines.push_back(truncateText(oss.str(), maxCharsPerLine));
    }

    return joinStrings(lines, "\n");
}

inline json buildFunctionalBrainRuntime(const json &sessionState, BrainProfileOptions options = {}) {
    const auto functional = isFunctionalBrainState(sessionState)
                                ? sessionState
                                : buildCognitiveState(sessionState, toCognitiveStateOptions(options));
    const auto corticalSystems = functional.value("corticalSystems", json::object());

    const auto visual = collectJsonStrings(corticalSystems.value("visualCortex", json::object()).value("observations", json::array()), options.maxRegionItems);
    const auto auditory = collectJsonStrings(corticalSystems.value("auditoryCortex", json::object()).value("observations", json::array()), options.maxRegionItems);
    const auto bindings = collectJsonStrings(corticalSystems.value("multimodalAssociation", json::object()).value("bindings", json::array()), options.maxRegionItems);
    const auto actions = collectJsonStrings(corticalSystems.value("motorPlanner", json::object()).value("predictedActions", json::array()), options.maxRegionItems);
    const auto replay = collectJsonStrings(corticalSystems.value("episodicMemory", json::object()).value("episodeTrace", json::array()), options.maxRegionItems);
    const auto focus = collectJsonStrings(functional.value("attentionFocus", json::array()), options.maxAttentionItems);
    const auto goals = collectJsonStrings(functional.value("activeGoals", json::array()), options.maxGoals);

    json salience = json::array();
    for (std::size_t index = 0; index < focus.size(); ++index) {
        const double score = std::max(0.2, 1.0 - 0.18 * static_cast<double>(index));
        salience.push_back(json{{"target", focus[index]}, {"score", score}});
    }

    std::vector<std::string> rewardDrivers;
    std::unordered_set<std::string> rewardSeen;
    for (const auto &goal : goals) {
        appendUniqueLimited(rewardDrivers, rewardSeen, "reward goal: " + goal, options.maxRegionItems, options.maxChars);
    }
    if (!bindings.empty()) {
        appendUniqueLimited(rewardDrivers, rewardSeen, "preserve multimodal consistency", options.maxRegionItems, options.maxChars);
    }
    if (functional.value("uncertainty", 0.5) > 0.45) {
        appendUniqueLimited(rewardDrivers, rewardSeen, "reduce uncertainty before committing", options.maxRegionItems, options.maxChars);
    }

    std::string selectedAction;
    if (!actions.empty()) {
        selectedAction = actions.front();
    } else if (!focus.empty()) {
        selectedAction = "inspect and stabilize " + focus.front();
    } else {
        selectedAction = "gather more scene evidence";
    }

    const std::string consolidationTarget = !replay.empty()
                                                ? replay.front()
                                                : functional.value("episodeSummary", functional.value("sceneSummary", std::string()));

    json executiveLoop = json::array({json{{"phase", "perceive"}, {"regions", json::array({"visualCortex", "auditoryCortex"})}, {"summary", truncateText(joinStrings(visual, " || ") + (auditory.empty() ? std::string() : std::string(" || ") + joinStrings(auditory, " || ")), options.maxChars)}},
                                      json{{"phase", "attend"}, {"regions", json::array({"thalamus", "prefrontal"})}, {"summary", truncateText(joinStrings(focus, ", "), options.maxChars)}},
                                      json{{"phase", "bind"}, {"regions", json::array({"multimodalAssociation", "hippocampus"})}, {"summary", truncateText(joinStrings(bindings, " || "), options.maxChars)}},
                                      json{{"phase", "evaluate"}, {"regions", json::array({"prefrontal", "basalGanglia"})}, {"summary", truncateText(joinStrings(rewardDrivers, " || "), options.maxChars)}},
                                      json{{"phase", "act"}, {"regions", json::array({"motorPlanner", "somatosensoryCortex"})}, {"summary", truncateText(selectedAction, options.maxChars)}},
                                      json{{"phase", "replay"}, {"regions", json::array({"episodicMemory", "hippocampus"})}, {"summary", truncateText(consolidationTarget, options.maxChars)}}});

    return json{{"ok", true},
                {"profile", "functional"},
                {"mode", "online-functional-loop"},
                {"selectedAction", truncateText(selectedAction, options.maxChars)},
                {"consolidationTarget", truncateText(consolidationTarget, options.maxChars)},
                {"functionalRuntime", json{{"perceptionLoop", json{{"visual", toJsonArray(visual)},
                                                                      {"auditory", toJsonArray(auditory)},
                                                                      {"multimodalBindings", toJsonArray(bindings)}}},
                                            {"attentionController", json{{"salience", salience},
                                                                          {"focus", toJsonArray(focus)}}},
                                            {"valueSystem", json{{"rewardDrivers", toJsonArray(rewardDrivers)},
                                                                  {"uncertainty", functional.value("uncertainty", 0.5)}}},
                                            {"executiveController", json{{"loop", executiveLoop},
                                                                           {"activeGoals", toJsonArray(goals)}}},
                                            {"actionBuffer", json{{"queuedActions", toJsonArray(actions)},
                                                                   {"selectedAction", truncateText(selectedAction, options.maxChars)}}},
                                            {"replayLoop", json{{"episodeTrace", toJsonArray(replay)},
                                                                 {"consolidationTarget", truncateText(consolidationTarget, options.maxChars)}}}}}};
}

inline json buildStructuralBrainCore(const json &sessionState, BrainProfileOptions options = {}) {
    const auto functional = buildCognitiveState(sessionState, toCognitiveStateOptions(options));
    const json corticalSystems = functional.value("corticalSystems", json::object());

    std::vector<std::string> occipital = collectJsonStrings(corticalSystems.value("visualCortex", json::object()).value("observations", json::array()), options.maxRegionItems);
    {
        std::unordered_set<std::string> seen;
        for (const auto &entry : occipital) {
            seen.insert(lowerCopy(entry));
        }
        for (const auto &entry : collectJsonStrings(corticalSystems.value("visualCortex", json::object()).value("objects", json::array()), options.maxRegionItems)) {
            appendUniqueLimited(occipital, seen, entry, options.maxRegionItems, options.maxChars);
        }
    }
    const auto auditory = collectJsonStrings(corticalSystems.value("auditoryCortex", json::object()).value("observations", json::array()), options.maxRegionItems);
    std::vector<std::string> wernicke = collectJsonStrings(corticalSystems.value("languageNetwork", json::object()).value("utterances", json::array()), options.maxRegionItems);
    {
        std::unordered_set<std::string> seen;
        for (const auto &entry : wernicke) {
            seen.insert(lowerCopy(entry));
        }
        for (const auto &entry : collectJsonStrings(corticalSystems.value("languageNetwork", json::object()).value("semanticTokens", json::array()), options.maxRegionItems)) {
            appendUniqueLimited(wernicke, seen, entry, options.maxRegionItems, options.maxChars);
        }
    }
    std::vector<std::string> broca = collectJsonStrings(corticalSystems.value("motorPlanner", json::object()).value("intentions", json::array()), options.maxRegionItems);
    {
        std::unordered_set<std::string> seen;
        for (const auto &entry : broca) {
            seen.insert(lowerCopy(entry));
        }
        for (const auto &entry : collectJsonStrings(functional.value("activeGoals", json::array()), options.maxRegionItems)) {
            appendUniqueLimited(broca, seen, entry, options.maxRegionItems, options.maxChars);
        }
    }
    const auto parietal = collectJsonStrings(corticalSystems.value("visualCortex", json::object()).value("spatialMap", json::array()), options.maxRegionItems);
    const auto somatosensory = collectJsonStrings(corticalSystems.value("somatosensoryCortex", json::object()).value("bodySchema", json::array()), options.maxRegionItems);
    const auto motor = collectJsonStrings(corticalSystems.value("motorPlanner", json::object()).value("predictedActions", json::array()), options.maxRegionItems);
    const auto cerebellumSeed = collectJsonStrings(functional.value("attentionFocus", json::array()), options.maxRegionItems);

    std::vector<std::string> cerebellum;
    {
        std::unordered_set<std::string> seen;
        for (const auto &entry : cerebellumSeed) {
            appendUniqueLimited(cerebellum, seen, "timing loop for " + entry, options.maxRegionItems, options.maxChars);
        }
        if (cerebellum.empty()) {
            appendUniqueLimited(cerebellum, seen, "timing loop for active scene stabilization", options.maxRegionItems, options.maxChars);
        }
    }

    json corticalMap{{"occipitalCortex", toJsonArray(occipital)},
                     {"auditoryCortex", toJsonArray(auditory)},
                     {"wernickeArea", toJsonArray(wernicke)},
                     {"brocaArea", toJsonArray(broca)},
                     {"parietalAssociationCortex", toJsonArray(parietal)},
                     {"somatosensoryCortex", toJsonArray(somatosensory)},
                     {"motorCortex", toJsonArray(motor)},
                     {"prefrontalCortex", functional.value("brainRegions", json::object()).value("prefrontal", json::array())},
                     {"hippocampus", functional.value("brainRegions", json::object()).value("hippocampus", json::array())},
                     {"thalamus", functional.value("brainRegions", json::object()).value("thalamus", json::array())},
                     {"basalGanglia", functional.value("brainRegions", json::object()).value("basalGanglia", json::array())},
                     {"cerebellum", toJsonArray(cerebellum)}};

    json pathways = json::array({json{{"from", "occipitalCortex"}, {"to", "parietalAssociationCortex"}, {"signal", "visual-spatial relay"}},
                                 json{{"from", "auditoryCortex"}, {"to", "wernickeArea"}, {"signal", "auditory-language relay"}},
                                 json{{"from", "wernickeArea"}, {"to", "brocaArea"}, {"signal", "speech planning relay"}},
                                 json{{"from", "thalamus"}, {"to", "prefrontalCortex"}, {"signal", "salience gating"}},
                                 json{{"from", "hippocampus"}, {"to", "prefrontalCortex"}, {"signal", "episodic recall"}},
                                 json{{"from", "basalGanglia"}, {"to", "motorCortex"}, {"signal", "action selection"}},
                                 json{{"from", "motorCortex"}, {"to", "cerebellum"}, {"signal", "timing correction"}}});

    return json{{"ok", true},
                {"profile", "structural"},
                {"purpose", "research"},
                {"workingMemory", functional.value("workingMemory", json::array())},
                {"attentionFocus", functional.value("attentionFocus", json::array())},
                {"activeGoals", functional.value("activeGoals", json::array())},
                {"reasoningAgenda", functional.value("reasoningAgenda", json::object())},
                {"uncertainty", functional.value("uncertainty", 0.5)},
                {"modalities", functional.value("modalities", json::array())},
                {"sceneSummary", functional.value("sceneSummary", std::string())},
                {"episodeSummary", functional.value("episodeSummary", std::string())},
                {"brainRegions", functional.value("brainRegions", json::object())},
                {"corticalMap", corticalMap},
                {"pathways", pathways}};
}

inline json buildHumanThoughtModel(const json &sessionState, BrainProfileOptions options = {}) {
    const auto structural = isStructuralBrainState(sessionState) ? sessionState : buildStructuralBrainCore(sessionState, options);
    const auto map = structural.value("corticalMap", json::object());

    const auto occipital = collectJsonStrings(map.value("occipitalCortex", json::array()), 2);
    const auto auditory = collectJsonStrings(map.value("auditoryCortex", json::array()), 2);
    const auto hippocampus = collectJsonStrings(map.value("hippocampus", json::array()), 2);
    const auto prefrontal = collectJsonStrings(map.value("prefrontalCortex", json::array()), 2);
    const auto basalGanglia = collectJsonStrings(map.value("basalGanglia", json::array()), 2);
    const auto motor = collectJsonStrings(map.value("motorCortex", json::array()), 2);
    const auto broca = collectJsonStrings(map.value("brocaArea", json::array()), 2);
    const auto wernicke = collectJsonStrings(map.value("wernickeArea", json::array()), 2);

    json phases = json::array({json{{"stage", "sensory_registration"},
                                    {"regions", json::array({"occipitalCortex", "auditoryCortex"})},
                                    {"strategy", "compress multisensory evidence into a few vivid anchors"},
                                    {"evidence", toJsonArray(occipital)}},
                                json{{"stage", "salience_gating"},
                                    {"regions", json::array({"thalamus", "parietalAssociationCortex"})},
                                    {"strategy", "keep only the most behaviorally relevant cues in attention"},
                                    {"evidence", structural.value("attentionFocus", json::array())}},
                                json{{"stage", "associative_recall"},
                                    {"regions", json::array({"hippocampus", "wernickeArea"})},
                                    {"strategy", "retrieve the closest remembered pattern and verbal concept"},
                                    {"evidence", toJsonArray(hippocampus.empty() ? wernicke : hippocampus)}},
                                json{{"stage", "counterfactual_branching"},
                                    {"regions", json::array({"prefrontalCortex", "brocaArea"})},
                                    {"strategy", "simulate only a few plausible next states instead of exhaustive search"},
                                    {"evidence", toJsonArray(prefrontal.empty() ? broca : prefrontal)}},
                                json{{"stage", "action_commitment"},
                                    {"regions", json::array({"basalGanglia", "motorCortex", "cerebellum"})},
                                    {"strategy", "choose a satisficing action and stabilize timing"},
                                    {"evidence", toJsonArray(!motor.empty() ? motor : basalGanglia)}},
                                json{{"stage", "conscious_report"},
                                    {"regions", json::array({"brocaArea", "wernickeArea"})},
                                    {"strategy", "turn the selected branch into a compact verbal explanation"},
                                    {"evidence", toJsonArray(!broca.empty() ? broca : wernicke)}}});

    return json{{"ok", true},
                {"mode", "reverse-human-thought-inference"},
                {"thoughtHeuristics", json::array({"sensory chunking", "analogy recall", "branch pruning", "satisficing action selection", "verbal self-report"})},
                {"phases", phases},
                {"dominantStrategy", "Use sensory anchors plus episodic analogies to prune search before explicit reasoning."}};
}


inline json buildBrainProfile(const json &sessionState, BrainProfileOptions options = {}) {
    if (options.kind == BrainProfileKind::Functional) {
        auto functional = buildCognitiveState(sessionState, toCognitiveStateOptions(options));
        const auto runtime = buildFunctionalBrainRuntime(functional, options);
        functional["mode"] = runtime.value("mode", std::string("online-functional-loop"));
        functional["selectedAction"] = runtime.value("selectedAction", std::string());
        functional["consolidationTarget"] = runtime.value("consolidationTarget", std::string());
        functional["functionalRuntime"] = runtime.value("functionalRuntime", json::object());
        functional["responsePlan"] = buildResponsePlan(functional, 4, options.maxChars);
        return functional;
    }

    auto structural = buildStructuralBrainCore(sessionState, options);
    structural["mode"] = "reverse-human-thought-inference";
    structural["humanThoughtModel"] = buildHumanThoughtModel(structural, options);
    structural["responsePlan"] = buildResponsePlan(structural, 4, options.maxChars);
    return structural;
}

inline json buildBrainProfiles(const json &sessionState, BrainProfileOptions options = {}) {
    BrainProfileOptions functionalOptions = options;
    functionalOptions.kind = BrainProfileKind::Functional;
    BrainProfileOptions structuralOptions = options;
    structuralOptions.kind = BrainProfileKind::Structural;
    return json{{"ok", true},
                {"application", buildBrainProfile(sessionState, functionalOptions)},
                {"research", buildBrainProfile(sessionState, structuralOptions)}};
}

inline std::string buildCognitivePromptContext(const json &cognitiveState, std::size_t maxCharsPerLine = 160) {
    if (!cognitiveState.is_object()) {
        return std::string();
    }
    std::vector<std::string> lines;
    if (cognitiveState.contains("profile") && cognitiveState["profile"].is_string()) {
        const std::string purpose = cognitiveState.value("purpose", std::string());
        std::string profileLine = "world_brain|profile: " + cognitiveState["profile"].get<std::string>();
        if (!purpose.empty()) {
            profileLine += " (" + purpose + ")";
        }
        lines.push_back(truncateText(profileLine, maxCharsPerLine));
    }
    if (cognitiveState.contains("attentionFocus") && cognitiveState["attentionFocus"].is_array() && !cognitiveState["attentionFocus"].empty()) {
        std::vector<std::string> focus;
        for (const auto &entry : cognitiveState["attentionFocus"]) {
            if (entry.is_string()) {
                focus.push_back(entry.get<std::string>());
            }
        }
        if (!focus.empty()) {
            lines.push_back("world_cognition|focus: " + truncateText(joinStrings(focus, ", "), maxCharsPerLine));
        }
    }
    if (cognitiveState.contains("activeGoals") && cognitiveState["activeGoals"].is_array() && !cognitiveState["activeGoals"].empty()) {
        std::vector<std::string> goals;
        for (const auto &entry : cognitiveState["activeGoals"]) {
            if (entry.is_string()) {
                goals.push_back(entry.get<std::string>());
            }
        }
        if (!goals.empty()) {
            lines.push_back("world_goals|active: " + truncateText(joinStrings(goals, "; "), maxCharsPerLine));
        }
    }
    if (cognitiveState.contains("workingMemory") && cognitiveState["workingMemory"].is_array() && !cognitiveState["workingMemory"].empty()) {
        std::vector<std::string> wm;
        for (const auto &entry : cognitiveState["workingMemory"]) {
            if (entry.is_string()) {
                wm.push_back(entry.get<std::string>());
            }
            if (wm.size() >= 2) {
                break;
            }
        }
        if (!wm.empty()) {
            lines.push_back("world_working_memory|top: " + truncateText(joinStrings(wm, " || "), maxCharsPerLine));
        }
    }
    if (cognitiveState.contains("uncertainty") && cognitiveState["uncertainty"].is_number()) {
        std::ostringstream oss;
        oss << "world_uncertainty|level: " << std::fixed << std::setprecision(2) << cognitiveState["uncertainty"].get<double>();
        lines.push_back(oss.str());
    }

    if (cognitiveState.contains("selectedAction") && cognitiveState["selectedAction"].is_string()) {
        lines.push_back("world_action|selected: " + truncateText(cognitiveState["selectedAction"].get<std::string>(), maxCharsPerLine));
    }
    if (cognitiveState.contains("consolidationTarget") && cognitiveState["consolidationTarget"].is_string()) {
        lines.push_back("world_replay|target: " + truncateText(cognitiveState["consolidationTarget"].get<std::string>(), maxCharsPerLine));
    }

    if (cognitiveState.contains("corticalSystems") && cognitiveState["corticalSystems"].is_object()) {
        const auto &systems = cognitiveState["corticalSystems"];
        const auto visual = collectJsonStrings(systems.value("visualCortex", json::object()).value("observations", json::array()), 2);
        const auto auditory = collectJsonStrings(systems.value("auditoryCortex", json::object()).value("observations", json::array()), 2);
        const auto motor = collectJsonStrings(systems.value("motorPlanner", json::object()).value("intentions", json::array()), 2);
        if (!visual.empty()) {
            lines.push_back("world_visual|active: " + truncateText(joinStrings(visual, " || "), maxCharsPerLine));
        }
        if (!auditory.empty()) {
            lines.push_back("world_audio|active: " + truncateText(joinStrings(auditory, " || "), maxCharsPerLine));
        }
        if (!motor.empty()) {
            lines.push_back("world_motor|plan: " + truncateText(joinStrings(motor, " || "), maxCharsPerLine));
        }
        if (cognitiveState.contains("functionalRuntime") && cognitiveState["functionalRuntime"].is_object()) {
            const auto runtime = cognitiveState["functionalRuntime"];
            const auto rewards = collectJsonStrings(runtime.value("valueSystem", json::object()).value("rewardDrivers", json::array()), 2);
            if (!rewards.empty()) {
                lines.push_back("world_reward|drivers: " + truncateText(joinStrings(rewards, " || "), maxCharsPerLine));
            }
        }
    } else if (cognitiveState.contains("corticalMap") && cognitiveState["corticalMap"].is_object()) {
        const auto &map = cognitiveState["corticalMap"];
        const auto occipital = collectJsonStrings(map.value("occipitalCortex", json::array()), 2);
        const auto auditory = collectJsonStrings(map.value("auditoryCortex", json::array()), 2);
        const auto prefrontal = collectJsonStrings(map.value("prefrontalCortex", json::array()), 2);
        if (!occipital.empty()) {
            lines.push_back("world_occipital|active: " + truncateText(joinStrings(occipital, " || "), maxCharsPerLine));
        }
        if (!auditory.empty()) {
            lines.push_back("world_auditory_cortex|active: " + truncateText(joinStrings(auditory, " || "), maxCharsPerLine));
        }
        if (!prefrontal.empty()) {
            lines.push_back("world_prefrontal|plan: " + truncateText(joinStrings(prefrontal, " || "), maxCharsPerLine));
        }
        if (cognitiveState.contains("humanThoughtModel") && cognitiveState["humanThoughtModel"].is_object()) {
            const auto heuristics = collectJsonStrings(cognitiveState["humanThoughtModel"].value("thoughtHeuristics", json::array()), 2);
            if (!heuristics.empty()) {
                lines.push_back("world_human_thought|heuristics: " + truncateText(joinStrings(heuristics, " || "), maxCharsPerLine));
            }
        }
    }
    return joinStrings(lines, "\n");
}

inline void appendGraphContextLine(std::string &graphContext, const std::string &line) {
    if (line.empty()) {
        return;
    }
    if (graphContext.empty()) {
        graphContext = line;
    } else {
        graphContext += "\n" + line;
    }
}

inline std::string buildReasoningAssemblyCacheKey(const std::string &initialGraphContext,
                                                  const json &worldState,
                                                  const ReasoningAssemblyOptions &options) {
    const auto hotspot = analyzeReasoningHotspot("reasoning-assembly", worldState, initialGraphContext);
    std::ostringstream oss;
    oss << hotspot.canonicalKey << '|'
        << options.promptOptions.maxRecentEvidence << '|'
        << options.promptOptions.maxEvidenceChars << '|'
        << options.promptOptions.maxSummaryChars << '|'
        << options.promptOptions.maxObjectSlots << '|'
        << options.brainOptions.maxChars << '|'
        << brainProfileKindToString(options.brainOptions.kind) << '|'
        << options.includeBrainContext << '|'
        << options.includeReasoningAgenda << '|'
        << options.includeReasoningPlan << '|'
        << options.maxContextChars << '|'
        << options.maxPlanItems;
    return buildStableCacheKey("reasoning-assembly", oss.str(), worldState.value("sessionId", std::string()));
}

inline json buildReasoningAssembly(const std::string &initialGraphContext,
                                   const json &worldState,
                                   const ReasoningAssemblyOptions &options) {
    std::string graphContext = mergePromptContext(initialGraphContext, worldState, options.promptOptions);
    json cognitiveBrainState = json::object();
    if (options.includeBrainContext || options.includeReasoningAgenda || options.includeReasoningPlan) {
        cognitiveBrainState = buildBrainProfile(worldState, options.brainOptions);
    }

    const std::string cognitivePrompt = options.includeBrainContext ? buildCognitivePromptContext(cognitiveBrainState, options.maxContextChars) : std::string();
    if (options.includeBrainContext) {
        appendGraphContextLine(graphContext, cognitivePrompt);
    }

    const std::string agendaPrompt = options.includeReasoningAgenda ? buildReasoningAgendaPromptContext(cognitiveBrainState, options.maxContextChars) : std::string();
    if (options.includeReasoningAgenda) {
        appendGraphContextLine(graphContext, agendaPrompt);
    }

    json responsePlan = json::object();
    std::string responsePlanPrompt;
    if (options.includeReasoningPlan && cognitiveBrainState.is_object()) {
        responsePlan = buildResponsePlan(cognitiveBrainState, options.maxPlanItems, options.maxContextChars);
        responsePlanPrompt = buildResponsePlanPromptContext(responsePlan, options.maxContextChars);
        appendGraphContextLine(graphContext, responsePlanPrompt);
    }

    double worldUncertainty = -1.0;
    bool worldReflectionSuggested = false;
    if (cognitiveBrainState.is_object()) {
        worldUncertainty = cognitiveBrainState.value("uncertainty", -1.0);
        if (cognitiveBrainState.contains("reasoningAgenda") && cognitiveBrainState["reasoningAgenda"].is_object()) {
            worldReflectionSuggested = cognitiveBrainState["reasoningAgenda"].value("shouldReflect", false);
        }
    }

    return json{{"graphContext", graphContext},
                {"worldState", worldState},
                {"cognitiveBrainState", cognitiveBrainState},
                {"responsePlan", responsePlan},
                {"worldUncertainty", worldUncertainty},
                {"worldReflectionSuggested", worldReflectionSuggested},
                {"promptContext", buildPromptContext(worldState, options.promptOptions)},
                {"cognitivePromptContext", cognitivePrompt},
                {"reasoningAgendaPromptContext", agendaPrompt},
                {"responsePlanPromptContext", responsePlanPrompt}};
}

inline bool isSensoryModality(const std::string &modality) {
    const auto lowered = lowerCopy(trimCopy(modality));
    return lowered == "vision" || lowered == "image" || lowered == "video" || lowered == "speech" || lowered == "audio";
}

inline std::string extractVideoCapturePath(const json &evidence) {
    if (!evidence.contains("metadata") || !evidence["metadata"].is_object()) {
        return {};
    }

    const auto &metadata = evidence["metadata"];
    auto readString = [&](const json &obj, const char *key) -> std::string {
        if (obj.contains(key) && obj[key].is_string()) {
            return lowerCopy(trimCopy(obj[key].get<std::string>()));
        }
        return {};
    };

    std::string capturePath = readString(metadata, "cameraInterface");
    if (capturePath.empty()) {
        capturePath = readString(metadata, "sensorInterface");
    }
    if (capturePath.empty()) {
        capturePath = readString(metadata, "captureBus");
    }
    if (capturePath.empty()) {
        capturePath = readString(metadata, "cameraBus");
    }
    if (capturePath.empty()) {
        capturePath = readString(metadata, "transport");
    }
    if (capturePath.empty() && metadata.contains("vjepa2") && metadata["vjepa2"].is_object()) {
        capturePath = readString(metadata["vjepa2"], "cameraInterface");
    }
    return capturePath;
}

inline bool isDirectVideoCapturePath(const std::string &capturePath) {
    const auto lowered = lowerCopy(trimCopy(capturePath));
    return lowered.find("mipi") != std::string::npos || lowered.find("csi") != std::string::npos;
}

inline std::string describeVideoCapturePath(const std::string &capturePath) {
    const auto normalized = lowerCopy(trimCopy(capturePath));
    if (normalized.empty()) {
        return {};
    }
    if (isDirectVideoCapturePath(normalized)) {
        return " from the direct micro-mipi / CSI camera path";
    }
    return " from the " + normalized + " video path";
}

inline std::size_t countKeywordOverlap(const std::string &lhs, const std::string &rhs, std::size_t maxKeywords = 10) {
    std::unordered_set<std::string> lhsKeywords;
    for (const auto &keyword : extractKeywords(lhs, maxKeywords)) {
        lhsKeywords.insert(keyword);
    }
    std::size_t overlap = 0;
    for (const auto &keyword : extractKeywords(rhs, maxKeywords)) {
        if (lhsKeywords.contains(keyword)) {
            ++overlap;
        }
    }
    return overlap;
}

inline double scoreVjepa2Candidate(const std::string &source,
                                   const std::string &target,
                                   const std::string &languageAnchor,
                                   const std::string &capturePath) {
    const auto normalizedSource = lowerCopy(trimCopy(source));
    double priority = 0.40;

    if (normalizedSource == "fusion_video_timeline_text") {
        priority += 0.80;
    } else if (normalizedSource == "fusion_video_text") {
        priority += 0.72;
    } else if (normalizedSource.find("focus") != std::string::npos) {
        priority += 0.44;
    } else if (normalizedSource.find("late") != std::string::npos) {
        priority += 0.40;
    } else if (normalizedSource.find("medium") != std::string::npos || normalizedSource.find("_mid") != std::string::npos) {
        priority += 0.24;
    } else if (normalizedSource.find("coarse") != std::string::npos || normalizedSource.find("early") != std::string::npos) {
        priority += 0.06;
    }

    static const std::unordered_set<std::string> salientKeywords = {
        "move",       "moving",      "motion",      "accelerate",  "accelerates", "accelerating",
        "approach",   "approaching", "target",      "targets",     "trajectory",  "collision",
        "avoid",      "obstacle",    "risk",        "gate",        "fence",       "silhouette",
        "silhouettes", "track",      "tracking",    "velocity",    "fast",        "fastest"
    };

    std::size_t salientHits = 0;
    for (const auto &keyword : extractKeywords(target, 14)) {
        if (salientKeywords.contains(keyword)) {
            ++salientHits;
        }
    }
    priority += std::min<double>(0.28, static_cast<double>(salientHits) * 0.04);

    const auto overlap = countKeywordOverlap(target, languageAnchor, 12);
    priority += std::min<double>(0.32, static_cast<double>(overlap) * 0.08);

    if (isDirectVideoCapturePath(capturePath)) {
        priority += 0.10;
        if (normalizedSource.find("focus") != std::string::npos ||
            normalizedSource.find("late") != std::string::npos ||
            normalizedSource.find("fusion") != std::string::npos) {
            priority += 0.08;
        }
    }

    return priority;
}

inline std::vector<std::pair<std::string, std::string>> buildVjepa2CompressionViews(const json &evidence,
                                                                                     std::size_t maxChars,
                                                                                     std::size_t maxLevels) {
    std::vector<std::pair<std::string, std::string>> out;
    std::unordered_set<std::string> seen;

    auto appendView = [&](const std::string &level, const std::string &text) {
        if (out.size() >= maxLevels) {
            return;
        }
        std::string normalizedLevel = lowerCopy(trimCopy(level));
        std::string normalizedText = truncateText(trimCopy(text), maxChars);
        if (normalizedLevel.empty() || normalizedText.empty()) {
            return;
        }
        const std::string dedupe = normalizedLevel + "|" + normalizedText;
        if (!seen.insert(dedupe).second) {
            return;
        }
        out.push_back({normalizedLevel, normalizedText});
    };

    if (evidence.contains("metadata") && evidence["metadata"].is_object()) {
        const auto &metadata = evidence["metadata"];

        auto appendFromObject = [&](const json &obj) {
            if (!obj.is_object()) {
                return;
            }
            if (obj.contains("coarse") && obj["coarse"].is_string()) {
                appendView("coarse", obj["coarse"].get<std::string>());
            }
            if (obj.contains("medium") && obj["medium"].is_string()) {
                appendView("medium", obj["medium"].get<std::string>());
            }
            if (obj.contains("focus") && obj["focus"].is_string()) {
                appendView("focus", obj["focus"].get<std::string>());
            }
            if (obj.contains("detail") && obj["detail"].is_string()) {
                appendView("focus", obj["detail"].get<std::string>());
            }
        };

        if (metadata.contains("vjepa2")) {
            appendFromObject(metadata["vjepa2"]);
        }
        if (metadata.contains("videoCompression")) {
            appendFromObject(metadata["videoCompression"]);
        }
        if (metadata.contains("compressionLevels") && metadata["compressionLevels"].is_array()) {
            std::size_t levelIndex = 0;
            for (const auto &entry : metadata["compressionLevels"]) {
                if (!entry.is_string()) {
                    continue;
                }
                const std::string level = levelIndex == 0 ? "coarse" : (levelIndex == 1 ? "medium" : "focus");
                appendView(level, entry.get<std::string>());
                ++levelIndex;
                if (out.size() >= maxLevels) {
                    break;
                }
            }
        }
    }

    PromptContextOptions promptOptions;
    promptOptions.maxEvidenceChars = maxChars;
    const std::string baseSummary = buildPromptEvidenceText(evidence, promptOptions);
    if (!baseSummary.empty()) {
        appendView("coarse", truncateText(baseSummary, std::max<std::size_t>(24, maxChars / 3)));
        appendView("medium", truncateText(baseSummary, std::max<std::size_t>(32, (maxChars * 2) / 3)));
        appendView("focus", truncateText(baseSummary, maxChars));
    }

    return out;
}

inline std::vector<std::pair<std::string, std::string>> buildVjepa2TemporalWindows(const json &evidence,
                                                                                     std::size_t maxChars,
                                                                                     std::size_t maxWindows) {
    std::vector<std::pair<std::string, std::string>> out;
    std::unordered_set<std::string> seen;

    auto appendWindow = [&](const std::string &name, const std::string &text) {
        if (out.size() >= maxWindows) {
            return;
        }
        const std::string normalizedName = lowerCopy(trimCopy(name));
        const std::string normalizedText = truncateText(trimCopy(text), maxChars);
        if (normalizedName.empty() || normalizedText.empty()) {
            return;
        }
        const std::string dedupe = normalizedName + "|" + normalizedText;
        if (!seen.insert(dedupe).second) {
            return;
        }
        out.push_back({normalizedName, normalizedText});
    };

    auto appendFromTimelineObject = [&](const json &obj) {
        if (!obj.is_object()) {
            return;
        }
        if (obj.contains("early") && obj["early"].is_string()) {
            appendWindow("early", obj["early"].get<std::string>());
        }
        if (obj.contains("mid") && obj["mid"].is_string()) {
            appendWindow("mid", obj["mid"].get<std::string>());
        }
        if (obj.contains("late") && obj["late"].is_string()) {
            appendWindow("late", obj["late"].get<std::string>());
        }
    };

    if (evidence.contains("metadata") && evidence["metadata"].is_object()) {
        const auto &metadata = evidence["metadata"];
        if (metadata.contains("vjepa2") && metadata["vjepa2"].is_object()) {
            const auto &vjepa2 = metadata["vjepa2"];
            if (vjepa2.contains("timeline")) {
                appendFromTimelineObject(vjepa2["timeline"]);
            }
        }
        if (metadata.contains("videoTimeline") && metadata["videoTimeline"].is_array()) {
            std::size_t index = 0;
            for (const auto &entry : metadata["videoTimeline"]) {
                if (!entry.is_string()) {
                    continue;
                }
                const std::string name = index == 0 ? "early" : (index == 1 ? "mid" : "late");
                appendWindow(name, entry.get<std::string>());
                ++index;
                if (out.size() >= maxWindows) {
                    break;
                }
            }
        }
        if (metadata.contains("videoWindows") && metadata["videoWindows"].is_object()) {
            appendFromTimelineObject(metadata["videoWindows"]);
        }
    }

    if (out.empty()) {
        const auto compressionViews = buildVjepa2CompressionViews(evidence, maxChars, std::max<std::size_t>(1, maxWindows));
        for (std::size_t index = 0; index < compressionViews.size() && out.size() < maxWindows; ++index) {
            const std::string name = index == 0 ? "early" : (index == 1 ? "mid" : "late");
            appendWindow(name, compressionViews[index].second);
        }
    }

    return out;
}

inline std::vector<GroundedLearningSample> buildGroundedLearningSamples(const json &sessionState, GroundedLearningOptions options = {}) {
    std::vector<GroundedLearningSample> samples;
    json sceneState = (sessionState.contains("sceneState") && sessionState["sceneState"].is_object()) ? sessionState["sceneState"] : json::object();
    json episode = (sessionState.contains("episode") && sessionState["episode"].is_object()) ? sessionState["episode"] : json::object();
    json recentEvidence = (sessionState.contains("recentEvidence") && sessionState["recentEvidence"].is_array()) ? sessionState["recentEvidence"] : json::array();

    PromptContextOptions promptOptions;
    promptOptions.maxRecentEvidence = options.maxRecentEvidence;
    promptOptions.maxEvidenceChars = std::min<std::size_t>(options.maxTargetChars, 160);
    promptOptions.maxSummaryChars = std::min<std::size_t>(options.maxTargetChars + 80, 240);
    promptOptions.maxObjectSlots = 6;

    const std::string promptContext = truncateText(buildPromptContext(sessionState, promptOptions), options.maxGraphChars);
    const std::string sceneSummary = truncateText(trimCopy(sceneState.value("summary", std::string())), options.maxTargetChars);
    const std::string episodeSummary = truncateText(trimCopy(episode.value("summary", std::string())), options.maxTargetChars);

    bool sawVision = false;
    bool sawSpeech = false;
    bool sawVideo = false;
    std::string recentTextSummary;
    std::unordered_set<std::string> seenTargets;

    struct RankedCandidate {
        double priority;
        std::string input;
        std::string target;
        std::string source;
    };
    struct VideoSelection {
        double priority{-1.0};
        std::string target;
        std::string capturePath;
    };

    std::vector<RankedCandidate> rankedCandidates;
    VideoSelection bestVideoSelection;
    VideoSelection bestTemporalSelection;

    auto appendSample = [&](std::string input, std::string target, std::string source) {
        input = truncateText(trimCopy(input), options.maxInputChars);
        target = truncateText(trimCopy(target), options.maxTargetChars);
        source = lowerCopy(trimCopy(source));
        if (input.empty() || target.empty() || source.empty()) {
            return;
        }
        const std::string dedupeKey = source + "|" + target;
        if (!seenTargets.insert(dedupeKey).second) {
            return;
        }
        samples.push_back(GroundedLearningSample{std::move(input), std::move(target), promptContext, std::move(source)});
    };

    auto queueCandidate = [&](double priority, std::string input, std::string target, std::string source) {
        input = truncateText(trimCopy(input), options.maxInputChars);
        target = truncateText(trimCopy(target), options.maxTargetChars);
        source = lowerCopy(trimCopy(source));
        if (input.empty() || target.empty() || source.empty()) {
            return;
        }
        rankedCandidates.push_back(RankedCandidate{priority, std::move(input), std::move(target), std::move(source)});
    };

    auto considerSelection = [&](VideoSelection &selection,
                                 const std::string &source,
                                 const std::string &target,
                                 const std::string &capturePath) {
        const double priority = scoreVjepa2Candidate(source, target, recentTextSummary, capturePath);
        if (priority > selection.priority) {
            selection.priority = priority;
            selection.target = target;
            selection.capturePath = capturePath;
        }
    };

    if (recentEvidence.is_array() && !recentEvidence.empty()) {
        std::size_t start = recentEvidence.size() > options.maxRecentEvidence ? recentEvidence.size() - options.maxRecentEvidence : 0;
        for (std::size_t index = start; index < recentEvidence.size(); ++index) {
            if (!recentEvidence[index].is_object()) {
                continue;
            }
            const auto &evidence = recentEvidence[index];
            const std::string modality = lowerCopy(trimCopy(evidence.value("modality", std::string())));
            if (modality == "text") {
                const std::string textSummary = buildPromptEvidenceText(evidence, promptOptions);
                if (!textSummary.empty()) {
                    recentTextSummary = textSummary;
                }
            }
        }
        for (std::size_t index = start; index < recentEvidence.size(); ++index) {
            if (!recentEvidence[index].is_object()) {
                continue;
            }
            const auto &evidence = recentEvidence[index];
            std::string modality = lowerCopy(trimCopy(evidence.value("modality", std::string())));
            if (!isSensoryModality(modality)) {
                continue;
            }
            std::string target = buildPromptEvidenceText(evidence, promptOptions);
            if (target.empty()) {
                continue;
            }
            if (modality == "vision" || modality == "image") {
                sawVision = true;
                queueCandidate(0.62, "Ground the current visual observation into explicit knowledge.", target, "vision");
            } else if (modality == "video") {
                sawVision = true;
                sawVideo = true;
                const std::string capturePath = extractVideoCapturePath(evidence);
                const std::string captureHint = describeVideoCapturePath(capturePath);
                if (options.includeVideoCompressionSamples) {
                    const auto views = buildVjepa2CompressionViews(evidence,
                                                                   std::min<std::size_t>(options.maxTargetChars, 160),
                                                                   std::max<std::size_t>(1, options.maxVideoCompressionLevels));
                    for (const auto &view : views) {
                        const std::string source = "video_" + view.first;
                        queueCandidate(scoreVjepa2Candidate(source, view.second, recentTextSummary, capturePath),
                                       "Ground the video stream" + captureHint + " at v-jepa2 " + view.first + " level into explicit knowledge.",
                                       view.second,
                                       source);
                        considerSelection(bestVideoSelection, source, view.second, capturePath);
                    }
                } else {
                    queueCandidate(scoreVjepa2Candidate("video", target, recentTextSummary, capturePath),
                                   "Ground the current video stream" + captureHint + " into explicit knowledge.",
                                   target,
                                   "video");
                    considerSelection(bestVideoSelection, "video", target, capturePath);
                }
                if (options.includeVideoTemporalSamples) {
                    const auto windows = buildVjepa2TemporalWindows(evidence,
                                                                    std::min<std::size_t>(options.maxTargetChars, 160),
                                                                    std::max<std::size_t>(1, options.maxVideoTemporalWindows));
                    for (const auto &window : windows) {
                        const std::string source = "video_timeline_" + window.first;
                        queueCandidate(scoreVjepa2Candidate(source, window.second, recentTextSummary, capturePath),
                                       "Ground the video stream" + captureHint + " temporal window (" + window.first + ") into explicit knowledge.",
                                       window.second,
                                       source);
                        considerSelection(bestVideoSelection, source, window.second, capturePath);
                        considerSelection(bestTemporalSelection, source, window.second, capturePath);
                    }
                }
            } else if (modality == "speech" || modality == "audio") {
                sawSpeech = true;
                queueCandidate(0.62, "Ground the current auditory observation into explicit knowledge.", target, "speech");
            }
        }
    }

    if (options.includeFusionSample && sawVision && sawSpeech) {
        queueCandidate(0.84,
                       "Fuse the current visual and auditory observations into one consistent knowledge statement.",
                       !episodeSummary.empty() ? episodeSummary : sceneSummary,
                       "fusion");
    }

    if (options.includeFusionSample && sawVideo && !recentTextSummary.empty()) {
        std::vector<std::string> fusionParts;
        const std::string goalAnchor = !episodeSummary.empty() ? episodeSummary : sceneSummary;
        if (!bestVideoSelection.target.empty()) {
            fusionParts.push_back(bestVideoSelection.target);
        } else if (!goalAnchor.empty()) {
            fusionParts.push_back(goalAnchor);
        }
        if (isDirectVideoCapturePath(bestVideoSelection.capturePath)) {
            fusionParts.push_back("capture path: " + bestVideoSelection.capturePath);
        }
        fusionParts.push_back("language anchor: " + recentTextSummary);
        if (!goalAnchor.empty()) {
            fusionParts.push_back("episode anchor: " + goalAnchor);
        }
        const std::string fusionTarget = truncateText(joinStrings(fusionParts, " | "), options.maxTargetChars);
        queueCandidate(scoreVjepa2Candidate("fusion_video_text", fusionTarget, recentTextSummary, bestVideoSelection.capturePath),
                       "Align the highest-salience v-jepa2 video abstraction" + describeVideoCapturePath(bestVideoSelection.capturePath) + " with the latest language context.",
                       fusionTarget,
                       "fusion_video_text");

        if (options.includeVideoTemporalSamples && !bestTemporalSelection.target.empty()) {
            std::vector<std::string> temporalParts;
            temporalParts.push_back(bestTemporalSelection.target);
            if (isDirectVideoCapturePath(bestTemporalSelection.capturePath)) {
                temporalParts.push_back("capture path: " + bestTemporalSelection.capturePath);
            }
            temporalParts.push_back("language anchor: " + recentTextSummary);
            if (!goalAnchor.empty()) {
                temporalParts.push_back("episode anchor: " + goalAnchor);
            }
            const std::string temporalTarget = truncateText(joinStrings(temporalParts, " | "), options.maxTargetChars);
            queueCandidate(scoreVjepa2Candidate("fusion_video_timeline_text", temporalTarget, recentTextSummary, bestTemporalSelection.capturePath),
                           "Align the highest-value video temporal focus" + describeVideoCapturePath(bestTemporalSelection.capturePath) + " with language guidance.",
                           temporalTarget,
                           "fusion_video_timeline_text");
        }
    }

    if (options.includeSceneRecallSample && (sawVision || sawSpeech)) {
        queueCandidate(0.34,
                       "What is currently known from the recent sensory observations?",
                       !sceneSummary.empty() ? sceneSummary : episodeSummary,
                       "scene");
    }

    std::stable_sort(rankedCandidates.begin(), rankedCandidates.end(), [](const RankedCandidate &lhs, const RankedCandidate &rhs) {
        return lhs.priority > rhs.priority;
    });

    for (const auto &candidate : rankedCandidates) {
        appendSample(candidate.input, candidate.target, candidate.source);
        if (samples.size() >= options.maxSamples) {
            samples.resize(options.maxSamples);
            return samples;
        }
    }

    if (samples.size() > options.maxSamples) {
        samples.resize(options.maxSamples);
    }
    return samples;
}

inline std::vector<GroundedLearningSample> buildFunctionalBrainLearningSamples(const json &sessionState, GroundedLearningOptions options = {}) {
    std::vector<GroundedLearningSample> samples;
    BrainProfileOptions brainOptions;
    brainOptions.kind = BrainProfileKind::Functional;
    brainOptions.maxChars = options.maxTargetChars;
    const auto brain = buildBrainProfile(sessionState, brainOptions);

    PromptContextOptions promptOptions;
    promptOptions.maxRecentEvidence = options.maxRecentEvidence;
    promptOptions.maxEvidenceChars = std::min<std::size_t>(options.maxTargetChars, 160);
    promptOptions.maxSummaryChars = std::min<std::size_t>(options.maxTargetChars + 80, 240);
    promptOptions.maxObjectSlots = 6;
    std::string graph = buildPromptContext(sessionState, promptOptions);
    const auto brainPrompt = buildCognitivePromptContext(brain, 160);
    if (!brainPrompt.empty()) {
        if (!graph.empty()) {
            graph += "\n";
        }
        graph += brainPrompt;
    }
    graph = truncateText(graph, options.maxGraphChars);

    auto appendSample = [&](std::string input, std::string target, std::string source) {
        input = truncateText(trimCopy(input), options.maxInputChars);
        target = truncateText(trimCopy(target), options.maxTargetChars);
        source = lowerCopy(trimCopy(source));
        if (input.empty() || target.empty() || source.empty()) {
            return;
        }
        samples.push_back(GroundedLearningSample{std::move(input), std::move(target), graph, std::move(source)});
    };

    if (brain.contains("selectedAction") && brain["selectedAction"].is_string()) {
        appendSample("Given the current multisensory world state, what should the application brain do next?",
                     brain["selectedAction"].get<std::string>(),
                     "functional_action");
    }
    if (brain.contains("consolidationTarget") && brain["consolidationTarget"].is_string()) {
        appendSample("Which memory trace should the application brain consolidate now?",
                     brain["consolidationTarget"].get<std::string>(),
                     "functional_replay");
    }
    const auto focus = collectJsonStrings(brain.value("attentionFocus", json::array()), 3);
    if (!focus.empty()) {
        appendSample("What sensory targets deserve attention right now?",
                     joinStrings(focus, ", "),
                     "functional_attention");
    }

    if (samples.size() > options.maxSamples) {
        samples.resize(options.maxSamples);
    }
    return samples;
}

inline json buildSceneVoxelPosition(const std::string &seed,
                                    std::size_t ordinal,
                                    const VirtualSceneOptions &options,
                                    std::size_t preferredLayer = 0,
                                    bool pinLayer = false) {
    const std::size_t width = std::max<std::size_t>(1, options.mapWidth);
    const std::size_t height = std::max<std::size_t>(1, options.mapHeight);
    const std::size_t depth = std::max<std::size_t>(1, options.mapDepth);
    const std::uint64_t hash = stableHash64(seed + "#" + std::to_string(ordinal));
    const std::size_t x = static_cast<std::size_t>(hash % width);
    const std::size_t y = static_cast<std::size_t>((hash / width) % height);
    const std::size_t planeSize = std::max<std::size_t>(1, width * height);
    std::size_t z = static_cast<std::size_t>((hash / planeSize) % depth);
    if (pinLayer) {
        z = std::min(preferredLayer, depth - 1);
    }
    return json{{"x", x},
                {"y", y},
                {"z", z},
                {"cellId", std::to_string(x) + ":" + std::to_string(y) + ":" + std::to_string(z)}};
}

inline std::string describeSceneVoxelPosition(const json &position) {
    return "cell " + std::to_string(position.value("x", 0)) + ":" +
           std::to_string(position.value("y", 0)) + ":" +
           std::to_string(position.value("z", 0));
}

inline std::vector<std::string> deriveEmbodiedCapabilities(const std::string &role,
                                                           const std::vector<std::string> &objectLabels,
                                                           const std::vector<std::string> &sceneTags,
                                                           const std::string &sceneSummary,
                                                           std::size_t maxCount,
                                                           std::size_t maxChars) {
    auto capabilities = deriveBodySchema(objectLabels, sceneTags, sceneSummary, maxCount, maxChars);
    std::unordered_set<std::string> seen;
    for (const auto &item : capabilities) {
        seen.insert(lowerCopy(item));
    }

    auto appendCapability = [&](const std::string &value) {
        appendUniqueLimited(capabilities, seen, value, maxCount, maxChars);
    };

    if (role == "observer") {
        appendCapability("maintain line-of-sight tracking and grounded perception updates");
    } else if (role == "planner") {
        appendCapability("simulate short-horizon trajectories before committing body movement");
    } else if (role == "critic") {
        appendCapability("verify whether embodied transitions remain consistent with shared evidence");
    } else if (role == "memory") {
        appendCapability("bind episodic traces to stable landmarks and repeated paths");
    } else if (role == "explorer") {
        appendCapability("probe adjacent cells for affordances and route alternatives");
    } else {
        appendCapability("coordinate with nearby agents through embodied presence and short dialogue loops");
    }
    return capabilities;
}

inline std::string buildAgentDialogueUtterance(const std::string &speakerRole,
                                               const std::string &listenerRole,
                                               const std::string &focus,
                                               const std::string &sceneSummary,
                                               const std::string &cellLabel,
                                               std::size_t step,
                                               std::size_t turn,
                                               std::size_t maxChars) {
    std::string utterance;
    if (speakerRole == "planner") {
        utterance = "planner to " + listenerRole + " from " + cellLabel + ": coordinate around " + focus;
    } else if (speakerRole == "critic") {
        utterance = "critic to " + listenerRole + " from " + cellLabel + ": verify the transition around " + focus;
    } else if (speakerRole == "memory") {
        utterance = "memory to " + listenerRole + " from " + cellLabel + ": anchor this embodied state for recall";
    } else if (speakerRole == "explorer") {
        utterance = "explorer to " + listenerRole + " from " + cellLabel + ": I can probe an alternate route near " + focus;
    } else if (speakerRole == "observer") {
        utterance = "observer to " + listenerRole + " from " + cellLabel + ": current evidence still centers on " + focus;
    } else {
        utterance = speakerRole + " to " + listenerRole + " from " + cellLabel + ": stay aligned on " + focus;
    }
    if (!sceneSummary.empty()) {
        utterance += "; context " + sceneSummary;
    }
    utterance += " [step " + std::to_string(step + 1) + ", turn " + std::to_string(turn + 1) + "]";
    return truncateText(utterance, maxChars);
}

inline json buildEcologyVideoAbstractions(const json &sessionState, const VirtualSceneOptions &options) {
    json out = json::array();
    if (!options.includeEcologyFromVideo || options.maxEcologyClusters == 0) {
        return out;
    }

    json sceneState = (sessionState.contains("sceneState") && sessionState["sceneState"].is_object()) ? sessionState["sceneState"] : json::object();
    json recentEvidence = (sessionState.contains("recentEvidence") && sessionState["recentEvidence"].is_array()) ? sessionState["recentEvidence"] : json::array();
    const auto sceneTags = collectSceneTags(sceneState, 4);
    auto visualSummaries = collectRecentModalitySummaries(recentEvidence,
                                                          {"video", "vision", "image"},
                                                          std::max<std::size_t>(1, options.maxEcologyClusters),
                                                          std::min<std::size_t>(options.maxEventChars, 120));
    const std::string sceneSummary = trimCopy(sceneState.value("summary", std::string()));
    if (visualSummaries.empty() && !sceneSummary.empty()) {
        visualSummaries.push_back(truncateText(sceneSummary, std::min<std::size_t>(options.maxEventChars, 120)));
    }

    for (std::size_t index = 0; index < visualSummaries.size() && out.size() < options.maxEcologyClusters; ++index) {
        const std::string summary = visualSummaries[index];
        const auto compressionViews = buildVjepa2CompressionViews(
            json{{"graphSummary", summary}, {"metadata", json::object()}},
            std::min<std::size_t>(options.maxEventChars, 120),
            3);
        const auto temporalWindows = buildVjepa2TemporalWindows(
            json{{"graphSummary", summary}, {"metadata", json::object()}},
            std::min<std::size_t>(options.maxEventChars, 120),
            3);
        std::string coarseSummary;
        std::string mediumSummary;
        std::string focusSummary;
        for (const auto &view : compressionViews) {
            if (view.first == "coarse" && coarseSummary.empty()) {
                coarseSummary = view.second;
            } else if (view.first == "medium" && mediumSummary.empty()) {
                mediumSummary = view.second;
            } else if ((view.first == "focus" || view.first == "detail") && focusSummary.empty()) {
                focusSummary = view.second;
            }
        }
        if (coarseSummary.empty()) {
            coarseSummary = truncateText(summary, std::max<std::size_t>(24, options.maxEventChars / 3));
        }
        if (mediumSummary.empty()) {
            mediumSummary = truncateText(summary, std::max<std::size_t>(32, (options.maxEventChars * 2) / 3));
        }
        if (focusSummary.empty()) {
            focusSummary = truncateText(summary, options.maxEventChars);
        }
        std::string earlySummary;
        std::string midSummary;
        std::string lateSummary;
        for (const auto &window : temporalWindows) {
            if (window.first == "early" && earlySummary.empty()) {
                earlySummary = window.second;
            } else if (window.first == "mid" && midSummary.empty()) {
                midSummary = window.second;
            } else if (window.first == "late" && lateSummary.empty()) {
                lateSummary = window.second;
            }
        }
        if (earlySummary.empty()) {
            earlySummary = coarseSummary;
        }
        if (midSummary.empty()) {
            midSummary = mediumSummary;
        }
        if (lateSummary.empty()) {
            lateSummary = focusSummary;
        }
        const auto keywords = extractKeywords(summary + " " + joinStrings(sceneTags, " ", 3), 5);
        std::vector<std::string> descriptors;
        std::unordered_set<std::string> seen;
        if (!keywords.empty()) {
            appendUniqueLimited(descriptors, seen, "motion cluster: " + joinStrings(keywords, ", ", 3), 4, options.maxEventChars);
        }
        if (!sceneTags.empty()) {
            appendUniqueLimited(descriptors, seen, "habitat cues: " + joinStrings(sceneTags, ", ", 3), 4, options.maxEventChars);
        }
        appendUniqueLimited(descriptors,
                            seen,
                            "video abstraction models the ecology as a distributed field instead of enumerating species",
                            4,
                            options.maxEventChars);

        const std::string biomeLabel = !keywords.empty() ? "biome-" + keywords.front()
                                                         : (!sceneTags.empty() ? "biome-" + lowerCopy(sceneTags.front()) : std::string("biome-generic"));
        const auto position = buildSceneVoxelPosition("ecology:" + summary, index, options, 0, true);
        out.push_back(json{{"id", "ecosystem-cluster-" + std::to_string(index + 1)},
                           {"entityType", "non-intelligent-ecosystem"},
                           {"abstraction", "video-ecology-cluster"},
                           {"encoding", "v-jepa2"},
                           {"biomeLabel", biomeLabel},
                           {"sourceSummary", focusSummary},
                           {"compressedViews", json{{"coarse", coarseSummary},
                                                      {"medium", mediumSummary},
                                                      {"focus", focusSummary}}},
                           {"temporalWindows", json{{"early", earlySummary},
                                                      {"mid", midSummary},
                                                      {"late", lateSummary}}},
                           {"motionSignature", truncateText("distributed ecological motion around " + (!keywords.empty() ? joinStrings(keywords, ", ", 2) : std::string("ambient habitat")),
                                                            options.maxEventChars)},
                           {"descriptors", toJsonArray(descriptors)},
                           {"position", position},
                           {"body", json{{"form", "distributed-field"},
                                          {"presence", "ambient-zone"},
                                          {"sensorBasis", "video-abstraction"},
                                          {"interactionMode", "non-intelligent"}}}});
    }
    return out;
}

inline json buildEarthMapManifestFromOptions(const VirtualSceneOptions &options,
                                             const std::string &sceneSummary,
                                             const std::vector<std::string> &sceneTags) {
    json request = json::object();
    if (options.earthMapRequest.is_object()) {
        request = options.earthMapRequest;
    } else if (options.earthMapRequest.is_string()) {
        request["sourceUri"] = options.earthMapRequest.get<std::string>();
    }

    bool enabled = options.earthMapEnabled;
    if (request.contains("enabled") && request["enabled"].is_boolean()) {
        enabled = request["enabled"].get<bool>();
    }

    std::string sourceUri = trimCopy(request.value("sourceUri", request.value("uri", std::string())));
    const std::string format = lowerCopy(trimCopy(request.value("format", std::string("heightfield"))));
    const std::string coordinateFrame = trimCopy(request.value("coordinateFrame", std::string("wgs84-local-enu")));
    std::string regionLabel = trimCopy(request.value("regionLabel", std::string()));
    if (regionLabel.empty()) {
        regionLabel = !sceneTags.empty() ? lowerCopy(sceneTags.front()) + "-earth-sector" : std::string("global-earth");
    }
    const int lod = std::max(0, request.value("lod", 6));
    const double metersPerCell = std::max(1.0, request.value("metersPerCell", 750.0));

    if (enabled && sourceUri.empty() && format == "heightfield") {
        sourceUri = "static/earth_maps/china_relief_heightfield.json";
    }

    json geoBounds = json{{"latMin", -90.0}, {"latMax", 90.0}, {"lonMin", -180.0}, {"lonMax", 180.0}};
    if (request.contains("geoBounds") && request["geoBounds"].is_object()) {
        const auto &rawBounds = request["geoBounds"];
        geoBounds["latMin"] = rawBounds.value("latMin", geoBounds["latMin"].get<double>());
        geoBounds["latMax"] = rawBounds.value("latMax", geoBounds["latMax"].get<double>());
        geoBounds["lonMin"] = rawBounds.value("lonMin", geoBounds["lonMin"].get<double>());
        geoBounds["lonMax"] = rawBounds.value("lonMax", geoBounds["lonMax"].get<double>());
    }

    json importPlan = json::array({
        "stream map tiles or meshes into local training sectors",
        "load a heightfield or derive a terrain collider proxy",
        "align agent bodies to a local tangent frame",
        "progressively refine lod only where training activity concentrates"
    });

    std::string summary;
    if (enabled) {
        summary = "earth map " + format + " import for " + regionLabel + " uses " +
                  (!sourceUri.empty() ? sourceUri : std::string("an unspecified source")) +
                  " in " + coordinateFrame + " at lod " + std::to_string(lod);
        if (!sceneSummary.empty()) {
            summary += "; scene anchor " + sceneSummary;
        }
    } else {
        summary = "earth map import disabled for this simulated world";
    }

    return json{{"enabled", enabled},
                {"sourceUri", sourceUri},
                {"format", format.empty() ? std::string("heightfield") : format},
                {"coordinateFrame", coordinateFrame.empty() ? std::string("wgs84-local-enu") : coordinateFrame},
                {"regionLabel", regionLabel},
                {"lod", lod},
                {"metersPerCell", metersPerCell},
                {"geoBounds", geoBounds},
                {"importPlan", importPlan},
                {"status", enabled ? (sourceUri.empty() ? std::string("awaiting-source") : std::string("configured")) : std::string("disabled")},
                {"summary", truncateText(summary, 220)}};
}

inline std::string derivePhysicsBodyShape(const json &entity) {
    const std::string entityType = lowerCopy(trimCopy(entity.value("entityType", std::string())));
    const std::string role = lowerCopy(trimCopy(entity.value("role", std::string())));
    if (entityType == "non-intelligent-ecosystem") {
        return "sensor-volume";
    }
    if (role == "planner" || role == "memory") {
        return "capsule";
    }
    if (role == "critic") {
        return "box";
    }
    if (role == "explorer" || role == "scout") {
        return "sphere";
    }
    return "capsule";
}

inline double derivePhysicsBodyMass(const json &entity, std::size_t ordinal) {
    const std::string entityType = lowerCopy(trimCopy(entity.value("entityType", std::string())));
    if (entityType == "non-intelligent-ecosystem") {
        return 0.0;
    }
    const std::string role = lowerCopy(trimCopy(entity.value("role", std::string())));
    if (role == "critic") {
        return 85.0;
    }
    if (role == "planner") {
        return 78.0;
    }
    return 65.0 + static_cast<double>(ordinal % 5) * 4.0;
}

inline json buildPhysicsScenePlan(const VirtualSceneOptions &options,
                                  const json &agents,
                                  const json &ecologyEntities,
                                  const json &earthMap,
                                  const std::string &sceneSummary) {
    const std::string backend = lowerCopy(trimCopy(options.physicsBackend.empty() ? std::string("bullet3") : options.physicsBackend));
    json rigidBodies = json::array();
    json constraints = json::array();

    if (!options.physicsEnabled) {
        return json{{"enabled", false},
                    {"backend", backend},
                    {"substeps", options.physicsSubsteps},
                    {"status", "disabled"},
                    {"earthMap", earthMap},
                    {"rigidBodies", rigidBodies},
                    {"constraints", constraints},
                    {"summary", "physics scene disabled for this rollout"}};
    }

    if (earthMap.value("enabled", false)) {
        rigidBodies.push_back(json{{"id", "earth-terrain"},
                                   {"shape", earthMap.value("format", std::string()) == "heightfield" ? std::string("heightfield") : std::string("triangle-mesh")},
                                   {"massKg", 0.0},
                                   {"bodyClass", "static-terrain"},
                                   {"regionLabel", earthMap.value("regionLabel", std::string("global-earth"))}});
    } else {
        rigidBodies.push_back(json{{"id", "world-ground-plane"},
                                   {"shape", "plane"},
                                   {"massKg", 0.0},
                                   {"bodyClass", "static-ground"}});
    }

    for (std::size_t index = 0; index < agents.size(); ++index) {
        const auto &agent = agents[index];
        if (!agent.is_object()) {
            continue;
        }
        rigidBodies.push_back(json{{"id", agent.value("id", std::string())},
                                   {"shape", derivePhysicsBodyShape(agent)},
                                   {"massKg", derivePhysicsBodyMass(agent, index)},
                                   {"bodyClass", "dynamic-agent"},
                                   {"role", agent.value("role", std::string())},
                                   {"position", agent.value("position", json::object())}});
        if (index > 0) {
            constraints.push_back(json{{"type", "formation-link"},
                                       {"bodyA", agents[index - 1].value("id", std::string())},
                                       {"bodyB", agent.value("id", std::string())},
                                       {"purpose", "maintain embodied coordination across adjacent agents"}});
        }
    }

    for (const auto &entity : ecologyEntities) {
        if (!entity.is_object()) {
            continue;
        }
        rigidBodies.push_back(json{{"id", entity.value("id", std::string())},
                                   {"shape", derivePhysicsBodyShape(entity)},
                                   {"massKg", derivePhysicsBodyMass(entity, rigidBodies.size())},
                                   {"bodyClass", "ecology-field"},
                                   {"position", entity.value("position", json::object())}});
    }

    std::string summary = "physics plan maps " + std::to_string(agents.size()) +
                          " intelligent bodies and " + std::to_string(ecologyEntities.size()) +
                          " ecological fields onto backend " + backend +
                          " with " + std::to_string(std::max<std::size_t>(1, options.physicsSubsteps)) + " substeps";
    if (earthMap.value("enabled", false)) {
        summary += "; earth terrain source " + earthMap.value("regionLabel", std::string("global-earth"));
    }
    if (!sceneSummary.empty()) {
        summary += "; scene " + sceneSummary;
    }

    return json{{"enabled", true},
                {"backend", backend.empty() ? std::string("bullet3") : backend},
                {"substeps", std::max<std::size_t>(1, options.physicsSubsteps)},
                {"status", "planned"},
                {"gravity", json::array({0.0, -9.81, 0.0})},
                {"earthMap", earthMap},
                {"rigidBodies", rigidBodies},
                {"constraints", constraints},
                {"summary", truncateText(summary, 220)}};
}

inline json buildWorldMap3D(const VirtualSceneOptions &options,
                            const json &agents,
                            const json &ecologyEntities,
                            const std::vector<std::string> &objectLabels,
                            const std::vector<std::string> &sceneTags,
                            const std::string &sceneSummary,
                            const json &earthMap) {
    if (!options.include3DMap) {
        return json::object();
    }

    const std::size_t width = std::max<std::size_t>(1, options.mapWidth);
    const std::size_t height = std::max<std::size_t>(1, options.mapHeight);
    const std::size_t depth = std::max<std::size_t>(1, options.mapDepth);
    json layers = json::array();
    for (std::size_t z = 0; z < depth; ++z) {
        layers.push_back(json{{"z", z},
                              {"name", z == 0 ? std::string("ground") : std::string("layer-") + std::to_string(z)},
                              {"features", json::array()}});
    }

    auto addFeatureToLayer = [&](const json &feature) {
        int zIndex = 0;
        if (feature.contains("position") && feature["position"].is_object()) {
            zIndex = feature["position"].value("z", 0);
        }
        if (zIndex < 0) {
            zIndex = 0;
        }
        if (static_cast<std::size_t>(zIndex) >= layers.size()) {
            zIndex = static_cast<int>(layers.size() - 1);
        }
        layers[static_cast<std::size_t>(zIndex)]["features"].push_back(feature);
    };

    json landmarks = json::array();
    for (std::size_t index = 0; index < objectLabels.size(); ++index) {
        const auto position = buildSceneVoxelPosition("landmark:" + objectLabels[index], index, options);
        const json landmark{{"id", "landmark-" + std::to_string(index + 1)},
                            {"label", objectLabels[index]},
                            {"kind", "object-anchor"},
                            {"position", position},
                            {"summary", truncateText("stable world anchor for " + objectLabels[index], options.maxEventChars)}};
        landmarks.push_back(landmark);
        addFeatureToLayer(landmark);
    }

    json occupancy = json::array();
    if (agents.is_array()) {
        for (const auto &agent : agents) {
            if (!agent.is_object()) {
                continue;
            }
            const auto position = agent.value("position", json::object());
            occupancy.push_back(json{{"entityId", agent.value("id", std::string())},
                                     {"kind", agent.value("entityType", std::string("intelligent-agent"))},
                                     {"position", position}});
            addFeatureToLayer(json{{"id", agent.value("id", std::string())},
                                   {"label", agent.value("role", std::string("agent")) + std::string(" body")},
                                   {"kind", "intelligent-agent"},
                                   {"position", position},
                                   {"summary", truncateText("embodied agent grounded in the shared world", options.maxEventChars)}});
        }
    }

    if (ecologyEntities.is_array()) {
        for (const auto &entity : ecologyEntities) {
            if (!entity.is_object()) {
                continue;
            }
            const auto position = entity.value("position", json::object());
            occupancy.push_back(json{{"entityId", entity.value("id", std::string())},
                                     {"kind", entity.value("entityType", std::string("non-intelligent-ecosystem"))},
                                     {"position", position}});
            addFeatureToLayer(json{{"id", entity.value("id", std::string())},
                                   {"label", entity.value("biomeLabel", std::string("ecology"))},
                                   {"kind", "ecology-zone"},
                                   {"position", position},
                                   {"summary", entity.value("motionSignature", std::string())}});
        }
    }

    json worldMap{{"dimensions", json{{"width", width}, {"height", height}, {"depth", depth}}},
                  {"coordinateSystem", earthMap.value("enabled", false) ? earthMap.value("coordinateFrame", std::string("wgs84-ecef")) : std::string("voxel-grid")},
                  {"sceneEnvelope", json{{"summary", truncateText(sceneSummary, options.maxEventChars)},
                                          {"tags", toJsonArray(sceneTags)},
                                          {"landmarkCount", landmarks.size()}}},
                  {"layers", layers},
                  {"landmarks", landmarks},
                  {"occupancy", occupancy}};
    if (earthMap.value("enabled", false)) {
        worldMap["earthMap"] = earthMap;
    }
    return worldMap;
}

inline json buildVirtualSceneRollout(const json &sessionState, VirtualSceneOptions options = {}) {
    BrainProfileOptions brainOptions;
    brainOptions.kind = options.brainProfile;
    const auto brainState = buildBrainProfile(sessionState, brainOptions);
    json sceneState = (sessionState.contains("sceneState") && sessionState["sceneState"].is_object()) ? sessionState["sceneState"] : json::object();
    const auto objectLabels = collectObjectLabels(sceneState, std::max<std::size_t>(options.maxAgents, std::size_t(3)));
    const auto sceneTags = collectSceneTags(sceneState, 6);
    const std::string sceneSummary = trimCopy(brainState.value("sceneSummary", std::string()));
    const std::string episodeSummary = trimCopy(brainState.value("episodeSummary", std::string()));
    const std::string baseSummary = !episodeSummary.empty() ? episodeSummary : sceneSummary;
    const auto agenda = buildReasoningAgenda(brainState, 4, options.maxEventChars);
    const auto focus = collectJsonStrings(brainState.value("attentionFocus", json::array()), 4);
    const auto goals = collectJsonStrings(brainState.value("activeGoals", json::array()), 4);
    const json earthMap = buildEarthMapManifestFromOptions(options, baseSummary, sceneTags);
    const json mobilityPlan = buildGroundMobilityPlan(sessionState,
                                                      agenda,
                                                      focus,
                                                      goals,
                                                      4,
                                                      options.maxEventChars,
                                                      options.includeEmbodiedAgents,
                                                      earthMap);

    static const std::vector<std::string> kRoleCycle = {
        "observer", "planner", "critic", "memory", "explorer", "mediator", "scout", "builder"};

    json agents = json::array();
    json entities = json::array();
    const std::size_t agentCount = std::max<std::size_t>(1, options.maxAgents);
    for (std::size_t index = 0; index < agentCount; ++index) {
        const std::string &role = kRoleCycle[index % kRoleCycle.size()];
        const std::string focus = !objectLabels.empty() ? objectLabels[index % objectLabels.size()]
                                                        : (!sceneTags.empty() ? sceneTags[index % sceneTags.size()]
                                                                              : std::string("scene-sector-") + std::to_string(index + 1));
        const auto position = buildSceneVoxelPosition("agent:" + role + ":" + focus, index, options, options.mapDepth > 1 ? index % options.mapDepth : 0, options.include3DMap);
        const auto capabilities = deriveEmbodiedCapabilities(role, objectLabels, sceneTags, baseSummary, 4, options.maxEventChars);
        json agent{{"id", "agent-" + role + "-" + std::to_string(index + 1)},
                   {"role", role},
                   {"focus", focus},
                   {"entityType", "intelligent-agent"},
                   {"position", position},
                   {"body", json{{"form", options.includeEmbodiedAgents ? "embodied-avatar" : "observer-proxy"},
                                  {"presence", options.includeEmbodiedAgents ? "physical-actor" : "simulated-observer"},
                                  {"locomotion", role == "explorer" ? "adaptive-roaming" : "grounded-walking"},
                                  {"positionAnchor", position},
                                  {"capabilities", toJsonArray(capabilities)}}},
                   {"dialogPolicy", json{{"maxTurns", options.maxDialogueTurns},
                                          {"style", role == "critic" ? "verification"
                                                                        : (role == "planner" ? "coordination" : "situational")}}}};
        if (mobilityPlan.is_object() && !mobilityPlan.empty() && agent.contains("body") && agent["body"].is_object()) {
            agent["body"]["mobilityProfile"] = json{{"vehicleClass", mobilityPlan.value("vehicleClass", std::string("all-terrain-four-wheel-rover"))},
                                                       {"environmentConstraint", mobilityPlan.value("environmentConstraint", std::string("ground-only"))},
                                                       {"controlLayer", mobilityPlan.value("controlLayer", std::string("high-level mobility planning"))}};
        }
        agents.push_back(agent);
        entities.push_back(agent);
    }

    json ecologyEntities = buildEcologyVideoAbstractions(sessionState, options);
    for (const auto &entity : ecologyEntities) {
        entities.push_back(entity);
    }

    json timeline = json::array();
    json dialogueTranscript = json::array();
    std::string predictedSummary = baseSummary;
    for (std::size_t step = 0; step < std::max<std::size_t>(1, options.maxSteps); ++step) {
        json events = json::array();
        json dialogues = json::array();
        json environmentalSignals = json::array();
        for (std::size_t index = 0; index < agents.size(); ++index) {
            const auto &agent = agents[index];
            const std::string role = agent.value("role", std::string("observer"));
            const std::string focus = agent.value("focus", std::string("scene"));
            const auto stepPosition = buildSceneVoxelPosition(agent.value("id", std::string()) + ":step:" + std::to_string(step + 1),
                                                             index + step,
                                                             options,
                                                             options.mapDepth > 1 ? (index + step) % options.mapDepth : 0,
                                                             options.include3DMap);
            const std::string cellLabel = describeSceneVoxelPosition(stepPosition);
            std::string event;
            if (role == "observer") {
                event = "observer body moves through " + cellLabel + " and tracks " + focus + " using current scene evidence";
            } else if (role == "planner") {
                event = "planner body repositions at " + cellLabel + " and predicts " + focus + " may change state in step " + std::to_string(step + 1);
                if (options.includeCounterfactual) {
                    event += " under a counterfactual branch";
                }
                if (mobilityPlan.contains("bigBrain") && mobilityPlan["bigBrain"].is_object() &&
                    mobilityPlan["bigBrain"].contains("routePolicy") && mobilityPlan["bigBrain"]["routePolicy"].is_string()) {
                    event += " while following route policy " + mobilityPlan["bigBrain"]["routePolicy"].get<std::string>();
                }
            } else if (role == "critic") {
                event = "critic body checks whether multimodal evidence about " + focus + " stays consistent from " + cellLabel + " at step " + std::to_string(step + 1);
            } else if (role == "memory") {
                event = "memory body consolidates the evolving scene around " + focus + " near " + cellLabel + " at step " + std::to_string(step + 1);
            } else if (role == "explorer") {
                event = "explorer body probes an alternate interaction with " + focus + " from " + cellLabel + " at step " + std::to_string(step + 1);
            } else {
                event = role + " body coordinates around " + focus + " from " + cellLabel + " at step " + std::to_string(step + 1);
            }
            event = truncateText(event + ": " + (!baseSummary.empty() ? baseSummary : std::string("dynamic virtual scene")), options.maxEventChars);
            events.push_back(json{{"agentId", agent.value("id", std::string())},
                                  {"role", role},
                                  {"focus", focus},
                                  {"position", stepPosition},
                                  {"event", event}});
            if (role == "planner") {
                predictedSummary = event;
            }
        }

        if (agents.size() > 1 && options.maxDialogueTurns > 0) {
            const std::size_t dialogueTurns = std::min<std::size_t>(options.maxDialogueTurns, agents.size() - 1);
            for (std::size_t turn = 0; turn < dialogueTurns; ++turn) {
                const std::size_t speakerIndex = (step + turn) % agents.size();
                const std::size_t listenerIndex = (speakerIndex + 1) % agents.size();
                const auto &speaker = agents[speakerIndex];
                const auto &listener = agents[listenerIndex];
                const auto speakerPosition = buildSceneVoxelPosition(speaker.value("id", std::string()) + ":dialogue:" + std::to_string(step + 1),
                                                                    speakerIndex + turn,
                                                                    options,
                                                                    options.mapDepth > 1 ? (speakerIndex + step) % options.mapDepth : 0,
                                                                    options.include3DMap);
                const std::string utterance = buildAgentDialogueUtterance(speaker.value("role", std::string("agent")),
                                                                          listener.value("role", std::string("agent")),
                                                                          speaker.value("focus", std::string("scene")),
                                                                          baseSummary,
                                                                          describeSceneVoxelPosition(speakerPosition),
                                                                          step,
                                                                          turn,
                                                                          options.maxEventChars);
                const json dialogue{{"step", step + 1},
                                    {"turn", turn + 1},
                                    {"speakerId", speaker.value("id", std::string())},
                                    {"speakerRole", speaker.value("role", std::string())},
                                    {"listenerId", listener.value("id", std::string())},
                                    {"listenerRole", listener.value("role", std::string())},
                                    {"utterance", utterance}};
                dialogues.push_back(dialogue);
                dialogueTranscript.push_back(dialogue);
            }
        }

        if (ecologyEntities.is_array()) {
            for (const auto &entity : ecologyEntities) {
                if (!entity.is_object()) {
                    continue;
                }
                environmentalSignals.push_back(json{{"entityId", entity.value("id", std::string())},
                                                    {"biomeLabel", entity.value("biomeLabel", std::string())},
                                                    {"summary", truncateText(entity.value("motionSignature", std::string()) + " near " + describeSceneVoxelPosition(entity.value("position", json::object())),
                                                                             options.maxEventChars)}});
            }
        }

        timeline.push_back(json{{"step", step + 1},
                                {"events", events},
                                {"dialogues", dialogues},
                                {"environmentalSignals", environmentalSignals}});
    }

    const json worldMap3D = buildWorldMap3D(options, agents, ecologyEntities, objectLabels, sceneTags, baseSummary, earthMap);
    const json physicsScene = buildPhysicsScenePlan(options, agents, ecologyEntities, earthMap, baseSummary);

    return json{{"ok", true},
                {"brainProfile", brainProfileKindToString(options.brainProfile)},
                {"agents", agents},
                {"entities", entities},
                {"nonIntelligentEntities", ecologyEntities},
                {"dialogueTranscript", dialogueTranscript},
                {"worldMap3D", worldMap3D},
                {"physicsScene", physicsScene},
                {"mobilityPlan", mobilityPlan},
                {"timeline", timeline},
                {"predictedSceneSummary", truncateText(predictedSummary, options.maxEventChars)},
                {"brainState", brainState},
                {"cognitiveState", brainState}};
}

inline std::vector<GroundedLearningSample> buildVirtualSceneTrainingSamples(const json &sessionState, VirtualSceneOptions options = {}) {
    std::vector<GroundedLearningSample> samples;
    const auto rollout = buildVirtualSceneRollout(sessionState, options);
    const auto cognitivePrompt = buildCognitivePromptContext(rollout.value("brainState", json::object()));

    PromptContextOptions promptOptions;
    promptOptions.maxRecentEvidence = 4;
    promptOptions.maxEvidenceChars = std::min<std::size_t>(options.maxEventChars, 140);
    promptOptions.maxSummaryChars = 220;
    promptOptions.maxObjectSlots = 6;
    std::string graph = buildPromptContext(sessionState, promptOptions);
    if (!cognitivePrompt.empty()) {
        if (!graph.empty()) {
            graph += "\n";
        }
        graph += cognitivePrompt;
    }
    graph = truncateText(graph, 512);

    if (!rollout.contains("timeline") || !rollout["timeline"].is_array()) {
        return samples;
    }

    auto append = [&](std::string input, std::string target, std::string source) {
        if (samples.size() >= options.maxTrainSamples) {
            return;
        }
        input = truncateText(trimCopy(input), 160);
        target = truncateText(trimCopy(target), options.maxEventChars);
        source = lowerCopy(trimCopy(source));
        if (input.empty() || target.empty() || source.empty()) {
            return;
        }
        samples.push_back(GroundedLearningSample{std::move(input), std::move(target), graph, std::move(source)});
    };

    for (const auto &step : rollout["timeline"]) {
        if (!step.is_object() || !step.contains("events") || !step["events"].is_array()) {
            continue;
        }
        for (const auto &event : step["events"]) {
            if (!event.is_object()) {
                continue;
            }
            const std::string role = event.value("role", std::string("observer"));
            const std::string target = event.value("event", std::string());
            if (role == "observer") {
                append("In the virtual scene, what did the observer agent notice?", target, "sim_observer");
            } else if (role == "planner") {
                append("In the virtual scene, what is the next plausible transition?", target, "sim_planner");
            } else if (role == "critic") {
                append("In the virtual scene, how should consistency be checked?", target, "sim_critic");
            } else if (role == "memory") {
                append("In the virtual scene, what should be consolidated into episodic memory?", target, "sim_memory");
            } else {
                append("In the virtual scene, what alternative action should be explored?", target, "sim_explorer");
            }
        }
        if (step.contains("dialogues") && step["dialogues"].is_array()) {
            for (const auto &dialogue : step["dialogues"]) {
                if (!dialogue.is_object()) {
                    continue;
                }
                append("In the embodied virtual scene, what coordination message was exchanged?",
                       dialogue.value("utterance", std::string()),
                       "sim_dialogue");
            }
        }
        if (step.contains("environmentalSignals") && step["environmentalSignals"].is_array()) {
            for (const auto &signal : step["environmentalSignals"]) {
                if (!signal.is_object()) {
                    continue;
                }
                append("What ecological signal did the world model infer from the recent video abstraction?",
                       signal.value("summary", std::string()),
                       "sim_ecology");
            }
        }
    }

    if (rollout.contains("worldMap3D") && rollout["worldMap3D"].is_object()) {
        std::vector<std::string> occupancyEntities;
        if (rollout["worldMap3D"].contains("occupancy") && rollout["worldMap3D"]["occupancy"].is_array()) {
            for (const auto &entry : rollout["worldMap3D"]["occupancy"]) {
                if (!entry.is_object()) {
                    continue;
                }
                const std::string entityId = trimCopy(entry.value("entityId", std::string()));
                if (!entityId.empty()) {
                    occupancyEntities.push_back(entityId);
                }
                if (occupancyEntities.size() >= 4) {
                    break;
                }
            }
        }
        if (!occupancyEntities.empty()) {
            append("Which entities are grounded into the simulated 3D world map?",
                   joinStrings(occupancyEntities, ", "),
                   "sim_map");
        }
    }

    if (rollout.contains("physicsScene") && rollout["physicsScene"].is_object()) {
        const auto &physicsScene = rollout["physicsScene"];
        if (physicsScene.value("enabled", false) && physicsScene.contains("summary") && physicsScene["summary"].is_string()) {
            append("How should the physical simulation world be configured for embodied training?",
                   physicsScene["summary"].get<std::string>(),
                   "sim_physics");
        }
        if (physicsScene.contains("earthMap") && physicsScene["earthMap"].is_object() &&
            physicsScene["earthMap"].value("enabled", false) &&
            physicsScene["earthMap"].contains("summary") && physicsScene["earthMap"]["summary"].is_string()) {
            append("Which earth-scale terrain source anchors the simulation?",
                   physicsScene["earthMap"]["summary"].get<std::string>(),
                   "sim_earth_map");
        }
    }

    if (rollout.contains("mobilityPlan") && rollout["mobilityPlan"].is_object()) {
        const auto &mobilityPlan = rollout["mobilityPlan"];
        const std::string strategicGoal = mobilityPlan.value("strategicGoal", std::string());
        std::string mobilitySummary = strategicGoal;
        if (mobilityPlan.contains("littleBrain") && mobilityPlan["littleBrain"].is_object()) {
            const auto speedPolicy = mobilityPlan["littleBrain"].value("speedPolicy", std::string());
            if (!speedPolicy.empty()) {
                mobilitySummary += mobilitySummary.empty() ? speedPolicy : std::string(" | ") + speedPolicy;
            }
        }
        if (!mobilitySummary.empty()) {
            append("How should the ground rover coordinate strategic route planning with local stabilization?",
                   mobilitySummary,
                   "sim_rover_mobility");
        }
    }

    if (options.includeSceneRecallSample && rollout.contains("predictedSceneSummary") && rollout["predictedSceneSummary"].is_string()) {
        append("Summarize the simulated scene rollout.", rollout["predictedSceneSummary"].get<std::string>(), "sim_scene");
    }

    if (samples.size() > options.maxTrainSamples) {
        samples.resize(options.maxTrainSamples);
    }
    return samples;
}

inline json simulateVirtualScene(const json &sessionState, VirtualSceneOptions options = {}) {
    auto rollout = buildVirtualSceneRollout(sessionState, options);
    json trainSamples = json::array();
    for (const auto &sample : buildVirtualSceneTrainingSamples(sessionState, options)) {
        trainSamples.push_back(json{{"input", sample.input},
                                    {"target", sample.target},
                                    {"graph", sample.graph},
                                    {"source", sample.source}});
    }
    rollout["trainSamples"] = trainSamples;
    return rollout;
}

class JsonStore {
public:
    explicit JsonStore(std::shared_ptr<KeyValueStore> store) : store_(std::move(store)) {}

    std::optional<json> get(const std::string &key) const {
        if (store_) {
            return store_->get(key);
        }
        std::lock_guard<std::mutex> lock(mu_);
        auto it = memory_.find(key);
        if (it == memory_.end()) {
            return std::nullopt;
        }
        return std::make_optional<json>(it->second);
    }

    void put(const std::string &key, const json &value) {
        if (store_) {
            store_->put(key, value);
            return;
        }
        std::lock_guard<std::mutex> lock(mu_);
        memory_[key] = value;
    }

    void del(const std::string &key) {
        if (store_) {
            store_->del(key);
            return;
        }
        std::lock_guard<std::mutex> lock(mu_);
        memory_.erase(key);
    }

    std::vector<std::pair<std::string, json>> entries(const std::string &prefix) const {
        if (store_) {
            return store_->entries(prefix);
        }
        std::vector<std::pair<std::string, json>> out;
        std::lock_guard<std::mutex> lock(mu_);
        for (const auto &entry : memory_) {
            if (entry.first.rfind(prefix, 0) == 0) {
                out.push_back(entry);
            }
        }
        std::sort(out.begin(), out.end(), [](const auto &lhs, const auto &rhs) {
            return lhs.first < rhs.first;
        });
        return out;
    }

    bool persistent() const {
        return static_cast<bool>(store_);
    }

private:
    std::shared_ptr<KeyValueStore> store_;
    mutable std::mutex mu_;
    std::unordered_map<std::string, json> memory_;
};

class WorldModelStore {
public:
    struct Options {
        std::size_t maxRecentEvidence{24};
        std::size_t maxRecentSceneEvidence{12};
        std::size_t maxObjectSlots{16};
        std::size_t maxTags{16};
        std::size_t maxGraphLines{8};
    };

    WorldModelStore(std::shared_ptr<KeyValueStore> kvmStore,
                    std::shared_ptr<KeyValueStore> memeStore,
                    std::shared_ptr<KeyValueStore> sessionStore)
        : WorldModelStore(std::move(kvmStore), std::move(memeStore), std::move(sessionStore), Options{}) {}

    WorldModelStore(std::shared_ptr<KeyValueStore> kvmStore,
                    std::shared_ptr<KeyValueStore> memeStore,
                    std::shared_ptr<KeyValueStore> sessionStore,
                    Options options)
        : kvm_(std::move(kvmStore)),
          meme_(std::move(memeStore)),
          session_(std::move(sessionStore)),
          options_(options) {}

    json ingestEvidence(json evidence) {
        const std::int64_t createdAt = evidence.value("createdAt", nowMs());
        std::string sessionId = trimCopy(evidence.value("sessionId", std::string()));
        if (sessionId.empty()) {
            sessionId = generateSessionId(createdAt, evidence);
        }

        std::string modality = lowerCopy(trimCopy(evidence.value("modality", std::string("text"))));
        if (modality.empty()) {
            modality = "text";
        }

        std::string text = trimCopy(evidence.value("text", std::string()));
        std::string graphSummary = trimCopy(evidence.value("graphSummary", std::string()));
        std::string rawLocation = trimCopy(evidence.value("rawLocation", std::string()));
        json metadata = (evidence.contains("metadata") && evidence["metadata"].is_object()) ? evidence["metadata"] : json::object();

        if (graphSummary.empty() && metadata.contains("graphContext") && metadata["graphContext"].is_string()) {
            graphSummary = trimCopy(metadata["graphContext"].get<std::string>());
        }
        if (graphSummary.empty() && !text.empty()) {
            graphSummary = truncateText(text, 160);
        }

        if (!evidence.contains("tags") || !evidence["tags"].is_array()) {
            evidence["tags"] = json::array();
            for (const auto &token : extractKeywords(text + " " + graphSummary, options_.maxTags)) {
                evidence["tags"].push_back(token);
            }
        }

        if (!evidence.contains("objectSlots") || !evidence["objectSlots"].is_array()) {
            evidence["objectSlots"] = buildObjectSlots(metadata, options_.maxObjectSlots);
        }

        std::string evidenceId = trimCopy(evidence.value("id", std::string()));
        if (evidenceId.empty()) {
            evidenceId = generateEvidenceId(sessionId, modality, text, graphSummary, rawLocation, createdAt);
        }

        evidence["id"] = evidenceId;
        evidence["sessionId"] = sessionId;
        evidence["modality"] = modality;
        evidence["text"] = text;
        evidence["graphSummary"] = graphSummary;
        evidence["rawLocation"] = rawLocation;
        evidence["metadata"] = metadata;
        evidence["createdAt"] = createdAt;
        evidence["embeddingSize"] = inferEmbeddingSize(evidence, metadata);

        kvm_.put(evidenceKey(sessionId, evidenceId), evidence);

        auto sessionDoc = session_.get(sessionKey(sessionId)).value_or(defaultSessionDoc(sessionId));
        auto sceneDoc = meme_.get(sceneKey(sessionId)).value_or(defaultSceneDoc(sessionId));
        auto episodeDoc = session_.get(episodeKey(sessionId)).value_or(defaultEpisodeDoc(sessionId));
        auto predictionDoc = session_.get(predictionKey(sessionId)).value_or(defaultPredictionDoc(sessionId));

        updateSessionDoc(sessionDoc, evidence);
        updateSceneDoc(sceneDoc, evidence);
        updateEpisodeDoc(episodeDoc, evidence, sceneDoc);
        const auto recentEvidence = loadRecentEvidence(sessionId, sessionDoc, options_.maxRecentEvidence);
        updatePredictionDoc(predictionDoc, sessionDoc, sceneDoc, episodeDoc, evidence, recentEvidence);

        session_.put(sessionKey(sessionId), sessionDoc);
        meme_.put(sceneKey(sessionId), sceneDoc);
        session_.put(episodeKey(sessionId), episodeDoc);
        session_.put(predictionKey(sessionId), predictionDoc);

        {
            std::lock_guard<std::mutex> lock(mu_);
            ++ingestCount_;
            sessionEvidenceCounts_[sessionId] = static_cast<std::size_t>(sessionDoc.value("evidenceCount", 0));
        }

        return json{{"ok", true},
                    {"evidence", evidence},
                    {"sceneState", sceneDoc},
                    {"episode", episodeDoc},
                    {"prediction", predictionDoc},
                    {"storage", storageDescriptor()}};
    }

    json sessionState(const std::string &sessionId, std::size_t limit = 8) const {
        const auto trimmed = trimCopy(sessionId);
        auto sessionDoc = session_.get(sessionKey(trimmed)).value_or(defaultSessionDoc(trimmed));
        auto sceneDoc = meme_.get(sceneKey(trimmed)).value_or(defaultSceneDoc(trimmed));
        auto episodeDoc = session_.get(episodeKey(trimmed)).value_or(defaultEpisodeDoc(trimmed));
        auto predictionDoc = session_.get(predictionKey(trimmed)).value_or(defaultPredictionDoc(trimmed));

        json recentEvidence = loadRecentEvidence(trimmed, sessionDoc, limit);

        return json{{"ok", true},
                    {"sessionId", trimmed},
                    {"session", sessionDoc},
                    {"sceneState", sceneDoc},
                    {"episode", episodeDoc},
                    {"prediction", predictionDoc},
                    {"recentEvidence", recentEvidence},
                    {"storage", storageDescriptor()}};
    }

    bool resetSession(const std::string &sessionId) {
        const auto trimmed = trimCopy(sessionId);
        auto sessionDoc = session_.get(sessionKey(trimmed));
        if (sessionDoc && sessionDoc->contains("recentEvidenceIds") && (*sessionDoc)["recentEvidenceIds"].is_array()) {
            for (const auto &entry : (*sessionDoc)["recentEvidenceIds"]) {
                if (entry.is_string()) {
                    kvm_.del(evidenceKey(trimmed, entry.get<std::string>()));
                }
            }
        }
        session_.del(sessionKey(trimmed));
        session_.del(episodeKey(trimmed));
        session_.del(predictionKey(trimmed));
        meme_.del(sceneKey(trimmed));
        std::lock_guard<std::mutex> lock(mu_);
        return sessionEvidenceCounts_.erase(trimmed) > 0 || static_cast<bool>(sessionDoc);
    }

    json status() const {
        std::lock_guard<std::mutex> lock(mu_);
        json sessions = json::object();
        for (const auto &entry : sessionEvidenceCounts_) {
            sessions[entry.first] = entry.second;
        }
        return json{{"ok", true},
                    {"ingestCount", ingestCount_},
                    {"trackedSessions", sessionEvidenceCounts_.size()},
                    {"sessionEvidenceCounts", sessions},
                    {"storage", storageDescriptor()}};
    }

private:
    JsonStore kvm_;
    JsonStore meme_;
    JsonStore session_;
    Options options_;
    mutable std::mutex mu_;
    std::uint64_t ingestCount_{0};
    std::unordered_map<std::string, std::size_t> sessionEvidenceCounts_;

    json storageDescriptor() const {
        return json{{"kvm", kvm_.persistent() ? "database" : "memory"},
                    {"meme", meme_.persistent() ? "database" : "memory"},
                    {"session", session_.persistent() ? "database" : "memory"}};
    }

    static std::string generateSessionId(std::int64_t createdAt, const json &evidence) {
        const auto basis = evidence.dump();
        std::ostringstream oss;
        oss << "wm_" << createdAt << "_" << std::hex << stableHash64(basis);
        return oss.str();
    }

    static std::string generateEvidenceId(const std::string &sessionId,
                                          const std::string &modality,
                                          const std::string &text,
                                          const std::string &graphSummary,
                                          const std::string &rawLocation,
                                          std::int64_t createdAt) {
        std::ostringstream oss;
        oss << "ev_" << createdAt << "_" << std::hex
            << stableHash64(sessionId + "|" + modality + "|" + text + "|" + graphSummary + "|" + rawLocation);
        return oss.str();
    }

    static std::string evidenceKey(const std::string &sessionId, const std::string &evidenceId) {
        return "world:evidence:" + sessionId + ":" + evidenceId;
    }

    static std::string sessionKey(const std::string &sessionId) {
        return "world:session:" + sessionId;
    }

    static std::string sceneKey(const std::string &sessionId) {
        return "world:scene:" + sessionId;
    }

    static std::string episodeKey(const std::string &sessionId) {
        return "world:episode:" + sessionId + ":current";
    }

    static std::string predictionKey(const std::string &sessionId) {
        return "world:prediction:" + sessionId + ":current";
    }

    static std::size_t inferEmbeddingSize(const json &evidence, const json &metadata) {
        if (evidence.contains("embedding") && evidence["embedding"].is_array()) {
            return evidence["embedding"].size();
        }
        if (metadata.contains("embedding") && metadata["embedding"].is_array()) {
            return metadata["embedding"].size();
        }
        if (metadata.contains("imageEmbedding") && metadata["imageEmbedding"].is_array()) {
            return metadata["imageEmbedding"].size();
        }
        return 0;
    }

    static json defaultSessionDoc(const std::string &sessionId) {
        return json{{"sessionId", sessionId},
                    {"evidenceCount", 0},
                    {"modalities", json::object()},
                    {"recentEvidenceIds", json::array()},
                    {"recentTags", json::array()},
                    {"predictionCalibration", json::object()},
                    {"lastUpdatedAt", 0}};
    }

    static json defaultSceneDoc(const std::string &sessionId) {
        return json{{"sessionId", sessionId},
                    {"evidenceCount", 0},
                    {"modalities", json::array()},
                    {"objectSlots", json::array()},
                    {"relations", json::array()},
                    {"recentEvidenceIds", json::array()},
                    {"tags", json::array()},
                    {"graphContext", json::array()},
                    {"summary", ""},
                    {"lastUpdatedAt", 0}};
    }

    static json defaultEpisodeDoc(const std::string &sessionId) {
        return json{{"sessionId", sessionId},
                    {"episodeId", "ep_current"},
                    {"evidenceIds", json::array()},
                    {"turnCount", 0},
                    {"summary", ""},
                  {"expectedNextState", json::object()},
                  {"observedNextState", json::object()},
                  {"predictionCalibration", json::object()},
                    {"createdAt", nowMs()},
                    {"updatedAt", 0}};
    }

        static json defaultPredictionDoc(const std::string &sessionId) {
          return json{{"sessionId", sessionId},
                  {"version", 1},
                  {"entities", json::array()},
                  {"relations", json::array()},
                  {"goals", json::array()},
                  {"hypotheses", json::array()},
                  {"contradictions", json::array()},
                  {"expectedNextState",
                   json{{"summary", ""},
                      {"entities", json::array()},
                      {"relations", json::array()},
                      {"goals", json::array()},
                      {"hypotheses", json::array()},
                      {"confidence", 0.0},
                      {"horizon", "next-observation"},
                      {"sourceEvidenceIds", json::array()},
                      {"createdAt", 0}}},
                  {"observedNextState",
                   json{{"summary", ""},
                      {"entities", json::array()},
                      {"relations", json::array()},
                      {"goals", json::array()},
                      {"hypotheses", json::array()},
                      {"contradictions", json::array()},
                      {"alignmentScore", 0.0},
                      {"matchedExpectedState", false},
                      {"sourceEvidenceId", ""},
                      {"observedAt", 0}}},
                  {"calibration",
                   json{{"samples", 0},
                      {"matched", 0},
                      {"mismatched", 0},
                      {"avgAlignmentScore", 0.0},
                      {"lastAlignmentScore", 0.0},
                      {"lastReason", ""},
                      {"corrections", json::array()},
                      {"lastUpdatedAt", 0}}},
                  {"lastUpdatedAt", 0}};
        }

    static bool jsonArrayContainsString(const json &arr, const std::string &value) {
        if (!arr.is_array()) {
            return false;
        }
        for (const auto &entry : arr) {
            if (entry.is_string() && entry.get<std::string>() == value) {
                return true;
            }
        }
        return false;
    }

    void appendRecentId(json &arr, const std::string &id, std::size_t maxCount) const {
        if (!arr.is_array()) {
            arr = json::array();
        }
        if (!jsonArrayContainsString(arr, id)) {
            arr.push_back(id);
        }
        while (arr.size() > maxCount) {
            arr.erase(arr.begin());
        }
    }

    void mergeTags(json &target, const json &evidence) const {
        if (!target.is_array()) {
            target = json::array();
        }
        std::unordered_set<std::string> seen;
        for (const auto &entry : target) {
            if (entry.is_string()) {
                seen.insert(entry.get<std::string>());
            }
        }
        if (evidence.contains("tags") && evidence["tags"].is_array()) {
            for (const auto &entry : evidence["tags"]) {
                if (!entry.is_string()) {
                    continue;
                }
                const auto token = entry.get<std::string>();
                if (token.empty() || !seen.insert(token).second) {
                    continue;
                }
                target.push_back(token);
                if (target.size() >= options_.maxTags) {
                    break;
                }
            }
        }
        while (target.size() > options_.maxTags) {
            target.erase(target.begin());
        }
    }

    static json buildObjectSlots(const json &metadata, std::size_t maxCount) {
        json out = json::array();
        std::unordered_set<std::string> seenLabels;
        if (metadata.contains("detections") && metadata["detections"].is_array()) {
            for (const auto &entry : metadata["detections"]) {
                if (!entry.is_object()) {
                    continue;
                }
                std::string label;
                if (entry.contains("label") && entry["label"].is_string()) {
                    label = entry["label"].get<std::string>();
                } else if (entry.contains("className") && entry["className"].is_string()) {
                    label = entry["className"].get<std::string>();
                } else if (entry.contains("name") && entry["name"].is_string()) {
                    label = entry["name"].get<std::string>();
                }
                label = trimCopy(label);
                if (label.empty() || !seenLabels.insert(lowerCopy(label)).second) {
                    continue;
                }
                json slot{{"label", label}, {"source", "vision"}};
                if (entry.contains("confidence")) {
                    slot["confidence"] = entry["confidence"];
                }
                if (entry.contains("bbox")) {
                    slot["bbox"] = entry["bbox"];
                }
                out.push_back(slot);
                if (out.size() >= maxCount) {
                    return out;
                }
            }
        }
        if (out.empty() && metadata.contains("details") && metadata["details"].is_array()) {
            for (const auto &entry : metadata["details"]) {
                if (!entry.is_string()) {
                    continue;
                }
                std::string label = trimCopy(entry.get<std::string>());
                if (label.empty() || !seenLabels.insert(lowerCopy(label)).second) {
                    continue;
                }
                out.push_back(json{{"label", truncateText(label, 80)}, {"source", "detail"}});
                if (out.size() >= maxCount) {
                    return out;
                }
            }
        }
        return out;
    }

    void mergeObjectSlots(json &target, const json &evidence) const {
        if (!target.is_array()) {
            target = json::array();
        }
        std::unordered_set<std::string> seen;
        for (const auto &entry : target) {
            if (entry.is_object() && entry.contains("label") && entry["label"].is_string()) {
                seen.insert(lowerCopy(entry["label"].get<std::string>()));
            }
        }
        if (!evidence.contains("objectSlots") || !evidence["objectSlots"].is_array()) {
            return;
        }
        for (const auto &entry : evidence["objectSlots"]) {
            if (!entry.is_object() || !entry.contains("label") || !entry["label"].is_string()) {
                continue;
            }
            const auto key = lowerCopy(entry["label"].get<std::string>());
            if (!seen.insert(key).second) {
                continue;
            }
            target.push_back(entry);
            if (target.size() >= options_.maxObjectSlots) {
                break;
            }
        }
        while (target.size() > options_.maxObjectSlots) {
            target.erase(target.begin());
        }
    }

    void mergeGraphContext(json &target, const std::string &graphSummary) const {
        if (!target.is_array()) {
            target = json::array();
        }
        std::unordered_set<std::string> seen;
        for (const auto &entry : target) {
            if (entry.is_string()) {
                seen.insert(entry.get<std::string>());
            }
        }
        std::istringstream iss(graphSummary);
        std::string line;
        while (std::getline(iss, line)) {
            line = trimCopy(line);
            if (line.empty() || !seen.insert(line).second) {
                continue;
            }
            target.push_back(line);
            if (target.size() >= options_.maxGraphLines) {
                break;
            }
        }
        while (target.size() > options_.maxGraphLines) {
            target.erase(target.begin());
        }
    }

    static std::string joinGraphContext(const json &graphContext) {
        if (!graphContext.is_array()) {
            return std::string();
        }
        std::ostringstream oss;
        bool first = true;
        for (const auto &entry : graphContext) {
            if (!entry.is_string()) {
                continue;
            }
            if (!first) {
                oss << '\n';
            }
            first = false;
            oss << entry.get<std::string>();
        }
        return oss.str();
    }

    json loadRecentEvidence(const std::string &sessionId, const json &sessionDoc, std::size_t limit) const {
        json recentEvidence = json::array();
        if (!sessionDoc.contains("recentEvidenceIds") || !sessionDoc["recentEvidenceIds"].is_array()) {
            return recentEvidence;
        }

        auto ids = sessionDoc["recentEvidenceIds"];
        const std::size_t size = ids.size();
        const std::size_t start = size > limit ? size - limit : 0;
        for (std::size_t i = start; i < size; ++i) {
            if (!ids[i].is_string()) {
                continue;
            }
            auto evidence = kvm_.get(evidenceKey(sessionId, ids[i].get<std::string>()));
            if (evidence) {
                recentEvidence.push_back(*evidence);
            }
        }
        return recentEvidence;
    }

    json evidenceField(const json &evidence, const char *key) const {
        if (key == nullptr || *key == '\0') {
            return json();
        }
        if (evidence.contains(key)) {
            return evidence[key];
        }
        if (evidence.contains("metadata") && evidence["metadata"].is_object() && evidence["metadata"].contains(key)) {
            return evidence["metadata"][key];
        }
        return json();
    }

    static std::string bestStringValue(const json &value) {
        if (value.is_string()) {
            return truncateText(trimCopy(value.get<std::string>()), 120);
        }
        if (!value.is_object()) {
            return std::string();
        }
        for (const auto *key : {"label", "name", "title", "summary", "text", "object", "subject", "predicate"}) {
            if (value.contains(key) && value[key].is_string()) {
                return truncateText(trimCopy(value[key].get<std::string>()), 120);
            }
        }
        return std::string();
    }

    static std::vector<std::string> collectFlexibleStrings(const json &value, std::size_t maxCount) {
        std::vector<std::string> out;
        std::unordered_set<std::string> seen;

        auto appendItem = [&](const std::string &text) {
            appendUniqueLimited(out, seen, text, maxCount, 120);
        };

        if (value.is_null()) {
            return out;
        }
        if (value.is_string() || value.is_object()) {
            const auto text = bestStringValue(value);
            if (!text.empty()) {
                appendItem(text);
            }
            return out;
        }
        if (!value.is_array()) {
            return out;
        }

        for (const auto &entry : value) {
            const auto text = bestStringValue(entry);
            if (text.empty()) {
                continue;
            }
            appendItem(text);
            if (out.size() >= maxCount) {
                break;
            }
        }
        return out;
    }

    static std::string relationSignature(const json &relation) {
        if (relation.is_string()) {
            return lowerCopy(trimCopy(relation.get<std::string>()));
        }
        if (!relation.is_object()) {
            return std::string();
        }
        const auto subject = lowerCopy(trimCopy(relation.value("subject", std::string())));
        const auto predicate = lowerCopy(trimCopy(relation.value("predicate", std::string())));
        const auto object = lowerCopy(trimCopy(relation.value("object", std::string())));
        if (!subject.empty() || !predicate.empty() || !object.empty()) {
            return subject + "|" + predicate + "|" + object;
        }
        return lowerCopy(trimCopy(relation.value("summary", std::string())));
    }

    json normalizeRelationEntry(const json &value) const {
        if (value.is_string()) {
            const auto summary = truncateText(trimCopy(value.get<std::string>()), 120);
            if (summary.empty()) {
                return json();
            }
            return json{{"subject", "world"}, {"predicate", "described_as"}, {"object", summary}};
        }
        if (!value.is_object()) {
            return json();
        }

        std::string subject = trimCopy(value.value("subject", value.value("source", std::string())));
        std::string predicate = trimCopy(value.value("predicate", value.value("relation", std::string())));
        std::string object = trimCopy(value.value("object", value.value("target", std::string())));
        const std::string summary = trimCopy(value.value("summary", std::string()));
        if (subject.empty() && predicate.empty() && object.empty() && summary.empty()) {
            return json();
        }
        if (subject.empty()) {
            subject = "world";
        }
        if (predicate.empty()) {
            predicate = "related_to";
        }
        if (object.empty()) {
            object = summary.empty() ? std::string("context") : summary;
        }

        json relation{{"subject", truncateText(subject, 80)},
                      {"predicate", truncateText(predicate, 80)},
                      {"object", truncateText(object, 120)}};
        if (value.contains("confidence") && value["confidence"].is_number()) {
            relation["confidence"] = value["confidence"];
        }
        return relation;
    }

    json normalizeRelations(const json &value) const {
        json out = json::array();
        std::unordered_set<std::string> seen;
        const std::size_t maxCount = std::max<std::size_t>(std::size_t(6), options_.maxObjectSlots * 3);

        auto appendRelation = [&](const json &entry) {
            const auto normalized = normalizeRelationEntry(entry);
            if (!normalized.is_object()) {
                return;
            }
            const auto signature = relationSignature(normalized);
            if (signature.empty() || !seen.insert(signature).second) {
                return;
            }
            out.push_back(normalized);
        };

        if (value.is_object() || value.is_string()) {
            appendRelation(value);
        } else if (value.is_array()) {
            for (const auto &entry : value) {
                appendRelation(entry);
                if (out.size() >= maxCount) {
                    break;
                }
            }
        }

        return out;
    }

    void mergeRelationArray(json &target, const json &incoming, std::size_t maxCount) const {
        if (!target.is_array()) {
            target = json::array();
        }

        std::unordered_set<std::string> seen;
        for (const auto &entry : target) {
            const auto signature = relationSignature(entry);
            if (!signature.empty()) {
                seen.insert(signature);
            }
        }

        auto appendRelation = [&](const json &entry) {
            const auto normalized = normalizeRelationEntry(entry);
            if (!normalized.is_object()) {
                return;
            }
            const auto signature = relationSignature(normalized);
            if (signature.empty() || !seen.insert(signature).second) {
                return;
            }
            target.push_back(normalized);
        };

        if (incoming.is_object() || incoming.is_string()) {
            appendRelation(incoming);
        } else if (incoming.is_array()) {
            for (const auto &entry : incoming) {
                appendRelation(entry);
                if (target.size() >= maxCount) {
                    break;
                }
            }
        }

        while (target.size() > maxCount) {
            target.erase(target.begin());
        }
    }

    json deriveRelations(const json &evidence) const {
        json out = json::array();
        const auto labels = collectObjectLabels(evidence, std::min<std::size_t>(options_.maxObjectSlots, 4));
        const auto modality = trimCopy(evidence.value("modality", std::string("text")));
        const std::size_t maxCount = std::max<std::size_t>(std::size_t(6), options_.maxObjectSlots * 3);

        if (!modality.empty()) {
            mergeRelationArray(out, json{{"subject", "world"}, {"predicate", "observed_via"}, {"object", modality}, {"confidence", 1.0}}, maxCount);
        }
        for (const auto &label : labels) {
            mergeRelationArray(out,
                               json{{"subject", label},
                                    {"predicate", "observed_in"},
                                    {"object", modality.empty() ? std::string("evidence") : modality},
                                    {"confidence", 1.0}},
                               maxCount);
        }
        for (std::size_t i = 0; i < labels.size(); ++i) {
            for (std::size_t j = i + 1; j < labels.size(); ++j) {
                mergeRelationArray(out,
                                   json{{"subject", labels[i]},
                                        {"predicate", "co_observed_with"},
                                        {"object", labels[j]},
                                        {"confidence", 0.6}},
                                   maxCount);
            }
        }

        return out;
    }

    json recentEvidenceIds(const json &sessionDoc, std::size_t maxCount) const {
        json out = json::array();
        if (!sessionDoc.contains("recentEvidenceIds") || !sessionDoc["recentEvidenceIds"].is_array()) {
            return out;
        }
        const auto &ids = sessionDoc["recentEvidenceIds"];
        const std::size_t size = ids.size();
        const std::size_t start = size > maxCount ? size - maxCount : 0;
        for (std::size_t i = start; i < size; ++i) {
            if (ids[i].is_string()) {
                out.push_back(ids[i]);
            }
        }
        return out;
    }

    static bool stateHasSignal(const json &state) {
        return state.is_object() &&
               (!trimCopy(state.value("summary", std::string())).empty() ||
                (state.contains("entities") && state["entities"].is_array() && !state["entities"].empty()) ||
                (state.contains("relations") && state["relations"].is_array() && !state["relations"].empty()));
    }

    json normalizeStatePayload(const json &value, const json &fallback) const {
        json out = fallback.is_object() ? fallback : json::object();
        if (value.is_null()) {
            return out;
        }
        if (value.is_string()) {
            const auto summary = truncateText(trimCopy(value.get<std::string>()), 180);
            if (!summary.empty()) {
                out["summary"] = summary;
            }
            return out;
        }
        if (!value.is_object()) {
            return out;
        }

        const auto summary = truncateText(trimCopy(value.value("summary", value.value("text", std::string()))), 180);
        if (!summary.empty()) {
            out["summary"] = summary;
        }

        const auto entities = collectFlexibleStrings(value.contains("entities") ? value["entities"] : json(), options_.maxObjectSlots);
        if (!entities.empty()) {
            out["entities"] = toJsonArray(entities);
        }

        const auto relations = normalizeRelations(value.contains("relations") ? value["relations"] : json());
        if (!relations.empty()) {
            out["relations"] = relations;
        }

        const auto goals = collectFlexibleStrings(value.contains("goals") ? value["goals"] : json(), 4);
        if (!goals.empty()) {
            out["goals"] = toJsonArray(goals);
        }
        const auto hypotheses = collectFlexibleStrings(value.contains("hypotheses") ? value["hypotheses"] : json(), 4);
        if (!hypotheses.empty()) {
            out["hypotheses"] = toJsonArray(hypotheses);
        }
        const auto contradictions = collectFlexibleStrings(value.contains("contradictions") ? value["contradictions"] : json(), 4);
        if (!contradictions.empty()) {
            out["contradictions"] = toJsonArray(contradictions);
        }

        for (const auto *key : {"horizon", "sourceEvidenceId"}) {
            if (value.contains(key) && value[key].is_string()) {
                out[key] = truncateText(trimCopy(value[key].get<std::string>()), 120);
            }
        }
        for (const auto *key : {"createdAt", "observedAt"}) {
            if (value.contains(key) && value[key].is_number_integer()) {
                out[key] = value[key];
            }
        }
        for (const auto *key : {"confidence", "alignmentScore"}) {
            if (value.contains(key) && value[key].is_number()) {
                out[key] = clampUnitValue(value[key].get<double>());
            }
        }
        if (value.contains("matchedExpectedState") && value["matchedExpectedState"].is_boolean()) {
            out["matchedExpectedState"] = value["matchedExpectedState"];
        }
        if (value.contains("sourceEvidenceIds") && value["sourceEvidenceIds"].is_array()) {
            out["sourceEvidenceIds"] = json::array();
            for (const auto &entry : value["sourceEvidenceIds"]) {
                if (entry.is_string()) {
                    out["sourceEvidenceIds"].push_back(entry);
                }
            }
        }

        return out;
    }

    static std::string summarizeObservation(const json &evidence, const json &sceneDoc, const json &episodeDoc) {
        std::string summary = trimCopy(evidence.value("graphSummary", std::string()));
        if (summary.empty()) {
            summary = trimCopy(evidence.value("text", std::string()));
        }
        if (summary.empty()) {
            summary = trimCopy(sceneDoc.value("summary", episodeDoc.value("summary", std::string())));
        }
        return truncateText(summary, 180);
    }

    json buildObservedNextState(const json &evidence, const json &sceneDoc, const json &episodeDoc) const {
        auto entities = collectFlexibleStrings(evidenceField(evidence, "entities"), options_.maxObjectSlots);
        if (entities.empty()) {
            entities = collectObjectLabels(evidence, std::min<std::size_t>(options_.maxObjectSlots, 5));
        }
        if (entities.empty()) {
            entities = collectObjectLabels(sceneDoc, std::min<std::size_t>(options_.maxObjectSlots, 5));
        }

        auto goals = collectFlexibleStrings(evidenceField(evidence, "goals"), 4);
        if (goals.empty()) {
            goals = collectFlexibleStrings(evidenceField(evidence, "goal"), 1);
        }
        const auto hypotheses = collectFlexibleStrings(evidenceField(evidence, "hypotheses"), 4);
        const auto contradictions = collectFlexibleStrings(evidenceField(evidence, "contradictions"), 4);

        json relations = normalizeRelations(evidenceField(evidence, "relations"));
        if (relations.empty()) {
            relations = deriveRelations(evidence);
        }

        json fallback{{"summary", summarizeObservation(evidence, sceneDoc, episodeDoc)},
                      {"entities", toJsonArray(entities)},
                      {"relations", relations},
                      {"goals", toJsonArray(goals)},
                      {"hypotheses", toJsonArray(hypotheses)},
                      {"contradictions", toJsonArray(contradictions)},
                      {"sourceEvidenceId", evidence.value("id", std::string())},
                      {"observedAt", evidence.value("createdAt", nowMs())}};
        return normalizeStatePayload(evidenceField(evidence, "observedNextState"), fallback);
    }

    json buildExpectedNextState(const json &sessionDoc,
                                const json &sceneDoc,
                                const json &episodeDoc,
                                const json &evidence,
                                const json &cognitiveState,
                                const json &agenda) const {
        auto entities = collectFlexibleStrings(evidenceField(evidence, "entities"), options_.maxObjectSlots);
        if (entities.empty()) {
            entities = collectObjectLabels(sceneDoc, std::min<std::size_t>(options_.maxObjectSlots, 5));
        }

        json relations = normalizeRelations(evidenceField(evidence, "relations"));
        if (relations.empty() && sceneDoc.contains("relations") && sceneDoc["relations"].is_array()) {
            relations = sceneDoc["relations"];
        }

        std::vector<std::string> goals = collectJsonStrings(cognitiveState.value("activeGoals", json::array()), 4);
        std::vector<std::string> explicitGoals = collectFlexibleStrings(evidenceField(evidence, "goals"), 4);
        if (explicitGoals.empty()) {
            explicitGoals = collectFlexibleStrings(evidenceField(evidence, "goal"), 1);
        }
        std::unordered_set<std::string> goalSeen;
        std::vector<std::string> mergedGoals;
        for (const auto &item : goals) {
            appendUniqueLimited(mergedGoals, goalSeen, item, 4, 120);
        }
        for (const auto &item : explicitGoals) {
            appendUniqueLimited(mergedGoals, goalSeen, item, 4, 120);
        }

        std::vector<std::string> hypotheses = collectJsonStrings(agenda.value("hypotheses", json::array()), 4);
        const auto explicitHypotheses = collectFlexibleStrings(evidenceField(evidence, "hypotheses"), 4);
        std::unordered_set<std::string> hypothesisSeen;
        std::vector<std::string> mergedHypotheses;
        for (const auto &item : hypotheses) {
            appendUniqueLimited(mergedHypotheses, hypothesisSeen, item, 4, 120);
        }
        for (const auto &item : explicitHypotheses) {
            appendUniqueLimited(mergedHypotheses, hypothesisSeen, item, 4, 120);
        }

        const auto focus = collectJsonStrings(cognitiveState.value("attentionFocus", json::array()), 1);
        const std::string sceneSummary = trimCopy(sceneDoc.value("summary", std::string()));
        const std::string episodeSummary = trimCopy(episodeDoc.value("summary", std::string()));
        const std::string nextStep = trimCopy(agenda.value("nextStep", std::string()));
        std::ostringstream summary;
        if (!sceneSummary.empty()) {
            summary << "expected continuation of " << sceneSummary;
        } else if (!episodeSummary.empty()) {
            summary << "expected continuation of " << episodeSummary;
        } else if (!focus.empty()) {
            summary << "expected next state around " << focus.front();
        } else {
            summary << "expected continuation of the current scene";
        }
        if (!focus.empty()) {
            summary << "; monitor " << focus.front();
        }
        if (!nextStep.empty()) {
            summary << "; next step " << nextStep;
        }

        json fallback{{"summary", truncateText(summary.str(), 180)},
                      {"entities", toJsonArray(entities)},
                      {"relations", relations},
                      {"goals", toJsonArray(mergedGoals)},
                      {"hypotheses", toJsonArray(mergedHypotheses)},
                      {"confidence", clampUnitValue(1.0 - cognitiveState.value("uncertainty", 0.5))},
                      {"horizon", "next-observation"},
                      {"sourceEvidenceIds", recentEvidenceIds(sessionDoc, 4)},
                      {"createdAt", evidence.value("createdAt", nowMs())}};
        return normalizeStatePayload(evidenceField(evidence, "expectedNextState"), fallback);
    }

    double computeAlignmentScore(const json &expectedState, const json &observedState) const {
        const auto expectedSummaryTokens = extractKeywords(trimCopy(expectedState.value("summary", std::string())), 8);
        const auto observedSummaryTokens = extractKeywords(trimCopy(observedState.value("summary", std::string())), 8);
        const auto expectedEntities = collectJsonStrings(expectedState.value("entities", json::array()), 8);
        const auto observedEntities = collectJsonStrings(observedState.value("entities", json::array()), 8);
        const auto expectedRelations = collectRelationSummaries(expectedState.value("relations", json::array()), 8);
        const auto observedRelations = collectRelationSummaries(observedState.value("relations", json::array()), 8);
        const auto expectedGoals = collectJsonStrings(expectedState.value("goals", json::array()), 4);
        const auto observedGoals = collectJsonStrings(observedState.value("goals", json::array()), 4);

        double weightedScore = 0.0;
        double weightTotal = 0.0;
        auto addComponent = [&](double score, double weight, bool enabled) {
            if (!enabled) {
                return;
            }
            weightedScore += score * weight;
            weightTotal += weight;
        };

        addComponent(computeStringSetOverlap(expectedSummaryTokens, observedSummaryTokens),
                     0.45,
                     !expectedSummaryTokens.empty() || !observedSummaryTokens.empty());
        addComponent(computeStringSetOverlap(expectedEntities, observedEntities),
                     0.30,
                     !expectedEntities.empty() || !observedEntities.empty());
        addComponent(computeStringSetOverlap(expectedRelations, observedRelations),
                     0.15,
                     !expectedRelations.empty() || !observedRelations.empty());
        addComponent(computeStringSetOverlap(expectedGoals, observedGoals),
                     0.10,
                     !expectedGoals.empty() || !observedGoals.empty());

        if (weightTotal <= 0.0) {
            return 0.0;
        }
        return clampUnitValue(weightedScore / weightTotal);
    }

    void updatePredictionDoc(json &predictionDoc,
                             json &sessionDoc,
                             json &sceneDoc,
                             json &episodeDoc,
                             const json &evidence,
                             const json &recentEvidence) const {
        const json previousExpected = (predictionDoc.contains("expectedNextState") && predictionDoc["expectedNextState"].is_object())
                                          ? predictionDoc["expectedNextState"]
                                          : json::object();
        json observedState = buildObservedNextState(evidence, sceneDoc, episodeDoc);

        if (!predictionDoc.contains("calibration") || !predictionDoc["calibration"].is_object()) {
            predictionDoc["calibration"] = defaultPredictionDoc(evidence.value("sessionId", std::string()))["calibration"];
        }
        json &calibration = predictionDoc["calibration"];
        if (!calibration.contains("corrections") || !calibration["corrections"].is_array()) {
            calibration["corrections"] = json::array();
        }

        if (stateHasSignal(previousExpected)) {
            const double alignmentScore = computeAlignmentScore(previousExpected, observedState);
            const bool matched = alignmentScore >= 0.45;
            const int previousSamples = calibration.value("samples", 0);
            const double previousAverage = calibration.value("avgAlignmentScore", 0.0);
            calibration["samples"] = previousSamples + 1;
            calibration["matched"] = calibration.value("matched", 0) + (matched ? 1 : 0);
            calibration["mismatched"] = calibration.value("mismatched", 0) + (matched ? 0 : 1);
            calibration["avgAlignmentScore"] = (previousAverage * static_cast<double>(previousSamples) + alignmentScore) /
                                                 static_cast<double>(previousSamples + 1);
            calibration["lastAlignmentScore"] = alignmentScore;
            calibration["lastUpdatedAt"] = evidence.value("createdAt", nowMs());
            calibration["lastReason"] = matched ? std::string("observed state aligned with expectation")
                                                  : std::string("observed state diverged from expected next state");
            observedState["alignmentScore"] = alignmentScore;
            observedState["matchedExpectedState"] = matched;

            std::ostringstream correction;
            correction << (matched ? "match" : "mismatch")
                       << " alignment=" << std::fixed << std::setprecision(2) << alignmentScore;
            const auto expectedSummary = trimCopy(previousExpected.value("summary", std::string()));
            const auto observedSummary = trimCopy(observedState.value("summary", std::string()));
            if (!expectedSummary.empty()) {
                correction << " | expected=" << truncateText(expectedSummary, 72);
            }
            if (!observedSummary.empty()) {
                correction << " | observed=" << truncateText(observedSummary, 72);
            }
            calibration["corrections"].push_back(truncateText(correction.str(), 180));
            while (calibration["corrections"].size() > 6) {
                calibration["corrections"].erase(calibration["corrections"].begin());
            }
        } else {
            observedState["alignmentScore"] = 0.0;
            observedState["matchedExpectedState"] = false;
        }

        json predictiveSessionState{{"sessionId", evidence.value("sessionId", std::string())},
                                    {"session", sessionDoc},
                                    {"sceneState", sceneDoc},
                                    {"episode", episodeDoc},
                                    {"recentEvidence", recentEvidence},
                                    {"prediction", predictionDoc}};
        CognitiveStateOptions cognitiveOptions;
        cognitiveOptions.maxWorkingMemory = 4;
        cognitiveOptions.maxAttentionItems = 4;
        cognitiveOptions.maxGoals = 4;
        cognitiveOptions.maxRegionItems = 4;
        cognitiveOptions.maxChars = 140;
        const auto cognitiveState = buildCognitiveState(predictiveSessionState, cognitiveOptions);
        const auto agenda = buildReasoningAgendaFromCognitiveState(cognitiveState, 4, 140);
        const auto expectedState = buildExpectedNextState(sessionDoc, sceneDoc, episodeDoc, evidence, cognitiveState, agenda);

        std::vector<std::string> entities = collectFlexibleStrings(evidenceField(evidence, "entities"), options_.maxObjectSlots);
        const auto sceneEntities = collectObjectLabels(sceneDoc, std::min<std::size_t>(options_.maxObjectSlots, 6));
        const auto observedEntities = collectJsonStrings(observedState.value("entities", json::array()), options_.maxObjectSlots);
        std::unordered_set<std::string> entitySeen;
        std::vector<std::string> mergedEntities;
        for (const auto &item : entities) {
            appendUniqueLimited(mergedEntities, entitySeen, item, options_.maxObjectSlots, 120);
        }
        for (const auto &item : sceneEntities) {
            appendUniqueLimited(mergedEntities, entitySeen, item, options_.maxObjectSlots, 120);
        }
        for (const auto &item : observedEntities) {
            appendUniqueLimited(mergedEntities, entitySeen, item, options_.maxObjectSlots, 120);
        }

        std::vector<std::string> goals = collectJsonStrings(cognitiveState.value("activeGoals", json::array()), 4);
        const auto explicitGoals = collectFlexibleStrings(evidenceField(evidence, "goals"), 4);
        const auto explicitGoal = collectFlexibleStrings(evidenceField(evidence, "goal"), 1);
        std::unordered_set<std::string> goalSeen;
        std::vector<std::string> mergedGoals;
        for (const auto &item : goals) {
            appendUniqueLimited(mergedGoals, goalSeen, item, 4, 120);
        }
        for (const auto &item : explicitGoals) {
            appendUniqueLimited(mergedGoals, goalSeen, item, 4, 120);
        }
        for (const auto &item : explicitGoal) {
            appendUniqueLimited(mergedGoals, goalSeen, item, 4, 120);
        }

        std::vector<std::string> hypotheses = collectJsonStrings(agenda.value("hypotheses", json::array()), 4);
        const auto explicitHypotheses = collectFlexibleStrings(evidenceField(evidence, "hypotheses"), 4);
        std::unordered_set<std::string> hypothesisSeen;
        std::vector<std::string> mergedHypotheses;
        for (const auto &item : hypotheses) {
            appendUniqueLimited(mergedHypotheses, hypothesisSeen, item, 4, 120);
        }
        for (const auto &item : explicitHypotheses) {
            appendUniqueLimited(mergedHypotheses, hypothesisSeen, item, 4, 120);
        }

        std::vector<std::string> contradictions = collectJsonStrings(agenda.value("contradictions", json::array()), 4);
        const auto explicitContradictions = collectFlexibleStrings(evidenceField(evidence, "contradictions"), 4);
        std::unordered_set<std::string> contradictionSeen;
        std::vector<std::string> mergedContradictions;
        for (const auto &item : contradictions) {
            appendUniqueLimited(mergedContradictions, contradictionSeen, item, 4, 120);
        }
        for (const auto &item : explicitContradictions) {
            appendUniqueLimited(mergedContradictions, contradictionSeen, item, 4, 120);
        }
        if (stateHasSignal(previousExpected) && observedState.value("matchedExpectedState", false) == false) {
            appendUniqueLimited(mergedContradictions,
                                contradictionSeen,
                                "expected next state diverged from the latest observation",
                                4,
                                120);
        }

        json relations = sceneDoc.contains("relations") && sceneDoc["relations"].is_array() ? sceneDoc["relations"] : json::array();
        mergeRelationArray(relations, observedState.value("relations", json::array()), std::max<std::size_t>(std::size_t(6), options_.maxObjectSlots * 3));

        predictionDoc["sessionId"] = evidence.value("sessionId", std::string());
        predictionDoc["entities"] = toJsonArray(mergedEntities);
        predictionDoc["relations"] = relations;
        predictionDoc["goals"] = toJsonArray(mergedGoals);
        predictionDoc["hypotheses"] = toJsonArray(mergedHypotheses);
        predictionDoc["contradictions"] = toJsonArray(mergedContradictions);
        predictionDoc["observedNextState"] = observedState;
        predictionDoc["expectedNextState"] = expectedState;
        predictionDoc["lastUpdatedAt"] = evidence.value("createdAt", nowMs());

        sessionDoc["predictionCalibration"] = calibration;
        sessionDoc["lastExpectedNextState"] = expectedState.value("summary", std::string());
        sessionDoc["lastObservedNextState"] = observedState.value("summary", std::string());

        episodeDoc["expectedNextState"] = expectedState;
        episodeDoc["observedNextState"] = observedState;
        episodeDoc["predictionCalibration"] = calibration;
    }

    void updateSessionDoc(json &sessionDoc, const json &evidence) const {
        const auto evidenceId = evidence.value("id", std::string());
        sessionDoc["sessionId"] = evidence.value("sessionId", std::string());
        sessionDoc["evidenceCount"] = sessionDoc.value("evidenceCount", 0) + 1;
        sessionDoc["lastUpdatedAt"] = evidence.value("createdAt", nowMs());
        sessionDoc["lastModality"] = evidence.value("modality", std::string());
        appendRecentId(sessionDoc["recentEvidenceIds"], evidenceId, options_.maxRecentEvidence);
        mergeTags(sessionDoc["recentTags"], evidence);

        if (!sessionDoc.contains("modalities") || !sessionDoc["modalities"].is_object()) {
            sessionDoc["modalities"] = json::object();
        }
        const auto modality = evidence.value("modality", std::string("text"));
        sessionDoc["modalities"][modality] = sessionDoc["modalities"].value(modality, 0) + 1;
    }

    void updateSceneDoc(json &sceneDoc, const json &evidence) const {
        sceneDoc["sessionId"] = evidence.value("sessionId", std::string());
        sceneDoc["evidenceCount"] = sceneDoc.value("evidenceCount", 0) + 1;
        sceneDoc["lastUpdatedAt"] = evidence.value("createdAt", nowMs());
        appendRecentId(sceneDoc["recentEvidenceIds"], evidence.value("id", std::string()), options_.maxRecentSceneEvidence);
        mergeTags(sceneDoc["tags"], evidence);
        mergeObjectSlots(sceneDoc["objectSlots"], evidence);
        mergeRelationArray(sceneDoc["relations"], normalizeRelations(evidenceField(evidence, "relations")), std::max<std::size_t>(std::size_t(6), options_.maxObjectSlots * 3));
        mergeRelationArray(sceneDoc["relations"], deriveRelations(evidence), std::max<std::size_t>(std::size_t(6), options_.maxObjectSlots * 3));
        mergeGraphContext(sceneDoc["graphContext"], evidence.value("graphSummary", std::string()));

        if (!sceneDoc.contains("modalities") || !sceneDoc["modalities"].is_array()) {
            sceneDoc["modalities"] = json::array();
        }
        const auto modality = evidence.value("modality", std::string("text"));
        if (!jsonArrayContainsString(sceneDoc["modalities"], modality)) {
            sceneDoc["modalities"].push_back(modality);
        }

        std::ostringstream summary;
        summary << "Observed " << sceneDoc.value("evidenceCount", 0) << " evidence item(s)";
        if (sceneDoc.contains("modalities") && sceneDoc["modalities"].is_array() && !sceneDoc["modalities"].empty()) {
            summary << " across modalities ";
            for (std::size_t i = 0; i < sceneDoc["modalities"].size(); ++i) {
                if (i > 0) {
                    summary << ", ";
                }
                summary << sceneDoc["modalities"][i].get<std::string>();
            }
        }
        if (sceneDoc.contains("objectSlots") && sceneDoc["objectSlots"].is_array() && !sceneDoc["objectSlots"].empty()) {
            summary << ". Objects: ";
            std::size_t limit = std::min<std::size_t>(sceneDoc["objectSlots"].size(), 5);
            for (std::size_t i = 0; i < limit; ++i) {
                if (i > 0) {
                    summary << ", ";
                }
                const auto &slot = sceneDoc["objectSlots"][i];
                summary << slot.value("label", std::string("unknown"));
            }
        }
        const auto graphContext = joinGraphContext(sceneDoc["graphContext"]);
        if (!graphContext.empty()) {
            summary << ". Context: " << truncateText(graphContext, 180);
        }
        sceneDoc["summary"] = summary.str();
    }

    void updateEpisodeDoc(json &episodeDoc, const json &evidence, const json &sceneDoc) const {
        episodeDoc["sessionId"] = evidence.value("sessionId", std::string());
        episodeDoc["updatedAt"] = evidence.value("createdAt", nowMs());
        episodeDoc["turnCount"] = episodeDoc.value("turnCount", 0) + 1;
        appendRecentId(episodeDoc["evidenceIds"], evidence.value("id", std::string()), options_.maxRecentEvidence);
        const auto graphSummary = evidence.value("graphSummary", std::string());
        if (!graphSummary.empty()) {
            episodeDoc["summary"] = truncateText(graphSummary, 220);
        } else {
            episodeDoc["summary"] = sceneDoc.value("summary", std::string());
        }
    }
};

} // namespace world_model