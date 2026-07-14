/* v51_runtime.cpp - V51 runtime implementation
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

#include "v51_runtime.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Fragment {
    std::string text;
    std::unordered_set<std::string> tokens;
    std::string signature;
    double relevance{0.0};
    double domain{0.0};
    double novelty{0.0};
    double conflict{0.0};
    double anchor{0.0};
    double noise{0.0};
    double familiarity{0.0};
    double recency{0.0};
    double attention{0.0};
    double centrality{0.0};
    double peripheral{0.0};
    double memoryStrength{0.0};
    double score{0.0};
    int index{0};
};

struct Event {
    std::string type;
    std::string index;
    std::vector<double> vec;
    double confidence{0.0};
    std::string source;
    std::vector<int> relations;
    std::string text;
};

struct SessionState {
    std::deque<std::string> shortContext;
    std::deque<std::string> taskMemory;
    std::deque<std::string> longMemory;
    std::deque<std::string> cacheMemory;
    std::deque<std::string> workingMemory;
    std::deque<std::string> episodicMemory;
    std::deque<std::string> semanticMemory;
    std::deque<std::string> sensoryBuffer;
    std::unordered_map<std::string, int> tokenSeen;
    std::unordered_map<std::string, int> fragmentSeen;
    std::unordered_map<std::string, double> learnerWeights;
    int64_t tick{0};
    double lastPosteriorScore{0.0};
    double lastAttentionScale{0.0};
    double lastPruningIntensity{0.0};
    double lastAdaptiveThreshold{0.0};
    bool lastMirrorPassed{true};
    Clock::time_point lastTouched{Clock::now()};
};

struct LearnerState {
    double alpha{1.0};
    double beta{1.0};
    double gamma{1.0};
    double delta{1.0};
    double threshold{0.7};
};

static std::string toLowerCopy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        if (c >= 'A' && c <= 'Z') {
            return static_cast<char>(c - 'A' + 'a');
        }
        return static_cast<char>(c);
    });
    return s;
}

static std::vector<std::string> splitFragments(const std::string &input) {
    std::vector<std::string> out;
    std::string current;
    current.reserve(input.size());
    for (char ch : input) {
        current.push_back(ch);
        if (ch == '\n' || ch == '.' || ch == '!' || ch == '?' || ch == ';' || ch == ',') {
            std::string trimmed;
            trimmed.reserve(current.size());
            for (char c : current) {
                if (c == '\r') {
                    continue;
                }
                trimmed.push_back(c);
            }
            auto begin = trimmed.find_first_not_of(" \t\n");
            if (begin != std::string::npos) {
                auto end = trimmed.find_last_not_of(" \t\n");
                out.push_back(trimmed.substr(begin, end - begin + 1));
            }
            current.clear();
        }
    }
    if (!current.empty()) {
        auto begin = current.find_first_not_of(" \t\n\r");
        if (begin != std::string::npos) {
            auto end = current.find_last_not_of(" \t\n\r");
            out.push_back(current.substr(begin, end - begin + 1));
        }
    }
    return out;
}

static std::vector<std::string> extractTokens(const std::string &text) {
    std::vector<std::string> tokens;
    std::string current;
    for (unsigned char uc : text) {
        char c = static_cast<char>(uc);
        bool alnum = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
        if (alnum) {
            current.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        } else {
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
        }
    }
    if (!current.empty()) {
        tokens.push_back(current);
    }
    return tokens;
}

static double jaccard(const std::unordered_set<std::string> &a, const std::unordered_set<std::string> &b) {
    if (a.empty() || b.empty()) {
        return 0.0;
    }
    std::size_t inter = 0;
    for (const auto &x : a) {
        if (b.find(x) != b.end()) {
            inter += 1;
        }
    }
    std::size_t uni = a.size() + b.size() - inter;
    if (uni == 0) {
        return 0.0;
    }
    return static_cast<double>(inter) / static_cast<double>(uni);
}

static bool hasNegationConflict(const std::string &text) {
    const std::string lower = toLowerCopy(text);
    const bool hasMust = (lower.find("must") != std::string::npos) || (lower.find("should") != std::string::npos) || (lower.find("need") != std::string::npos);
    const bool hasNot = (lower.find(" not ") != std::string::npos) || (lower.find("cannot") != std::string::npos) || (lower.find("never") != std::string::npos);
    return hasMust && hasNot;
}

static std::string joinTop(const std::vector<Fragment> &fragments, std::size_t n) {
    std::ostringstream oss;
    std::size_t count = 0;
    for (const auto &f : fragments) {
        if (f.text.empty()) {
            continue;
        }
        if (count > 0) {
            oss << "\n";
        }
        oss << "- " << f.text;
        count += 1;
        if (count >= n) {
            break;
        }
    }
    return oss.str();
}

static std::string buildFragmentSignature(const std::unordered_set<std::string> &tokens) {
    if (tokens.empty()) {
        return std::string();
    }
    std::vector<std::string> ordered(tokens.begin(), tokens.end());
    std::sort(ordered.begin(), ordered.end());
    if (ordered.size() > 6) {
        ordered.resize(6);
    }
    std::ostringstream oss;
    for (std::size_t i = 0; i < ordered.size(); ++i) {
        if (i > 0) {
            oss << '|';
        }
        oss << ordered[i];
    }
    return oss.str();
}

static double computeNoisePenalty(const std::string &text, std::size_t tokenCount) {
    if (text.empty()) {
        return 1.0;
    }
    double punctuation = 0.0;
    double digits = 0.0;
    for (unsigned char ch : text) {
        if (std::isdigit(ch) != 0) {
            digits += 1.0;
        }
        if (std::ispunct(ch) != 0) {
            punctuation += 1.0;
        }
    }
    const double denom = std::max(1.0, static_cast<double>(text.size()));
    const double punctuationRatio = punctuation / denom;
    const double digitRatio = digits / denom;
    const double shortPenalty = tokenCount <= 1 ? 0.35 : 0.0;
    const double longPenalty = text.size() > 220 ? std::min(0.35, (static_cast<double>(text.size()) - 220.0) / 320.0) : 0.0;
    return std::clamp(shortPenalty + longPenalty + punctuationRatio * 1.4 + digitRatio * 0.5, 0.0, 1.0);
}

static double edgeDistanceFactor(int index, std::size_t totalCount) {
    if (totalCount <= 2) {
        return 1.0;
    }
    const double center = static_cast<double>(totalCount - 1) / 2.0;
    const double distance = std::abs(static_cast<double>(index) - center);
    const double denom = std::max(1.0, center);
    return 0.5 + 0.5 * (distance / denom);
}

static void pushUniqueLimited(std::deque<std::string> &dq, const std::string &value, std::size_t limit) {
    if (value.empty()) {
        return;
    }
    auto it = std::find(dq.begin(), dq.end(), value);
    if (it != dq.end()) {
        dq.erase(it);
    }
    dq.push_back(value);
    while (dq.size() > limit) {
        dq.pop_front();
    }
}

static void compactMemory(std::deque<std::string> &dq, std::size_t limit) {
    std::unordered_set<std::string> seen;
    std::deque<std::string> compact;
    for (auto it = dq.rbegin(); it != dq.rend(); ++it) {
        if (seen.insert(*it).second) {
            compact.push_front(*it);
        }
        if (compact.size() >= limit) {
            break;
        }
    }
    dq = std::move(compact);
}

static std::vector<std::string> collectRecentMemory(const std::deque<std::string> &dq, std::size_t limit) {
    std::vector<std::string> out;
    std::unordered_set<std::string> seen;
    for (auto it = dq.rbegin(); it != dq.rend() && out.size() < limit; ++it) {
        if (seen.insert(*it).second) {
            out.push_back(*it);
        }
    }
    std::reverse(out.begin(), out.end());
    return out;
}

static std::unordered_set<std::string> collectMemoryTokens(const std::deque<std::string> &dq, std::size_t limit) {
    std::unordered_set<std::string> tokens;
    const auto lines = collectRecentMemory(dq, limit);
    for (const auto &line : lines) {
        for (const auto &token : extractTokens(line)) {
            tokens.insert(token);
        }
    }
    return tokens;
}

static double tokenOverlapScore(const std::unordered_set<std::string> &tokens, const std::unordered_set<std::string> &memoryTokens) {
    if (tokens.empty() || memoryTokens.empty()) {
        return 0.0;
    }
    std::size_t overlap = 0;
    for (const auto &token : tokens) {
        if (memoryTokens.find(token) != memoryTokens.end()) {
            ++overlap;
        }
    }
    return static_cast<double>(overlap) / static_cast<double>(tokens.size());
}

static double estimateContextScale(std::size_t pieceCount, std::size_t totalTokens) {
    const double pieceScale = pieceCount <= 2 ? 0.0 : std::min(1.0, static_cast<double>(pieceCount - 2) / 8.0);
    const double tokenScale = totalTokens <= 48 ? 0.0 : std::min(1.0, static_cast<double>(totalTokens - 48) / 240.0);
    return std::clamp(0.6 * pieceScale + 0.4 * tokenScale, 0.0, 1.0);
}

static double computeAdaptiveThreshold(const LearnerState &learner, double contextScale) {
    return std::clamp(learner.threshold - 0.10 * (1.0 - contextScale) + 0.16 * contextScale, 0.45, 1.15);
}

static double computePruningIntensity(std::size_t pieceCount, double contextScale) {
    if (pieceCount <= 3) {
        return 0.06;
    }
    return std::clamp(0.08 + 0.84 * contextScale, 0.08, 0.95);
}

static Json::Value stringsToJson(const std::vector<std::string> &lines) {
    Json::Value out(Json::arrayValue);
    for (const auto &line : lines) {
        out.append(line);
    }
    return out;
}

static std::string formatMemorySection(const char *label, const std::vector<std::string> &lines) {
    if (lines.empty()) {
        return std::string();
    }
    std::ostringstream oss;
    oss << label << ":\n";
    for (const auto &line : lines) {
        oss << "- " << line << "\n";
    }
    return oss.str();
}

static Json::Value vectorToJson(const std::vector<double> &v) {
    Json::Value out(Json::arrayValue);
    for (double x : v) {
        out.append(x);
    }
    return out;
}

static std::string buildHumanLikeMemorySummary(const SessionState &session) {
    const auto working = collectRecentMemory(session.workingMemory, 3);
    const auto episodic = collectRecentMemory(session.episodicMemory, 2);
    const auto semantic = collectRecentMemory(session.semanticMemory, 2);
    const auto sensory = collectRecentMemory(session.sensoryBuffer, 2);

    std::ostringstream oss;
    bool hasSection = false;
    for (const auto &section : {formatMemorySection("工作记忆", working),
                                formatMemorySection("情景记忆", episodic),
                                formatMemorySection("语义记忆", semantic),
                                formatMemorySection("感官缓冲", sensory)}) {
        if (section.empty()) {
            continue;
        }
        if (hasSection) {
            oss << "\n";
        }
        oss << section;
        hasSection = true;
    }
    return oss.str();
}

} // namespace

struct V51RuntimeEngine::Impl {
    mutable std::mutex mu;
    std::unordered_map<std::string, SessionState> sessions;
    LearnerState learner;

    SessionState &touchSession(const std::string &sessionId) {
        auto &s = sessions[sessionId];
        s.lastTouched = Clock::now();
        s.tick += 1;
        return s;
    }

    std::vector<std::string> parseDomainHints(const Json::Value &request) const {
        std::vector<std::string> hints;
        if (request.isMember("domainHints") && request["domainHints"].isArray()) {
            for (const auto &x : request["domainHints"]) {
                if (x.isString()) {
                    hints.push_back(toLowerCopy(x.asString()));
                }
            }
        }
        if (request.isMember("domain") && request["domain"].isString()) {
            hints.push_back(toLowerCopy(request["domain"].asString()));
        }
        return hints;
    }

    std::vector<Event> buildEvents(const Json::Value &request) const {
        std::vector<Event> events;
        int seq = 0;
        auto append = [&](const std::string &type, const std::string &text, const std::string &source, double conf) {
            if (text.empty()) {
                return;
            }
            Event e;
            e.type = type;
            e.source = source;
            e.index = std::to_string(seq++);
            e.text = text;
            const auto tokens = extractTokens(text);
            double digits = 0.0;
            double punct = 0.0;
            for (unsigned char ch : text) {
                if (ch >= '0' && ch <= '9') {
                    digits += 1.0;
                }
                if (!std::isalnum(ch) && !std::isspace(ch)) {
                    punct += 1.0;
                }
            }
            const double denom = std::max(1.0, static_cast<double>(text.size()));
            e.vec = {static_cast<double>(tokens.size()), digits / denom, punct / denom};
            e.confidence = std::clamp(conf, 0.0, 1.0);
            events.push_back(std::move(e));
        };

        if (request.isMember("text") && request["text"].isString()) {
            append("text", request["text"].asString(), "user", 0.95);
        }
        if (request.isMember("imageContext") && request["imageContext"].isString()) {
            append("image", request["imageContext"].asString(), "vision", 0.85);
        }
        if (request.isMember("speechContext") && request["speechContext"].isString()) {
            append("audio", request["speechContext"].asString(), "speech", 0.8);
        }
        if (request.isMember("videoContext") && request["videoContext"].isString()) {
            append("video", request["videoContext"].asString(), "video", 0.8);
        }

        for (std::size_t i = 0; i < events.size(); ++i) {
            for (std::size_t j = i + 1; j < events.size(); ++j) {
                const auto t1 = extractTokens(events[i].text);
                const auto t2 = extractTokens(events[j].text);
                std::unordered_set<std::string> s1(t1.begin(), t1.end());
                std::unordered_set<std::string> s2(t2.begin(), t2.end());
                if (j - i <= 1 || jaccard(s1, s2) > 0.0) {
                    events[i].relations.push_back(static_cast<int>(j));
                    events[j].relations.push_back(static_cast<int>(i));
                }
            }
        }

        return events;
    }

    double computeMemoryFamiliarity(const SessionState &session,
                                    const std::unordered_set<std::string> &tokens,
                                    const std::string &signature) const {
        if (tokens.empty()) {
            return 0.0;
        }
        double seenScore = 0.0;
        for (const auto &token : tokens) {
            auto it = session.tokenSeen.find(token);
            if (it != session.tokenSeen.end()) {
                seenScore += std::min(1.0, std::log1p(static_cast<double>(it->second)) / std::log(6.0));
            }
        }
        seenScore /= static_cast<double>(tokens.size());

        double repeatScore = 0.0;
        auto repeatIt = session.fragmentSeen.find(signature);
        if (repeatIt != session.fragmentSeen.end()) {
            repeatScore = std::min(1.0, 0.25 * static_cast<double>(repeatIt->second));
        }

        const auto workingTokens = collectMemoryTokens(session.workingMemory, 4);
        const auto episodicTokens = collectMemoryTokens(session.episodicMemory, 4);
        const auto semanticTokens = collectMemoryTokens(session.semanticMemory, 4);

        const double workingOverlap = tokenOverlapScore(tokens, workingTokens);
        const double episodicOverlap = tokenOverlapScore(tokens, episodicTokens);
        const double semanticOverlap = tokenOverlapScore(tokens, semanticTokens);

        return std::clamp(0.25 * seenScore + 0.15 * repeatScore + 0.25 * workingOverlap + 0.15 * episodicOverlap + 0.20 * semanticOverlap,
                          0.0,
                          1.5);
    }

    double computeRecencyBias(const SessionState &session, const std::unordered_set<std::string> &tokens) const {
        const auto shortTokens = collectMemoryTokens(session.shortContext, 4);
        const auto sensoryTokens = collectMemoryTokens(session.sensoryBuffer, 4);
        const double shortOverlap = tokenOverlapScore(tokens, shortTokens);
        const double sensoryOverlap = tokenOverlapScore(tokens, sensoryTokens);
        return std::clamp(0.55 * shortOverlap + 0.45 * sensoryOverlap, 0.0, 1.0);
    }

    std::vector<Fragment> summarizeAndPrune(const std::string &text, const std::vector<std::string> &domainHints, SessionState &session,
                                           std::size_t *candidateCount = nullptr,
                                           double *contextScaleOut = nullptr,
                                           double *adaptiveThresholdOut = nullptr,
                                           double *pruningIntensityOut = nullptr) const {
        const auto pieces = splitFragments(text);
        if (candidateCount) {
            *candidateCount = pieces.size();
        }
        std::size_t totalTokens = 0;
        std::unordered_map<std::string, int> freq;
        for (const auto &p : pieces) {
            const auto tokens = extractTokens(p);
            totalTokens += tokens.size();
            for (const auto &tok : tokens) {
                freq[tok] += 1;
            }
        }

        const double contextScale = estimateContextScale(pieces.size(), totalTokens);
        const double adaptiveThreshold = computeAdaptiveThreshold(learner, contextScale);
        const double pruningIntensity = computePruningIntensity(pieces.size(), contextScale);
        if (contextScaleOut) {
            *contextScaleOut = contextScale;
        }
        if (adaptiveThresholdOut) {
            *adaptiveThresholdOut = adaptiveThreshold;
        }
        if (pruningIntensityOut) {
            *pruningIntensityOut = pruningIntensity;
        }

        std::vector<Fragment> fragments;
        fragments.reserve(pieces.size());
        int idx = 0;
        for (const auto &piece : pieces) {
            Fragment f;
            f.text = piece;
            f.index = idx++;
            const auto toks = extractTokens(piece);
            f.signature = buildFragmentSignature(std::unordered_set<std::string>(toks.begin(), toks.end()));
            for (const auto &tok : toks) {
                f.tokens.insert(tok);
                f.relevance += 1.0 / static_cast<double>(std::max(1, freq[tok]));
                const auto seen = session.tokenSeen.find(tok);
                if (seen == session.tokenSeen.end()) {
                    f.novelty += 1.0;
                } else {
                    f.novelty += 1.0 / static_cast<double>(1 + seen->second);
                }
                for (const auto &hint : domainHints) {
                    if (!hint.empty() && tok.find(hint) != std::string::npos) {
                        f.domain += 1.0;
                    }
                }
                auto weightIt = session.learnerWeights.find(tok);
                if (weightIt != session.learnerWeights.end()) {
                    f.anchor += weightIt->second;
                }
            }
            if (hasNegationConflict(piece)) {
                f.conflict += 1.0;
            }

            const double tokenCount = std::max(1.0, static_cast<double>(f.tokens.size()));
            const double signalDensity = std::clamp(tokenCount / 6.0, 0.35, 1.0);
            const double brevityPenalty = tokenCount < 4.0 ? 0.16 * (4.0 - tokenCount) : 0.0;
            f.relevance /= tokenCount;
            f.novelty /= tokenCount;
            f.domain /= tokenCount;
            f.anchor /= tokenCount;
            f.noise = computeNoisePenalty(piece, static_cast<std::size_t>(tokenCount));

            double historicalRepeats = 0.0;
            auto seenIt = session.fragmentSeen.find(f.signature);
            if (seenIt != session.fragmentSeen.end()) {
                historicalRepeats = std::min(1.5, 0.25 * static_cast<double>(seenIt->second));
            }

            f.familiarity = computeMemoryFamiliarity(session, f.tokens, f.signature);
            f.recency = computeRecencyBias(session, f.tokens);

            const double structure = std::min(1.4, 0.35 + 0.12 * tokenCount);
            const double decay = static_cast<double>(session.tick) * 0.0015;
            f.attention = std::max(0.0,
                                   0.22 * f.relevance * signalDensity +
                                       0.24 * f.domain +
                                       (0.12 + 0.08 * (1.0 - contextScale)) * f.novelty * signalDensity +
                                       (0.16 + 0.22 * contextScale) * f.familiarity +
                                       (0.10 + 0.18 * contextScale) * f.recency +
                                       0.10 * std::max(0.0, f.anchor) -
                                       (0.10 + 0.25 * pruningIntensity) * f.noise -
                                       0.10 * f.conflict);
            f.memoryStrength = 0.28 * historicalRepeats + 0.28 * f.familiarity + 0.18 * f.recency + 0.16 * std::max(0.0, f.anchor) + 0.10 * f.domain;
            const double life = learner.alpha * structure +
                                learner.beta * f.relevance * signalDensity +
                                learner.gamma * ((0.55 * f.novelty * signalDensity) + 0.45 * f.domain) +
                                0.42 * f.memoryStrength +
                                0.55 * f.attention +
                                0.25 * historicalRepeats +
                                0.30 * f.anchor -
                                learner.delta * (f.conflict + decay + brevityPenalty + (0.35 + 0.75 * pruningIntensity) * f.noise);
            f.score = life;
            fragments.push_back(std::move(f));
        }

        std::sort(fragments.begin(), fragments.end(), [](const Fragment &a, const Fragment &b) {
            return a.score > b.score;
        });

        std::size_t budget = fragments.size();
        if (fragments.size() > 3) {
            const double keepRatio = std::clamp(0.98 - 0.58 * pruningIntensity, 0.40, 1.0);
            const std::size_t desired = static_cast<std::size_t>(std::llround(static_cast<double>(fragments.size()) * keepRatio));
            const std::size_t minKeep = pruningIntensity < 0.25 ? 3 : 2;
            budget = std::max<std::size_t>(minKeep, std::min<std::size_t>(12, std::min<std::size_t>(fragments.size(), desired)));
        }
        std::vector<Fragment> kept;
        kept.reserve(budget);
        for (const auto &f : fragments) {
            if (kept.size() >= budget) {
                break;
            }
            if (f.score + 1e-9 >= adaptiveThreshold || f.attention >= adaptiveThreshold * 0.60 || f.familiarity >= 0.45 || f.domain > 0.0 || f.anchor > 0.15) {
                kept.push_back(f);
            }
        }
        if (kept.size() < std::min<std::size_t>(2, fragments.size())) {
            for (const auto &f : fragments) {
                if (kept.size() >= std::min<std::size_t>(2, fragments.size())) {
                    break;
                }
                auto dup = std::find_if(kept.begin(), kept.end(), [&](const Fragment &existing) {
                    return existing.index == f.index;
                });
                if (dup == kept.end()) {
                    kept.push_back(f);
                }
            }
        }
        return kept;
    }

    void applyGraphPropagation(std::vector<Fragment> &fragments, double pruningIntensity) const {
        if (fragments.size() <= 1) {
            return;
        }
        const std::size_t n = fragments.size();
        std::vector<std::vector<double>> adj(n, std::vector<double>(n, 0.0));
        std::vector<double> centralityRaw(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                double w = jaccard(fragments[i].tokens, fragments[j].tokens);
                if (std::abs(static_cast<int>(fragments[i].index) - static_cast<int>(fragments[j].index)) <= 1) {
                    w += 0.2;
                }
                if (fragments[i].domain > 0.0 && fragments[j].domain > 0.0) {
                    w += 0.1;
                }
                adj[i][j] = w;
                adj[j][i] = w;
                centralityRaw[i] += w;
                centralityRaw[j] += w;
            }
        }

        const double maxCentrality = *std::max_element(centralityRaw.begin(), centralityRaw.end());

        std::vector<double> score(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            score[i] = fragments[i].score;
            fragments[i].centrality = maxCentrality > 0.0 ? centralityRaw[i] / maxCentrality : 0.0;
        }

        for (int it = 0; it < 2; ++it) {
            std::vector<double> next = score;
            for (std::size_t i = 0; i < n; ++i) {
                double sumStrongW = 0.0;
                double sumStrongS = 0.0;
                double sumWeakW = 0.0;
                double sumWeakS = 0.0;
                for (std::size_t j = 0; j < n; ++j) {
                    if (i == j || adj[i][j] <= 0.0) {
                        continue;
                    }
                    if (adj[i][j] >= 0.18) {
                        sumStrongW += adj[i][j];
                        sumStrongS += adj[i][j] * score[j];
                    } else {
                        sumWeakW += adj[i][j];
                        sumWeakS += adj[i][j] * score[j];
                    }
                }
                const double support = sumStrongW > 0.0 ? (sumStrongS / sumStrongW) : score[i] * 0.6;
                const double distraction = sumWeakW > 0.0 ? (sumWeakS / std::max(0.001, sumWeakW)) : 0.0;
                const double supportRetention = std::clamp(1.0 - 0.55 * fragments[i].noise + 0.25 * fragments[i].familiarity + 0.20 * fragments[i].domain,
                                                           0.25,
                                                           1.25);
                const double anchorBoost = 0.14 * fragments[i].centrality +
                                           0.16 * fragments[i].domain +
                                           0.10 * std::max(0.0, fragments[i].anchor) +
                                           0.12 * fragments[i].attention +
                                           0.10 * fragments[i].familiarity +
                                           0.08 * fragments[i].recency;
                const double peripheralPenalty = edgeDistanceFactor(fragments[i].index, n) * (1.0 - fragments[i].centrality) *
                                                 (0.12 + 0.88 * pruningIntensity) *
                                                 (0.30 + 0.45 * fragments[i].noise) *
                                                 (1.0 - std::min(1.0, fragments[i].domain + std::max(0.0, fragments[i].anchor) + 0.6 * fragments[i].familiarity + 0.3 * fragments[i].recency));
                const double noisePenalty = (0.08 + 0.22 * pruningIntensity) * fragments[i].noise;
                fragments[i].peripheral = peripheralPenalty;
                next[i] = std::max(0.0,
                                   0.48 * score[i] +
                                       0.28 * support * supportRetention +
                                       anchorBoost -
                                       (0.10 + 0.12 * pruningIntensity) * distraction -
                                       peripheralPenalty -
                                       noisePenalty);
            }
            score.swap(next);
        }

        for (std::size_t i = 0; i < n; ++i) {
            fragments[i].score = score[i];
        }
        std::sort(fragments.begin(), fragments.end(), [](const Fragment &a, const Fragment &b) {
            return a.score > b.score;
        });
    }

    void updateMemory(SessionState &session, const std::vector<Fragment> &fragments, double adaptiveThreshold) const {
        for (const auto &f : fragments) {
            pushUniqueLimited(session.shortContext, f.text, 8);

            const int repeats = ++session.fragmentSeen[f.signature];
            const bool strongFocus = f.attention >= adaptiveThreshold * 0.75 || f.score >= adaptiveThreshold - 0.05 || f.domain >= 0.25 || f.familiarity >= 0.40;
            const bool episodic = (f.novelty >= 0.55 && f.score >= learner.threshold - 0.05) || f.conflict > 0.0;
            const bool semantic = (repeats >= 2 || f.familiarity >= 0.55 || (f.score >= adaptiveThreshold + 0.15 && f.peripheral <= 0.30 && f.noise <= 0.45)) && f.conflict < 0.5;
            const bool sensory = f.score >= adaptiveThreshold - 0.25 && f.noise <= 0.95;

            if (sensory) {
                pushUniqueLimited(session.sensoryBuffer, f.text, 18);
                pushUniqueLimited(session.cacheMemory, f.text, 40);
            }
            if (strongFocus) {
                pushUniqueLimited(session.workingMemory, f.text, 8);
                pushUniqueLimited(session.taskMemory, f.text, 24);
            }
            if (episodic) {
                pushUniqueLimited(session.episodicMemory, f.text, 28);
            }
            if (semantic) {
                pushUniqueLimited(session.semanticMemory, f.text, 48);
                pushUniqueLimited(session.longMemory, f.text, 80);
            }
            for (const auto &tok : f.tokens) {
                session.tokenSeen[tok] += 1;
            }
        }

        compactMemory(session.workingMemory, 6);
        compactMemory(session.episodicMemory, 24);
        compactMemory(session.semanticMemory, 40);
        compactMemory(session.sensoryBuffer, 12);
        compactMemory(session.cacheMemory, 20);
        compactMemory(session.taskMemory, 20);
        compactMemory(session.longMemory, 60);
    }

    std::pair<double, bool> posteriorAndMirror(const std::string &draft, const std::vector<Fragment> &fragments) const {
        if (draft.empty()) {
            return {0.0, false};
        }
        std::unordered_set<std::string> targetTokens;
        for (std::size_t i = 0; i < std::min<std::size_t>(3, fragments.size()); ++i) {
            for (const auto &tok : fragments[i].tokens) {
                if (tok.size() >= 3) {
                    targetTokens.insert(tok);
                }
            }
        }
        if (targetTokens.empty()) {
            return {0.8, true};
        }
        const auto outTokVec = extractTokens(draft);
        std::unordered_set<std::string> outTok(outTokVec.begin(), outTokVec.end());
        std::size_t hit = 0;
        for (const auto &tok : targetTokens) {
            if (outTok.find(tok) != outTok.end()) {
                hit += 1;
            }
        }
        double coverage = static_cast<double>(hit) / static_cast<double>(targetTokens.size());

        const std::string lower = toLowerCopy(draft);
        bool mirrorPass = true;
        if (lower.find("cannot") != std::string::npos && lower.find("must") != std::string::npos) {
            mirrorPass = false;
        }
        if (lower.find("error") != std::string::npos && lower.find("success") != std::string::npos) {
            mirrorPass = false;
        }

        const double score = std::clamp(0.2 + 0.8 * coverage - (mirrorPass ? 0.0 : 0.2), 0.0, 1.0);
        return {score, mirrorPass};
    }

    Json::Value buildProcessResult(const Json::Value &request) {
        std::string sessionId = request.isMember("sessionId") && request["sessionId"].isString() ? request["sessionId"].asString() : std::string();
        if (sessionId.empty()) {
            const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count();
            sessionId = "v51-" + std::to_string(now);
        }

        const auto domainHints = parseDomainHints(request);
        std::string allText;
        if (request.isMember("text") && request["text"].isString()) {
            allText += request["text"].asString();
        }
        if (request.isMember("imageContext") && request["imageContext"].isString()) {
            if (!allText.empty()) allText += "\n";
            allText += request["imageContext"].asString();
        }
        if (request.isMember("speechContext") && request["speechContext"].isString()) {
            if (!allText.empty()) allText += "\n";
            allText += request["speechContext"].asString();
        }
        if (allText.empty()) {
            allText = "empty-input";
        }

        auto &session = touchSession(sessionId);
        std::size_t candidateCount = 0;
        double contextScale = 0.0;
        double adaptiveThreshold = learner.threshold;
        double pruningIntensity = 0.0;
        auto fragments = summarizeAndPrune(allText, domainHints, session, &candidateCount, &contextScale, &adaptiveThreshold, &pruningIntensity);
        applyGraphPropagation(fragments, pruningIntensity);
        updateMemory(session, fragments, adaptiveThreshold);

        const std::string memorySummary = buildHumanLikeMemorySummary(session);

        std::ostringstream draft;
        draft << "[v5.1] 结构化摘要:\n" << joinTop(fragments, 4);
        if (!domainHints.empty()) {
            draft << "\n\n域提示: ";
            for (std::size_t i = 0; i < domainHints.size(); ++i) {
                if (i) draft << ", ";
                draft << domainHints[i];
            }
        }
        if (!memorySummary.empty()) {
            draft << "\n\n" << memorySummary;
        }

        auto [posteriorScore, mirrorPass] = posteriorAndMirror(draft.str(), fragments);
        if (posteriorScore < 0.65 || !mirrorPass) {
            draft << "\n\n[修正] 输出触发后验修正，已加强关键约束覆盖与冲突规避。";
            auto repaired = posteriorAndMirror(draft.str(), fragments);
            posteriorScore = repaired.first;
            mirrorPass = repaired.second;
        }

        session.lastPosteriorScore = posteriorScore;
        session.lastAttentionScale = contextScale;
        session.lastPruningIntensity = pruningIntensity;
        session.lastAdaptiveThreshold = adaptiveThreshold;
        session.lastMirrorPassed = mirrorPass;

        const auto events = buildEvents(request);

        Json::Value result;
        result["ok"] = true;
        result["sessionId"] = sessionId;
        result["response"] = draft.str();
        result["posteriorScore"] = posteriorScore;
        result["mirrorPassed"] = mirrorPass;
        result["memorySummary"] = memorySummary;

        Json::Value attention;
        attention["contextScale"] = contextScale;
        attention["pruningIntensity"] = pruningIntensity;
        attention["adaptiveThreshold"] = adaptiveThreshold;
        attention["candidateCount"] = static_cast<Json::UInt64>(candidateCount);
        result["attention"] = attention;

        Json::Value pruning;
        pruning["candidateCount"] = static_cast<Json::UInt64>(candidateCount);
        pruning["keptCount"] = static_cast<Json::UInt64>(fragments.size());
        pruning["suppressedCount"] = static_cast<Json::UInt64>(candidateCount >= fragments.size() ? candidateCount - fragments.size() : 0);
        result["pruning"] = pruning;

        Json::Value frag(Json::arrayValue);
        for (const auto &f : fragments) {
            Json::Value item;
            item["text"] = f.text;
            item["score"] = f.score;
            item["relevance"] = f.relevance;
            item["domain"] = f.domain;
            item["novelty"] = f.novelty;
            item["conflict"] = f.conflict;
            item["anchor"] = f.anchor;
            item["noise"] = f.noise;
            item["familiarity"] = f.familiarity;
            item["recency"] = f.recency;
            item["attention"] = f.attention;
            item["centrality"] = f.centrality;
            item["peripheral"] = f.peripheral;
            item["memoryStrength"] = f.memoryStrength;
            frag.append(item);
        }
        result["fragments"] = frag;

        Json::Value eventJson(Json::arrayValue);
        for (const auto &e : events) {
            Json::Value item;
            item["type"] = e.type;
            item["index"] = e.index;
            item["vector"] = vectorToJson(e.vec);
            item["confidence"] = e.confidence;
            item["source"] = e.source;
            Json::Value links(Json::arrayValue);
            for (int x : e.relations) {
                links.append(x);
            }
            item["relations"] = links;
            eventJson.append(item);
        }
        result["events"] = eventJson;

        Json::Value mem;
        mem["short"] = static_cast<Json::UInt64>(session.shortContext.size());
        mem["task"] = static_cast<Json::UInt64>(session.taskMemory.size());
        mem["long"] = static_cast<Json::UInt64>(session.longMemory.size());
        mem["cache"] = static_cast<Json::UInt64>(session.cacheMemory.size());
        Json::Value humanLike;
        humanLike["workingCount"] = static_cast<Json::UInt64>(session.workingMemory.size());
        humanLike["episodicCount"] = static_cast<Json::UInt64>(session.episodicMemory.size());
        humanLike["semanticCount"] = static_cast<Json::UInt64>(session.semanticMemory.size());
        humanLike["sensoryCount"] = static_cast<Json::UInt64>(session.sensoryBuffer.size());
        humanLike["workingFocus"] = stringsToJson(collectRecentMemory(session.workingMemory, 4));
        humanLike["episodicFocus"] = stringsToJson(collectRecentMemory(session.episodicMemory, 4));
        humanLike["semanticFocus"] = stringsToJson(collectRecentMemory(session.semanticMemory, 4));
        humanLike["sensoryFocus"] = stringsToJson(collectRecentMemory(session.sensoryBuffer, 4));
        mem["humanLike"] = humanLike;
        result["memory"] = mem;
        return result;
    }

    Json::Value buildLearnResult(const Json::Value &request) {
        const std::string sessionId = request.isMember("sessionId") && request["sessionId"].isString() ? request["sessionId"].asString() : std::string("global");
        auto &session = touchSession(sessionId);
        const bool hasExplicitFeedback = request.isMember("feedback") && request["feedback"].isNumeric();
        const double feedback = std::clamp(hasExplicitFeedback ? request["feedback"].asDouble() : 0.0, -1.0, 1.0);
        const double rate = std::clamp(request.isMember("learningRate") ? request["learningRate"].asDouble() : 0.08, 0.001, 0.5);

        auto readDouble = [](const Json::Value &obj, const char *key, double fallback) {
            return obj.isObject() && obj.isMember(key) && obj[key].isNumeric() ? obj[key].asDouble() : fallback;
        };
        auto readBool = [](const Json::Value &obj, const char *key, bool fallback) {
            return obj.isObject() && obj.isMember(key) && obj[key].isBool() ? obj[key].asBool() : fallback;
        };
        auto readString = [](const Json::Value &obj, const char *key, const std::string &fallback) {
            return obj.isObject() && obj.isMember(key) && obj[key].isString() ? obj[key].asString() : fallback;
        };

        const Json::Value residual = request.isMember("residual") && request["residual"].isObject() ? request["residual"] : Json::Value(Json::objectValue);
        const bool hasPredictedVerify = (residual.isMember("predictedVerifyScore") && residual["predictedVerifyScore"].isNumeric()) ||
                                        (request.isMember("predictedVerifyScore") && request["predictedVerifyScore"].isNumeric());
        const bool hasObservedVerify = (residual.isMember("observedVerifyScore") && residual["observedVerifyScore"].isNumeric()) ||
                                       (request.isMember("observedVerifyScore") && request["observedVerifyScore"].isNumeric());
        const double predictedVerify = readDouble(residual, "predictedVerifyScore", readDouble(request, "predictedVerifyScore", 0.0));
        const double observedVerify = readDouble(residual, "observedVerifyScore", readDouble(request, "observedVerifyScore", predictedVerify));
        const double verifyResidual = (hasPredictedVerify || hasObservedVerify) ? (observedVerify - predictedVerify) : 0.0;
        const bool accepted = readBool(residual, "accepted", readBool(request, "accepted", true));
        const bool executed = readBool(residual, "executed", readBool(request, "executed", false));
        const bool scheduled = readBool(residual, "scheduled", readBool(request, "scheduled", false));
        const bool willActuate = readBool(residual, "willActuate", readBool(request, "willActuate", executed || scheduled));
        const std::string gateReason = readString(residual, "gateReason", readString(request, "gateReason", std::string()));
        const std::string actionMode = readString(residual, "actionMode", readString(request, "actionMode", std::string("unknown")));
        const std::string powerMode = readString(residual, "powerMode", readString(request, "powerMode", std::string("normal")));
        const std::string motionState = executed ? "executed" : (scheduled ? "scheduled" : (accepted ? "blocked" : "rejected"));

        double residualFeedback = 0.0;
        if (hasPredictedVerify || hasObservedVerify) {
            residualFeedback += std::clamp(verifyResidual * 0.9, -0.6, 0.6);
        }
        if (executed) {
            residualFeedback += 0.2;
        } else if (scheduled) {
            residualFeedback += 0.08;
        } else if (accepted && willActuate) {
            residualFeedback -= 0.35;
        } else if (!accepted && willActuate) {
            residualFeedback -= 0.2;
        }
        if (!gateReason.empty() && gateReason.find("disallow-move") != std::string::npos) {
            residualFeedback += 0.12;
        }
        if (powerMode == "low" || powerMode == "eco") {
            residualFeedback += accepted ? 0.04 : 0.02;
        }
        residualFeedback = std::clamp(residualFeedback, -1.0, 1.0);
        const double effectiveFeedback = std::clamp(hasExplicitFeedback ? feedback * 0.7 + residualFeedback * 0.3 : residualFeedback, -1.0, 1.0);

        learner.alpha = std::clamp(learner.alpha + rate * effectiveFeedback, 0.2, 3.0);
        learner.beta = std::clamp(learner.beta + rate * effectiveFeedback * 0.8, 0.2, 3.0);
        learner.gamma = std::clamp(learner.gamma + rate * effectiveFeedback * 0.6, 0.1, 3.0);
        learner.delta = std::clamp(learner.delta - rate * effectiveFeedback * 0.5, 0.1, 3.0);
        learner.threshold = std::clamp(learner.threshold - rate * effectiveFeedback * 0.2, 0.3, 1.3);

        std::vector<std::string> keywords;
        std::unordered_set<std::string> seenKeywords;
        auto pushKeyword = [&](const std::string &raw) {
            const std::string key = toLowerCopy(raw);
            if (key.empty() || !seenKeywords.insert(key).second) {
                return;
            }
            keywords.push_back(key);
        };

        if (request.isMember("keywords") && request["keywords"].isArray()) {
            for (const auto &k : request["keywords"]) {
                if (k.isString()) {
                    pushKeyword(k.asString());
                }
            }
        }
        pushKeyword("motion_" + motionState);
        pushKeyword("mode_" + actionMode);
        pushKeyword("power_" + powerMode);
        if (!gateReason.empty()) {
            pushKeyword(gateReason);
        }
        if (verifyResidual >= 0.15) {
            pushKeyword("verify_gain");
        } else if (verifyResidual <= -0.15) {
            pushKeyword("verify_drop");
        }
        if (accepted && !executed && !scheduled && willActuate) {
            pushKeyword("actuation_gap");
        }

        for (const auto &key : keywords) {
            session.learnerWeights[key] += effectiveFeedback * rate;
        }

        std::ostringstream residualSummary;
        residualSummary << "mobility-residual state=" << motionState
                        << " mode=" << actionMode
                        << " accepted=" << (accepted ? "true" : "false")
                        << " executed=" << (executed ? "true" : "false")
                        << " power=" << powerMode;
        if (!gateReason.empty()) {
            residualSummary << " gate=" << gateReason;
        }
        if (hasPredictedVerify || hasObservedVerify) {
            residualSummary << " verifyDelta=" << verifyResidual;
        }

        std::ostringstream semanticSummary;
        semanticSummary << "mobility-policy power=" << powerMode;
        if (!gateReason.empty()) {
            semanticSummary << " gate=" << gateReason;
        }

        pushUniqueLimited(session.workingMemory, residualSummary.str(), 24);
        if (std::abs(effectiveFeedback) >= 0.05) {
            pushUniqueLimited(session.episodicMemory, residualSummary.str(), 28);
        }
        pushUniqueLimited(session.semanticMemory, semanticSummary.str(), 48);
        if (hasPredictedVerify || hasObservedVerify) {
            std::ostringstream sensorySummary;
            sensorySummary << "verify predicted=" << predictedVerify << " observed=" << observedVerify << " residual=" << verifyResidual;
            pushUniqueLimited(session.sensoryBuffer, sensorySummary.str(), 24);
        }
        compactMemory(session.workingMemory, 20);
        compactMemory(session.episodicMemory, 24);
        compactMemory(session.semanticMemory, 40);
        compactMemory(session.sensoryBuffer, 20);
        session.tick += 1;
        session.lastTouched = Clock::now();

        Json::Value out;
        out["ok"] = true;
        out["sessionId"] = sessionId;
        out["feedback"] = hasExplicitFeedback ? feedback : effectiveFeedback;
        out["effectiveFeedback"] = effectiveFeedback;
        out["derivedFeedback"] = residualFeedback;
        out["learningRate"] = rate;
        Json::Value params;
        params["alpha"] = learner.alpha;
        params["beta"] = learner.beta;
        params["gamma"] = learner.gamma;
        params["delta"] = learner.delta;
        params["threshold"] = learner.threshold;
        out["params"] = params;
        Json::Value residualAnalysis;
        residualAnalysis["accepted"] = accepted;
        residualAnalysis["executed"] = executed;
        residualAnalysis["scheduled"] = scheduled;
        residualAnalysis["willActuate"] = willActuate;
        residualAnalysis["motionState"] = motionState;
        residualAnalysis["actionMode"] = actionMode;
        residualAnalysis["powerMode"] = powerMode;
        residualAnalysis["gateReason"] = gateReason;
        residualAnalysis["predictedVerifyScore"] = predictedVerify;
        residualAnalysis["observedVerifyScore"] = observedVerify;
        residualAnalysis["verifyResidual"] = verifyResidual;
        out["residualAnalysis"] = residualAnalysis;
        Json::Value keywordJson(Json::arrayValue);
        for (const auto &key : keywords) {
            keywordJson.append(key);
        }
        out["keywords"] = keywordJson;
        Json::Value memory;
        memory["workingFocus"] = stringsToJson(collectRecentMemory(session.workingMemory, 3));
        memory["episodicFocus"] = stringsToJson(collectRecentMemory(session.episodicMemory, 3));
        memory["semanticFocus"] = stringsToJson(collectRecentMemory(session.semanticMemory, 3));
        memory["sensoryFocus"] = stringsToJson(collectRecentMemory(session.sensoryBuffer, 3));
        out["memory"] = memory;
        out["sessionWeightCount"] = static_cast<Json::UInt64>(session.learnerWeights.size());
        return out;
    }
};

V51RuntimeEngine::V51RuntimeEngine()
    : impl_(new Impl()) {}

V51RuntimeEngine::~V51RuntimeEngine() {
    delete impl_;
}

Json::Value V51RuntimeEngine::process(const Json::Value &request) {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->buildProcessResult(request);
}

Json::Value V51RuntimeEngine::learn(const Json::Value &request) {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->buildLearnResult(request);
}

Json::Value V51RuntimeEngine::status(const std::string &sessionId) const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    Json::Value out;
    out["ok"] = true;
    out["sessionCount"] = static_cast<Json::UInt64>(impl_->sessions.size());
    Json::Value params;
    params["alpha"] = impl_->learner.alpha;
    params["beta"] = impl_->learner.beta;
    params["gamma"] = impl_->learner.gamma;
    params["delta"] = impl_->learner.delta;
    params["threshold"] = impl_->learner.threshold;
    out["params"] = params;

    if (!sessionId.empty()) {
        auto it = impl_->sessions.find(sessionId);
        if (it != impl_->sessions.end()) {
            const auto &s = it->second;
            Json::Value session;
            session["sessionId"] = sessionId;
            session["tick"] = static_cast<Json::Int64>(s.tick);
            session["posteriorScore"] = s.lastPosteriorScore;
            session["attentionScale"] = s.lastAttentionScale;
            session["pruningIntensity"] = s.lastPruningIntensity;
            session["adaptiveThreshold"] = s.lastAdaptiveThreshold;
            session["mirrorPassed"] = s.lastMirrorPassed;
            session["short"] = static_cast<Json::UInt64>(s.shortContext.size());
            session["task"] = static_cast<Json::UInt64>(s.taskMemory.size());
            session["long"] = static_cast<Json::UInt64>(s.longMemory.size());
            session["cache"] = static_cast<Json::UInt64>(s.cacheMemory.size());
            Json::Value humanLike;
            humanLike["workingCount"] = static_cast<Json::UInt64>(s.workingMemory.size());
            humanLike["episodicCount"] = static_cast<Json::UInt64>(s.episodicMemory.size());
            humanLike["semanticCount"] = static_cast<Json::UInt64>(s.semanticMemory.size());
            humanLike["sensoryCount"] = static_cast<Json::UInt64>(s.sensoryBuffer.size());
            humanLike["workingFocus"] = stringsToJson(collectRecentMemory(s.workingMemory, 4));
            humanLike["episodicFocus"] = stringsToJson(collectRecentMemory(s.episodicMemory, 4));
            humanLike["semanticFocus"] = stringsToJson(collectRecentMemory(s.semanticMemory, 4));
            humanLike["sensoryFocus"] = stringsToJson(collectRecentMemory(s.sensoryBuffer, 4));
            session["humanLike"] = humanLike;
            session["memorySummary"] = buildHumanLikeMemorySummary(s);
            session["tokenSeen"] = static_cast<Json::UInt64>(s.tokenSeen.size());
            out["session"] = session;
        } else {
            out["session"] = Json::Value(Json::nullValue);
        }
    }
    return out;
}
