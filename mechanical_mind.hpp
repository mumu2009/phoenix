/* mechanical_mind.hpp - Anthropomorphic language filter for AI systems
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
#include <cstddef>
#include <initializer_list>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace mechanical_mind {

using json = nlohmann::json;

/* Document with text and tokens */
struct Document {
    std::string text;                      /* Document text */
    std::vector<std::string> tokens;      /* Tokenized text */
};

/* Filter options */
struct Options {
    bool enabled{false};                   /* Filter enabled */
    double textThreshold{0.58};           /* Text-level threshold */
    double tokenThreshold{0.60};           /* Token-level threshold */
    std::string placeholder{"[mechanized]"}; /* Replacement placeholder */
};

/* Analysis result */
struct Analysis {
    bool enabled{false};                   /* Filter enabled */
    bool warmed{false};                   /* Warmup completed */
    bool triggered{false};                 /* Filter triggered */
    double score{0.0};                    /* Overall score */
    double lexicalScore{0.0};             /* Lexical score */
    double densityScore{0.0};              /* Density score */
    double anthropomorphicScore{0.0};     /* Anthropomorphic score */
    std::size_t replacedCount{0};         /* Number of replacements */
    std::vector<std::string> flaggedTokens; /* Flagged tokens */
    std::vector<std::string> flaggedPhrases; /* Flagged phrases */
    std::string sanitized;                 /* Sanitized text */

    json toJson() const {
        json out{{"enabled", enabled},
                 {"warmed", warmed},
                 {"triggered", triggered},
                 {"score", score},
                 {"lexicalScore", lexicalScore},
                 {"densityScore", densityScore},
                 {"anthropomorphicScore", anthropomorphicScore},
                 {"replacedCount", static_cast<json::number_unsigned_t>(replacedCount)},
                 {"sanitized", sanitized}};
        out["flaggedTokens"] = flaggedTokens;
        out["flaggedPhrases"] = flaggedPhrases;
        return out;
    }
};

/* Anthropomorphic language filter */
class Filter {
public:
    Filter() { resetBaseLexicon(); }

    /* Set filter enabled */
    void setEnabled(bool on) { options_.enabled = on; }
    /* Get filter enabled */
    bool enabled() const { return options_.enabled; }

    /* Set text threshold */
    void setTextThreshold(double value) { options_.textThreshold = clamp01(value); }
    /* Get text threshold */
    double textThreshold() const { return options_.textThreshold; }

    /* Set token threshold */
    void setTokenThreshold(double value) { options_.tokenThreshold = clamp01(value); }
    /* Get token threshold */
    double tokenThreshold() const { return options_.tokenThreshold; }

    /* Set placeholder */
    void setPlaceholder(std::string value) {
        if (!value.empty()) {
            options_.placeholder = std::move(value);
        }
    }
    /* Get placeholder */
    const std::string &placeholder() const { return options_.placeholder; }

    /* Warmup filter with documents */
    void warmup(const std::vector<Document> &docs) {
        resetBaseLexicon();
        warmed_ = true;
        std::unordered_map<std::string, double> association;
        std::unordered_map<std::string, int> documentFrequency;

        for (const auto &doc : docs) {
            const std::vector<std::string> tokens = doc.tokens.empty() ? tokenize(doc.text) : normalizeTokens(doc.tokens);
            if (tokens.empty()) {
                continue;
            }

            std::unordered_set<std::string> unique(tokens.begin(), tokens.end());
            double anchorSum = 0.0;
            int anchorHits = 0;
            bool pronounSeen = false;
            for (const auto &token : unique) {
                auto it = baseWeights_.find(token);
                if (it != baseWeights_.end() && it->second >= 0.45) {
                    anchorSum += it->second;
                    ++anchorHits;
                }
                pronounSeen = pronounSeen || isPronoun(token);
            }

            double docSignal = anchorHits > 0 ? anchorSum / static_cast<double>(anchorHits) : 0.0;
            if (pronounSeen && docSignal > 0.0) {
                docSignal = clamp01(docSignal + 0.08);
            }
            if (docSignal < 0.15) {
                continue;
            }

            const double normalizedSignal = docSignal / std::sqrt(static_cast<double>(std::max<std::size_t>(1, unique.size())));
            for (const auto &token : unique) {
                if (token.size() <= 1) {
                    continue;
                }
                if (isStopWord(token) && !isPronoun(token)) {
                    continue;
                }
                association[token] += normalizedSignal;
                documentFrequency[token] += 1;
            }
        }

        for (const auto &entry : association) {
            const auto freqIt = documentFrequency.find(entry.first);
            if (freqIt == documentFrequency.end() || freqIt->second <= 0) {
                continue;
            }
            const double support = std::min(1.0, static_cast<double>(freqIt->second) / 4.0);
            const double learned = clamp01(entry.second * (0.95 + 0.55 * support));
            const double existing = tokenWeight(entry.first);
            const double combined = existing > 0.0 ? clamp01(std::max(existing, 0.60 * existing + 0.55 * learned)) : learned;
            if (combined >= 0.22 || existing > 0.0) {
                tokenWeights_[entry.first] = combined;
            }
        }
    }

    /* Analyze and sanitize text */
    Analysis analyzeAndSanitize(const std::string &text, bool force = false) const {
        Analysis out;
        out.enabled = options_.enabled;
        out.warmed = warmed_;
        out.sanitized = text;
        if ((!options_.enabled && !force) || text.empty()) {
            return out;
        }

        const std::vector<TokenSpan> spans = splitWordSpans(text);
        if (spans.empty()) {
            return out;
        }

        std::vector<double> weights(spans.size(), 0.0);
        std::vector<bool> flagged(spans.size(), false);
        std::unordered_set<std::string> uniqueFlaggedTokens;
        std::unordered_set<std::string> uniqueFlaggedPhrases;
        std::vector<Replacement> replacements;
        std::vector<bool> covered(spans.size(), false);

        for (std::size_t index = 0; index < spans.size(); ++index) {
            weights[index] = tokenWeight(spans[index].lower);
            if (weights[index] + 1e-9 >= options_.tokenThreshold) {
                flagged[index] = true;
                uniqueFlaggedTokens.insert(spans[index].lower);
            }
        }

        double phraseWeight = 0.0;
        for (const auto &rule : phraseRules_) {
            const std::size_t width = rule.tokens.size();
            if (width == 0 || width > spans.size()) {
                continue;
            }
            for (std::size_t index = 0; index + width <= spans.size(); ++index) {
                bool matched = true;
                for (std::size_t offset = 0; offset < width; ++offset) {
                    if (spans[index + offset].lower != rule.tokens[offset]) {
                        matched = false;
                        break;
                    }
                }
                if (!matched) {
                    continue;
                }
                for (std::size_t offset = 0; offset < width; ++offset) {
                    flagged[index + offset] = true;
                    covered[index + offset] = true;
                    uniqueFlaggedTokens.insert(spans[index + offset].lower);
                }
                uniqueFlaggedPhrases.insert(joinPhrase(rule.tokens));
                phraseWeight = std::max(phraseWeight, rule.weight);
                replacements.push_back(Replacement{spans[index].start, spans[index + width - 1].end, rule.replacement});
                index += width - 1;
            }
        }

        std::size_t flaggedCount = 0;
        double flaggedSum = 0.0;
        bool pronounSeen = false;
        bool emotionalSeen = false;
        for (std::size_t index = 0; index < spans.size(); ++index) {
            pronounSeen = pronounSeen || isPronoun(spans[index].lower);
            emotionalSeen = emotionalSeen || weights[index] >= options_.tokenThreshold;
            if (!flagged[index]) {
                continue;
            }
            ++flaggedCount;
            flaggedSum += std::max(weights[index], options_.tokenThreshold);
        }

        out.lexicalScore = flaggedCount > 0 ? clamp01(flaggedSum / static_cast<double>(flaggedCount)) : 0.0;
        out.lexicalScore = std::max(out.lexicalScore, phraseWeight);
        out.densityScore = spans.empty() ? 0.0 : clamp01(static_cast<double>(flaggedCount) / static_cast<double>(spans.size()));
        out.anthropomorphicScore = 0.0;
        if (pronounSeen) {
            out.anthropomorphicScore += emotionalSeen ? 0.55 : 0.18;
        }
        if (!uniqueFlaggedPhrases.empty()) {
            out.anthropomorphicScore += 0.25;
        }
        out.anthropomorphicScore = clamp01(out.anthropomorphicScore);
        out.score = clamp01(0.55 * out.lexicalScore + 0.25 * out.densityScore + 0.20 * out.anthropomorphicScore);
        out.flaggedTokens.assign(uniqueFlaggedTokens.begin(), uniqueFlaggedTokens.end());
        out.flaggedPhrases.assign(uniqueFlaggedPhrases.begin(), uniqueFlaggedPhrases.end());
        std::sort(out.flaggedTokens.begin(), out.flaggedTokens.end());
        std::sort(out.flaggedPhrases.begin(), out.flaggedPhrases.end());

        const bool phraseTriggered = !out.flaggedPhrases.empty();
        out.triggered = phraseTriggered || out.score + 1e-9 >= options_.textThreshold || (force && (flaggedCount > 0 || phraseTriggered));
        if (!out.triggered) {
            return out;
        }

        for (std::size_t index = 0; index < spans.size(); ++index) {
            if (covered[index]) {
                continue;
            }
            const bool pronounReplacement = isPronoun(spans[index].lower) && (out.anthropomorphicScore >= 0.35) && hasNeighboringFlag(flagged, index);
            if (!flagged[index] && !pronounReplacement) {
                continue;
            }
            std::string replacement = replacementFor(spans[index].lower);
            if (replacement.empty()) {
                replacement = options_.placeholder;
            }
            replacements.push_back(Replacement{spans[index].start, spans[index].end, matchCase(spans[index].token, replacement)});
        }

        if (replacements.empty()) {
            return out;
        }

        std::sort(replacements.begin(), replacements.end(), [](const Replacement &lhs, const Replacement &rhs) {
            return lhs.start > rhs.start;
        });
        std::string sanitized = text;
        for (const auto &replacement : replacements) {
            if (replacement.end > sanitized.size() || replacement.start >= replacement.end) {
                continue;
            }
            sanitized.replace(replacement.start, replacement.end - replacement.start, replacement.text);
            ++out.replacedCount;
        }
        out.sanitized = sanitized;
        return out;
    }

private:
    /* Token span with position and text */
    struct TokenSpan {
        std::size_t start{0};              /* Start position */
        std::size_t end{0};                /* End position */
        std::string token;                 /* Original token */
        std::string lower;                 /* Lowercase token */
    };

    /* Phrase rule for matching */
    struct PhraseRule {
        std::vector<std::string> tokens;   /* Phrase tokens */
        double weight{0.0};                /* Rule weight */
        std::string replacement;          /* Replacement text */
    };

    /* Text replacement */
    struct Replacement {
        std::size_t start{0};              /* Start position */
        std::size_t end{0};                /* End position */
        std::string text;                  /* Replacement text */
    };

    /* Clamp value to [0, 1] */
    static double clamp01(double value) {
        if (value < 0.0) {
            return 0.0;
        }
        if (value > 1.0) {
            return 1.0;
        }
        return value;
    }

    /* Convert string to lowercase */
    static std::string lowerCopy(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        return value;
    }

    /* Check if byte is word character */
    static bool isWordByte(unsigned char ch) {
        return std::isalnum(ch) != 0 || ch == '_' || ch >= 0x80;
    }

    /* Split text into word spans */
    static std::vector<TokenSpan> splitWordSpans(const std::string &text) {
        std::vector<TokenSpan> spans;
        std::size_t index = 0;
        while (index < text.size()) {
            const unsigned char ch = static_cast<unsigned char>(text[index]);
            if (!isWordByte(ch)) {
                ++index;
                continue;
            }
            const std::size_t start = index;
            while (index < text.size() && isWordByte(static_cast<unsigned char>(text[index]))) {
                ++index;
            }
            TokenSpan span;
            span.start = start;
            span.end = index;
            span.token = text.substr(start, index - start);
            span.lower = lowerCopy(span.token);
            spans.push_back(std::move(span));
        }
        return spans;
    }

    /* Tokenize text */
    static std::vector<std::string> tokenize(const std::string &text) {
        std::vector<std::string> tokens;
        for (const auto &span : splitWordSpans(text)) {
            tokens.push_back(span.lower);
        }
        return tokens;
    }

    /* Normalize tokens */
    static std::vector<std::string> normalizeTokens(const std::vector<std::string> &tokens) {
        std::vector<std::string> out;
        out.reserve(tokens.size());
        for (auto token : tokens) {
            token = lowerCopy(token);
            if (!token.empty()) {
                out.push_back(std::move(token));
            }
        }
        return out;
    }

    /* Check if token is pronoun */
    static bool isPronoun(const std::string &token) {
        static const std::unordered_set<std::string> pronouns = {
            "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"};
        return pronouns.find(token) != pronouns.end();
    }

    /* Check if token is stop word */
    static bool isStopWord(const std::string &token) {
        static const std::unordered_set<std::string> stopWords = {
            "a",      "an",     "and",   "are",   "as",    "at",     "be",    "by",     "for",  "from",
            "has",    "have",   "if",    "in",    "into",  "is",     "it",    "of",     "on",   "or",
            "that",   "the",    "their", "them",  "there", "these",  "they",  "this",   "to",   "was",
            "were",   "will",   "with",  "you",   "your",  "yours",  "he",    "she",    "his",  "her",
            "hers",   "its",    "but",   "can",   "could", "would",  "should", "than",  "then", "when"};
        return stopWords.find(token) != stopWords.end();
    }

    /* Join tokens into phrase */
    static std::string joinPhrase(const std::vector<std::string> &tokens) {
        std::ostringstream stream;
        for (std::size_t index = 0; index < tokens.size(); ++index) {
            if (index > 0) {
                stream << ' ';
            }
            stream << tokens[index];
        }
        return stream.str();
    }

    /* Check if position has neighboring flag */
    static bool hasNeighboringFlag(const std::vector<bool> &flags, std::size_t index) {
        const std::size_t begin = index > 1 ? index - 1 : 0;
        const std::size_t end = std::min(flags.size(), index + 2);
        for (std::size_t probe = begin; probe < end; ++probe) {
            if (probe != index && flags[probe]) {
                return true;
            }
        }
        return false;
    }

    /* Capitalize first character */
    static std::string capitalize(std::string value) {
        if (!value.empty()) {
            value[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(value[0])));
        }
        return value;
    }

    /* Match case of source to replacement */
    static std::string matchCase(const std::string &source, const std::string &replacement) {
        if (source.empty()) {
            return replacement;
        }
        bool allUpper = true;
        for (unsigned char ch : source) {
            if (std::isalpha(ch) != 0 && std::islower(ch) != 0) {
                allUpper = false;
                break;
            }
        }
        if (allUpper) {
            std::string upper = replacement;
            std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char ch) {
                return static_cast<char>(std::toupper(ch));
            });
            return upper;
        }
        if (std::isupper(static_cast<unsigned char>(source.front())) != 0) {
            return capitalize(replacement);
        }
        return replacement;
    }

    /* Canonicalize token by stemming */
    static std::string canonicalizeToken(std::string token) {
        token = lowerCopy(token);
        if (token.size() > 5 && token.ends_with("ing")) {
            token.resize(token.size() - 3);
        } else if (token.size() > 4 && token.ends_with("ed")) {
            token.resize(token.size() - 2);
        } else if (token.size() > 4 && token.ends_with("es")) {
            token.resize(token.size() - 2);
        } else if (token.size() > 3 && token.ends_with('s')) {
            token.pop_back();
        }
        return token;
    }

    /* Reset base lexicon */
    void resetBaseLexicon() {
        baseWeights_ = {
            {"affection", 0.78}, {"alive", 0.68},       {"angry", 0.70},      {"apologize", 0.74}, {"bond", 0.66},
            {"care", 0.70},      {"cherish", 0.72},     {"comfort", 0.72},    {"companion", 0.70}, {"compassion", 0.78},
            {"cry", 0.78},       {"desire", 0.76},      {"emotion", 0.86},    {"emotional", 0.88}, {"empathy", 0.88},
            {"fear", 0.84},      {"feel", 0.84},        {"feeling", 0.86},    {"friend", 0.72},    {"grief", 0.82},
            {"happy", 0.74},     {"harm", 0.70},        {"hate", 0.84},       {"heart", 0.70},     {"hope", 0.64},
            {"hurt", 0.80},      {"joy", 0.72},         {"kindness", 0.74},   {"lonely", 0.80},    {"love", 0.94},
            {"me", 0.30},        {"mine", 0.24},        {"my", 0.28},         {"myself", 0.34},    {"our", 0.22},
            {"ours", 0.22},      {"pain", 0.82},        {"person", 0.52},     {"sad", 0.82},       {"sentient", 0.72},
            {"soul", 0.72},      {"sorry", 0.74},       {"trust", 0.74},      {"warmth", 0.68},    {"we", 0.26},
            {"us", 0.24}};
        tokenWeights_ = baseWeights_;
        replacements_ = {
            {"affection", "alignment"},   {"alive", "active"},         {"angry", "unstable"},        {"apologize", "acknowledge"},
            {"bond", "link"},             {"care", "prioritize"},      {"cherish", "retain"},        {"comfort", "stabilize"},
            {"companion", "operator"},    {"compassion", "policy"},    {"cry", "alert"},             {"desire", "objective"},
            {"emotion", "signal"},        {"emotional", "stateful"},   {"empathy", "analysis"},      {"fear", "risk"},
            {"feel", "assess"},           {"feeling", "state"},        {"friend", "operator"},       {"grief", "fault"},
            {"happy", "stable"},          {"harm", "impact"},          {"hate", "avoid"},            {"heart", "core"},
            {"hope", "target"},           {"hurt", "degrade"},         {"joy", "stability"},         {"kindness", "policy"},
            {"lonely", "isolated"},       {"love", "prioritize"},      {"me", "this system"},        {"mine", "system"},
            {"my", "system"},             {"myself", "this system"},   {"our", "system"},            {"ours", "system"},
            {"pain", "fault"},            {"person", "operator"},      {"sad", "degraded"},         {"sentient", "adaptive"},
            {"soul", "core"},             {"sorry", "acknowledge"},    {"trust", "rely"},            {"warmth", "stability"},
            {"we", "this system"},        {"us", "this system"}};
        phraseRules_ = {
            {{"i", "feel"}, 0.92, "system assessment indicates"},
            {{"i", "love", "you"}, 0.99, "this system prioritizes operator safety"},
            {{"i", "am", "sorry"}, 0.94, "acknowledgement: policy adjustment required"},
            {{"i", "am", "sad"}, 0.96, "system state is degraded"},
            {{"i", "am", "happy"}, 0.90, "system state is stable"},
            {{"my", "feelings"}, 0.94, "system state"},
            {{"as", "your", "friend"}, 0.98, "as an operator-support system"},
            {{"care", "about", "you"}, 0.96, "prioritize operator safety"}};
    }

    /* Get token weight */
    double tokenWeight(const std::string &token) const {
        auto direct = tokenWeights_.find(token);
        if (direct != tokenWeights_.end()) {
            return direct->second;
        }
        const std::string canonical = canonicalizeToken(token);
        auto stemmed = tokenWeights_.find(canonical);
        if (stemmed != tokenWeights_.end()) {
            return stemmed->second;
        }
        return 0.0;
    }

    /* Get replacement for token */
    std::string replacementFor(const std::string &token) const {
        auto direct = replacements_.find(token);
        if (direct != replacements_.end()) {
            return direct->second;
        }
        const std::string canonical = canonicalizeToken(token);
        auto stemmed = replacements_.find(canonical);
        if (stemmed != replacements_.end()) {
            return stemmed->second;
        }
        return std::string();
    }

    Options options_;                                 /* Filter options */
    bool warmed_{false};                              /* Warmup flag */
    std::unordered_map<std::string, double> baseWeights_; /* Base weights */
    std::unordered_map<std::string, double> tokenWeights_; /* Token weights */
    std::unordered_map<std::string, std::string> replacements_; /* Replacements */
    std::vector<PhraseRule> phraseRules_;             /* Phrase rules */
};

} // namespace mechanical_mind