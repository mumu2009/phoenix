/* plugin_semantics.hpp - memetic-semantic plugin intent matching.

   WHY
   ---
   Plugin selection used to be word recognition: each addon ran its own
   keyword/tally heuristic in consider() and the host simply took the highest
   tally (addon.cpp collectOffers: `if (score < minScore) continue;`).  A
   plugin that genuinely matched the *situation* but used different wording
   could therefore never be selected - the classic "词语驱动" failure.

   This layer decides the offer in two channels instead:

     lexical : Jaccard over stopword-filtered content tokens (order-free,
               wording-insensitive up to synonymy);
     memetic : relatedness between the memes of the situation and the memes of
               the plugin's declared intent, scored by the caller's meme graph
               (`related(a,b)`).  This is what lets "prove the plan is
               consistent" reach a plugin described as "first-order logic
               prover": the graph knows consistency ~ logic, the words do not.

   The plugin's own score stays a feature (it encodes the plugin's private
   confidence), but it no longer has veto power: it is multiplied by the
   semantic evidence instead of gating the offer.

   Everything here is pure: no globals, no locking, no I/O, so it is unit
   tested directly (tests/gtest/unit/addons/test_plugin_semantics.cpp).
*/
#pragma once

#include <algorithm>
#include <cctype>
#include <functional>
#include <string>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

namespace phoenix {
namespace plugin {

using json = nlohmann::json;

inline std::string lowerAscii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

/* Content tokens.  UTF-8 bytes >= 0x80 are kept (Chinese words have no ASCII
   word boundaries, so each run of such bytes becomes one token). */
inline std::vector<std::string> semanticTokens(const std::string &text) {
    std::vector<std::string> out;
    std::string cur;
    for (unsigned char c : text) {
        if (std::isalnum(c) || c == '_' || c == '-') {
            cur.push_back(static_cast<char>(std::tolower(c)));
        } else if (c >= 0x80) {
            cur.push_back(static_cast<char>(c));
        } else {
            if (!cur.empty()) {
                out.push_back(cur);
                cur.clear();
            }
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

inline bool semanticStopWord(const std::string &w) {
    static const std::unordered_set<std::string> kStop = {
        "the",  "and",  "for",  "that",  "this",  "with", "from",  "are",
        "was",  "were", "been", "have",   "not",   "but",  "you",   "your",
        "into", "about","would","could",  "should","there","their", "which",
        "when", "where","what", "will",   "shall", "than", "then",  "also",
        "only", "such", "some", "any",    "all",   "can",  "may",   "a",
        "an",   "of",   "to",   "in",     "on",    "is",   "it",    "as",
        "at",   "by",   "or",   "be",     "we",    "i",    "if",    "so"};
    return kStop.count(w) > 0;
}

inline std::unordered_set<std::string>
contentTokens(const std::string &text) {
    std::unordered_set<std::string> out;
    for (const auto &w : semanticTokens(text)) {
        if (w.size() < 2) continue;
        if (semanticStopWord(w)) continue;
        out.insert(w);
    }
    return out;
}

/* Jaccard over content tokens.  Both empty => 0 (no evidence either way),
   unlike memetic's near-duplicate helper which treats empty==empty as 1. */
inline double lexicalAffinity(const std::string &situation,
                              const std::string &profile) {
    const auto a = contentTokens(situation);
    const auto b = contentTokens(profile);
    if (a.empty() || b.empty()) return 0.0;
    int inter = 0;
    for (const auto &w : a)
        if (b.count(w)) ++inter;
    const int uni = static_cast<int>(a.size() + b.size()) - inter;
    if (uni <= 0) return 0.0;
    return static_cast<double>(inter) / static_cast<double>(uni);
}

/* Graph relatedness between the two meme sets: mean over the strongest
   partner of every situation meme (so one decisive link is enough, and a
   large profile cannot dilute it). */
inline double memeticAffinity(
    const std::string &situation, const std::string &profile,
    const std::function<double(const std::string &, const std::string &)>
        &related) {
    if (!related) return 0.0;
    const auto a = contentTokens(situation);
    const auto b = contentTokens(profile);
    if (a.empty() || b.empty()) return 0.0;
    double sum = 0.0;
    int n = 0;
    for (const auto &x : a) {
        double best = 0.0;
        for (const auto &y : b) {
            if (x == y) {
                best = 1.0;
                break;
            }
            double r = 0.0;
            try {
                r = related(x, y);
            } catch (...) {
                r = 0.0;
            }
            if (!(r >= 0.0) || r > 1.0) r = 0.0; /* NaN / out-of-range */
            if (r > best) best = r;
        }
        sum += best;
        ++n;
    }
    if (n == 0) return 0.0;
    return sum / static_cast<double>(n);
}

/* Combined semantic evidence in [0,1].  memeticWeight = 0 reproduces the
   lexical-only behaviour (the pre-v10.2 baseline). */
inline double semanticAffinity(
    const std::string &situation, const std::string &profile,
    const std::function<double(const std::string &, const std::string &)>
        &related = nullptr,
    double memeticWeight = 0.5) {
    if (!(memeticWeight >= 0.0) || memeticWeight > 1.0) memeticWeight = 0.5;
    const double lex = lexicalAffinity(situation, profile);
    const double mem = related ? memeticAffinity(situation, profile, related)
                              : 0.0;
    /* Without a graph the lexical channel carries the whole decision. */
    const double w = related ? memeticWeight : 0.0;
    return std::max(0.0, std::min(1.0, (1.0 - w) * lex + w * mem));
}

/* The plugin's declared intent as one text blob: name + type + its own
   reason plus any tag/keyword arrays it publishes in meta. */
inline std::string pluginProfile(const std::string &name,
                                 const std::string &type,
                                 const std::string &reason,
                                 const json &meta = json::object()) {
    std::string out = name;
    out += ' ';
    out += type;
    if (!reason.empty()) {
        out += ' ';
        out += reason;
    }
    auto appendList = [&out](const json &v) {
        if (!v.is_array()) return;
        for (const auto &e : v) {
            if (e.is_string()) {
                out += ' ';
                out += e.get<std::string>();
            }
        }
    };
    if (meta.is_object()) {
        for (const char *k : {"tags", "tag", "keywords", "keyword", "topics",
                              "description", "intent"}) {
            if (meta.contains(k)) {
                if (meta[k].is_string()) {
                    out += ' ';
                    out += meta[k].get<std::string>();
                } else {
                    appendList(meta[k]);
                }
            }
        }
    }
    return out;
}

/* Final offer score.

     final = pluginScore * (0.5 + 0.5 * semantic)

   The plugin's own confidence still matters (a plugin that says "I'm sure"
   is never ignored), but it can no longer override the semantic evidence: no
   semantic overlap halves the score rather than deleting the offer, and full
   semantic overlap keeps it at full strength.  This is deliberately fail-open
   - the semantic channel is approximate, so it re-ranks, it does not veto. */
inline double blendOfferScore(double pluginScore, double semantic,
                              double semanticWeight = 0.5) {
    if (!(pluginScore >= 0.0) || pluginScore > 1.0) pluginScore = 0.0;
    if (!(semantic >= 0.0) || semantic > 1.0) semantic = 0.0;
    if (!(semanticWeight >= 0.0) || semanticWeight > 1.0) semanticWeight = 0.5;
    const double boost = (1.0 - semanticWeight) + semanticWeight * semantic;
    return pluginScore * boost;
}

}  // namespace plugin
}  // namespace phoenix
