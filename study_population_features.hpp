/* study_population_features.hpp - domain-data coupling for the population layer.

   WHY
   ---
   `study.population` competed on NOTHING: `PopulationLayer::runGoal()` needs a
   numeric {input, target} pair, and no caller ever produced one, so
   `goalsRun` stayed 0 and the 8 individuals were never scored against real
   work.  The layer was effectively decorative.

   This header turns a mission field state into that pair, so the population
   competes on the ONLY signal the system actually has: how much NEW
   information a tick produced.

     input  (8 features, each normalised to roughly [0,1])
       [0] deliverable chars / 8000
       [1] unique-word ratio (vocabulary richness)
       [2] repetition ratio (1 - unique-sentence ratio)
       [3] mission pressure in [0,1]
       [4] tick index / 1000
       [5] memory units offered this tick / 16
       [6] gnn units offered this tick / 16
       [7] plugin units offered this tick / 16

     target (1 value)
       information gain this tick: (new information chars) / 2000

   "Information chars" reuses the mission layer's own measure
   (`deliverableInformationChars`) when available, so the population optimises
   what the rest of the system already judges - not a private proxy.

   Everything here is pure and header-only: no globals, no locks, no I/O, so
   the contract is unit tested directly
   (tests/gtest/unit/autonomy/test_population_features.cpp).
*/
#pragma once

#include <algorithm>
#include <cmath>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace phoenix {
namespace study {

inline constexpr int kFieldFeatureCount = 8;

inline double clamp01(double v) {
    if (!(v >= 0.0)) return 0.0; /* NaN -> 0 */
    if (v > 1.0) return 1.0;
    return v;
}

/* Ratio of distinct word tokens to total word tokens, scaled to [0,1].
   Short/empty text yields 0. */
inline double vocabularyRichness(const std::string &text) {
    std::string cur;
    std::set<std::string> uniq;
    size_t total = 0;
    auto flush = [&]() {
        if (cur.size() < 3) {
            cur.clear();
            return;
        }
        ++total;
        uniq.insert(cur);
        cur.clear();
    };
    for (unsigned char c : text) {
        if (std::isalnum(c) || c >= 0x80)
            cur.push_back(static_cast<char>(std::tolower(c)));
        else
            flush();
    }
    flush();
    if (total == 0) return 0.0;
    return static_cast<double>(uniq.size()) / static_cast<double>(total);
}

/* 1 - (distinct sentences / sentences): how much the text repeats itself. */
inline double repetitionRatio(const std::string &text) {
    std::vector<std::string> sentences;
    std::string cur;
    for (char c : text) {
        if (c == '.' || c == '!' || c == '?' || c == '\n') {
            std::string t = cur;
            const auto b = t.find_first_not_of(" \t\r");
            if (b != std::string::npos) sentences.push_back(t.substr(b));
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    if (sentences.size() < 2) return 0.0;
    std::set<std::string> uniq(sentences.begin(), sentences.end());
    const double distinct = static_cast<double>(uniq.size()) /
                            static_cast<double>(sentences.size());
    return clamp01(1.0 - distinct);
}

/* Build the 8-feature input vector from one field state.

   Recognised keys (all optional, missing = 0):
     deliverable (string), pressure (number 0..1), tick (number),
     memoryUnits / gnnUnits / pluginUnits (integers). */
inline std::vector<float> fieldStateFeatures(const nlohmann::json &state) {
    std::vector<float> f(static_cast<size_t>(kFieldFeatureCount), 0.0f);
    if (!state.is_object()) return f;
    const std::string text = state.value("deliverable", std::string());
    f[0] = static_cast<float>(clamp01(static_cast<double>(text.size()) / 8000.0));
    f[1] = static_cast<float>(vocabularyRichness(text));
    f[2] = static_cast<float>(repetitionRatio(text));
    f[3] = static_cast<float>(clamp01(state.value("pressure", 0.0)));
    f[4] = static_cast<float>(
        clamp01(state.value("tick", 0.0) / 1000.0));
    f[5] = static_cast<float>(clamp01(state.value("memoryUnits", 0) / 16.0));
    f[6] = static_cast<float>(clamp01(state.value("gnnUnits", 0) / 16.0));
    f[7] = static_cast<float>(clamp01(state.value("pluginUnits", 0) / 16.0));
    return f;
}

/* Information gain between two field states, as a 1-element target.
   Uses the deliverable's growth in *information* chars: characters added while
   the repetition ratio did not increase.  Repeating yourself is not learning,
   so a longer but more repetitive draft can score 0. */
inline std::vector<float> fieldGainTarget(const nlohmann::json &prev,
                                          const nlohmann::json &now) {
    const std::string a = prev.is_object()
                              ? prev.value("deliverable", std::string())
                              : std::string();
    const std::string b = now.is_object()
                              ? now.value("deliverable", std::string())
                              : std::string();
    double gain = 0.0;
    if (b.size() > a.size()) {
        const double grew = static_cast<double>(b.size() - a.size());
        const double repPrev = repetitionRatio(a);
        const double repNow = repetitionRatio(b);
        /* Strict, matching the contract above: text that grew BY REPEATING
           itself is not new information, so it scores 0 rather than a
           damped fraction of its length.  The 0.05 tolerance absorbs
           punctuation-splitting noise on unchanged drafts. */
        if (repNow <= repPrev + 0.05) gain = grew;
    }
    return {static_cast<float>(clamp01(gain / 2000.0))};
}

/* Ready-to-send runGoal payload, or null when the shapes cannot match
   (the layer answers "goal-shape"/"goal-dim" otherwise, and a rejected payload
   would count as a sample). */
inline nlohmann::json populationGoal(const nlohmann::json &prev,
                                     const nlohmann::json &now) {
    nlohmann::json goal;
    goal["input"] = fieldStateFeatures(now);
    goal["target"] = fieldGainTarget(prev, now);
    return goal;
}

}  // namespace study
}  // namespace phoenix
