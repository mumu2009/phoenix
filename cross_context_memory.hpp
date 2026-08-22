/* cross_context_memory.hpp - explicit cross-context long-term memory (v8.x)
   Copyright (C) 2026 079 Project

   Context isolation rule (see doc/v8.3/context_isolation.md):
   - PER-CONTEXT state (workspace, deliverable, plan, sensations, appraisal)
     is strictly separated by contextTag (mission:<id> / chat:<session>);
   - CROSS-CONTEXT memory (this module + GNN graph + mission experience)
     is the ONE place contexts may exchange knowledge.  Chat turns and
     finished missions deposit summaries here; new contexts recall the
     most similar entries and inject them into their prompts, so every
     context starts with what the system already learned, without ever
     leaking live context into another.

   Storage: one JSON file (default runtime_store/cross_context_memory.json),
   capped at 500 entries.  Retrieval: word-set overlap (cheap, no external
   embedding dependency - safe for header-only / gtest compilation). */
#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace phoenix {
namespace memory {

struct CcmEntry {
    std::string sourceTag;  /* "chat:<session>" or "mission:<id>" */
    std::string text;       /* the deposited summary */
    uint64_t atMs{0};
};

inline std::vector<CcmEntry> ccmLoad(const std::string &storePath) {
    std::vector<CcmEntry> out;
    std::ifstream in(storePath);
    if (!in) return out;
    nlohmann::json j;
    in >> j;
    if (!j.is_array()) return out;
    for (const auto &e : j) {
        if (!e.is_object()) continue;
        out.push_back({e.value("sourceTag", std::string()),
                       e.value("text", std::string()),
                       e.value("atMs", 0ull)});
    }
    return out;
}

inline void ccmSave(const std::string &storePath,
                    const std::vector<CcmEntry> &entries) {
    std::filesystem::create_directories(
        std::filesystem::path(storePath).parent_path());
    nlohmann::json arr = nlohmann::json::array();
    for (const auto &e : entries) {
        arr.push_back(nlohmann::json{{"sourceTag", e.sourceTag},
                                     {"text", e.text},
                                     {"atMs", e.atMs}});
    }
    std::ofstream out(storePath);
    out << arr.dump(2);
}

/* Word-set overlap between two texts (same cheap metric as the mission
   experience store). */
inline double ccmOverlap(const std::string &a, const std::string &b) {
    auto words = [](const std::string &s) {
        std::vector<std::string> w;
        std::string cur;
        for (const char c : s) {
            const bool alpha = (c >= 'a' && c <= 'z') ||
                               (c >= 'A' && c <= 'Z') ||
                               (c >= '0' && c <= '9');
            if (alpha) {
                cur.push_back(static_cast<char>(::tolower(c)));
            } else if (!cur.empty()) {
                w.push_back(cur);
                cur.clear();
            }
        }
        if (!cur.empty()) w.push_back(cur);
        return w;
    };
    auto va = words(a);
    auto vb = words(b);
    if (va.empty() || vb.empty()) return 0.0;
    std::sort(va.begin(), va.end());
    std::sort(vb.begin(), vb.end());
    size_t i = 0, j = 0, inter = 0;
    while (i < va.size() && j < vb.size()) {
        if (va[i] == vb[j]) { ++inter; ++i; ++j; }
        else if (va[i] < vb[j]) ++i;
        else ++j;
    }
    const size_t un = va.size() + vb.size() - inter;
    return un == 0 ? 0.0 : static_cast<double>(inter) / static_cast<double>(un);
}

/* Deposit one cross-context memory entry (cap 500, newest kept).  Callers:
   chat pipeline after each turn, mission done branch after completion. */
inline void ccmRemember(const std::string &storePath,
                        const std::string &sourceTag,
                        const std::string &text) {
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    auto entries = ccmLoad(storePath);
    entries.push_back({sourceTag, text, 0});
    if (entries.size() > 500) entries.erase(entries.begin());
    ccmSave(storePath, entries);
}

/* Recall the top-k most similar entries for a query (excludes nothing -
   cross-context by design; the caller decides how to frame it). */
inline std::vector<CcmEntry> ccmRecall(const std::string &storePath,
                                       const std::string &query, size_t k = 3) {
    const auto entries = ccmLoad(storePath);
    std::vector<std::pair<double, CcmEntry>> scored;
    for (const auto &e : entries)
        scored.push_back({ccmOverlap(query, e.text), e});
    std::sort(scored.begin(), scored.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });
    std::vector<CcmEntry> out;
    for (size_t i = 0; i < scored.size() && i < k; ++i) {
        if (scored[i].first <= 0.05) break;
        out.push_back(scored[i].second);
    }
    return out;
}

}  // namespace memory
}  // namespace phoenix
