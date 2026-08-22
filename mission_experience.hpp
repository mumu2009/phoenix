/* mission_experience.hpp - long-term mission experience store (v8.x B2)
   Copyright (C) 2026 079 Project

   Every completed mission deposits a structural summary (section titles +
   a short excerpt).  A new mission retrieves the most similar past
   experiences and injects them into the deliberator's STATIC prefix, so
   tasks of the same kind start faster and better (the data source for
   autonomous improvement).  Storage is one JSON file, capped at 200
   entries - no external DB dependency. */
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
namespace mission {

struct ExperienceEntry {
    std::string goal;
    std::string summary;
    uint64_t atMs{0};
};

inline std::vector<ExperienceEntry> experienceLoad(const std::string &storePath) {
    std::vector<ExperienceEntry> out;
    std::ifstream in(storePath);
    if (!in) return out;
    nlohmann::json j;
    in >> j;
    if (!j.is_array()) return out;
    for (const auto &e : j) {
        if (!e.is_object()) continue;
        out.push_back({e.value("goal", std::string()),
                       e.value("summary", std::string()),
                       e.value("atMs", 0ull)});
    }
    return out;
}

inline void experienceSave(const std::string &storePath,
                            const std::vector<ExperienceEntry> &entries) {
    std::filesystem::create_directories(
        std::filesystem::path(storePath).parent_path());
    nlohmann::json arr = nlohmann::json::array();
    for (const auto &e : entries) {
        arr.push_back(nlohmann::json{{"goal", e.goal},
                                     {"summary", e.summary},
                                     {"atMs", e.atMs}});
    }
    std::ofstream out(storePath);
    out << arr.dump(2);
}

/* Word-set overlap score between two texts (cheap, dependency-free). */
inline double experienceOverlap(const std::string &a, const std::string &b) {
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

/* Append one experience (cap 200, newest kept). */
inline void experienceAdd(const std::string &storePath, const std::string &goal,
                          const std::string &summary) {
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    auto entries = experienceLoad(storePath);
    entries.push_back({goal, summary, 0});
    if (entries.size() > 200) entries.erase(entries.begin());
    experienceSave(storePath, entries);
}

/* Top-k most similar past missions for a new goal. */
inline std::vector<ExperienceEntry> experienceTop(const std::string &storePath,
                                                  const std::string &goal,
                                                  size_t k = 3) {
    const auto entries = experienceLoad(storePath);
    std::vector<std::pair<double, ExperienceEntry>> scored;
    for (const auto &e : entries)
        scored.push_back({experienceOverlap(goal, e.goal), e});
    std::sort(scored.begin(), scored.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });
    std::vector<ExperienceEntry> out;
    for (size_t i = 0; i < scored.size() && i < k; ++i) {
        if (scored[i].first <= 0.05) break;
        out.push_back(scored[i].second);
    }
    return out;
}

}  // namespace mission
}  // namespace phoenix
