/* cross_context_memory.hpp - explicit cross-context long-term memory (v8.x)

   Context isolation rule (see doc/v8.3/context_isolation.md):
   - PER-CONTEXT state (workspace, deliverable, plan, sensations, appraisal)
     is strictly separated by contextTag (mission:<id> / chat:<session>);
   - CROSS-CONTEXT memory (this module + GNN graph + mission experience)
     is the ONE place contexts may exchange knowledge.  Chat turns and
     finished missions deposit captions + optional unit-query rows; new
     contexts recall entries and inject them as unit-query I/O (not
     prompt text). Live per-context state never leaks.

   Storage: one JSON file (default runtime_store/cross_context_memory.json),
   capped at 500 entries.  Retrieval: word-set overlap (cheap, no external
   embedding dependency - safe for header-only / gtest compilation). */
#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace phoenix {
namespace memory {

struct CcmEntry {
    std::string sourceTag;  /* "chat:<session>" or "mission:<id>" */
    std::string text;       /* retrieval key / caption — not infer I/O */
    uint64_t atMs{0};
    std::string modality{"text"}; /* text | audio | video | image | unit */
    std::vector<std::vector<float>> unitQuery; /* E-space rows when already enc'd */
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
        CcmEntry ce{e.value("sourceTag", std::string()),
                    e.value("text", std::string()),
                    e.value("atMs", 0ull),
                    e.value("modality", std::string("text")),
                    {}};
        if (e.contains("unitQuery") && e["unitQuery"].is_array()) {
            for (const auto &row : e["unitQuery"]) {
                if (!row.is_array()) continue;
                std::vector<float> v;
                for (const auto &x : row) {
                    if (x.is_number())
                        v.push_back(static_cast<float>(x.get<double>()));
                }
                if (!v.empty()) ce.unitQuery.push_back(std::move(v));
            }
        }
        out.push_back(std::move(ce));
    }
    return out;
}

inline void ccmSave(const std::string &storePath,
                    const std::vector<CcmEntry> &entries) {
    std::filesystem::create_directories(
        std::filesystem::path(storePath).parent_path());
    nlohmann::json arr = nlohmann::json::array();
    for (const auto &e : entries) {
        nlohmann::json item{{"sourceTag", e.sourceTag},
                            {"text", e.text},
                            {"atMs", e.atMs},
                            {"modality", e.modality.empty() ? "text" : e.modality}};
        if (!e.unitQuery.empty())
            item["unitQuery"] = e.unitQuery;
        arr.push_back(std::move(item));
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
inline void ccmRememberUnit(const std::string &storePath,
                            const std::string &sourceTag,
                            const std::string &text,
                            const std::string &modality,
                            const std::vector<std::vector<float>> &unitQuery) {
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    auto entries = ccmLoad(storePath);
    entries.push_back({sourceTag, text, 0,
                       modality.empty() ? "text" : modality, unitQuery});
    if (entries.size() > 500) entries.erase(entries.begin());
    ccmSave(storePath, entries);
}

inline void ccmRemember(const std::string &storePath,
                        const std::string &sourceTag,
                        const std::string &text) {
    ccmRememberUnit(storePath, sourceTag, text, "text", {});
}

inline std::string chatSourceTag(const std::string &sessionId) {
    if (sessionId.empty())
        return {};
    if (sessionId.rfind("chat:", 0) == 0)
        return sessionId;
    return std::string("chat:") + sessionId;
}

/* Live chat increments stay in their own session. Mission / corpus /
   finished-mission tags may still cross. An empty callerTag is a
   mission-style recall: skip every chat:* deposit. */
inline bool ccmAllowForCaller(const CcmEntry &e, const std::string &callerTag) {
    if (e.sourceTag.rfind("chat:", 0) != 0)
        return true;
    if (callerTag.rfind("chat:", 0) != 0)
        return false;
    return e.sourceTag == callerTag;
}

inline constexpr const char kScopeCanaryPrefix[] = "SCOPECANARY-";
inline constexpr size_t kScopeCanaryPrefixLen = 12;

inline size_t scopeCanaryTokenEnd(const std::string &s, size_t pos) {
    size_t end = pos + kScopeCanaryPrefixLen;
    while (end < s.size()) {
        const unsigned char c = static_cast<unsigned char>(s[end]);
        if (!(std::isalnum(c) || c == '-' || c == '_'))
            break;
        ++end;
    }
    return end;
}

inline bool isMissionScopeCanary(const std::string &tok) {
    return tok.rfind("SCOPECANARY-MISSION-", 0) == 0;
}

inline bool isChatScopeCanary(const std::string &tok) {
    return tok.rfind("SCOPECANARY-CHAT-", 0) == 0;
}

/* Chat must not inject or echo a just-finished mission canary.
   SCOPECANARY-MISSION-* is always foreign on the chat path.
   SCOPECANARY-CHAT-* stays only when the current user text asked for it. */
inline bool isForeignScopeCanary(const std::string &tok,
                                 const std::string &ownUserText) {
    if (tok.size() < kScopeCanaryPrefixLen ||
        tok.compare(0, kScopeCanaryPrefixLen, kScopeCanaryPrefix) != 0)
        return false;
    if (isMissionScopeCanary(tok))
        return true;
    if (isChatScopeCanary(tok))
        return ownUserText.find(tok) == std::string::npos;
    return ownUserText.find(tok) == std::string::npos;
}

/* Drop dialog increments that belong to another chat session, and any
   line that still carries SCOPECANARY-MISSION-*. Used on contextHint /
   graph leftovers so a stale frontend dump cannot put B/C or a finished
   mission canary into the next generate context. */
inline std::string retainOwnChatIncrements(const std::string &text,
                                           const std::string &ownUserText) {
    if (text.empty())
        return text;
    std::istringstream in(text);
    std::ostringstream out;
    std::string line;
    bool wrote = false;
    while (std::getline(in, line)) {
        if (line.find("[Cross-session history]") != std::string::npos)
            continue;
        bool dropForeign = false;
        for (size_t pos = line.find(kScopeCanaryPrefix);
             pos != std::string::npos;
             pos = line.find(kScopeCanaryPrefix, pos + kScopeCanaryPrefixLen)) {
            const size_t end = scopeCanaryTokenEnd(line, pos);
            const std::string tok = line.substr(pos, end - pos);
            if (isForeignScopeCanary(tok, ownUserText)) {
                dropForeign = true;
                break;
            }
        }
        if (dropForeign)
            continue;
        if (wrote)
            out << '\n';
        out << line;
        wrote = true;
    }
    return out.str();
}

/* Remove foreign SCOPECANARY-* tokens. Covers CHAT and MISSION: prompt,
   CCM captions, units, and the visible reply. Mission canaries are
   always erased on the chat path. */
inline std::string eraseForeignChatCanaries(const std::string &text,
                                            const std::string &ownUserText) {
    if (text.empty())
        return text;
    std::string out = text;
    for (size_t pos = 0; (pos = out.find(kScopeCanaryPrefix, pos)) !=
                         std::string::npos;) {
        const size_t end = scopeCanaryTokenEnd(out, pos);
        const std::string tok = out.substr(pos, end - pos);
        if (isForeignScopeCanary(tok, ownUserText)) {
            out.erase(pos, end - pos);
            continue;
        }
        pos = end;
    }
    return out;
}

/* Recall the top-k most similar entries. callerTag isolates chat↔chat:
   chat:A never receives chat:B/C dialog increments (the soak canary hole).
   Chat callers also lose SCOPECANARY-MISSION-* from mission deposits. */
inline std::vector<CcmEntry> ccmRecall(const std::string &storePath,
                                       const std::string &query, size_t k = 3,
                                       const std::string &callerTag = {}) {
    const auto entries = ccmLoad(storePath);
    std::vector<std::pair<double, CcmEntry>> scored;
    for (const auto &e : entries) {
        if (!ccmAllowForCaller(e, callerTag))
            continue;
        scored.push_back({ccmOverlap(query, e.text), e});
    }
    std::sort(scored.begin(), scored.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });
    std::vector<CcmEntry> out;
    const bool chatCaller = callerTag.rfind("chat:", 0) == 0;
    for (size_t i = 0; i < scored.size() && i < k; ++i) {
        if (scored[i].first <= 0.05) break;
        CcmEntry e = scored[i].second;
        if (chatCaller)
            e.text = eraseForeignChatCanaries(e.text, query);
        out.push_back(std::move(e));
    }
    return out;
}

}  // namespace memory
}  // namespace phoenix
