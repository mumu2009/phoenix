/* mission_workspace.hpp - sandboxed file-IO channel for the mission worker
   Copyright (C) 2026 079 Project

   The mission worker (and the Mission console) gets a REAL file workspace:
   the agent drafts a minimal version of its deliverable, then iteratively
   reads / rewrites / expands files instead of appending forever.
   All operations are sandboxed to <workspaceRoot>/<missionId>/ - no path
   traversal, no absolute paths, per-file size cap.
*/
#pragma once

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

namespace phoenix {
namespace mission {

namespace fs = std::filesystem;

/* Sanitize a workspace scope.  A scope may contain '/' (e.g.
   "mission-1/children/child-0-0"): every segment is sanitized separately so
   the parent (mission scope) and each helper box (children/<id> scope) get
   their own sandboxed subtree. */
inline std::string sanitizeScope(const std::string &scope) {
    std::string out;
    std::string seg;
    for (const char c : scope) {
        if (c == '/') {
            if (seg.empty()) seg = "x";
            if (!out.empty()) out.push_back('/');
            out += seg;
            seg.clear();
            continue;
        }
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') || c == '-' || c == '_';
        seg.push_back(ok ? c : '_');
    }
    if (!seg.empty()) {
        if (!out.empty()) out.push_back('/');
        out += seg;
    }
    return out.empty() ? std::string("shared") : out;
}

/**
 * Execute one sandboxed workspace operation.
 *
 * payload: {"action": "list"|"read"|"write"|"append"|"delete",
 *           "path": "relative/path.md" (no "..", no absolute),
 *           "content": "..." (write/append only)}
 *
 * Returns {ok: bool, result/error, bytes} - never throws.
 */
inline nlohmann::json workspaceExecute(const std::string &workspaceRoot,
                                       const std::string &missionId,
                                       const nlohmann::json &payload) {
    /* Align with MissionLifecycle in-memory deliverable cap (4 MiB).  The old
       256 KiB cap silently blocked append once tutorials grew, so the UI
       memory view and deliverable.md diverged / looked "truncated". */
    constexpr size_t kMaxFileBytes = 4u * 1024u * 1024u;
    const std::string action = payload.value("action", std::string());
    const std::string rel = payload.value("path", std::string());

    if (action.empty()) {
        return nlohmann::json{{"ok", false}, {"error", "action required"}};
    }
    const fs::path root = fs::absolute(fs::path(workspaceRoot)) / sanitizeScope(missionId);

    auto resolve = [&](const std::string &p) -> std::pair<bool, fs::path> {
        if (p.empty() || p[0] == '/' || p[0] == '\\') return {false, {}};
        const fs::path relPath(p);
        if (relPath.is_absolute()) return {false, {}};
        for (const auto &part : relPath) {
            if (part == "..") return {false, {}};
        }
        fs::path target = root / relPath;
        std::error_code ec;
        const fs::path canon = fs::weakly_canonical(target, ec);
        if (ec) return {false, {}};
        const fs::path canonRoot = fs::weakly_canonical(root, ec);
        if (ec) return {false, {}};
        std::string rootStr = canonRoot.string();
        std::string targetStr = canon.string();
        if (targetStr.size() < rootStr.size() ||
            targetStr.compare(0, rootStr.size(), rootStr) != 0) return {false, {}};
        return {true, target};
    };

    auto readFile = [&](const fs::path &p) -> nlohmann::json {
        std::ifstream in(p, std::ios::binary);
        if (!in) return nlohmann::json{{"ok", false}, {"error", "cannot read: " + p.string()}};
        std::ostringstream ss;
        ss << in.rdbuf();
        std::string body = ss.str();
        bool truncated = false;
        if (body.size() > kMaxFileBytes) {
            /* Keep the sliding tail so the deliberator still sees recent work
               instead of hard-failing the whole tick. */
            body = body.substr(body.size() - kMaxFileBytes);
            truncated = true;
        }
        nlohmann::json out = {{"ok", true}, {"bytes", body.size()}, {"content", body}};
        if (truncated) out["truncated"] = true;
        return out;
    };

    if (action == "list") {
        std::error_code ec;
        fs::create_directories(root, ec);
        nlohmann::json files = nlohmann::json::array();
        if (fs::exists(root, ec)) {
            for (const auto &entry : fs::directory_iterator(root, ec)) {
                if (entry.is_regular_file()) {
                    files.push_back(nlohmann::json{{"name", entry.path().filename().string()},
                                                   {"bytes", static_cast<uint64_t>(entry.file_size(ec))}});
                }
            }
        }
        return nlohmann::json{{"ok", true}, {"files", files}};
    }

    if (action == "read") {
        const auto [ok, target] = resolve(rel);
        if (!ok) return nlohmann::json{{"ok", false}, {"error", "invalid path (sandboxed)"}};
        return readFile(target);
    }

    if (action == "write" || action == "append") {
        const std::string content = payload.value("content", std::string());
        if (content.size() > kMaxFileBytes) {
            return nlohmann::json{{"ok", false}, {"error", "content exceeds size cap"}};
        }
        const auto [ok, target] = resolve(rel);
        if (!ok) return nlohmann::json{{"ok", false}, {"error", "invalid path (sandboxed)"}};
        std::error_code ec;
        fs::create_directories(target.parent_path(), ec);
        if (action == "append") {
            const nlohmann::json existing = readFile(target);
            if (!existing.value("ok", false) && fs::exists(target, ec)) {
                return existing;
            }
            std::string merged = (existing.value("ok", false)
                                      ? existing.value("content", std::string())
                                      : std::string()) +
                                 content;
            bool trimmed = false;
            if (merged.size() > kMaxFileBytes) {
                /* Prefer keeping the newest text (same policy as in-memory
                   Mission::deliverable). */
                merged = merged.substr(merged.size() - kMaxFileBytes);
                trimmed = true;
            }
            std::ofstream out(target, std::ios::binary | std::ios::trunc);
            if (!out) return nlohmann::json{{"ok", false}, {"error", "cannot write: " + target.string()}};
            out << merged;
            nlohmann::json res = {{"ok", true}, {"bytes", merged.size()}};
            if (trimmed) res["trimmedToCap"] = true;
            return res;
        }
        std::ofstream out(target, std::ios::binary | std::ios::trunc);
        if (!out) return nlohmann::json{{"ok", false}, {"error", "cannot write: " + target.string()}};
        out << content;
        return nlohmann::json{{"ok", true}, {"bytes", content.size()}};
    }

    if (action == "delete") {
        const auto [ok, target] = resolve(rel);
        if (!ok) return nlohmann::json{{"ok", false}, {"error", "invalid path (sandboxed)"}};
        std::error_code ec;
        if (!fs::remove(target, ec)) {
            return nlohmann::json{{"ok", false}, {"error", "cannot delete: " + target.string()}};
        }
        return nlohmann::json{{"ok", true}, {"bytes", 0}};
    }

    return nlohmann::json{{"ok", false}, {"error", "unknown action: " + action}};
}


/* ======================= v8.x L1 lookup cache =======================
 * Response memoization + repetition guard for the mission deliberator.
 * On RDK CPU one tick costs minutes; when the dynamic part of the prompt
 * is (near-)identical to the previous tick the model would repeat itself,
 * so we skip the LLM entirely and replay nothing (the workspace text is
 * already there).  Similarity is a lightweight 4-gram Jaccard over the
 * dynamic prompt key - no external embedding dependency, safe to compile
 * standalone (gtests include this header without main.cpp parts). */

namespace ws_cache_detail {
inline uint64_t wsHash4(const std::string &s, size_t start, size_t end) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = start; i < end; ++i) {
        h ^= static_cast<unsigned char>(s[i]);
        h *= 1099511628211ULL;
    }
    return h;
}
}  // namespace ws_cache_detail

inline double workspaceJaccard(const std::string &a, const std::string &b) {
    if (a == b) return 1.0;
    if (a.size() < 4 || b.size() < 4) return a == b ? 1.0 : 0.0;
    std::vector<uint64_t> ga, gb;
    ga.reserve(a.size());
    gb.reserve(b.size());
    for (size_t i = 0; i + 4 <= a.size(); ++i)
        ga.push_back(ws_cache_detail::wsHash4(a, i, i + 4));
    for (size_t i = 0; i + 4 <= b.size(); ++i)
        gb.push_back(ws_cache_detail::wsHash4(b, i, i + 4));
    std::sort(ga.begin(), ga.end());
    std::sort(gb.begin(), gb.end());
    size_t inter = 0, i = 0, j = 0;
    while (i < ga.size() && j < gb.size()) {
        if (ga[i] == gb[j]) { ++inter; ++i; ++j; }
        else if (ga[i] < gb[j]) ++i;
        else ++j;
    }
    const size_t un = ga.size() + gb.size() - inter;
    return un == 0 ? 1.0 : static_cast<double>(inter) / static_cast<double>(un);
}

/* Shared cache state: C++17 inline variables so EVERY translation unit
 * (main.cpp + gtests) shares ONE table - a function-local static here
 * would split Get and Put into two unrelated tables and silently disable
 * the whole L1 cache. */
namespace ws_cache_detail {
struct WsCacheEntry {
    std::string key;
    std::string reply;
};
inline std::mutex gWsCacheMu;
inline std::unordered_map<std::string, std::vector<WsCacheEntry>> gWsCacheTable;
}  // namespace ws_cache_detail

/* Get a cached response for (scope, key).  Returns false when nothing is
 * similar enough (>= simThreshold, default 0.95). */
inline bool workspaceCacheGet(const std::string &scope, const std::string &key,
                              double simThreshold) {
    using namespace ws_cache_detail;
    std::lock_guard<std::mutex> lock(gWsCacheMu);
    const auto it = gWsCacheTable.find(scope);
    if (it == gWsCacheTable.end()) return false;
    for (const auto &e : it->second) {
        if (workspaceJaccard(e.key, key) >= simThreshold) return true;
    }
    return false;
}

/* Store/refresh the newest (key, reply) for a scope; LRU cap 64 entries. */
inline void workspaceCachePut(const std::string &scope, const std::string &key,
                              const std::string &reply) {
    using namespace ws_cache_detail;
    std::lock_guard<std::mutex> lock(gWsCacheMu);
    auto &v = gWsCacheTable[scope];
    v.push_back({key, reply});
    if (v.size() > 64) v.erase(v.begin());
}

}  // namespace mission
}  // namespace phoenix
