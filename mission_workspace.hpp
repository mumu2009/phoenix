/* mission_workspace.hpp - sandboxed file-IO channel for the mission worker
   Copyright (C) 2026 079 Project

   The mission worker (and the Mission console) gets a REAL file workspace:
   the agent drafts a minimal version of its deliverable, then iteratively
   reads / rewrites / expands files instead of appending forever.
   All operations are sandboxed to <workspaceRoot>/<missionId>/ - no path
   traversal, no absolute paths, per-file size cap.
*/
#pragma once

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

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
    constexpr size_t kMaxFileBytes = 256u * 1024u;
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
        const std::string body = ss.str();
        if (body.size() > kMaxFileBytes) {
            return nlohmann::json{{"ok", false}, {"error", "file exceeds size cap"}};
        }
        return nlohmann::json{{"ok", true}, {"bytes", body.size()}, {"content", body}};
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
            const std::string merged = (existing.value("ok", false)
                                            ? existing.value("content", std::string())
                                            : std::string()) + content;
            if (merged.size() > kMaxFileBytes) {
                return nlohmann::json{{"ok", false}, {"error", "merged file exceeds size cap"}};
            }
            std::ofstream out(target, std::ios::binary | std::ios::trunc);
            if (!out) return nlohmann::json{{"ok", false}, {"error", "cannot write: " + target.string()}};
            out << merged;
            return nlohmann::json{{"ok", true}, {"bytes", merged.size()}};
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

}  // namespace mission
}  // namespace phoenix
