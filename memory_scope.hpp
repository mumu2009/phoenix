/* memory_scope.hpp - Chat / Mission trainable-memory namespace

   Live trainable state is bucketed by MemoryScope{kind, id}.
   Chat without a mission goal uses the chat sessionId.  A mission never
   shares a bucket with chat or with another mission.  Read-only cold
   stores (official GNN ingest, experience recall, CCM) may be shared. */
#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace phoenix {
namespace memory {

enum class MemoryKind { Chat = 0, Mission = 1 };

struct MemoryScope {
  MemoryKind kind{MemoryKind::Chat};
  std::string id;

  bool valid() const { return !id.empty(); }

  const char *kindName() const {
    return kind == MemoryKind::Mission ? "mission" : "chat";
  }

  /** Canonical key: "chat:<id>" or "mission:<id>". */
  std::string key() const {
    std::string out = kindName();
    out.push_back(':');
    out += id;
    return out;
  }

  bool operator==(const MemoryScope &o) const {
    return kind == o.kind && id == o.id;
  }
  bool operator!=(const MemoryScope &o) const { return !(*this == o); }
};

inline std::string sanitizeScopeId(std::string raw) {
  while (!raw.empty() &&
         std::isspace(static_cast<unsigned char>(raw.front())))
    raw.erase(raw.begin());
  while (!raw.empty() && std::isspace(static_cast<unsigned char>(raw.back())))
    raw.pop_back();
  for (char &c : raw) {
    if (c == '/' || c == '\\')
      c = '_';
  }
  return raw;
}

inline MemoryScope makeChatScope(std::string sessionId) {
  sessionId = sanitizeScopeId(std::move(sessionId));
  if (sessionId.empty())
    sessionId = "anonymous";
  return MemoryScope{MemoryKind::Chat, std::move(sessionId)};
}

inline MemoryScope makeMissionScope(std::string missionOrGoalId) {
  missionOrGoalId = sanitizeScopeId(std::move(missionOrGoalId));
  if (missionOrGoalId.empty())
    missionOrGoalId = "anonymous";
  return MemoryScope{MemoryKind::Mission, std::move(missionOrGoalId)};
}

/** Strip helper-box suffix: "mid/children/child-0" -> "mid". */
inline std::string missionIdFromWorkspaceScope(const std::string &scope) {
  const auto sp = scope.find("/children/");
  if (sp == std::string::npos)
    return sanitizeScopeId(scope);
  return sanitizeScopeId(scope.substr(0, sp));
}

/** Parse "chat:X" / "mission:X".  Bare ids become Chat. */
inline MemoryScope parseMemoryScope(const std::string &raw) {
  const std::string s = sanitizeScopeId(raw);
  if (s.rfind("mission:", 0) == 0)
    return makeMissionScope(s.substr(8));
  if (s.rfind("chat:", 0) == 0)
    return makeChatScope(s.substr(5));
  return makeChatScope(s);
}

/**
 * Resolve a scope from explicit ids.
 * A non-empty missionId / goalId always wins (Mission).  Chat never
 * inherits a "default" running mission — that is how Helios leaked
 * into an unrelated ops chat.
 */
inline MemoryScope memoryScopeFromIds(std::string sessionId,
                                      std::string missionId) {
  missionId = sanitizeScopeId(std::move(missionId));
  if (!missionId.empty()) {
    if (missionId.rfind("mission:", 0) == 0)
      missionId = missionId.substr(8);
    return makeMissionScope(std::move(missionId));
  }
  return parseMemoryScope(std::move(sessionId));
}

/**
 * Request / iterate payload -> scope.
 * Honours memoryKind / kind / missionId / goalId / contextTag / sessionId.
 * Does NOT consult any process-global default mission.
 */
inline MemoryScope memoryScopeFromPayload(const nlohmann::json &p) {
  if (!p.is_object())
    return makeChatScope("anonymous");
  const std::string kind =
      p.value("memoryKind", p.value("kind", std::string()));
  std::string missionId =
      p.value("missionId", p.value("goalId", std::string()));
  std::string sessionId = p.value("sessionId", std::string());
  const std::string contextTag = p.value("contextTag", std::string());
  if (missionId.empty() && contextTag.rfind("mission:", 0) == 0)
    missionId = contextTag.substr(8);
  if (kind == "mission" || kind == "Mission") {
    std::string id = missionId.empty() ? sessionId : missionId;
    if (id.rfind("mission:", 0) == 0)
      id = id.substr(8);
    return makeMissionScope(std::move(id));
  }
  return memoryScopeFromIds(std::move(sessionId), std::move(missionId));
}

/**
 * /api/chat and frontend chat-proxy: a leftover missionId on a chat
 * body must not steal the mission trainable bucket.  Only an explicit
 * memoryKind/kind=mission attaches chat traffic to a mission scope.
 */
inline MemoryScope memoryScopeForChatRoute(const nlohmann::json &p) {
  if (!p.is_object())
    return makeChatScope("anonymous");
  const std::string kind =
      p.value("memoryKind", p.value("kind", std::string()));
  if (kind == "mission" || kind == "Mission")
    return memoryScopeFromPayload(p);
  return makeChatScope(p.value("sessionId", std::string()));
}

inline std::string pressureSourceFor(const MemoryScope &s) {
  if (s.kind == MemoryKind::Mission)
    return "mission:" + s.id + ":pressure";
  return "chat:" + s.id + ":sensation";
}

/**
 * Scoped FIFO eviction for shared append-only buffers (e.g. the frontend
 * episodic memory vector).  Each owner (scope key) keeps at most
 * perScopeCap entries: a busy scope evicts ITS OWN oldest entry, never
 * another scope's.  globalCap is only a memory guard across all owners.
 * `ownerOf(entry)` must return the entry's scope key.
 */
template <typename EntryT, typename FOwner>
inline void evictScopedFifoForInsert(std::vector<EntryT> &entries,
                                     const std::string &ownerKey,
                                     std::size_t perScopeCap,
                                     std::size_t globalCap,
                                     FOwner ownerOf) {
  if (perScopeCap > 0) {
    std::size_t own = 0;
    for (const auto &e : entries)
      if (ownerOf(e) == ownerKey)
        ++own;
    if (own >= perScopeCap) {
      for (auto it = entries.begin(); it != entries.end(); ++it) {
        if (ownerOf(*it) == ownerKey) {
          entries.erase(it);
          break;
        }
      }
    }
  }
  if (globalCap > 0 && entries.size() >= globalCap)
    entries.erase(entries.begin());
}

inline MemoryScope &tlsCurrentMemoryScope() {
  thread_local MemoryScope cur = makeChatScope("anonymous");
  return cur;
}

inline MemoryScope currentMemoryScope() { return tlsCurrentMemoryScope(); }

inline void setCurrentMemoryScope(MemoryScope s) {
  if (!s.valid())
    s = makeChatScope("anonymous");
  tlsCurrentMemoryScope() = std::move(s);
}

/** RAII current-scope binder for graphLink / concept-matrix / dialog paths. */
class MemoryScopeGuard {
 public:
  explicit MemoryScopeGuard(MemoryScope s) : prev_(currentMemoryScope()) {
    setCurrentMemoryScope(std::move(s));
  }
  ~MemoryScopeGuard() { setCurrentMemoryScope(prev_); }
  MemoryScopeGuard(const MemoryScopeGuard &) = delete;
  MemoryScopeGuard &operator=(const MemoryScopeGuard &) = delete;

 private:
  MemoryScope prev_;
};

}  // namespace memory
}  // namespace phoenix
