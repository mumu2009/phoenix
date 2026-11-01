/* Conditional profile rank.

   Used only when the lexical ranker has already returned no file.
   Each file is reduced to at most eight distinctive words: length at
   least 3, not a listed function word, and absent from every other
   candidate body. The frozen causal model then scores the description
   after the prefix "Notes about: " plus those words. The decision
   threshold is the calendar run's lower median, 0.2818081893704152.
   It is applied inside tools/meme_cond_resident.py and is not refit
   here. The model is loaded once and answers on 127.0.0.1:8107.
   One failed start latches this call off. An empty path in a real
   reply is a decision, not a failure. This call does not write aliases.
*/
#pragma once

#include "memetic_existence.hpp"
#include "subprocess.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace phoenix {
namespace memetic {

/* Test seam: the probe substitutes a fake exchange so the cache and
   gate can be measured without loading the 1B model. Live code leaves
   it null and uses the real TCP exchange. */
using CondExchangeFn = std::string (*)(const std::string &host, int port,
                                       const std::string &line, int timeoutMs);

inline CondExchangeFn &condExchangeHook() {
  static CondExchangeFn hook = nullptr;
  return hook;
}

/* Number of real model exchanges since process start. A cache hit does
   not increment it. */
inline std::atomic<int> &condProfileExchangeCount() {
  static std::atomic<int> count{0};
  return count;
}

inline std::atomic<int> &condProfileCacheHitCount() {
  static std::atomic<int> count{0};
  return count;
}

inline std::vector<std::string> condTokens(const std::string &text) {
  std::vector<std::string> out;
  std::string cur;
  auto flush = [&]() {
    if (cur.size() >= 3)
      out.push_back(cur);
    cur.clear();
  };
  for (unsigned char c : text) {
    if (std::isalnum(c))
      cur.push_back(static_cast<char>(std::tolower(c)));
    else
      flush();
  }
  flush();
  return out;
}

inline std::string distinctiveProfile(
    const std::string &selfId, const std::string &body,
    const std::vector<std::pair<std::string, std::string>> &candidates) {
  std::unordered_map<std::string, int> tf;
  for (const auto &w : condTokens(body)) {
    if (!condFunctionWord(w))
      tf[w] += 1;
  }
  std::unordered_set<std::string> blocked;
  for (const auto &other : candidates) {
    if (other.first == selfId)
      continue;
    for (const auto &w : condTokens(other.second))
      blocked.insert(w);
  }
  std::vector<std::pair<int, std::string>> ranked;
  for (const auto &kv : tf) {
    if (blocked.count(kv.first))
      continue;
    ranked.emplace_back(kv.second, kv.first);
  }
  std::sort(ranked.begin(), ranked.end(),
            [](const std::pair<int, std::string> &a,
               const std::pair<int, std::string> &b) {
              if (a.first != b.first)
                return a.first > b.first;
              return a.second < b.second;
            });
  std::string out;
  int kept = 0;
  for (const auto &row : ranked) {
    if (kept == 8)
      break;
    if (!out.empty())
      out.push_back(' ');
    out += row.second;
    ++kept;
  }
  return out;
}

inline std::string condJsonEscape(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') {
      out.push_back('\\');
      out.push_back(static_cast<char>(c));
    } else if (c == '\n' || c == '\r') {
      out.push_back(' ');
    } else {
      out.push_back(static_cast<char>(c));
    }
  }
  return out;
}

inline std::string condProfilePath(
    const std::string &description,
    const std::vector<std::pair<std::string, std::string>> &profiles) {
  if (description.empty() || profiles.size() < 2)
    return std::string();
  const std::filesystem::path python =
      std::filesystem::path("runtime_store") / "intent_train_venv" /
      "Scripts" / "python.exe";
  const std::filesystem::path script =
      std::filesystem::path("tools") / "meme_cond_resident.py";
  std::error_code ec;
  if (!std::filesystem::is_regular_file(python, ec) ||
      !std::filesystem::is_regular_file(script, ec))
    return std::string();
  std::ostringstream payload;
  payload << "{\"description\":\"" << condJsonEscape(description)
          << "\",\"profiles\":[";
  bool first = true;
  for (const auto &row : profiles) {
    if (row.first.empty() || row.second.empty())
      continue;
    if (!first)
      payload << ',';
    first = false;
    payload << "{\"id\":\"" << condJsonEscape(row.first) << "\",\"text\":\""
            << condJsonEscape(row.second) << "\"}";
  }
  payload << "]}";
  if (first)
    return std::string();
  const std::string line = payload.str() + "\n";
  /* Teacher cache: the same (description, profiles) is answered once.
     Only the path is stored; no candidate body is copied. Bounded at
     256 entries. A cached decision - including "no file" - is returned
     without a model exchange, which is what lets the promoted alias
     route locally after the two hits. */
  static std::mutex cacheMu;
  static std::unordered_map<std::string, std::string> cache;
  {
    std::lock_guard<std::mutex> lock(cacheMu);
    auto it = cache.find(line);
    if (it != cache.end()) {
      condProfileCacheHitCount().fetch_add(1, std::memory_order_relaxed);
      return it->second;
    }
  }
  static std::mutex gate;
  static bool starting = false;
  static bool latched = false;
  {
    std::lock_guard<std::mutex> lock(gate);
    if (latched)
      return std::string();
  }
  auto exchange = [&]() {
    CondExchangeFn &hook = condExchangeHook();
    condProfileExchangeCount().fetch_add(1, std::memory_order_relaxed);
    if (hook != nullptr)
      return hook("127.0.0.1", 8107, line, 45000);
    return subprocess::tcpExchange("127.0.0.1", 8107, line, 45000);
  };
  std::string reply = exchange();
  if (reply.empty()) {
    bool launch = false;
    {
      std::lock_guard<std::mutex> lock(gate);
      if (latched)
        return std::string();
      if (!starting) {
        starting = true;
        launch = true;
      }
    }
    bool spawned = true;
    if (launch)
      spawned = subprocess::spawnDetached(python.string(), {script.string()});
    bool up = false;
    const int attempts = spawned ? 90 : 3;
    for (int attempt = 0; attempt < attempts; ++attempt) {
      if (subprocess::tcpReachable("127.0.0.1", 8107, 200)) {
        up = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    if (up)
      reply = exchange();
    if (reply.empty()) {
      std::lock_guard<std::mutex> lock(gate);
      latched = true;
      return std::string();
    }
  }
  {
    std::lock_guard<std::mutex> lock(gate);
    starting = false;
  }
  auto parsePath = [&](const std::string &text) {
    const auto key = text.find("\"path\"");
    if (key == std::string::npos)
      return std::string();
    const auto colon = text.find(':', key);
    const auto quote =
        text.find('"', colon == std::string::npos ? key : colon);
    if (quote == std::string::npos)
      return std::string();
    const auto end = text.find('"', quote + 1);
    if (end == std::string::npos)
      return std::string();
    return text.substr(quote + 1, end - quote - 1);
  };
  const std::string path = parsePath(reply);
  {
    std::lock_guard<std::mutex> lock(cacheMu);
    if (cache.size() >= 256)
      cache.erase(cache.begin());
    cache[line] = path;
  }
  return path;
}

inline std::string reverseWordOrder(const std::string &text) {
  std::vector<std::string> words;
  std::string cur;
  for (char c : text) {
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
      if (!cur.empty()) {
        words.push_back(cur);
        cur.clear();
      }
    } else {
      cur.push_back(c);
    }
  }
  if (!cur.empty())
    words.push_back(cur);
  std::string out;
  for (size_t i = words.size(); i-- > 0;) {
    if (!out.empty())
      out.push_back(' ');
    out += words[i];
  }
  return out;
}

/* Stability audit for one teacher hit.

   The two-hit gate cannot tell whether the teacher is right twice; it
   can tell whether the association is stable. Ask the same description
   with the same set of profile words in the opposite order. A path that
   survives both orders is accepted; a different path, or an empty
   answer on either side, is rejected. This filters unstable teacher
   associations, not wrong-but-stable ones (theory.md 11.3 stays
   honest about that). The decision threshold T lives in the resident
   and is not touched. */
inline std::string condProfilePathAudited(
    const std::string &description,
    const std::vector<std::pair<std::string, std::string>> &profiles,
    bool *stable = nullptr) {
  if (stable)
    *stable = false;
  if (description.empty() || profiles.size() < 2)
    return std::string();
  const std::string forward = condProfilePath(description, profiles);
  std::vector<std::pair<std::string, std::string>> reversed;
  reversed.reserve(profiles.size());
  for (const auto &row : profiles)
    reversed.emplace_back(row.first, reverseWordOrder(row.second));
  const std::string backward = condProfilePath(description, reversed);
  if (forward.empty() || backward.empty() || forward != backward)
    return std::string();
  if (stable)
    *stable = true;
  return forward;
}

}  // namespace memetic
}  // namespace phoenix
