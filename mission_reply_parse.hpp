/* mission_reply_parse.hpp - parse mission deliberator LLM replies safely */
#pragma once

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>

#include <sstream>
#include <utility>

#include <nlohmann/json.hpp>

namespace phoenix {
namespace mission {

inline std::string trimCopy(const std::string &s) {
  const auto b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return std::string();
  const auto e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

/** Word-boundary match. Not a phrase catalog. */
inline bool hasAlnumWord(const std::string &lower, const std::string &w) {
  if (w.empty() || lower.empty()) return false;
  size_t p = 0;
  while ((p = lower.find(w, p)) != std::string::npos) {
    const bool left =
        p == 0 || !std::isalnum(static_cast<unsigned char>(lower[p - 1]));
    const bool right =
        p + w.size() >= lower.size() ||
        !std::isalnum(static_cast<unsigned char>(lower[p + w.size()]));
    if (left && right) return true;
    ++p;
  }
  return false;
}

/** Short paragraph that asks a second person a question.
    Structure only — no style/title phrase list. */
inline bool isReaderAddressText(const std::string &raw) {
  const std::string t = trimCopy(raw);
  if (t.size() < 12 || t.size() > 280) return false;
  if (t.find('?') == std::string::npos) return false;
  std::string lower = t;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  return hasAlnumWord(lower, "you") || hasAlnumWord(lower, "your") ||
         lower.find("you'd") != std::string::npos;
}

/** Optional utility: drop markdown fence marker lines. Persist / resume
    paths must not call this — fences are document syntax, not a defect. */
inline std::string stripMarkdownFences(std::string s) {
  std::string out;
  size_t i = 0;
  while (i < s.size()) {
    const size_t eol = s.find('\n', i);
    std::string line =
        eol == std::string::npos ? s.substr(i) : s.substr(i, eol - i);
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    const std::string t = trimCopy(line);
    if (t.rfind("```", 0) != 0) {
      if (!out.empty())
        out.push_back('\n');
      out += line;
    }
    if (eol == std::string::npos)
      break;
    i = eol + 1;
  }
  size_t p = 0;
  while ((p = out.find("```", p)) != std::string::npos)
    out.erase(p, 3);
  return trimCopy(out);
}

/** A generate that is only ``` / ```lang. Resume ended on a closer and
    the 8B opened another fence then stopped. */
inline bool looksLikeFenceOnlyReply(const std::string &s) {
  const std::string t = trimCopy(s);
  if (t.size() < 3 || t.rfind("```", 0) != 0) return false;
  size_t i = 3;
  while (i < t.size() && t[i] != '\n' && t[i] != '\r') ++i;
  while (i < t.size() &&
         (t[i] == '\n' || t[i] == '\r' || t[i] == ' ' || t[i] == '\t'))
    ++i;
  return i >= t.size();
}

inline size_t countMarkdownFences(const std::string &s) {
  size_t n = 0;
  for (size_t i = 0; i + 2 < s.size(); ++i) {
    if (s.compare(i, 3, "```") == 0) {
      ++n;
      i += 2;
    }
  }
  return n;
}

inline bool endsWithFenceCloser(const std::string &s) {
  const auto p = s.rfind("```");
  if (p == std::string::npos) return false;
  for (size_t i = p + 3; i < s.size(); ++i) {
    if (s[i] != ' ' && s[i] != '\n' && s[i] != '\r' && s[i] != '\t')
      return false;
  }
  return true;
}

/** Cycle gate only: do not prime the next generate with a trailing ```.
    Inner fences stay. Does not rewrite the on-disk draft. */
inline std::string trimTrailingFenceCloser(std::string s) {
  while (endsWithFenceCloser(s)) {
    const auto p = s.rfind("```");
    if (p == std::string::npos) break;
    s.resize(p);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' ||
                          s.back() == ' ' || s.back() == '\t'))
      s.pop_back();
  }
  return s;
}

/** Fence-only reply against on-disk parity.
    Odd ``` → write the closer once. Even → empty (drop). Never write
    blank lines on drop — that grew the file and left the same resume. */
inline std::string resolveFenceOnlyReply(const std::string &file) {
  if (countMarkdownFences(file) % 2 == 1)
    return "```\n\n";
  return std::string();
}

/** Resume suffix only. Disk is not rewritten.
    Even ``` count, and the suffix does not end on a closer. */
inline std::string balanceResumeFences(std::string tail) {
  /* Drop an unclosed trailing opener, then drop a trailing closer so the
     next tick is not primed with ```. Keep opener+body of a closed block. */
  if (countMarkdownFences(tail) % 2 == 1 && !endsWithFenceCloser(tail)) {
    const auto p = tail.rfind("```");
    if (p != std::string::npos) {
      tail.resize(p);
      while (!tail.empty() && (tail.back() == '\n' || tail.back() == '\r' ||
                               tail.back() == ' ' || tail.back() == '\t'))
        tail.pop_back();
    }
  }
  return trimTrailingFenceCloser(std::move(tail));
}

/** One fence contract for a generate tick.
    Persist: fences are Markdown syntax. Never strip them from disk.
    Resume: balanceResumeFences on the suffix only.
    Fence-only generate: resolveFenceOnlyReply. Odd writes once; even drops.
    Safety stays in MemeBarrier. This is not a style/content judge. */
struct MarkdownFenceTick {
  std::string chunk;
  bool write{true};
  bool closingOddFence{false};
  const char *reason{"ok"};
};

inline MarkdownFenceTick applyMarkdownFenceProtocol(
    const std::string &generated, const std::string &onDisk) {
  MarkdownFenceTick out;
  out.chunk = generated;
  if (looksLikeFenceOnlyReply(out.chunk)) {
    out.chunk = resolveFenceOnlyReply(onDisk);
    if (out.chunk.empty()) {
      out.write = false;
      out.reason = "drop-fence-only";
      return out;
    }
    out.closingOddFence = true;
    out.reason = "close-odd";
    return out;
  }
  if (trimCopy(out.chunk).empty()) {
    out.chunk.clear();
    out.write = false;
    out.reason = "drop-empty";
  }
  return out;
}

/** Control / diagnostic lines must never be appended to deliverable.md. */
inline bool isMissionMetaReply(const std::string &s) {
  const std::string t = trimCopy(s);
  if (t.empty()) return true;
  if (t.rfind("[loop-pause:", 0) == 0) return true;
  static const char *prefixes[] = {
      "[tool:",      "[file:",        "[parse-fail]", "[plan-refused]",
      "[plan]",      "[box]",         "[loop]",       "[auto-verify]",
      "[auto-complete]", "[cache]",    "[llm-error]",  "[loop-break]",
  };
  for (const char *p : prefixes) {
    if (t.rfind(p, 0) == 0) return true;
  }
  /* Do not refuse tone, headings, or factual claims. Only drop
     control/diagnostic lines so they never become deliverable.md. */
  return false;
}

/** Flatten executeAgiActionByName() JSON into a user-visible tool reply. */
inline std::string agiActionReply(const nlohmann::json &tr) {
  if (!tr.is_object()) return std::string();
  if (tr.contains("reply") && tr["reply"].is_string())
    return tr["reply"].get<std::string>();
  const auto &res = tr.value("result", nlohmann::json::object());
  if (res.is_object() && res.contains("reply") && res["reply"].is_string())
    return res["reply"].get<std::string>();
  return std::string();
}

/** Skip an optional leading "JSON" word before the first '{'.
    Do not skip ``` / ```json — fenced blocks are document examples. */
inline size_t firstJsonBrace(const std::string &s) {
  const std::string lowerPrefixes[] = {"json"};
  size_t i = 0;
  while (i < s.size()) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    if (i >= s.size()) break;
    bool skipped = false;
    for (const auto &pfx : lowerPrefixes) {
      if (s.size() - i >= pfx.size()) {
        bool match = true;
        for (size_t k = 0; k < pfx.size(); ++k) {
          if (std::tolower(static_cast<unsigned char>(s[i + k])) !=
              static_cast<unsigned char>(pfx[k])) {
            match = false;
            break;
          }
        }
        if (match) {
          i += pfx.size();
          skipped = true;
          break;
        }
      }
    }
    if (!skipped) break;
  }
  const size_t brace = s.find('{', i);
  return brace == std::string::npos ? s.size() : brace;
}

/**
 * Extract every top-level JSON object embedded in a noisy reply
 * (e.g. `JSON {"action":...} JSON {"action":...}`).
 */
inline std::vector<nlohmann::json> extractJsonObjects(const std::string &text) {
  std::vector<nlohmann::json> out;
  size_t pos = firstJsonBrace(text);
  while (pos < text.size()) {
    if (text[pos] != '{') {
      pos = text.find('{', pos + 1);
      continue;
    }
    int depth = 0;
    bool inStr = false;
    bool esc = false;
    size_t end = std::string::npos;
    for (size_t i = pos; i < text.size(); ++i) {
      const char c = text[i];
      if (inStr) {
        if (esc) {
          esc = false;
        } else if (c == '\\') {
          esc = true;
        } else if (c == '"') {
          inStr = false;
        }
        continue;
      }
      if (c == '"') {
        inStr = true;
        continue;
      }
      if (c == '{')
        ++depth;
      else if (c == '}') {
        --depth;
        if (depth == 0) {
          end = i;
          break;
        }
      }
    }
    if (end == std::string::npos) break;
    const std::string slice = text.substr(pos, end - pos + 1);
    try {
      auto j = nlohmann::json::parse(slice);
      if (j.is_object()) out.push_back(std::move(j));
    } catch (...) {
    }
    pos = text.find('{', end + 1);
  }
  return out;
}

inline std::string summarizeEmotionForPrompt(const nlohmann::json &et) {
  if (!et.is_object() || et.empty()) return std::string();
  auto num = [&](const char *k) -> double {
    if (!et.contains(k) || !et[k].is_number()) return 0.0;
    return et[k].get<double>();
  };
  char buf[192];
  std::snprintf(buf, sizeof(buf),
                "arousal=%.2f valence=%.2f (internal only - never copy into "
                "deliverable)",
                num("arousal"), num("valence"));
  return std::string(buf);
}

/** True when the reply itself is a control object, not prose that
    happens to quote the JSON schema. Buried `{...}` used to mark a
    markdown continuation as structured and the write path swallowed it.
    Fenced ```json / unlabeled ``` blocks are examples in the draft —
    only a bare `{` or a `JSON {` prefix is a tool turn. */
inline bool replyLeadsWithJson(const std::string &s) {
  const std::string t = trimCopy(s);
  if (t.empty()) return false;
  if (t.front() == '{') return true;
  if (t.rfind("JSON", 0) == 0 || t.rfind("json", 0) == 0) {
    if (t.size() >= 3 && t.compare(0, 3, "```") == 0)
      return false;
    return true;
  }
  return false;
}

/** True when the LLM reply is structured control JSON, not markdown body. */
inline bool looksLikeStructuredMissionReply(const std::string &s) {
  const std::string t = trimCopy(s);
  if (t.empty() || !replyLeadsWithJson(t)) return false;
  if (t.rfind("JSON", 0) == 0 || t.rfind("json", 0) == 0) return true;

  const auto objs = extractJsonObjects(t);
  for (const auto &obj : objs) {
    if (!obj.is_object()) continue;
    if (obj.contains("action") || obj.contains("tool") || obj.contains("plan") ||
        obj.contains("done") || obj.contains("loop"))
      return true;
  }

  if (!t.empty() && t.front() == '{' &&
      (t.find("\"action\"") != std::string::npos ||
       t.find("\"tool\"") != std::string::npos ||
       t.find("\"plan\"") != std::string::npos))
    return true;

  if (t.find("\"arousal\"") != std::string::npos &&
      t.find("\"valence\"") != std::string::npos)
    return true;
  return false;
}

inline bool isDeliverablePath(const std::string &path) {
  if (path == "deliverable.md") return true;
  return path.size() > 14 &&
         path.compare(path.size() - 14, 14, "deliverable.md") == 0;
}

/** High-bit / non-English run without mission keywords = language drift. */
inline bool deliverableChunkLooksDrifted(const std::string &s) {
  if (s.size() < 40) return false;
  size_t hi = 0;
  for (unsigned char c : s)
    if (c >= 0x80) ++hi;
  if (hi * 8 < s.size()) return false;
  std::string lower = s;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  static const char *keep[] = {"chapter", "requirement", "architecture",
                               "deliverable"};
  for (const char *k : keep)
    if (lower.find(k) != std::string::npos) return false;
  return true;
}

/** Drop a prefix of `add` that already sits at the end of `prev`.
    Token-budget cuts often make the next tick repeat the unfinished sentence. */
inline std::string stripOverlappingContinuation(const std::string &prev,
                                                std::string add) {
  add = trimCopy(add);
  if (prev.empty() || add.empty()) return add;
  const size_t maxN = std::min({add.size(), prev.size(), size_t{480}});
  size_t best = 0;
  for (size_t n = maxN; n >= 40; --n) {
    if (prev.compare(prev.size() - n, n, add, 0, n) == 0) {
      best = n;
      break;
    }
  }
  if (best == 0) {
    const size_t para = prev.find_last_of('\n');
    const std::string last = trimCopy(
        para == std::string::npos ? prev : prev.substr(para + 1));
    if (last.size() >= 40 && add.compare(0, last.size(), last) == 0)
      best = last.size();
  }
  if (best == 0) return add;
  return trimCopy(add.substr(best));
}

/** Join ticks with a blank line so 64-token cuts do not smash words. */
inline std::string joinDeliverableText(const std::string &prev,
                                       const std::string &add) {
  const std::string chunk = trimCopy(add);
  if (chunk.empty()) return prev;
  if (prev.empty()) return chunk;
  std::string out = prev;
  const unsigned char last = static_cast<unsigned char>(out.back());
  const unsigned char first = static_cast<unsigned char>(chunk.front());
  /* "2." + "5 billion" must stay one number. A blank line here is what
     locked 727316: resume ended in "2." and every later tick was 15 chars. */
  if (last == '.' && out.size() >= 2 &&
      std::isdigit(static_cast<unsigned char>(out[out.size() - 2])) &&
      std::isdigit(first)) {
    out += chunk;
    return out;
  }
  if (!std::isspace(last) && !std::isspace(first)) out += "\n\n";
  out += chunk;
  return out;
}

/** A model `write` that is shorter than the on-disk body is a fragment or
    a context-window snapshot, not a full rewrite.  Keep `prev` and append
    only the part of `incoming` that is not already the file tail. */
inline std::string recoverDeliverableFromShorterWrite(const std::string &prev,
                                                      const std::string &incoming) {
  if (incoming.empty()) return prev;
  if (prev.empty()) return incoming;
  if (incoming.size() >= prev.size()) return incoming;
  if (prev.compare(prev.size() - incoming.size(), incoming.size(), incoming) ==
      0)
    return prev;
  const std::string add = stripOverlappingContinuation(prev, incoming);
  if (add.empty()) return prev;
  return joinDeliverableText(prev, add);
}

inline bool looksLikeTailCompletion(const std::string &prev,
                                    const std::string &add) {
  const std::string a = trimCopy(add);
  if (a.empty() || a.size() > 80) return false;
  if (prev.empty()) return false;
  size_t i = prev.size();
  while (i > 0 &&
         std::isspace(static_cast<unsigned char>(prev[i - 1])))
    --i;
  if (i == 0) return false;
  const unsigned char last = static_cast<unsigned char>(prev[i - 1]);
  const unsigned char first = static_cast<unsigned char>(a.front());
  if (last == '.' && i >= 2 &&
      std::isdigit(static_cast<unsigned char>(prev[i - 2])))
    return std::isdigit(first) != 0;
  if (std::isdigit(last)) return std::isalnum(first) != 0;
  return false;
}

/** Drop filename / fence wrappers the model sometimes prefixes. */
inline std::string stripDeliverableWrappers(std::string body) {
  body = trimCopy(body);
  auto dropFirstLineIf = [&](const char *pfx) {
    if (body.rfind(pfx, 0) != 0) return;
    const size_t eol = body.find('\n');
    body = eol == std::string::npos ? std::string() : body.substr(eol + 1);
    body = trimCopy(body);
  };
  dropFirstLineIf("**deliverable.md**");
  dropFirstLineIf("```markdown");
  dropFirstLineIf("```md");
  dropFirstLineIf("```Markdown");
  return body;
}

/** Cut before leaked ChatML / special tokens. Those are control, not prose. */
inline std::string cutAtChatMarkup(std::string s) {
  static const char *marks[] = {
      "<|eom_id|>",
      "<|eot_id|>",
      "<|im_end|>",
      "<|im_start|>",
      "<|start_header_id|>",
      "<|end_header_id|>",
      "<|end_of_text|>",
      "<|begin_of_text|>",
  };
  size_t best = std::string::npos;
  for (const char *m : marks) {
    const auto p = s.find(m);
    if (p != std::string::npos && (best == std::string::npos || p < best))
      best = p;
  }
  if (best != std::string::npos)
    s.resize(best);
  return s;
}

/** Strip leaked control / ChatML before appending plain markdown.
    Do not rewrite document syntax (fences stay). */
inline std::string sanitizePlainDeliverableChunk(std::string s) {
  if (looksLikeStructuredMissionReply(s)) return std::string();
  s = cutAtChatMarkup(std::move(s));
  const std::string bad[] = {
      "Current internal state",
      "Current Internal State",
      "\"arousal\"",
      "internal only - never copy",
  };
  for (const auto &needle : bad) {
    const auto pos = s.find(needle);
    if (pos != std::string::npos) {
      s = s.substr(0, pos);
      break;
    }
  }
  s = trimCopy(s);
  return s;
}

/** Sanitize content for deliverable.md write/append JSON actions. */
inline std::string sanitizeDeliverableActionContent(const std::string &content) {
  return sanitizePlainDeliverableChunk(content);
}

/** True when a line is leaked control JSON, not document prose. */
inline bool isPollutedDeliverableLine(const std::string &line) {
  const std::string t = trimCopy(line);
  if (t.empty()) return false;
  if (t.rfind("JSON {", 0) == 0 || t.rfind("json {", 0) == 0) return true;
  if (t.find("\"action\"") != std::string::npos &&
      t.find("deliverable") != std::string::npos)
    return true;
  if (t.rfind("### Current internal state", 0) == 0 ||
      t.rfind("## Current Internal State", 0) == 0)
    return true;
  if (t.front() == '{' && t.find("\"arousal\"") != std::string::npos &&
      t.find("\"valence\"") != std::string::npos)
    return true;
  return false;
}

/** Remove leaked JSON / internal-state lines from an existing deliverable. */
inline std::string scrubPollutedDeliverable(std::string body) {
  if (body.empty()) return body;
  std::ostringstream out;
  bool skipBlock = false;
  size_t pos = 0;
  while (pos < body.size()) {
    size_t eol = body.find('\n', pos);
    if (eol == std::string::npos) eol = body.size();
    std::string line = body.substr(pos, eol - pos);
    const std::string t = trimCopy(line);
    if (!skipBlock && isPollutedDeliverableLine(t)) {
      skipBlock = true;
      pos = (eol < body.size()) ? eol + 1 : eol;
      continue;
    }
    if (skipBlock) {
      if (!t.empty() && isPollutedDeliverableLine(t)) {
        pos = (eol < body.size()) ? eol + 1 : eol;
        continue;
      }
      if (t.empty() || t[0] == '#') skipBlock = false;
      else {
        pos = (eol < body.size()) ? eol + 1 : eol;
        continue;
      }
    }
    out << line;
    if (eol < body.size()) out << '\n';
    pos = eol + 1;
  }
  return trimCopy(out.str());
}

/** True for markdown section headings used in deliverable.md. */
inline bool isDeliverableSectionLine(const std::string &line) {
  const std::string t = trimCopy(line);
  if (t.empty()) return false;
  /* Only real headings. Any **bold** line used to count, so a title like
     **Deliverable: ...** or a list **Total budget** reopened the gate and
     heading-cut discarded the rest of the tick (727316: 1255 -> 39). */
  if (t.front() == '#') return true;
  if (t.rfind("**Chapter", 0) == 0) return true;
  if (t.rfind("Chapter ", 0) == 0 && t.size() > 10) return true;
  return false;
}

inline std::string normalizeSectionKey(std::string line) {
  line = trimCopy(line);
  while (!line.empty() && line[0] == '#') line.erase(line.begin());
  line = trimCopy(line);
  while (line.size() >= 4 && line.rfind("**", 0) == 0) {
    const auto end = line.find("**", 2);
    if (end == std::string::npos) {
      line = trimCopy(line.substr(2));
      break;
    }
    line = trimCopy(line.substr(2, end - 2));
  }
  while (!line.empty() && (line.back() == ':' || line.back() == '.'))
    line.pop_back();
  line = trimCopy(line);
  std::transform(line.begin(), line.end(), line.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  return line;
}

inline std::vector<std::string>
collectDeliverableSectionLines(const std::string &body) {
  std::vector<std::string> out;
  size_t pos = 0;
  while (pos < body.size()) {
    const size_t eol = body.find('\n', pos);
    const std::string line = body.substr(
        pos, eol == std::string::npos ? std::string::npos : eol - pos);
    if (isDeliverableSectionLine(line)) out.push_back(trimCopy(line));
    if (eol == std::string::npos) break;
    pos = eol + 1;
  }
  return out;
}

/** Numbers taken from the assignment. Used to state operating
    conditions — never to copy Background Assumptions / Required
    Chapters prose into the chat or the file. */
struct GoalOperationalFacts {
  std::string year;
  std::string delay;
  std::string duration;
  bool landed = false;
  bool noRealtime = false;
  bool uncrewed = false;
};

inline std::string lowerCopy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

inline int firstIntIn(const std::string &s) {
  size_t i = 0;
  while (i < s.size() && !std::isdigit(static_cast<unsigned char>(s[i]))) ++i;
  if (i >= s.size()) return 0;
  return std::atoi(s.c_str() + static_cast<long>(i));
}

inline std::pair<int, int> parseDelaySpan(const std::string &delay) {
  int lo = 0;
  int hi = 0;
  size_t i = 0;
  while (i < delay.size() && !std::isdigit(static_cast<unsigned char>(delay[i])))
    ++i;
  if (i >= delay.size()) return {0, 0};
  lo = std::atoi(delay.c_str() + static_cast<long>(i));
  while (i < delay.size() && std::isdigit(static_cast<unsigned char>(delay[i])))
    ++i;
  while (i < delay.size() &&
         (delay[i] == '-' || delay[i] == ' ' ||
          static_cast<unsigned char>(delay[i]) == 0xe2))
    ++i;
  if (i < delay.size() && std::isdigit(static_cast<unsigned char>(delay[i])))
    hi = std::atoi(delay.c_str() + static_cast<long>(i));
  if (hi <= 0) hi = lo;
  return {lo, hi};
}

inline int parseDurationDays(const std::string &duration) {
  const int n = firstIntIn(duration);
  if (n <= 0) return 0;
  const std::string lower = lowerCopy(duration);
  if (lower.find("year") != std::string::npos) return n * 365;
  if (lower.find("sol") != std::string::npos) return n;
  if (lower.find("day") != std::string::npos) return n;
  return n;
}

inline int parseGoalWordCount(const std::string &goal) {
  const std::string lower = lowerCopy(goal);
  size_t p = 0;
  int best = 0;
  while ((p = lower.find("word", p)) != std::string::npos) {
    size_t start = p;
    while (start > 0) {
      const unsigned char c = static_cast<unsigned char>(goal[start - 1]);
      if (std::isdigit(c) || c == ',' || c == ' ')
        --start;
      else
        break;
    }
    std::string num;
    for (size_t i = start; i < p; ++i) {
      const unsigned char c = static_cast<unsigned char>(goal[i]);
      if (std::isdigit(c)) num.push_back(static_cast<char>(c));
    }
    if (!num.empty()) best = std::max(best, std::atoi(num.c_str()));
    p += 4;
  }
  return best;
}

inline GoalOperationalFacts extractGoalOperationalFacts(const std::string &goal) {
  GoalOperationalFacts out;
  if (goal.empty()) return out;
  const std::string lower = lowerCopy(goal);
  out.landed = lower.find("successfully landed") != std::string::npos ||
               lower.find("has landed") != std::string::npos ||
               lower.find("already landed") != std::string::npos;
  out.noRealtime = lower.find("without real-time") != std::string::npos ||
                   lower.find("without realtime") != std::string::npos ||
                   lower.find("no real-time human") != std::string::npos ||
                   lower.find("no realtime human") != std::string::npos;
  out.uncrewed = lower.find("uncrewed") != std::string::npos ||
                 lower.find("unmanned") != std::string::npos;
  {
    const auto p = goal.find("It is ");
    if (p != std::string::npos) {
      size_t i = p + 6;
      while (i < goal.size() &&
             std::isdigit(static_cast<unsigned char>(goal[i])))
        ++i;
      if (i > p + 9) out.year = goal.substr(p + 6, i - (p + 6));
    }
  }
  if (out.year.empty()) {
    const auto p = lower.find("year ");
    if (p != std::string::npos) {
      size_t i = p + 5;
      while (i < goal.size() &&
             std::isdigit(static_cast<unsigned char>(goal[i])))
        ++i;
      if (i >= p + 9) out.year = goal.substr(p + 5, i - (p + 5));
    }
  }
  {
    const auto p = goal.find("minutes");
    if (p != std::string::npos) {
      size_t start = p;
      while (start > 0) {
        const unsigned char c =
            static_cast<unsigned char>(goal[start - 1]);
        if (std::isdigit(c) || c == '-' || c == ' ' || c == 0xe2 ||
            c == 0x80 || c == 0x93 || c == 0x92)
          --start;
        else
          break;
      }
      std::string d = trimCopy(goal.substr(start, (p + 7) - start));
      for (size_t i = 0; i + 2 < d.size();) {
        if (static_cast<unsigned char>(d[i]) == 0xe2 &&
            static_cast<unsigned char>(d[i + 1]) == 0x80 &&
            (static_cast<unsigned char>(d[i + 2]) == 0x93 ||
             static_cast<unsigned char>(d[i + 2]) == 0x92)) {
          d.replace(i, 3, "-");
        } else {
          ++i;
        }
      }
      if (!d.empty()) out.delay = d;
    }
  }
  {
    static const char *units[] = {" sols", " days", " years"};
    for (const char *unit : units) {
      const auto p = goal.find(unit);
      if (p == std::string::npos) continue;
      size_t start = p;
      while (start > 0 &&
             (std::isdigit(static_cast<unsigned char>(goal[start - 1])) ||
              goal[start - 1] == ' ' || goal[start - 1] == ','))
        --start;
      std::string d = trimCopy(goal.substr(start, (p + std::strlen(unit)) - start));
      if (!d.empty()) {
        out.duration = d;
        break;
      }
    }
  }
  return out;
}

/** Compact constants for the system turn / empty-file seed.
    Only numbers and spans copied from this goal. Do not invent
    uncrewed / Helios / station nouns that the goal did not say. */
inline std::string formatGoalConstraintPin(const std::string &goal,
                                          size_t maxChars = 480) {
  const auto f = extractGoalOperationalFacts(goal);
  if (f.year.empty() && f.delay.empty() && f.duration.empty())
    return std::string();
  std::string src;
  if (!f.year.empty()) src += f.year + ". ";
  if (!f.delay.empty()) src += f.delay + ". ";
  if (!f.duration.empty()) src += f.duration + ". ";
  src = trimCopy(src);
  if (src.size() > maxChars) src.resize(maxChars);
  return src + "\n\n";
}

inline std::string stripDesignThePrefix(std::string title) {
  title = trimCopy(title);
  const std::string prefix = "Design the ";
  if (title.size() > prefix.size() &&
      title.compare(0, prefix.size(), prefix) == 0)
    title = trimCopy(title.substr(prefix.size()));
  auto stripWrap = [](std::string s) {
    while (!s.empty()) {
      const unsigned char c = static_cast<unsigned char>(s[0]);
      if (c == '"' || c == '\'') {
        s.erase(0, 1);
        continue;
      }
      if (c == 0xe2 && s.size() >= 3 &&
          static_cast<unsigned char>(s[1]) == 0x80) {
        s.erase(0, 3);
        continue;
      }
      break;
    }
    while (!s.empty()) {
      const unsigned char c = static_cast<unsigned char>(s.back());
      if (c == '"' || c == '\'') {
        s.pop_back();
        continue;
      }
      if (s.size() >= 3 &&
          static_cast<unsigned char>(s[s.size() - 3]) == 0xe2) {
        s.resize(s.size() - 3);
        continue;
      }
      break;
    }
    return trimCopy(s);
  };
  title = stripWrap(title);
  std::string cleaned;
  cleaned.reserve(title.size());
  for (size_t i = 0; i < title.size();) {
    const unsigned char c = static_cast<unsigned char>(title[i]);
    if (c == '"' || c == '\'') {
      ++i;
      continue;
    }
    if (c == 0xe2 && i + 2 < title.size() &&
        static_cast<unsigned char>(title[i + 1]) == 0x80) {
      const unsigned char t = static_cast<unsigned char>(title[i + 2]);
      if (t == 0x9c || t == 0x9d || t == 0x98 || t == 0x99) {
        i += 3;
        continue;
      }
    }
    cleaned.push_back(title[i]);
    ++i;
  }
  return trimCopy(cleaned);
}

inline std::string extractGoalTitle(const std::string &goal) {
  std::string title;
  const auto p = goal.find("Task Name");
  if (p != std::string::npos) {
    std::istringstream in(goal.substr(p));
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
      line = trimCopy(line);
      if (!line.empty()) {
        title = stripDesignThePrefix(line);
        break;
      }
    }
  }
  if (title.size() > 160) title.resize(160);
  return title;
}

inline bool deliverableChunkLooksDrifted(const std::string &s,
                                         const std::string &goal) {
  if (!deliverableChunkLooksDrifted(s)) return false;
  if (goal.empty()) return true;
  const std::string title = lowerCopy(extractGoalTitle(goal));
  const auto facts = extractGoalOperationalFacts(goal);
  std::string lower = s;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (!title.empty() && title.size() >= 4 &&
      lower.find(title.substr(0, std::min(title.size(), size_t{24}))) !=
          std::string::npos)
    return false;
  if (!facts.year.empty() && lower.find(lowerCopy(facts.year)) != std::string::npos)
    return false;
  if (!facts.duration.empty() &&
      lower.find(lowerCopy(facts.duration.substr(
          0, std::min(facts.duration.size(), size_t{8})))) != std::string::npos)
    return false;
  return true;
}

/** Empty draft stays empty. Do not invent year/delay text. */
inline std::string formatEmptyFileSeed(const std::string &) {
  return std::string();
}

/** Untyped working brief from THIS goal: text before the chapter table.
    Not a user/assistant turn. Empty if the goal has no such prefix. */
inline std::string extractGoalWorkingContext(const std::string &goal,
                                             size_t maxChars = 8000) {
  if (goal.empty() || maxChars == 0) return std::string();
  std::string src = goal;
  size_t cut = std::string::npos;
  size_t scan = 0;
  while (scan < src.size()) {
    size_t eol = src.find('\n', scan);
    if (eol == std::string::npos) eol = src.size();
    std::string line = trimCopy(src.substr(scan, eol - scan));
    while (!line.empty() && line[0] == '#') {
      line.erase(line.begin());
      line = trimCopy(line);
    }
    if (line.rfind("Chapter ", 0) == 0 && line.size() > 8 &&
        std::isdigit(static_cast<unsigned char>(line[8]))) {
      cut = scan;
      break;
    }
    scan = (eol == src.size()) ? src.size() : eol + 1;
  }
  if (cut != std::string::npos) src = src.substr(0, cut);
  src = trimCopy(src);
  /* The line immediately before "Chapter N" is usually a catalog
     section title (no sentence), not working prose. Leaving it as
     the causal tip made the 8B continue a chapter table. */
  if (cut != std::string::npos && !src.empty()) {
    const auto lastNl = src.find_last_of('\n');
    const std::string last =
        trimCopy(lastNl == std::string::npos ? src : src.substr(lastNl + 1));
    const bool headingOnly =
        !last.empty() && last.size() < 96 &&
        last.find('.') == std::string::npos &&
        last.find('?') == std::string::npos &&
        last.find('!') == std::string::npos;
    if (headingOnly)
      src = lastNl == std::string::npos
                ? std::string()
                : trimCopy(src.substr(0, lastNl));
  }
  /* Later heading sections are author-instructions (word count,
     output format). Those belong in RAG, not the causal stream —
     putting them next to the tip made the 8B restate the brief. */
  {
    const auto facts = extractGoalOperationalFacts(src);
    const bool haveFacts = !facts.year.empty() || !facts.delay.empty() ||
                           !facts.duration.empty();
    auto mentions = [&](const std::string &t) {
      if (t.empty()) return false;
      if (!facts.year.empty() && t.find(facts.year) != std::string::npos)
        return true;
      if (!facts.delay.empty() && t.find(facts.delay) != std::string::npos)
        return true;
      if (!facts.duration.empty() &&
          t.find(facts.duration) != std::string::npos)
        return true;
      return false;
    };
    if (haveFacts) {
      std::string kept;
      std::string heading;
      std::string body;
      auto flush = [&]() {
        if (!mentions(body) && !mentions(heading)) return;
        if (!kept.empty()) return;
        if (!heading.empty()) kept += heading + "\n";
        kept += trimCopy(body);
      };
      size_t pos = 0;
      while (pos <= src.size()) {
        size_t eol = src.find('\n', pos);
        if (eol == std::string::npos) eol = src.size();
        const std::string line = trimCopy(src.substr(pos, eol - pos));
        const bool headingOnly =
            !line.empty() && line.size() < 96 &&
            line.find('.') == std::string::npos &&
            line.find('?') == std::string::npos &&
            line.find('!') == std::string::npos;
        if (headingOnly || pos == src.size()) {
          flush();
          heading = headingOnly ? line : std::string();
          body.clear();
        } else if (!line.empty()) {
          if (!body.empty()) body += "\n";
          body += line;
        }
        if (eol == src.size()) break;
        pos = eol + 1;
      }
      flush();
      if (!kept.empty()) {
        const std::string title = extractGoalTitle(goal);
        if (!title.empty() && kept.find(title) == std::string::npos)
          kept = title + "\n\n" + kept;
        src = trimCopy(kept);
      }
    }
  }
  if (src.size() > maxChars) {
    src.resize(maxChars);
    const auto nl = src.rfind('\n');
    if (nl != std::string::npos && nl > maxChars / 2)
      src.resize(nl);
    src = trimCopy(src);
  }
  if (!src.empty() && src.back() != '\n') src.push_back('\n');
  return src;
}

/** Chapter lines taken from THIS goal (not a built-in outline). */
inline std::vector<std::string>
collectGoalChapterHeadings(const std::string &goal) {
  std::vector<std::string> out;
  size_t pos = 0;
  while (pos < goal.size()) {
    size_t eol = goal.find('\n', pos);
    if (eol == std::string::npos) eol = goal.size();
    std::string line = trimCopy(goal.substr(pos, eol - pos));
    pos = (eol == goal.size()) ? goal.size() : eol + 1;
    while (!line.empty() && line[0] == '#') {
      line.erase(line.begin());
      line = trimCopy(line);
    }
    if (line.rfind("Chapter ", 0) == 0 && line.size() > 8 &&
        std::isdigit(static_cast<unsigned char>(line[8])))
      out.push_back(line);
  }
  return out;
}

/** First chapter heading in the goal that the draft has not opened. */
inline std::string firstMissingGoalChapter(const std::string &goal,
                                           const std::string &draft) {
  const std::string dlow = lowerCopy(draft);
  for (const auto &h : collectGoalChapterHeadings(goal)) {
    std::string key = h;
    const auto colon = key.find(':');
    if (colon != std::string::npos) key = trimCopy(key.substr(0, colon));
    if (dlow.find(lowerCopy(key)) == std::string::npos) return h;
  }
  return std::string();
}

/** Web-searchable queries from THIS goal. Title / background / the first
    chapter the draft has not opened. No chapter table dump, no invented
    nouns. Owned by the search plugin's understanding of a situation. */
inline std::vector<std::string>
buildPluginSearchQueries(const std::string &goal, const std::string &draft = "",
                         size_t maxQueries = 3, size_t maxChars = 140) {
  std::vector<std::string> out;
  if (goal.empty() || maxQueries == 0) return out;
  auto clip = [&](std::string s) {
    s = trimCopy(s);
    if (s.size() <= maxChars) return s;
    s.resize(maxChars);
    const auto sp = s.find_last_of(" \t");
    if (sp != std::string::npos && sp > maxChars / 2) s.resize(sp);
    return trimCopy(s);
  };
  auto add = [&](std::string s) {
    s = clip(stripDesignThePrefix(trimCopy(s)));
    if (s.size() < 12) return;
    for (const auto &e : out) {
      if (e == s) return;
    }
    out.push_back(std::move(s));
  };
  const std::string title = extractGoalTitle(goal);
  const std::string work = extractGoalWorkingContext(goal, 1200);
  size_t pos = 0;
  int paras = 0;
  while (pos < work.size() && paras < 2 && out.size() < maxQueries) {
    size_t eol = work.find('\n', pos);
    if (eol == std::string::npos) eol = work.size();
    const std::string line = trimCopy(work.substr(pos, eol - pos));
    pos = (eol == work.size()) ? work.size() : eol + 1;
    if (line.size() < 40) continue;
    if (line.rfind("Task ", 0) == 0 || line.rfind("Chapter ", 0) == 0 ||
        line.rfind("#", 0) == 0)
      continue;
    if (line.find('.') == std::string::npos &&
        line.find('?') == std::string::npos &&
        line.find('!') == std::string::npos)
      continue;
    const std::string stripped = stripDesignThePrefix(line);
    if (!title.empty() &&
        (stripped == title ||
         line.find(title.substr(0, std::min(title.size(), size_t{12}))) !=
             std::string::npos))
      continue;
    std::string focused = line;
    size_t si = 0;
    while (si < focused.size()) {
      const size_t dot = focused.find(". ", si);
      if (dot == std::string::npos) break;
      const std::string sent = trimCopy(focused.substr(si, dot - si + 1));
      if (sent.size() >= 40) {
        focused = trimCopy(focused.substr(si));
        si = 0;
        break;
      }
      si = dot + 2;
    }
    if (si > 0 && si < focused.size())
      focused = trimCopy(focused.substr(si));
    add(focused);
    ++paras;
  }
  if (!title.empty()) add(title);
  const std::string missing = firstMissingGoalChapter(goal, draft);
  if (!missing.empty() && out.size() < maxQueries) {
    std::string topic = missing;
    const auto colon = topic.find(':');
    if (colon != std::string::npos) topic = trimCopy(topic.substr(colon + 1));
    std::string prefix = title;
    if (prefix.size() > 48) {
      const auto sp = prefix.find(' ', prefix.size() / 3);
      if (sp != std::string::npos) prefix = trimCopy(prefix.substr(sp));
    }
    if (!topic.empty()) {
      if (!prefix.empty()) add(prefix + " " + topic);
      else add(topic);
    }
  }
  if (out.empty()) {
    std::string fallback = trimCopy(extractGoalWorkingContext(goal, maxChars));
    if (fallback.empty()) fallback = clip(goal);
    add(fallback);
  }
  if (out.size() > maxQueries) out.resize(maxQueries);
  return out;
}

inline std::string buildPluginSearchQuery(const std::string &goal,
                                          size_t maxChars = 140) {
  const auto qs = buildPluginSearchQueries(goal, "", 1, maxChars);
  return qs.empty() ? std::string() : qs.front();
}

/** Display / log clip only. Not a module-to-module protocol.
    Knowledge travels as unit-query packets, not as a causal listicle. */
inline std::string formatPluginRetrievedPrefix(const std::string &raw,
                                              size_t maxChars = 4000) {
  std::string s = trimCopy(raw);
  if (s.empty() || maxChars == 0) return std::string();
  if (s.size() > maxChars) {
    s.resize(maxChars);
    const auto nl = s.rfind('\n');
    if (nl != std::string::npos && nl > maxChars / 2) s.resize(nl);
    s = trimCopy(s);
  }
  if (!s.empty() && s.back() != '\n') s.push_back('\n');
  return s;
}

/** Tokens of length >= minLen from this string (alnum, lower). */
inline std::vector<std::string> alnumTokensMinLen(const std::string &s,
                                                  size_t minLen) {
  std::vector<std::string> toks;
  std::string cur;
  for (unsigned char c : s) {
    if (std::isalnum(c))
      cur.push_back(static_cast<char>(std::tolower(c)));
    else if (!cur.empty()) {
      if (cur.size() >= minLen) toks.push_back(cur);
      cur.clear();
    }
  }
  if (cur.size() >= minLen) toks.push_back(cur);
  return toks;
}

inline int goalWorkTokenOverlap(const std::string &text,
                                const std::string &goal) {
  const std::string work = extractGoalWorkingContext(goal, 2000);
  std::unordered_set<std::string> focus;
  for (const auto &t : alnumTokensMinLen(work, 5)) focus.insert(t);
  int n = 0;
  std::unordered_set<std::string> seen;
  for (const auto &t : alnumTokensMinLen(text, 5)) {
    if (focus.count(t) && seen.insert(t).second) ++n;
  }
  return n;
}

/** Share THIS goal's extracted year / delay / duration — not generic
    topic overlap (a planet page shares "mars" with a surface-station
    assignment). Duration-only is ignored when year or delay exist:
    many unrelated mission pages mention a sol count. */
inline bool hitMentionsGoalYear(const std::string &text,
                                const GoalOperationalFacts &f) {
  if (f.year.empty() || text.empty()) return false;
  return hasAlnumWord(lowerCopy(text), lowerCopy(f.year));
}

inline bool hitMentionsGoalDelay(const std::string &text,
                                 const GoalOperationalFacts &f) {
  if (f.delay.empty() || text.empty()) return false;
  const std::string lower = lowerCopy(text);
  const std::string d = lowerCopy(f.delay);
  if (lower.find(d) != std::string::npos) return true;
  const bool hasUnit =
      (d.find("minute") != std::string::npos &&
       lower.find("minute") != std::string::npos) ||
      (d.find("second") != std::string::npos &&
       lower.find("second") != std::string::npos) ||
      (d.find("hour") != std::string::npos &&
       lower.find("hour") != std::string::npos);
  if (!hasUnit) return false;
  const auto span = parseDelaySpan(f.delay);
  if (span.first > 0 &&
      hasAlnumWord(lower, std::to_string(span.first)))
    return true;
  if (span.second > 0 && span.second != span.first &&
      hasAlnumWord(lower, std::to_string(span.second)))
    return true;
  return false;
}

inline bool hitMentionsGoalDuration(const std::string &text,
                                    const GoalOperationalFacts &f) {
  if (f.duration.empty() || text.empty()) return false;
  const std::string lower = lowerCopy(text);
  const std::string d = lowerCopy(f.duration);
  if (lower.find(d) != std::string::npos) return true;
  const int n = firstIntIn(f.duration);
  if (n <= 0 || lower.find(std::to_string(n)) == std::string::npos)
    return false;
  if (d.find("sol") != std::string::npos &&
      (hasAlnumWord(lower, "sol") || hasAlnumWord(lower, "sols")))
    return true;
  if (d.find("year") != std::string::npos && hasAlnumWord(lower, "year"))
    return true;
  if (d.find("day") != std::string::npos &&
      (hasAlnumWord(lower, "day") || hasAlnumWord(lower, "days")))
    return true;
  return false;
}

inline bool searchHitAlignedToGoal(const std::string &text,
                                   const std::string &goal) {
  if (text.empty() || goal.empty()) return false;
  const auto facts = extractGoalOperationalFacts(goal);
  const bool haveYearOrDelay =
      !facts.year.empty() || !facts.delay.empty();
  if (haveYearOrDelay)
    return hitMentionsGoalYear(text, facts) ||
           hitMentionsGoalDelay(text, facts);
  if (!facts.duration.empty())
    return hitMentionsGoalDuration(text, facts);
  return goalWorkTokenOverlap(text, goal) >= 3;
}

inline nlohmann::json keepSearchHitsAlignedToGoal(const nlohmann::json &hits,
                                                 const std::string &goal,
                                                 size_t maxKeep = 6) {
  nlohmann::json out = nlohmann::json::array();
  if (!hits.is_array() || goal.empty()) return out;
  for (const auto &h : hits) {
    if (!h.is_object()) continue;
    const std::string title = h.value("title", std::string());
    const std::string snip =
        h.value("snippet", h.value("text", std::string()));
    if (snip.size() < 40) continue;
    if (!searchHitAlignedToGoal(title + "\n" + snip, goal)) continue;
    out.push_back(h);
    if (out.size() >= maxKeep) break;
  }
  return out;
}

/** Unit-query packets from search hits. Same shape as unitQueryIOToJson. */
inline nlohmann::json searchHitsToUnitQueries(const nlohmann::json &hits,
                                             size_t maxUnits = 8) {
  nlohmann::json arr = nlohmann::json::array();
  if (!hits.is_array()) return arr;
  for (const auto &h : hits) {
    if (!h.is_object()) continue;
    const std::string title = h.value("title", std::string());
    std::string snip = h.value("snippet", h.value("text", std::string()));
    if (snip.size() < 40) continue;
    const std::string content =
        title.empty() ? snip : (title + "\n" + snip);
    arr.push_back(nlohmann::json{{"modality", "text"}, {"content", content}});
    if (arr.size() >= maxUnits) break;
  }
  return arr;
}

/** First working sentence from THIS goal (after a short year-stamp).
    Goes on the causal opening so continuation is the assignment, not a
    chapter table and not an empty heading. */
inline std::string extractGoalLeadParagraph(const std::string &goal,
                                            size_t maxChars = 480) {
  const std::string work = extractGoalWorkingContext(goal, 1600);
  size_t pos = 0;
  while (pos < work.size()) {
    size_t eol = work.find('\n', pos);
    if (eol == std::string::npos) eol = work.size();
    std::string line = trimCopy(work.substr(pos, eol - pos));
    pos = (eol == work.size()) ? work.size() : eol + 1;
    if (line.size() < 40) continue;
    if (line.find('.') == std::string::npos &&
        line.find('?') == std::string::npos &&
        line.find('!') == std::string::npos)
      continue;
    size_t si = 0;
    while (si < line.size()) {
      const size_t dot = line.find(". ", si);
      if (dot == std::string::npos) break;
      const std::string sent = trimCopy(line.substr(si, dot - si + 1));
      if (sent.size() >= 24) {
        line = trimCopy(line.substr(si));
        si = 0;
        break;
      }
      si = dot + 2;
    }
    if (si > 0 && si < line.size())
      line = trimCopy(line.substr(si));
    if (line.size() > maxChars) {
      line.resize(maxChars);
      const auto cut = line.find_last_of(".?!");
      if (cut != std::string::npos && cut > maxChars / 2)
        line.resize(cut + 1);
    }
    line = trimCopy(line);
    if (!line.empty() && line.back() != '\n') line += "\n\n";
    return line;
  }
  return std::string();
}

/** First chapter line from THIS goal, as a markdown heading.
    Pin-only completion continues this, not the assignment closer. */
inline std::string extractGoalOpeningHeading(const std::string &goal) {
  if (goal.empty()) return std::string();
  size_t pos = 0;
  while (pos < goal.size()) {
    size_t eol = goal.find('\n', pos);
    if (eol == std::string::npos) eol = goal.size();
    std::string line = trimCopy(goal.substr(pos, eol - pos));
    pos = eol + 1;
    if (line.rfind("## Chapter ", 0) == 0 || line.rfind("Chapter ", 0) == 0) {
      if (line.rfind("#", 0) != 0) line = "## " + line;
      return "\n" + line + "\n\n";
    }
  }
  return std::string();
}

/** Pin is already the file seed / prefix. Putting it on resume made
    sampling continue from the duration number (1 sol = USD). */
inline std::string formatEmptyFileResume(const std::string &) {
  return std::string();
}

/** No recent draft yet. Does not invent year/delay text. */
inline bool deliverableIsOnlyPin(const std::string &body,
                                 const std::string &) {
  return trimCopy(body).empty();
}

inline std::string formatExistingSectionsBlock(const std::string &body) {
  const auto lines = collectDeliverableSectionLines(body);
  if (lines.empty()) return std::string();
  std::ostringstream oss;
  oss << "Already-written headings (do not output these again):\n";
  for (const auto &ln : lines) oss << "  [done] " << ln << "\n";
  return oss.str();
}

/** Lines the agent wrote in plan.md (numbered / bullets / headings). */
inline std::vector<std::string> collectPlanItems(const std::string &plan) {
  std::vector<std::string> items;
  size_t pos = 0;
  while (pos < plan.size() && items.size() < 24) {
    size_t eol = plan.find('\n', pos);
    if (eol == std::string::npos) eol = plan.size();
    std::string line = trimCopy(plan.substr(pos, eol - pos));
    pos = eol + 1;
    if (line.empty() || line.rfind("[", 0) == 0) continue;
    bool structured = false;
    size_t i = 0;
    while (i < line.size() &&
           std::isdigit(static_cast<unsigned char>(line[i])))
      ++i;
    if (i > 0 && i < line.size() && (line[i] == '.' || line[i] == ')')) {
      line = trimCopy(line.substr(i + 1));
      structured = true;
    } else if (line[0] == '-' || line[0] == '*') {
      line = trimCopy(line.substr(1));
      structured = true;
    } else if (line[0] == '#') {
      while (!line.empty() && line[0] == '#') line.erase(line.begin());
      line = trimCopy(line);
      structured = true;
    }
    if (structured && line.size() >= 3) items.push_back(std::move(line));
  }
  return items;
}

/** Extract outline item titles from a goal/outline blob (Chapter lines). */
inline std::vector<std::string>
collectOutlineItems(const std::string &outlineOrGoal) {
  std::vector<std::string> items;
  size_t pos = 0;
  while (pos < outlineOrGoal.size() && items.size() < 24) {
    const size_t hit = outlineOrGoal.find("Chapter ", pos);
    if (hit == std::string::npos) break;
    size_t lineEnd = outlineOrGoal.find('\n', hit);
    if (lineEnd == std::string::npos) lineEnd = outlineOrGoal.size();
    std::string line = trimCopy(outlineOrGoal.substr(hit, lineEnd - hit));
    if (!line.empty()) items.push_back(line);
    pos = lineEnd + 1;
  }
  return items;
}

inline bool sectionKeyMatches(const std::string &outlineKey,
                              const std::string &doneKey) {
  if (outlineKey.empty() || doneKey.empty()) return false;
  if (outlineKey == doneKey) return true;
  if (doneKey.find(outlineKey) != std::string::npos) return true;
  if (outlineKey.find(doneKey) != std::string::npos) return true;
  return false;
}

/** I/O unit is a whole paragraph, not a token stream.
    Tokenizer/detokenizer may stream internally; this drops an incomplete
    last paragraph/sentence so deliverable.md never stores a mid-word cut. */
inline std::string keepCompleteParagraphIo(const std::string &raw) {
  if (raw.empty()) return std::string();
  auto isDecimalDot = [](const std::string &s, size_t i) -> bool {
    if (i >= s.size() || s[i] != '.') return false;
    return i > 0 && i + 1 < s.size() &&
           std::isdigit(static_cast<unsigned char>(s[i - 1])) &&
           std::isdigit(static_cast<unsigned char>(s[i + 1]));
  };
  auto completeEnd = [&](const std::string &p) -> bool {
    const std::string t = trimCopy(p);
    if (t.empty()) return false;
    if (t[0] == '#') return false;
    const char c = t.back();
    if (c == ':' && t.size() < 80) return false;
    if ((t[0] == '*' || t[0] == '-') && t.size() >= 24 && c != ':')
      return true;
    if (c == '.' || c == '!' || c == '?' || c == ':' || c == '"' ||
        c == '\'' || c == ')' || c == '`')
      return true;
    if (c == '*' || c == '-') return true;
    return t.size() >= 120;
  };
  std::vector<std::string> paras;
  size_t pos = 0;
  while (pos < raw.size()) {
    size_t eol = raw.find("\n\n", pos);
    if (eol == std::string::npos) {
      paras.push_back(raw.substr(pos));
      break;
    }
    paras.push_back(raw.substr(pos, eol - pos));
    pos = eol + 2;
  }
  if (paras.size() == 1) {
    if (completeEnd(paras[0])) return trimCopy(paras[0]);
    const std::string &one = paras[0];
    size_t last = std::string::npos;
    for (size_t i = one.size(); i-- > 0;) {
      if (one[i] == '!' || one[i] == '?') {
        last = i;
        break;
      }
      if (one[i] == '.' && !isDecimalDot(one, i)) {
        last = i;
        break;
      }
    }
    if (last == std::string::npos || last < 20) return std::string();
    return trimCopy(one.substr(0, last + 1));
  }
  while (!paras.empty() && !completeEnd(paras.back()))
    paras.pop_back();
  if (paras.empty()) return std::string();
  std::ostringstream oss;
  for (size_t i = 0; i < paras.size(); ++i) {
    if (i) oss << "\n\n";
    oss << paras[i];
  }
  return trimCopy(oss.str());
}

/** True when the deliverable ends mid-sentence (no clean paragraph break). */
inline bool deliverableEndsIncomplete(const std::string &body) {
  const std::string t = trimCopy(body);
  if (t.empty()) return false;
  const char c = t.back();
  if (c == '.' || c == '!' || c == '?' || c == ':' || c == ')' || c == '"' ||
      c == '\'' || c == '`')
    return false;
  if (c == '*' || c == '-' || c == '#') return false; /* list/heading end */
  return true;
}

/**
 * Structural continuation cue for the next deliberator tick.
 * Root cause of repeated chapter titles: each tick looked like a fresh
 * "write the document" job (full OUTLINE starting at Chapter 1) without a
 * hard resume anchor. This block makes the task explicitly APPEND-only.
 */
inline bool isGoalChapterHeadingLine(const std::string &line) {
  const std::string t = trimCopy(line);
  if (t.rfind("Chapter ", 0) == 0) {
    size_t i = 8;
    if (i >= t.size() || !std::isdigit(static_cast<unsigned char>(t[i])))
      return false;
    while (i < t.size() && std::isdigit(static_cast<unsigned char>(t[i]))) ++i;
    return i < t.size() && (t[i] == ':' || t[i] == '.' || t[i] == ' ' ||
                            t[i] == '\t');
  }
  return trimCopy(line).rfind("Appendix", 0) == 0;
}

/** Chapter/appendix heading lines taken from the assignment itself. */
inline std::string formatGoalChapterLines(const std::string &goal) {
  std::vector<std::string> heads;
  {
    std::istringstream in(goal);
    std::string line;
    while (std::getline(in, line)) {
      if (isGoalChapterHeadingLine(line)) heads.push_back(trimCopy(line));
    }
  }
  if (heads.size() < 3) return std::string();
  std::ostringstream oss;
  oss << "The goal itself names these chapters. They are the assignment, "
         "not headings copied from the draft:\n";
  for (const auto &h : heads) oss << "  " << h << "\n";
  return oss.str();
}

inline std::vector<std::string> splitAllParagraphs(const std::string &t) {
  std::vector<std::string> paras;
  size_t pos = 0;
  while (pos < t.size()) {
    size_t eol = t.find("\n\n", pos);
    if (eol == std::string::npos) eol = t.size();
    std::string para = trimCopy(t.substr(pos, eol - pos));
    if (eol >= t.size())
      pos = eol;
    else
      pos = eol + 2;
    if (!para.empty()) paras.push_back(std::move(para));
  }
  return paras;
}

inline std::string joinParagraphs(const std::vector<std::string> &paras,
                                  size_t from, size_t to) {
  std::ostringstream oss;
  bool first = true;
  const size_t end = (to < paras.size()) ? to : paras.size();
  for (size_t i = from; i < end; ++i) {
    if (!first) oss << "\n\n";
    first = false;
    oss << paras[i];
  }
  return oss.str();
}

inline std::string prefixBeforeRepeatedShortLine(const std::string &reply);

/** Causal resume only: whole-paragraph tail, even fences, no closer.
    Outline / heading lists stay in RAG (buildMissionOutline), not here.
    Disk is not rewritten. */
inline std::string formatContinuationContext(const std::string &body,
                                             const std::string &outlineOrGoal,
                                             size_t resumeChars = 720) {
  if (body.empty()) return std::string();

  std::ostringstream oss;
  (void)outlineOrGoal;
  std::string src = body;
  {
    std::string lower = src;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    (void)lower;
    src = prefixBeforeRepeatedShortLine(src);
  }
  while (!src.empty()) {
    const auto lastBreak = src.rfind("\n\n");
    const std::string last =
        lastBreak == std::string::npos ? src : src.substr(lastBreak + 2);
    if (!isReaderAddressText(last)) break;
    if (lastBreak == std::string::npos) {
      src.clear();
      break;
    }
    src.resize(lastBreak);
    while (!src.empty() && (src.back() == '\n' || src.back() == '\r' ||
                            src.back() == ' '))
      src.pop_back();
  }
  if (src.empty()) src = body;
  /* Do not prime the next tick with a trailing paragraph that already
     appears earlier in the draft — that is how 5807996 locked onto a
     slogan and every later generate was dropped as a repeat. */
  {
    const auto paras = splitAllParagraphs(src);
    size_t end = paras.size();
    while (end > 1) {
      const std::string &p = paras[end - 1];
      if (p.size() < 72) break;
      bool earlier = false;
      for (size_t i = 0; i + 1 < end; ++i) {
        if (paras[i] == p) {
          earlier = true;
          break;
        }
      }
      if (!earlier) break;
      --end;
    }
    if (end < paras.size() && end > 0)
      src = joinParagraphs(paras, 0, end);
  }
  /* Resume is whole paragraphs. A mid-word char window made the next
     tick continue "FAULT" as "ISOLATION = 4". */
  size_t start = src.size() > resumeChars ? src.size() - resumeChars : 0;
  if (start > 0) {
    const auto aligned = src.find("\n\n", start);
    if (aligned != std::string::npos && aligned + 2 < src.size())
      start = aligned + 2;
    else {
      const auto prev = src.rfind("\n\n", start);
      if (prev != std::string::npos)
        start = prev + 2;
    }
  }
  std::string tail = src.substr(start);
  {
    const auto paras = splitAllParagraphs(tail);
    std::vector<std::string> kept;
    bool dropRest = false;
    for (const auto &p : paras) {
      const std::string t = trimCopy(p);
      std::string low = t;
      std::transform(low.begin(), low.end(), low.begin(),
                     [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                     });
      if (low.rfind("plan:", 0) == 0 ||
          low.rfind("plan :", 0) == 0) {
        dropRest = true;
        continue;
      }
      if (dropRest && (t.empty() || t[0] == '-' ||
                       (t.size() > 2 && t[1] == '.' && std::isdigit(
                           static_cast<unsigned char>(t[0])))))
        continue;
      dropRest = false;
      kept.push_back(p);
    }
    if (!kept.empty())
      tail = joinParagraphs(kept, 0, kept.size());
  }
  /* Drop a trailing fragment only when it is unfinished. A finished
     short paragraph (first sentence after the pin) is the resume tip. */
  const auto lastBreak = tail.rfind("\n\n");
  if (lastBreak != std::string::npos && tail.size() - lastBreak < 72) {
    const std::string last = trimCopy(tail.substr(lastBreak + 2));
    const bool finished =
        !last.empty() &&
        (last.back() == '.' || last.back() == '!' || last.back() == '?' ||
         last.back() == ':' || last[0] == '#' || last[0] == '*');
    if (!finished)
      tail = tail.substr(0, lastBreak);
  }
  tail = balanceResumeFences(std::move(tail));
  oss << tail;
  if (!tail.empty() && tail.back() != '\n')
    oss << '\n';
  return oss.str();
}

/** Resume is the draft tail only. A pin-only file is not a window. */
inline std::string formatMissionResumeSuffix(const std::string &body,
                                            const std::string &goal,
                                            size_t resumeChars = 720) {
  if (body.empty() || deliverableIsOnlyPin(body, goal))
    return std::string();
  return formatContinuationContext(body, std::string(), resumeChars);
}

/** Recent context is the draft window. Split by size so n_resume can
    protect the tail. Do not invent year/delay text. */
inline std::pair<std::string, std::string>
splitDeliverableCausal(const std::string &body, const std::string &,
                       size_t resumeChars = 8000) {
  if (body.empty()) return {std::string(), std::string()};
  const size_t cap = resumeChars == 0 ? body.size() : resumeChars;
  if (body.size() <= cap) return {body, std::string()};
  return {body.substr(0, body.size() - cap), body.substr(body.size() - cap)};
}

inline const char *contradictedOperationalFact(const std::string &reply,
                                               const std::string &goal);

/** Computable defects in the model's own draft. Facts only — not an outline. */
inline std::string inspectDeliverableHealth(const std::string &body,
                                           const std::string &goal = "") {
  if (body.size() < 80) return std::string();
  std::vector<std::string> notes;

  size_t fences = 0;
  for (size_t i = 0; i + 2 < body.size(); ++i)
    if (body.compare(i, 3, "```") == 0) ++fences;
  if (fences % 2 == 1)
    notes.emplace_back("- markdown fence is unclosed (odd ``` count); close "
                       "it before new prose");

  const std::string head =
      trimCopy(body.substr(0, std::min(body.size(), size_t{64})));
  if (head.rfind("**deliverable.md**", 0) == 0 ||
      head.rfind("```markdown", 0) == 0)
    notes.emplace_back("- draft is wrapped in a filename/fence; write the "
                       "document body only");

  int chapterAsH3 = 0;
  std::unordered_set<int> chapterMajors;
  std::unordered_set<int> subsectionMajors;
  size_t pos = 0;
  while (pos < body.size()) {
    size_t eol = body.find('\n', pos);
    if (eol == std::string::npos) eol = body.size();
    const std::string line = trimCopy(body.substr(pos, eol - pos));
    pos = eol + 1;
    if (line.rfind("### Chapter", 0) == 0 ||
        line.rfind("#### Chapter", 0) == 0)
      ++chapterAsH3;
    const size_t ch = line.find("Chapter ");
    if (ch != std::string::npos && line.size() > ch + 8 &&
        std::isdigit(static_cast<unsigned char>(line[ch + 8])))
      chapterMajors.insert(std::atoi(line.c_str() + static_cast<long>(ch + 8)));
    if (line.rfind("### ", 0) == 0 && line.size() > 4 &&
        std::isdigit(static_cast<unsigned char>(line[4])))
      subsectionMajors.insert(std::atoi(line.c_str() + 4));
  }
  if (chapterAsH3)
    notes.emplace_back("- a Chapter title is at ### / ####; keep chapters at "
                       "## and nest subsections under them");
  for (int maj : subsectionMajors) {
    if (maj > 0 && !chapterMajors.count(maj)) {
      notes.emplace_back("- subsection " + std::to_string(maj) +
                         ".x appears before a Chapter " +
                         std::to_string(maj) + " heading");
      break;
    }
  }

  const auto secs = collectDeliverableSectionLines(body);
  std::unordered_set<std::string> seenKeys;
  int dups = 0;
  for (const auto &ln : secs) {
    if (!seenKeys.insert(normalizeSectionKey(ln)).second) ++dups;
  }
  if (dups)
    notes.emplace_back("- some headings appear more than once; do not reopen "
                       "them");

  if (body.size() >= 200) {
    const size_t win = 56;
    const std::string tail =
        body.substr(body.size() - std::min(body.size(), size_t{900}));
    std::string probe;
    size_t p = tail.size();
    while (p > 0) {
      size_t start = tail.rfind('\n', p - 1);
      if (start == std::string::npos)
        start = 0;
      else
        ++start;
      const std::string para = trimCopy(tail.substr(start, p - start));
      if (para.size() >= win) {
        probe = para.substr(0, win);
        break;
      }
      if (start == 0) break;
      p = start - 1;
    }
    if (probe.size() >= win) {
      const std::string hay = body.substr(0, body.size() - 80);
      const size_t first = hay.find(probe);
      if (first != std::string::npos &&
          hay.find(probe, first + win) != std::string::npos)
        notes.emplace_back("- the last section repeats a paragraph already "
                           "written; continue with new design after the last "
                           "sentence");
    }
  }

  {
    /* Latest span only: the whole draft may discuss the constraint
       correctly. This is RAG, not a refuse gate. */
    const std::string tail =
        body.substr(body.size() - std::min(body.size(), size_t{800}));
    if (const char *fact = contradictedOperationalFact(tail, goal))
      notes.emplace_back(std::string("- latest span contradicts ") + fact);
  }
  {
    const auto lastBreak = body.rfind("\n\n");
    const std::string last =
        lastBreak == std::string::npos ? body : body.substr(lastBreak + 2);
    if (isReaderAddressText(last))
      notes.emplace_back("- the last paragraph addresses a reader; continue "
                         "the design and do not answer it");
  }

  if (!goal.empty() && body.size() >= 400 &&
      !deliverableIsOnlyPin(body, goal)) {
    const std::string work = extractGoalWorkingContext(goal, 8000);
    std::unordered_set<std::string> goalWords;
    std::string cur;
    auto flush = [&]() {
      if (cur.size() >= 6) goalWords.insert(cur);
      cur.clear();
    };
    for (unsigned char c : work) {
      if (std::isalnum(c))
        cur.push_back(static_cast<char>(std::tolower(c)));
      else
        flush();
    }
    flush();
    int hit = 0;
    if (goalWords.size() >= 8) {
      const std::string blow = lowerCopy(body);
      for (const auto &w : goalWords) {
        if (blow.find(w) != std::string::npos) ++hit;
        if (hit >= 2) break;
      }
      if (hit < 2)
        notes.emplace_back("- the draft shares almost none of this "
                           "assignment's working terms; continue from those "
                           "terms");
    }
  }

  if (notes.empty()) return std::string();
  std::ostringstream oss;
  oss << "Self-check (facts about the current draft):\n";
  for (const auto &n : notes) oss << n << "\n";
  oss << "\n";
  return oss.str();
}

/** Drop paragraphs already in the draft. Continuation often restates the
    last resume paragraph first; throwing the whole reply away for that
    leftover is what stalled 5807996. */
inline std::string stripAlreadyWrittenParagraphs(const std::string &reply,
                                                 const std::string &existing) {
  if (existing.empty()) return trimCopy(reply);
  const auto paras = splitAllParagraphs(reply);
  std::vector<std::string> kept;
  kept.reserve(paras.size());
  for (const auto &p : paras) {
    /* Exact span already in the file — including short title lines.
       Do not key off heading syntax; the same string is enough. */
    if (p.size() >= 16 && existing.find(p) != std::string::npos) continue;
    kept.push_back(p);
  }
  return trimCopy(joinParagraphs(kept, 0, kept.size()));
}

/** True when the same long paragraph appears twice inside this reply.
    Draft overlap is handled by stripAlreadyWrittenParagraphs. */
inline bool replyRepeatsDraftParagraph(const std::string &reply,
                                       const std::string &existing) {
  (void)existing;
  const std::string t = trimCopy(reply);
  if (t.size() < 72) return false;
  std::vector<std::string> paras;
  size_t pos = 0;
  while (pos < t.size()) {
    size_t eol = t.find("\n\n", pos);
    if (eol == std::string::npos) eol = t.find('\n', pos);
    if (eol == std::string::npos) eol = t.size();
    std::string para = trimCopy(t.substr(pos, eol - pos));
    if (eol >= t.size())
      pos = eol;
    else if (t.compare(eol, 2, "\n\n") == 0)
      pos = eol + 2;
    else
      pos = eol + 1;
    if (para.size() >= 72) paras.push_back(std::move(para));
  }
  if (paras.empty() && t.size() >= 72) paras.push_back(t);
  for (size_t i = 0; i < paras.size(); ++i) {
    const std::string &p = paras[i];
    int copies = 0;
    for (const auto &q : paras)
      if (q == p) ++copies;
    if (copies >= 2) return true;
  }
  return false;
}

/** One-liner that only restates this goal's extracted facts as brochure. */
inline bool replyIsBrochurePinRestatement(const std::string &reply,
                                          const std::string &goal = "") {
  std::string t = trimCopy(reply);
  if (t.size() < 24 || t.size() > 220) return false;
  if (t.find('\n') != std::string::npos) {
    std::string u = t;
    while (!u.empty() && (u.back() == '\n' || u.back() == '\r')) u.pop_back();
    if (u.find('\n') != std::string::npos) return false;
    t = std::move(u);
  }
  const std::string lower = lowerCopy(t);
  const auto facts = extractGoalOperationalFacts(goal);
  int hits = 0;
  if (!facts.year.empty() && lower.find(lowerCopy(facts.year)) != std::string::npos)
    ++hits;
  if (!facts.delay.empty()) {
    const auto span = parseDelaySpan(facts.delay);
    if (span.first > 0 &&
        lower.find(std::to_string(span.first)) != std::string::npos)
      ++hits;
  }
  if (!facts.duration.empty() &&
      lower.find(std::to_string(firstIntIn(facts.duration))) != std::string::npos)
    ++hits;
  if (facts.uncrewed &&
      (hasAlnumWord(lower, "uncrewed") || hasAlnumWord(lower, "unmanned")))
    ++hits;
  if (facts.noRealtime &&
      (lower.find("no real-time") != std::string::npos ||
       lower.find("real-time human") != std::string::npos))
    ++hits;
  if (hits < 1) return false;
  int words = 0;
  for (size_t i = 0; i < t.size();) {
    while (i < t.size() &&
           !std::isalnum(static_cast<unsigned char>(t[i])))
      ++i;
    if (i >= t.size()) break;
    ++words;
    while (i < t.size() && std::isalnum(static_cast<unsigned char>(t[i])))
      ++i;
  }
  if (words > 24) return false;
  if (lower.find("because") != std::string::npos) return false;
  return true;
}

/** Cut at the second copy of a long sentence/line. Token penalties did
    not stop an 8-copy slogan loop that used single newlines, so
    replyRepeatsDraftParagraph (\n\n only) never saw two paragraphs. */
inline std::string prefixBeforeSelfRepeat(const std::string &reply) {
  if (reply.size() < 160) return reply;
  std::vector<size_t> starts;
  starts.push_back(0);
  for (size_t i = 0; i + 1 < reply.size(); ++i) {
    if ((reply[i] == '.' || reply[i] == '!' || reply[i] == '?') &&
        (reply[i + 1] == ' ' || reply[i + 1] == '\n'))
      starts.push_back(i + 2);
    else if (reply[i] == '\n')
      starts.push_back(i + 1);
  }
  size_t cut = std::string::npos;
  for (size_t a = 0; a < starts.size(); ++a) {
    const size_t endA =
        (a + 1 < starts.size()) ? starts[a + 1] : reply.size();
    if (endA <= starts[a] + 72) continue;
    const std::string sent =
        trimCopy(reply.substr(starts[a], endA - starts[a]));
    if (sent.size() < 72) continue;
    const auto p = reply.find(sent, endA);
    if (p != std::string::npos && (cut == std::string::npos || p < cut))
      cut = p;
  }
  if (cut == std::string::npos) return reply;
  return trimCopy(reply.substr(0, cut));
}

/** Cut a same-reply slogan loop, then keep only paragraphs that are new. */
inline std::string keepUniqueContinuation(const std::string &reply,
                                          const std::string &existing) {
  return stripAlreadyWrittenParagraphs(
      prefixBeforeSelfRepeat(prefixBeforeRepeatedShortLine(reply)), existing);
}

/** True when a real-time mention is denied in the same window. */
inline bool realtimeMentionIsNegated(const std::string &lower, size_t at,
                                     size_t n) {
  const size_t lo = at > 40 ? at - 40 : 0;
  const size_t hi = std::min(lower.size(), at + n + 72);
  const std::string win = lower.substr(lo, hi - lo);
  return win.find("without") != std::string::npos ||
         win.find("impossible") != std::string::npos ||
         win.find("no real") != std::string::npos ||
         win.find("not real") != std::string::npos ||
         win.find("cannot") != std::string::npos;
}

/** Compared to THIS goal's extracted flags/spans. No topic phrase catalog. */
inline const char *contradictedOperationalFact(const std::string &reply,
                                               const std::string &goal) {
  if (reply.empty() || goal.empty()) return nullptr;
  const auto facts = extractGoalOperationalFacts(goal);
  const std::string lower = lowerCopy(trimCopy(reply));
  const bool mentionsEarth = hasAlnumWord(lower, "earth");
  const auto span = parseDelaySpan(facts.delay);

  if (facts.noRealtime) {
    bool live = false;
    size_t p = 0;
    while ((p = lower.find("real", p)) != std::string::npos) {
      const bool hyphen = p + 9 <= lower.size() &&
                          lower.compare(p, 9, "real-time") == 0;
      const bool glued = p + 8 <= lower.size() &&
                         lower.compare(p, 8, "realtime") == 0;
      const bool spaced = p + 9 <= lower.size() &&
                          lower.compare(p, 9, "real time") == 0;
      const size_t n = hyphen || spaced ? 9 : glued ? 8 : 0;
      if (n && !realtimeMentionIsNegated(lower, p, n)) {
        if (mentionsEarth || lower.find("communicat") != std::string::npos)
          live = true;
      }
      ++p;
    }
    if (live) return "real-time";
    if (mentionsEarth &&
        (hasAlnumWord(lower, "remote") ||
         hasAlnumWord(lower, "remotely")) &&
        (hasAlnumWord(lower, "control") ||
         hasAlnumWord(lower, "controlled") ||
         hasAlnumWord(lower, "controller") ||
         hasAlnumWord(lower, "controllers")))
      return "earth+remote control";
    if (mentionsEarth &&
        (hasAlnumWord(lower, "controller") ||
         hasAlnumWord(lower, "controllers")))
      return "earth+remote control";
  }

  if (span.second > 0) {
    const bool delayCtx = lower.find("delay") != std::string::npos ||
                          lower.find("communicat") != std::string::npos ||
                          mentionsEarth;
    if (delayCtx) {
      if ((lower.find("hour") != std::string::npos) && span.second <= 60 &&
          (lower.find("over") != std::string::npos ||
           lower.find("several") != std::string::npos ||
           lower.find("hours later") != std::string::npos ||
           lower.find("more than") != std::string::npos))
        return lower.find("later") != std::string::npos ? "delay-hours"
                                                       : "delay>max";
      int nums[4] = {0, 0, 0, 0};
      int nN = 0;
      for (size_t i = 0; i < lower.size() && nN < 4; ++i) {
        if (!std::isdigit(static_cast<unsigned char>(lower[i]))) continue;
        int v = 0;
        size_t j = i;
        while (j < lower.size() &&
               std::isdigit(static_cast<unsigned char>(lower[j]))) {
          v = v * 10 + (lower[j] - '0');
          ++j;
        }
        if (lower.find("minute", j) != std::string::npos &&
            lower.find("minute", j) < j + 16)
          nums[nN++] = v;
        i = j;
      }
      if (nN >= 2 && (nums[0] != span.first || nums[1] != span.second))
        return "delay-range";
    }
  }

  if (facts.landed) {
    if (lower.find("launch") != std::string::npos)
      return "prelaunch";
    if (hasAlnumWord(lower, "orbit") || hasAlnumWord(lower, "orbiting") ||
        lower.find("station's orbit") != std::string::npos)
      return "surface-station-orbit";
  }

  if (facts.uncrewed &&
      (lower.find("humans on") != std::string::npos ||
       lower.find("on board") != std::string::npos ||
       lower.find("onboard crew") != std::string::npos))
    return "crew-on-spacecraft";
  return nullptr;
}

/** Detector only. Live path puts the fact into RAG / observe();
    do not refuse the append. */
inline bool replyContradictsOperationalFacts(const std::string &reply,
                                             const std::string &goal = "") {
  return contradictedOperationalFact(reply, goal) != nullptr;
}

/** Cut when the same short line repeats. No phrase list — structure only. */
inline std::string prefixBeforeRepeatedShortLine(const std::string &reply) {
  if (reply.size() < 32) return reply;
  std::vector<std::string> lines;
  std::vector<size_t> offs;
  size_t i = 0;
  while (i < reply.size()) {
    size_t nl = reply.find('\n', i);
    if (nl == std::string::npos) nl = reply.size();
    offs.push_back(i);
    lines.push_back(trimCopy(reply.substr(i, nl - i)));
    i = (nl == reply.size()) ? reply.size() : nl + 1;
    if (nl == reply.size()) break;
  }
  std::unordered_map<std::string, int> cnt;
  for (const auto &ln : lines) {
    if (ln.size() >= 4 && ln.size() <= 64) cnt[ln]++;
  }
  std::string worst;
  int worstN = 0;
  for (const auto &kv : cnt) {
    if (kv.second > worstN) {
      worstN = kv.second;
      worst = kv.first;
    }
  }
  if (worstN < 4 || worst.empty()) return reply;
  int seen = 0;
  for (size_t k = 0; k < lines.size(); ++k) {
    if (lines[k] != worst) continue;
    ++seen;
    if (seen == 2) return trimCopy(reply.substr(0, offs[k]));
  }
  return reply;
}

/** True when most of the reply is one short line looping. */
inline bool replyLooksLikeShortLineLoop(const std::string &reply) {
  const std::string t = trimCopy(reply);
  if (t.size() < 32) return false;
  const std::string cut = prefixBeforeRepeatedShortLine(t);
  return !cut.empty() && cut.size() * 4 < t.size();
}

/** Structural loop cut only. Phrase lists are not a knowledge fix. */
inline std::string prefixBeforeForbiddenCloser(const std::string &reply) {
  return prefixBeforeRepeatedShortLine(reply);
}

/**
 * True when the reply re-opens an already-written section.
 * Catches both a leading heading line and the common TinyLlama failure mode
 * of gluing "**Chapter N**" onto the previous incomplete sentence with no
 * newline (e.g. "...following**Chapter 1: ...**").
 *
 * Mid-prose references like "see **Chapter 1** above" are ignored: only
 * heading-at-start and newline-prefixed headings are treated as restarts.
 */
inline bool replyRestartsExistingHeading(const std::string &reply,
                                         const std::string &existing) {
  if (reply.empty() || existing.empty()) return false;

  std::vector<std::string> doneChapterKeys;
  for (const auto &ln : collectDeliverableSectionLines(existing)) {
    const std::string key = normalizeSectionKey(ln);
    if (!key.empty()) doneChapterKeys.push_back(key);
  }
  if (doneChapterKeys.empty()) return false;

  auto extractHeadingCandidate = [](const std::string &raw) -> std::string {
    std::string cand = trimCopy(raw);
    if (cand.rfind("**", 0) == 0) {
      const size_t close = cand.find("**", 2);
      if (close != std::string::npos) cand = cand.substr(0, close + 2);
    }
    return cand;
  };

  auto headingIsContinuation = [](const std::string &raw) -> bool {
    std::string lower = raw;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    return lower.find("continued") != std::string::npos ||
           lower.find("(cont.") != std::string::npos ||
           lower.find("(cont)") != std::string::npos;
  };

  auto matchesDone = [&](const std::string &raw) -> bool {
    if (headingIsContinuation(raw)) return false;
    if (!isDeliverableSectionLine(raw) && raw.rfind("**Chapter", 0) != 0 &&
        raw.rfind("Chapter ", 0) != 0)
      return false;
    const std::string key = normalizeSectionKey(raw);
    for (const auto &dk : doneChapterKeys) {
      if (sectionKeyMatches(key, dk)) return true;
    }
    return false;
  };

  /* 1) first non-empty line is a done heading (incl. glued at reply start) */
  size_t pos = 0;
  while (pos < reply.size()) {
    const size_t eol = reply.find('\n', pos);
    const std::string line = trimCopy(reply.substr(
        pos, eol == std::string::npos ? std::string::npos : eol - pos));
    if (!line.empty()) {
      if (matchesDone(extractHeadingCandidate(line))) return true;
      break;
    }
    if (eol == std::string::npos) break;
    pos = eol + 1;
  }

  /* 2) later paragraph that re-opens a done heading (newline-prefixed) */
  const char *markers[] = {"\n**Chapter", "\n**", "\n### ", "\n## ", "\n# "};
  for (const char *m : markers) {
    size_t hit = 0;
    const std::string marker(m);
    while ((hit = reply.find(marker, hit)) != std::string::npos) {
      const size_t start = hit + 1;
      size_t end = reply.find('\n', start);
      if (end == std::string::npos) end = reply.size();
      if (matchesDone(extractHeadingCandidate(reply.substr(start, end - start))))
        return true;
      hit = start + 1;
    }
  }

  /* 3) mid-prose glue: "...following**Chapter 1: ...**" (no newline before **) */
  size_t glued = 0;
  while ((glued = reply.find("**Chapter", glued)) != std::string::npos) {
    if (glued > 0 && reply[glued - 1] != '\n' &&
        !std::isspace(static_cast<unsigned char>(reply[glued - 1]))) {
      size_t end = reply.find('\n', glued);
      if (end == std::string::npos) end = reply.size();
      if (matchesDone(extractHeadingCandidate(reply.substr(glued, end - glued))))
        return true;
    }
    glued += 1;
  }
  /* 4) mid-line markdown heading: "…. ## Autonomous …" */
  size_t h2 = 0;
  while ((h2 = reply.find("## ", h2)) != std::string::npos) {
    if (h2 > 0 && reply[h2 - 1] != '\n') {
      size_t end = reply.find('\n', h2);
      if (end == std::string::npos) end = reply.size();
      if (matchesDone(extractHeadingCandidate(reply.substr(h2, end - h2))))
        return true;
    }
    h2 += 1;
  }
  return false;
}

/** Unused on the write path. Kept for tests. Repeat handling is
    keepUniqueContinuation (exact text already in the draft). */
inline std::string prefixBeforeRestartedHeading(const std::string &reply,
                                               const std::string &existing) {
  if (reply.empty() || existing.empty()) return reply;
  if (!replyRestartsExistingHeading(reply, existing)) return reply;

  std::vector<std::string> doneKeys;
  for (const auto &ln : collectDeliverableSectionLines(existing)) {
    const std::string key = normalizeSectionKey(ln);
    if (!key.empty()) doneKeys.push_back(key);
  }
  auto isDone = [&](const std::string &raw) -> bool {
    const std::string t = trimCopy(raw);
    if (t.empty() || !isDeliverableSectionLine(t)) return false;
    const std::string key = normalizeSectionKey(t);
    for (const auto &dk : doneKeys) {
      if (sectionKeyMatches(key, dk)) return true;
    }
    return false;
  };

  auto cutMidLine = [&](const std::string &line) -> size_t {
    size_t best = std::string::npos;
    auto consider = [&](size_t at) {
      if (at == std::string::npos || at == 0) return;
      if (isDone(line.substr(at))) {
        if (best == std::string::npos || at < best) best = at;
      }
    };
    consider(line.find("## "));
    consider(line.find("# "));
    consider(line.find("**Chapter"));
    size_t bold = line.find("**");
    if (bold != std::string::npos && bold > 0) consider(bold);
    return best;
  };

  std::string out;
  size_t pos = 0;
  bool any = false;
  while (pos < reply.size()) {
    size_t eol = reply.find('\n', pos);
    if (eol == std::string::npos) eol = reply.size();
    const std::string line = reply.substr(pos, eol - pos);
    if (isDone(line)) {
      /* Skip a restated heading; keep new prose on both sides.
         Unique-cut drops paragraphs that are already in the draft. */
      if (eol >= reply.size()) break;
      pos = eol + 1;
      continue;
    }
    const size_t mid = cutMidLine(line);
    if (mid != std::string::npos) {
      if (any) out.push_back('\n');
      out += line.substr(0, mid);
      return trimCopy(out);
    }
    if (any) out.push_back('\n');
    out += line;
    any = true;
    if (eol >= reply.size()) break;
    pos = eol + 1;
  }
  return trimCopy(out);
}

/** Scale pressure from THIS goal's duration / requested word count. */
inline double inferPressureTauSec(const std::string &goal, double configuredTau) {
  double tau = configuredTau >= 1.0 ? configuredTau : 1.0;
  const auto facts = extractGoalOperationalFacts(goal);
  const int days = parseDurationDays(facts.duration);
  if (days >= 100) tau = std::max(tau, 14.0 * 86400.0);
  const int words = parseGoalWordCount(goal);
  if (words >= 20000) tau = std::max(tau, 7.0 * 86400.0);
  if (goal.size() > 1500) tau = std::max(tau, 3.0 * 86400.0);
  return tau;
}

/** Raise the draft floor from THIS goal's requested word count. */
inline int inferMinDeliverableChars(const std::string &goal, int configured) {
  int minChars = configured > 0 ? configured : 1200;
  const int words = parseGoalWordCount(goal);
  if (words >= 50000) minChars = std::max(minChars, 20000);
  else if (words >= 30000) minChars = std::max(minChars, 12000);
  else if (words >= 10000) minChars = std::max(minChars, 6000);
  if (goal.size() > 1500) minChars = std::max(minChars, 6000);
  return minChars;
}

/** When the prompt is over budget, keep the START (instructions / headings /
    early goal) and the END (later goal chapters / resume point).  Keeping
    only the tail made the model reread the last subsection and spin. */
inline std::string clipKeepHeadAndTail(const std::string &s, size_t budget) {
  if (s.size() <= budget) return s;
  if (budget < 80) return s.substr(s.size() - budget);
  const size_t headN = std::max<size_t>(32, budget / 3);
  const size_t tailN = budget - headN;
  return s.substr(0, headN) + "\n" + s.substr(s.size() - tailN);
}

/** Clip a long assignment without dropping its own chapter/appendix lines.
    This is the goal text, not an injected document outline. */
inline std::string clipGoalKeepRequiredStructure(const std::string &goal,
                                                 size_t budget) {
  if (budget == 0) return std::string();
  if (goal.size() <= budget) return goal;

  std::vector<std::string> lines;
  {
    std::istringstream in(goal);
    std::string line;
    while (std::getline(in, line)) lines.push_back(line);
  }

  std::vector<size_t> chIdx;
  for (size_t i = 0; i < lines.size(); ++i) {
    if (isGoalChapterHeadingLine(lines[i])) chIdx.push_back(i);
  }
  if (chIdx.size() < 3) return clipKeepHeadAndTail(goal, budget);

  std::ostringstream pre;
  for (size_t i = 0; i < chIdx.front(); ++i) pre << lines[i] << "\n";
  const std::string preText = pre.str();

  std::ostringstream mid;
  for (size_t k = 0; k < chIdx.size(); ++k) {
    const size_t start = chIdx[k];
    const size_t end =
        (k + 1 < chIdx.size()) ? chIdx[k + 1] : lines.size();
    for (size_t i = start; i < end; ++i) mid << lines[i] << "\n";
  }
  const std::string midText = mid.str();
  std::string assembled = preText + midText;
  if (assembled.size() <= budget) return assembled;

  std::ostringstream compact;
  const size_t preBudget = std::min(preText.size(), std::max<size_t>(budget / 4, 240));
  compact << clipKeepHeadAndTail(preText, preBudget);
  if (!compact.str().empty() && compact.str().back() != '\n') compact << "\n";
  for (size_t idx : chIdx) compact << lines[idx] << "\n";
  std::string compactText = compact.str();
  if (compactText.size() > budget)
    return clipKeepHeadAndTail(compactText, budget);
  const size_t leftover = budget - compactText.size();
  if (leftover > 120 && !midText.empty())
    compactText += clipKeepHeadAndTail(midText, leftover);
  return compactText;
}

inline std::pair<std::string, std::string>
fitMissionPromptSplit(const std::string &staticPart,
                      const std::string &dynamicPart, size_t maxChars) {
  if (maxChars == 0) return {"", ""};
  if (staticPart.size() + dynamicPart.size() <= maxChars)
    return {staticPart, dynamicPart};
  /* 4096 slots * ngram=2 holds the working brief AND the draft tail.
     Do not throw away the static prefix at maxChars/8. Keep all of it
     whenever any resume room remains. */
  std::string st = staticPart;
  if (st.size() >= maxChars) {
    const size_t dynKeep =
        std::min(dynamicPart.size(), std::max<size_t>(maxChars / 5, 200));
    const size_t stKeep = maxChars > dynKeep ? maxChars - dynKeep : maxChars / 2;
    return {clipKeepHeadAndTail(st, stKeep),
            clipKeepHeadAndTail(dynamicPart, dynKeep)};
  }
  const size_t dynBudget = maxChars - st.size();
  std::string dyn = dynamicPart;
  if (dyn.size() > dynBudget)
    dyn = dyn.substr(dyn.size() - dynBudget);
  return {st, dyn};
}

inline std::string fitMissionPromptParts(const std::string &staticPart,
                                         const std::string &dynamicPart,
                                         size_t maxChars) {
  const auto parts = fitMissionPromptSplit(staticPart, dynamicPart, maxChars);
  return parts.first + parts.second;
}

/** Pull "Chapter N:" headings from a long goal when present. */
inline std::string outlineFromGoalChapters(const std::string &goal) {
  std::vector<std::string> chapters;
  size_t pos = 0;
  while (pos < goal.size()) {
    const size_t hit = goal.find("Chapter ", pos);
    if (hit == std::string::npos) break;
    size_t lineEnd = goal.find('\n', hit);
    if (lineEnd == std::string::npos) lineEnd = goal.size();
    std::string line = trimCopy(goal.substr(hit, lineEnd - hit));
    if (!line.empty()) chapters.push_back(line);
    pos = lineEnd + 1;
    if (chapters.size() >= 24) break;
  }
  if (chapters.size() < 3) return std::string();
  std::ostringstream oss;
  oss << "[OUTLINE - from goal required chapters]\n";
  for (size_t i = 0; i < chapters.size(); ++i)
    oss << (i + 1) << ". " << chapters[i] << "\n";
  oss << "[OUTLINE END]";
  return oss.str();
}

}  // namespace mission
}  // namespace phoenix
