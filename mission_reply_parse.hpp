/* mission_reply_parse.hpp - parse mission deliberator LLM replies safely
   Copyright (C) 2026 079 Project */
#pragma once

#include <cctype>
#include <string>
#include <vector>

#include <sstream>

#include <nlohmann/json.hpp>

namespace phoenix {
namespace mission {

inline std::string trimCopy(const std::string &s) {
  const auto b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return std::string();
  const auto e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
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

/** Skip optional "JSON" / markdown fences before the first '{'. */
inline size_t firstJsonBrace(const std::string &s) {
  const std::string lowerPrefixes[] = {"json", "```json", "```"};
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
