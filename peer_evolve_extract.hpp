#pragma once

#include <cctype>
#include <string>

namespace phoenix {
namespace peer {

/* The compiler sees the last function they wrote. This does not insert
   tokens, rename calls, or close parentheses they left open. */

inline bool evolveWordByte(unsigned char c) {
  return std::isalnum(c) || c == '_';
}

inline void evolveReplaceAll(std::string &s, const std::string &from,
                             const std::string &to) {
  if (from.empty()) return;
  std::size_t pos = 0;
  while ((pos = s.find(from, pos)) != std::string::npos) {
    s.replace(pos, from.size(), to);
    pos += to.size();
  }
}

inline std::string evolveCloseFunction(std::string fn) {
  int depth = 0;
  bool seen = false;
  for (std::size_t i = 0; i < fn.size(); ++i) {
    if (fn[i] == '{') {
      ++depth;
      seen = true;
    } else if (fn[i] == '}') {
      --depth;
      if (seen && depth == 0) return fn.substr(0, i + 1) + "\n";
    }
  }
  if (seen && depth > 0) {
    while (depth-- > 0) fn += "\n}\n";
    return fn;
  }
  return {};
}

inline std::size_t evolveLastCall(const std::string &s, const std::string &word) {
  if (word.empty() || s.size() < word.size()) return std::string::npos;
  std::size_t searchFrom = s.size();
  while (searchFrom > 0) {
    const std::size_t pos = s.rfind(word, searchFrom - 1);
    if (pos == std::string::npos) return std::string::npos;
    const bool leftOk =
        pos == 0 || !evolveWordByte(static_cast<unsigned char>(s[pos - 1]));
    const std::size_t after = pos + word.size();
    const bool rightOk = after >= s.size() ||
                         !evolveWordByte(static_cast<unsigned char>(s[after]));
    if (leftOk && rightOk) {
      std::size_t k = after;
      while (k < s.size() && std::isspace(static_cast<unsigned char>(s[k]))) ++k;
      if (k < s.size() && s[k] == '(') return pos;
    }
    if (pos == 0) return std::string::npos;
    searchFrom = pos;
  }
  return std::string::npos;
}

inline bool evolveHasControl(const std::string &s) {
  return evolveLastCall(s, "for") != std::string::npos ||
         evolveLastCall(s, "while") != std::string::npos;
}

inline void evolveTrimToDeclaration(std::string &body) {
  const std::size_t forAt = evolveLastCall(body, "for");
  const std::size_t whileAt = evolveLastCall(body, "while");
  std::size_t use = std::string::npos;
  if (forAt != std::string::npos) use = forAt;
  if (whileAt != std::string::npos &&
      (use == std::string::npos || whileAt > use))
    use = whileAt;
  if (use == std::string::npos) return;
  const char *marks[] = {"uint32_t", "uint64_t", "unsigned", "int state",
                         "auto state"};
  std::size_t latest = std::string::npos;
  for (const char *m : marks) {
    const auto p = body.rfind(m, use);
    if (p != std::string::npos && (latest == std::string::npos || p > latest))
      latest = p;
  }
  std::size_t keep = 0;
  if (latest != std::string::npos) keep = latest;
  else if (use > 480) keep = use - 480;
  if (keep > 0) body = body.substr(keep);
}

inline std::string extractEvolve(const std::string &goal, std::string text) {
  evolveReplaceAll(text, "```cpp", "\n");
  evolveReplaceAll(text, "```c++", "\n");
  evolveReplaceAll(text, "```", "\n");
  const std::string marker = "std::string evolveCompute()";
  const auto inFile = text.rfind(marker);
  if (inFile != std::string::npos)
    return evolveCloseFunction(text.substr(inFile));
  const auto inGoal = goal.rfind(marker);
  if (inGoal == std::string::npos) return {};
  std::string body = text;
  /* A matrix routine is one function with several loops. Trimming to the
     last declaration would drop the arrays. */
  if (goal.find("ALGO-OPT") == std::string::npos)
    evolveTrimToDeclaration(body);
  if (body.find_first_not_of(" \t\r\n") == std::string::npos) return {};
  return evolveCloseFunction(goal.substr(inGoal) + "\n" + body);
}

}  // namespace peer
}  // namespace phoenix
