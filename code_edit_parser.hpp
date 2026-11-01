/* code_edit_parser.hpp - recognise a code edit in a model reply.
 *
 * The self-mod / peer-evolve trace only counts a tick as a source edit
 * when the reply carries FILE/OLD/NEW or a unified diff.  Small expert
 * models often answer with a fenced code block instead.  This parser is
 * the plugin-side normaliser: it turns that block into the textual patch
 * form the trace already understands.  It never writes files itself. */

#pragma once

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace phoenix {
namespace code_edit {

inline std::string trimCopy(const std::string &s) {
  size_t a = 0, b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}

inline bool hasPatchForm(const std::string &text) {
  if (text.find("diff --git ") != std::string::npos) return true;
  if (text.find("--- a/") != std::string::npos) return true;
  const auto f = text.find("FILE ");
  if (f == std::string::npos) return false;
  const auto o = text.find("\nOLD", f);
  const auto n = text.find("\nNEW", f);
  return o != std::string::npos && n != std::string::npos && o < n;
}

inline std::string findPathLine(const std::string &text) {
  size_t pos = 0;
  while (pos < text.size()) {
    size_t eol = text.find('\n', pos);
    if (eol == std::string::npos) eol = text.size();
    const std::string line = trimCopy(text.substr(pos, eol - pos));
    if (line.rfind("FILE ", 0) == 0)
      return trimCopy(line.substr(5));
    const auto p = line.find(" path=");
    if (p != std::string::npos) return trimCopy(line.substr(p + 6));
    pos = (eol == text.size()) ? text.size() : eol + 1;
  }
  return std::string();
}

/* Returns a FILE/OLD/NEW patch, or empty when the reply has no code.
 * `hintPath` is used when the reply does not name a file. */
inline std::string extractCodeEdit(const std::string &reply,
                                   const std::string &hintPath) {
  if (reply.empty()) return std::string();
  if (hasPatchForm(reply)) return reply;
  std::string path = findPathLine(reply);
  if (path.empty()) path = hintPath;
  if (path.empty() || path.find("..") != std::string::npos) return std::string();
  /* Fenced block first.  ```lang ... ``` */
  const auto fence = reply.find("```");
  if (fence != std::string::npos) {
    size_t body = reply.find('\n', fence);
    if (body == std::string::npos) return std::string();
    body += 1;
    const auto close = reply.find("```", body);
    if (close != std::string::npos && close > body) {
      const std::string code = reply.substr(body, close - body);
      if (trimCopy(code).size() >= 8)
        return "FILE " + path + "\nOLD\n\nNEW\n" + code + "\n";
    }
  }
  /* Raw code body: a C++ preprocessor line or a def/function. */
  const bool raw = reply.find("#include") != std::string::npos ||
                   reply.find("\ndef ") != std::string::npos ||
                   reply.find("\nvoid ") != std::string::npos ||
                   reply.find("\nint ") != std::string::npos;
  if (raw && reply.size() >= 8)
    return "FILE " + path + "\nOLD\n\nNEW\n" + reply + "\n";
  return std::string();
}

}  // namespace code_edit
}  // namespace phoenix
