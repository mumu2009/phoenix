#pragma once

#include "runtime_opt/shm_channel.hpp"
#include "runtime_opt/will_gate.hpp"

#include <string>
#include <string_view>

namespace phoenix {
namespace runtime_opt {

struct ExoMemConfig {
  bool enabled = true;
  bool sandboxHelpers = true;
};

struct ShmFrame {
  std::string kind;
  std::string scope;
  std::string payload;
};

inline std::string encodeFrame(std::string_view kind, std::string_view scope,
                               std::string_view payload) {
  return std::string(kind) + "\t" + std::string(scope) + "\t" +
         std::string(payload);
}

inline bool decodeFrame(std::string_view raw, ShmFrame &out) {
  const auto a = raw.find('\t');
  if (a == std::string_view::npos)
    return false;
  const auto b = raw.find('\t', a + 1);
  if (b == std::string_view::npos)
    return false;
  out.kind = std::string(raw.substr(0, a));
  out.scope = std::string(raw.substr(a + 1, b - a - 1));
  out.payload = std::string(raw.substr(b + 1));
  return !out.kind.empty();
}

inline bool shmWriteFrame(ShmChannel &ch, std::string_view kind,
                          std::string_view scope, std::string_view payload) {
  return ch.write(encodeFrame(kind, scope, payload));
}

inline bool shmReadFrame(const ShmChannel &ch, ShmFrame &out) {
  std::string raw;
  if (!ch.read(raw))
    return false;
  return decodeFrame(raw, out);
}

inline WillDecision sandboxHelperCmd(std::string_view cmd) {
  return willDecide(cmd, cmd);
}

} // namespace runtime_opt
} // namespace phoenix
