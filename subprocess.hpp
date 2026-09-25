/* subprocess.hpp - One-shot subprocess runner with timeout and capture.

   Used by the cli-json addon to run whitelisted command templates (the
   "any software becomes a plugin" bridge).  Direct exec, no shell, so
   arguments never go through an interpreter - the config whitelist is the
   trust boundary, not string quoting.
*/
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace phoenix {
namespace subprocess {

struct RunResult {
  bool started{false};
  bool timedOut{false};
  int exitCode{-1};
  std::string stdoutText;
  std::string stderrText;

  std::string error;
};

struct RunRequest {
  std::string command;
  std::vector<std::string> args;
  std::string stdinText;
  int timeoutMs{5000};
  size_t maxOutputBytes{1024 * 1024};
};

/* Run one command to completion (or timeout).  Direct exec (no shell). */
RunResult run(const RunRequest &req);

} /* namespace subprocess */
} /* namespace phoenix */
