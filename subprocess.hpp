/* subprocess.hpp - One-shot subprocess runner with timeout and capture.
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   079 Project is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with 079 Project.  If not, see <http://www.gnu.org/licenses/>.

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
