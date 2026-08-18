/* subprocess.cpp - Implementation, see header. */
#include "subprocess.hpp"

#include <chrono>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#else
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace phoenix {
namespace subprocess {

namespace {

std::string quoteArg(const std::string &a) {
  if (a.empty()) return "\"\"";
  bool need = a.find_first_of(" \t\"") != std::string::npos;
  if (!need) return a;
  std::string out = "\"";
  for (char c : a) {
    if (c == '"' || c == '\\') out.push_back('\\');
    out.push_back(c);
  }
  out.push_back('"');
  return out;
}

std::string capString(std::string s, size_t maxBytes) {
  if (s.size() > maxBytes) s.resize(maxBytes);
  return s;
}

#ifdef _WIN32
RunResult runWindows(const RunRequest &req) {
  RunResult r;
  HANDLE childOutRead = nullptr, childOutWrite = nullptr;
  HANDLE childErrRead = nullptr, childErrWrite = nullptr;
  HANDLE childInRead = nullptr, childInWrite = nullptr;
  SECURITY_ATTRIBUTES sa{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
  if (!CreatePipe(&childOutRead, &childOutWrite, &sa, 0) ||
      !CreatePipe(&childErrRead, &childErrWrite, &sa, 0) ||
      !CreatePipe(&childInRead, &childInWrite, &sa, 0)) {
    r.error = "CreatePipe failed";
    return r;
  }
  SetHandleInformation(childOutRead, HANDLE_FLAG_INHERIT, 0);
  SetHandleInformation(childErrRead, HANDLE_FLAG_INHERIT, 0);
  SetHandleInformation(childInWrite, HANDLE_FLAG_INHERIT, 0);

  std::string cmdline = quoteArg(req.command);
  for (const auto &a : req.args) cmdline += " " + quoteArg(a);

  STARTUPINFOA si{};
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdInput = childInRead;
  si.hStdOutput = childOutWrite;
  si.hStdError = childErrWrite;
  PROCESS_INFORMATION pi{};
  std::vector<char> cmd(cmdline.begin(), cmdline.end());
  cmd.push_back('\0');
  if (!CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW,
                      nullptr, nullptr, &si, &pi)) {
    r.error = "CreateProcessA failed (" + std::to_string(GetLastError()) + ")";
    CloseHandle(childOutRead); CloseHandle(childOutWrite);
    CloseHandle(childErrRead); CloseHandle(childErrWrite);
    CloseHandle(childInRead); CloseHandle(childInWrite);
    return r;
  }
  CloseHandle(childOutWrite);
  CloseHandle(childErrWrite);
  CloseHandle(childInRead);
  r.started = true;

  /* write stdin */
  if (!req.stdinText.empty()) {
    size_t off = 0;
    while (off < req.stdinText.size()) {
      DWORD n = 0;
      if (!WriteFile(childInWrite, req.stdinText.data() + off,
                     static_cast<DWORD>(req.stdinText.size() - off), &n, nullptr) || n == 0)
        break;
      off += n;
    }
  }
  CloseHandle(childInWrite);

  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(req.timeoutMs);
  auto readAll = [&](HANDLE h, std::string &into) {
    char buf[8192];
    DWORD n = 0;
    while (PeekNamedPipe(h, nullptr, 0, nullptr, &n, nullptr) && n > 0) {
      DWORD got = 0;
      if (!ReadFile(h, buf, sizeof(buf), &got, nullptr) || got == 0) break;
      into.append(buf, got);
      if (into.size() > req.maxOutputBytes) { into.resize(req.maxOutputBytes); return; }
    }
  };

  while (std::chrono::steady_clock::now() < deadline) {
    DWORD code = 0;
    if (GetExitCodeProcess(pi.hProcess, &code) && code != STILL_ACTIVE) {
      r.exitCode = static_cast<int>(code);
      break;
    }
    readAll(childOutRead, r.stdoutText);
    readAll(childErrRead, r.stderrText);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  if (r.exitCode < 0) {
    /* timed out */
    r.timedOut = true;
    TerminateProcess(pi.hProcess, 1);
    r.exitCode = -1;
    r.error = "command timed out after " + std::to_string(req.timeoutMs) + "ms";
  }
  /* drain whatever remains (must also run on the success path) */
  for (int pass = 0; pass < 200 && !r.timedOut; ++pass) {
    readAll(childOutRead, r.stdoutText);
    readAll(childErrRead, r.stderrText);
    DWORD avail = 0;
    if (!PeekNamedPipe(childOutRead, nullptr, 0, nullptr, &avail, nullptr) || avail == 0)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  readAll(childOutRead, r.stdoutText);
  readAll(childErrRead, r.stderrText);
  CloseHandle(childOutRead);
  CloseHandle(childErrRead);
  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);
  r.stdoutText = capString(std::move(r.stdoutText), req.maxOutputBytes);
  r.stderrText = capString(std::move(r.stderrText), req.maxOutputBytes);
  return r;
}
#else
RunResult runPosix(const RunRequest &req) {
  RunResult r;
  int outP[2], errP[2], inP[2];
  if (pipe(outP) != 0 || pipe(errP) != 0 || pipe(inP) != 0) {
    r.error = "pipe failed";
    return r;
  }
  pid_t pid = fork();
  if (pid < 0) { r.error = "fork failed"; return r; }
  if (pid == 0) {
    dup2(inP[0], STDIN_FILENO);
    dup2(outP[1], STDOUT_FILENO);
    dup2(errP[1], STDERR_FILENO);
    close(inP[0]); close(inP[1]); close(outP[0]); close(outP[1]); close(errP[0]); close(errP[1]);
    std::vector<char *> argv;
    argv.push_back(const_cast<char *>(req.command.c_str()));
    for (const auto &a : req.args) argv.push_back(const_cast<char *>(a.c_str()));
    argv.push_back(nullptr);
    execvp(req.command.c_str(), argv.data());
    _exit(127);
  }
  close(inP[0]); close(outP[1]); close(errP[1]);
  r.started = true;
  if (!req.stdinText.empty()) {
    size_t off = 0;
    while (off < req.stdinText.size()) {
      ssize_t n = write(inP[1], req.stdinText.data() + off, req.stdinText.size() - off);
      if (n <= 0) break;
      off += static_cast<size_t>(n);
    }
  }
  close(inP[1]);
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(req.timeoutMs);
  bool done = false;
  while (std::chrono::steady_clock::now() < deadline) {
    int status = 0;
    pid_t w = waitpid(pid, &status, WNOHANG);
    if (w == pid) {
      r.exitCode = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
      done = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  if (!done) {
    r.timedOut = true;
    kill(pid, SIGKILL);
    waitpid(pid, nullptr, 0);
    r.error = "command timed out after " + std::to_string(req.timeoutMs) + "ms";
  }
  /* drain pipes */
  char buf[8192];
  for (;;) {
    ssize_t n = read(outP[0], buf, sizeof(buf));
    if (n <= 0) break;
    r.stdoutText.append(buf, static_cast<size_t>(n));
    if (r.stdoutText.size() > req.maxOutputBytes) { r.stdoutText.resize(req.maxOutputBytes); break; }
  }
  for (;;) {
    ssize_t n = read(errP[0], buf, sizeof(buf));
    if (n <= 0) break;
    r.stderrText.append(buf, static_cast<size_t>(n));
    if (r.stderrText.size() > req.maxOutputBytes) { r.stderrText.resize(req.maxOutputBytes); break; }
  }
  close(outP[0]); close(errP[0]);
  return r;
}
#endif

} /* namespace */

RunResult run(const RunRequest &req) {
  if (req.command.empty()) {
    RunResult r;
    r.error = "empty command";
    return r;
  }
#ifdef _WIN32
  return runWindows(req);
#else
  return runPosix(req);
#endif
}

} /* namespace subprocess */
} /* namespace phoenix */
