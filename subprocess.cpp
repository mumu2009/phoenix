/* subprocess.cpp - Implementation, see header. */
#include "subprocess.hpp"

#include <cerrno>
#include <chrono>
#include <thread>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace phoenix {
namespace subprocess {

namespace {

#ifdef _WIN32
/* Child environment without the embedded-interpreter variables: a spawned
   python must find its own home (its own stdlib/numpy), not the parent's
   PYTHONHOME / AI_PYTHON_HOME / PYTHONPATH used by the embedded CPython. */
std::string envBlockForChild() {
  std::string out;
  char *env = GetEnvironmentStringsA();
  if (env) {
    for (char *p = env; *p;) {
      std::string e(p);
      const bool drop = e.rfind("PYTHONHOME=", 0) == 0 ||
                        e.rfind("AI_PYTHON_HOME=", 0) == 0 ||
                        e.rfind("PYTHONPATH=", 0) == 0;
      if (!drop) {
        out += e;
        out.push_back('\0');
      }
      p += e.size() + 1;
    }
    FreeEnvironmentStringsA(env);
  }
  out.push_back('\0');
  return out;
}
#endif

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
#ifdef _WIN32
  std::string childEnv = envBlockForChild();
#endif
  if (!CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW,
#ifdef _WIN32
                      childEnv.empty() ? nullptr : childEnv.data(),
#else
                      nullptr,
#endif
                      nullptr, &si, &pi)) {
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

namespace detail {

std::string detachedCommand(const std::string &command,
                            const std::vector<std::string> &args) {
  auto quote = [](const std::string &a) {
    if (a.empty())
      return std::string("\"\"");
    if (a.find_first_of(" \t\"") == std::string::npos)
      return a;
    std::string out = "\"";
    for (char c : a) {
      if (c == '"' || c == '\\')
        out.push_back('\\');
      out.push_back(c);
    }
    out.push_back('"');
    return out;
  };
  std::string cmdline = quote(command);
  for (const auto &a : args)
    cmdline += " " + quote(a);
  return cmdline;
}

#ifdef _WIN32
struct SocketHold {
  SOCKET fd{INVALID_SOCKET};
  SocketHold() = default;
  explicit SocketHold(SOCKET s) : fd(s) {}
  SocketHold(SocketHold &&other) noexcept : fd(other.fd) {
    other.fd = INVALID_SOCKET;
  }
  ~SocketHold() {
    if (fd != INVALID_SOCKET)
      closesocket(fd);
  }
  SocketHold(const SocketHold &) = delete;
  SocketHold &operator=(const SocketHold &) = delete;
};

bool startupNet() {
  static bool once = false;
  static bool ok = false;
  if (!once) {
    WSADATA data{};
    ok = WSAStartup(MAKEWORD(2, 2), &data) == 0;
    once = true;
  }
  return ok;
}

SocketHold connectTcp(const std::string &host, int port, int timeoutMs) {
  SocketHold hold;
  if (!startupNet() || timeoutMs <= 0)
    return hold;
  SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (s == INVALID_SOCKET)
    return hold;
  u_long nb = 1;
  ioctlsocket(s, FIONBIO, &nb);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<u_short>(port));
  if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
    closesocket(s);
    return hold;
  }
  int connected = connect(s, reinterpret_cast<sockaddr *>(&addr), sizeof(addr));
  if (connected != 0 && WSAGetLastError() != WSAEWOULDBLOCK) {
    closesocket(s);
    return hold;
  }
  fd_set writable;
  FD_ZERO(&writable);
  FD_SET(s, &writable);
  timeval tv{};
  tv.tv_sec = timeoutMs / 1000;
  tv.tv_usec = (timeoutMs % 1000) * 1000;
  if (select(0, nullptr, &writable, nullptr, &tv) <= 0) {
    closesocket(s);
    return hold;
  }
  int soerr = 0;
  int slen = sizeof(soerr);
  if (getsockopt(s, SOL_SOCKET, SO_ERROR, reinterpret_cast<char *>(&soerr), &slen) != 0 ||
      soerr != 0) {
    closesocket(s);
    return hold;
  }
  hold.fd = s;
  return hold;
}
#else
struct SocketHold {
  int fd{-1};
  SocketHold() = default;
  explicit SocketHold(int s) : fd(s) {}
  SocketHold(SocketHold &&other) noexcept : fd(other.fd) { other.fd = -1; }
  ~SocketHold() {
    if (fd >= 0)
      close(fd);
  }
  SocketHold(const SocketHold &) = delete;
  SocketHold &operator=(const SocketHold &) = delete;
};

SocketHold connectTcp(const std::string &host, int port, int timeoutMs) {
  SocketHold hold;
  if (timeoutMs <= 0)
    return hold;
  int s = socket(AF_INET, SOCK_STREAM, 0);
  if (s < 0)
    return hold;
  int flags = fcntl(s, F_GETFL, 0);
  fcntl(s, F_SETFL, flags | O_NONBLOCK);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
    close(s);
    return hold;
  }
  int connected = connect(s, reinterpret_cast<sockaddr *>(&addr), sizeof(addr));
  if (connected != 0 && errno != EINPROGRESS) {
    close(s);
    return hold;
  }
  fd_set writable;
  FD_ZERO(&writable);
  FD_SET(s, &writable);
  timeval tv{};
  tv.tv_sec = timeoutMs / 1000;
  tv.tv_usec = (timeoutMs % 1000) * 1000;
  if (select(s + 1, nullptr, &writable, nullptr, &tv) <= 0) {
    close(s);
    return hold;
  }
  int soerr = 0;
  socklen_t slen = sizeof(soerr);
  if (getsockopt(s, SOL_SOCKET, SO_ERROR, &soerr, &slen) != 0 || soerr != 0) {
    close(s);
    return hold;
  }
  hold.fd = s;
  return hold;
}
#endif

}  // namespace detail

bool spawnDetached(const std::string &command, const std::vector<std::string> &args) {
  if (command.empty())
    return false;
#ifdef _WIN32
  std::string cmdline = detail::detachedCommand(command, args);
  std::vector<char> cmd(cmdline.begin(), cmdline.end());
  cmd.push_back('\0');
  SECURITY_ATTRIBUTES sa{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
  HANDLE log = CreateFileA("runtime_store\\cond_resident.log", GENERIC_WRITE,
                           FILE_SHARE_READ, &sa, CREATE_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
  HANDLE nul = CreateFileA("NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                           &sa, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  STARTUPINFOA si{};
  si.cb = sizeof(si);
  if (log != INVALID_HANDLE_VALUE && nul != INVALID_HANDLE_VALUE) {
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = nul;
    si.hStdOutput = log;
    si.hStdError = log;
  }
  PROCESS_INFORMATION pi{};
#ifdef _WIN32
  std::string childEnv = envBlockForChild();
#endif
  const BOOL ok = CreateProcessA(
      nullptr, cmd.data(), nullptr, nullptr,
      si.dwFlags == STARTF_USESTDHANDLES ? TRUE : FALSE,
      CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
#ifdef _WIN32
      childEnv.empty() ? nullptr : childEnv.data(),
#else
      nullptr,
#endif
      nullptr, &si, &pi);
  if (log != INVALID_HANDLE_VALUE)
    CloseHandle(log);
  if (nul != INVALID_HANDLE_VALUE)
    CloseHandle(nul);
  if (!ok)
    return false;
  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);
  return true;
#else
  pid_t pid = fork();
  if (pid < 0)
    return false;
  if (pid == 0) {
    setsid();
    int log = open("runtime_store/cond_resident.log",
                   O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (log >= 0) {
      dup2(log, STDOUT_FILENO);
      dup2(log, STDERR_FILENO);
      if (log > 2)
        close(log);
    }
    int nul = open("/dev/null", O_RDONLY);
    if (nul >= 0) {
      dup2(nul, STDIN_FILENO);
      if (nul > 2)
        close(nul);
    }
    std::vector<char *> argv;
    argv.push_back(const_cast<char *>(command.c_str()));
    for (const auto &a : args)
      argv.push_back(const_cast<char *>(a.c_str()));
    argv.push_back(nullptr);
    execvp(command.c_str(), argv.data());
    _exit(127);
  }
  return true;
#endif
}

bool tcpReachable(const std::string &host, int port, int timeoutMs) {
  detail::SocketHold hold = detail::connectTcp(host, port, timeoutMs);
#ifdef _WIN32
  return hold.fd != INVALID_SOCKET;
#else
  return hold.fd >= 0;
#endif
}

std::string tcpExchange(const std::string &host, int port, const std::string &line,
                        int timeoutMs) {
  if (line.empty())
    return std::string();
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  detail::SocketHold hold = detail::connectTcp(host, port, timeoutMs);
#ifdef _WIN32
  if (hold.fd == INVALID_SOCKET)
    return std::string();
  const auto sock = hold.fd;
#else
  if (hold.fd < 0)
    return std::string();
  const auto sock = hold.fd;
#endif
  size_t off = 0;
  while (off < line.size()) {
    if (std::chrono::steady_clock::now() >= deadline)
      return std::string();
#ifdef _WIN32
    int n = send(sock, line.data() + off, static_cast<int>(line.size() - off), 0);
    if (n > 0) {
      off += static_cast<size_t>(n);
      continue;
    }
    if (n < 0 && WSAGetLastError() == WSAEWOULDBLOCK) {
#else
    ssize_t n = send(sock, line.data() + off, line.size() - off, 0);
    if (n > 0) {
      off += static_cast<size_t>(n);
      continue;
    }
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
#endif
      fd_set writable;
      FD_ZERO(&writable);
      FD_SET(sock, &writable);
      timeval tv{};
      tv.tv_usec = 200000;
#ifdef _WIN32
      select(0, nullptr, &writable, nullptr, &tv);
#else
      select(sock + 1, nullptr, &writable, nullptr, &tv);
#endif
      continue;
    }
    return std::string();
  }
  std::string got;
  while (got.find('\n') == std::string::npos) {
    if (std::chrono::steady_clock::now() >= deadline)
      return std::string();
    char buf[4096];
#ifdef _WIN32
    int n = recv(sock, buf, sizeof(buf), 0);
    if (n > 0) {
      got.append(buf, static_cast<size_t>(n));
      if (got.size() > 1000000)
        break;
      continue;
    }
    if (n < 0 && WSAGetLastError() == WSAEWOULDBLOCK) {
#else
    ssize_t n = recv(sock, buf, sizeof(buf), 0);
    if (n > 0) {
      got.append(buf, static_cast<size_t>(n));
      if (got.size() > 1000000)
        break;
      continue;
    }
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
#endif
      fd_set readable;
      FD_ZERO(&readable);
      FD_SET(sock, &readable);
      timeval tv{};
      tv.tv_usec = 200000;
#ifdef _WIN32
      select(0, &readable, nullptr, nullptr, &tv);
#else
      select(sock + 1, &readable, nullptr, nullptr, &tv);
#endif
      continue;
    }
    break;
  }
  return got;
}

} /* namespace subprocess */
} /* namespace phoenix */

