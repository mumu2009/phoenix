/* mcp_client.cpp - Implementation, see header for the design. */
#include "mcp_client.hpp"

#include <chrono>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#else
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace phoenix {
namespace mcp {

using json = nlohmann::json;

/* --------------------------- config --------------------------- */

json McpServerConfig::toJson() const {
  json envObj = json::object();
  for (const auto &kv : env) envObj[kv.first] = kv.second;
  return json{{"name", name}, {"command", command}, {"args", args},
              {"env", envObj}, {"timeoutMs", timeoutMs}, {"startupMs", startupMs},
              {"enabled", enabled}};
}

McpServerConfig McpServerConfig::fromJson(const json &j) {
  McpServerConfig c;
  if (!j.is_object()) return c;
  c.name = j.value("name", std::string());
  c.command = j.value("command", std::string());
  if (j.contains("args") && j["args"].is_array())
    for (const auto &a : j["args"]) if (a.is_string()) c.args.push_back(a.get<std::string>());
  if (j.contains("env") && j["env"].is_object())
    for (auto it = j["env"].begin(); it != j["env"].end(); ++it)
      if (it.value().is_string()) c.env[it.key()] = it.value().get<std::string>();
  c.timeoutMs = j.value("timeoutMs", 10000);
  c.startupMs = j.value("startupMs", 15000);
  c.enabled = j.value("enabled", true);
  return c;
}

json McpTool::toJson() const {
  return json{{"name", name}, {"description", description},
              {"inputSchema", inputSchema}};
}

/* ------------------------- protocol helpers ------------------------- */

std::string buildRequest(int64_t id, const std::string &method, const json &params) {
  json req{{"jsonrpc", "2.0"}, {"id", id}, {"method", method}, {"params", params}};
  std::string line = req.dump();
  line.push_back('\n');
  return line;
}

std::string buildNotification(const std::string &method, const json &params) {
  json req{{"jsonrpc", "2.0"}, {"method", method}, {"params", params}};
  std::string line = req.dump();
  line.push_back('\n');
  return line;
}

ParsedMessage parseMessage(const std::string &line) {
  ParsedMessage m;
  /* trim trailing \r/\n and whitespace */
  std::string s = line;
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r' ||
                        s.back() == ' ' || s.back() == '\t'))
    s.pop_back();
  if (s.empty()) return m;
  json j;
  try {
    j = json::parse(s);
  } catch (...) {
    return m;
  }
  if (!j.is_object()) return m;
  if (!j.contains("jsonrpc") || j["jsonrpc"] != "2.0") return m;
  m.valid = true;
  if (j.contains("id")) {
    m.hasId = true;
    if (j["id"].is_number_integer()) m.id = j["id"].get<int64_t>();
  }
  const bool hasResult = j.contains("result");
  const bool hasError = j.contains("error") && j["error"].is_object();
  if (m.hasId && (hasResult || hasError)) {
    m.isResponse = true;
    if (hasResult) m.result = j["result"];
    if (hasError) m.error = j["error"];
  } else if (j.contains("method") && j["method"].is_string()) {
    m.isRequest = true;
    m.method = j["method"].get<std::string>();
    if (j.contains("params")) m.params = j["params"];
  }
  return m;
}

/* ------------------------- subprocess spawn ------------------------- */

namespace {

struct SpawnResult {
  bool ok{false};
  int inFd{-1};   /* parent reads  */
  int outFd{-1};  /* parent writes */
  std::string error;
#ifdef _WIN32
  HANDLE process{nullptr};
  HANDLE thread{nullptr};
#else
  pid_t pid{-1};
#endif
};

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

#ifdef _WIN32
SpawnResult spawnSubprocess(const McpServerConfig &cfg) {
  SpawnResult r;
  HANDLE readPipe = nullptr, writePipe = nullptr;
  SECURITY_ATTRIBUTES sa{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
  if (!CreatePipe(&readPipe, &writePipe, &sa, 0)) { r.error = "CreatePipe read failed"; return r; }
  HANDLE readPipe2 = nullptr, writePipe2 = nullptr;
  if (!CreatePipe(&readPipe2, &writePipe2, &sa, 0)) {
    r.error = "CreatePipe write failed";
    CloseHandle(readPipe); CloseHandle(writePipe);
    return r;
  }
  /* child inherits only its ends */
  SetHandleInformation(writePipe, HANDLE_FLAG_INHERIT, 0);
  SetHandleInformation(readPipe2, HANDLE_FLAG_INHERIT, 0);

  std::string cmdline = quoteArg(cfg.command);
  for (const auto &a : cfg.args) cmdline += " " + quoteArg(a);

  STARTUPINFOA si{};
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdInput = readPipe;
  si.hStdOutput = writePipe2;
  si.hStdError = writePipe2;
  PROCESS_INFORMATION pi{};

  std::vector<char> cmd(cmdline.begin(), cmdline.end());
  cmd.push_back('\0');

  /* environment: inherit + overrides */
  std::string envBlock;
  if (!cfg.env.empty()) {
    for (const auto &kv : cfg.env) envBlock += kv.first + "=" + kv.second + "\0";
    envBlock += "\0";
  }
  void *envPtr = envBlock.empty() ? nullptr : (void *)envBlock.data();
  DWORD flags = CREATE_NO_WINDOW;
  if (envBlock.empty()) flags |= 0;

  if (!CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, TRUE, flags,
                      envPtr, nullptr, &si, &pi)) {
    r.error = "CreateProcessA failed (code " + std::to_string(GetLastError()) + ")";
    CloseHandle(readPipe); CloseHandle(writePipe);
    CloseHandle(readPipe2); CloseHandle(writePipe2);
    return r;
  }
  CloseHandle(readPipe);
  CloseHandle(writePipe2);
  r.ok = true;
  r.inFd = _open_osfhandle(reinterpret_cast<intptr_t>(readPipe2), 0);
  r.outFd = _open_osfhandle(reinterpret_cast<intptr_t>(writePipe), 0);
  r.process = pi.hProcess;
  CloseHandle(pi.hThread); /* not needed after spawn */
  return r;
}

#else
SpawnResult spawnSubprocess(const McpServerConfig &cfg) {
  SpawnResult r;
  int p1[2], p2[2];
  if (pipe(p1) != 0 || pipe(p2) != 0) { r.error = "pipe failed"; return r; }
  pid_t pid = fork();
  if (pid < 0) { r.error = "fork failed"; return r; }
  if (pid == 0) {
    /* child */
    dup2(p1[0], STDIN_FILENO);
    dup2(p2[1], STDOUT_FILENO);
    dup2(p2[1], STDERR_FILENO);
    close(p1[0]); close(p1[1]); close(p2[0]); close(p2[1]);
    std::vector<char *> argv;
    argv.push_back(const_cast<char *>(cfg.command.c_str()));
    for (const auto &a : cfg.args) argv.push_back(const_cast<char *>(a.c_str()));
    argv.push_back(nullptr);
    for (const auto &kv : cfg.env) setenv(kv.first.c_str(), kv.second.c_str(), 1);
    execvp(cfg.command.c_str(), argv.data());
    _exit(127);
  }
  close(p1[0]);
  close(p2[1]);
  r.ok = true;
  r.inFd = p2[0];
  r.outFd = p1[1];
  r.pid = pid;
  return r;
}

#endif

} /* namespace */

/* ------------------------------ client ------------------------------ */

McpClient::~McpClient() { shutdown(); }

bool McpClient::start(std::string &err) {
  if (running_.load(std::memory_order_acquire)) { err = "already running"; return false; }
  if (cfg_.command.empty()) { err = "empty server command"; return false; }
  SpawnResult r = spawnSubprocess(cfg_);
  if (!r.ok) { err = r.error; return false; }
  inFd_ = r.inFd;
  outFd_ = r.outFd;
  ownsFds_ = true;
#ifdef _WIN32
  procHandle_ = reinterpret_cast<void *>(r.process);
#else
  procPid_ = r.pid;
#endif
  return startWithFds(inFd_, outFd_, true, err);
}

bool McpClient::startWithFds(int inFd, int outFd, bool ownsFds, std::string &err) {
  inFd_ = inFd;
  outFd_ = outFd;
  ownsFds_ = ownsFds;
  running_.store(true, std::memory_order_release);
  reader_ = std::thread([this] { readerLoop(); });
  /* handshake */
  const json initParams = json{
      {"protocolVersion", "2024-11-05"},
      {"capabilities", json::object()},
      {"clientInfo", json{{"name", "phoenix"}, {"version", "7.0"}}}};
  const json resp = request("initialize", initParams, cfg_.startupMs);
  if (!resp.value("ok", false)) {
    err = "initialize failed: " + resp.value("error", std::string("?"));
    shutdown();
    return false;
  }
  notify("notifications/initialized", json::object());
  return true;
}

void McpClient::readerLoop() {
  /* line-buffered reader over a raw fd; no stdio FILE* to avoid buffering
     conflicts with the writer side. */
  std::string pending;
  char buf[8192];
  while (running_.load(std::memory_order_acquire)) {
#ifdef _WIN32
    int n = _read(inFd_, buf, sizeof(buf));
    if (n <= 0) break;
#else
    /* poll so shutdown() can interrupt the wait */
    struct pollfd pfd;
    pfd.fd = inFd_;
    pfd.events = POLLIN;
    const int pr = poll(&pfd, 1, 200);
    if (pr < 0) break;
    if (pr == 0) continue;
    ssize_t n = read(inFd_, buf, sizeof(buf));
    if (n <= 0) break;
#endif
    pending.append(buf, static_cast<size_t>(n));
    size_t pos = 0;
    while (true) {
      const size_t nl = pending.find('\n', pos);
      if (nl == std::string::npos) break;
      std::string line = pending.substr(pos, nl - pos);
      pos = nl + 1;
      const ParsedMessage m = parseMessage(line);
      if (m.valid && m.isResponse && m.hasId) {
        std::lock_guard<std::mutex> lock(mu_);
        responses_[m.id] = json{{"ok", true}, {"result", m.result}, {"error", m.error}};
        cv_.notify_all();
      }
    }
    if (pos > 0) pending.erase(0, pos);
    if (pending.size() > 16u * 1024u * 1024u) pending.clear(); /* safety */
  }
  running_.store(false, std::memory_order_release);
  cv_.notify_all();
}

bool McpClient::writeLineRaw(const std::string &line) {
  if (outFd_ < 0) return false;
  size_t sent = 0;
  while (sent < line.size()) {
#ifdef _WIN32
    int n = _write(outFd_, line.data() + sent, static_cast<unsigned>(line.size() - sent));
#else
    ssize_t n = write(outFd_, line.data() + sent, line.size() - sent);
#endif
    if (n <= 0) return false;
    sent += static_cast<size_t>(n);
  }
  return true;
}

bool McpClient::writeLine(const std::string &line) {
  if (!running_.load(std::memory_order_acquire)) return false;
  size_t sent = 0;
  while (sent < line.size()) {
#ifdef _WIN32
    int n = _write(outFd_, line.data() + sent, static_cast<unsigned>(line.size() - sent));
#else
    ssize_t n = write(outFd_, line.data() + sent, line.size() - sent);
#endif
    if (n <= 0) return false;
    sent += static_cast<size_t>(n);
  }
  return true;
}

json McpClient::request(const std::string &method, const json &params, int timeoutMs) {
  if (!running_.load(std::memory_order_acquire)) {
    return json{{"ok", false}, {"error", "client not running"}};
  }
  int64_t id = 0;
  {
    std::lock_guard<std::mutex> lock(mu_);
    id = nextId_++;
  }
  if (!writeLine(buildRequest(id, method, params))) {
    return json{{"ok", false}, {"error", "write failed"}};
  }
  std::unique_lock<std::mutex> lock(mu_);
  const bool got = cv_.wait_for(lock, std::chrono::milliseconds(timeoutMs), [&] {
    return responses_.count(id) > 0 || !running_.load(std::memory_order_acquire);
  });
  auto it = responses_.find(id);
  if (it == responses_.end()) {
    return json{{"ok", false}, {"error", "timeout or client stopped"}, {"method", method}};
  }
  json out = it->second;
  responses_.erase(it);
  return out;
}

void McpClient::notify(const std::string &method, const json &params) {
  if (running_.load(std::memory_order_acquire)) writeLine(buildNotification(method, params));
}

std::vector<McpTool> McpClient::listTools(std::string &error) {
  std::vector<McpTool> out;
  const json resp = request("tools/list", json::object(), cfg_.timeoutMs);
  if (!resp.value("ok", false) || !resp["error"].is_null()) {
    error = "tools/list failed: " +
            (resp.contains("error") && resp["error"].is_string()
                 ? resp["error"].get<std::string>()
                 : "server error");
    return out;
  }
  const json &result = resp["result"];
  if (result.contains("tools") && result["tools"].is_array()) {
    for (const auto &t : result["tools"]) {
      McpTool tool;
      tool.name = t.value("name", std::string());
      tool.description = t.value("description", std::string());
      if (t.contains("inputSchema")) tool.inputSchema = t["inputSchema"];
      if (!tool.name.empty()) out.push_back(tool);
    }
  }
  return out;
}

json McpClient::callTool(const std::string &tool, const json &args, std::string &error) {
  const json resp = request("tools/call", json{{"name", tool}, {"arguments", args}},
                            cfg_.timeoutMs);
  if (!resp.value("ok", false)) {
    error = "tools/call failed";
    return json{{"ok", false}, {"error", "transport failure"}};
  }
  if (!resp["error"].is_null()) {
    error = resp["error"].is_object() && resp["error"].contains("message")
                ? resp["error"]["message"].get<std::string>()
                : "server error";
    return json{{"ok", false}, {"error", error}};
  }
  const json &result = resp["result"];
  const bool isError = result.value("isError", false);
  json content = result.contains("content") ? result["content"] : json::array();
  return json{{"ok", !isError}, {"isError", isError}, {"content", content}};
}

json McpClient::readResource(const std::string &uri, std::string &error) {
  const json resp = request("resources/read", json{{"uri", uri}}, cfg_.timeoutMs);
  if (!resp.value("ok", false) || !resp["error"].is_null()) {
    error = "resources/read failed";
    return json{{"ok", false}, {"error", error}};
  }
  return json{{"ok", true}, {"result", resp["result"]}};
}

json McpClient::getPrompt(const std::string &name, const json &args, std::string &error) {
  const json resp = request("prompts/get", json{{"name", name}, {"arguments", args}},
                            cfg_.timeoutMs);
  if (!resp.value("ok", false) || !resp["error"].is_null()) {
    error = "prompts/get failed";
    return json{{"ok", false}, {"error", error}};
  }
  return json{{"ok", true}, {"result", resp["result"]}};
}

void McpClient::shutdown() {
  const bool wasRunning = running_.exchange(false, std::memory_order_acq_rel);
  if (wasRunning) {
    /* best-effort graceful: shutdown request + exit notification, then the
       reader unblocks when the child closes its stdout (or is terminated). */
    writeLineRaw(buildRequest(nextId_++, "shutdown", json::object()));
    writeLineRaw(buildNotification("exit", json::object()));
  }
  cv_.notify_all();
  /* unblock the reader: on Windows terminating the process closes the pipe;
     on POSIX SIGTERM makes the child exit. */
  if (ownsFds_) {
#ifdef _WIN32
    if (procHandle_) { TerminateProcess(procHandle_, 0); CloseHandle(procHandle_); procHandle_ = nullptr; }
    if (inFd_ >= 0) { _close(inFd_); inFd_ = -1; }
    if (outFd_ >= 0) { _close(outFd_); outFd_ = -1; }
#else
    /* SIGTERM the child first so a non-cooperative server cannot linger
       after the client is gone. */
    if (procPid_ > 0) {
      kill(procPid_, SIGTERM);
      waitpid(procPid_, nullptr, 0);
      procPid_ = -1;
    }
    if (inFd_ >= 0) { close(inFd_); inFd_ = -1; }
    if (outFd_ >= 0) { close(outFd_); outFd_ = -1; }
#endif
  }
  if (reader_.joinable()) reader_.join();
}

/* ------------------------------ manager ----------------------------- */

void McpManager::configure(const json &cfg, std::string &err) {
  std::lock_guard<std::mutex> lock(mu_);
  servers_.clear();
  if (!cfg.is_object()) { err = "mcp config must be an object"; return; }
  enabled_ = cfg.value("enabled", false);
  if (cfg.contains("servers") && cfg["servers"].is_array()) {
    for (const auto &s : cfg["servers"]) {
      McpServerConfig c = McpServerConfig::fromJson(s);
      if (!c.name.empty() && !c.command.empty()) servers_.push_back(c);
    }
  }
}

json McpManager::startAll() {
  std::lock_guard<std::mutex> lock(mu_);
  json report = json::array();
  if (!enabled_) return json{{"ok", true}, {"enabled", false}, {"servers", report}};
  for (const auto &cfg : servers_) {
    if (!cfg.enabled) { report.push_back(json{{"server", cfg.name}, {"ok", false}, {"error", "disabled"}}); continue; }
    auto client = std::make_unique<McpClient>(cfg);
    std::string err;
    if (client->start(err)) {
      report.push_back(json{{"server", cfg.name}, {"ok", true}, {"running", true}});
      clients_.push_back(std::move(client));
    } else {
      report.push_back(json{{"server", cfg.name}, {"ok", false}, {"error", err}});
    }
  }
  return json{{"ok", true}, {"enabled", true}, {"servers", report}};
}

void McpManager::stopAll() {
  std::lock_guard<std::mutex> lock(mu_);
  for (auto &c : clients_) c->shutdown();
  clients_.clear();
}

json McpManager::status() const {
  std::lock_guard<std::mutex> lock(mu_);
  json arr = json::array();
  for (const auto &c : clients_) arr.push_back(json{{"server", c->serverName()}, {"running", c->running()}});
  return json{{"enabled", enabled_}, {"configured", servers_.size()}, {"clients", arr}};
}

json McpManager::listTools() const {
  std::lock_guard<std::mutex> lock(mu_);
  json out = json::array();
  for (const auto &c : clients_) {
    std::string err;
    auto tools = c->listTools(err);
    for (const auto &t : tools)
      out.push_back(json{{"server", c->serverName()}, {"tool", t.toJson()}});
  }
  return out;
}

json McpManager::callTool(const std::string &server, const std::string &tool,
                          const json &args) const {
  std::lock_guard<std::mutex> lock(mu_);
  for (const auto &c : clients_) {
    if (c->serverName() != server) continue;
    std::string err;
    json r = c->callTool(tool, args, err);
    if (!err.empty() && !r.contains("error")) r["error"] = err;
    return r;
  }
  return json{{"ok", false}, {"error", "server not found: " + server}};
}

} /* namespace mcp */
} /* namespace phoenix */
