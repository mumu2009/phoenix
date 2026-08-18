/* mcp_client.hpp - Model Context Protocol (MCP) client.
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

   MCP compatibility layer: launches external MCP servers (JSON-RPC 2.0 over
   newline-delimited stdio, per the MCP specification) as subprocesses and
   exposes their tools so the planner can use the mainstream plugin market.

   Scope (honest): stdio transport only (the original and most portable MCP
   transport; SSE/streamable-HTTP are out of scope here); client role only.
   Tools are snapshotted after "initialize" and exposed as AGI actions with
   category "mcp" (addonType "mcp").

   Threading: one reader thread per client dispatches responses by id into a
   condition-variable map; request/response matching is strictly by JSON-RPC
   id, so interleaved notifications are ignored safely.
*/
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

namespace phoenix {
namespace mcp {

using json = nlohmann::json;

struct McpServerConfig {
  std::string name;
  std::string command;
  std::vector<std::string> args;
  std::map<std::string, std::string> env;
  int timeoutMs{10000};
  int startupMs{15000};
  bool enabled{true};

  json toJson() const;
  static McpServerConfig fromJson(const json &j);
};

struct McpTool {
  std::string name;
  std::string description;
  json inputSchema = json::object();
  json toJson() const;
};

/* ------------------------- protocol helpers ------------------------- */

/* JSON-RPC request line: {"jsonrpc":"2.0","id":N,"method":"...","params":...} */
std::string buildRequest(int64_t id, const std::string &method, const json &params);
/* JSON-RPC notification line (no id). */
std::string buildNotification(const std::string &method, const json &params);

struct ParsedMessage {
  bool valid{false};
  bool isResponse{false};   /* has "id" and ("result"|"error") */
  bool isRequest{false};    /* has "method" */
  bool hasId{false};
  int64_t id{0};
  std::string method;
  json result;
  json error;
  json params;
};

/* Parse one newline-delimited JSON message (stdio transport framing). */
ParsedMessage parseMessage(const std::string &line);

/* ------------------------------ client ------------------------------ */

class McpClient {
public:
  explicit McpClient(McpServerConfig cfg) : cfg_(std::move(cfg)) {}
  ~McpClient();

  McpClient(const McpClient &) = delete;
  McpClient &operator=(const McpClient &) = delete;

  /* Launch the server subprocess and perform the initialize handshake.
     Returns false and fills err on any failure. */
  bool start(std::string &err);

  bool running() const { return running_.load(std::memory_order_acquire); }
  const std::string &serverName() const { return cfg_.name; }

  /* Blocking JSON-RPC request.  Returns {"ok":true,"result":...} or
     {"ok":false,"error":"...", ...}. */
  json request(const std::string &method, const json &params, int timeoutMs);

  /* Fire-and-forget notification. */
  void notify(const std::string &method, const json &params);

  /* tools/list -> vector of tools (empty + lastError on failure). */
  std::vector<McpTool> listTools(std::string &error);

  /* tools/call -> {"ok":bool,"isError":bool,"content":[...], "error":...} */
  json callTool(const std::string &tool, const json &args, std::string &error);

  /* resources/read -> content object (or error). */
  json readResource(const std::string &uri, std::string &error);

  /* prompts/get -> prompt object (or error). */
  json getPrompt(const std::string &name, const json &args, std::string &error);

  /* Graceful shutdown: shutdown request + exit notification, join reader. */
  void shutdown();

  /* Test hook: run the client loop over an existing connected fd pair
     (fake in-process server).  ownsFds=false => fds are not closed. */
  bool startWithFds(int inFd, int outFd, bool ownsFds, std::string &err);

private:
  void readerLoop();
  bool writeLine(const std::string &line);
  bool writeLineRaw(const std::string &line);

  McpServerConfig cfg_;
  std::atomic<bool> running_{false};
  int inFd_{-1};   /* client reads from here  */
  int outFd_{-1};  /* client writes to here   */
  bool ownsFds_{true};
  void *procHandle_{nullptr};
  std::thread reader_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::map<int64_t, json> responses_;
  int64_t nextId_{1};
  std::string lastError_;
};

/* ------------------------------ manager ----------------------------- */

class McpManager {
public:
  void configure(const json &cfg, std::string &err);
  const std::vector<McpServerConfig> &servers() const { return servers_; }
  json startAll();  /* per-server status report */
  void stopAll();
  json status() const;

  /* Aggregate tool snapshot across all running servers:
     [{"server":name, "tool":{...}}, ...] */
  json listTools() const;

  /* Call a tool on a specific server. */
  json callTool(const std::string &server, const std::string &tool,
                const json &args) const;

private:
  mutable std::mutex mu_;
  std::vector<McpServerConfig> servers_;
  std::vector<std::unique_ptr<McpClient>> clients_;
  bool enabled_{false};
};

} /* namespace mcp */
} /* namespace phoenix */
