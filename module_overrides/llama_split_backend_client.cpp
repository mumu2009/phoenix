/* llama_split_backend_client.cpp - Phoenix client for the patched
   llama-server's split backend.  See llama_split_backend_client.hpp for
   the public API.

   Single-instance path (every stage except infer is optional):
     I/O enc (paragraph in, unit-query sequence out) -> preprocess -> gnn
     -> infer -> I/O dec (sequence in, paragraph text out)
   Two infer streams: causal prefix+resume (raw text); memory/GNN as RAG mix.
   Internal pair: infer -> dec -> enc -> infer.
 */

#include "llama_split_backend_client.hpp"

#include "inference_abort.hpp"
#include "inference_unit_pipeline.hpp"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
typedef int sockopt_len_t;
using ioctl_arg_t = unsigned long;
#else
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <unistd.h>
typedef int SOCKET;
#define INVALID_SOCKET (-1)
#define SOCKET_ERROR (-1)
#define WSADATA int
#define MAKEWORD(a, b) (0)
#define WSAStartup(a, b) (void)0
#define WSAGetLastError() (errno)
#define WSAEWOULDBLOCK EWOULDBLOCK
#define closesocket close
#define ioctlsocket ioctl
using ioctl_arg_t = int;
typedef socklen_t sockopt_len_t;
#endif

namespace phoenix {
namespace v7 {

using nlohmann::json;

namespace {

struct HttpResult {
  bool connectFailed = false;
  int status = 0;
  std::string body;
  std::string error;
};

void splitHostPort(const std::string &baseUrl, std::string &host, int &port) {
  host = "127.0.0.1";
  port = 8082;
  std::string ep = baseUrl;
  if (ep.substr(0, 7) == "http://") ep = ep.substr(7);
  else if (ep.substr(0, 8) == "https://") ep = ep.substr(8);
  auto slash = ep.find('/');
  if (slash != std::string::npos) ep = ep.substr(0, slash);
  auto col = ep.find(':');
  if (col != std::string::npos) {
    host = ep.substr(0, col);
    try {
      port = std::stoi(ep.substr(col + 1));
    } catch (...) {
      port = 80;
    }
  } else {
    host = ep;
    port = 80;
  }
}

// Raw Winsock HTTP/1.1 request (GET or POST) with a connect() timeout,
// mirroring the approach used by chatWithExternalAdapter() in
// main_hub_parts/112_section_before_contexthint.inc.
HttpResult httpRequest(const std::string &host, int port,
                        const std::string &method, const std::string &path,
                        const std::string &body, int timeoutMs) {
  HttpResult result;

  std::ostringstream req;
  req << method << " " << path << " HTTP/1.1\r\n"
      << "Host: " << host << ":" << port << "\r\n";
  if (!body.empty() || method == "POST") {
    req << "Content-Type: application/json\r\n"
        << "Content-Length: " << body.size() << "\r\n";
  }
  req << "Connection: close\r\n\r\n" << body;
  std::string reqStr = req.str();

  std::string rawResponse;
  {
    const uint64_t epoch = phoenix::inference::currentAbortEpoch();
    auto wouldBlock = []() {
#ifdef _WIN32
      return WSAGetLastError() == WSAEWOULDBLOCK;
#else
      return errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR;
#endif
    };
    auto sliceSelect = [&](SOCKET s, bool wantRead, bool wantWrite,
                           int sliceMs) -> int {
      fd_set rfds, wfds;
      FD_ZERO(&rfds);
      FD_ZERO(&wfds);
      if (wantRead)
        FD_SET(s, &rfds);
      if (wantWrite)
        FD_SET(s, &wfds);
      struct timeval tv;
      tv.tv_sec = sliceMs / 1000;
      tv.tv_usec = (sliceMs % 1000) * 1000;
      return select((int)s + 1, wantRead ? &rfds : nullptr,
                    wantWrite ? &wfds : nullptr, nullptr, &tv);
    };

    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
    SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCKET) {
      result.connectFailed = true;
      result.error = "socket create failed";
      return result;
    }
    int timeoutVal = std::max(1000, timeoutMs);

    struct sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<unsigned short>(port));
    inet_pton(AF_INET, host.c_str(), &addr.sin_addr);

    ioctl_arg_t nonblk = 1;
    ioctlsocket(sock, FIONBIO, &nonblk);
    int cr = connect(sock, (struct sockaddr *)&addr, sizeof(addr));
    bool connected = false;
    // On Windows non-blocking connect returns WSAEWOULDBLOCK; on Linux it
    // returns EINPROGRESS.  Both mean "connection in progress, use select()".
    if (cr == 0) {
      connected = true;
    } else if (cr == SOCKET_ERROR &&
               (WSAGetLastError() == WSAEWOULDBLOCK ||
                WSAGetLastError() == EINPROGRESS)) {
      int waited = 0;
      while (waited < timeoutVal) {
        if (phoenix::inference::shouldAbort(epoch)) {
          closesocket(sock);
          result.error = "aborted";
          return result;
        }
        int sel = sliceSelect(sock, false, true, 500);
        if (sel == 1) {
          int err = 0;
          sockopt_len_t errlen = sizeof(err);
          getsockopt(sock, SOL_SOCKET, SO_ERROR, (char *)&err, &errlen);
          connected = (err == 0);
          break;
        }
        if (sel < 0)
          break;
        waited += 500;
      }
    }
    if (!connected) {
      closesocket(sock);
      result.connectFailed = true;
      result.error = "connect failed to " + host + ":" + std::to_string(port) +
                      " (err=" + std::to_string(WSAGetLastError()) + ")";
      return result;
    }

    int sent = 0;
    int sendWaited = 0;
    while (sent < (int)reqStr.size() && sendWaited < timeoutVal) {
      if (phoenix::inference::shouldAbort(epoch)) {
        closesocket(sock);
        result.error = "aborted";
        return result;
      }
      int n = send(sock, reqStr.c_str() + sent, (int)reqStr.size() - sent, 0);
      if (n > 0) {
        sent += n;
        continue;
      }
      if (n < 0 && wouldBlock()) {
        if (sliceSelect(sock, false, true, 500) < 0)
          break;
        sendWaited += 500;
        continue;
      }
      break;
    }
    char buf[8192];
    int recvWaited = 0;
    while (recvWaited < timeoutVal) {
      if (phoenix::inference::shouldAbort(epoch)) {
        result.error = "aborted";
        break;
      }
      int n = recv(sock, buf, sizeof(buf), 0);
      if (n > 0) {
        rawResponse.append(buf, n);
        continue;
      }
      if (n == 0)
        break;
      if (n < 0 && wouldBlock()) {
        int sel = sliceSelect(sock, true, false, 500);
        if (sel < 0)
          break;
        recvWaited += 500;
        continue;
      }
      break;
    }
    closesocket(sock);
    if (result.error == "aborted")
      return result;
  }

  auto hdrEnd = rawResponse.find("\r\n\r\n");
  std::string httpBody;
  int httpStatus = 0;
  if (hdrEnd != std::string::npos) {
    httpBody = rawResponse.substr(hdrEnd + 4);
    auto sp1 = rawResponse.find(' ');
    if (sp1 != std::string::npos) {
      auto sp2 = rawResponse.find(' ', sp1 + 1);
      try {
        httpStatus = std::stoi(rawResponse.substr(sp1 + 1, sp2 - sp1 - 1));
      } catch (...) {
      }
    }
    bool chunked =
        rawResponse.find("Transfer-Encoding: chunked") != std::string::npos ||
        rawResponse.find("transfer-encoding: chunked") != std::string::npos;
    if (chunked) {
      std::string decoded;
      size_t pos = 0;
      while (pos < httpBody.size()) {
        auto crlf = httpBody.find("\r\n", pos);
        if (crlf == std::string::npos) break;
        size_t chunkSize = 0;
        try {
          chunkSize = std::stoul(httpBody.substr(pos, crlf - pos), nullptr, 16);
        } catch (...) {
          break;
        }
        if (chunkSize == 0) break;
        pos = crlf + 2;
        if (pos + chunkSize > httpBody.size()) break;
        decoded.append(httpBody, pos, chunkSize);
        pos += chunkSize + 2;
      }
      httpBody = decoded;
    }
  } else if (!rawResponse.empty()) {
    httpBody = rawResponse;
    httpStatus = 200;
  }

  result.status = httpStatus;
  result.body = httpBody;
  return result;
}

bool postJson(const std::string &host, int port, const std::string &path,
              const json &payload, int timeoutMs, json &outJson,
              std::string &error) {
  HttpResult r =
      httpRequest(host, port, "POST", path, payload.dump(), timeoutMs);
  if (r.connectFailed) {
    error = r.error;
    return false;
  }
  if (r.status < 200 || r.status >= 300) {
    error = path + " bad status " + std::to_string(r.status) + ": " +
            r.body.substr(0, 300);
    return false;
  }
  outJson = json::parse(r.body, nullptr, false);
  if (outJson.is_discarded()) {
    error = path + " invalid json response";
    return false;
  }
  return true;
}

bool getJson(const std::string &host, int port, const std::string &path,
             int timeoutMs, json &outJson, std::string &error) {
  HttpResult r = httpRequest(host, port, "GET", path, "", timeoutMs);
  if (r.connectFailed) {
    error = r.error;
    return false;
  }
  if (r.status < 200 || r.status >= 300) {
    error = path + " bad status " + std::to_string(r.status) + ": " +
            r.body.substr(0, 300);
    return false;
  }
  outJson = json::parse(r.body, nullptr, false);
  if (outJson.is_discarded()) {
    error = path + " invalid json response";
    return false;
  }
  return true;
}

// Sanitizes a string so that it is valid UTF-8 and safe to embed in a JSON
// payload without triggering nlohmann::json::dump() type_error.316.
std::string sanitizeUtf8(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  size_t i = 0;
  while (i < s.size()) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    size_t len = 1;
    if (c >= 0xC0 && c <= 0xDF) len = 2;
    else if (c >= 0xE0 && c <= 0xEF) len = 3;
    else if (c >= 0xF0 && c <= 0xF7) len = 4;
    else if (c >= 0x80) {
      out.push_back('?');
      ++i;
      continue;
    }
    if (i + len > s.size()) {
      out.push_back('?');
      ++i;
      continue;
    }
    bool valid = true;
    for (size_t j = 1; j < len; ++j) {
      unsigned char cb = static_cast<unsigned char>(s[i + j]);
      if ((cb & 0xC0) != 0x80) {
        valid = false;
        break;
      }
    }
    if (valid) {
      out.append(s, i, len);
      i += len;
    } else {
      out.push_back('?');
      ++i;
    }
  }
  return out;
}

// Builds the final user-facing prompt the same way chatWithExternalAdapter()
// does: strip an already-injected "[Context hint ...]" wrapper (if present)
// and re-inject the (possibly truncated) graph context as a "Context:\n"
// block ahead of the user text.
// (legacy buildPrompt removed — split path uses system+user messages only)

// The gateway's Ahead-memory context block ends with the CURRENT user text.
// Feeding the question twice (context + actual message) drives instruct
// models into degenerate loops; strip the trailing echo and any now-empty
// "[Ahead memory]" marker before the context is used as a system message.
std::string cleanGraphContextForSystem(const std::string &graphContext,
                                       const std::string &text) {
  std::string ctx = graphContext;
  const auto rtrim = [](std::string &s) {
    while (!s.empty() &&
           (s.back() == ' ' || s.back() == '\t' || s.back() == '\n' || s.back() == '\r'))
      s.pop_back();
  };
  if (!text.empty()) {
    std::string tail = text;
    rtrim(tail);
    if (!tail.empty()) {
      const size_t pos = ctx.rfind(tail);
      if (pos != std::string::npos && pos + tail.size() == ctx.size()) {
        ctx = ctx.substr(0, pos);
      }
    }
    rtrim(ctx);
    const std::string marker = "[Ahead memory]";
    const size_t mpos = ctx.rfind(marker);
    if (mpos != std::string::npos) {
      bool onlyMarkerLeft = true;
      for (size_t i = mpos + marker.size(); i < ctx.size(); ++i) {
        if (ctx[i] != ' ' && ctx[i] != '\t' && ctx[i] != '\n' && ctx[i] != '\r') {
          onlyMarkerLeft = false;
          break;
        }
      }
      if (onlyMarkerLeft) {
        ctx = ctx.substr(0, mpos);
        rtrim(ctx);
      }
    }
  }
  return ctx;
}

// Extract token ids from /phx/enc or /tokenize response shapes.
std::vector<int> tokensFromJson(const json &resp) {
  std::vector<int> out;
  const json *arr = nullptr;
  if (resp.is_array()) {
    arr = &resp;
  } else if (resp.is_object()) {
    if (resp.contains("tokens") && resp["tokens"].is_array())
      arr = &resp["tokens"];
    else if (resp.contains("ids") && resp["ids"].is_array())
      arr = &resp["ids"];
  }
  if (!arr) return out;
  for (const auto &t : *arr) {
    if (t.is_number_integer()) out.push_back(t.get<int>());
  }
  return out;
}

// Maps one word to token ids via the patched server's /phx/enc only.
std::vector<int> tokenizeWord(const std::string &host, int port,
                              const std::string &word, int timeoutMs) {
  json resp;
  std::string err;
  if (!postJson(host, port, "/phx/enc", json{{"content", word}}, timeoutMs, resp,
                err)) {
    return {};
  }
  return tokensFromJson(resp);
}

json normalizeLogitBias(const std::string &host, int port, const json &rawBias,
                        int timeoutMs);

// Forward sampling / slot / bias keys onto a /phx/generate payload.
void applyGenerateInferenceOptions(json &payload, const std::string &host,
                                   int port, int timeoutMs,
                                   const json &options) {
  if (!options.is_object()) return;
  if (options.contains("temperature") && options["temperature"].is_number())
    payload["temperature"] = options["temperature"];
  if (options.contains("top_p") && options["top_p"].is_number())
    payload["top_p"] = options["top_p"];
  for (const char *key : {"top_k", "min_p", "presence_penalty",
                          "frequency_penalty", "repeat_penalty", "seed"}) {
    if (options.contains(key) && options[key].is_number())
      payload[key] = options[key];
  }
  if (options.contains("n_parallel") && options["n_parallel"].is_number_integer())
    payload["n_parallel"] = options["n_parallel"];
  if (options.contains("parallel_mode") && options["parallel_mode"].is_string())
    payload["parallel_mode"] = options["parallel_mode"];
  if (options.contains("cache_prompt") && options["cache_prompt"].is_boolean())
    payload["cache_prompt"] = options["cache_prompt"].get<bool>();
  if (options.contains("id_slot") && options["id_slot"].is_number_integer())
    payload["id_slot"] = options["id_slot"].get<int>();
  if (options.contains("repeat_last_n") &&
      options["repeat_last_n"].is_number_integer())
    payload["repeat_last_n"] = options["repeat_last_n"].get<int>();
  if (options.contains("max_side_tokens") &&
      options["max_side_tokens"].is_number_integer())
    payload["max_side_tokens"] = options["max_side_tokens"].get<int>();
  if (options.contains("memoryWeight") && options["memoryWeight"].is_number())
    payload["memoryWeight"] = options["memoryWeight"];
  if (options.contains("gnnWeight") && options["gnnWeight"].is_number())
    payload["gnnWeight"] = options["gnnWeight"];
  if (options.contains("rag_beta") && options["rag_beta"].is_number())
    payload["rag_beta"] = options["rag_beta"];
  if (options.contains("leftover_beta") && options["leftover_beta"].is_number())
    payload["leftover_beta"] = options["leftover_beta"];
  if (options.contains("leftover_k") && options["leftover_k"].is_number_integer())
    payload["leftover_k"] = options["leftover_k"].get<int>();
  if (options.contains("ngram_merge") && options["ngram_merge"].is_number_integer())
    payload["ngram_merge"] = options["ngram_merge"].get<int>();
  if (options.contains("ngram_protect_last") &&
      options["ngram_protect_last"].is_number_integer())
    payload["ngram_protect_last"] = options["ngram_protect_last"].get<int>();
  if (options.contains("rag_mix_cap") && options["rag_mix_cap"].is_number_integer())
    payload["rag_mix_cap"] = options["rag_mix_cap"].get<int>();
  if (options.contains("num_predict") && options["num_predict"].is_number_integer())
    payload["max_tokens"] =
        std::max(1, options["num_predict"].get<int>());
  if (options.contains("logit_bias") && options["logit_bias"].is_object() &&
      !options["logit_bias"].empty()) {
    json mapped = normalizeLogitBias(host, port, options["logit_bias"], timeoutMs);
    if (!mapped.empty()) payload["logit_bias"] = mapped;
  }
  if (options.contains("loop_mode") && options["loop_mode"].is_string())
    payload["loop_mode"] = options["loop_mode"].get<std::string>();
  if (options.contains("feedback_mode") && options["feedback_mode"].is_string())
    payload["feedback_mode"] = options["feedback_mode"].get<std::string>();
  if (options.contains("prefix_hidden") && options["prefix_hidden"].is_array())
    payload["prefix_hidden"] = options["prefix_hidden"];
  if (options.contains("context_content") && options["context_content"].is_string())
    payload["context_content"] = options["context_content"];
  if (options.contains("gnn_content") && options["gnn_content"].is_string())
    payload["gnn_content"] = options["gnn_content"];
  if (options.contains("context_units") && options["context_units"].is_array())
    payload["context_units"] = options["context_units"];
  if (options.contains("gnn_units") && options["gnn_units"].is_array())
    payload["gnn_units"] = options["gnn_units"];
  if (options.contains("resume_suffix") && options["resume_suffix"].is_string())
    payload["resume_suffix"] = options["resume_suffix"];
  if (options.contains("stop") && options["stop"].is_array())
    payload["stop"] = options["stop"];
  if (options.contains("return_hidden") && options["return_hidden"].is_boolean())
    payload["return_hidden"] = options["return_hidden"].get<bool>();
  if (options.contains("decode_text") && options["decode_text"].is_boolean())
    payload["decode_text"] = options["decode_text"].get<bool>();
}

std::vector<std::vector<float>> extractHiddenRows(const json &resp) {
  std::vector<std::vector<float>> rows;
  if (!resp.is_object() || !resp.contains("hidden") ||
      !resp["hidden"].is_array())
    return rows;
  return inference::jsonToUnitRows(resp["hidden"]);
}

bool phxEncodePrompt(const std::string &host, int port,
                     const std::string &formattedPrompt, int timeoutMs,
                     std::vector<std::vector<float>> &outHidden,
                     std::string &error) {
  json resp;
  if (!postJson(host, port, "/phx/enc",
                json{{"content", formattedPrompt},
                     {"add_special", true},
                     {"granularity", "token"}},
                timeoutMs, resp, error))
    return false;
  outHidden = extractHiddenRows(resp);
  if (outHidden.empty()) {
    error = "phx/enc: empty hidden";
    return false;
  }
  return true;
}

json runPhxGenerate(const std::string &host, int port, json payload,
                    int timeoutMs, std::string &error) {
  json genResp;
  if (!postJson(host, port, "/phx/generate", payload, timeoutMs, genResp,
                error))
    return json{{"ok", false}, {"error", error}};
  if (!genResp.is_object() || !genResp.contains("text") ||
      !genResp["text"].is_string()) {
    error = "phx/generate: missing text";
    return json{{"ok", false}, {"error", error}};
  }
  return json{{"ok", true}, {"reply", genResp["text"].get<std::string>()},
              {"raw", genResp}};
}

// Rewrites a string-keyed logit_bias map into the integer-token-id map that
// llama-server accepts.  Entries that cannot be mapped are dropped (never
// send garbage); numeric-string keys are used as-is.
json normalizeLogitBias(const std::string &host, int port,
                        const json &rawBias, int timeoutMs) {
  json out = json::object();
  if (!rawBias.is_object()) return out;
  for (auto it = rawBias.begin(); it != rawBias.end(); ++it) {
    const std::string key = it.key();
    if (!it.value().is_number()) continue;
    const double v = it.value().get<double>();
    bool allDigits = !key.empty();
    for (char ch : key) {
      if (ch < '0' || ch > '9') { allDigits = false; break; }
    }
    if (allDigits) {
      out[key] = v;
      continue;
    }
    for (int id : tokenizeWord(host, port, key, timeoutMs)) {
      out[std::to_string(id)] = v;
    }
  }
  return out;
}

// Only cache a successful probe. A 4s miss while llama is in /phx/generate
// (--parallel 1 + phx_mutex) used to lock the whole mission on "unavailable".
// The unit path encodes inside /phx/generate; this probe is optional.
bool splitBackendReachable(const std::string &host, int port) {
  static std::mutex mu;
  static std::map<std::string, bool> cache;
  const std::string key = host + ":" + std::to_string(port);
  {
    std::lock_guard<std::mutex> lk(mu);
    const auto it = cache.find(key);
    if (it != cache.end() && it->second) return true;
  }
  json resp;
  std::string err;
  const bool ok =
      postJson(host, port, "/phx/enc", json{{"content", "probe"}}, 8000, resp,
               err) &&
      resp.is_object() && (resp.contains("tokens") || resp.contains("hidden") ||
                           resp.contains("n_tokens"));
  if (ok) {
    std::lock_guard<std::mutex> lk(mu);
    cache[key] = true;
  }
  return ok;
}

// Returns token count for content via /tokenize (falls back to chars/4).
int countContentTokens(const std::string &host, int port,
                       const std::string &content, int timeoutMs) {
  json resp;
  std::string err;
  if (!postJson(host, port, "/tokenize", json{{"content", content}}, timeoutMs,
                resp, err)) {
    return static_cast<int>(content.size() / 4);
  }
  return static_cast<int>(tokensFromJson(resp).size());
}

/* Causal continuation keeps the tail. Head+ellipsis made the 8B
   jump topics (game review after a 3-number pin). */
std::string clipPromptTail(const std::string &s, size_t charBudget) {
  if (s.size() <= charBudget) return s;
  return s.substr(s.size() - charBudget);
}

// Truncate formatted prompt so prompt_tokens + genReserve fits ctx budget.
std::string fitPromptToTokenBudget(const std::string &host, int port,
                                   std::string prompt, int ctxTokens,
                                   int genReserve, int timeoutMs,
                                   int ngramMerge = 1) {
  const int fac = ngramMerge >= 2 ? std::min(ngramMerge, 3) : 1;
  const int budget =
      std::max(256, (ctxTokens - genReserve - 64) * fac);
  int nTok = countContentTokens(host, port, prompt, timeoutMs);
  if (nTok <= budget) return prompt;
  size_t lo = prompt.size() / 4;
  size_t hi = prompt.size();
  std::string best = clipPromptTail(prompt, lo);
  while (lo + 1 < hi) {
    const size_t mid = (lo + hi) / 2;
    const std::string candidate = clipPromptTail(prompt, mid);
    nTok = countContentTokens(host, port, candidate, timeoutMs);
    if (nTok <= budget) {
      best = candidate;
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return best;
}

// Native llama-server /completion (slot path + proper sampler).  Text-only;
// keeps /phx/* for hidden-state multimodal unit queries.
json textCompletionFallback(const std::string &host, int port,
                            const std::string &formattedPrompt,
                            int effectiveMaxTokens, int timeoutMs,
                            const json &inferenceOptions) {
  json out;
  json payload = {{"prompt", formattedPrompt},
                  {"n_predict", effectiveMaxTokens},
                  {"stream", false},
                  {"cache_prompt", false}};
  if (inferenceOptions.is_object()) {
    if (inferenceOptions.contains("temperature") &&
        inferenceOptions["temperature"].is_number())
      payload["temperature"] = inferenceOptions["temperature"];
    if (inferenceOptions.contains("top_p") &&
        inferenceOptions["top_p"].is_number())
      payload["top_p"] = inferenceOptions["top_p"];
    for (const char *key :
         {"top_k", "min_p", "presence_penalty", "frequency_penalty", "seed"}) {
      if (inferenceOptions.contains(key) &&
          inferenceOptions[key].is_number())
        payload[key] = inferenceOptions[key];
    }
    if (inferenceOptions.contains("num_predict") &&
        inferenceOptions["num_predict"].is_number_integer())
      payload["n_predict"] =
          std::max(1, inferenceOptions["num_predict"].get<int>());
  }
  std::string error;
  json resp;
  if (!postJson(host, port, "/completion", payload, timeoutMs, resp, error)) {
    out["ok"] = false;
    out["error"] = "completion: " + error;
    return out;
  }
  std::string text;
  if (resp.is_object() && resp.contains("content") &&
      resp["content"].is_string()) {
    text = resp["content"].get<std::string>();
  } else if (resp.is_array() && !resp.empty() && resp[0].is_object() &&
             resp[0].contains("content") &&
             resp[0]["content"].is_string()) {
    text = resp[0]["content"].get<std::string>();
  }
  if (text.empty()) {
    out["ok"] = false;
    out["error"] = "completion: missing content in response";
    return out;
  }
  out["ok"] = true;
  out["reply"] = text;
  return out;
}

// Unified unit iteration: I/O enc -> memory/gnn UQ -> infer -> I/O dec.
json unitIterationPipeline(const std::string &host, int port,
                           const std::string &formattedPrompt,
                           int effectiveMaxTokens, int timeoutMs,
                           const json &inferenceOptions) {
  json out;
  std::string error;
  try {
    const auto pipeCfg = inference::pipelineConfigFromJson(inferenceOptions);
    // Encode on the server inside /phx/generate.  Shipping prompt hidden
    // (n_tokens * n_embd floats) over HTTP corrupts long mission prompts.
    double temperature = 0.0;
    double topP = 0.9;
    if (inferenceOptions.is_object()) {
      if (inferenceOptions.contains("temperature") &&
          inferenceOptions["temperature"].is_number())
        temperature = inferenceOptions["temperature"].get<double>();
      if (inferenceOptions.contains("top_p") && inferenceOptions["top_p"].is_number())
        topP = inferenceOptions["top_p"].get<double>();
    }

    auto buildPayload = [&](const std::string &feedbackMode) {
      json payload = {{"content", formattedPrompt},
                      {"max_tokens", effectiveMaxTokens},
                      {"temperature", temperature},
                      {"top_p", topP},
                      {"decode_text", true},
                      {"return_hidden", false},
                      {"loop_mode", pipeCfg.loopMode.empty() ? "unit"
                                                             : pipeCfg.loopMode},
                      {"feedback_mode", feedbackMode}};
      applyGenerateInferenceOptions(payload, host, port, timeoutMs,
                                    inferenceOptions);
      payload.erase("prefix_hidden");
      payload.erase("memory_units");
      if (!pipeCfg.memoryEnabled) {
        payload.erase("context_content");
        payload.erase("context_units");
      }
      if (!pipeCfg.gnnEnabled) {
        payload.erase("gnn_content");
        payload.erase("gnn_units");
      }
      return payload;
    };

    json attempt =
        runPhxGenerate(host, port, buildPayload(pipeCfg.feedbackMode),
                       timeoutMs, error);
    if (!attempt.value("ok", false) && pipeCfg.autoFeedbackFallback &&
        pipeCfg.feedbackMode != "dec_enc") {
      attempt = runPhxGenerate(host, port, buildPayload("dec_enc"), timeoutMs,
                               error);
      if (attempt.value("ok", false))
        attempt["feedbackFallback"] = "dec_enc";
    }
    if (!attempt.value("ok", false)) {
      out["ok"] = false;
      out["error"] = attempt.value("error", error);
      return out;
    }
    out["ok"] = true;
    out["reply"] = attempt.value("reply", std::string());
    if (attempt.contains("feedbackFallback"))
      out["feedbackFallback"] = attempt["feedbackFallback"];
    return out;
  } catch (const std::exception &e) {
    out["ok"] = false;
    out["error"] = std::string("unit pipeline exception: ") + e.what();
    return out;
  } catch (...) {
    out["ok"] = false;
    out["error"] = "unit pipeline unknown error";
    return out;
  }
}

std::string trimWsCopy(std::string s) {
  size_t a = 0;
  while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  size_t b = s.size();
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}

/* Raw completion prefix. No ChatML, no system/user/assistant roles.
   One agent cannot be two people; autonomy is a single text stream. */
std::string assembleCausalText(const std::string &prefix,
                               const std::string &body) {
  const std::string a = trimWsCopy(prefix);
  const std::string b = trimWsCopy(body);
  if (a.empty()) return b;
  if (b.empty()) return a;
  if (b.size() >= a.size() && b.compare(0, a.size(), a) == 0) return b;
  std::string out = a;
  if (out.back() != '\n') out.push_back('\n');
  out += b;
  return out;
}

json hiddenStatePipeline(const std::string &host, int port,
                          const std::string &text,
                          const std::string &graphContext, int effectiveMaxTokens,
                          int timeoutMs, const json &inferenceOptions) {
  json out;
  std::string error;
  try {
    const std::string cleanCtx = cleanGraphContextForSystem(graphContext, text);
    std::string formattedPrompt =
        assembleCausalText(cleanCtx, text);
    std::string resume;
    if (inferenceOptions.is_object() &&
        inferenceOptions.contains("resume_suffix") &&
        inferenceOptions["resume_suffix"].is_string())
      resume = inferenceOptions["resume_suffix"].get<std::string>();
    /* Do not fold resume into content. Server needs resume_tokens so
       n_resume drives RAG protect. Empty content is illegal; if the
       prefix is empty, use resume as content and drop the suffix. */
    json genOpts =
        inferenceOptions.is_object() ? inferenceOptions : json::object();
    if (formattedPrompt.empty()) {
      formattedPrompt = resume;
      resume.clear();
      genOpts.erase("resume_suffix");
    }
    if (formattedPrompt.empty()) {
      out["ok"] = false;
      out["error"] = "empty causal text";
      return out;
    }

    const int ctxBudget =
        inferenceOptions.is_object() &&
                inferenceOptions.contains("ctxTokenBudget") &&
                inferenceOptions["ctxTokenBudget"].is_number_integer()
            ? std::max(512, inferenceOptions["ctxTokenBudget"].get<int>())
            : 3584;
    const int ngramFit =
        inferenceOptions.is_object() &&
                inferenceOptions.contains("ngram_merge") &&
                inferenceOptions["ngram_merge"].is_number_integer()
            ? inferenceOptions["ngram_merge"].get<int>()
            : 2;
    formattedPrompt = fitPromptToTokenBudget(
        host, port, std::move(formattedPrompt), ctxBudget,
        effectiveMaxTokens + 64, timeoutMs, ngramFit);

    if (phoenix::inference::shutdownRequested()) {
      out["ok"] = false;
      out["error"] = "aborted";
      return out;
    }
    const bool useNative = genOpts.value("useNativeCompletion", false);
    const bool unitPipeline = genOpts.value("unitPipeline", true);

    auto isAbortErr = [](const json &r) {
      const std::string err = r.value("error", std::string());
      return phoenix::inference::shutdownRequested() ||
             err.find("aborted") != std::string::npos;
    };
    if (unitPipeline) {
      json unit = unitIterationPipeline(host, port, formattedPrompt,
                                        effectiveMaxTokens, timeoutMs,
                                        genOpts);
      if (unit.value("ok", false) &&
          !unit.value("reply", std::string()).empty())
        return unit;
      if (isAbortErr(unit)) {
        out["ok"] = false;
        out["error"] = "aborted";
        return out;
      }
      std::cerr << "[llama] unit /phx/generate failed; falling back to "
                   "/completion: "
                << unit.value("error", std::string("empty reply")) << std::endl;
    }
    if (phoenix::inference::shutdownRequested()) {
      out["ok"] = false;
      out["error"] = "aborted";
      return out;
    }
    if (useNative || unitPipeline) {
      json native = textCompletionFallback(host, port, formattedPrompt,
                                           effectiveMaxTokens, timeoutMs,
                                           genOpts);
      const std::string nativeErr = native.value("error", std::string());
      if (!native.value("ok", false) &&
          nativeErr.find("status 0") != std::string::npos) {
        native = textCompletionFallback(host, port, formattedPrompt,
                                        effectiveMaxTokens, timeoutMs,
                                        genOpts);
      }
      if (native.value("ok", false)) return native;
      if (isAbortErr(native)) {
        out["ok"] = false;
        out["error"] = "aborted";
        return out;
      }
      if (useNative) {
        out["ok"] = false;
        out["error"] =
            "completion: " + native.value("error", std::string("failed"));
        return out;
      }
    }

    double temperature = 0.0;
    double topP = 0.9;
    if (inferenceOptions.is_object()) {
      if (inferenceOptions.contains("temperature") &&
          inferenceOptions["temperature"].is_number())
        temperature = inferenceOptions["temperature"].get<double>();
      if (inferenceOptions.contains("top_p") && inferenceOptions["top_p"].is_number())
        topP = inferenceOptions["top_p"].get<double>();
    }

    json generatePayload = {{"content", formattedPrompt},
                            {"max_tokens", effectiveMaxTokens},
                            {"temperature", temperature},
                            {"top_p", topP},
                            {"decode_text", true},
                            {"loop_mode", "token"}};
    applyGenerateInferenceOptions(generatePayload, host, port, timeoutMs,
                                  genOpts);

    json genResp;
    if (!postJson(host, port, "/phx/generate", generatePayload, timeoutMs, genResp,
                  error)) {
      out["ok"] = false;
      out["error"] = "phx/generate: " + error;
      return out;
    }
    if (!genResp.is_object() || !genResp.contains("text") ||
        !genResp["text"].is_string()) {
      out["ok"] = false;
      out["error"] = "phx/generate: missing text in response";
      return out;
    }

    out["ok"] = true;
    out["reply"] = genResp["text"].get<std::string>();
    return out;
  } catch (const std::exception &e) {
    out["ok"] = false;
    out["error"] = std::string("exception: ") + e.what();
    return out;
  } catch (...) {
    out["ok"] = false;
    out["error"] = "unknown error";
    return out;
  }
}

}  // namespace

json llamaSplitChat(const std::string &baseUrl, int timeoutMs,
                     const std::string &model, const std::string &text,
                     const std::string &graphContext, int maxTokens,
                     const json &inferenceOptions) {
  json out;
  out["provider"] = "llamacpp";
  std::string selectedModel = model.empty() ? std::string("llamacpp") : model;
  out["model"] = selectedModel;
  out["reply"] = "";

  std::string host;
  int port = 8082;
  std::string endpoint = baseUrl.empty() ? "http://127.0.0.1:8082" : baseUrl;
  splitHostPort(endpoint, host, port);

  int effectiveMaxTokens = std::max(1, maxTokens);
  if (inferenceOptions.is_object()) {
    if (inferenceOptions.contains("num_predict") &&
        inferenceOptions["num_predict"].is_number_integer())
      effectiveMaxTokens =
          std::max(1, inferenceOptions["num_predict"].get<int>());
  }
  int callTimeoutMs = timeoutMs;
  if (inferenceOptions.is_object() &&
      inferenceOptions.contains("timeoutMs") &&
      inferenceOptions["timeoutMs"].is_number_integer()) {
    callTimeoutMs = std::max(1000, inferenceOptions["timeoutMs"].get<int>());
  }

  /* Prefix is the pin / working facts. Body is the recent window.
     skipGraphContext only means GNN is in units, not that the prefix
     should be dropped when a body exists. */
  std::string ctxForPipeline;
  if (inferenceOptions.is_object() &&
      inferenceOptions.contains("system_content") &&
      inferenceOptions["system_content"].is_string()) {
    ctxForPipeline = inferenceOptions["system_content"].get<std::string>();
  } else {
    ctxForPipeline = graphContext;
  }

  const bool unitPipeline =
      !inferenceOptions.is_object() ||
      inferenceOptions.value("unitPipeline", true);
  if (unitPipeline && !splitBackendReachable(host, port)) {
    std::cerr << "[llama] /phx/enc probe missed (slot busy or timeout); "
                 "continuing on /phx/generate"
              << std::endl;
  }

  json pipelineResult = hiddenStatePipeline(
      host, port, text, ctxForPipeline, effectiveMaxTokens, callTimeoutMs,
      inferenceOptions);
  if (pipelineResult.is_object() && pipelineResult.value("ok", false)) {
    out["ok"] = true;
    out["reply"] = pipelineResult.value("reply", std::string());
    if (pipelineResult.contains("feedbackFallback"))
      out["feedbackFallback"] = pipelineResult["feedbackFallback"];
    return out;
  }

  std::string pipelineError =
      pipelineResult.is_object() ? pipelineResult.value("error", std::string())
                                  : std::string("unit pipeline failed");
  out["ok"] = false;
  out["reply"] = "";
  out["error"] = "split pipeline failed: " + pipelineError;
  return out;
}

}  // namespace v7
}  // namespace phoenix
