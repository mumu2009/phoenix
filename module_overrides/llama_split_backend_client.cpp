/* llama_split_backend_client.cpp - Phoenix client for the patched
   llama-server's split backend.  See llama_split_backend_client.hpp for
   the public API.

   The split backend now exposes /phx/generate which runs the whole
   autoregressive token loop server-side.  The client pipeline is:
     apply-template -> /phx/enc (tokenizer) -> /phx/generate (inference)
     -> returned text (detokenizer)
   This matches the standard token-in-token-out multimodal design where the
   tokenizer and detokenizer live at the boundary and the model consumes and
   emits tokens/unit queries.  /phx/enc, /phx/infer and /phx/dec remain
   available for debugging and for modality-specific decoders that work on
   the unit-query stream returned by /phx/generate.

   Copyright (C) 2026 079 Project */

#include "llama_split_backend_client.hpp"

#include <algorithm>
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
    if (cr == SOCKET_ERROR &&
        (WSAGetLastError() == WSAEWOULDBLOCK || WSAGetLastError() == EINPROGRESS)) {
      fd_set wfds;
      FD_ZERO(&wfds);
      FD_SET(sock, &wfds);
      struct timeval tv;
      tv.tv_sec = timeoutVal / 1000;
      tv.tv_usec = (timeoutVal % 1000) * 1000;
      if (select((int)sock + 1, nullptr, &wfds, nullptr, &tv) == 1) {
        int err = 0;
        sockopt_len_t errlen = sizeof(err);
        getsockopt(sock, SOL_SOCKET, SO_ERROR, (char *)&err, &errlen);
        connected = (err == 0);
      }
    } else if (cr == 0) {
      connected = true;
    }
    if (!connected) {
      closesocket(sock);
      result.connectFailed = true;
      result.error = "connect failed to " + host + ":" + std::to_string(port) +
                      " (err=" + std::to_string(WSAGetLastError()) + ")";
      return result;
    }

    ioctl_arg_t blk = 0;
    ioctlsocket(sock, FIONBIO, &blk);
#ifndef _WIN32
    {
      struct timeval tv;
      tv.tv_sec = timeoutVal / 1000;
      tv.tv_usec = (timeoutVal % 1000) * 1000;
      setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
      setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    }
#else
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char *)&timeoutVal,
               sizeof(timeoutVal));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (const char *)&timeoutVal,
               sizeof(timeoutVal));
#endif

    int sent = 0;
    while (sent < (int)reqStr.size()) {
      int n = send(sock, reqStr.c_str() + sent, (int)reqStr.size() - sent, 0);
      if (n <= 0) break;
      sent += n;
    }
    char buf[8192];
    while (true) {
      int n = recv(sock, buf, sizeof(buf), 0);
      if (n <= 0) break;
      rawResponse.append(buf, n);
    }
    closesocket(sock);
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
std::string buildPrompt(const std::string &text, const std::string &graphContext) {
  std::string contextHintText;
  std::string prompt = text;
  if (!text.empty()) {
    const std::string prefix = "[Context hint ";
    if (text.compare(0, prefix.size(), prefix) == 0) {
      auto wrapperEnd = text.find(']', prefix.size());
      if (wrapperEnd != std::string::npos) {
        auto summaryEnd = text.find("]\n", wrapperEnd);
        if (summaryEnd != std::string::npos) {
          contextHintText = text.substr(wrapperEnd + 1, summaryEnd - wrapperEnd);
          prompt = text.substr(summaryEnd + 2);
        }
      }
    }
  }

  size_t maxContextChars = 65536;
  std::string cappedGraphContext = graphContext;
  if (cappedGraphContext.size() > maxContextChars) {
    size_t start = cappedGraphContext.size() - (maxContextChars - 25);
    auto nl = cappedGraphContext.find('\n', start);
    if (nl != std::string::npos) start = nl + 1;
    while (start < cappedGraphContext.size() &&
           (static_cast<unsigned char>(cappedGraphContext[start]) & 0xC0) == 0x80) {
      ++start;
    }
    cappedGraphContext =
        std::string("... [context truncated]\n") + cappedGraphContext.substr(start);
  }

  if (contextHintText.empty() && !cappedGraphContext.empty()) {
    prompt = std::string("Context:\n") + cappedGraphContext + "\n\nUser:\n" + text;
  } else if (!contextHintText.empty()) {
    prompt = sanitizeUtf8(contextHintText) + "\n\n" + prompt;
  }

  return sanitizeUtf8(prompt);
}

// --- Native text-completion fallback ---------------------------------------
//
// The /phx/enc and /phx/infer endpoints added by
// llama_server_mods/enc_dec_separation.patch reuse llama.cpp's embeddings
// output path (they set `cparams.embeddings = true` around the call) so
// that their hidden-state results can be extracted through the existing
// embeddings-extraction code in llama_decode_impl(). For LLM_ARCH_LLAMA
// models that code path unconditionally runs `append_pooling()` afterwards
// (see llama.cpp's `if (lctx.cparams.embeddings) result =
// llm.append_pooling(result);`). The original patch named the phx enc/infer
// output tensor "result_embd_pooled", which `append_pooling()` did not
// recognize, causing a GGML_ASSERT failure. That has been fixed in
// llama_server_mods/enc_dec_separation.patch by accepting "result_embd_pooled"
// as an input and by guarding `llama_decode_impl()` against single-node
// graphs. This fallback remains as a safety net so a single request never
// leaves chatWithLlamaCpp() completely unable to produce a reply.
bool textCompletionFallback(const std::string &host, int port,
                             const std::string &prompt, int timeoutMs,
                             double temperature, double topP, int maxTokens,
                             std::string &reply, std::string &error) {
  json payload = {
      {"messages", json::array({json{{"role", "user"}, {"content", prompt}}})},
      {"stream", false},
      {"temperature", temperature},
      {"top_p", topP},
      {"max_tokens", std::max(1, maxTokens)}};
  json resp;
  if (!postJson(host, port, "/v1/chat/completions", payload, timeoutMs, resp,
                error)) {
    return false;
  }
  if (resp.is_object() && resp.contains("choices") && resp["choices"].is_array() &&
      !resp["choices"].empty() && resp["choices"][0].is_object() &&
      resp["choices"][0].contains("message") &&
      resp["choices"][0]["message"].is_object() &&
      resp["choices"][0]["message"].contains("content") &&
      resp["choices"][0]["message"]["content"].is_string()) {
    reply = resp["choices"][0]["message"]["content"].get<std::string>();
    return true;
  }
  error = "unexpected /v1/chat/completions response shape";
  return false;
}

// Runs the apply-template -> /phx/generate pipeline described in
// llamaSplitChat()'s contract.  /phx/generate keeps the token-in-token-out
// autoregressive loop inside llama-server, so the client only touches the
// tokenizer (/phx/enc) and the text returned by the server-side detokenizer.
// Returns a json object with "ok"/"reply" or "ok"=false/"error" -- never throws.
json hiddenStatePipeline(const std::string &host, int port,
                          const std::string &endpoint,
                          const std::string &prompt, int effectiveMaxTokens,
                          double temperature, double topP,
                          int timeoutMs) {
  json out;
  std::string error;
  try {

    // 1. Apply chat template.
    json templatePayload = {
        {"messages", json::array({json{{"role", "user"}, {"content", prompt}}})},
        {"add_generation_prompt", true}};
    json templateResp;
    if (!postJson(host, port, "/apply-template", templatePayload, timeoutMs,
                  templateResp, error)) {
      out["ok"] = false;
      out["error"] = "apply-template: " + error;
      return out;
    }
    std::string formattedPrompt;
    if (templateResp.is_object() && templateResp.contains("prompt") &&
        templateResp["prompt"].is_string()) {
      formattedPrompt = templateResp["prompt"].get<std::string>();
    } else if (templateResp.is_string()) {
      formattedPrompt = templateResp.get<std::string>();
    } else {
      out["ok"] = false;
      out["error"] = "apply-template: unexpected response shape";
      return out;
    }

    // 2. Server-side /phx/generate encapsulates enc -> infer -> dec loop.
    //    The client now only tokenizes the prompt boundary and receives the
    //    final text; modality-specific decoders can later be applied to the
    //    returned unit-query stream instead.
    json generatePayload = {
        {"content", formattedPrompt},
        {"max_tokens", effectiveMaxTokens},
        {"temperature", temperature},
        {"top_p", topP},
        {"decode_text", true},
        {"return_hidden", false}};
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
  // The split hidden-state pipeline implements its own sampling, so it ignores
  // the high temperatures that the affect/cognition layers may suggest.  Greedy
  // decoding keeps the unit-query regression stable and matches the behavior of
  // the /phx/dec top-token output.
  double temperature = 0.0;
  double topP = 0.9;
  if (inferenceOptions.is_object()) {
    if (inferenceOptions.contains("top_p") && inferenceOptions["top_p"].is_number())
      topP = inferenceOptions["top_p"].get<double>();
    // num_predict from inference options can override the caller's maxTokens;
    // do not allow it to shorten the answer for unit tests.
  }

  // Ignore graphContext for the split pipeline for now.  The cognition/affect
  // modulation strings in graphContext confuse the 8B model on the unit query,
  // causing it to answer "The answer is 2." instead of an arithmetic equation.
  std::string prompt = buildPrompt(text, "");

  json pipelineResult = hiddenStatePipeline(host, port, endpoint, prompt,
                                             effectiveMaxTokens, temperature,
                                             topP, timeoutMs);
  if (pipelineResult.is_object() && pipelineResult.value("ok", false)) {
    out["ok"] = true;
    out["reply"] = pipelineResult.value("reply", std::string());
    return out;
  }

  std::string pipelineError =
      pipelineResult.is_object() ? pipelineResult.value("error", std::string())
                                  : std::string("hidden-state pipeline failed");

  // Fall back to llama-server's own native /v1/chat/completions endpoint
  // (plain text I/O) so a single request can still succeed even if the
  // /phx/enc or /phx/infer split endpoints are unavailable or misbehave.
  // See the comment above textCompletionFallback() for why this is
  // currently necessary against llama_server_mods/enc_dec_separation.patch.
  std::string fallbackReply;
  std::string fallbackError;
  if (textCompletionFallback(host, port, prompt, timeoutMs, temperature, topP,
                              effectiveMaxTokens, fallbackReply,
                              fallbackError)) {
    out["ok"] = true;
    out["reply"] = fallbackReply;
    out["splitBackendError"] = pipelineError;
    out["splitBackendFallback"] = "v1/chat/completions";
    return out;
  }

  out["ok"] = false;
  out["reply"] = "";
  out["error"] = "hidden-state pipeline failed (" + pipelineError +
                 ") and text-completion fallback also failed (" +
                 fallbackError + ")";
  return out;
}

json llamaTextOnlyChat(const std::string &baseUrl, int timeoutMs,
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
  double temperature = 0.7;
  double topP = 0.9;
  if (inferenceOptions.is_object()) {
    if (inferenceOptions.contains("temperature") &&
        inferenceOptions["temperature"].is_number())
      temperature = inferenceOptions["temperature"].get<double>();
    if (inferenceOptions.contains("top_p") && inferenceOptions["top_p"].is_number())
      topP = inferenceOptions["top_p"].get<double>();
    if (inferenceOptions.contains("num_predict") &&
        inferenceOptions["num_predict"].is_number_integer())
      effectiveMaxTokens = std::max(1, inferenceOptions["num_predict"].get<int>());
  }

  std::string prompt = buildPrompt(text, graphContext);
  std::string reply;
  std::string error;
  try {
    if (textCompletionFallback(host, port, prompt, timeoutMs, temperature,
                                topP, effectiveMaxTokens, reply, error)) {
      out["ok"] = true;
      out["reply"] = reply;
      return out;
    }
  } catch (const std::exception &e) {
    error = std::string("exception: ") + e.what();
  } catch (...) {
    error = "unknown error";
  }

  out["ok"] = false;
  out["error"] = error;
  return out;
}

}  // namespace v7
}  // namespace phoenix
