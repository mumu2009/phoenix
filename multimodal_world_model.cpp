/* multimodal_world_model.cpp - External LLaVA / Qwen2-Audio enc/dec client
   Copyright (C) 2026 079 Project */

#include "multimodal_world_model.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iomanip>
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
#include <netdb.h>
#include <netinet/in.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <sys/socket.h>
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
typedef socklen_t sockopt_len_t;
using ioctl_arg_t = int;
#endif

namespace phoenix {
namespace io {

using nlohmann::json;

namespace {

static const char kBase64Chars[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string base64Encode(const std::vector<uint8_t> &data) {
  std::string out;
  if (data.empty()) return out;
  size_t i = 0;
  while (i + 2 < data.size()) {
    uint32_t b = (uint32_t(data[i]) << 16) | (uint32_t(data[i + 1]) << 8) |
                 uint32_t(data[i + 2]);
    out.push_back(kBase64Chars[(b >> 18) & 0x3F]);
    out.push_back(kBase64Chars[(b >> 12) & 0x3F]);
    out.push_back(kBase64Chars[(b >> 6) & 0x3F]);
    out.push_back(kBase64Chars[b & 0x3F]);
    i += 3;
  }
  if (i < data.size()) {
    uint32_t b = uint32_t(data[i]) << 16;
    if (i + 1 < data.size()) b |= uint32_t(data[i + 1]) << 8;
    out.push_back(kBase64Chars[(b >> 18) & 0x3F]);
    out.push_back(kBase64Chars[(b >> 12) & 0x3F]);
    out.push_back((i + 1 < data.size()) ? kBase64Chars[(b >> 6) & 0x3F] : '=');
    out.push_back('=');
  }
  return out;
}

std::vector<uint8_t> base64Decode(const std::string &in) {
  std::vector<int> table(256, -1);
  for (int i = 0; i < 64; ++i) table[static_cast<unsigned char>(kBase64Chars[i])] = i;
  std::vector<uint8_t> out;
  if (in.empty()) return out;
  size_t i = 0;
  int val = 0, bits = 0;
  for (char c : in) {
    if (c == '=') break;
    int v = table[static_cast<unsigned char>(c)];
    if (v < 0) continue;
    val = (val << 6) | v;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(static_cast<uint8_t>((val >> bits) & 0xFF));
    }
  }
  return out;
}

struct HttpResult {
  bool connectFailed = false;
  int status = 0;
  std::string body;
  std::string error;
};

void splitHostPort(const std::string &baseUrl, std::string &host, int &port) {
  host = "127.0.0.1";
  port = 8085;
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
    if (cr == SOCKET_ERROR && WSAGetLastError() == WSAEWOULDBLOCK) {
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

}  // namespace

MultimodalImageWorldModel::MultimodalImageWorldModel(
    const MultimodalEncDecConfig &cfg)
    : cfg_(cfg) {}

MultimodalImageWorldModel::~MultimodalImageWorldModel() = default;

MultimodalImageEncResult MultimodalImageWorldModel::encode(
    const std::vector<uint8_t> &imageBytes,
    int width,
    int height,
    const std::string &mimeType) {
  MultimodalImageEncResult result;

  std::string host;
  int port = 8085;
  splitHostPort(cfg_.baseUrl, host, port);

  json req = {{"image", base64Encode(imageBytes)},
              {"mime_type", mimeType.empty() ? "image/png" : mimeType},
              {"width", width},
              {"height", height},
              {"return_sequence", true}};

  json resp;
  std::string error;
  if (!postJson(host, port, "/enc/image", req, cfg_.timeoutMs, resp, error)) {
    result.error = error;
    return result;
  }

  if (resp.value("ok", false) == false) {
    result.error = resp.value("error", "service returned ok=false");
    return result;
  }

  result.model = resp.value("model", cfg_.imageEncoderModel);
  if (resp.contains("unit_queries") && resp["unit_queries"].is_array()) {
    for (const auto &q : resp["unit_queries"]) {
      if (q.is_array()) {
        result.unitQueries.push_back(q.get<std::vector<float>>());
      }
    }
  }
  if (resp.contains("unit_query") && resp["unit_query"].is_array()) {
    result.meanUnitQuery = resp["unit_query"].get<std::vector<float>>();
  } else if (!result.unitQueries.empty()) {
    const size_t dim = result.unitQueries[0].size();
    result.meanUnitQuery.assign(dim, 0.0f);
    for (const auto &q : result.unitQueries) {
      for (size_t i = 0; i < dim; ++i) result.meanUnitQuery[i] += q[i];
    }
    for (float &v : result.meanUnitQuery) v /= float(result.unitQueries.size());
  }

  return result;
}

MultimodalDecResult MultimodalImageWorldModel::decode(
    const std::vector<float> &meanUnitQuery,
    const std::vector<std::vector<float>> &unitQueries,
    const std::string &mimeType,
    int width,
    int height) {
  MultimodalDecResult result;

  std::string host;
  int port = 8085;
  splitHostPort(cfg_.baseUrl, host, port);

  json req = {{"mean_unit_query", meanUnitQuery},
              {"unit_queries", unitQueries},
              {"mime_type", mimeType.empty() ? "image/png" : mimeType},
              {"width", width},
              {"height", height}};

  json resp;
  std::string error;
  if (!postJson(host, port, "/dec/image", req, cfg_.timeoutMs, resp, error)) {
    result.error = error;
    return result;
  }

  if (resp.value("ok", false) == false) {
    result.error = resp.value("error", "service returned ok=false");
    return result;
  }

  result.mimeType = resp.value("mime_type", mimeType);
  result.model = resp.value("model", "multimodal-image-decoder");
  if (resp.contains("payload") && resp["payload"].is_string()) {
    result.payload = base64Decode(resp["payload"].get<std::string>());
  } else if (resp.contains("image") && resp["image"].is_string()) {
    result.payload = base64Decode(resp["image"].get<std::string>());
  }
  if (result.payload.empty()) {
    result.error = "decoder returned empty payload";
  }

  return result;
}

nlohmann::json MultimodalImageWorldModel::status() const {
  if (statusLoaded_) return lastStatus_;

  std::string host;
  int port = 8085;
  splitHostPort(cfg_.baseUrl, host, port);

  json resp;
  std::string error;
  if (getJson(host, port, "/status", cfg_.timeoutMs, resp, error)) {
    lastStatus_ = resp;
    statusLoaded_ = true;
  } else {
    lastStatus_ = {{"ok", false}, {"error", error}};
  }
  return lastStatus_;
}

const std::string &MultimodalImageWorldModel::model() const {
  return cfg_.imageEncoderModel;
}

MultimodalAudioWorldModel::MultimodalAudioWorldModel(
    const MultimodalEncDecConfig &cfg)
    : cfg_(cfg) {}

MultimodalAudioWorldModel::~MultimodalAudioWorldModel() = default;

MultimodalAudioEncResult MultimodalAudioWorldModel::encode(
    const std::vector<uint8_t> &audioBytes,
    int sampleRate,
    const std::string &mimeType) {
  MultimodalAudioEncResult result;

  std::string host;
  int port = 8085;
  splitHostPort(cfg_.baseUrl, host, port);

  json req = {{"audio", base64Encode(audioBytes)},
              {"mime_type", mimeType.empty() ? "audio/wav" : mimeType},
              {"sample_rate", sampleRate},
              {"return_sequence", true}};

  json resp;
  std::string error;
  if (!postJson(host, port, "/enc/audio", req, cfg_.timeoutMs, resp, error)) {
    result.error = error;
    return result;
  }

  if (resp.value("ok", false) == false) {
    result.error = resp.value("error", "service returned ok=false");
    return result;
  }

  result.model = resp.value("model", cfg_.audioEncoderModel);
  if (resp.contains("unit_queries") && resp["unit_queries"].is_array()) {
    for (const auto &q : resp["unit_queries"]) {
      if (q.is_array()) {
        result.unitQueries.push_back(q.get<std::vector<float>>());
      }
    }
  }
  if (resp.contains("unit_query") && resp["unit_query"].is_array()) {
    result.meanUnitQuery = resp["unit_query"].get<std::vector<float>>();
  } else if (!result.unitQueries.empty()) {
    const size_t dim = result.unitQueries[0].size();
    result.meanUnitQuery.assign(dim, 0.0f);
    for (const auto &q : result.unitQueries) {
      for (size_t i = 0; i < dim; ++i) result.meanUnitQuery[i] += q[i];
    }
    for (float &v : result.meanUnitQuery)
      v /= float(result.unitQueries.size());
  }

  return result;
}

MultimodalDecResult MultimodalAudioWorldModel::decode(
    const std::vector<float> &meanUnitQuery,
    const std::vector<std::vector<float>> &unitQueries,
    const std::string &mimeType,
    size_t lengthHint) {
  MultimodalDecResult result;

  std::string host;
  int port = 8085;
  splitHostPort(cfg_.baseUrl, host, port);

  json req = {{"mean_unit_query", meanUnitQuery},
              {"unit_queries", unitQueries},
              {"mime_type", mimeType.empty() ? "audio/wav" : mimeType},
              {"length_hint", lengthHint}};

  json resp;
  std::string error;
  if (!postJson(host, port, "/dec/audio", req, cfg_.timeoutMs, resp, error)) {
    result.error = error;
    return result;
  }

  if (resp.value("ok", false) == false) {
    result.error = resp.value("error", "service returned ok=false");
    return result;
  }

  result.mimeType = resp.value("mime_type", mimeType);
  result.model = resp.value("model", "multimodal-audio-decoder");
  if (resp.contains("payload") && resp["payload"].is_string()) {
    result.payload = base64Decode(resp["payload"].get<std::string>());
  } else if (resp.contains("audio") && resp["audio"].is_string()) {
    result.payload = base64Decode(resp["audio"].get<std::string>());
  }
  if (result.payload.empty()) {
    result.error = "decoder returned empty payload";
  }

  return result;
}

nlohmann::json MultimodalAudioWorldModel::status() const {
  if (statusLoaded_) return lastStatus_;

  std::string host;
  int port = 8085;
  splitHostPort(cfg_.baseUrl, host, port);

  json resp;
  std::string error;
  if (getJson(host, port, "/status", cfg_.timeoutMs, resp, error)) {
    lastStatus_ = resp;
    statusLoaded_ = true;
  } else {
    lastStatus_ = {{"ok", false}, {"error", error}};
  }
  return lastStatus_;
}

const std::string &MultimodalAudioWorldModel::model() const {
  return cfg_.audioEncoderModel;
}

}  // namespace io
}  // namespace phoenix
