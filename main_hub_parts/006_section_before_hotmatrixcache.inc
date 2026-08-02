/* Auto-generated from main.cpp
   Segment: section_before_HotMatrixCache
   Original lines: 1715-1873
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
   along with 079 Project.  If not, see <http://www.gnu.org/licenses/>. */

static ProcessReservedArena &globalReservedArena() {
  static ProcessReservedArena arena;
  return arena;
}

static void initProcessMemoryReservation() {
  static std::once_flag once;
  std::call_once(once, []() {
    double mb = phoenix::resolveConfig<double>("main.reservedMemMb", 256.0, "AI_RESERVED_MEM_MB");
    if (!std::isfinite(mb) || mb < 8.0)
      mb = 256.0;
    size_t bytes = (size_t)(mb * 1024.0 * 1024.0);
    globalReservedArena().initialize(bytes);
  });
}

static bool recvExact(SocketHandle sock, uint8_t *dst, size_t bytes,
                      int timeoutMs) {
  size_t got = 0;
  while (got < bytes) {
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(sock, &rfds);
    timeval tv;
    tv.tv_sec = timeoutMs / 1000;
    tv.tv_usec = (timeoutMs % 1000) * 1000;
    int rc = select((int)(sock + 1), &rfds, nullptr, nullptr, &tv);
    if (rc <= 0)
      return false;
#ifdef _WIN32
    int n = ::recv(sock, reinterpret_cast<char *>(dst + got),
                   (int)(bytes - got), 0);
#else
    ssize_t n = ::recv(sock, dst + got, bytes - got, 0);
#endif
    if (n <= 0)
      return false;
    got += (size_t)n;
  }
  return true;
}

static bool sendPacket(SocketHandle sock, const std::vector<uint8_t> &payload) {
  if (payload.size() > (size_t)std::numeric_limits<uint32_t>::max())
    return false;
  uint32_t len = htonl((uint32_t)payload.size());
  std::string header(reinterpret_cast<const char *>(&len), sizeof(len));
  if (!sendAll(sock, header))
    return false;

  size_t sent = 0;
  while (sent < payload.size()) {
#ifdef _WIN32
    int n = ::send(sock, reinterpret_cast<const char *>(payload.data() + sent),
                   (int)(payload.size() - sent), 0);
#else
    ssize_t n = ::send(sock, payload.data() + sent, payload.size() - sent, 0);
#endif
    if (n <= 0)
      return false;
    sent += (size_t)n;
  }
  return true;
}

static bool recvPacket(SocketHandle sock, std::vector<uint8_t> &payload,
                       int timeoutMs,
                       size_t maxBytes = 64ull * 1024ull * 1024ull) {
  payload.clear();
  uint32_t lenBe = 0;
  if (!recvExact(sock, reinterpret_cast<uint8_t *>(&lenBe), sizeof(lenBe),
                 timeoutMs))
    return false;
  uint32_t len = ntohl(lenBe);
  if (len == 0)
    return true;
  if ((size_t)len > maxBytes)
    return false;
  payload.resize((size_t)len);
  return recvExact(sock, payload.data(), payload.size(), timeoutMs);
}

static bool recvLine(SocketHandle sock, std::string &out, int timeoutMs) {
  out.clear();
  char ch;
  while (true) {
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(sock, &rfds);
    timeval tv;
    tv.tv_sec = timeoutMs / 1000;
    tv.tv_usec = (timeoutMs % 1000) * 1000;
    int rc = select((int)(sock + 1), &rfds, nullptr, nullptr, &tv);
    if (rc <= 0)
      return false;
#ifdef _WIN32
    int n = ::recv(sock, &ch, 1, 0);
#else
    ssize_t n = ::recv(sock, &ch, 1, 0);
#endif
    if (n <= 0)
      return false;
    if (ch == '\n')
      break;
    if (ch != '\r')
      out.push_back(ch);
  }
  return true;
}

static std::string urlEncode(const std::string &s) {
  std::ostringstream oss;
  for (unsigned char c : s) {
    if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
      oss << c;
    } else if (c == ' ') {
      oss << '%';
      oss << std::hex << std::uppercase << (int)c << std::nouppercase
          << std::dec;
    } else {
      oss << '%';
      oss << std::hex << std::uppercase << (int)c << std::nouppercase
          << std::dec;
    }
  }
  return oss.str();
}

static std::string urlDecode(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size(); i++) {
    char c = s[i];
    if (c == '+') {
      out.push_back(' ');
      continue;
    }
    if (c == '%' && i + 2 < s.size()) {
      auto hex = s.substr(i + 1, 2);
      char *end = nullptr;
      long v = std::strtol(hex.c_str(), &end, 16);
      if (end && *end == '\0') {
        out.push_back(static_cast<char>(v));
        i += 2;
        continue;
      }
    }
    out.push_back(c);
  }
  return out;
}

// HotMatrixCache 提供矩阵热点缓存与命中统计能力。
// 调用方式：在矩阵计算前后分别调用 get/put 管理热点项。
// 实现思路：本地缓存优先，按策略可同步到外部缓存层。
// 注意事项：缓存命中前需校验维度与版本一致性。
// 注意事项：缓存不可替代源数据正确性校验。
// 注意事项：高并发场景需配合外层限流策略。
