/* Auto-generated from main.cpp
   Segment: section_before_FetchResult
   Original lines: 1987-2093
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

static json parseFormUrlEncoded(const std::string &body) {
  json out = json::object();
  size_t start = 0;
  while (start <= body.size()) {
    size_t amp = body.find('&', start);
    if (amp == std::string::npos)
      amp = body.size();
    std::string pair = body.substr(start, amp - start);
    size_t eq = pair.find('=');
    std::string key = eq == std::string::npos ? pair : pair.substr(0, eq);
    std::string val = eq == std::string::npos ? "" : pair.substr(eq + 1);
    key = urlDecode(key);
    val = urlDecode(val);
    if (!key.empty()) {
      if (!out.contains(key)) {
        out[key] = val;
      } else if (out[key].is_array()) {
        out[key].push_back(val);
      } else {
        json arr = json::array();
        arr.push_back(out[key]);
        arr.push_back(val);
        out[key] = arr;
      }
    }
    if (amp == body.size())
      break;
    start = amp + 1;
  }
  return out;
}

static std::string normalizeUrlSimple(const std::string &url) {
  if (url.empty())
    return {};
  std::string out = url;
  auto hashPos = out.find('#');
  if (hashPos != std::string::npos)
    out.erase(hashPos);
  auto qPos = out.find('?');
  if (qPos != std::string::npos) {
    std::string base = out.substr(0, qPos);
    std::string query = out.substr(qPos + 1);
    std::vector<std::string> parts;
    std::string cur;
    for (char c : query) {
      if (c == '&') {
        if (!cur.empty())
          parts.push_back(cur);
        cur.clear();
      } else
        cur.push_back(c);
    }
    if (!cur.empty())
      parts.push_back(cur);
    std::vector<std::string> kept;
    for (auto &p : parts) {
      auto eq = p.find('=');
      std::string key = (eq == std::string::npos) ? p : p.substr(0, eq);
      std::string lower = key;
      std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
      if (lower.rfind("utm_", 0) == 0 || lower == "fbclid" || lower == "gclid")
        continue;
      kept.push_back(p);
    }
    if (!kept.empty()) {
      std::string q;
      for (size_t i = 0; i < kept.size(); i++) {
        if (i)
          q += "&";
        q += kept[i];
      }
      out = base + "?" + q;
    } else {
      out = base;
    }
  }
  return out;
}

static bool sameSiteSimple(const std::string &baseUrl,
                           const std::string &targetUrl) {
  auto hostOf = [](const std::string &u) {
    auto pos = u.find("//");
    if (pos == std::string::npos)
      return std::string();
    auto start = pos + 2;
    auto end = u.find('/', start);
    std::string host = u.substr(
        start, end == std::string::npos ? u.size() - start : end - start);
    if (host.rfind("www.", 0) == 0)
      host = host.substr(4);
    return host;
  };
  auto h1 = hostOf(baseUrl);
  auto h2 = hostOf(targetUrl);
  return !h1.empty() && h1 == h2;
}

// FetchResult 用于承载一次 HTTP 抓取的完整结果。
// 调用方式：由抓取函数返回给上层解析与清洗流程。
// 实现思路：集中保存正文、状态码、最终 URL 与字节数。
// 注意事项：字段可能为空或为默认值，需由调用方兜底。
// 注意事项：status 不等于 200 时不代表 body 一定为空。
// 注意事项：该结构仅做数据传递，不承担重试策略。
#if 0
#ifdef HAVE_CURL
#endif
#endif
