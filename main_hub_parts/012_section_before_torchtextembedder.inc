/* Auto-generated from main.cpp
   Segment: section_before_TorchTextEmbedder
   Original lines: 2114-2491
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

#ifdef HAVE_CURL
/* curlWriteCb handles libcurl chunked download data writing */
// 调用方式：由 libcurl 在收到网络数据时自动回调。
// 实现思路：按最大容量限制写入并返回已消费字节数。
// 注意事项：返回 0 会触发 curl 提前终止当前下载。
static size_t curlWriteCb(char *ptr, size_t size, size_t nmemb,
                          void *userdata) {
  auto *ctx = reinterpret_cast<CurlWriteCtx *>(userdata);
  size_t total = size * nmemb;
  if (ctx->maxBytes > 0 && ctx->out->size() + total > ctx->maxBytes) {
    size_t allowed = ctx->maxBytes - ctx->out->size();
    if (allowed > 0)
      ctx->out->append(ptr, allowed);
    return 0;
  }
  ctx->out->append(ptr, total);
  return total;
}

// fetchUrl 执行单次 HTTP 获取并返回结构化结果。
// 调用方式：传入 URL、超时、UA 与可选体积限制。
// 实现思路：配置 curl 选项并通过回调累计响应数据。
// 注意事项：网络异常时返回默认结果，调用方需判空校验。
static FetchResult
fetchUrl(const std::string &url, int timeoutMs = 10000,
         const std::string &userAgent = "079ProjectCrawler/1.0",
         size_t maxBytes = 0) {
  FetchResult res;
  CURL *curl = curl_easy_init();
  if (!curl)
    return res;
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, timeoutMs);
  curl_easy_setopt(curl, CURLOPT_USERAGENT, userAgent.c_str());
  CurlWriteCtx ctx{&res.body, maxBytes};
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlWriteCb);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ctx);
  curl_easy_perform(curl);
  char *ct = nullptr;
  curl_easy_getinfo(curl, CURLINFO_CONTENT_TYPE, &ct);
  if (ct)
    res.contentType = ct;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &res.status);
  char *finalUrl = nullptr;
  curl_easy_getinfo(curl, CURLINFO_EFFECTIVE_URL, &finalUrl);
  if (finalUrl)
    res.finalUrl = finalUrl;
  res.bytes = res.body.size();
  curl_easy_cleanup(curl);
  return res;
}
#endif

#ifdef HAVE_GUMBO
static void gumboExtractText(GumboNode *node, std::string &out) {
  if (!node)
    return;
  if (node->type == GUMBO_NODE_TEXT) {
    out.append(node->v.text.text);
    out.push_back(' ');
    return;
  }
  if (node->type != GUMBO_NODE_ELEMENT && node->type != GUMBO_NODE_DOCUMENT)
    return;
  const GumboVector *children = &node->v.element.children;
  for (size_t i = 0; i < children->length; i++) {
    gumboExtractText(static_cast<GumboNode *>(children->data[i]), out);
  }
}

static std::string extractTextFromHtml(const std::string &html) {
  GumboOutput *output = gumbo_parse(html.c_str());
  std::string text;
  gumboExtractText(output->root, text);
  gumbo_destroy_output(&kGumboDefaultOptions, output);
  return text;
}
#endif

#ifdef HAVE_POPPLER
static std::string extractTextFromPdfFile(const fs::path &file) {
  std::unique_ptr<poppler::document> doc(
      poppler::document::load_from_file(file.string()));
  if (!doc)
    return {};
  std::string out;
  for (int i = 0; i < doc->pages(); i++) {
    auto page = std::unique_ptr<poppler::page>(doc->create_page(i));
    if (!page)
      continue;
    auto bytes = page->text().to_utf8();
    out.append(bytes.begin(), bytes.end());
    out.push_back('\n');
  }
  return out;
}
#endif

static std::vector<std::string> parseJsStringLiterals(const std::string &src) {
  std::vector<std::string> out;
  const size_t n = src.size();
  for (size_t i = 0; i < n; i++) {
    char quote = src[i];
    if (quote != '\'' && quote != '"')
      continue;
    std::string buf;
    i++;
    for (; i < n; i++) {
      char c = src[i];
      if (c == '\\' && i + 1 < n) {
        char next = src[i + 1];
        if (next == 'n') {
          buf.push_back('\n');
          i++;
          continue;
        }
        if (next == 'r') {
          buf.push_back('\r');
          i++;
          continue;
        }
        if (next == 't') {
          buf.push_back('\t');
          i++;
          continue;
        }
        if (next == '\\' || next == '\'' || next == '"') {
          buf.push_back(next);
          i++;
          continue;
        }
        buf.push_back(next);
        i++;
        continue;
      }
      if (c == quote)
        break;
      buf.push_back(c);
    }
    if (!buf.empty())
      out.push_back(buf);
  }
  return out;
}

static bool loadNaturalStopwords(std::unordered_set<std::string> &out) {
  auto normalize = [](std::string s) {
    s = trimCopy(s);
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char ch) { return (char)std::tolower(ch); });
    return s;
  };

  auto collectFromText = [&](const std::string &content) {
    size_t before = out.size();
    std::istringstream iss(content);
    std::string line;
    while (std::getline(iss, line)) {
      line = trimCopy(line);
      if (line.empty())
        continue;
      if (line.rfind("//", 0) == 0 || line.rfind("#", 0) == 0)
        continue;
      for (char sep : {',', ';', '\t', ' '})
        std::replace(line.begin(), line.end(), sep, '\n');
      std::istringstream ps(line);
      std::string token;
      while (std::getline(ps, token)) {
        token = normalize(token);
        if (token.empty())
          continue;
        if (token.find_first_of("\r\n") != std::string::npos)
          continue;
        out.insert(token);
      }
    }
    return out.size() > before;
  };

  auto collectFromJs = [&](const std::string &content) {
    size_t before = out.size();
    auto words = parseJsStringLiterals(content);
    for (const auto &w : words) {
      std::string token = normalize(w);
      if (token.empty())
        continue;
      if (token.find_first_of(" \t\r\n") != std::string::npos)
        continue;
      out.insert(token);
    }
    return out.size() > before;
  };

  std::vector<fs::path> roots;
  roots.push_back(fs::current_path());
  if (!g_selfPath.empty()) {
    auto exeDir = fs::absolute(fs::path(g_selfPath)).parent_path();
    roots.push_back(exeDir);
    roots.push_back(exeDir.parent_path());
  }
  roots.push_back(fs::absolute(fs::path(__FILE__).parent_path()));
  roots.push_back(fs::absolute(fs::path(__FILE__).parent_path().parent_path()));

  std::vector<fs::path> candidates;
  for (const auto &root : roots) {
    candidates.push_back(root / "node_modules" / "natural" / "lib" / "natural" /
                         "util" / "stopwords.js");
    candidates.push_back(root / "robots" / "stopwords.txt");
    candidates.push_back(root / "doc" / "stopwords.txt");
    candidates.push_back(root / "stopwords.txt");
  }

  std::string envPaths = phoenix::resolveConfig<std::string>("main.stopwordsPath", "", "STOPWORDS_PATH");
  if (!envPaths.empty()) {
    for (char &ch : envPaths) {
      if (ch == ';' || ch == ',')
        ch = '\n';
    }
    std::istringstream ep(envPaths);
    std::string one;
    while (std::getline(ep, one)) {
      one = trimCopy(one);
      if (!one.empty())
        candidates.push_back(fs::path(one));
    }
  }

  for (const auto &file : candidates) {
    if (!fs::exists(file) || !fs::is_regular_file(file))
      continue;
    std::string content = readTextFile(file);
    if (content.empty())
      continue;
    bool isJs = file.extension() == ".js" || file.extension() == ".mjs" ||
                file.extension() == ".cjs";
    if (isJs)
      collectFromJs(content);
    else
      collectFromText(content);
  }

  if (out.empty()) {
    static const std::array<const char *, 64> fallback = {
        "a",     "an",    "the",    "and",   "or",   "but",   "if",   "then",
        "else",  "for",   "to",     "of",    "in",   "on",    "at",   "by",
        "with",  "as",    "is",     "are",   "was",  "were",  "be",   "been",
        "being", "do",    "does",   "did",   "have", "has",   "had",  "not",
        "this",  "that",  "these",  "those", "it",   "its",   "from", "into",
        "about", "over",  "under",  "than",  "too",  "very",  "can",  "could",
        "will",  "would", "should", "i",     "you",  "he",    "she",  "we",
        "they",  "them",  "my",     "your",  "our",  "their", "的",   "了"};
    for (auto *w : fallback)
      out.insert(std::string(w));
  }

  return !out.empty();
}

static const std::unordered_set<std::string> &getStopWords() {
  static std::unordered_set<std::string> stop;
  static bool loaded = false;
  if (!loaded) {
    loaded = true;
    loadNaturalStopwords(stop);
  }
  return stop;
}

static bool isAsciiWordToken(const std::string &token) {
  if (token.empty())
    return false;
  for (unsigned char c : token) {
    if (c >= 0x80)
      return false;
    if (!std::isalnum(c) && c != '_' && c != '-')
      return false;
  }
  return true;
}

static std::vector<std::string> tokenize(const std::string &text) {
  auto isCjk = [](uint32_t cp) { return cp >= 0x4e00 && cp <= 0x9fff; };
  auto decode = [](const std::string &s, size_t i, uint32_t &cp, size_t &len) {
    unsigned char c = (unsigned char)s[i];
    if (c < 0x80) {
      cp = c;
      len = 1;
      return true;
    }
    if ((c >> 5) == 0x6 && i + 1 < s.size()) {
      cp = ((c & 0x1f) << 6) | (s[i + 1] & 0x3f);
      len = 2;
      return true;
    }
    if ((c >> 4) == 0xe && i + 2 < s.size()) {
      cp = ((c & 0x0f) << 12) | ((s[i + 1] & 0x3f) << 6) | (s[i + 2] & 0x3f);
      len = 3;
      return true;
    }
    if ((c >> 3) == 0x1e && i + 3 < s.size()) {
      cp = ((c & 0x07) << 18) | ((s[i + 1] & 0x3f) << 12) |
           ((s[i + 2] & 0x3f) << 6) | (s[i + 3] & 0x3f);
      len = 4;
      return true;
    }
    cp = c;
    len = 1;
    return false;
  };

  std::vector<std::string> parts;
  std::string cur;
  auto flush = [&]() {
    if (cur.empty())
      return;
    if (isAsciiWordToken(cur) && getStopWords().count(cur)) {
      cur.clear();
      return;
    }
    parts.push_back(cur);
    cur.clear();
  };

  for (size_t i = 0; i < text.size();) {
    uint32_t cp = 0;
    size_t len = 1;
    decode(text, i, cp, len);
    if (cp == '\r' || cp == '\n' || cp == '\t' || cp == ' ') {
      flush();
      i += len;
      continue;
    }
    if (cp < 0x80) {
      unsigned char c = (unsigned char)cp;
      if (std::isalnum(c) || c == '_' || c == '-') {
        cur.push_back((char)std::tolower(c));
      } else {
        flush();
      }
      i += len;
      continue;
    }
    if (isCjk(cp)) {
      cur.append(text.substr(i, len));
    } else {
      flush();
    }
    i += len;
  }
  flush();
  return parts;
}

static double cosineSim(const std::vector<float> &a,
                        const std::vector<float> &b) {
  if (a.empty() || b.empty() || a.size() != b.size())
    return 0.0;
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (size_t i = 0; i < a.size(); i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na == 0.0 || nb == 0.0)
    return 0.0;
  return dot / (std::sqrt(na) * std::sqrt(nb));
}

// TorchTextEmbedder 提供轻量文本向量编码能力。
// 调用方式：在需要语义检索或相似度计算时调用。
// 实现思路：构建词表后通过 Embedding 与线性头生成向量。
// 注意事项：首次初始化可能耗时，建议在热身阶段完成。
// 注意事项：词表规模与最大长度会影响内存与精度。
// 注意事项：未就绪状态下调用需由上层进行降级处理。
#ifdef HAVE_TORCH
#endif
