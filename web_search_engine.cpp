/* web_search_engine.cpp - Implementation, see header for the design. */
#include "web_search_engine.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <map>
#include <mutex>
#include <regex>
#include <sstream>

#ifdef HAVE_CURL
#include <curl/curl.h>
#endif

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
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
using ioctl_arg_t = int;
typedef socklen_t sockopt_len_t;
#endif

namespace phoenix {
namespace websearch {

using json = nlohmann::json;

nlohmann::json SearchResult::toJson() const {
  return {{"title", title}, {"url", url}, {"snippet", snippet}, {"score", score}};
}

/* ---------------------------- URL encode ---------------------------- */

std::string urlEncode(const std::string &s) {
  static const char hex[] = "0123456789ABCDEF";
  std::string out;
  for (unsigned char c : s) {
    if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
      out.push_back(static_cast<char>(c));
    } else {
      out.push_back('%');
      out.push_back(hex[c >> 4]);
      out.push_back(hex[c & 0xF]);
    }
  }
  return out;
}

/* -------------------------- HTML helpers ---------------------------- */

static std::string decodeEntity(const std::smatch &m) {
  static const std::map<std::string, std::string> named = {
      {"amp", "&"}, {"lt", "<"}, {"gt", ">"}, {"quot", "\""}, {"apos", "'"},
      {"nbsp", " "}, {"mdash", "-"}, {"ndash", "-"}, {"hellip", "..."},
      {"rsquo", "'"}, {"lsquo", "'"}, {"rdquo", "\""}, {"ldquo", "\""},
      {"copy", "(c)"}, {"reg", "(R)"}, {"trade", "(TM)"}, {"deg", " degree "}};
  const std::string body = m[1].str();
  if (!body.empty() && body[0] == '#') {
    try {
      long cp = std::stol(body.substr(1));
      std::string u;
      if (cp < 0x80) u.push_back(static_cast<char>(cp));
      else if (cp < 0x800) {
        u.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        u.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
      } else {
        u.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        u.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        u.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
      }
      return u;
    } catch (...) {
      return m[0].str();
    }
  }
  auto it = named.find(body);
  return it == named.end() ? m[0].str() : it->second;
}

static std::string htmlDecode(const std::string &s) {
  static const std::regex re("&(#?[a-zA-Z0-9]+);");
  std::string out = s;
  std::smatch m;
  std::string::const_iterator start = out.cbegin();
  std::string acc;
  while (std::regex_search(start, out.cend(), m, re)) {
    acc.append(start, m[0].first);
    acc += decodeEntity(m);
    start = m[0].second;
  }
  acc.append(start, out.cend());
  return acc;
}

std::string htmlToText(const std::string &s) {
  /* decode entities first, then strip tags: "&lt;b&gt;" must vanish, not
     become a literal "<b>". script/style/noscript bodies are dropped so a
     page excerpt is prose, not bundled JavaScript. */
  std::string t = htmlDecode(s);
  auto startsWithIgnore = [](const std::string &src, size_t at,
                             const char *pat) {
    for (size_t i = 0; pat[i]; ++i) {
      if (at + i >= src.size()) return false;
      const char a = static_cast<char>(std::tolower(static_cast<unsigned char>(src[at + i])));
      const char b = static_cast<char>(std::tolower(static_cast<unsigned char>(pat[i])));
      if (a != b) return false;
    }
    return true;
  };
  std::string stripped;
  bool inTag = false;
  bool skip = false;
  const char *skipEnd = nullptr;
  for (size_t i = 0; i < t.size(); ++i) {
    if (!inTag && t[i] == '<') {
      if (startsWithIgnore(t, i, "<script")) {
        skip = true;
        skipEnd = "</script";
        inTag = true;
        continue;
      }
      if (startsWithIgnore(t, i, "<style")) {
        skip = true;
        skipEnd = "</style";
        inTag = true;
        continue;
      }
      if (startsWithIgnore(t, i, "<noscript")) {
        skip = true;
        skipEnd = "</noscript";
        inTag = true;
        continue;
      }
      inTag = true;
      continue;
    }
    if (inTag && t[i] == '>') {
      inTag = false;
      continue;
    }
    if (skip && skipEnd && startsWithIgnore(t, i, skipEnd)) {
      skip = false;
      skipEnd = nullptr;
      inTag = true;
      continue;
    }
    if (!inTag && !skip) stripped.push_back(t[i]);
  }
  t = std::move(stripped);
  /* collapse whitespace */
  std::string out;
  bool space = false;
  for (char c : t) {
    if (std::isspace(static_cast<unsigned char>(c))) {
      if (!space && !out.empty()) out.push_back(' ');
      space = true;
    } else {
      out.push_back(c);
      space = false;
    }
  }
  return out;
}

/* ------------------------ query / URL helpers ------------------------ */

static std::string lowerAscii(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

static std::vector<std::string> queryWords(const std::string &q) {
  std::vector<std::string> out;
  std::string cur;
  for (unsigned char c : q) {
    if (std::isalnum(c) || c == '-')
      cur.push_back(static_cast<char>(std::tolower(c)));
    else if (!cur.empty()) {
      bool alpha = false;
      for (char x : cur) {
        if (std::isalpha(static_cast<unsigned char>(x))) {
          alpha = true;
          break;
        }
      }
      if (cur.size() >= 4 && alpha) out.push_back(cur);
      cur.clear();
    }
  }
  if (cur.size() >= 4) {
    bool alpha = false;
    for (char x : cur) {
      if (std::isalpha(static_cast<unsigned char>(x))) {
        alpha = true;
        break;
      }
    }
    if (alpha) out.push_back(cur);
  }
  return out;
}

static int tokenOverlap(const std::string &text, const std::vector<std::string> &toks) {
  const std::string low = lowerAscii(text);
  int n = 0;
  for (const auto &t : toks) {
    if (low.find(t) != std::string::npos) ++n;
  }
  return n;
}

static bool isSerpOrLoginUrl(const std::string &url) {
  const std::string u = lowerAscii(url);
  if (u.rfind("javascript:", 0) == 0 || u.rfind("mailto:", 0) == 0) return true;
  if (u.find("bing.com/search") != std::string::npos) return true;
  if (u.find("bing.com/ck/") != std::string::npos) return true;
  if (u.find("duckduckgo.com") != std::string::npos) return true;
  if (u.find("baidu.com/s?") != std::string::npos) return true;
  if (u.find("account.microsoft.com") != std::string::npos) return true;
  if (u.find("login.live.com") != std::string::npos) return true;
  return false;
}

#ifdef HAVE_CURL
static size_t curlWriteBody(char *ptr, size_t size, size_t nmemb, void *ud) {
  auto *out = static_cast<std::string *>(ud);
  const size_t n = size * nmemb;
  if (out->size() + n > 2u * 1024u * 1024u) return 0;
  out->append(ptr, n);
  return n;
}

static std::string curlGetText(const std::string &url, int timeoutMs,
                               int maxRedirects, const std::string &userAgent) {
  static std::once_flag once;
  std::call_once(once, []() { curl_global_init(CURL_GLOBAL_DEFAULT); });
  CURL *c = curl_easy_init();
  if (!c) return "";
  std::string body;
  curl_easy_setopt(c, CURLOPT_URL, url.c_str());
  curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(c, CURLOPT_MAXREDIRS, static_cast<long>(std::max(1, maxRedirects)));
  curl_easy_setopt(c, CURLOPT_TIMEOUT_MS, static_cast<long>(std::max(1000, timeoutMs)));
  curl_easy_setopt(c, CURLOPT_USERAGENT, userAgent.c_str());
  curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, curlWriteBody);
  curl_easy_setopt(c, CURLOPT_WRITEDATA, &body);
  curl_easy_setopt(c, CURLOPT_ACCEPT_ENCODING, "");
  curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);
  const CURLcode rc = curl_easy_perform(c);
  long status = 0;
  curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &status);
  curl_easy_cleanup(c);
  if (rc != CURLE_OK || status < 200 || status >= 300) return "";
  return body;
}
#endif

std::string pageExcerpt(const std::string &html, const std::string &query,
                        size_t maxChars) {
  std::string text = htmlToText(html);
  if (text.size() > 40000) text.resize(40000);
  const auto toks = queryWords(query);
  if (toks.empty() || text.empty()) {
    if (text.size() > maxChars) text.resize(maxChars);
    return text;
  }
  std::string out;
  size_t i = 0;
  while (i < text.size() && out.size() < maxChars) {
    size_t end = text.find(". ", i);
    if (end == std::string::npos) end = text.size();
    else end += 1;
    std::string sent = text.substr(i, end - i);
    i = end + (end < text.size() && text[end] == ' ' ? 1 : 0);
    while (!sent.empty() && std::isspace(static_cast<unsigned char>(sent.front())))
      sent.erase(sent.begin());
    if (sent.size() < 40) continue;
    if (tokenOverlap(sent, toks) <= 0) continue;
    if (!out.empty()) out.push_back(' ');
    out += sent;
  }
  if (out.empty()) {
    out = text.substr(0, std::min(text.size(), maxChars));
  }
  if (out.size() > maxChars) out.resize(maxChars);
  return out;
}

/* --------------------------- raw HTTP GET --------------------------- */

namespace {
struct SplitUrl {
  std::string host;
  int port{80};
  std::string path{"/"};
};

SplitUrl splitHttpUrl(const std::string &urlIn) {
  SplitUrl out;
  std::string url = urlIn;
  if (url.rfind("http://", 0) == 0) url = url.substr(7);
  auto slash = url.find('/');
  std::string hostport = slash == std::string::npos ? url : url.substr(0, slash);
  out.path = slash == std::string::npos ? "/" : url.substr(slash);
  if (out.path.empty()) out.path = "/";
  auto colon = hostport.find(':');
  if (colon != std::string::npos) {
    out.host = hostport.substr(0, colon);
    try { out.port = std::stoi(hostport.substr(colon + 1)); } catch (...) { out.port = 80; }
  } else {
    out.host = hostport;
  }
  return out;
}
} /* namespace */

std::string httpGetText(const std::string &url, int timeoutMs, int maxRedirects,
                        const std::string &userAgent) {
#ifdef HAVE_CURL
  if (url.rfind("http://", 0) == 0 || url.rfind("https://", 0) == 0)
    return curlGetText(url, timeoutMs, maxRedirects, userAgent);
#endif
  if (url.rfind("https://", 0) == 0)
    return "";
  std::string current = url;
  for (int hop = 0; hop <= maxRedirects; ++hop) {
    const SplitUrl u = splitHttpUrl(current);
    if (u.host.empty()) return "";
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
    SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCKET) return "";
    const int timeoutVal = std::max(1000, timeoutMs);
    struct sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<unsigned short>(u.port));
    if (inet_pton(AF_INET, u.host.c_str(), &addr.sin_addr) != 1) {
      /* resolve hostname */
      struct hostent *he = gethostbyname(u.host.c_str());
      if (!he) { closesocket(sock); return ""; }
      std::memcpy(&addr.sin_addr, he->h_addr_list[0], static_cast<size_t>(he->h_length));
    }
    ioctl_arg_t nonblk = 1;
    ioctlsocket(sock, FIONBIO, &nonblk);
    bool connected = false;
    const int cr = connect(sock, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr));
    if (cr == SOCKET_ERROR &&
        (WSAGetLastError() == WSAEWOULDBLOCK || WSAGetLastError() == EINPROGRESS)) {
      fd_set wfds;
      FD_ZERO(&wfds);
      FD_SET(sock, &wfds);
      struct timeval tv;
      tv.tv_sec = timeoutVal / 1000;
      tv.tv_usec = (timeoutVal % 1000) * 1000;
      if (select(static_cast<int>(sock) + 1, nullptr, &wfds, nullptr, &tv) == 1) {
        int err = 0;
        sockopt_len_t errlen = sizeof(err);
        getsockopt(sock, SOL_SOCKET, SO_ERROR, reinterpret_cast<char *>(&err), &errlen);
        connected = (err == 0);
      }
    } else if (cr == 0) {
      connected = true;
    }
    if (!connected) { closesocket(sock); return ""; }
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
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char *>(&timeoutVal),
               sizeof(timeoutVal));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char *>(&timeoutVal),
               sizeof(timeoutVal));
#endif
    std::string req = "GET " + u.path + " HTTP/1.1\r\nHost: " + u.host +
                      "\r\nUser-Agent: " + userAgent +
                      "\r\nAccept: text/html,application/json\r\n"
                      "Accept-Encoding: identity\r\nConnection: close\r\n\r\n";
    size_t sent = 0;
    while (sent < req.size()) {
#ifdef _WIN32
      int n = send(sock, req.c_str() + sent, static_cast<int>(req.size() - sent), 0);
#else
      ssize_t n = send(sock, req.c_str() + sent, req.size() - sent, 0);
#endif
      if (n <= 0) { closesocket(sock); return ""; }
      sent += static_cast<size_t>(n);
    }
    std::string resp;
    char buf[8192];
    while (true) {
#ifdef _WIN32
      int n = recv(sock, buf, sizeof(buf), 0);
#else
      ssize_t n = recv(sock, buf, sizeof(buf), 0);
#endif
      if (n <= 0) break;
      resp.append(buf, static_cast<size_t>(n));
      if (resp.size() > 8u * 1024u * 1024u) break; /* 8MB safety cap */
    }
    closesocket(sock);
    if (resp.empty()) return "";
    auto headerEnd = resp.find("\r\n\r\n");
    if (headerEnd == std::string::npos) return "";
    std::string header = resp.substr(0, headerEnd);
    std::string body = resp.substr(headerEnd + 4);
    /* status line */
    std::istringstream hs(header);
    std::string statusLine;
    std::getline(hs, statusLine);
    int status = 0;
    {
      std::istringstream ss(statusLine);
      std::string http;
      ss >> http >> status;
    }
    if (status >= 300 && status < 400) {
      /* find Location */
      auto loc = header.find("\r\nLocation:");
      if (loc != std::string::npos) {
        std::string line = header.substr(loc + 12);
        auto eol = line.find("\r\n");
        if (eol != std::string::npos) line = line.substr(0, eol);
        /* trim */
        auto a = line.find_first_not_of(" \t");
        auto b = line.find_last_not_of(" \t");
        if (a != std::string::npos) line = line.substr(a, b - a + 1);
        current = line;
        continue;
      }
      return "";
    }
    if (status < 200 || status >= 300) return "";
    return body;
  }
  return "";
}

/* ------------------------- DuckDuckGo Lite -------------------------- */

std::vector<SearchResult> parseDdgLiteHtml(const std::string &html) {
  std::vector<SearchResult> out;
  /* lite.duckduckgo.com/lite: each hit is
     <a rel="nofollow" href="URL" class="result-link">TITLE</a> ...
     <td class="result-snippet">SNIPPET</td> */
  static const std::regex linkRe(
      R"ddg(<a[^>]*href="([^"]+)"[^>]*class="[^"]*result-link[^"]*"[^>]*>([\s\S]*?)</a>)ddg",
      std::regex::icase);
  static const std::regex snippetRe(
      R"ddg(<td[^>]*class="[^"]*result-snippet[^"]*"[^>]*>([\s\S]*?)</td>)ddg",
      std::regex::icase);
  std::sregex_iterator lit(html.begin(), html.end(), linkRe), lend;
  std::vector<std::pair<size_t, std::string>> snippets;
  for (std::sregex_iterator sit(html.begin(), html.end(), snippetRe), send;
       sit != send; ++sit) {
    snippets.emplace_back(static_cast<size_t>(sit->position()), sit->str(1));
  }
  size_t si = 0;
  for (; lit != lend; ++lit) {
    std::string url = lit->str(1);
    std::string title = htmlToText(lit->str(2));
    if (title.empty()) title = url;
    /* find first snippet after this link */
    while (si < snippets.size() && snippets[si].first < static_cast<size_t>(lit->position())) ++si;
    std::string snippet;
    if (si < snippets.size()) snippet = htmlToText(snippets[si].second);
    if (snippet.size() > 500) snippet.resize(500);
    /* DuckDuckGo wraps redirect URLs; unwrap uddg= param */
    auto uddg = url.find("uddg=");
    if (uddg != std::string::npos) {
      std::string inner = url.substr(uddg + 5);
      auto amp = inner.find('&');
      if (amp != std::string::npos) inner = inner.substr(0, amp);
      /* percent-decode inner */
      std::string decoded;
      for (size_t i = 0; i < inner.size(); ++i) {
        if (inner[i] == '%' && i + 2 < inner.size()) {
          int v = 0;
          auto hexv = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            return -1;
          };
          int h = hexv(inner[i + 1]), l = hexv(inner[i + 2]);
          if (h >= 0 && l >= 0) { v = h * 16 + l; decoded.push_back(static_cast<char>(v)); i += 2; }
          else decoded.push_back(inner[i]);
        } else {
          decoded.push_back(inner[i]);
        }
      }
      if (!decoded.empty()) url = decoded;
    }
    out.push_back(SearchResult{title, url, snippet, 0.0});
  }
  return out;
}

std::vector<SearchResult> parseBingHtml(const std::string &html) {
  std::vector<SearchResult> out;
  static const std::regex linkRe(
      R"bing(<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>([\s\S]*?)</a>)bing",
      std::regex::icase);
  std::sregex_iterator it(html.begin(), html.end(), linkRe), end;
  for (; it != end; ++it) {
    std::string url = it->str(1);
    std::string title = htmlToText(it->str(2));
    if (title.empty() || isSerpOrLoginUrl(url)) continue;
    std::string snippet;
    const size_t after = static_cast<size_t>(it->position() + it->length());
    const size_t cap = html.find("b_caption", after);
    if (cap != std::string::npos && cap < after + 2500) {
      const size_t p0 = html.find("<p", cap);
      if (p0 != std::string::npos && p0 < cap + 400) {
        const size_t gt = html.find('>', p0);
        const size_t p1 = html.find("</p>", gt == std::string::npos ? p0 : gt);
        if (gt != std::string::npos && p1 != std::string::npos)
          snippet = htmlToText(html.substr(gt + 1, p1 - gt - 1));
      }
    }
    if (snippet.size() > 600) snippet.resize(600);
    out.push_back(SearchResult{title, url, snippet, 0.0});
  }
  return out;
}

/* ------------------------- endpoint JSON ---------------------------- */

std::vector<SearchResult> parseEndpointJson(const nlohmann::json &raw) {
  std::vector<SearchResult> out;
  const nlohmann::json *arr = nullptr;
  if (raw.is_array()) {
    arr = &raw;
  } else if (raw.is_object() && raw.contains("results") && raw["results"].is_array()) {
    arr = &raw["results"];
  } else if (raw.is_object() && raw.contains("items") && raw["items"].is_array()) {
    arr = &raw["items"];
  }
  if (!arr) return out;
  for (const auto &item : *arr) {
    if (!item.is_object()) continue;
    auto gs = [&](const char *k) -> std::string {
      if (item.contains(k) && item[k].is_string()) return item[k].get<std::string>();
      return "";
    };
    SearchResult r;
    r.title = gs("title");
    r.url = gs("url");
    if (r.url.empty()) r.url = gs("link");
    r.snippet = gs("snippet");
    if (r.snippet.empty()) r.snippet = gs("description");
    if (r.snippet.empty()) r.snippet = gs("text");
    double score = 0.0;
    if (item.contains("score") && item["score"].is_number()) score = item["score"].get<double>();
    r.score = score;
    if (!r.url.empty()) out.push_back(r);
  }
  return out;
}

/* --------------------------- engine --------------------------- */

nlohmann::json WebSearchEngine::searchBackend(const std::string &backend,
                                              const std::string &query,
                                              const WebSearchConfig &cfg) const {
  if (backend == "bing") {
    const std::string q = urlEncode(query);
    const std::vector<std::string> urls = {
        "https://www.bing.com/search?q=" + q + "&setlang=en-US&mkt=en-US",
        "https://cn.bing.com/search?q=" + q + "&setlang=en&mkt=en-US",
    };
    json lastErr = json{{"ok", false}, {"backend", backend}, {"error", "http fetch failed"}};
    for (const auto &url : urls) {
      const std::string html =
          httpGetText(url, cfg.timeoutMs, cfg.maxRedirects, cfg.userAgent);
      if (html.empty()) continue;
      auto results = parseBingHtml(html);
      if (results.empty()) {
        lastErr["error"] = "no parseable hits";
        continue;
      }
      json arr = json::array();
      for (const auto &r : results) arr.push_back(r.toJson());
      return json{{"ok", true}, {"backend", backend}, {"results", arr}};
    }
    return lastErr;
  }
  if (backend == "ddg_lite") {
    const std::string url = "http://lite.duckduckgo.com/lite/?q=" + urlEncode(query);
    const std::string html = httpGetText(url, cfg.timeoutMs, cfg.maxRedirects, cfg.userAgent);
    if (html.empty()) {
      return json{{"ok", false}, {"backend", backend}, {"error", "http fetch failed"}};
    }
    auto results = parseDdgLiteHtml(html);
    json arr = json::array();
    for (const auto &r : results) arr.push_back(r.toJson());
    return json{{"ok", true}, {"backend", backend}, {"results", arr}};
  }
  if (backend == "endpoint") {
    if (cfg.endpoint.empty()) {
      return json{{"ok", false}, {"backend", backend}, {"error", "no endpoint configured"}};
    }
    std::string url = cfg.endpoint;
    url += url.find('?') == std::string::npos ? "?q=" : "&q=";
    url += urlEncode(query);
    const std::string body = httpGetText(url, cfg.timeoutMs, cfg.maxRedirects, cfg.userAgent);
    if (body.empty()) {
      return json{{"ok", false}, {"backend", backend}, {"error", "http fetch failed"}};
    }
    nlohmann::json raw;
    try {
      raw = nlohmann::json::parse(body);
    } catch (...) {
      return json{{"ok", false}, {"backend", backend},
                  {"error", "endpoint returned non-JSON"}, {"rawHead", body.substr(0, 200)}};
    }
    auto results = parseEndpointJson(raw);
    json arr = json::array();
    for (const auto &r : results) arr.push_back(r.toJson());
    return json{{"ok", true}, {"backend", backend}, {"results", arr}};
  }
  return json{{"ok", false}, {"backend", backend}, {"error", "unknown backend"}};
}

nlohmann::json WebSearchEngine::search(const std::string &query,
                                       const nlohmann::json &options) const {
  if (query.empty()) return json{{"ok", false}, {"error", "empty query"}};
  WebSearchConfig cfg = cfg_;
  if (options.contains("timeoutMs") && options["timeoutMs"].is_number())
    cfg.timeoutMs = options["timeoutMs"].get<int>();
  if (options.contains("maxResults") && options["maxResults"].is_number())
    cfg.maxResults = options["maxResults"].get<size_t>();
  if (options.contains("userAgent") && options["userAgent"].is_string())
    cfg.userAgent = options["userAgent"].get<std::string>();
  if (options.contains("endpoint") && options["endpoint"].is_string())
    cfg.endpoint = options["endpoint"].get<std::string>();
  if (options.contains("backends") && options["backends"].is_array()) {
    cfg.backends.clear();
    for (const auto &b : options["backends"])
      if (b.is_string()) cfg.backends.push_back(b.get<std::string>());
  }
  if (cfg.backends.empty()) {
    cfg.backends.push_back("bing");
    cfg.backends.push_back("ddg_lite");
  }
  if (!cfg.enabled) return json{{"ok", false}, {"error", "web search disabled"}};
  if (options.contains("fetchPages") && options["fetchPages"].is_number())
    cfg.fetchPages = options["fetchPages"].get<size_t>();

  std::vector<SearchResult> merged;
  std::vector<std::string> usedUrls;
  json sources = json::array();
  for (const auto &backend : cfg.backends) {
    json one = searchBackend(backend, query, cfg);
    sources.push_back(json{{"backend", backend},
                           {"ok", one.value("ok", false)},
                           {"error", one.value("error", std::string())}});
    if (!one.value("ok", false)) continue;
    for (const auto &item : one["results"]) {
      SearchResult r;
      r.title = item.value("title", "");
      r.url = item.value("url", "");
      r.snippet = item.value("snippet", "");
      r.score = item.value("score", 0.0);
      if (r.url.empty() || isSerpOrLoginUrl(r.url)) continue;
      if (std::find(usedUrls.begin(), usedUrls.end(), r.url) != usedUrls.end()) continue;
      usedUrls.push_back(r.url);
      merged.push_back(r);
      if (merged.size() >= cfg.maxResults) break;
    }
    if (merged.size() >= cfg.maxResults) break;
  }
  if (merged.empty()) {
    return json{{"ok", false}, {"query", query}, {"sources", sources},
                {"error", "no results from any backend"}};
  }
  const auto toks = queryWords(query);
  for (auto &r : merged)
    r.score = static_cast<double>(tokenOverlap(r.title + " " + r.snippet, toks));
  std::sort(merged.begin(), merged.end(),
            [](const SearchResult &a, const SearchResult &b) { return a.score > b.score; });
  if (!toks.empty()) {
    const double best = merged.front().score;
    if (best >= 2.0) {
      merged.erase(std::remove_if(merged.begin(), merged.end(),
                                  [](const SearchResult &r) { return r.score < 1.0; }),
                   merged.end());
    }
  }
  const size_t pages = std::min(cfg.fetchPages, merged.size());
  for (size_t i = 0; i < pages; ++i) {
    if (isSerpOrLoginUrl(merged[i].url)) continue;
    const std::string html =
        httpGetText(merged[i].url, cfg.timeoutMs, cfg.maxRedirects, cfg.userAgent);
    if (html.empty()) continue;
    std::string ex = pageExcerpt(html, query, 720);
    if (ex.size() < 80) continue;
    if (merged[i].snippet.size() < ex.size()) merged[i].snippet = std::move(ex);
    else if (ex.find(merged[i].snippet) == std::string::npos)
      merged[i].snippet += " " + ex;
    if (merged[i].snippet.size() > 900) merged[i].snippet.resize(900);
  }
  json results = json::array();
  for (const auto &r : merged) results.push_back(r.toJson());
  return json{{"ok", true}, {"query", query}, {"results", results},
              {"sources", sources}, {"count", merged.size()}};
}

} /* namespace websearch */
} /* namespace phoenix */
