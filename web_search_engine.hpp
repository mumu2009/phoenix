/* web_search_engine.hpp - Dependency-free online search engine.
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

   Purpose: give the evolving agent a *material source* (web search) that
   works out of the box with no API key and no external SDK:
    - built-in backend "ddg_lite": DuckDuckGo Lite HTML over plain HTTP,
      parsed into structured {title,url,snippet} results;
    - backend "endpoint": a configurable search HTTP endpoint returning JSON
      (the shape the gateway's OnlineResearcher already uses), so a private
      search API can replace the public backend.

   Transport: raw-socket HTTP/1.1 GET (Winsock on Windows, POSIX elsewhere)
   with connect/read timeouts and up to 5 redirects.  HTTPS is intentionally
   NOT attempted here: the gateway's existing HAVE_CURL path covers HTTPS
   endpoints.  Callers should prefer HTTPS-capable paths when configured.

   All of this is stateless and thread-safe; results are deduplicated by URL
   and capped.  No pseudoscience: this is a plain HTTP client + HTML parser.
*/
#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace phoenix {
namespace websearch {

struct SearchResult {
  std::string title;
  std::string url;
  std::string snippet;
  double score{0.0};
  nlohmann::json toJson() const;
};

struct WebSearchConfig {
  bool enabled{true};
  int timeoutMs{5000};
  size_t maxResults{8};
  int maxRedirects{5};
  std::string userAgent{"PhoenixWebSearch/1.0"};
  std::vector<std::string> backends; /* "ddg_lite" and/or "endpoint" */
  std::string endpoint;              /* JSON endpoint for backend "endpoint" */
};

/* Raw HTTP GET with timeouts + redirects.  Returns body on 2xx, empty on any
   failure.  Plain HTTP only. */
std::string httpGetText(const std::string &url, int timeoutMs, int maxRedirects,
                        const std::string &userAgent);

/* Parse DuckDuckGo Lite HTML into results (title/url/snippet). */
std::vector<SearchResult> parseDdgLiteHtml(const std::string &html);

/* Parse a JSON search-endpoint response into results.
   Accepts {"results":[{title,url,snippet,...}]} or a top-level array. */
std::vector<SearchResult> parseEndpointJson(const nlohmann::json &raw);

/* URL-encode (RFC 3986, UTF-8 preserved). */
std::string urlEncode(const std::string &s);

/* HTML entity decode + strip tags (for snippets). */
std::string htmlToText(const std::string &s);

class WebSearchEngine {
public:
  explicit WebSearchEngine(WebSearchConfig cfg) : cfg_(std::move(cfg)) {}
  const WebSearchConfig &config() const { return cfg_; }
  void setConfig(WebSearchConfig cfg) { cfg_ = std::move(cfg); }

  /* search(query, options):
     options.timeoutMs / maxResults / backends / endpoint / userAgent override
     the config for one call.  Returns:
       {"ok":true, "query":..., "results":[{title,url,snippet,score}],
        "sources":[...]}
       {"ok":false, "query":..., "error":...} */
  nlohmann::json search(const std::string &query,
                        const nlohmann::json &options) const;

  /* Search a single backend; used by search() and directly by tests. */
  nlohmann::json searchBackend(const std::string &backend,
                               const std::string &query,
                               const WebSearchConfig &cfg) const;

private:
  WebSearchConfig cfg_;
};

} /* namespace websearch */
} /* namespace phoenix */
