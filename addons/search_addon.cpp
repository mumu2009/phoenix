/* search_addon.cpp - Search addon: web material source for the evolving agent.
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

   Enhanced search pipeline:
    1. gateway handler first (index-aware OnlineResearcher when available);
    2. otherwise (or when the handler reports failure) the dependency-free
       WebSearchEngine (phoenix::websearch) with backends "ddg_lite" and/or
       "endpoint" from config search.* / payload searchOptions;
    3. structured multi-result reply (top hits with title/url/snippet) so the
       agent gets real material, not one flat string.

   Config (config/phoenix.json):
     "search": { "enabled": true, "backends": ["ddg_lite"], "endpoint": "",
                 "timeoutMs": 5000, "maxResults": 8, "userAgent": "..." }
*/
#include "search_addon.hpp"

#include "../phoenix_config.hpp"
#include "../web_search_engine.hpp"

#include <algorithm>
#include <cctype>
#include <vector>

namespace addon::builtins {

namespace {

std::string trimCopy(const std::string &s) {
  auto start = s.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) return "";
  auto end = s.find_last_not_of(" \t\r\n");
  return s.substr(start, end - start + 1);
}

std::string lowerCopy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return s;
}

std::string stripPrefix(const std::string &text) {
  std::string t = trimCopy(text);
  std::string l = lowerCopy(t);
  const std::vector<std::string> prefixes = {
      "search:", "web:", "lookup:", "research:", "搜索:", "检索:", "查询:"
  };
  for (const auto &p : prefixes) {
    if (l.rfind(p, 0) == 0) return trimCopy(t.substr(p.size()));
  }
  return t;
}

phoenix::websearch::WebSearchConfig engineConfig(const json &options) {
  phoenix::websearch::WebSearchConfig cfg;
  cfg.enabled = phoenix::cfgOr<bool>("search.enabled", true);
  cfg.timeoutMs = phoenix::cfgOr<int>("search.timeoutMs", 5000);
  cfg.maxResults = static_cast<size_t>(phoenix::cfgOr<int>("search.maxResults", 8));
  cfg.userAgent = phoenix::cfgOr<std::string>("search.userAgent", "PhoenixWebSearch/1.0");
  cfg.endpoint = phoenix::cfgOr<std::string>("search.endpoint", std::string());
  {
    nlohmann::json bs = phoenix::cfgOr<nlohmann::json>("search.backends", nlohmann::json::array());
    if (bs.is_array() && !bs.empty()) {
      for (const auto &b : bs) if (b.is_string()) cfg.backends.push_back(b.get<std::string>());
    }
  }
  /* per-request overrides */
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
    for (const auto &b : options["backends"]) if (b.is_string()) cfg.backends.push_back(b.get<std::string>());
  }
  if (cfg.backends.empty()) cfg.backends.push_back("ddg_lite");
  return cfg;
}

std::string formatResults(const nlohmann::json &results, size_t maxItems, size_t maxChars) {
  std::string reply;
  size_t n = 0;
  for (const auto &r : results) {
    if (n >= maxItems) break;
    ++n;
    if (n > 1) reply += "\n";
    reply += std::to_string(n) + ". " + r.value("title", std::string());
    if (!r.value("url", std::string()).empty()) reply += "\n   " + r.value("url", std::string());
    std::string snippet = r.value("snippet", std::string());
    if (!snippet.empty()) {
      if (snippet.size() > 240) snippet.resize(240);
      reply += "\n   " + snippet;
    }
    if (reply.size() > maxChars) break;
  }
  if (reply.size() > maxChars) reply.resize(maxChars);
  return reply;
}

class SearchAddon : public Addon {
public:
  explicit SearchAddon(std::string name) : name_(std::move(name)) {}
  std::string name() const override { return name_; }
  std::string type() const override { return "search"; }

  AddonResult handle(const std::string &text, const json &payload) override {
    AddonResult res;
    std::string addonType = lowerCopy(trimCopy(payload.value("__addonType", std::string())));
    bool explicitSearch = (addonType == "search" || addonType == "research" || addonType == "web");
    if (!explicitSearch && !addonType.empty()) return res;

    std::string query = stripPrefix(text);
    if (query.empty()) return res;

    json options = payload.contains("searchOptions") && payload["searchOptions"].is_object()
        ? payload["searchOptions"]
        : json::object();
    if (!options.contains("preferIndex")) options["preferIndex"] = true;

    nlohmann::json results;
    std::string source = "local";
    bool got = false;

    /* 1) gateway handler first (index-aware OnlineResearcher). */
    bool handlerOk = false;
    json lookup;
    handlerOk = addon::invokeAddonOnlineLookup(json(query), options, lookup);
    if (handlerOk && !lookup.is_null()) {
      /* handler may return snippet/suggestions (legacy) or results array */
      if (lookup.contains("results") && lookup["results"].is_array()) {
        results = lookup["results"];
        source = lookup.value("source", "remote");
        got = true;
      } else if (lookup.value("ok", false) && !lookup.value("cached", false)) {
        source = lookup.value("source", "remote");
      }
      if (!got) {
        /* legacy flat forms: convert to a one-item result if text present */
        std::string snippet;
        if (lookup.contains("snippet") && lookup["snippet"].is_string())
          snippet = lookup["snippet"].get<std::string>();
        else if (lookup.contains("text") && lookup["text"].is_string())
          snippet = lookup["text"].get<std::string>();
        else if (lookup.contains("suggestions") && lookup["suggestions"].is_array() &&
                 !lookup["suggestions"].empty()) {
          const auto &first = lookup["suggestions"][0];
          if (first.is_object() && first.contains("words") && first["words"].is_array()) {
            for (const auto &w : first["words"]) {
              if (!snippet.empty()) snippet += " ";
              snippet += w.is_string() ? w.get<std::string>() : w.dump();
            }
          }
        }
        if (!snippet.empty()) {
          results = json::array({json{{"title", query}, {"url", ""}, {"snippet", snippet}}});
          source = lookup.value("source", "local");
          got = true;
        }
      }
    }

    /* 2) dependency-free web engine fallback (ddg_lite / endpoint). */
    if (!got && options.value("allowWeb", true)) {
      const phoenix::websearch::WebSearchEngine engine(engineConfig(options));
      const json out = engine.search(query, options);
      if (out.value("ok", false) && out.contains("results") && out["results"].is_array() &&
          !out["results"].empty()) {
        results = out["results"];
        source = "web:" + out.value("sources", json::array()).dump();
        got = true;
      }
    }

    if (!got || !results.is_array() || results.empty()) return res;

    res.handled = true;
    const size_t replyItems = static_cast<size_t>(options.value("replyItems", 3));
    const size_t replyChars = static_cast<size_t>(options.value("replyChars", 1200));
    res.reply = formatResults(results, replyItems, replyChars);
    res.meta = json{{"addon", "search"}, {"name", name_}, {"query", query},
                    {"source", source}, {"results", results}};
    return res;
  }

private:
  std::string name_;
};

} /* namespace */

std::shared_ptr<Addon> createSearchAddon(const std::string &name) {
  const std::string addonName = name.empty() ? std::string("search") : name;
  return std::make_shared<SearchAddon>(addonName);
}

} /* namespace addon::builtins */
