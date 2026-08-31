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
    1. live web (Bing first; DDG only if that network can reach it);
    2. local inverted index of previously ingested pages (not the assignment
       restated, not wikitext seed when the web already answered);
    3. assignment paragraphs only as last-resort fallback.

   Config (config/phoenix.json):
     "search": { "enabled": true, "backends": ["bing", "ddg_lite"], "endpoint": "",
                 "timeoutMs": 8000, "maxResults": 8, "fetchPages": 2 }
*/
#include "search_addon.hpp"

#include "../mission_reply_parse.hpp"
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
  cfg.timeoutMs = phoenix::cfgOr<int>("search.timeoutMs", 8000);
  cfg.maxResults = static_cast<size_t>(phoenix::cfgOr<int>("search.maxResults", 8));
  cfg.fetchPages = static_cast<size_t>(phoenix::cfgOr<int>("search.fetchPages", 2));
  cfg.userAgent = phoenix::cfgOr<std::string>(
      "search.userAgent",
      "Mozilla/5.0 (compatible; PhoenixWebSearch/1.0)");
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
  if (cfg.backends.empty()) {
    cfg.backends.push_back("bing");
    cfg.backends.push_back("ddg_lite");
  }
  if (options.contains("fetchPages") && options["fetchPages"].is_number())
    cfg.fetchPages = options["fetchPages"].get<size_t>();
  return cfg;
}

std::vector<std::string> queryTokens(const std::string &q) {
  std::vector<std::string> toks;
  std::string cur;
  for (unsigned char c : q) {
    if (std::isalnum(c) || c == '-')
      cur.push_back(static_cast<char>(std::tolower(c)));
    else if (!cur.empty()) {
      if (cur.size() >= 3) toks.push_back(cur);
      cur.clear();
    }
  }
  if (cur.size() >= 3) toks.push_back(cur);
  return toks;
}

int overlapScore(const std::string &para, const std::vector<std::string> &toks) {
  const std::string lower = lowerCopy(para);
  int n = 0;
  for (const auto &t : toks) {
    if (lower.find(t) != std::string::npos) ++n;
  }
  return n;
}

json scoreLocalParagraphs(const std::string &text, const std::string &query,
                          size_t maxHits) {
  json arr = json::array();
  auto toks = queryTokens(query);
  if (text.empty() || toks.empty()) return arr;
  struct Hit {
    int score;
    std::string text;
  };
  std::vector<Hit> hits;
  size_t i = 0;
  while (i < text.size()) {
    size_t eol = text.find('\n', i);
    if (eol == std::string::npos) eol = text.size();
    std::string line = trimCopy(text.substr(i, eol - i));
    i = (eol == text.size()) ? text.size() : eol + 1;
    if (line.size() < 40) continue;
    std::string heading = line;
    while (!heading.empty() && heading[0] == '#') {
      heading.erase(heading.begin());
      heading = trimCopy(heading);
    }
    if (heading.rfind("Chapter ", 0) == 0 && heading.size() > 8 &&
        std::isdigit(static_cast<unsigned char>(heading[8])))
      continue;
    const int s = overlapScore(line, toks);
    if (s >= 1) hits.push_back({s, std::move(line)});
  }
  std::sort(hits.begin(), hits.end(),
            [](const Hit &a, const Hit &b) { return a.score > b.score; });
  for (size_t n = 0; n < hits.size() && n < maxHits; ++n) {
    if (n > 0 && hits[n].score < 2 && hits[0].score >= 2) break;
    std::string snip = hits[n].text;
    if (snip.size() > 360) snip.resize(360);
    arr.push_back(json{{"title", "assigned-goal"},
                       {"url", "mission://goal"},
                       {"snippet", snip}});
  }
  return arr;
}

void appendLookupHits(const json &lookup, json &results) {
  auto push = [&](const json &item) {
    if (!item.is_object()) return;
    std::string snippet = item.value("snippet", std::string());
    if (snippet.empty()) snippet = item.value("text", std::string());
    if (snippet.empty()) return;
    results.push_back(json{{"title", item.value("title", std::string())},
                           {"url", item.value("url", std::string())},
                           {"snippet", snippet}});
  };
  if (lookup.contains("results") && lookup["results"].is_array()) {
    for (const auto &r : lookup["results"]) push(r);
    return;
  }
  if (lookup.contains("urls") && lookup["urls"].is_array()) {
    for (const auto &u : lookup["urls"]) push(u);
  }
}

std::string formatResults(const nlohmann::json &results, size_t maxItems, size_t maxChars) {
  std::string reply;
  size_t n = 0;
  for (const auto &r : results) {
    if (n >= maxItems) break;
    ++n;
    if (n > 1) reply += "\n\n";
    reply += std::to_string(n) + ". " + r.value("title", std::string());
    if (!r.value("url", std::string()).empty()) reply += "\n   " + r.value("url", std::string());
    std::string snippet = r.value("snippet", std::string());
    if (!snippet.empty()) {
      if (snippet.size() > 480) snippet.resize(480);
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

  float consider(const json &situation) const override {
    const std::string goal = situation.value(
        "goal", situation.value("text", std::string()));
    if (goal.size() < 40) return 0.f;
    if (queryTokens(goal).size() < 4) return 0.f;
    const std::string phase = situation.value("phase", std::string());
    if (phase == "mission-deliberate") return 0.86f;
    return 0.62f;
  }

  AddonResult contribute(const json &situation) override {
    const std::string goal = situation.value(
        "goal", situation.value("text", std::string()));
    const std::string draft = situation.value("draft", std::string());
    const auto queries =
        phoenix::mission::buildPluginSearchQueries(goal, draft, 3);
    json payload = situation.is_object() ? situation : json::object();
    payload["__addonType"] = "search";
    if (!payload.contains("missionGoal") ||
        !payload["missionGoal"].is_string() ||
        payload["missionGoal"].get<std::string>().empty())
      payload["missionGoal"] =
          phoenix::mission::extractGoalWorkingContext(goal, 8000);
    json options = payload.contains("searchOptions") &&
                           payload["searchOptions"].is_object()
                       ? payload["searchOptions"]
                       : json::object();
    if (!options.contains("allowWeb"))
      options["allowWeb"] = situation.value("allowWeb", true);
    if (!options.contains("useIndex")) options["useIndex"] = true;
    if (!options.contains("preferIndex")) options["preferIndex"] = false;
    if (!options.contains("localText") ||
        !options["localText"].is_string() ||
        options["localText"].get<std::string>().empty())
      options["localText"] = payload["missionGoal"];
    if (!options.contains("fetchPages"))
      options["fetchPages"] = phoenix::cfgOr<int>("search.fetchPages", 2);
    if (!options.contains("replyItems")) options["replyItems"] = 5;
    if (!options.contains("replyChars")) options["replyChars"] = 4800;
    payload["searchOptions"] = options;

    AddonResult combined;
    json allHits = json::array();
    std::string srcAll;
    std::string replyAll;
    for (const auto &q : queries) {
      auto one = handle("search: " + q, payload);
      if (!one.handled) continue;
      if (!replyAll.empty()) replyAll += "\n\n";
      replyAll += one.reply;
      const std::string src = one.meta.value("source", std::string());
      if (!src.empty() && srcAll.find(src) == std::string::npos) {
        if (!srcAll.empty()) srcAll += "+";
        srcAll += src;
      }
      if (one.meta.contains("results") && one.meta["results"].is_array()) {
        for (const auto &hit : one.meta["results"]) {
          if (hit.is_object()) allHits.push_back(hit);
        }
      }
    }
    json kept = phoenix::mission::keepSearchHitsAlignedToGoal(allHits, goal, 6);
    if (kept.empty()) {
      const std::string work =
          phoenix::mission::extractGoalWorkingContext(goal, 8000);
      const std::string local = work.empty() ? goal : work;
      kept = phoenix::mission::keepSearchHitsAlignedToGoal(
          scoreLocalParagraphs(local, goal, 4), goal, 4);
      if (!kept.empty()) {
        if (!srcAll.empty()) srcAll += "+";
        srcAll += "goal";
      }
    }
    if (kept.empty()) return combined;
    combined.handled = true;
    combined.reply = formatResults(kept, 5, 4800);
    combined.units = phoenix::mission::searchHitsToUnitQueries(kept);
    combined.meta = json{{"addon", "search"},
                         {"name", name_},
                         {"source", srcAll},
                         {"queries", queries},
                         {"results", kept},
                         {"reason", "goal has enough lexical content to research"}};
    (void)replyAll;
    return combined;
  }

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
    const bool allowWeb = options.value("allowWeb", true);
    if (!options.contains("preferIndex")) options["preferIndex"] = !allowWeb;
    if (!options.contains("useIndex"))
      options["useIndex"] = options.value("preferIndex", !allowWeb);
    if (!options.contains("fetchPages"))
      options["fetchPages"] = phoenix::cfgOr<int>("search.fetchPages", 2);

    nlohmann::json results = json::array();
    std::string source;
    bool got = false;

    auto pushHits = [&](const json &hits) {
      if (!hits.is_array()) return;
      for (const auto &h : hits) {
        if (!h.is_object()) continue;
        const std::string url = h.value("url", std::string());
        const std::string snippet = h.value("snippet", h.value("text", std::string()));
        if (snippet.empty()) continue;
        results.push_back(json{{"title", h.value("title", std::string())},
                               {"url", url},
                               {"snippet", snippet}});
      }
    };

    /* Live web first: assignment text is not discovered knowledge. */
    if (allowWeb) {
      const phoenix::websearch::WebSearchEngine engine(engineConfig(options));
      const json out = engine.search(query, options);
      if (out.value("ok", false) && out.contains("results") &&
          out["results"].is_array() && !out["results"].empty()) {
        pushHits(out["results"]);
        source = "web";
        got = !results.empty();
      }
    }

    /* Previously ingested pages. Skip corpus/assignment echoes once web hit. */
    json lookup;
    const bool handlerOk =
        addon::invokeAddonOnlineLookup(json(query), options, lookup);
    if (handlerOk && !lookup.is_null()) {
      json indexHits = json::array();
      appendLookupHits(lookup, indexHits);
      json kept = json::array();
      for (const auto &h : indexHits) {
        if (!h.is_object()) continue;
        const std::string url = h.value("url", std::string());
        if (got && (url.rfind("corpus://", 0) == 0 ||
                    url.rfind("mission://", 0) == 0))
          continue;
        kept.push_back(h);
      }
      if (!kept.empty()) {
        pushHits(kept);
        source = source.empty() ? lookup.value("source", "index")
                                : (source + "+" + lookup.value("source", "index"));
        got = true;
      } else if (!got) {
        std::string snippet;
        if (lookup.contains("snippet") && lookup["snippet"].is_string())
          snippet = lookup["snippet"].get<std::string>();
        else if (lookup.contains("text") && lookup["text"].is_string())
          snippet = lookup["text"].get<std::string>();
        if (!snippet.empty()) {
          results.push_back(
              json{{"title", query}, {"url", ""}, {"snippet", snippet}});
          source = lookup.value("source", "local");
          got = true;
        }
      }
    }

    std::string localText;
    if (options.contains("localText") && options["localText"].is_string())
      localText = options["localText"].get<std::string>();
    if (localText.empty() && payload.contains("missionGoal") &&
        payload["missionGoal"].is_string())
      localText = payload["missionGoal"].get<std::string>();
    if (!got && !localText.empty()) {
      const json localHits = scoreLocalParagraphs(localText, query, 3);
      if (localHits.is_array() && !localHits.empty()) {
        pushHits(localHits);
        source = "goal";
        got = !results.empty();
      }
    }

    if (!got || !results.is_array() || results.empty()) return res;

    const std::string alignGoal = payload.value(
        "missionGoal", payload.value("goal", std::string()));
    if (!alignGoal.empty()) {
      json aligned = phoenix::mission::keepSearchHitsAlignedToGoal(
          results, alignGoal, options.value("replyItems", 5));
      if (aligned.empty() && !localText.empty()) {
        aligned = phoenix::mission::keepSearchHitsAlignedToGoal(
            scoreLocalParagraphs(localText, query, 4), alignGoal, 4);
        if (!aligned.empty())
          source = source.empty() ? std::string("goal") : (source + "+goal");
      }
      results = std::move(aligned);
    }
    if (!results.is_array() || results.empty()) return res;

    res.handled = true;
    const size_t replyItems = static_cast<size_t>(options.value("replyItems", 5));
    const size_t replyChars = static_cast<size_t>(options.value("replyChars", 2400));
    res.reply = formatResults(results, replyItems, replyChars);
    res.units = phoenix::mission::searchHitsToUnitQueries(results);
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
