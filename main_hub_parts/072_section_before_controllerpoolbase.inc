// Auto-generated from main.cpp
// Segment: section_before_ControllerPoolBase
// Original lines: 9527-9881
// Disabled due to pre-existing compilation errors unrelated to LoRA integration
#if 0
#endif

json OnlineResearcher::lookup(const json &input, const json &options) {
  std::string rawText;
  std::vector<std::string> words;
  std::function<std::string(const json &)> jsStringForJoin;
  jsStringForJoin = [&](const json &v) -> std::string {
    if (v.is_null())
      return "";
    if (v.is_string())
      return v.get<std::string>();
    if (v.is_boolean())
      return v.get<bool>() ? "true" : "false";
    if (v.is_number())
      return v.dump();
    if (v.is_array()) {
      std::string out;
      bool first = true;
      for (auto &item : v) {
        if (!first)
          out += ",";
        first = false;
        out += jsStringForJoin(item);
      }
      return out;
    }
    return "[object Object]";
  };
  auto jsStringOrEmpty = [&](const json &v) -> std::string {
    if (v.is_null())
      return "";
    if (v.is_boolean())
      return v.get<bool>() ? "true" : "";
    if (v.is_number()) {
      double n = v.get<double>();
      if (n == 0.0)
        return "";
      return v.dump();
    }
    if (v.is_string())
      return v.get<std::string>();
    if (v.is_array()) {
      std::string out;
      bool first = true;
      for (auto &item : v) {
        if (!first)
          out += ",";
        first = false;
        out += jsStringForJoin(item);
      }
      return out;
    }
    return "[object Object]";
  };

  if (input.is_array()) {
    for (auto &t : input) {
      words.push_back(jsStringForJoin(t));
    }
    for (size_t i = 0; i < words.size(); i++) {
      if (i)
        rawText += " ";
      rawText += words[i];
    }
  } else {
    rawText = jsStringOrEmpty(input);
    words = tokenize(rawText);
  }

  bool useIndex = options.value("useIndex", false) ||
                  options.value("mode", "search") == "aspect";
  if (useIndex) {
    auto idx = queryLocalIndex(words, options.value("limitUrls", 12),
                               options.value("limitAspects", 10),
                               options.value("kClusters", 4));
    if (options.value("preferIndex", true) ||
        options.value("mode", "search") == "aspect") {
      return idx;
    }
  }

  if (!enabled_ || options.value("disableRemote", false)) {
    return fallback(words);
  }

  auto crawlReq = options.value("crawl", json::object());
  std::string urlFromInput = extractFirstUrl(rawText);
  bool shouldCrawl =
      options.value("mode", "search") == "crawl" ||
      options.value("mode", "search") == "seed" || (!urlFromInput.empty()) ||
      (crawlReq.contains("startUrl") && crawlReq["startUrl"].is_string());
  if (shouldCrawl) {
    std::string startUrl = crawlReq.value("startUrl", urlFromInput);
    if (startUrl.empty() && options.value("mode", "search") != "seed") {
      return json{
          {"ok", false}, {"source", "crawl"}, {"reason", "startUrl-required"}};
    }
    auto payload = crawl(startUrl, crawlReq, options);
    if (!options.value("forceRemote", false)) {
      std::string cacheSeed = startUrl.empty() ? "seed" : startUrl;
      auto key = normalizeKey(tokenize("crawl " + cacheSeed));
      remember(key, payload);
    }
    return payload;
  }

  if (words.empty()) {
    return json{{"ok", false}, {"source", "local"}, {"reason", "empty-query"}};
  }
  std::string key = normalizeKey(words);
  if (!options.value("forceRemote", false)) {
    auto cached = fromCache(key);
    if (cached) {
      json payload = *cached;
      payload["cached"] = true;
      return payload;
    }
  }

  json payload;
  bool remoteOk = false;
#ifdef HAVE_CURL
  if (!endpoint_.empty() && !options.value("skipRemote", false)) {
    std::string url = endpoint_;
    if (url.find('?') == std::string::npos)
      url += "?q=" + urlEncode(rawText);
    else
      url += "&q=" + urlEncode(rawText);
    int timeoutMs = options.value("timeout", 5000);
    auto res = fetchUrl(url, timeoutMs);
    if (!res.body.empty()) {
      json raw;
      try {
        raw = json::parse(res.body);
      } catch (...) {
        raw = json{{"raw", res.body}};
      }
      std::string snippet;
      if (raw.is_string())
        snippet = raw.get<std::string>().substr(0, 280);
      else
        snippet = raw.dump().substr(0, 280);
      payload = json{{"ok", true},
                     {"source", "remote"},
                     {"query", rawText},
                     {"snippet", snippet}};
      if (options.value("includeRaw", false))
        payload["raw"] = raw;
      remoteOk = true;
    }
  }
#endif
  if (!remoteOk) {
    payload = fallback(words);
  }

  remember(key, payload);
  return payload;
}

// ------------------ Controller / Pool ------------------

class Controller;

// ControllerPoolBase 定义控制器池抽象接口。
// 调用方式：网关与学习模块依赖该接口屏蔽实现差异。
// 实现思路：统一控制器发现、调用、指标与运维操作。
// 注意事项：不同实现的时延与一致性语义可能不同。
// 注意事项：调用方应处理空控制器与异常返回。
// 注意事项：该接口不约束具体线程模型。
