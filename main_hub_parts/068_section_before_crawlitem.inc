// Auto-generated from main.cpp
// Segment: section_before_CrawlItem
// Original lines: 8832-9330


std::string
OnlineResearcher::normalizeKey(const std::vector<std::string> &words) const {
  std::ostringstream oss;
  int limit = std::min<int>(32, (int)words.size());
  for (int i = 0; i < limit; i++) {
    if (i)
      oss << " ";
    std::string w = words[i];
    for (auto &c : w)
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    oss << w;
  }
  return oss.str();
}

void OnlineResearcher::pruneCache() {
  if ((int)cache_.size() <= cacheSize_)
    return;
  std::vector<std::pair<std::string, CacheEntry>> items(cache_.begin(),
                                                        cache_.end());
  std::sort(items.begin(), items.end(),
            [](auto &a, auto &b) { return a.second.ts < b.second.ts; });
  while ((int)items.size() > cacheSize_) {
    cache_.erase(items.front().first);
    items.erase(items.begin());
  }
}

void OnlineResearcher::remember(const std::string &key, const json &payload) {
  cache_[key] =
      CacheEntry{(int64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::system_clock::now().time_since_epoch())
                     .count(),
                 payload};
  pruneCache();
}

std::optional<json> OnlineResearcher::fromCache(const std::string &key) {
  auto it = cache_.find(key);
  if (it == cache_.end())
    return std::nullopt;
  int64_t nowMs =
      (int64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();
  if (nowMs - it->second.ts > cooldownMs_) {
    cache_.erase(it);
    return std::nullopt;
  }
  return it->second.payload;
}

std::vector<std::string> OnlineResearcher::loadSeedUrls(int maxSeeds) const {
  std::vector<std::string> out;
  fs::path seeds = runtime_ ? (runtime_->config().robotsDir / "seeds.txt")
                            : (getProjectRoot() / "robots" / "seeds.txt");
  if (!fs::exists(seeds))
    return out;
  std::ifstream in(seeds);
  if (!in)
    return out;
  std::string line;
  while (std::getline(in, line)) {
    line = trimCopy(line);
    if (line.empty())
      continue;
    if (line[0] == '#')
      continue;
    out.push_back(line);
    if (maxSeeds > 0 && (int)out.size() >= maxSeeds)
      break;
  }
  return out;
}

std::vector<std::string>
OnlineResearcher::extractLinksFromHtml(const std::string &html,
                                       const std::string &baseUrl) const {
  std::vector<std::string> links;
  if (html.empty())
    return links;
  std::unordered_set<std::string> uniq;
  auto makeAbsolute = [&](const std::string &link) -> std::string {
    if (link.empty())
      return {};
    if (link.rfind("http://", 0) == 0 || link.rfind("https://", 0) == 0)
      return link;
    if (link.rfind("//", 0) == 0)
      return std::string("https:") + link;
    if (baseUrl.empty())
      return link;
    auto schemePos = baseUrl.find("://");
    auto pathPos = schemePos == std::string::npos
                       ? std::string::npos
                       : baseUrl.find('/', schemePos + 3);
    std::string root =
        (pathPos == std::string::npos) ? baseUrl : baseUrl.substr(0, pathPos);
    if (!link.empty() && link[0] == '/')
      return root + link;
    std::string baseDir = baseUrl;
    auto lastSlash = baseDir.find_last_of('/');
    if (lastSlash != std::string::npos)
      baseDir = baseDir.substr(0, lastSlash + 1);
    return baseDir + link;
  };
#ifdef HAVE_GUMBO
  GumboOutput *output = gumbo_parse(html.c_str());
  if (output) {
    std::function<void(GumboNode *)> walk = [&](GumboNode *node) {
      if (!node)
        return;
      if (node->type == GUMBO_NODE_ELEMENT) {
        GumboElement &el = node->v.element;
        auto addAttr = [&](const char *attrName) {
          GumboAttribute *attr = gumbo_get_attribute(&el.attributes, attrName);
          if (!attr || !attr->value)
            return;
          std::string link = attr->value;
          if (link.rfind("javascript:", 0) == 0 ||
              link.rfind("mailto:", 0) == 0)
            return;
          link = makeAbsolute(link);
          link = normalizeUrlSimple(link);
          if (link.empty())
            return;
          if (uniq.insert(link).second)
            links.push_back(link);
        };
        addAttr("href");
        addAttr("src");
        GumboVector *children = &el.children;
        for (size_t i = 0; i < children->length; i++) {
          walk(static_cast<GumboNode *>(children->data[i]));
        }
      } else if (node->type == GUMBO_NODE_DOCUMENT) {
        GumboVector *children = &node->v.document.children;
        for (size_t i = 0; i < children->length; i++) {
          walk(static_cast<GumboNode *>(children->data[i]));
        }
      }
    };
    walk(output->root);
    gumbo_destroy_output(&kGumboDefaultOptions, output);
  }
#else
  std::regex hrefRe(R"(\b(?:href|src)\s*=\s*['"]([^'"#]+)['"])",
                    std::regex::icase);
  std::smatch m;
  std::string::const_iterator searchStart(html.cbegin());
  while (std::regex_search(searchStart, html.cend(), m, hrefRe)) {
    std::string link = m[1].str();
    if (link.rfind("javascript:", 0) == 0 || link.rfind("mailto:", 0) == 0) {
      searchStart = m.suffix().first;
      continue;
    }
    link = makeAbsolute(link);
    link = normalizeUrlSimple(link);
    if (!link.empty() && uniq.insert(link).second)
      links.push_back(link);
    searchStart = m.suffix().first;
  }
#endif
  return links;
}

std::string OnlineResearcher::cleanTextAdvanced(const std::string &text) const {
  if (text.empty())
    return {};
  auto decodeEntities = [](const std::string &s) {
    std::string out = s;
    auto replaceAll = [&](const std::string &from, const std::string &to) {
      size_t pos = 0;
      while ((pos = out.find(from, pos)) != std::string::npos) {
        out.replace(pos, from.size(), to);
        pos += to.size();
      }
    };
    replaceAll("&nbsp;", " ");
    replaceAll("&amp;", "&");
    replaceAll("&lt;", "<");
    replaceAll("&gt;", ">");
    replaceAll("&quot;", "\"");
    replaceAll("&#39;", "'");
    return out;
  };
  std::string raw = decodeEntities(text);
  std::string cleaned;
  cleaned.reserve(raw.size());
  for (char c : raw) {
    unsigned char uc = static_cast<unsigned char>(c);
    if (uc < 0x09)
      continue;
    if (uc == 0x0b || uc == 0x0c)
      continue;
    cleaned.push_back(c);
  }
  std::unordered_set<std::string> seenLines;
  std::ostringstream oss;
  std::istringstream iss(cleaned);
  std::string line;
  auto alphaRatio = [](const std::string &s) {
    if (s.empty())
      return 0.0;
    int alpha = 0;
    int total = 0;
    for (unsigned char c : s) {
      if (std::isalpha(c))
        alpha++;
      if (!std::isspace(c))
        total++;
    }
    if (total == 0)
      return 0.0;
    return (double)alpha / (double)total;
  };
  while (std::getline(iss, line)) {
    line = trimCopy(line);
    if (line.empty())
      continue;
    if (line.size() < 15 && alphaRatio(line) < 0.4)
      continue;
    if (line.size() > 600)
      line = line.substr(0, 600);
    if (!seenLines.insert(line).second)
      continue;
    if (!oss.str().empty())
      oss << "\n";
    oss << line;
    if ((int)oss.str().size() > 50000)
      break;
  }
  std::string out = oss.str();
  if (out.size() > 50000)
    out.resize(50000);
  return out;
}

std::vector<std::string>
OnlineResearcher::pickAspects(const std::vector<std::string> &tokens,
                              int maxAspects) const {
  std::unordered_map<std::string, int> freq;
  auto &stop = getStopWords();
  for (const auto &t : tokens) {
    if (t.size() < 2)
      continue;
    if (stop.count(t))
      continue;
    freq[t] += 1;
  }
  std::vector<std::pair<std::string, int>> ranked(freq.begin(), freq.end());
  std::sort(ranked.begin(), ranked.end(),
            [](auto &a, auto &b) { return a.second > b.second; });
  std::vector<std::string> out;
  for (auto &p : ranked) {
    out.push_back(p.first);
    if ((int)out.size() >= maxAspects)
      break;
  }
  return out;
}

void OnlineResearcher::indexDocument(const DocEntry &doc) {
  if (doc.url.empty())
    return;
  std::lock_guard<std::mutex> lock(indexMu_);
  docs_[doc.url] = doc;
  urlToContent_[doc.url] = doc.text;
  auto &stop = getStopWords();
  for (const auto &t : doc.tokens) {
    if (t.size() < 2)
      continue;
    if (stop.count(t))
      continue;
    invertedIndex_[t].insert(doc.url);
  }
  for (const auto &a : doc.aspects) {
    if (a.empty())
      continue;
    aspectToUrls_[a].insert(doc.url);
  }
}

json OnlineResearcher::queryLocalIndex(const std::vector<std::string> &words,
                                       int limitUrls, int limitAspects,
                                       int maxClusters) {
  json out{{"ok", true}, {"source", "index"}, {"query", ""}};
  if (words.empty())
    return json{{"ok", false}, {"source", "index"}, {"reason", "empty-query"}};
  std::unordered_map<std::string, int> scores;
  std::vector<std::string> queryTokens = words;
  std::ostringstream oss;
  for (size_t i = 0; i < queryTokens.size(); i++) {
    if (i)
      oss << " ";
    oss << queryTokens[i];
  }
  out["query"] = oss.str();
  {
    std::lock_guard<std::mutex> lock(indexMu_);
    for (const auto &t : queryTokens) {
      auto it = invertedIndex_.find(t);
      if (it == invertedIndex_.end())
        continue;
      for (const auto &url : it->second)
        scores[url] += 1;
    }
  }
  std::vector<std::pair<std::string, int>> ranked(scores.begin(), scores.end());
  std::sort(ranked.begin(), ranked.end(),
            [](auto &a, auto &b) { return a.second > b.second; });
  json urls = json::array();
  std::vector<DocEntry> docs;
  {
    std::lock_guard<std::mutex> lock(indexMu_);
    for (auto &p : ranked) {
      auto it = docs_.find(p.first);
      if (it == docs_.end())
        continue;
      json item{{"url", it->second.url},
                {"score", p.second},
                {"title", it->second.title}};
      if (!it->second.aspects.empty())
        item["aspects"] = it->second.aspects;
      urls.push_back(item);
      docs.push_back(it->second);
      if ((int)urls.size() >= limitUrls)
        break;
    }
  }
  out["urls"] = urls;

  json aspects = json::array();
  {
    std::unordered_map<std::string, int> aspectScore;
    std::lock_guard<std::mutex> lock(indexMu_);
    for (const auto &t : queryTokens) {
      auto it = aspectToUrls_.find(t);
      if (it != aspectToUrls_.end()) {
        aspectScore[t] = (int)it->second.size();
      }
    }
    std::vector<std::pair<std::string, int>> rankedA(aspectScore.begin(),
                                                     aspectScore.end());
    std::sort(rankedA.begin(), rankedA.end(),
              [](auto &a, auto &b) { return a.second > b.second; });
    for (auto &p : rankedA) {
      aspects.push_back(json{{"aspect", p.first}, {"count", p.second}});
      if ((int)aspects.size() >= limitAspects)
        break;
    }
  }
  out["aspects"] = aspects;

  if (!docs.empty() && maxClusters > 0) {
    std::vector<std::vector<float>> vectors;
    vectors.reserve(docs.size());
    for (auto &d : docs)
      vectors.push_back(d.embedding);
    int k = std::min<int>(maxClusters,
                          std::max(1, (int)std::sqrt((double)vectors.size())));
    std::vector<int> assignments;
    auto centers = kMeans(vectors, k, 8, assignments);
    json clusters = json::array();
    for (int c = 0; c < k; c++) {
      json group = json::array();
      for (size_t i = 0; i < assignments.size(); i++) {
        if (assignments[i] == c) {
          group.push_back(json{{"url", docs[i].url}, {"title", docs[i].title}});
        }
      }
      if (!group.empty())
        clusters.push_back(
            json{{"cluster", c}, {"size", (int)group.size()}, {"urls", group}});
    }
    out["clusters"] = clusters;
  }
  return out;
}

std::vector<std::vector<float>>
OnlineResearcher::kMeans(const std::vector<std::vector<float>> &vectors, int k,
                         int iters, std::vector<int> &assignments) {
  std::vector<std::vector<float>> centers;
  assignments.assign(vectors.size(), 0);
  if (vectors.empty() || k <= 0)
    return centers;
  int dim = (int)vectors[0].size();
  centers.resize(k, std::vector<float>(dim, 0.0f));
  for (int i = 0; i < k; i++) {
    centers[i] = vectors[(size_t)i % vectors.size()];
  }
  auto dist2 = [&](const std::vector<float> &a, const std::vector<float> &b) {
    double d = 0.0;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; i++) {
      double diff = (double)a[i] - (double)b[i];
      d += diff * diff;
    }
    return d;
  };
  for (int it = 0; it < iters; it++) {
    for (size_t i = 0; i < vectors.size(); i++) {
      double best = std::numeric_limits<double>::max();
      int bestK = 0;
      for (int c = 0; c < k; c++) {
        double d = dist2(vectors[i], centers[c]);
        if (d < best) {
          best = d;
          bestK = c;
        }
      }
      assignments[i] = bestK;
    }
    std::vector<std::vector<float>> next(k, std::vector<float>(dim, 0.0f));
    std::vector<int> counts(k, 0);
    for (size_t i = 0; i < vectors.size(); i++) {
      int c = assignments[i];
      counts[c]++;
      for (int d = 0; d < dim; d++)
        next[c][d] += vectors[i][d];
    }
    for (int c = 0; c < k; c++) {
      if (counts[c] == 0)
        continue;
      for (int d = 0; d < dim; d++)
        next[c][d] /= (float)counts[c];
    }
    centers.swap(next);
  }
  return centers;
}

json OnlineResearcher::fallback(const std::vector<std::string> &words) {
  if (!runtime_)
    return json{
        {"ok", false}, {"source", "local"}, {"reason", "runtime-missing"}};
  if (words.empty())
    return json{{"ok", false}, {"source", "local"}, {"reason", "empty-query"}};
  auto mapping = runtime_->mapWordsToMemesOrdered(words);
  auto seeds = mapping.seeds;
  if (seeds.empty()) {
    return json{{"ok", false}, {"source", "local"}, {"reason", "no-memes"}};
  }
  auto result =
      runtime_->runPropagation(seeds, json{{"seedOrder", mapping.order}});
  json suggestions = json::array();
  int limit = std::min<int>(10, (int)result.windowInfo.ids.size());
  for (int i = 0; i < limit; i++) {
    std::string memeId = result.windowInfo.ids[i];
    double strength = (i < (int)result.activation.size() &&
                       std::isfinite(result.activation[i]))
                          ? result.activation[i]
                          : 0.0;
    std::vector<std::string> wordsForMeme;
    for (auto &w : runtime_->kvm().getMemeWords(memeId)) {
      wordsForMeme.push_back(w);
      if (wordsForMeme.size() >= 6)
        break;
    }
    suggestions.push_back(json{
        {"memeId", memeId}, {"strength", strength}, {"words", wordsForMeme}});
  }
  std::ostringstream oss;
  for (size_t i = 0; i < words.size(); i++) {
    if (i)
      oss << " ";
    oss << words[i];
  }
  return json{{"ok", true},
              {"source", "local"},
              {"query", oss.str()},
              {"suggestions", suggestions}};
}

json OnlineResearcher::crawl(const std::string &startUrl, const json &crawlReq,
                             const json &options) {
  json out;
  out["ok"] = true;
  out["source"] = "crawl";
  out["pages"] = json::array();
#if 0
#ifdef HAVE_CURL
  int timeoutMs =
      clampInt(crawlReq.value("timeoutMs", json()), 1000, 60000, 12000);
  int maxPages = clampInt(crawlReq.value("maxPages", json()), 1, 500, 60);
  int maxDepth = clampInt(crawlReq.value("maxDepth", json()), 0, 10, 3);
  int maxSeeds = clampInt(crawlReq.value("maxSeeds", json()), 0, 5000, 200);
  bool includePdf = crawlReq.value("includePdf", true);
  bool sameSite = crawlReq.value("sameSite", true);
  int maxBytesPerPage = clampInt(crawlReq.value("maxBytesPerPage", json()),
                                 8 * 1024, 10 * 1024 * 1024, 2 * 1024 * 1024);
  int maxPdfBytes = clampInt(crawlReq.value("maxPdfBytes", json()), 64 * 1024,
                             40 * 1024 * 1024, 20 * 1024 * 1024);
  (void)maxPdfBytes;
  std::string userAgent = crawlReq.value("userAgent", "079ProjectCrawler/1.0");
  bool useSeeds = crawlReq.value("useSeeds", true);
  int kClusters = clampInt(crawlReq.value("kClusters", json()), 0, 12, 4);
  int maxAspects = clampInt(crawlReq.value("maxAspects", json()), 2, 30, 12);
  json errors = json::array();
#endif
#endif
  (void)options;
  return out;
}
