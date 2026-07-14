/* Auto-generated from main.cpp
   Segment: class_OnlineResearcher
   Original lines: 3497-3565
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

class OnlineResearcher {
public:
  OnlineResearcher(RuntimeState *runtime, const std::string &endpoint,
                   int cooldownMs = 5 * 60 * 1000, int cacheSize = 128)
      : runtime_(runtime), endpoint_(endpoint), cooldownMs_(cooldownMs),
        cacheSize_(cacheSize) {}

  // setEndpoint 更新在线检索服务地址。
  // 调用方式：配置热更新后调用以切换后端。
  // 实现思路：直接替换内部 endpoint 字段。
  // 注意事项：新地址需由调用方保证可达与合法。
  void setEndpoint(const std::string &endpoint) { endpoint_ = endpoint; }
  // setEnabled 开启或关闭在线检索功能。
  // 调用方式：运行期开关控制时调用。
  // 实现思路：更新布尔标记供查询路径判断。
  // 注意事项：关闭后将走本地降级路径。
  void setEnabled(bool enabled) { enabled_ = enabled; }

  json lookup(const json &input, const json &options);

private:
  struct CacheEntry {
    int64_t ts{0};
    json payload;
  };
  struct DocEntry {
    std::string url;
    std::string title;
    std::string text;
    std::vector<std::string> tokens;
    std::vector<float> embedding;
    std::vector<std::string> aspects;
  };

  RuntimeState *runtime_{nullptr};
  std::string endpoint_;
  int cooldownMs_{5 * 60 * 1000};
  int cacheSize_{128};
  bool enabled_{true};
  std::unordered_map<std::string, CacheEntry> cache_;
  std::unordered_map<std::string, DocEntry> docs_;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      invertedIndex_;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      aspectToUrls_;
  std::unordered_map<std::string, std::string> urlToContent_;
  std::mutex indexMu_;

  std::string normalizeKey(const std::vector<std::string> &words) const;
  void pruneCache();
  void remember(const std::string &key, const json &payload);
  std::optional<json> fromCache(const std::string &key);
  json fallback(const std::vector<std::string> &words);
  json crawl(const std::string &startUrl, const json &crawlReq,
             const json &options);
  std::vector<std::string> loadSeedUrls(int maxSeeds) const;
  std::vector<std::string>
  extractLinksFromHtml(const std::string &html,
                       const std::string &baseUrl) const;
  std::string cleanTextAdvanced(const std::string &text) const;
  std::vector<std::string> pickAspects(const std::vector<std::string> &tokens,
                                       int maxAspects) const;
  void indexDocument(const DocEntry &doc);
  json queryLocalIndex(const std::vector<std::string> &words, int limitUrls,
                       int limitAspects, int maxClusters);
  static std::vector<std::vector<float>>
  kMeans(const std::vector<std::vector<float>> &vectors, int k, int iters,
         std::vector<int> &assignments);
};
