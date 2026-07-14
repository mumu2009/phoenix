/* Auto-generated from main.cpp
   Segment: class_KVMStore
   Original lines: 4089-4240
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

class KVMStore
/* KVMStore maintains key-value memory data read/write and retrieval logic */
// 调用方式：在会话和长期记忆路径中保存与提取状态。
// 实现思路：围绕键前缀、命名空间和索引执行操作。
// 注意事项：后端能力会影响一致性与时延表现。
// 注意事项：批量操作应明确事务边界与重试策略。
// 注意事项：该类不负责业务层权限校验。
{
public:
  explicit KVMStore(std::shared_ptr<KeyValueStore> store, int maxCacheEntries)
      : store_(std::move(store)), maxCacheEntries_(maxCacheEntries) {}

  std::vector<std::string> getWordMemeSet(const std::string &word) {
    std::string key = "word:" + word;
    auto cached = cacheGet(key);
    if (cached)
      return cached->order;
    auto out = loadOrdered(key);
    return out.order;
  }

  std::vector<std::string> getMemeWords(const std::string &memeId) {
    std::string key = "meme:" + memeId;
    auto cached = cacheGet(key);
    if (cached)
      return cached->order;
    auto out = loadOrdered(key);
    return out.order;
  }

  void link(const std::string &word, const std::string &memeId) {
    std::string wordKey = "word:" + word;
    std::string memeKey = "meme:" + memeId;
    auto wordEntry = loadOrdered(wordKey);
    auto memeEntry = loadOrdered(memeKey);
    if (!wordEntry.set.count(memeId)) {
      wordEntry.set.insert(memeId);
      wordEntry.order.push_back(memeId);
      store_->put(wordKey, json(wordEntry.order));
      cacheSet(wordKey, wordEntry);
    }
    if (!memeEntry.set.count(word)) {
      memeEntry.set.insert(word);
      memeEntry.order.push_back(word);
      store_->put(memeKey, json(memeEntry.order));
      cacheSet(memeKey, memeEntry);
    }
  }

  void unlink(const std::string &word, const std::string &memeId) {
    std::string wordKey = "word:" + word;
    std::string memeKey = "meme:" + memeId;
    auto wordEntry = loadOrdered(wordKey);
    auto memeEntry = loadOrdered(memeKey);
    if (wordEntry.set.erase(memeId)) {
      auto it =
          std::find(wordEntry.order.begin(), wordEntry.order.end(), memeId);
      if (it != wordEntry.order.end())
        wordEntry.order.erase(it);
      store_->put(wordKey, json(wordEntry.order));
      cacheSet(wordKey, wordEntry);
    }
    if (memeEntry.set.erase(word)) {
      auto it = std::find(memeEntry.order.begin(), memeEntry.order.end(), word);
      if (it != memeEntry.order.end())
        memeEntry.order.erase(it);
      store_->put(memeKey, json(memeEntry.order));
      cacheSet(memeKey, memeEntry);
    }
  }

  json exportEntries() {
    std::lock_guard<std::mutex> lock(cacheMu_);
    json out;
    json words = json::object();
    json memes = json::object();
    for (auto &kv : cache_) {
      if (kv.first.rfind("word:", 0) == 0) {
        words[kv.first.substr(5)] = json(kv.second.value.order);
      } else if (kv.first.rfind("meme:", 0) == 0) {
        memes[kv.first.substr(5)] = json(kv.second.value.order);
      }
    }
    out["words"] = words;
    out["memes"] = memes;
    return out;
  }

private:
  std::shared_ptr<KeyValueStore> store_;
  struct OrderedSet {
    std::vector<std::string> order;
    std::unordered_set<std::string> set;
  };
  struct CacheEntry {
    OrderedSet value;
    std::list<std::string>::iterator it;
  };
  mutable std::mutex cacheMu_;
  std::unordered_map<std::string, CacheEntry> cache_;
  std::list<std::string> lru_;
  int maxCacheEntries_{50000};

  std::optional<OrderedSet> cacheGet(const std::string &key) {
    std::lock_guard<std::mutex> lock(cacheMu_);
    if (maxCacheEntries_ <= 0)
      return std::nullopt;
    auto it = cache_.find(key);
    if (it == cache_.end())
      return std::nullopt;
    lru_.erase(it->second.it);
    lru_.push_back(key);
    it->second.it = std::prev(lru_.end());
    return it->second.value;
  }

  void cacheSet(const std::string &key, const OrderedSet &value) {
    std::lock_guard<std::mutex> lock(cacheMu_);
    if (maxCacheEntries_ <= 0)
      return;
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      it->second.value = value;
      lru_.erase(it->second.it);
      lru_.push_back(key);
      it->second.it = std::prev(lru_.end());
    } else {
      lru_.push_back(key);
      cache_[key] = CacheEntry{value, std::prev(lru_.end())};
    }
    while ((int)cache_.size() > maxCacheEntries_) {
      auto oldest = lru_.front();
      lru_.pop_front();
      cache_.erase(oldest);
    }
  }

  OrderedSet loadOrdered(const std::string &key) {
    auto cached = cacheGet(key);
    if (cached)
      return *cached;
    OrderedSet out;
    auto v = store_->get(key);
    if (v && v->is_array()) {
      for (auto &x : *v) {
        if (!x.is_string())
          continue;
        std::string s = x.get<std::string>();
        if (out.set.insert(s).second)
          out.order.push_back(s);
      }
    }
    cacheSet(key, out);
    return out;
  }
};
