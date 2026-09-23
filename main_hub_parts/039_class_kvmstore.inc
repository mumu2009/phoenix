/* Auto-generated from main.cpp
   Segment: class_KVMStore
   Original lines: 4089-4240 */

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

  int wordTf(const std::string &memeId, const std::string &word) {
    if (memeId.empty() || word.empty())
      return 0;
    auto memeEntry = loadOrdered("meme:" + memeId);
    auto it = memeEntry.tf.find(word);
    if (it != memeEntry.tf.end() && it->second > 0)
      return it->second;
    return memeEntry.set.count(word) ? 1 : 0;
  }

  void link(const std::string &word, const std::string &memeId) {
    link(word, memeId, true);
  }

  /* countTf: ingest increments meme-conditional tf. Query-time
     re-bind of an existing pair must not inflate weights. */
  void link(const std::string &word, const std::string &memeId, bool countTf) {
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
    bool freshWord = !memeEntry.set.count(word);
    if (freshWord) {
      memeEntry.set.insert(word);
      memeEntry.order.push_back(word);
      memeEntry.tf[word] = 1;
    } else if (countTf) {
      int cur = 1;
      auto it = memeEntry.tf.find(word);
      if (it != memeEntry.tf.end() && it->second > 0)
        cur = it->second;
      memeEntry.tf[word] = cur + 1;
    }
    if (freshWord || countTf) {
      store_->put(memeKey, json(memeEntry.order));
      json tfJson = json::object();
      for (const auto &w : memeEntry.order) {
        int t = 1;
        auto it = memeEntry.tf.find(w);
        if (it != memeEntry.tf.end() && it->second > 0)
          t = it->second;
        tfJson[w] = t;
      }
      store_->put("meme_tf:" + memeId, tfJson);
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
      memeEntry.tf.erase(word);
      store_->put(memeKey, json(memeEntry.order));
      json tfJson = json::object();
      for (const auto &w : memeEntry.order) {
        int t = 1;
        auto jt = memeEntry.tf.find(w);
        if (jt != memeEntry.tf.end() && jt->second > 0)
          t = jt->second;
        tfJson[w] = t;
      }
      store_->put("meme_tf:" + memeId, tfJson);
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
    std::unordered_map<std::string, int> tf;
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
    if (key.rfind("meme:", 0) == 0) {
      auto tfv = store_->get("meme_tf:" + key.substr(5));
      if (tfv && tfv->is_object()) {
        for (auto &kv : tfv->items()) {
          if (!kv.value().is_number())
            continue;
          int t = kv.value().get<int>();
          if (t > 0)
            out.tf[kv.key()] = t;
        }
      }
      for (const auto &w : out.order) {
        if (!out.tf.count(w) || out.tf[w] <= 0)
          out.tf[w] = 1;
      }
    }
    cacheSet(key, out);
    return out;
  }
};
