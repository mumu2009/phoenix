/* Auto-generated from main.cpp
   Segment: class_HotMatrixCache
   Original lines: 1874-1986
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

class HotMatrixCache {
public:
  HotMatrixCache(std::string redisUrl, int db, std::string prefix)
      : redisUrl_(std::move(redisUrl)), db_(db), prefix_(std::move(prefix)) {
#ifdef HAVE_REDIS
    try {
      sw::redis::ConnectionOptions opts(redisUrl_);
      opts.db = db_;
      redis_ = std::make_shared<sw::redis::Redis>(opts);
    } catch (...) {
      redis_.reset();
    }
#endif
  }

  bool enabled() const {
#ifdef HAVE_REDIS
    return (bool)redis_;
#else
    return false;
#endif
  }

  bool getMatrix(const std::string &key, int rows, int cols,
                 std::vector<std::vector<float>> &out) {
#ifdef HAVE_REDIS
    if (!redis_)
      return false;
    try {
      auto val = redis_->get(prefix_ + ":data:" + key);
      if (!val)
        return false;
      json payload = json::parse(*val, nullptr, false);
      if (payload.is_discarded())
        return false;
      int r = payload.value("rows", 0);
      int c = payload.value("cols", 0);
      if (r <= 0 || c <= 0)
        return false;
      if (rows > 0 && rows != r)
        return false;
      if (cols > 0 && cols != c)
        return false;
      auto data = payload.value("data", std::vector<float>{});
      if (data.size() != (size_t)r * (size_t)c)
        return false;
      out.assign((size_t)r, std::vector<float>((size_t)c, 0.0f));
      for (int i = 0; i < r; i++) {
        auto rowBegin = data.begin() + (size_t)i * (size_t)c;
        std::copy(rowBegin, rowBegin + c, out[(size_t)i].begin());
      }
      redis_->incr(prefix_ + ":hits:" + key);
      return true;
    } catch (...) {
      return false;
    }
#else
    (void)key;
    (void)rows;
    (void)cols;
    (void)out;
    return false;
#endif
  }

  void recordAndMaybeSet(const std::string &key,
                         const std::vector<std::vector<float>> &matrix,
                         int ttlSeconds, int hotThreshold = 3) {
#ifdef HAVE_REDIS
    if (!redis_)
      return;
    int hits = 0;
    {
      std::lock_guard<std::mutex> lock(mu_);
      hits = ++localHits_[key];
    }
    if (hits < hotThreshold)
      return;
    try {
      int rows = (int)matrix.size();
      int cols = rows > 0 ? (int)matrix[0].size() : 0;
      std::vector<float> flat;
      flat.reserve((size_t)rows * (size_t)cols);
      for (int i = 0; i < rows; i++) {
        const auto &row = matrix[(size_t)i];
        flat.insert(flat.end(), row.begin(), row.begin() + cols);
      }
      json payload{{"rows", rows}, {"cols", cols}, {"data", flat}};
      std::string k = prefix_ + ":data:" + key;
      redis_->set(k, payload.dump());
      if (ttlSeconds > 0)
        redis_->expire(k, ttlSeconds);
    } catch (...) {
    }
#else
    (void)key;
    (void)matrix;
    (void)ttlSeconds;
    (void)hotThreshold;
#endif
  }

private:
  std::string redisUrl_;
  int db_{1};
  std::string prefix_;
#ifdef HAVE_REDIS
  std::shared_ptr<sw::redis::Redis> redis_;
#endif
  std::mutex mu_;
  std::unordered_map<std::string, int> localHits_;
};
