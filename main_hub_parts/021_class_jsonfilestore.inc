/* Auto-generated from main.cpp
   Segment: class_JsonFileStore
   Original lines: 3144-3353
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

class JsonFileStore : public KeyValueStore {
public:
  JsonFileStore(const std::string &name, const fs::path &rootDir)
      : name_(name), rootDir_(rootDir) {
    ensureDir(rootDir_);
    file_ = rootDir_ / (name_ + ".json");
    binFile_ = rootDir_ / (name_ + ".msgpack");
    loadFromDisk();
    startFlushThread();
  }

  ~JsonFileStore() override {
    stopFlushThread();
    flush();
  }

  std::optional<json> get(const std::string &key) override {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = data_.find(key);
    if (it == data_.end())
      return std::nullopt;
    return it->second;
  }

  void put(const std::string &key, const json &value) override {
    std::lock_guard<std::mutex> lock(mu_);
    if (!data_.count(key))
      order_.push_back(key);
    data_[key] = value;
    dirty_ = true;
  }

  void del(const std::string &key) override {
    std::lock_guard<std::mutex> lock(mu_);
    if (data_.erase(key) > 0) {
      order_.erase(std::remove(order_.begin(), order_.end(), key),
                   order_.end());
    }
    dirty_ = true;
  }

  std::vector<std::pair<std::string, json>>
  entries(const std::string &prefix) override {
    std::vector<std::pair<std::string, json>> out;
    std::lock_guard<std::mutex> lock(mu_);
    for (auto &key : order_) {
      auto it = data_.find(key);
      if (it == data_.end())
        continue;
      if (prefix.empty() || key.rfind(prefix, 0) == 0)
        out.push_back(*it);
    }
    return out;
  }

  void flush() override {
    std::lock_guard<std::mutex> lock(mu_);
    if (!dirty_)
      return;

    nlohmann::ordered_json doc = nlohmann::ordered_json::object();
    nlohmann::ordered_json items = nlohmann::ordered_json::object();
    nlohmann::ordered_json order = nlohmann::ordered_json::array();
    for (auto &key : order_) {
      auto it = data_.find(key);
      if (it != data_.end()) {
        items[key] = it->second;
        order.push_back(key);
      }
    }
    doc["version"] = 2;
    doc["items"] = std::move(items);
    doc["order"] = std::move(order);

    try {
      auto packed = json::to_msgpack(doc);
      fs::path tmpBin = binFile_;
      tmpBin += ".tmp";
      {
        std::ofstream bout(tmpBin, std::ios::binary | std::ios::trunc);
        bout.write(reinterpret_cast<const char *>(packed.data()),
                   (std::streamsize)packed.size());
      }
      if (fs::exists(binFile_))
        fs::remove(binFile_);
      fs::rename(tmpBin, binFile_);
    } catch (...) {
    }

    try {
      fs::path tmpJson = file_;
      tmpJson += ".tmp";
      std::ofstream jout(tmpJson, std::ios::trunc);
      jout << doc.dump(2);
      if (fs::exists(file_))
        fs::remove(file_);
      fs::rename(tmpJson, file_);
    } catch (...) {
    }

    dirty_ = false;
  }

private:
  bool loadOrderedObject(const nlohmann::ordered_json &j) {
    if (!j.is_object())
      return false;
    data_.clear();
    order_.clear();

    auto readLegacy = [&]() {
      for (auto it = j.begin(); it != j.end(); ++it) {
        json value = it.value();
        if (value.is_string()) {
          const auto raw = value.get<std::string>();
          try {
            data_[it.key()] = json::parse(raw);
          } catch (...) {
            data_[it.key()] = value;
          }
        } else {
          data_[it.key()] = value;
        }
        order_.push_back(it.key());
      }
    };

    if (!j.contains("items") || !j["items"].is_object()) {
      readLegacy();
      return true;
    }

    const auto &items = j["items"];
    if (j.contains("order") && j["order"].is_array()) {
      for (const auto &id : j["order"]) {
        if (!id.is_string())
          continue;
        std::string key = id.get<std::string>();
        auto it = items.find(key);
        if (it == items.end())
          continue;
        data_[key] = *it;
        order_.push_back(key);
      }
    }
    for (auto it = items.begin(); it != items.end(); ++it) {
      if (!data_.count(it.key())) {
        data_[it.key()] = it.value();
        order_.push_back(it.key());
      }
    }
    return true;
  }

  void loadFromDisk() {
    if (fs::exists(binFile_)) {
      try {
        std::ifstream in(binFile_, std::ios::binary);
        std::vector<uint8_t> raw((std::istreambuf_iterator<char>(in)),
                                 std::istreambuf_iterator<char>());
        if (!raw.empty()) {
          auto j = json::from_msgpack(raw);
          nlohmann::ordered_json o = j;
          if (loadOrderedObject(o))
            return;
        }
      } catch (...) {
      }
    }

    if (fs::exists(file_)) {
      try {
        std::ifstream in(file_);
        nlohmann::ordered_json j;
        in >> j;
        loadOrderedObject(j);
      } catch (...) {
      }
    }
  }

  void startFlushThread() {
    running_.store(true);
    flushThread_ = std::thread([this]() {
      while (running_.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        if (!running_.load())
          break;
        flush();
      }
    });
  }

  void stopFlushThread() {
    running_.store(false);
    if (flushThread_.joinable())
      flushThread_.join();
  }

  std::string name_;
  fs::path rootDir_;
  fs::path file_;
  fs::path binFile_;
  bool dirty_{false};
  std::unordered_map<std::string, json> data_;
  std::vector<std::string> order_;
  std::mutex mu_;
  std::atomic<bool> running_{false};
  std::thread flushThread_;
};
