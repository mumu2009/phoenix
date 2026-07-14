/* Auto-generated from main.cpp
   Segment: class_NamespacedStore
   Original lines: 3361-3402
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

class NamespacedStore : public KeyValueStore {
public:
  NamespacedStore(std::shared_ptr<KeyValueStore> base, const std::string &ns)
      : base_(std::move(base)) {
    std::string trimmed = ns;
    trimmed.erase(trimmed.begin(), std::find_if(trimmed.begin(), trimmed.end(),
                                                [](unsigned char ch) {
                                                  return !std::isspace(ch);
                                                }));
    trimmed.erase(
        std::find_if(trimmed.rbegin(), trimmed.rend(),
                     [](unsigned char ch) { return !std::isspace(ch); })
            .base(),
        trimmed.end());
    if (trimmed.empty())
      throw std::runtime_error("NamespacedStore requires a namespace");
    prefix_ = "ns:" + trimmed + ":";
  }

  std::optional<json> get(const std::string &key) override {
    return base_->get(prefix_ + key);
  }
  void put(const std::string &key, const json &value) override {
    base_->put(prefix_ + key, value);
  }
  void del(const std::string &key) override { base_->del(prefix_ + key); }
  std::vector<std::pair<std::string, json>>
  entries(const std::string &prefix) override {
    auto all = base_->entries(prefix_ + prefix);
    std::vector<std::pair<std::string, json>> out;
    for (auto &kv : all) {
      if (kv.first.rfind(prefix_, 0) == 0) {
        out.push_back({kv.first.substr(prefix_.size()), kv.second});
      }
    }
    return out;
  }

private:
  std::shared_ptr<KeyValueStore> base_;
  std::string prefix_;
};
