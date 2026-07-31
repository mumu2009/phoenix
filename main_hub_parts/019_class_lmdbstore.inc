/* Auto-generated from main.cpp
   Segment: class_LmdbStore
   Original lines: 3016-3136
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

class LmdbStore : public KeyValueStore {
public:
  LmdbStore(const std::string &name, const fs::path &rootDir,
            std::size_t mapSizeBytes, bool encodeJSON = true)
      : name_(name), rootDir_(rootDir), mapSizeBytes_(mapSizeBytes),
        encodeJSON_(encodeJSON) {
    ensureDir(rootDir_ / name_);
    fs::path envPath = rootDir_ / name_;
    if (mdb_env_create(&env_) != 0) {
      env_ = nullptr;
      return;
    }
    mdb_env_set_maxreaders(env_, 64);
    mdb_env_set_mapsize(env_, mapSizeBytes_);
    if (mdb_env_open(env_, envPath.string().c_str(), 0, 0664) != 0) {
      mdb_env_close(env_);
      env_ = nullptr;
      return;
    }
    MDB_txn *txn = nullptr;
    if (mdb_txn_begin(env_, nullptr, 0, &txn) != 0)
      return;
    if (mdb_dbi_open(txn, "default", MDB_CREATE, &dbi_) != 0) {
      mdb_txn_abort(txn);
      return;
    }
    mdb_txn_commit(txn);
    ok_ = true;
  }

  ~LmdbStore() override {
    if (env_) {
      mdb_dbi_close(env_, dbi_);
      mdb_env_close(env_);
    }
  }

  bool ok() const { return ok_; }

  std::optional<json> get(const std::string &key) override {
    if (!ok_)
      return std::nullopt;
    MDB_txn *txn = nullptr;
    MDB_val k{key.size(), (void *)key.data()};
    MDB_val v;
    if (mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn) != 0)
      return std::nullopt;
    int rc = mdb_get(txn, dbi_, &k, &v);
    mdb_txn_abort(txn);
    if (rc != 0)
      return std::nullopt;
    std::string raw((char *)v.mv_data, v.mv_size);
    try {
      return encodeJSON_ ? json::parse(raw) : json(raw);
    } catch (...) {
      return json(raw);
    }
  }

  void put(const std::string &key, const json &value) override {
    if (!ok_)
      return;
    MDB_txn *txn = nullptr;
    mdb_txn_begin(env_, nullptr, 0, &txn);
    std::string encoded = encodeJSON_ ? value.dump() : value.get<std::string>();
    MDB_val k{key.size(), (void *)key.data()};
    MDB_val v{encoded.size(), (void *)encoded.data()};
    mdb_put(txn, dbi_, &k, &v, 0);
    mdb_txn_commit(txn);
  }

  void del(const std::string &key) override {
    if (!ok_)
      return;
    MDB_txn *txn = nullptr;
    mdb_txn_begin(env_, nullptr, 0, &txn);
    MDB_val k{key.size(), (void *)key.data()};
    mdb_del(txn, dbi_, &k, nullptr);
    mdb_txn_commit(txn);
  }

  std::vector<std::pair<std::string, json>>
  entries(const std::string &prefix) override {
    std::vector<std::pair<std::string, json>> out;
    if (!ok_)
      return out;
    MDB_txn *txn = nullptr;
    MDB_cursor *cursor = nullptr;
    mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn);
    mdb_cursor_open(txn, dbi_, &cursor);
    MDB_val k, v;
    std::string start = prefix;
    k.mv_size = start.size();
    k.mv_data = (void *)start.data();
    int rc = mdb_cursor_get(cursor, &k, &v, MDB_SET_RANGE);
    while (rc == 0) {
      std::string key((char *)k.mv_data, k.mv_size);
      if (!prefix.empty() && key.rfind(prefix, 0) != 0)
        break;
      std::string raw((char *)v.mv_data, v.mv_size);
      try {
        out.push_back({key, encodeJSON_ ? json::parse(raw) : json(raw)});
      } catch (...) {
        out.push_back({key, json(raw)});
      }
      rc = mdb_cursor_get(cursor, &k, &v, MDB_NEXT);
    }
    mdb_cursor_close(cursor);
    mdb_txn_abort(txn);
    return out;
  }

private:
  std::string name_;
  fs::path rootDir_;
  std::size_t mapSizeBytes_;
  bool encodeJSON_{true};
  bool ok_{false};
  MDB_env *env_{nullptr};
  MDB_dbi dbi_{0};
};
