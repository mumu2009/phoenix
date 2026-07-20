/* DATABASE_079.hpp - Database and persistence layer for 079 Project
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

#pragma once

#include <filesystem>
#include <mutex>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

/* Abstract key-value store interface.
   Provides a common interface for different storage backends (SQLite, Redis, etc.). */
class KeyValueStore {
public:
    virtual ~KeyValueStore() = default;
    virtual std::optional<nlohmann::json> get(const std::string &key) = 0; /* Get value by key */
    virtual void put(const std::string &key, const nlohmann::json &value) = 0; /* Store key-value pair */
    virtual void del(const std::string &key) = 0; /* Delete key */
    virtual std::vector<std::pair<std::string, nlohmann::json>> entries(const std::string &prefix) = 0; /* List entries with prefix */
    virtual void flush() {} /* Flush pending writes */
};

class SqliteKeyValueStore;
class HotCache;
struct DbHandle;

/* Main database manager for 079 Project.
   Manages multiple key-value stores with hot/cold caching and Redis integration. */
class Database079 {
public:
    /* Collection of named stores for different data types */
    struct Stores {
        std::shared_ptr<KeyValueStore> kvm;      /* Key-value memory store */
        std::shared_ptr<KeyValueStore> meme;     /* Meme/graph store */
        std::shared_ptr<KeyValueStore> session;  /* Session store */
    };

    Database079(std::filesystem::path dbPath,
                std::filesystem::path legacyDir,
                std::string redisUrl,
                int redisDb,
                std::string redisPrefix);

    ~Database079(); /* Close database connection on destruction */

    bool open(); /* Open database connection */
    void close(); /* Close database connection */

    std::shared_ptr<KeyValueStore> createStore(const std::string &name); /* Create a new named store */

    bool getInferenceCache(const std::string &key, nlohmann::json &out); /* Get inference result from cache */
    void setInferenceCache(const std::string &key, const nlohmann::json &value, int ttlSeconds = 86400); /* Set inference cache */
    void configureInferenceSwap(std::size_t hotLimit, int promoteHits, int hotTtlSeconds, int rebalanceEvery); /* Configure hot/cold swap */
    nlohmann::json getSwapStats() const; /* Get cache swap statistics */

    bool isOpen() const { return opened_; } /* Check if database is open */

    static bool isSqliteStore(const std::shared_ptr<KeyValueStore> &store); /* Check if store is SQLite-backed */

private:
    std::filesystem::path dbPath_;        /* Database file path */
    std::filesystem::path legacyDir_;     /* Legacy data directory */
    std::string redisUrl_;                /* Redis connection URL */
    int redisDb_{1};                      /* Redis database index */
    std::string redisPrefix_;             /* Redis key prefix */
    std::shared_ptr<DbHandle> handle_;    /* Database handle */
    std::shared_ptr<HotCache> hot_;       /* Hot cache instance */
    bool opened_{false};                  /* Database open state */
    mutable std::mutex swapMu_;           /* Swap mutex */
    std::unordered_map<std::string, int> recentInferenceTouches_; /* Recent access counts */
    std::size_t inferenceHotLimit_{512};  /* Hot cache entry limit */
    int inferencePromoteHits_{2};         /* Hits needed to promote to hot */
    int inferenceHotTtlSeconds_{3600};    /* Hot cache TTL in seconds */
    int inferenceRebalanceEvery_{64};    /* Rebalance every N accesses */
    std::uint64_t inferenceAccesses_{0};  /* Total cache accesses */
    std::uint64_t inferenceHotHits_{0};  /* Hot cache hits */
    std::uint64_t inferenceColdLoads_{0}; /* Cold cache loads */
    std::uint64_t inferenceWrites_{0};   /* Cache writes */
    std::uint64_t inferencePromotions_{0}; /* Promotions to hot */
    std::uint64_t inferenceDemotions_{0};  /* Demotions to cold */
};
