/* DATABASE_079.cpp - Database implementation
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

#include "DATABASE_079.hpp"
#include "loggerCXXH.hpp"

#include <chrono>
#include <fstream>
#include <mutex>
#include <unordered_map>

#ifdef HAVE_SQLITE
#include <sqlite3.h>
#endif

#ifdef HAVE_REDIS
#include <sw/redis++/redis++.h>
#endif

using json = nlohmann::json;
namespace fs = std::filesystem;

/* DbHandle encapsulates underlying SQLite connection and mutex */
// 调用方式：由 Database079 在 open() 成功后创建并持有。
// 实现思路：将 sqlite3* 与 std::mutex 聚合，统一并发访问入口。
// 注意事项：仅在 HAVE_SQLITE 打开时包含 db 字段，避免无效引用。
// 注意事项：访问 db 前需确保连接已打开且生命周期仍然有效。
// 注意事项：上层需通过 lock_guard 控制线程安全。
struct DbHandle
{
#ifdef HAVE_SQLITE
    sqlite3 *db{nullptr};
#endif
    std::mutex mu;
};

// 返回当前 Unix 毫秒时间戳。
// 调用方式：用于写入 updatedAt 或缓存时间相关字段。
// 实现思路：基于 system_clock 转换为 milliseconds。
// 注意事项：仅提供相对时序参考，不保证跨机器绝对同步。
static int64_t nowMs()
{
    return (int64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

// 获取数据库模块统一日志实例。
// 调用方式：在数据库读写、告警和错误路径中直接调用。
// 实现思路：返回 LoggerCXX 的单例引用。
// 注意事项：日志开关与级别由 LoggerCXX 全局配置控制。
static LoggerCXX &dbLogger()
{
    return LoggerCXX::instance();
}

// HotCache 提供 Redis 热缓存能力以减少数据库访问压力。
// 调用方式：由 Database079 创建后注入到具体 Store 中使用。
// 实现思路：以 key/value 字符串形式读写，支持可选过期时间。
// 注意事项：在未启用 HAVE_REDIS 时自动退化为空实现。
// 注意事项：缓存失败不应影响主流程，所有异常内部吞掉。
// 注意事项：key 前缀由调用方控制，避免命名冲突。
class HotCache
{
public:
    // 构造缓存对象并尝试连接 Redis。
    // 调用方式：传入连接地址、数据库编号和 key 前缀。
    // 实现思路：创建 redis++ 连接对象并保存共享指针。
    // 注意事项：连接失败时会清空 redis_，后续调用将安全返回。
    HotCache(std::string redisUrl, int db, std::string prefix)
        : redisUrl_(std::move(redisUrl)), db_(db), prefix_(std::move(prefix))
    {
#ifdef HAVE_REDIS
        try
        {
            sw::redis::ConnectionOptions opts(redisUrl_);
            opts.db = db_;
            redis_ = std::make_shared<sw::redis::Redis>(opts);
        }
        catch (...)
        {
            redis_.reset();
        }
#endif
    }

    // 从热缓存读取指定 key 的字符串值。
    // 调用方式：调用成功时返回 true，并通过 out 输出内容。
    // 实现思路：拼接前缀后执行 GET，命中即赋值。
    // 注意事项：当 Redis 不可用或异常时返回 false，不抛出异常。
    bool get(const std::string &key, std::string &out)
    {
#ifdef HAVE_REDIS
        if (!redis_)
            return false;
        try
        {
            auto val = redis_->get(prefix_ + key);
            if (!val)
                return false;
            out = *val;
            return true;
        }
        catch (...)
        {
            return false;
        }
#else
        (void)key;
        (void)out;
        return false;
#endif
    }

    // 写入热缓存并按需设置 TTL。
    // 调用方式：传入业务 key、序列化值和过期秒数。
    // 实现思路：先 SET 再在 ttlSeconds>0 时执行 EXPIRE。
    // 注意事项：缓存写失败仅影响命中率，不影响主存储正确性。
    void set(const std::string &key, const std::string &value, int ttlSeconds)
    {
#ifdef HAVE_REDIS
        if (!redis_)
            return;
        try
        {
            redis_->set(prefix_ + key, value);
            if (ttlSeconds > 0)
                redis_->expire(prefix_ + key, ttlSeconds);
        }
        catch (...)
        {
        }
#else
        (void)key;
        (void)value;
        (void)ttlSeconds;
#endif
    }

    // 删除热缓存中的指定键。
    // 调用方式：在删除或失效业务数据后同步调用。
    // 实现思路：执行 Redis DEL 清理对应前缀 key。
    // 注意事项：删除失败会被吞掉，避免打断主流程。
    void del(const std::string &key)
    {
#ifdef HAVE_REDIS
        if (!redis_)
            return;
        try
        {
            redis_->del(prefix_ + key);
        }
        catch (...)
        {
        }
#else
        (void)key;
#endif
    }

private:
    std::string redisUrl_;
    int db_{1};
    std::string prefix_;
#ifdef HAVE_REDIS
    std::shared_ptr<sw::redis::Redis> redis_;
#endif
};

#ifdef HAVE_SQLITE
// 执行一段不返回结果集的 SQL 语句。
// 调用方式：用于 PRAGMA、建表等初始化语句。
// 实现思路：调用 sqlite3_exec 并检查返回码。
// 注意事项：失败时释放错误字符串并返回 false。
static bool execSql(sqlite3 *db, const std::string &sql)
{
    char *err = nullptr;
    int rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &err);
    if (rc != SQLITE_OK)
    {
        if (err)
            sqlite3_free(err);
        return false;
    }
    return true;
}

// 将原始字符串解析为 JSON 值。
// 调用方式：用于从存储层反序列化 value 字段。
// 实现思路：优先按 JSON 解析，失败时降级为普通字符串 JSON。
// 注意事项：空串返回空 JSON 值而非错误。
static std::optional<json> parseJsonValue(const std::string &raw)
{
    if (raw.empty())
        return json();
    try
    {
        return json::parse(raw);
    }
    catch (...)
    {
        return json(raw);
    }
}
#endif

// SqliteKeyValueStore 提供命名空间隔离的键值存储实现。
// 调用方式：由 Database079::createStore 创建并返回给业务层。
// 实现思路：主数据落地 SQLite，热数据可选 Redis 加速。
// 注意事项：所有读写通过 handle_ 锁保护，确保线程安全。
// 注意事项：构造时会执行旧 JSON 文件迁移逻辑。
// 注意事项：在未启用 SQLite 时接口会退化为安全空操作。
class SqliteKeyValueStore : public KeyValueStore
{
public:
    // 构造命名空间存储对象并触发必要迁移。
    // 调用方式：传入共享句柄、命名空间、旧数据目录和热缓存实例。
    // 实现思路：保存参数后调用 migrateIfNeeded()。
    // 注意事项：迁移仅在目标命名空间无数据时执行。
    SqliteKeyValueStore(std::shared_ptr<DbHandle> handle,
                        std::string ns,
                        fs::path legacyDir,
                        std::shared_ptr<HotCache> hot)
        : handle_(std::move(handle)), ns_(std::move(ns)), legacyDir_(std::move(legacyDir)), hot_(std::move(hot))
    {
        migrateIfNeeded();
    }

    // 读取指定 key 的 JSON 值。
    // 调用方式：命中返回 optional<json>，未命中返回 nullopt。
    // 实现思路：先查热缓存，再查 SQLite 并回填缓存。
    // 注意事项：调用方应处理 nullopt 与解析失败场景。
    std::optional<json> get(const std::string &key) override
    {
#ifdef HAVE_SQLITE
        if (!handle_ || !handle_->db)
            return std::nullopt;
        if (dbLogger().enabled())
            dbLogger().log(LoggerCXX::Type::COMPUTE, std::string("kv get ns=") + ns_ + " key=" + key);
        if (hot_)
        {
            std::string cached;
            if (hot_->get(cacheKey(key), cached))
            {
                auto parsed = parseJsonValue(cached);
                if (parsed)
                {
                    promote(key);
                    return parsed;
                }
            }
        }
        std::lock_guard<std::mutex> lock(handle_->mu);
        sqlite3_stmt *stmt = nullptr;
        const char *sql = "SELECT value FROM kv_store WHERE namespace=? AND key=?";
        if (sqlite3_prepare_v2(handle_->db, sql, -1, &stmt, nullptr) != SQLITE_OK)
            return std::nullopt;
        sqlite3_bind_text(stmt, 1, ns_.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, key.c_str(), -1, SQLITE_TRANSIENT);
        std::optional<json> out = std::nullopt;
        if (sqlite3_step(stmt) == SQLITE_ROW)
        {
            const unsigned char *raw = sqlite3_column_text(stmt, 0);
            std::string value = raw ? (const char *)raw : "";
            out = parseJsonValue(value);
            if (hot_)
            {
                hot_->set(cacheKey(key), value, hotTtlSeconds_);
            }
        }
        sqlite3_finalize(stmt);
        return out;
#else
        (void)key;
        return std::nullopt;
#endif
    }

    // 写入或覆盖指定 key 的 JSON 值。
    // 调用方式：传入 key 与 json，接口无返回值。
    // 实现思路：使用 UPSERT 语句写入 kv_store，并更新时间戳。
    // 注意事项：写入成功后会同步刷新热缓存。
    void put(const std::string &key, const json &value) override
    {
#ifdef HAVE_SQLITE
        if (!handle_ || !handle_->db)
            return;
        if (dbLogger().enabled())
            dbLogger().log(LoggerCXX::Type::COMPUTE, std::string("kv put ns=") + ns_ + " key=" + key);
        std::string raw = value.dump();
        std::lock_guard<std::mutex> lock(handle_->mu);
        sqlite3_stmt *stmt = nullptr;
        const char *sql = "INSERT INTO kv_store(namespace,key,value,updatedAt) VALUES(?,?,?,?)"
                          " ON CONFLICT(namespace,key) DO UPDATE SET value=excluded.value, updatedAt=excluded.updatedAt";
        if (sqlite3_prepare_v2(handle_->db, sql, -1, &stmt, nullptr) != SQLITE_OK)
            return;
        sqlite3_bind_text(stmt, 1, ns_.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, key.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 3, raw.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int64(stmt, 4, nowMs());
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
        if (hot_)
            hot_->set(cacheKey(key), raw, hotTtlSeconds_);
#else
        (void)key;
        (void)value;
#endif
    }

    // 删除指定 key。
    // 调用方式：用于业务侧明确删除数据。
    // 实现思路：执行 DELETE 语句并清理热缓存副本。
    // 注意事项：未命中删除属于正常情况，不视为错误。
    void del(const std::string &key) override
    {
#ifdef HAVE_SQLITE
        if (!handle_ || !handle_->db)
            return;
        if (dbLogger().enabled())
            dbLogger().log(LoggerCXX::Type::COMPUTE, std::string("kv del ns=") + ns_ + " key=" + key);
        std::lock_guard<std::mutex> lock(handle_->mu);
        sqlite3_stmt *stmt = nullptr;
        const char *sql = "DELETE FROM kv_store WHERE namespace=? AND key=?";
        if (sqlite3_prepare_v2(handle_->db, sql, -1, &stmt, nullptr) != SQLITE_OK)
            return;
        sqlite3_bind_text(stmt, 1, ns_.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, key.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
        if (hot_)
            hot_->del(cacheKey(key));
#else
        (void)key;
#endif
    }

    // 按前缀遍历当前命名空间下的键值对。
    // 调用方式：传入 prefix，返回按 key 排序的结果列表。
    // 实现思路：SQL LIKE 查询并逐行解析 JSON。
    // 注意事项：结果可能较大，调用方应控制 prefix 粒度。
    std::vector<std::pair<std::string, json>> entries(const std::string &prefix) override
    {
        std::vector<std::pair<std::string, json>> out;
#ifdef HAVE_SQLITE
        if (!handle_ || !handle_->db)
            return out;
        if (dbLogger().enabled())
            dbLogger().log(LoggerCXX::Type::COMPUTE, std::string("kv entries ns=") + ns_ + " prefix=" + prefix);
        std::lock_guard<std::mutex> lock(handle_->mu);
        sqlite3_stmt *stmt = nullptr;
        std::string like = prefix;
        like += "%";
        const char *sql = "SELECT key, value FROM kv_store WHERE namespace=? AND key LIKE ? ORDER BY key";
        if (sqlite3_prepare_v2(handle_->db, sql, -1, &stmt, nullptr) != SQLITE_OK)
            return out;
        sqlite3_bind_text(stmt, 1, ns_.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, like.c_str(), -1, SQLITE_TRANSIENT);
        while (sqlite3_step(stmt) == SQLITE_ROW)
        {
            const unsigned char *k = sqlite3_column_text(stmt, 0);
            const unsigned char *v = sqlite3_column_text(stmt, 1);
            std::string key = k ? (const char *)k : "";
            std::string val = v ? (const char *)v : "";
            auto parsed = parseJsonValue(val);
            out.push_back({key, parsed.value_or(json())});
        }
        sqlite3_finalize(stmt);
#else
        (void)prefix;
#endif
        return out;
    }

private:
    std::shared_ptr<DbHandle> handle_;
    std::string ns_;
    fs::path legacyDir_;
    std::shared_ptr<HotCache> hot_;
    std::unordered_map<std::string, int> hits_;
    int hotThreshold_{6};
    int hotTtlSeconds_{3600};

    // 生成热缓存专用键名。
    // 调用方式：内部用于 get/put/del 的缓存层映射。
    // 实现思路：使用命名空间+业务 key 拼接成唯一键。
    // 注意事项：返回值不做转义，需保证输入 key 可安全拼接。
    std::string cacheKey(const std::string &key) const
    {
        return "kv:" + ns_ + ":" + key;
    }

    // 提升本地命中计数，用于后续热点判断。
    // 调用方式：在缓存命中或访问路径中内部调用。
    // 实现思路：命中计数字典按 key 自增。
    // 注意事项：当前实现仅保存在内存，不持久化。
    void promote(const std::string &key)
    {
        auto &h = hits_[key];
        h += 1;
    }

    // 若存在旧 JSON 文件则迁移到 SQLite。
    // 调用方式：构造时自动调用，一般无需手动触发。
    // 实现思路：先检测命名空间是否已有数据，空时导入旧文件。
    // 注意事项：迁移过程容错处理，异常不会中断主流程。
    void migrateIfNeeded()
    {
#ifdef HAVE_SQLITE
        if (!handle_ || !handle_->db)
            return;
        if (legacyDir_.empty())
            return;
        fs::path legacy = legacyDir_ / (ns_ + ".json");
        if (!fs::exists(legacy))
            return;
        // check if namespace already has data
        {
            std::lock_guard<std::mutex> lock(handle_->mu);
            sqlite3_stmt *stmt = nullptr;
            const char *sql = "SELECT COUNT(1) FROM kv_store WHERE namespace=?";
            if (sqlite3_prepare_v2(handle_->db, sql, -1, &stmt, nullptr) != SQLITE_OK)
                return;
            sqlite3_bind_text(stmt, 1, ns_.c_str(), -1, SQLITE_TRANSIENT);
            int count = 0;
            if (sqlite3_step(stmt) == SQLITE_ROW)
                count = sqlite3_column_int(stmt, 0);
            sqlite3_finalize(stmt);
            if (count > 0)
                return;
        }
        try
        {
            std::ifstream in(legacy);
            json j;
            in >> j;
            if (!j.is_object())
                return;
            for (auto it = j.begin(); it != j.end(); ++it)
            {
                put(it.key(), it.value());
            }
        }
        catch (...)
        {
        }
#endif
    }
};

// 构造数据库管理器并记录配置。
// 调用方式：传入 SQLite 路径、旧数据目录与 Redis 配置。
// 实现思路：仅保存参数，真正连接在 open() 中完成。
// 注意事项：构造阶段不访问磁盘与网络，成本较低。
Database079::Database079(fs::path dbPath,
                         fs::path legacyDir,
                         std::string redisUrl,
                         int redisDb,
                         std::string redisPrefix)
    : dbPath_(std::move(dbPath)),
      legacyDir_(std::move(legacyDir)),
      redisUrl_(std::move(redisUrl)),
      redisDb_(redisDb),
      redisPrefix_(std::move(redisPrefix)) {}

// 打开数据库并初始化所需表结构。
// 调用方式：服务启动时先调用，成功返回 true。
// 实现思路：连接 SQLite、执行 PRAGMA、建表并初始化热缓存。
// 注意事项：在未启用 SQLite 编译选项时会返回 false。
bool Database079::open()
{
#ifdef HAVE_SQLITE
    handle_ = std::make_shared<DbHandle>();
    if (sqlite3_open(dbPath_.string().c_str(), &handle_->db) != SQLITE_OK)
    {
        if (handle_->db)
            sqlite3_close(handle_->db);
        handle_.reset();
        if (dbLogger().enabled())
            dbLogger().log(LoggerCXX::Type::ERROR, std::string("db open failed path=") + dbPath_.string());
        return false;
    }
    execSql(handle_->db, "PRAGMA journal_mode=WAL;");
    execSql(handle_->db, "PRAGMA synchronous=NORMAL;");
    execSql(handle_->db,
            "CREATE TABLE IF NOT EXISTS kv_store("
            "namespace TEXT NOT NULL,"
            "key TEXT NOT NULL,"
            "value TEXT NOT NULL,"
            "updatedAt INTEGER,"
            "PRIMARY KEY(namespace,key)"
            ");");
    execSql(handle_->db,
            "CREATE TABLE IF NOT EXISTS inference_cache("
            "key TEXT PRIMARY KEY,"
            "value TEXT NOT NULL,"
            "updatedAt INTEGER"
            ");");
        execSql(handle_->db,
            "CREATE TABLE IF NOT EXISTS inference_cache_stats("
            "key TEXT PRIMARY KEY,"
            "totalHits INTEGER DEFAULT 0,"
            "hotHits INTEGER DEFAULT 0,"
            "coldLoads INTEGER DEFAULT 0,"
            "writes INTEGER DEFAULT 0,"
            "lastAccessMs INTEGER DEFAULT 0,"
            "lastPromotedMs INTEGER DEFAULT 0,"
            "isHot INTEGER DEFAULT 0"
            ");");
    hot_ = std::make_shared<HotCache>(redisUrl_, redisDb_, redisPrefix_ + ":");
    opened_ = true;
    if (dbLogger().enabled())
        dbLogger().log(LoggerCXX::Type::LOG, std::string("db opened path=") + dbPath_.string());
    return true;
#else
    opened_ = false;
    if (dbLogger().enabled())
        dbLogger().log(LoggerCXX::Type::WARNING, "db open skipped: sqlite disabled");
    return false;
#endif
}

// 关闭数据库连接并释放句柄。
// 调用方式：进程退出或资源回收时调用。
// 实现思路：关闭 sqlite3 连接并重置共享句柄状态。
// 注意事项：多次调用安全；关闭后需重新 open 才可使用。
void Database079::close()
{
#ifdef HAVE_SQLITE
    if (handle_ && handle_->db)
    {
        sqlite3_close(handle_->db);
        handle_->db = nullptr;
    }
    handle_.reset();
    opened_ = false;
    if (dbLogger().enabled())
        dbLogger().log(LoggerCXX::Type::LOG, "db closed");
#endif
}

void Database079::configureInferenceSwap(std::size_t hotLimit, int promoteHits, int hotTtlSeconds, int rebalanceEvery)
{
    std::lock_guard<std::mutex> lock(swapMu_);
    inferenceHotLimit_ = std::max<std::size_t>(32, hotLimit);
    inferencePromoteHits_ = std::max(1, promoteHits);
    inferenceHotTtlSeconds_ = std::max(60, hotTtlSeconds);
    inferenceRebalanceEvery_ = std::max(8, rebalanceEvery);
}

nlohmann::json Database079::getSwapStats() const
{
    std::lock_guard<std::mutex> lock(swapMu_);
    return {
        {"hotLimit", inferenceHotLimit_},
        {"promoteHits", inferencePromoteHits_},
        {"hotTtlSeconds", inferenceHotTtlSeconds_},
        {"rebalanceEvery", inferenceRebalanceEvery_},
        {"accesses", inferenceAccesses_},
        {"hotHits", inferenceHotHits_},
        {"coldLoads", inferenceColdLoads_},
        {"writes", inferenceWrites_},
        {"promotions", inferencePromotions_},
        {"demotions", inferenceDemotions_},
        {"trackedKeys", recentInferenceTouches_.size()}
    };
}

// 创建指定命名空间的键值存储实例。
// 调用方式：业务侧按模块名/场景名传入 namespace。
// 实现思路：确保数据库可用后返回 SqliteKeyValueStore。
// 注意事项：若 SQLite 不可用将返回空指针。
std::shared_ptr<KeyValueStore> Database079::createStore(const std::string &name)
{
#ifdef HAVE_SQLITE
    if (!handle_ || !handle_->db)
    {
        if (!open())
            return nullptr;
    }
    if (!hot_)
        hot_ = std::make_shared<HotCache>(redisUrl_, redisDb_, redisPrefix_ + ":");
    if (dbLogger().enabled())
        dbLogger().log(LoggerCXX::Type::COMPUTE, std::string("db createStore namespace=") + name);
    return std::make_shared<SqliteKeyValueStore>(handle_, name, legacyDir_, hot_);
#else
    (void)name;
    if (dbLogger().enabled())
        dbLogger().log(LoggerCXX::Type::WARNING, "db createStore skipped: sqlite disabled");
    return nullptr;
#endif
}

// 读取推理缓存。
// 调用方式：传入 key，命中时通过 out 返回 JSON。
// 实现思路：优先 Redis，未命中再查 SQLite 并回填。
// 注意事项：解析失败会返回 false，调用方需回退到重算路径。
bool Database079::getInferenceCache(const std::string &key, json &out)
{
#ifdef HAVE_SQLITE
    if (!handle_ || !handle_->db)
        return false;
    if (hot_)
    {
        std::string raw;
        if (hot_->get("infer:" + key, raw))
        {
            try
            {
                out = json::parse(raw);
                {
                    std::lock_guard<std::mutex> swapLock(swapMu_);
                    inferenceAccesses_ += 1;
                    inferenceHotHits_ += 1;
                    recentInferenceTouches_[key] += 1;
                }
                std::lock_guard<std::mutex> lock(handle_->mu);
                sqlite3_stmt *statStmt = nullptr;
                const char *statSql =
                    "INSERT INTO inference_cache_stats(key,totalHits,hotHits,coldLoads,writes,lastAccessMs,lastPromotedMs,isHot) VALUES(?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(key) DO UPDATE SET totalHits=totalHits+1, hotHits=hotHits+1, lastAccessMs=excluded.lastAccessMs, isHot=1";
                if (sqlite3_prepare_v2(handle_->db, statSql, -1, &statStmt, nullptr) == SQLITE_OK)
                {
                    sqlite3_bind_text(statStmt, 1, key.c_str(), -1, SQLITE_TRANSIENT);
                    sqlite3_bind_int64(statStmt, 2, 1);
                    sqlite3_bind_int64(statStmt, 3, 1);
                    sqlite3_bind_int64(statStmt, 4, 0);
                    sqlite3_bind_int64(statStmt, 5, 0);
                    sqlite3_bind_int64(statStmt, 6, nowMs());
                    sqlite3_bind_int64(statStmt, 7, nowMs());
                    sqlite3_bind_int(statStmt, 8, 1);
                    sqlite3_step(statStmt);
                    sqlite3_finalize(statStmt);
                }
                if (dbLogger().enabled())
                    dbLogger().log(LoggerCXX::Type::COMPUTE, std::string("db infer cache hit key=") + key);
                return true;
            }
            catch (...)
            {
                if (dbLogger().enabled())
                    dbLogger().log(LoggerCXX::Type::WARNING, std::string("db infer cache parse fail key=") + key);
                return false;
            }
        }
    }
    std::lock_guard<std::mutex> lock(handle_->mu);
    sqlite3_stmt *stmt = nullptr;
    const char *sql = "SELECT value FROM inference_cache WHERE key=?";
    if (sqlite3_prepare_v2(handle_->db, sql, -1, &stmt, nullptr) != SQLITE_OK)
        return false;
    sqlite3_bind_text(stmt, 1, key.c_str(), -1, SQLITE_TRANSIENT);
    bool ok = false;
    bool loadedCold = false;
    std::vector<std::string> demoteKeys;
    std::string coldRaw;
    if (sqlite3_step(stmt) == SQLITE_ROW)
    {
        const unsigned char *v = sqlite3_column_text(stmt, 0);
        std::string val = v ? (const char *)v : "";
        try
        {
            out = json::parse(val);
            ok = true;
            loadedCold = true;
            coldRaw = val;
            if (dbLogger().enabled())
                dbLogger().log(LoggerCXX::Type::COMPUTE, std::string("db infer cache load key=") + key);
        }
        catch (...)
        {
            ok = false;
            if (dbLogger().enabled())
                dbLogger().log(LoggerCXX::Type::WARNING, std::string("db infer row parse fail key=") + key);
        }
    }
    sqlite3_finalize(stmt);
    bool promoteToHot = false;
    if (ok && loadedCold)
    {
        std::lock_guard<std::mutex> swapLock(swapMu_);
        inferenceAccesses_ += 1;
        inferenceColdLoads_ += 1;
        int &touch = recentInferenceTouches_[key];
        touch += 1;
        promoteToHot = touch >= inferencePromoteHits_;
    }
    {
        std::lock_guard<std::mutex> lock(handle_->mu);
        if (ok && loadedCold)
        {
            sqlite3_stmt *statStmt = nullptr;
            const char *statSql =
                "INSERT INTO inference_cache_stats(key,totalHits,hotHits,coldLoads,writes,lastAccessMs,lastPromotedMs,isHot) VALUES(?,?,?,?,?,?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET totalHits=totalHits+1, coldLoads=coldLoads+1, lastAccessMs=excluded.lastAccessMs, lastPromotedMs=CASE WHEN excluded.isHot=1 THEN excluded.lastPromotedMs ELSE lastPromotedMs END, isHot=MAX(isHot, excluded.isHot)";
            if (sqlite3_prepare_v2(handle_->db, statSql, -1, &statStmt, nullptr) == SQLITE_OK)
            {
                sqlite3_bind_text(statStmt, 1, key.c_str(), -1, SQLITE_TRANSIENT);
                sqlite3_bind_int64(statStmt, 2, 1);
                sqlite3_bind_int64(statStmt, 3, 0);
                sqlite3_bind_int64(statStmt, 4, 1);
                sqlite3_bind_int64(statStmt, 5, 0);
                sqlite3_bind_int64(statStmt, 6, nowMs());
                sqlite3_bind_int64(statStmt, 7, promoteToHot ? nowMs() : 0);
                sqlite3_bind_int(statStmt, 8, promoteToHot ? 1 : 0);
                sqlite3_step(statStmt);
                sqlite3_finalize(statStmt);
            }
        }

        bool shouldRebalance = false;
        {
            std::lock_guard<std::mutex> swapLock(swapMu_);
            shouldRebalance = inferenceRebalanceEvery_ > 0 && (inferenceAccesses_ % (std::uint64_t)inferenceRebalanceEvery_) == 0;
        }
        if (shouldRebalance)
        {
            sqlite3_stmt *countStmt = nullptr;
            const char *countSql = "SELECT COUNT(1) FROM inference_cache_stats WHERE isHot=1";
            int hotCount = 0;
            if (sqlite3_prepare_v2(handle_->db, countSql, -1, &countStmt, nullptr) == SQLITE_OK)
            {
                if (sqlite3_step(countStmt) == SQLITE_ROW)
                    hotCount = sqlite3_column_int(countStmt, 0);
                sqlite3_finalize(countStmt);
            }
            if (hotCount > (int)inferenceHotLimit_)
            {
                sqlite3_stmt *selectStmt = nullptr;
                std::string selectSql = "SELECT key FROM inference_cache_stats WHERE isHot=1 ORDER BY lastAccessMs ASC LIMIT ?";
                if (sqlite3_prepare_v2(handle_->db, selectSql.c_str(), -1, &selectStmt, nullptr) == SQLITE_OK)
                {
                    sqlite3_bind_int(selectStmt, 1, hotCount - (int)inferenceHotLimit_);
                    while (sqlite3_step(selectStmt) == SQLITE_ROW)
                    {
                        const unsigned char *rawKey = sqlite3_column_text(selectStmt, 0);
                        if (rawKey)
                            demoteKeys.emplace_back(reinterpret_cast<const char *>(rawKey));
                    }
                    sqlite3_finalize(selectStmt);
                }
                sqlite3_stmt *demoteStmt = nullptr;
                const char *demoteSql = "UPDATE inference_cache_stats SET isHot=0 WHERE key=?";
                if (sqlite3_prepare_v2(handle_->db, demoteSql, -1, &demoteStmt, nullptr) == SQLITE_OK)
                {
                    for (const auto &demoteKey : demoteKeys)
                    {
                        sqlite3_reset(demoteStmt);
                        sqlite3_clear_bindings(demoteStmt);
                        sqlite3_bind_text(demoteStmt, 1, demoteKey.c_str(), -1, SQLITE_TRANSIENT);
                        sqlite3_step(demoteStmt);
                    }
                    sqlite3_finalize(demoteStmt);
                }
            }
        }
    }
    if (ok && promoteToHot && hot_)
    {
        hot_->set("infer:" + key, coldRaw.empty() ? out.dump() : coldRaw, inferenceHotTtlSeconds_);
        std::lock_guard<std::mutex> swapLock(swapMu_);
        inferencePromotions_ += 1;
    }
    if (hot_)
    {
        for (const auto &demoteKey : demoteKeys)
            hot_->del("infer:" + demoteKey);
        if (!demoteKeys.empty())
        {
            std::lock_guard<std::mutex> swapLock(swapMu_);
            inferenceDemotions_ += demoteKeys.size();
        }
    }
    return ok;
#else
    (void)key;
    (void)out;
    return false;
#endif
}

// 写入推理缓存。
// 调用方式：传入 key、json 内容和 TTL 秒数。
// 实现思路：持久化写入 inference_cache，并同步热缓存。
// 注意事项：TTL 仅作用于热缓存，SQLite 仍会保留数据。
void Database079::setInferenceCache(const std::string &key, const json &value, int ttlSeconds)
{
#ifdef HAVE_SQLITE
    if (!handle_ || !handle_->db)
        return;
    std::string raw = value.dump();
    {
        std::lock_guard<std::mutex> lock(handle_->mu);
        sqlite3_stmt *stmt = nullptr;
        const char *sql = "INSERT INTO inference_cache(key,value,updatedAt) VALUES(?,?,?)"
                          " ON CONFLICT(key) DO UPDATE SET value=excluded.value, updatedAt=excluded.updatedAt";
        if (sqlite3_prepare_v2(handle_->db, sql, -1, &stmt, nullptr) != SQLITE_OK)
            return;
        sqlite3_bind_text(stmt, 1, key.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, raw.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int64(stmt, 3, nowMs());
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }
    bool keepHot = false;
    {
        std::lock_guard<std::mutex> swapLock(swapMu_);
        inferenceWrites_ += 1;
        int &touch = recentInferenceTouches_[key];
        touch += 1;
        keepHot = touch >= inferencePromoteHits_;
    }
    {
        std::lock_guard<std::mutex> lock(handle_->mu);
        sqlite3_stmt *statStmt = nullptr;
        const char *statSql =
            "INSERT INTO inference_cache_stats(key,totalHits,hotHits,coldLoads,writes,lastAccessMs,lastPromotedMs,isHot) VALUES(?,?,?,?,?,?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET writes=writes+1, lastAccessMs=excluded.lastAccessMs, lastPromotedMs=CASE WHEN excluded.isHot=1 THEN excluded.lastPromotedMs ELSE lastPromotedMs END, isHot=MAX(isHot, excluded.isHot)";
        if (sqlite3_prepare_v2(handle_->db, statSql, -1, &statStmt, nullptr) == SQLITE_OK)
        {
            sqlite3_bind_text(statStmt, 1, key.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_int64(statStmt, 2, 0);
            sqlite3_bind_int64(statStmt, 3, 0);
            sqlite3_bind_int64(statStmt, 4, 0);
            sqlite3_bind_int64(statStmt, 5, 1);
            sqlite3_bind_int64(statStmt, 6, nowMs());
            sqlite3_bind_int64(statStmt, 7, keepHot ? nowMs() : 0);
            sqlite3_bind_int(statStmt, 8, keepHot ? 1 : 0);
            sqlite3_step(statStmt);
            sqlite3_finalize(statStmt);
        }
    }
    if (hot_ && keepHot)
    {
        hot_->set("infer:" + key, raw, std::max(ttlSeconds, inferenceHotTtlSeconds_));
        std::lock_guard<std::mutex> swapLock(swapMu_);
        inferencePromotions_ += 1;
    }
    if (dbLogger().enabled())
        dbLogger().log(LoggerCXX::Type::COMPUTE, std::string("db infer cache set key=") + key + " ttl=" + std::to_string(ttlSeconds));
#else
    (void)key;
    (void)value;
    (void)ttlSeconds;
#endif
}

// 判断给定存储对象是否为 SQLite 实现。
// 调用方式：用于运行时分支判断或调试校验。
// 实现思路：使用 dynamic_pointer_cast 进行类型识别。
// 注意事项：仅用于 RTTI 场景，不应作为核心业务逻辑依赖。
bool Database079::isSqliteStore(const std::shared_ptr<KeyValueStore> &store)
{
    return std::dynamic_pointer_cast<SqliteKeyValueStore>(store) != nullptr;
}
