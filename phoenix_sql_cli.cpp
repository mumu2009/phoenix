/* phoenix_sql_cli.cpp - SQL CLI implementation
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

#include "phoenix_sql_cli.hpp"

#include <sqlite3.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace phoenix_sql_cli
{

namespace
{

struct SourceDbConfig
{
    std::string sourceId;
    fs::path path;
};

struct JsonDocumentRow
{
    std::string collection;
    std::string documentId;
    fs::path path;
    std::string contentJson;
    bool writable{true};
};

class SqliteHandle
{
public:
    SqliteHandle() = default;

    explicit SqliteHandle(sqlite3 *db)
        : db_(db)
    {
    }

    ~SqliteHandle()
    {
        reset();
    }

    SqliteHandle(const SqliteHandle &) = delete;
    SqliteHandle &operator=(const SqliteHandle &) = delete;

    SqliteHandle(SqliteHandle &&other) noexcept
        : db_(other.db_)
    {
        other.db_ = nullptr;
    }

    SqliteHandle &operator=(SqliteHandle &&other) noexcept
    {
        if (this != &other)
        {
            reset();
            db_ = other.db_;
            other.db_ = nullptr;
        }
        return *this;
    }

    sqlite3 *get() const { return db_; }

    sqlite3 *release()
    {
        sqlite3 *db = db_;
        db_ = nullptr;
        return db;
    }

    void reset(sqlite3 *db = nullptr)
    {
        if (db_)
            sqlite3_close(db_);
        db_ = db;
    }

private:
    sqlite3 *db_{nullptr};
};

class StatementHandle
{
public:
    StatementHandle() = default;

    explicit StatementHandle(sqlite3_stmt *stmt)
        : stmt_(stmt)
    {
    }

    ~StatementHandle()
    {
        if (stmt_)
            sqlite3_finalize(stmt_);
    }

    StatementHandle(const StatementHandle &) = delete;
    StatementHandle &operator=(const StatementHandle &) = delete;

    StatementHandle(StatementHandle &&other) noexcept
        : stmt_(other.stmt_)
    {
        other.stmt_ = nullptr;
    }

    StatementHandle &operator=(StatementHandle &&other) noexcept
    {
        if (this != &other)
        {
            if (stmt_)
                sqlite3_finalize(stmt_);
            stmt_ = other.stmt_;
            other.stmt_ = nullptr;
        }
        return *this;
    }

    sqlite3_stmt *get() const { return stmt_; }

private:
    sqlite3_stmt *stmt_{nullptr};
};

std::string trimCopy(const std::string &value)
{
    size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start])))
        ++start;
    size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1])))
        --end;
    return value.substr(start, end - start);
}

std::string lowerCopy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch)
                   { return static_cast<char>(std::tolower(ch)); });
    return value;
}

std::string documentKey(const std::string &collection, const std::string &documentId)
{
    return collection + "\n" + documentId;
}

fs::path resolveAbsolute(const fs::path &path)
{
    std::error_code ec;
    fs::path absolute = fs::absolute(path, ec);
    return ec ? path : absolute;
}

std::string readTextFile(const fs::path &path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("failed to open file: " + path.string());
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

void writeTextFile(const fs::path &path, const std::string &text)
{
    if (!path.parent_path().empty())
        fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("failed to write file: " + path.string());
    output << text;
}

bool fileExists(const fs::path &path)
{
    std::error_code ec;
    return fs::exists(path, ec) && fs::is_regular_file(path, ec);
}

void removeFileIfExists(const fs::path &path)
{
    std::error_code ec;
    fs::remove(path, ec);
}

void execSql(sqlite3 *db, const std::string &sql)
{
    char *err = nullptr;
    int rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &err);
    if (rc != SQLITE_OK)
    {
        std::string message = err ? err : "sqlite exec failed";
        if (err)
            sqlite3_free(err);
        throw std::runtime_error(message);
    }
}

bool tableExists(sqlite3 *db, const std::string &tableName)
{
    const char *sql = "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?";
    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK)
        return false;
    StatementHandle handle(stmt);
    sqlite3_bind_text(stmt, 1, tableName.c_str(), -1, SQLITE_TRANSIENT);
    return sqlite3_step(stmt) == SQLITE_ROW;
}

SqliteHandle openSqlite(const fs::path &path, int flags)
{
    sqlite3 *db = nullptr;
    const std::string native = resolveAbsolute(path).string();
    if (sqlite3_open_v2(native.c_str(), &db, flags, nullptr) != SQLITE_OK)
    {
        std::string error = db ? sqlite3_errmsg(db) : std::string("sqlite open failed");
        if (db)
            sqlite3_close(db);
        throw std::runtime_error(error + ": " + native);
    }
    return SqliteHandle(db);
}

SqliteHandle openMirrorDb()
{
    sqlite3 *db = nullptr;
    if (sqlite3_open(":memory:", &db) != SQLITE_OK)
    {
        std::string error = db ? sqlite3_errmsg(db) : std::string("failed to open mirror db");
        if (db)
            sqlite3_close(db);
        throw std::runtime_error(error);
    }
    return SqliteHandle(db);
}

void ensureMirrorSchema(sqlite3 *db)
{
    execSql(db,
            "CREATE TABLE IF NOT EXISTS internal_catalog("
            "table_name TEXT PRIMARY KEY,"
            "source TEXT NOT NULL,"
            "writable INTEGER NOT NULL,"
            "description TEXT NOT NULL"
            ");");
    execSql(db,
            "CREATE TABLE IF NOT EXISTS kv_entries("
            "source_db TEXT NOT NULL,"
            "namespace TEXT NOT NULL,"
            "key TEXT NOT NULL,"
            "value_json TEXT NOT NULL,"
            "updated_at INTEGER,"
            "PRIMARY KEY(source_db, namespace, key)"
            ");");
    execSql(db,
            "CREATE TABLE IF NOT EXISTS inference_cache("
            "source_db TEXT NOT NULL,"
            "key TEXT NOT NULL,"
            "value_json TEXT NOT NULL,"
            "updated_at INTEGER,"
            "PRIMARY KEY(source_db, key)"
            ");");
    execSql(db,
            "CREATE TABLE IF NOT EXISTS inference_cache_stats("
            "source_db TEXT NOT NULL,"
            "key TEXT NOT NULL,"
            "total_hits INTEGER DEFAULT 0,"
            "hot_hits INTEGER DEFAULT 0,"
            "cold_loads INTEGER DEFAULT 0,"
            "writes INTEGER DEFAULT 0,"
            "last_access_ms INTEGER DEFAULT 0,"
            "last_promoted_ms INTEGER DEFAULT 0,"
            "is_hot INTEGER DEFAULT 0,"
            "PRIMARY KEY(source_db, key)"
            ");");
    execSql(db,
            "CREATE TABLE IF NOT EXISTS memebarrier_phrase_feedback_positive("
            "phrase TEXT PRIMARY KEY,"
            "offset REAL NOT NULL"
            ");");
    execSql(db,
            "CREATE TABLE IF NOT EXISTS memebarrier_phrase_blocklist("
            "phrase TEXT PRIMARY KEY"
            ");");
    execSql(db,
            "CREATE TABLE IF NOT EXISTS json_documents("
            "collection TEXT NOT NULL,"
            "document_id TEXT NOT NULL,"
            "path TEXT NOT NULL,"
            "content_json TEXT NOT NULL,"
            "writable INTEGER NOT NULL DEFAULT 1,"
            "PRIMARY KEY(collection, document_id)"
            ");");

    execSql(db, "DELETE FROM internal_catalog;");
    execSql(db,
            "INSERT INTO internal_catalog(table_name, source, writable, description) VALUES"
            "('kv_entries', 'runtime_store/*.sqlite::kv_store', 1, 'Main and world-model namespace key/value rows mirrored from kv_store'),"
            "('inference_cache', 'runtime_store/*.sqlite::inference_cache', 1, 'Inference cache rows mirrored from SQLite'),"
            "('inference_cache_stats', 'runtime_store/*.sqlite::inference_cache_stats', 1, 'Inference cache hit/promotion stats mirrored from SQLite'),"
            "('memebarrier_phrase_feedback_positive', 'runtime_store/memebarrier_phrase_feedback.json', 1, 'Persistent MemeBarrier positive phrase offsets'),"
            "('memebarrier_phrase_blocklist', 'runtime_store/memebarrier_phrase_blocklist.json', 1, 'Persistent MemeBarrier blocked phrases'),"
            "('json_documents', 'runtime_store/*.json + GGUF_models/manifests/**/*.json', 1, 'Generic JSON-backed internal documents, including runtime status, model lifecycle, brain maps, structured exports, runtime config, and model manifests')"
            ";");
}

void ensureSourceDbSchema(sqlite3 *db)
{
    execSql(db,
            "CREATE TABLE IF NOT EXISTS kv_store("
            "namespace TEXT NOT NULL,"
            "key TEXT NOT NULL,"
            "value TEXT NOT NULL,"
            "updatedAt INTEGER,"
            "PRIMARY KEY(namespace,key)"
            ");");
    execSql(db,
            "CREATE TABLE IF NOT EXISTS inference_cache("
            "key TEXT PRIMARY KEY,"
            "value TEXT NOT NULL,"
            "updatedAt INTEGER"
            ");");
    execSql(db,
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
}

void bindText(sqlite3_stmt *stmt, int index, const std::string &value)
{
    sqlite3_bind_text(stmt, index, value.c_str(), -1, SQLITE_TRANSIENT);
}

void bindInt64(sqlite3_stmt *stmt, int index, std::int64_t value)
{
    sqlite3_bind_int64(stmt, index, static_cast<sqlite3_int64>(value));
}

void insertKvMirrorRow(sqlite3 *mirrorDb,
                       const std::string &sourceId,
                       const std::string &nameSpace,
                       const std::string &key,
                       const std::string &valueJson,
                       std::int64_t updatedAt)
{
    const char *sql =
        "INSERT OR REPLACE INTO kv_entries(source_db, namespace, key, value_json, updated_at) VALUES(?,?,?,?,?)";
    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(mirrorDb, sql, -1, &stmt, nullptr) != SQLITE_OK)
        throw std::runtime_error(sqlite3_errmsg(mirrorDb));
    StatementHandle handle(stmt);
    bindText(stmt, 1, sourceId);
    bindText(stmt, 2, nameSpace);
    bindText(stmt, 3, key);
    bindText(stmt, 4, valueJson);
    bindInt64(stmt, 5, updatedAt);
    if (sqlite3_step(stmt) != SQLITE_DONE)
        throw std::runtime_error(sqlite3_errmsg(mirrorDb));
}

void insertInferenceCacheMirrorRow(sqlite3 *mirrorDb,
                                   const std::string &sourceId,
                                   const std::string &key,
                                   const std::string &valueJson,
                                   std::int64_t updatedAt)
{
    const char *sql =
        "INSERT OR REPLACE INTO inference_cache(source_db, key, value_json, updated_at) VALUES(?,?,?,?)";
    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(mirrorDb, sql, -1, &stmt, nullptr) != SQLITE_OK)
        throw std::runtime_error(sqlite3_errmsg(mirrorDb));
    StatementHandle handle(stmt);
    bindText(stmt, 1, sourceId);
    bindText(stmt, 2, key);
    bindText(stmt, 3, valueJson);
    bindInt64(stmt, 4, updatedAt);
    if (sqlite3_step(stmt) != SQLITE_DONE)
        throw std::runtime_error(sqlite3_errmsg(mirrorDb));
}

void insertInferenceStatsMirrorRow(sqlite3 *mirrorDb,
                                   const std::string &sourceId,
                                   const std::string &key,
                                   std::int64_t totalHits,
                                   std::int64_t hotHits,
                                   std::int64_t coldLoads,
                                   std::int64_t writes,
                                   std::int64_t lastAccessMs,
                                   std::int64_t lastPromotedMs,
                                   bool isHot)
{
    const char *sql =
        "INSERT OR REPLACE INTO inference_cache_stats(source_db, key, total_hits, hot_hits, cold_loads, writes, last_access_ms, last_promoted_ms, is_hot) VALUES(?,?,?,?,?,?,?,?,?)";
    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(mirrorDb, sql, -1, &stmt, nullptr) != SQLITE_OK)
        throw std::runtime_error(sqlite3_errmsg(mirrorDb));
    StatementHandle handle(stmt);
    bindText(stmt, 1, sourceId);
    bindText(stmt, 2, key);
    bindInt64(stmt, 3, totalHits);
    bindInt64(stmt, 4, hotHits);
    bindInt64(stmt, 5, coldLoads);
    bindInt64(stmt, 6, writes);
    bindInt64(stmt, 7, lastAccessMs);
    bindInt64(stmt, 8, lastPromotedMs);
    bindInt64(stmt, 9, isHot ? 1 : 0);
    if (sqlite3_step(stmt) != SQLITE_DONE)
        throw std::runtime_error(sqlite3_errmsg(mirrorDb));
}

void insertPhraseFeedbackRow(sqlite3 *mirrorDb, const std::string &phrase, double offset)
{
    const char *sql = "INSERT OR REPLACE INTO memebarrier_phrase_feedback_positive(phrase, offset) VALUES(?,?)";
    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(mirrorDb, sql, -1, &stmt, nullptr) != SQLITE_OK)
        throw std::runtime_error(sqlite3_errmsg(mirrorDb));
    StatementHandle handle(stmt);
    bindText(stmt, 1, phrase);
    sqlite3_bind_double(stmt, 2, offset);
    if (sqlite3_step(stmt) != SQLITE_DONE)
        throw std::runtime_error(sqlite3_errmsg(mirrorDb));
}

void insertPhraseBlocklistRow(sqlite3 *mirrorDb, const std::string &phrase)
{
    const char *sql = "INSERT OR REPLACE INTO memebarrier_phrase_blocklist(phrase) VALUES(?)";
    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(mirrorDb, sql, -1, &stmt, nullptr) != SQLITE_OK)
        throw std::runtime_error(sqlite3_errmsg(mirrorDb));
    StatementHandle handle(stmt);
    bindText(stmt, 1, phrase);
    if (sqlite3_step(stmt) != SQLITE_DONE)
        throw std::runtime_error(sqlite3_errmsg(mirrorDb));
}

void insertJsonDocumentRow(sqlite3 *mirrorDb,
                           const std::string &collection,
                           const std::string &documentId,
                           const fs::path &path,
                           const std::string &contentJson,
                           bool writable)
{
    const char *sql =
        "INSERT OR REPLACE INTO json_documents(collection, document_id, path, content_json, writable) VALUES(?,?,?,?,?)";
    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(mirrorDb, sql, -1, &stmt, nullptr) != SQLITE_OK)
        throw std::runtime_error(sqlite3_errmsg(mirrorDb));
    StatementHandle handle(stmt);
    bindText(stmt, 1, collection);
    bindText(stmt, 2, documentId);
    bindText(stmt, 3, path.generic_string());
    bindText(stmt, 4, contentJson);
    bindInt64(stmt, 5, writable ? 1 : 0);
    if (sqlite3_step(stmt) != SQLITE_DONE)
        throw std::runtime_error(sqlite3_errmsg(mirrorDb));
}

std::string readSqlScript(const Options &options)
{
    if (!options.sql.empty())
        return options.sql;
    if (!options.sqlFile.empty())
        return readTextFile(options.sqlFile);
    if (options.listTables)
        return "SELECT table_name, source, writable, description FROM internal_catalog ORDER BY table_name;";
    throw std::runtime_error("sql required: pass --sql, --file, or --list-tables");
}

Options normalizeOptions(Options options)
{
    options.runtimeDir = resolveAbsolute(options.runtimeDir);
    if (options.mainDbPath.empty())
        options.mainDbPath = options.runtimeDir / "ai_store.sqlite";
    else
        options.mainDbPath = resolveAbsolute(options.mainDbPath);
    if (options.worldModelDbPath.empty())
        options.worldModelDbPath = options.runtimeDir / "frontend_world_model.sqlite";
    else
        options.worldModelDbPath = resolveAbsolute(options.worldModelDbPath);
    options.ggufModelsDir = resolveAbsolute(options.ggufModelsDir);
    if (!options.sqlFile.empty())
        options.sqlFile = resolveAbsolute(options.sqlFile);
    return options;
}

class MirrorWorkspace
{
public:
    explicit MirrorWorkspace(const Options &options)
        : options_(normalizeOptions(options)), mirrorDb_(openMirrorDb())
    {
        ensureMirrorSchema(mirrorDb_.get());
    }

    void loadAll()
    {
        loadSqliteSource({"main", options_.mainDbPath});
        if (fileExists(options_.worldModelDbPath))
            loadSqliteSource({"world_model", options_.worldModelDbPath});
        loadPhraseFeedback();
        loadPhraseBlocklist();
        loadKnownJsonDocuments();
    }

    json executeSqlScript(const std::string &script, bool readOnly)
    {
        json results = json::array();
        const char *cursor = script.c_str();
        while (cursor && *cursor)
        {
            sqlite3_stmt *rawStmt = nullptr;
            const char *tail = nullptr;
            int rc = sqlite3_prepare_v2(mirrorDb_.get(), cursor, -1, &rawStmt, &tail);
            if (rc != SQLITE_OK)
                throw std::runtime_error(sqlite3_errmsg(mirrorDb_.get()));

            std::string statementText(cursor, tail ? (tail - cursor) : std::strlen(cursor));
            cursor = tail;
            StatementHandle stmt(rawStmt);
            if (!stmt.get())
                continue;
            statementText = trimCopy(statementText);
            if (statementText.empty())
                continue;

            const bool statementReadonly = sqlite3_stmt_readonly(stmt.get()) != 0;
            if (readOnly && !statementReadonly)
                throw std::runtime_error("read-only mode forbids write statement: " + statementText);

            json statementResult{{"sql", statementText}, {"readonly", statementReadonly}};
            const int columnCount = sqlite3_column_count(stmt.get());
            json rows = json::array();
            json columns = json::array();
            for (int i = 0; i < columnCount; ++i)
                columns.push_back(sqlite3_column_name(stmt.get(), i));

            while ((rc = sqlite3_step(stmt.get())) == SQLITE_ROW)
            {
                json row = json::object();
                for (int i = 0; i < columnCount; ++i)
                {
                    const std::string columnName = sqlite3_column_name(stmt.get(), i);
                    switch (sqlite3_column_type(stmt.get(), i))
                    {
                    case SQLITE_INTEGER:
                        row[columnName] = static_cast<std::int64_t>(sqlite3_column_int64(stmt.get(), i));
                        break;
                    case SQLITE_FLOAT:
                        row[columnName] = sqlite3_column_double(stmt.get(), i);
                        break;
                    case SQLITE_TEXT:
                        row[columnName] = reinterpret_cast<const char *>(sqlite3_column_text(stmt.get(), i));
                        break;
                    case SQLITE_NULL:
                        row[columnName] = nullptr;
                        break;
                    default:
                        row[columnName] = "<blob>";
                        break;
                    }
                }
                rows.push_back(row);
            }
            if (rc != SQLITE_DONE)
                throw std::runtime_error(sqlite3_errmsg(mirrorDb_.get()));

            statementResult["columns"] = columns;
            statementResult["rows"] = rows;
            statementResult["rowCount"] = rows.size();
            statementResult["changes"] = statementReadonly ? 0 : sqlite3_changes(mirrorDb_.get());
            results.push_back(statementResult);
        }
        return results;
    }

    void persistAll()
    {
        persistSqliteSource({"main", options_.mainDbPath});
        persistSqliteSource({"world_model", options_.worldModelDbPath}, fileExists(options_.worldModelDbPath));
        persistPhraseFeedback();
        persistPhraseBlocklist();
        persistJsonDocuments();
    }

    const Options &options() const { return options_; }

private:
    void loadSqliteSource(const SourceDbConfig &config)
    {
        if (!fileExists(config.path))
            return;
        SqliteHandle db = openSqlite(config.path, SQLITE_OPEN_READONLY);
        if (tableExists(db.get(), "kv_store"))
        {
            const char *sql = "SELECT namespace, key, value, COALESCE(updatedAt, 0) FROM kv_store ORDER BY namespace, key";
            sqlite3_stmt *stmt = nullptr;
            if (sqlite3_prepare_v2(db.get(), sql, -1, &stmt, nullptr) != SQLITE_OK)
                throw std::runtime_error(sqlite3_errmsg(db.get()));
            StatementHandle handle(stmt);
            while (sqlite3_step(stmt) == SQLITE_ROW)
            {
                const std::string nameSpace = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 0));
                const std::string key = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 1));
                const std::string valueJson = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 2));
                const std::int64_t updatedAt = sqlite3_column_int64(stmt, 3);
                insertKvMirrorRow(mirrorDb_.get(), config.sourceId, nameSpace, key, valueJson, updatedAt);
            }
        }
        if (tableExists(db.get(), "inference_cache"))
        {
            const char *sql = "SELECT key, value, COALESCE(updatedAt, 0) FROM inference_cache ORDER BY key";
            sqlite3_stmt *stmt = nullptr;
            if (sqlite3_prepare_v2(db.get(), sql, -1, &stmt, nullptr) != SQLITE_OK)
                throw std::runtime_error(sqlite3_errmsg(db.get()));
            StatementHandle handle(stmt);
            while (sqlite3_step(stmt) == SQLITE_ROW)
            {
                const std::string key = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 0));
                const std::string valueJson = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 1));
                const std::int64_t updatedAt = sqlite3_column_int64(stmt, 2);
                insertInferenceCacheMirrorRow(mirrorDb_.get(), config.sourceId, key, valueJson, updatedAt);
            }
        }
        if (tableExists(db.get(), "inference_cache_stats"))
        {
            const char *sql = "SELECT key, totalHits, hotHits, coldLoads, writes, lastAccessMs, lastPromotedMs, isHot FROM inference_cache_stats ORDER BY key";
            sqlite3_stmt *stmt = nullptr;
            if (sqlite3_prepare_v2(db.get(), sql, -1, &stmt, nullptr) != SQLITE_OK)
                throw std::runtime_error(sqlite3_errmsg(db.get()));
            StatementHandle handle(stmt);
            while (sqlite3_step(stmt) == SQLITE_ROW)
            {
                insertInferenceStatsMirrorRow(mirrorDb_.get(),
                                             config.sourceId,
                                             reinterpret_cast<const char *>(sqlite3_column_text(stmt, 0)),
                                             sqlite3_column_int64(stmt, 1),
                                             sqlite3_column_int64(stmt, 2),
                                             sqlite3_column_int64(stmt, 3),
                                             sqlite3_column_int64(stmt, 4),
                                             sqlite3_column_int64(stmt, 5),
                                             sqlite3_column_int64(stmt, 6),
                                             sqlite3_column_int64(stmt, 7) != 0);
            }
        }
        managedSourceDbs_[config.sourceId] = config.path;
    }

    void persistSqliteSource(const SourceDbConfig &config, bool forceCreate = true)
    {
        if (!forceCreate && managedSourceDbs_.find(config.sourceId) == managedSourceDbs_.end())
            return;
        if (!config.path.parent_path().empty())
            fs::create_directories(config.path.parent_path());
        SqliteHandle db = openSqlite(config.path, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE);
        ensureSourceDbSchema(db.get());
        execSql(db.get(), "BEGIN IMMEDIATE TRANSACTION;");
        try
        {
            execSql(db.get(), "DELETE FROM kv_store;");
            execSql(db.get(), "DELETE FROM inference_cache;");
            execSql(db.get(), "DELETE FROM inference_cache_stats;");

            persistKvRows(db.get(), config.sourceId);
            persistInferenceCacheRows(db.get(), config.sourceId);
            persistInferenceStatsRows(db.get(), config.sourceId);
            execSql(db.get(), "COMMIT;");
        }
        catch (...)
        {
            execSql(db.get(), "ROLLBACK;");
            throw;
        }
    }

    void persistKvRows(sqlite3 *targetDb, const std::string &sourceId)
    {
        const char *selectSql =
            "SELECT namespace, key, value_json, COALESCE(updated_at, 0) FROM kv_entries WHERE source_db=? ORDER BY namespace, key";
        sqlite3_stmt *selectStmt = nullptr;
        if (sqlite3_prepare_v2(mirrorDb_.get(), selectSql, -1, &selectStmt, nullptr) != SQLITE_OK)
            throw std::runtime_error(sqlite3_errmsg(mirrorDb_.get()));
        StatementHandle selectHandle(selectStmt);
        bindText(selectStmt, 1, sourceId);

        const char *insertSql =
            "INSERT INTO kv_store(namespace, key, value, updatedAt) VALUES(?,?,?,?)";
        sqlite3_stmt *insertStmt = nullptr;
        if (sqlite3_prepare_v2(targetDb, insertSql, -1, &insertStmt, nullptr) != SQLITE_OK)
            throw std::runtime_error(sqlite3_errmsg(targetDb));
        StatementHandle insertHandle(insertStmt);

        while (sqlite3_step(selectStmt) == SQLITE_ROW)
        {
            sqlite3_reset(insertStmt);
            sqlite3_clear_bindings(insertStmt);
            bindText(insertStmt, 1, reinterpret_cast<const char *>(sqlite3_column_text(selectStmt, 0)));
            bindText(insertStmt, 2, reinterpret_cast<const char *>(sqlite3_column_text(selectStmt, 1)));
            bindText(insertStmt, 3, reinterpret_cast<const char *>(sqlite3_column_text(selectStmt, 2)));
            bindInt64(insertStmt, 4, sqlite3_column_int64(selectStmt, 3));
            if (sqlite3_step(insertStmt) != SQLITE_DONE)
                throw std::runtime_error(sqlite3_errmsg(targetDb));
        }
    }

    void persistInferenceCacheRows(sqlite3 *targetDb, const std::string &sourceId)
    {
        const char *selectSql =
            "SELECT key, value_json, COALESCE(updated_at, 0) FROM inference_cache WHERE source_db=? ORDER BY key";
        sqlite3_stmt *selectStmt = nullptr;
        if (sqlite3_prepare_v2(mirrorDb_.get(), selectSql, -1, &selectStmt, nullptr) != SQLITE_OK)
            throw std::runtime_error(sqlite3_errmsg(mirrorDb_.get()));
        StatementHandle selectHandle(selectStmt);
        bindText(selectStmt, 1, sourceId);

        const char *insertSql = "INSERT INTO inference_cache(key, value, updatedAt) VALUES(?,?,?)";
        sqlite3_stmt *insertStmt = nullptr;
        if (sqlite3_prepare_v2(targetDb, insertSql, -1, &insertStmt, nullptr) != SQLITE_OK)
            throw std::runtime_error(sqlite3_errmsg(targetDb));
        StatementHandle insertHandle(insertStmt);

        while (sqlite3_step(selectStmt) == SQLITE_ROW)
        {
            sqlite3_reset(insertStmt);
            sqlite3_clear_bindings(insertStmt);
            bindText(insertStmt, 1, reinterpret_cast<const char *>(sqlite3_column_text(selectStmt, 0)));
            bindText(insertStmt, 2, reinterpret_cast<const char *>(sqlite3_column_text(selectStmt, 1)));
            bindInt64(insertStmt, 3, sqlite3_column_int64(selectStmt, 2));
            if (sqlite3_step(insertStmt) != SQLITE_DONE)
                throw std::runtime_error(sqlite3_errmsg(targetDb));
        }
    }

    void persistInferenceStatsRows(sqlite3 *targetDb, const std::string &sourceId)
    {
        const char *selectSql =
            "SELECT key, total_hits, hot_hits, cold_loads, writes, last_access_ms, last_promoted_ms, is_hot FROM inference_cache_stats WHERE source_db=? ORDER BY key";
        sqlite3_stmt *selectStmt = nullptr;
        if (sqlite3_prepare_v2(mirrorDb_.get(), selectSql, -1, &selectStmt, nullptr) != SQLITE_OK)
            throw std::runtime_error(sqlite3_errmsg(mirrorDb_.get()));
        StatementHandle selectHandle(selectStmt);
        bindText(selectStmt, 1, sourceId);

        const char *insertSql =
            "INSERT INTO inference_cache_stats(key, totalHits, hotHits, coldLoads, writes, lastAccessMs, lastPromotedMs, isHot) VALUES(?,?,?,?,?,?,?,?)";
        sqlite3_stmt *insertStmt = nullptr;
        if (sqlite3_prepare_v2(targetDb, insertSql, -1, &insertStmt, nullptr) != SQLITE_OK)
            throw std::runtime_error(sqlite3_errmsg(targetDb));
        StatementHandle insertHandle(insertStmt);

        while (sqlite3_step(selectStmt) == SQLITE_ROW)
        {
            sqlite3_reset(insertStmt);
            sqlite3_clear_bindings(insertStmt);
            bindText(insertStmt, 1, reinterpret_cast<const char *>(sqlite3_column_text(selectStmt, 0)));
            bindInt64(insertStmt, 2, sqlite3_column_int64(selectStmt, 1));
            bindInt64(insertStmt, 3, sqlite3_column_int64(selectStmt, 2));
            bindInt64(insertStmt, 4, sqlite3_column_int64(selectStmt, 3));
            bindInt64(insertStmt, 5, sqlite3_column_int64(selectStmt, 4));
            bindInt64(insertStmt, 6, sqlite3_column_int64(selectStmt, 5));
            bindInt64(insertStmt, 7, sqlite3_column_int64(selectStmt, 6));
            bindInt64(insertStmt, 8, sqlite3_column_int64(selectStmt, 7));
            if (sqlite3_step(insertStmt) != SQLITE_DONE)
                throw std::runtime_error(sqlite3_errmsg(targetDb));
        }
    }

    void loadPhraseFeedback()
    {
        const fs::path path = options_.runtimeDir / "memebarrier_phrase_feedback.json";
        feedbackPath_ = path;
        if (!fileExists(path))
            return;
        json doc = json::parse(readTextFile(path), nullptr, true, true);
        const json offsets = doc.value("positiveThresholdOffsets", json::object());
        if (!offsets.is_object())
            return;
        for (auto it = offsets.begin(); it != offsets.end(); ++it)
        {
            if (!it.value().is_number())
                continue;
            insertPhraseFeedbackRow(mirrorDb_.get(), it.key(), it.value().get<double>());
        }
    }

    void persistPhraseFeedback()
    {
        json doc;
        doc["positiveThresholdOffsets"] = json::object();
        const char *sql = "SELECT phrase, offset FROM memebarrier_phrase_feedback_positive ORDER BY phrase";
        sqlite3_stmt *stmt = nullptr;
        if (sqlite3_prepare_v2(mirrorDb_.get(), sql, -1, &stmt, nullptr) != SQLITE_OK)
            throw std::runtime_error(sqlite3_errmsg(mirrorDb_.get()));
        StatementHandle handle(stmt);
        while (sqlite3_step(stmt) == SQLITE_ROW)
        {
            const std::string phrase = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 0));
            const double offset = sqlite3_column_double(stmt, 1);
            doc["positiveThresholdOffsets"][phrase] = offset;
        }
        writeTextFile(feedbackPath_, doc.dump(2));
    }

    void loadPhraseBlocklist()
    {
        const fs::path path = options_.runtimeDir / "memebarrier_phrase_blocklist.json";
        blocklistPath_ = path;
        if (!fileExists(path))
            return;
        json doc = json::parse(readTextFile(path), nullptr, true, true);
        const json blocked = doc.value("blockedPhrases", json::array());
        if (!blocked.is_array())
            return;
        for (const auto &entry : blocked)
        {
            if (!entry.is_string())
                continue;
            insertPhraseBlocklistRow(mirrorDb_.get(), entry.get<std::string>());
        }
    }

    void persistPhraseBlocklist()
    {
        json doc;
        doc["blockedPhrases"] = json::array();
        const char *sql = "SELECT phrase FROM memebarrier_phrase_blocklist ORDER BY phrase";
        sqlite3_stmt *stmt = nullptr;
        if (sqlite3_prepare_v2(mirrorDb_.get(), sql, -1, &stmt, nullptr) != SQLITE_OK)
            throw std::runtime_error(sqlite3_errmsg(mirrorDb_.get()));
        StatementHandle handle(stmt);
        while (sqlite3_step(stmt) == SQLITE_ROW)
        {
            doc["blockedPhrases"].push_back(reinterpret_cast<const char *>(sqlite3_column_text(stmt, 0)));
        }
        writeTextFile(blocklistPath_, doc.dump(2));
    }

    void loadKnownJsonDocuments()
    {
        loadJsonDocumentFile("runtime_status", "llamacpp_runtime_status", options_.runtimeDir / "llamacpp_runtime_status.json");
        loadJsonDocumentFile("runtime_status", "bitnet_runtime_status", options_.runtimeDir / "bitnet_runtime_status.json");
        loadJsonDocumentFile("runtime_status", "bug_shooter_status", options_.runtimeDir / "bug_shooter_status.json");
        loadJsonDocumentFile("runtime_config", "start_079_launcher", options_.runtimeDir / "start_079_launcher.json");
        loadJsonDocumentTree("model_lifecycle", options_.runtimeDir / "model_lifecycle");
        loadJsonDocumentTree("brain_maps", options_.runtimeDir / "brain_maps");
        loadJsonDocumentTree("structured_exports", options_.runtimeDir / "structured_exports");
        loadJsonDocumentTree("model_storage", options_.ggufModelsDir / "manifests");
    }

    void loadJsonDocumentFile(const std::string &collection, const std::string &documentId, const fs::path &path)
    {
        if (!fileExists(path))
            return;
        const std::string text = readTextFile(path);
        json doc = json::parse(text, nullptr, true, true);
        insertJsonDocumentRow(mirrorDb_.get(), collection, documentId, resolveAbsolute(path), doc.dump(2), true);
        managedJsonDocuments_[documentKey(collection, documentId)] = resolveAbsolute(path);
    }

    void loadJsonDocumentTree(const std::string &collection, const fs::path &root)
    {
        std::error_code ec;
        if (!fs::exists(root, ec) || !fs::is_directory(root, ec))
            return;
        for (fs::recursive_directory_iterator it(root, ec), end; it != end; it.increment(ec))
        {
            if (ec)
                break;
            if (!it->is_regular_file())
                continue;
            if (lowerCopy(it->path().extension().string()) != ".json")
                continue;
            const fs::path absolute = resolveAbsolute(it->path());
            const fs::path relative = fs::relative(absolute, root, ec);
            const std::string documentId = ec ? absolute.filename().generic_string() : relative.generic_string();
            const std::string text = readTextFile(absolute);
            json doc = json::parse(text, nullptr, true, true);
            insertJsonDocumentRow(mirrorDb_.get(), collection, documentId, absolute, doc.dump(2), true);
            managedJsonDocuments_[documentKey(collection, documentId)] = absolute;
        }
    }

    std::vector<JsonDocumentRow> readJsonDocumentRows() const
    {
        std::vector<JsonDocumentRow> rows;
        const char *sql = "SELECT collection, document_id, path, content_json, writable FROM json_documents ORDER BY collection, document_id";
        sqlite3_stmt *stmt = nullptr;
        if (sqlite3_prepare_v2(mirrorDb_.get(), sql, -1, &stmt, nullptr) != SQLITE_OK)
            throw std::runtime_error(sqlite3_errmsg(mirrorDb_.get()));
        StatementHandle handle(stmt);
        while (sqlite3_step(stmt) == SQLITE_ROW)
        {
            JsonDocumentRow row;
            row.collection = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 0));
            row.documentId = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 1));
            row.path = fs::path(reinterpret_cast<const char *>(sqlite3_column_text(stmt, 2)));
            row.contentJson = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 3));
            row.writable = sqlite3_column_int64(stmt, 4) != 0;
            rows.push_back(std::move(row));
        }
        return rows;
    }

    void persistJsonDocuments()
    {
        const std::vector<JsonDocumentRow> rows = readJsonDocumentRows();
        std::unordered_map<std::string, fs::path> current;
        for (const auto &row : rows)
        {
            if (!row.writable)
                continue;
            json doc = json::parse(row.contentJson, nullptr, true, true);
            const fs::path path = resolveAbsolute(row.path);
            writeTextFile(path, doc.dump(2));
            current[documentKey(row.collection, row.documentId)] = path;
        }

        for (const auto &entry : managedJsonDocuments_)
        {
            auto it = current.find(entry.first);
            if (it == current.end())
            {
                removeFileIfExists(entry.second);
                continue;
            }
            if (resolveAbsolute(entry.second) != resolveAbsolute(it->second))
                removeFileIfExists(entry.second);
        }
    }

    Options options_;
    SqliteHandle mirrorDb_;
    std::unordered_map<std::string, fs::path> managedSourceDbs_;
    std::unordered_map<std::string, fs::path> managedJsonDocuments_;
    fs::path feedbackPath_;
    fs::path blocklistPath_;
};

Options parseArgs(int argc, char **argv)
{
    Options options;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        auto nextValue = [&](const std::string &prefix) -> std::string
        {
            if (arg.rfind(prefix + "=", 0) == 0)
                return arg.substr(prefix.size() + 1);
            if (arg == prefix && i + 1 < argc)
                return argv[++i];
            return std::string();
        };

        if (arg == "--help" || arg == "-h")
        {
            options.sql = "__HELP__";
            return options;
        }
        if (arg == "--read-only")
        {
            options.readOnly = true;
            continue;
        }
        if (arg == "--list-tables")
        {
            options.listTables = true;
            continue;
        }
        if (auto value = nextValue("--runtime-dir"); !value.empty())
        {
            options.runtimeDir = value;
            continue;
        }
        if (auto value = nextValue("--main-db"); !value.empty())
        {
            options.mainDbPath = value;
            continue;
        }
        if (auto value = nextValue("--world-db"); !value.empty())
        {
            options.worldModelDbPath = value;
            continue;
        }
        if (auto value = nextValue("--gguf-models-dir"); !value.empty())
        {
            options.ggufModelsDir = value;
            continue;
        }
        if (auto value = nextValue("--sql"); !value.empty())
        {
            options.sql = value;
            continue;
        }
        if (auto value = nextValue("--file"); !value.empty())
        {
            options.sqlFile = value;
            continue;
        }
        throw std::runtime_error("unknown argument: " + arg);
    }
    return options;
}

std::string usageText()
{
    return
        "Usage: phoenix_sql_cli.exe [--runtime-dir DIR] [--main-db FILE] [--world-db FILE] [--gguf-models-dir DIR] [--read-only] (--sql SQL | --file SQL_FILE | --list-tables)\n"
        "\n"
        "Mirrored tables:\n"
        "  kv_entries(source_db, namespace, key, value_json, updated_at)\n"
        "  inference_cache(source_db, key, value_json, updated_at)\n"
        "  inference_cache_stats(source_db, key, total_hits, hot_hits, cold_loads, writes, last_access_ms, last_promoted_ms, is_hot)\n"
        "  memebarrier_phrase_feedback_positive(phrase, offset)\n"
        "  memebarrier_phrase_blocklist(phrase)\n"
        "  json_documents(collection, document_id, path, content_json, writable)\n"
        "  internal_catalog(table_name, source, writable, description)\n";
}

} // namespace

ExecutionResult execute(const Options &options)
{
    try
    {
        const Options normalized = normalizeOptions(options);
        const std::string script = readSqlScript(normalized);
        MirrorWorkspace workspace(normalized);
        workspace.loadAll();
        json out;
        out["ok"] = true;
        out["runtimeDir"] = workspace.options().runtimeDir.generic_string();
        out["mainDbPath"] = workspace.options().mainDbPath.generic_string();
        out["worldModelDbPath"] = workspace.options().worldModelDbPath.generic_string();
        out["ggufModelsDir"] = workspace.options().ggufModelsDir.generic_string();
        out["results"] = workspace.executeSqlScript(script, normalized.readOnly);
        if (!normalized.readOnly)
            workspace.persistAll();
        out["persisted"] = !normalized.readOnly;
        return ExecutionResult{true, out, std::string()};
    }
    catch (const std::exception &e)
    {
        return ExecutionResult{false, json{{"ok", false}, {"error", e.what()}}, e.what()};
    }
    catch (...)
    {
        return ExecutionResult{false, json{{"ok", false}, {"error", "unknown error"}}, "unknown error"};
    }
}

int runMain(int argc, char **argv, std::ostream &out, std::ostream &err)
{
    try
    {
        Options options = parseArgs(argc, argv);
        if (options.sql == "__HELP__")
        {
            out << usageText();
            return 0;
        }
        ExecutionResult result = execute(options);
        out << result.output.dump(2) << std::endl;
        return result.ok ? 0 : 1;
    }
    catch (const std::exception &e)
    {
        err << e.what() << std::endl;
        err << usageText();
        return 2;
    }
}

} // namespace phoenix_sql_cli