#include "../phoenix_sql_cli.hpp"

#include <sqlite3.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

namespace fs = std::filesystem;
using json = nlohmann::json;

void requireTrue(bool condition, const std::string &message)
{
    if (!condition)
        throw std::runtime_error(message);
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

fs::path makeRoot(const std::string &name)
{
    const fs::path root = fs::current_path() / "build" / "testdata" / name;
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);
    return root;
}

void writeJson(const fs::path &path, const json &doc)
{
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("failed to write file: " + path.string());
    output << doc.dump(2);
}

json readJson(const fs::path &path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("failed to open file: " + path.string());
    return json::parse(input, nullptr, true, true);
}

void initSourceDb(const fs::path &path)
{
    fs::create_directories(path.parent_path());
    sqlite3 *db = nullptr;
    if (sqlite3_open(path.string().c_str(), &db) != SQLITE_OK)
    {
        std::string error = db ? sqlite3_errmsg(db) : std::string("sqlite open failed");
        if (db)
            sqlite3_close(db);
        throw std::runtime_error(error);
    }
    execSql(db,
            "CREATE TABLE kv_store(namespace TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, updatedAt INTEGER, PRIMARY KEY(namespace,key));"
            "CREATE TABLE inference_cache(key TEXT PRIMARY KEY, value TEXT NOT NULL, updatedAt INTEGER);"
            "CREATE TABLE inference_cache_stats(key TEXT PRIMARY KEY, totalHits INTEGER DEFAULT 0, hotHits INTEGER DEFAULT 0, coldLoads INTEGER DEFAULT 0, writes INTEGER DEFAULT 0, lastAccessMs INTEGER DEFAULT 0, lastPromotedMs INTEGER DEFAULT 0, isHot INTEGER DEFAULT 0);"
            "INSERT INTO kv_store(namespace,key,value,updatedAt) VALUES('kvm','alpha','{\"value\":1}',111);"
            "INSERT INTO inference_cache(key,value,updatedAt) VALUES('cache-key','{\"score\":0.7}',222);"
            "INSERT INTO inference_cache_stats(key,totalHits,hotHits,coldLoads,writes,lastAccessMs,lastPromotedMs,isHot) VALUES('cache-key',5,3,2,1,333,444,1);");
    sqlite3_close(db);
}

std::string readSingleValue(const fs::path &dbPath, const std::string &sql)
{
    sqlite3 *db = nullptr;
    if (sqlite3_open(dbPath.string().c_str(), &db) != SQLITE_OK)
    {
        std::string error = db ? sqlite3_errmsg(db) : std::string("sqlite open failed");
        if (db)
            sqlite3_close(db);
        throw std::runtime_error(error);
    }
    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK)
    {
        std::string error = sqlite3_errmsg(db);
        sqlite3_close(db);
        throw std::runtime_error(error);
    }
    std::string value;
    if (sqlite3_step(stmt) == SQLITE_ROW)
    {
        const unsigned char *raw = sqlite3_column_text(stmt, 0);
        value = raw ? reinterpret_cast<const char *>(raw) : "";
    }
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return value;
}

void testCrudPersistsAcrossSources()
{
    const fs::path root = makeRoot("phoenix_sql_cli_case");
    const fs::path runtimeDir = root / "runtime_store";
    const fs::path ggufDir = root / "GGUF_models";
    const fs::path mainDbPath = runtimeDir / "ai_store.sqlite";
    initSourceDb(mainDbPath);

    writeJson(runtimeDir / "memebarrier_phrase_feedback.json", json{{"positiveThresholdOffsets", json{{"safe phrase", 0.05}}}});
    writeJson(runtimeDir / "memebarrier_phrase_blocklist.json", json{{"blockedPhrases", json::array({"blocked phrase"})}});
    writeJson(runtimeDir / "llamacpp_runtime_status.json", json{{"status", "ready"}, {"ok", true}});
    writeJson(runtimeDir / "model_lifecycle" / "manifest.json", json{{"activeTarget", "local"}, {"activeVersion", "v1"}, {"updateSeq", 1}, {"events", json::array()}});
    writeJson(ggufDir / "manifests" / "demo.json", json{{"model", "demo"}});

    phoenix_sql_cli::Options options;
    options.runtimeDir = runtimeDir;
    options.ggufModelsDir = ggufDir;

    const fs::path newManifestPath = ggufDir / "manifests" / "new_manifest.json";
    options.sql =
        "UPDATE kv_entries SET value_json='{\"value\":2}' WHERE source_db='main' AND namespace='kvm' AND key='alpha';"
        "INSERT INTO kv_entries(source_db, namespace, key, value_json, updated_at) VALUES('main','session','beta','{\"active\":true}',123);"
        "UPDATE memebarrier_phrase_feedback_positive SET offset=0.2 WHERE phrase='safe phrase';"
        "INSERT INTO memebarrier_phrase_blocklist(phrase) VALUES('new blocked phrase');"
        "DELETE FROM memebarrier_phrase_blocklist WHERE phrase='blocked phrase';"
        "UPDATE json_documents SET content_json='{\"status\":\"warming\",\"ok\":true}' WHERE collection='runtime_status' AND document_id='llamacpp_runtime_status';"
        "INSERT INTO json_documents(collection, document_id, path, content_json, writable) VALUES('model_storage','new_manifest','" + newManifestPath.generic_string() + "','{\"model\":\"new\"}',1);";

    const auto result = phoenix_sql_cli::execute(options);
    requireTrue(result.ok, "phoenix_sql_cli execute should succeed");
    requireTrue(result.output["persisted"].get<bool>(), "write mode should persist mirrored changes");

    requireTrue(readSingleValue(mainDbPath, "SELECT value FROM kv_store WHERE namespace='kvm' AND key='alpha';") == "{\"value\":2}", "kv entry update should persist to source database");
    requireTrue(readSingleValue(mainDbPath, "SELECT value FROM kv_store WHERE namespace='session' AND key='beta';") == "{\"active\":true}", "kv entry insert should persist to source database");

    const json feedbackDoc = readJson(runtimeDir / "memebarrier_phrase_feedback.json");
    requireTrue(std::abs(feedbackDoc["positiveThresholdOffsets"]["safe phrase"].get<double>() - 0.2) < 1e-9, "phrase feedback update should persist to json file");

    const json blocklistDoc = readJson(runtimeDir / "memebarrier_phrase_blocklist.json");
    requireTrue(blocklistDoc["blockedPhrases"].size() == 1, "blocklist should reflect insert + delete");
    requireTrue(blocklistDoc["blockedPhrases"][0].get<std::string>() == "new blocked phrase", "blocklist should contain the inserted phrase");

    const json runtimeStatus = readJson(runtimeDir / "llamacpp_runtime_status.json");
    requireTrue(runtimeStatus["status"].get<std::string>() == "warming", "runtime status update should persist through json_documents");

    const json newManifest = readJson(newManifestPath);
    requireTrue(newManifest["model"].get<std::string>() == "new", "new model manifest should be created through json_documents insert");
}

void testSelectCatalogAndDocuments()
{
    const fs::path root = makeRoot("phoenix_sql_cli_catalog");
    const fs::path runtimeDir = root / "runtime_store";
    const fs::path ggufDir = root / "GGUF_models";
    initSourceDb(runtimeDir / "ai_store.sqlite");
    writeJson(runtimeDir / "memebarrier_phrase_feedback.json", json{{"positiveThresholdOffsets", json{{"safe phrase", 0.05}}}});
    writeJson(runtimeDir / "memebarrier_phrase_blocklist.json", json{{"blockedPhrases", json::array({"blocked phrase"})}});
    writeJson(ggufDir / "manifests" / "demo.json", json{{"model", "demo"}});

    phoenix_sql_cli::Options options;
    options.runtimeDir = runtimeDir;
    options.ggufModelsDir = ggufDir;
    options.sql = "SELECT table_name FROM internal_catalog ORDER BY table_name; SELECT document_id FROM json_documents WHERE collection='model_storage' ORDER BY document_id;";

    const auto result = phoenix_sql_cli::execute(options);
    requireTrue(result.ok, "select query should succeed");
    requireTrue(result.output["results"].size() == 2, "two SELECT statements should produce two result sets");
    requireTrue(result.output["results"][0]["rows"].size() >= 6, "catalog should expose mirrored table inventory");
    requireTrue(result.output["results"][1]["rows"].size() == 1, "model storage manifest should appear in json_documents");
    requireTrue(result.output["results"][1]["rows"][0]["document_id"].get<std::string>() == "demo.json", "document id should use manifest-relative path");
}

} // namespace

int main()
{
    try
    {
        testCrudPersistsAcrossSources();
        testSelectCatalogAndDocuments();
        std::cout << "phoenix_sql_cli_tests: ok" << std::endl;
        return 0;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "phoenix_sql_cli_tests: failed: " << ex.what() << std::endl;
        return 1;
    }
}