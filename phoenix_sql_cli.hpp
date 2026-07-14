/* phoenix_sql_cli.hpp - SQL command-line interface */

#pragma once

#include <filesystem>
#include <iosfwd>
#include <string>

#include <nlohmann/json.hpp>

namespace phoenix_sql_cli
{

namespace fs = std::filesystem;
using json = nlohmann::json;

/* SQL CLI options */
struct Options
{
    fs::path runtimeDir{"runtime_store"}; /* Runtime directory */
    fs::path mainDbPath;                  /* Main database path */
    fs::path worldModelDbPath;            /* World model database path */
    fs::path ggufModelsDir{"GGUF_models"}; /* GGUF models directory */
    std::string sql;                      /* SQL query string */
    fs::path sqlFile;                     /* SQL file path */
    bool readOnly{false};                 /* Read-only mode */
    bool listTables{false};               /* List tables flag */
};

/* SQL execution result */
struct ExecutionResult
{
    bool ok{false};                       /* Success flag */
    json output = json::object();         /* Output data */
    std::string error;                    /* Error message */
};

/* Execute SQL with options */
ExecutionResult execute(const Options &options);
/* Run main CLI */
int runMain(int argc, char **argv, std::ostream &out, std::ostream &err);

} // namespace phoenix_sql_cli