/* phoenix_sql_cli.hpp - SQL command-line interface
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