/* gguf_tensor_parser.hpp - GGUF model file parser and inspector
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

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

#include <nlohmann/json.hpp>

namespace gguf_tensor_parser {

namespace fs = std::filesystem;
using json = nlohmann::json;

/* Options for GGUF file inspection */
struct InspectOptions {
    std::size_t largestTensorCount{12}; /* Number of largest tensors to report */
    std::size_t kvPreviewCount{8};      /* Number of key-value pairs to preview */
    std::size_t tokenPreviewCount{16};  /* Number of tokens to preview */
};

/* Result of GGUF file inspection */
struct InspectResult {
    bool exists{false};              /* File exists */
    bool valid{false};               /* File is valid GGUF format */
    std::string error;               /* Error message if any */
    json report = json::object();    /* Inspection report */

    json toJson() const;             /* Serialize to JSON */
};

/* Inspect a GGUF file and return detailed information */
InspectResult inspectFile(const fs::path &path, const InspectOptions &options = {});

/* Build brain map document from inspection results */
json buildBrainMapDocument(const std::string &provider,
                           const std::string &modelPath,
                           const InspectResult &inspection,
                           const fs::path &calculatorRoot,
                           const fs::path &divingAgreementRoot,
                           int64_t generatedAtMs);

/* Build structured export bundle for model distribution */
json buildStructuredExportBundle(const std::string &provider,
                                 const std::string &modelPath,
                                 const InspectResult &inspection,
                                 int64_t generatedAtMs);

/* Write structured export files to disk */
json writeStructuredExportFiles(const json &bundle,
                                const fs::path &outputRoot,
                                std::string *error = nullptr);

} // namespace gguf_tensor_parser