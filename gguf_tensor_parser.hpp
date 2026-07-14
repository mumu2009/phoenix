/* gguf_tensor_parser.hpp - GGUF model file parser and inspector */

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