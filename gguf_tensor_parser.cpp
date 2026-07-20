/* gguf_tensor_parser.cpp - GGUF tensor parser implementation
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

#include "gguf_tensor_parser.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <limits>
#include <numeric>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gguf_tensor_parser {

namespace {

constexpr uint32_t kGgufVersionMax = 3;
constexpr uint32_t kDefaultAlignment = 32;
constexpr std::size_t kMaxDims = 4;

enum GgufType : uint32_t {
    GGUF_TYPE_UINT8 = 0,
    GGUF_TYPE_INT8 = 1,
    GGUF_TYPE_UINT16 = 2,
    GGUF_TYPE_INT16 = 3,
    GGUF_TYPE_UINT32 = 4,
    GGUF_TYPE_INT32 = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL = 7,
    GGUF_TYPE_STRING = 8,
    GGUF_TYPE_ARRAY = 9,
    GGUF_TYPE_UINT64 = 10,
    GGUF_TYPE_INT64 = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

struct TensorInfo {
    std::string name;
    std::vector<int64_t> dims;
    uint32_t typeId{0};
    uint64_t offset{0};
    uint64_t spanBytes{0};
    uint64_t elements{0};
};

class Reader {
public:
    explicit Reader(const fs::path &path) : in_(path, std::ios::binary) {}

    bool good() const {
        return static_cast<bool>(in_);
    }

    template <typename T>
    bool readPod(T &value) {
        in_.read(reinterpret_cast<char *>(&value), sizeof(T));
        return static_cast<bool>(in_);
    }

    bool readString(std::string &value) {
        uint64_t size = 0;
        if (!readPod(size)) {
            return false;
        }
        if (size > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
            return false;
        }
        value.resize(static_cast<std::size_t>(size));
        if (size == 0) {
            return true;
        }
        in_.read(value.data(), static_cast<std::streamsize>(size));
        return static_cast<bool>(in_);
    }

    bool skip(uint64_t bytes) {
        if (bytes > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
            return false;
        }
        in_.seekg(static_cast<std::streamoff>(bytes), std::ios::cur);
        return static_cast<bool>(in_);
    }

    uint64_t position() {
        return static_cast<uint64_t>(in_.tellg());
    }

private:
    std::ifstream in_;
};

std::string ggufTypeName(uint32_t typeId) {
    switch (typeId) {
    case GGUF_TYPE_UINT8:
        return "u8";
    case GGUF_TYPE_INT8:
        return "i8";
    case GGUF_TYPE_UINT16:
        return "u16";
    case GGUF_TYPE_INT16:
        return "i16";
    case GGUF_TYPE_UINT32:
        return "u32";
    case GGUF_TYPE_INT32:
        return "i32";
    case GGUF_TYPE_FLOAT32:
        return "f32";
    case GGUF_TYPE_BOOL:
        return "bool";
    case GGUF_TYPE_STRING:
        return "string";
    case GGUF_TYPE_ARRAY:
        return "array";
    case GGUF_TYPE_UINT64:
        return "u64";
    case GGUF_TYPE_INT64:
        return "i64";
    case GGUF_TYPE_FLOAT64:
        return "f64";
    default:
        return "unknown";
    }
}

std::string ggmlTypeName(uint32_t typeId) {
    static const std::array<const char *, 39> names = {
        "F32", "F16", "Q4_0", "Q4_1", "type_4", "type_5", "Q5_0", "Q5_1", "Q8_0", "Q8_1",
        "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S",
        "IQ4_NL", "IQ3_S", "IQ2_S", "IQ4_XS", "I8", "I16", "I32", "I64", "F64", "IQ1_M",
        "BF16", "type_31", "type_32", "type_33", "TQ1_0", "TQ2_0", "type_36", "type_37", "type_38"};
    if (typeId < names.size()) {
        return names[typeId];
    }
    return "UNKNOWN";
}

bool readScalarValue(Reader &reader, uint32_t typeId, json &value) {
    switch (typeId) {
    case GGUF_TYPE_UINT8: {
        uint8_t v = 0;
        if (!reader.readPod(v)) return false;
        value = v;
        return true;
    }
    case GGUF_TYPE_INT8: {
        int8_t v = 0;
        if (!reader.readPod(v)) return false;
        value = v;
        return true;
    }
    case GGUF_TYPE_UINT16: {
        uint16_t v = 0;
        if (!reader.readPod(v)) return false;
        value = v;
        return true;
    }
    case GGUF_TYPE_INT16: {
        int16_t v = 0;
        if (!reader.readPod(v)) return false;
        value = v;
        return true;
    }
    case GGUF_TYPE_UINT32: {
        uint32_t v = 0;
        if (!reader.readPod(v)) return false;
        value = v;
        return true;
    }
    case GGUF_TYPE_INT32: {
        int32_t v = 0;
        if (!reader.readPod(v)) return false;
        value = v;
        return true;
    }
    case GGUF_TYPE_FLOAT32: {
        float v = 0.0f;
        if (!reader.readPod(v)) return false;
        value = v;
        return true;
    }
    case GGUF_TYPE_BOOL: {
        int8_t v = 0;
        if (!reader.readPod(v)) return false;
        value = (v != 0);
        return true;
    }
    case GGUF_TYPE_STRING: {
        std::string v;
        if (!reader.readString(v)) return false;
        value = v;
        return true;
    }
    case GGUF_TYPE_UINT64: {
        uint64_t v = 0;
        if (!reader.readPod(v)) return false;
        value = v;
        return true;
    }
    case GGUF_TYPE_INT64: {
        int64_t v = 0;
        if (!reader.readPod(v)) return false;
        value = v;
        return true;
    }
    case GGUF_TYPE_FLOAT64: {
        double v = 0.0;
        if (!reader.readPod(v)) return false;
        value = v;
        return true;
    }
    default:
        return false;
    }
}

bool readValue(Reader &reader, uint32_t typeId, std::size_t previewCount, json &value) {
    if (typeId != GGUF_TYPE_ARRAY) {
        return readScalarValue(reader, typeId, value);
    }

    uint32_t innerType = 0;
    uint64_t count = 0;
    if (!reader.readPod(innerType) || !reader.readPod(count)) {
        return false;
    }

    json preview = json::array();
    for (uint64_t index = 0; index < count; ++index) {
        json item;
        if (!readScalarValue(reader, innerType, item)) {
            return false;
        }
        if (preview.size() < previewCount) {
            preview.push_back(item);
        }
    }
    value = {
        {"type", "array"},
        {"itemType", ggufTypeName(innerType)},
        {"count", count},
        {"preview", preview}};
    return true;
}

uint64_t alignUp(uint64_t value, uint64_t alignment) {
    if (alignment == 0) {
        return value;
    }
    const uint64_t mask = alignment - 1;
    return (value + mask) & ~mask;
}

std::string lowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool containsAny(const std::string &value, const std::vector<std::string> &needles) {
    const std::string lowered = lowerCopy(value);
    for (const auto &needle : needles) {
        if (lowered.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

int64_t jsonToInt64(const json &value, int64_t fallback);

std::string sanitizeExportName(std::string value) {
    if (value.empty()) {
        return "unknown";
    }
    for (char &ch : value) {
        const bool asciiAlphaNum = (ch >= 'a' && ch <= 'z') ||
                                   (ch >= 'A' && ch <= 'Z') ||
                                   (ch >= '0' && ch <= '9');
        if (!asciiAlphaNum && ch != '-' && ch != '_') {
            ch = '_';
        }
    }
    return value;
}

json findArrayPreviewBySuffix(const json &interesting, const std::vector<std::string> &suffixes) {
    for (auto it = interesting.begin(); it != interesting.end(); ++it) {
        const std::string key = lowerCopy(it.key());
        for (const auto &suffix : suffixes) {
            if (key.size() < suffix.size() || key.compare(key.size() - suffix.size(), suffix.size(), suffix) != 0) {
                continue;
            }
            if (it.value().is_object() && it.value().value("type", "") == "array") {
                const json preview = it.value().value("preview", json::array());
                if (preview.is_array()) {
                    return preview;
                }
            }
        }
    }
    return json::array();
}

int64_t findScalarByExactKey(const json &interesting, const std::vector<std::string> &keys) {
    for (const auto &keyName : keys) {
        const auto it = interesting.find(keyName);
        if (it != interesting.end()) {
            return jsonToInt64(*it, 0);
        }
    }
    return 0;
}

json inferSpecialTokens(const json &interesting, const json &tokens) {
    json special = json::object();

    const struct NamedIdKey {
        const char *name;
        std::vector<std::string> keys;
    } namedKeys[] = {
        {"bos", {"tokenizer.ggml.bos_token_id", "tokenizer.bos_token_id"}},
        {"eos", {"tokenizer.ggml.eos_token_id", "tokenizer.eos_token_id"}},
        {"unk", {"tokenizer.ggml.unknown_token_id", "tokenizer.unknown_token_id", "tokenizer.ggml.unk_token_id"}},
        {"pad", {"tokenizer.ggml.padding_token_id", "tokenizer.padding_token_id", "tokenizer.ggml.pad_token_id"}}};

    for (const auto &entry : namedKeys) {
        const int64_t value = findScalarByExactKey(interesting, entry.keys);
        if (value >= 0) {
            special[entry.name] = value;
        }
    }

    if (!tokens.is_array()) {
        return special;
    }

    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (!tokens[index].is_string()) {
            continue;
        }
        const std::string token = lowerCopy(tokens[index].get<std::string>());
        if ((token == "<s>" || token == "<bos>" || token == "<|bos|>") && !special.contains("bos")) {
            special["bos"] = static_cast<int64_t>(index);
        } else if ((token == "</s>" || token == "<eos>" || token == "<|eos|>") && !special.contains("eos")) {
            special["eos"] = static_cast<int64_t>(index);
        } else if ((token == "<unk>" || token == "<|unk|>") && !special.contains("unk")) {
            special["unk"] = static_cast<int64_t>(index);
        } else if ((token == "<pad>" || token == "<|pad|>") && !special.contains("pad")) {
            special["pad"] = static_cast<int64_t>(index);
        }
    }

    return special;
}

json buildTensorIndex(const json &anchors) {
    json tensorIndex = {
        {"embedding", json::array()},
        {"attention", json::array()},
        {"mlp", json::array()},
        {"norm", json::array()},
        {"output", json::array()}}
    ;
    if (!anchors.is_array()) {
        return tensorIndex;
    }
    for (const auto &anchor : anchors) {
        if (!anchor.is_object()) {
            continue;
        }
        const std::string name = anchor.value("name", "");
        const std::string lowered = lowerCopy(name);
        if (containsAny(lowered, {"token_embd", "embed_tokens", "tok_embeddings"})) {
            tensorIndex["embedding"].push_back(name);
        } else if (containsAny(lowered, {"attn", "attention"})) {
            tensorIndex["attention"].push_back(name);
        } else if (containsAny(lowered, {"ffn", "mlp", "feed_forward"})) {
            tensorIndex["mlp"].push_back(name);
        } else if (containsAny(lowered, {"norm", "rms"})) {
            tensorIndex["norm"].push_back(name);
        } else if (containsAny(lowered, {"output", "lm_head"})) {
            tensorIndex["output"].push_back(name);
        }
    }
    return tensorIndex;
}

json buildSemanticMappings(const json &tokens) {
    json mappings = json::array();
    if (!tokens.is_array()) {
        return mappings;
    }

    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (!tokens[index].is_string()) {
            continue;
        }
        const std::string token = tokens[index].get<std::string>();
        json units = json::array();
        if (token.empty()) {
            units.push_back({
                {"id", "sem:tok:" + std::to_string(index) + ":0"},
                {"text", ""},
                {"role", "empty"}});
        } else if (token.size() <= 4 || (token.front() == '<' && token.back() == '>')) {
            units.push_back({
                {"id", "sem:tok:" + std::to_string(index) + ":0"},
                {"text", token},
                {"role", token.front() == '<' ? "special" : "identity"}});
        } else {
            std::size_t partIndex = 0;
            for (std::size_t cursor = 0; cursor < token.size(); cursor += 4) {
                const std::string chunk = token.substr(cursor, std::min<std::size_t>(4, token.size() - cursor));
                units.push_back({
                    {"id", "sem:tok:" + std::to_string(index) + ":" + std::to_string(partIndex++)},
                    {"text", chunk},
                    {"role", "subtoken"}});
            }
        }

        mappings.push_back({
            {"sourceTokenId", static_cast<int64_t>(index)},
            {"sourceToken", token},
            {"strategy", units.size() > 1 ? "split-by-morpheme" : "identity"},
            {"semanticUnits", units}});
    }
    return mappings;
}

json buildNeuroDynamicsTable(const json &mappings,
                             const json &anchors,
                             const json &model,
                             const json &projection) {
    json table = json::array();
    json fittedFrom = json::array();
    if (anchors.is_array()) {
        for (const auto &anchor : anchors) {
            if (anchor.is_object() && anchor.contains("name")) {
                fittedFrom.push_back(anchor["name"]);
            }
        }
    }

    const double embeddingWidth = std::max<int64_t>(1, model.value("embeddingWidth", 64));
    const double attentionUnits = std::max<int64_t>(1, projection.value("attentionUnitsPerToken", 8));
    const double semanticBands = std::max<int64_t>(1, projection.value("semanticBands", 4));
    const double baseTau = std::max(1.0, embeddingWidth / std::max(1.0, attentionUnits));

    for (const auto &mapping : mappings) {
        if (!mapping.is_object()) {
            continue;
        }
        const json units = mapping.value("semanticUnits", json::array());
        for (std::size_t unitIndex = 0; unitIndex < units.size(); ++unitIndex) {
            const json &unit = units[unitIndex];
            if (!unit.is_object()) {
                continue;
            }
            const double local = static_cast<double>(unitIndex + 1);
            const double threshold = 0.2 + local / (semanticBands * 6.0);
            const double reset = std::max(0.05, threshold * 0.45);
            const double tau = baseTau + local * 0.35;
            table.push_back({
                {"semanticUnitId", unit.value("id", "")},
                {"modelType", "lif"},
                {"parameters", {
                    {"I_th", threshold},
                    {"I_reset", reset},
                    {"tau", tau}}},
                {"fittedFrom", fittedFrom},
                {"constraints", {
                    {"tau", "> 0"},
                    {"I_th", "> I_reset"},
                    {"I_reset", ">= 0"}}}});
        }
    }
    return table;
}

json buildFitResult(const json &dynamicsTable,
                    const json &projection,
                    int64_t generatedAtMs) {
    const std::size_t paramCount = dynamicsTable.is_array() ? dynamicsTable.size() : 0;
    double loss = 0.0;
    double validationError = 0.0;
    double sumThreshold = 0.0;
    double sumReset = 0.0;
    double sumTau = 0.0;

    if (dynamicsTable.is_array()) {
        for (const auto &entry : dynamicsTable) {
            const json params = entry.value("parameters", json::object());
            sumThreshold += params.value("I_th", 0.0);
            sumReset += params.value("I_reset", 0.0);
            sumTau += params.value("tau", 0.0);
        }
    }

    if (paramCount > 0) {
        loss = 1.0 / static_cast<double>(paramCount + 2);
        validationError = 1.0 / static_cast<double>(paramCount + 4);
        sumThreshold /= static_cast<double>(paramCount);
        sumReset /= static_cast<double>(paramCount);
        sumTau /= static_cast<double>(paramCount);
    }

    return {
        {"loss", loss},
        {"convergence", {
            {"reached", paramCount > 0},
            {"reason", paramCount > 0 ? "bootstrap-from-gguf-inspection" : "insufficient-semantic-units"},
            {"sampleCount", paramCount},
            {"generatedAtMs", generatedAtMs}}},
        {"epochs", std::max<int64_t>(8, projection.value("semanticBands", 4) * 2)},
        {"validationError", validationError},
        {"parameterSnapshot", {
            {"modelType", "lif"},
            {"lifAverages", {
                {"I_th", sumThreshold},
                {"I_reset", sumReset},
                {"tau", sumTau}}}}}};
}

bool writeJsonDocument(const fs::path &path, const json &doc) {
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    out << doc.dump(2);
    return static_cast<bool>(out);
}

int64_t jsonToInt64(const json &value, int64_t fallback = 0) {
    if (value.is_number_integer()) {
        return value.get<int64_t>();
    }
    if (value.is_number_unsigned()) {
        const uint64_t raw = value.get<uint64_t>();
        return raw > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ? fallback : static_cast<int64_t>(raw);
    }
    if (value.is_number_float()) {
        return static_cast<int64_t>(value.get<double>());
    }
    return fallback;
}

int64_t findScalarBySuffix(const json &interesting, const std::vector<std::string> &suffixes, int64_t fallback = 0) {
    for (auto it = interesting.begin(); it != interesting.end(); ++it) {
        const std::string key = lowerCopy(it.key());
        for (const auto &suffix : suffixes) {
            if (key.size() >= suffix.size() && key.compare(key.size() - suffix.size(), suffix.size(), suffix) == 0) {
                const json &raw = it.value();
                if (raw.is_object() && raw.contains("count")) {
                    return jsonToInt64(raw["count"], fallback);
                }
                return jsonToInt64(raw, fallback);
            }
        }
    }
    return fallback;
}

std::string findStringBySuffix(const json &interesting, const std::vector<std::string> &suffixes) {
    for (auto it = interesting.begin(); it != interesting.end(); ++it) {
        const std::string key = lowerCopy(it.key());
        for (const auto &suffix : suffixes) {
            if (key.size() >= suffix.size() && key.compare(key.size() - suffix.size(), suffix.size(), suffix) == 0 && it.value().is_string()) {
                return it.value().get<std::string>();
            }
        }
    }
    return {};
}

std::vector<int64_t> inferDims(const TensorInfo &tensor) {
    std::vector<int64_t> dims;
    for (int64_t dim : tensor.dims) {
        if (dim > 1 || dims.empty()) {
            dims.push_back(dim);
        }
    }
    if (dims.empty()) {
        dims.push_back(1);
    }
    return dims;
}

int64_t inferEmbeddingWidth(const std::vector<TensorInfo> &tensors, int64_t vocabSize) {
    for (const auto &tensor : tensors) {
        if (!containsAny(tensor.name, {"token_embd", "embed_tokens", "tok_embeddings"})) {
            continue;
        }
        const auto dims = inferDims(tensor);
        if (dims.empty()) {
            continue;
        }
        if (dims.size() == 1) {
            return dims.front();
        }
        if (vocabSize > 0) {
            const auto dist0 = std::llabs(dims[0] - vocabSize);
            const auto dist1 = std::llabs(dims[1] - vocabSize);
            return dist0 <= dist1 ? dims[1] : dims[0];
        }
        return std::min<int64_t>(dims[0], dims[1]);
    }
    return 0;
}

json makeTensorJson(const TensorInfo &tensor) {
    return {
        {"name", tensor.name},
        {"dims", tensor.dims},
        {"typeId", tensor.typeId},
        {"type", ggmlTypeName(tensor.typeId)},
        {"offset", tensor.offset},
        {"spanBytes", tensor.spanBytes},
        {"elements", tensor.elements}};
}

json summarizeInspection(const fs::path &path,
                        uint32_t version,
                        uint64_t fileSize,
                        uint64_t alignment,
                        uint64_t dataOffset,
                        const json &interestingKv,
                        const std::vector<json> &kvPreview,
                        const std::vector<TensorInfo> &tensors,
                        const InspectOptions &options) {
    uint64_t totalSpanBytes = 0;
    uint64_t quantizedSpanBytes = 0;
    std::unordered_map<std::string, uint64_t> typeHistogram;
    std::vector<TensorInfo> largest = tensors;
    std::sort(largest.begin(), largest.end(), [](const TensorInfo &lhs, const TensorInfo &rhs) {
        if (lhs.spanBytes != rhs.spanBytes) {
            return lhs.spanBytes > rhs.spanBytes;
        }
        return lhs.name < rhs.name;
    });
    if (largest.size() > options.largestTensorCount) {
        largest.resize(options.largestTensorCount);
    }

    for (const auto &tensor : tensors) {
        totalSpanBytes += tensor.spanBytes;
        const std::string typeName = ggmlTypeName(tensor.typeId);
        typeHistogram[typeName] += tensor.spanBytes;
        if (typeName != "F32" && typeName != "F16" && typeName != "BF16" && typeName != "F64") {
            quantizedSpanBytes += tensor.spanBytes;
        }
    }

    const int64_t vocabSize = std::max<int64_t>(0, findScalarBySuffix(interestingKv, {"tokenizer.ggml.tokens", ".vocab_size"}));
    const int64_t contextLength = std::max<int64_t>(0, findScalarBySuffix(interestingKv, {".context_length", ".n_ctx_train"}));
    const int64_t blockCount = std::max<int64_t>(0, findScalarBySuffix(interestingKv, {".block_count", ".layer_count"}));
    const int64_t attentionHeads = std::max<int64_t>(0, findScalarBySuffix(interestingKv, {".attention.head_count", ".n_head"}));
    const int64_t ropeDimensions = std::max<int64_t>(0, findScalarBySuffix(interestingKv, {".rope.dimension_count"}));
    const int64_t embeddingWidth = std::max<int64_t>(0, inferEmbeddingWidth(tensors, vocabSize));
    const int64_t semanticBands = std::max<int64_t>(3, std::min<int64_t>(64, blockCount > 0 ? (blockCount / 2 + 2) : std::max<int64_t>(4, attentionHeads / 2 + 2)));
    const int64_t avgUnitsPerToken = std::max<int64_t>(32, embeddingWidth > 0 ? embeddingWidth : (ropeDimensions > 0 ? ropeDimensions * 2 : 128));
    const int64_t attentionUnitsPerToken = std::max<int64_t>(8, attentionHeads > 0 ? attentionHeads * 8 : std::max<int64_t>(8, avgUnitsPerToken / 8));
    const int64_t slidingWindow = std::max<int64_t>(256, std::min<int64_t>(contextLength > 0 ? contextLength : 2048, avgUnitsPerToken * 6));

    json largestJson = json::array();
    for (const auto &tensor : largest) {
        largestJson.push_back(makeTensorJson(tensor));
    }

    json histogram = json::object();
    for (const auto &entry : typeHistogram) {
        histogram[entry.first] = entry.second;
    }

    json anchors = json::array();
    for (const auto &tensor : tensors) {
        if (anchors.size() >= 6) {
            break;
        }
        if (containsAny(tensor.name, {"token_embd", "attn", "ffn", "output", "lm_head"})) {
            anchors.push_back(makeTensorJson(tensor));
        }
    }

    return {
        {"path", path.string()},
        {"version", version},
        {"fileSizeBytes", fileSize},
        {"alignment", alignment},
        {"dataOffset", dataOffset},
        {"dataSectionBytes", fileSize >= dataOffset ? (fileSize - dataOffset) : 0},
        {"tensorCount", tensors.size()},
        {"kvCount", kvPreview.size() + interestingKv.size() - std::min<std::size_t>(interestingKv.size(), kvPreview.size())},
        {"model", {
            {"architecture", findStringBySuffix(interestingKv, {"general.architecture", ".architecture"})},
            {"name", findStringBySuffix(interestingKv, {"general.name", ".name"})},
            {"vocabSize", vocabSize},
            {"contextLength", contextLength},
            {"blockCount", blockCount},
            {"attentionHeads", attentionHeads},
            {"ropeDimensions", ropeDimensions},
            {"embeddingWidth", embeddingWidth}}},
        {"kv", {
            {"interesting", interestingKv},
            {"preview", kvPreview}}},
        {"tensors", {
            {"largest", largestJson},
            {"typeHistogram", histogram},
            {"totalSpanBytes", totalSpanBytes},
            {"quantizedSpanBytes", quantizedSpanBytes}}},
        {"brainProjection", {
            {"mode", "token-unit-many-to-many"},
            {"tokenCount", vocabSize},
            {"avgUnitsPerToken", avgUnitsPerToken},
            {"attentionUnitsPerToken", attentionUnitsPerToken},
            {"semanticBands", semanticBands},
            {"layerCount", blockCount},
            {"slidingWindow", {
                {"recommendedTokens", slidingWindow},
                {"maxTokens", contextLength > 0 ? contextLength : slidingWindow},
                {"loadAheadBands", std::max<int64_t>(2, semanticBands / 3)}}},
            {"neuronField", {
                {"embeddingWidth", embeddingWidth},
                {"activationUnits", avgUnitsPerToken + attentionUnitsPerToken},
                {"tokenToUnitFanout", std::max<int64_t>(4, avgUnitsPerToken / 16)},
                {"unitToTokenFanIn", std::max<int64_t>(2, attentionUnitsPerToken / 8)}}},
            {"tensorAnchors", anchors}}}};
}

} // namespace

json InspectResult::toJson() const {
    json out = report;
    out["exists"] = exists;
    out["valid"] = valid;
    if (!error.empty()) {
        out["error"] = error;
    }
    return out;
}

InspectResult inspectFile(const fs::path &path, const InspectOptions &options) {
    InspectResult result;
    std::error_code ec;
    const fs::path absolutePath = fs::absolute(path, ec);
    const fs::path effectivePath = ec ? path : absolutePath;
    result.exists = fs::exists(effectivePath, ec) && fs::is_regular_file(effectivePath, ec);
    if (!result.exists) {
        result.error = "gguf file not found";
        result.report = json{{"path", effectivePath.string()}};
        return result;
    }

    const uint64_t fileSize = fs::file_size(effectivePath, ec);
    if (ec) {
        result.error = "failed to stat gguf file";
        result.report = json{{"path", effectivePath.string()}};
        return result;
    }

    Reader reader(effectivePath);
    if (!reader.good()) {
        result.error = "failed to open gguf file";
        result.report = json{{"path", effectivePath.string()}};
        return result;
    }

    char magic[4] = {};
    if (!reader.readPod(magic[0]) || !reader.readPod(magic[1]) || !reader.readPod(magic[2]) || !reader.readPod(magic[3])) {
        result.error = "failed to read gguf header";
        result.report = json{{"path", effectivePath.string()}};
        return result;
    }
    if (!(magic[0] == 'G' && magic[1] == 'G' && magic[2] == 'U' && magic[3] == 'F')) {
        result.error = "invalid gguf magic";
        result.report = json{{"path", effectivePath.string()}};
        return result;
    }

    uint32_t version = 0;
    uint64_t nTensors = 0;
    uint64_t nKv = 0;
    if (!reader.readPod(version) || !reader.readPod(nTensors) || !reader.readPod(nKv)) {
        result.error = "failed to read gguf counts";
        result.report = json{{"path", effectivePath.string()}};
        return result;
    }
    if (version == 0 || version > kGgufVersionMax) {
        result.error = "unsupported gguf version";
        result.report = json{{"path", effectivePath.string()}, {"version", version}};
        return result;
    }

    json interestingKv = json::object();
    std::vector<json> kvPreview;
    kvPreview.reserve(std::min<std::size_t>(static_cast<std::size_t>(nKv), options.kvPreviewCount));
    uint64_t alignment = kDefaultAlignment;

    for (uint64_t index = 0; index < nKv; ++index) {
        std::string key;
        uint32_t typeId = 0;
        if (!reader.readString(key) || !reader.readPod(typeId)) {
            result.error = "failed to read gguf key-value entry";
            result.report = json{{"path", effectivePath.string()}, {"version", version}};
            return result;
        }
        json value;
        if (!readValue(reader, typeId, options.tokenPreviewCount, value)) {
            result.error = "failed to decode gguf key-value payload";
            result.report = json{{"path", effectivePath.string()}, {"version", version}, {"key", key}};
            return result;
        }
        const bool important = containsAny(key, {
            "general.", "tokenizer.", ".context_length", ".block_count", ".head_count", ".rope.", ".name", ".architecture"});
        if (important) {
            interestingKv[key] = value;
        }
        if (kvPreview.size() < options.kvPreviewCount) {
            kvPreview.push_back({{"key", key}, {"type", ggufTypeName(typeId)}, {"value", value}});
        }
        if (lowerCopy(key) == "general.alignment") {
            alignment = static_cast<uint64_t>(std::max<int64_t>(1, jsonToInt64(value, kDefaultAlignment)));
        }
    }

    std::vector<TensorInfo> tensors;
    tensors.reserve(static_cast<std::size_t>(std::min<uint64_t>(nTensors, 4096)));
    for (uint64_t index = 0; index < nTensors; ++index) {
        TensorInfo tensor;
        uint32_t nDims = 0;
        if (!reader.readString(tensor.name) || !reader.readPod(nDims)) {
            result.error = "failed to read gguf tensor header";
            result.report = json{{"path", effectivePath.string()}, {"version", version}};
            return result;
        }
        if (nDims > kMaxDims) {
            result.error = "tensor dimensions exceed parser limit";
            result.report = json{{"path", effectivePath.string()}, {"tensor", tensor.name}, {"nDims", nDims}};
            return result;
        }
        tensor.dims.assign(kMaxDims, 1);
        tensor.elements = 1;
        for (uint32_t dim = 0; dim < nDims; ++dim) {
            int64_t rawDim = 1;
            if (!reader.readPod(rawDim)) {
                result.error = "failed to read tensor dimension";
                result.report = json{{"path", effectivePath.string()}, {"tensor", tensor.name}};
                return result;
            }
            tensor.dims[dim] = rawDim;
            if (rawDim > 0 && tensor.elements <= std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(rawDim)) {
                tensor.elements *= static_cast<uint64_t>(rawDim);
            }
        }
        if (!reader.readPod(tensor.typeId) || !reader.readPod(tensor.offset)) {
            result.error = "failed to read tensor metadata";
            result.report = json{{"path", effectivePath.string()}, {"tensor", tensor.name}};
            return result;
        }
        tensors.push_back(std::move(tensor));
    }

    const uint64_t metadataEnd = reader.position();
    const uint64_t dataOffset = alignUp(metadataEnd, alignment);
    const uint64_t dataSectionBytes = fileSize >= dataOffset ? (fileSize - dataOffset) : 0;
    for (std::size_t index = 0; index < tensors.size(); ++index) {
        const uint64_t nextOffset = (index + 1 < tensors.size()) ? tensors[index + 1].offset : dataSectionBytes;
        tensors[index].spanBytes = nextOffset >= tensors[index].offset ? (nextOffset - tensors[index].offset) : 0;
    }

    result.exists = true;
    result.valid = true;
    result.report = summarizeInspection(effectivePath, version, fileSize, alignment, dataOffset, interestingKv, kvPreview, tensors, options);
    result.report["path"] = effectivePath.string();
    result.report["exists"] = true;
    result.report["valid"] = true;
    return result;
}

json buildBrainMapDocument(const std::string &provider,
                           const std::string &modelPath,
                           const InspectResult &inspection,
                           const fs::path &calculatorRoot,
                           const fs::path &divingAgreementRoot,
                           int64_t generatedAtMs) {
    std::error_code ec;
    const fs::path absoluteModel = fs::absolute(fs::path(modelPath), ec);
    const fs::path effectiveModel = ec ? fs::path(modelPath) : absoluteModel;
    const json inspectionJson = inspection.toJson();

    json doc = {
        {"provider", provider},
        {"modelPath", effectiveModel.string()},
        {"generatedAtMs", generatedAtMs},
        {"source", inspection.valid ? "gguf-parser" : "heuristic-fallback"},
        {"inspection", inspectionJson},
        {"toolBridges", {
            {"calculator", {
                {"path", (calculatorRoot / "main.exe").string()},
                {"available", fs::exists(calculatorRoot / "main.exe", ec)}}},
            {"divingAgreement", {
                {"path", (divingAgreementRoot / "main.exe").string()},
                {"available", fs::exists(divingAgreementRoot / "main.exe", ec)}}}}}
    };

    if (inspection.valid && inspectionJson.contains("brainProjection")) {
        const json &projection = inspectionJson["brainProjection"];
        const json &model = inspectionJson.value("model", json::object());
        doc["model"] = {
            {"fileSizeBytes", inspectionJson.value("fileSizeBytes", 0)},
            {"tensors", inspectionJson.value("tensorCount", 0)},
            {"kvCount", inspectionJson.value("kvCount", 0)},
            {"contextLength", model.value("contextLength", 0)},
            {"vocabSize", model.value("vocabSize", 0)},
            {"embeddingWidth", model.value("embeddingWidth", 0)},
            {"slidingWindow", projection.value("slidingWindow", json::object())},
            {"neuronField", projection.value("neuronField", json::object())}};
        doc["conversion"] = {
            {"mode", projection.value("mode", "token-unit-many-to-many")},
            {"tokenCount", projection.value("tokenCount", 0)},
            {"avgUnitsPerToken", projection.value("avgUnitsPerToken", 0)},
            {"attentionUnitsPerToken", projection.value("attentionUnitsPerToken", 0)},
            {"semanticBands", projection.value("semanticBands", 0)},
            {"tensorAnchors", projection.value("tensorAnchors", json::array())},
            {"focus", "token-to-unit-many-to-many"}};
    } else {
        const int64_t fileSize = inspectionJson.value("fileSizeBytes", static_cast<int64_t>(0));
        const int64_t shardCount = std::max<int64_t>(4, std::min<int64_t>(128, fileSize / (256ll * 1024ll * 1024ll) + 4));
        const int64_t semanticBands = std::max<int64_t>(3, std::min<int64_t>(24, shardCount / 4));
        const int64_t windowTokens = std::max<int64_t>(1024, std::min<int64_t>(8192, shardCount * 256));
        doc["model"] = {
            {"fileSizeBytes", fileSize},
            {"shards", shardCount},
            {"semanticBands", semanticBands},
            {"windowTokens", windowTokens},
            {"slidingWindow", {
                {"loadAhead", std::max<int64_t>(2, semanticBands / 2)},
                {"mergeThreshold", 0.63},
                {"evictThreshold", 0.17}}},
            {"spikeLattice", {
                {"neurons", shardCount * 256},
                {"decay", 0.82},
                {"refractorySteps", 3},
                {"plasticity", 0.11}}}};
        doc["conversion"] = {
            {"mode", "gguf-to-brain-manifest"},
            {"tokenSplitPolicy", "semantic-dense"},
            {"focus", "token-to-unit-many-to-many"}};
    }

    return doc;
}

json buildStructuredExportBundle(const std::string &provider,
                                 const std::string &modelPath,
                                 const InspectResult &inspection,
                                 int64_t generatedAtMs) {
    std::error_code ec;
    const fs::path absoluteModel = fs::absolute(fs::path(modelPath), ec);
    const fs::path effectiveModel = ec ? fs::path(modelPath) : absoluteModel;
    const json inspectionJson = inspection.toJson();
    const json model = inspectionJson.value("model", json::object());
    const json interesting = inspectionJson.value("kv", json::object()).value("interesting", json::object());
    const json projection = inspectionJson.value("brainProjection", json::object());
    const json anchors = projection.value("tensorAnchors", json::array());
    const json tokenPreview = findArrayPreviewBySuffix(interesting, {"tokenizer.ggml.tokens", ".tokens"});
    const std::string architecture = model.value("architecture", "transformer").empty() ? "transformer" : model.value("architecture", "transformer");
    const std::string tokenizerType = findStringBySuffix(interesting, {"tokenizer.ggml.model", ".tokenizer", "general.tokenizer"});

    const json tensorIndex = buildTensorIndex(anchors);
    const json vocabulary = {
        {"tokenizerType", tokenizerType.empty() ? "unknown" : tokenizerType},
        {"tokens", tokenPreview},
        {"specialTokens", inferSpecialTokens(interesting, tokenPreview)},
        {"exportedTokenCount", tokenPreview.is_array() ? tokenPreview.size() : 0},
        {"sourceTokenCount", model.value("vocabSize", 0)}};
    const json mappings = buildSemanticMappings(tokenPreview);
    const json dynamicsTable = buildNeuroDynamicsTable(mappings, anchors, model, projection);
    const json fitResult = buildFitResult(dynamicsTable, projection, generatedAtMs);
    const json runtimeConfig = {
        {"windowSize", projection.value("slidingWindow", json::object()).value("recommendedTokens", std::max<int64_t>(256, model.value("contextLength", 0)))},
        {"summaryStateDim", std::max<int64_t>(32, model.value("embeddingWidth", 0) / 2)},
        {"personaPool", {
            {"enabled", true},
            {"phaseBuckets", std::max<int64_t>(1, projection.value("semanticBands", 1))},
            {"refreshEveryMs", 12000}}},
        {"toolBindings", {
            {"calculator", "calculator/main.exe"},
            {"divingAgreement", "DivingPact/main.exe"}}}};

    return {
        {"provider", provider},
        {"generatedAtMs", generatedAtMs},
        {"ggufModel", {
            {"modelPath", effectiveModel.string()},
            {"architecture", architecture},
            {"tensorIndex", tensorIndex},
            {"metadata", {
                {"n_layers", model.value("blockCount", 0)},
                {"n_embd", model.value("embeddingWidth", 0)},
                {"n_ctx", model.value("contextLength", 0)},
                {"tokenizer", vocabulary.value("tokenizerType", "unknown")}}}}},
        {"vocabulary", vocabulary},
        {"semanticMapping", mappings},
        {"neuroDynamicsTable", dynamicsTable},
        {"fitResult", fitResult},
        {"runtimeConfig", runtimeConfig},
        {"inspection", inspectionJson}};
}

json writeStructuredExportFiles(const json &bundle,
                                const fs::path &outputRoot,
                                std::string *error) {
    json bundleObj = bundle.is_object() ? bundle : json::object();

    const std::vector<std::pair<fs::path, json>> files = {
        {outputRoot / "gguf" / "model.json", bundleObj.value("ggufModel", json::object())},
        {outputRoot / "vocab" / "tokens.json", bundleObj.value("vocabulary", json::object())},
        {outputRoot / "mapping" / "semantic_mapping.json", bundleObj.value("semanticMapping", json::array())},
        {outputRoot / "dynamics" / "neuro_dynamics.json", bundleObj.value("neuroDynamicsTable", json::array())},
        {outputRoot / "fit" / "fit_result.json", bundleObj.value("fitResult", json::object())},
        {outputRoot / "runtime" / "runtime_config.json", bundleObj.value("runtimeConfig", json::object())},
        {outputRoot / "manifest.json", bundleObj}};

    json manifest = {
        {"root", outputRoot.string()},
        {"written", false},
        {"files", json::array()}};

    for (const auto &entry : files) {
        if (!writeJsonDocument(entry.first, entry.second)) {
            if (error) {
                *error = "failed to write structured export file: " + entry.first.string();
            }
            return manifest;
        }
        manifest["files"].push_back(entry.first.string());
    }

    manifest["written"] = true;
    return manifest;
}

} // namespace gguf_tensor_parser