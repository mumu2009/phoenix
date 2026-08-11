/* external_mixed_modal_io.cpp - Implementation for external mixed-modal I/O
   Copyright (C) 2026 079 Project */

#include "external_mixed_modal_io.hpp"
#include "multimodal_world_model.hpp"
#include "phoenix_config.hpp"
#include "transformer.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <unordered_map>

namespace phoenix {
namespace io {

std::string MixedModalPacket::modalityToString(MixedModalModality m) {
    switch (m) {
        case MixedModalModality::Text: return "text";
        case MixedModalModality::Image: return "image";
        case MixedModalModality::Audio: return "audio";
        case MixedModalModality::Video: return "video";
        case MixedModalModality::Sensor: return "sensor";
        case MixedModalModality::Structured: return "structured";
        default: return "unknown";
    }
}

MixedModalModality MixedModalPacket::stringToModality(const std::string &s) {
    static const std::unordered_map<std::string, MixedModalModality> map = {
        {"text", MixedModalModality::Text},
        {"image", MixedModalModality::Image},
        {"audio", MixedModalModality::Audio},
        {"video", MixedModalModality::Video},
        {"sensor", MixedModalModality::Sensor},
        {"structured", MixedModalModality::Structured},
    };
    auto it = map.find(s);
    return it != map.end() ? it->second : MixedModalModality::Unknown;
}

nlohmann::json MixedModalPacket::toJson() const {
    nlohmann::json j;
    j["id"] = id;
    j["modality"] = modalityToString(modality);
    j["payload"] = payload;
    j["mimeType"] = mimeType;
    j["source"] = source;
    j["timestampMs"] = timestampMs;
    j["metadata"] = metadata;
    return j;
}

MixedModalPacket MixedModalPacket::fromJson(const nlohmann::json &j) {
    MixedModalPacket p;
    if (!j.is_object()) return p;
    if (j.contains("id") && j["id"].is_string()) p.id = j["id"].get<std::string>();
    if (j.contains("modality") && j["modality"].is_string()) {
        p.modality = stringToModality(j["modality"].get<std::string>());
    }
    if (j.contains("payload") && j["payload"].is_array()) {
        p.payload = j["payload"].get<std::vector<uint8_t>>();
    }
    if (j.contains("mimeType") && j["mimeType"].is_string()) p.mimeType = j["mimeType"].get<std::string>();
    if (j.contains("source") && j["source"].is_string()) p.source = j["source"].get<std::string>();
    if (j.contains("timestampMs") && j["timestampMs"].is_number_unsigned()) {
        p.timestampMs = j["timestampMs"].get<uint64_t>();
    } else if (p.timestampMs == 0) {
        p.timestampMs = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
    }
    if (j.contains("metadata") && j["metadata"].is_object()) p.metadata = j["metadata"];
    return p;
}

namespace {

size_t conceptDimension(size_t requested) {
    return requested == 0 ? 64 : std::min<size_t>(requested, 4096);
}

std::vector<float> normalizeConcept(std::vector<float> value) {
    return phoenix::multimodal::normalizeVector(value);
}

MultimodalEncDecConfig buildMultimodalEncDecConfig() {
    MultimodalEncDecConfig cfg;
    cfg.baseUrl = phoenix::resolveConfigAsString("multimodal.encDecBaseUrl", "http://127.0.0.1:8085");
    cfg.timeoutMs = phoenix::resolveConfig<int>("multimodal.encDecTimeoutMs", 120000);
    return cfg;
}

MultimodalImageWorldModel &imageWorldModel(const MultimodalEncDecConfig &cfg) {
    static std::mutex mu;
    static std::unordered_map<std::string, std::unique_ptr<MultimodalImageWorldModel>> cache;
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(cfg.baseUrl);
    if (it != cache.end()) return *it->second;
    auto model = std::make_unique<MultimodalImageWorldModel>(cfg);
    auto &ref = *model;
    cache.emplace(cfg.baseUrl, std::move(model));
    return ref;
}

std::string imageVariantFromMetadata(const nlohmann::json &metadata) {
    (void)metadata;
    return "llava-1.5-7b";
}

MultimodalAudioWorldModel &audioWorldModel(const MultimodalEncDecConfig &cfg) {
    static std::mutex mu;
    static std::unordered_map<std::string, std::unique_ptr<MultimodalAudioWorldModel>> cache;
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(cfg.baseUrl);
    if (it != cache.end()) return *it->second;
    auto model = std::make_unique<MultimodalAudioWorldModel>(cfg);
    auto &ref = *model;
    cache.emplace(cfg.baseUrl, std::move(model));
    return ref;
}

std::string speechVariantFromMetadata(const nlohmann::json &metadata) {
    (void)metadata;
    return "qwen2-audio-7b";
}

int sampleRateFromMetadata(const nlohmann::json &metadata) {
    if (metadata.contains("sampleRate") && metadata["sampleRate"].is_number_integer()) {
        return metadata["sampleRate"].get<int>();
    }
    return 16000;
}

std::vector<float> metadataConcept(const nlohmann::json &metadata, size_t dim) {
    for (const char *key : {"conceptVector", "worldModelVector", "encoderVector"}) {
        auto it = metadata.find(key);
        if (it != metadata.end() && it->is_array()) {
            try {
                return normalizeConcept(phoenix::multimodal::projectToDimension(
                    it->get<std::vector<float>>(), dim, 0x574d4f44U));
            } catch (const nlohmann::json::exception &) {
                return {};
            }
        }
    }
    return {};
}

std::vector<float> tokenEncoderConcept(const std::string &text, size_t dim) {
    std::vector<float> vector(dim, 0.0f);
    std::string token;
    auto addToken = [&]() {
        if (token.empty()) return;
        const size_t bucket = std::hash<std::string>{}(token) % dim;
        const size_t companion = std::hash<std::string>{}(token + "#encoder") % dim;
        vector[bucket] += 1.0f;
        vector[companion] += 0.5f;
        token.clear();
    };
    for (unsigned char ch : text) {
        if (std::isalnum(ch) || ch >= 0x80) {
            token.push_back(static_cast<char>(std::tolower(ch)));
        } else {
            addToken();
        }
    }
    addToken();
    return normalizeConcept(std::move(vector));
}

std::vector<float> mediaConcept(const std::vector<uint8_t> &payload, size_t dim, unsigned int seed) {
    std::vector<float> features(dim, 0.0f);
    if (payload.empty()) return features;
    for (size_t i = 0; i < payload.size(); ++i) {
        const float sample = (static_cast<float>(payload[i]) - 127.5f) / 127.5f;
        const size_t primary = (i * 1315423911u + payload[i] + seed) % dim;
        const size_t derivative = (i * 2654435761u + seed) % dim;
        features[primary] += sample;
        if (i > 0) features[derivative] += sample - (static_cast<float>(payload[i - 1]) - 127.5f) / 127.5f;
    }
    return normalizeConcept(std::move(features));
}

/**
 * @brief A persistent concept matrix for multimodal SemanticUnits.
 *
 * Implements a residual-driven associative memory:
 *   - Inference computes the residual between a new concept and the nearest
 *     stored prototype.  If the residual is below the add threshold the
 *     prototype is updated (moving average); otherwise the new concept is
 *     added as a new prototype.  This is the "reasoning optimizes the matrix"
 *     step.
 *   - Pretraining uses the same path but forces an immediate save.
 */
class PersistentConceptMatrix {
 public:
    struct Result {
        phoenix::multimodal::SemanticUnit unit;
        float residual = 0.0f;
        std::string action;
    };

    PersistentConceptMatrix() = default;

    Result addOrUpdate(const phoenix::multimodal::SemanticUnit &query, bool pretrain) {
        if (query.semanticVector.empty()) return {query, 0.0f, "empty"};
        load();
        std::lock_guard<std::mutex> lock(mu_);

        size_t nearestIdx = 0;
        float sim = 0.0f;
        const bool hasNearest = findNearestLocked(query, nearestIdx, sim);
        const float residual = hasNearest ? (1.0f - sim) : 1.0f;

        Result result;
        result.residual = residual;

        if (!hasNearest || residual > addThreshold_) {
            if (entries_.size() >= maxEntries_) evictOldestLocked();
            phoenix::multimodal::SemanticUnit u = query;
            if (u.timestampMs == 0) u.timestampMs = nowMs();
            entries_.push_back({u, 1, residual});
            result.unit = std::move(u);
            result.action = "added";
            ++additions_;
        } else {
            auto &entry = entries_[nearestIdx];
            const size_t n = entry.count;
            std::vector<float> &stored = entry.unit.semanticVector;
            const size_t dim = std::min(stored.size(), query.semanticVector.size());
            for (size_t i = 0; i < dim; ++i) {
                stored[i] = (stored[i] * static_cast<float>(n) + query.semanticVector[i]) / static_cast<float>(n + 1);
            }
            stored = phoenix::multimodal::normalizeVector(stored);
            entry.unit.timestampMs = nowMs();
            entry.unit.metadata["lastResidual"] = std::to_string(residual);
            ++entry.count;
            entry.residual = residual;
            result.unit = entry.unit;
            result.action = "updated";
            ++updates_;
        }

        if (pretrain || ++inferenceSaves_ % 32 == 0) save();
        return result;
    }

    nlohmann::json status() const {
        std::lock_guard<std::mutex> lock(mu_);
        nlohmann::json j;
        j["count"] = entries_.size();
        j["maxEntries"] = maxEntries_;
        j["additions"] = additions_;
        j["updates"] = updates_;
        j["addThreshold"] = addThreshold_;
        j["learnThreshold"] = learnThreshold_;
        j["path"] = path_;
        j["loaded"] = loaded_;
        return j;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mu_);
        entries_.clear();
        loaded_ = true;
        additions_ = 0;
        updates_ = 0;
        inferenceSaves_ = 0;
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

 private:
    struct Entry {
        phoenix::multimodal::SemanticUnit unit;
        size_t count = 1;
        float residual = 0.0f;
    };

    bool findNearestLocked(const phoenix::multimodal::SemanticUnit &query, size_t &idx, float &sim) const {
        const size_t dim = query.semanticVector.size();
        float best = -1.0f;
        bool found = false;
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (entries_[i].unit.semanticVector.size() != dim) continue;
            const float s = phoenix::multimodal::cosineSimilarity(query.semanticVector, entries_[i].unit.semanticVector);
            if (s > best) {
                best = s;
                idx = i;
                found = true;
            }
        }
        sim = best;
        return found;
    }

    void evictOldestLocked() {
        if (entries_.empty()) return;
        size_t idx = 0;
        uint64_t oldest = entries_[0].unit.timestampMs;
        for (size_t i = 1; i < entries_.size(); ++i) {
            if (entries_[i].unit.timestampMs < oldest) {
                oldest = entries_[i].unit.timestampMs;
                idx = i;
            }
        }
        entries_.erase(entries_.begin() + idx);
    }

    uint64_t nowMs() const {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count());
    }

    void load() {
        if (loaded_) return;
        loaded_ = true;
        std::ifstream in(path_);
        if (!in) return;
        try {
            nlohmann::json j;
            in >> j;
            if (!j.is_array()) return;
            std::lock_guard<std::mutex> lock(mu_);
            for (const auto &item : j) {
                if (!item.is_object() || !item.contains("unit")) continue;
                Entry e;
                e.unit = phoenix::multimodal::SemanticUnit::fromJson(item["unit"]);
                e.count = item.value("count", 1u);
                e.residual = item.value("residual", 0.0f);
                entries_.push_back(std::move(e));
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(mu_);
            entries_.clear();
        }
    }

    bool save() const {
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(path_).parent_path(), ec);
        nlohmann::json j = nlohmann::json::array();
        // Caller (addOrUpdate) already holds mu_, so do not re-lock.
        for (const auto &e : entries_) {
            nlohmann::json item;
            item["unit"] = e.unit.toJson();
            item["count"] = e.count;
            item["residual"] = e.residual;
            j.push_back(item);
        }
        std::ofstream out(path_, std::ios::trunc);
        if (!out) return false;
        out << j.dump(2);
        return static_cast<bool>(out);
    }

    mutable std::mutex mu_;
    std::vector<Entry> entries_;
    size_t maxEntries_ = 4096;
    float addThreshold_ = 0.5f;
    float learnThreshold_ = 0.2f;
    std::string path_ = "runtime_store/concept_matrix.json";
    bool loaded_ = false;
    size_t additions_ = 0;
    size_t updates_ = 0;
    mutable size_t inferenceSaves_ = 0;
};

PersistentConceptMatrix gConceptMatrix;

constexpr const char *kDefaultTextEncoderCheckpoint = "runtime_store/transformer_text_encoder.json";

struct TransformerTextEncoder {
    transformer::TransformerParams params_;
    std::unique_ptr<transformer::TransformerModel> model_;
    std::unique_ptr<transformer::Tokenizer> tokenizer_;
    bool loaded_ = false;
    std::string checkpointPath_;
    std::string loadError_;

    static transformer::TransformerParams paramsFromJson(const nlohmann::json &j) {
        transformer::TransformerParams p;
        if (!j.is_object()) return p;
        if (j.contains("vocabSize") && j["vocabSize"].is_number_integer()) p.vocabSize = j["vocabSize"].get<int>();
        if (j.contains("dModel") && j["dModel"].is_number_integer()) p.dModel = j["dModel"].get<int>();
        if (j.contains("nHeads") && j["nHeads"].is_number_integer()) p.nHeads = j["nHeads"].get<int>();
        if (j.contains("nLayers") && j["nLayers"].is_number_integer()) p.nLayers = j["nLayers"].get<int>();
        if (j.contains("dFF") && j["dFF"].is_number_integer()) p.dFF = j["dFF"].get<int>();
        if (j.contains("maxLen") && j["maxLen"].is_number_integer()) p.maxLen = j["maxLen"].get<int>();
        if (j.contains("maxTokens") && j["maxTokens"].is_number_integer()) p.maxTokens = j["maxTokens"].get<int>();
        if (j.contains("tokenizerMode") && j["tokenizerMode"].is_string()) p.tokenizerMode = j["tokenizerMode"].get<std::string>();
        return p;
    }

    explicit TransformerTextEncoder(std::string checkpointPath)
        : params_(transformer::TransformerParams{}), tokenizer_(std::make_unique<transformer::Tokenizer>(params_.vocabSize, params_.tokenizerMode)), checkpointPath_(std::move(checkpointPath)) {}

    void load() {
        if (loaded_ || checkpointPath_.empty()) return;
        std::ifstream input(checkpointPath_);
        if (!input) {
            loadError_ = "checkpoint not found";
            return;
        }
        try {
            nlohmann::json state;
            input >> state;
            nlohmann::json pjson = (state.is_object() && state.contains("params") && state["params"].is_object()) ? state["params"] : state;
            params_ = paramsFromJson(pjson);
            tokenizer_ = std::make_unique<transformer::Tokenizer>(params_.vocabSize, params_.tokenizerMode);
            model_ = std::make_unique<transformer::TransformerModel>(params_);
            if (state.is_object() && state.contains("stateDict") && state["stateDict"].is_object()) {
                std::string err;
                (void)model_->loadStateDict(state["stateDict"], err);
                loadError_ = err;
            }
            loaded_ = true;
        } catch (const std::exception &ex) {
            loadError_ = ex.what();
            model_.reset();
            loaded_ = false;
        }
    }

    std::vector<float> encode(const std::string &text, size_t dim) {
        if (!loaded_) load();
        if (!model_) return {};
        if (!tokenizer_) return {};
        auto tokens = tokenizer_->encode(text, params_.maxLen, false);
        if (tokens.empty()) return {};
        auto hidden = model_->encode(tokens);
        if (hidden.empty() || hidden[0].empty()) return {};
        std::vector<float> pooled(hidden[0].size(), 0.0f);
        for (const auto &row : hidden) {
            for (size_t i = 0; i < pooled.size() && i < row.size(); ++i) {
                pooled[i] += row[i];
            }
        }
        const float invN = 1.0f / static_cast<float>(hidden.size());
        for (float &v : pooled) v *= invN;
        auto conceptVector = phoenix::multimodal::projectToDimension(pooled, dim, 0x54455845U);
        return normalizeConcept(std::move(conceptVector));
    }

    nlohmann::json status() const {
        nlohmann::json j;
        j["loaded"] = loaded_;
        j["checkpoint"] = checkpointPath_;
        if (!loadError_.empty()) j["error"] = loadError_;
        if (loaded_) {
            j["params"] = {{"vocabSize", params_.vocabSize}, {"dModel", params_.dModel}, {"nHeads", params_.nHeads}, {"nLayers", params_.nLayers}, {"tokenizerMode", params_.tokenizerMode}};
        }
        return j;
    }
};

std::mutex gTextEncoderMutex;
std::unique_ptr<TransformerTextEncoder> gTextEncoder;

std::string textEncoderCheckpointFromMetadata(const nlohmann::json &metadata) {
    if (metadata.contains("textEncoderCheckpoint") && metadata["textEncoderCheckpoint"].is_string()) {
        return metadata["textEncoderCheckpoint"].get<std::string>();
    }
    return kDefaultTextEncoderCheckpoint;
}

std::vector<float> transformerTextEncoderConcept(const std::string &text, size_t dim, const std::string &checkpointPath) {
    if (checkpointPath.empty()) return tokenEncoderConcept(text, dim);
    std::lock_guard<std::mutex> lock(gTextEncoderMutex);
    if (!gTextEncoder || gTextEncoder->checkpointPath_ != checkpointPath) {
        gTextEncoder = std::make_unique<TransformerTextEncoder>(checkpointPath);
    }
    auto encoded = gTextEncoder->encode(text, dim);
    if (encoded.empty()) return tokenEncoderConcept(text, dim);
    return encoded;
}

nlohmann::json textEncoderStatus() {
    std::lock_guard<std::mutex> lock(gTextEncoderMutex);
    if (gTextEncoder) return gTextEncoder->status();
    nlohmann::json j;
    j["loaded"] = false;
    j["checkpoint"] = kDefaultTextEncoderCheckpoint;
    j["note"] = "not initialized; using token-encoder fallback";
    return j;
}

phoenix::multimodal::Modality semanticModality(MixedModalModality modality) {
    switch (modality) {
        case MixedModalModality::Image: return phoenix::multimodal::Modality::Image;
        case MixedModalModality::Audio: return phoenix::multimodal::Modality::Audio;
        case MixedModalModality::Video: return phoenix::multimodal::Modality::Video;
        case MixedModalModality::Sensor: return phoenix::multimodal::Modality::Sensor;
        case MixedModalModality::Structured: return phoenix::multimodal::Modality::Structured;
        default: return phoenix::multimodal::Modality::Text;
    }
}

}  // namespace

/**
 * @brief Encode an external mixed-modal packet into a modality-agnostic SemanticUnit.
 *
 * Resolution order:
 *   1. If metadata carries a pre-computed concept/world-model/encoder vector, use it.
 *   2. Text payloads are tokenised with a lightweight encoder adapter.
 *   3. Image/Video payloads are dispatched to the external LLaVA-based image world model.
 *      The mean unit query becomes the semantic vector; the per-patch unit query
 *      sequence is preserved for attention-based decoding.
 *   4. Audio payloads are dispatched to the external Qwen2-Audio world model.
 *   5. Structured/sensor payloads fall back to a generic hashed concept.
 *   6. The resulting SemanticUnit is submitted to the persistent ConceptMatrix.
 *      The residual to the nearest prototype is used to either add a new prototype
 *      (large residual) or refine an existing one (small residual).  The returned
 *      unit carries the matrix-corrected concept vector.
 */
phoenix::multimodal::SemanticUnit MixedModalConceptBridge::encode(const MixedModalPacket &packet,
                                                                   size_t targetDim,
                                                                   const std::string &contentHint) {
    const size_t dim = conceptDimension(targetDim);
    phoenix::multimodal::SemanticUnit unit;
    unit.id = packet.id.empty() ? phoenix::multimodal::generateSemanticId(contentHint) : packet.id;
    unit.modality = semanticModality(packet.modality);
    unit.confidence = packet.payload.empty() ? 0.0f : 1.0f;
    unit.timestampMs = packet.timestampMs;
    unit.metadata["source"] = packet.source;
    unit.metadata["mimeType"] = packet.mimeType;
    unit.metadata["modality"] = phoenix::multimodal::modalityToString(unit.modality);
    unit.modalWeights[phoenix::multimodal::modalityToString(unit.modality)] = 1.0f;

    const auto suppliedConcept = metadataConcept(packet.metadata, dim);
    if (!suppliedConcept.empty()) {
        unit.semanticVector = suppliedConcept;
        unit.metadata["conceptEncoder"] = "world-model";
    } else if (packet.modality == MixedModalModality::Text) {
        unit.content.assign(packet.payload.begin(), packet.payload.end());
        const auto checkpoint = textEncoderCheckpointFromMetadata(packet.metadata);
        auto encoded = transformerTextEncoderConcept(unit.content, dim, checkpoint);
        if (encoded.empty()) {
            unit.semanticVector = tokenEncoderConcept(unit.content, dim);
            unit.metadata["conceptEncoder"] = "token-encoder-adapter";
        } else {
            unit.semanticVector = std::move(encoded);
            unit.metadata["conceptEncoder"] = "transformer-text-encoder";
            unit.metadata["textEncoderCheckpoint"] = checkpoint;
        }
    } else if (packet.modality == MixedModalModality::Image || packet.modality == MixedModalModality::Video) {
        int width = 0, height = 0;
        if (packet.metadata.contains("width") && packet.metadata["width"].is_number()) {
            width = packet.metadata["width"].get<int>();
        }
        if (packet.metadata.contains("height") && packet.metadata["height"].is_number()) {
            height = packet.metadata["height"].get<int>();
        }
        auto &imageModel = imageWorldModel(buildMultimodalEncDecConfig());
        auto result = imageModel.encode(packet.payload, width, height, packet.mimeType);
        unit.semanticVector = std::move(result.meanUnitQuery);
        if (!result.unitQueries.empty()) {
            unit.unitQuerySequence = std::move(result.unitQueries);
        }
        if (width > 0) unit.metadata["width"] = std::to_string(width);
        if (height > 0) unit.metadata["height"] = std::to_string(height);
        unit.metadata["conceptEncoder"] = imageVariantFromMetadata(packet.metadata);
        unit.metadata["multimodalBackend"] = "external-python-service";
        if (unit.semanticVector.empty()) {
            unit.confidence = 0.0f;
            unit.metadata["multimodalEncodeError"] = result.error.empty() ? "image encoder returned empty mean unit query" : result.error;
        } else if (!result.error.empty()) {
            unit.metadata["multimodalEncodeError"] = result.error;
        }
    } else if (packet.modality == MixedModalModality::Audio) {
        const int sampleRate = sampleRateFromMetadata(packet.metadata);
        auto &audioModel = audioWorldModel(buildMultimodalEncDecConfig());
        auto result = audioModel.encode(packet.payload, sampleRate, packet.mimeType);
        unit.semanticVector = std::move(result.meanUnitQuery);
        if (!result.unitQueries.empty()) {
            unit.unitQuerySequence = std::move(result.unitQueries);
        }
        unit.metadata["conceptEncoder"] = speechVariantFromMetadata(packet.metadata);
        unit.metadata["multimodalBackend"] = "external-python-service";
        if (unit.semanticVector.empty()) {
            unit.confidence = 0.0f;
            unit.metadata["multimodalEncodeError"] = result.error.empty() ? "audio encoder returned empty mean unit query" : result.error;
        } else if (!result.error.empty()) {
            unit.metadata["multimodalEncodeError"] = result.error;
        }
    } else {
        unit.semanticVector = mediaConcept(packet.payload, dim, 0x53545255U);
        unit.metadata["conceptEncoder"] = "structured-concept-adapter";
    }

    // Concept-matrix reasoning: residual to nearest prototype decides whether
    // to add a new concept or refine an existing one.  The returned vector is
    // the (possibly corrected) matrix prototype.
    auto matrixResult = gConceptMatrix.addOrUpdate(unit, false);
    unit.semanticVector = std::move(matrixResult.unit.semanticVector);
    unit.metadata["conceptMatrixResidual"] = std::to_string(matrixResult.residual);
    unit.metadata["conceptMatrixAction"] = matrixResult.action;

    return unit;
}

/**
 * @brief Pre-train the audio world model and the shared concept matrix.
 *
 * Encodes the audio clip with the external Qwen2-Audio service, encodes the
 * transcript with the transformer text encoder, and stores both as associated
 * concepts in the persistent concept matrix.
 */
bool MixedModalConceptBridge::pretrainSpeech(const MixedModalPacket &audio,
                                             const std::string &transcript,
                                             size_t targetDim) {
    if (audio.modality != MixedModalModality::Audio || audio.payload.empty() || transcript.empty()) return false;
    const size_t dim = conceptDimension(targetDim);
    const int sampleRate = sampleRateFromMetadata(audio.metadata);
    auto &model = audioWorldModel(buildMultimodalEncDecConfig());
    auto result = model.encode(audio.payload, sampleRate, audio.mimeType);
    if (result.meanUnitQuery.empty()) return false;

    const auto correlationId = phoenix::multimodal::generateSemanticId(transcript);
    phoenix::multimodal::SemanticUnit audioUnit;
    audioUnit.id = audio.id.empty() ? correlationId + "-audio" : audio.id;
    audioUnit.modality = phoenix::multimodal::Modality::Audio;
    audioUnit.semanticVector = std::move(result.meanUnitQuery);
    if (!result.unitQueries.empty()) {
        audioUnit.unitQuerySequence = std::move(result.unitQueries);
    }
    audioUnit.confidence = 1.0f;
    audioUnit.timestampMs = audio.timestampMs;
    audioUnit.metadata["source"] = audio.source;
    audioUnit.metadata["mimeType"] = audio.mimeType;
    audioUnit.metadata["conceptEncoder"] = speechVariantFromMetadata(audio.metadata);
    audioUnit.metadata["multimodalBackend"] = "external-python-service";
    audioUnit.metadata["transcript"] = transcript;
    audioUnit.associationIds.push_back(correlationId);
    gConceptMatrix.addOrUpdate(audioUnit, true);

    const auto textConcept = transformerTextEncoderConcept(transcript, dim, kDefaultTextEncoderCheckpoint);
    if (!textConcept.empty()) {
        phoenix::multimodal::SemanticUnit textUnit;
        textUnit.id = correlationId + "-text";
        textUnit.modality = phoenix::multimodal::Modality::Text;
        textUnit.semanticVector = textConcept;
        textUnit.content = transcript;
        textUnit.confidence = 1.0f;
        textUnit.timestampMs = audio.timestampMs;
        textUnit.metadata["source"] = "transcript";
        textUnit.metadata["conceptEncoder"] = "transformer-text-encoder";
        textUnit.associationIds.push_back(correlationId);
        gConceptMatrix.addOrUpdate(textUnit, true);
    }

    return true;
}

/**
 * @brief Pre-train the image world model and the shared concept matrix.
 *
 * Encodes the image with the external LLaVA service, optionally encodes the
 * caption with the transformer text encoder, and stores both as associated
 * concepts in the persistent concept matrix.
 */
bool MixedModalConceptBridge::pretrainImage(const MixedModalPacket &image,
                                            const std::string &caption,
                                            size_t targetDim) {
    if ((image.modality != MixedModalModality::Image && image.modality != MixedModalModality::Video) ||
        image.payload.empty())
        return false;
    const size_t dim = conceptDimension(targetDim);
    int width = 0, height = 0;
    if (image.metadata.contains("width") && image.metadata["width"].is_number()) {
        width = image.metadata["width"].get<int>();
    }
    if (image.metadata.contains("height") && image.metadata["height"].is_number()) {
        height = image.metadata["height"].get<int>();
    }
    auto &model = imageWorldModel(buildMultimodalEncDecConfig());
    auto result = model.encode(image.payload, width, height, image.mimeType);
    if (result.meanUnitQuery.empty()) return false;

    const auto correlationId = phoenix::multimodal::generateSemanticId(caption);
    phoenix::multimodal::SemanticUnit imageUnit;
    imageUnit.id = image.id.empty() ? (caption.empty() ? phoenix::multimodal::generateSemanticId()
                                                       : correlationId + "-image")
                                    : image.id;
    imageUnit.modality = phoenix::multimodal::Modality::Image;
    imageUnit.semanticVector = std::move(result.meanUnitQuery);
    if (!result.unitQueries.empty()) {
        imageUnit.unitQuerySequence = std::move(result.unitQueries);
    }
    imageUnit.confidence = 1.0f;
    imageUnit.timestampMs = image.timestampMs;
    imageUnit.metadata["source"] = image.source;
    imageUnit.metadata["mimeType"] = image.mimeType;
    if (width > 0) imageUnit.metadata["width"] = std::to_string(width);
    if (height > 0) imageUnit.metadata["height"] = std::to_string(height);
    imageUnit.metadata["conceptEncoder"] = imageVariantFromMetadata(image.metadata);
    imageUnit.metadata["multimodalBackend"] = "external-python-service";
    if (!caption.empty()) {
        imageUnit.metadata["caption"] = caption;
        imageUnit.associationIds.push_back(correlationId);
    }
    gConceptMatrix.addOrUpdate(imageUnit, true);

    if (!caption.empty()) {
        const auto textConcept = transformerTextEncoderConcept(caption, dim, kDefaultTextEncoderCheckpoint);
        if (!textConcept.empty()) {
            phoenix::multimodal::SemanticUnit textUnit;
            textUnit.id = correlationId + "-text";
            textUnit.modality = phoenix::multimodal::Modality::Text;
            textUnit.semanticVector = textConcept;
            textUnit.content = caption;
            textUnit.confidence = 1.0f;
            textUnit.timestampMs = image.timestampMs;
            textUnit.metadata["source"] = "caption";
            textUnit.metadata["conceptEncoder"] = "transformer-text-encoder";
            textUnit.associationIds.push_back(correlationId);
            gConceptMatrix.addOrUpdate(textUnit, true);
        }
    }

    return true;
}

/** Reset all persistent concept state. */
void MixedModalConceptBridge::reset() {
    gConceptMatrix.clear();
}

/**
 * @brief Adapt an abstract SemanticUnit back into an external MixedModalPacket.
 *
 * For text the original content bytes are emitted directly.  For image and video
 * the external LLaVA-based decoder is used.  For audio the external Qwen2-Audio
 * decoder is used.  For all other modalities a JSON concept payload is produced
 * and marked with requiresModalityDecoder so the downstream modality-specific
 * renderer can materialise the entity.
 */
MixedModalPacket MixedModalConceptBridge::decode(const phoenix::multimodal::SemanticUnit &unit,
                                                   MixedModalModality target,
                                                   const std::string &source) {
    MixedModalPacket packet;
    packet.id = unit.id;
    packet.modality = target;
    packet.source = source;
    if (target == MixedModalModality::Text) {
        packet.mimeType = "text/plain; charset=utf-8";
        packet.payload.assign(unit.content.begin(), unit.content.end());
    } else if (target == MixedModalModality::Image || target == MixedModalModality::Video) {
        int width = 224, height = 224;
        auto wit = unit.metadata.find("width");
        if (wit != unit.metadata.end() && !wit->second.empty()) {
            try {
                width = std::stoi(wit->second);
            } catch (...) {
                width = 224;
            }
        }
        auto hit = unit.metadata.find("height");
        if (hit != unit.metadata.end() && !hit->second.empty()) {
            try {
                height = std::stoi(hit->second);
            } catch (...) {
                height = 224;
            }
        }
        auto &model = imageWorldModel(buildMultimodalEncDecConfig());
        auto result = model.decode(unit.semanticVector, unit.unitQuerySequence, "image/png", width, height);
        packet.payload = std::move(result.payload);
        packet.mimeType = result.mimeType.empty() ? "image/png" : result.mimeType;
        if (!result.error.empty()) {
            packet.metadata["imageDecodeError"] = result.error;
        }
        packet.metadata["sourceModality"] = phoenix::multimodal::modalityToString(unit.modality);
        packet.metadata["conceptVector"] = unit.semanticVector;
    } else if (target == MixedModalModality::Audio) {
        size_t lengthHint = 0;
        auto it = unit.metadata.find("lengthHint");
        if (it != unit.metadata.end() && !it->second.empty()) {
            try {
                lengthHint = static_cast<size_t>(std::stoull(it->second));
            } catch (...) {
                lengthHint = 0;
            }
        }
        auto &model = audioWorldModel(buildMultimodalEncDecConfig());
        auto result = model.decode(unit.semanticVector, unit.unitQuerySequence, "audio/wav", lengthHint);
        packet.payload = std::move(result.payload);
        packet.mimeType = result.mimeType.empty() ? "audio/wav" : result.mimeType;
        if (!result.error.empty()) {
            packet.metadata["audioDecodeError"] = result.error;
        }
        packet.metadata["sourceModality"] = phoenix::multimodal::modalityToString(unit.modality);
        packet.metadata["conceptVector"] = unit.semanticVector;
    } else {
        packet.mimeType = "application/json";
        nlohmann::json conceptPayload{{"semanticVector", unit.semanticVector},
                                      {"sourceModality", phoenix::multimodal::modalityToString(unit.modality)},
                                      {"content", unit.content}};
        const auto serialized = conceptPayload.dump();
        packet.payload.assign(serialized.begin(), serialized.end());
        packet.metadata["requiresModalityDecoder"] = true;
    }
    return packet;
}

/**
 * @brief Report the state of the multimodal image/audio world models,
 *        the text encoder, and the shared concept matrix.
 */
nlohmann::json MixedModalConceptBridge::status() {
    nlohmann::json j;
    try {
        const auto cfg = buildMultimodalEncDecConfig();
        j["imageWorldModel"] = imageWorldModel(cfg).status();
    } catch (...) {
        j["imageWorldModel"] = {{"error", "image world model not available"}};
    }
    try {
        const auto cfg = buildMultimodalEncDecConfig();
        j["audioWorldModel"] = audioWorldModel(cfg).status();
    } catch (...) {
        j["audioWorldModel"] = {{"error", "audio world model not available"}};
    }
    j["textEncoder"] = textEncoderStatus();
    j["conceptMatrix"] = gConceptMatrix.status();
    return j;
}

/**
 * @brief Convenience dispatcher: packet -> concept bridge -> SemanticUnit.
 */
phoenix::multimodal::SemanticUnit MixedModalPacket::toSemanticUnit(size_t targetDim,
                                                                   const std::string &contentHint) const {
    return MixedModalConceptBridge::encode(*this, targetDim, contentHint);
}

void MixedModalInputBuffer::push(MixedModalPacket packet) {
    if (packet.timestampMs == 0) {
        packet.timestampMs = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
    }
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_.push_back(std::move(packet));
}

std::vector<MixedModalPacket> MixedModalInputBuffer::flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<MixedModalPacket> out;
    out.swap(buffer_);
    return out;
}

bool MixedModalInputBuffer::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return buffer_.empty();
}

size_t MixedModalInputBuffer::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return buffer_.size();
}

nlohmann::json MixedModalInputBuffer::toJson() const {
    std::lock_guard<std::mutex> lock(mutex_);
    nlohmann::json arr = nlohmann::json::array();
    for (const auto &p : buffer_) arr.push_back(p.toJson());
    return arr;
}

void MixedModalOutputQueue::push(MixedModalPacket packet) {
    if (packet.timestampMs == 0) {
        packet.timestampMs = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
    }
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push_back(std::move(packet));
}

std::vector<MixedModalPacket> MixedModalOutputQueue::drain(size_t max) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (max == 0 || max >= queue_.size()) {
        std::vector<MixedModalPacket> out;
        out.swap(queue_);
        return out;
    }
    std::vector<MixedModalPacket> out(queue_.begin(), queue_.begin() + max);
    queue_.erase(queue_.begin(), queue_.begin() + max);
    return out;
}

bool MixedModalOutputQueue::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
}

size_t MixedModalOutputQueue::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

nlohmann::json MixedModalOutputQueue::toJson() const {
    std::lock_guard<std::mutex> lock(mutex_);
    nlohmann::json arr = nlohmann::json::array();
    for (const auto &p : queue_) arr.push_back(p.toJson());
    return arr;
}

void MixedModalChannelRegistry::registerSource(const std::string &name, const std::string &mimeType) {
    std::lock_guard<std::mutex> lock(mutex_);
    sources_[name] = mimeType;
}

std::vector<std::string> MixedModalChannelRegistry::sources() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> out;
    out.reserve(sources_.size());
    for (const auto &kv : sources_) out.push_back(kv.first);
    return out;
}

nlohmann::json MixedModalChannelRegistry::toJson() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return sources_;
}

}  // namespace io
}  // namespace phoenix
