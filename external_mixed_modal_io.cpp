/* external_mixed_modal_io.cpp - Implementation for external mixed-modal I/O
   Copyright (C) 2026 079 Project */

#include "external_mixed_modal_io.hpp"
#include "video_model.hpp"
#include "audio_model.hpp"
#include "phoenix_config.hpp"
#include "transformer.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <numeric>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#ifdef HAVE_SQLITE
#include <sqlite3.h>
#endif

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

static size_t gLastRequestedTargetDim = 64;

void recordRequestedTargetDim(size_t dim) {
    if (dim > 0) gLastRequestedTargetDim = dim;
}

std::vector<float> normalizeConcept(std::vector<float> value) {
    return phoenix::multimodal::normalizeVector(value);
}

VideoModel &videoEncoder(size_t targetDim = 0) {
    static std::mutex mu;
    static std::unordered_map<int, std::unique_ptr<VideoModel>> cache;
    std::lock_guard<std::mutex> lock(mu);
    int key = static_cast<int>(targetDim);
    auto it = cache.find(key);
    if (it != cache.end()) return *it->second;
    auto model = createVideoEncoder("video-encoder", static_cast<int>(targetDim));
    auto &ref = *model;
    cache.emplace(key, std::move(model));
    return ref;
}

VideoModel &videoDecoder(size_t targetDim = 0) {
    static std::mutex mu;
    static std::unordered_map<int, std::unique_ptr<VideoModel>> cache;
    std::lock_guard<std::mutex> lock(mu);
    int key = static_cast<int>(targetDim);
    auto it = cache.find(key);
    if (it != cache.end()) return *it->second;
    auto model = createVideoDecoder("video-encoder", static_cast<int>(targetDim));
    auto &ref = *model;
    cache.emplace(key, std::move(model));
    return ref;
}

std::string imageVariantFromMetadata(const nlohmann::json &metadata) {
    (void)metadata;
    return "video-encoder";
}

AudioModel &audioEncoder(size_t targetDim = 0) {
    static std::mutex mu;
    static std::unordered_map<int, std::unique_ptr<AudioModel>> cache;
    std::lock_guard<std::mutex> lock(mu);
    int key = static_cast<int>(targetDim);
    auto it = cache.find(key);
    if (it != cache.end()) return *it->second;
    auto model = createAudioEncoder("audio-encoder", static_cast<int>(targetDim));
    auto &ref = *model;
    cache.emplace(key, std::move(model));
    return ref;
}

AudioModel &audioDecoder(size_t targetDim = 0) {
    static std::mutex mu;
    static std::unordered_map<int, std::unique_ptr<AudioModel>> cache;
    std::lock_guard<std::mutex> lock(mu);
    int key = static_cast<int>(targetDim);
    auto it = cache.find(key);
    if (it != cache.end()) return *it->second;
    auto model = createAudioDecoder("audio-encoder", static_cast<int>(targetDim));
    auto &ref = *model;
    cache.emplace(key, std::move(model));
    return ref;
}

std::string speechVariantFromMetadata(const nlohmann::json &metadata) {
    (void)metadata;
    return "audio-encoder";
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
    ~PersistentConceptMatrix() {
        std::lock_guard<std::mutex> lock(mu_);
        if (!entries_.empty() || !deletedIds_.empty()) save();
        closeDb();
    }

    Result addOrUpdate(const phoenix::multimodal::SemanticUnit &query, bool pretrain) {
        if (query.semanticVector.empty()) return {query, 0.0f, "empty"};
        std::lock_guard<std::mutex> lock(mu_);
        load();

        int64_t nearestId = -1;
        float sim = 0.0f;
        const bool hasNearest = findNearestLocked(query, nearestId, sim);
        const float residual = hasNearest ? (1.0f - sim) : 1.0f;

        Result result;
        result.residual = residual;

        if (!hasNearest || residual > addThreshold_) {
            if (entries_.size() >= maxEntries_) evictOldestLocked();
            phoenix::multimodal::SemanticUnit u = query;
            if (u.timestampMs == 0) u.timestampMs = nowMs();
            int64_t id = nextId_++;
            Entry e;
            e.unit = std::move(u);
            e.count = 1;
            e.residual = residual;
            e.id = id;
            e.dirty = true;
            maybeRecomputeAndAddToLsh(e, id);
            auto it = entries_.emplace(id, std::move(e));
            result.unit = it.first->second.unit;
            result.action = "added";
            ++additions_;
        } else {
            auto it = entries_.find(nearestId);
            if (it != entries_.end()) {
                Entry &entry = it->second;
                removeFromLsh(entry, nearestId);
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
                entry.dirty = true;
                maybeRecomputeAndAddToLsh(entry, nearestId);
                result.unit = entry.unit;
                result.action = "updated";
                ++updates_;
            } else {
                result.action = "error";
            }
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
        for (const auto &kv : entries_) {
            deletedIds_.push_back(kv.first);
        }
        entries_.clear();
        lshCache_.clear();
        nextId_ = 1;
        loaded_ = true;
        additions_ = 0;
        updates_ = 0;
        inferenceSaves_ = 0;
#ifdef HAVE_SQLITE
        save();
#else
        std::error_code ec;
        std::filesystem::remove(path_, ec);
        deletedIds_.clear();
#endif
    }

 private:
    struct Entry {
        phoenix::multimodal::SemanticUnit unit;
        size_t count = 1;
        float residual = 0.0f;
        int64_t id = 0;
        std::vector<uint64_t> lshSignatures;
        bool dirty = true;
    };

    struct LSHIndex {
        int dim = 0;
        int L = 0;
        int b = 0;
        std::vector<std::vector<float>> hyperplanes;
        std::vector<std::unordered_map<uint64_t, std::vector<int64_t>>> buckets;
    };

    static constexpr int kLshL = 8;
    static constexpr int kLshBits = 16;
    static constexpr size_t kBruteForceLimit = 256;
    static constexpr size_t kDimBruteForceLimit = 32;
    static constexpr float kExactThreshold = 1.0f - 1e-4f;
    static constexpr float kLshFallbackThreshold = 0.95f;
    static constexpr uint64_t kLshSeedA = 0x9E3779B97F4A7C15ULL;
    static constexpr uint64_t kLshSeedB = 0x123456789ABCDEFULL;

    bool bruteForceFindLocked(const phoenix::multimodal::SemanticUnit &query,
                              int64_t &bestId,
                              float &bestSim) const {
        const size_t dim = query.semanticVector.size();
        float best = -1.0f;
        bool found = false;
        int64_t foundId = -1;
        for (const auto &kv : entries_) {
            if (kv.second.unit.semanticVector.size() != dim) continue;
            const float s = phoenix::multimodal::cosineSimilarity(query.semanticVector, kv.second.unit.semanticVector);
            if (s > best) {
                best = s;
                foundId = kv.first;
                found = true;
                if (s >= kExactThreshold) break;
            }
        }
        if (found) {
            bestId = foundId;
            bestSim = best;
        }
        return found;
    }

    bool findNearestLocked(const phoenix::multimodal::SemanticUnit &query,
                           int64_t &bestId,
                           float &bestSim) {
        const size_t dim = query.semanticVector.size();
        const size_t n = entries_.size();
        if (n <= kBruteForceLimit || dim < kDimBruteForceLimit) {
            return bruteForceFindLocked(query, bestId, bestSim);
        }

        LSHIndex *lsh = ensureLshIndex(dim);
        if (!lsh) return bruteForceFindLocked(query, bestId, bestSim);

        const std::vector<uint64_t> qSigs = computeLshSignatures(query.semanticVector, *lsh);
        std::unordered_set<int64_t> candidates;
        candidates.reserve(lsh->L * 16);
        for (int l = 0; l < lsh->L; ++l) {
            const auto &bucket = lsh->buckets[l];
            auto it = bucket.find(qSigs[l]);
            if (it != bucket.end()) {
                for (int64_t id : it->second) candidates.insert(id);
            }
        }

        if (candidates.empty()) {
            return bruteForceFindLocked(query, bestId, bestSim);
        }

        float best = -1.0f;
        bool found = false;
        int64_t foundId = -1;
        for (int64_t id : candidates) {
            auto it = entries_.find(id);
            if (it == entries_.end()) continue;
            if (it->second.unit.semanticVector.size() != dim) continue;
            const float s = phoenix::multimodal::cosineSimilarity(query.semanticVector, it->second.unit.semanticVector);
            if (s > best) {
                best = s;
                foundId = id;
                found = true;
                if (s >= kExactThreshold) break;
            }
        }

        if (!found || best < kLshFallbackThreshold) {
            return bruteForceFindLocked(query, bestId, bestSim);
        }

        bestId = foundId;
        bestSim = best;
        return true;
    }

    void evictOldestLocked() {
        if (entries_.empty()) return;
        int64_t oldestId = -1;
        uint64_t oldestTs = std::numeric_limits<uint64_t>::max();
        for (const auto &kv : entries_) {
            if (kv.second.unit.timestampMs < oldestTs) {
                oldestTs = kv.second.unit.timestampMs;
                oldestId = kv.first;
            }
        }
        if (oldestId < 0) return;
        auto it = entries_.find(oldestId);
        if (it != entries_.end()) {
            removeFromLsh(it->second, oldestId);
            deletedIds_.push_back(oldestId);
            entries_.erase(it);
        }
    }

    uint64_t nowMs() const {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count());
    }

    std::vector<uint64_t> computeLshSignatures(const std::vector<float> &v,
                                               const LSHIndex &lsh) const {
        std::vector<uint64_t> sigs(lsh.L, 0);
        for (int l = 0; l < lsh.L; ++l) {
            uint64_t sig = 0;
            const float *hp = lsh.hyperplanes[l].data();
            for (int bit = 0; bit < lsh.b; ++bit) {
                const float *p = hp + static_cast<size_t>(bit) * lsh.dim;
                double dot = std::inner_product(v.begin(), v.end(), p, 0.0);
                if (dot >= 0.0) sig |= (1ULL << bit);
            }
            sigs[l] = sig;
        }
        return sigs;
    }

    LSHIndex *ensureLshIndex(size_t dim) {
        auto it = lshCache_.find(dim);
        if (it != lshCache_.end()) return it->second.get();
        if (dim == 0) return nullptr;
        auto lsh = std::make_unique<LSHIndex>();
        lsh->dim = static_cast<int>(dim);
        lsh->L = kLshL;
        lsh->b = kLshBits;
        lsh->hyperplanes.resize(lsh->L, std::vector<float>(lsh->b * lsh->dim));
        lsh->buckets.resize(lsh->L);
        uint64_t seed = static_cast<uint64_t>(dim) * kLshSeedA + kLshSeedB;
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> normal(0.0, 1.0);
        for (int l = 0; l < lsh->L; ++l) {
            for (int bit = 0; bit < lsh->b; ++bit) {
                float *p = lsh->hyperplanes[l].data() + static_cast<size_t>(bit) * lsh->dim;
                double sum2 = 0.0;
                for (int i = 0; i < lsh->dim; ++i) {
                    double v = normal(rng);
                    p[i] = static_cast<float>(v);
                    sum2 += v * v;
                }
                if (sum2 == 0.0) {
                    double inv = 1.0 / std::sqrt(static_cast<double>(dim));
                    for (int i = 0; i < lsh->dim; ++i) p[i] = static_cast<float>(inv);
                } else {
                    double inv = 1.0 / std::sqrt(sum2);
                    for (int i = 0; i < lsh->dim; ++i) p[i] *= static_cast<float>(inv);
                }
            }
        }
        for (auto &kv : entries_) {
            if (kv.second.unit.semanticVector.size() != dim) continue;
            kv.second.lshSignatures = computeLshSignatures(kv.second.unit.semanticVector, *lsh);
            for (int l = 0; l < lsh->L; ++l) {
                lsh->buckets[l][kv.second.lshSignatures[l]].push_back(kv.first);
            }
        }
        LSHIndex *ptr = lsh.get();
        lshCache_[dim] = std::move(lsh);
        return ptr;
    }

    void maybeRecomputeAndAddToLsh(Entry &e, int64_t id) {
        const size_t dim = e.unit.semanticVector.size();
        auto it = lshCache_.find(dim);
        if (it == lshCache_.end()) {
            e.lshSignatures.clear();
            return;
        }
        LSHIndex &lsh = *it->second;
        e.lshSignatures = computeLshSignatures(e.unit.semanticVector, lsh);
        for (int l = 0; l < lsh.L; ++l) {
            lsh.buckets[l][e.lshSignatures[l]].push_back(id);
        }
    }

    void removeFromLsh(Entry &e, int64_t id) {
        if (e.lshSignatures.empty()) return;
        const size_t dim = e.unit.semanticVector.size();
        auto it = lshCache_.find(dim);
        if (it == lshCache_.end()) return;
        LSHIndex &lsh = *it->second;
        if (e.lshSignatures.size() != static_cast<size_t>(lsh.L)) return;
        for (int l = 0; l < lsh.L; ++l) {
            auto bucketIt = lsh.buckets[l].find(e.lshSignatures[l]);
            if (bucketIt == lsh.buckets[l].end()) continue;
            auto &vec = bucketIt->second;
            auto idIt = std::find(vec.begin(), vec.end(), id);
            if (idIt != vec.end()) {
                *idIt = vec.back();
                vec.pop_back();
                if (vec.empty()) lsh.buckets[l].erase(bucketIt);
            }
        }
        e.lshSignatures.clear();
    }

    void load() {
        if (loaded_) return;
#ifdef HAVE_SQLITE
        if (!openDb()) {
            loaded_ = true;
            return;
        }
        const char *sql = "SELECT id, unit_json, count, residual, timestamp_ms FROM concept_matrix;";
        sqlite3_stmt *stmt = nullptr;
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) == SQLITE_OK) {
            while (sqlite3_step(stmt) == SQLITE_ROW) {
                int64_t id = sqlite3_column_int64(stmt, 0);
                const unsigned char *uj = sqlite3_column_text(stmt, 1);
                if (!uj) continue;
                std::string ujStr = reinterpret_cast<const char*>(uj);
                if (ujStr.empty()) continue;
                int count = sqlite3_column_int(stmt, 2);
                double residual = sqlite3_column_double(stmt, 3);
                int64_t ts = sqlite3_column_int64(stmt, 4);
                try {
                    nlohmann::json j = nlohmann::json::parse(ujStr);
                    Entry e;
                    e.unit = phoenix::multimodal::SemanticUnit::fromJson(j);
                    e.count = count > 0 ? static_cast<size_t>(count) : 1;
                    e.residual = static_cast<float>(residual);
                    e.id = id;
                    e.dirty = false;
                    if (e.unit.timestampMs == 0) {
                        e.unit.timestampMs = ts > 0 ? static_cast<uint64_t>(ts) : nowMs();
                    }
                    entries_.emplace(id, std::move(e));
                    if (id >= nextId_) nextId_ = id + 1;
                } catch (...) {}
            }
            sqlite3_finalize(stmt);
        }
#else
        std::ifstream in(path_);
        if (!in) { loaded_ = true; return; }
        try {
            nlohmann::json j;
            in >> j;
            if (j.is_array()) {
                for (const auto &item : j) {
                    if (!item.is_object() || !item.contains("unit")) continue;
                    Entry e;
                    e.unit = phoenix::multimodal::SemanticUnit::fromJson(item["unit"]);
                    e.count = item.value("count", 1u);
                    e.residual = item.value("residual", 0.0f);
                    e.id = nextId_++;
                    e.dirty = false;
                    entries_.emplace(e.id, std::move(e));
                }
            }
        } catch (...) {
            entries_.clear();
        }
#endif
        loaded_ = true;
    }

    bool save() {
#ifdef HAVE_SQLITE
        if (!openDb()) return false;
        const char *sql = "INSERT OR REPLACE INTO concept_matrix (id, unit_json, count, residual, timestamp_ms) VALUES (?, ?, ?, ?, ?);";
        sqlite3_stmt *stmt = nullptr;
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
        execSql("BEGIN TRANSACTION;");
        for (auto &kv : entries_) {
            if (!kv.second.dirty) continue;
            std::string uj = kv.second.unit.toJson().dump();
            sqlite3_bind_int64(stmt, 1, kv.first);
            sqlite3_bind_text(stmt, 2, uj.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_int64(stmt, 3, static_cast<sqlite3_int64>(kv.second.count));
            sqlite3_bind_double(stmt, 4, static_cast<double>(kv.second.residual));
            sqlite3_bind_int64(stmt, 5, static_cast<sqlite3_int64>(kv.second.unit.timestampMs));
            sqlite3_step(stmt);
            sqlite3_reset(stmt);
            sqlite3_clear_bindings(stmt);
            kv.second.dirty = false;
        }
        sqlite3_finalize(stmt);

        if (!deletedIds_.empty()) {
            const char *delSql = "DELETE FROM concept_matrix WHERE id=?;";
            sqlite3_stmt *delStmt = nullptr;
            if (sqlite3_prepare_v2(db_, delSql, -1, &delStmt, nullptr) == SQLITE_OK) {
                for (int64_t id : deletedIds_) {
                    sqlite3_bind_int64(delStmt, 1, id);
                    sqlite3_step(delStmt);
                    sqlite3_reset(delStmt);
                    sqlite3_clear_bindings(delStmt);
                }
                sqlite3_finalize(delStmt);
            }
            deletedIds_.clear();
        }
        execSql("COMMIT;");
        return true;
#else
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(path_).parent_path(), ec);
        nlohmann::json j = nlohmann::json::array();
        for (const auto &kv : entries_) {
            nlohmann::json item;
            item["unit"] = kv.second.unit.toJson();
            item["count"] = kv.second.count;
            item["residual"] = kv.second.residual;
            j.push_back(item);
        }
        std::ofstream out(path_, std::ios::trunc);
        if (!out) return false;
        out << j.dump(2);
        deletedIds_.clear();
        return static_cast<bool>(out);
#endif
    }

#ifdef HAVE_SQLITE
    bool openDb() {
        if (db_) return true;
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(path_).parent_path(), ec);
        int rc = sqlite3_open(path_.c_str(), &db_);
        if (rc != SQLITE_OK) {
            if (db_) {
                sqlite3_close(db_);
                db_ = nullptr;
            }
            return false;
        }
        execSql("PRAGMA journal_mode=WAL;");
        execSql("PRAGMA synchronous=NORMAL;");
        execSql("CREATE TABLE IF NOT EXISTS concept_matrix ("
                "id INTEGER PRIMARY KEY,"
                "unit_json TEXT NOT NULL,"
                "count INTEGER NOT NULL,"
                "residual REAL NOT NULL,"
                "timestamp_ms INTEGER NOT NULL"
                ");");
        execSql("CREATE INDEX IF NOT EXISTS idx_concept_matrix_ts ON concept_matrix(timestamp_ms);");
        return true;
    }
#endif

    void closeDb() {
#ifdef HAVE_SQLITE
        if (db_) {
            sqlite3_close(db_);
            db_ = nullptr;
        }
#endif
    }

    bool execSql(const std::string &sql) {
#ifdef HAVE_SQLITE
        if (!db_) return false;
        char *err = nullptr;
        int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err);
        if (rc != SQLITE_OK) {
            if (err) sqlite3_free(err);
            return false;
        }
        return true;
#else
        (void)sql;
        return true;
#endif
    }

    mutable std::mutex mu_;
    std::unordered_map<int64_t, Entry> entries_;
    std::unordered_map<size_t, std::unique_ptr<LSHIndex>> lshCache_;
    std::vector<int64_t> deletedIds_;
    int64_t nextId_ = 1;
    size_t maxEntries_ = 4096;
    float addThreshold_ = 0.5f;
    float learnThreshold_ = 0.2f;
#ifdef HAVE_SQLITE
    std::string path_ = "runtime_store/concept_matrix.db";
    sqlite3 *db_ = nullptr;
#else
    std::string path_ = "runtime_store/concept_matrix.json";
#endif
    bool loaded_ = false;
    size_t additions_ = 0;
    size_t updates_ = 0;
    size_t inferenceSaves_ = 0;
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
    recordRequestedTargetDim(targetDim);
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
        auto &imageModel = videoEncoder(dim);
        unit.semanticVector = imageModel.encode(packet.payload, width, height, packet.mimeType);
        if (width > 0) unit.metadata["width"] = std::to_string(width);
        if (height > 0) unit.metadata["height"] = std::to_string(height);
        unit.metadata["conceptEncoder"] = imageVariantFromMetadata(packet.metadata);
        unit.metadata["multimodalBackend"] = imageModel.status().value("backend", std::string("video-encoder"));
        if (unit.semanticVector.empty()) {
            unit.confidence = 0.0f;
            std::string err = imageModel.status().value("error", std::string());
            unit.metadata["videoEncoderError"] = err.empty() ? "video encoder returned empty concept vector" : err;
        }
    } else if (packet.modality == MixedModalModality::Audio) {
        const int sampleRate = sampleRateFromMetadata(packet.metadata);
        auto &speechModel = audioEncoder(dim);
        unit.semanticVector = speechModel.encode(packet.payload, sampleRate, packet.mimeType);
        unit.metadata["conceptEncoder"] = speechVariantFromMetadata(packet.metadata);
        unit.metadata["multimodalBackend"] = speechModel.status().value("backend", std::string("audio-encoder"));
        if (unit.semanticVector.empty()) {
            unit.confidence = 0.0f;
            std::string err = speechModel.status().value("error", std::string());
            unit.metadata["multimodalEncodeError"] = err.empty() ? "audio encoder returned empty concept vector" : err;
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
    recordRequestedTargetDim(targetDim);
    if (audio.modality != MixedModalModality::Audio || audio.payload.empty() || transcript.empty()) return false;
    const size_t dim = conceptDimension(targetDim);
    const int sampleRate = sampleRateFromMetadata(audio.metadata);
    auto &model = audioEncoder(dim);
    auto encoded = model.encode(audio.payload, sampleRate, audio.mimeType);
    if (encoded.empty()) return false;

    const auto correlationId = phoenix::multimodal::generateSemanticId(transcript);
    phoenix::multimodal::SemanticUnit audioUnit;
    audioUnit.id = audio.id.empty() ? correlationId + "-audio" : audio.id;
    audioUnit.modality = phoenix::multimodal::Modality::Audio;
    audioUnit.semanticVector = std::move(encoded);
    audioUnit.confidence = 1.0f;
    audioUnit.timestampMs = audio.timestampMs;
    audioUnit.metadata["source"] = audio.source;
    audioUnit.metadata["mimeType"] = audio.mimeType;
    audioUnit.metadata["conceptEncoder"] = speechVariantFromMetadata(audio.metadata);
    audioUnit.metadata["multimodalBackend"] = model.status().value("backend", std::string("audio-encoder"));
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
    recordRequestedTargetDim(targetDim);
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
    auto &model = videoEncoder(dim);
    auto encoded = model.encode(image.payload, width, height, image.mimeType);
    if (encoded.empty()) return false;

    const auto correlationId = phoenix::multimodal::generateSemanticId(caption);
    phoenix::multimodal::SemanticUnit imageUnit;
    imageUnit.id = image.id.empty() ? (caption.empty() ? phoenix::multimodal::generateSemanticId()
                                                       : correlationId + "-image")
                                    : image.id;
    imageUnit.modality = phoenix::multimodal::Modality::Image;
    imageUnit.semanticVector = std::move(encoded);
    imageUnit.confidence = 1.0f;
    imageUnit.timestampMs = image.timestampMs;
    imageUnit.metadata["source"] = image.source;
    imageUnit.metadata["mimeType"] = image.mimeType;
    if (width > 0) imageUnit.metadata["width"] = std::to_string(width);
    if (height > 0) imageUnit.metadata["height"] = std::to_string(height);
    imageUnit.metadata["conceptEncoder"] = imageVariantFromMetadata(image.metadata);
    imageUnit.metadata["multimodalBackend"] = model.status().value("backend", std::string("video-encoder"));
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
        auto &model = videoDecoder(unit.semanticVector.size());
        packet.payload = model.decode(unit.semanticVector, "image/png");
        packet.mimeType = "image/png";
        packet.metadata["multimodalBackend"] = model.status().value("backend", std::string("video-decoder"));
        if (packet.payload.empty()) {
            std::string err = model.status().value("error", std::string());
            packet.metadata["videoDecodeError"] = err.empty() ? "video decoder returned empty payload" : err;
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
        auto &model = audioDecoder(unit.semanticVector.size());
        packet.payload = model.decode(unit.semanticVector, "audio/pcm", lengthHint);
        packet.mimeType = "audio/pcm";
        packet.metadata["multimodalBackend"] = model.status().value("backend", std::string("audio-decoder"));
        if (packet.payload.empty()) {
            std::string err = model.status().value("error", std::string());
            packet.metadata["audioDecodeError"] = err.empty() ? "audio decoder returned empty payload" : err;
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
 * @brief Report the state of the multimodal video/audio world models,
 *        the text encoder, and the shared concept matrix.
 */
nlohmann::json MixedModalConceptBridge::status() {
    nlohmann::json j;
    try {
        auto imgStatus = videoEncoder(gLastRequestedTargetDim).status();
        if (!imgStatus.contains("dimension") && imgStatus.contains("targetDim")) {
            imgStatus["dimension"] = imgStatus["targetDim"];
        }
        j["videoWorldModel"] = std::move(imgStatus);
    } catch (...) {
        j["videoWorldModel"] = {{"error", "video world model not available"}};
    }
    try {
        auto spkStatus = audioEncoder(gLastRequestedTargetDim).status();
        if (!spkStatus.contains("dimension") && spkStatus.contains("targetDim")) {
            spkStatus["dimension"] = spkStatus["targetDim"];
        }
        j["audioWorldModel"] = std::move(spkStatus);
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
