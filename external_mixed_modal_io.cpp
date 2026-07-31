/* external_mixed_modal_io.cpp - Implementation for external mixed-modal I/O
   Copyright (C) 2026 079 Project */

#include "external_mixed_modal_io.hpp"
#include "jpea_v2_image_world_model.hpp"
#include "jpea_v2_speech_world_model.hpp"
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

JpeaV2ImageWorldModel &imageWorldModel(const std::string &variant, int targetDim) {
    static std::mutex mu;
    static std::unordered_map<std::string, std::unique_ptr<JpeaV2ImageWorldModel>> cache;
    std::lock_guard<std::mutex> lock(mu);
    std::string key = variant + ":" + std::to_string(targetDim);
    auto it = cache.find(key);
    if (it != cache.end()) return *it->second;
    auto model = createJpeaV2ImageWorldModel(variant, targetDim);
    auto &ref = *model;
    cache.emplace(key, std::move(model));
    return ref;
}

std::string imageVariantFromMetadata(const nlohmann::json &metadata) {
    if (metadata.contains("jpeaVariant") && metadata["jpeaVariant"].is_string()) {
        return metadata["jpeaVariant"].get<std::string>();
    }
    if (metadata.contains("worldModel") && metadata["worldModel"].is_string()) {
        return metadata["worldModel"].get<std::string>();
    }
    return "ijepa_vith14_1k";
}

JpeaV2SpeechWorldModel &speechWorldModel(const std::string &variant, int targetDim) {
    static std::mutex mu;
    static std::unordered_map<std::string, std::unique_ptr<JpeaV2SpeechWorldModel>> cache;
    std::lock_guard<std::mutex> lock(mu);
    std::string key = variant + ":" + std::to_string(targetDim);
    auto it = cache.find(key);
    if (it != cache.end()) return *it->second;
    auto model = createJpeaV2SpeechWorldModel(variant, targetDim);
    auto &ref = *model;
    cache.emplace(key, std::move(model));
    return ref;
}

std::string speechVariantFromMetadata(const nlohmann::json &metadata) {
    if (metadata.contains("jpeaSpeechVariant") && metadata["jpeaSpeechVariant"].is_string()) {
        return metadata["jpeaSpeechVariant"].get<std::string>();
    }
    if (metadata.contains("worldModel") && metadata["worldModel"].is_string()) {
        return metadata["worldModel"].get<std::string>();
    }
    return "jpea_v2_speech_16k";
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

struct SpeechConceptModel {
    size_t dimension{0};
    size_t samples{0};
    std::vector<float> meanAlignment;
    bool loaded{false};

    void load() {
        if (loaded) return;
        loaded = true;
        std::ifstream input("runtime_store/speech_concept_model.json");
        if (!input) return;
        try {
            nlohmann::json state;
            input >> state;
            dimension = state.value("dimension", 0u);
            samples = state.value("samples", 0u);
            meanAlignment = state.value("meanAlignment", std::vector<float>{});
            if (dimension == 0 || meanAlignment.size() != dimension) {
                dimension = 0;
                samples = 0;
                meanAlignment.clear();
            }
        } catch (const nlohmann::json::exception &) {
            dimension = 0;
            samples = 0;
            meanAlignment.clear();
        }
    }

    bool save() const {
        std::error_code error;
        std::filesystem::create_directories("runtime_store", error);
        std::ofstream output("runtime_store/speech_concept_model.json", std::ios::trunc);
        if (!output) return false;
        output << nlohmann::json{{"dimension", dimension}, {"samples", samples}, {"meanAlignment", meanAlignment}}.dump(2);
        return static_cast<bool>(output);
    }
};

std::mutex gSpeechModelMutex;
SpeechConceptModel gSpeechModel;

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
 *   3. Image/Video payloads are dispatched to the image world model.  If it cannot
 *      produce an embedding (e.g. no BPU/decoder or invalid input) a deterministic
 *      media-concept fallback is used.
 *   4. Audio payloads are dispatched to the speech world model.  If it cannot
 *      produce an embedding, a deterministic media-concept fallback is used.  When
 *      a persistent speech concept model is available and the embedding is valid,
 *      it is aligned and normalised.
 *   5. Structured/sensor payloads fall back to a generic hashed concept.
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
        auto variant = imageVariantFromMetadata(packet.metadata);
        auto encoded = imageWorldModel(variant, static_cast<int>(dim))
                           .encode(packet.payload, width, height, packet.mimeType);
        if (encoded.empty()) {
            encoded = mediaConcept(packet.payload, dim, 0x494D4147U);
            unit.metadata["jpeaBackend"] = "fallback";
        }
        unit.semanticVector = std::move(encoded);
        unit.metadata["conceptEncoder"] = "jpea-v2-image-world-model";
        unit.metadata["jpeaVariant"] = variant;
    } else if (packet.modality == MixedModalModality::Audio) {
        auto variant = speechVariantFromMetadata(packet.metadata);
        const int sampleRate = sampleRateFromMetadata(packet.metadata);
        auto encoded = speechWorldModel(variant, static_cast<int>(dim))
                           .encode(packet.payload, sampleRate, packet.mimeType);
        if (encoded.empty()) {
            encoded = mediaConcept(packet.payload, dim, 0x41554449U);
            unit.metadata["jpeaSpeechBackend"] = "fallback";
        }
        {
            std::lock_guard<std::mutex> lock(gSpeechModelMutex);
            gSpeechModel.load();
            if (gSpeechModel.dimension == dim && gSpeechModel.meanAlignment.size() == dim &&
                encoded.size() == dim) {
                for (size_t i = 0; i < dim; ++i) encoded[i] += gSpeechModel.meanAlignment[i];
                encoded = normalizeConcept(std::move(encoded));
            }
        }
        unit.semanticVector = std::move(encoded);
        unit.metadata["conceptEncoder"] = "jpea-v2-speech-world-model";
        unit.metadata["jpeaSpeechVariant"] = variant;
        if (!unit.metadata.contains("jpeaSpeechBackend"))
            unit.metadata["jpeaSpeechBackend"] = "jpea-v2";
        unit.metadata["speechModelSamples"] = std::to_string(gSpeechModel.samples);
    } else {
        unit.semanticVector = mediaConcept(packet.payload, dim, 0x53545255U);
        unit.metadata["conceptEncoder"] = "structured-concept-adapter";
    }

    return unit;
}

/**
 * @brief Persistently align an audio concept vector with its transcript.
 *
 * Computes the per-element offset between the acoustic concept vector and the
 * text-encoder vector of the transcript, then updates a running mean alignment
 * stored in runtime_store/speech_concept_model.json.  This model is loaded on
 * demand by encode() and is not part of the mutable runtime session state.
 */
bool MixedModalConceptBridge::pretrainSpeech(const MixedModalPacket &audio,
                                             const std::string &transcript,
                                             size_t targetDim) {
    if (audio.modality != MixedModalModality::Audio || audio.payload.empty() || transcript.empty()) return false;
    const size_t dim = conceptDimension(targetDim);
    const auto variant = speechVariantFromMetadata(audio.metadata);
    const int sampleRate = sampleRateFromMetadata(audio.metadata);
    auto &model = speechWorldModel(variant, static_cast<int>(dim));
    const auto acoustic = model.encode(audio.payload, sampleRate, audio.mimeType);
    const auto linguistic = transformerTextEncoderConcept(transcript, dim, kDefaultTextEncoderCheckpoint);
    if (acoustic.size() != dim || linguistic.size() != dim) return false;
    model.contrastiveAdapt(audio.payload, sampleRate, audio.mimeType, linguistic, 0.1f);
    std::lock_guard<std::mutex> lock(gSpeechModelMutex);
    gSpeechModel.load();
    if (gSpeechModel.dimension != dim) {
        gSpeechModel.dimension = dim;
        gSpeechModel.samples = 0;
        gSpeechModel.meanAlignment.assign(dim, 0.0f);
    }
    const float previous = static_cast<float>(gSpeechModel.samples);
    for (size_t i = 0; i < dim; ++i) {
        const float alignment = linguistic[i] - acoustic[i];
        gSpeechModel.meanAlignment[i] = (gSpeechModel.meanAlignment[i] * previous + alignment) / (previous + 1.0f);
    }
    ++gSpeechModel.samples;
    return gSpeechModel.save();
}

/**
 * @brief Adapt an abstract SemanticUnit back into an external MixedModalPacket.
 *
 * For text the original content bytes are emitted directly.  For all other
 * modalities a JSON concept payload is produced and marked with
 * requiresModalityDecoder so the downstream modality-specific renderer can
 * materialise the entity.
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
        packet.mimeType = "image/png";
        auto variant = "ijepa_vith14_1k";
        packet.payload = imageWorldModel(variant, static_cast<int>(unit.semanticVector.size()))
                             .decode(unit.semanticVector, packet.mimeType);
        // If no decoder is configured, emit a 1x1 PNG placeholder so the
        // downstream pipeline still receives a valid image packet.
        if (packet.payload.empty()) {
            static const std::vector<uint8_t> kPlaceholder1x1Png{
                0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00,
                0x00, 0x0d, 0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01,
                0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1f,
                0x15, 0xc4, 0x89, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x44, 0x41,
                0x54, 0x78, 0xda, 0x63, 0xfc, 0xcf, 0xc0, 0x50, 0x0f, 0x00,
                0x04, 0x85, 0x01, 0x80, 0x84, 0xa3, 0x92, 0x23, 0x00, 0x00,
                0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82};
            packet.payload = kPlaceholder1x1Png;
            packet.metadata["imageDecodeFallback"] = true;
        }
        packet.metadata["sourceModality"] = phoenix::multimodal::modalityToString(unit.modality);
        packet.metadata["conceptVector"] = unit.semanticVector;
    } else if (target == MixedModalModality::Audio) {
        packet.mimeType = "audio/pcm";
        auto variant = "jpea_v2_speech_16k";
        size_t lengthHint = 0;
        auto it = unit.metadata.find("lengthHint");
        if (it != unit.metadata.end() && !it->second.empty()) {
            try {
                lengthHint = static_cast<size_t>(std::stoull(it->second));
            } catch (...) {
                lengthHint = 0;
            }
        }
        packet.payload = speechWorldModel(variant, static_cast<int>(unit.semanticVector.size()))
                             .decode(unit.semanticVector, packet.mimeType, lengthHint);
        // If the speech model cannot synthesize a waveform, return a JSON
        // concept payload as the portable fallback.
        if (packet.payload.empty()) {
            packet.mimeType = "application/json";
            nlohmann::json conceptPayload{{"semanticVector", unit.semanticVector},
                                          {"sourceModality", phoenix::multimodal::modalityToString(unit.modality)},
                                          {"lengthHint", lengthHint}};
            const auto serialized = conceptPayload.dump();
            packet.payload.assign(serialized.begin(), serialized.end());
            packet.metadata["requiresModalityDecoder"] = true;
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
 * @brief Report the state of the persistent speech concept model.
 */
nlohmann::json MixedModalConceptBridge::status() {
    nlohmann::json j;
    {
        std::lock_guard<std::mutex> lock(gSpeechModelMutex);
        gSpeechModel.load();
        j["speechWorldModel"] = {{"persistent", true}, {"dimension", gSpeechModel.dimension}, {"samples", gSpeechModel.samples}};
    }
    try {
        j["imageWorldModel"] = imageWorldModel("ijepa_vith14_1k", 64).status();
    } catch (...) {
        j["imageWorldModel"] = {{"error", "image world model not available"}};
    }
    j["textEncoder"] = textEncoderStatus();
    try {
        j["jpeaSpeechWorldModel"] = speechWorldModel("jpea_v2_speech_16k", 64).status();
    } catch (...) {
        j["jpeaSpeechWorldModel"] = {{"error", "speech world model not available"}};
    }
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
