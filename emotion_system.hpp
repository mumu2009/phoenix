/* emotion_system.hpp - Emotion modeling and analysis system for 079 Project
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

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <mutex>
#include <functional>
#include <array>
#include <unordered_map>
#include <optional>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <nlohmann/json.hpp>

namespace phoenix {
namespace emotion {

/* Emotion dimensions based on psychological models (e.g., PAD model).
   Represents emotional state as an 8-dimensional tensor. */
struct EmotionTensor {
    float valence;      /* Pleasure/displeasure (-1.0 to 1.0) */
    float arousal;      /* Activation/deactivation (-1.0 to 1.0) */
    float dominance;    /* Control/submission (-1.0 to 1.0) */
    float trust;        /* Trust/suspicion (-1.0 to 1.0) */
    float joy;          /* Joy/sadness (-1.0 to 1.0) */
    float fear;         /* Fear/courage (-1.0 to 1.0) */
    float anger;        /* Anger/calm (-1.0 to 1.0) */
    float surprise;     /* Surprise/boredom (-1.0 to 1.0) */
    
    EmotionTensor() : valence(0.0f), arousal(0.0f), dominance(0.0f),
                      trust(0.0f), joy(0.0f), fear(0.0f), anger(0.0f), surprise(0.0f) {}

    EmotionTensor(float v, float a, float d, float tr, float j, float f, float an, float s)
        : valence(v), arousal(a), dominance(d), trust(tr), joy(j), fear(f), anger(an), surprise(s) {}

    /* Convert to JSON for storage */
    nlohmann::json toJson() const {
        return {
            {"valence", valence},
            {"arousal", arousal},
            {"dominance", dominance},
            {"trust", trust},
            {"joy", joy},
            {"fear", fear},
            {"anger", anger},
            {"surprise", surprise}
        };
    }

    /* Load from JSON */
    static EmotionTensor fromJson(const nlohmann::json& j) {
        EmotionTensor t;
        if (!j.is_object()) return t;
        auto getFloat = [&j](const std::string& key, float fallback) -> float {
            auto it = j.find(key);
            if (it == j.end()) return fallback;
            if (it->is_number()) {
                try { return it->get<float>(); } catch (...) { return fallback; }
            }
            if (it->is_string()) {
                try { return std::stof(it->get<std::string>()); } catch (...) { return fallback; }
            }
            return fallback;
        };
        t.valence   = getFloat("valence", t.valence);
        t.arousal   = getFloat("arousal", t.arousal);
        t.dominance = getFloat("dominance", t.dominance);
        t.trust     = getFloat("trust", t.trust);
        t.joy       = getFloat("joy", t.joy);
        t.fear      = getFloat("fear", t.fear);
        t.anger     = getFloat("anger", t.anger);
        t.surprise  = getFloat("surprise", t.surprise);
        return t;
    }

    /* Linear interpolation between two emotion tensors */
    static EmotionTensor lerp(const EmotionTensor& a, const EmotionTensor& b, float t) {
        EmotionTensor result;
        result.valence = a.valence + t * (b.valence - a.valence);
        result.arousal = a.arousal + t * (b.arousal - a.arousal);
        result.dominance = a.dominance + t * (b.dominance - a.dominance);
        result.trust = a.trust + t * (b.trust - a.trust);
        result.joy = a.joy + t * (b.joy - a.joy);
        result.fear = a.fear + t * (b.fear - a.fear);
        result.anger = a.anger + t * (b.anger - a.anger);
        result.surprise = a.surprise + t * (b.surprise - a.surprise);
        return result;
    }

    /* Calculate Euclidean distance between emotion tensors */
    float distance(const EmotionTensor& other) const {
        float dv = valence - other.valence;
        float da = arousal - other.arousal;
        float dd = dominance - other.dominance;
        float dt = trust - other.trust;
        float dj = joy - other.joy;
        float df = fear - other.fear;
        float dang = anger - other.anger;
        float ds = surprise - other.surprise;
        return std::sqrt(dv*dv + da*da + dd*dd + dt*dt + dj*dj + df*df + dang*dang + ds*ds);
    }

    /* Human-readable directive derived from the tensor (used in prompt text). */
    std::string modulationHint() const {
        std::ostringstream oss;
        if (arousal >= 0.3f) oss << "high-arousal; ";
        else if (arousal <= -0.3f) oss << "low-arousal; ";
        if (fear >= 0.3f) oss << "risk-averse; ";
        else if (fear <= -0.3f) oss << "risk-tolerant; ";
        if (trust >= 0.3f) oss << "trust-high; ";
        else if (trust <= -0.3f) oss << "trust-low verify; ";
        if (joy >= 0.3f) oss << "optimistic; ";
        else if (joy <= -0.3f) oss << "cautious; ";
        if (dominance >= 0.3f) oss << "directive; ";
        else if (dominance <= -0.3f) oss << "deferential; ";
        if (valence >= 0.3f) oss << "approach; ";
        else if (valence <= -0.3f) oss << "avoid; ";
        if (surprise >= 0.3f) oss << "novelty-attention; ";
        if (oss.str().empty()) oss << "neutral; ";
        std::string s = oss.str();
        s.pop_back(); s.pop_back();
        return s;
    }

    /* Map the emotion tensor to LLM inference options.
       Returns a JSON object with temperature, top_p, presence_penalty and
       frequency_penalty adjustments.  These are *additive* to the caller's
       base values and can be clamped by the caller. */
    nlohmann::json inferenceOptions(float baseTemp = 0.7f,
                                    float baseTopP = 0.9f) const {
        float temperature = baseTemp + 0.35f * arousal - 0.15f * fear;
        float topP = baseTopP + 0.08f * surprise - 0.05f * dominance;
        float presence = -0.2f * trust + 0.15f * joy;
        float frequency = 0.15f * anger - 0.1f * valence;
        temperature = std::max(0.1f, std::min(1.5f, temperature));
        topP = std::max(0.1f, std::min(1.0f, topP));
        presence = std::max(-2.0f, std::min(2.0f, presence));
        frequency = std::max(-2.0f, std::min(2.0f, frequency));
        return {
            {"temperature", temperature},
            {"top_p", topP},
            {"presence_penalty", presence},
            {"frequency_penalty", frequency}
        };
    }
};

/* Emotion state with temporal information */
struct EmotionState {
    EmotionTensor current;                           /* Current emotion state */
    EmotionTensor baseline;                          /* Baseline emotion for this session */
    std::chrono::system_clock::time_point timestamp; /* State timestamp */
    std::string sessionId;                          /* Session identifier */
    int64_t turnNumber;                              /* Turn number in conversation */

    EmotionState() : turnNumber(0) {
        timestamp = std::chrono::system_clock::now();
    }

    /* Calculate emotional stability (how much current deviates from baseline) */
    float stability() const;

    /* Get emotional intensity (magnitude of emotion vector) */
    float intensity() const {
        return std::sqrt(
            current.valence * current.valence +
            current.arousal * current.arousal +
            current.dominance * current.dominance
        );
    }
};

/* Storage backend interface for emotion data */
class EmotionStorage {
public:
    virtual ~EmotionStorage() = default;

    virtual bool saveEmotionState(const std::string& sessionId, const EmotionState& state) = 0; /* Save emotion state */
    virtual std::optional<EmotionState> loadEmotionState(const std::string& sessionId) = 0; /* Load emotion state */
    virtual bool deleteEmotionState(const std::string& sessionId) = 0; /* Delete emotion state */
    virtual std::vector<std::string> listSessions() = 0; /* List all sessions */
};

/* SQLite-based storage implementation */
class SQLiteEmotionStorage : public EmotionStorage {
public:
    explicit SQLiteEmotionStorage(const std::string& dbPath);
    ~SQLiteEmotionStorage() override;

    bool saveEmotionState(const std::string& sessionId, const EmotionState& state) override;
    std::optional<EmotionState> loadEmotionState(const std::string& sessionId) override;
    bool deleteEmotionState(const std::string& sessionId) override;
    std::vector<std::string> listSessions() override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/* LLM weight adjustment interface for runtime fine-tuning */
class LLMWeightAdjuster {
public:
    virtual ~LLMWeightAdjuster() = default;

    /* Apply emotion-based weight adjustments (runtime only, no disk modification) */
    virtual bool applyEmotionWeights(const EmotionTensor& emotion,
                                     const std::string& layerPattern) = 0;

    /* Reset weights to baseline */
    virtual bool resetWeights() = 0;

    /* Get current adjustment magnitude */
    virtual float getAdjustmentMagnitude() const = 0;
};

/* Emotion analysis interface */
class EmotionAnalyzer {
public:
    virtual ~EmotionAnalyzer() = default;

    /* Analyze text and extract emotion */
    virtual EmotionTensor analyzeText(const std::string& text) = 0;

    /* Analyze audio features and extract emotion */
    virtual EmotionTensor analyzeAudio(const std::vector<float>& audioFeatures) = 0;

    /* Combine multiple emotion sources */
    virtual EmotionTensor combineEmotions(const std::vector<EmotionTensor>& emotions,
                                         const std::vector<float>& weights) = 0;
};

class EmotionVocabWeightTable; // forward declaration

/* Stage configuration for the vocabulary-level emotion weight table. */
struct EmotionVocabWeightTableStageConfig {
    float influence{1.0f};
    int halfLifeTurns{1};
    float decay{0.95f};
};

/* Configuration for the vocabulary-level emotion weight table.
   Declared outside the class so EmotionSystem::Config can reference it
   before EmotionVocabWeightTable is defined. */
struct EmotionVocabWeightTableConfig {
    bool enabled{true};
    float learningRate{0.05f};
    float maxBias{2.0f};
    float minBias{-2.0f};
    float decay{0.95f};
    float momentum{0.9f};
    float tokenBoostExponent{1.2f};
    float minTokenScore{0.05f};
    std::array<EmotionVocabWeightTableStageConfig, 4> stageConfigs;
    std::unordered_map<std::string, std::vector<std::string>> seedLexicon;
    std::string vocabTablePath;
    std::string vocabCachePath;
    bool applyToPrompt{true};
    bool applyLogitBias{true};
};

/* Main emotion system class */
class EmotionSystem {
public:
    struct Config {
        bool enabled{true};                       /* Enable emotion system */
        std::string storageBackend{"sqlite"};     /* Storage backend type */
        std::string storagePath{"emotion_states.db"}; /* Storage path */
        float emotionDecayRate{0.95f};            /* Emotion decay per turn */
        float emotionInfluence{0.3f};            /* How much emotion affects LLM */
        bool enableRuntimeFineTuning{true};       /* Enable runtime fine-tuning */
        int historyLength{10};                    /* How many emotion states to keep */
        EmotionVocabWeightTableConfig vocabTableConfig; /* Vocabulary-level weight table */
    };

    explicit EmotionSystem(const Config& config);
    ~EmotionSystem();

    /* Process a user message and update emotion state */
    EmotionState processMessage(const std::string& sessionId,
                               const std::string& text,
                               int64_t turnNumber);

    /* Get current emotion state for a session */
    std::optional<EmotionState> getEmotionState(const std::string& sessionId);

    /* Apply emotion-based LLM weight adjustments */
    bool applyToLLM(const std::string& sessionId, LLMWeightAdjuster* adjuster);

    /* Get emotion context for prompt injection */
    std::string getEmotionContext(const std::string& sessionId);

    /* Set custom emotion analyzer */
    void setEmotionAnalyzer(std::shared_ptr<EmotionAnalyzer> analyzer);

    /* Enable/disable the system */
    void setEnabled(bool enabled);
    bool isEnabled() const;

    /* Vocabulary-level weight table accessors */
    std::shared_ptr<EmotionVocabWeightTable> vocabTable();
    std::shared_ptr<const EmotionVocabWeightTable> vocabTable() const;

    /* Observe tokens for the vocabulary weight table (uses the last emotion state). */
    void observeVocab(const std::string& sessionId,
                    const std::vector<std::string>& tokens, int turnNumber);

    /* Get logit_bias JSON and prompt modulation for a session/input. */
    nlohmann::json getVocabLogitBias(const std::string& sessionId,
                                     const std::vector<std::string>& inputTokens,
                                     int turnNumber) const;
    std::string getVocabPromptModulation(const std::string& sessionId,
                                         const std::vector<std::string>& inputTokens,
                                         int turnNumber) const;

    /* Update the vocabulary table from a (prompt, reply, reward) triple. */
    void updateVocabFromResponse(const std::string& sessionId,
                                 const std::vector<std::string>& promptTokens,
                                 const std::vector<std::string>& replyTokens,
                                 float reward, int turnNumber);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/* Plugin interface for extending emotion system */
class EmotionPlugin {
public:
    virtual ~EmotionPlugin() = default;

    virtual std::string name() const = 0; /* Plugin name */
    virtual std::string version() const = 0; /* Plugin version */

    /* Called when emotion state is updated */
    virtual void onEmotionUpdate(const std::string& sessionId,
                                const EmotionState& oldState,
                                const EmotionState& newState) {}

    /* Can modify emotion before it's applied */
    virtual EmotionTensor modifyEmotion(const std::string& sessionId,
                                       const EmotionTensor& emotion) {
        return emotion;
    }

    /* Can provide additional context based on emotion */
    virtual std::string getAdditionalContext(const std::string& sessionId,
                                           const EmotionState& state) {
        return "";
    }
};

/* Vocabulary-level emotion weight table.
   Maintains per-session, per-stage sparse biases over tokens.
   Four stages correspond to different context ranges:
     Immediate: current utterance
     Short:     recent session history (few turns)
     Context:   session-level baseline/persona
     Long:      cross-session persistent profile
   The table can be applied to prompts (soft modulation) or exported as
   logit_bias maps for external llama-server backends. */
class EmotionVocabWeightTable {
public:
    enum class Stage { Immediate = 0, Short = 1, Context = 2, Long = 3, Count = 4 };

    struct TokenWeight {
        float bias{0.0f};
        float momentum{0.0f};
        int lastUpdateTurn{0};
    };

    using StageConfig = EmotionVocabWeightTableStageConfig;
    using Config = EmotionVocabWeightTableConfig;

    explicit EmotionVocabWeightTable(const Config& cfg = Config{});
    ~EmotionVocabWeightTable() = default;

    /* Observe a set of tokens under a given emotional signal. */
    void observe(Stage stage, const std::string& sessionId,
                 const std::vector<std::string>& tokens,
                 const EmotionTensor& emotion, int turnNumber);

    /* Compute per-token bias for the current input across all stages. */
    std::unordered_map<std::string, float> computeTokenBias(
        const std::string& sessionId,
        const std::vector<std::string>& inputTokens,
        int turnNumber) const;

    /* Learn from (prompt, reply, reward). Positive reward reinforces
       co-occurring tokens; negative reward suppresses them. */
    void updateFromResponse(const std::string& sessionId,
                            const std::vector<std::string>& promptTokens,
                            const std::vector<std::string>& replyTokens,
                            float reward, int turnNumber);

    /* Build a short prompt-modulation string from the strongest biases. */
    std::string buildPromptModulation(const std::string& sessionId,
                                      const std::vector<std::string>& inputTokens,
                                      int turnNumber) const;

    /* Export biases as a JSON object suitable for logit_bias payloads. */
    nlohmann::json getLogitBiasJson(const std::string& sessionId,
                                    const std::vector<std::string>& inputTokens,
                                    int turnNumber) const;

    nlohmann::json toJson() const;
    void fromJson(const nlohmann::json& j);

    bool load();
    bool save() const;

    bool enabled() const { return cfg_.enabled; }

private:
    Config cfg_;
    mutable std::mutex mu_;
    std::unordered_map<std::string, std::array<std::unordered_map<std::string, TokenWeight>,
        static_cast<size_t>(Stage::Count)>> sessionWeights_;
    std::unordered_map<std::string, TokenWeight> longTermWeights_;
    std::unordered_map<std::string, int> sessionLastTurn_;

    float decayFactor(int stageIdx, int deltaTurns) const;
    void applyDecay(TokenWeight& w, int stageIdx, int deltaTurns) const;
    float emotionToSignal(const EmotionTensor& e, const std::string& token) const;
    std::vector<std::string> expandWithEmotion(const std::vector<std::string>& tokens,
                                               const EmotionTensor& emotion) const;
    std::vector<std::pair<std::string, float>> rankBiases(
        const std::unordered_map<std::string, float>& biases, size_t topN) const;
};

/* Plugin manager */
class EmotionPluginManager {
public:
    void registerPlugin(std::shared_ptr<EmotionPlugin> plugin); /* Register a plugin */
    void unregisterPlugin(const std::string& name); /* Unregister a plugin */
    std::shared_ptr<EmotionPlugin> getPlugin(const std::string& name); /* Get plugin by name */
    std::vector<std::string> listPlugins() const; /* List all plugins */

    /* Notify all plugins of emotion update */
    void notifyEmotionUpdate(const std::string& sessionId,
                           const EmotionState& oldState,
                           const EmotionState& newState);

    /* Collect additional context from all plugins */
    std::string collectPluginContext(const std::string& sessionId,
                                    const EmotionState& state);

private:
    std::map<std::string, std::shared_ptr<EmotionPlugin>> plugins_; /* Registered plugins */
    mutable std::mutex mutex_; /* Mutex for thread safety */
};

} // namespace emotion
} // namespace phoenix
