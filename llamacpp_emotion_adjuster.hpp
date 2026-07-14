/* llamacpp_emotion_adjuster.hpp - Emotion-based weight adjustment for llama.cpp */

#pragma once

#include "emotion_system.hpp"
#include <memory>
#include <string>
#include <map>
#include <vector>
#include <mutex>

namespace phoenix {
namespace emotion {

/* LlamaCpp-specific weight adjustment implementation */
class LlamaCppEmotionWeightAdjuster : public LLMWeightAdjuster {
public:
    struct Config {
        float maxAdjustment{0.1f};        /* Maximum weight adjustment magnitude */
        float adjustmentDecay{0.98f};      /* Decay of adjustments over time */
        bool adjustAttention{true};        /* Adjust attention weights */
        bool adjustFeedForward{true};      /* Adjust feed-forward weights */
        bool adjustOutput{true};           /* Adjust output layer weights */
        std::vector<int> targetLayers;     /* Specific layers to adjust (empty = all) */
    };

    explicit LlamaCppEmotionWeightAdjuster(const Config& config);
    ~LlamaCppEmotionWeightAdjuster() override;

    /* Apply emotion-based weight adjustments */
    bool applyEmotionWeights(const EmotionTensor& emotion,
                             const std::string& layerPattern) override;

    /* Reset weights to baseline */
    bool resetWeights() override;

    /* Get current adjustment magnitude */
    float getAdjustmentMagnitude() const override;

    /* Set baseline weights (should be called after model loading) */
    bool setBaselineWeights(const std::string& layerPattern);

    /* Get current weight adjustments for monitoring */
    std::map<std::string, float> getCurrentAdjustments() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/* Runtime weight cache for emotion-based adjustments */
class EmotionWeightCache {
public:
    struct WeightEntry {
        std::vector<float> baseline;      /* Baseline weight values */
        std::vector<float> current;       /* Current adjusted values */
        std::string layerName;            /* Layer name */
        int64_t lastUpdateTurn;           /* Last update turn number */
    };

    explicit EmotionWeightCache(size_t maxEntries = 1000);

    /* Store baseline weights */
    bool storeBaseline(const std::string& layerName, const std::vector<float>& weights);

    /* Get baseline weights */
    std::optional<std::vector<float>> getBaseline(const std::string& layerName) const;

    /* Apply emotion-based adjustment to weights */
    std::vector<float> applyAdjustment(const std::string& layerName,
                                       const EmotionTensor& emotion,
                                       float maxAdjustment);

    /* Reset all weights to baseline */
    void resetAll();

    /* Get current adjustment statistics */
    size_t getEntryCount() const;
    float getTotalAdjustmentMagnitude() const;

private:
    std::map<std::string, WeightEntry> cache_; /* Weight cache */
    mutable std::mutex mutex_;                 /* Mutex for thread safety */
    size_t maxEntries_;                        /* Maximum cache entries */
};

/* Emotion-to-weight mapping strategies */
enum class EmotionMappingStrategy {
    LINEAR,        /* Direct linear mapping */
    EXPONENTIAL,   /* Exponential mapping for stronger effects */
    SIGMOID,       /* Sigmoid mapping for bounded effects */
    CUSTOM         /* Custom mapping function */
};

class EmotionWeightMapper {
public:
    explicit EmotionWeightMapper(EmotionMappingStrategy strategy = EmotionMappingStrategy::LINEAR);

    /* Map emotion tensor to weight adjustment factor */
    float mapToWeightAdjustment(const EmotionTensor& emotion,
                                const std::string& weightType) const;

    /* Set custom mapping function */
    using MappingFunction = std::function<float(const EmotionTensor&, const std::string&)>;
    void setCustomMapping(MappingFunction func);

    /* Set mapping strategy */
    void setStrategy(EmotionMappingStrategy strategy);

private:
    EmotionMappingStrategy strategy_; /* Current mapping strategy */
    MappingFunction customFunc_;      /* Custom mapping function */

    float linearMap(const EmotionTensor& emotion, const std::string& weightType) const;
    float exponentialMap(const EmotionTensor& emotion, const std::string& weightType) const;
    float sigmoidMap(const EmotionTensor& emotion, const std::string& weightType) const;
};

/* Hook interface for integrating with llamacpp */
class LlamaCppEmotionHooks {
public:
    /* Called before each forward pass */
    static void onForwardPassBegin(const std::string& sessionId);

    /* Called after each forward pass */
    static void onForwardPassEnd(const std::string& sessionId);

    /* Called when model is loaded */
    static void onModelLoaded();

    /* Called when model is unloaded */
    static void onModelUnloaded();

    /* Set the global emotion weight adjuster */
    static void setWeightAdjuster(std::shared_ptr<LlamaCppEmotionWeightAdjuster> adjuster);

    /* Get the global emotion weight adjuster */
    static std::shared_ptr<LlamaCppEmotionWeightAdjuster> getWeightAdjuster();

    /* Enable/disable emotion-based adjustments */
    static void setEnabled(bool enabled);
    static bool isEnabled();

private:
    static std::shared_ptr<LlamaCppEmotionWeightAdjuster> globalAdjuster_; /* Global adjuster instance */
    static bool enabled_;       /* Emotion adjustments enabled */
    static std::mutex mutex_;   /* Mutex for thread safety */
};

} // namespace emotion
} // namespace phoenix
