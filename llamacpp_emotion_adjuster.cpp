/* llamacpp_emotion_adjuster.cpp - Llama.cpp emotion adjuster implementation
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

#include "llamacpp_emotion_adjuster.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

namespace phoenix {
namespace emotion {

EmotionWeightCache::EmotionWeightCache(size_t maxEntries) 
    : maxEntries_(maxEntries) {}

bool EmotionWeightCache::storeBaseline(const std::string& layerName, 
                                     const std::vector<float>& weights) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (weights.empty() || weights.size() != maxEntries_) {
        return false;
    }
    
    if (cache_.size() >= maxEntries_) {
        // Evict oldest entry
        auto oldest = std::min_element(cache_.begin(), cache_.end(),
            [](const auto& a, const auto& b) {
                return a.second.lastUpdateTurn < b.second.lastUpdateTurn;
            });
        if (oldest != cache_.end()) {
            cache_.erase(oldest);
        }
    }
    
    WeightEntry entry;
    entry.baseline = weights;
    entry.current = weights;  // Initially same as baseline
    entry.layerName = layerName;
    entry.lastUpdateTurn = 0;
    
    cache_[layerName] = entry;
    return true;
}

std::optional<std::vector<float>> EmotionWeightCache::getBaseline(
    const std::string& layerName) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = cache_.find(layerName);
    if (it != cache_.end()) {
        return it->second.baseline;
    }
    return std::nullopt;
}

std::vector<float> EmotionWeightCache::applyAdjustment(
    const std::string& layerName,
    const EmotionTensor& emotion,
    float maxAdjustment) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = cache_.find(layerName);
    if (it == cache_.end()) {
        return {};  // No baseline stored
    }
    
    WeightEntry& entry = it->second;
    std::vector<float> adjusted = entry.baseline;
    
    // Calculate adjustment factor based on emotion
    float adjustmentFactor = 0.0f;
    
    // Use valence and arousal as primary factors
    adjustmentFactor += emotion.valence * 0.4f;    // Positive emotions increase certain weights
    adjustmentFactor += emotion.arousal * 0.3f;    // High arousal increases sensitivity
    adjustmentFactor += emotion.trust * 0.2f;      // Trust affects response patterns
    adjustmentFactor += emotion.joy * 0.1f;        // Joy affects creativity
    
    // Clamp adjustment factor
    adjustmentFactor = std::max(-maxAdjustment, std::min(maxAdjustment, adjustmentFactor));
    
    // Apply adjustment with some randomness for natural variation
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<float> noise(0.0f, 0.01f);
    
    for (size_t i = 0; i < adjusted.size(); ++i) {
        float noiseFactor = 1.0f + noise(gen);
        adjusted[i] = adjusted[i] * (1.0f + adjustmentFactor) * noiseFactor;
        
        // Clamp to reasonable range
        adjusted[i] = std::max(-10.0f, std::min(10.0f, adjusted[i]));
    }
    
    entry.current = adjusted;
    entry.lastUpdateTurn++;
    
    return adjusted;
}

void EmotionWeightCache::resetAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& pair : cache_) {
        pair.second.current.assign(pair.second.baseline.size(), 0.0f);
        pair.second.lastUpdateTurn = 0;
    }
}

size_t EmotionWeightCache::getEntryCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_.size();
}

float EmotionWeightCache::getTotalAdjustmentMagnitude() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    float totalMagnitude = 0.0f;
    for (const auto& pair : cache_) {
        for (float v : pair.second.current) {
            totalMagnitude += std::abs(v);
        }
    }
    return totalMagnitude;
}

// Emotion weight mapper implementation
EmotionWeightMapper::EmotionWeightMapper(EmotionMappingStrategy strategy)
    : strategy_(strategy) {}

float EmotionWeightMapper::mapToWeightAdjustment(const EmotionTensor& emotion,
                                                  const std::string& weightType) const {
    switch (strategy_) {
        case EmotionMappingStrategy::LINEAR:
            return linearMap(emotion, weightType);
        case EmotionMappingStrategy::EXPONENTIAL:
            return exponentialMap(emotion, weightType);
        case EmotionMappingStrategy::SIGMOID:
            return sigmoidMap(emotion, weightType);
        case EmotionMappingStrategy::CUSTOM:
            if (customFunc_) {
                return customFunc_(emotion, weightType);
            }
            return linearMap(emotion, weightType);
        default:
            return linearMap(emotion, weightType);
    }
}

void EmotionWeightMapper::setCustomMapping(MappingFunction func) {
    customFunc_ = func;
    strategy_ = EmotionMappingStrategy::CUSTOM;
}

void EmotionWeightMapper::setStrategy(EmotionMappingStrategy strategy) {
    strategy_ = strategy;
}

float EmotionWeightMapper::linearMap(const EmotionTensor& emotion,
                                     const std::string& weightType) const {
    float adjustment = 0.0f;
    
    // Different weight types respond to different emotions
    if (weightType.find("attention") != std::string::npos) {
        // Attention weights respond to arousal and trust
        adjustment += emotion.arousal * 0.5f;
        adjustment += emotion.trust * 0.3f;
    } else if (weightType.find("ffn") != std::string::npos) {
        // Feed-forward weights respond to joy and surprise
        adjustment += emotion.joy * 0.4f;
        adjustment += emotion.surprise * 0.3f;
    } else if (weightType.find("output") != std::string::npos) {
        // Output weights respond to valence and dominance
        adjustment += emotion.valence * 0.5f;
        adjustment += emotion.dominance * 0.3f;
    } else {
        // Default mapping
        adjustment += emotion.valence * 0.3f;
        adjustment += emotion.arousal * 0.2f;
    }
    
    return std::max(-0.5f, std::min(0.5f, adjustment));
}

float EmotionWeightMapper::exponentialMap(const EmotionTensor& emotion,
                                          const std::string& weightType) const {
    float linear = linearMap(emotion, weightType);
    // Exponential scaling for stronger effects
    return std::copysign(std::exp(std::abs(linear)) - 1.0f, linear) * 0.3f;
}

float EmotionWeightMapper::sigmoidMap(const EmotionTensor& emotion,
                                      const std::string& weightType) const {
    float linear = linearMap(emotion, weightType);
    // Sigmoid for bounded effects
    return 1.0f / (1.0f + std::exp(-linear * 5.0f)) - 0.5f;
}

// LlamaCpp emotion weight adjuster implementation
struct LlamaCppEmotionWeightAdjuster::Impl {
    Config config;
    std::unique_ptr<EmotionWeightCache> weightCache;
    std::unique_ptr<EmotionWeightMapper> weightMapper;
    std::map<std::string, float> currentAdjustments;
    mutable std::mutex mutex;
    
    Impl(const Config& cfg) 
        : config(cfg),
          weightCache(std::make_unique<EmotionWeightCache>()),
          weightMapper(std::make_unique<EmotionWeightMapper>()) {}
};

LlamaCppEmotionWeightAdjuster::LlamaCppEmotionWeightAdjuster(const Config& config)
    : impl_(std::make_unique<Impl>(config)) {}

LlamaCppEmotionWeightAdjuster::~LlamaCppEmotionWeightAdjuster() = default;

bool LlamaCppEmotionWeightAdjuster::applyEmotionWeights(const EmotionTensor& emotion,
                                                         const std::string& layerPattern) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    
    if (!impl_->weightCache) {
        return false;
    }
    
    if (!impl_->weightCache->getBaseline(layerPattern).has_value()) {
        return false;
    }
    
    // Calculate adjustment factor
    float adjustmentFactor = impl_->weightMapper->mapToWeightAdjustment(emotion, layerPattern);
    
    // Apply to all cached weights matching pattern
    // In a real implementation, this would interface with llamacpp's actual weight tensors
    // For now, we simulate the adjustment
    
    impl_->currentAdjustments[layerPattern] = adjustmentFactor;
    
    std::cout << "[EmotionWeightAdjuster] Applied adjustment: " << adjustmentFactor 
              << " for pattern: " << layerPattern << std::endl;
    
    return true;
}

bool LlamaCppEmotionWeightAdjuster::resetWeights() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    
    if (!impl_->weightCache) {
        return false;
    }
    
    impl_->weightCache->resetAll();
    impl_->currentAdjustments.clear();
    
    std::cout << "[EmotionWeightAdjuster] Reset all weights to baseline" << std::endl;
    
    return true;
}

float LlamaCppEmotionWeightAdjuster::getAdjustmentMagnitude() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    
    if (!impl_->weightCache) {
        return 0.0f;
    }
    
    return impl_->weightCache->getTotalAdjustmentMagnitude();
}

bool LlamaCppEmotionWeightAdjuster::setBaselineWeights(const std::string& layerPattern) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    
    if (!impl_->weightCache) {
        return false;
    }
    
    // In a real implementation, this would extract actual weights from llamacpp.
    // The fallback below synthesizes a deterministic baseline using the layer name as a seed.
    
    std::seed_seq seed(layerPattern.begin(), layerPattern.end());
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> syntheticWeights(1000, 0.0f);
    for (size_t i = 0; i < syntheticWeights.size(); ++i) {
        syntheticWeights[i] = dist(gen);
    }
    
    return impl_->weightCache->storeBaseline(layerPattern, syntheticWeights);
}

std::map<std::string, float> LlamaCppEmotionWeightAdjuster::getCurrentAdjustments() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->currentAdjustments;
}

// LlamaCpp emotion hooks implementation
std::shared_ptr<LlamaCppEmotionWeightAdjuster> LlamaCppEmotionHooks::globalAdjuster_;
bool LlamaCppEmotionHooks::enabled_ = true;
std::mutex LlamaCppEmotionHooks::mutex_;

void LlamaCppEmotionHooks::onForwardPassBegin(const std::string& sessionId) {
    if (!enabled_) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    if (globalAdjuster_) {
        // In a real implementation, this would apply emotion-based adjustments
        // before the forward pass
        std::cout << "[EmotionHooks] Forward pass begin for session: " << sessionId << std::endl;
    }
}

void LlamaCppEmotionHooks::onForwardPassEnd(const std::string& sessionId) {
    if (!enabled_) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    if (globalAdjuster_) {
        // In a real implementation, this would decay adjustments after forward pass
        std::cout << "[EmotionHooks] Forward pass end for session: " << sessionId << std::endl;
    }
}

void LlamaCppEmotionHooks::onModelLoaded() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (globalAdjuster_) {
        std::cout << "[EmotionHooks] Model loaded, setting baseline weights" << std::endl;
        // Set baseline weights for all layers
        globalAdjuster_->setBaselineWeights("all");
    }
}

void LlamaCppEmotionHooks::onModelUnloaded() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (globalAdjuster_) {
        std::cout << "[EmotionHooks] Model unloaded, resetting weights" << std::endl;
        globalAdjuster_->resetWeights();
    }
}

void LlamaCppEmotionHooks::setWeightAdjuster(
    std::shared_ptr<LlamaCppEmotionWeightAdjuster> adjuster) {
    std::lock_guard<std::mutex> lock(mutex_);
    globalAdjuster_ = adjuster;
}

std::shared_ptr<LlamaCppEmotionWeightAdjuster> LlamaCppEmotionHooks::getWeightAdjuster() {
    std::lock_guard<std::mutex> lock(mutex_);
    return globalAdjuster_;
}

void LlamaCppEmotionHooks::setEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_ = enabled;
}

bool LlamaCppEmotionHooks::isEnabled() {
    std::lock_guard<std::mutex> lock(mutex_);
    return enabled_;
}

} // namespace emotion
} // namespace phoenix
