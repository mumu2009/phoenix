/* emotion_system.cpp - Emotion system implementation
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

#include "emotion_system.hpp"
#include <sqlite3.h>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace phoenix {
namespace emotion {

/* SQLite implementation details */
struct SQLiteEmotionStorage::Impl {
    sqlite3* db{nullptr};
    std::string dbPath;
    
    bool open() {
        int rc = sqlite3_open(dbPath.c_str(), &db);
        if (rc != SQLITE_OK) {
            std::cerr << "[SQLiteEmotionStorage] Failed to open database: " << sqlite3_errmsg(db) << std::endl;
            return false;
        }
        
        // Create table if not exists
        const char* createTableSQL = 
            "CREATE TABLE IF NOT EXISTS emotion_states ("
            "session_id TEXT PRIMARY KEY,"
            "valence REAL,"
            "arousal REAL,"
            "dominance REAL,"
            "trust REAL,"
            "joy REAL,"
            "fear REAL,"
            "anger REAL,"
            "surprise REAL,"
            "baseline_valence REAL,"
            "baseline_arousal REAL,"
            "baseline_dominance REAL,"
            "baseline_trust REAL,"
            "baseline_joy REAL,"
            "baseline_fear REAL,"
            "baseline_anger REAL,"
            "baseline_surprise REAL,"
            "timestamp INTEGER,"
            "turn_number INTEGER"
            ");";
        
        char* errMsg = nullptr;
        rc = sqlite3_exec(db, createTableSQL, nullptr, nullptr, &errMsg);
        if (rc != SQLITE_OK) {
            std::cerr << "[SQLiteEmotionStorage] Failed to create table: " << errMsg << std::endl;
            sqlite3_free(errMsg);
            return false;
        }
        
        return true;
    }
    
    void close() {
        if (db) {
            sqlite3_close(db);
            db = nullptr;
        }
    }
};

SQLiteEmotionStorage::SQLiteEmotionStorage(const std::string& dbPath)
    : impl_(std::make_unique<Impl>()) {
    impl_->dbPath = dbPath;
    impl_->open();
}

SQLiteEmotionStorage::~SQLiteEmotionStorage() {
    impl_->close();
}

bool SQLiteEmotionStorage::saveEmotionState(const std::string& sessionId, const EmotionState& state) {
    if (!impl_->db) {
        if (!impl_->open()) return false;
    }
    
    const char* insertSQL = 
        "INSERT OR REPLACE INTO emotion_states ("
        "session_id, valence, arousal, dominance, trust, joy, fear, anger, surprise,"
        "baseline_valence, baseline_arousal, baseline_dominance, baseline_trust, baseline_joy, baseline_fear, baseline_anger, baseline_surprise,"
        "timestamp, turn_number"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db, insertSQL, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "[SQLiteEmotionStorage] Failed to prepare statement: " << sqlite3_errmsg(impl_->db) << std::endl;
        return false;
    }
    
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        state.timestamp.time_since_epoch()).count();
    
    sqlite3_bind_text(stmt, 1, sessionId.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_double(stmt, 2, state.current.valence);
    sqlite3_bind_double(stmt, 3, state.current.arousal);
    sqlite3_bind_double(stmt, 4, state.current.dominance);
    sqlite3_bind_double(stmt, 5, state.current.trust);
    sqlite3_bind_double(stmt, 6, state.current.joy);
    sqlite3_bind_double(stmt, 7, state.current.fear);
    sqlite3_bind_double(stmt, 8, state.current.anger);
    sqlite3_bind_double(stmt, 9, state.current.surprise);
    sqlite3_bind_double(stmt, 10, state.baseline.valence);
    sqlite3_bind_double(stmt, 11, state.baseline.arousal);
    sqlite3_bind_double(stmt, 12, state.baseline.dominance);
    sqlite3_bind_double(stmt, 13, state.baseline.trust);
    sqlite3_bind_double(stmt, 14, state.baseline.joy);
    sqlite3_bind_double(stmt, 15, state.baseline.fear);
    sqlite3_bind_double(stmt, 16, state.baseline.anger);
    sqlite3_bind_double(stmt, 17, state.baseline.surprise);
    sqlite3_bind_int64(stmt, 18, timestamp);
    sqlite3_bind_int64(stmt, 19, state.turnNumber);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        std::cerr << "[SQLiteEmotionStorage] Failed to execute statement: " << sqlite3_errmsg(impl_->db) << std::endl;
        return false;
    }
    
    return true;
}

std::optional<EmotionState> SQLiteEmotionStorage::loadEmotionState(const std::string& sessionId) {
    if (!impl_->db) {
        if (!impl_->open()) return std::nullopt;
    }
    
    const char* selectSQL = 
        "SELECT valence, arousal, dominance, trust, joy, fear, anger, surprise,"
        "baseline_valence, baseline_arousal, baseline_dominance, baseline_trust, baseline_joy, baseline_fear, baseline_anger, baseline_surprise,"
        "timestamp, turn_number FROM emotion_states WHERE session_id = ?;";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db, selectSQL, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "[SQLiteEmotionStorage] Failed to prepare statement: " << sqlite3_errmsg(impl_->db) << std::endl;
        return std::nullopt;
    }
    
    sqlite3_bind_text(stmt, 1, sessionId.c_str(), -1, SQLITE_STATIC);
    
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW) {
        sqlite3_finalize(stmt);
        return std::nullopt;
    }
    
    EmotionState state;
    state.current.valence = sqlite3_column_double(stmt, 0);
    state.current.arousal = sqlite3_column_double(stmt, 1);
    state.current.dominance = sqlite3_column_double(stmt, 2);
    state.current.trust = sqlite3_column_double(stmt, 3);
    state.current.joy = sqlite3_column_double(stmt, 4);
    state.current.fear = sqlite3_column_double(stmt, 5);
    state.current.anger = sqlite3_column_double(stmt, 6);
    state.current.surprise = sqlite3_column_double(stmt, 7);
    
    state.baseline.valence = sqlite3_column_double(stmt, 8);
    state.baseline.arousal = sqlite3_column_double(stmt, 9);
    state.baseline.dominance = sqlite3_column_double(stmt, 10);
    state.baseline.trust = sqlite3_column_double(stmt, 11);
    state.baseline.joy = sqlite3_column_double(stmt, 12);
    state.baseline.fear = sqlite3_column_double(stmt, 13);
    state.baseline.anger = sqlite3_column_double(stmt, 14);
    state.baseline.surprise = sqlite3_column_double(stmt, 15);
    
    int64_t timestamp = sqlite3_column_int64(stmt, 16);
    state.timestamp = std::chrono::system_clock::time_point(std::chrono::milliseconds(timestamp));
    state.turnNumber = sqlite3_column_int64(stmt, 17);
    state.sessionId = sessionId;
    
    sqlite3_finalize(stmt);
    return state;
}

bool SQLiteEmotionStorage::deleteEmotionState(const std::string& sessionId) {
    if (!impl_->db) {
        if (!impl_->open()) return false;
    }
    
    const char* deleteSQL = "DELETE FROM emotion_states WHERE session_id = ?;";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db, deleteSQL, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "[SQLiteEmotionStorage] Failed to prepare statement: " << sqlite3_errmsg(impl_->db) << std::endl;
        return false;
    }
    
    sqlite3_bind_text(stmt, 1, sessionId.c_str(), -1, SQLITE_STATIC);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    return rc == SQLITE_DONE;
}

std::vector<std::string> SQLiteEmotionStorage::listSessions() {
    std::vector<std::string> sessions;
    
    if (!impl_->db) {
        if (!impl_->open()) return sessions;
    }
    
    const char* selectSQL = "SELECT session_id FROM emotion_states;";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db, selectSQL, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "[SQLiteEmotionStorage] Failed to prepare statement: " << sqlite3_errmsg(impl_->db) << std::endl;
        return sessions;
    }
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char* sessionId = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        if (sessionId) {
            sessions.push_back(std::string(sessionId));
        }
    }
    
    sqlite3_finalize(stmt);
    return sessions;
}

// Simple rule-based emotion analyzer
class RuleBasedEmotionAnalyzer : public EmotionAnalyzer {
public:
    EmotionTensor analyzeText(const std::string& text) override {
        EmotionTensor emotion;
        
        // Simple keyword-based emotion detection
        std::string lowerText = text;
        std::transform(lowerText.begin(), lowerText.end(), lowerText.begin(), ::tolower);
        
        // Positive/joy keywords
        if (lowerText.find("happy") != std::string::npos ||
            lowerText.find("great") != std::string::npos ||
            lowerText.find("love") != std::string::npos ||
            lowerText.find("wonderful") != std::string::npos) {
            emotion.joy += 0.5f;
            emotion.valence += 0.4f;
        }
        
        // Negative/sad keywords
        if (lowerText.find("sad") != std::string::npos ||
            lowerText.find("bad") != std::string::npos ||
            lowerText.find("hate") != std::string::npos ||
            lowerText.find("terrible") != std::string::npos) {
            emotion.joy -= 0.5f;
            emotion.valence -= 0.4f;
        }
        
        // Fear keywords
        if (lowerText.find("afraid") != std::string::npos ||
            lowerText.find("scared") != std::string::npos ||
            lowerText.find("fear") != std::string::npos ||
            lowerText.find("worried") != std::string::npos) {
            emotion.fear += 0.6f;
            emotion.arousal += 0.3f;
        }
        
        // Anger keywords
        if (lowerText.find("angry") != std::string::npos ||
            lowerText.find("furious") != std::string::npos ||
            lowerText.find("mad") != std::string::npos ||
            lowerText.find("hate") != std::string::npos) {
            emotion.anger += 0.7f;
            emotion.arousal += 0.5f;
        }
        
        // Surprise keywords
        if (lowerText.find("surprise") != std::string::npos ||
            lowerText.find("shock") != std::string::npos ||
            lowerText.find("amazing") != std::string::npos ||
            lowerText.find("unexpected") != std::string::npos) {
            emotion.surprise += 0.6f;
            emotion.arousal += 0.4f;
        }
        
        // Trust keywords
        if (lowerText.find("trust") != std::string::npos ||
            lowerText.find("believe") != std::string::npos ||
            lowerText.find("confident") != std::string::npos) {
            emotion.trust += 0.5f;
            emotion.valence += 0.2f;
        }
        
        // Clamp values to [-1, 1]
        auto clamp = [](float& v) { v = std::max(-1.0f, std::min(1.0f, v)); };
        clamp(emotion.valence);
        clamp(emotion.arousal);
        clamp(emotion.dominance);
        clamp(emotion.trust);
        clamp(emotion.joy);
        clamp(emotion.fear);
        clamp(emotion.anger);
        clamp(emotion.surprise);
        
        return emotion;
    }
    
    EmotionTensor analyzeAudio(const std::vector<float>& audioFeatures) override {
        // Placeholder for audio-based emotion analysis
        // In a real implementation, this would use a neural network
        EmotionTensor emotion;
        if (audioFeatures.size() >= 8) {
            emotion.valence = audioFeatures[0];
            emotion.arousal = audioFeatures[1];
            emotion.dominance = audioFeatures[2];
            emotion.trust = audioFeatures[3];
            emotion.joy = audioFeatures[4];
            emotion.fear = audioFeatures[5];
            emotion.anger = audioFeatures[6];
            emotion.surprise = audioFeatures[7];
        }
        return emotion;
    }
    
    EmotionTensor combineEmotions(const std::vector<EmotionTensor>& emotions,
                                  const std::vector<float>& weights) override {
        if (emotions.empty()) return EmotionTensor();
        
        EmotionTensor combined;
        float totalWeight = 0.0f;
        
        for (size_t i = 0; i < emotions.size() && i < weights.size(); ++i) {
            float w = weights[i];
            combined.valence += emotions[i].valence * w;
            combined.arousal += emotions[i].arousal * w;
            combined.dominance += emotions[i].dominance * w;
            combined.trust += emotions[i].trust * w;
            combined.joy += emotions[i].joy * w;
            combined.fear += emotions[i].fear * w;
            combined.anger += emotions[i].anger * w;
            combined.surprise += emotions[i].surprise * w;
            totalWeight += w;
        }
        
        if (totalWeight > 0.0f) {
            combined.valence /= totalWeight;
            combined.arousal /= totalWeight;
            combined.dominance /= totalWeight;
            combined.trust /= totalWeight;
            combined.joy /= totalWeight;
            combined.fear /= totalWeight;
            combined.anger /= totalWeight;
            combined.surprise /= totalWeight;
        }
        
        return combined;
    }
};

// Emotion system implementation
struct EmotionSystem::Impl {
    Config config;
    std::unique_ptr<EmotionStorage> storage;
    std::shared_ptr<EmotionAnalyzer> analyzer;
    std::map<std::string, EmotionState> sessionStates;
    std::map<std::string, std::vector<EmotionState>> sessionHistory;
    EmotionPluginManager pluginManager;
    std::mutex mutex;
    
    Impl(const Config& cfg) : config(cfg) {
        if (cfg.storageBackend == "sqlite") {
            storage = std::make_unique<SQLiteEmotionStorage>(cfg.storagePath);
        }
        analyzer = std::make_shared<RuleBasedEmotionAnalyzer>();
    }
};

EmotionSystem::EmotionSystem(const Config& config)
    : impl_(std::make_unique<Impl>(config)) {}

EmotionSystem::~EmotionSystem() = default;

EmotionState EmotionSystem::processMessage(const std::string& sessionId,
                                           const std::string& text,
                                           int64_t turnNumber) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    
    if (!impl_->config.enabled) {
        EmotionState state;
        state.sessionId = sessionId;
        state.turnNumber = turnNumber;
        return state;
    }
    
    // Load existing state or create new one
    EmotionState state;
    auto existingState = impl_->storage->loadEmotionState(sessionId);
    if (existingState) {
        state = *existingState;
    } else {
        state.sessionId = sessionId;
        state.baseline = EmotionTensor(); // Neutral baseline
    }
    
    EmotionState oldState = state;
    state.turnNumber = turnNumber;
    state.timestamp = std::chrono::system_clock::now();
    
    // Analyze emotion from text
    EmotionTensor newEmotion = impl_->analyzer->analyzeText(text);
    
    // Apply plugin modifications
    for (const auto& pair : impl_->pluginManager.listPlugins()) {
        auto plugin = impl_->pluginManager.getPlugin(pair);
        if (plugin) {
            newEmotion = plugin->modifyEmotion(sessionId, newEmotion);
        }
    }
    
    // Apply emotion decay
    newEmotion = EmotionTensor::lerp(state.current, newEmotion, 1.0f - impl_->config.emotionDecayRate);
    
    state.current = newEmotion;
    
    // Update history
    impl_->sessionHistory[sessionId].push_back(state);
    if (impl_->sessionHistory[sessionId].size() > static_cast<size_t>(impl_->config.historyLength)) {
        impl_->sessionHistory[sessionId].erase(impl_->sessionHistory[sessionId].begin());
    }
    
    // Save to storage
    impl_->storage->saveEmotionState(sessionId, state);
    
    // Notify plugins
    impl_->pluginManager.notifyEmotionUpdate(sessionId, oldState, state);
    
    impl_->sessionStates[sessionId] = state;
    return state;
}

std::optional<EmotionState> EmotionSystem::getEmotionState(const std::string& sessionId) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    
    auto it = impl_->sessionStates.find(sessionId);
    if (it != impl_->sessionStates.end()) {
        return it->second;
    }
    
    return impl_->storage->loadEmotionState(sessionId);
}

bool EmotionSystem::applyToLLM(const std::string& sessionId, LLMWeightAdjuster* adjuster) {
    if (!impl_->config.enabled || !impl_->config.enableRuntimeFineTuning || !adjuster) {
        return false;
    }
    
    auto state = getEmotionState(sessionId);
    if (!state) {
        return false;
    }
    
    // Apply emotion-based weight adjustments
    // This is a simplified implementation - real implementation would be more sophisticated
    float influence = impl_->config.emotionInfluence;
    
    // Apply adjustments based on emotion dimensions
    // In a real implementation, this would adjust specific layer weights
    return adjuster->applyEmotionWeights(state->current, "all");
}

std::string EmotionSystem::getEmotionContext(const std::string& sessionId) {
    if (!impl_->config.enabled) {
        return "";
    }
    
    auto state = getEmotionState(sessionId);
    if (!state) {
        return "";
    }
    
    std::ostringstream context;
    context << "[Emotional Context]\n";
    context << "Valence: " << std::fixed << std::setprecision(2) << state->current.valence << "\n";
    context << "Arousal: " << std::fixed << std::setprecision(2) << state->current.arousal << "\n";
    context << "Dominance: " << std::fixed << std::setprecision(2) << state->current.dominance << "\n";
    context << "Trust: " << std::fixed << std::setprecision(2) << state->current.trust << "\n";
    context << "Joy: " << std::fixed << std::setprecision(2) << state->current.joy << "\n";
    context << "Fear: " << std::fixed << std::setprecision(2) << state->current.fear << "\n";
    context << "Anger: " << std::fixed << std::setprecision(2) << state->current.anger << "\n";
    context << "Surprise: " << std::fixed << std::setprecision(2) << state->current.surprise << "\n";
    context << "Stability: " << std::fixed << std::setprecision(2) << state->stability() << "\n";
    context << "Intensity: " << std::fixed << std::setprecision(2) << state->intensity() << "\n";
    
    // Add plugin context
    std::string pluginContext = impl_->pluginManager.collectPluginContext(sessionId, *state);
    if (!pluginContext.empty()) {
        context << "\n[Plugin Context]\n" << pluginContext;
    }
    
    return context.str();
}

void EmotionSystem::setEmotionAnalyzer(std::shared_ptr<EmotionAnalyzer> analyzer) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->analyzer = analyzer;
}

void EmotionSystem::setEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->config.enabled = enabled;
}

bool EmotionSystem::isEnabled() const {
    return impl_->config.enabled;
}

// Plugin manager implementation
void EmotionPluginManager::registerPlugin(std::shared_ptr<EmotionPlugin> plugin) {
    std::lock_guard<std::mutex> lock(mutex_);
    plugins_[plugin->name()] = plugin;
}

void EmotionPluginManager::unregisterPlugin(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    plugins_.erase(name);
}

std::shared_ptr<EmotionPlugin> EmotionPluginManager::getPlugin(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = plugins_.find(name);
    if (it != plugins_.end()) {
        return it->second;
    }
    return nullptr;
}

std::vector<std::string> EmotionPluginManager::listPlugins() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> names;
    for (const auto& pair : plugins_) {
        names.push_back(pair.first);
    }
    return names;
}

void EmotionPluginManager::notifyEmotionUpdate(const std::string& sessionId,
                                               const EmotionState& oldState,
                                               const EmotionState& newState) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& pair : plugins_) {
        pair.second->onEmotionUpdate(sessionId, oldState, newState);
    }
}

std::string EmotionPluginManager::collectPluginContext(const std::string& sessionId,
                                                       const EmotionState& state) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream context;
    for (auto& pair : plugins_) {
        std::string pluginContext = pair.second->getAdditionalContext(sessionId, state);
        if (!pluginContext.empty()) {
            context << "[" << pair.first << "]\n" << pluginContext << "\n";
        }
    }
    return context.str();
}

} // namespace emotion
} // namespace phoenix
