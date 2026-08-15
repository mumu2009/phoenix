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
#include <filesystem>
#include <fstream>
#include <deque>

namespace phoenix {
namespace emotion {

namespace {
    /* Lightweight word tokenizer used for the emotion vocabulary table. */
    std::vector<std::string> tokenizeText(const std::string& text) {
        std::vector<std::string> tokens;
        std::string cur;
        for (char ch : text) {
            if (std::isalnum(static_cast<unsigned char>(ch)) || ch == '\'') {
                cur.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
            } else if (!cur.empty()) {
                tokens.push_back(cur);
                cur.clear();
            }
        }
        if (!cur.empty()) tokens.push_back(cur);
        return tokens;
    }
}

float EmotionState::stability() const {
    float d = current.distance(baseline);
    if (std::isnan(d) || std::isinf(d)) return 0.0f;
    float s = 1.0f - (d / 4.0f); /* Normalize to 0-1 */
    return std::clamp(s, 0.0f, 1.0f);
}

/* SQLite implementation details */
struct SQLiteEmotionStorage::Impl {
    sqlite3* db{nullptr};
    std::string dbPath;

    // RAII helper: opens the database in the constructor and closes it on
    // destruction. This avoids holding a file handle across operations.
    struct ScopedDb {
        Impl* impl{nullptr};
        explicit ScopedDb(Impl* i) : impl(i) {
            if (impl && !impl->open()) impl = nullptr;
        }
        ~ScopedDb() {
            if (impl) impl->close();
        }
        bool ok() const { return impl != nullptr; }
    };

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
    // Opening is deferred to each operation via ScopedDb so we do not hold a
    // file handle while the EmotionSystem is alive.
}

SQLiteEmotionStorage::~SQLiteEmotionStorage() {
    impl_->close();
}

bool SQLiteEmotionStorage::saveEmotionState(const std::string& sessionId, const EmotionState& state) {
    Impl::ScopedDb scoped(impl_.get());
    if (!scoped.ok()) return false;
    
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
    Impl::ScopedDb scoped(impl_.get());
    if (!scoped.ok()) return std::nullopt;
    
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
    Impl::ScopedDb scoped(impl_.get());
    if (!scoped.ok()) return false;
    
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
    
    Impl::ScopedDb scoped(impl_.get());
    if (!scoped.ok()) return sessions;
    
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

// VAD-lexicon based emotion analyzer.
//
// A compact affective lexicon maps lowercased words to PAD coordinates
// (valence, arousal, dominance) in [-1, 1], following the affective-norms
// tradition (Russell, 1980, "A circumplex model of affect"; Warriner et al.,
// 2013, "Norms of valence, arousal, and dominance for 13,915 English lemmas").
// The analyzer mean-pools the matched coordinates over ALL tokens (unmatched
// words contribute zero), which (a) bounds the signal by match density instead
// of the saturated keyword counting used previously, and (b) removes the
// double-counting caused by overlapping keyword categories.  The pooled
// (V, A, D) triple is converted to the 8-d tensor through padToTensor().
namespace {
struct VAD { float v; float a; float d; };

const std::unordered_map<std::string, VAD>& vadLexicon() {
    static const std::unordered_map<std::string, VAD> lex = {
        // joy / positive
        {"happy",     {0.90f,  0.70f,  0.50f}},
        {"glad",      {0.80f,  0.50f,  0.40f}},
        {"great",     {0.80f,  0.60f,  0.40f}},
        {"wonderful", {0.90f,  0.60f,  0.50f}},
        {"excellent", {0.90f,  0.60f,  0.60f}},
        {"love",      {0.90f,  0.80f,  0.30f}},
        {"joy",       {0.90f,  0.80f,  0.50f}},
        {"amazing",   {0.90f,  0.80f,  0.40f}},
        // sadness / negative
        {"sad",       {-0.80f, -0.40f, -0.50f}},
        {"bad",       {-0.70f,  0.20f, -0.30f}},
        {"terrible",  {-0.90f,  0.40f, -0.40f}},
        {"awful",     {-0.90f,  0.30f, -0.50f}},
        {"hate",      {-0.90f,  0.60f,  0.20f}},
        // fear
        {"afraid",    {-0.80f,  0.70f, -0.70f}},
        {"scared",    {-0.80f,  0.70f, -0.70f}},
        {"fear",      {-0.80f,  0.60f, -0.60f}},
        {"worried",   {-0.60f,  0.50f, -0.50f}},
        {"anxious",   {-0.60f,  0.60f, -0.50f}},
        // anger
        {"angry",     {-0.80f,  0.80f,  0.70f}},
        {"furious",   {-0.90f,  0.90f,  0.80f}},
        {"mad",       {-0.70f,  0.70f,  0.60f}},
        // surprise
        {"surprise",  { 0.20f,  0.80f,  0.00f}},
        {"shock",     {-0.40f,  0.90f, -0.60f}},
        {"unexpected",{ 0.00f,  0.70f, -0.20f}},
        // trust
        {"trust",     { 0.60f,  0.10f,  0.40f}},
        {"believe",   { 0.50f,  0.10f,  0.30f}},
        {"confident", { 0.70f,  0.40f,  0.80f}},
        {"reliable",  { 0.60f,  0.10f,  0.50f}},
    };
    return lex;
}
}  // namespace

class RuleBasedEmotionAnalyzer : public EmotionAnalyzer {
public:
    EmotionTensor analyzeText(const std::string& text) override {
        const auto toks = tokenizeText(text);
        if (toks.empty()) return EmotionTensor();
        const auto& lex = vadLexicon();
        float V = 0.0f, A = 0.0f, D = 0.0f;
        size_t matched = 0;
        for (const auto& t : toks) {
            auto it = lex.find(t);
            if (it == lex.end()) continue;
            V += it->second.v;
            A += it->second.a;
            D += it->second.d;
            ++matched;
        }
        if (matched == 0) return EmotionTensor();
        // Mean-pool over ALL tokens (unmatched -> 0) so intensity scales with
        // match density rather than being saturated by a single keyword.
        const float invN = 1.0f / static_cast<float>(toks.size());
        return padToTensor(V * invN, A * invN, D * invN);
    }

    EmotionTensor analyzeAudio(const std::vector<float>& audioFeatures) override {
        // Audio-feature based emotion analysis.
        // If the upstream audio tower provides >= 3 features, interpret the
        // first three as a (valence, arousal, dominance) triplet; a single
        // feature is treated as an arousal-only (energy/loudness) signal.  The
        // previous implementation mapped eight raw acoustic features directly
        // onto the 8-d affect space, conflating acoustic energy with the
        // appraisal dimensions.
        if (audioFeatures.size() >= 3) {
            return padToTensor(audioFeatures[0], audioFeatures[1], audioFeatures[2]);
        }
        if (audioFeatures.size() == 1) {
            return padToTensor(0.0f, audioFeatures[0], 0.0f);
        }
        return EmotionTensor();
    }

    EmotionTensor combineEmotions(const std::vector<EmotionTensor>& emotions,
                                  const std::vector<float>& weights) override {
        if (emotions.empty()) return EmotionTensor();
        // Confidence-weighted mean.  Non-positive weights are treated as zero
        // so a low-confidence source decays gracefully rather than being
        // averaged in with unit weight.
        const size_t n = std::min(emotions.size(), weights.size());
        EmotionTensor combined;
        float totalWeight = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            const float w = weights[i] > 0.0f ? weights[i] : 0.0f;
            if (w == 0.0f) continue;
            const auto& e = emotions[i];
            combined.valence += e.valence * w;
            combined.arousal += e.arousal * w;
            combined.dominance += e.dominance * w;
            combined.trust += e.trust * w;
            combined.joy += e.joy * w;
            combined.fear += e.fear * w;
            combined.anger += e.anger * w;
            combined.surprise += e.surprise * w;
            totalWeight += w;
        }
        if (totalWeight <= 0.0f) return EmotionTensor();
        const float inv = 1.0f / totalWeight;
        combined.valence *= inv;
        combined.arousal *= inv;
        combined.dominance *= inv;
        combined.trust *= inv;
        combined.joy *= inv;
        combined.fear *= inv;
        combined.anger *= inv;
        combined.surprise *= inv;
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
    std::unordered_map<std::string, std::deque<std::pair<int, std::string>>> sessionTexts;
    EmotionPluginManager pluginManager;
    std::shared_ptr<EmotionVocabWeightTable> vocabTable;
    std::mutex mutex;
    
    Impl(const Config& cfg) : config(cfg) {
        if (cfg.storageBackend == "sqlite") {
            storage = std::make_unique<SQLiteEmotionStorage>(cfg.storagePath);
        }
        analyzer = std::make_shared<RuleBasedEmotionAnalyzer>();
        EmotionVocabWeightTable::Config vcfg = cfg.vocabTableConfig;
        // Default stage configs if not supplied.
        if (vcfg.stageConfigs[0].halfLifeTurns <= 0) {
            vcfg.stageConfigs[0] = {1.0f, 1, 0.95f};
            vcfg.stageConfigs[1] = {0.7f, 5, 0.95f};
            vcfg.stageConfigs[2] = {0.4f, 20, 0.95f};
            vcfg.stageConfigs[3] = {0.2f, 200, 0.95f};
        }
        if (vcfg.seedLexicon.empty()) {
            vcfg.seedLexicon = {
                {"positive", {"happy", "great", "love", "wonderful", "glad", "excellent"}},
                {"negative", {"sad", "bad", "hate", "terrible", "angry", "awful"}},
                {"fear", {"afraid", "scared", "fear", "worried", "anxious"}},
                {"trust", {"trust", "believe", "confident", "reliable"}}
            };
        }
        vocabTable = std::make_shared<EmotionVocabWeightTable>(vcfg);
    }
};

EmotionSystem::EmotionSystem(const Config& config)
    : impl_(std::make_unique<Impl>(config)) {}

EmotionSystem::~EmotionSystem() {
    if (impl_ && impl_->vocabTable) {
        try {
            impl_->vocabTable->save();
        } catch (...) {
            // Persistence failures must not throw during destruction.
        }
    }
}

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
    
    // Apply emotion dynamics: a first-order homeostatic update.
    //   E = lerp(E, newEmotion, 1 - decayRate)      // stimulus coupling (EMA)
    //   E = lerp(E, baseline,    homeostasisRate)   // opponent-process pull
    // The second term is the opponent-process homeostatic pull toward the
    // session baseline (Solomon & Corbit, 1974, "An opponent-process theory of
    // motivation").  See doc/v7.0/algorithm.md section 15 for the derivation.
    newEmotion = EmotionTensor::lerp(state.current, newEmotion, 1.0f - impl_->config.emotionDecayRate);
    newEmotion = EmotionTensor::lerp(newEmotion, state.baseline, impl_->config.homeostasisRate);
    state.current = newEmotion;

    // Update vocabulary-level emotion weight table across stages.
    if (impl_->vocabTable) {
        auto toks = tokenizeText(text);
        int t = static_cast<int>(turnNumber);
        impl_->vocabTable->observe(EmotionVocabWeightTable::Stage::Immediate,
                                   sessionId, toks, newEmotion, t);
        auto& texts = impl_->sessionTexts[sessionId];
        texts.push_back({t, text});
        if (texts.size() > 30) texts.pop_front();

        std::string shortAgg;
        int shortCount = 0;
        for (auto it = texts.rbegin(); it != texts.rend() && shortCount < 5; ++it, ++shortCount) {
            shortAgg += it->second + " ";
        }
        if (!shortAgg.empty()) {
            impl_->vocabTable->observe(EmotionVocabWeightTable::Stage::Short,
                                       sessionId, tokenizeText(shortAgg), newEmotion, t);
        }
        std::string ctxAgg;
        for (const auto& rec : texts) ctxAgg += rec.second + " ";
        if (!ctxAgg.empty()) {
            impl_->vocabTable->observe(EmotionVocabWeightTable::Stage::Context,
                                       sessionId, tokenizeText(ctxAgg), newEmotion, t);
        }
    }

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

std::shared_ptr<EmotionVocabWeightTable> EmotionSystem::vocabTable() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->vocabTable;
}

std::shared_ptr<const EmotionVocabWeightTable> EmotionSystem::vocabTable() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->vocabTable;
}

void EmotionSystem::observeVocab(const std::string& sessionId,
                                 const std::vector<std::string>& tokens,
                                 int turnNumber) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->vocabTable) return;
    EmotionTensor emotion;
    auto it = impl_->sessionStates.find(sessionId);
    if (it != impl_->sessionStates.end()) emotion = it->second.current;
    impl_->vocabTable->observe(EmotionVocabWeightTable::Stage::Short,
                               sessionId, tokens, emotion, turnNumber);
}

nlohmann::json EmotionSystem::getVocabLogitBias(
    const std::string& sessionId,
    const std::vector<std::string>& inputTokens,
    int turnNumber) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->vocabTable) return {};
    return impl_->vocabTable->getLogitBiasJson(sessionId, inputTokens, turnNumber);
}

std::string EmotionSystem::getVocabPromptModulation(
    const std::string& sessionId,
    const std::vector<std::string>& inputTokens,
    int turnNumber) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->vocabTable) return "";
    return impl_->vocabTable->buildPromptModulation(sessionId, inputTokens, turnNumber);
}

void EmotionSystem::updateVocabFromResponse(
    const std::string& sessionId,
    const std::vector<std::string>& promptTokens,
    const std::vector<std::string>& replyTokens,
    float reward,
    int turnNumber) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->vocabTable) return;
    impl_->vocabTable->updateFromResponse(sessionId, promptTokens, replyTokens,
                                          reward, turnNumber);
}

// Plugin manager implementation
void EmotionPluginManager::registerPlugin(std::shared_ptr<EmotionPlugin> plugin) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!plugin)
        return;
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

// Vocabulary-level emotion weight table implementation

EmotionVocabWeightTable::EmotionVocabWeightTable(const Config& cfg) : cfg_(cfg) {
    if (cfg_.vocabTablePath.empty()) {
        cfg_.vocabTablePath = "./runtime_store/emotion_vocab_weight_table.json";
    }
    for (auto& sc : cfg_.stageConfigs) {
        if (sc.halfLifeTurns <= 0) sc.halfLifeTurns = 1;
        if (sc.decay <= 0.0f || sc.decay > 1.0f) sc.decay = 0.95f;
    }
    load();
}

float EmotionVocabWeightTable::decayFactor(int stageIdx, int deltaTurns) const {
    if (deltaTurns <= 0) return 1.0f;
    float halfLife = static_cast<float>(cfg_.stageConfigs[stageIdx].halfLifeTurns);
    return std::pow(cfg_.decay, static_cast<float>(deltaTurns) / std::max(1.0f, halfLife));
}

void EmotionVocabWeightTable::applyDecay(TokenWeight& w, int stageIdx, int deltaTurns) const {
    if (deltaTurns <= 0 || std::abs(w.bias) < 1e-6f) return;
    w.bias *= decayFactor(stageIdx, deltaTurns);
}

float EmotionVocabWeightTable::emotionToSignal(const EmotionTensor& e,
                                               const std::string& token) const {
    for (const auto& kv : cfg_.seedLexicon) {
        const auto& words = kv.second;
        if (std::find(words.begin(), words.end(), token) != words.end()) {
            const std::string& cat = kv.first;
            if (cat == "positive") return e.valence + e.joy;
            if (cat == "negative") return -(e.valence + e.joy);
            if (cat == "fear") return e.fear + e.arousal * 0.5f;
            if (cat == "trust") return e.trust;
            return 0.0f;
        }
    }
    // Default projection for any content word.
    return e.valence * 0.3f + e.joy * 0.3f - e.fear * 0.2f - e.anger * 0.2f;
}

std::vector<std::string> EmotionVocabWeightTable::expandWithEmotion(
    const std::vector<std::string>& tokens,
    const EmotionTensor& /*emotion*/) const {
    std::vector<std::string> out;
    out.reserve(tokens.size());
    for (const auto& t : tokens) {
        if (!t.empty()) out.push_back(t);
    }
    return out;
}

void EmotionVocabWeightTable::observe(Stage stage, const std::string& sessionId,
                                     const std::vector<std::string>& tokens,
                                     const EmotionTensor& emotion, int turnNumber) {
    if (!cfg_.enabled || tokens.empty()) return;
    std::lock_guard<std::mutex> lock(mu_);
    sessionLastTurn_[sessionId] = turnNumber;
    size_t sIdx = static_cast<size_t>(stage);
    auto& stageMap = sessionWeights_[sessionId][sIdx];
    auto expanded = expandWithEmotion(tokens, emotion);
    for (const auto& tok : expanded) {
        if (tok.empty() || tok.size() > 24) continue;
        auto& w = stageMap[tok];
        int delta = std::max(0, turnNumber - w.lastUpdateTurn);
        applyDecay(w, static_cast<int>(sIdx), delta);
        float signal = emotionToSignal(emotion, tok);
        float target = signal * cfg_.stageConfigs[sIdx].influence;
        float grad = target - w.bias;
        w.momentum = cfg_.momentum * w.momentum + (1.0f - cfg_.momentum) * grad;
        w.bias += cfg_.learningRate * (grad + w.momentum);
        w.bias = std::max(cfg_.minBias, std::min(cfg_.maxBias, w.bias));
        w.lastUpdateTurn = turnNumber;
        if (stage == Stage::Long) {
            auto& g = longTermWeights_[tok];
            int gd = std::max(0, turnNumber - g.lastUpdateTurn);
            applyDecay(g, static_cast<int>(sIdx), gd);
            g.bias = 0.95f * g.bias + 0.05f * w.bias;
            g.lastUpdateTurn = turnNumber;
        }
    }
}

std::unordered_map<std::string, float> EmotionVocabWeightTable::computeTokenBias(
    const std::string& sessionId,
    const std::vector<std::string>& inputTokens,
    int turnNumber) const {
    std::unordered_map<std::string, float> out;
    if (!cfg_.enabled) return out;
    std::lock_guard<std::mutex> lock(mu_);
    auto sit = sessionWeights_.find(sessionId);
    for (const auto& tok : inputTokens) {
        if (tok.empty()) continue;
        float total = 0.0f;
        for (size_t s = 0; s < static_cast<size_t>(Stage::Count); ++s) {
            float halfLife = static_cast<float>(std::max(1, cfg_.stageConfigs[s].halfLifeTurns));
            float stageInfluence = cfg_.stageConfigs[s].influence;
            if (sit != sessionWeights_.end()) {
                auto it = sit->second[s].find(tok);
                if (it != sit->second[s].end()) {
                    int delta = std::max(0, turnNumber - it->second.lastUpdateTurn);
                    float d = std::pow(cfg_.decay, static_cast<float>(delta) / halfLife);
                    total += stageInfluence * it->second.bias * d;
                }
            }
            if (static_cast<Stage>(s) == Stage::Long) {
                auto git = longTermWeights_.find(tok);
                if (git != longTermWeights_.end()) {
                    int delta = std::max(0, turnNumber - git->second.lastUpdateTurn);
                    float d = std::pow(cfg_.decay, static_cast<float>(delta) / halfLife);
                    total += stageInfluence * git->second.bias * d;
                }
            }
        }
        if (std::abs(total) >= cfg_.minTokenScore) {
            total = std::max(cfg_.minBias, std::min(cfg_.maxBias, total));
            out[tok] = total;
        }
    }
    return out;
}

void EmotionVocabWeightTable::updateFromResponse(
    const std::string& sessionId,
    const std::vector<std::string>& promptTokens,
    const std::vector<std::string>& replyTokens,
    float reward,
    int turnNumber) {
    if (!cfg_.enabled) return;
    std::lock_guard<std::mutex> lock(mu_);
    sessionLastTurn_[sessionId] = turnNumber;
    auto& imm = sessionWeights_[sessionId][static_cast<size_t>(Stage::Immediate)];
    auto apply = [&](const std::vector<std::string>& toks, float multiplier) {
        for (const auto& tok : toks) {
            if (tok.empty()) continue;
            auto& w = imm[tok];
            int delta = std::max(0, turnNumber - w.lastUpdateTurn);
            applyDecay(w, 0, delta);
            float grad = cfg_.learningRate * reward * multiplier;
            w.momentum = cfg_.momentum * w.momentum + (1.0f - cfg_.momentum) * grad;
            w.bias += grad + w.momentum;
            w.bias = std::max(cfg_.minBias, std::min(cfg_.maxBias, w.bias));
            w.lastUpdateTurn = turnNumber;
        }
    };
    apply(promptTokens, 0.6f);
    apply(replyTokens, 1.0f);

    if (reward > 0.0f && !promptTokens.empty() && !replyTokens.empty()) {
        auto& ctx = sessionWeights_[sessionId][static_cast<size_t>(Stage::Context)];
        for (const auto& p : promptTokens) {
            if (p.empty()) continue;
            (void)p;
            for (const auto& r : replyTokens) {
                if (r.empty()) continue;
                auto& w = ctx[r];
                int delta = std::max(0, turnNumber - w.lastUpdateTurn);
                applyDecay(w, static_cast<int>(Stage::Context), delta);
                float grad = cfg_.learningRate * reward * 0.3f;
                w.bias += grad;
                w.bias = std::max(cfg_.minBias, std::min(cfg_.maxBias, w.bias));
                w.lastUpdateTurn = turnNumber;
            }
        }
    }
}

std::vector<std::pair<std::string, float>> EmotionVocabWeightTable::rankBiases(
    const std::unordered_map<std::string, float>& biases, size_t topN) const {
    std::vector<std::pair<std::string, float>> ranked;
    ranked.reserve(biases.size());
    for (const auto& kv : biases) {
        if (std::abs(kv.second) >= cfg_.minTokenScore) ranked.push_back(kv);
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) {
                  return std::abs(a.second) > std::abs(b.second);
              });
    if (ranked.size() > topN) ranked.resize(topN);
    return ranked;
}

std::string EmotionVocabWeightTable::buildPromptModulation(
    const std::string& sessionId,
    const std::vector<std::string>& inputTokens,
    int turnNumber) const {
    auto biases = computeTokenBias(sessionId, inputTokens, turnNumber);
    auto ranked = rankBiases(biases, 16);
    if (ranked.empty()) return "";
    std::ostringstream pos, neg;
    size_t posCount = 0, negCount = 0;
    for (const auto& kv : ranked) {
        if (kv.second > 0.0f && posCount < 4) {
            if (posCount++) pos << " ";
            pos << kv.first;
        } else if (kv.second < 0.0f && negCount < 4) {
            if (negCount++) neg << " ";
            neg << kv.first;
        }
    }
    if (posCount == 0 && negCount == 0) return "";
    std::ostringstream out;
    out << "[emotion emphasis]";
    if (posCount > 0) out << " emphasize:" << pos.str();
    if (negCount > 0) out << " avoid:" << neg.str();
    out << "[/emotion emphasis]";
    return out.str();
}

nlohmann::json EmotionVocabWeightTable::getLogitBiasJson(
    const std::string& sessionId,
    const std::vector<std::string>& inputTokens,
    int turnNumber) const {
    nlohmann::json j = nlohmann::json::object();
    if (!cfg_.enabled) return j;
    auto biases = computeTokenBias(sessionId, inputTokens, turnNumber);
    for (const auto& kv : biases) {
        if (std::abs(kv.second) >= cfg_.minTokenScore) {
            j[kv.first] = kv.second;
        }
    }
    return j;
}

nlohmann::json EmotionVocabWeightTable::toJson() const {
    nlohmann::json j;
    j["enabled"] = cfg_.enabled;
    j["learningRate"] = cfg_.learningRate;
    j["maxBias"] = cfg_.maxBias;
    j["minBias"] = cfg_.minBias;
    j["decay"] = cfg_.decay;
    j["momentum"] = cfg_.momentum;
    j["tokenBoostExponent"] = cfg_.tokenBoostExponent;
    j["minTokenScore"] = cfg_.minTokenScore;
    j["applyToPrompt"] = cfg_.applyToPrompt;
    j["applyLogitBias"] = cfg_.applyLogitBias;
    j["vocabTablePath"] = cfg_.vocabTablePath;
    j["vocabCachePath"] = cfg_.vocabCachePath;

    nlohmann::json sc = nlohmann::json::array();
    for (const auto& s : cfg_.stageConfigs) {
        sc.push_back({{"influence", s.influence},
                      {"halfLifeTurns", s.halfLifeTurns},
                      {"decay", s.decay}});
    }
    j["stageConfigs"] = sc;

    nlohmann::json lex = nlohmann::json::object();
    for (const auto& kv : cfg_.seedLexicon) lex[kv.first] = kv.second;
    j["seedLexicon"] = lex;

    nlohmann::json sw = nlohmann::json::object();
    for (const auto& s : sessionWeights_) {
        nlohmann::json arr = nlohmann::json::array();
        for (size_t i = 0; i < static_cast<size_t>(Stage::Count); ++i) {
            nlohmann::json stageObj = nlohmann::json::object();
            for (const auto& kv : s.second[i]) {
                stageObj[kv.first] = {{"bias", kv.second.bias},
                                      {"momentum", kv.second.momentum},
                                      {"lastUpdateTurn", kv.second.lastUpdateTurn}};
            }
            arr.push_back(stageObj);
        }
        sw[s.first] = arr;
    }
    j["sessionWeights"] = sw;

    nlohmann::json lw = nlohmann::json::object();
    for (const auto& kv : longTermWeights_) {
        lw[kv.first] = {{"bias", kv.second.bias},
                        {"momentum", kv.second.momentum},
                        {"lastUpdateTurn", kv.second.lastUpdateTurn}};
    }
    j["longTermWeights"] = lw;

    nlohmann::json sl = nlohmann::json::object();
    for (const auto& kv : sessionLastTurn_) sl[kv.first] = kv.second;
    j["sessionLastTurn"] = sl;
    return j;
}

void EmotionVocabWeightTable::fromJson(const nlohmann::json& j) {
    if (!j.is_object()) return;
    if (j.contains("sessionWeights") && j["sessionWeights"].is_object()) {
        for (auto& [sid, arr] : j["sessionWeights"].items()) {
            if (!arr.is_array()) continue;
            auto& stages = sessionWeights_[sid];
            for (size_t i = 0; i < std::min(static_cast<size_t>(arr.size()),
                                             static_cast<size_t>(Stage::Count)); ++i) {
                if (!arr[i].is_object()) continue;
                for (auto& [tok, wj] : arr[i].items()) {
                    TokenWeight w;
                    if (wj.is_object()) {
                        w.bias = wj.value("bias", 0.0f);
                        w.momentum = wj.value("momentum", 0.0f);
                        w.lastUpdateTurn = wj.value("lastUpdateTurn", 0);
                    }
                    stages[i][tok] = w;
                }
            }
        }
    }
    if (j.contains("longTermWeights") && j["longTermWeights"].is_object()) {
        for (auto& [tok, wj] : j["longTermWeights"].items()) {
            TokenWeight w;
            if (wj.is_object()) {
                w.bias = wj.value("bias", 0.0f);
                w.momentum = wj.value("momentum", 0.0f);
                w.lastUpdateTurn = wj.value("lastUpdateTurn", 0);
            }
            longTermWeights_[tok] = w;
        }
    }
    if (j.contains("sessionLastTurn") && j["sessionLastTurn"].is_object()) {
        for (auto& [sid, v] : j["sessionLastTurn"].items()) {
            if (v.is_number_integer()) sessionLastTurn_[sid] = v.get<int>();
        }
    }
}

bool EmotionVocabWeightTable::load() {
    namespace fs = std::filesystem;
    try {
        if (!fs::exists(cfg_.vocabTablePath)) return false;
        std::ifstream f(cfg_.vocabTablePath);
        if (!f.good()) return false;
        nlohmann::json j;
        f >> j;
        fromJson(j);
        return true;
    } catch (...) {
        return false;
    }
}

bool EmotionVocabWeightTable::save() const {
    namespace fs = std::filesystem;
    try {
        fs::create_directories(fs::path(cfg_.vocabTablePath).parent_path());
        std::ofstream f(cfg_.vocabTablePath);
        if (!f.good()) return false;
        f << toJson().dump(2);
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace emotion
} // namespace phoenix
