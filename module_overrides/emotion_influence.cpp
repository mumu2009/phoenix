/* emotion_influence.cpp - Evidence-based emotion influence on LLM generation
   Copyright (C) 2026 079 Project */

#include "emotion_influence.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace phoenix {
namespace emotion_influence {

EmotionInfluenceConfig EmotionInfluenceConfig::fromJson(const nlohmann::json &j) {
    EmotionInfluenceConfig c;
    if (j.contains("strength")) c.strength = j["strength"].get<float>();
    if (j.contains("maxBias")) c.maxBias = j["maxBias"].get<float>();
    if (j.contains("minBias")) c.minBias = j["minBias"].get<float>();
    if (j.contains("baseTemperature")) c.baseTemperature = j["baseTemperature"].get<float>();
    if (j.contains("baseTopP")) c.baseTopP = j["baseTopP"].get<float>();
    if (j.contains("decay")) c.decay = j["decay"].get<float>();
    if (j.contains("affectiveLexicon") && j["affectiveLexicon"].is_object()) {
        for (auto &[k, v] : j["affectiveLexicon"].items()) {
            if (v.is_array()) {
                for (const auto &item : v) c.affectiveLexicon[k].push_back(item.get<std::string>());
            }
        }
    }
    return c;
}

nlohmann::json EmotionInfluenceConfig::toJson() const {
    nlohmann::json lex;
    for (const auto &[k, v] : affectiveLexicon) lex[k] = v;
    return {
        {"strength", strength},
        {"maxBias", maxBias},
        {"minBias", minBias},
        {"baseTemperature", baseTemperature},
        {"baseTopP", baseTopP},
        {"decay", decay},
        {"affectiveLexicon", lex}
    };
}

struct EmotionInfluence::State {
    float accumulatedArousal{0.0f};
    float accumulatedValence{0.0f};
    int turn{0};
};

EmotionInfluence::EmotionInfluence(const EmotionInfluenceConfig &cfg) : cfg_(cfg) {
    if (cfg_.affectiveLexicon.empty()) {
        // Default lexicon is loosely grounded in affective-norm research:
        // high valence -> positive / approach, low valence -> negative / avoid,
        // high arousal -> active / urgent, low arousal -> calm / reflective.
        cfg_.affectiveLexicon = {
            {"joy", {"wonderful", "glad", "delighted", "cheerful"}},
            {"trust", {"trust", "reliable", "confident", "dependable"}},
            {"fear", {"worry", "anxious", "cautious", "concerned"}},
            {"anger", {"frustrated", "annoyed", "outraged", "irritated"}},
            {"surprise", {"unexpected", "amazing", "astonishing", "curious"}},
            {"sadness", {"sorry", "regret", "unfortunately", "sad"}},
            {"dominance", {"must", "should", "directly", "clearly"}},
            {"submission", {"please", "suggest", "consider", "perhaps"}}
        };
    }
}

EmotionInfluence::~EmotionInfluence() = default;

void EmotionInfluence::setAffectiveLexicon(
    const std::unordered_map<std::string, std::vector<std::string>> &lexicon) {
    cfg_.affectiveLexicon = lexicon;
}

float EmotionInfluence::tokenScore_(const std::string &emotionKey, float emotionValue) const {
    return cfg_.strength * std::tanh(emotionValue) * 0.5f;
}

std::string EmotionInfluence::buildPromptModulation_(
    const phoenix::emotion::EmotionTensor &e) const {
    std::ostringstream oss;
    oss << "<emotion: ";
    if (e.arousal >= 0.3f) oss << "high-arousal; ";
    else if (e.arousal <= -0.3f) oss << "low-arousal; ";
    if (e.valence >= 0.3f) oss << "positive; ";
    else if (e.valence <= -0.3f) oss << "negative; ";
    if (e.dominance >= 0.3f) oss << "directive; ";
    else if (e.dominance <= -0.3f) oss << "deferential; ";
    if (e.fear >= 0.3f) oss << "risk-averse; ";
    if (e.anger >= 0.3f) oss << "tense; ";
    if (e.joy >= 0.3f) oss << "optimistic; ";
    if (e.surprise >= 0.3f) oss << "attentive; ";
    std::string s = oss.str();
    if (s.size() > 10) s.pop_back(), s.pop_back();
    s += ">";
    return s;
}

nlohmann::json EmotionInfluence::buildInferenceOptions_(
    const phoenix::emotion::EmotionTensor &e) const {
    float temperature = cfg_.baseTemperature + 0.35f * e.arousal - 0.15f * e.fear;
    float topP = cfg_.baseTopP + 0.08f * e.surprise - 0.05f * e.dominance;
    float presence = -0.2f * e.trust + 0.15f * e.joy;
    float frequency = 0.15f * e.anger - 0.1f * e.valence;
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

EmotionInfluenceResult EmotionInfluence::compute(
    const phoenix::emotion::EmotionTensor &emotion,
    const std::string & /*context*/,
    int64_t /*turn*/) const {
    EmotionInfluenceResult r;
    nlohmann::json logitBias = nlohmann::json::object();

    // Map each emotion dimension to token scores for its token group.
    auto addTokens = [&](const std::vector<std::string> &tokens, float score) {
        if (tokens.empty()) return;
        float clamped = std::max(cfg_.minBias, std::min(cfg_.maxBias, score));
        for (const auto &tok : tokens) {
            logitBias[tok] = clamped;
        }
    };

    auto it = cfg_.affectiveLexicon.find("joy");
    if (it != cfg_.affectiveLexicon.end()) addTokens(it->second, tokenScore_("joy", emotion.joy + emotion.valence));
    it = cfg_.affectiveLexicon.find("trust");
    if (it != cfg_.affectiveLexicon.end()) addTokens(it->second, tokenScore_("trust", emotion.trust));
    it = cfg_.affectiveLexicon.find("fear");
    if (it != cfg_.affectiveLexicon.end()) addTokens(it->second, tokenScore_("fear", emotion.fear - emotion.valence));
    it = cfg_.affectiveLexicon.find("anger");
    if (it != cfg_.affectiveLexicon.end()) addTokens(it->second, tokenScore_("anger", emotion.anger - emotion.valence));
    it = cfg_.affectiveLexicon.find("surprise");
    if (it != cfg_.affectiveLexicon.end()) addTokens(it->second, tokenScore_("surprise", emotion.surprise));
    it = cfg_.affectiveLexicon.find("dominance");
    if (it != cfg_.affectiveLexicon.end()) addTokens(it->second, tokenScore_("dominance", emotion.dominance));
    it = cfg_.affectiveLexicon.find("submission");
    if (it != cfg_.affectiveLexicon.end()) addTokens(it->second, tokenScore_("submission", -emotion.dominance));

    r.logitBias = logitBias;
    r.promptModulation = buildPromptModulation_(emotion);
    r.inferenceOptions = buildInferenceOptions_(emotion);

    std::ostringstream ex;
    ex << "emotion influence: arousal=" << emotion.arousal
       << " valence=" << emotion.valence
       << " dominance=" << emotion.dominance;
    r.explanation = ex.str();
    return r;
}

EmotionInfluenceResult EmotionInfluence::computeStateful(
    const phoenix::emotion::EmotionTensor &emotion,
    const std::string &sessionId,
    const std::string &context,
    int64_t turn) {
    auto &st = sessionStates_[sessionId];
    if (!st) st = std::make_unique<State>();
    st->turn = static_cast<int>(turn);
    st->accumulatedArousal = cfg_.decay * st->accumulatedArousal + emotion.arousal;
    st->accumulatedValence = cfg_.decay * st->accumulatedValence + emotion.valence;
    auto r = compute(emotion, context, turn);
    r.explanation += " (session " + sessionId + " accumulated valence=" +
                     std::to_string(st->accumulatedValence) + ")";
    return r;
}

void EmotionInfluence::resetSession(const std::string &sessionId) {
    sessionStates_.erase(sessionId);
}

nlohmann::json EmotionInfluence::status() const {
    nlohmann::json j;
    j["config"] = cfg_.toJson();
    j["sessions"] = sessionStates_.size();
    return j;
}

} // namespace emotion_influence
} // namespace phoenix
