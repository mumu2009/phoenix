/* emotion_influence.cpp - Evidence-based emotion influence on LLM generation */

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
    (void)emotionKey;
    const float cap = std::min(0.75f, std::max(0.0f, cfg_.maxBias));
    return std::clamp(cfg_.strength * emotionValue, -cap, cap);
}

std::string EmotionInfluence::buildPromptModulation_(
    const phoenix::emotion::EmotionTensor &) const {
    /* Chat causal text must not carry an affect sentence. */
    return std::string();
}

nlohmann::json EmotionInfluence::buildInferenceOptions_(
    const phoenix::emotion::EmotionTensor &e) const {
    return phoenix::emotion::mapChatSampling(e, cfg_.strength);
}

EmotionInfluenceResult EmotionInfluence::compute(
    const phoenix::emotion::EmotionTensor &emotion,
    const std::string & /*context*/,
    int64_t /*turn*/) const {
    EmotionInfluenceResult r;
    r.logitBias = phoenix::emotion::affectLogitBias(emotion, cfg_.strength);
    auto dimOf = [&](const std::string &key) -> float {
        if (key == "joy" || key == "positive")
            return 0.60f * emotion.joy + 0.15f * emotion.valence;
        if (key == "trust") return 0.65f * emotion.trust;
        if (key == "fear") return 0.70f * emotion.fear;
        if (key == "anger" || key == "negative") return 0.70f * emotion.anger;
        if (key == "surprise") return 0.65f * emotion.surprise;
        if (key == "sadness") return -0.55f * emotion.joy;
        if (key == "dominance") return 0.55f * emotion.dominance;
        if (key == "submission") return -0.55f * emotion.dominance;
        return 0.0f;
    };
    for (const auto &kv : cfg_.affectiveLexicon) {
        const float clamped = tokenScore_(kv.first, dimOf(kv.first));
        if (std::fabs(clamped) < 0.02f) continue;
        for (const auto &tok : kv.second) {
            if (!tok.empty()) r.logitBias[tok] = clamped;
        }
    }
    r.promptModulation = buildPromptModulation_(emotion);
    r.inferenceOptions = buildInferenceOptions_(emotion);
    if (r.inferenceOptions.is_object())
        r.inferenceOptions["logit_bias"] = r.logitBias;

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
