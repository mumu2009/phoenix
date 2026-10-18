/* emotion_system.hpp - Emotion modeling and analysis system */

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

    /* Wide sampling map used by mission blending.  Chat must call
       mapChatSampling(), which keeps temperature inside [0.15, 0.40]. */
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
/**
 * @brief Canonical PAD -> 8-dimension affect mapping.
 *
 * Grounds the 8-d emotion tensor in the Pleasure-Arousal-Dominance (PAD)
 * model (Mehrabian & Russell, 1974) as the three core dimensions, and derives
 * the five appraisal emotions (trust, joy, fear, anger, surprise) as documented
 * functions of PAD per appraisal theory (Ortony, Clore & Collins, 1988;
 * Smith & Lazarus, 1990).
 *
 * Let V = valence, A = arousal, D = dominance, each in [-1, 1]. Then:
 *   valence   = V
 *   arousal   = A
 *   dominance = D
 *   trust     = (V + D) / 2     // approach + control -> trust; else suspicion
 *   joy       = (V + A) / 2     // approach + energy  -> joy;  else sadness
 *   fear      = (-V - D) / 2    // threat  + submission -> fear; else courage
 *   anger     = (-V + D) / 2    // threat  + control   -> anger; else calm
 *   surprise  = A * (1 - |D|)   // energy x uncertainty  -> surprise
 *
 * All outputs lie in [-1, 1] (see doc/v7.0/algorithm.md section 15 for the
 * boundedness proof).  The mapping is linear in (V, A, D) except for the |D|
 * term in surprise, hence Lipschitz-continuous.
 */
inline EmotionTensor padToTensor(float valence, float arousal, float dominance) {
    const float V = std::clamp(valence, -1.0f, 1.0f);
    const float A = std::clamp(arousal, -1.0f, 1.0f);
    const float D = std::clamp(dominance, -1.0f, 1.0f);
    EmotionTensor t;
    t.valence = V;
    t.arousal = A;
    t.dominance = D;
    t.trust = (V + D) * 0.5f;
    t.joy = (V + A) * 0.5f;
    t.fear = (-V - D) * 0.5f;
    t.anger = (-V + D) * 0.5f;
    t.surprise = A * (1.0f - std::abs(D));
    return t;
}

/**
 * @brief Benefit/harm appraisal -> 8-dimension affect tensor.
 *
 * Maps a benefit-harm (趋利避害) appraisal to PAD and then to the 8-d tensor.
 * Given benefit B and harm H in [0, 1]:
 *   U = (B - H) / (B + H + eps)      // net utility, [-1, 1]
 *   A = 2 * min(B + H, 1) - 1        // arousal energy mapped to [-1, 1]
 * and dominance is set equal to U because the instinct layer carries no
 * independent coping/blame signal (approach implies dominance, avoidance
 * implies submission).  The result is padToTensor(U, A, U).
 *
 * This single canonical function replaces the previous ad-hoc 8x5 fixed
 * matrix in InstinctEngine::evaluate().  See algorithm.md section 7/15 for
 * the monotonicity and boundedness proofs.
 */
inline EmotionTensor fromAppraisal(float benefit, float harm) {
    const float B = std::clamp(benefit, 0.0f, 1.0f);
    const float H = std::clamp(harm, 0.0f, 1.0f);
    const float eps = 1e-6f;
    const float U = (B - H) / (B + H + eps);
    const float A = 2.0f * std::min(B + H, 1.0f) - 1.0f;
    return padToTensor(U, A, U);
}

/**
 * Finer appraisal on top of benefit/harm.  Neutral extras
 * (novelty = 0, control = 0.5, uncertainty = 0, otherAgency = 0)
 * reproduce fromAppraisal(benefit, harm) exactly.
 *
 *   U  = (B - H) / (B + H + eps)
 *   A0 = 2 * min(B + H, 1) - 1
 *   A  = clamp(A0 + 0.55 * novelty * (1 - A0), -1, 1)
 *   D  = clamp(U + 0.65 * (2*control - 1) * (1 - |U|), -1, 1)
 *   t  = padToTensor(U, A, D)
 *   surprise *= (1 + 0.75 * uncertainty)     then clamp
 *   blame = otherAgency * H
 *   anger += 0.40 * blame * (1 - anger)
 *   trust -= 0.35 * blame * (1 + trust)
 */
struct FineAppraisal {
    float benefit{0.0f};
    float harm{0.0f};
    float novelty{0.0f};
    float control{0.5f};
    float uncertainty{0.0f};
    float otherAgency{0.0f};
};

inline EmotionTensor fromFineAppraisal(const FineAppraisal &in) {
    const float B = std::clamp(in.benefit, 0.0f, 1.0f);
    const float H = std::clamp(in.harm, 0.0f, 1.0f);
    const float N = std::clamp(in.novelty, 0.0f, 1.0f);
    const float C = std::clamp(in.control, 0.0f, 1.0f);
    const float Unc = std::clamp(in.uncertainty, 0.0f, 1.0f);
    const float O = std::clamp(in.otherAgency, 0.0f, 1.0f);
    const float eps = 1e-6f;
    const float U = (B - H) / (B + H + eps);
    const float A0 = 2.0f * std::min(B + H, 1.0f) - 1.0f;
    const float A = std::clamp(A0 + 0.55f * N * (1.0f - A0), -1.0f, 1.0f);
    const float pull = 2.0f * C - 1.0f;
    const float D = std::clamp(U + 0.65f * pull * (1.0f - std::fabs(U)), -1.0f, 1.0f);
    EmotionTensor t = padToTensor(U, A, D);
    t.surprise = std::clamp(t.surprise * (1.0f + 0.75f * Unc), -1.0f, 1.0f);
    const float blame = O * H;
    t.anger = std::clamp(t.anger + 0.40f * blame * (1.0f - t.anger), -1.0f, 1.0f);
    t.trust = std::clamp(t.trust - 0.35f * blame * (1.0f + t.trust), -1.0f, 1.0f);
    return t;
}

/** Same two lerps as EmotionSystem::processMessage. */
inline EmotionTensor homeostaticUpdate(const EmotionTensor &current,
                                       const EmotionTensor &stimulus,
                                       const EmotionTensor &baseline,
                                       float decayRate,
                                       float homeostasisRate) {
    const float couple = std::clamp(1.0f - decayRate, 0.0f, 1.0f);
    const float pull = std::clamp(homeostasisRate, 0.0f, 1.0f);
    const EmotionTensor coupled = EmotionTensor::lerp(current, stimulus, couple);
    return EmotionTensor::lerp(coupled, baseline, pull);
}

inline std::vector<float> tensorToDrive(const EmotionTensor &t) {
    return {t.valence, t.arousal, t.dominance, t.trust,
            t.joy, t.fear, t.anger, t.surprise};
}

/** Word -> (valence, arousal, dominance), each in [-1, 1]. */
inline const std::unordered_map<std::string, std::array<float, 3>> &vadLexicon() {
    static const std::unordered_map<std::string, std::array<float, 3>> lex = {
        {"happy", {0.90f, 0.70f, 0.50f}},
        {"glad", {0.80f, 0.50f, 0.40f}},
        {"great", {0.80f, 0.60f, 0.40f}},
        {"wonderful", {0.90f, 0.60f, 0.50f}},
        {"excellent", {0.90f, 0.60f, 0.60f}},
        {"love", {0.90f, 0.80f, 0.30f}},
        {"joy", {0.90f, 0.80f, 0.50f}},
        {"amazing", {0.90f, 0.80f, 0.40f}},
        {"calm", {0.40f, -0.50f, 0.30f}},
        {"peaceful", {0.55f, -0.40f, 0.20f}},
        {"hopeful", {0.60f, 0.30f, 0.25f}},
        {"proud", {0.70f, 0.45f, 0.65f}},
        {"grateful", {0.75f, 0.25f, 0.20f}},
        {"relief", {0.50f, -0.20f, 0.15f}},
        {"safe", {0.45f, -0.30f, 0.35f}},
        {"curious", {0.25f, 0.45f, 0.10f}},
        {"sad", {-0.80f, -0.40f, -0.50f}},
        {"bad", {-0.70f, 0.20f, -0.30f}},
        {"terrible", {-0.90f, 0.40f, -0.40f}},
        {"awful", {-0.90f, 0.30f, -0.50f}},
        {"hate", {-0.90f, 0.60f, 0.20f}},
        {"grief", {-0.85f, -0.20f, -0.55f}},
        {"lonely", {-0.60f, -0.30f, -0.40f}},
        {"ashamed", {-0.70f, 0.20f, -0.60f}},
        {"guilty", {-0.65f, 0.25f, -0.45f}},
        {"bored", {-0.30f, -0.60f, -0.10f}},
        {"afraid", {-0.80f, 0.70f, -0.70f}},
        {"scared", {-0.80f, 0.70f, -0.70f}},
        {"fear", {-0.80f, 0.60f, -0.60f}},
        {"worried", {-0.60f, 0.50f, -0.50f}},
        {"anxious", {-0.60f, 0.60f, -0.50f}},
        {"panic", {-0.85f, 0.95f, -0.80f}},
        {"danger", {-0.70f, 0.75f, -0.40f}},
        {"angry", {-0.80f, 0.80f, 0.70f}},
        {"furious", {-0.90f, 0.90f, 0.80f}},
        {"mad", {-0.70f, 0.70f, 0.60f}},
        {"rage", {-0.90f, 0.95f, 0.75f}},
        {"tense", {-0.40f, 0.55f, -0.15f}},
        {"surprise", {0.20f, 0.80f, 0.00f}},
        {"shock", {-0.40f, 0.90f, -0.60f}},
        {"unexpected", {0.00f, 0.70f, -0.20f}},
        {"confused", {-0.20f, 0.35f, -0.30f}},
        {"trust", {0.60f, 0.10f, 0.40f}},
        {"believe", {0.50f, 0.10f, 0.30f}},
        {"confident", {0.70f, 0.40f, 0.80f}},
        {"reliable", {0.60f, 0.10f, 0.50f}},
        {"\xE5\xAE\xB3\xE6\x80\x95", {-0.80f, 0.70f, -0.70f}}, /* 害怕 */
        {"\xE9\xAB\x98\xE5\x85\xB4", {0.85f, 0.70f, 0.40f}},   /* 高兴 */
        {"\xE7\x94\x9F\xE6\xB0\x94", {-0.75f, 0.75f, 0.55f}}, /* 生气 */
        {"\xE9\x9A\xBE\xE8\xBF\x87", {-0.75f, -0.35f, -0.45f}}, /* 难过 */
        {"\xE5\xAE\x89\xE5\xBF\x83", {0.45f, -0.40f, 0.25f}}, /* 安心 */
        {"\xE5\xA5\xBD\xE5\xA5\x87", {0.20f, 0.50f, 0.05f}},   /* 好奇 */
        {"\xE8\xAE\xA8\xE5\x8E\x8C", {-0.80f, 0.55f, 0.15f}}, /* 讨厌 */
        {"\xE4\xBF\xA1\xE4\xBB\xBB", {0.60f, 0.10f, 0.40f}},   /* 信任 */
        {"\xE6\x83\x8A\xE8\xAE\xB6", {0.15f, 0.80f, -0.10f}}, /* 惊讶 */
        {"\xE5\xAD\xA4\xE7\x8B\xAC", {-0.60f, -0.25f, -0.40f}}, /* 孤独 */
    };
    return lex;
}

/** Mean-pool lexicon hits. Unmatched ASCII tokens contribute 0.
    CJK keys are substring hits and each adds one to the pool size. */
inline EmotionTensor surfaceAffect(const std::string &text) {
    std::string lower;
    lower.reserve(text.size());
    bool anyHigh = false;
    for (unsigned char ch : text) {
        if (ch < 128) {
            lower.push_back(static_cast<char>(std::tolower(ch)));
        } else {
            lower.push_back(static_cast<char>(ch));
            anyHigh = true;
        }
    }
    std::vector<std::string> toks;
    std::string cur;
    for (unsigned char ch : lower) {
        if (std::isalnum(ch) || ch == '\'') {
            cur.push_back(static_cast<char>(ch));
        } else if (!cur.empty()) {
            toks.push_back(cur);
            cur.clear();
        }
    }
    if (!cur.empty()) toks.push_back(cur);
    const auto &lex = vadLexicon();
    float V = 0.0f, A = 0.0f, D = 0.0f;
    float pool = static_cast<float>(toks.size());
    for (const auto &t : toks) {
        auto it = lex.find(t);
        if (it == lex.end()) continue;
        V += it->second[0];
        A += it->second[1];
        D += it->second[2];
    }
    if (anyHigh) {
        for (const auto &kv : lex) {
            bool cjk = false;
            for (unsigned char c : kv.first) {
                if (c >= 128) { cjk = true; break; }
            }
            if (!cjk) continue;
            if (lower.find(kv.first) == std::string::npos) continue;
            V += kv.second[0];
            A += kv.second[1];
            D += kv.second[2];
            pool += 1.0f;
        }
    }
    if (pool <= 0.0f) return EmotionTensor();
    const float inv = 1.0f / pool;
    return padToTensor(V * inv, A * inv, D * inv);
}

/** Continuous logit nudge. Neutral affect emits an empty object.
    Values stay in [-0.75, 0.75]. These words are sampler biases, not prompt text. */
inline nlohmann::json affectLogitBias(const EmotionTensor &e, float strength = 1.0f) {
    const float k = std::clamp(strength, 0.0f, 2.0f);
    constexpr float kCap = 0.75f;
    nlohmann::json j = nlohmann::json::object();
    auto put = [&](const char *word, float raw) {
        const float b = std::clamp(k * raw, -kCap, kCap);
        if (std::fabs(b) < 0.02f) return;
        j[word] = b;
    };
    put("glad", 0.60f * e.joy + 0.15f * e.valence);
    put("wonderful", 0.55f * e.joy);
    put("worry", 0.70f * e.fear);
    put("anxious", 0.65f * e.fear);
    put("frustrated", 0.70f * e.anger);
    put("annoyed", 0.55f * e.anger);
    put("trust", 0.65f * e.trust);
    put("reliable", 0.50f * e.trust);
    put("unexpected", 0.65f * e.surprise);
    put("curious", 0.45f * e.surprise);
    put("clearly", 0.55f * e.dominance);
    put("perhaps", -0.55f * e.dominance);
    return j;
}

/** Chat sampling. Temperature lands in [0.15, 0.40], centered at 0.30.
    Penalties are floors plus a non-negative lift, so emotion can raise
    them and cannot drag them under the chat floors. */
inline nlohmann::json mapChatSampling(const EmotionTensor &e, float biasStrength = 1.0f) {
    constexpr float kTNeutral = 0.30f;
    constexpr float kTAmp = 0.10f;
    constexpr float kTMin = 0.15f;
    constexpr float kTMax = 0.40f;
    const float sTemp = std::clamp(0.55f * e.arousal - 0.35f * e.fear + 0.10f * e.surprise,
                                   -1.0f, 1.0f);
    const float temperature = std::clamp(kTNeutral + kTAmp * sTemp, kTMin, kTMax);
    const float topP = std::clamp(0.90f + 0.04f * e.surprise - 0.03f * e.dominance,
                                  0.50f, 0.95f);
    const float uFreq = std::clamp(0.5f * std::max(e.anger, 0.0f) + 0.5f * std::max(-e.valence, 0.0f),
                                   0.0f, 1.0f);
    const float uPres = std::clamp(0.5f * std::max(-e.trust, 0.0f) + 0.5f * std::max(e.joy, 0.0f),
                                   0.0f, 1.0f);
    const float uRep = std::clamp(0.6f * std::max(e.anger, 0.0f) + 0.4f * std::max(e.surprise, 0.0f),
                                  0.0f, 1.0f);
    return nlohmann::json{
        {"temperature", temperature},
        {"top_p", topP},
        {"frequency_penalty", 0.35f + 0.25f * uFreq},
        {"presence_penalty", 0.15f + 0.20f * uPres},
        {"repeat_penalty", 1.18f + 0.22f * uRep},
        {"logit_bias", affectLogitBias(e, biasStrength)}
    };
}

/** Read-only thought nudge. Does not accept a SelfMind and returns no text. */
inline EmotionTensor blendThoughtAffect(const EmotionTensor &base,
                                        const std::string &action,
                                        const std::string &text,
                                        float gain) {
    float v = 0.0f, a = 0.0f, d = 0.0f;
    if (action == "recall") { v = 0.15f; a = -0.10f; d = 0.12f; }
    else if (action == "prefer") { v = 0.28f; a = 0.06f; d = 0.16f; }
    else if (action == "continue") { v = 0.05f; a = 0.12f; d = 0.05f; }
    else if (action == "ask") { v = 0.00f; a = 0.22f; d = -0.08f; }
    else if (action == "service") { v = -0.22f; a = 0.08f; d = -0.24f; }
    const float g = std::clamp(gain, 0.0f, 1.0f);
    const EmotionTensor nud = padToTensor(v, a, d);
    const EmotionTensor lex = surfaceAffect(text);
    const float lg = 0.50f * g;
    auto mix = [&](float oldv, float nv, float lv) {
        return std::clamp(oldv + g * nv + lg * lv, -1.0f, 1.0f);
    };
    EmotionTensor out;
    out.valence = mix(base.valence, nud.valence, lex.valence);
    out.arousal = mix(base.arousal, nud.arousal, lex.arousal);
    out.dominance = mix(base.dominance, nud.dominance, lex.dominance);
    out.trust = mix(base.trust, nud.trust, lex.trust);
    out.joy = mix(base.joy, nud.joy, lex.joy);
    out.fear = mix(base.fear, nud.fear, lex.fear);
    out.anger = mix(base.anger, nud.anger, lex.anger);
    out.surprise = mix(base.surprise, nud.surprise, lex.surprise);
    return out;
}

/** Chat causal string is the user text only. No affect labels, no thought. */
inline std::string chatCausalText(const std::string &userText) {
    if (userText.empty()) return std::string();
    if (userText.back() == '\n') return userText;
    return userText + "\n";
}


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
        float emotionDecayRate{0.95f};            /* Emotion decay per turn (retention of previous state) */
        float homeostasisRate{0.05f};        /* Homeostatic pull toward baseline per turn */
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
