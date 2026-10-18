/* Standalone checks for the chat affect map.
   Does not link phoenix_main and does not touch llama-server.
   Run from the repository root. */

#include "emotion_system.hpp"
#include "emotion_influence.hpp"
#include "instinct.hpp"
#include "self_mind.hpp"

#include <cmath>
#include <iostream>
#include <string>

namespace {

int gFails = 0;

void check(bool cond, const char *name) {
    if (cond) {
        std::cout << "PASS " << name << "\n";
    } else {
        std::cerr << "FAIL " << name << "\n";
        ++gFails;
    }
}

bool near(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

bool tensorNear(const phoenix::emotion::EmotionTensor &a,
                const phoenix::emotion::EmotionTensor &b,
                float eps = 1e-4f) {
    return near(a.valence, b.valence, eps) && near(a.arousal, b.arousal, eps) &&
           near(a.dominance, b.dominance, eps) && near(a.trust, b.trust, eps) &&
           near(a.joy, b.joy, eps) && near(a.fear, b.fear, eps) &&
           near(a.anger, b.anger, eps) && near(a.surprise, b.surprise, eps);
}

bool inUnit(const phoenix::emotion::EmotionTensor &t) {
    const float v[] = {t.valence, t.arousal, t.dominance, t.trust,
                       t.joy, t.fear, t.anger, t.surprise};
    for (float x : v) {
        if (!std::isfinite(x) || x < -1.0f - 1e-5f || x > 1.0f + 1e-5f)
            return false;
    }
    return true;
}

}  // namespace

int main() {
    using phoenix::emotion::EmotionTensor;
    using phoenix::emotion::FineAppraisal;
    using phoenix::emotion::affectLogitBias;
    using phoenix::emotion::blendThoughtAffect;
    using phoenix::emotion::chatCausalText;
    using phoenix::emotion::fromAppraisal;
    using phoenix::emotion::fromFineAppraisal;
    using phoenix::emotion::homeostaticUpdate;
    using phoenix::emotion::mapChatSampling;
    using phoenix::emotion::surfaceAffect;

    {
        const float pairs[][2] = {{0, 0}, {1, 0}, {0, 1}, {0.3f, 0.7f}, {0.5f, 0.5f}};
        bool same = true;
        for (const auto &p : pairs) {
            FineAppraisal fa;
            fa.benefit = p[0];
            fa.harm = p[1];
            if (!tensorNear(fromFineAppraisal(fa), fromAppraisal(p[0], p[1]), 1e-5f))
                same = false;
        }
        check(same, "fine-appraisal-matches-benefit-harm-when-neutral");
    }

    {
        FineAppraisal base;
        base.benefit = 0.2f;
        base.harm = 0.2f;
        FineAppraisal lifted = base;
        lifted.novelty = 1.0f;
        const auto a0 = fromFineAppraisal(base);
        const auto a1 = fromFineAppraisal(lifted);
        check(a1.arousal > a0.arousal + 0.2f, "novelty-raises-arousal");
        check(inUnit(a1), "novelty-stays-in-unit-box");
    }

    {
        FineAppraisal low;
        low.benefit = 0.4f;
        low.harm = 0.4f;
        low.control = 0.0f;
        FineAppraisal high = low;
        high.control = 1.0f;
        check(fromFineAppraisal(high).dominance > fromFineAppraisal(low).dominance + 0.4f,
              "control-raises-dominance");
        FineAppraisal mid = low;
        mid.control = 0.5f;
        check(near(fromFineAppraisal(mid).dominance, fromAppraisal(0.4f, 0.4f).dominance),
              "control-half-matches-utility-dominance");
    }

    {
        FineAppraisal calm;
        calm.benefit = 0.2f;
        calm.harm = 0.2f;
        FineAppraisal unsure = calm;
        unsure.uncertainty = 1.0f;
        check(std::fabs(fromFineAppraisal(unsure).surprise) >
                  std::fabs(fromFineAppraisal(calm).surprise) + 0.05f,
              "uncertainty-scales-surprise");
    }

    {
        FineAppraisal selfCaused;
        selfCaused.benefit = 0.1f;
        selfCaused.harm = 0.8f;
        FineAppraisal blamed = selfCaused;
        blamed.otherAgency = 1.0f;
        const auto a = fromFineAppraisal(selfCaused);
        const auto b = fromFineAppraisal(blamed);
        check(b.anger > a.anger + 0.05f, "other-agency-raises-anger");
        check(b.trust < a.trust - 0.05f, "other-agency-lowers-trust");
        check(inUnit(b), "blame-stays-in-unit-box");
    }

    {
        bool bounded = true;
        for (int b = 0; b <= 4; ++b) {
            for (int h = 0; h <= 4; ++h) {
                for (int n = 0; n <= 2; ++n) {
                    FineAppraisal fa;
                    fa.benefit = 0.25f * static_cast<float>(b);
                    fa.harm = 0.25f * static_cast<float>(h);
                    fa.novelty = 0.5f * static_cast<float>(n);
                    fa.control = 0.5f * static_cast<float>(n);
                    fa.uncertainty = 0.5f * static_cast<float>(n);
                    fa.otherAgency = 0.5f * static_cast<float>(n);
                    if (!inUnit(fromFineAppraisal(fa))) bounded = false;
                }
            }
        }
        check(bounded, "fine-appraisal-grid-bounded");
    }

    {
        EmotionTensor neutral;
        const auto opts = mapChatSampling(neutral);
        check(near(opts["temperature"].get<float>(), 0.30f, 1e-5f), "neutral-temp-is-0.30");
        check(near(opts["top_p"].get<float>(), 0.90f, 1e-5f), "neutral-top-p-is-0.90");
        check(near(opts["frequency_penalty"].get<float>(), 0.35f, 1e-5f), "neutral-freq-is-floor");
        check(near(opts["presence_penalty"].get<float>(), 0.15f, 1e-5f), "neutral-presence-is-floor");
        check(near(opts["repeat_penalty"].get<float>(), 1.18f, 1e-5f), "neutral-repeat-is-floor");
        check(opts["logit_bias"].empty(), "neutral-logit-bias-empty");
    }

    {
        bool band = true;
        bool floors = true;
        bool biasCap = true;
        const float grid[] = {-1.0f, -0.4f, 0.0f, 0.4f, 1.0f};
        for (float arousal : grid) {
            for (float fear : grid) {
                for (float surprise : grid) {
                    for (float joy : grid) {
                        EmotionTensor e;
                        e.arousal = arousal;
                        e.fear = fear;
                        e.surprise = surprise;
                        e.joy = joy;
                        e.anger = fear;
                        e.valence = -joy;
                        e.trust = joy;
                        e.dominance = arousal;
                        const auto opts = mapChatSampling(e);
                        const float t = opts["temperature"].get<float>();
                        const float p = opts["top_p"].get<float>();
                        if (!(t >= 0.15f - 1e-5f && t <= 0.40f + 1e-5f)) band = false;
                        if (!(p >= 0.50f - 1e-5f && p <= 0.95f + 1e-5f)) band = false;
                        if (opts["frequency_penalty"].get<float>() < 0.35f - 1e-5f) floors = false;
                        if (opts["presence_penalty"].get<float>() < 0.15f - 1e-5f) floors = false;
                        if (opts["repeat_penalty"].get<float>() < 1.18f - 1e-5f) floors = false;
                        for (auto it = opts["logit_bias"].begin(); it != opts["logit_bias"].end(); ++it) {
                            const float b = it->get<float>();
                            if (b < -0.75f - 1e-4f || b > 0.75f + 1e-4f) biasCap = false;
                        }
                    }
                }
            }
        }
        check(band, "chat-sampling-stays-in-band");
        check(floors, "chat-penalties-respect-floors");
        check(biasCap, "logit-bias-capped");
    }

    {
        EmotionTensor low;
        low.arousal = -1.0f;
        low.fear = 1.0f;
        EmotionTensor high;
        high.arousal = 1.0f;
        high.fear = -1.0f;
        const float tLow = mapChatSampling(low)["temperature"].get<float>();
        const float tHigh = mapChatSampling(high)["temperature"].get<float>();
        check(tHigh > tLow + 0.05f, "arousal-raises-temperature-inside-band");
        check(tLow >= 0.15f && tHigh <= 0.40f, "arousal-endpoints-inside-band");
        EmotionTensor angry;
        angry.anger = 1.0f;
        angry.valence = -1.0f;
        check(mapChatSampling(angry)["frequency_penalty"].get<float>() > 0.35f + 0.1f,
              "anger-lifts-frequency-penalty");
        EmotionTensor joyful;
        joyful.joy = 1.0f;
        check(mapChatSampling(joyful)["presence_penalty"].get<float>() > 0.15f,
              "joy-lifts-presence-penalty");
        const auto bias = affectLogitBias(joyful);
        check(bias.contains("glad") && bias["glad"].get<float>() > 0.2f, "joy-biases-glad");
    }

    {
        const auto heard = surfaceAffect("I am happy");
        const auto expect = phoenix::emotion::padToTensor(0.90f / 3.0f, 0.70f / 3.0f, 0.50f / 3.0f);
        check(tensorNear(heard, expect, 1e-4f), "english-happy-mean-pools-like-before");
        const char *fear = "\xE6\x88\x91\xE5\xBE\x88\xE5\xAE\xB3\xE6\x80\x95"; /* 我很害怕 */
        const auto zh = surfaceAffect(fear);
        check(zh.fear > 0.2f && zh.valence < 0.0f, "chinese-fear-word-moves-pad");
        check(surfaceAffect("no lexicon hits here").distance(EmotionTensor{}) < 1e-6f,
              "unknown-words-stay-neutral");
    }

    {
        EmotionTensor cur;
        cur.valence = 1.0f;
        EmotionTensor baseline;
        baseline.valence = -0.2f;
        const auto one = homeostaticUpdate(cur, baseline, baseline, 0.80f, 0.25f);
        /* couple 0.20: 1 + 0.20*(-1.2) = 0.76; then pull 0.25: 0.76 + 0.25*(-0.96) = 0.52. */
        check(near(one.valence, 0.52f, 1e-4f), "homeostasis-one-step");
        for (int i = 0; i < 40; ++i)
            cur = homeostaticUpdate(cur, baseline, baseline, 0.80f, 0.25f);
        check(std::fabs(cur.valence - baseline.valence) < 0.02f,
              "homeostasis-converges-when-stimulus-is-baseline");
        check(inUnit(cur), "homeostasis-stays-bounded");
        /* A standing neutral stimulus does not erase a non-zero baseline.
           Fixed point of c <- 0.6 c + 0.25 b is c = 0.625 b. */
        EmotionTensor pulled;
        pulled.valence = 1.0f;
        EmotionTensor neutralStim;
        for (int i = 0; i < 80; ++i)
            pulled = homeostaticUpdate(pulled, neutralStim, baseline, 0.80f, 0.25f);
        check(near(pulled.valence, 0.625f * baseline.valence, 0.02f),
              "homeostasis-neutral-stimulus-fixed-point");
    }

    {
        EmotionTensor origin;
        const auto prefer = blendThoughtAffect(origin, "prefer", "", 0.22f);
        const auto service = blendThoughtAffect(origin, "service", "", 0.22f);
        check(prefer.valence > 0.02f && service.valence < -0.02f, "thought-action-moves-valence");
        check(prefer.valence > service.valence, "prefer-is-above-service");
        check(tensorNear(blendThoughtAffect(origin, "", "", 0.22f), origin),
              "empty-thought-is-identity");
        const auto glad = blendThoughtAffect(origin, "continue", "I feel glad and calm", 0.0f);
        check(tensorNear(glad, origin), "zero-gain-ignores-text");
        const auto moved = blendThoughtAffect(origin, "", "I feel glad", 0.40f);
        check(moved.valence > origin.valence, "thought-text-can-raise-valence");
        check(inUnit(moved) && inUnit(service), "thought-blend-stays-in-unit-box");
    }

    {
        phoenix::selfmind::SelfMind mind;
        mind.configure(nlohmann::json{{"enabled", true}, {"learningRate", 0.35}});
        const auto tick = mind.tickThought(nlohmann::json{{"userPrompt", "ZZZ-QUESTION-TOKEN?"}});
        const auto snap = mind.toJson();
        const std::string thought = tick.value("text", std::string());
        (void)blendThoughtAffect(EmotionTensor{}, tick.value("action", std::string()), thought, 0.22f);
        check(mind.toJson() == snap, "blend-does-not-write-self-mind");
        check(thought.find("ZZZ-QUESTION-TOKEN") == std::string::npos, "thought-ignores-question");
        const std::string user = "What is two plus two?";
        const std::string causal = chatCausalText(user);
        check(causal == user + "\n", "causal-text-is-user-text");
        check(causal.find(thought) == std::string::npos, "causal-text-omits-thought");
        check(causal.find("Affect:") == std::string::npos, "causal-text-omits-affect-label");
        check(causal.find("<emotion") == std::string::npos, "causal-text-omits-emotion-tag");
        check(causal.find("[self-kernel]") == std::string::npos, "causal-text-omits-kernel");
        check(causal.find("high-arousal") == std::string::npos, "causal-text-omits-modulation");
        const auto felt = phoenix::instinct::sensationFromThought(
            tick.value("action", std::string()), "chat:s1");
        check(felt.source == "chat:s1:thought", "thought-sensation-is-scoped");
        check(felt.source.find(thought) == std::string::npos, "sensation-source-is-not-the-thought");
        phoenix::primal::PrimalSensationEngine eng;
        eng.add(felt);
        eng.add(phoenix::instinct::sensationFromThought("service", "mission:m"));
        check(eng.activeFor("chat:s1").size() == 1, "chat-sees-only-its-thought");
        check(eng.activeFor("mission:m").size() == 1, "mission-sees-only-its-thought");
        check(eng.activeFor("chat:other").empty(), "other-chat-does-not-see-it");
    }

    {
        phoenix::emotion_influence::EmotionInfluence influence;
        EmotionTensor hot;
        hot.arousal = 1.0f;
        hot.joy = 0.8f;
        hot.fear = -0.4f;
        const auto r = influence.compute(hot, "happy", 3);
        check(r.promptModulation.empty(), "influence-prompt-modulation-empty");
        const float t = r.inferenceOptions["temperature"].get<float>();
        check(t >= 0.15f && t <= 0.40f, "influence-temperature-in-chat-band");
        check(r.inferenceOptions["frequency_penalty"].get<float>() >= 0.35f - 1e-5f,
              "influence-frequency-at-least-floor");
        bool capped = true;
        for (auto it = r.logitBias.begin(); it != r.logitBias.end(); ++it) {
            const float b = it->get<float>();
            if (b < -0.75f - 1e-4f || b > 0.75f + 1e-4f) capped = false;
        }
        check(capped, "influence-logit-bias-capped");
    }

    {
        auto eng = phoenix::instinct::InstinctEngine::defaultEngine();
        phoenix::primal::PrimalSensation nov;
        nov.type = phoenix::primal::SensationType::Novelty;
        nov.intensity = 0.9f;
        nov.valence = 0.0f;
        const auto with = eng.evaluate({nov});
        const auto plain = fromAppraisal(with.benefitScore, with.harmScore);
        check(with.novelty > 0.5f, "instinct-records-novelty");
        check(with.driveVector.size() == 8, "drive-is-8");
        check(with.driveVector[1] > plain.arousal + 0.2f, "instinct-novelty-lifts-arousal");
        phoenix::primal::PrimalSensation pleasure;
        pleasure.type = phoenix::primal::SensationType::Pleasure;
        pleasure.intensity = 1.0f;
        pleasure.valence = 0.6f;
        const auto pleased = eng.evaluate({pleasure});
        const auto pleasedPlain = fromAppraisal(pleased.benefitScore, pleased.harmScore);
        check(pleased.control > 0.5f, "pleasure-raises-control");
        check(pleased.driveVector[2] > pleasedPlain.dominance, "control-changes-dominance");
        const auto none = eng.evaluate({});
        const auto nonePlain = fromAppraisal(none.benefitScore, none.harmScore);
        check(near(none.driveVector[0], nonePlain.valence) &&
                  near(none.driveVector[1], nonePlain.arousal) &&
                  near(none.driveVector[7], nonePlain.surprise),
              "no-sensation-matches-old-appraisal");
    }

    if (gFails) {
        std::cerr << gFails << " failed\n";
        return 1;
    }
    std::cout << "all passed\n";
    return 0;
}
