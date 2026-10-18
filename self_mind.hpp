/* self_mind.hpp - Person model: dialogue RL, inner thought, mirror/Turing probes.

   The chat window stays a separate channel. This module is the layer under it:
   a first-person continuation state learned from dialogue, plus thoughts that
   fire without being an answer to a question.

   Learning is a full-information contextual bandit (one-step TD, gamma = 0):
   every speech act is scored against the human centroid and the service-window
   centroid, then the linear policy is updated. It does not fine-tune the LLM.

   Mirror and Turing here are behavioral probes of that self-model. They are
   not a claim of phenomenal consciousness, and the Turing probe is not a
   human-judge imitation game.
*/
#pragma once

#include <nlohmann/json.hpp>
#include <mutex>
#include <string>
#include <vector>

namespace phoenix {
namespace selfmind {

constexpr int kFeatDim = 24;
constexpr int kPolicyDim = kFeatDim + 1; /* last component is a bias */
constexpr int kActs = 5;

class SelfMind {
public:
    SelfMind() = default;
    SelfMind(const SelfMind &) = delete;
    SelfMind &operator=(const SelfMind &) = delete;

    nlohmann::json configure(const nlohmann::json &payload);
    nlohmann::json ingest(const nlohmann::json &payload);
    nlohmann::json observeUtterance(const std::string &role, const std::string &text);
    /* Ignores userPrompt / questions. The thought is the next inner line. */
    nlohmann::json tickThought(const nlohmann::json &ctx = nlohmann::json::object());
    nlohmann::json probe();
    nlohmann::json status() const;
    double scoreText(const std::string &text) const;
    std::string kernelText() const;
    bool enabled() const;
    bool learnFromLiveTurns() const;

    nlohmann::json toJson() const;
    void fromJson(const nlohmann::json &j);

private:
    struct Feat {
        float v[kFeatDim]{};
        float fp{0.f};
        float bot{0.f};
    };

    nlohmann::json configureLocked(const nlohmann::json &payload);
    nlohmann::json ingestLocked(const nlohmann::json &payload, bool allowPath);
    nlohmann::json observeLocked(const std::string &role, const std::string &text);
    nlohmann::json tickLocked(const nlohmann::json &ctx);
    nlohmann::json mirrorLocked();
    nlohmann::json turingLocked();
    nlohmann::json statusLocked() const;
    nlohmann::json toJsonLocked() const;
    void fromJsonLocked(const nlohmann::json &j);
    std::string kernelLocked() const;
    double scoreLocked(const Feat &f) const;
    std::string renderAct(int act) const;
    bool recognizeLocked(const std::string &text) const;
    void policyPhi(float phi[kPolicyDim]) const;
    void updatePolicy(const float phi[kPolicyDim]);
    int chooseAct(const float phi[kPolicyDim]) const;
    void ema(Feat &dst, const Feat &x, int &count);
    static Feat featurize(const std::string &text);
    static float cosine(const Feat &a, const Feat &b);
    static std::string utf8Prefix(const std::string &s, size_t maxChars);

    mutable std::mutex mu_;
    bool enabled_{true};
    std::string name_{"self"};
    float alpha_{0.35f};
    int holdoutEvery_{4};
    bool learnFromLiveTurns_{false};
    std::string corpusPath_;
    std::string loadedCorpusPath_;
    std::string lastError_;

    Feat human_{};
    Feat chatbot_{};
    Feat self_{};
    Feat foreign_{};
    int humanCount_{0};
    int chatbotCount_{0};
    int selfCount_{0};
    int foreignCount_{0};
    int humanOrdinal_{0};

    float w_[kActs][kPolicyDim]{};
    int episodes_{0};
    int spontaneousCount_{0};

    std::vector<std::string> exemplars_;
    std::vector<std::string> holdout_;
    std::vector<std::string> thoughts_;
    std::vector<std::string> thoughtActs_;
    std::vector<std::string> foreignLines_;
    std::string lastThought_;
    std::string lastAction_;
    std::string marked_;
    std::string lastPartner_;
};

}  // namespace selfmind
}  // namespace phoenix
