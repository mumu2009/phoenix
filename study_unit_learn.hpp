/* study_unit_learn.hpp - Deepened GNN learning on units.

   One BPE token is one unit. Episodes enter as unit rows and meme edges.
   Nothing here builds a chat causal string, and nothing here writes a
   self-mind thought. Recall returns unit rows for a side channel.
*/
#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace phoenix {
namespace study {

/** Small byte-pair encoder. Token id 0..255 are raw bytes; later ids are merges. */
class BytePairEncoder {
public:
    void fit(const std::string &text, int maxMerges);
    std::vector<int> encode(const std::string &text) const;
    int mergeCount() const { return static_cast<int>(merges_.size()); }
    const std::string &surface(int id) const;

    nlohmann::json toJson() const;
    void fromJson(const nlohmann::json &j);

private:
    struct Merge {
        int left{0};
        int right{0};
        int id{0};
    };
    std::vector<Merge> merges_;
    std::vector<std::string> surfaces_;
};

/** Full-process unit graph: encode, meme import, window edges, K-step
    propagation, then a linear next-unit residual on fixed meme vectors. */
class UnitLearner {
public:
    UnitLearner() = default;

    nlohmann::json configure(const nlohmann::json &payload);
    nlohmann::json fitEncoder(const std::string &text);
    std::vector<int> encode(const std::string &text) const;
    nlohmann::json learnEpisode(const std::string &text);
    nlohmann::json importUnits(const nlohmann::json &payload);
    nlohmann::json recall(const nlohmann::json &payload) const;
    nlohmann::json status() const;
    /** Stats only. Drops every text field so a caller cannot splice it
        into the chat causal string. */
    nlohmann::json sideChannel(const nlohmann::json &learned) const;
    std::vector<float> unitRow(int id) const;
    bool enabled() const;

    nlohmann::json toJson() const;
    void fromJson(const nlohmann::json &j);

private:
    struct Node {
        std::vector<float> base;
        std::vector<float> state;
        int count{0};
    };

    nlohmann::json configureLocked(const nlohmann::json &payload);
    nlohmann::json learnLocked(const std::string &text);
    nlohmann::json importLocked(const nlohmann::json &payload);
    nlohmann::json recallLocked(const nlohmann::json &payload) const;
    nlohmann::json statusLocked() const;
    nlohmann::json toJsonLocked() const;
    void fromJsonLocked(const nlohmann::json &j);
    void ensureNode(int id);
    void initBase(std::vector<float> &dst, int id) const;
    void writeBase(int id, const std::vector<float> &row);
    void addEdge(int from, int to, float w);
    void propagate();
    std::vector<float> applyW(const std::vector<float> &h) const;
    static std::vector<float> normalize(std::vector<float> v);
    static std::vector<float> fitRow(const std::vector<float> &row, int dim);

    mutable std::mutex mu_;
    bool enabled_{false};
    int dim_{16};
    int steps_{3};
    int maxMerges_{32};
    float lr_{0.35f};
    float alpha_{0.45f};
    int maxNodes_{4096};
    int maxEdges_{8192};
    BytePairEncoder encoder_;
    bool encoderReady_{false};
    std::unordered_map<int, Node> nodes_;
    std::unordered_map<std::uint64_t, float> edges_;
    std::vector<float> W_;
    double lastLoss_{-1.0};
    int episodes_{0};
};

}  // namespace study
}  // namespace phoenix
