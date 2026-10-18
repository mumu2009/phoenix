/* tiny_expert.hpp - Goal-local model small enough to train on this host.

   A single hidden layer. It never loads an 8B checkpoint. Summon can
   register the saved weights as kind "tiny-expert" with no server URL.
*/
#pragma once

#include <nlohmann/json.hpp>

#include <mutex>
#include <string>
#include <vector>

namespace phoenix {
namespace study {

class TinyExpert {
public:
    TinyExpert() = default;

    nlohmann::json configure(const nlohmann::json &payload);
    nlohmann::json trainForGoal(const nlohmann::json &payload);
    nlohmann::json predict(const nlohmann::json &payload) const;
    nlohmann::json save(const std::string &path) const;
    nlohmann::json load(const std::string &path);
    bool registerSummon(const std::string &name) const;
    nlohmann::json status() const;
    bool enabled() const;
    int paramCount() const;

    nlohmann::json toJson() const;
    void fromJson(const nlohmann::json &j);

private:
    struct Sample {
        std::vector<float> x;
        int y{0};
    };

    nlohmann::json configureLocked(const nlohmann::json &payload);
    void initWeightsLocked(unsigned seed);
    double epochLocked(std::vector<Sample> &data, float lr, unsigned seed);
    std::vector<float> forwardLocked(const std::vector<float> &x, std::vector<float> *hidden) const;
    nlohmann::json predictLocked(const std::vector<float> &x) const;

    mutable std::mutex mu_;
    bool enabled_{false};
    int vocab_{32};
    int hidden_{16};
    int classes_{32};
    float lr_{0.15f};
    int epochs_{8};
    std::string goal_;
    std::string source_;
    std::string license_;
    std::string weightsPath_;
    std::string corpusPath_;
    std::vector<float> W1_;
    std::vector<float> b1_;
    std::vector<float> W2_;
    std::vector<float> b2_;
    bool ready_{false};
    double lastLoss_{-1.0};
};

}  // namespace study
}  // namespace phoenix
