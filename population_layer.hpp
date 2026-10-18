/* population_layer.hpp - Capped population above the multi-instance layer.

   One individual is one multi-instance layer, represented by a small
   parameter vector. This is not session replication and not the sealed
   in-process (1+lambda) contest. Default off. Each goal may nudge n-1
   individuals, score the forward output, and keep the best half inside
   a fixed population.
*/
#pragma once

#include <nlohmann/json.hpp>

#include <mutex>
#include <string>
#include <vector>

namespace phoenix {
namespace study {

inline int populationParamCount(int inputDim, int hidden, int outputDim) {
    return hidden * inputDim + hidden + outputDim * hidden + outputDim;
}

/** Lay a perfect readout: hidden preactivation is 0 so y equals the bias. */
inline std::vector<float> populationPerfectParams(int inputDim, int hidden, int outputDim,
                                                 const std::vector<float> &target) {
    std::vector<float> p(static_cast<size_t>(populationParamCount(inputDim, hidden, outputDim)), 0.f);
    const int b2 = hidden * inputDim + hidden + outputDim * hidden;
    for (int k = 0; k < outputDim && k < static_cast<int>(target.size()); ++k) {
        p[static_cast<size_t>(b2 + k)] = target[static_cast<size_t>(k)];
    }
    return p;
}

class PopulationLayer {
public:
    static constexpr int kHardCap = 16;

    PopulationLayer() = default;

    nlohmann::json configure(const nlohmann::json &payload);
    nlohmann::json runGoal(const nlohmann::json &goal);
    nlohmann::json setIndividual(const nlohmann::json &payload);
    nlohmann::json status() const;
    std::vector<float> forwardIndex(int index, const std::vector<float> &input) const;
    bool enabled() const;

    nlohmann::json toJson() const;
    void fromJson(const nlohmann::json &j);

private:
    struct Individual {
        std::string id;
        std::vector<float> params;
        float lastError{1.0e9f};
        bool lastCorrect{false};
        bool scored{false};
    };

    nlohmann::json configureLocked(const nlohmann::json &payload);
    nlohmann::json runGoalLocked(const nlohmann::json &goal);
    nlohmann::json statusLocked() const;
    void reseedLocked(int n);
    std::vector<float> forwardParams(const std::vector<float> &params,
                                     const std::vector<float> &input) const;
    static float errorOf(const std::vector<float> &y, const std::vector<float> &target);

    mutable std::mutex mu_;
    bool enabled_{false};
    int n_{0};
    int maxPopulation_{8};
    int inputDim_{4};
    int hidden_{4};
    int outputDim_{2};
    float mutationScale_{0.05f};
    float tolerance_{0.2f};
    unsigned seed_{1};
    unsigned rng_{1};
    int goalsRun_{0};
    int keptLast_{0};
    int mutatedLast_{0};
    std::vector<Individual> pop_;
};

}  // namespace study
}  // namespace phoenix
