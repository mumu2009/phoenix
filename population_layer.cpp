/* population_layer.cpp - Fixed-size population, slight mutation, keep best half. */

#include "population_layer.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace phoenix {
namespace study {

namespace {

float clampf(float v, float lo, float hi) { return std::max(lo, std::min(hi, v)); }

}  // namespace

nlohmann::json PopulationLayer::configure(const nlohmann::json &payload) {
    std::lock_guard<std::mutex> lock(mu_);
    return configureLocked(payload);
}

nlohmann::json PopulationLayer::configureLocked(const nlohmann::json &payload) {
    const nlohmann::json &p = payload.is_object() ? payload : nlohmann::json::object();
    if (p.contains("enabled")) enabled_ = p.value("enabled", false);
    if (p.contains("mutationScale"))
        mutationScale_ = clampf(p.value("mutationScale", mutationScale_), 0.f, 1.f);
    if (p.contains("tolerance"))
        tolerance_ = clampf(p.value("tolerance", tolerance_), 1.0e-4f, 10.f);
    if (p.contains("seed")) {
        seed_ = static_cast<unsigned>(std::max(1, p.value("seed", 1)));
        rng_ = seed_;
    }
    if (p.contains("maxPopulation")) {
        maxPopulation_ = std::max(2, std::min(kHardCap, p.value("maxPopulation", maxPopulation_)));
    }
    bool reseed = false;
    if (p.contains("inputDim") || p.contains("hidden") || p.contains("outputDim")) {
        inputDim_ = std::max(1, std::min(16, p.value("inputDim", inputDim_)));
        hidden_ = std::max(1, std::min(16, p.value("hidden", hidden_)));
        outputDim_ = std::max(1, std::min(8, p.value("outputDim", outputDim_)));
        reseed = true;
    }
    if (p.contains("n")) {
        const int requested = p.value("n", n_);
        if (requested < 2) {
            return nlohmann::json{{"ok", false},
                                  {"reason", "n-below-2"},
                                  {"n", n_},
                                  {"maxPopulation", maxPopulation_},
                                  {"enabled", enabled_}};
        }
        n_ = std::min(requested, maxPopulation_);
        reseed = true;
    } else if (reseed && n_ >= 2) {
        n_ = std::min(n_, maxPopulation_);
    } else if (p.value("enabled", false) && n_ < 2 && !p.contains("n")) {
        n_ = std::min(4, maxPopulation_);
        reseed = true;
    }
    if (reseed) reseedLocked(n_);
    auto st = statusLocked();
    st["ok"] = true;
    return st;
}

void PopulationLayer::reseedLocked(int n) {
    n = std::max(0, std::min(n, maxPopulation_));
    n_ = n;
    pop_.clear();
    pop_.reserve(static_cast<size_t>(n_));
    std::mt19937 rng(seed_);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    const int pc = populationParamCount(inputDim_, hidden_, outputDim_);
    for (int i = 0; i < n_; ++i) {
        Individual ind;
        ind.id = "ind-" + std::to_string(i);
        ind.params.resize(static_cast<size_t>(pc));
        for (float &v : ind.params) v = dist(rng);
        pop_.push_back(std::move(ind));
    }
    rng_ = rng();
}

nlohmann::json PopulationLayer::setIndividual(const nlohmann::json &payload) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!payload.is_object()) return nlohmann::json{{"ok", false}, {"reason", "payload"}};
    const int index = payload.value("index", -1);
    if (index < 0 || index >= static_cast<int>(pop_.size())) {
        return nlohmann::json{{"ok", false}, {"reason", "index"}};
    }
    if (!payload.contains("params") || !payload["params"].is_array()) {
        return nlohmann::json{{"ok", false}, {"reason", "params"}};
    }
    const int pc = populationParamCount(inputDim_, hidden_, outputDim_);
    if (static_cast<int>(payload["params"].size()) != pc) {
        return nlohmann::json{{"ok", false}, {"reason", "param-count"}, {"expected", pc}};
    }
    Individual &ind = pop_[static_cast<size_t>(index)];
    for (int i = 0; i < pc; ++i) ind.params[static_cast<size_t>(i)] = payload["params"][i].get<float>();
    if (payload.contains("lastError")) {
        ind.lastError = payload.value("lastError", ind.lastError);
        ind.scored = true;
    }
    return nlohmann::json{{"ok", true}, {"id", ind.id}};
}

nlohmann::json PopulationLayer::runGoal(const nlohmann::json &goal) {
    std::lock_guard<std::mutex> lock(mu_);
    return runGoalLocked(goal);
}

nlohmann::json PopulationLayer::runGoalLocked(const nlohmann::json &goal) {
    if (!enabled_) {
        return nlohmann::json{{"ok", true},
                              {"enabled", false},
                              {"ran", false},
                              {"mutated", 0},
                              {"spawnedInstances", 0},
                              {"reason", "disabled"}};
    }
    if (n_ < 2 || static_cast<int>(pop_.size()) < 2) {
        return nlohmann::json{{"ok", false}, {"reason", "n-below-2"}, {"ran", false}};
    }
    if (!goal.is_object() || !goal.contains("input") || !goal.contains("target") ||
        !goal["input"].is_array() || !goal["target"].is_array()) {
        return nlohmann::json{{"ok", false}, {"reason", "goal-shape"}, {"ran", false}};
    }
    std::vector<float> input;
    std::vector<float> target;
    for (const auto &v : goal["input"])
        if (v.is_number()) input.push_back(v.get<float>());
    for (const auto &v : goal["target"])
        if (v.is_number()) target.push_back(v.get<float>());
    if (static_cast<int>(input.size()) != inputDim_ || static_cast<int>(target.size()) != outputDim_) {
        return nlohmann::json{{"ok", false}, {"reason", "goal-dim"}, {"ran", false}};
    }

    int anchor = 0;
    float bestBefore = pop_[0].lastError;
    for (int i = 1; i < n_; ++i) {
        if (pop_[static_cast<size_t>(i)].lastError < bestBefore) {
            bestBefore = pop_[static_cast<size_t>(i)].lastError;
            anchor = i;
        }
    }
    const std::vector<float> anchorBefore = pop_[static_cast<size_t>(anchor)].params;
    const std::string anchorId = pop_[static_cast<size_t>(anchor)].id;

    std::mt19937 rng(rng_ ? rng_ : seed_);
    std::normal_distribution<float> noise(0.f, std::max(mutationScale_, 1.0e-4f));
    int mutated = 0;
    for (int i = 0; i < n_; ++i) {
        if (i == anchor) continue;
        Individual &ind = pop_[static_cast<size_t>(i)];
        bool changed = false;
        for (float &v : ind.params) {
            const float before = v;
            v = clampf(v + noise(rng), -2.f, 2.f);
            if (v != before) changed = true;
        }
        if (!changed && !ind.params.empty()) {
            ind.params[0] = clampf(ind.params[0] + std::max(mutationScale_, 1.0e-3f), -2.f, 2.f);
            changed = true;
        }
        if (changed) ++mutated;
    }
    rng_ = rng();
    const bool anchorSame = pop_[static_cast<size_t>(anchor)].params == anchorBefore;

    struct Score {
        int index;
        float error;
    };
    std::vector<Score> scores;
    scores.reserve(static_cast<size_t>(n_));
    int correctCount = 0;
    for (int i = 0; i < n_; ++i) {
        const auto y = forwardParams(pop_[static_cast<size_t>(i)].params, input);
        const float err = errorOf(y, target);
        const bool ok = err <= tolerance_;
        pop_[static_cast<size_t>(i)].lastError = err;
        pop_[static_cast<size_t>(i)].lastCorrect = ok;
        pop_[static_cast<size_t>(i)].scored = true;
        if (ok) ++correctCount;
        scores.push_back(Score{i, err});
    }
    std::sort(scores.begin(), scores.end(), [&](const Score &a, const Score &b) {
        if (a.error != b.error) return a.error < b.error;
        if (a.index == anchor) return true;
        if (b.index == anchor) return false;
        return a.index < b.index;
    });
    const int keep = std::max(1, n_ / 2);
    std::vector<Individual> survivors;
    survivors.reserve(static_cast<size_t>(keep));
    for (int i = 0; i < keep; ++i)
        survivors.push_back(pop_[static_cast<size_t>(scores[static_cast<size_t>(i)].index)]);
    std::vector<Individual> next;
    next.reserve(static_cast<size_t>(n_));
    for (int i = 0; i < keep; ++i) next.push_back(survivors[static_cast<size_t>(i)]);
    for (int slot = keep; slot < n_; ++slot) {
        Individual copy = survivors[static_cast<size_t>((slot - keep) % keep)];
        copy.id = survivors[static_cast<size_t>((slot - keep) % keep)].id + "~" + std::to_string(slot);
        next.push_back(std::move(copy));
    }
    pop_.swap(next);
    n_ = static_cast<int>(pop_.size());
    ++goalsRun_;
    keptLast_ = keep;
    mutatedLast_ = mutated;

    float bestAfter = pop_.empty() ? 1.0e9f : pop_[0].lastError;
    for (const auto &ind : pop_) bestAfter = std::min(bestAfter, ind.lastError);
    bool anchorKept = false;
    for (int i = 0; i < keep; ++i) {
        if (survivors[static_cast<size_t>(i)].id == anchorId) anchorKept = true;
    }
    return nlohmann::json{{"ok", true},
                          {"enabled", true},
                          {"ran", true},
                          {"n", n_},
                          {"mutated", mutated},
                          {"kept", keep},
                          {"keepFraction", static_cast<double>(keep) / static_cast<double>(n_)},
                          {"anchor", anchorId},
                          {"anchorUnchanged", anchorSame},
                          {"anchorKept", anchorKept},
                          {"bestError", bestAfter},
                          {"correctCount", correctCount},
                          {"populationSize", n_},
                          {"maxPopulation", maxPopulation_},
                          {"spawnedInstances", 0},
                          {"replacedSlots", n_ - keep}};
}

std::vector<float> PopulationLayer::forwardIndex(int index, const std::vector<float> &input) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (index < 0 || index >= static_cast<int>(pop_.size())) return {};
    return forwardParams(pop_[static_cast<size_t>(index)].params, input);
}

std::vector<float> PopulationLayer::forwardParams(const std::vector<float> &params,
                                                 const std::vector<float> &input) const {
    std::vector<float> y(static_cast<size_t>(outputDim_), 0.f);
    const int need = populationParamCount(inputDim_, hidden_, outputDim_);
    if (static_cast<int>(params.size()) != need || static_cast<int>(input.size()) != inputDim_) return y;
    std::vector<float> h(static_cast<size_t>(hidden_), 0.f);
    const int b1 = hidden_ * inputDim_;
    for (int j = 0; j < hidden_; ++j) {
        float s = params[static_cast<size_t>(b1 + j)];
        for (int i = 0; i < inputDim_; ++i)
            s += params[static_cast<size_t>(j * inputDim_ + i)] * input[static_cast<size_t>(i)];
        h[static_cast<size_t>(j)] = std::tanh(s);
    }
    const int w2 = hidden_ * inputDim_ + hidden_;
    const int b2 = w2 + outputDim_ * hidden_;
    for (int k = 0; k < outputDim_; ++k) {
        float s = params[static_cast<size_t>(b2 + k)];
        for (int j = 0; j < hidden_; ++j)
            s += params[static_cast<size_t>(w2 + k * hidden_ + j)] * h[static_cast<size_t>(j)];
        y[static_cast<size_t>(k)] = s;
    }
    return y;
}

float PopulationLayer::errorOf(const std::vector<float> &y, const std::vector<float> &target) {
    if (y.size() != target.size() || y.empty()) return 1.0e9f;
    float s = 0.f;
    for (size_t i = 0; i < y.size(); ++i) s += std::fabs(y[i] - target[i]);
    return s / static_cast<float>(y.size());
}

nlohmann::json PopulationLayer::status() const {
    std::lock_guard<std::mutex> lock(mu_);
    return statusLocked();
}

nlohmann::json PopulationLayer::statusLocked() const {
    nlohmann::json inds = nlohmann::json::array();
    for (const auto &ind : pop_) {
        nlohmann::json params = nlohmann::json::array();
        for (float v : ind.params) params.push_back(v);
        inds.push_back(nlohmann::json{{"id", ind.id},
                                      {"lastError", ind.lastError},
                                      {"lastCorrect", ind.lastCorrect},
                                      {"params", params}});
    }
    float best = 1.0e9f;
    for (const auto &ind : pop_) best = std::min(best, ind.lastError);
    return nlohmann::json{{"enabled", enabled_},
                          {"n", n_},
                          {"maxPopulation", maxPopulation_},
                          {"hardCap", kHardCap},
                          {"inputDim", inputDim_},
                          {"hidden", hidden_},
                          {"outputDim", outputDim_},
                          {"mutationScale", mutationScale_},
                          {"tolerance", tolerance_},
                          {"goalsRun", goalsRun_},
                          {"keptLast", keptLast_},
                          {"mutatedLast", mutatedLast_},
                          {"bestError", pop_.empty() ? 1.0e9 : best},
                          {"spawnedInstances", 0},
                          {"individuals", inds}};
}

bool PopulationLayer::enabled() const {
    std::lock_guard<std::mutex> lock(mu_);
    return enabled_;
}

nlohmann::json PopulationLayer::toJson() const {
    std::lock_guard<std::mutex> lock(mu_);
    auto j = statusLocked();
    j["seed"] = seed_;
    j["rng"] = rng_;
    return j;
}

void PopulationLayer::fromJson(const nlohmann::json &j) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!j.is_object()) return;
    enabled_ = j.value("enabled", false);
    maxPopulation_ = std::max(2, std::min(kHardCap, j.value("maxPopulation", maxPopulation_)));
    inputDim_ = j.value("inputDim", inputDim_);
    hidden_ = j.value("hidden", hidden_);
    outputDim_ = j.value("outputDim", outputDim_);
    mutationScale_ = j.value("mutationScale", mutationScale_);
    tolerance_ = j.value("tolerance", tolerance_);
    seed_ = j.value("seed", seed_);
    rng_ = j.value("rng", rng_);
    goalsRun_ = j.value("goalsRun", 0);
    keptLast_ = j.value("keptLast", 0);
    mutatedLast_ = j.value("mutatedLast", 0);
    pop_.clear();
    if (j.contains("individuals") && j["individuals"].is_array()) {
        for (const auto &item : j["individuals"]) {
            if (!item.is_object()) continue;
            if (static_cast<int>(pop_.size()) >= maxPopulation_) break;
            Individual ind;
            ind.id = item.value("id", std::string());
            ind.lastError = item.value("lastError", 1.0e9f);
            ind.lastCorrect = item.value("lastCorrect", false);
            ind.scored = true;
            if (item.contains("params") && item["params"].is_array()) {
                for (const auto &v : item["params"])
                    if (v.is_number()) ind.params.push_back(v.get<float>());
            }
            pop_.push_back(std::move(ind));
        }
    }
    n_ = static_cast<int>(pop_.size());
}

}  // namespace study
}  // namespace phoenix
