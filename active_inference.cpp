/* active_inference.cpp - Active inference / MPC implementation
   Copyright (C) 2026 079 Project */

#include "active_inference.hpp"

#include <algorithm>
#include <cmath>

namespace phoenix {
namespace agi {

nlohmann::json ExpectedFreeEnergy::toJson() const {
  return {{"pragmatic", pragmatic}, {"intrinsic", intrinsic},
          {"epistemic", epistemic}, {"total", total()}};
}

LatentTransitionModel::LatentTransitionModel(size_t dim, size_t actionDim)
    : dim_(dim), actionDim_(actionDim),
      A_(dim * dim, 0.0f), B_(actionDim ? dim * actionDim : 0, 0.0f),
      b_(dim, 0.0f) {
  // Identity dynamics as a safe default: z_{t+1} = z_t (no-op predictor).
  for (size_t i = 0; i < dim; ++i) A_[i * dim + i] = 1.0f;
}

void LatentTransitionModel::reset(size_t dim, size_t actionDim) {
  dim_ = dim;
  actionDim_ = actionDim;
  A_.assign(dim * dim, 0.0f);
  B_.assign(actionDim ? dim * actionDim : 0, 0.0f);
  b_.assign(dim, 0.0f);
  for (size_t i = 0; i < dim; ++i) A_[i * dim + i] = 1.0f;
}

std::vector<float> LatentTransitionModel::predict(
    const std::vector<float> &z, const std::vector<float> &a) const {
  std::vector<float> out(dim_, 0.0f);
  for (size_t i = 0; i < dim_; ++i) {
    float acc = b_[i];
    const size_t row = i * dim_;
    for (size_t j = 0; j < dim_ && j < z.size(); ++j) {
      acc += A_[row + j] * z[j];
    }
    if (!a.empty()) {
      const size_t brow = i * actionDim_;
      for (size_t j = 0; j < actionDim_ && j < a.size(); ++j) {
        acc += B_[brow + j] * a[j];
      }
    }
    out[i] = acc;
  }
  return out;
}

double LatentTransitionModel::surprise(const std::vector<float> &observed,
                                       const std::vector<float> &predicted) const {
  double s = 0.0;
  const size_t n = std::min(observed.size(), predicted.size());
  for (size_t i = 0; i < n; ++i) {
    const double e = static_cast<double>(observed[i]) - static_cast<double>(predicted[i]);
    s += e * e;
  }
  return 0.5 * s;  // -ln N(e; 0, I) up to constants.
}

void LatentTransitionModel::update(const std::vector<float> &z,
                                   const std::vector<float> &a,
                                   const std::vector<float> &zNext, float lr) {
  const std::vector<float> pred = predict(z, a);
  std::vector<float> err(dim_, 0.0f);
  for (size_t i = 0; i < dim_; ++i) {
    err[i] = (i < zNext.size() ? zNext[i] : 0.0f) - pred[i];
  }
  for (size_t i = 0; i < dim_; ++i) {
    const size_t row = i * dim_;
    for (size_t j = 0; j < dim_ && j < z.size(); ++j) {
      A_[row + j] += lr * err[i] * z[j];
    }
    b_[i] += lr * err[i];
  }
  if (!a.empty()) {
    for (size_t i = 0; i < dim_; ++i) {
      const size_t brow = i * actionDim_;
      for (size_t j = 0; j < actionDim_ && j < a.size(); ++j) {
        B_[brow + j] += lr * err[i] * a[j];
      }
    }
  }
}

nlohmann::json LatentTransitionModel::status() const {
  return {{"dim", dim_}, {"actionDim", actionDim_},
          {"model", "linear-latent-transition"}};
}

ActiveInferenceController::ActiveInferenceController(size_t dim, size_t actionDim,
                                                     size_t horizon)
    : model_(dim, actionDim), horizon_(horizon), w_(dim, 0.0f) {}

void ActiveInferenceController::configure(size_t dim, size_t actionDim, size_t horizon) {
  model_.reset(dim, actionDim);
  horizon_ = horizon ? horizon : 1;
  w_.assign(dim, 0.0f);
  prefsBootstrapped_ = false;
  surpriseEma_ = 0.0;
  lastSurprise_ = 0.0;
}

void ActiveInferenceController::setPreferences(const std::vector<float> &w) { w_ = w; }

void ActiveInferenceController::bootstrapPreferences(const std::vector<float> &w) {
  if (!prefsBootstrapped_) {
    w_ = w;
    prefsBootstrapped_ = true;
  }
}

void ActiveInferenceController::setActions(std::vector<Action> actions) {
  actions_ = std::move(actions);
}

void ActiveInferenceController::setHorizon(size_t h) { horizon_ = h ? h : 1; }

double ActiveInferenceController::utility(const std::vector<float> &z) const {
  double u = 0.0;
  const size_t n = std::min(z.size(), w_.size());
  for (size_t i = 0; i < n; ++i) u += static_cast<double>(w_[i]) * z[i];
  // Risk aversion: prospect-theory curvature u' = sign(u)·|u|^gamma.
  if (riskAversion_ != 1.0f) {
    u = (u >= 0.0 ? 1.0 : -1.0) *
        std::pow(std::abs(u), static_cast<double>(riskAversion_));
  }
  return u;
}

ExpectedFreeEnergy ActiveInferenceController::evaluate(
    const std::vector<float> &z, const Action &action, double driveCost,
    double pragW, double intrinW, double epistW) const {
  ExpectedFreeEnergy efe;
  std::vector<float> zt = z;
  for (size_t t = 0; t < horizon_; ++t) {
    std::vector<float> zNext = model_.predict(zt, action.embedding);
    const double u = utility(zNext);
    // Epistemic proxy during imagination: the expected latent-state change
    // magnitude biases exploration toward novel territory.  The true prediction
    // error (surprise) is computed against real transitions in observe().
    const double surpr = model_.surprise(zNext, zt);
    efe.pragmatic -= pragW * u;
    efe.epistemic -= epistW * surpr;
    zt = std::move(zNext);
  }
  efe.intrinsic = intrinW * driveCost * static_cast<double>(horizon_);
  return efe;
}

ActiveInferenceController::Plan ActiveInferenceController::plan(
    const std::vector<float> &z, double driveCost,
    double pragW, double intrinW, double epistW) const {
  Plan p;
  for (size_t i = 0; i < actions_.size(); ++i) {
    const ExpectedFreeEnergy efe =
        evaluate(z, actions_[i], driveCost, pragW, intrinW, epistW);
    const double c = efe.total();
    if (c < p.bestCost) {
      p.bestCost = c;
      p.bestAction = static_cast<int>(i);
      p.efe = efe;
    }
  }
  return p;
}

double ActiveInferenceController::observe(const std::vector<float> &z,
                                          const std::vector<float> &a,
                                          const std::vector<float> &zNext, float lr) {
  const std::vector<float> pred = model_.predict(z, a);
  const double surpr = model_.surprise(zNext, pred);
  model_.update(z, a, zNext, lr);
  lastSurprise_ = surpr;
  surpriseEma_ = surpriseEma_ <= 0.0 ? surpr : 0.9 * surpriseEma_ + 0.1 * surpr;
  Episodes_.push_back({z, a, zNext, static_cast<float>(utility(zNext)),
                       static_cast<float>(surpr), 0.0f});
  if (Episodes_.size() > kMaxEpisodes_) Episodes_.erase(Episodes_.begin());
  return surpr;
}

double ActiveInferenceController::observeRewarded(
    const std::vector<float> &z, const std::vector<float> &a,
    const std::vector<float> &zNext, double reward,
    float lr, double alpha, double gamma) {
  const std::vector<float> pred = model_.predict(z, a);
  const double surpr = model_.surprise(zNext, pred);
  model_.update(z, a, zNext, lr);
  // Self-evolution: the realised benefit-harm netUtility is the reward that
  // trains the value head via TD(0).
  learnValue(z, zNext, reward, alpha, gamma);
  lastSurprise_ = surpr;
  surpriseEma_ = surpriseEma_ <= 0.0 ? surpr : 0.9 * surpriseEma_ + 0.1 * surpr;
  Episodes_.push_back({z, a, zNext, static_cast<float>(utility(zNext)),
                       static_cast<float>(surpr), static_cast<float>(reward)});
  if (Episodes_.size() > kMaxEpisodes_) Episodes_.erase(Episodes_.begin());
  return surpr;
}

double ActiveInferenceController::learnValue(const std::vector<float> &z,
                                             const std::vector<float> &zNext,
                                             double reward, double alpha,
                                             double gamma) {
  if (z.empty() || w_.empty()) return 0.0;
  const double v = utility(z);
  const double vNext = utility(zNext);
  const double delta = reward + gamma * vNext - v;  // TD(0) error.
  const size_t n = std::min(z.size(), w_.size());
  for (size_t i = 0; i < n; ++i) {
    w_[i] = static_cast<float>(
        std::clamp(static_cast<double>(w_[i]) + alpha * delta * z[i],
                   -valueClamp_, valueClamp_));
  }
  return delta;
}

double ActiveInferenceController::consolidate(size_t maxReplays, double alpha,
                                              double gamma) {
  if (Episodes_.empty()) return 0.0;
  const size_t n = std::min(maxReplays, Episodes_.size());
  const size_t start = Episodes_.size() - n;
  double totalSurprise = 0.0;
  for (size_t i = start; i < Episodes_.size(); ++i) {
    const auto &ep = Episodes_[i];
    model_.update(ep.z, ep.a, ep.zNext, 0.005f);
    totalSurprise += model_.surprise(ep.zNext, model_.predict(ep.z, ep.a));
    learnValue(ep.z, ep.zNext, static_cast<double>(ep.reward), alpha, gamma);
  }
  return totalSurprise / static_cast<double>(n);
}

double ActiveInferenceController::explorationMultiplier() const {
  if (surpriseEma_ <= 1e-6) return 1.0;
  // Recent surprise below its EMA -> environment getting predictable -> the
  // epistemic term should be amplified to explore; above -> exploit.
  const double ratio = (surpriseEma_ - lastSurprise_) / surpriseEma_;
  return std::clamp(1.0 + ratio, 0.5, 2.0);
}

nlohmann::json ActiveInferenceController::status() const {
  return {{"horizon", horizon_}, {"actions", actions_.size()},
          {"episodes", Episodes_.size()},
          {"surpriseEma", surpriseEma_},
          {"lastSurprise", lastSurprise_},
          {"explorationMultiplier", explorationMultiplier()},
          {"prefsBootstrapped", prefsBootstrapped_},
          {"valueClamp", valueClamp_},
          {"model", model_.status()}};
}

}  // namespace agi
}  // namespace phoenix
