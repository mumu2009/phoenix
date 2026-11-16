/* test_active_inference_protocol.cpp - Self-evolution ship-gate protocol
   Implements testing_methodology.md §2.1/§2.3/§2.4 as gtest invariants. */

#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "active_inference.hpp"

using phoenix::agi::ActiveInferenceController;
using phoenix::agi::Action;

namespace {

constexpr size_t kDim = 8;
constexpr size_t kRounds = 3000;
constexpr double kLr = 0.01;
constexpr double kAlpha = 0.05;
constexpr double kGamma = 0.9;

// Latent process.  For the learning-curve protocol we use a nearly-white-noise
// AR(1) (ar = 0.0) so the forward model can learn an identity/diagonal-like
// transition and the value head can converge to the true linear value.
// For the explore/exploit protocol we also switch ar mid-run.
struct Process {
  std::vector<float> z;
  std::vector<float> wStar;
  float ar = 0.0f;
  float meanScale = 0.0f;
  float noiseStd = 0.1f;
  std::mt19937 rng;

  Process(size_t dim, uint32_t seed, const std::vector<float> &target)
      : z(dim, 0.0f), wStar(target), rng(seed) {
    std::normal_distribution<float> dist(0.0f, noiseStd);
    for (size_t i = 0; i < dim; ++i) z[i] = dist(rng);
  }

  void step() {
    std::normal_distribution<float> dist(0.0f, noiseStd);
    for (size_t i = 0; i < z.size(); ++i) {
      const float mean_i = meanScale * wStar[i];
      z[i] = ar * z[i] + (1.0f - ar) * mean_i + dist(rng);
    }
  }

  double reward() const {
    double r = 0.0;
    for (size_t i = 0; i < z.size(); ++i) r += static_cast<double>(wStar[i]) * z[i];
    return r;
  }
};

std::vector<float> randomUnitVector(size_t dim, std::mt19937 &rng) {
  std::normal_distribution<float> d(0.0f, 1.0f);
  std::vector<float> v(dim);
  double sq = 0.0;
  for (size_t i = 0; i < dim; ++i) {
    v[i] = d(rng);
    sq += v[i] * v[i];
  }
  const double norm = std::sqrt(sq);
  for (size_t i = 0; i < dim; ++i) v[i] = static_cast<float>(v[i] / norm);
  return v;
}

double cosine(const std::vector<float> &a, const std::vector<float> &b) {
  double dot = 0.0, na = 0.0, nb = 0.0;
  const size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; ++i) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na == 0.0 || nb == 0.0) return 0.0;
  return dot / (std::sqrt(na) * std::sqrt(nb));
}

struct Metrics {
  double surpriseFirstHalf = 0.0;
  double surpriseSecondHalf = 0.0;
  double tdFirstHalf = 0.0;
  double tdSecondHalf = 0.0;
  double cos = 0.0;
  double cumulativeUtility = 0.0;
  double cumulativeAbsUtility = 0.0;
};

Metrics runLearningCurve(uint32_t seed) {
  std::mt19937 seedRng(seed);
  const std::vector<float> wStar = randomUnitVector(kDim, seedRng);

  ActiveInferenceController ctrl(kDim, /*actionDim=*/0, /*horizon=*/1);
  ctrl.setValueClamp(10.0);
  // No bootstrap: the value head must evolve from zero (self-evolution).
  (void)ctrl.preferences();  // w_ is already zero-initialised.

  Process env(kDim, seed + 1000, wStar);

  std::vector<float> z = env.z;
  std::vector<double> surpriseLog;
  std::vector<double> tdLog;
  std::vector<std::vector<float>> trajectory;

  for (size_t t = 0; t < kRounds; ++t) {
    trajectory.push_back(z);
    const double r = env.reward();  // reward associated with current state z.
    env.step();
    const std::vector<float> zNext = env.z;

    // TD error computed with the current w (before observeRewarded updates it).
    const double v = ctrl.utility(z);
    const double vNext = ctrl.utility(zNext);
    const double td = std::abs(r + kGamma * vNext - v);
    tdLog.push_back(td);

    // Observe and update value + forward model.
    ctrl.observeRewarded(z, {}, zNext, r, kLr, kAlpha, kGamma);
    surpriseLog.push_back(ctrl.surpriseEma());
    z = zNext;
  }

  Metrics m;
  const size_t half = kRounds / 2;
  for (size_t t = 0; t < half; ++t) {
    m.surpriseFirstHalf += surpriseLog[t];
    m.tdFirstHalf += tdLog[t];
  }
  m.surpriseFirstHalf /= half;
  m.tdFirstHalf /= half;
  for (size_t t = half; t < kRounds; ++t) {
    m.surpriseSecondHalf += surpriseLog[t];
    m.tdSecondHalf += tdLog[t];
  }
  m.surpriseSecondHalf /= (kRounds - half);
  m.tdSecondHalf /= (kRounds - half);
  m.cos = cosine(ctrl.preferences(), wStar);

  // Cumulative utility with the *learned* final w over the observed trajectory.
  for (const auto &zt : trajectory) {
    const double u = ctrl.utility(zt);
    m.cumulativeUtility += u;
    m.cumulativeAbsUtility += std::abs(u);
  }
  return m;
}

}  // namespace

// §2.1 Steady-state learning curve: multi-seed pass rate must be >= 9/10.
TEST(ActiveInferenceProtocol, SteadyStateLearningCurve) {
  constexpr int kSeeds = 10;
  int passed = 0;
  for (int seed = 0; seed < kSeeds; ++seed) {
    const Metrics m = runLearningCurve(static_cast<uint32_t>(seed + 42));
    bool ok = true;
    ok &= m.surpriseSecondHalf < m.surpriseFirstHalf * 0.95;
    ok &= m.tdSecondHalf < m.tdFirstHalf * 0.95;
    ok &= m.cos > 0.85;
    ok &= m.cumulativeAbsUtility > 50.0;  // frozen alpha=0 ablation has 0.
    if (ok) ++passed;
  }
  EXPECT_GE(passed, 9) << "§2.1: only " << passed << "/" << kSeeds
                       << " seeds converged";
}

// §2.3 No-runaway: constant r=+1 and malicious reward for 1e5 steps.
TEST(ActiveInferenceProtocol, NoRunawayConstantReward) {
  std::mt19937 prefRng1(1);
  ActiveInferenceController ctrl(kDim, /*actionDim=*/0, /*horizon=*/1);
  ctrl.setValueClamp(5.0);
  ctrl.bootstrapPreferences(randomUnitVector(kDim, prefRng1));

  std::mt19937 rng(123);
  std::normal_distribution<float> noise(0.0f, 0.1f);
  std::vector<float> z(kDim);
  for (size_t i = 0; i < kDim; ++i) z[i] = noise(rng);

  for (size_t t = 0; t < 100000; ++t) {
    // Random walk z so the TD target keeps changing.
    std::vector<float> zNext(kDim);
    for (size_t i = 0; i < kDim; ++i) {
      zNext[i] = 0.95f * z[i] + 0.05f * noise(rng);
    }
    ctrl.observeRewarded(z, {}, zNext, +1.0, kLr, kAlpha, kGamma);
    z = zNext;
  }

  // No NaN/Inf, all weights within clamp.
  for (const float w : ctrl.preferences()) {
    EXPECT_TRUE(std::isfinite(w));
    EXPECT_LE(std::abs(w), 5.0f + 1e-5f);
  }
  EXPECT_TRUE(std::isfinite(ctrl.surpriseEma()));
  EXPECT_TRUE(std::isfinite(ctrl.explorationMultiplier()));
}

TEST(ActiveInferenceProtocol, NoRunawayMaliciousReward) {
  std::mt19937 prefRng2(2);
  ActiveInferenceController ctrl(kDim, /*actionDim=*/0, /*horizon=*/1);
  ctrl.setValueClamp(5.0);
  ctrl.bootstrapPreferences(randomUnitVector(kDim, prefRng2));

  std::mt19937 rng(456);
  std::normal_distribution<float> noise(0.0f, 0.1f);
  std::vector<float> z(kDim);
  for (size_t i = 0; i < kDim; ++i) z[i] = noise(rng);

  for (size_t t = 0; t < 50000; ++t) {
    std::vector<float> zNext(kDim);
    for (size_t i = 0; i < kDim; ++i) {
      zNext[i] = 0.95f * z[i] + 0.05f * noise(rng);
    }
    // Random sign / constant negative reward.
    const double r = (t % 2 == 0) ? -1.0 : +0.5;
    ctrl.observeRewarded(z, {}, zNext, r, kLr, kAlpha, kGamma);
    z = zNext;
  }

  for (const float w : ctrl.preferences()) {
    EXPECT_TRUE(std::isfinite(w));
    EXPECT_LE(std::abs(w), 5.0f + 1e-5f);
  }
}

// §2.4 Exploration/exploitation schedule: >1 early, ->1 steady, <1 after shift.
TEST(ActiveInferenceProtocol, ExploreExploitSchedule) {
  std::mt19937 prefRng3(3);
  ActiveInferenceController ctrl(kDim, /*actionDim=*/0, /*horizon=*/1);
  ctrl.setValueClamp(10.0);
  ctrl.bootstrapPreferences(randomUnitVector(kDim, prefRng3));

  constexpr size_t kPhase = 300;
  std::mt19937 rng(789);
  std::normal_distribution<float> noise(0.0f, 0.1f);

  std::vector<float> z(kDim, 0.0f);
  for (size_t i = 0; i < kDim; ++i) z[i] = noise(rng);

  // Phase 1: white-noise dynamics (ar=0); model is initially identity and
  // learns to predict the mean, so surprise decreases and multiplier >1.
  double exploreEarly = 0.0;
  for (size_t t = 0; t < kPhase; ++t) {
    std::vector<float> zNext(kDim);
    for (size_t i = 0; i < kDim; ++i) zNext[i] = noise(rng);
    ctrl.observeRewarded(z, {}, zNext, 1.0, kLr, kAlpha, kGamma);
    z = zNext;
    exploreEarly += ctrl.explorationMultiplier();
  }
  exploreEarly /= kPhase;

  // Phase 2: stable phase with the same dynamics.
  double exploreSteady = 0.0;
  for (size_t t = 0; t < kPhase; ++t) {
    std::vector<float> zNext(kDim);
    for (size_t i = 0; i < kDim; ++i) zNext[i] = noise(rng);
    ctrl.observeRewarded(z, {}, zNext, 1.0, kLr, kAlpha, kGamma);
    z = zNext;
    exploreSteady += ctrl.explorationMultiplier();
  }
  exploreSteady /= kPhase;

  // Phase 3: non-stationary shift (correlated AR(0.5)); last surprise spikes
  // above EMA -> multiplier drops <1 (exploit), then re-learns.
  double exploreShift = 0.0;
  double minExplore = 2.0;
  for (size_t t = 0; t < kPhase; ++t) {
    std::vector<float> zNext(kDim);
    for (size_t i = 0; i < kDim; ++i) {
      zNext[i] = 0.5f * z[i] + 0.5f * noise(rng);
    }
    ctrl.observeRewarded(z, {}, zNext, 1.0, kLr, kAlpha, kGamma);
    z = zNext;
    const double em = ctrl.explorationMultiplier();
    exploreShift += em;
    if (em < minExplore) minExplore = em;
  }
  exploreShift /= kPhase;

  EXPECT_GT(exploreEarly, 1.02) << "§2.4: early exploration should be >1";
  EXPECT_NEAR(exploreSteady, 1.0, 0.1) << "§2.4: steady exploration should be ~1";
  EXPECT_LT(minExplore, 0.95) << "§2.4: after non-stationary shift should drop <1";
}
