/* test_controller_persistence.cpp - ActiveInferenceController full-state
   serialization (long-term evolution survives restarts). */

#include "active_inference.hpp"

#include <gtest/gtest.h>

#include <vector>

using phoenix::agi::Action;
using phoenix::agi::ActiveInferenceController;

namespace {

std::vector<float> unitVec(size_t n, uint32_t seed) {
    std::vector<float> v(n);
    unsigned s = seed;
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        v[i] = static_cast<float>((s % 2000u) - 1000u) / 1000.0f;
        sum += v[i] * v[i];
    }
    const double norm = std::sqrt(sum);
    for (auto &x : v) x = static_cast<float>(x / norm);
    return v;
}

}  // namespace

TEST(ControllerPersistence, JsonRoundTripPreservesLearnedState) {
    ActiveInferenceController ctl(8, 2, 3);
    ctl.setActions({Action{"a", {1.0f, 0.0f}}, Action{"b", {0.0f, 1.0f}}});
    ctl.bootstrapPreferences(unitVec(8, 42));
    /* learn a few transitions so w / model / EMA actually change */
    for (int i = 0; i < 20; ++i) {
        std::vector<float> z = unitVec(8, 100 + i);
        std::vector<float> a = {i % 2 == 0 ? 1.0f : 0.0f,
                                i % 2 == 0 ? 0.0f : 1.0f};
        std::vector<float> zNext = unitVec(8, 200 + i);
        ctl.observeRewarded(z, a, zNext, 0.5, 0.01f, 0.05f, 0.9);
    }

    nlohmann::json j = ctl.toJson();
    ActiveInferenceController restored = ActiveInferenceController::fromJson(j);

    EXPECT_EQ(restored.preferences().size(), ctl.preferences().size());
    for (size_t i = 0; i < ctl.preferences().size(); ++i) {
        EXPECT_FLOAT_EQ(restored.preferences()[i], ctl.preferences()[i]);
    }
    EXPECT_DOUBLE_EQ(restored.surpriseEma(), ctl.surpriseEma());
    EXPECT_EQ(restored.actions().size(), 2u);
    EXPECT_EQ(restored.actions()[0].name, "a");
    EXPECT_GT(restored.episodeCount(), 0u);
    EXPECT_TRUE(restored.preferencesBootstrapped());
    EXPECT_EQ(restored.model().dim(), 8u);
    EXPECT_EQ(restored.model().actionDim(), 2u);
}
