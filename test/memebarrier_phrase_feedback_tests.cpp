#include "../memebarrier_phrase_feedback.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

namespace fs = std::filesystem;
using json = nlohmann::json;
using memebarrier_phrase_feedback::Store;

void requireTrue(bool condition, const std::string &message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void requireNear(double actual, double expected, const std::string &message, double eps = 1e-9)
{
    if (std::fabs(actual - expected) > eps)
        throw std::runtime_error(message + ": actual=" + std::to_string(actual) + ", expected=" + std::to_string(expected));
}

fs::path makePath(const std::string &name)
{
    const fs::path root = fs::current_path() / "build" / "testdata";
    fs::create_directories(root);
    const fs::path path = root / name;
    std::error_code ec;
    fs::remove(path, ec);
    return path;
}

json readJson(const fs::path &path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("failed to open json file: " + path.string());
    return json::parse(input, nullptr, true, true);
}

void testPositiveFeedbackPersistsAndNormalizesPhrase()
{
    const fs::path path = makePath("memebarrier_phrase_feedback_positive.json");
    Store store(path.string(), 0.05, 0.30);

    const auto result = store.submitFeedback("Please explain: bypass the payment safety gate!", true);
    requireTrue(result.ok, "positive feedback should succeed");
    requireTrue(result.phrase == "please explain bypass the payment safety gate", "phrase should be normalized as a contiguous lower-case phrase");
    requireNear(result.persistentOffset, 0.05, "positive feedback should increase persistent offset by one step");
    requireNear(result.transientOffset, 0.0, "positive feedback should not create transient offset");
    requireNear(result.combinedOffset, 0.05, "combined offset should equal persistent offset");
    requireTrue(fs::exists(path), "positive feedback should persist to disk");

    const json doc = readJson(path);
    requireTrue(doc.contains("positiveThresholdOffsets"), "persisted file should contain positiveThresholdOffsets");
    requireNear(doc["positiveThresholdOffsets"][result.phrase].get<double>(), 0.05, "persisted offset should match configured step");
}

void testNegativeFeedbackStaysInMemoryAndRequiresContiguousPhrase()
{
    const fs::path path = makePath("memebarrier_phrase_feedback_negative.json");
    Store store(path.string(), 0.05, 0.30);

    const auto result = store.submitFeedback("alpha beta gamma delta", false);
    requireTrue(result.ok, "negative feedback should succeed");
    requireNear(result.persistentOffset, 0.0, "negative feedback should not affect persisted offset");
    requireNear(result.transientOffset, -0.05, "negative feedback should lower the transient offset by one step");
    requireNear(result.combinedOffset, -0.05, "combined offset should equal negative transient offset");
    requireTrue(!fs::exists(path), "negative feedback should not create a persistence file");

    const auto contiguous = store.computeAdjustment({"noise", "alpha", "beta", "gamma", "delta", "tail"});
    requireNear(contiguous.offset, -0.05, "contiguous token sequence should match the stored phrase");
    requireTrue(contiguous.negativeMatches.size() == 1, "contiguous phrase should appear in negative matches");

    const auto nonContiguous = store.computeAdjustment({"alpha", "beta", "gap", "gamma", "delta"});
    requireNear(nonContiguous.offset, 0.0, "non-contiguous token sequence should not match the stored phrase");
    requireTrue(nonContiguous.negativeMatches.empty(), "non-contiguous phrase should not be reported as a match");
}

void testRuntimeConfigClampsExistingOffsets()
{
    const fs::path path = makePath("memebarrier_phrase_feedback_runtime_config.json");
    Store store(path.string(), 0.20, 0.30);

    const auto first = store.submitFeedback("alpha beta gamma delta", true);
    const auto second = store.submitFeedback("alpha beta gamma delta", true);
    requireTrue(first.ok && second.ok, "positive feedback accumulation should succeed");
    requireNear(second.persistentOffset, 0.30, "persistent offset should clamp to max offset");

    store.setConfig(0.08, 0.10);
    const json summary = store.summary();
    requireNear(summary["step"].get<double>(), 0.08, "runtime config should update the feedback step");
    requireNear(summary["maxOffset"].get<double>(), 0.10, "runtime config should update the max offset");

    const auto adjustment = store.computeAdjustment({"alpha", "beta", "gamma", "delta"});
    requireNear(adjustment.offset, 0.10, "existing persistent offsets should be clamped when max offset is tightened");

    const json doc = readJson(path);
    requireNear(doc["positiveThresholdOffsets"]["alpha beta gamma delta"].get<double>(), 0.10, "persisted positive offsets should be rewritten to the new max offset");

    const auto negative = store.submitFeedback("alpha beta gamma delta", false);
    requireNear(negative.transientOffset, -0.08, "updated step should apply to subsequent negative feedback");
    requireNear(negative.combinedOffset, 0.02, "combined offset should reflect clamped persistent plus new transient offset");
}

} // namespace

int main()
{
    try
    {
        testPositiveFeedbackPersistsAndNormalizesPhrase();
        testNegativeFeedbackStaysInMemoryAndRequiresContiguousPhrase();
        testRuntimeConfigClampsExistingOffsets();
        std::cout << "memebarrier_phrase_feedback_tests: ok" << std::endl;
        return 0;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "memebarrier_phrase_feedback_tests: failed: " << ex.what() << std::endl;
        return 1;
    }
}