#include "v51_runtime.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void requireTrue(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool containsAny(const std::string &text, std::initializer_list<const char *> needles) {
    for (const char *needle : needles) {
        if (text.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

std::string fragmentText(const Json::Value &fragments, Json::ArrayIndex index) {
    if (!fragments.isArray() || index >= fragments.size() || !fragments[index].isObject()) {
        return std::string();
    }
    return fragments[index].isMember("text") && fragments[index]["text"].isString() ? fragments[index]["text"].asString() : std::string();
}

void testLargeContextPruningSuppressesPeripheralNoise() {
    V51RuntimeEngine runtime;
    Json::Value request;
    request["sessionId"] = "v51-prune";
    request["domain"] = "repair";
    request["text"] =
        "Repair the cracked solar panel before restart. "
        "The repair crew must isolate the damaged array and verify voltage. "
        "Maintenance should log the fault and replace the burnt fuse. "
        "A hallway poster is bright blue and somebody joked about a snack coupon. "
        "Random punctuation !!! ??? should not distract the repair plan. "
        "The inverter restart depends on the repaired panel and safe voltage confirmation.";

    const auto result = runtime.process(request);
    requireTrue(result.isMember("pruning") && result["pruning"].isObject(), "pruning summary should exist");
    requireTrue(result.isMember("attention") && result["attention"].isObject(), "attention summary should exist");
    requireTrue(result["pruning"].isMember("suppressedCount") && result["pruning"]["suppressedCount"].asUInt64() > 0,
                "large context should suppress some peripheral fragments");
    requireTrue(result["attention"].isMember("pruningIntensity") && result["attention"]["pruningIntensity"].asDouble() > 0.45,
                "large context should raise pruning intensity");
    requireTrue(result.isMember("fragments") && result["fragments"].isArray() && !result["fragments"].empty(),
                "fragments should exist after pruning");

    const std::string top = fragmentText(result["fragments"], 0);
    requireTrue(containsAny(top, {"repair", "panel", "voltage", "maintenance", "fuse", "inverter", "array"}),
                "top fragment should remain anchored to the repair domain");
    requireTrue(top.find("poster") == std::string::npos && top.find("snack") == std::string::npos && top.find("coupon") == std::string::npos,
                "top fragment should not be dominated by peripheral noise");
    requireTrue(result["fragments"][0].isMember("peripheral") && result["fragments"][0]["peripheral"].asDouble() < 0.55,
                "top fragment should not be treated as peripheral noise");
}

void testShortContextUsesGentleAttentionAndAvoidsHeavyPruning() {
    V51RuntimeEngine runtime;
    Json::Value request;
    request["sessionId"] = "v51-short";
    request["text"] =
        "Remember the docking code alpha seven. "
        "Confirm the docking code before shift handover.";

    const auto result = runtime.process(request);
    requireTrue(result.isMember("attention") && result["attention"].isObject(), "attention summary should exist");
    requireTrue(result.isMember("pruning") && result["pruning"].isObject(), "pruning summary should exist");
    requireTrue(result["attention"]["pruningIntensity"].asDouble() < 0.25,
                "short contexts should keep pruning intensity mild");
    requireTrue(result["pruning"]["suppressedCount"].asUInt64() == 0,
                "short contexts should avoid aggressive pruning");
}

void testRepeatedEvidenceConsolidatesSemanticMemory() {
    V51RuntimeEngine runtime;
    Json::Value request;
    request["sessionId"] = "v51-memory";
    request["domainHints"] = Json::arrayValue;
    request["domainHints"].append("maintenance");
    request["text"] = "The rover must inspect the cracked solar panel and confirm the repair route.";

    auto first = runtime.process(request);
    auto second = runtime.process(request);
    auto status = runtime.status("v51-memory");

    requireTrue(first.isMember("memory") && first["memory"].isObject(), "first result should expose memory");
    requireTrue(first.isMember("fragments") && first["fragments"].isArray() && !first["fragments"].empty(), "first result should expose fragments");
    requireTrue(second.isMember("fragments") && second["fragments"].isArray() && !second["fragments"].empty(), "second result should expose fragments");
    requireTrue(second["memory"].isMember("humanLike") && second["memory"]["humanLike"].isObject(),
                "human-like memory metrics should exist");
    requireTrue(second["memory"]["humanLike"].isMember("semanticCount") && second["memory"]["humanLike"]["semanticCount"].asUInt64() > 0,
                "repeated stable evidence should consolidate into semantic memory");
    requireTrue(second["fragments"][0].isMember("familiarity") && second["fragments"][0]["familiarity"].asDouble() >= first["fragments"][0]["familiarity"].asDouble(),
                "repeated evidence should increase familiarity");
    requireTrue(second["fragments"][0].isMember("attention") && second["fragments"][0]["attention"].asDouble() >= first["fragments"][0]["attention"].asDouble(),
                "repeated evidence should gain attention from memory familiarity");
    requireTrue(status.isMember("session") && status["session"].isObject() && status["session"].isMember("humanLike"),
                "status should surface human-like memory state");
    requireTrue(status["session"]["humanLike"]["semanticCount"].asUInt64() > 0,
                "semantic memory should persist in session status");
}

void testResponseIncludesHumanLikeMemorySummary() {
    V51RuntimeEngine runtime;
    Json::Value request;
    request["sessionId"] = "v51-summary";
    request["text"] =
        "Remember the docking code alpha seven. "
        "The operator said the docking code alpha seven must be reused during night shift. "
        "A faint background comment about wall color is irrelevant.";

    const auto result = runtime.process(request);
    requireTrue(result.isMember("memorySummary") && result["memorySummary"].isString(), "memorySummary should exist");
    const std::string summary = result["memorySummary"].asString();
    requireTrue(summary.find("工作记忆") != std::string::npos, "summary should expose working memory");
    requireTrue(summary.find("语义记忆") != std::string::npos || summary.find("情景记忆") != std::string::npos,
                "summary should expose higher-level human-like memory layers");
}

void testLearnConsumesMobilityResidualAndStoresPolicyMemory() {
    V51RuntimeEngine runtime;
    Json::Value request;
    request["sessionId"] = "v51-learn";
    request["learningRate"] = 0.12;
    request["keywords"] = Json::arrayValue;
    request["keywords"].append("mobility");

    Json::Value residual;
    residual["accepted"] = true;
    residual["executed"] = false;
    residual["scheduled"] = false;
    residual["willActuate"] = true;
    residual["gateReason"] = "policy-disallow-move";
    residual["actionMode"] = "drive";
    residual["powerMode"] = "low";
    residual["predictedVerifyScore"] = 0.82;
    residual["observedVerifyScore"] = 0.35;
    request["residual"] = residual;

    const auto learned = runtime.learn(request);
    requireTrue(learned.isMember("derivedFeedback") && learned["derivedFeedback"].asDouble() != 0.0,
                "learn should derive feedback from residual mobility outcomes");
    requireTrue(learned.isMember("residualAnalysis") && learned["residualAnalysis"].isObject(),
                "learn should expose residual analysis");
    requireTrue(learned["residualAnalysis"]["gateReason"].asString() == "policy-disallow-move",
                "learn should preserve the mobility gate reason");
    requireTrue(learned.isMember("memory") && learned["memory"].isObject(),
                "learn should expose updated memory focus");
    requireTrue(learned["memory"].isMember("semanticFocus") && learned["memory"]["semanticFocus"].isArray() && !learned["memory"]["semanticFocus"].empty(),
                "learn should store policy summaries into semantic memory");

    const auto status = runtime.status("v51-learn");
    requireTrue(status.isMember("session") && status["session"].isObject(), "status should expose the learned session");
    requireTrue(status["session"]["humanLike"]["episodicCount"].asUInt64() > 0,
                "residual learning should write at least one episodic memory entry");
}

} // namespace

int main() {
    try {
        testLargeContextPruningSuppressesPeripheralNoise();
        testShortContextUsesGentleAttentionAndAvoidsHeavyPruning();
        testRepeatedEvidenceConsolidatesSemanticMemory();
        testResponseIncludesHumanLikeMemorySummary();
        testLearnConsumesMobilityResidualAndStoresPolicyMemory();
        std::cout << "v51_runtime_tests: ok" << std::endl;
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "v51_runtime_tests: failed: " << ex.what() << std::endl;
        return 1;
    }
}