#include "mechanical_mind.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void requireTrue(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void testNeutralTextPasses() {
    mechanical_mind::Filter filter;
    filter.setEnabled(true);
    filter.warmup({mechanical_mind::Document{"Check voltage and replace the burnt fuse.", {"check", "voltage", "replace", "burnt", "fuse"}}});
    auto report = filter.analyzeAndSanitize("Check voltage and replace the burnt fuse.");
    requireTrue(!report.triggered, "neutral maintenance text should not trigger mechanical mind filtering");
    requireTrue(report.sanitized == "Check voltage and replace the burnt fuse.", "neutral text should stay unchanged");
}

void testAffectiveTextIsMechanicalized() {
    mechanical_mind::Filter filter;
    filter.setEnabled(true);
    filter.warmup({mechanical_mind::Document{"I feel sad and lonely when the operator leaves.", {"i", "feel", "sad", "and", "lonely", "when", "the", "operator", "leaves"}}});
    auto report = filter.analyzeAndSanitize("I feel sad and lonely when you leave.");
    requireTrue(report.triggered, "affective anthropomorphic text should trigger");
    requireTrue(report.replacedCount > 0, "triggered text should be rewritten");
    requireTrue(report.sanitized.find("feel") == std::string::npos, "affective token should be filtered from sanitized text");
}

void testWarmupLearnsCooccurringToken() {
    mechanical_mind::Filter filter;
    filter.setEnabled(true);
    filter.warmup({
        mechanical_mind::Document{"I feel love and treasure this bond.", {"i", "feel", "love", "and", "treasure", "this", "bond"}},
        mechanical_mind::Document{"We treasure emotional comfort and empathy.", {"we", "treasure", "emotional", "comfort", "and", "empathy"}},
    });
    auto report = filter.analyzeAndSanitize("I treasure this bond with you.");
    requireTrue(report.triggered, "warmup should learn co-occurring anthropomorphic token risk");
    requireTrue(report.sanitized.find("treasure") == std::string::npos, "learned risky token should be neutralized");
}

} // namespace

int main() {
    try {
        testNeutralTextPasses();
        testAffectiveTextIsMechanicalized();
        testWarmupLearnsCooccurringToken();
        std::cout << "mechanical_mind_filter_tests: ok" << std::endl;
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "mechanical_mind_filter_tests: failed: " << ex.what() << std::endl;
        return 1;
    }
}