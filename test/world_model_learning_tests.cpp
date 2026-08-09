#include "world_model.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using json = nlohmann::json;

void requireTrue(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void requireContains(const std::string &text, const std::string &needle, const std::string &message) {
    requireTrue(text.find(needle) != std::string::npos, message + " missing=[" + needle + "]\nactual:\n" + text);
}

void testBuildGroundedLearningSamplesIncludesVisionSpeechAndFusion() {
    json state = {
        {"sessionId", "learn-1"},
        {"sceneState", {{"summary", "Observed a cat near a window while the user said it is sleeping."},
                         {"objectSlots", json::array({json{{"label", "cat"}}, json{{"label", "window"}}})}}},
        {"episode", {{"summary", "The current multimodal episode says a cat is sleeping on the window ledge."}}},
        {"recentEvidence", json::array({json{{"modality", "vision"}, {"graphSummary", "cat on window ledge"}},
                                         json{{"modality", "speech"}, {"text", "the cat is sleeping on the ledge"}}})}
    };

    auto samples = world_model::buildGroundedLearningSamples(state);

    requireTrue(samples.size() >= 3, "expected grounded learning samples for vision, speech, and fusion");
    requireContains(samples[0].graph, "world_scene|summary:", "grounded sample graph should include world prompt context");

    bool sawVision = false;
    bool sawSpeech = false;
    bool sawFusion = false;
    for (const auto &sample : samples) {
        if (sample.source == "vision") {
            sawVision = true;
            requireContains(sample.target, "cat on window ledge", "vision sample should preserve visual evidence");
        }
        if (sample.source == "speech") {
            sawSpeech = true;
            requireContains(sample.target, "sleeping", "speech sample should preserve auditory evidence");
        }
        if (sample.source == "fusion") {
            sawFusion = true;
            requireContains(sample.target, "multimodal episode", "fusion sample should use episode or scene summary");
        }
    }

    requireTrue(sawVision, "vision grounded sample missing");
    requireTrue(sawSpeech, "speech grounded sample missing");
    requireTrue(sawFusion, "fusion grounded sample missing");
}

void testBuildGroundedLearningSamplesSkipsNonSensoryOnlyState() {
    json state = {
        {"sessionId", "learn-2"},
        {"sceneState", {{"summary", "Only text state."}, {"objectSlots", json::array()}}},
        {"episode", {{"summary", "Only text episode."}}},
        {"recentEvidence", json::array({json{{"modality", "text"}, {"text", "plain text only"}}})}
    };

    auto samples = world_model::buildGroundedLearningSamples(state);
    requireTrue(samples.empty(), "non-sensory-only state should not emit multimodal grounded samples");
}

void testGroundedLearningOptionsLimitAndTruncateSamples() {
    json state = {
        {"sessionId", "learn-3"},
        {"sceneState", {{"summary", "Observed multiple sensory items."}, {"objectSlots", json::array()}}},
        {"episode", {{"summary", "Episode summary should drive fusion."}}},
        {"recentEvidence", json::array({json{{"modality", "vision"}, {"graphSummary", "first very long visual evidence summary that should be truncated before training"}},
                                         json{{"modality", "speech"}, {"text", "second very long speech evidence summary that should also be truncated before training"}},
                                         json{{"modality", "vision"}, {"graphSummary", "third evidence should be dropped by the sample limit"}}})}
    };

    world_model::GroundedLearningOptions options;
    options.maxSamples = 2;
    options.maxTargetChars = 28;

    auto samples = world_model::buildGroundedLearningSamples(state, options);

    requireTrue(samples.size() == 2, "grounded learning sample count should respect configured maxSamples");
    requireTrue(samples[0].target.find("...") != std::string::npos || samples[1].target.find("...") != std::string::npos,
                "at least one grounded target should be truncated");
}

void testVideoVjepa2CompressionAndTextFusionSamples() {
    json state = {
        {"sessionId", "learn-4"},
        {"sceneState", {{"summary", "A rover tracks moving silhouettes while operator guidance arrives."}, {"objectSlots", json::array()}}},
        {"episode", {{"summary", "The rover should follow the strongest moving target and keep safe distance."}}},
        {"recentEvidence", json::array({json{{"modality", "video"},
                                              {"graphSummary", "rover sees moving silhouettes near a fence"},
                                              {"metadata", json{{"vjepa2", json{{"coarse", "motion near fence"},
                                                                                   {"medium", "two silhouettes moving left to right"},
                                                                                   {"focus", "primary silhouette accelerates toward gate"}}}}}},
                                     json{{"modality", "text"}, {"text", "prioritize the fastest target but avoid collision"}}})}
    };

    world_model::GroundedLearningOptions options;
    options.maxSamples = 8;
    options.maxVideoCompressionLevels = 3;

    auto samples = world_model::buildGroundedLearningSamples(state, options);
    requireTrue(!samples.empty(), "video-based state should emit grounded learning samples");

    bool sawVideoCoarse = false;
    bool sawVideoMedium = false;
    bool sawVideoFocus = false;
    bool sawVideoTimelineLate = false;
    bool sawVideoTextFusion = false;
    bool sawVideoTimelineTextFusion = false;
    for (const auto &sample : samples) {
        if (sample.source == "video_coarse") {
            sawVideoCoarse = true;
            requireContains(sample.target, "motion near fence", "video coarse sample should keep coarse summary");
        }
        if (sample.source == "video_medium") {
            sawVideoMedium = true;
            requireContains(sample.target, "silhouettes", "video medium sample should keep medium summary");
        }
        if (sample.source == "video_focus") {
            sawVideoFocus = true;
            requireContains(sample.target, "accelerates", "video focus sample should keep focus summary");
        }
        if (sample.source == "fusion_video_text") {
            sawVideoTextFusion = true;
            requireContains(sample.target, "language anchor", "video-text fusion sample should include language anchor");
        }
        if (sample.source == "video_timeline_late") {
            sawVideoTimelineLate = true;
            requireContains(sample.target, "accelerates", "video timeline late sample should preserve late-window summary");
        }
        if (sample.source == "fusion_video_timeline_text") {
            sawVideoTimelineTextFusion = true;
            requireContains(sample.target, "language anchor", "timeline-text fusion sample should include language anchor");
        }
    }

    requireTrue(sawVideoCoarse, "missing v-jepa2 coarse video sample");
    requireTrue(sawVideoMedium, "missing v-jepa2 medium video sample");
    requireTrue(sawVideoFocus, "missing v-jepa2 focus video sample");
    requireTrue(sawVideoTimelineLate, "missing v-jepa2 timeline late sample");
    requireTrue(sawVideoTextFusion, "missing video-text fusion sample");
    requireTrue(sawVideoTimelineTextFusion, "missing video-timeline-text fusion sample");
}

void testVideoPriorityKeepsActionableSamplesUnderTightBudget() {
    json videoEvidence = json::object();
    videoEvidence["modality"] = "video";
    videoEvidence["graphSummary"] = "sensor sees silhouettes shifting near the gate";
    videoEvidence["metadata"] = json{{"cameraInterface", "micro-mipi-csi"},
                                      {"vjepa2", json{{"coarse", "motion near fence"},
                                                       {"medium", "two silhouettes move toward the gate"},
                                                       {"focus", "primary silhouette accelerates toward gate"},
                                                       {"timeline", json{{"early", "target appears at fence edge"},
                                                                          {"mid", "target turns toward gate"},
                                                                          {"late", "target accelerates toward gate while obstacle closes in"}}}}}};

    json state = {
        {"sessionId", "learn-5"},
        {"sceneState", {{"summary", "A direct camera feed watches a gate while the operator issues safety guidance."}, {"objectSlots", json::array()}}},
        {"episode", {{"summary", "Track the fastest moving target while staying collision-safe."}}},
        {"recentEvidence", json::array({
            videoEvidence,
            json{{"modality", "text"}, {"text", "prioritize the fastest target and avoid collision at the gate"}}
        })}
    };

    world_model::GroundedLearningOptions options;
    options.maxSamples = 4;
    options.maxVideoCompressionLevels = 3;
    options.maxVideoTemporalWindows = 3;

    auto samples = world_model::buildGroundedLearningSamples(state, options);

    requireTrue(samples.size() == 4, "tight video sample budget should still return four highest-value samples");

    bool sawFocus = false;
    bool sawLate = false;
    bool sawFusion = false;
    bool sawTemporalFusion = false;
    bool sawCoarse = false;
    bool sawDirectCameraHint = false;
    for (const auto &sample : samples) {
        if (sample.source == "video_focus") {
            sawFocus = true;
            requireContains(sample.target, "accelerates", "focus sample should preserve the most actionable motion detail");
        }
        if (sample.source == "video_timeline_late") {
            sawLate = true;
            requireContains(sample.target, "obstacle", "late timeline sample should preserve near-term risk");
        }
        if (sample.source == "fusion_video_text") {
            sawFusion = true;
            requireContains(sample.target, "language anchor", "video fusion should include the latest language anchor");
        }
        if (sample.source == "fusion_video_timeline_text") {
            sawTemporalFusion = true;
            requireContains(sample.target, "capture path: micro-mipi-csi", "timeline fusion should keep direct camera path metadata");
        }
        if (sample.source == "video_coarse") {
            sawCoarse = true;
        }
        if (sample.input.find("micro-mipi / CSI") != std::string::npos) {
            sawDirectCameraHint = true;
        }
    }

    requireTrue(sawFocus, "tight budget should preserve focus-level video evidence");
    requireTrue(sawLate, "tight budget should preserve the most recent temporal window");
    requireTrue(sawFusion, "tight budget should preserve video-text fusion");
    requireTrue(sawTemporalFusion, "tight budget should preserve video-timeline-text fusion");
    requireTrue(!sawCoarse, "tight budget should drop coarse video evidence before higher-value samples");
    requireTrue(sawDirectCameraHint, "direct micro-mipi camera hint should reach the grounded prompt");
}

} // namespace

int main() {
    try {
        testBuildGroundedLearningSamplesIncludesVisionSpeechAndFusion();
        testBuildGroundedLearningSamplesSkipsNonSensoryOnlyState();
        testGroundedLearningOptionsLimitAndTruncateSamples();
        testVideoVjepa2CompressionAndTextFusionSamples();
        testVideoPriorityKeepsActionableSamplesUnderTightBudget();
        std::cout << "world_model_learning_tests: ok" << std::endl;
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "world_model_learning_tests: failed: " << ex.what() << std::endl;
        return 1;
    }
}