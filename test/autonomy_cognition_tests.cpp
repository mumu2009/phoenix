#include "autonomy_stack.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

void requireTrue(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void testObserveBuildsSessionBacklogAndActionQueue() {
    autonomy::CognitionAutonomyManager manager;

    json worldState = {
        {"sessionId", "cognition-1"},
        {"evidenceCount", 3},
        {"recentEvidence", json::array({json{{"modality", "vision"}, {"graphSummary", "rover near damaged solar array"}}})}
    };
    json payload = {
        {"sessionId", "cognition-1"},
        {"reasoningAgenda", {
            {"hypotheses", json::array({"the damaged array is causing the unstable output"})},
            {"contradictions", json::array({"telemetry is unstable and repair safety is not confirmed"})},
            {"openQuestions", json::array({"what evidence would reduce uncertainty fastest?"})},
            {"nextStep", "inspect the damaged array before attempting repair"},
            {"shouldReflect", true},
            {"reflectionReason", "uncertainty is high"},
            {"uncertainty", 0.74}
        }},
        {"responsePlan", {
            {"goalFrame", "stabilize the power diagnosis"},
            {"simulationTargets", json::array({"simulate next state around the damaged array"})},
            {"critiqueChecklist", json::array({"avoid unsupported certainty"})},
            {"mobilityPlan", {
                {"strategicGoal", "inspect the damaged panel with a ground-safe rover route"},
                {"bigBrain", {{"routePolicy", "prefer a contour-following route with traction reserve"},
                               {"routeWaypoints", json::array({"stage on stable ground", "approach the damaged panel from the uphill side"})},
                               {"simulationTargets", json::array({"simulate rover traction along the first route segment"})}}},
                {"littleBrain", {{"speedPolicy", "cautious crawl with stop-to-check windows"}}}
            }},
            {"revisionBudget", 0.74}
        }},
        {"worldUncertainty", 0.74},
        {"worldReflectionSuggested", true},
        {"verify", {{"score", 0.41}}},
        {"reply", "The damaged array may be responsible, but the evidence is incomplete."}
    };

    auto observed = manager.observe(payload, worldState);

    requireTrue(observed.value("ok", false), "observe should succeed");
    requireTrue(observed.contains("result") && observed["result"].is_object(), "observe should return a result object");
    requireTrue(observed["result"].value("shouldIterate", false), "high-uncertainty session should be marked for iteration");
    requireTrue(observed["result"].contains("subgoals") && observed["result"]["subgoals"].is_array() && !observed["result"]["subgoals"].empty(),
                "observe should produce subgoals");
    requireTrue(observed["result"].contains("reflectionTasks") && observed["result"]["reflectionTasks"].is_array() && !observed["result"]["reflectionTasks"].empty(),
                "observe should produce reflection tasks");
    requireTrue(observed["result"].contains("actionQueue") && observed["result"]["actionQueue"].is_array() && !observed["result"]["actionQueue"].empty(),
                "observe should produce an action queue");
    requireTrue(observed["result"].contains("heads") && observed["result"]["heads"].is_array() && !observed["result"]["heads"].empty(),
                "observe should expose weighted cognition heads");
    requireTrue(observed["result"].contains("dominantHead") && observed["result"]["dominantHead"].is_string() && !observed["result"]["dominantHead"].get<std::string>().empty(),
                "observe should expose the dominant cognition head");
    requireTrue(observed["result"].contains("mobilityPlan") && observed["result"]["mobilityPlan"].is_object(),
                "observe should retain the mobility plan in session state");
    requireTrue(observed["result"]["actionQueue"].is_array(), "action queue should stay array-typed");
    bool sawGroundRouteAction = false;
    for (const auto &action : observed["result"]["actionQueue"]) {
        if (action.is_string() && action.get<std::string>() == "plan-ground-route") {
            sawGroundRouteAction = true;
            break;
        }
    }
    requireTrue(sawGroundRouteAction, "observe should schedule ground-route planning when a mobility plan is present");
}

void testIterateProducesWorldEvidenceAndRuntimePatch() {
    autonomy::CognitionAutonomyManager manager;

    json worldState = {
        {"sessionId", "cognition-2"},
        {"evidenceCount", 2},
        {"recentEvidence", json::array({json{{"modality", "speech"}, {"text", "repair path remains uncertain"}}})}
    };
    json payload = {
        {"sessionId", "cognition-2"},
        {"reasoningAgenda", {
            {"hypotheses", json::array({"a conveyor timing fault is causing the jam"})},
            {"contradictions", json::array({"sensor fault and mechanical jam are both plausible"})},
            {"openQuestions", json::array({"which subsystem should be inspected first?"})},
            {"nextStep", "inspect the timing subsystem before restarting the conveyor"},
            {"shouldReflect", true},
            {"reflectionReason", "multiple root causes remain plausible"},
            {"uncertainty", 0.68}
        }},
        {"responsePlan", {
            {"goalFrame", "stabilize the conveyor diagnosis"},
            {"simulationTargets", json::array({"simulate restart risk after timing inspection"})},
            {"critiqueChecklist", json::array({"resolve the competing root-cause hypotheses"})},
            {"mobilityPlan", {
                {"strategicGoal", "move the inspection rover to the timing subsystem along a safe ground corridor"},
                {"bigBrain", {{"routePolicy", "prefer the shortest stable ground corridor with periodic evidence refresh"},
                               {"routeWaypoints", json::array({"hold a staging position", "finish at timing subsystem"})},
                               {"simulationTargets", json::array({"simulate stop distance before the final waypoint"})}}},
                {"littleBrain", {{"speedPolicy", "measured cruise with conservative braking reserve"}}}
            }},
            {"revisionBudget", 0.68}
        }},
        {"worldUncertainty", 0.68},
        {"worldReflectionSuggested", true},
        {"verify", {{"score", 0.36}}}
    };

    auto observed = manager.observe(payload, worldState);
    requireTrue(observed.value("ok", false), "observe should succeed before iterate");

    auto iterated = manager.iterate(json{{"sessionId", "cognition-2"}}, worldState);
    requireTrue(iterated.value("ok", false), "iterate should succeed");
    requireTrue(iterated.contains("result") && iterated["result"].is_object(), "iterate should return a result object");
    requireTrue(iterated["result"].contains("worldEvidence") && iterated["result"]["worldEvidence"].is_array() && !iterated["result"]["worldEvidence"].empty(),
                "iterate should emit world evidence suggestions");
    requireTrue(iterated["result"].contains("runtimeFeaturePatch") && iterated["result"]["runtimeFeaturePatch"].is_object(),
                "iterate should emit a runtime patch suggestion");
    requireTrue(iterated["result"].contains("scheduledHeads") && iterated["result"]["scheduledHeads"].is_array() && !iterated["result"]["scheduledHeads"].empty(),
                "iterate should expose weighted scheduled heads");
    requireTrue(iterated["result"]["runtimeFeaturePatch"].value("reasoningPlannerEnabled", false),
                "iterate should keep planner enabled in runtime patch suggestions");
    bool sawMobilityEvidence = false;
    for (const auto &evidence : iterated["result"]["worldEvidence"]) {
        if (evidence.is_object() && evidence.contains("metadata") && evidence["metadata"].is_object() &&
            evidence["metadata"].value("kind", std::string()) == "mobility") {
            sawMobilityEvidence = true;
            break;
        }
    }
    requireTrue(sawMobilityEvidence, "iterate should emit explicit mobility evidence when rover planning is present");

    auto session = manager.session("cognition-2");
    requireTrue(session.value("ok", false), "session lookup should succeed");
    requireTrue(session["result"].contains("lastActionQueue") && session["result"]["lastActionQueue"].is_array(),
                "session snapshot should retain the last action queue");
    requireTrue(session["result"].contains("lastScheduledHeads") && session["result"]["lastScheduledHeads"].is_array() && !session["result"]["lastScheduledHeads"].empty(),
                "session snapshot should retain the last weighted head schedule");
}

void testObserveHandlesMalformedHeadInputsAndDeduplicates() {
    autonomy::CognitionAutonomyManager manager;

    const std::string oversizedQuestion(220, 'Q');
    json worldState = {{"sessionId", "cognition-malformed"}, {"evidenceCount", 0}};
    json payload = {
        {"sessionId", "cognition-malformed"},
        {"reasoningAgenda", {
            {"hypotheses", json::array({"same hypothesis", "same hypothesis", 42, nullptr})},
            {"contradictions", json::array({"conflict A", "conflict A", true})},
            {"openQuestions", json::array({oversizedQuestion, oversizedQuestion, "", json::object()})},
            {"nextStep", "  stabilize inputs  "},
            {"shouldReflect", true},
            {"reflectionReason", "messy payload should not break scheduling"},
            {"uncertainty", 0.81}
        }},
        {"responsePlan", {
            {"goalFrame", "stabilize the malformed cognition payload"},
            {"simulationTargets", json::array({"simulate parser fallback", 7, "simulate parser fallback"})},
            {"critiqueChecklist", json::array({"remove duplicates", false, "remove duplicates"})},
            {"mobilityPlan", {{"strategicGoal", ""}, {"bigBrain", {{"routeWaypoints", json::array({"checkpoint", "checkpoint"})}}}}},
            {"revisionBudget", 0.81}
        }},
        {"worldUncertainty", 0.81},
        {"verify", {{"score", 0.22}}}
    };

    auto observed = manager.observe(payload, worldState);
    requireTrue(observed.value("ok", false), "observe should tolerate malformed and duplicate payload content");
    requireTrue(observed["result"].contains("heads") && observed["result"]["heads"].is_array() && !observed["result"]["heads"].empty(),
                "observe should still produce weighted heads for malformed samples");

    std::unordered_set<std::string> actionSeen;
    for (const auto &action : observed["result"]["actionQueue"]) {
        requireTrue(action.is_string(), "action queue entries should stay string-typed after malformed inputs");
        const std::string actionText = action.get<std::string>();
        requireTrue(actionSeen.insert(actionText).second, "action queue should de-duplicate repeated scheduled tasks");
    }

    bool sawTruncatedFocus = false;
    for (const auto &head : observed["result"]["heads"]) {
        requireTrue(head.is_object(), "weighted heads should stay object-typed");
        requireTrue(head.contains("triggerEvents") && head["triggerEvents"].is_array(), "weighted heads should expose trigger events");
        if (!head.contains("focus") || !head["focus"].is_array()) {
            continue;
        }
        for (const auto &focus : head["focus"]) {
            if (!focus.is_string()) {
                continue;
            }
            requireTrue(focus.get<std::string>().size() <= 120, "head focus text should be truncated to bounded size");
            if (focus.get<std::string>().find("...") != std::string::npos) {
                sawTruncatedFocus = true;
            }
        }
    }
    requireTrue(sawTruncatedFocus, "oversized malformed prompts should be truncated inside head focus summaries");
}

void testSeedPayloadColdStartsAutonomyWithoutDialogHistory() {
    autonomy::CognitionAutonomyManager manager;

    json worldState = {
        {"sessionId", "cognition-seed"},
        {"evidenceCount", 0},
        {"recentEvidence", json::array()}
    };
    json payload = autonomy::buildCognitionAutonomySeedPayload(
        "cognition-seed",
        "investigate how to reduce repeated runtime failures and produce a safer next action plan",
        0.72);

    auto observed = manager.observe(payload, worldState);
    requireTrue(observed.value("ok", false), "seed payload should cold-start cognition observe");
    requireTrue(observed["result"].value("shouldIterate", false), "seed payload should request iteration when uncertainty is high");
    requireTrue(observed["result"].contains("subgoals") && observed["result"]["subgoals"].is_array() && !observed["result"]["subgoals"].empty(),
                "seed payload should generate startup subgoals");

    auto iterated = manager.iterate(json{{"sessionId", "cognition-seed"}}, worldState);
    requireTrue(iterated.value("ok", false), "seeded cognition session should iterate without prior dialog history");
    requireTrue(iterated["result"].contains("worldEvidence") && iterated["result"]["worldEvidence"].is_array() && !iterated["result"]["worldEvidence"].empty(),
                "seeded cognition iteration should emit world evidence");

    auto session = manager.session("cognition-seed");
    requireTrue(session.value("ok", false), "seeded cognition session lookup should succeed");
    requireTrue(session["result"].value("observations", 0) >= 1, "seeded cognition session should record the first observation");
    requireTrue(session["result"].value("dominantHead", std::string()).size() > 0,
                "seeded cognition session should select a dominant head");
}

void testExportAndImportStateRestoresSessionSchedule() {
    autonomy::CognitionAutonomyManager manager;

    json worldState = {
        {"sessionId", "cognition-persist"},
        {"evidenceCount", 2},
        {"recentEvidence", json::array({json{{"modality", "cognition"}, {"graphSummary", "route to the timing subsystem remains unstable"}}})}
    };
    json payload = {
        {"sessionId", "cognition-persist"},
        {"reasoningAgenda", {
            {"hypotheses", json::array({"the timing rail fault is primary"})},
            {"contradictions", json::array({"sensor drift is still plausible"})},
            {"openQuestions", json::array({"which subsystem should be sampled first?"})},
            {"nextStep", "inspect the timing rail before restarting the line"},
            {"shouldReflect", true},
            {"reflectionReason", "restored sessions should retain weighted scheduling"},
            {"uncertainty", 0.71}
        }},
        {"responsePlan", {
            {"goalFrame", "stabilize the timing diagnosis"},
            {"simulationTargets", json::array({"simulate timing restart after inspection"})},
            {"critiqueChecklist", json::array({"keep competing root causes explicit"})},
            {"mobilityPlan", {
                {"strategicGoal", "move inspection rover along the safe ground corridor"},
                {"bigBrain", {{"routePolicy", "prefer stable ground with periodic re-checks"},
                               {"routeWaypoints", json::array({"stage at the aisle", "finish at timing rail"})}}},
                {"littleBrain", {{"speedPolicy", "measured crawl"}}}
            }},
            {"revisionBudget", 0.71}
        }},
        {"worldUncertainty", 0.71},
        {"worldReflectionSuggested", true},
        {"verify", {{"score", 0.29}}}
    };

    auto observed = manager.observe(payload, worldState);
    requireTrue(observed.value("ok", false), "observe should succeed before export/import round trip");
    auto iterated = manager.iterate(json{{"sessionId", "cognition-persist"}}, worldState);
    requireTrue(iterated.value("ok", false), "iterate should succeed before export/import round trip");

    auto before = manager.session("cognition-persist");
    requireTrue(before.value("ok", false), "session lookup should succeed before export");

    json exported = manager.exportState();
    requireTrue(exported.contains("sessions") && exported["sessions"].is_object() && exported["sessions"].contains("cognition-persist"),
                "exported state should include the persisted cognition session");

    autonomy::CognitionAutonomyManager restored;
    auto imported = restored.importState(exported);
    requireTrue(imported.value("ok", false), "importState should accept exported manager state");

    auto status = restored.status();
    requireTrue(status.value("ok", false), "restored manager status should succeed");
    requireTrue(status["result"].value("sessionsTracked", 0) >= 1,
                "restored manager should report tracked sessions");

    auto after = restored.session("cognition-persist");
    requireTrue(after.value("ok", false), "restored session lookup should succeed");
    requireTrue(after["result"].contains("lastScheduledHeads") && after["result"]["lastScheduledHeads"].is_array() && !after["result"]["lastScheduledHeads"].empty(),
                "restored session should keep last scheduled heads");
    requireTrue(after["result"].contains("lastActionQueue") && after["result"]["lastActionQueue"].is_array() && !after["result"]["lastActionQueue"].empty(),
                "restored session should keep last action queue");
    requireTrue(after["result"].value("dominantHead", std::string()) == before["result"].value("dominantHead", std::string()),
                "restored session should preserve the dominant cognition head");
}

} // namespace

int main() {
    try {
        testObserveBuildsSessionBacklogAndActionQueue();
        testIterateProducesWorldEvidenceAndRuntimePatch();
        testObserveHandlesMalformedHeadInputsAndDeduplicates();
        testSeedPayloadColdStartsAutonomyWithoutDialogHistory();
        testExportAndImportStateRestoresSessionSchedule();
        std::cout << "autonomy_cognition_tests: ok" << std::endl;
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "autonomy_cognition_tests: failed: " << ex.what() << std::endl;
        return 1;
    }
}