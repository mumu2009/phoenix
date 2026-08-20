#include "world_model.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

using json = nlohmann::json;

void requireTrue(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool jsonArrayContainsString(const json &arr, const std::string &value) {
    if (!arr.is_array()) {
        return false;
    }
    for (const auto &entry : arr) {
        if (entry.is_string() && entry.get<std::string>() == value) {
            return true;
        }
    }
    return false;
}

bool jsonObjectContainsKey(const json &obj, const std::string &key) {
    return obj.is_object() && obj.contains(key);
}

void testBuildBrainProfilesProvidesFunctionalAndStructuralVersions() {
    json state = {
        {"sessionId", "brain-profiles-1"},
        {"sceneState", {{"summary", "Observed a person reading near a lamp while audio mentions quiet music."},
                         {"objectSlots", json::array({json{{"label", "person"}}, json{{"label", "lamp"}}, json{{"label", "book"}}})},
                         {"tags", json::array({"reading", "music", "lamp"})}}},
        {"episode", {{"summary", "The person keeps reading while music continues in the room."}}},
        {"recentEvidence", json::array({json{{"modality", "vision"}, {"graphSummary", "person reading near lamp with book"}},
                                         json{{"modality", "speech"}, {"text", "soft music is playing in the room"}}})}
    };

    auto brainProfiles = world_model::buildBrainProfiles(state);

    requireTrue(brainProfiles.is_object(), "brain profiles should be an object");
    requireTrue(brainProfiles.contains("application") && brainProfiles["application"].is_object(),
                "functional application profile should exist");
    requireTrue(brainProfiles.contains("research") && brainProfiles["research"].is_object(),
                "structural research profile should exist");
    requireTrue(brainProfiles["application"].value("profile", std::string()) == "functional",
                "application profile should be functional");
    requireTrue(brainProfiles["research"].value("profile", std::string()) == "structural",
                "research profile should be structural");
        requireTrue(jsonObjectContainsKey(brainProfiles["application"], "corticalSystems"),
                    "functional profile should expose cortical systems");
        requireTrue(jsonObjectContainsKey(brainProfiles["application"], "functionalRuntime"),
                    "functional profile should expose runtime loops");
    requireTrue(jsonObjectContainsKey(brainProfiles["application"]["corticalSystems"], "visualCortex"),
                "functional profile should include visual cortex");
    requireTrue(jsonObjectContainsKey(brainProfiles["application"]["corticalSystems"], "auditoryCortex"),
                "functional profile should include auditory cortex");
    requireTrue(jsonObjectContainsKey(brainProfiles["application"]["corticalSystems"], "languageNetwork"),
                "functional profile should include language network");
    requireTrue(jsonObjectContainsKey(brainProfiles["research"], "corticalMap"),
                "structural profile should expose cortical map");
    requireTrue(jsonObjectContainsKey(brainProfiles["research"], "humanThoughtModel"),
                "structural profile should expose human thought model");
    requireTrue(jsonObjectContainsKey(brainProfiles["research"], "consciousComputePlan"),
                "structural profile should expose conscious compute plan");
    requireTrue(jsonObjectContainsKey(brainProfiles["research"], "collectiveComputePlan"),
                "structural profile should expose collective compute plan");
    requireTrue(jsonObjectContainsKey(brainProfiles["research"]["corticalMap"], "occipitalCortex"),
                "structural profile should include occipital cortex");
    requireTrue(jsonObjectContainsKey(brainProfiles["research"]["corticalMap"], "auditoryCortex"),
                "structural profile should include auditory cortex");
    requireTrue(jsonObjectContainsKey(brainProfiles["research"]["corticalMap"], "prefrontalCortex"),
                "structural profile should include prefrontal cortex");
}

void testBuildCognitiveStateIncludesWorkingMemoryGoalsAndRegions() {
    json state = {
        {"sessionId", "brain-1"},
        {"sceneState", {{"summary", "Observed a cat near a window while speech says it is sleeping."},
                         {"objectSlots", json::array({json{{"label", "cat"}}, json{{"label", "window"}}})},
                         {"tags", json::array({"cat", "window", "sleeping"})}}},
        {"episode", {{"summary", "The cat is resting on the window ledge in the current episode."}}},
        {"recentEvidence", json::array({json{{"modality", "vision"}, {"graphSummary", "cat on window ledge"}},
                                         json{{"modality", "speech"}, {"text", "the cat is sleeping on the ledge"}}})}
    };

    auto cognitive = world_model::buildCognitiveState(state);

    requireTrue(cognitive.is_object(), "cognitive state should be an object");
    requireTrue(cognitive.contains("workingMemory") && cognitive["workingMemory"].is_array() && !cognitive["workingMemory"].empty(),
                "working memory should be populated");
    requireTrue(cognitive.contains("activeGoals") && cognitive["activeGoals"].is_array(), "active goals should exist");
    requireTrue(jsonArrayContainsString(cognitive["activeGoals"], "align multimodal evidence"),
                "multimodal alignment goal should be present");
    requireTrue(cognitive.contains("brainRegions") && cognitive["brainRegions"].is_object(), "brain regions should exist");
    requireTrue(cognitive["brainRegions"].contains("prefrontal") && cognitive["brainRegions"]["prefrontal"].is_array(),
                "prefrontal region should be populated");
    requireTrue(cognitive["brainRegions"].contains("hippocampus") && cognitive["brainRegions"]["hippocampus"].is_array(),
                "hippocampus region should be populated");
}

void testWorldModelStorePersistsPredictiveStateAndCalibration() {
    world_model::WorldModelStore store(std::shared_ptr<KeyValueStore>{},
                                       std::shared_ptr<KeyValueStore>{},
                                       std::shared_ptr<KeyValueStore>{});

    json first = {
        {"sessionId", "wm-predictive-1"},
        {"modality", "vision"},
        {"graphSummary", "cat resting on the window ledge beside a plant"},
        {"metadata",
         {
             {"detections", json::array({json{{"label", "cat"}, {"confidence", 0.98}}, json{{"label", "window"}}, json{{"label", "plant"}}})},
             {"goals", json::array({"track object state transitions"})},
             {"hypotheses", json::array({"the cat will remain on the ledge"})},
             {"relations", json::array({json{{"subject", "cat"}, {"predicate", "near"}, {"object", "window"}}})},
             {"expectedNextState",
              json{{"summary", "cat likely remains on the window ledge while the scene stays quiet"},
                   {"entities", json::array({"cat", "window"})},
                   {"goals", json::array({"track object state transitions"})}}}
         }}
    };
    store.ingestEvidence(first);

    json second = {
        {"sessionId", "wm-predictive-1"},
        {"modality", "speech"},
        {"text", "the cat is still sleeping on the same ledge near the window"},
        {"graphSummary", "cat remains asleep near the window ledge"},
        {"metadata",
         {
             {"goals", json::array({"consolidate episodic memory"})},
             {"observedNextState",
              json{{"summary", "cat remains asleep near the window ledge"},
                   {"entities", json::array({"cat", "window"})},
                   {"goals", json::array({"track object state transitions"})}}}
         }}
    };
    store.ingestEvidence(second);

    auto state = store.sessionState("wm-predictive-1", 4);

    requireTrue(state.contains("prediction") && state["prediction"].is_object(), "session state should expose prediction state");
    const auto &prediction = state["prediction"];
    requireTrue(prediction["entities"].is_array() && jsonArrayContainsString(prediction["entities"], "cat"),
                "prediction state should persist normalized entities");
    requireTrue(prediction["relations"].is_array() && !prediction["relations"].empty(),
                "prediction state should persist relations");
    requireTrue(prediction["goals"].is_array() && !prediction["goals"].empty(),
                "prediction state should persist goals");
    requireTrue(prediction["hypotheses"].is_array() && !prediction["hypotheses"].empty(),
                "prediction state should persist hypotheses");
    requireTrue(prediction.contains("expectedNextState") && prediction["expectedNextState"].is_object(),
                "prediction state should expose expected next state");
    requireTrue(prediction.contains("observedNextState") && prediction["observedNextState"].is_object(),
                "prediction state should expose observed next state");
    requireTrue(prediction["calibration"].value("samples", 0) >= 1,
                "prediction calibration should record at least one comparison sample");
    requireTrue(prediction["calibration"].value("matched", 0) >= 1,
                "prediction calibration should record a matched observation when expectation aligns");
    requireTrue(state["episode"].contains("expectedNextState") && state["episode"]["expectedNextState"].is_object(),
                "episode state should persist expected next state");
    requireTrue(state["episode"].contains("predictionCalibration") && state["episode"]["predictionCalibration"].is_object(),
                "episode state should persist prediction calibration");
}

void testWorldModelStoreDetectsPredictionMismatchAndFeedsPromptContext() {
    world_model::WorldModelStore store(std::shared_ptr<KeyValueStore>{},
                                       std::shared_ptr<KeyValueStore>{},
                                       std::shared_ptr<KeyValueStore>{});

    json first = {
        {"sessionId", "wm-predictive-2"},
        {"modality", "vision"},
        {"graphSummary", "cat watching rain from the window sill"},
        {"metadata",
         {
             {"detections", json::array({json{{"label", "cat"}}, json{{"label", "window"}}})},
             {"expectedNextState",
              json{{"summary", "cat likely stays by the window and keeps watching the rain"},
                   {"entities", json::array({"cat", "window"})}}}
         }}
    };
    store.ingestEvidence(first);

    json second = {
        {"sessionId", "wm-predictive-2"},
        {"modality", "vision"},
        {"graphSummary", "dog splashes through a muddy puddle near the open gate"},
        {"metadata",
         {
             {"detections", json::array({json{{"label", "dog"}}, json{{"label", "gate"}}, json{{"label", "puddle"}}})},
             {"observedNextState",
              json{{"summary", "dog splashes through a muddy puddle near the open gate"},
                   {"entities", json::array({"dog", "gate", "puddle"})}}}
         }}
    };
    store.ingestEvidence(second);

    auto state = store.sessionState("wm-predictive-2", 4);
    const auto &prediction = state["prediction"];

    requireTrue(prediction["calibration"].value("mismatched", 0) >= 1,
                "prediction calibration should mark mismatched observations");
    requireTrue(prediction["contradictions"].is_array() && !prediction["contradictions"].empty(),
                "prediction state should surface contradictions after a mismatch");
    requireTrue(prediction["observedNextState"].value("matchedExpectedState", true) == false,
                "observed next state should record that the expectation was not matched");

    const std::string promptContext = world_model::buildPromptContext(state);
    requireTrue(promptContext.find("world_prediction|expected:") != std::string::npos,
                "prompt context should include the expected next state");
    requireTrue(promptContext.find("world_prediction|observed:") != std::string::npos,
                "prompt context should include the observed next state");
    requireTrue(promptContext.find("world_prediction|calibration:") != std::string::npos,
                "prompt context should include calibration feedback");
    requireTrue(promptContext.find("world_relations|active:") != std::string::npos,
                "prompt context should include normalized relation summaries");
}

void testSimulateVirtualSceneCreatesMultipleAgentsAndTimeline() {
    json state = {
        {"sessionId", "brain-2"},
        {"sceneState", {{"summary", "Observed a child holding a ball near a door."},
                         {"objectSlots", json::array({json{{"label", "child"}}, json{{"label", "ball"}}, json{{"label", "door"}}})}}},
        {"episode", {{"summary", "The child may move toward the door with the ball."}}},
        {"recentEvidence", json::array({json{{"modality", "vision"}, {"graphSummary", "child holding ball near door"}},
                                         json{{"modality", "speech"}, {"text", "the child is about to leave with the ball"}}})}
    };

    world_model::VirtualSceneOptions options;
    options.maxAgents = 3;
    options.maxSteps = 2;
    options.maxTrainSamples = 5;

    auto simulation = world_model::simulateVirtualScene(state, options);

    requireTrue(simulation.is_object(), "simulation should return an object");
    requireTrue(simulation.contains("agents") && simulation["agents"].is_array() && simulation["agents"].size() == 3,
                "simulation should create three virtual agents");
    requireTrue(simulation.contains("timeline") && simulation["timeline"].is_array() && !simulation["timeline"].empty(),
                "simulation timeline should not be empty");
    requireTrue(simulation.contains("trainSamples") && simulation["trainSamples"].is_array() && !simulation["trainSamples"].empty(),
                "simulation should emit train samples");
}

void testVirtualSceneTrainingSampleLimitAndTruncation() {
    json state = {
        {"sessionId", "brain-3"},
        {"sceneState", {{"summary", "Observed an agent near a workstation with many dynamic objects around it."},
                         {"objectSlots", json::array({json{{"label", "agent"}}, json{{"label", "workstation"}}})}}},
        {"episode", {{"summary", "The workstation scene is evolving quickly and should be predicted."}}},
        {"recentEvidence", json::array({json{{"modality", "vision"}, {"graphSummary", "agent near workstation with long detailed visual context that should be truncated"}},
                                         json{{"modality", "speech"}, {"text", "the workstation scene is changing and the agent might move soon"}}})}
    };

    world_model::VirtualSceneOptions options;
    options.maxAgents = 4;
    options.maxSteps = 3;
    options.maxTrainSamples = 2;
    options.maxEventChars = 28;

    auto samples = world_model::buildVirtualSceneTrainingSamples(state, options);

    requireTrue(samples.size() == 2, "virtual scene training should respect maxTrainSamples");
    requireTrue(samples[0].target.find("...") != std::string::npos || samples[1].target.find("...") != std::string::npos,
                "virtual scene training sample targets should truncate long events");
}

void testStructuralBrainModePropagatesIntoSimulation() {
    json state = {
        {"sessionId", "brain-4"},
        {"sceneState", {{"summary", "Observed a robot arm sorting colored blocks while a speaker announces the next target."},
                         {"objectSlots", json::array({json{{"label", "robot arm"}}, json{{"label", "red block"}}, json{{"label", "blue block"}}})}}},
        {"episode", {{"summary", "The robot arm is expected to place the red block into the target bin."}}},
        {"recentEvidence", json::array({json{{"modality", "vision"}, {"graphSummary", "robot arm near red and blue blocks"}},
                                         json{{"modality", "speech"}, {"text", "place the red block into the target bin"}}})}
    };

    world_model::VirtualSceneOptions options;
    options.maxAgents = 3;
    options.maxSteps = 2;
    options.maxTrainSamples = 4;
    options.brainProfile = world_model::BrainProfileKind::Structural;

    auto simulation = world_model::simulateVirtualScene(state, options);

    requireTrue(simulation.is_object(), "simulation should return an object");
    requireTrue(simulation.value("brainProfile", std::string()) == "structural",
                "simulation should record structural brain mode");
    requireTrue(simulation.contains("brainState") && simulation["brainState"].is_object(),
                "simulation should expose the selected brain state");
    requireTrue(simulation["brainState"].value("profile", std::string()) == "structural",
                "simulation brain state should match the selected structural mode");
}

void testVirtualSceneBuildsEmbodied3DWorldAndEcologyFromVideo() {
    json state = {
        {"sessionId", "brain-embodied-1"},
        {"sceneState", {{"summary", "Observed multiple agents near a marshland walkway while distant motion ripples across the habitat."},
                         {"objectSlots", json::array({json{{"label", "walkway"}}, json{{"label", "drone pad"}}, json{{"label", "sensor mast"}}})},
                         {"tags", json::array({"marshland", "migration", "waterline"})}}},
        {"episode", {{"summary", "The agents should coordinate movement along the walkway while monitoring the habitat."}}},
        {"recentEvidence", json::array({json{{"modality", "video"}, {"graphSummary", "broad migrating motion across wet grassland with clustered splashes near the waterline"}},
                                         json{{"modality", "vision"}, {"graphSummary", "agents standing near walkway and sensor mast"}},
                                         json{{"modality", "speech"}, {"text", "hold formation near the walkway and report ecological changes"}}})}
    };

    world_model::VirtualSceneOptions options;
    options.maxAgents = 6;
    options.maxSteps = 2;
    options.maxTrainSamples = 8;
    options.mapWidth = 7;
    options.mapHeight = 5;
    options.mapDepth = 4;
    options.maxDialogueTurns = 2;
    options.maxEcologyClusters = 2;
    options.physicsEnabled = true;
    options.physicsBackend = "bullet3";
    options.physicsSubsteps = 5;
    options.earthMapEnabled = true;
    options.earthMapRequest = json{{"enabled", true},
                                   {"sourceUri", "static/earth_maps/china_relief_heightfield.json"},
                                   {"format", "heightfield"},
                                   {"coordinateFrame", "wgs84-local-enu"},
                                   {"regionLabel", "china-relief-demo"},
                                   {"lod", 6}};

    auto simulation = world_model::simulateVirtualScene(state, options);

    requireTrue(simulation.contains("agents") && simulation["agents"].is_array() && simulation["agents"].size() == 6,
                "simulation should respect the requested embodied agent count");
    requireTrue(simulation.contains("entities") && simulation["entities"].is_array() && simulation["entities"].size() >= 6,
                "simulation should expose a combined entity roster");
    requireTrue(simulation["agents"][0].contains("body") && simulation["agents"][0]["body"].is_object(),
                "agents should expose embodied body metadata");
    requireTrue(simulation["agents"][0].contains("position") && simulation["agents"][0]["position"].is_object(),
                "agents should expose grounded 3D positions");
    requireTrue(simulation.contains("worldMap3D") && simulation["worldMap3D"].is_object(),
                "simulation should expose a 3D world map");
    requireTrue(simulation["worldMap3D"].contains("dimensions") && simulation["worldMap3D"]["dimensions"].is_object(),
                "3D world map should include dimensions");
    requireTrue(simulation["worldMap3D"]["dimensions"].value("depth", 0) == 4,
                "3D world map should preserve the requested map depth");
    requireTrue(simulation.contains("dialogueTranscript") && simulation["dialogueTranscript"].is_array() && !simulation["dialogueTranscript"].empty(),
                "embodied agents should exchange dialogue turns");
    requireTrue(simulation.contains("nonIntelligentEntities") && simulation["nonIntelligentEntities"].is_array() && !simulation["nonIntelligentEntities"].empty(),
                "simulation should include non-intelligent ecological entities");
    requireTrue(simulation["nonIntelligentEntities"][0].value("abstraction", std::string()) == "video-ecology-cluster",
                "ecological entities should come from video abstraction clusters");
    requireTrue(simulation.contains("timeline") && simulation["timeline"].is_array() && simulation["timeline"][0].contains("dialogues") &&
                    simulation["timeline"][0]["dialogues"].is_array() && !simulation["timeline"][0]["dialogues"].empty(),
                "timeline should retain per-step dialogue exchanges");
    requireTrue(simulation.contains("physicsScene") && simulation["physicsScene"].is_object(),
                "simulation should expose a physics scene plan");
    requireTrue(simulation["physicsScene"].value("enabled", false),
                "physics scene should be enabled when requested");
    requireTrue(simulation["physicsScene"].value("backend", std::string()) == "bullet3",
                "physics scene should retain the requested backend");
    requireTrue(simulation["physicsScene"].contains("rigidBodies") && simulation["physicsScene"]["rigidBodies"].is_array() && !simulation["physicsScene"]["rigidBodies"].empty(),
                "physics scene should include rigid body seeds");
    requireTrue(simulation["physicsScene"].contains("earthMap") && simulation["physicsScene"]["earthMap"].is_object() && simulation["physicsScene"]["earthMap"].value("enabled", false),
                "physics scene should include an earth map import manifest");
    requireTrue(simulation["physicsScene"]["earthMap"].value("format", std::string()) == "heightfield",
                "earth map manifest should preserve the configured format");
    requireTrue(simulation["physicsScene"]["earthMap"].value("sourceUri", std::string()) == "static/earth_maps/china_relief_heightfield.json",
                "earth map manifest should preserve the bundled heightfield source");
    requireTrue(simulation["worldMap3D"].contains("earthMap") && simulation["worldMap3D"]["earthMap"].is_object(),
                "3D world map should expose the earth map reference when available");
    requireTrue(simulation.contains("mobilityPlan") && simulation["mobilityPlan"].is_object(),
                "simulation should expose a rover mobility plan");
    requireTrue(simulation["mobilityPlan"].value("environmentConstraint", std::string()) == "ground-only",
                "simulation mobility plan should remain ground-only");
    requireTrue(simulation["agents"][0].contains("body") && simulation["agents"][0]["body"].is_object() &&
                    simulation["agents"][0]["body"].contains("mobilityProfile") && simulation["agents"][0]["body"]["mobilityProfile"].is_object(),
                "agent bodies should expose a mobility profile for rover planning");
}

void testFunctionalBrainRuntimeIncludesControlAndActionLoops() {
    json state = {
        {"sessionId", "brain-runtime-1"},
        {"sceneState", {{"summary", "Observed a drone near a landing pad while audio requests a safe descent."},
                         {"objectSlots", json::array({json{{"label", "drone"}}, json{{"label", "landing pad"}}})},
                         {"tags", json::array({"drone", "landing", "descent"})}}},
        {"episode", {{"summary", "The drone should descend carefully onto the landing pad."}}},
        {"recentEvidence", json::array({json{{"modality", "vision"}, {"graphSummary", "drone hovering over landing pad"}},
                                         json{{"modality", "speech"}, {"text", "descend slowly and land on the pad"}}})}
    };

    world_model::BrainProfileOptions options;
    options.kind = world_model::BrainProfileKind::Functional;
    auto brain = world_model::buildBrainProfile(state, options);

    requireTrue(brain.is_object(), "functional brain profile should be an object");
    requireTrue(brain.contains("selectedAction") && brain["selectedAction"].is_string() && !brain["selectedAction"].get<std::string>().empty(),
                "functional brain should expose a selected action");
    requireTrue(brain.contains("functionalRuntime") && brain["functionalRuntime"].is_object(),
                "functional brain should contain runtime state");
    requireTrue(jsonObjectContainsKey(brain["functionalRuntime"], "executiveController"),
                "functional runtime should contain executive controller");
    requireTrue(jsonObjectContainsKey(brain["functionalRuntime"], "valueSystem"),
                "functional runtime should contain value system");
    requireTrue(jsonObjectContainsKey(brain["functionalRuntime"], "actionBuffer"),
                "functional runtime should contain action buffer");
}

void testCognitiveStateIncludesReasoningAgenda() {
    json state = {
        {"sessionId", "brain-agenda-1"},
        {"sceneState", {{"summary", "Observed a rover near a damaged solar panel while an operator reports unstable power."},
                         {"objectSlots", json::array({json{{"label", "rover"}}, json{{"label", "solar panel"}}})},
                         {"tags", json::array({"rover", "power", "repair"})}}},
        {"episode", {{"summary", "The rover should inspect the damaged panel before attempting a repair."}}},
        {"recentEvidence", json::array({json{{"modality", "vision"}, {"graphSummary", "rover parked near cracked solar panel"}},
                                         json{{"modality", "speech"}, {"text", "power output keeps dropping and repair is not yet safe"}}})}
    };

    auto cognitive = world_model::buildCognitiveState(state);

    requireTrue(cognitive.is_object(), "cognitive state should be an object");
    requireTrue(cognitive.contains("reasoningAgenda") && cognitive["reasoningAgenda"].is_object(),
                "cognitive state should expose a reasoning agenda");
    requireTrue(cognitive["reasoningAgenda"].contains("hypotheses") && cognitive["reasoningAgenda"]["hypotheses"].is_array() && !cognitive["reasoningAgenda"]["hypotheses"].empty(),
                "reasoning agenda should include hypotheses");
    requireTrue(cognitive["reasoningAgenda"].contains("openQuestions") && cognitive["reasoningAgenda"]["openQuestions"].is_array() && !cognitive["reasoningAgenda"]["openQuestions"].empty(),
                "reasoning agenda should include open questions");
    requireTrue(cognitive["reasoningAgenda"].contains("nextStep") && cognitive["reasoningAgenda"]["nextStep"].is_string() && !cognitive["reasoningAgenda"]["nextStep"].get<std::string>().empty(),
                "reasoning agenda should include a next step");

    world_model::BrainProfileOptions structuralOptions;
    structuralOptions.kind = world_model::BrainProfileKind::Structural;
    auto structural = world_model::buildBrainProfile(state, structuralOptions);
    requireTrue(structural.contains("reasoningAgenda") && structural["reasoningAgenda"].is_object(),
                "structural brain profile should also carry the reasoning agenda");
}

void testBrainProfileIncludesResponsePlan() {
    json state = {
        {"sessionId", "brain-plan-1"},
        {"sceneState", {{"summary", "Observed a maintenance bot near a jammed conveyor while sensors report intermittent faults."},
                         {"objectSlots", json::array({json{{"label", "maintenance bot"}}, json{{"label", "conveyor"}}})},
                         {"tags", json::array({"maintenance", "fault", "conveyor"})}}},
        {"episode", {{"summary", "The bot should diagnose the conveyor jam before resuming movement."}}},
        {"recentEvidence", json::array({json{{"modality", "vision"}, {"graphSummary", "maintenance bot parked beside jammed conveyor"}},
                                         json{{"modality", "speech"}, {"text", "fault state is intermittent and the root cause is not confirmed"}}})}
    };

    world_model::BrainProfileOptions structuralOptions;
    structuralOptions.kind = world_model::BrainProfileKind::Structural;
    auto structural = world_model::buildBrainProfile(state, structuralOptions);

    requireTrue(structural.contains("responsePlan") && structural["responsePlan"].is_object(),
                "brain profile should expose a response plan");
    requireTrue(structural["responsePlan"].contains("answerOutline") && structural["responsePlan"]["answerOutline"].is_array() && !structural["responsePlan"]["answerOutline"].empty(),
                "response plan should include answer outline");
    requireTrue(structural["responsePlan"].contains("critiqueChecklist") && structural["responsePlan"]["critiqueChecklist"].is_array() && !structural["responsePlan"]["critiqueChecklist"].empty(),
                "response plan should include critique checklist");
    requireTrue(structural["responsePlan"].contains("simulationTargets") && structural["responsePlan"]["simulationTargets"].is_array() && !structural["responsePlan"]["simulationTargets"].empty(),
                "response plan should include simulation targets");
}

void testResponsePlanBuildsGroundMobilityPlanForRover() {
    json state = {
        {"sessionId", "brain-rover-plan-1"},
        {"sceneState", {{"summary", "Observed a rover near a damaged solar panel on rough terrain with a narrow approach path."},
                         {"objectSlots", json::array({json{{"label", "rover"}}, json{{"label", "damaged solar panel"}}, json{{"label", "approach path"}}})},
                         {"tags", json::array({"rover", "terrain", "repair", "slope"})}}},
        {"episode", {{"summary", "The rover should inspect the damaged panel without losing traction on the slope."}}},
        {"recentEvidence", json::array({json{{"modality", "vision"}, {"graphSummary", "rover parked on uneven ground near damaged panel"}},
                                         json{{"modality", "speech"}, {"text", "approach slowly and keep a safe braking margin on the slope"}}})}
    };

    auto plan = world_model::buildResponsePlan(state, 4, 140);

    requireTrue(plan.contains("mobilityPlan") && plan["mobilityPlan"].is_object(),
                "response plan should expose a ground mobility plan for rover scenes");
    requireTrue(plan["mobilityPlan"].value("vehicleClass", std::string()) == "all-terrain-four-wheel-rover",
                "mobility plan should target the four-wheel rover platform");
    requireTrue(plan["mobilityPlan"].value("environmentConstraint", std::string()) == "ground-only",
                "mobility plan should preserve ground-only constraints");
    requireTrue(plan["mobilityPlan"].contains("bigBrain") && plan["mobilityPlan"]["bigBrain"].is_object(),
                "mobility plan should expose big-brain route planning");
    requireTrue(plan["mobilityPlan"]["bigBrain"].contains("simulationTargets") &&
                    plan["mobilityPlan"]["bigBrain"]["simulationTargets"].is_array() &&
                    !plan["mobilityPlan"]["bigBrain"]["simulationTargets"].empty(),
                "mobility plan should request route simulations");
    requireTrue(jsonArrayContainsString(plan["simulationTargets"], "simulate rover traction along the first route segment"),
                "response plan should merge rover simulation targets into the main simulation list");
    requireTrue(plan["mobilityPlan"].contains("actionConstraints") && plan["mobilityPlan"]["actionConstraints"].is_object(),
                "mobility plan should expose action constraints for the action adapter");
    requireTrue(plan["mobilityPlan"].contains("littleBrain") && plan["mobilityPlan"]["littleBrain"].is_object() &&
                    plan["mobilityPlan"]["littleBrain"].contains("selectedAction") &&
                    plan["mobilityPlan"]["littleBrain"]["selectedAction"].is_string() &&
                    !plan["mobilityPlan"]["littleBrain"]["selectedAction"].get<std::string>().empty(),
                "mobility plan should expose a selected action for the adapter boundary");
}



