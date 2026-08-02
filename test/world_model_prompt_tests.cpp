#include "world_model.hpp"

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

using json = nlohmann::json;

class MemoryKeyValueStore : public KeyValueStore {
public:
    std::optional<json> get(const std::string &key) override {
        auto it = data_.find(key);
        if (it == data_.end()) {
            return std::nullopt;
        }
        return it->second;
    }

    void put(const std::string &key, const json &value) override {
        data_[key] = value;
    }

    void del(const std::string &key) override {
        data_.erase(key);
    }

    std::vector<std::pair<std::string, json>> entries(const std::string &prefix) override {
        std::vector<std::pair<std::string, json>> out;
        for (const auto &entry : data_) {
            if (entry.first.rfind(prefix, 0) == 0) {
                out.push_back(entry);
            }
        }
        return out;
    }

private:
    std::unordered_map<std::string, json> data_;
};

void requireTrue(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void requireContains(const std::string &text, const std::string &needle, const std::string &message) {
    requireTrue(text.find(needle) != std::string::npos, message + " missing=[" + needle + "]\nactual:\n" + text);
}

std::size_t countSubstring(const std::string &text, const std::string &needle) {
    if (needle.empty()) {
        return 0;
    }
    std::size_t count = 0;
    std::size_t cursor = 0;
    while ((cursor = text.find(needle, cursor)) != std::string::npos) {
        ++count;
        cursor += needle.size();
    }
    return count;
}

void testBuildPromptContextIncludesSceneEpisodeAndEvidence() {
    json state = {
        {"sessionId", "s-1"},
        {"sceneState",
         {{"summary", "Observed 3 evidence item(s) across modalities text, vision. Objects: cat, window. Context: cat on window ledge"},
          {"objectSlots", json::array({json{{"label", "cat"}}, json{{"label", "window"}}})}}},
        {"episode", {{"summary", "The cat moved near the window and is now resting there."}}},
        {"recentEvidence",
         json::array({json{{"modality", "vision"}, {"graphSummary", "cat on window ledge"}},
                      json{{"modality", "text"}, {"text", "User said the cat is sleeping on the ledge after moving there."}}})}
    };

    const std::string promptContext = world_model::buildPromptContext(state);

    requireContains(promptContext, "world_scene|summary:", "scene summary should be present");
    requireContains(promptContext, "world_episode|summary:", "episode summary should be present");
    requireContains(promptContext, "world_objects|labels: cat, window", "object labels should be present");
    requireContains(promptContext, "world_recent|vision:", "vision evidence should be present");
    requireContains(promptContext, "world_recent|text:", "text evidence should be present");
}

void testMergePromptContextLeavesGraphUntouchedWhenWorldStateEmpty() {
    json state = {
        {"sessionId", "s-2"},
        {"sceneState", {{"summary", ""}, {"objectSlots", json::array()}}},
        {"episode", {{"summary", ""}}},
        {"recentEvidence", json::array()}
    };

    const std::string graphContext = "gnn_stage2|align=0.75|w=0.82:cat window ledge";
    const std::string merged = world_model::mergePromptContext(graphContext, state);
    requireTrue(merged == graphContext, "empty world state should not change graph context");
}

void testBuildLlamacppWorldPromptShellPrioritizesSceneVideoAndPlan() {
    const std::string graphContext =
        "gnn_stage2|align=0.88|w=0.91:cat near gate\n"
        "world_recent|text: operator reports movement near the gate\n"
        "capture path: micro-mipi-csi\n"
        "world_scene|summary: cat approaches the side gate while the fence remains closed\n"
        "world_plan|goal: explain the most actionable visual change\n"
        "addon_contract|tool=search";

    const std::string shell = world_model::buildLlamacppWorldPromptShell(graphContext, 5, 120);

    requireContains(shell, "World model context for llama.cpp", "llamacpp shell should include a provider-specific header");
    requireContains(shell, "world_plan|goal: explain the most actionable visual change", "llamacpp shell should keep the current goal");
    requireContains(shell, "world_scene|summary: cat approaches the side gate while the fence remains closed", "llamacpp shell should keep scene summary");
    requireContains(shell, "capture path: micro-mipi-csi", "llamacpp shell should keep direct capture path hints");
    requireContains(shell, "world_recent|text: operator reports movement near the gate", "llamacpp shell should keep recent evidence");
    requireTrue(shell.find("addon_contract|tool=search") == std::string::npos, "llamacpp shell should drop internal routing-only lines");
}

void testBuildLlamacppWorldPromptShellConsumesBulletizedModelGraphContext() {
    const std::string graphContext =
        "Supporting evidence:\n"
        "- world_plan|goal: inspect the fastest moving target\n"
        "- world_scene|summary: rover watches the gate from a direct camera feed\n"
        "- capture path: micro-mipi-csi\n"
        "- world_agenda|next_step: verify whether the target is accelerating toward the gate\n"
        "- world_plan|action: slow rover and hold lane\n"
        "- world_plan|action_guard: verify>=0.60 || uncertainty<=0.45\n"
        "- world_plan|simulate: simulate next state around target at the gate\n"
        "- world_reflection|needed: yes | reason=trajectory is noisy | priority=0.62\n";

    const std::string shell = world_model::buildLlamacppWorldPromptShell(graphContext, 8, 120);

    requireTrue(shell.find("Supporting evidence:") == std::string::npos,
                "llamacpp shell should drop the generic evidence wrapper header");
    requireTrue(shell.find("- - world_plan|action") == std::string::npos,
                "llamacpp shell should normalize bulletized model graph context lines");
    requireContains(shell, "world_agenda|next_step: verify whether the target is accelerating toward the gate",
                    "llamacpp shell should preserve bulletized next-step guidance");
    requireContains(shell, "world_plan|action: slow rover and hold lane",
                    "llamacpp shell should preserve bulletized action guidance");
    requireContains(shell, "world_plan|action_guard: verify>=0.60 || uncertainty<=0.45",
                    "llamacpp shell should preserve bulletized action guards");
    requireContains(shell, "world_plan|simulate: simulate next state around target at the gate",
                    "llamacpp shell should preserve bulletized simulation targets");
    requireContains(shell, "world_reflection|needed: yes",
                    "llamacpp shell should preserve bulletized reflection requirements");

    const auto goalPos = shell.find("world_plan|goal: inspect the fastest moving target");
    const auto nextStepPos = shell.find("world_agenda|next_step: verify whether the target is accelerating toward the gate");
    const auto scenePos = shell.find("world_scene|summary: rover watches the gate from a direct camera feed");
    const auto actionGuardPos = shell.find("world_plan|action_guard: verify>=0.60 || uncertainty<=0.45");
    const auto reflectionPos = shell.find("world_reflection|needed: yes | reason=trajectory is noisy | priority=0.62");

    requireTrue(goalPos != std::string::npos && nextStepPos != std::string::npos && goalPos < nextStepPos,
                "llamacpp shell should rank the current goal before next-step guidance");
    requireTrue(nextStepPos != std::string::npos && scenePos != std::string::npos && nextStepPos < scenePos,
                "llamacpp shell should rank next-step guidance ahead of lower-priority scene context when the line budget is tight");
    requireTrue(actionGuardPos != std::string::npos && reflectionPos != std::string::npos && actionGuardPos < reflectionPos,
                "llamacpp shell should rank action guards ahead of reflection reminders");
}

void testPromptContextRespectsEvidenceLimitsAndTruncation() {
    json state = {
        {"sessionId", "s-3"},
        {"sceneState", {{"summary", "Observed several events."}, {"objectSlots", json::array()}}},
        {"episode", {{"summary", "Episode summary."}}},
        {"recentEvidence",
         json::array({json{{"modality", "text"}, {"text", "first evidence that should be dropped because only the most recent items are kept"}},
                      json{{"modality", "text"}, {"text", "second evidence also should be dropped before prompt assembly"}},
                      json{{"modality", "vision"}, {"graphSummary", "third evidence is long enough to need truncation in the final prompt context line"}},
                      json{{"modality", "speech"}, {"text", "fourth evidence is also long enough to be truncated for graph safety"}}})}
    };

    world_model::PromptContextOptions options;
    options.maxRecentEvidence = 2;
    options.maxEvidenceChars = 24;

    const std::string promptContext = world_model::buildPromptContext(state, options);

    requireTrue(countSubstring(promptContext, "world_recent|") == 2, "prompt context should keep only the configured number of recent evidence lines");
    requireTrue(promptContext.find("first evidence") == std::string::npos, "older evidence should be dropped from prompt context");
    requireTrue(promptContext.find("second evidence") == std::string::npos, "older evidence should be dropped from prompt context");
    requireContains(promptContext, "...", "truncated evidence should use ellipsis");
}

void testReasoningAgendaPromptContextHighlightsQuestionsAndReflection() {
    json state = {
        {"sessionId", "s-4"},
        {"sceneState", {{"summary", "Observed a single noisy text report without stable object tracking."}, {"objectSlots", json::array()}}},
        {"episode", {{"summary", ""}}},
        {"recentEvidence", json::array({json{{"modality", "text"}, {"text", "the scene is unstable and the next action is unclear"}}})}
    };

    const auto cognitive = world_model::buildCognitiveState(state);
    const std::string agendaContext = world_model::buildReasoningAgendaPromptContext(cognitive, 120);

    requireTrue(cognitive.contains("reasoningAgenda") && cognitive["reasoningAgenda"].is_object(),
                "cognitive state should expose a reasoning agenda");
    requireContains(agendaContext, "world_agenda|question:", "agenda prompt should surface open questions");
    requireContains(agendaContext, "world_agenda|next_step:", "agenda prompt should surface the next step");
    requireContains(agendaContext, "world_reflection|needed:", "agenda prompt should request reflection when uncertainty is high");
}

void testResponsePlanPromptContextHighlightsOutlineCriticAndSimulation() {
    json state = {
        {"sessionId", "s-5"},
        {"sceneState", {{"summary", "Observed a rover near a damaged solar array while telemetry remains unstable."},
                         {"objectSlots", json::array({json{{"label", "rover"}}, json{{"label", "solar array"}}})}}},
        {"episode", {{"summary", "The rover should diagnose the damaged array before attempting a repair."}}},
        {"recentEvidence", json::array({json{{"modality", "vision"}, {"graphSummary", "rover parked near cracked solar array"}},
                                         json{{"modality", "speech"}, {"text", "power output is unstable and the repair path is unclear"}}})}
    };

    const auto cognitive = world_model::buildCognitiveState(state);
    const auto responsePlan = world_model::buildResponsePlan(cognitive, 4, 120);
    const std::string planContext = world_model::buildResponsePlanPromptContext(responsePlan, 120);

    requireTrue(responsePlan.is_object(), "response plan should be an object");
    requireContains(planContext, "world_plan|goal:", "response plan prompt should expose a goal frame");
    requireContains(planContext, "world_plan|outline:", "response plan prompt should expose answer outline");
    requireContains(planContext, "world_plan|critic:", "response plan prompt should expose critique checklist");
    requireContains(planContext, "world_plan|simulate:", "response plan prompt should expose simulation targets");
    requireContains(planContext, "world_plan|action:", "response plan prompt should expose the selected body action");
    requireContains(planContext, "world_plan|action_guard:", "response plan prompt should expose action guardrails");
}

void testSelectiveKvCachePromotesRepeatedColdHits() {
    auto coldStore = std::make_shared<MemoryKeyValueStore>();
    world_model::SelectiveKvCache::Options options;
    options.hotLimit = 4;
    options.promoteHits = 2;
    options.hotTtlMs = 60000;
    world_model::SelectiveKvCache cache(coldStore, options);

    world_model::HotspotAnalysis analysis;
    analysis.scope = "reply-hotspot";
    analysis.canonicalKey = "reply-hotspot:test";
    analysis.hotScore = 0.4;
    analysis.hotCandidate = false;

    cache.put("reply:alpha", json{{"reply", "cached answer"}}, analysis);
    requireTrue(!cache.hasHot("reply:alpha"), "entry should stay cold after the initial write when not marked hot");

    auto first = cache.get("reply:alpha");
    requireTrue(first.has_value() && (*first)["reply"] == "cached answer", "first cold load should return payload");
    requireTrue(cache.hasHot("reply:alpha"), "repeated cold hits should promote the entry into the hot cache");

    auto second = cache.get("reply:alpha");
    requireTrue(second.has_value() && (*second)["reply"] == "cached answer", "promoted hot entry should still return the same payload");
}

void testReasoningAssemblyBuildsCacheableBundle() {
    json state = {
        {"sessionId", "cache-bundle-1"},
        {"sceneState", {{"summary", "Observed a rover near the gate while telemetry remains noisy."},
                         {"objectSlots", json::array({json{{"label", "rover"}}, json{{"label", "gate"}}})},
                         {"tags", json::array({"rover", "gate", "telemetry"})}}},
        {"episode", {{"summary", "The rover should confirm the gate state before moving."}}},
        {"recentEvidence", json::array({json{{"modality", "vision"}, {"graphSummary", "rover pauses beside the gate"}},
                                         json{{"modality", "speech"}, {"text", "telemetry says the gate status is uncertain"}}})}
    };

    world_model::ReasoningAssemblyOptions options;
    options.includeBrainContext = true;
    options.includeReasoningAgenda = true;
    options.includeReasoningPlan = true;
    options.maxContextChars = 140;

    const std::string cacheKey = world_model::buildReasoningAssemblyCacheKey("capture path: micro-mipi-csi", state, options);
    const json bundle = world_model::buildReasoningAssembly("capture path: micro-mipi-csi", state, options);

    requireTrue(!cacheKey.empty(), "reasoning assembly cache key should not be empty");
    requireTrue(bundle.is_object(), "reasoning assembly should return an object bundle");
    requireContains(bundle.value("graphContext", std::string()), "world_scene|summary:", "reasoning bundle should include prompt context in graphContext");
    requireTrue(bundle.contains("cognitiveBrainState") && bundle["cognitiveBrainState"].is_object(), "reasoning bundle should carry cognitive brain state");
    requireTrue(bundle.contains("responsePlan") && bundle["responsePlan"].is_object(), "reasoning bundle should carry response plan when enabled");
}

void testCompiledInvocationPlanSelectsEmbeddedLlamaPath() {
    const auto plan = world_model::buildCompiledInvocationPlan("llamacpp", true, false);

    requireTrue(plan.kind == world_model::CompiledInvocationKind::LlamaCppEmbedded,
                "llama.cpp provider should select the embedded llama path");
    requireTrue(plan.useExternalAdapter, "llama.cpp plan should still use the external adapter transport");
    requireTrue(plan.useLlamacppWorldShell, "llama.cpp plan with context should request the llama world shell");
    requireTrue(plan.injectRuntimeIdentity, "llama.cpp plan should embed runtime identity instructions");
    requireTrue(plan.requestPath == "/v1/chat/completions", "llama.cpp plan should embed the OpenAI-compatible request path");
}

void testCompiledInvocationPlanSelectsNativeDirectPath() {
    const auto plan = world_model::buildCompiledInvocationPlan("native", false, false);

    requireTrue(plan.kind == world_model::CompiledInvocationKind::NativeDirect,
                "native provider should bypass the external adapter and stay on the direct transformer path");
    requireTrue(!plan.useExternalAdapter, "native direct plan should not use external adapter transport");
    requireTrue(plan.compiledEmbedding, "native direct plan should be marked as a compiled embedding path");
}

} // namespace

int main() {
    try {
        testBuildPromptContextIncludesSceneEpisodeAndEvidence();
        testMergePromptContextLeavesGraphUntouchedWhenWorldStateEmpty();
        testBuildLlamacppWorldPromptShellPrioritizesSceneVideoAndPlan();
        testBuildLlamacppWorldPromptShellConsumesBulletizedModelGraphContext();
        testPromptContextRespectsEvidenceLimitsAndTruncation();
        testReasoningAgendaPromptContextHighlightsQuestionsAndReflection();
        testResponsePlanPromptContextHighlightsOutlineCriticAndSimulation();
        testSelectiveKvCachePromotesRepeatedColdHits();
        testReasoningAssemblyBuildsCacheableBundle();
        testCompiledInvocationPlanSelectsEmbeddedLlamaPath();
        testCompiledInvocationPlanSelectsNativeDirectPath();
        std::cout << "world_model_prompt_tests: ok" << std::endl;
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "world_model_prompt_tests: failed: " << ex.what() << std::endl;
        return 1;
    }
}